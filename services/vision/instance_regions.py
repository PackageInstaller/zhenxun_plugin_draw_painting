"""Visible full-character regions for crowded anime illustrations.

The detector is the RTMDet-Ins stage of AnimeInsSeg.  It is deliberately run on
CPU: the long-lived WD, Camie, person and face CUDA arenas already reserve up to
9.75 GiB, so another resident CUDA session would recreate the OOM failure this
plugin previously recovered from.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

from zhenxun.services.log import logger

from .assets import (
    FOREGROUND_MODEL,
    INSTANCE_MODEL,
    INSTANCE_REFINER,
    INSTANCE_VERSION,
    PERSON_MODEL,
)
from .faces import FaceBox, unrotate_box
from .subjects import AnimeBoxDetector, PersonBox

DETECTION_THRESHOLD = 0.28
MASK_THRESHOLD = 0.45
REFINE_THRESHOLD = 0.4
FOREGROUND_THRESHOLD = 0.35
MAX_CANDIDATES = 100
MASK_INPUT_SIZE = 640


@dataclass(slots=True)
class InferenceRegion:
    """The exact pixels and bounds used for one person's tag inference."""

    view: Image.Image
    box: tuple[int, int, int, int]
    mode: str
    outline: list[list[tuple[int, int]]]
    confidence: float | None = None
    proposal_box: tuple[int, int, int, int] | None = None
    contaminated: bool = False
    # Source-resolution ownership mask.  Kept out of every serialized entry;
    # only the query-time subject salience scorer consumes it.
    mask: np.ndarray | None = None


@dataclass(slots=True)
class _Candidate:
    point: tuple[float, float]
    stride: int
    kernel: np.ndarray
    box_canvas: tuple[float, float, float, float]
    box_source: tuple[int, int, int, int]
    score: float


@dataclass(slots=True)
class _Proposal:
    """One person's refined probability map before global adjudication."""

    face_index: int
    score: float
    proposal_box: tuple[int, int, int, int]
    view_image: Image.Image
    refined: np.ndarray
    angle: int = 0
    frame_offset: tuple[int, int] = (0, 0)


def _area(box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _intersection(left, right) -> float:
    return max(0.0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0.0, min(left[3], right[3]) - max(left[1], right[1])
    )


def _iou(left, right) -> float:
    overlap = _intersection(left, right)
    return overlap / max(1.0, _area(left) + _area(right) - overlap)


def _assign(scores: np.ndarray, minimum: float) -> dict[int, int]:
    """Maximum-weight one-to-one assignment, with a dependency-free fallback."""
    if scores.size == 0:
        return {}
    try:
        from scipy.optimize import linear_sum_assignment

        rows, columns = linear_sum_assignment(-scores)
        return {
            int(row): int(column)
            for row, column in zip(rows, columns, strict=True)
            if scores[row, column] >= minimum
        }
    except Exception:
        edges = sorted(
            (
                (float(scores[row, column]), row, column)
                for row in range(scores.shape[0])
                for column in range(scores.shape[1])
                if scores[row, column] >= minimum
            ),
            reverse=True,
        )
        result, used_rows, used_columns = {}, set(), set()
        for _, row, column in edges:
            if row not in used_rows and column not in used_columns:
                result[row] = column
                used_rows.add(row)
                used_columns.add(column)
        return result


def match_faces_to_people(
    faces: list[FaceBox], people: list[PersonBox]
) -> dict[int, int]:
    """Use the face as an identity anchor and a person box as a full-body proposal."""
    scores = np.full((len(faces), len(people)), -1.0, dtype=np.float32)
    for face_index, face in enumerate(faces):
        face_area = _area(face.xyxy)
        face_height = max(1, face.xyxy[3] - face.xyxy[1])
        face_center = (
            (face.xyxy[0] + face.xyxy[2]) / 2,
            (face.xyxy[1] + face.xyxy[3]) / 2,
        )
        for person_index, person in enumerate(people):
            coverage = _intersection(face.xyxy, person.xyxy) / max(1.0, face_area)
            px0, py0, px1, py1 = person.xyxy
            contains_center = (
                px0 <= face_center[0] <= px1 and py0 <= face_center[1] <= py1
            )
            if coverage < 0.55 or not contains_center or person.area < face_area * 2:
                continue
            relative_y = (face_center[1] - py0) / max(1, py1 - py0)
            # The "face sits in the upper part" rule only describes full-body
            # boxes.  A torso crop ends shortly below the chin, so the face
            # legitimately fills most of it and must not be rejected.
            body_below = py1 - face.xyxy[3]
            if body_below >= face_height * 1.2 and relative_y > 0.62:
                continue
            horizontal = abs(face_center[0] - (px0 + px1) / 2) / max(1, px1 - px0)
            scores[face_index, person_index] = (
                1.8 * coverage
                + 0.6 * person.confidence
                + 0.35 * max(0.0, 1.0 - relative_y)
                + 0.2 * max(0.0, 1.0 - horizontal)
            )
    return _assign(scores, 1.25)


def _letterbox(image: Image.Image):
    resized = image.copy()
    resized.thumbnail((MASK_INPUT_SIZE, MASK_INPUT_SIZE), Image.Resampling.LANCZOS)
    # AnimeInsSeg's official MMDetection pipeline pads on the right/bottom.
    # Center padding (used by a third-party demo) shifts priors and turns the
    # mask into a coarse half-plane on tall illustrations.
    left = top = 0
    canvas = Image.new("RGB", (MASK_INPUT_SIZE, MASK_INPUT_SIZE), (114, 114, 114))
    canvas.paste(resized, (left, top))
    # The exported backbone consumes RGB-normalized tensors.  MMDetection's
    # original pipeline starts from BGR files and converts them to RGB in its
    # data preprocessor; PIL has already performed that conversion here.
    array = np.asarray(canvas, dtype=np.float32)
    mean = np.asarray((123.675, 116.28, 103.53), dtype=np.float32)
    std = np.asarray((58.395, 57.12, 57.375), dtype=np.float32)
    tensor = ((array - mean) / std).transpose(2, 0, 1)[None]
    return tensor, resized.size, (left, top)


def _source_box(box, source_size, resized_size, padding):
    width, height = source_size
    resized_width, resized_height = resized_size
    left, top = padding
    values = (
        (box[0] - left) / max(1, resized_width) * width,
        (box[1] - top) / max(1, resized_height) * height,
        (box[2] - left) / max(1, resized_width) * width,
        (box[3] - top) / max(1, resized_height) * height,
    )
    x0, y0, x1, y1 = (
        int(round(np.clip(value, 0, limit)))
        for value, limit in zip(values, (width, height, width, height), strict=True)
    )
    return x0, y0, x1, y1


def _decode_candidates(outputs, source_size, resized_size, padding):
    candidates: list[_Candidate] = []
    for stride in (8, 16, 32):
        logits = outputs[f"scores.stride{stride}"][0, 0]
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
        rows, columns = np.where(probabilities >= DETECTION_THRESHOLD)
        for row, column in zip(rows, columns, strict=True):
            score = float(probabilities[row, column])
            left, top, right, bottom = outputs[f"bboxes.stride{stride}"][
                0, :, row, column
            ]
            point = (float(column * stride), float(row * stride))
            box = (
                point[0] - float(left),
                point[1] - float(top),
                point[0] + float(right),
                point[1] + float(bottom),
            )
            source_box = _source_box(box, source_size, resized_size, padding)
            if _area(source_box) < source_size[0] * source_size[1] * 0.01:
                continue
            candidates.append(
                _Candidate(
                    point,
                    stride,
                    outputs[f"coeffs.stride{stride}"][0, :, row, column].copy(),
                    box,
                    source_box,
                    score,
                )
            )
    candidates.sort(key=lambda candidate: candidate.score, reverse=True)
    # Collapse adjacent feature-map cells for the same instance.  A high
    # threshold deliberately keeps heavily overlapping *different* people.
    distinct: list[_Candidate] = []
    for candidate in candidates:
        if any(_iou(candidate.box_source, old.box_source) >= 0.86 for old in distinct):
            continue
        distinct.append(candidate)
        if len(distinct) >= MAX_CANDIDATES:
            break
    return distinct


def _dynamic_mask(proto: np.ndarray, candidate: _Candidate) -> np.ndarray:
    channels, height, width = proto.shape
    y_grid, x_grid = np.mgrid[:height, :width].astype(np.float32)
    relative = np.stack(
        (
            (candidate.point[0] - x_grid * 8) / (candidate.stride * 8),
            (candidate.point[1] - y_grid * 8) / (candidate.stride * 8),
        )
    )
    features = np.concatenate((relative, proto), axis=0).reshape(channels + 2, -1)
    kernel = candidate.kernel.astype(np.float32, copy=False)
    sizes = (8 * (channels + 2), 64, 8, 8, 8, 1)
    if sum(sizes) != kernel.size:
        raise ValueError(f"动漫实例掩码参数形状异常: {kernel.shape}")
    parts = np.split(kernel, np.cumsum(sizes)[:-1])
    first = parts[0].reshape(8, channels + 2) @ features + parts[3][:, None]
    second = parts[1].reshape(8, 8) @ np.maximum(first, 0) + parts[4][:, None]
    logits = parts[2].reshape(1, 8) @ np.maximum(second, 0) + parts[5][:, None]
    return (1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))).reshape(
        height, width
    )


def _restore_mask(probability, source_size, resized_size, padding):
    canvas = Image.fromarray(np.uint8(np.clip(probability, 0, 1) * 255)).resize(
        (MASK_INPUT_SIZE, MASK_INPUT_SIZE), Image.Resampling.BILINEAR
    )
    left, top = padding
    resized_width, resized_height = resized_size
    content = canvas.crop((left, top, left + resized_width, top + resized_height))
    restored = content.resize(source_size, Image.Resampling.BILINEAR)
    return np.asarray(restored) >= round(MASK_THRESHOLD * 255)


def _resize_pad(image: Image.Image, size: int, fill):
    scale = size / max(image.size)
    resized = image.resize(
        (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
        Image.Resampling.BILINEAR,
    )
    # Match the training/inference utility: padding is right/bottom only.
    left = top = 0
    canvas = Image.new(image.mode, (size, size), fill)
    canvas.paste(resized, (left, top))
    return canvas, resized.size, (left, top)


def _mask_box(mask: np.ndarray):
    rows, columns = np.where(mask)
    if not len(columns):
        return None
    return (
        int(columns.min()),
        int(rows.min()),
        int(columns.max()) + 1,
        int(rows.max()) + 1,
    )


def _face_mask_coverage(mask: np.ndarray, face: FaceBox) -> float:
    x0, y0, x1, y1 = face.xyxy
    area = max(1, (x1 - x0) * (y1 - y0))
    return float(mask[y0:y1, x0:x1].sum()) / area


def _outline(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    result = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(contour) < mask.size * 0.0003:
            continue
        perimeter = cv2.arcLength(contour, True)
        simplified = cv2.approxPolyDP(contour, max(1.0, perimeter * 0.0015), True)
        points = [(int(point[0][0]), int(point[0][1])) for point in simplified]
        if len(points) >= 3:
            result.append(points)
        if sum(len(item) for item in result) >= 900:
            break
    return result


def _remove_tiny_components(mask: np.ndarray) -> np.ndarray:
    """Discard specks before deriving a crop; they must not enlarge its bounds."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if count <= 1:
        return mask
    minimum = max(16, round(mask.size * 0.0003))
    keep = np.zeros(mask.shape, dtype=bool)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= minimum:
            keep |= labels == label
    return keep


def _pad_box(box, size, ratio=0.04):
    x0, y0, x1, y1 = box
    width, height = x1 - x0, y1 - y0
    return (
        max(0, round(x0 - width * ratio)),
        max(0, round(y0 - height * ratio)),
        min(size[0], round(x1 + width * ratio)),
        min(size[1], round(y1 + height * ratio)),
    )


def _masked_view(image: Image.Image, mask: np.ndarray, box):
    crop = np.asarray(image.crop(box)).copy()
    local_mask = mask[box[1] : box[3], box[0] : box[2]]
    crop[~local_mask] = 255
    return Image.fromarray(crop, "RGB")


class AnimeInstanceSegmenter:
    """Decode all RTMDet-Ins candidates, not only the global best instance."""

    def __init__(self) -> None:
        options = ort.SessionOptions()
        options.log_severity_level = 3
        options.intra_op_num_threads = 4
        options.enable_mem_pattern = False
        self.session = ort.InferenceSession(
            str(INSTANCE_MODEL.path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.refiner = ort.InferenceSession(
            str(INSTANCE_REFINER.path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.refiner_input = self.refiner.get_inputs()[0].name
        self.refiner_output = self.refiner.get_outputs()[0].name
        self.foreground = ort.InferenceSession(
            str(FOREGROUND_MODEL.path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.foreground_input = self.foreground.get_inputs()[0].name
        self.foreground_output = self.foreground.get_outputs()[0].name
        self._people_detector: AnimeBoxDetector | None = None
        logger.info(f"动漫人物实例分割已加载: CPU, {INSTANCE_VERSION}")

    def _refine(self, image: Image.Image, coarse: np.ndarray) -> np.ndarray:
        size = int(self.refiner.get_inputs()[0].shape[2])
        image_pad, resized_size, padding = _resize_pad(image, size, (0, 0, 0))
        mask_image = Image.fromarray(coarse.astype(np.uint8) * 255, "L")
        mask_pad, _, _ = _resize_pad(mask_image, size, 0)
        rgb = np.asarray(image_pad, dtype=np.float32).transpose(2, 0, 1) / 255.0
        mask = np.asarray(mask_pad, dtype=np.float32)[None] / 255.0
        tensor = np.concatenate((rgb, mask), axis=0)[None]
        logits = self.refiner.run(
            [self.refiner_output], {self.refiner_input: tensor}
        )[0][0, 0]
        probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
        left, top = padding
        width, height = resized_size
        cropped = probability[top : top + height, left : left + width]
        restored = Image.fromarray(np.float32(cropped), "F").resize(
            image.size, Image.Resampling.BILINEAR
        )
        return np.asarray(restored).copy()

    def _foreground_probability(self, image: Image.Image) -> np.ndarray:
        """Anime semantic foreground removes scenery from instance proposals."""
        size = int(self.foreground.get_inputs()[0].shape[2])
        scale = size / max(image.size)
        resized = image.resize(
            (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
            Image.Resampling.BILINEAR,
        )
        left = (size - resized.width) // 2
        top = (size - resized.height) // 2
        canvas = Image.new("RGB", (size, size), (0, 0, 0))
        canvas.paste(resized, (left, top))
        tensor = (
            np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
        )
        probability = self.foreground.run(
            [self.foreground_output], {self.foreground_input: tensor}
        )[0][0, 0]
        probability = probability[
            top : top + resized.height, left : left + resized.width
        ]
        restored = Image.fromarray(np.float32(probability), "F").resize(
            image.size, Image.Resampling.BILINEAR
        )
        return np.asarray(restored)

    def segment(
        self,
        image: Image.Image,
        faces: list[FaceBox],
        people: list[PersonBox],
        *,
        return_masks: bool = False,
    ) -> dict[int, InferenceRegion]:
        """Segment every face in its own upright frame, adjudicate globally.

        Faces carry the rotation at which the detector saw them upright.  For
        each rotation group the whole pipeline (person detection, RTMDet-Ins,
        refiner) runs on the derotated image; masks are mapped back to the
        original frame where a single winner-take-all pass keeps pixel
        ownership disjoint across people and orientations.
        """
        if not faces:
            return {}
        foreground = self._foreground_probability(image) >= FOREGROUND_THRESHOLD
        accepted: dict[int, InferenceRegion] = {}
        accepted_masks: list[np.ndarray] = []
        for angle in sorted({face.rotation for face in faces}):
            (
                proposals,
                variant,
                faces_v,
                face_people,
                people_v,
            ) = self._variant_proposals(image, faces, people, angle)
            accepted, accepted_masks = self._adjudicate(
                image, faces, proposals, foreground, accepted, accepted_masks,
                return_masks,
            )
            # Crowded scenes can decode two touching people as ONE merged
            # instance; the loser of winner-take-all then has no usable
            # pixels.  Re-running inside that person's own proposal crop
            # separates them.
            matched_here = [
                index
                for index, face in enumerate(faces)
                if face.rotation == angle
                and index in face_people
                and index not in accepted
            ]
            for face_index in matched_here[:3]:
                rescue = self._rescue_proposal(
                    image, faces, variant, faces_v, people_v, face_people,
                    face_index, angle,
                )
                if rescue is not None:
                    accepted, accepted_masks = self._adjudicate(
                        image, faces, [rescue], foreground, accepted,
                        accepted_masks, return_masks,
                    )
        # A rotated face can still segment better in the upright frame than in
        # its own derotated variant (merged neighbours differ per frame), so
        # leftovers always get one upright sweep before falling back.
        leftovers = [
            index for index in range(len(faces)) if index not in accepted
        ]
        if leftovers:
            (
                proposals,
                variant,
                faces_v,
                face_people,
                people_v,
            ) = self._variant_proposals(
                image, faces, people, 0, force_targets=leftovers
            )
            accepted, accepted_masks = self._adjudicate(
                image, faces, proposals, foreground, accepted, accepted_masks,
                return_masks,
            )
            matched_upright = [
                index
                for index in leftovers
                if index in face_people and index not in accepted
            ]
            for face_index in matched_upright[:3]:
                rescue = self._rescue_proposal(
                    image, faces, variant, faces_v, people_v, face_people,
                    face_index, 0,
                )
                if rescue is not None:
                    accepted, accepted_masks = self._adjudicate(
                        image, faces, [rescue], foreground, accepted,
                        accepted_masks, return_masks,
                    )
        return accepted
    def _person_boxes_for_variant(self, variant: Image.Image) -> list[PersonBox]:
        """Upright person detection inside a derotated variant (CPU only)."""
        if self._people_detector is None:
            self._people_detector = AnimeBoxDetector(
                PERSON_MODEL.path, 0.0, 0.324, 0.003, cpu_only=True
            )
        return self._people_detector.detect(variant)

    def _variant_proposals(
        self,
        image: Image.Image,
        faces: list[FaceBox],
        people: list[PersonBox],
        angle: int,
        force_targets: list[int] | None = None,
    ):
        variant = image if angle == 0 else image.rotate(angle, expand=True)
        faces_v = [
            replace(
                face, xyxy=unrotate_box(face.xyxy, (-angle) % 360, *variant.size)
            )
            for face in faces
        ]
        people_v = people if angle == 0 else self._person_boxes_for_variant(variant)
        people_v = list(people_v)
        face_people = match_faces_to_people(faces_v, people_v)
        # Heavily occluded pile poses can hide a body from the person detector
        # entirely.  A proportion-expanded box around the face still gives the
        # instance assignment a target; the acceptance gates keep judging the
        # resulting mask on its own evidence.
        wanted = (
            set(range(len(faces)))
            if force_targets is not None
            else {
                index for index, face in enumerate(faces) if face.rotation == angle
            }
        )
        expanded = False
        for index in wanted:
            if index in face_people:
                continue
            fx0, fy0, fx1, fy1 = faces_v[index].xyxy
            face_width = max(1, fx1 - fx0)
            face_height = max(1, fy1 - fy0)
            people_v.append(
                PersonBox(
                    (
                        max(0, round(fx0 - face_width * 1.2)),
                        max(0, round(fy0 - face_height * 0.8)),
                        min(variant.width, round(fx1 + face_width * 1.2)),
                        min(variant.height, round(fy1 + face_height * 3.2)),
                    ),
                    0.0,
                    True,
                )
            )
            expanded = True
        if expanded:
            face_people = match_faces_to_people(faces_v, people_v)
        ordered = (
            force_targets if force_targets is not None else sorted(wanted)
        )
        targets = [index for index in ordered if index in face_people]
        proposals = self._build_proposals(
            variant, faces_v, people_v, face_people, targets
        )
        for proposal in proposals:
            proposal.angle = angle
            if angle:
                proposal.proposal_box = unrotate_box(
                    proposal.proposal_box, angle, *image.size
                )
        return proposals, variant, faces_v, face_people, people_v

    def _build_proposals(
        self, view_image, faces_list, people_list, face_people, targets
    ):
        """Letterbox → candidates → assignment → refined maps; gates come later."""
        proposals: list[_Proposal] = []
        if not targets:
            return proposals
        tensor, resized_size, padding = _letterbox(view_image)
        raw = self.session.run(self.output_names, {self.input_name: tensor})
        outputs = dict(zip(self.output_names, raw, strict=True))
        candidates = _decode_candidates(
            outputs, view_image.size, resized_size, padding
        )
        if not candidates:
            return proposals
        scores = np.zeros((len(targets), len(candidates)), dtype=np.float32)
        for row, face_index in enumerate(targets):
            proposal = people_list[face_people[face_index]]
            for column, candidate in enumerate(candidates):
                overlap = _iou(proposal.xyxy, candidate.box_source)
                if overlap >= 0.12:
                    scores[row, column] = overlap + candidate.score * 0.35
        assignments = _assign(scores, 0.25)
        proto = outputs["mask_proto"][0].astype(np.float32, copy=False)
        for row, candidate_index in assignments.items():
            face_index = targets[row]
            candidate = candidates[candidate_index]
            probability = _dynamic_mask(proto, candidate)
            coarse = _restore_mask(
                probability, view_image.size, resized_size, padding
            )
            refined = self._refine(view_image, coarse)
            # Detector logits outside the candidate box are not part of the
            # instance.  Cropping them prevents low-frequency mask spill.
            guard = np.zeros(refined.shape, dtype=bool)
            gx0, gy0, gx1, gy1 = _pad_box(candidate.box_source, view_image.size, 0.03)
            guard[gy0:gy1, gx0:gx1] = True
            refined[~guard] = 0.0
            proposals.append(
                _Proposal(
                    face_index=face_index,
                    score=candidate.score,
                    proposal_box=people_list[face_people[face_index]].xyxy,
                    view_image=view_image,
                    refined=refined,
                )
            )
        return proposals

    def _rescue_proposal(
        self,
        image: Image.Image,
        faces: list[FaceBox],
        variant: Image.Image,
        faces_v: list[FaceBox],
        people_v: list[PersonBox],
        face_people: dict[int, int],
        face_index: int,
        angle: int,
    ) -> _Proposal | None:
        px0, py0, px1, py1 = people_v[face_people[face_index]].xyxy
        pad_x = max(8, int((px1 - px0) * 0.08))
        pad_y = max(8, int((py1 - py0) * 0.08))
        ox, oy = max(0, px0 - pad_x), max(0, py0 - pad_y)
        cx1, cy1 = min(variant.width, px1 + pad_x), min(variant.height, py1 + pad_y)
        if cx1 - ox < 160 or cy1 - oy < 160:
            return None
        crop = variant.crop((ox, oy, cx1, cy1))
        local_faces, origin = [], []
        for index, face_v in enumerate(faces_v):
            clipped = (
                max(0, face_v.xyxy[0] - ox),
                max(0, face_v.xyxy[1] - oy),
                min(crop.width, face_v.xyxy[2] - ox),
                min(crop.height, face_v.xyxy[3] - oy),
            )
            if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
                continue
            if _area(clipped) < _area(face_v.xyxy) * 0.5:
                continue
            local_faces.append(replace(faces[index], xyxy=clipped))
            origin.append(index)
        if face_index not in origin:
            return None
        local_people = []
        for person in people_v:
            clipped = (
                max(0, person.xyxy[0] - ox),
                max(0, person.xyxy[1] - oy),
                min(crop.width, person.xyxy[2] - ox),
                min(crop.height, person.xyxy[3] - oy),
            )
            if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
                continue
            if _area(clipped) < person.area * 0.25:
                continue
            local_people.append(replace(person, xyxy=clipped))
        local_face_people = match_faces_to_people(local_faces, local_people)
        local_index = origin.index(face_index)
        if local_index not in local_face_people:
            return None
        # Assign EVERY locally matched face, not only the rescue target: the
        # one-to-one assignment is what separates two touching people onto
        # different instances inside the crop.
        local_matched = [
            index
            for index in range(len(local_faces))
            if index in local_face_people
        ]
        built = self._build_proposals(
            crop, local_faces, local_people, local_face_people, local_matched
        )
        if len(built) > 1:
            # Local winner-take-all: two touching people inside the crop must
            # not keep each other's pixels, exactly as in the global pass.
            local_winners = np.stack([item.refined for item in built]).argmax(
                axis=0
            )
            for position, item in enumerate(built):
                item.refined = np.where(
                    local_winners == position, item.refined, 0.0
                ).astype(np.float32)
        rescue = next(
            (item for item in built if item.face_index == local_index), None
        )
        if rescue is None:
            return None
        proposal = rescue
        proposal.face_index = face_index
        proposal.angle = angle
        box_v = (
            proposal.proposal_box[0] + ox,
            proposal.proposal_box[1] + oy,
            proposal.proposal_box[2] + ox,
            proposal.proposal_box[3] + oy,
        )
        proposal.proposal_box = (
            unrotate_box(box_v, angle, *image.size)
            if angle
            else tuple(int(value) for value in box_v)
        )
        # refined stays in the CROP frame; _adjudicate embeds it into the
        # variant canvas at this offset before rotating to the original frame.
        proposal.frame_offset = (ox, oy)
        return proposal

    def _mask_in_view_frame(
        self, mask: np.ndarray, proposal: _Proposal
    ) -> np.ndarray:
        canvas = Image.fromarray(mask.astype(np.uint8) * 255, "L")
        if proposal.angle:
            canvas = canvas.rotate(
                proposal.angle, expand=True, resample=Image.Resampling.NEAREST
            )
        restored = np.asarray(canvas) > 127
        ox, oy = proposal.frame_offset
        return restored[
            oy : oy + proposal.view_image.height,
            ox : ox + proposal.view_image.width,
        ]

    def _adjudicate(
        self,
        image: Image.Image,
        faces: list[FaceBox],
        proposals: list[_Proposal],
        foreground: np.ndarray,
        accepted: dict[int, InferenceRegion],
        accepted_masks: list[np.ndarray],
        return_masks: bool,
    ):
        """Winner-take-all in the ORIGINAL frame; gates decide acceptance."""
        if not proposals:
            return accepted, accepted_masks
        owned = np.zeros((image.height, image.width), dtype=bool)
        for mask in accepted_masks:
            owned |= mask
        maps = []
        for proposal in proposals:
            ox, oy = proposal.frame_offset
            refined = proposal.refined
            if proposal.angle in (90, 270):
                variant_width, variant_height = image.height, image.width
            else:
                variant_width, variant_height = image.width, image.height
            variant_canvas = np.zeros(
                (variant_height, variant_width), dtype=np.float32
            )
            variant_canvas[
                oy : oy + refined.shape[0], ox : ox + refined.shape[1]
            ] = refined
            if proposal.angle:
                # Variant frame -> original frame: PIL rotate(-angle) with
                # expand swaps the axes back exactly.
                canvas = np.asarray(
                    Image.fromarray(variant_canvas, "F").rotate(
                        -proposal.angle,
                        expand=True,
                        resample=Image.Resampling.BILINEAR,
                    )
                )
            else:
                canvas = variant_canvas
            maps.append(canvas)
        probabilities = np.stack(maps)
        winners = probabilities.argmax(axis=0)
        for index, proposal in enumerate(proposals):
            mask = (
                foreground
                & (winners == index)
                & (probabilities[index] >= REFINE_THRESHOLD)
                & ~owned
            )
            # Smooth only pixel-scale seams; never merge separated people.
            mask = cv2.morphologyEx(
                mask.astype(np.uint8),
                cv2.MORPH_CLOSE,
                np.ones((3, 3), dtype=np.uint8),
            ).astype(bool)
            mask = _remove_tiny_components(mask)
            bounds = _mask_box(mask)
            if bounds is None:
                continue
            face = faces[proposal.face_index]
            own_coverage = _face_mask_coverage(mask, face)
            neighbor_coverage = max(
                (
                    _face_mask_coverage(mask, other)
                    for other_index, other in enumerate(faces)
                    if other_index != proposal.face_index
                ),
                default=0.0,
            )
            # Orientation-agnostic body extent: a lying figure extends along
            # the same axis as its lying face, so compare the long spans.
            face_span = max(
                face.xyxy[2] - face.xyxy[0], face.xyxy[3] - face.xyxy[1]
            )
            mask_span = max(bounds[2] - bounds[0], bounds[3] - bounds[1])
            if (
                own_coverage < 0.34
                or neighbor_coverage > 0.28
                or mask.sum()
                < max(
                    _area(face.xyxy) * 2.5,
                    _area(proposal.proposal_box) * 0.12,
                )
                or mask_span < face_span * 1.8
            ):
                continue
            view_mask = self._mask_in_view_frame(mask, proposal)
            view_bounds = _mask_box(view_mask)
            if view_bounds is None:
                continue
            crop_box = _pad_box(bounds, image.size)
            accepted[proposal.face_index] = InferenceRegion(
                _masked_view(
                    proposal.view_image,
                    view_mask,
                    _pad_box(view_bounds, proposal.view_image.size),
                ),
                crop_box,
                "instance_mask",
                _outline(mask),
                confidence=proposal.score,
                proposal_box=tuple(int(value) for value in proposal.proposal_box),
                mask=mask.copy() if return_masks else None,
            )
            accepted_masks.append(mask)
        return accepted, accepted_masks


def masked_upper_view(
    image: Image.Image, face: FaceBox, region: InferenceRegion
) -> tuple[Image.Image, tuple[int, int, int, int]] | None:
    """A mask-clean head-and-shoulders retry view around one face.

    Interlocked group poses (joint heart gestures, linked arms) hand a
    companion's limbs to this person's full-body mask, which the taggers then
    read as "multiple people".  A head-and-shoulders band keeps the person's
    own hair, face and collar while the mask still whites out every pixel
    that does not belong to them.
    """
    if region.mask is None:
        return None
    fx0, fy0, fx1, fy1 = face.xyxy
    face_width = max(1, fx1 - fx0)
    face_height = max(1, fy1 - fy0)
    pad_x = max(64, round(face_width * 0.9))
    box = (
        max(region.box[0], round(fx0 - pad_x)),
        region.box[1],
        min(region.box[2], round(fx1 + pad_x)),
        min(region.box[3], round(fy1 + face_height * 1.15)),
    )
    if box[2] - box[0] < 64 or box[3] - box[1] < 64:
        return None
    return _masked_view(image, region.mask, box), box


def masked_face_view(
    image: Image.Image, face: FaceBox, region: InferenceRegion
) -> tuple[Image.Image, tuple[int, int, int, int]] | None:
    """A mask-clean head-band retry view around one face.

    Group poses where companions rest their hands on this person's shoulders
    leave foreign limbs inside even a head-and-shoulders band.  The face band
    (hair top to chin) removes them while keeping the strongest gender cues:
    hair, face and collar.
    """
    if region.mask is None:
        return None
    fx0, fy0, fx1, fy1 = face.xyxy
    face_width = max(1, fx1 - fx0)
    face_height = max(1, fy1 - fy0)
    pad_x = max(48, round(face_width * 0.5))
    box = (
        max(region.box[0], round(fx0 - pad_x)),
        max(region.box[1], round(fy0 - face_height * 0.55)),
        min(region.box[2], round(fx1 + pad_x)),
        min(region.box[3], round(fy1 + face_height * 0.35)),
    )
    if box[2] - box[0] < 48 or box[3] - box[1] < 48:
        return None
    return _masked_view(image, region.mask, box), box


__all__ = [
    "AnimeInstanceSegmenter",
    "InferenceRegion",
    "masked_face_view",
    "masked_upper_view",
    "match_faces_to_people",
]

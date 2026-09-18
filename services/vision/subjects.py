from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from .assets import PERSON_MODEL
from .runtime import create_session


@dataclass(frozen=True, slots=True)
class PersonBox:
    xyxy: tuple[int, int, int, int]
    confidence: float
    small_verified: bool = False

    @property
    def area(self) -> int:
        x0, y0, x1, y1 = self.xyxy
        return max(0, x1 - x0) * max(0, y1 - y0)


def intersection(left: PersonBox, right: PersonBox) -> int:
    a, b = left.xyxy, right.xyxy
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )


def select_subject(
    boxes: list[PersonBox], width: int, height: int
) -> tuple[PersonBox | None, str]:
    """Area + centrality is a heuristic, not a semantic identity guarantee."""
    if not boxes:
        return None, "no_person"

    def salience(box: PersonBox) -> float:
        x0, y0, x1, y1 = box.xyxy
        distance = abs((x0 + x1) / (2 * width) - 0.5)
        # Confidence measures box reliability, not narrative importance. Do not
        # let a small, clearly drawn companion outweigh a large occluded lead.
        return box.area * (1.0 - 0.3 * distance) * max(0.5, box.confidence)

    ranked = sorted(boxes, key=salience, reverse=True)
    main = ranked[0]
    if (
        main.area < width * height * 0.08 and not main.small_verified
    ) or main.confidence < 0.4:
        return None, "weak_detection"
    if main.confidence < 0.5 and (
        main.area < width * height * 0.12
        or (len(ranked) > 1 and salience(main) < 1.8 * salience(ranked[1]))
    ):
        return None, "weak_detection"
    if len(ranked) > 1:
        if salience(main) < 1.7 * salience(ranked[1]):
            return None, "ambiguous_dominance"
        for other in ranked[1:]:
            if intersection(main, other) / max(1, other.area) > 0.15:
                # Keep the spatial candidate, but never trust its full crop.
                return main, "overlapping_people"
    return main, "tentative_detection" if main.confidence < 0.5 else "candidate"


def subject_focus_box(
    main: PersonBox, boxes: list[PersonBox]
) -> tuple[int, int, int, int] | None:
    """A conservative upper-body view, ending before an overlapping companion.

    This is not segmentation: upper companions and too-small remaining regions
    cannot be isolated safely, and must keep their unknown status.
    """
    mx0, y0, mx1, y1 = main.xyxy
    height, width = y1 - y0, mx1 - mx0
    # Preserve head AND torso. Head-only crops can change perceived gender and
    # should not be used to certify body features. Exclude peripheral shoulders
    # where smaller companions are often occluded by the main person's box.
    x0, x1 = mx0 + int(width * 0.15), mx1 - int(width * 0.15)
    bottom = y0 + int(height * 0.65)
    for other in boxes:
        region = PersonBox((x0, y0, x1, bottom), 1.0)
        if other is main or other == main or not intersection(region, other):
            continue
        # Trim a side sliver before discarding the entire torso below it.
        ox0, _, ox1, _ = other.xyxy
        overlap_width = min(x1, ox1) - max(x0, ox0)
        if overlap_width <= (x1 - x0) * 0.35:
            margin = max(2, int(width * 0.02))
            if ox0 > (x0 + x1) / 2:
                x1 = min(x1, ox0 - margin)
                continue
            if ox1 < (x0 + x1) / 2:
                x0 = max(x0, ox1 + margin)
                continue
        if other.xyxy[1] <= y0 + height * 0.25:
            return None
        bottom = min(bottom, other.xyxy[1] - max(2, int(height * 0.02)))
    if bottom - y0 < max(64, height * 0.5) or x1 - x0 < max(64, width * 0.45):
        return None
    return x0, y0, x1, bottom


def refine_subject_box(main: PersonBox, local_boxes: list[PersonBox]) -> PersonBox:
    """Accept only a higher-confidence, substantially overlapping re-detection.

    Coordinates from the detector's main-crop pass are translated here. The
    size/IoU gates prevent switching to a small, more easily detected companion.
    """
    x0, y0, _, _ = main.xyxy
    candidates = []
    for box in local_boxes:
        mapped = PersonBox(
            (box.xyxy[0] + x0, box.xyxy[1] + y0, box.xyxy[2] + x0, box.xyxy[3] + y0),
            box.confidence,
            main.small_verified,
        )
        overlap = intersection(main, mapped)
        iou = overlap / max(1, main.area + mapped.area - overlap)
        if (
            iou >= 0.55
            and mapped.area >= main.area * 0.6
            and mapped.confidence >= max(0.55, main.confidence + 0.05)
        ):
            candidates.append(mapped)
    return max(candidates, key=lambda box: box.confidence, default=main)


class AnimeBoxDetector:
    def __init__(
        self,
        path: Path,
        memory_gib: float,
        confidence: float,
        min_area: float,
        *,
        cpu_only: bool = False,
    ) -> None:
        if cpu_only:
            # Rotated-variant detection must never grow the CUDA arenas that
            # the long-lived tagging sessions already reserve.
            options = ort.SessionOptions()
            options.log_severity_level = 3
            options.intra_op_num_threads = 4
            options.enable_mem_pattern = False
            self.session = ort.InferenceSession(
                str(path), sess_options=options, providers=["CPUExecutionProvider"]
            )
        else:
            self.session = create_session(path, memory_gib)
        self.confidence_threshold = confidence
        self.min_area = min_area
        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        metadata = self.session.get_modelmeta().custom_metadata_map
        dimensions = json.loads(metadata.get("imgsz", "[640, 640]"))
        self.height, self.width = dimensions
        if isinstance(model_input.shape[2], int):
            self.height = model_input.shape[2]
        if isinstance(model_input.shape[3], int):
            self.width = model_input.shape[3]

    def detect(self, image: Image.Image) -> list[PersonBox]:
        # Follow the upstream deepghs fixed-size RGB preprocessing.
        resized = image.resize((self.width, self.height), Image.Resampling.BICUBIC)
        array = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1) / 255.0
        output = self.session.run(None, {self.input_name: array[None]})[0][0]
        if output.ndim != 2 or output.shape[0] != 5:
            raise ValueError(f"人物检测模型输出形状异常: {output.shape}")
        candidates = output.T
        candidates = candidates[candidates[:, 4] >= self.confidence_threshold]
        candidates = candidates[np.argsort(candidates[:, 4])[::-1]][:300]
        boxes: list[PersonBox] = []
        for cx, cy, width, height, score in candidates:
            coords = (
                (cx - width / 2) / self.width * image.width,
                (cy - height / 2) / self.height * image.height,
                (cx + width / 2) / self.width * image.width,
                (cy + height / 2) / self.height * image.height,
            )
            clipped = tuple(
                int(round(np.clip(value, 0, limit)))
                for value, limit in zip(
                    coords, (image.width, image.height) * 2, strict=True
                )
            )
            box = PersonBox(clipped, float(score))
            if box.area < image.width * image.height * self.min_area:
                continue
            if any(
                intersection(box, old)
                / max(1, box.area + old.area - intersection(box, old))
                > 0.5
                for old in boxes
            ):
                continue
            boxes.append(box)
        return boxes


class AnimePersonDetector(AnimeBoxDetector):
    def __init__(self) -> None:
        super().__init__(PERSON_MODEL.path, 0.5, 0.324, 0.003)

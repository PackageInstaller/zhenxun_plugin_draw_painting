"""Explainable, CPU-only visual ranking for detected anime people.

This module deliberately does not decide identity, gender, or whether a crop is
safe for tagging.  It only estimates which detected person is the visual lead.
The result is made exclusively from JSON-compatible values so it can be kept in
``TagPrediction.analysis`` and audited later.

The weights are conservative starting values, not a learned claim about artistic
intent.  In particular, a close result abstains instead of assigning a narrative
role to one member of a group.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import numpy as np
from PIL import Image

from .subjects import PersonBox

SALIENCE_VERSION = "subject-salience-v3"
MAX_ANALYSIS_SIDE = 512
# A standing figure fills roughly half of its bounding box.  Dividing the
# visible-body mask area by this factor reconstructs an occupancy estimate
# that merged or padding-heavy detection boxes cannot fake.
EXPECTED_BODY_FILL = 0.55

_WEIGHTS = {
    "scale": 0.22,
    "composition": 0.14,
    "face_scale": 0.08,
    "local_luma": 0.05,
    "local_chroma": 0.05,
    "global_luma": 0.03,
    "global_chroma": 0.03,
    "focus": 0.09,
    "modeling": 0.04,
    "exposure": 0.06,
    "saturation": 0.04,
    "emphasis": 0.12,
    "frontness": 0.07,
    "detector": 0.05,
    "completeness": 0.02,
}
_APPEARANCE_KEYS = (
    "local_luma",
    "local_chroma",
    "global_luma",
    "global_chroma",
    "focus",
    "modeling",
    "exposure",
    "saturation",
)
# Artistic emphasis: the cue family in which illustrations single out their
# lead - sharp contours, wide tonal modelling, brighter exposure and stronger
# saturation than the companions.  Grounded in center-surround saliency
# (Itti/Koch/Niebur, PAMI 1998) and aesthetic-assessment work where the
# brightly lit, colour-vivid, detail-rich subject region marks the intended
# focus.  A decisive lead may promote an emphasis leader over a larger but
# flat, dark companion that only wins on box size and detector confidence.
_EMPHASIS_KEYS = ("focus", "modeling", "exposure", "saturation")
_EMPHASIS_GAP = 0.12
_EMPHASIS_PROMOTION_GAP = 0.18


def _clip(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _content_box(image: Image.Image) -> tuple[int, int, int, int]:
    candidate = image.info.get("painting_content_box")
    if isinstance(candidate, tuple | list) and len(candidate) == 4:
        try:
            x0, y0, x1, y1 = (int(value) for value in candidate)
        except (TypeError, ValueError):
            pass
        else:
            x0, x1 = max(0, x0), min(image.width, x1)
            y0, y1 = max(0, y0), min(image.height, y1)
            if x1 > x0 and y1 > y0:
                return x0, y0, x1, y1
    if "A" in image.getbands():
        alpha = image.getchannel("A").getbbox()
        if alpha:
            return alpha
    return 0, 0, image.width, image.height


def _prepare_image(image: Image.Image):
    content = _content_box(image)
    source = image.crop(content).convert("RGBA")
    background = Image.new("RGBA", source.size, (255, 255, 255, 255))
    background.alpha_composite(source)
    rgb = background.convert("RGB")
    scale = min(1.0, MAX_ANALYSIS_SIDE / max(1, *rgb.size))
    target = (
        max(1, round(rgb.width * scale)),
        max(1, round(rgb.height * scale)),
    )
    if target != rgb.size:
        rgb = rgb.resize(target, Image.Resampling.LANCZOS)
    ycbcr = np.asarray(rgb.convert("YCbCr"), dtype=np.uint8)
    return ycbcr, content


def _scaled_box(box, content, size) -> tuple[int, int, int, int]:
    left, top, right, bottom = content
    width, height = right - left, bottom - top
    sx, sy = size[0] / max(1, width), size[1] / max(1, height)
    x0 = int(round((box[0] - left) * sx))
    y0 = int(round((box[1] - top) * sy))
    x1 = int(round((box[2] - left) * sx))
    y1 = int(round((box[3] - top) * sy))
    x0, x1 = max(0, x0), min(size[0], x1)
    y0, y1 = max(0, y0), min(size[1], y1)
    return x0, y0, x1, y1


def _area(box) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _intersection(left, right) -> int:
    return max(0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0, min(left[3], right[3]) - max(left[1], right[1])
    )


def _rectangle_mask(box, shape) -> np.ndarray:
    result = np.zeros(shape, dtype=bool)
    if _area(box):
        result[box[1] : box[3], box[0] : box[2]] = True
    return result


def _supplied_mask(masks, index, image_size, content, target_size):
    if masks is None:
        return None
    value = masks.get(index) if isinstance(masks, Mapping) else (
        masks[index] if index < len(masks) else None
    )
    if value is None:
        return None
    array = np.asarray(value)
    if array.ndim != 2:
        return None
    if array.shape == (image_size[1], image_size[0]):
        x0, y0, x1, y1 = content
        array = array[y0:y1, x0:x1]
    elif array.shape != (content[3] - content[1], content[2] - content[0]):
        return None
    mask = Image.fromarray(np.uint8(array > 0) * 255, "L")
    if mask.size != target_size:
        mask = mask.resize(target_size, Image.Resampling.NEAREST)
    result = np.asarray(mask) > 0
    return result if result.any() else None


def _js(left: np.ndarray, right: np.ndarray) -> float:
    left = left.astype(np.float64, copy=False)
    right = right.astype(np.float64, copy=False)
    if not left.sum() or not right.sum():
        return 0.5
    left /= left.sum()
    right /= right.sum()
    middle = (left + right) / 2
    left_mask, right_mask = left > 0, right > 0
    value = 0.5 * np.sum(left[left_mask] * np.log(left[left_mask] / middle[left_mask]))
    value += 0.5 * np.sum(
        right[right_mask] * np.log(right[right_mask] / middle[right_mask])
    )
    return _clip(value / math.log(2))


def _luma_hist(values: np.ndarray) -> np.ndarray:
    return np.histogram(values, bins=16, range=(0, 256))[0]


def _chroma_hist(cb: np.ndarray, cr: np.ndarray) -> np.ndarray:
    return np.histogram2d(cb, cr, bins=12, range=((0, 256), (0, 256)))[0]


def _sigmoid(value: float) -> float:
    if value >= 0:
        term = math.exp(-min(value, 60))
        return 1 / (1 + term)
    term = math.exp(max(value, -60))
    return term / (1 + term)


def _effective(value: float, reliability: float) -> float:
    return _clip(0.5 + reliability * (value - 0.5))


def _appearance(ycbcr, masks, exact, boxes, ring_ratio=0.12):
    height, width = ycbcr.shape[:2]
    union = np.zeros((height, width), dtype=bool)
    for mask in masks:
        union |= mask
    background = ~union
    background_valid = int(background.sum()) >= 256
    global_luma = _luma_hist(ycbcr[..., 0][background]) if background_valid else None
    global_chroma = (
        _chroma_hist(ycbcr[..., 1][background], ycbcr[..., 2][background])
        if background_valid
        else None
    )
    luminance = ycbcr[..., 0].astype(np.float32)
    gy, gx = np.gradient(luminance / 255.0)
    gradient = np.hypot(gx, gy)
    background_luma = (
        float(luminance[background].mean()) if background_valid else None
    )
    background_chroma = (
        float(
            np.hypot(
                ycbcr[..., 1][background].astype(np.float32) - 128.0,
                ycbcr[..., 2][background].astype(np.float32) - 128.0,
            ).mean()
        )
        if background_valid
        else None
    )
    raw = []
    for index, (mask, box) in enumerate(zip(masks, boxes, strict=True)):
        other_union = np.zeros_like(union)
        for other_index, other in enumerate(masks):
            if other_index != index:
                other_union |= other
        own_area = max(1, int(mask.sum()))
        overlap = int(np.logical_and(mask, other_union).sum()) / own_area
        safe = mask & ~other_union
        safe_fraction = int(safe.sum()) / own_area
        if int(safe.sum()) < max(64, round(own_area * 0.15)):
            safe = mask
            safe_fraction *= 0.2
        base_reliability = 0.95 if exact[index] else 0.55
        reliability = _clip(base_reliability * (1.0 - min(overlap, 0.9)))

        x0, y0, x1, y1 = box
        padding = max(2, round(max(x1 - x0, y1 - y0) * ring_ratio))
        ring_box = (
            max(0, x0 - padding),
            max(0, y0 - padding),
            min(width, x1 + padding),
            min(height, y1 + padding),
        )
        ring = _rectangle_mask(ring_box, (height, width)) & ~union
        ring_valid = int(ring.sum()) >= max(256, round(own_area * 0.05))
        foreground_valid = int(safe.sum()) >= 64

        if foreground_valid and ring_valid:
            local_luma = _js(
                _luma_hist(ycbcr[..., 0][safe]), _luma_hist(ycbcr[..., 0][ring])
            )
            local_chroma = _js(
                _chroma_hist(ycbcr[..., 1][safe], ycbcr[..., 2][safe]),
                _chroma_hist(ycbcr[..., 1][ring], ycbcr[..., 2][ring]),
            )
            ring_energy = float(np.quantile(gradient[ring], 0.75))
        else:
            local_luma = local_chroma = 0.5
            ring_energy = None
        if foreground_valid and background_valid:
            foreground_luma = _luma_hist(ycbcr[..., 0][safe])
            foreground_chroma = _chroma_hist(
                ycbcr[..., 1][safe], ycbcr[..., 2][safe]
            )
            scene_luma = _js(foreground_luma, global_luma)
            scene_chroma = _js(foreground_chroma, global_chroma)
        else:
            scene_luma = scene_chroma = 0.5
        energy = (
            float(np.quantile(gradient[safe], 0.75)) if foreground_valid else 0.0
        )
        # Light modelling (chiaroscuro): lead characters are usually lit with a
        # wider luminance range than supporting cast or background art.  The
        # p90-p10 spread inside the person's own pixels measures that emphasis
        # without confusing it with person-vs-background contrast (local_luma).
        if foreground_valid:
            luma = ycbcr[..., 0][safe].astype(np.float32)
            modeling_raw = _clip(
                (float(np.quantile(luma, 0.9)) - float(np.quantile(luma, 0.1)))
                / 128.0
            )
        else:
            modeling_raw = 0.5
        if foreground_valid:
            person_luma = float(luminance[safe].mean())
            person_chroma = float(
                np.hypot(
                    ycbcr[..., 1][safe].astype(np.float32) - 128.0,
                    ycbcr[..., 2][safe].astype(np.float32) - 128.0,
                ).mean()
            )
        else:
            person_luma = person_chroma = None
        raw.append(
            {
                "local_luma_raw": local_luma,
                "local_chroma_raw": local_chroma,
                "global_luma_raw": scene_luma,
                "global_chroma_raw": scene_chroma,
                "energy": energy,
                "ring_energy": ring_energy,
                "modeling_raw": modeling_raw,
                "person_luma": person_luma,
                "person_chroma": person_chroma,
                "reliability": reliability,
                "completeness": _clip(safe_fraction),
                "local_valid": ring_valid and foreground_valid,
                "global_valid": background_valid and foreground_valid,
                "modeling_valid": foreground_valid,
            }
        )
    maximum_energy = max((item["energy"] for item in raw), default=0.0)
    luma_all = [item["person_luma"] for item in raw]
    chroma_all = [item["person_chroma"] for item in raw]
    for position, item in enumerate(raw):
        relative = (
            math.sqrt(item["energy"] / maximum_energy)
            if maximum_energy > 1e-7
            else 0.5
        )
        ring_energy = item["ring_energy"]
        ring_focus = (
            _sigmoid(
                math.log((item["energy"] + 1e-5) / (ring_energy + 1e-5)) / 0.6
            )
            if ring_energy is not None
            else 0.5
        )
        focus = 0.55 * relative + 0.45 * ring_focus
        reliability = item["reliability"]
        item["local_luma"] = _effective(
            item["local_luma_raw"], reliability if item["local_valid"] else 0
        )
        item["local_chroma"] = _effective(
            item["local_chroma_raw"], reliability if item["local_valid"] else 0
        )
        item["global_luma"] = _effective(
            item["global_luma_raw"], reliability if item["global_valid"] else 0
        )
        item["global_chroma"] = _effective(
            item["global_chroma_raw"], reliability if item["global_valid"] else 0
        )
        # Exposure/saturation are COMPANION-relative: the lead is lit and
        # coloured beyond the other depicted people, not beyond a sky or
        # window that may outshine every character.  A lone person is neutral.
        others_luma = [
            value
            for index, value in enumerate(luma_all)
            if index != position and value is not None
        ]
        others_chroma = [
            value
            for index, value in enumerate(chroma_all)
            if index != position and value is not None
        ]
        if item["person_luma"] is None:
            exposure_raw = saturation_raw = 0.5
        else:
            luma_reference = (
                float(np.median(others_luma))
                if others_luma
                else (
                    background_luma
                    if background_luma is not None
                    else item["person_luma"]
                )
            )
            chroma_reference = (
                float(np.median(others_chroma))
                if others_chroma
                else (
                    background_chroma
                    if background_chroma is not None
                    else item["person_chroma"]
                )
            )
            exposure_raw = _clip(
                0.5 + (item["person_luma"] - luma_reference) / 120.0
            )
            saturation_raw = _clip(
                0.5 + (item["person_chroma"] - chroma_reference) / 60.0
            )
        item["focus"] = _effective(focus, reliability)
        item["modeling"] = _effective(
            item["modeling_raw"], reliability if item["modeling_valid"] else 0
        )
        item["exposure"] = _effective(
            exposure_raw, reliability if item["modeling_valid"] else 0
        )
        item["saturation"] = _effective(
            saturation_raw, reliability if item["modeling_valid"] else 0
        )
    return raw


def _gaussian_distance(point, target, sigma) -> float:
    dx, dy = point[0] - target[0], point[1] - target[1]
    return math.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma))


def _geometry(boxes, confidences, size, effective_areas=None):
    width, height = size
    areas = [max(1, _area(box)) for box in boxes]
    heights = [max(1, box[3] - box[1]) for box in boxes]
    maximum_height = max(heights)
    if effective_areas is None:
        effective_areas = areas
    effective_areas = [
        max(1.0, min(box_area, visible))
        for box_area, visible in zip(areas, effective_areas, strict=True)
    ]
    maximum_effective = max(effective_areas)
    centers = [
        ((box[0] + box[2]) / (2 * width), (box[1] + box[3]) / (2 * height))
        for box in boxes
    ]
    total_area = max(1, sum(areas))
    centroid = (
        sum(point[0] * area for point, area in zip(centers, areas, strict=True))
        / total_area,
        sum(point[1] * area for point, area in zip(centers, areas, strict=True))
        / total_area,
    )
    thirds = ((1 / 3, 1 / 3), (2 / 3, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 2 / 3))
    depth = []
    result = []
    for box, area, box_height, center, confidence, effective in zip(
        boxes, areas, heights, centers, confidences, effective_areas, strict=True
    ):
        absolute = _clip(
            math.sqrt(effective / max(1, width * height)) / math.sqrt(0.35)
        )
        scale = (
            0.45 * math.sqrt(effective / maximum_effective)
            + 0.30 * box_height / maximum_height
            + 0.25 * absolute
        )
        center_score = _gaussian_distance(center, (0.5, 0.5), 0.32)
        group_score = _gaussian_distance(center, centroid, 0.28)
        thirds_score = max(_gaussian_distance(center, point, 0.20) for point in thirds)
        composition = 0.62 * center_score + 0.25 * group_score + 0.13 * thirds_score
        detector = _clip((confidence - 0.324) / (0.85 - 0.324))
        bottom = _clip(box[3] / max(1, height))
        depth.append(
            0.55 * box_height / maximum_height + 0.30 * bottom + 0.15 * detector
        )
        result.append(
            {
                "scale": _clip(scale),
                "composition": _clip(composition),
                "detector": detector,
            }
        )
    for index, box in enumerate(boxes):
        weighted, total = 0.0, 0.0
        for other_index, other in enumerate(boxes):
            if index == other_index:
                continue
            overlap = _intersection(box, other) / max(
                1, min(areas[index], areas[other_index])
            )
            if overlap < 0.10:
                continue
            weighted += overlap * math.tanh((depth[index] - depth[other_index]) / 0.12)
            total += overlap
        result[index]["frontness"] = _clip(
            0.5 + 0.5 * weighted / total if total else 0.5
        )
    return result


def _score(cues) -> float:
    return sum(_WEIGHTS[key] * cues[key] for key in _WEIGHTS)


def _jitter_boxes(boxes, size, factor):
    width, height = size
    result = []
    for x0, y0, x1, y1 in boxes:
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        half_width = max(0.5, (x1 - x0) * factor / 2)
        half_height = max(0.5, (y1 - y0) * factor / 2)
        result.append(
            (
                max(0, round(cx - half_width)),
                max(0, round(cy - half_height)),
                min(width, round(cx + half_width)),
                min(height, round(cy + half_height)),
            )
        )
    return result


def _variant_scores(cues, boxes, confidences, size, effective_areas=None):
    variants = [[_score(item) for item in cues]]
    for factor in (1.03, 0.97):
        geometry = _geometry(
            _jitter_boxes(boxes, size, factor),
            confidences,
            size,
            [value * factor * factor for value in (effective_areas or [])]
            or None,
        )
        variants.append(
            [
                _score(
                    {
                        **item,
                        "scale": update["scale"],
                        "composition": update["composition"],
                        "frontness": update["frontness"],
                    }
                )
                for item, update in zip(cues, geometry, strict=True)
            ]
        )
    for strength in (0.8, 1.2):
        variants.append(
            [
                _score(
                    {
                        **item,
                        **{
                            key: _clip(0.5 + strength * (item[key] - 0.5))
                            for key in _APPEARANCE_KEYS
                        },
                    }
                )
                for item in cues
            ]
        )
    return variants


def _unique_winner(scores, eligible):
    ordered = sorted(eligible, key=lambda index: scores[index], reverse=True)
    if not ordered:
        return None
    if len(ordered) > 1 and abs(scores[ordered[0]] - scores[ordered[1]]) < 1e-9:
        return None
    return ordered[0]


def _family_values(cue):
    return (
        0.64 * cue["scale"] + 0.36 * cue["composition"],
        0.16 * cue["local_luma"]
        + 0.16 * cue["local_chroma"]
        + 0.08 * cue["global_luma"]
        + 0.08 * cue["global_chroma"]
        + 0.20 * cue["focus"]
        + 0.12 * cue["modeling"]
        + 0.12 * cue["exposure"]
        + 0.08 * cue["saturation"],
        0.55 * cue["frontness"]
        + 0.30 * cue["detector"]
        + 0.15 * cue["completeness"],
    )


def _emphasis(cue) -> float:
    return sum(cue[key] for key in _EMPHASIS_KEYS) / len(_EMPHASIS_KEYS)


def _apply_emphasis_promotion(
    top: int,
    eligible: Sequence[int],
    cues: Sequence[dict],
    base_scores: Sequence[float],
    people: Sequence[PersonBox],
    stability: float,
) -> tuple[int, bool]:
    """Let a decisive emphasis leader take the subject role from the argmax.

    The argmax can favour a larger, more confidently detected companion whose
    visible body is flat and dark.  When the emphasis leader (sharp, modelled,
    lit, coloured) beats that companion by a wide margin, keeps an exact mask
    and does not trail far behind overall, artistic intent outranks box size.
    """
    leader = max(eligible, key=lambda index: _emphasis(cues[index]))
    emphasis_margin = _emphasis(cues[leader]) - _emphasis(cues[top])
    if (
        leader != top
        and emphasis_margin >= _EMPHASIS_PROMOTION_GAP
        and base_scores[top] - base_scores[leader] <= 0.10
        and base_scores[leader] >= 0.52
        and cues[leader]["appearance_reliability"] >= 0.9
        and stability >= 0.8
        and (people[leader].confidence >= 0.5 or people[leader].small_verified)
    ):
        return leader, True
    return top, False


def _round(value) -> float:
    return round(float(value), 6)


def rank_subjects(
    image: Image.Image,
    boxes: Sequence[PersonBox],
    *,
    masks: Mapping[int, np.ndarray] | Sequence[np.ndarray | None] | None = None,
    detection: Mapping | None = None,
    faces: Mapping[int, Sequence[int]] | None = None,
) -> dict:
    """Rank people and abstain unless a visual lead is stable and separated.

    ``masks`` may contain source-resolution boolean instance masks.  ``faces``
    may map a person index to its anchor face box: in group illustration the
    protagonist is drawn with the largest, closest face, so face scale is a
    geometry cue in its own right.  Callers can omit either; bounding boxes
    then act as deliberately low-reliability appearance proxies.  No image
    arrays or model objects are returned.
    """
    people = list(boxes)
    if not people:
        return {
            "version": SALIENCE_VERSION,
            "status": "no_person",
            "selected_index": None,
            "suggested_index": None,
            "score": 0.0,
            "margin": 0.0,
            "stability": 1.0,
            "co_primary_indices": [],
            "ranked": [],
        }
    ycbcr, content = _prepare_image(image)
    size = (ycbcr.shape[1], ycbcr.shape[0])
    scaled = [_scaled_box(person.xyxy, content, size) for person in people]
    confidences = [_clip(person.confidence) for person in people]
    candidate_masks, exact = [], []
    for index, box in enumerate(scaled):
        supplied = _supplied_mask(masks, index, image.size, content, size)
        if supplied is not None:
            candidate_masks.append(supplied)
            exact.append(True)
        else:
            candidate_masks.append(_rectangle_mask(box, ycbcr.shape[:2]))
            exact.append(False)
    appearance = _appearance(ycbcr, candidate_masks, exact, scaled)
    # Visible-body occupancy beats box area as a scale cue: merged or padded
    # detection boxes otherwise inflate a leaning companion into the lead.
    effective_areas = [
        float(mask.sum()) / EXPECTED_BODY_FILL if is_exact else float(box_area)
        for mask, is_exact, box_area in zip(
            candidate_masks, exact, (_area(box) for box in scaled), strict=True
        )
    ]
    geometry = _geometry(scaled, confidences, size, effective_areas)
    face_areas = [
        float(_area(faces[index])) if faces and index in faces else None
        for index in range(len(people))
    ]
    known_faces = [value for value in face_areas if value]
    maximum_face = max(known_faces) if known_faces else None
    face_scales = [
        _clip(math.sqrt(value / maximum_face)) if value and maximum_face else 0.5
        for value in face_areas
    ]
    cues = []
    eligible = []
    content_area = max(1, size[0] * size[1])
    for index, (person, box, visual, spatial) in enumerate(
        zip(people, scaled, appearance, geometry, strict=True)
    ):
        area_fraction = _area(box) / content_area
        is_eligible = not (
            person.confidence < 0.4
            or (area_fraction < 0.08 and not person.small_verified)
            or (
                person.confidence < 0.5
                and area_fraction < 0.12
                and not person.small_verified
            )
        )
        if is_eligible:
            eligible.append(index)
        cues.append(
            {
                "scale": spatial["scale"],
                "composition": spatial["composition"],
                "face_scale": face_scales[index],
                "local_luma": visual["local_luma"],
                "local_chroma": visual["local_chroma"],
                "global_luma": visual["global_luma"],
                "global_chroma": visual["global_chroma"],
                "focus": visual["focus"],
                "modeling": visual["modeling"],
                "exposure": visual["exposure"],
                "saturation": visual["saturation"],
                "frontness": spatial["frontness"],
                "detector": spatial["detector"],
                "completeness": visual["completeness"],
                "appearance_reliability": visual["reliability"],
                "eligible": is_eligible,
            }
        )
    for cue in cues:
        cue["emphasis"] = sum(cue[key] for key in _EMPHASIS_KEYS) / len(
            _EMPHASIS_KEYS
        )
    variants = _variant_scores(cues, scaled, confidences, size, effective_areas)
    base_scores = variants[0]
    ranked_indices = sorted(range(len(people)), key=lambda i: (-base_scores[i], i))
    ranked = [
        {
            "index": index,
            "box": [int(value) for value in people[index].xyxy],
            "confidence": _round(people[index].confidence),
            "small_verified": bool(people[index].small_verified),
            "eligible": bool(cues[index]["eligible"]),
            "score": _round(base_scores[index]),
            "cues": {
                key: _round(value)
                for key, value in cues[index].items()
                if key not in {"eligible"}
            },
        }
        for index in ranked_indices
    ]
    if not eligible:
        return {
            "version": SALIENCE_VERSION,
            "status": "weak_detection",
            "selected_index": None,
            "suggested_index": None,
            "score": 0.0,
            "margin": 0.0,
            "stability": 0.0,
            "co_primary_indices": [],
            "ranked": ranked,
        }
    ordered = sorted(eligible, key=lambda index: (-base_scores[index], index))
    top = ordered[0]
    runner_up = ordered[1] if len(ordered) > 1 else None
    margin = base_scores[top] - (base_scores[runner_up] if runner_up is not None else 0)
    stability = sum(
        _unique_winner(scores, eligible) == top for scores in variants
    ) / len(variants)
    co_primary = [
        index for index in ordered if base_scores[top] - base_scores[index] <= 0.08
    ]
    incomplete = bool(
        detection
        and (
            detection.get("budget_exhausted")
            or detection.get("candidate_complete") is False
        )
    )
    status, selected = "candidate", top
    promoted_by_emphasis = False
    if len(eligible) == 1:
        person = people[top]
        if person.confidence < 0.5 and not person.small_verified:
            status = "tentative_detection"
    elif incomplete:
        status, selected = "candidate_set_incomplete", None
    else:
        top_families = _family_values(cues[top])
        family_support = 0
        for family_index, value in enumerate(top_families):
            other = max(
                _family_values(cues[index])[family_index]
                for index in eligible
                if index != top
            )
            family_support += value - other >= 0.02
        appearance_top = top_families[1]
        appearance_runner = max(
            _family_values(cues[index])[1] for index in eligible if index != top
        )
        legacy = []
        for index in eligible:
            x0, _, x1, _ = scaled[index]
            center = (x0 + x1) / (2 * max(1, size[0]))
            legacy.append(
                (
                    _area(scaled[index])
                    * (1.0 - 0.3 * abs(center - 0.5))
                    * max(0.5, people[index].confidence),
                    index,
                )
            )
        legacy.sort(reverse=True)
        legacy_ratio = legacy[0][0] / max(1e-9, legacy[1][0])
        legacy_rescue = (
            legacy[0][1] == top
            and legacy_ratio >= 1.7
            and margin >= 0.08
            and appearance_runner - appearance_top <= 0.15
            and stability >= 0.8
        )
        regular_accept = (
            base_scores[top] >= 0.52
            and margin >= 0.14
            and stability >= 0.8
            and (family_support >= 2 or margin >= 0.18)
            and (people[top].confidence >= 0.5 or people[top].small_verified)
        )
        # Artistic emphasis: a lead that is sharply drawn, modelled, lit and
        # coloured well beyond every companion is the intended focus even when
        # a large dark companion keeps the overall scores close.  Requires an
        # exact mask so box proxies cannot fake the appearance evidence, and
        # the emphasis gap is only measured against opponents whose appearance
        # evidence is equally reliable - a box proxy mixes in background
        # pixels (bright bedding, sky) and must not veto the decision.
        reliable_opponents = [
            index
            for index in eligible
            if index != top and cues[index]["appearance_reliability"] >= 0.9
        ]
        emphasis_gap = (
            _emphasis(cues[top])
            - max(
                (_emphasis(cues[index]) for index in reliable_opponents),
                default=0.5,
            )
            if reliable_opponents
            else _emphasis(cues[top]) - 0.5
        )
        emphasis_accept = (
            base_scores[top] >= 0.52
            and margin >= (0.0 if reliable_opponents else 0.03)
            and stability >= 0.8
            and emphasis_gap >= _EMPHASIS_GAP
            and cues[top]["appearance_reliability"] >= 0.9
            and (people[top].confidence >= 0.5 or people[top].small_verified)
        )
        if (
            not emphasis_accept
            and not reliable_opponents
            and base_scores[top] >= 0.52
            and margin >= 0.03
            and stability >= 0.8
            and _emphasis(cues[top]) >= 0.55
            and cues[top]["appearance_reliability"] >= 0.9
            and (people[top].confidence >= 0.5 or people[top].small_verified)
        ):
            # No reliable opposing appearance evidence at all: the lead's own
            # emphasis carries the decision.
            emphasis_accept = True
        promoted_by_emphasis = False
        if not (regular_accept or emphasis_accept):
            leader, promoted_by_emphasis = _apply_emphasis_promotion(
                top, eligible, cues, base_scores, people, stability
            )
            if promoted_by_emphasis:
                top = leader
                margin = base_scores[top] - max(
                    base_scores[index]
                    for index in eligible
                    if index != top
                )
        if margin < 0.01 and not (emphasis_accept or promoted_by_emphasis):
            status, selected = "ambiguous_salience", None
        elif margin < 0.10 and not (
            regular_accept or emphasis_accept or promoted_by_emphasis
        ):
            status, selected = "ambiguous_salience", None
        elif margin < 0.14 and not (
            regular_accept or emphasis_accept or promoted_by_emphasis
        ):
            status, selected = "tentative_salience", None
        elif not (
            regular_accept
            or emphasis_accept
            or promoted_by_emphasis
            or legacy_rescue
        ):
            status, selected = "salience_unreliable", None
    return {
        "version": SALIENCE_VERSION,
        "status": status,
        "selected_index": selected,
        "suggested_index": top,
        "score": _round(base_scores[top]),
        "margin": _round(margin),
        "stability": _round(stability),
        "co_primary_indices": co_primary,
        "promoted_by_emphasis": promoted_by_emphasis,
        "ranked": ranked,
    }


def salience_subject_decision(
    salience: Mapping,
    people: Sequence[PersonBox],
    census_people: Sequence[Mapping],
    face_people: Mapping[int, int],
) -> dict | None:
    """Translate a stable salience winner into a query subject decision.

    Returns ``None`` unless the scorer confidently, stably selects exactly one
    person.  A face anchor is an identity hint, not the subject itself: the
    winning person may have no census entry (back-facing, undetected face) and
    then the decision is box-only and carries no gender claims.
    """
    if not isinstance(salience, Mapping) or salience.get("status") != "candidate":
        return None
    selected = salience.get("selected_index")
    if selected is None or not 0 <= int(selected) < len(people):
        return None
    selected = int(selected)
    person = people[selected]
    face_index = next(
        (face for face, index in face_people.items() if index == selected), None
    )
    entry = None
    if face_index is not None and face_index < len(census_people):
        entry = census_people[face_index]
    decision = {
        "person_index": selected,
        "box": tuple(int(value) for value in person.xyxy),
        "gender": "unknown",
        "gender_status": "no_face_anchor",
        "tags": {},
        "male_probability": 0.0,
        "female_probability": 0.0,
        "context_box": None,
        "context_mode": "full_body",
    }
    if entry is None:
        return decision
    decision["context_box"] = entry.get("context_box")
    mode = entry.get("context_mode")
    decision["context_mode"] = mode if isinstance(mode, str) else "face_crop"
    gender = entry.get("gender")
    if gender in ("male", "female"):
        decision["gender"] = gender
        decision["gender_status"] = entry.get("status", "gender_confident")
        decision["tags"] = dict(entry.get("tags") or {})
        decision["male_probability"] = float(entry.get("male_probability") or 0.0)
        decision["female_probability"] = float(entry.get("female_probability") or 0.0)
        # Per-person model scores let the archive policy re-score a move using
        # neighbour-free evidence instead of the old whole-crop logits.
        models = entry.get("gender_models")
        if isinstance(models, dict):
            decision["gender_models"] = {
                model: [float(value) for value in pair]
                for model, pair in models.items()
                if isinstance(pair, tuple | list)
                and len(pair) == 2
                and all(isinstance(value, int | float) for value in pair)
            }
    else:
        decision["gender_status"] = "per_person_gender_uncertain"
    return decision


__all__ = [
    "MAX_ANALYSIS_SIDE",
    "SALIENCE_VERSION",
    "rank_subjects",
    "salience_subject_decision",
]

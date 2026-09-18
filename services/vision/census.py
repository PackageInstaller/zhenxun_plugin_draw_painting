"""Per-person diagnostics; identity masks and gender decisions stay separate."""

import numpy as np
from PIL import Image

from zhenxun.services.log import logger

from .faces import FaceBox, face_context
from .fusion import fuse_general, subject_gender
from .instance_regions import (
    AnimeInstanceSegmenter,
    InferenceRegion,
    masked_face_view,
    masked_upper_view,
)
from .subjects import PersonBox
from .taggers import CamieTaggerModel, WDTaggerModel
from .types import MULTIPLE_TAGS, ModelTags

MAX_ANALYZED_FACES = 24
CENSUS_VERSION = "instance-census-v1:face-784dc4c0bb69"


def personal_scores(tags: ModelTags) -> tuple[float, float]:
    return tags.general.get("1boy", 0.0), tags.general.get("1girl", 0.0)


def face_gender(left: ModelTags, right: ModelTags, face: FaceBox):
    lm, lf = personal_scores(left)
    rm, rf = personal_scores(right)
    if any(
        tags.general.get(tag, 0.0) >= 0.6
        for tags in (left, right)
        for tag in MULTIPLE_TAGS
    ):
        return "unknown", "multiple_in_crop", min(lm, rm), min(lf, rf)
    # Counts below the veto still cannot contribute to individual scores.
    clean = [
        ModelTags({"1boy": male, "1girl": female}, {}, {})
        for male, female in ((lm, lf), (rm, rf))
    ]
    result = subject_gender(*clean)
    if result[1] != "gender_uncertain" or face.confidence < 0.6 or face.support < 2:
        return result
    # Diagnostic counts have a different purpose from automatic library moves.
    # A repeatedly located face + two same-direction votes with a 25-point
    # margin can be displayed as an estimate; it never changes archival policy.
    if min(lf, rf) >= 0.7 and min(lf - lm, rf - rm) >= 0.25:
        return "female", "face_consensus", min(lm, rm), min(lf, rf)
    if min(lm, rm) >= 0.7 and min(lm - lf, rm - rf) >= 0.25:
        return "male", "face_consensus", min(lm, rm), min(lf, rf)
    return result


def overlaps_neighbor(region, face: FaceBox, faces: list[FaceBox]) -> bool:
    a, b, c, d = region
    return any(
        other.xyxy != face.xyxy
        and min(c, other.xyxy[2]) > max(a, other.xyxy[0])
        and min(d, other.xyxy[3]) > max(b, other.xyxy[1])
        for other in faces
    )


def _decision(left, right, face: FaceBox, *, contaminated: bool = False) -> dict:
    if contaminated:
        return {"gender": "unknown", "status": "neighbor_overlap", "tags": {}}
    if isinstance(left, Exception) or isinstance(right, Exception):
        return {"gender": "unknown", "status": "inference_failed", "tags": {}}
    gender, status, male, female = face_gender(left, right, face)
    return {
        "gender": gender,
        "status": status,
        "male_probability": male,
        "female_probability": female,
        "gender_models": {"wd": personal_scores(left), "camie": personal_scores(right)},
        "tags": fuse_general(left, right, strict=True) if gender != "unknown" else {},
    }


def analyze_people(
    image: Image.Image,
    faces: list[FaceBox],
    wd: WDTaggerModel,
    camie: CamieTaggerModel,
    *,
    people: list[PersonBox] | None = None,
    segmenter: AnimeInstanceSegmenter | None = None,
    return_masks: bool = False,
) -> dict | tuple[dict, dict[int, np.ndarray]]:
    selected = faces[:MAX_ANALYZED_FACES]
    instance_regions: dict[int, InferenceRegion] = {}
    if segmenter is not None and people:
        try:
            instance_regions = segmenter.segment(
                image, selected, people, return_masks=return_masks
            )
        except Exception as exc:
            # Optional instance isolation must never make the diagnostic command
            # less reliable than its conservative face-crop fallback.
            logger.warning(f"动漫人物实例隔离失败，改用人脸邻域: {type(exc).__name__}")

    contexts: list[InferenceRegion] = []
    for index, face in enumerate(selected):
        if region := instance_regions.get(index):
            contexts.append(region)
            continue
        view, box = face_context(image, face, neighbors=faces)
        contexts.append(
            InferenceRegion(
                view,
                box,
                "face_crop",
                [],
                contaminated=overlaps_neighbor(box, face, faces),
            )
        )
    wd_results = wd.predict_images([region.view for region in contexts])
    camie_results = camie.predict_images([region.view for region in contexts])
    entries = []
    for face, region, left, right in zip(
        selected, contexts, wd_results, camie_results, strict=True
    ):
        entry = {
            "box": face.xyxy,
            "context_box": region.box,
            "context_mode": region.mode,
            "context_outline": region.outline,
            "confidence": round(face.confidence, 4),
            "rotation": face.rotation,
            "support": face.support,
            "gender": "unknown",
            "status": "inference_failed",
            "tags": {},
        }
        if region.outline:
            # Keep the visible-body outline even when a later face-only retry
            # becomes the final classification input.
            entry["instance_outline"] = region.outline
            entry["instance_box"] = region.box
        if region.proposal_box is not None:
            entry["proposal_box"] = region.proposal_box
        if region.confidence is not None:
            entry["instance_confidence"] = round(region.confidence, 4)
        entry.update(
            _decision(
                left,
                right,
                face,
                contaminated=region.contaminated,
            )
        )
        entries.append(entry)
    pending = [
        index for index, person in enumerate(entries) if person["gender"] == "unknown"
    ]
    # Interlocked poses can make even a clean full-body mask read as "multiple
    # people" (joined heart gestures, hands resting on shoulders).  Retry chain
    # per person, each mode tried at most once so the loop always terminates:
    # head-and-shoulders band of the SAME mask, then a face band, then the
    # legacy tight rectangular crop.
    attempted: dict[int, set[str]] = {}
    while pending:
        tried: list[int] = []
        retries: list[tuple[Image.Image, tuple[int, int, int, int], str, bool]] = []
        for index in pending:
            modes = attempted.setdefault(index, set())
            region = instance_regions.get(index)
            retry = None
            if region is not None:
                if "instance_mask_upper" not in modes:
                    retry = masked_upper_view(image, selected[index], region)
                    mode = "instance_mask_upper"
                elif "instance_mask_face" not in modes:
                    retry = masked_face_view(image, selected[index], region)
                    mode = "instance_mask_face"
            if retry is None and "face_crop_tight" not in modes:
                view, box = face_context(
                    image, selected[index], tight=True, neighbors=faces
                )
                retry = (view, box)
                mode = "face_crop_tight"
            if retry is None:
                continue
            modes.add(mode)
            if mode == "face_crop_tight":
                retries.append(
                    (
                        retry[0],
                        retry[1],
                        mode,
                        overlaps_neighbor(retry[1], selected[index], faces),
                    )
                )
            else:
                retries.append((retry[0], retry[1], mode, False))
            tried.append(index)
        if not retries:
            break
        retry_wd = wd.predict_images([item[0] for item in retries])
        retry_camie = camie.predict_images([item[0] for item in retries])
        next_pending: list[int] = []
        for index, (view, box, mode, contaminated), left, right in zip(
            tried, retries, retry_wd, retry_camie, strict=True
        ):
            decision = _decision(
                left,
                right,
                selected[index],
                contaminated=contaminated,
            )
            person = entries[index]
            person["initial_gender_models"] = person.get("gender_models", {})
            person["retry_status"] = decision["status"]
            if decision["status"] == "inference_failed":
                next_pending.append(index)
                continue
            # A narrower view is a WORSE gender witness than an uncontaminated
            # full-body mask (tomboy faces read as male from head-only crops).
            # Its consensus estimate never overwrites the clean full view; it
            # stays as an auditable note.  Contaminated initial views are the
            # exception - any cleaner retry beats them.
            estimate_only = (
                decision["status"] == "face_consensus"
                and person["status"]
                not in ("multiple_in_crop", "neighbor_overlap")
            )
            if estimate_only:
                person["retry_gender_estimate"] = decision["gender"]
                person["retry_gender_models"] = decision.get("gender_models", {})
                continue
            person["initial_context_box"] = person["context_box"]
            person["initial_context_mode"] = person["context_mode"]
            for key in ("male_probability", "female_probability", "gender_models"):
                person.pop(key, None)
            person.update(decision)
            person["context_box"] = box
            person["context_mode"] = mode
            person["tight_crop"] = mode == "face_crop_tight"
            if mode == "face_crop_tight":
                person["context_outline"] = []
            elif person["gender"] == "unknown":
                # The mask band was not enough; keep its scores as the audit
                # trail and let the tight crop try once more.
                next_pending.append(index)
        pending = [
            index
            for index in next_pending
            # Continue only while the tight crop has not been tried yet.
            if "face_crop_tight" not in attempted.get(index, set())
        ]
    female = sum(person["gender"] == "female" for person in entries)
    male = sum(person["gender"] == "male" for person in entries)
    result = {
        "version": CENSUS_VERSION,
        "detected": len(faces),
        "analyzed": len(entries),
        "female": female,
        "male": male,
        "unknown": len(faces) - female - male,
        "people": entries,
    }
    if not return_masks:
        return result
    masks = {
        index: region.mask
        for index, region in instance_regions.items()
        if region.mask is not None
    }
    return result, masks

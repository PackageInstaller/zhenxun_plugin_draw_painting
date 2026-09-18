from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from threading import Lock

import numpy as np
from PIL import Image

from zhenxun.services.log import logger

from .census import analyze_people
from .faces import AnimeFaceDetector
from .fusion import fuse_general, subject_gender
from .instance_regions import AnimeInstanceSegmenter, match_faces_to_people
from .runtime import open_rgb
from .slicing import detect_with_slices
from .subject_salience import rank_subjects, salience_subject_decision
from .subjects import (
    AnimePersonDetector,
    PersonBox,
    refine_subject_box,
    select_subject,
    subject_focus_box,
)
from .taggers import CamieTaggerModel, WDTaggerModel
from .types import ModelTags, PredictionOutcome, TagPrediction


def _iou(left, right) -> float:
    overlap = max(0.0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0.0, min(left[3], right[3]) - max(left[1], right[1])
    )
    left_area = max(1.0, (left[2] - left[0]) * (left[3] - left[1]))
    right_area = max(1.0, (right[2] - right[0]) * (right[3] - right[1]))
    return overlap / (left_area + right_area - overlap)


def apply_salience_subject(prediction: TagPrediction, analysis: dict, override: dict):
    """Point the image subject at the visually salient person.

    The indexing pass picks a subject by person-box geometry and tags its crop.
    When the salience scorer confidently chooses the same person, its stored
    fields are kept (only refreshed with cleaner per-person model scores);
    when it chooses a different one, the neighbour-free per-person census
    decision of that person becomes the displayed subject instead.
    """
    box = override["box"]
    previous = analysis.get("subject_box")
    agrees = previous is not None and _iou(previous, box) >= 0.5
    # "face_consensus" is a display-only estimate: the census module documents
    # that it must never change archival policy, so only a true per-person
    # "confident" decision may mark the image subject as confident.
    confident = override["gender"] in ("male", "female") and override.get(
        "gender_status"
    ) == "confident"
    gender_scores = override.get("gender_models")

    def applied(source: str = "visual_salience") -> tuple[TagPrediction, dict]:
        context_box = override.get("context_box")
        subject_view = (
            "instance_mask"
            if override.get("context_mode") == "instance_mask"
            else "full_body"
        )
        updated = {
            **analysis,
            "subject_source": source,
            "subject_box": list(box),
            "feature_box": list(context_box) if context_box else list(box),
            "subject_view": subject_view,
            "salience_gender_status": override.get("gender_status"),
        }
        if gender_scores:
            updated["subject_gender"] = gender_scores
        return (
            replace(
                prediction,
                subject_status="confident" if confident else "tentative_detection",
                subject_gender=override["gender"],
                subject_tags=override["tags"],
                male_probability=override["male_probability"] if confident else 0,
                female_probability=override["female_probability"] if confident else 0,
                analysis=updated,
            ),
            updated,
        )

    if agrees and prediction.subject_status == "confident":
        # Same person: a decided subject is never overwritten unless the
        # neighbour-free masked view disagrees with the old whole-crop verdict.
        if override["gender"] in ("male", "female") and (
            override["gender"] != prediction.subject_gender or confident
        ):
            return applied("visual_salience_confirmed")
        updated = {**analysis, "subject_source": "visual_salience_confirmed"}
        if gender_scores:
            updated["subject_gender"] = gender_scores
        return replace(prediction, analysis=updated), updated
    return applied()


def _co_primary_gender(people: list[dict]):
    """Two or more confirmed people sharing one gender: co-primary subjects.

    Mirrored pairs and twin compositions have no single lead; their shared
    gender is still the image's subject gender.  Requires every CONFIRMED
    person to agree - one confident opposite vote disqualifies the pair.
    """
    confirmed = [
        person
        for person in people
        if person.get("status") == "confident"
        and person.get("gender") in ("male", "female")
    ]
    if len(confirmed) < 2 or len({person["gender"] for person in confirmed}) != 1:
        return None
    gender = confirmed[0]["gender"]
    male = min(person.get("male_probability", 0.0) for person in confirmed)
    female = min(person.get("female_probability", 0.0) for person in confirmed)
    models: dict[str, list[float]] = {}
    for model in ("wd", "camie"):
        pairs = [
            person.get("gender_models", {}).get(model)
            for person in confirmed
            if isinstance(person.get("gender_models", {}).get(model), list | tuple)
        ]
        if len(pairs) == len(confirmed):
            models[model] = [
                min(float(pair[index]) for pair in pairs) for index in (0, 1)
            ]
    return gender, male, female, models


def multi_person_subject_pass(
    image: Image.Image,
    prediction: TagPrediction,
    boxes: list[PersonBox],
    *,
    face_detector,
    segmenter: AnimeInstanceSegmenter | None,
    wd: WDTaggerModel,
    camie: CamieTaggerModel,
    salience_fn=rank_subjects,
) -> TagPrediction:
    """Full-evidence subject decision shared by indexing and feature query.

    Faces anchor identity, instance masks hand the taggers neighbour-free
    pixels, the salience scorer (scale/composition/perspective/colour/light)
    picks the visual lead, and that person's census gender becomes the
    image-level subject gender used for library classification.
    """
    if len(boxes) < 2:
        return prediction
    faces = face_detector.detect(image, scene_tags=prediction.general_tags)
    census, instance_masks = analyze_people(
        image,
        faces,
        wd,
        camie,
        people=boxes,
        segmenter=segmenter,
        return_masks=True,
    )
    analysis = {**prediction.analysis, "census": census}
    face_people = match_faces_to_people(faces, boxes)
    person_masks: dict[int, np.ndarray] = {}
    for face_index, mask in instance_masks.items():
        person_index = face_people.get(face_index)
        if person_index is not None:
            person_masks.setdefault(person_index, mask)
    salience = salience_fn(
        image,
        boxes,
        masks=person_masks or None,
        detection=analysis.get("detection"),
        faces={
            person_index: faces[face_index].xyxy
            for face_index, person_index in face_people.items()
        },
    )
    def face_of(person_index):
        return next(
            (
                face
                for face, index in face_people.items()
                if index == person_index
            ),
            None,
        )

    salience["selected_face_index"] = face_of(salience.get("selected_index"))
    salience["suggested_face_index"] = face_of(salience.get("suggested_index"))
    analysis["subject_salience"] = salience
    override = salience_subject_decision(
        salience, boxes, census.get("people", []), face_people
    )
    if override is None:
        co_primary = _co_primary_gender(census.get("people", []))
        if co_primary is not None:
            gender, male, female, models = co_primary
            updated = {**analysis, "subject_source": "co_primary_census"}
            if models:
                updated["subject_gender"] = models
            return replace(
                prediction,
                subject_status="co_primary",
                subject_gender=gender,
                male_probability=male,
                female_probability=female,
                analysis=updated,
            )
        return replace(prediction, analysis=analysis)
    new_prediction, _ = apply_salience_subject(prediction, analysis, override)
    return new_prediction


class EnsembleModel:
    """Serialized sessions, cross-image batching, at most 9.75 GiB CUDA arenas."""

    def __init__(self) -> None:
        self.detector = AnimePersonDetector()
        self.wd = WDTaggerModel()
        self.camie = CamieTaggerModel()
        self._face_detector: AnimeFaceDetector | None = None
        self._instance_segmenter: AnimeInstanceSegmenter | None = None
        self._instance_unavailable = False
        self._lock = Lock()

    def predict_paths(self, paths: list[Path]) -> list[PredictionOutcome]:
        with self._lock:
            results = []
            # Bound decoded originals as well as tensor batches for large PNGs.
            for offset in range(0, len(paths), 8):
                results.extend(self._predict_batch(paths[offset : offset + 8]))
            return results

    def _prepare_one(self, path: Path):
        image = open_rgb(path)
        boxes, detection = detect_with_slices(image, self.detector.detect)
        main, reason = select_subject(boxes, image.width, image.height)
        original_box = main.xyxy if main else None
        if main is None and reason == "weak_detection" and boxes:
            # Re-detect a weak large candidate at its native crop resolution.
            # Never fall back to a whole-scene gender vote or pick a member of
            # an ambiguous group. Same-person IoU/area gates still apply.
            weak = max(boxes, key=lambda box: box.area)
            if weak.area >= image.width * image.height * 0.08:
                refined = refine_subject_box(
                    weak, self.detector.detect(image.crop(weak.xyxy))
                )
                boxes = [refined if box == weak else box for box in boxes]
                main, reason = select_subject(boxes, image.width, image.height)
        if main is not None and reason != "candidate":
            refined = refine_subject_box(
                main, self.detector.detect(image.crop(main.xyxy))
            )
            boxes = [refined if box == main else box for box in boxes]
            # Refinement changes geometry and reliability: do not carry the old
            # overlap/dominance decision into the new crop.
            main, reason = select_subject(boxes, image.width, image.height)
        views = [image]
        if main is not None:
            views.append(image.crop(main.xyxy))
        return image, boxes, main, reason, original_box, detection, views

    def _get_segmenter(self) -> AnimeInstanceSegmenter | None:
        """Lazy CPU segmenter; unavailable stays sticky for the session."""
        if self._instance_unavailable:
            return None
        try:
            if self._instance_segmenter is None:
                self._instance_segmenter = AnimeInstanceSegmenter()
            return self._instance_segmenter
        except Exception as exc:
            self._instance_unavailable = True
            logger.warning(
                "动漫人物实例分割不可用，改用人脸邻域: " f"{type(exc).__name__}"
            )
            return None

    def _subject_pass(self, image, outcome: PredictionOutcome, boxes):
        """Silent-path multi-person upgrade; never fails the whole prediction."""
        if not isinstance(outcome, TagPrediction) or len(boxes) < 2:
            return outcome
        try:
            if self._face_detector is None:
                self._face_detector = AnimeFaceDetector()
            return multi_person_subject_pass(
                image,
                outcome,
                boxes,
                face_detector=self._face_detector,
                segmenter=self._get_segmenter(),
                wd=self.wd,
                camie=self.camie,
            )
        except Exception as exc:
            logger.warning(f"多人主体决策失败，保留几何主体决策: {type(exc).__name__}")
            return outcome

    def _predict_batch(self, paths: list[Path]) -> list[PredictionOutcome]:
        prepared, views = [], []
        for path in paths:
            try:
                item = self._prepare_one(path)
                prepared.append((item, len(views)))
                views.extend(item[-1])
            except Exception as exc:
                prepared.append(exc)
        if not views:
            return prepared
        wd = self.wd.predict_images(views)
        camie = self.camie.predict_images(views)
        results = []
        for path, item in zip(paths, prepared, strict=True):
            if isinstance(item, Exception):
                results.append(item)
                continue
            state, start = item
            end = start + len(state[-1])
            outcome = self._predict_one(path, state, (wd[start:end], camie[start:end]))
            results.append(self._subject_pass(state[0], outcome, state[1]))
        return results

    def predict_query(self, path: Path) -> TagPrediction:
        """Extra census only for user uploads; share model instances and lock."""
        with self._lock:
            prediction = self._predict_one(path)
            if isinstance(prediction, Exception):
                raise prediction
            image = open_rgb(path)
            people = []
            for item in prediction.analysis.get("people", []):
                box = item.get("box")
                if not isinstance(box, list | tuple) or len(box) != 4:
                    continue
                people.append(
                    PersonBox(
                        tuple(int(value) for value in box),
                        float(item.get("confidence", 0.0)),
                        bool(item.get("small_verified", False)),
                    )
                )
            # Multi-person uploads share the indexing path's full-evidence
            # subject decision (face anchors + instance masks + salience).
            prediction = self._subject_pass(image, prediction, people)
            analysis = prediction.analysis
            if "census" not in analysis:
                if self._face_detector is None:
                    self._face_detector = AnimeFaceDetector()
                faces = self._face_detector.detect(
                    image, scene_tags=prediction.general_tags
                )
                census = analyze_people(
                    image,
                    faces,
                    self.wd,
                    self.camie,
                    people=people or None,
                    segmenter=self._get_segmenter(),
                )
                analysis = {**analysis, "census": census}
                prediction = replace(prediction, analysis=analysis)
                analysis = prediction.analysis
            if analysis.get("subject_source") is None:
                areas = sorted(
                    (
                        (person["box"][2] - person["box"][0])
                        * (person["box"][3] - person["box"][1])
                        for person in analysis.get("census", {}).get("people", [])
                        if isinstance(person.get("box"), list | tuple)
                        and len(person["box"]) == 4
                    ),
                    reverse=True,
                )
                if len(areas) > 1 and areas[0] < areas[1] * 1.7:
                    # Similar-sized faces in a group are not one giant lead
                    # person, and no stable visual lead was confirmed either.
                    analysis = {
                        **analysis,
                        "subject_box": None,
                        "feature_box": None,
                        "subject_view": "none",
                        "subject_gender": {},
                    }
                    prediction = replace(
                        prediction,
                        subject_status="ambiguous_dominance",
                        subject_gender="unknown",
                        subject_tags={},
                        male_probability=0,
                        female_probability=0,
                        analysis=analysis,
                    )
            return prediction

    def _predict_one(
        self, path: Path, prepared=None, predictions=None
    ) -> PredictionOutcome:
        try:
            image, boxes, main, reason, original_box, detection, views = (
                prepared or self._prepare_one(path)
            )
            wd_results, camie_results = predictions or (
                self.wd.predict_images(views),
                self.camie.predict_images(views),
            )
            for result in (*wd_results, *camie_results):
                if isinstance(result, Exception):
                    # Never label a partial single-model result as fused/current.
                    raise result
            wd_scene, camie_scene = wd_results[0], camie_results[0]
            if not isinstance(wd_scene, ModelTags) or not isinstance(
                camie_scene, ModelTags
            ):
                raise TypeError("场景标签结果无效")
            tags: dict[str, float] = {}
            gender, status, male, female = "unknown", reason, 0.0, 0.0
            crop_scores = {}
            candidate_scores = {}
            feature_box = None
            subject_view = "none"
            candidate_tags = {}
            if main is not None:
                wd_subject, camie_subject = wd_results[1], camie_results[1]
                if not isinstance(wd_subject, ModelTags) or not isinstance(
                    camie_subject, ModelTags
                ):
                    raise TypeError("主体标签结果无效")
                gender, status, male, female = subject_gender(wd_subject, camie_subject)
                candidate_scores = {
                    "wd": wd_subject.gender_scores(),
                    "camie": camie_subject.gender_scores(),
                }
                candidate_tags = fuse_general(wd_subject, camie_subject, strict=True)
                crop_scores = candidate_scores
                feature_box = main.xyxy
                subject_view = "full_body"
                if reason != "candidate" or status != "confident":
                    focus = subject_focus_box(main, boxes)
                    if focus is not None:
                        wd_focus = self.wd.predict_images([image.crop(focus)])[0]
                        camie_focus = self.camie.predict_images([image.crop(focus)])[0]
                        if isinstance(wd_focus, Exception):
                            raise wd_focus
                        if isinstance(camie_focus, Exception):
                            raise camie_focus
                        full_gender, full_status = gender, status
                        gender, status, male, female = subject_gender(
                            wd_focus, camie_focus
                        )
                        if (
                            full_status == status == "confident"
                            and gender != full_gender
                        ):
                            gender, status = "unknown", "view_conflict"
                        crop_scores = {
                            "wd": wd_focus.gender_scores(),
                            "camie": camie_focus.gender_scores(),
                        }
                        feature_box, subject_view = focus, "upper_body"
                        # A focused view cannot certify the whole body's traits.
                        candidate_tags = fuse_general(
                            wd_focus, camie_focus, strict=True
                        )
                    elif reason != "candidate":
                        gender, status = "unknown", reason
                if status == "confident":
                    tags = candidate_tags
            return TagPrediction(
                width=image.width,
                height=image.height,
                male_probability=male,
                female_probability=female,
                general_tags=fuse_general(wd_scene, camie_scene, strict=False),
                character_tags={**wd_scene.characters, **camie_scene.characters},
                rating_tags=camie_scene.ratings,
                subject_tags=tags,
                subject_status=status,
                subject_gender=gender,
                analysis={
                    "detection": detection,
                    "subject_selection_revision": 4,
                    "people": [
                        {
                            "box": box.xyxy,
                            "confidence": round(box.confidence, 4),
                            "small_verified": box.small_verified,
                        }
                        for box in boxes
                    ],
                    "subject_box": main.xyxy if main else None,
                    "initial_subject_box": original_box,
                    "feature_box": feature_box,
                    "subject_view": subject_view,
                    "selection_status": reason,
                    "scene_gender": {
                        "wd": wd_scene.gender_scores(),
                        "camie": camie_scene.gender_scores(),
                    },
                    "subject_gender": crop_scores,
                    "candidate_gender": candidate_scores,
                    "candidate_tags": candidate_tags if status != "confident" else {},
                },
            )
        except Exception as exc:
            return exc

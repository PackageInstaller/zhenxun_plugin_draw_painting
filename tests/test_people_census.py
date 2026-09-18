"""Offline geometry, conservative census and rendering regressions."""

# ruff: noqa: PT009, PT027
from dataclasses import replace
import importlib
from io import BytesIO
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
import test_subject_features as base

faces = importlib.import_module("_painting_tests.services.vision.faces")
census = importlib.import_module("_painting_tests.services.vision.census")
instance_regions = importlib.import_module(
    "_painting_tests.services.vision.instance_regions"
)
report = importlib.import_module("_painting_tests.services.feature_report")
people_report = importlib.import_module("_painting_tests.services.people_report")
subjects = importlib.import_module("_painting_tests.services.vision.subjects")


class GeometryTests(unittest.TestCase):
    def test_face_to_person_matching_is_one_to_one_and_order_independent(self):
        anchors = [
            faces.FaceBox((100, 100, 180, 180), 0.9, 0),
            faces.FaceBox((430, 110, 510, 190), 0.9, 0),
        ]
        # Deliberately reverse the people and add one large distractor proposal.
        people = [
            subjects.PersonBox((350, 60, 600, 480), 0.8),
            subjects.PersonBox((20, 50, 300, 490), 0.85),
            subjects.PersonBox((0, 0, 700, 500), 0.35),
        ]
        self.assertEqual(
            {0: 1, 1: 0},
            instance_regions.match_faces_to_people(anchors, people),
        )

    def test_neighbor_aware_context_preserves_anchor_for_all_rotations(self):
        image = Image.new("RGB", (1296, 1812))
        for angle in (0, 90, 180, 270):
            first = faces.FaceBox((367, 341, 687, 629), 0.88, angle)
            second = faces.FaceBox((741, 470, 1121, 792), 0.88, angle)
            for target in (first, second):
                for tight in (False, True):
                    _, region = faces.face_context(
                        image, target, tight=tight, neighbors=[first, second]
                    )
                    self.assertLessEqual(region[0], target.xyxy[0])
                    self.assertLessEqual(region[1], target.xyxy[1])
                    self.assertGreaterEqual(region[2], target.xyxy[2])
                    self.assertGreaterEqual(region[3], target.xyxy[3])
                    self.assertFalse(
                        census.overlaps_neighbor(region, target, [first, second])
                    )

    def test_two_tile_halves_use_intact_anchor_not_two_people(self):
        full = faces.FaceBox((883, 16, 1836, 1071), 0.716, 0)
        left = faces.FaceBox((912, 0, 1548, 1087), 0.725, 0, clipped=True)
        right = faces.FaceBox((1248, 16, 1802, 1079), 0.627, 0, clipped=True)
        result = faces.merge_faces([left, right, full])
        self.assertEqual(1, len(result))
        self.assertEqual(full.xyxy, result[0].xyxy)
        self.assertEqual(3, result[0].support)

    def test_original_image_edge_is_not_artificial_clipping(self):
        self.assertFalse(
            faces.tile_clips_face((0, 0, 100, 100), (0, 0, 100, 100), (100, 100))
        )
        self.assertTrue(
            faces.tile_clips_face((0, 20, 40, 60), (50, 0, 150, 100), (200, 100))
        )
        self.assertFalse(
            faces.merge_faces(
                [
                    faces.FaceBox((0, 0, 40, 60), 0.99, 0, support=4, clipped=True),
                ]
            )
        )

    def test_nearby_distinct_small_faces_stay_separate(self):
        first = faces.FaceBox((1806, 1200, 1958, 1332), 0.609, 270, support=4)
        second = faces.FaceBox((2000, 1425, 2166, 1551), 0.772, 0, support=19)
        self.assertEqual(2, len(faces.merge_faces([first, second])))

    def test_detail_trigger_uses_multigirl_hint_or_large_image_small_face(self):
        small = faces.FaceBox((2000, 1425, 2166, 1551), 0.77, 0)
        self.assertTrue(faces.needs_detail_scan((4096, 4096), [small], {}))
        self.assertTrue(
            faces.needs_detail_scan((800, 800), [], {"multiple_girls": 0.51})
        )
        self.assertFalse(
            faces.needs_detail_scan((800, 800), [], {"multiple_girls": 0.49})
        )
        large = faces.FaceBox((883, 16, 1836, 1071), 0.71, 0)
        self.assertFalse(faces.needs_detail_scan((2796, 1290), [large], {"1boy": 0.99}))
        self.assertFalse(faces.needs_detail_scan((4096, 4096), [small, small], {}))
        self.assertTrue(
            faces.needs_detail_scan((4096, 4096), [small, small], {"3girls": 0.6})
        )

    def test_square_detail_tiles_overlap_and_remain_bounded(self):
        for size in ((4096, 4096), (2377, 1081), (1, 1)):
            regions = faces.detail_regions(*size)
            self.assertLessEqual(len(regions), 9)
            self.assertEqual(len(regions), len(set(regions)))
            for left, top, right, bottom in regions:
                self.assertTrue(0 <= left < right <= size[0])
                self.assertTrue(0 <= top < bottom <= size[1])
        self.assertIn((1024, 1024, 3072, 3072), faces.detail_regions(4096, 4096))

    def test_scene_count_labels_are_not_presented_as_detected_count(self):
        for tag in ("1girl", "6girls", "6+girls", "8girls", "multiple_girls"):
            self.assertTrue(people_report.is_count_tag(tag))
        self.assertFalse(people_report.is_count_tag("long_hair"))

    def test_all_right_angle_mappings_round_trip(self):
        original = (20, 30, 90, 130)
        for angle in (0, 90, 180, 270):
            rotated_size = (200, 500) if angle in (90, 270) else (500, 200)
            rotated = faces.unrotate_box(original, (-angle) % 360, *rotated_size)
            self.assertEqual(original, faces.unrotate_box(rotated, angle, 500, 200))

    def test_rotated_duplicate_is_counted_once(self):
        detections = [
            faces.FaceBox((100, 100, 200, 200), 0.8, angle)
            for angle in (0, 90, 180, 270)
        ]
        detections.append(faces.FaceBox((205, 100, 305, 200), 0.9, 0))
        result = faces.merge_faces(detections)
        self.assertEqual(2, len(result))
        self.assertEqual(4, result[0].support)

    def test_marginal_detection_needs_support(self):
        face = faces.FaceBox((10, 10, 50, 50), 0.4, 0)
        self.assertFalse(faces.merge_faces([face]))
        self.assertEqual(1, len(faces.merge_faces([face, replace(face, rotation=90)])))

    def test_tiling_is_bounded_and_inside_source(self):
        for width, height in ((2376, 1080), (600, 4000), (1000, 1000)):
            regions = faces.scan_regions(width, height)
            self.assertLessEqual(len(regions), 5)
            for x0, y0, x1, y1 in regions:
                self.assertTrue(0 <= x0 < x1 <= width)
                self.assertTrue(0 <= y0 < y1 <= height)

    def test_face_context_keeps_anchor_inside_for_every_rotation(self):
        image = Image.new("RGB", (500, 300))
        for angle in (0, 90, 180, 270):
            face = faces.FaceBox((100, 100, 180, 180), 0.9, angle)
            view, region = faces.face_context(image, face, tight=True)
            self.assertGreater(view.width, 0)
            self.assertLessEqual(region[0], face.xyxy[0])
            self.assertLessEqual(region[1], face.xyxy[1])
            self.assertGreaterEqual(region[2], face.xyxy[2])
            self.assertGreaterEqual(region[3], face.xyxy[3])


class CensusTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (1000, 500))
        self.face = faces.FaceBox((50, 100, 100, 150), 0.8, 0, support=3)
        self.female = base.ModelTags(
            {"1girl": 0.99, "1boy": 0.01, "long_hair": 0.9}, {}, {}
        )

    def test_face_threshold_does_not_relax_library_threshold(self):
        left = base.ModelTags({"1girl": 0.74, "1boy": 0.46}, {}, {})
        right = base.ModelTags({"1girl": 0.86, "1boy": 0.47}, {}, {})
        self.assertEqual("unknown", base.subject_gender(left, right)[0])
        self.assertEqual("female", census.face_gender(left, right, self.face)[0])
        self.assertEqual(
            "unknown", census.face_gender(left, right, replace(self.face, support=1))[0]
        )
        conflict = base.ModelTags({"1girl": 0.46, "1boy": 0.74}, {}, {})
        self.assertEqual("unknown", census.face_gender(left, conflict, self.face)[0])

    def test_group_scores_never_become_individual_gender_scores(self):
        tags = base.ModelTags(
            {"1girl": 0.2, "1boy": 0.01, "multiple_girls": 0.99}, {}, {}
        )
        decision = census._decision(tags, tags, self.face)
        self.assertEqual("multiple_in_crop", decision["status"])
        self.assertEqual(0.2, decision["female_probability"])
        self.assertFalse(decision["tags"])
        tags = base.ModelTags(
            {"1girl": 0.1, "1boy": 0.1, "male_focus": 0.99, "2girls": 0.59}, {}, {}
        )
        self.assertEqual("unknown", census.face_gender(tags, tags, self.face)[0])

    def test_unseparable_neighbor_vetoes_even_confident_gender(self):
        first = replace(self.face, xyxy=(100, 100, 200, 200))
        second = replace(self.face, xyxy=(150, 150, 250, 250))
        wd, camie = Mock(), Mock()
        for model in (wd, camie):
            model.predict_images.return_value = [self.female] * 2
        result = census.analyze_people(self.image, [first, second], wd, camie)
        self.assertEqual(2, result["unknown"])
        for person in result["people"]:
            self.assertEqual("neighbor_overlap", person["status"])
            self.assertFalse(person["tags"])
            self.assertNotIn("female_probability", person)

    def test_disjoint_instance_masks_allow_overlapping_person_bounds(self):
        first = replace(self.face, xyxy=(100, 100, 200, 200))
        second = replace(self.face, xyxy=(350, 100, 450, 200))
        regions = {
            0: instance_regions.InferenceRegion(
                Image.new("RGB", (500, 450), "red"),
                (0, 0, 500, 450),
                "instance_mask",
                [[(0, 0), (250, 0), (250, 450), (0, 450)]],
                confidence=0.8,
                proposal_box=(0, 0, 400, 500),
            ),
            1: instance_regions.InferenceRegion(
                Image.new("RGB", (500, 450), "blue"),
                (250, 0, 750, 450),
                "instance_mask",
                [[(250, 0), (750, 0), (750, 450), (250, 450)]],
                confidence=0.7,
                proposal_box=(250, 0, 750, 500),
            ),
        }
        segmenter = SimpleNamespace(segment=Mock(return_value=regions))
        wd, camie = Mock(), Mock()
        for model in (wd, camie):
            model.predict_images.return_value = [self.female, self.female]
        result = census.analyze_people(
            self.image,
            [first, second],
            wd,
            camie,
            people=[
                subjects.PersonBox((0, 0, 400, 500), 0.8),
                subjects.PersonBox((250, 0, 750, 500), 0.7),
            ],
            segmenter=segmenter,
        )
        self.assertEqual((2, 0), (result["female"], result["unknown"]))
        self.assertTrue(
            all(
                person["context_mode"] == "instance_mask"
                for person in result["people"]
            )
        )
        self.assertTrue(all(person["instance_outline"] for person in result["people"]))
        self.assertEqual((500, 450), wd.predict_images.call_args.args[0][0].size)

    def test_failed_confirmation_reports_latest_crop_not_initial_score(self):
        mixed = base.ModelTags({"1girl": 0.95, "2girls": 0.99}, {}, {})
        uncertain = base.ModelTags({"1girl": 0.4, "1boy": 0.3}, {}, {})
        wd, camie = Mock(), Mock()
        for model in (wd, camie):
            model.predict_images.side_effect = [[mixed], [uncertain]]
        result = census.analyze_people(self.image, [self.face], wd, camie)
        person = result["people"][0]
        self.assertEqual("gender_uncertain", person["status"])
        self.assertEqual(0.4, person["female_probability"])
        self.assertTrue(person["tight_crop"])
        self.assertEqual((0.0, 0.95), person["initial_gender_models"]["wd"])

    def test_unknown_report_explains_contamination_not_absent_features(self):
        self.assertIn(
            "多人", people_report.person_status_text({"status": "multiple_in_crop"})
        )
        self.assertIn(
            "隔离", people_report.person_status_text({"status": "neighbor_overlap"})
        )
        self.assertIn(
            "证据", people_report.person_status_text({"status": "gender_uncertain"})
        )

    def test_eight_faces_eight_votes_not_one_scene_count_tag(self):
        anchors = [
            replace(self.face, xyxy=(index * 110 + 10, 100, index * 110 + 60, 150))
            for index in range(8)
        ]
        wd, camie = Mock(), Mock()
        wd.predict_images.return_value = [self.female] * 8
        camie.predict_images.return_value = [self.female] * 8
        result = census.analyze_people(self.image, anchors, wd, camie)
        self.assertEqual(
            (8, 8, 0, 0),
            (result["detected"], result["female"], result["male"], result["unknown"]),
        )
        self.assertEqual(8, len(result["people"]))

    def test_unknown_retry_uses_tighter_view(self):
        mixed = base.ModelTags({"1girl": 0.99, "multiple_girls": 0.9}, {}, {})
        wd, camie = Mock(), Mock()
        for model in (wd, camie):
            model.predict_images.side_effect = [[mixed], [self.female]]
        result = census.analyze_people(self.image, [self.face], wd, camie)
        self.assertEqual(1, result["female"])
        self.assertTrue(result["people"][0]["tight_crop"])

    def test_failed_models_remain_unknown(self):
        wd, camie = Mock(), Mock()
        wd.predict_images.return_value = [RuntimeError("failed")]
        camie.predict_images.return_value = [self.female]
        result = census.analyze_people(self.image, [self.face], wd, camie)
        self.assertEqual(1, result["unknown"])
        self.assertEqual(0, result["female"])

    def test_group_render_and_analysis_limit(self):
        wd, camie = Mock(), Mock()
        wd.predict_images.return_value = [self.female] * 2
        camie.predict_images.return_value = [self.female] * 2
        with patch.object(census, "MAX_ANALYZED_FACES", 2):
            result = census.analyze_people(self.image, [self.face] * 3, wd, camie)
        self.assertEqual(
            (3, 2, 1), (result["detected"], result["analyzed"], result["unknown"])
        )
        value = replace(
            base.prediction(), width=1000, height=500, analysis={"census": result}
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.png"
            self.image.save(path)
            rendered = report.render_feature_report(path, value)
        with Image.open(BytesIO(rendered)) as output:
            self.assertEqual("JPEG", output.format)
            self.assertEqual(1600, output.width)


if __name__ == "__main__":
    unittest.main()


class SalienceIntegrationTests(unittest.TestCase):
    """Query-path subject override: salience decides, census evidence speaks."""

    def setUp(self):
        self.image = Image.new("RGB", (1000, 500))
        self.face = faces.FaceBox((50, 100, 100, 150), 0.8, 0, support=3)
        self.female = base.ModelTags({"1girl": 0.99, "1boy": 0.01}, {}, {})

    def test_return_masks_exposes_face_indexed_masks(self):
        mask = np.zeros((500, 1000), dtype=bool)
        mask[100:400, 100:400] = True
        regions = {
            0: instance_regions.InferenceRegion(
                Image.new("RGB", (300, 300), "red"),
                (100, 100, 400, 400),
                "instance_mask",
                [],
                confidence=0.8,
                proposal_box=(0, 0, 400, 500),
                mask=mask,
            )
        }
        segmenter = SimpleNamespace(segment=Mock(return_value=regions))
        wd, camie = Mock(), Mock()
        for model in (wd, camie):
            model.predict_images.return_value = [self.female]
        result, masks = census.analyze_people(
            self.image,
            [self.face],
            wd,
            camie,
            people=[subjects.PersonBox((0, 0, 400, 500), 0.8)],
            segmenter=segmenter,
            return_masks=True,
        )
        self.assertEqual("female", result["people"][0]["gender"])
        self.assertEqual({0: mask}, masks)
        segmenter.segment.assert_called_once()
        self.assertTrue(segmenter.segment.call_args.kwargs["return_masks"])

    def test_default_census_return_stays_a_plain_mapping(self):
        wd, camie = Mock(), Mock()
        for model in (wd, camie):
            model.predict_images.return_value = [self.female]
        result = census.analyze_people(
            self.image, [self.face], wd, camie
        )
        self.assertIsInstance(result, dict)
        self.assertNotIsInstance(result, tuple)

    def _prediction(self, **kwargs):
        defaults = {
            "width": 1000,
            "height": 500,
            "male_probability": 0.0,
            "female_probability": 0.0,
            "general_tags": {},
            "character_tags": {},
            "rating_tags": {},
            "subject_status": "ambiguous_dominance",
            "subject_gender": "unknown",
            "analysis": {"subject_box": None},
        }
        defaults.update(kwargs)
        return base_prediction_type(**defaults)

    def test_salient_person_replaces_indexing_subject(self):
        ensemble = importlib.import_module("_painting_tests.services.vision.ensemble")
        prediction = self._prediction()
        analysis = {"subject_box": None, "census": {"people": []}}
        override = {
            "person_index": 1,
            "box": (500, 100, 900, 950),
            "gender": "female",
            "gender_status": "confident",
            "tags": {"long_hair": 0.95},
            "male_probability": 0.01,
            "female_probability": 0.97,
            "context_box": (500, 80, 920, 960),
            "context_mode": "instance_mask",
        }
        new_prediction, updated = ensemble.apply_salience_subject(
            prediction, analysis, override
        )
        self.assertEqual("confident", new_prediction.subject_status)
        self.assertEqual("female", new_prediction.subject_gender)
        self.assertEqual({"long_hair": 0.95}, new_prediction.subject_tags)
        self.assertEqual(0.97, new_prediction.female_probability)
        self.assertEqual("visual_salience", updated["subject_source"])
        self.assertEqual("instance_mask", updated["subject_view"])
        self.assertEqual([500, 80, 920, 960], updated["feature_box"])

    def test_agreeing_confident_subject_is_never_downgraded(self):
        ensemble = importlib.import_module("_painting_tests.services.vision.ensemble")
        prediction = self._prediction(
            subject_status="confident",
            subject_gender="female",
            subject_tags={"twintails": 0.9},
            female_probability=0.95,
            analysis={"subject_box": [520, 120, 880, 900]},
        )
        override = {
            "person_index": 1,
            "box": (500, 100, 900, 950),
            "gender": "female",
            "gender_status": "confident",
            "tags": {"long_hair": 0.95},
            "male_probability": 0.01,
            "female_probability": 0.97,
            "context_box": None,
            "context_mode": "full_body",
        }
        new_prediction, updated = ensemble.apply_salience_subject(
            prediction, {"subject_box": [520, 120, 880, 900]}, override
        )
        # Never downgraded: status and gender stay; the confident same-person
        # verdict is refreshed with its neighbour-free masked-view evidence.
        self.assertEqual("confident", new_prediction.subject_status)
        self.assertEqual("female", new_prediction.subject_gender)
        self.assertEqual("long_hair", next(iter(new_prediction.subject_tags)))
        self.assertEqual(
            "visual_salience_confirmed", updated["subject_source"]
        )

    def test_faceless_salient_person_downgrades_to_tentative_box(self):
        ensemble = importlib.import_module("_painting_tests.services.vision.ensemble")
        prediction = self._prediction(
            subject_status="confident",
            subject_gender="female",
            analysis={"subject_box": [10, 10, 300, 480]},
        )
        override = {
            "person_index": 1,
            "box": (500, 100, 900, 950),
            "gender": "unknown",
            "gender_status": "no_face_anchor",
            "tags": {},
            "male_probability": 0.0,
            "female_probability": 0.0,
            "context_box": None,
            "context_mode": "full_body",
        }
        new_prediction, updated = ensemble.apply_salience_subject(
            prediction, {"subject_box": [10, 10, 300, 480]}, override
        )
        self.assertEqual("tentative_detection", new_prediction.subject_status)
        self.assertEqual("unknown", new_prediction.subject_gender)
        self.assertFalse(new_prediction.subject_tags)
        self.assertEqual([500, 100, 900, 950], updated["subject_box"])


def base_prediction_type(**kwargs):
    from _painting_tests.services.vision.types import TagPrediction

    return TagPrediction(**kwargs)


class MultiPersonSubjectPassTests(unittest.TestCase):
    """The shared full-evidence subject pass used by indexing and query."""

    def setUp(self):
        self.image = Image.new("RGB", (1000, 500))
        self.people = [
            subjects.PersonBox((0, 0, 400, 500), 0.8),
            subjects.PersonBox((550, 50, 950, 480), 0.75),
        ]
        self.faces = [
            faces.FaceBox((100, 100, 180, 180), 0.8, 0, support=3),
            faces.FaceBox((650, 120, 730, 200), 0.8, 0, support=3),
        ]
        self.female = base.ModelTags({"1girl": 0.99, "1boy": 0.01}, {}, {})

    def _prediction(self, **kwargs):
        from _painting_tests.services.vision.types import TagPrediction

        defaults = {
            "width": 1000,
            "height": 500,
            "male_probability": 0.0,
            "female_probability": 0.0,
            "general_tags": {},
            "character_tags": {},
            "rating_tags": {},
            "subject_status": "ambiguous_dominance",
            "subject_gender": "unknown",
            "analysis": {"subject_box": None, "detection": {}},
        }
        defaults.update(kwargs)
        return TagPrediction(**defaults)

    def _models(self):
        wd, camie = Mock(), Mock()
        for model in (wd, camie):
            model.predict_images.return_value = [self.female, self.female]
        face_detector = SimpleNamespace(
            detect=Mock(return_value=list(self.faces))
        )
        return wd, camie, face_detector

    def test_salient_person_decision_reaches_the_prediction(self):
        ensemble = importlib.import_module("_painting_tests.services.vision.ensemble")
        wd, camie, face_detector = self._models()
        salience_fn = Mock(
            return_value={
                "status": "candidate",
                "selected_index": 1,
                "score": 0.8,
                "margin": 0.3,
                "ranked": [],
            }
        )
        prediction = ensemble.multi_person_subject_pass(
            self.image,
            self._prediction(),
            self.people,
            face_detector=face_detector,
            segmenter=None,
            wd=wd,
            camie=camie,
            salience_fn=salience_fn,
        )
        self.assertEqual("confident", prediction.subject_status)
        self.assertEqual("female", prediction.subject_gender)
        analysis = prediction.analysis
        self.assertEqual("visual_salience", analysis["subject_source"])
        self.assertIn("census", analysis)
        self.assertEqual(1, analysis["subject_salience"]["selected_face_index"])
        self.assertEqual(
            [100, 100, 180, 180], list(analysis["census"]["people"][0]["box"])
        )
        self.assertEqual(
            [0.01, 0.99], list(analysis["subject_gender"]["camie"])
        )
        salience_fn.assert_called_once()
        self.assertIsNone(salience_fn.call_args.kwargs["masks"])

    def test_single_person_short_circuits_without_face_detection(self):
        ensemble = importlib.import_module("_painting_tests.services.vision.ensemble")
        wd, camie, face_detector = self._models()
        prediction = self._prediction()
        result = ensemble.multi_person_subject_pass(
            self.image,
            prediction,
            self.people[:1],
            face_detector=face_detector,
            segmenter=None,
            wd=wd,
            camie=camie,
        )
        self.assertIs(prediction, result)
        face_detector.detect.assert_not_called()

    def test_face_consensus_estimate_never_confirms_the_subject(self):
        ensemble = importlib.import_module("_painting_tests.services.vision.ensemble")
        override = {
            "person_index": 1,
            "box": (550, 50, 950, 480),
            "gender": "female",
            "gender_status": "face_consensus",
            "tags": {"long_hair": 0.9},
            "male_probability": 0.05,
            "female_probability": 0.9,
            "context_box": (540, 40, 960, 490),
            "context_mode": "instance_mask",
            "gender_models": {"wd": [0.05, 0.9], "camie": [0.05, 0.9]},
        }
        new_prediction, updated = ensemble.apply_salience_subject(
            self._prediction(), {"subject_box": None}, override
        )
        self.assertNotEqual("confident", new_prediction.subject_status)
        self.assertEqual("female", new_prediction.subject_gender)
        self.assertEqual(0, new_prediction.female_probability)
        # The estimate stays auditable but cannot drive a library move.
        self.assertEqual([0.05, 0.9], list(updated["subject_gender"]["wd"]))


class RetryChainTerminationTests(unittest.TestCase):
    def test_mask_upper_then_tight_chain_is_bounded(self):
        image = Image.new("RGB", (1000, 500))
        face = faces.FaceBox((100, 100, 180, 180), 0.8, 0, support=3)
        mask = np.zeros((500, 1000), dtype=bool)
        mask[90:460, 60:420] = True
        view = Image.new("RGB", (360, 370), "red")
        regions = {
            0: instance_regions.InferenceRegion(
                view,
                (60, 90, 420, 460),
                "instance_mask",
                [],
                confidence=0.7,
                proposal_box=(40, 80, 440, 480),
                mask=mask,
            )
        }
        segmenter = SimpleNamespace(segment=Mock(return_value=regions))
        uncertain = base.ModelTags({"1girl": 0.4, "1boy": 0.3}, {}, {})
        wd, camie = Mock(), Mock()
        for model in (wd, camie):
            model.predict_images.side_effect = (
                lambda *args, **kwargs: [uncertain] * len(args[0])
            )
        result = census.analyze_people(
            image,
            [face],
            wd,
            camie,
            people=[subjects.PersonBox((40, 80, 440, 480), 0.8)],
            segmenter=segmenter,
        )
        person = result["people"][0]
        self.assertEqual("unknown", person["gender"])
        self.assertEqual("face_crop_tight", person["context_mode"])
        self.assertTrue(person["tight_crop"])
        # initial full-mask view + upper band + face band + tight crop.
        self.assertEqual(4, wd.predict_images.call_count)
        self.assertEqual(4, camie.predict_images.call_count)

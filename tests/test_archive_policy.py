"""Fused-score archive policy and cached startup migration regressions."""

# ruff: noqa: PT009
import asyncio
from dataclasses import replace
import importlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import test_subject_features as base

policy = importlib.import_module("_painting_tests.services.archive_policy")


class ArchivePolicyTests(unittest.TestCase):
    def test_explicit_non_human_tags_override_incidental_gender_logits(self):
        cases = (
            # Actual fused evidence from 赛尔号_克多尔_默认.png.
            (
                {"no_humans": 0.7906, "pokemon_(creature)": 0.9676},
                {"wd": [0.0046, 0.0271], "camie": [0.3374, 0.7048]},
                "Others",
            ),
            # Actual fused evidence from 赛尔号_潇潇_默认.png.
            (
                {"no_humans": 0.6773, "pokemon_(creature)": 0.5836},
                {"wd": [0.0083, 0.0146], "camie": [0.5955, 0.6082]},
                "Others",
            ),
            ({"no_humans": 0.8}, {"wd": [0.9, 0.9], "camie": [0.9, 0.9]}, "Others"),
        )
        for tags, scores, expected in cases:
            with self.subTest(tags=tags):
                self.assertEqual(
                    expected,
                    policy.target_library(
                        "wives",
                        "gender_uncertain",
                        "unknown",
                        {},
                        scores,
                        None,
                        None,
                        tags,
                    ),
                )

    def test_non_human_rule_requires_strong_direct_and_safe_corroboration(self):
        rejected = (
            {"no_humans": 0.6499, "pokemon_(creature)": 0.99},
            {"no_humans": 0.79},
            {"no_humans": 0.7, "pokemon_(creature)": 0.4999},
            {"no_humans": 0.7, "monster_girl": 0.99},
            {"no_humans": 0.7, "robot_girl": 0.99},
            {"no_humans": 0.7, "mecha_musume": 0.99},
            {"no_humans": float("nan"), "animal_focus": 0.99},
        )
        high_gender = {"wd": [0.8, 0.8], "camie": [0.8, 0.8]}
        for tags in rejected:
            with self.subTest(tags=tags):
                self.assertIsNone(
                    policy.target_library(
                        "wives",
                        "gender_uncertain",
                        "unknown",
                        {},
                        high_gender,
                        None,
                        None,
                        tags,
                    )
                )

    def test_others_requires_both_real_fused_scores_strictly_below_thirty(self):
        for library in ("wives", "husbands"):
            for male, female, expected in (
                (0.1, 0.2, "Others"),
                (0.2999, 0.2999, "Others"),
                (0.3, 0.2, None),
                (0.2, 0.3, None),
                (0.1, 0.8, None),
            ):
                self.assertEqual(
                    expected,
                    policy.target_library(
                        library,
                        "no_person",
                        "unknown",
                        {},
                        {"wd": [male, female], "camie": [male, female]},
                    ),
                )
        # Without scene corroboration a single model's weak gender read is
        # NOT enough to archive: the other model may simply fail on the art
        # style (stylized characters, conflicting taggers stay in library).
        self.assertIsNone(
            policy.target_library(
                "wives",
                "no_person",
                "unknown",
                {},
                {"wd": [0, 0], "camie": [0.8, 0.8]},
            ),
        )
        # With the scene itself admitting no humans, one model's solo gender
        # read IS a phantom: consensus rules and the image archives.
        self.assertEqual(
            "Others",
            policy.target_library(
                "wives",
                "no_person",
                "unknown",
                {},
                {"wd": [0, 0], "camie": [0.8, 0.8]},
                scene_tags={"no_humans": 0.6},
            ),
        )
        # ...unless BOTH models are near-certain about a subject lead; a
        # moderate consensus (0.75) on a creature read is still a phantom.
        self.assertEqual(
            "Others",
            policy.target_library(
                "wives",
                "gender_uncertain",
                "unknown",
                {},
                {"wd": [0, 0], "camie": [0.8, 0.8]},
                subject_scores={"wd": [0.05, 0.75], "camie": [0.1, 0.8]},
                scene_tags={"no_humans": 0.6},
            ),
        )
        self.assertIsNone(
            policy.target_library(
                "wives",
                "gender_uncertain",
                "unknown",
                {},
                {"wd": [0, 0], "camie": [0.8, 0.8]},
                subject_scores={"wd": [0.02, 0.85], "camie": [0.05, 0.9]},
                scene_tags={"no_humans": 0.6},
            ),
        )
        for data in (
            None,
            {},
            {"wd": [0, 0]},
            {"wd": [0, 0], "camie": [float("nan"), 0]},
            {"wd": [0, 0], "camie": [-1, 0]},
        ):
            self.assertIsNone(
                policy.target_library("wives", "no_person", "unknown", {}, data)
            )
        self.assertIsNone(
            policy.target_library(
                "Others",
                "no_person",
                "unknown",
                {},
                {"wd": [0, 0], "camie": [0, 0]},
            )
        )

    def test_existing_low_scores_move_to_others_and_handle_name_collisions(self):
        for identical in (True, False):
            with (
                self.subTest(identical=identical),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                wives, others = root / "wives", root / "Others"
                wives.mkdir()
                others.mkdir()
                path = wives / "game_nonperson.png"
                path.write_bytes(b"new image")
                os.utime(path, (100, 100))
                existing = others / path.name
                existing.write_bytes(b"new image" if identical else b"original")
                service = base.ImageFeatureService()
                service.store = base.ImageFeatureStore(root / "features.db")
                service.store.initialize()
                service._libraries = lambda: (("wives", wives),)
                value = replace(
                    base.prediction(),
                    subject_status="no_person",
                    subject_gender="unknown",
                    subject_tags={},
                    analysis={"scene_gender": {"wd": [0.1, 0.2], "camie": [0.1, 0.2]}},
                )
                stat = path.stat()
                service.store.save_prediction(
                    path,
                    "wives",
                    stat.st_size,
                    stat.st_mtime_ns,
                    base.service_module._sha256_file(path),
                    value,
                )
                self.assertEqual(1, asyncio.run(service._scan(initial=True)))
                with (
                    patch.object(
                        base.service_module.paths, "OTHERS_IMAGES_FOLDER", str(others)
                    ),
                    patch.object(
                        base.service_module.ModelManager, "predict_paths"
                    ) as infer,
                ):
                    asyncio.run(service._process_batch(service._take_batch()))
                    infer.assert_not_called()
                self.assertFalse(path.exists())
                target = existing if identical else others / "game_nonperson_重复1.png"
                self.assertEqual(b"new image", target.read_bytes())
                if not identical:
                    self.assertEqual(b"original", existing.read_bytes())
                self.assertEqual(set(), service.store.archive_candidates())
                self.assertEqual(
                    "Others",
                    service.store.get_cached(
                        target, stat.st_size, stat.st_mtime_ns
                    ).library,
                )

    def test_startup_reuses_cached_non_human_tags_without_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wives, others = root / "wives", root / "Others"
            wives.mkdir()
            others.mkdir()
            path = wives / "赛尔号_潇潇_默认.png"
            path.write_bytes(b"cached non-human")
            os.utime(path, (100, 100))
            service = base.ImageFeatureService()
            service.store = base.ImageFeatureStore(root / "features.db")
            service.store.initialize()
            service._libraries = lambda: (("wives", wives),)
            value = replace(
                base.prediction(),
                subject_status="weak_detection",
                subject_gender="unknown",
                subject_tags={},
                general_tags={
                    "no_humans": 0.6773,
                    "pokemon_(creature)": 0.5836,
                },
                analysis={
                    "scene_gender": {
                        "wd": [0.0083, 0.0146],
                        "camie": [0.5955, 0.6082],
                    }
                },
            )
            stat = path.stat()
            service.store.save_prediction(
                path,
                "wives",
                stat.st_size,
                stat.st_mtime_ns,
                base.service_module._sha256_file(path),
                value,
            )
            self.assertEqual(1, asyncio.run(service._scan(initial=True)))
            with (
                patch.object(
                    base.service_module.paths, "OTHERS_IMAGES_FOLDER", str(others)
                ),
                patch.object(
                    base.service_module.ModelManager, "predict_paths"
                ) as infer,
            ):
                asyncio.run(service._process_batch(service._take_batch()))
                infer.assert_not_called()
            self.assertFalse(path.exists())
            self.assertTrue((others / path.name).exists())

    def test_strict_seventy_percent_boundary_both_directions(self):
        for library, gender, tag, destination in (
            ("wives", "male", "1boy", "husbands"),
            ("husbands", "female", "1girl", "wives"),
        ):
            for score in (0.69, 0.70, 0.7001, 0.95):
                with self.subTest(library=library, score=score):
                    self.assertEqual(
                        destination if score > 0.70 else None,
                        policy.target_library(
                            library, "confident", gender, {tag: score}
                        ),
                    )

    def test_uses_fused_subject_tags_not_minimum_or_scene_scores(self):
        value = replace(
            base.prediction(),
            subject_gender="male",
            male_probability=0.8455,
            female_probability=0.0391,
            subject_tags={"1boy": 0.9076, "male_focus": 0.9313},
        )
        self.assertEqual(
            "husbands", base.ImageFeatureService._target_library("wives", value)
        )
        value = replace(value, male_probability=0.99, subject_tags={"1boy": 0.70})
        self.assertIsNone(base.ImageFeatureService._target_library("wives", value))

    def test_uncertain_conflicting_missing_or_same_gender_never_moves(self):
        cases = (
            ("gender_uncertain", "male", {"1boy": 0.99}),
            ("confident", "unknown", {"1boy": 0.99}),
            ("confident", "female", {"1boy": 0.99}),
            ("confident", "male", {}),
            ("confident", "male", {"1boy": float("nan")}),
            ("confident", "male", {"1boy": 0.95, "1girl": 0.8}),
            ("confident", "male", {"1boy": 0.75, "1girl": 0.69}),
        )
        for status, gender, tags in cases:
            with self.subTest(status=status, gender=gender, tags=tags):
                self.assertIsNone(policy.target_library("wives", status, gender, tags))

    def test_startup_requeues_existing_candidate_and_moves_using_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wives, husbands = root / "wives", root / "husbands"
            wives.mkdir()
            husbands.mkdir()
            path = wives / "AliceFiction_00570.png"
            path.write_bytes(b"test-image")
            os.utime(path, (100, 100))
            stat = path.stat()
            service = base.ImageFeatureService()
            service.store = base.ImageFeatureStore(root / "features.db")
            service.store.initialize()
            service._libraries = lambda: (("wives", wives), ("husbands", husbands))
            value = replace(
                base.prediction(),
                subject_gender="male",
                subject_tags={"1boy": 0.9076},
                male_probability=0.8455,
            )
            service.store.save_prediction(
                path,
                "wives",
                stat.st_size,
                stat.st_mtime_ns,
                base.service_module._sha256_file(path),
                value,
            )
            self.assertFalse(
                service.store.needs_processing(path, stat.st_size, stat.st_mtime_ns)
            )
            self.assertEqual(1, asyncio.run(service._scan(initial=True)))
            batch = service._take_batch()
            with patch.object(
                base.service_module.ModelManager, "predict_paths"
            ) as infer:
                asyncio.run(service._process_batch(batch))
                infer.assert_not_called()
            self.assertFalse(path.exists())
            target = husbands / path.name
            self.assertEqual(b"test-image", target.read_bytes())
            self.assertEqual(set(), service.store.archive_candidates())
            self.assertEqual(0, asyncio.run(service._scan(initial=True)))


if __name__ == "__main__":
    unittest.main()


class PhantomGenderTests(unittest.TestCase):
    def test_mascot_with_single_model_phantom_moves_to_others(self):
        # Real case: a Seer elf - wd sees no humans (0.008/0.009), camie
        # phantom-reads "1boy 0.54"; camie's crop scores claim girl 0.70.
        # Neither establishes a person, so the image belongs in Others.
        decision = policy.target_library(
            "wives",
            "gender_uncertain",
            "unknown",
            {},
            scene_gender={
                "wd": [0.008, 0.009],
                "camie": [0.543, 0.366],
            },
            subject_scores={
                "wd": [0.075, 0.042],
                "camie": [0.608, 0.705],
            },
            scene_tags={"no_humans": 0.77},
        )
        self.assertEqual("Others", decision)

    def test_conflicting_or_style_blind_taggers_stay_in_library(self):
        # Real false positives of the naive min rule: stylized characters
        # where WD outputs near-zero or opposite logits.  No scene evidence
        # of non-humans -> never archive a clearly gendered character.
        cases = (
            # Reverse:1999 rabbit lady - WD style-blind, camie reads male.
            {"wd": [0.061, 0.016], "camie": [0.873, 0.433]},
            # Technoroid - the two taggers directly contradict each other.
            {"wd": [0.882, 0.136], "camie": [0.27, 0.998]},
            # Project Sekai - WD style-blind, camie reads female.
            {"wd": [0.005, 0.049], "camie": [0.417, 0.59]},
        )
        for scene in cases:
            self.assertIsNone(
                policy.target_library(
                    "wives",
                    "gender_uncertain",
                    "unknown",
                    {},
                    scene,
                    {"wd": [0.1, 0.1], "camie": [0.5, 0.6]},
                    scene_tags={"no_humans": 0.1},
                )
            )

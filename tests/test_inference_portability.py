"""Portable cache, raw-score archive and real cross-image scheduling contracts."""

# ruff: noqa: PT009
from dataclasses import replace
import importlib
from pathlib import Path
import sqlite3
import tempfile
from threading import Lock
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
import test_subject_features as base

ensemble = importlib.import_module("_painting_tests.services.vision.ensemble")
policy = importlib.import_module("_painting_tests.services.archive_policy")


class PortabilityTests(unittest.TestCase):
    def test_subject_refresh_only_invalidates_affected_old_decisions_once(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "features.db"
            store = base.ImageFeatureStore(database)
            store.initialize()
            store.save_prediction(
                "wives/good.png", "wives", 10, 20, "good", base.prediction()
            )
            store.save_prediction(
                "wives/weak.png",
                "wives",
                10,
                20,
                "weak",
                replace(base.prediction(), subject_status="weak_detection"),
            )
            with store._connect() as connection:
                connection.execute(
                    "DELETE FROM feature_metadata WHERE key = 'subject_selection_v3'"
                )
            refreshed = base.ImageFeatureStore(database)
            refreshed.initialize()
            self.assertIsNotNone(refreshed.find_by_hash("good"))
            self.assertIsNone(refreshed.find_by_hash("weak"))
            self.assertTrue(refreshed.needs_processing("wives/weak.png", 10, 20))
            refreshed.save_prediction(
                "wives/weak.png",
                "wives",
                10,
                20,
                "weak",
                replace(base.prediction(), subject_status="weak_detection"),
            )
            restarted = base.ImageFeatureStore(database)
            restarted.initialize()
            self.assertIsNotNone(restarted.find_by_hash("weak"))

    def test_timestamp_only_change_reuses_content_after_reconciliation(self):
        with tempfile.TemporaryDirectory() as directory:
            store = base.ImageFeatureStore(Path(directory) / "features.db")
            store.initialize()
            store.save_prediction(
                "D:/bot/wives/a.png", "wives", 10, 20, "old", base.prediction()
            )
            store.reconcile_library("wives", {"wives/a.png": (10, 30)})
            self.assertIsNotNone(store.find_by_hash("old"))
            self.assertIsNone(store.find_by_hash("different-content"))
            self.assertFalse(store.find_paths_by_tags("wives", ("black_hair",)))

    def test_drive_root_and_case_independent_keys(self):
        key = "wives/游戏_角色.png"
        for path in (r"C:\old\wives\游戏_角色.PNG", "D:/new/wives/游戏_角色.png", key):
            self.assertEqual(key, base.normalize_image_path(path))
        self.assertNotEqual(
            key, base.normalize_image_path("D:/new/husbands/游戏_角色.png")
        )

    def test_migration_keeps_best_row_and_index_with_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "features.db"
            store = base.ImageFeatureStore(database)
            store.initialize()
            store.save_prediction(
                "C:/old/wives/game_name.png",
                "wives",
                10,
                20,
                "digest",
                base.prediction(),
            )
            with store._connect() as connection:
                original_id = connection.execute(
                    "SELECT id FROM image_features"
                ).fetchone()[0]
                connection.execute(
                    "UPDATE image_features SET path = ?",
                    ("D:/new/wives/game_name.png",),
                )
                # An older installation row must not win over the current tag.
                connection.execute(
                    "INSERT INTO image_features (path, library, file_size, mtime_ns, "
                    "model_version, status, processed_at) "
                    "VALUES (?, 'wives', 1, 2, ?, 'missing', 1)",
                    ("C:/old/wives/game_name.png", base.store_module.MODEL_VERSION),
                )
                connection.execute("DELETE FROM feature_metadata")
            migrated = base.ImageFeatureStore(database)
            migrated.initialize()
            cached = migrated.get_cached("E:/another/wives/game_name.png", 10, 20)
            self.assertEqual("digest", cached.content_sha256)
            self.assertEqual(
                {"wives/game_name.png"},
                migrated.find_paths_by_tags("wives", ("black_hair",)),
            )
            with migrated._connect() as connection:
                rows = connection.execute(
                    "SELECT id, path FROM image_features"
                ).fetchall()
                self.assertEqual(
                    [(original_id, "wives/game_name.png")], [tuple(row) for row in rows]
                )
            backup = database.with_suffix(".pre_relative_paths.bak")
            connection = sqlite3.connect(backup)
            try:
                self.assertEqual(
                    2,
                    connection.execute(
                        "SELECT COUNT(*) FROM image_features"
                    ).fetchone()[0],
                )
            finally:
                connection.close()
            # A real replacement still invalidates the old signature.
            self.assertIsNone(
                migrated.get_cached("E:/another/wives/game_name.png", 11, 21)
            )

    def test_raw_opposite_gender_and_subject_scene_conflict(self):
        raw = {"wd": [0.8, 0.75], "camie": [0.8, 0.75]}
        self.assertIsNone(
            policy.target_library(
                "wives", "confident", "male", {"1boy": 0.8}, subject_scores=raw
            )
        )
        raw = {"wd": [0.1, 0.95], "camie": [0.1, 0.95]}
        scene = {"wd": [0.1, 0.1], "camie": [0.1, 0.1]}
        self.assertIsNone(
            policy.target_library(
                "wives", "confident", "female", {"1girl": 0.95}, scene, raw
            )
        )
        self.assertEqual(
            "wives",
            policy.target_library(
                "husbands", "confident", "female", {}, subject_scores=raw
            ),
        )


class BatchTests(unittest.TestCase):
    def test_failed_batches_shrink_without_dropping_results(self):
        module = importlib.import_module("_painting_tests.services.vision.taggers")
        model = module.BatchTagger.__new__(module.BatchTagger)
        model.batch_size = 8
        model.input_name = "images"
        model.prepare = lambda image: np.zeros((1,), dtype=np.float32)
        model.decode = lambda row: base.ModelTags({}, {}, {})
        sizes = []

        def run(_outputs, inputs):
            count = len(inputs["images"])
            sizes.append(count)
            if count > 2:
                raise RuntimeError("simulated memory limit")
            return [np.zeros((count, 1))]

        model.session = Mock()
        model.session.run.side_effect = run
        results = model.predict_images([Image.new("RGB", (1, 1))] * 11)
        self.assertEqual(11, len(results))
        self.assertTrue(all(isinstance(item, base.ModelTags) for item in results))
        self.assertEqual(2, model.batch_size)
        self.assertEqual([2, 1], sizes[-2:])

    def model(self):
        model = ensemble.EnsembleModel.__new__(ensemble.EnsembleModel)
        model._lock = Lock()
        model.detector = Mock()
        model.detector.detect.return_value = [base.PersonBox((0, 0, 100, 200), 0.9)]
        tags = base.ModelTags({"1girl": 0.99, "1boy": 0.01}, {}, {})
        model.wd = Mock()
        model.camie = Mock()
        for tagger in (model.wd, model.camie):
            tagger.predict_images.side_effect = lambda images: [tags] * len(images)
        return model

    def test_cross_image_batch_and_corrupt_file_isolation(self):
        model = self.model()
        image = Image.new("RGB", (100, 200))
        with patch.object(
            ensemble, "open_rgb", side_effect=[image, ValueError("bad image"), image]
        ):
            results = model.predict_paths([Path("a"), Path("bad"), Path("b")])
        self.assertEqual(3, len(results))
        self.assertIsInstance(results[1], Exception)
        self.assertEqual("female", results[0].subject_gender)
        self.assertEqual("female", results[2].subject_gender)
        for tagger in (model.wd, model.camie):
            self.assertEqual(1, tagger.predict_images.call_count)
            self.assertEqual(4, len(tagger.predict_images.call_args.args[0]))

    def test_weak_candidate_rechecked_without_guessing_group(self):
        model = self.model()
        weak = base.PersonBox((0, 0, 100, 200), 0.39)
        refined = replace(weak, confidence=0.9)
        model.detector.detect.side_effect = [[weak], [refined]]
        with patch.object(
            ensemble, "open_rgb", return_value=Image.new("RGB", (100, 200))
        ):
            result = model._predict_one(Path("a"))
        self.assertEqual("confident", result.subject_status)
        self.assertEqual("candidate", result.analysis["selection_status"])

    def test_refinement_recomputes_selection_reason(self):
        model = self.model()
        weak = base.PersonBox((0, 0, 100, 200), 0.45)
        model.detector.detect.side_effect = [[weak], [replace(weak, confidence=0.9)]]
        with patch.object(
            ensemble, "open_rgb", return_value=Image.new("RGB", (100, 200))
        ):
            result = model._predict_one(Path("a"))
        self.assertEqual("candidate", result.analysis["selection_status"])
        self.assertEqual("full_body", result.analysis["subject_view"])


if __name__ == "__main__":
    unittest.main()

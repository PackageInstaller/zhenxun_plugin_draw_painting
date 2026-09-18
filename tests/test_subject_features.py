"""Run from the bot root: python -m unittest discover -s <plugin>/tests.

No model inference, production database writes or bot startup hooks are run.
"""

# Standard-library unittest runner; keep its diagnostic assertion methods.
# ruff: noqa: PT009

import asyncio
from dataclasses import replace
import importlib
import logging
from pathlib import Path
import sqlite3
import sys
import tempfile
import types
import unittest

# Load only service modules, bypassing plugin registration and record.db setup.
package = types.ModuleType("_painting_tests")
package.__path__ = [str(Path(__file__).resolve().parents[1])]
sys.modules[package.__name__] = package
test_log = types.ModuleType("zhenxun.services.log")
test_log.logger = logging.getLogger("painting-tests")
previous_log = sys.modules.get("zhenxun.services.log")
sys.modules["zhenxun.services.log"] = test_log
try:
    request_module = importlib.import_module("_painting_tests.services.draw_request")
    store_module = importlib.import_module(
        "_painting_tests.services.image_feature_store"
    )
    service_module = importlib.import_module("_painting_tests.services.image_features")
    name_module = importlib.import_module("_painting_tests.services.painting_name")
    fusion_module = importlib.import_module("_painting_tests.services.vision.fusion")
    subject_module = importlib.import_module("_painting_tests.services.vision.subjects")
    types_module = importlib.import_module("_painting_tests.services.vision.types")
finally:
    if previous_log is None:
        sys.modules.pop("zhenxun.services.log", None)
    else:
        sys.modules["zhenxun.services.log"] = previous_log

find_character_variants = request_module.find_character_variants
parse_draw_request = request_module.parse_draw_request
ImageFeatureStore = store_module.ImageFeatureStore
normalize_image_path = store_module.normalize_image_path
ImageFeatureService = service_module.ImageFeatureService
_QueuedImage = service_module._QueuedImage
parse_painting_name = name_module.parse_painting_name
fuse_general = fusion_module.fuse_general
subject_gender = fusion_module.subject_gender
PersonBox = subject_module.PersonBox
select_subject = subject_module.select_subject
ModelTags = types_module.ModelTags
TagPrediction = types_module.TagPrediction


def prediction() -> TagPrediction:
    return TagPrediction(
        100,
        200,
        0.01,
        0.99,
        {"white_hair": 0.95, "1boy": 0.9, "1girl": 0.99},
        {},
        {},
        subject_tags={"black_hair": 0.9, "long_hair": 0.8, "1girl": 0.99},
        subject_status="confident",
        subject_gender="female",
        analysis={"subject_box": [0, 0, 100, 200]},
    )


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = ImageFeatureStore(self.root / "features.db")
        self.store.initialize()
        self.path = self.root / "wives" / "游戏_角色_原皮_备注.png"

    def tearDown(self):
        self.temp.cleanup()

    def save(self, value=None, path=None, library="wives"):
        self.store.save_prediction(
            path or self.path, library, 10, 20, "digest", value or prediction()
        )

    def test_subject_not_scene_and_library_gender(self):
        self.save()
        self.assertFalse(self.store.find_paths_by_tags("wives", ("white_hair",)))
        expected = {normalize_image_path(self.path)}
        self.assertEqual(
            expected,
            self.store.find_paths_by_tags(
                "wives", ("black_hair", "long_hair", "black_hair")
            ),
        )
        self.save(replace(prediction(), subject_gender="male"))
        self.assertFalse(self.store.find_paths_by_tags("wives", ("black_hair",)))
        self.save(replace(prediction(), subject_status="gender_uncertain"))
        self.assertFalse(self.store.find_paths_by_tags("wives", ("black_hair",)))

    def test_prediction_round_trip(self):
        self.save()
        cached = self.store.get_cached(self.path, 10, 20)
        self.assertEqual(cached.to_prediction(), prediction())

    def test_explicit_scene_counts_do_not_leak_other_scene_features(self):
        value = prediction()
        self.save(
            replace(value, general_tags={**value.general_tags, "multiple_girls": 0.9})
        )
        self.assertTrue(
            self.store.find_paths_by_tags("wives", ("multiple_girls", "black_hair"))
        )
        self.assertFalse(
            self.store.find_paths_by_tags("wives", ("multiple_girls", "white_hair"))
        )

    def test_deleted_rename_reuse_then_purge(self):
        self.save()
        self.assertEqual((1, 0), self.store.reconcile_library("wives", {}, now=100))
        self.assertFalse(self.store.find_paths_by_tags("wives", ("black_hair",)))
        self.assertIsNotNone(self.store.find_by_hash("digest"))
        self.assertEqual((0, 0), self.store.reconcile_library("wives", {}, now=200))
        self.assertEqual((0, 1), self.store.reconcile_library("wives", {}, now=86600))
        self.assertIsNone(self.store.find_by_hash("digest"))
        with self.store._connect() as connection:
            self.assertEqual(
                0,
                connection.execute(
                    "SELECT COUNT(*) FROM image_feature_tags"
                ).fetchone()[0],
            )

    def test_reappearing_file_not_purged(self):
        self.save()
        self.store.reconcile_library("wives", {}, now=100)
        signatures = {normalize_image_path(self.path): (10, 20)}
        self.assertEqual(
            (0, 0), self.store.reconcile_library("wives", signatures, now=100000)
        )
        self.assertIsNotNone(self.store.find_by_hash("digest"))

    def test_changed_file_invalidated_and_other_library_untouched(self):
        self.save()
        self.save(path=self.root / "husbands" / "游戏_男性.png", library="husbands")
        signatures = {normalize_image_path(self.path): (11, 21)}
        self.store.reconcile_library("wives", signatures)
        self.assertFalse(self.store.find_paths_by_tags("wives", ("black_hair",)))
        self.assertEqual(1, self.store.status_counts()["tagged"])
        self.assertEqual(1, self.store.status_counts()["pending"])

    def test_scan_finds_offline_deletion_and_skips_unavailable_folder(self):
        service = ImageFeatureService()
        service.store = self.store
        folder = self.root / "wives"
        folder.mkdir()
        service._libraries = lambda: (("wives", folder),)
        self.save()
        asyncio.run(service._scan(initial=True))
        self.assertEqual(1, self.store.status_counts()["missing"])
        self.save()
        folder.rmdir()
        asyncio.run(service._scan(initial=False))
        self.assertEqual(1, self.store.status_counts()["tagged"])

    def test_replacement_during_inference_not_saved_or_moved(self):
        self.path.parent.mkdir()
        self.path.touch()
        service = ImageFeatureService()
        service.store = self.store
        service._store_and_maybe_move(
            _QueuedImage(self.path, "wives"),
            "old",
            prediction(),
            100,
            10,
            allow_move=True,
        )
        self.assertEqual({}, self.store.status_counts())
        self.assertTrue(self.path.exists())

    def test_name_suffix_and_exact_variant_identity(self):
        names = [
            "游戏_角色_变体_备注1_备注2_备注3.png",
            "游戏_角色_另一立绘.png",
            "游戏_角色二_变体.png",
            "游戏_角色.txt",
        ]
        for name in names:
            (self.root / name).touch()
        parsed = parse_painting_name(names[0])
        self.assertEqual(("变体", "备注1", "备注2", "备注3"), parsed.suffix)
        self.assertEqual(
            set(names[:2]), set(find_character_variants(self.root, names[0]))
        )
        self.assertEqual("角色.v2", parse_painting_name("游戏_角色.v2").character)
        request = parse_draw_request("抽老婆 游戏 -n 角色 -t 黑发+长发", "老婆")
        self.assertEqual(("黑发", "长发"), request.feature_names)

    def test_legacy_scene_index_not_reinterpreted(self):
        path = self.root / "legacy.db"
        with sqlite3.connect(path) as connection:
            connection.execute("""CREATE TABLE image_features (
                path TEXT PRIMARY KEY, content_sha256 TEXT, library TEXT NOT NULL,
                file_size INTEGER NOT NULL, mtime_ns INTEGER NOT NULL,
                model_version TEXT NOT NULL, width INTEGER, height INTEGER,
                male_probability REAL, female_probability REAL,
                general_tags_json TEXT NOT NULL DEFAULT '{}',
                character_tags_json TEXT NOT NULL DEFAULT '{}',
                rating_tags_json TEXT NOT NULL DEFAULT '{}', status TEXT NOT NULL,
                error TEXT, processed_at REAL NOT NULL, moved_from TEXT)""")
            connection.execute(
                """INSERT INTO image_features
                (path,library,file_size,mtime_ns,model_version,status,processed_at,
                 general_tags_json) VALUES (?,?,?,?,?,?,?,?)""",
                (
                    str(self.path),
                    "wives",
                    10,
                    20,
                    "old-model",
                    "tagged",
                    1,
                    '{"white_hair":0.99}',
                ),
            )
        connection.close()
        store = ImageFeatureStore(path)
        store.initialize()
        self.assertFalse(store.find_paths_by_tags("wives", ("white_hair",)))
        with store._connect() as connection:
            row = connection.execute("SELECT * FROM image_features").fetchone()
            self.assertEqual("legacy", row["subject_status"])
            self.assertIn("white_hair", row["general_tags_json"])
            self.assertEqual(
                "ok", connection.execute("PRAGMA integrity_check").fetchone()[0]
            )


class FusionTests(unittest.TestCase):
    def test_occluded_dominant_person_not_discarded_at_half_threshold(self):
        main = PersonBox((521, 503, 1284, 1773), 0.4668)
        companion = PersonBox((1179, 968, 1734, 1655), 0.6325)
        selected, reason = select_subject([companion, main], 2048, 2048)
        self.assertEqual(main, selected)
        self.assertEqual("overlapping_people", reason)
        unreliable = PersonBox(main.xyxy, 0.35)
        self.assertIsNone(select_subject([unreliable], 2048, 2048)[0])

    def test_refinement_cannot_switch_to_smaller_companion(self):
        main = PersonBox((100, 100, 900, 1500), 0.45)
        child = PersonBox((10, 900, 210, 1200), 0.99)
        self.assertEqual(main, subject_module.refine_subject_box(main, [child]))
        tight = PersonBox((100, 20, 790, 1380), 0.65)
        refined = subject_module.refine_subject_box(main, [child, tight])
        self.assertEqual((200, 120, 890, 1480), refined.xyxy)

    def test_focus_avoids_side_companion_and_retains_torso(self):
        main = PersonBox((720, 520, 1280, 1757), 0.627)
        child = PersonBox((1179, 968, 1734, 1655), 0.63)
        region = subject_module.subject_focus_box(main, [main, child])
        self.assertIsNotNone(region)
        focus = PersonBox(region, 1)
        self.assertEqual(0, subject_module.intersection(focus, child))
        self.assertGreater(region[3] - region[1], (1757 - 520) * 0.5)
        blocking = PersonBox((820, 600, 1180, 1600), 0.8)
        self.assertIsNone(subject_module.subject_focus_box(main, [main, blocking]))

    def test_high_consensus_requires_margin_in_each_model(self):
        wd = ModelTags({"1girl": 0.99, "1boy": 0.01}, {}, {})
        camie = ModelTags({"1girl": 0.93, "1boy": 0.60}, {}, {})
        self.assertEqual("female", subject_gender(wd, camie)[0])
        conflict = ModelTags({"1girl": 0.93, "1boy": 0.85}, {}, {})
        self.assertEqual("unknown", subject_gender(wd, conflict)[0])

    def test_overlap_and_ambiguous_dominance_are_excluded(self):
        large = PersonBox((100, 0, 900, 1000), 0.95)
        overlap = PersonBox((300, 200, 500, 500), 0.9)
        self.assertEqual(
            "overlapping_people", select_subject([large, overlap], 1000, 1000)[1]
        )
        left, right = (
            PersonBox((0, 0, 450, 1000), 0.9),
            PersonBox((550, 0, 1000, 1000), 0.9),
        )
        self.assertEqual(
            "ambiguous_dominance", select_subject([left, right], 1000, 1000)[1]
        )
        self.assertEqual(large, select_subject([large], 1000, 1000)[0])

    def test_conflict_and_multipeople_do_not_get_subject_gender(self):
        male = ModelTags({"1boy": 0.99, "1girl": 0.01}, {}, {})
        female = ModelTags({"1boy": 0.01, "1girl": 0.99}, {}, {})
        self.assertEqual("unknown", subject_gender(male, female)[0])
        multi = ModelTags({"1girl": 0.99, "multiple_girls": 0.9}, {}, {})
        self.assertEqual("multiple_in_crop", subject_gender(multi, multi)[1])
        self.assertEqual("male", subject_gender(male, male)[0])

    def test_low_scores_are_not_mistaken_for_missing_tags(self):
        wd = ModelTags({"long_hair": 0.9, "white_hair": 0.1}, {}, {})
        camie = ModelTags(
            {"long_hair": 0.9, "white_hair": 0.95, "new_tag": 0.8}, {}, {}
        )
        fused = fuse_general(wd, camie, strict=True)
        self.assertNotIn("white_hair", fused)
        self.assertIn("new_tag", fused)
        self.assertIn("long_hair", fused)


if __name__ == "__main__":
    unittest.main()

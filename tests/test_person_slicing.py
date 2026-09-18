"""Small-person recovery without changing group-role ranking."""

# ruff: noqa: PT009
from dataclasses import replace
import importlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import test_subject_features as base

slicing = importlib.import_module("_painting_tests.services.vision.slicing")
runtime = importlib.import_module("_painting_tests.services.vision.runtime")
policy = importlib.import_module("_painting_tests.services.archive_policy")
Box = base.PersonBox
Hit = slicing.SliceHit


class SlicingTests(unittest.TestCase):
    def test_grid_covers_far_edges_without_duplicate_offsets(self):
        self.assertEqual([0], slicing.starts(400, 1024))
        positions = slicing.starts(4096, 1024)
        self.assertEqual([0, 768, 1536, 2304, 3072], positions)
        self.assertEqual(1024, slicing.starts(2048, 1024)[-1])

    def test_easy_image_skips_expensive_slices(self):
        calls = []
        box = Box((0, 0, 1200, 1800), 0.9)

        def detect(image):
            calls.append(image.size)
            return [box]

        result, info = slicing.detect_with_slices(
            Image.new("RGB", (2048, 2048)), detect
        )
        self.assertEqual([box], result)
        self.assertEqual(1, len(calls))
        self.assertEqual(0, info["slice_passes"])

    def test_mapping_offset_small_subject_and_fragment_removal(self):
        hits = [
            Hit(Box((500, 600, 650, 900), 0.8), (0, 0, 1024, 1024)),
            Hit(Box((502, 602, 652, 902), 0.7), (300, 400, 1324, 1424)),
            Hit(Box((500, 700, 650, 900), 0.95), (500, 700, 1524, 1724), True),
        ]
        boxes = slicing.merge_hits(hits)
        self.assertEqual(1, len(boxes))
        self.assertTrue(boxes[0].small_verified)
        self.assertEqual((500, 600, 650, 900), boxes[0].xyxy)
        self.assertEqual("candidate", base.select_subject(boxes, 4096, 4096)[1])
        self.assertEqual(
            "weak_detection",
            base.select_subject([replace(boxes[0], small_verified=False)], 4096, 4096)[
                1
            ],
        )
        self.assertEqual([], slicing.merge_hits([hits[-1]]))

    def test_distinct_people_and_duplicate_source_do_not_fake_support(self):
        first = Hit(Box((100, 100, 300, 500), 0.9), (0, 0, 1024, 1024))
        second = Hit(Box((350, 100, 550, 500), 0.9), first.source)
        boxes = slicing.merge_hits([first, first, second])
        self.assertEqual(2, len(boxes))
        self.assertFalse(any(box.small_verified for box in boxes))

    def test_alpha_region_maps_coordinates_back_to_original(self):
        image = Image.new("RGB", (2048, 2048))
        image.info["painting_content_box"] = (500, 700, 1100, 1300)

        def detect(crop):
            return [Box((50, 60, 250, 400), 0.8)] if crop.size == (600, 600) else []

        boxes, _ = slicing.detect_with_slices(image, detect)
        self.assertEqual((550, 760, 750, 1100), boxes[0].xyxy)

    def test_budget_and_archive_guard(self):
        with patch.object(slicing, "MAX_SLICE_PASSES", 2):
            _, info = slicing.detect_with_slices(
                Image.new("RGB", (2048, 2048)), lambda image: []
            )
        self.assertEqual(2, info["slice_passes"])
        self.assertTrue(info["budget_exhausted"])
        self.assertIsNone(
            policy.target_library(
                "wives",
                "no_person",
                "unknown",
                {},
                {"wd": [0.1, 0.1], "camie": [0.1, 0.1]},
                detection=info,
            )
        )

    def test_alpha_metadata_and_hidden_rgb_are_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "alpha.png"
            image = Image.new("RGBA", (100, 100), (255, 0, 0, 0))
            image.paste((0, 0, 0, 255), (30, 40, 70, 80))
            image.save(path)
            rgb = runtime.open_rgb(path)
            self.assertEqual((100, 100), rgb.size)
            self.assertEqual((255, 255, 255), rgb.getpixel((0, 0)))
            self.assertEqual((22, 32, 78, 88), rgb.info["painting_content_box"])

    def test_selective_cache_refresh_runs_once(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "features.db"
            store = base.ImageFeatureStore(database)
            store.initialize()
            for name, value in (
                (
                    "large",
                    replace(
                        base.prediction(),
                        width=2048,
                        height=2048,
                        subject_status="no_person",
                    ),
                ),
                ("small", replace(base.prediction(), subject_status="no_person")),
                ("good", base.prediction()),
            ):
                store.save_prediction(f"wives/{name}.png", "wives", 10, 20, name, value)
            with store._connect() as connection:
                connection.execute(
                    "DELETE FROM feature_metadata WHERE key='sliced_detection_v1'"
                )
            migrated = base.ImageFeatureStore(database)
            migrated.initialize()
            self.assertIsNone(migrated.find_by_hash("large"))
            self.assertIsNotNone(migrated.find_by_hash("small"))
            self.assertIsNotNone(migrated.find_by_hash("good"))


if __name__ == "__main__":
    unittest.main()

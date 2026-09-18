"""Display thresholds and uncapped layouts do not change inference policy."""

# ruff: noqa: PT009
from dataclasses import replace
import importlib
from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw
import test_subject_features as base

presentation = importlib.import_module("_painting_tests.services.feature_presentation")
report = importlib.import_module("_painting_tests.services.feature_report")
census = importlib.import_module("_painting_tests.services.vision.census")
faces = importlib.import_module("_painting_tests.services.vision.faces")


class PresentationTests(unittest.TestCase):
    def test_inclusive_threshold_stable_order_and_finite_scores(self):
        self.assertEqual(
            list(
                presentation.display_tags(
                    {
                        "z": 0.5,
                        "a": 0.5,
                        "high": 0.99,
                        "low": 0.4999,
                        "nan": float("nan"),
                        "inf": float("inf"),
                    }
                )
            ),
            ["high", "a", "z"],
        )

    def test_no_top_n_cap_and_no_long_name_truncation(self):
        tags = {f"trait_{index:03}": 0.8 for index in range(60)}
        self.assertEqual(60, len(presentation.feature_labels(tags)))
        text = "long_feature_name_" * 20
        wrapped = presentation.wrap_text(text, report._font(23), 250)
        self.assertGreater(len(wrapped), 1)
        self.assertEqual(text, "".join(wrapped))

    def test_census_preserves_all_accepted_traits(self):
        tags = {f"trait_{index:03}": 0.8 for index in range(60)}
        model = base.ModelTags({**tags, "1girl": 0.99, "1boy": 0.01}, {}, {})
        face = faces.FaceBox((10, 10, 50, 50), 0.9, 0, support=3)
        result = census._decision(model, model, face)
        self.assertTrue(tags.keys() <= result["tags"].keys())

    def test_single_and_group_render_all_tags_inside_expanded_canvas(self):
        tags = {f"trait_{index:03}": 0.8 for index in range(60)}
        tags["excluded_low"] = 0.4999
        person = {
            "box": [0, 0, 80, 100],
            "gender": "female",
            "tags": tags,
            "female_probability": 0.99,
            "male_probability": 0.01,
        }
        for group in (False, True):
            with self.subTest(group=group), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "input.png"
                Image.new("RGB", (100, 200)).save(path)
                analysis = (
                    {
                        "census": {
                            "people": [person] * 5,
                            "detected": 5,
                            "analyzed": 5,
                            "female": 5,
                            "male": 0,
                            "unknown": 0,
                        }
                    }
                    if group
                    else {}
                )
                value = replace(
                    base.prediction(),
                    subject_tags=tags,
                    general_tags=tags,
                    analysis=analysis,
                )
                drawn = []
                original_text = ImageDraw.ImageDraw.text

                def record(draw, xy, text, *args, **kwargs):
                    drawn.append((xy, text))
                    return original_text(draw, xy, text, *args, **kwargs)

                with patch.object(ImageDraw.ImageDraw, "text", record):
                    rendered = report.render_feature_report(path, value)
                with Image.open(BytesIO(rendered)) as output:
                    self.assertGreater(output.height, 2000)
                    self.assertTrue(all(y + 28 <= output.height for (_, y), _ in drawn))
                text = "".join(line for _, line in drawn)
                self.assertNotIn("excluded_low", text)
                for index in range(60):
                    self.assertIn(f"trait_{index:03}", text)


if __name__ == "__main__":
    unittest.main()

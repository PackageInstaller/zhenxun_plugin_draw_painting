"""The displayed context must be the real final inference crop."""

# ruff: noqa: PT009
import importlib
import unittest
from unittest.mock import Mock, patch

import test_subject_features  # noqa: F401

overlay = importlib.import_module("_painting_tests.services.region_overlay")
people_report = importlib.import_module("_painting_tests.services.people_report")


class RegionOverlayTests(unittest.TestCase):
    def test_mapping_clips_to_display_and_rejects_invalid_boxes(self):
        self.assertEqual(
            (10, 20, 109, 219),
            overlay.map_region((-5, -5, 1100, 1100), (1000, 1000), (10, 20, 110, 220)),
        )
        for invalid in (None, [1, 2], [2, 2, 1, 1], [0, 0, float("nan"), 10]):
            self.assertIsNone(overlay.map_region(invalid, (100, 100), (0, 0, 100, 100)))

    def test_outline_mapping_clips_and_rejects_malformed_contours(self):
        self.assertEqual(
            [[(10, 20), (60, 20), (109, 219)]],
            overlay.map_outline(
                [[(-5, -5), (50, 0), (110, 100)]],
                (100, 100),
                (10, 20, 110, 220),
            ),
        )
        self.assertEqual(
            [],
            overlay.map_outline(
                [[(0, 0), (float("nan"), 10), (20, 20)]],
                (100, 100),
                (0, 0, 100, 100),
            ),
        )

    def test_instance_outline_final_input_and_face_share_person_color(self):
        draw, font = Mock(), Mock()
        font.getlength.return_value = 120
        person = {
            "box": (40, 40, 60, 60),
            "context_box": (30, 20, 70, 80),
            "context_mode": "face_crop_tight",
            "context_outline": [],
            "initial_context_box": (0, 0, 100, 100),
            "instance_box": (0, 0, 100, 100),
            "instance_outline": [[(0, 0), (100, 0), (50, 100)]],
            "tight_crop": True,
            "gender": "female",
        }
        with (
            patch.object(overlay, "dashed_rectangle") as dashed,
            patch.object(overlay, "draw_instance_outline") as instance,
        ):
            overlay.draw_person_regions(
                draw, [person], (100, 100), (10, 10, 210, 210), font
            )
        self.assertEqual(
            overlay.map_region(person["context_box"], (100, 100), (10, 10, 210, 210)),
            dashed.call_args.args[1],
        )
        self.assertEqual(
            overlay.map_outline(
                person["instance_outline"], (100, 100), (10, 10, 210, 210)
            ),
            instance.call_args.args[1],
        )
        self.assertEqual(overlay.person_color(0), instance.call_args.args[2])
        labels = [call.args[1] for call in draw.text.call_args_list]
        self.assertIn("#1 模型输入边界", labels)
        self.assertIn("#1 女性·人脸", labels)
        self.assertNotEqual(overlay.person_color(0), overlay.person_color(1))

    def test_instance_outline_takes_priority_over_context_outline(self):
        draw, font = Mock(), Mock()
        font.getlength.return_value = 100
        instance_outline = [[(10, 10), (50, 10), (30, 80)]]
        with patch.object(overlay, "draw_instance_outline") as instance:
            overlay.draw_person_regions(
                draw,
                [
                    {
                        "box": (20, 20, 30, 30),
                        "context_box": (0, 0, 80, 90),
                        "context_outline": [[(0, 0), (5, 0), (5, 5)]],
                        "instance_outline": instance_outline,
                        "gender": "unknown",
                    }
                ],
                (100, 100),
                (0, 0, 100, 100),
                font,
            )
        self.assertEqual(
            overlay.map_outline(instance_outline, (100, 100), (0, 0, 100, 100)),
            instance.call_args.args[1],
        )

    def test_missing_context_does_not_invent_inference_box(self):
        draw, font = Mock(), Mock()
        font.getlength.return_value = 100
        with patch.object(overlay, "dashed_rectangle") as dashed:
            overlay.draw_person_regions(
                draw,
                [{"box": (10, 10, 50, 50), "gender": "unknown"}],
                (100, 100),
                (0, 0, 100, 100),
                font,
            )
        dashed.assert_not_called()

    def test_report_distinguishes_mask_input_from_tight_crop_fallback(self):
        masked = {
            "context_box": (1, 2, 90, 100),
            "context_mode": "instance_mask",
            "context_outline": [[(1, 2), (90, 2), (40, 100)]],
            "instance_confidence": 0.876,
        }
        lines = people_report.person_region_lines(masked)
        self.assertIn("可见身体掩码", "".join(lines))
        self.assertIn("轮廓外置白", "".join(lines))
        self.assertIn("87.6%", "".join(lines))

        fallback = {
            "context_box": (20, 20, 50, 60),
            "context_mode": "face_crop_tight",
            "instance_outline": masked["context_outline"],
        }
        fallback_text = "".join(people_report.person_region_lines(fallback))
        self.assertIn("人脸紧裁回退", fallback_text)
        self.assertIn("仅作参考", fallback_text)

    def test_report_says_visible_outline_does_not_complete_occlusion(self):
        notes = people_report.region_explanation_notes(
            [
                {
                    "context_mode": "instance_mask",
                    "context_outline": [[(0, 0), (10, 0), (10, 10)]],
                },
                {
                    "context_mode": "face_crop_tight",
                    "instance_outline": [[(0, 0), (5, 0), (5, 5)]],
                },
            ]
        )
        text = "".join(notes)
        self.assertIn("可见身体", text)
        self.assertIn("不会补全遮挡", text)
        self.assertIn("最终送入", text)
        self.assertIn("未用于最终特征推理", text)


if __name__ == "__main__":
    unittest.main()

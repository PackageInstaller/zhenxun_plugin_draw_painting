"""Deterministic regressions for explainable visual-lead ranking."""

# ruff: noqa: PT009
import importlib
import json
import unittest

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import test_subject_features as base

salience = importlib.import_module("_painting_tests.services.vision.subject_salience")


def person(box, confidence=0.9, *, verified=False):
    return base.PersonBox(box, confidence, verified)


class SubjectSalienceTests(unittest.TestCase):
    def test_empty_and_single_person_keep_legacy_admission(self):
        image = Image.new("RGB", (1000, 1000), "white")
        self.assertEqual("no_person", salience.rank_subjects(image, [])["status"])

        result = salience.rank_subjects(image, [person((100, 50, 900, 950))])
        self.assertEqual("candidate", result["status"])
        self.assertEqual(0, result["selected_index"])
        self.assertEqual(1.0, result["stability"])

        tentative = salience.rank_subjects(
            image, [person((100, 50, 900, 950), 0.45)]
        )
        self.assertEqual("tentative_detection", tentative["status"])
        self.assertEqual(0, tentative["selected_index"])

        weak = salience.rank_subjects(image, [person((10, 10, 110, 110), 0.9)])
        self.assertEqual("weak_detection", weak["status"])
        self.assertIsNone(weak["selected_index"])

    def test_result_is_json_serializable_and_contains_explanations(self):
        image = Image.new("RGB", (600, 400), "#808080")
        boxes = [person((20, 20, 260, 380)), person((340, 20, 580, 380))]
        result = salience.rank_subjects(image, boxes)
        json.dumps(result, ensure_ascii=False)
        self.assertEqual(salience.SALIENCE_VERSION, result["version"])
        self.assertEqual(2, len(result["ranked"]))
        self.assertEqual(
            {
                "scale",
                "composition",
                "face_scale",
                "local_luma",
                "local_chroma",
                "global_luma",
                "global_chroma",
                "focus",
                "modeling",
                "exposure",
                "saturation",
                "emphasis",
                "frontness",
                "detector",
                "completeness",
                "appearance_reliability",
            },
            set(result["ranked"][0]["cues"]),
        )

    def test_symmetric_co_leads_abstain_and_order_does_not_break_tie(self):
        image = Image.new("RGB", (1000, 700), "#8c8c8c")
        left = person((80, 60, 450, 660))
        right = person((550, 60, 920, 660))
        forward = salience.rank_subjects(image, [left, right])
        reverse = salience.rank_subjects(image, [right, left])
        self.assertEqual("ambiguous_salience", forward["status"])
        self.assertEqual("ambiguous_salience", reverse["status"])
        self.assertIsNone(forward["selected_index"])
        self.assertEqual({0, 1}, set(forward["co_primary_indices"]))
        by_box_forward = {
            tuple(item["box"]): item["score"] for item in forward["ranked"]
        }
        by_box_reverse = {
            tuple(item["box"]): item["score"] for item in reverse["ranked"]
        }
        self.assertEqual(by_box_forward, by_box_reverse)

    def test_large_lead_beats_small_high_confidence_companion(self):
        image = Image.new("RGB", (1000, 1000), "#b0b0b0")
        main = person((80, 30, 760, 980), 0.65)
        companion = person((760, 420, 990, 900), 0.99)
        result = salience.rank_subjects(image, [companion, main])
        self.assertEqual("candidate", result["status"])
        self.assertEqual(1, result["selected_index"])
        self.assertGreaterEqual(result["margin"], 0.08)

    def test_contrast_and_focus_are_measured_without_forcing_a_close_choice(self):
        image = Image.new("RGB", (900, 500), "#808080")
        draw = ImageDraw.Draw(image)
        draw.rectangle((40, 40, 390, 460), fill="#7f7f7f")
        draw.rectangle((510, 40, 860, 460), fill="#ffff00")
        for y in range(50, 460, 12):
            draw.line((510, y, 860, y), fill="#000080", width=5)
        boxes = [person((40, 40, 390, 460)), person((510, 40, 860, 460))]
        result = salience.rank_subjects(image, boxes)
        cues = {item["index"]: item["cues"] for item in result["ranked"]}
        self.assertGreater(cues[1]["local_chroma"], cues[0]["local_chroma"])
        self.assertGreater(cues[1]["focus"], cues[0]["focus"])
        self.assertEqual(1, result["suggested_index"])

    def test_global_resize_and_mirror_preserve_the_decision(self):
        image = Image.new("RGB", (800, 600), "#aaaaaa")
        boxes = [person((60, 20, 540, 590)), person((580, 260, 790, 590))]
        original = salience.rank_subjects(image, boxes)
        resized = salience.rank_subjects(
            image.resize((1600, 1200)),
            [
                person(tuple(value * 2 for value in box.xyxy), box.confidence)
                for box in boxes
            ],
        )
        mirrored = salience.rank_subjects(
            image.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
            [
                person(
                    (800 - box.xyxy[2], box.xyxy[1], 800 - box.xyxy[0], box.xyxy[3]),
                    box.confidence,
                )
                for box in boxes
            ],
        )
        self.assertEqual(original["status"], resized["status"])
        self.assertEqual(original["selected_index"], resized["selected_index"])
        self.assertEqual(original["status"], mirrored["status"])
        self.assertEqual(original["selected_index"], mirrored["selected_index"])
        self.assertAlmostEqual(original["margin"], resized["margin"], delta=0.015)

    def test_alpha_whitespace_does_not_change_scale_or_composition(self):
        compact = Image.new("RGBA", (400, 400), (120, 120, 120, 255))
        padded = Image.new("RGBA", (800, 800), (255, 255, 255, 0))
        padded.alpha_composite(compact, (200, 200))
        one = salience.rank_subjects(compact, [person((40, 20, 360, 390))])
        two = salience.rank_subjects(padded, [person((240, 220, 560, 590))])
        one_cues = one["ranked"][0]["cues"]
        two_cues = two["ranked"][0]["cues"]
        self.assertAlmostEqual(one_cues["scale"], two_cues["scale"], delta=0.01)
        self.assertAlmostEqual(
            one_cues["composition"], two_cues["composition"], delta=0.01
        )

    def test_incomplete_candidate_set_always_abstains_for_groups(self):
        image = Image.new("RGB", (800, 500), "gray")
        boxes = [person((0, 0, 500, 500)), person((550, 100, 790, 490))]
        result = salience.rank_subjects(
            image, boxes, detection={"candidate_complete": False}
        )
        self.assertEqual("candidate_set_incomplete", result["status"])
        self.assertIsNone(result["selected_index"])

    def test_blur_reduces_the_focus_cue(self):
        texture = Image.new("RGB", (300, 400), "white")
        draw = ImageDraw.Draw(texture)
        for x in range(0, 300, 10):
            draw.line((x, 0, x, 400), fill="black", width=3)
        image = Image.new("RGB", (800, 500), "#808080")
        image.paste(texture, (30, 50))
        image.paste(texture.filter(ImageFilter.GaussianBlur(5)), (470, 50))
        result = salience.rank_subjects(
            image, [person((30, 50, 330, 450)), person((470, 50, 770, 450))]
        )
        cues = {item["index"]: item["cues"] for item in result["ranked"]}
        self.assertGreater(cues[0]["focus"], cues[1]["focus"])


if __name__ == "__main__":
    unittest.main()


class SalienceSubjectDecisionTests(unittest.TestCase):
    def test_only_confident_winners_become_decisions(self):
        people = [person((0, 0, 400, 900)), person((500, 100, 900, 950))]
        face_people = {0: 0, 1: 1}
        census_people = [
            {"gender": "unknown", "status": "gender_uncertain", "tags": {}},
            {
                "gender": "female",
                "status": "confident",
                "tags": {"long_hair": 0.95},
                "male_probability": 0.01,
                "female_probability": 0.97,
                "context_box": (500, 80, 920, 960),
                "context_mode": "instance_mask",
            },
        ]
        ambiguous = salience.salience_subject_decision(
            {"status": "ambiguous_salience", "selected_index": None},
            people,
            census_people,
            face_people,
        )
        self.assertIsNone(ambiguous)
        decision = salience.salience_subject_decision(
            {"status": "candidate", "selected_index": 1},
            people,
            census_people,
            face_people,
        )
        self.assertEqual(1, decision["person_index"])
        self.assertEqual((500, 100, 900, 950), decision["box"])
        self.assertEqual("female", decision["gender"])
        self.assertEqual({"long_hair": 0.95}, decision["tags"])
        self.assertEqual((500, 80, 920, 960), decision["context_box"])
        self.assertEqual("instance_mask", decision["context_mode"])

    def test_faceless_salient_person_stays_box_only(self):
        people = [person((0, 0, 400, 900)), person((500, 100, 900, 950))]
        decision = salience.salience_subject_decision(
            {"status": "candidate", "selected_index": 0}, people, [], {}
        )
        self.assertEqual(0, decision["person_index"])
        self.assertEqual("unknown", decision["gender"])
        self.assertEqual("no_face_anchor", decision["gender_status"])
        self.assertFalse(decision["tags"])

    def test_uncertain_census_entry_does_not_invent_gender(self):
        people = [person((0, 0, 400, 900))]
        decision = salience.salience_subject_decision(
            {"status": "candidate", "selected_index": 0},
            people,
            [{"gender": "unknown", "status": "multiple_in_crop", "tags": {}}],
            {0: 0},
        )
        self.assertEqual("unknown", decision["gender"])
        self.assertEqual("per_person_gender_uncertain", decision["gender_status"])
        self.assertEqual("face_crop", decision["context_mode"])


class VisibleBodyScaleTests(unittest.TestCase):
    def test_sparse_mask_reduces_scale_despite_equal_box(self):
        image = Image.new("RGB", (800, 500), "#909090")
        boxes = [person((30, 50, 330, 450)), person((470, 50, 770, 450))]
        dense = np.zeros((500, 800), dtype=bool)
        dense[50:450, 30:330] = True
        sparse = np.zeros((500, 800), dtype=bool)
        sparse[50:450, 470:770:8] = True
        sparse[50:450:8, 470:770] = True
        result = salience.rank_subjects(
            image, boxes, masks={0: dense, 1: sparse}
        )
        cues = {item["index"]: item["cues"] for item in result["ranked"]}
        self.assertGreater(cues[0]["scale"], cues[1]["scale"])


class ArtisticEmphasisTests(unittest.TestCase):
    """Emphasis cues (focus/modeling/exposure/saturation) pick the lit lead."""

    def _scene(self):
        # Dark, muted canvas; a bright vivid textured figure left, a dark
        # desaturated figure right whose box is larger (scalefavours it).
        image = Image.new("RGB", (800, 500), "#20242c")
        draw = ImageDraw.Draw(image)
        for x in range(0, 800, 14):
            draw.line((x, 0, x, 500), fill="#1a1d24", width=4)
        lead = Image.new("RGB", (220, 420), "#e8e2f2")
        d = ImageDraw.Draw(lead)
        for y in range(0, 420, 9):
            d.line((0, y, 220, y), fill="#5131a8", width=3)
        image.paste(lead, (40, 40))
        dark = Image.new("RGB", (300, 460), "#262a30")
        d = ImageDraw.Draw(dark)
        for y in range(0, 460, 30):
            d.line((0, y, 300, y), fill="#20242a", width=4)
        image.paste(dark, (480, 20))
        return image

    def test_lit_vivid_lead_beats_larger_dark_companion(self):
        image = self._scene()
        boxes = [
            person((40, 40, 260, 460), 0.85),
            person((480, 20, 780, 480), 0.85),
        ]
        lead_mask = np.zeros((500, 800), dtype=bool)
        lead_mask[40:460, 40:260] = True
        dark_mask = np.zeros((500, 800), dtype=bool)
        dark_mask[20:480, 480:780] = True
        result = salience.rank_subjects(
            image, boxes, masks={0: lead_mask, 1: dark_mask}
        )
        cues = {item["index"]: item["cues"] for item in result["ranked"]}
        self.assertGreater(cues[0]["exposure"], cues[1]["exposure"] + 0.2)
        self.assertGreater(cues[0]["saturation"], cues[1]["saturation"] + 0.1)
        self.assertEqual("candidate", result["status"])
        self.assertEqual(0, result["selected_index"])

    def test_box_proxies_cannot_use_the_emphasis_path(self):
        image = self._scene()
        boxes = [
            person((40, 40, 260, 460), 0.85),
            person((480, 20, 780, 480), 0.85),
        ]
        result = salience.rank_subjects(image, boxes)
        # Without exact masks the appearance reliability stays low, so the
        # emphasis-specific acceptance and promotion paths stay closed; a
        # selection can only come from the regular margin rules.
        self.assertFalse(result["promoted_by_emphasis"])

    def test_emphasis_promotion_requires_exact_mask_and_close_score(self):
        def cue(emphasis, reliability):
            focus, rest = divmod(emphasis * 4, 1)
            return {
                "focus": emphasis,
                "modeling": emphasis,
                "exposure": emphasis,
                "saturation": emphasis,
                "appearance_reliability": reliability,
            }

        people = [person((0, 0, 500, 900), 0.9), person((600, 0, 900, 900), 0.9)]
        scores = [0.66, 0.60]
        cues = [cue(0.45, 0.95), cue(0.80, 0.95)]
        # Emphasis leader (index 1, 0.80 vs 0.45) trails by only 0.06 and has
        # an exact mask: it takes the subject role.
        self.assertEqual(
            (1, True),
            salience._apply_emphasis_promotion(
                0, [0, 1], cues, scores, people, 1.0
            ),
        )
        # A box-proxy leader (reliability 0.55) must never be promoted.
        cues_proxy = [cue(0.45, 0.95), cue(0.80, 0.55)]
        self.assertEqual(
            (0, False),
            salience._apply_emphasis_promotion(
                0, [0, 1], cues_proxy, scores, people, 1.0
            ),
        )
        # Trailing too far overall also blocks promotion.
        scores_far = [0.80, 0.60]
        self.assertEqual(
            (0, False),
            salience._apply_emphasis_promotion(
                0, [0, 1], cues, scores_far, people, 1.0
            ),
        )


class FaceScaleTests(unittest.TestCase):
    def test_largest_face_marks_the_lead_in_group_art(self):
        image = Image.new("RGB", (800, 500), "#808080")
        boxes = [person((30, 50, 330, 450)), person((470, 50, 770, 450))]
        result = salience.rank_subjects(
            image, boxes, faces={0: (110, 110, 250, 250), 1: (540, 90, 620, 170)}
        )
        cues = {item["index"]: item["cues"] for item in result["ranked"]}
        self.assertGreater(cues[0]["face_scale"], cues[1]["face_scale"])
        self.assertAlmostEqual(
            (80 * 80) / (140 * 140), cues[1]["face_scale"] ** 2, delta=1e-6
        )

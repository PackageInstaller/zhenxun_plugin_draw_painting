"""Partial-name selection must preserve exact-name priority and ownership."""

# ruff: noqa: PT009, PT027
import unittest

import test_subject_features as base

select = base.request_module.select_character_images


class CharacterSearchTests(unittest.TestCase):
    def test_typo_matches_all_variants_and_respects_claimed_name(self):
        images = ["悠久之树_妮菲亚_觉醒.png", "悠久之树_妮菲亚_原皮.png"]
        self.assertEqual(
            images, select([*images, "悠久之树_索菲娅_原皮.png"], "妮菲娅")
        )
        self.assertEqual([], select(images, "妮菲娅", excluded_characters={"妮菲亚"}))
        self.assertEqual([], select(images, "妮娅"))

    def test_tied_typos_ask_for_name_but_exact_and_partial_keep_priority(self):
        images = ["游戏_妮菲亚_原皮.png", "游戏_妮菲雅_原皮.png"]
        with self.assertRaisesRegex(
            base.request_module.DrawRequestError, "多个相近人物"
        ):
            select(images, "妮菲娅")
        self.assertEqual(images[:1], select(images, "妮菲亚"))
        self.assertEqual(images, select(images, "妮菲"))

    def test_typo_supports_english_and_rejects_unrelated_names(self):
        images = ["游戏_Shooter_原皮.png"]
        self.assertEqual(images, select(images, "shootre"))
        self.assertEqual([], select(images, "another"))

    def setUp(self):
        self.images = [
            "碧蓝航线_BLACK★ROCK SHOOTER_原皮.png",
            "碧蓝航线_BLACK★ROCK SHOOTER_表情1.png",
            "碧蓝航线_BLACK★ROCK SHOOTER（后排）_黑之女神.png",
            "碧蓝航线_其他角色_shooter.png",
        ]

    def test_partial_latin_ignores_case_spacing_and_symbols(self):
        for query in ("shooter", "ＳＨＯＯＴＥＲ", "black rock", "BLACK★ROCK"):
            self.assertEqual(self.images[:3], select(self.images, query))
        request = base.parse_draw_request("抽老婆碧蓝航线 -n shooter -t 黑发", "老婆")
        self.assertEqual("碧蓝航线", request.game_name)
        self.assertEqual("shooter", request.character_name)
        self.assertEqual(("黑发",), request.feature_names)

    def test_exact_name_has_priority_and_does_not_fall_back_when_claimed(self):
        name = "BLACK★ROCK SHOOTER"
        self.assertEqual(self.images[:2], select(self.images, name))
        self.assertEqual(
            [],
            select(
                self.images,
                name,
                excluded_characters={
                    base.request_module.normalize_draw_name(name),
                },
            ),
        )

    def test_partial_excludes_claimed_canonical_character_and_all_variants(self):
        self.assertEqual(
            self.images[2:3],
            select(
                self.images,
                "shooter",
                excluded_characters={"black★rock shooter"},
            ),
        )

    def test_no_skin_remark_matching_and_no_empty_symbol_wildcard(self):
        for query in ("★", "", "不存在", "表情1", "黑之女神", "碧蓝航线"):
            self.assertEqual([], select(self.images, query))
        self.assertEqual(self.images[3:], select(self.images, "其他"))


if __name__ == "__main__":
    unittest.main()

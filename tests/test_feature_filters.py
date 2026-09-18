"""Mixed-language filters, AND semantics, and forward-message catalog tests."""

# ruff: noqa: PT009
import asyncio
from dataclasses import replace
import importlib
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import AsyncMock, Mock, patch

from nonebot.adapters.onebot.v11 import GroupMessageEvent, PrivateMessageEvent
import test_subject_features as base

aliases = importlib.import_module("_painting_tests.services.feature_aliases")
catalog = importlib.import_module("_painting_tests.services.feature_catalog")
stub = types.ModuleType("_painting_tests.matchers")
stub.feature_list = types.SimpleNamespace(handle=lambda: lambda fn: fn)
previous = sys.modules.get(stub.__name__)
sys.modules[stub.__name__] = stub
try:
    handler = importlib.import_module("_painting_tests.handlers.feature_filters")
finally:
    if previous is None:
        sys.modules.pop(stub.__name__, None)
    else:
        sys.modules[stub.__name__] = previous

COMMAND = "抽老婆 -t 粉发 粉瞳 看向观众 发饰  铠甲 手持物品 water"
TAGS = (
    "pink_hair",
    "pink_eyes",
    "looking_at_viewer",
    "hair_ornament",
    "armor",
    "holding",
    "water",
)


class FeatureFilterTests(unittest.TestCase):
    def setUp(self):
        self.registry = aliases.FeatureAliasRegistry()

    def test_expanded_aliases_resolve_and_deduplicate_mixed_input(self):
        request = base.parse_draw_request(
            "抽老婆 -t 水 water 持剑 holding_sword "
            "分离袖 白色夹克 球鞋 弓 bow_(weapon)",
            "老婆",
        )
        self.assertEqual(
            (
                "water",
                "holding_sword",
                "detached_sleeves",
                "white_jacket",
                "sneakers",
                "bow_(weapon)",
            ),
            self.registry.resolve_many(request.feature_names).tags,
        )
        self.assertEqual("弓", self.registry.display_name("bow_(weapon)"))
        self.assertEqual("蝴蝶结", self.registry.display_name("bow"))

    def test_every_translation_and_alias_round_trips_through_command_parser(self):
        count = 0
        for definitions in self.registry.grouped_definitions().values():
            for definition in definitions:
                count += 1
                for value in (definition.tag, definition.name, *definition.aliases):
                    request = base.parse_draw_request(f"抽老婆 -t {value}", "老婆")
                    self.assertEqual(
                        (definition.tag,),
                        self.registry.resolve_many(
                            request.feature_names,
                        ).tags,
                    )
                self.assertEqual(
                    definition.name, self.registry.display_name(definition.tag)
                )
        self.assertGreaterEqual(count, 435)

    def test_mixed_chinese_english_and_case_are_allowed(self):
        request = base.parse_draw_request(COMMAND, "老婆")
        with patch.object(
            self.registry, "_known_model_tags", return_value=frozenset(TAGS)
        ):
            resolved = self.registry.resolve_many(request.feature_names)
            self.assertEqual(TAGS, resolved.tags)
            self.assertEqual(
                ("pink_hair", "water"),
                self.registry.resolve_many(("粉发", "PINK_HAIR", "Water")).tags,
            )

    def test_missing_armor_excludes_image_but_other_six_match(self):
        with tempfile.TemporaryDirectory() as folder:
            store = base.ImageFeatureStore(Path(folder) / "test.db")
            store.initialize()
            path = Path(folder) / "AsterTatariqus_101001002_UnitImage_HiRes.png"
            scores = dict(
                zip(
                    TAGS,
                    (0.8423, 0.5058, 0.7833, 0.7641, 0, 0.8246, 0.7211),
                    strict=True,
                )
            )
            scores.pop("armor")
            store.save_prediction(
                path,
                "wives",
                1,
                1,
                "test",
                replace(
                    base.prediction(),
                    subject_tags=scores,
                ),
            )
            self.assertEqual(set(), store.find_paths_by_tags("wives", TAGS))
            self.assertEqual(
                {base.normalize_image_path(path)},
                store.find_paths_by_tags(
                    "wives",
                    tuple(tag for tag in TAGS if tag != "armor"),
                ),
            )

    def test_catalog_contains_every_bilingual_mapping_in_bounded_pages(self):
        pages = catalog.feature_list_messages(self.registry)
        self.assertGreater(len(pages), 2)
        self.assertTrue(all(len(page) <= catalog.PAGE_CHAR_LIMIT for page in pages))
        text = "\n".join(pages)
        for definitions in self.registry.grouped_definitions().values():
            for definition in definitions:
                self.assertIn(f"{definition.name} = {definition.tag}", text)
                for alias in definition.aliases:
                    self.assertIn(alias, text)

    def test_handler_sends_forward_nodes_in_groups_and_private_chats(self):
        for event_type, api in (
            (GroupMessageEvent, "send_group_forward_msg"),
            (PrivateMessageEvent, "send_private_forward_msg"),
        ):
            with self.subTest(api=api):
                event = Mock(spec=event_type, group_id=123, user_id=456)
                bot = Mock(self_id="789", call_api=AsyncMock(), send=AsyncMock())
                asyncio.run(handler.handle_feature_list(bot, event))
                bot.send.assert_not_called()
                bot.call_api.assert_awaited_once()
                args, kwargs = bot.call_api.call_args
                self.assertEqual(api, args[0])
                self.assertGreater(len(kwargs["messages"]), 2)
                self.assertTrue(
                    all(node["type"] == "node" for node in kwargs["messages"])
                )


if __name__ == "__main__":
    unittest.main()

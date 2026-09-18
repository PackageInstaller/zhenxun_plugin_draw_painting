"""Offline query/report tests. No bot startup, downloads or real model sessions."""

# ruff: noqa: PT009, PT027
import asyncio
from dataclasses import replace
import importlib
from io import BytesIO
import logging
from pathlib import Path
import sys
import tempfile
from threading import Event
import types
import unittest
from unittest.mock import AsyncMock, Mock, patch

import httpx
from nonebot.adapters.onebot.v11 import Message, MessageSegment
from PIL import Image
import test_subject_features as base

query = importlib.import_module("_painting_tests.services.feature_query")
report = importlib.import_module("_painting_tests.services.feature_report")
ensemble = importlib.import_module("_painting_tests.services.vision.ensemble")

# Import just the command function with registration/DB guards replaced, so no
# production SQLite repository or lifecycle hooks are initialized by tests.
stubs = {}
for name in (
    "_painting_tests.matchers",
    "_painting_tests.services.command_guard",
    "zhenxun.services.log",
):
    stubs[name] = sys.modules.get(name)
matchers = types.ModuleType("_painting_tests.matchers")
matchers.feature_query = types.SimpleNamespace(handle=lambda **_: lambda fn: fn)
guard = types.ModuleType("_painting_tests.services.command_guard")
guard.CommandHandler = types.SimpleNamespace(dependency=lambda **_: None)
log = types.ModuleType("zhenxun.services.log")
log.logger = logging.getLogger("feature-query-tests")
sys.modules[matchers.__name__] = matchers
sys.modules[guard.__name__] = guard
sys.modules[log.__name__] = log
try:
    handler = importlib.import_module("_painting_tests.handlers.feature_query")
finally:
    for name, previous in stubs.items():
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


def image_bytes() -> bytes:
    output = BytesIO()
    Image.new("RGB", (100, 200), "white").save(output, format="PNG")
    return output.getvalue()


class ReportTests(unittest.TestCase):
    def test_unknown_is_not_reported_as_zero(self):
        value = replace(
            base.prediction(),
            male_probability=0,
            female_probability=0,
            subject_status="weak_detection",
            subject_gender="unknown",
            analysis={},
        )
        text = report.gender_summary(value)
        self.assertIn("尚未计算", text)
        self.assertNotIn("0.00%", text)
        self.assertIn("暂不可用", report.gender_summary(None))

    def test_uncertain_computed_scores_are_explicitly_candidates(self):
        value = replace(
            base.prediction(),
            subject_status="gender_uncertain",
            analysis={"subject_gender": {"wd": [0.1, 0.9]}},
        )
        self.assertIn("候选区域", report.gender_summary(value))
        self.assertNotIn("尚未计算", report.gender_summary(value))

    def test_render_is_valid_jpeg_and_original_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "original.png"
            content = image_bytes()
            path.write_bytes(content)
            value = replace(
                base.prediction(),
                analysis={
                    "people": [{"box": [0, 0, 100, 200], "confidence": 0.8}],
                    "subject_box": [0, 0, 100, 200],
                    "feature_box": [15, 0, 85, 130],
                    "subject_view": "upper_body",
                },
            )
            rendered = report.render_feature_report(path, value)
            with Image.open(BytesIO(rendered)) as result:
                self.assertEqual("JPEG", result.format)
                self.assertGreater(result.width, 1000)
            self.assertEqual(content, path.read_bytes())
            self.assertIn("上半身", report.report_text(value))

    def test_temporary_query_does_not_call_library_store(self):
        captured = []

        def predict(path):
            captured.append(path)
            self.assertTrue(path.is_file())
            return base.prediction()

        with (
            patch.object(query.ModelManager, "predict_query", side_effect=predict),
            patch.object(
                base.service_module.image_feature_service,
                "tag_image",
                side_effect=AssertionError("must not touch library"),
            ),
        ):
            rendered = query._inspect_image(image_bytes())
        self.assertTrue(rendered)
        self.assertFalse(captured[0].exists())

    def test_invalid_or_oversized_input_never_reaches_models(self):
        with patch.object(query.ModelManager, "predict_query") as predict:
            with self.assertRaises(query.FeatureQueryError):
                query._inspect_image(b"not an image")
            with patch.object(query, "MAX_IMAGE_PIXELS", 100):
                with self.assertRaises(query.FeatureQueryError):
                    query._inspect_image(image_bytes())
            predict.assert_not_called()

    def test_attachment_and_reply_selection(self):
        first = MessageSegment.image("https://gchat.qpic.cn/first")
        second = MessageSegment.image("https://gchat.qpic.cn/second")
        self.assertEqual(
            first.data, query.select_image_data(Message(first), Message(second))
        )
        self.assertEqual(
            second.data, query.select_image_data(Message("特征查询"), Message(second))
        )
        self.assertIsNone(query.select_image_data(Message("特征查询")))
        with self.assertRaises(query.FeatureQueryError):
            query.select_image_data(Message([first, second]))

    def test_untrusted_urls_rejected(self):
        for url in (
            "file:///C:/secret.png",
            "http://127.0.0.1/a",
            "http://example.com/a",
            "https://qpic.cn.evil.test/a",
            "https://user:pass@gchat.qpic.cn/a",
        ):
            with self.assertRaises(query.FeatureQueryError, msg=url):
                query.validate_image_url(url)
        query.validate_image_url("https://gchat.qpic.cn/a?token=test")


class QueryAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancellation_waits_for_native_inference_cleanup(self):
        started, release = Event(), Event()

        def inspect(_content):
            started.set()
            release.wait(timeout=3)
            return b"image"

        with patch.object(query, "_inspect_image", side_effect=inspect):
            task = asyncio.create_task(query.inspect_query_image(b"data"))
            try:
                self.assertTrue(await asyncio.to_thread(started.wait, 2))
                task.cancel()
                await asyncio.sleep(0)
                self.assertFalse(task.done())
            finally:
                release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task

    async def test_download_and_blocked_redirect(self):
        real_client = httpx.AsyncClient

        def responses(request):
            if request.url.path == "/redirect":
                return httpx.Response(
                    302, headers={"location": "http://127.0.0.1/secret"}
                )
            return httpx.Response(200, content=b"image")

        with patch.object(
            query.httpx,
            "AsyncClient",
            side_effect=lambda **kwargs: real_client(
                transport=httpx.MockTransport(responses), **kwargs
            ),
        ):
            self.assertEqual(
                b"image", await query.download_query_image("https://gchat.qpic.cn/a")
            )
            with self.assertRaises(query.FeatureQueryError):
                await query.download_query_image("https://gchat.qpic.cn/redirect")

    async def test_streaming_size_limit(self):
        real_client = httpx.AsyncClient
        with (
            patch.object(query, "MAX_IMAGE_BYTES", 3),
            patch.object(
                query.httpx,
                "AsyncClient",
                side_effect=lambda **kwargs: real_client(
                    transport=httpx.MockTransport(
                        lambda request: httpx.Response(200, content=b"large")
                    ),
                    **kwargs,
                ),
            ),
        ):
            with self.assertRaises(query.FeatureQueryError):
                await query.download_query_image("https://gchat.qpic.cn/a")

    async def test_handler_quotes_response_and_releases_slot(self):
        bot = types.SimpleNamespace(
            self_id="bot", send=AsyncMock(), get_image=AsyncMock(), call_api=AsyncMock()
        )
        event = types.SimpleNamespace(
            user_id=123,
            message_id=-456,
            reply=None,
            message=Message(
                [
                    MessageSegment.text("特征查询"),
                    MessageSegment("image", {"url": "https://gchat.qpic.cn/a"}),
                ]
            ),
        )
        with (
            patch.object(
                handler, "download_query_image", AsyncMock(return_value=b"in")
            ),
            patch.object(
                handler,
                "inspect_query_image",
                AsyncMock(return_value=b"out"),
            ),
        ):
            await handler.handle_feature_query(bot, event, Message())
        self.assertEqual(1, bot.send.await_count)
        self.assertTrue(
            all(call.kwargs["reply_message"] for call in bot.send.await_args_list)
        )
        sent = bot.send.await_args_list[-1].args[1]
        self.assertIsInstance(sent, MessageSegment)
        self.assertEqual("image", sent.type)
        self.assertEqual(
            ["282", "478"], [c.kwargs["emoji_id"] for c in bot.call_api.await_args_list]
        )
        self.assertTrue(
            all(c.args == ("set_msg_emoji_like",) for c in bot.call_api.await_args_list)
        )
        self.assertTrue(
            all(c.kwargs["message_id"] == -456 for c in bot.call_api.await_args_list)
        )
        self.assertFalse(handler._active_users)

    async def test_handler_error_releases_slot(self):
        bot = types.SimpleNamespace(
            self_id="bot", send=AsyncMock(), get_image=AsyncMock(), call_api=AsyncMock()
        )
        event = types.SimpleNamespace(
            user_id=123,
            message_id=456,
            reply=None,
            message=Message(
                MessageSegment("image", {"url": "https://gchat.qpic.cn/a"})
            ),
        )
        with patch.object(
            handler,
            "download_query_image",
            AsyncMock(side_effect=query.FeatureQueryError("过期")),
        ):
            await handler.handle_feature_query(bot, event, Message())
        self.assertIn("过期", bot.send.await_args_list[-1].args[1])
        self.assertEqual(
            ["282", "479"], [c.kwargs["emoji_id"] for c in bot.call_api.await_args_list]
        )
        self.assertFalse(handler._active_users)

    async def test_duplicate_user_does_not_enqueue_twice(self):
        bot = types.SimpleNamespace(
            self_id="bot", send=AsyncMock(), call_api=AsyncMock()
        )
        event = types.SimpleNamespace(
            user_id=123,
            message_id=456,
            reply=None,
            message=Message(
                MessageSegment("image", {"url": "https://gchat.qpic.cn/a"})
            ),
        )
        handler._active_users.add(("bot", 123))
        try:
            with patch.object(handler, "download_query_image", AsyncMock()) as download:
                await handler.handle_feature_query(bot, event, Message())
                download.assert_not_awaited()
        finally:
            handler._active_users.clear()

    async def test_reaction_failure_still_returns_image(self):
        bot = types.SimpleNamespace(
            self_id="bot",
            send=AsyncMock(),
            call_api=AsyncMock(side_effect=RuntimeError("unsupported")),
        )
        event = types.SimpleNamespace(
            user_id=123,
            message_id=456,
            reply=None,
            message=Message(
                MessageSegment("image", {"url": "https://gchat.qpic.cn/a"})
            ),
        )
        with (
            patch.object(
                handler, "download_query_image", AsyncMock(return_value=b"in")
            ),
            patch.object(
                handler, "inspect_query_image", AsyncMock(return_value=b"out")
            ),
        ):
            await handler.handle_feature_query(bot, event, Message())
        self.assertEqual(1, bot.send.await_count)
        self.assertEqual("image", bot.send.await_args.args[1].type)
        self.assertFalse(handler._active_users)

    async def test_cancelled_query_reacts_failure_and_releases_slot(self):
        bot = types.SimpleNamespace(
            self_id="bot", send=AsyncMock(), call_api=AsyncMock()
        )
        event = types.SimpleNamespace(
            user_id=123,
            message_id=456,
            reply=None,
            message=Message(
                MessageSegment("image", {"url": "https://gchat.qpic.cn/a"})
            ),
        )
        with (
            patch.object(
                handler, "download_query_image", AsyncMock(return_value=b"in")
            ),
            patch.object(
                handler,
                "inspect_query_image",
                AsyncMock(side_effect=asyncio.CancelledError()),
            ),
        ):
            with self.assertRaises(asyncio.CancelledError):
                await handler.handle_feature_query(bot, event, Message())
        self.assertEqual(
            ["282", "479"], [c.kwargs["emoji_id"] for c in bot.call_api.await_args_list]
        )
        bot.send.assert_not_awaited()
        self.assertFalse(handler._active_users)


class PipelineTests(unittest.TestCase):
    def test_ambiguous_composition_never_uses_scene_as_subject(self):
        model = ensemble.EnsembleModel.__new__(ensemble.EnsembleModel)
        model.detector = Mock()
        model.detector.detect.return_value = [
            base.PersonBox((0, 0, 450, 1000), 0.9),
            base.PersonBox((550, 0, 1000, 1000), 0.9),
        ]
        scene = base.ModelTags({"1girl": 0.99, "white_hair": 0.99}, {}, {})
        model.wd, model.camie = Mock(), Mock()
        for tagger in (model.wd, model.camie):
            tagger.predict_images.return_value = [scene]
        with patch.object(
            ensemble, "open_rgb", return_value=Image.new("RGB", (1000, 1000))
        ):
            result = model._predict_one(Path("unused.png"))
        self.assertEqual("ambiguous_dominance", result.subject_status)
        self.assertFalse(result.subject_tags)
        self.assertFalse(result.analysis["subject_gender"])
        self.assertIn("尚未计算", report.gender_summary(result))

    def test_cross_view_gender_conflict_stays_unknown(self):
        model = ensemble.EnsembleModel.__new__(ensemble.EnsembleModel)
        main = base.PersonBox((100, 0, 900, 1400), 0.46)
        model.detector = Mock()
        model.detector.detect.side_effect = (
            lambda image: [main] if image.size == (1000, 1500) else []
        )
        male = base.ModelTags({"1boy": 0.99, "1girl": 0.01}, {}, {})
        female = base.ModelTags({"1girl": 0.99, "1boy": 0.01}, {}, {})
        model.wd, model.camie = Mock(), Mock()
        for tagger in (model.wd, model.camie):
            tagger.predict_images.side_effect = [[male, male], [female]]
        with patch.object(
            ensemble, "open_rgb", return_value=Image.new("RGB", (1000, 1500))
        ):
            result = model._predict_one(Path("unused.png"))
        self.assertEqual("view_conflict", result.subject_status)
        self.assertEqual("unknown", result.subject_gender)
        self.assertFalse(result.subject_tags)

    def test_focused_tags_do_not_include_scene_companion(self):
        model = ensemble.EnsembleModel.__new__(ensemble.EnsembleModel)
        main = base.PersonBox((100, 0, 900, 1400), 0.46)
        child = base.PersonBox((800, 800, 1000, 1300), 0.65)
        model.detector = Mock()
        model.detector.detect.side_effect = (
            lambda image: [main, child] if image.size == (1000, 1500) else []
        )
        multi = base.ModelTags({"multiple_girls": 0.99, "white_hair": 0.99}, {}, {})
        focused = base.ModelTags(
            {"1girl": 0.99, "1boy": 0.01, "black_hair": 0.9}, {}, {}
        )
        model.wd = Mock()
        model.camie = Mock()
        for tagger in (model.wd, model.camie):
            tagger.predict_images.side_effect = [[multi, multi], [focused]]
        with patch.object(
            ensemble, "open_rgb", return_value=Image.new("RGB", (1000, 1500))
        ):
            result = model._predict_one(Path("unused.png"))
        self.assertNotIsInstance(result, Exception)
        self.assertEqual("confident", result.subject_status)
        self.assertEqual("upper_body", result.analysis["subject_view"])
        self.assertIn("black_hair", result.subject_tags)
        self.assertNotIn("white_hair", result.subject_tags)


if __name__ == "__main__":
    unittest.main()

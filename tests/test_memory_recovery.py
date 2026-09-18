"""No GPU allocations: simulate prolonged runs, poisoned arenas and recovery."""

# ruff: noqa: PT009, PT027
import asyncio
import importlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import weakref

import numpy as np
import test_subject_features as base

runtime = importlib.import_module("_painting_tests.services.vision.runtime")
taggers = importlib.import_module("_painting_tests.services.vision.taggers")
resources = importlib.import_module("_painting_tests.services.vision.resources")
OOM = (
    "[ONNXRuntimeError] BFCArena::AllocateRawInternal Available memory of "
    "433408 is smaller than requested bytes of 19267584"
)


class FakeSession:
    def __init__(self, *, fail=False, cuda=True):
        self.fail = fail
        self.cuda = cuda
        self.calls = []

    def get_providers(self):
        return ["CUDAExecutionProvider" if self.cuda else "CPUExecutionProvider"]

    def run(self, outputs, inputs, run_options=None):
        self.calls.append(run_options)
        if self.fail:
            raise RuntimeError(OOM)
        return [inputs["images"]]


class SessionRecoveryTests(unittest.TestCase):
    def test_memory_failure_after_many_successful_runs_recovers(self):
        old, fresh = FakeSession(), FakeSession()
        with patch.object(runtime, "_create_session", side_effect=[old, fresh]):
            session = runtime.create_session(Path("camie.onnx"), 5.5)
            for index in range(256):
                if index == 200:
                    old.fail = True
                feed = {"images": np.array([[index]])}
                self.assertEqual(index, session.run(None, feed)[0][0, 0])
            self.assertEqual(201, len(old.calls))
            self.assertEqual(56, len(fresh.calls))

    def test_rebuild_releases_old_session_before_allocating_new_weights(self):
        refs = []

        def factory(*args):
            if refs:
                self.assertIsNone(refs[-1](), "Old CUDA session still retained")
            value = FakeSession(fail=not refs)
            refs.append(weakref.ref(value))
            return value

        with patch.object(runtime, "_create_session", side_effect=factory):
            session = runtime.create_session(Path("camie.onnx"), 5.5)
            feed = {"images": np.ones((1, 2))}
            self.assertEqual((1, 2), session.run(None, feed)[0].shape)
            self.assertEqual(2, len(refs))

    def test_persistent_oom_is_bounded_then_retries_after_cooldown(self):
        created = []

        def factory(*args):
            value = FakeSession(fail=len(created) < 2)
            created.append(value)
            return value

        with (
            patch.object(runtime, "_create_session", side_effect=factory),
            patch.object(runtime.time, "monotonic", return_value=1000) as clock,
        ):
            session = runtime.create_session(Path("camie.onnx"), 5.5)
            feed = {"images": np.ones((1, 2))}
            for _ in range(100):
                with self.assertRaises(resources.InferenceResourceError):
                    session.run(None, feed)
            self.assertEqual(2, len(created))
            self.assertEqual([1, 1], [len(value.calls) for value in created])
            self.assertIsNone(session._session)
            clock.return_value = 1061
            self.assertEqual((1, 2), session.run(None, feed)[0].shape)
            self.assertEqual(3, len(created))

    def test_reload_weight_allocation_failure_is_transient(self):
        with patch.object(
            runtime,
            "_create_session",
            side_effect=[FakeSession(fail=True), MemoryError()],
        ) as factory:
            session = runtime.create_session(Path("camie.onnx"), 5.5)
            with self.assertRaises(resources.InferenceResourceError):
                session.run(None, {"images": np.ones((1, 2))})
            self.assertIsNone(session._session)
            self.assertEqual(2, factory.call_count)

    def test_non_memory_error_does_not_rebuild(self):
        native = FakeSession()
        native.run = Mock(side_effect=ValueError("invalid input shape"))
        with patch.object(runtime, "_create_session", return_value=native) as factory:
            session = runtime.create_session(Path("camie.onnx"), 5.5)
            with self.assertRaises(ValueError):
                session.run(None, {"images": np.ones((1, 2))})
            self.assertEqual(1, factory.call_count)

    def test_gpu_shrink_on_shape_change_and_every_32_runs(self):
        native = FakeSession()
        with patch.object(runtime, "_create_session", return_value=native):
            session = runtime.create_session(Path("camie.onnx"), 5.5)
            for _ in range(33):
                session.run(None, {"images": np.ones((1, 2))})
            session.run(None, {"images": np.ones((2, 2))})
        chosen = [i + 1 for i, option in enumerate(native.calls) if option is not None]
        self.assertEqual([1, 32, 34], chosen)
        for index in chosen:
            self.assertEqual(
                "gpu:0",
                native.calls[index - 1].get_run_config_entry(
                    "memory.enable_memory_arena_shrinkage"
                ),
            )

    def test_cpu_does_not_request_nonexistent_gpu_arena(self):
        native = FakeSession(cuda=False)
        with patch.object(runtime, "_create_session", return_value=native):
            session = runtime.create_session(Path("camie.onnx"), 5.5)
            session.run(None, {"images": np.ones((1, 2))})
        self.assertIsNone(native.calls[0])

    def test_oom_batch_is_halved_before_single_image_rebuild(self):
        created = []

        def factory(*args):
            value = FakeSession(fail=not created)
            created.append(value)
            return value

        with patch.object(runtime, "_create_session", side_effect=factory):
            model = taggers.BatchTagger.__new__(taggers.BatchTagger)
            model.session = runtime.create_session(Path("camie.onnx"), 5.5)
            model.input_name = "images"
            model.batch_size = 8
            model.decode = lambda row: int(row[0])
            values = model.run([np.array([i]) for i in range(8)])
        self.assertEqual(list(range(8)), values)
        self.assertEqual(1, model.batch_size)
        self.assertEqual(4, len(created[0].calls))  # 8 -> 4 -> 2 -> 1
        self.assertEqual(2, len(created))

    def test_session_cooldown_does_not_split_every_image_again(self):
        model = taggers.BatchTagger.__new__(taggers.BatchTagger)
        model.session = Mock()
        model.session.run.side_effect = resources.InferenceResourceError("cooldown")
        model.input_name = "images"
        model.batch_size = 8
        results = model.run([np.ones((1,))] * 8)
        self.assertEqual(8, len(results))
        self.assertEqual(1, model.session.run.call_count)
        self.assertTrue(all(value.__traceback__ is None for value in results))


class QueueRecoveryTests(unittest.TestCase):
    def test_legacy_errors_only_requeues_resource_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            store = base.ImageFeatureStore(Path(directory) / "features.db")
            store.initialize()
            store.mark_error("wives/oom.png", "wives", 10, 20, OOM)
            store.mark_error("wives/bad.png", "wives", 10, 20, "cannot identify image")
            store.save_prediction(
                "wives/good.png", "wives", 10, 20, "ok", base.prediction()
            )
            self.assertEqual(1, store.retry_resource_errors())
            self.assertEqual(0, store.retry_resource_errors())
            self.assertTrue(store.needs_processing("wives/oom.png", 10, 20))
            self.assertFalse(store.needs_processing("wives/bad.png", 10, 20))
            self.assertIsNotNone(store.get_cached("wives/good.png", 10, 20))

    def test_queue_preserves_oom_without_error_row_or_progress_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service = base.ImageFeatureService()
            service.store = base.ImageFeatureStore(root / "features.db")
            service.store.initialize()
            folder = root / "wives"
            folder.mkdir()
            paths = [folder / "retry.png", folder / "ok.png"]
            for index, path in enumerate(paths):
                path.write_bytes(bytes([index]))
                service._enqueue(base._QueuedImage(path, "wives"), priority=False)
            service._initial_pending = set(service._queued_paths)
            # Test the interruptible cooldown without sleeping.
            service._stop_event.set()
            batch = service._take_batch()
            with patch.object(
                base.service_module.ModelManager,
                "predict_paths",
                return_value=[resources.InferenceResourceError(OOM), base.prediction()],
            ):
                asyncio.run(service._process_queued_batch(batch))
            retry = base.normalize_image_path(paths[0])
            self.assertEqual({retry}, service._queued_paths)
            self.assertEqual({retry}, service._initial_pending)
            self.assertFalse(service._in_flight)
            self.assertEqual(1, len(service._queue))
            self.assertNotIn(retry, service.store.processing_index())
            self.assertEqual(
                "tagged",
                service.store.processing_index()[base.normalize_image_path(paths[1])][
                    3
                ],
            )

    def test_error_detection_does_not_confuse_bad_image_with_oom(self):
        self.assertTrue(resources.is_memory_error(OOM))
        self.assertTrue(resources.is_memory_error(MemoryError()))
        self.assertTrue(resources.is_memory_error("CUDA failure 2: out of memory"))
        self.assertFalse(resources.is_memory_error("invalid shape"))
        self.assertFalse(resources.is_memory_error("cannot identify image file"))


if __name__ == "__main__":
    unittest.main()

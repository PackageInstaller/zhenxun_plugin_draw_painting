import gc
from pathlib import Path
import time

import onnxruntime as ort
from PIL import Image, ImageOps

from zhenxun.services.log import logger

from .resources import InferenceResourceError, is_memory_error


def open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        image.seek(0)
        rgba = ImageOps.exif_transpose(image).convert("RGBA")
    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    background.alpha_composite(rgba)
    rgb = background.convert("RGB")
    bounds = rgba.getchannel("A").getbbox()
    if bounds:
        x0, y0, x1, y1 = bounds
        rgb.info["painting_content_box"] = (
            max(0, x0 - 8),
            max(0, y0 - 8),
            min(rgb.width, x1 + 8),
            min(rgb.height, y1 + 8),
        )
    return rgb


def _create_session(path: Path, memory_gib: float) -> ort.InferenceSession:
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
    providers: list = []
    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers.append(
            (
                "CUDAExecutionProvider",
                {
                    "device_id": "0",
                    "gpu_mem_limit": str(int(memory_gib * 1024**3)),
                    "arena_extend_strategy": "kSameAsRequested",
                    "cudnn_conv_algo_search": "HEURISTIC",
                    "cudnn_conv_use_max_workspace": "0",
                },
            )
        )
    providers.append("CPUExecutionProvider")
    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.intra_op_num_threads = 4
    # Scene, subject and face crops use varying batch shapes. Avoid retaining
    # shape-specific memory patterns across a many-hour indexing run.
    options.enable_mem_pattern = False
    session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
    logger.info(
        f"立绘模型 {path.name}: {session.get_providers()[0]}, "
        f"CUDA arena 上限 {memory_gib:g} GiB"
    )
    return session


class RecoveringSession:
    """Called under EnsembleModel's lock; never retain a poisoned CUDA arena."""

    RECOVERY_SECONDS = 60.0
    SHRINK_INTERVAL = 32

    def __init__(self, path: Path, memory_gib: float) -> None:
        self.path = path
        self.memory_gib = memory_gib
        self._session = _create_session(path, memory_gib)
        self._last_recovery = float("-inf")
        self._blocked_until = 0.0
        self._runs = 0
        self._shape = None

    def get_inputs(self):
        return self._session.get_inputs()

    def get_modelmeta(self):
        return self._session.get_modelmeta()

    def get_providers(self):
        return self._session.get_providers()

    def _release(self) -> None:
        self._session = None
        self._shape = None
        # Exception tracebacks must already have left their except block before
        # this point, otherwise ORT's run frame can keep the old session alive.
        gc.collect()

    def _run_once(self, outputs, inputs):
        shape = tuple((name, tuple(value.shape)) for name, value in inputs.items())
        self._runs += 1
        shrink = self._shape != shape or self._runs % self.SHRINK_INTERVAL == 0
        self._shape = shape
        options = None
        if shrink and "CUDAExecutionProvider" in self._session.get_providers():
            options = ort.RunOptions()
            options.add_run_config_entry(
                "memory.enable_memory_arena_shrinkage", "gpu:0"
            )
        return self._session.run(outputs, inputs, run_options=options)

    def _rebuild(self) -> str | None:
        self._release()
        self._last_recovery = time.monotonic()
        try:
            self._session = _create_session(self.path, self.memory_gib)
        except Exception as exc:
            # Includes failures while allocating model weights on reload.
            return str(exc) or type(exc).__name__
        return None

    def _unavailable(self, reason: str):
        self._release()
        self._blocked_until = time.monotonic() + self.RECOVERY_SECONDS
        logger.warning(
            f"立绘模型 {self.path.name} 显存恢复未成功，冷却 60 秒: {reason}"
        )
        raise InferenceResourceError(f"{self.path.name} 显存恢复等待重试: {reason}")

    def run(self, outputs, inputs):
        if time.monotonic() < self._blocked_until:
            raise InferenceResourceError(f"{self.path.name} 显存恢复冷却中")
        if self._session is None:
            error = self._rebuild()
            if error:
                return self._unavailable(error)
        try:
            return self._run_once(outputs, inputs)
        except Exception as exc:
            if not is_memory_error(exc):
                raise
            error = str(exc) or "out of memory"
        # Leave the exception handler before splitting/releasing native objects.
        count = next(iter(inputs.values())).shape[0]
        if count > 1:
            # BatchTagger halves the batch first; a rebuild is the last resort.
            raise RuntimeError(error)
        if time.monotonic() - self._last_recovery < self.RECOVERY_SECONDS:
            return self._unavailable(error)
        logger.warning(f"立绘模型 {self.path.name} 单张显存不足，释放会话后重建重试")
        error = self._rebuild()
        if error:
            return self._unavailable(error)
        try:
            return self._run_once(outputs, inputs)
        except Exception as exc:
            if not is_memory_error(exc):
                raise
            error = str(exc) or "out of memory"
        return self._unavailable(error)


def create_session(path: Path, memory_gib: float) -> RecoveringSession:
    return RecoveringSession(path, memory_gib)

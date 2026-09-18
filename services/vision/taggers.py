from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .assets import CAMIE_MODEL, CAMIE_TAGS, WD_MODEL, WD_TAGS
from .resources import InferenceResourceError
from .runtime import create_session
from .types import ModelTags


class BatchTagger:
    """Bounded ONNX batches; failed batches are retried as smaller batches."""

    def __init__(self, path: Path, memory_gib: float) -> None:
        self.session = create_session(path, memory_gib)
        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        batch_dimension = model_input.shape[0]
        self.batch_size = batch_dimension if isinstance(batch_dimension, int) else 8

    def prepare(self, image: Image.Image) -> np.ndarray:
        raise NotImplementedError

    def decode(self, values: np.ndarray) -> ModelTags:
        if values.shape != (len(self.names),) or not np.isfinite(values).all():
            raise ValueError("标签模型输出形状或数值无效")
        general: dict[str, float] = {}
        characters: dict[str, float] = {}
        ratings: dict[str, float] = {}
        for name, category, value in zip(
            self.names, self.categories, values, strict=True
        ):
            score = float(value)
            if category == "general":
                # Keep raw scores until fusion; thresholding before averaging
                # would mistake a discarded low score for an unsupported tag.
                general[name] = score
            elif category == "character" and score >= 0.85:
                characters[name] = score
            elif category == "rating":
                ratings[name] = score
        return ModelTags(general, characters, ratings)

    def run(self, arrays: list[np.ndarray]) -> list[ModelTags | Exception]:
        if len(arrays) > self.batch_size:
            size = self.batch_size
            return [
                result
                for offset in range(0, len(arrays), size)
                for result in self.run(arrays[offset : offset + size])
            ]
        try:
            outputs = self.session.run(None, {self.input_name: np.stack(arrays)})
            values = self.output_probabilities(outputs)
            if len(values) != len(arrays):
                raise ValueError("标签模型输出批次数量不一致")
            return [self.decode(row) for row in values]
        except Exception as exc:
            # Cooldown is session-wide: splitting cannot help, nor should an
            # error object retain native session/tensor frames across batches.
            error = exc.with_traceback(None)
            if isinstance(error, InferenceResourceError):
                return [error] * len(arrays)
        if len(arrays) == 1:
            return [error]
        half = max(1, len(arrays) // 2)
        self.batch_size = min(self.batch_size, half)
        return self.run(arrays[:half]) + self.run(arrays[half:])

    def output_probabilities(self, outputs: list[np.ndarray]) -> np.ndarray:
        return outputs[0]

    def predict_images(self, images: list[Image.Image]) -> list[ModelTags | Exception]:
        results: list[ModelTags | Exception] = []
        offset = 0
        # PIL resize / NumPy release the GIL. Parallel preprocessing prevents
        # high-resolution PNG crops from starving otherwise idle CUDA sessions.
        with ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="painting-prep"
        ) as pool:
            while offset < len(images):
                images_chunk = images[offset : offset + self.batch_size]
                try:
                    arrays = list(pool.map(self.prepare, images_chunk))
                    results.extend(self.run(arrays))
                except Exception as exc:
                    results.extend([exc] * len(images_chunk))
                offset += len(images_chunk)
        return results


class WDTaggerModel(BatchTagger):
    def __init__(
        self, model_path: Path = WD_MODEL.path, tags_path: Path = WD_TAGS.path
    ) -> None:
        with tags_path.open(encoding="utf-8", newline="") as source:
            rows = list(csv.DictReader(source))
        self.names = [row["name"] for row in rows]
        categories = {"0": "general", "4": "character", "9": "rating"}
        self.categories = [categories.get(row["category"], "other") for row in rows]
        super().__init__(model_path, 3.5)
        self.size = int(self.session.get_inputs()[0].shape[1])

    def prepare(self, image: Image.Image) -> np.ndarray:
        side = max(image.size)
        square = Image.new("RGB", (side, side), "white")
        square.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
        resized = square.resize((self.size, self.size), Image.Resampling.BICUBIC)
        return np.asarray(resized, dtype=np.float32)[:, :, ::-1].copy()


class CamieTaggerModel(BatchTagger):
    def __init__(self) -> None:
        metadata = json.loads(CAMIE_TAGS.path.read_text(encoding="utf-8"))
        mapping = metadata["dataset_info"]["tag_mapping"]
        self.names = [
            mapping["idx_to_tag"][str(index)]
            for index in range(metadata["dataset_info"]["total_tags"])
        ]
        self.categories = [mapping["tag_to_category"][name] for name in self.names]
        self.size = int(metadata["model_info"]["img_size"])
        super().__init__(CAMIE_MODEL.path, 5.5)

    def prepare(self, image: Image.Image) -> np.ndarray:
        ratio = self.size / max(image.size)
        resized = image.resize(
            (max(1, int(image.width * ratio)), max(1, int(image.height * ratio))),
            Image.Resampling.LANCZOS,
        )
        padded = Image.new("RGB", (self.size, self.size), (124, 116, 104))
        padded.paste(
            resized,
            ((self.size - resized.width) // 2, (self.size - resized.height) // 2),
        )
        array = np.asarray(padded, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        return np.transpose((array - mean) / std, (2, 0, 1)).copy()

    def output_probabilities(self, outputs: list[np.ndarray]) -> np.ndarray:
        # Official export: initial logits, full refined logits, candidate IDs.
        if len(outputs) < 2 or outputs[1].shape[-1] != len(self.names):
            raise ValueError("Camie refined_predictions 输出与标签表不一致")
        return 1.0 / (1.0 + np.exp(-np.clip(outputs[1], -80.0, 80.0)))

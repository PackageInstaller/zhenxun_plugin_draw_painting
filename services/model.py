"""Compatibility facade for the versioned, local multi-model vision pipeline."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from threading import Event, Lock
from typing import ClassVar

from huggingface_hub import hf_hub_download

from zhenxun.services.log import logger

from .vision.assets import (
    ASSETS,
    INSTANCE_ASSETS,
    MODEL_VERSION,
    WD_MODEL,
    WD_TAGS,
    ModelAsset,
)
from .vision.ensemble import EnsembleModel
from .vision.taggers import WDTaggerModel
from .vision.types import GENERAL_THRESHOLD, PredictionOutcome, TagPrediction


class ModelManager:
    _model: ClassVar[EnsembleModel | None] = None
    _load_lock: ClassVar[Lock] = Lock()
    _instance_download_lock: ClassVar[Lock] = Lock()
    _download_complete: ClassVar[Event] = Event()
    _verified_signatures: ClassVar[dict[Path, tuple[int, int]]] = {}
    _is_downloading = False
    _download_progress = 0.0

    @classmethod
    def model_dir(cls) -> Path:
        return WD_MODEL.path.parent

    @classmethod
    def model_path(cls) -> Path:
        return WD_MODEL.path

    @classmethod
    def tags_path(cls) -> Path:
        return WD_TAGS.path

    @classmethod
    def _verify_asset(cls, asset: ModelAsset) -> bool:
        try:
            stat = asset.path.stat()
            signature = (stat.st_size, stat.st_mtime_ns)
            if stat.st_size != asset.size:
                return False
            if cls._verified_signatures.get(asset.path) == signature:
                return True
            digest = hashlib.sha256()
            with asset.path.open("rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != asset.sha256:
                return False
        except OSError:
            return False
        cls._verified_signatures[asset.path] = signature
        return True

    @classmethod
    def verify_model(cls, _file_path: str | Path | None = None) -> bool:
        return all(cls._verify_asset(asset) for asset in ASSETS)

    @classmethod
    def _download_assets(cls, assets: tuple[ModelAsset, ...]) -> None:
        for asset in assets:
            if cls._verify_asset(asset):
                continue
            local_dir = asset.path
            for _part in Path(asset.filename).parts:
                local_dir = local_dir.parent
            logger.info(f"下载立绘识别组件: {asset.repo}/{asset.filename}")
            hf_hub_download(
                repo_id=asset.repo,
                revision=asset.revision,
                filename=asset.filename,
                local_dir=local_dir,
                force_download=asset.path.exists(),
            )
            if not cls._verify_asset(asset):
                raise RuntimeError(f"模型文件校验失败: {asset.filename}")

    @classmethod
    def _download_files(cls) -> None:
        cls._download_assets(ASSETS)

    @classmethod
    def ensure_instance_models(cls) -> bool:
        """Download query-only segmentation assets without touching core versioning."""
        if all(cls._verify_asset(asset) for asset in INSTANCE_ASSETS):
            return True
        with cls._instance_download_lock:
            if all(cls._verify_asset(asset) for asset in INSTANCE_ASSETS):
                return True
            try:
                cls._download_assets(INSTANCE_ASSETS)
                return True
            except Exception as exc:
                logger.warning(
                    "下载动漫人物实例分割组件失败，特征查询将使用兼容模式: "
                    f"{type(exc).__name__}"
                )
                return False

    @classmethod
    async def download_model(cls) -> bool:
        if await asyncio.to_thread(cls.is_model_ready):
            return True
        if cls._is_downloading:
            await asyncio.to_thread(cls._download_complete.wait)
            return cls.is_model_ready()
        cls._is_downloading = True
        cls._download_complete.clear()
        try:
            download = asyncio.create_task(asyncio.to_thread(cls._download_files))
            while not download.done():
                complete = sum(
                    a.size
                    for a in ASSETS
                    if cls._verified_signatures.get(a.path) is not None
                )
                cls._download_progress = complete / sum(a.size for a in ASSETS) * 100
                await asyncio.sleep(0.5)
            await download
            cls._download_progress = 100.0
            return True
        except Exception as exc:
            logger.error(f"下载多模型立绘识别组件失败: {exc}")
            return False
        finally:
            cls._is_downloading = False
            cls._download_complete.set()

    @classmethod
    def is_model_ready(cls) -> bool:
        return not cls._is_downloading and cls.verify_model()

    @classmethod
    def get_download_status(cls) -> tuple[bool, float]:
        return cls._is_downloading, cls._download_progress

    @classmethod
    def get_model(cls) -> EnsembleModel:
        with cls._load_lock:
            if cls._model is None:
                if not cls.verify_model():
                    raise RuntimeError("多模型立绘识别组件尚未下载完成")
                cls._model = EnsembleModel()
            return cls._model

    @classmethod
    def predict_paths(cls, image_paths: list[Path]) -> list[PredictionOutcome]:
        return cls.get_model().predict_paths(image_paths)

    @classmethod
    def predict_query(cls, image_path: Path) -> TagPrediction:
        cls.ensure_instance_models()
        return cls.get_model().predict_query(image_path)


async def determine_gender(img_path: str) -> tuple[float, float]:
    try:
        outcomes = await asyncio.to_thread(ModelManager.predict_paths, [Path(img_path)])
        prediction = outcomes[0]
        if isinstance(prediction, Exception):
            raise prediction
        return prediction.male_probability, prediction.female_probability
    except Exception as exc:
        logger.warning(f"识别图片主体性别失败: {exc}")
        return 0.0, 0.0


__all__ = [
    "GENERAL_THRESHOLD",
    "MODEL_VERSION",
    "ModelManager",
    "PredictionOutcome",
    "TagPrediction",
    "WDTaggerModel",
    "determine_gender",
]

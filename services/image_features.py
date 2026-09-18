from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
import hashlib
from pathlib import Path
import shutil
from stat import S_ISREG
import time

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from zhenxun.services.log import logger

from ..config import paths
from .archive_policy import (
    fused_gender_scores,
    fused_scene_gender_scores,
    target_library,
)
from .image_feature_store import (
    ERROR_RETRY_SECONDS,
    ImageFeatureStore,
    normalize_image_path,
)
from .model import MODEL_VERSION, ModelManager, TagPrediction
from .vision.resources import is_memory_error

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
WATCH_INTERVAL_SECONDS = 15.0
FILE_STABLE_SECONDS = 3.0
# Bounded work units; model sessions serialize scene/crop batches internally.
# CUDA arena budgets: WD 3.5 GiB + Camie 5.5 GiB + person detector 0.5 GiB.
INFERENCE_BATCH_SIZE = 8


@dataclass(frozen=True, slots=True)
class _QueuedImage:
    path: Path
    library: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as image_file:
        for block in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as image_file:
        for block in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_move_target(source: Path, desired_target: Path) -> tuple[Path, bool]:
    """Return an available target, or an identical target that may be replaced."""

    source_md5 = _md5_file(source)
    candidate = desired_target
    duplicate_index = 0
    while candidate.exists():
        if _md5_file(candidate) == source_md5:
            return candidate, True
        duplicate_index += 1
        candidate = desired_target.with_name(
            f"{desired_target.stem}_重复{duplicate_index}{desired_target.suffix}"
        )
    return candidate, False


def _feature_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("已用"),
        TimeElapsedColumn(),
        TextColumn("剩余"),
        TimeRemainingColumn(),
    )


class ImageFeatureService:
    """Index existing paintings and continuously tag new or changed files."""

    def __init__(self) -> None:
        self.store = ImageFeatureStore(paths.IMAGE_FEATURES_DB)
        self._queue: deque[_QueuedImage] = deque()
        self._queued_paths: set[str] = set()
        self._in_flight: set[str] = set()
        self._known_signatures: dict[str, tuple[int, int]] = {}
        self._task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()
        self._progress: Progress | None = None
        self._progress_task_id: int | None = None
        self._initial_pending: set[str] = set()

    @staticmethod
    def _libraries() -> tuple[tuple[str, Path], ...]:
        return (
            ("wives", Path(paths.WIVES_IMAGES_FOLDER)),
            ("husbands", Path(paths.HUSBANDS_IMAGES_FOLDER)),
        )

    def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self.store.initialize()
        self._stop_event = asyncio.Event()
        self._task = asyncio.create_task(
            self._run(),
            name="draw-painting-image-feature-index",
        )

    async def stop(self) -> None:
        self._stop_event.set()
        task = self._task
        if task is not None and not task.done():
            try:
                await asyncio.wait_for(task, timeout=30)
            except TimeoutError:
                task.cancel()
            except asyncio.CancelledError:
                pass
        self._task = None
        self._stop_progress()

    @staticmethod
    def _record_needs_processing(
        record: tuple[int, int, str, str, float] | None,
        file_size: int,
        mtime_ns: int,
        now: float,
    ) -> bool:
        if record is None:
            return True
        old_size, old_mtime, model_version, status, processed_at = record
        if (
            old_size != file_size
            or old_mtime != mtime_ns
            or model_version != MODEL_VERSION
        ):
            return True
        if status == "tagged":
            return False
        if status == "error":
            return now - processed_at >= ERROR_RETRY_SECONDS
        return True

    def _enqueue(self, item: _QueuedImage, *, priority: bool) -> bool:
        normalized = normalize_image_path(item.path)
        if normalized in self._queued_paths or normalized in self._in_flight:
            return False
        if priority:
            self._queue.appendleft(item)
        else:
            self._queue.append(item)
        self._queued_paths.add(normalized)
        return True

    async def _scan(self, *, initial: bool) -> int:
        if initial:
            retried = await asyncio.to_thread(self.store.retry_resource_errors)
            if retried:
                logger.info(f"重新排队 {retried} 张历史显存不足导致标记失败的立绘")
        database_index = await asyncio.to_thread(self.store.processing_index)
        # Re-evaluate existing tags after policy changes without full GPU retagging.
        archive_candidates = (
            await asyncio.to_thread(self.store.archive_candidates) if initial else set()
        )
        now = time.time()
        current_signatures: dict[str, tuple[int, int]] = {}
        added = 0

        for library, folder in self._libraries():
            try:
                entries = list(folder.iterdir())
            except OSError as exc:
                logger.error(f"扫描立绘目录失败 {folder}: {exc}")
                continue

            scanned: dict[str, tuple[int, int]] = {}
            scan_complete = True
            for image_path in entries:
                if image_path.suffix.casefold() not in IMAGE_SUFFIXES:
                    continue
                try:
                    stat = image_path.stat()
                    if not S_ISREG(stat.st_mode):
                        continue
                except FileNotFoundError:
                    continue
                except OSError:
                    scan_complete = False
                    continue
                normalized = normalize_image_path(image_path)
                signature = (stat.st_size, stat.st_mtime_ns)
                scanned[normalized] = signature
                current_signatures[normalized] = signature
                if now - stat.st_mtime < FILE_STABLE_SECONDS:
                    continue
                if (
                    normalized not in archive_candidates
                    and not (
                        not initial
                        and self._known_signatures.get(normalized) != signature
                    )
                    and not self._record_needs_processing(
                        database_index.get(normalized),
                        stat.st_size,
                        stat.st_mtime_ns,
                        now,
                    )
                ):
                    continue

                changed_at_runtime = (
                    not initial and self._known_signatures.get(normalized) != signature
                )
                if self._enqueue(
                    _QueuedImage(image_path, library),
                    priority=changed_at_runtime,
                ):
                    added += 1

            if scan_complete:
                missing, purged = await asyncio.to_thread(
                    self.store.reconcile_library, library, scanned
                )
                if missing or purged:
                    logger.info(
                        f"立绘索引核对 {library}: 失效 {missing} 条，"
                        f"清理持续缺失超过 24 小时的特征记录 {purged} 条"
                    )
        self._known_signatures = current_signatures
        return added

    def _start_progress(self) -> None:
        self._initial_pending = set(self._queued_paths)
        if not self._initial_pending:
            logger.info("立绘特征索引已是最新状态")
            return
        self._progress = _feature_progress()
        self._progress.start()
        self._progress_task_id = self._progress.add_task(
            "识别并记录立绘特征",
            total=len(self._initial_pending),
        )
        logger.info(f"发现 {len(self._initial_pending)} 张未标记或需更新的立绘")

    def _complete_initial(self, path: Path) -> None:
        normalized = normalize_image_path(path)
        if normalized not in self._initial_pending:
            return
        self._initial_pending.remove(normalized)
        if self._progress is not None and self._progress_task_id is not None:
            self._progress.update(self._progress_task_id, advance=1)
        if not self._initial_pending:
            self._stop_progress()
            counts = self.store.status_counts()
            logger.info(f"立绘初始特征扫描完成: {counts}")

    def _stop_progress(self) -> None:
        if self._progress is not None:
            self._progress.stop()
        self._progress = None
        self._progress_task_id = None

    def _take_batch(self) -> list[_QueuedImage]:
        batch: list[_QueuedImage] = []
        while self._queue and len(batch) < INFERENCE_BATCH_SIZE:
            item = self._queue.popleft()
            self._queued_paths.discard(normalize_image_path(item.path))
            self._in_flight.add(normalize_image_path(item.path))
            batch.append(item)
        return batch

    @staticmethod
    def _target_library(
        library: str,
        prediction: TagPrediction,
    ) -> str | None:
        return target_library(
            library,
            prediction.subject_status,
            prediction.subject_gender,
            prediction.subject_tags,
            prediction.analysis.get("scene_gender"),
            prediction.analysis.get("subject_gender"),
            prediction.analysis.get("detection"),
            prediction.general_tags,
        )

    def _folder_for_library(self, library: str) -> Path:
        if library == "Others":
            return Path(paths.OTHERS_IMAGES_FOLDER)
        for candidate, folder in self._libraries():
            if candidate == library:
                return folder
        raise ValueError(f"未知立绘目录: {library}")

    def _store_and_maybe_move(
        self,
        item: _QueuedImage,
        content_sha256: str,
        prediction: TagPrediction,
        file_size: int,
        mtime_ns: int,
        *,
        allow_move: bool,
    ) -> Path:
        # Inference may take long enough for the user to replace/delete a file.
        # Do not persist or move a prediction made for a previous file version.
        try:
            current = item.path.stat()
        except OSError:
            self.store.mark_missing(item.path)
            return item.path
        if (current.st_size, current.st_mtime_ns) != (file_size, mtime_ns):
            return item.path
        self.store.save_prediction(
            item.path,
            item.library,
            file_size,
            mtime_ns,
            content_sha256,
            prediction,
        )
        target_library = self._target_library(item.library, prediction)
        if not allow_move or target_library is None:
            return item.path

        desired_target = self._folder_for_library(target_library) / item.path.name
        try:
            desired_target.parent.mkdir(parents=True, exist_ok=True)
            target_path, should_replace = _resolve_move_target(
                item.path,
                desired_target,
            )
            if should_replace:
                item.path.replace(target_path)
                logger.info(f"目标立绘 MD5 相同，已直接覆盖: {target_path}")
            else:
                shutil.move(str(item.path), str(target_path))
                if target_path != desired_target:
                    logger.info(f"目标立绘内容不同，已重命名为: {target_path.name}")
            target_stat = target_path.stat()
            self.store.save_prediction(
                target_path,
                target_library,
                target_stat.st_size,
                target_stat.st_mtime_ns,
                content_sha256,
                prediction,
                moved_from=item.path,
            )
            self.store.mark_missing(item.path)
            male, female = fused_scene_gender_scores(
                prediction.analysis.get("subject_gender")
            ) or fused_gender_scores(prediction.subject_tags)
            if target_library == "Others":
                male, female = fused_scene_gender_scores(
                    prediction.analysis.get("scene_gender")
                )
            logger.info(
                f"立绘已自动从 {item.library} 移至 {target_library}: "
                f"{item.path.name} -> {target_path.name} "
                f"(融合 male={male:.2%}, female={female:.2%})"
            )
            return target_path
        except OSError as exc:
            logger.error(f"自动移动立绘失败 {item.path}: {exc}")
            return item.path

    async def _process_batch(self, batch: list[_QueuedImage]) -> list[_QueuedImage]:
        retry: list[_QueuedImage] = []
        prepared: list[tuple[_QueuedImage, int, int, str]] = []
        for item in batch:
            try:
                stat_before = item.path.stat()
                content_sha256 = await asyncio.to_thread(_sha256_file, item.path)
                stat_after = item.path.stat()
                if (
                    stat_before.st_size != stat_after.st_size
                    or stat_before.st_mtime_ns != stat_after.st_mtime_ns
                ):
                    continue
                prepared.append(
                    (
                        item,
                        stat_after.st_size,
                        stat_after.st_mtime_ns,
                        content_sha256,
                    )
                )
            except OSError as exc:
                logger.warning(f"读取待标记立绘失败 {item.path}: {exc}")

        to_infer: list[tuple[_QueuedImage, int, int, str]] = []
        for item, file_size, mtime_ns, content_sha256 in prepared:
            cached = await asyncio.to_thread(
                self.store.find_by_hash,
                content_sha256,
            )
            if cached is None:
                to_infer.append((item, file_size, mtime_ns, content_sha256))
                continue
            await asyncio.to_thread(
                self._store_and_maybe_move,
                item,
                content_sha256,
                cached.to_prediction(),
                file_size,
                mtime_ns,
                allow_move=True,
            )

        if to_infer:
            try:
                outcomes = await asyncio.to_thread(
                    ModelManager.predict_paths,
                    [item.path for item, _size, _mtime, _digest in to_infer],
                )
            except Exception as exc:
                if not is_memory_error(exc):
                    raise
                outcomes = [exc.with_traceback(None)] * len(to_infer)
            for prepared_item, outcome in zip(to_infer, outcomes, strict=True):
                item, file_size, mtime_ns, content_sha256 = prepared_item
                if isinstance(outcome, Exception):
                    if is_memory_error(outcome):
                        # Do not classify resource exhaustion as a corrupt image,
                        # persist partial model output, move it, or finish progress.
                        retry.append(item)
                        continue
                    await asyncio.to_thread(
                        self.store.mark_error,
                        item.path,
                        item.library,
                        file_size,
                        mtime_ns,
                        str(outcome),
                        content_sha256,
                    )
                    logger.warning(f"标记立绘失败 {item.path}: {outcome}")
                    continue
                await asyncio.to_thread(
                    self._store_and_maybe_move,
                    item,
                    content_sha256,
                    outcome,
                    file_size,
                    mtime_ns,
                    allow_move=True,
                )

        for item in batch:
            if item not in retry:
                self._complete_initial(item.path)
        return retry

    async def _process_queued_batch(self, batch: list[_QueuedImage]) -> None:
        retry: list[_QueuedImage] = []
        try:
            retry = await self._process_batch(batch)
        finally:
            self._in_flight.difference_update(
                normalize_image_path(item.path) for item in batch
            )
        for item in retry:
            self._enqueue(item, priority=False)
        if retry:
            logger.warning(
                f"立绘识别显存暂不可用，{len(retry)} 张保留待重试；"
                "暂停推理 60 秒，目录监控继续运行"
            )
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=60)
            except TimeoutError:
                pass

    async def _watch(self) -> None:
        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=WATCH_INTERVAL_SECONDS
                )
                return
            except TimeoutError:
                try:
                    added = await self._scan(initial=False)
                    if added:
                        logger.info(f"实时监控发现 {added} 张新增或变更立绘")
                except Exception as exc:
                    logger.warning(f"立绘目录核对失败，下次重试: {exc}")

    async def _run(self) -> None:
        watcher: asyncio.Task[None] | None = None
        try:
            await self._scan(initial=True)
            self._start_progress()
            watcher = asyncio.create_task(
                self._watch(), name="painting-directory-watch"
            )
            while not self._stop_event.is_set():
                if await ModelManager.download_model():
                    try:
                        await asyncio.to_thread(ModelManager.get_model)
                        break
                    except Exception as exc:
                        logger.warning(f"多模型初始化失败: {exc}")
                logger.warning("多模型尚不可用，目录监控继续运行，60 秒后重试")
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=60)
                    return
                except TimeoutError:
                    pass
            logger.info(f"多模型立绘分类工作批量: {INFERENCE_BATCH_SIZE}")

            while not self._stop_event.is_set():
                batch = self._take_batch()
                if batch:
                    await self._process_queued_batch(batch)
                    await asyncio.sleep(0)
                    continue

                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=WATCH_INTERVAL_SECONDS,
                    )
                except TimeoutError:
                    pass
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception(f"立绘特征监控异常退出: {exc}")
        finally:
            if watcher is not None:
                watcher.cancel()
                try:
                    await watcher
                except asyncio.CancelledError:
                    pass
            self._stop_progress()

    @staticmethod
    def _library_for_path(path: Path) -> str:
        parent = normalize_image_path(path.parent)
        if parent == normalize_image_path(paths.HUSBANDS_IMAGES_FOLDER):
            return "husbands"
        return "wives"

    async def tag_image(self, image_path: str | Path) -> TagPrediction:
        """Return and persist one image's traits without moving it mid-command."""

        await asyncio.to_thread(self.store.initialize)
        path = Path(image_path)
        stat = path.stat()
        cached = await asyncio.to_thread(
            self.store.get_cached,
            path,
            stat.st_size,
            stat.st_mtime_ns,
        )
        if cached is not None:
            return cached.to_prediction()

        digest = await asyncio.to_thread(_sha256_file, path)
        duplicate = await asyncio.to_thread(self.store.find_by_hash, digest)
        if duplicate is not None:
            prediction = duplicate.to_prediction()
        else:
            outcomes = await asyncio.to_thread(ModelManager.predict_paths, [path])
            outcome = outcomes[0]
            if isinstance(outcome, Exception):
                raise outcome
            prediction = outcome

        await asyncio.to_thread(
            self._store_and_maybe_move,
            _QueuedImage(path, self._library_for_path(path)),
            digest,
            prediction,
            stat.st_size,
            stat.st_mtime_ns,
            allow_move=False,
        )
        return prediction


image_feature_service = ImageFeatureService()


async def get_image_gender(image_path: str | Path) -> tuple[float, float]:
    try:
        prediction = await image_feature_service.tag_image(image_path)
        return prediction.male_probability, prediction.female_probability
    except Exception as exc:
        logger.warning(f"读取或生成图片特征失败 {image_path}: {exc}")
        return 0.0, 0.0


__all__ = ["ImageFeatureService", "get_image_gender", "image_feature_service"]

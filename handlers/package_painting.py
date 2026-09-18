from __future__ import annotations

import asyncio
from pathlib import Path
import time
from typing import Any

from nonebot import get_driver, on_command
from nonebot.adapters.onebot.v11 import (
    Bot,
    Event,
    GroupMessageEvent,
    MessageSegment,
)
from nonebot.params import CommandArg
from nonebot.permission import SUPERUSER

from zhenxun.services.log import logger

from ..config import paths
from ..services.alias_registry import game_alias_manager, get_game_name_from_alias
from ..services.painting_package import (
    PACKAGE_TTL_SECONDS,
    PaintingPackageStorage,
    PaintingPackageStore,
    build_painting_archive,
    count_game_paintings,
    list_game_prefixes,
    normalize_game_name,
    resolve_game_prefix,
    safe_archive_name,
)
from ..services.reactions import FAILURE_EMOJI_ID, SUCCESS_EMOJI_ID
from ..services.reactions import PROCESSING_EMOJI_ID as PACKAGING_EMOJI_ID
from ..services.reactions import set_command_emoji as _set_command_emoji

package_painting = on_command("打包立绘", priority=5, block=True, permission=SUPERUSER)

_data_dir = Path(paths.PLUGIN_DIR) / "data" / "painting_packages"
_work_dir = _data_dir / "work"
_store = PaintingPackageStore(_data_dir / "cache.db")
_storage = PaintingPackageStorage()
_package_locks: dict[str, asyncio.Lock] = {}
_cleanup_task: asyncio.Task[None] | None = None


def _message_id_from_result(result: Any) -> int | None:
    value = result.get("message_id") if isinstance(result, dict) else None
    if value is None:
        value = getattr(result, "message_id", None)
    try:
        message_id = int(value)
    except (TypeError, ValueError):
        return None
    return message_id if message_id != 0 else None


def _positive_int(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _message_sequence_from_info(info: Any) -> int | None:
    """Extract SnowLuma's positive QQ sequence used by reply elements."""
    if isinstance(info, dict):
        # SnowLuma message events expose message_seq. Its documented real_id may
        # still be the signed OneBot message hash on some versions.
        for key in ("message_seq", "real_id", "message_id"):
            if sequence := _positive_int(info.get(key)):
                return sequence
        return None
    for key in ("message_seq", "real_id", "message_id"):
        if sequence := _positive_int(getattr(info, key, None)):
            return sequence
    return None


async def _resolve_reply_sequence(bot: Bot, message_id: int) -> int | None:
    """Resolve a signed OneBot message ID to SnowLuma's replySeq."""
    for delay in (0.0, 0.1, 0.3):
        if delay:
            await asyncio.sleep(delay)
        try:
            info = await bot.get_msg(message_id=message_id)
        except Exception as e:
            logger.warning(f"获取消息序号失败 message_id={message_id}: {e}")
            continue
        if sequence := _message_sequence_from_info(info):
            return sequence
    return None


async def _send_command_reply(
    bot: Bot,
    event: GroupMessageEvent,
    message: str,
    reply_seq: int | None,
) -> Any:
    """Reply with SnowLuma's positive QQ sequence whenever it is available."""
    if reply_seq is None:
        reply_seq = await _resolve_reply_sequence(bot, event.message_id)
    if reply_seq is None:
        logger.warning(
            f"无法解析打包指令的 message_seq，回退到 OneBot message_id: "
            f"{event.message_id}"
        )
        reply_seq = event.message_id
    return await bot.send_group_msg(
        group_id=event.group_id,
        message=MessageSegment.reply(reply_seq) + MessageSegment.text(message),
    )


def _download_message(
    game_name: str,
    husbands_count: int,
    wives_count: int,
    url: str,
) -> str:
    return (
        f"「{game_name}」立绘打包完成\n"
        f"老公：{husbands_count} 张，"
        f"老婆：{wives_count} 张，"
        f"合计：{husbands_count + wives_count} 张\n"
        f"下载链接（压缩包上传后 1 天内有效）：\n{url}"
    )


async def _cache_download_message(
    bot: Bot,
    scope_key: str,
    game_key: str,
    game_name: str,
    result: Any,
    object_key: str,
    expires_at: float,
) -> None:
    message_id = _message_id_from_result(result)
    if message_id is None:
        logger.warning(f"立绘压缩包消息未返回 message_id，无法缓存: {game_name}")
        return

    reply_seq = await _resolve_reply_sequence(bot, message_id)
    if reply_seq is None:
        logger.warning(f"下载消息未返回可引用的 message_seq: message_id={message_id}")

    try:
        await asyncio.to_thread(
            _store.save_message,
            scope_key,
            game_key,
            message_id,
            reply_seq,
            object_key,
            expires_at,
        )
    except Exception as e:
        # The download message has already been sent and remains usable.
        logger.warning(f"保存立绘压缩包消息 ID 失败: {e}")


async def _delete_object(object_key: str) -> bool:
    try:
        await asyncio.to_thread(_storage.delete, object_key)
        await asyncio.to_thread(_store.mark_object_deleted, object_key)
        return True
    except Exception as e:
        logger.warning(f"删除过期立绘压缩包失败 {object_key}: {e}")
        return False


async def _cleanup_expired_packages() -> None:
    now = time.time()
    object_keys = await asyncio.to_thread(_store.list_expired_objects, now)
    for object_key in object_keys:
        await _delete_object(object_key)
    await asyncio.to_thread(_store.purge_expired_messages, now)


async def _cleanup_loop() -> None:
    while True:
        try:
            await _cleanup_expired_packages()
        except Exception as e:
            logger.warning(f"清理过期立绘压缩包失败: {e}")
        await asyncio.sleep(60 * 60)


driver = get_driver()


@driver.on_startup
async def _start_package_service() -> None:
    global _cleanup_task
    _work_dir.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(_store.initialize)
    if _cleanup_task is None or _cleanup_task.done():
        _cleanup_task = asyncio.create_task(_cleanup_loop())


@driver.on_shutdown
async def _stop_package_service() -> None:
    global _cleanup_task
    if _cleanup_task is None:
        return
    _cleanup_task.cancel()
    try:
        await _cleanup_task
    except asyncio.CancelledError:
        pass
    _cleanup_task = None


@package_painting.handle()
async def handle_package_painting(
    bot: Bot,
    event: Event,
    args=CommandArg(),
) -> None:
    if not isinstance(event, GroupMessageEvent):
        await package_painting.finish("打包立绘目前仅支持群聊使用。")

    requested_name = args.extract_plain_text().strip()
    if not requested_name:
        await _set_command_emoji(bot, event, FAILURE_EMOJI_ID)
        await package_painting.finish("请输入游戏名，例如：打包立绘 原神")

    try:
        known_game = game_alias_manager.get_game_info(requested_name) is not None
        canonical_name = await get_game_name_from_alias(requested_name)
        prefixes = await asyncio.to_thread(
            list_game_prefixes,
            Path(paths.HUSBANDS_IMAGES_FOLDER),
            Path(paths.WIVES_IMAGES_FOLDER),
        )
        game_prefix = resolve_game_prefix(
            canonical_name,
            prefixes,
            allow_fuzzy=known_game,
        )
    except Exception as e:
        logger.error(f"解析打包立绘游戏名失败 {requested_name}: {e}")
        await _set_command_emoji(bot, event, FAILURE_EMOJI_ID)
        await package_painting.finish("处理打包立绘指令失败，请稍后重试。")
    if game_prefix is None:
        await _set_command_emoji(bot, event, FAILURE_EMOJI_ID)
        await package_painting.finish(f"没有找到「{requested_name}」的立绘。")

    game_key = normalize_game_name(game_prefix)
    scope_key = f"{bot.self_id}:{event.group_id}"
    lock = _package_locks.setdefault(game_key, asyncio.Lock())

    async with lock:
        try:
            cached = await asyncio.to_thread(
                _store.get_valid,
                scope_key,
                game_key,
                time.time(),
            )
        except Exception as e:
            logger.error(f"读取 {game_prefix} 立绘压缩包缓存失败: {e}")
            await _set_command_emoji(bot, event, FAILURE_EMOJI_ID)
            await package_painting.finish("读取立绘压缩包缓存失败，请稍后重试。")
        if cached is not None:
            reply_seq = cached.reply_seq
            if reply_seq is None:
                reply_seq = await _resolve_reply_sequence(bot, cached.message_id)
                if reply_seq is not None:
                    await asyncio.to_thread(
                        _store.update_reply_seq,
                        scope_key,
                        game_key,
                        reply_seq,
                    )
            try:
                if reply_seq is None:
                    raise RuntimeError(
                        f"无法解析下载消息的 replySeq: {cached.message_id}"
                    )
                await bot.send_group_msg(
                    group_id=event.group_id,
                    message=(
                        MessageSegment.reply(reply_seq) + MessageSegment.text("1")
                    ),
                )
                await _set_command_emoji(bot, event, SUCCESS_EMOJI_ID)
                return
            except Exception as e:
                # The original message may have been recalled. Rebuild a usable
                # package, while retaining its object record for timed cleanup.
                logger.warning(f"引用已有立绘压缩包消息失败，准备重新打包: {e}")
                try:
                    await asyncio.to_thread(
                        _store.invalidate_message,
                        scope_key,
                        game_key,
                    )
                except Exception as invalidate_error:
                    logger.warning(
                        f"作废 {game_prefix} 立绘压缩包缓存失败: " f"{invalidate_error}"
                    )

        try:
            shared = await asyncio.to_thread(
                _store.get_shared_valid,
                game_key,
                time.time(),
            )
            if shared is None:
                husbands_count, wives_count = await asyncio.to_thread(
                    count_game_paintings,
                    game_prefix,
                    Path(paths.HUSBANDS_IMAGES_FOLDER),
                    Path(paths.WIVES_IMAGES_FOLDER),
                )
                shared = await asyncio.to_thread(
                    _store.promote_legacy_package,
                    game_key,
                    game_prefix,
                    husbands_count,
                    wives_count,
                    time.time(),
                )
        except Exception as e:
            logger.error(f"读取 {game_prefix} 共享立绘压缩包失败: {e}")
            await _set_command_emoji(bot, event, FAILURE_EMOJI_ID)
            await package_painting.finish("读取共享立绘压缩包失败，请稍后重试。")

        await _set_command_emoji(bot, event, PACKAGING_EMOJI_ID)
        command_reply_seq = await _resolve_reply_sequence(bot, event.message_id)

        if shared is not None:
            try:
                remaining_seconds = max(1, int(shared.expires_at - time.time()))
                url = await asyncio.to_thread(
                    _storage.generate_download_url,
                    shared.object_key,
                    remaining_seconds,
                )
                result = await _send_command_reply(
                    bot,
                    event,
                    _download_message(
                        shared.game_name,
                        shared.husbands_count,
                        shared.wives_count,
                        url,
                    ),
                    command_reply_seq,
                )
            except Exception as e:
                logger.error(f"复用 {game_prefix} 共享立绘压缩包失败: {e}")
                await _set_command_emoji(bot, event, FAILURE_EMOJI_ID)
                await package_painting.finish("复用共享立绘压缩包失败，请稍后重试。")

            await _set_command_emoji(bot, event, SUCCESS_EMOJI_ID)
            await _cache_download_message(
                bot,
                scope_key,
                game_key,
                shared.game_name,
                result,
                shared.object_key,
                shared.expires_at,
            )
            return

        try:
            await package_painting.send(f"开始打包「{game_prefix}」立绘，请稍候……")
        except Exception as e:
            logger.error(f"发送 {game_prefix} 立绘打包提示失败: {e}")
            await _set_command_emoji(bot, event, FAILURE_EMOJI_ID)
            return

        try:
            archive = await asyncio.to_thread(
                build_painting_archive,
                game_prefix,
                Path(paths.HUSBANDS_IMAGES_FOLDER),
                Path(paths.WIVES_IMAGES_FOLDER),
                _work_dir,
            )
        except FileNotFoundError:
            await _set_command_emoji(bot, event, FAILURE_EMOJI_ID)
            await package_painting.finish(f"没有找到「{game_prefix}」的立绘。")
        except Exception as e:
            logger.error(f"打包 {game_prefix} 立绘失败: {e}")
            await _set_command_emoji(bot, event, FAILURE_EMOJI_ID)
            await package_painting.finish("立绘打包失败，请稍后重试。")

        object_key: str | None = None
        expires_at: float | None = None
        result: Any = None
        try:
            object_key, url = await asyncio.to_thread(
                _storage.upload,
                archive.path,
                safe_archive_name(game_prefix),
            )
            expires_at = time.time() + PACKAGE_TTL_SECONDS
            await asyncio.to_thread(
                _store.save_shared_package,
                game_key,
                game_prefix,
                object_key,
                archive.husbands_count,
                archive.wives_count,
                expires_at,
            )

            result = await _send_command_reply(
                bot,
                event,
                _download_message(
                    game_prefix,
                    archive.husbands_count,
                    archive.wives_count,
                    url,
                ),
                command_reply_seq,
            )
        except Exception as e:
            logger.error(f"上传或发送 {game_prefix} 立绘压缩包失败: {e}")
            if object_key is not None:
                await _delete_object(object_key)
            await _set_command_emoji(bot, event, FAILURE_EMOJI_ID)
            await package_painting.finish("压缩包上传失败，请稍后重试。")
        finally:
            archive.path.unlink(missing_ok=True)

        await _set_command_emoji(bot, event, SUCCESS_EMOJI_ID)
        if object_key is not None and expires_at is not None:
            await _cache_download_message(
                bot,
                scope_key,
                game_key,
                game_prefix,
                result,
                object_key,
                expires_at,
            )

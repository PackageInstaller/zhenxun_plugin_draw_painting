from __future__ import annotations

import asyncio

from nonebot.adapters.onebot.v11 import Bot, Message, MessageEvent, MessageSegment
from nonebot.params import CommandArg

from zhenxun.services.log import logger

from ..matchers import feature_query
from ..services.command_guard import CommandHandler
from ..services.feature_query import (
    FeatureQueryError,
    download_query_image,
    inspect_query_image,
    select_image_data,
)
from ..services.reactions import (
    FAILURE_EMOJI_ID,
    PROCESSING_EMOJI_ID,
    SUCCESS_EMOJI_ID,
    set_command_emoji,
)

_active_users: set[tuple[str, int]] = set()
_MAX_PENDING = 3
_USAGE = "用法：特征查询 + 附带一张图片，或回复一张图片发送“特征查询”。"


@feature_query.handle(parameterless=[CommandHandler.dependency(block=True)])
async def handle_feature_query(
    bot: Bot, event: MessageEvent, args: Message = CommandArg()
) -> None:
    if args.extract_plain_text().strip() not in {"", "+", "＋"}:
        await set_command_emoji(bot, event, FAILURE_EMOJI_ID)
        await bot.send(event, _USAGE, reply_message=True)
        return
    try:
        data = select_image_data(
            event.message, event.reply.message if event.reply else None
        )
    except FeatureQueryError as exc:
        await set_command_emoji(bot, event, FAILURE_EMOJI_ID)
        await bot.send(event, str(exc), reply_message=True)
        return
    if data is None:
        await set_command_emoji(bot, event, FAILURE_EMOJI_ID)
        await bot.send(event, _USAGE, reply_message=True)
        return
    key = (bot.self_id, event.user_id)
    if key in _active_users:
        await set_command_emoji(bot, event, FAILURE_EMOJI_ID)
        await bot.send(
            event, "你已有特征查询正在处理中，请等待结果。", reply_message=True
        )
        return
    if len(_active_users) >= _MAX_PENDING:
        await set_command_emoji(bot, event, FAILURE_EMOJI_ID)
        await bot.send(event, "特征查询队列已满，请稍后重试。", reply_message=True)
        return
    _active_users.add(key)
    try:
        await set_command_emoji(bot, event, PROCESSING_EMOJI_ID)
        url = data.get("url")
        if not url and data.get("file"):
            info = await bot.get_image(file=data["file"])
            url = info.get("url")
        if not isinstance(url, str) or not url:
            raise FeatureQueryError("无法取得图片下载地址，请重新上传图片。")
        content = await download_query_image(url)
        annotated = await inspect_query_image(content)
        await bot.send(
            event,
            MessageSegment.image(annotated),
            reply_message=True,
        )
        await set_command_emoji(bot, event, SUCCESS_EMOJI_ID)
    except asyncio.CancelledError:
        await set_command_emoji(bot, event, FAILURE_EMOJI_ID)
        raise
    except FeatureQueryError as exc:
        await set_command_emoji(bot, event, FAILURE_EMOJI_ID)
        await bot.send(event, str(exc), reply_message=True)
    except Exception as exc:
        await set_command_emoji(bot, event, FAILURE_EMOJI_ID)
        logger.warning(f"立绘特征查询失败: {type(exc).__name__}")
        await bot.send(
            event, "特征查询处理失败，请稍后重试；原图未修改。", reply_message=True
        )
    finally:
        _active_users.discard(key)

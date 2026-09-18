from __future__ import annotations

from dataclasses import dataclass
import os
import re

from nonebot.adapters.onebot.v11 import Bot, Event
from nonebot.matcher import Matcher

from ..config import paths
from ..matchers import husbands_probability, wives_probability
from ..services.command_guard import CommandHandler
from ..services.numbers import parse_chinese_numeral
from ..services.statistics import calculate_game_stats, generate_and_send_stats


@dataclass(frozen=True)
class StatisticsKind:
    command_name: str
    image_folder: str
    matcher: type[Matcher]


WIFE = StatisticsKind("老婆概率", paths.WIVES_IMAGES_FOLDER, wives_probability)
HUSBAND = StatisticsKind(
    "老公概率",
    paths.HUSBANDS_IMAGES_FOLDER,
    husbands_probability,
)


async def _parse_limit(message: str, command_name: str, default: int) -> int:
    match = re.search(
        rf"{re.escape(command_name)}(\d+|[一二三四五六七八九十百]+)",
        message.replace(" ", ""),
    )
    if match is None:
        return default
    value = match.group(1)
    if value.isdigit():
        return int(value)
    return await parse_chinese_numeral(value) or default


async def _handle_statistics(bot: Bot, event: Event, kind: StatisticsKind) -> None:
    try:
        images = [
            name for name in os.listdir(kind.image_folder) if name.count("_") >= 1
        ]
        if not images:
            await bot.send(event, "没有找到任何有效的图片", reply_message=True)
            await kind.matcher.finish()
            return

        limit = await _parse_limit(
            event.get_plaintext().strip(),
            kind.command_name,
            len(images),
        )
        await generate_and_send_stats(
            bot,
            event,
            calculate_game_stats(images),
            limit,
            title=kind.command_name.replace("概率", "统计"),
        )
    except Exception:
        await bot.send(event, "处理请求时发生错误，请稍后重试", reply_message=True)
        await kind.matcher.finish()


@wives_probability.handle(parameterless=[CommandHandler.dependency(block=True)])
async def handle_wives_probability(bot: Bot, event: Event) -> None:
    await _handle_statistics(bot, event, WIFE)


@husbands_probability.handle(parameterless=[CommandHandler.dependency(block=True)])
async def handle_husbands_probability(bot: Bot, event: Event) -> None:
    await _handle_statistics(bot, event, HUSBAND)

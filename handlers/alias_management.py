from __future__ import annotations

import asyncio
from pathlib import Path
import shlex

from nonebot import on_command
from nonebot.adapters.onebot.v11 import Bot, Event
from nonebot.params import CommandArg
from nonebot.permission import SUPERUSER

from zhenxun.services.log import logger

from ..config import paths
from ..services.alias_registry import game_alias_manager
from ..services.game_aliases import AliasConfigError, add_game_aliases, normalize_alias
from ..services.painting_package import list_game_prefixes, resolve_game_prefix

alias_add = on_command(
    "别名添加",
    priority=5,
    block=True,
    permission=SUPERUSER,
)


def _split_arguments(raw_args: str) -> list[str]:
    lexer = shlex.shlex(raw_args, posix=True)
    lexer.whitespace_split = True
    lexer.commenters = ""
    # Only double quotes group values containing spaces. Apostrophes remain
    # usable in ordinary English game names and aliases.
    lexer.quotes = '"'
    return list(lexer)


async def _reply(bot: Bot, event: Event, message: str) -> None:
    await bot.send(event, message, reply_message=True)


@alias_add.handle()
async def handle_alias_add(bot: Bot, event: Event, args=CommandArg()) -> None:
    raw_args = args.extract_plain_text().strip()
    try:
        values = _split_arguments(raw_args)
    except ValueError as e:
        await _reply(bot, event, f"参数中的引号不完整：{e}")
        return

    if len(values) < 2:
        await _reply(
            bot,
            event,
            "格式：别名添加 原游戏名 别名1 [别名2 ...]\n"
            '名称或别名含空格时请使用英文双引号，例如：别名添加 原神 ys "Genshin Game"',
        )
        return

    requested_game, aliases = values[0], values[1:]
    configured_game = next(
        (
            game.name
            for game in game_alias_manager.games_config
            if normalize_alias(game.name) == normalize_alias(requested_game)
        ),
        None,
    )

    allow_create = False
    game_name = configured_game
    if game_name is None:
        prefixes = await asyncio.to_thread(
            list_game_prefixes,
            Path(paths.HUSBANDS_IMAGES_FOLDER),
            Path(paths.WIVES_IMAGES_FOLDER),
        )
        game_name = resolve_game_prefix(requested_game, prefixes)
        allow_create = game_name is not None
    if game_name is None:
        await _reply(
            bot,
            event,
            f"找不到原游戏「{requested_game}」。请使用别名列表中的标准游戏名，"
            "或本地立绘文件的准确游戏前缀。",
        )
        return

    try:
        result = await asyncio.to_thread(
            add_game_aliases,
            Path(paths.GAME_ALIASES_PATH),
            game_name,
            aliases,
            allow_create=allow_create,
        )
        game_alias_manager.reload_config()
        for alias in result.added:
            info = game_alias_manager.get_game_info(alias)
            if info is None or info.name != result.game_name:
                raise RuntimeError(f"刷新后无法解析新别名「{alias}」")
    except AliasConfigError as e:
        await _reply(bot, event, str(e))
        return
    except Exception as e:
        logger.error(f"添加游戏别名失败: {e}")
        await _reply(bot, event, "添加别名失败，请检查日志和别名配置。")
        return

    if not result.added:
        await _reply(
            bot,
            event,
            f"「{result.game_name}」没有新增别名；输入的别名均已存在。",
        )
        return

    message = f"已为「{result.game_name}」添加别名：{'、'.join(result.added)}"
    if result.duplicates:
        message += f"\n已跳过重复项：{'、'.join(result.duplicates)}"
    message += "\n别名缓存已刷新，立即生效。"
    await _reply(bot, event, message)

from __future__ import annotations

from dataclasses import dataclass
import os
import re

from nonebot.adapters.onebot.v11 import Bot, Event
from nonebot.matcher import Matcher
from nonebot.typing import T_State

from ..config import paths
from ..database import db_handler
from ..matchers import husbands_rename, wives_rename
from ..services.command_guard import CommandHandler
from ..services.messaging import send_image_message
from ..services.rename import perform_character_rename

NAME_PATTERN = re.compile(r"^[^\s_]+(_[^\s_]+)+$")


@dataclass(frozen=True)
class RenameKind:
    matcher: type[Matcher]
    card_type: str
    relation_label: str
    images_folder: str
    command: str
    draw_command: str


WIFE = RenameKind(
    matcher=wives_rename,
    card_type="Wife",
    relation_label="老婆",
    images_folder=paths.WIVES_IMAGES_FOLDER,
    command="老婆改名",
    draw_command="抽老婆",
)
HUSBAND = RenameKind(
    matcher=husbands_rename,
    card_type="Husband",
    relation_label="老公",
    images_folder=paths.HUSBANDS_IMAGES_FOLDER,
    command="老公改名",
    draw_command="抽老公",
)


def _matching_selected_images(kind: RenameKind, stored_name: str) -> list[str]:
    return [
        image
        for image in os.listdir(kind.images_folder)
        if os.path.splitext(image)[0] == stored_name
    ]


def _validation_error(stored_name: str, new_name: str) -> str | None:
    if not new_name:
        return "名字不能为空，请输入有效的名字"
    if new_name.split("_", maxsplit=1)[0] != stored_name.split("_", maxsplit=1)[0]:
        return "新名字中的游戏名与原始游戏名不一致，请保持游戏名一致。"
    if not NAME_PATTERN.fullmatch(new_name):
        return "名字格式不正确，请确保每段内容用下划线连接且无多余空格"
    return None


async def _perform_rename(
    kind: RenameKind,
    bot: Bot,
    event: Event,
    user_id: str,
    new_name: str,
) -> None:
    await perform_character_rename(
        bot,
        event,
        user_id,
        new_name,
        card_type=kind.card_type,
        images_folder=kind.images_folder,
        relation_label=kind.relation_label,
    )


async def _start_rename(
    kind: RenameKind, bot: Bot, event: Event, state: T_State
) -> None:
    user_id = str(event.get_user_id())
    stored_name = db_handler.get_card_name(user_id, kind.card_type)
    if not stored_name:
        await bot.send(
            event,
            f"你还没有{kind.relation_label}呢，快发送 {kind.draw_command} 来抽取吧",
            reply_message=True,
        )
        await kind.matcher.finish()

    matching_images = _matching_selected_images(kind, stored_name)
    if not matching_images:
        await bot.send(
            event,
            f"找不到你{kind.relation_label}的图片呢，请确认是否已被删除",
            reply_message=True,
        )
        await kind.matcher.finish()

    direct_match = re.fullmatch(
        rf"{re.escape(kind.command)}\s*(.+)", event.get_plaintext().strip()
    )
    if direct_match:
        new_name = direct_match.group(1).strip()
        if error := _validation_error(stored_name, new_name):
            await bot.send(event, error, reply_message=True)
        else:
            await _perform_rename(kind, bot, event, user_id, new_name)
        await kind.matcher.finish()

    image_paths = [os.path.join(kind.images_folder, image) for image in matching_images]
    image_names = "\n".join(os.path.splitext(image)[0] for image in matching_images)
    message = (
        f"当前图片名称如下：\n{image_names}\n"
        "名字格式：\n游戏名_角色名称_皮肤名称/阶段状态等信息(没有可不写)\n"
        "举例：解神者_少姜_蓝水乐园\n不知道或者点错了的话请发送 取消"
    )
    await send_image_message(bot, event, message, image_paths)
    state.update(
        awaiting_name=True,
        user_id=user_id,
        session_id=event.get_session_id(),
    )


async def _receive_new_name(
    kind: RenameKind, bot: Bot, event: Event, state: T_State
) -> None:
    user_id = str(event.get_user_id())
    if not state.get("awaiting_name") or state.get("user_id") != user_id:
        return

    new_name = event.get_plaintext().strip()
    if new_name == "取消":
        state["awaiting_name"] = False
        await bot.send(event, "已取消改名操作", reply_message=True)
        await kind.matcher.finish()

    stored_name = db_handler.get_card_name(user_id, kind.card_type)
    if not stored_name:
        state["awaiting_name"] = False
        await bot.send(event, f"当前没有{kind.relation_label}记录", reply_message=True)
        await kind.matcher.finish()

    if error := _validation_error(stored_name, new_name):
        await bot.send(event, error, reply_message=True)
        await kind.matcher.reject()

    await _perform_rename(kind, bot, event, user_id, new_name)
    state["awaiting_name"] = False
    await kind.matcher.finish()


@wives_rename.handle(parameterless=[CommandHandler.dependency(block=True)])
async def handle_wives_rename(bot: Bot, event: Event, state: T_State) -> None:
    await _start_rename(WIFE, bot, event, state)


@wives_rename.got("new_name", prompt="请输入新的名字：")
async def handle_got_new_wives_name(bot: Bot, event: Event, state: T_State) -> None:
    await _receive_new_name(WIFE, bot, event, state)


@husbands_rename.handle(parameterless=[CommandHandler.dependency(block=True)])
async def handle_husbands_rename(bot: Bot, event: Event, state: T_State) -> None:
    await _start_rename(HUSBAND, bot, event, state)


@husbands_rename.got("new_name", prompt="请输入新的名字：")
async def handle_got_new_husbands_name(bot: Bot, event: Event, state: T_State) -> None:
    await _receive_new_name(HUSBAND, bot, event, state)

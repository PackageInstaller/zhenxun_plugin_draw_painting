from __future__ import annotations

from dataclasses import dataclass
import os

from nonebot.adapters.onebot.v11 import Bot, Event, Message, MessageSegment
from nonebot.matcher import Matcher

from zhenxun.configs.config import BotConfig

from ..config import paths
from ..database import db_handler
from ..matchers import husbands_view, wives_view
from ..services.command_guard import CommandHandler
from ..services.draw_request import character_prefix, find_character_variants
from ..services.messaging import send_forward_msg_handler

FORWARD_IMAGE_LIMIT = 30


@dataclass(frozen=True)
class CollectionKind:
    role_name: str
    card_type: str
    image_folder: str
    matcher: type[Matcher]


WIFE = CollectionKind("老婆", "Wife", paths.WIVES_IMAGES_FOLDER, wives_view)
HUSBAND = CollectionKind(
    "老公",
    "Husband",
    paths.HUSBANDS_IMAGES_FOLDER,
    husbands_view,
)


def _forward_nodes(
    bot: Bot,
    kind: CollectionKind,
    display_name: str,
    images: list[str],
    *,
    page: int = 1,
    total_pages: int = 1,
):
    description = (
        f"第 {page}/{total_pages} 批，共 {len(images)} 张立绘"
        if total_pages > 1
        else "下面是所有立绘"
    )
    nodes = [
        {
            "type": "node",
            "data": {
                "name": str(BotConfig.self_nickname),
                "uin": bot.self_id,
                "content": f"你{kind.role_name}是 {display_name}：\n{description}\n",
            },
        }
    ]
    for image in images:
        stem = os.path.splitext(image)[0]
        image_path = os.path.join(kind.image_folder, image)
        nodes.append(
            {
                "type": "node",
                "data": {
                    "name": str(BotConfig.self_nickname),
                    "uin": bot.self_id,
                    "content": f"{stem}\n{MessageSegment.image(image_path)}",
                },
            }
        )
    return nodes


def _regular_message(kind: CollectionKind, display_name: str, images: list[str]):
    message = Message(f"你{kind.role_name}是 {display_name}：\n下面是所有立绘\n")
    for image in images:
        stem = os.path.splitext(image)[0]
        image_path = os.path.join(kind.image_folder, image)
        message += Message(f"{stem}\n") + MessageSegment.image(image_path) + "\n"
    return message


async def _finish_with_message(
    bot: Bot,
    event: Event,
    kind: CollectionKind,
    message: str,
) -> None:
    await bot.send(event, message, reply_message=True)
    await kind.matcher.finish()


async def _handle_collection(bot: Bot, event: Event, kind: CollectionKind) -> None:
    record = db_handler.get_card_name(str(event.get_user_id()), kind.card_type)
    if not record:
        await _finish_with_message(
            bot,
            event,
            kind,
            f"你还没有{kind.role_name}呢，快发送 抽{kind.role_name} 来抽取吧",
        )
        return

    display_name = character_prefix(record)
    images = find_character_variants(kind.image_folder, display_name)
    if not images:
        await _finish_with_message(
            bot,
            event,
            kind,
            f"找不到你{kind.role_name}的图片呢，请确认是否已被删除",
        )
        return

    try:
        if len(images) > 2:
            total_pages = (len(images) + FORWARD_IMAGE_LIMIT - 1) // FORWARD_IMAGE_LIMIT
            for offset in range(0, len(images), FORWARD_IMAGE_LIMIT):
                await send_forward_msg_handler(
                    bot,
                    event,
                    _forward_nodes(
                        bot,
                        kind,
                        display_name,
                        images[offset : offset + FORWARD_IMAGE_LIMIT],
                        page=offset // FORWARD_IMAGE_LIMIT + 1,
                        total_pages=total_pages,
                    ),
                )
        else:
            await bot.send(event, _regular_message(kind, display_name, images))
    except Exception as e:
        await bot.send(event, f"发送消息时发生错误：{e}", reply_message=True)
    await kind.matcher.finish()


@wives_view.handle(parameterless=[CommandHandler.dependency(block=True)])
async def handle_wives_view(bot: Bot, event: Event) -> None:
    await _handle_collection(bot, event, WIFE)


@husbands_view.handle(parameterless=[CommandHandler.dependency(block=True)])
async def handle_husbands_view(bot: Bot, event: Event) -> None:
    await _handle_collection(bot, event, HUSBAND)

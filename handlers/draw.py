from __future__ import annotations

import asyncio
from dataclasses import dataclass
import os
from pathlib import Path

from fuzzywuzzy import fuzz, process
from nonebot.adapters.onebot.v11 import Bot, Event, MessageSegment, Message
from nonebot.matcher import Matcher

from zhenxun.services.log import logger

from ..config import paths
from ..database import db_handler
from ..matchers import husbands_draw, wives_draw
from ..services.alias_registry import get_game_name_from_alias
from ..services.command_guard import CommandHandler
from ..services.draw_request import (
    DrawRequestError,
    find_character_variants,
    image_character_name,
    normalize_draw_name,
    parse_draw_request,
    select_character_images,
)
from ..services.draw_support import get_random_choice, improved_partial_word_match
from ..services.feature_aliases import FeatureAliasError, feature_alias_registry
from ..services.feature_query import render_path_recognition
from ..services.feature_report import gender_summary
from ..services.image_feature_store import normalize_image_path
from ..services.image_features import image_feature_service
from ..services.messaging import send_image_message


@dataclass(frozen=True)
class DrawKind:
    role_name: str
    card_type: str
    library: str
    image_folder: str
    matcher: type[Matcher]


WIFE = DrawKind("老婆", "Wife", "wives", paths.WIVES_IMAGES_FOLDER, wives_draw)
HUSBAND = DrawKind(
    "老公",
    "Husband",
    "husbands",
    paths.HUSBANDS_IMAGES_FOLDER,
    husbands_draw,
)


class DrawLookupError(ValueError):
    pass


def _library_games(images: list[str]) -> dict[str, str]:
    """Map normalized filename prefixes back to their display spelling."""
    games: dict[str, str] = {}
    for image in images:
        if "_" not in image:
            continue
        game_name = image.split("_", 1)[0]
        games.setdefault(game_name.casefold(), game_name)
    return games


async def _resolve_game(
    requested_game: str | None,
    library_games: dict[str, str],
) -> str | None:
    if requested_game is None:
        return None

    canonical = await get_game_name_from_alias(requested_game)
    normalized_games = set(library_games)
    fuzzy_matches = await improved_partial_word_match(
        list(canonical.casefold()),
        normalized_games,
    )
    if fuzzy_matches:
        return library_games[fuzzy_matches[0]]

    extracted = process.extractOne(
        canonical.casefold(),
        normalized_games,
        scorer=fuzz.partial_ratio,
    )
    if extracted is None or extracted[1] < 70:
        raise DrawLookupError("没有找到相关的游戏呢。")
    return library_games[extracted[0]]


def _selected_characters(game_name: str | None, kind: DrawKind) -> set[str]:
    if game_name is None:
        records = db_handler.get_all_selected_wives_or_husbands(kind.card_type)
    else:
        records = db_handler.get_selected_wives_or_husbands_by_game(
            game_name,
            kind.card_type,
        )
    return {
        normalize_draw_name(character)
        for record in records
        if (character := image_character_name(record)) is not None
    }


def _available_images(
    all_images: list[str],
    game_name: str | None,
    character_name: str | None,
    selected_characters: set[str],
) -> list[str]:
    if game_name is None:
        game_images = all_images
    else:
        normalized_game = game_name.casefold()
        game_images = [
            image
            for image in all_images
            if "_" in image and image.split("_", 1)[0].casefold() == normalized_game
        ]

    if character_name is not None:
        try:
            character_images = select_character_images(game_images, character_name)
        except DrawRequestError as exc:
            raise DrawLookupError(str(exc)) from exc
        if not character_images:
            raise DrawLookupError(
                f"没有在「{game_name}」中找到人物「{character_name}」。"
            )
        available = select_character_images(
            game_images, character_name, excluded_characters=selected_characters
        )
        if not available:
            raise DrawLookupError(
                f"「{game_name}」中匹配「{character_name}」的人物已经被抽走了。"
            )
        return available

    return [
        image
        for image in game_images
        if (character := image_character_name(image)) is not None
        and normalize_draw_name(character) not in selected_characters
    ]


async def _finish_with_message(
    bot: Bot,
    event: Event,
    kind: DrawKind,
    message: str,
) -> None:
    await bot.send(event, message, reply_message=True)
    await kind.matcher.finish()


async def _handle_draw(bot: Bot, event: Event, kind: DrawKind) -> None:
    try:
        request = parse_draw_request(event.get_plaintext(), kind.role_name)
    except DrawRequestError as e:
        await _finish_with_message(bot, event, kind, str(e))
        return

    try:
        resolved_features = feature_alias_registry.resolve_many(request.feature_names)
    except FeatureAliasError as e:
        await _finish_with_message(bot, event, kind, str(e))
        return

    all_images = list(os.listdir(kind.image_folder))
    try:
        game_name = await _resolve_game(
            request.game_name,
            _library_games(all_images),
        )
        selected = _selected_characters(game_name, kind)
        candidates = _available_images(
            all_images,
            game_name,
            request.character_name,
            selected,
        )
    except DrawLookupError as e:
        await _finish_with_message(bot, event, kind, str(e))
        return

    if candidates and resolved_features.tags:
        matching_paths = await asyncio.to_thread(
            image_feature_service.store.find_paths_by_tags,
            kind.library,
            resolved_features.tags,
        )
        candidates = [
            image
            for image in candidates
            if normalize_image_path(os.path.join(kind.image_folder, image))
            in matching_paths
        ]
        if not candidates:
            labels = " + ".join(resolved_features.labels)
            await _finish_with_message(
                bot,
                event,
                kind,
                f"没有找到同时满足「{labels}」且尚未被抽走的"
                f"{kind.role_name}。\n只匹配已完成多模型标记且主体性别明确的立绘；"
                "多人主体不明、模型冲突或仍在重新标记的图片暂不参与。\n"
                "中英文特征可混用；每一项都必须有对应的主体标签。"
                "可减少条件后逐个加回，检查哪项未被模型记录。",
            )
            return

    if not candidates:
        await _finish_with_message(
            bot,
            event,
            kind,
            f"所有{kind.role_name}都被抽完了捏",
        )
        return

    image_name = await get_random_choice(candidates)
    image_path = os.path.join(kind.image_folder, image_name)
    image_stem = os.path.splitext(image_name)[0]
    variants = find_character_variants(kind.image_folder, image_name)

    user_id = str(event.get_user_id())
    db_handler.update_draw_record(user_id, image_stem, kind.card_type)
    db_handler.log_draw_history_record(user_id, image_stem, kind.card_type)

    try:
        try:
            prediction = await image_feature_service.tag_image(image_path)
        except Exception as exc:
            logger.warning(f"抽取立绘时主体识别失败: {type(exc).__name__}")
            prediction = None
        if len(variants) > 1:
            variant_text = f"这个角色还有 {len(variants) - 1} 张立绘呢"
        else:
            variant_text = "这个角色只有一张立绘呢"
        feature_text = (
            f"\n特征筛选：{' + '.join(resolved_features.labels)}"
            if resolved_features.labels
            else ""
        )
        message = (
            f"你抽到的{kind.role_name}是 {image_stem}\n"
            f"{variant_text}\n"
            f"{gender_summary(prediction)}"
            f"{feature_text}"
        )
        # Draw cards (never the view commands) attach the annotated
        # recognition report in the SAME message; failures must never break a
        # draw, they just drop the reference image.
        annotated = None
        try:
            annotated = await asyncio.to_thread(
                render_path_recognition, Path(image_path)
            )
        except Exception:
            logger.warning(f"抽卡识别参考图生成失败，已跳过: {image_name}")
        segments = [
            MessageSegment.text(message),
            MessageSegment.image(f"file:///{image_path}"),
        ]
        if annotated is not None:
            segments.extend(
                (
                    MessageSegment.text(
                        "附：识别参考图（仅供了解判定依据，不影响抽取结果）"
                    ),
                    MessageSegment.image(annotated),
                )
            )
        await bot.send(event, Message(segments), reply_message=True)
    except Exception as e:
        await bot.send(event, f"发送消息时发生错误：{e}", reply_message=True)
    await kind.matcher.finish()


@wives_draw.handle(parameterless=[CommandHandler.dependency(block=True)])
async def handle_wives_draw(bot: Bot, event: Event) -> None:
    await _handle_draw(bot, event, WIFE)


@husbands_draw.handle(parameterless=[CommandHandler.dependency(block=True)])
async def handle_husbands_draw(bot: Bot, event: Event) -> None:
    await _handle_draw(bot, event, HUSBAND)

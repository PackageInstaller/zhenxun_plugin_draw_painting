from __future__ import annotations

from dataclasses import dataclass
import os

from nonebot.adapters.onebot.v11 import Bot, Event

from ..config import paths
from ..database import db_handler
from .painting_name import PaintingName, parse_painting_name


@dataclass(frozen=True)
class RenameResult:
    """Result of renaming all variants belonging to one character."""

    matched: tuple[str, ...]
    renamed: tuple[str, ...]
    failed: tuple[tuple[str, str], ...]
    current_updated: bool


def rename_character_images(
    user_id: str,
    new_name: str,
    *,
    card_type: str,
    images_folder: str,
) -> RenameResult:
    """Rename a selected character and its variants in one image library."""

    stored_name = db_handler.get_card_name(user_id, card_type)
    if not stored_name:
        return RenameResult((), (), (), False)

    stored = parse_painting_name(stored_name)
    replacement = parse_painting_name(new_name)
    if stored is None or replacement is None:
        return RenameResult((), (), (), False)

    matching_images = tuple(
        image
        for image in os.listdir(images_folder)
        if (parsed := parse_painting_name(image)) is not None
        and parsed.extension
        and parsed.identity == stored.identity
    )
    if not matching_images:
        return RenameResult((), (), (), False)

    renamed: list[str] = []
    failed: list[tuple[str, str]] = []
    current_updated = False

    for image in matching_images:
        old_base, extension = os.path.splitext(image)
        parsed = parse_painting_name(image)
        if parsed is None:
            continue
        is_selected = parsed.suffix == stored.suffix
        if is_selected:
            new_base = replacement.stem
        else:
            new_base = PaintingName(
                replacement.game, replacement.character, parsed.suffix
            ).stem

        old_path = os.path.join(images_folder, image)
        new_path = os.path.join(images_folder, f"{new_base}{extension}")
        try:
            if os.path.normcase(old_path) != os.path.normcase(new_path):
                os.rename(old_path, new_path)
            renamed.append(image)
            if is_selected:
                db_handler.update_renamed_record(user_id, old_base, new_name)
                current_updated = True
        except OSError as exc:
            failed.append((image, str(exc)))

    if current_updated:
        db_handler.update_draw_record(user_id, new_name, card_type)
    return RenameResult(
        matching_images,
        tuple(renamed),
        tuple(failed),
        current_updated,
    )


async def perform_character_rename(
    bot: Bot,
    event: Event,
    user_id: str,
    new_name: str,
    *,
    card_type: str,
    images_folder: str,
    relation_label: str,
) -> RenameResult:
    """Run the filesystem operation and report its outcome to the requester."""

    result = rename_character_images(
        user_id,
        new_name,
        card_type=card_type,
        images_folder=images_folder,
    )
    if not result.matched:
        await bot.send(event, "该图片已被删除，请重新抽取", reply_message=True)
        return result

    if result.failed:
        details = "\n".join(f"{name}: {reason}" for name, reason in result.failed)
        await bot.send(event, f"部分图片重命名失败：\n{details}", reply_message=True)

    if result.current_updated:
        await bot.send(
            event,
            f"已将你{relation_label}和相关立绘重命名为 {new_name}",
            reply_message=True,
        )
    else:
        await bot.send(
            event,
            "当前立绘未能完成重命名，抽取记录保持不变。",
            reply_message=True,
        )
    return result


async def perform_wife_rename(
    bot: Bot, event: Event, user_id: str, new_name: str
) -> RenameResult:
    """Backward-compatible wife rename entrypoint."""

    return await perform_character_rename(
        bot,
        event,
        user_id,
        new_name,
        card_type="Wife",
        images_folder=paths.WIVES_IMAGES_FOLDER,
        relation_label="老婆",
    )


async def perform_husband_rename(
    bot: Bot, event: Event, user_id: str, new_name: str
) -> RenameResult:
    """Backward-compatible husband rename entrypoint."""

    return await perform_character_rename(
        bot,
        event,
        user_id,
        new_name,
        card_type="Husband",
        images_folder=paths.HUSBANDS_IMAGES_FOLDER,
        relation_label="老公",
    )

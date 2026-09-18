from __future__ import annotations

import io
import os
from typing import TypedDict

from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from nonebot.adapters.onebot.v11 import Bot, GroupMessageEvent, MessageSegment

from ..config import paths


class GameStat(TypedDict):
    count: int
    characters: set[str]
    percentage: float
    char_count: int
    avg_skins: float


async def generate_and_send_stats(
    bot: Bot,
    event: GroupMessageEvent,
    game_stats: dict[str, GameStat],
    limit: int,
    *,
    title: str = "立绘统计",
) -> None:
    """Render paginated statistics and send them as a forwarded message."""

    font = FontProperties(fname=paths.FONT_PATH)
    display_games = sorted(
        game_stats.items(), key=lambda item: item[1]["count"], reverse=True
    )[:limit]
    batch_size = 10
    batches = [
        display_games[index : index + batch_size]
        for index in range(0, len(display_games), batch_size)
    ]
    forward_messages = []

    for page, batch in enumerate(batches, start=1):
        lines = []
        for rank, (game_name, stats) in enumerate(
            batch, start=(page - 1) * batch_size + 1
        ):
            lines.append(
                f"第 {rank} 位: {game_name}\n"
                f"    占比: {stats['percentage']:.2f}%\n"
                f"    角色数: {stats['char_count']}\n"
                f"    平均皮肤数: {stats['avg_skins']:.1f}"
            )

        max_length = max(
            (len(line) for block in lines for line in block.splitlines()), default=10
        )
        figure, axes = plt.subplots(
            figsize=(max(max_length * 0.15, 2), max(len(batch) * 0.6, 1))
        )
        axes.axis("off")
        figure.patch.set_facecolor("white")
        axes.text(
            0,
            0.5,
            "\n\n".join(lines),
            fontsize=12,
            ha="left",
            va="center",
            fontproperties=font,
            linespacing=1.5,
        )

        buffer = io.BytesIO()
        try:
            figure.savefig(
                buffer,
                format="png",
                bbox_inches="tight",
                transparent=False,
                dpi=120,
            )
            buffer.seek(0)
            content = MessageSegment.image(buffer) + MessageSegment.text(
                f"\n第 {page}/{len(batches)} 页"
            )
            forward_messages.append(
                {
                    "type": "node",
                    "data": {"name": title, "uin": bot.self_id, "content": content},
                }
            )
        finally:
            buffer.close()
            plt.close(figure)

    await bot.send_group_forward_msg(group_id=event.group_id, messages=forward_messages)


def calculate_game_stats(images: list[str]) -> dict[str, GameStat]:
    """Calculate image, character, percentage, and average-skin statistics."""

    total_images = len(images)
    game_stats: dict[str, GameStat] = {}
    for image in images:
        parts = os.path.splitext(image)[0].split("_")
        if len(parts) < 2:
            continue
        game_name, character_name = parts[:2]
        stats = game_stats.setdefault(
            game_name,
            {
                "count": 0,
                "characters": set(),
                "percentage": 0.0,
                "char_count": 0,
                "avg_skins": 0.0,
            },
        )
        stats["count"] += 1
        stats["characters"].add(character_name)

    for stats in game_stats.values():
        character_count = len(stats["characters"])
        stats["percentage"] = (
            stats["count"] / total_images * 100 if total_images else 0.0
        )
        stats["char_count"] = character_count
        stats["avg_skins"] = (
            stats["count"] / character_count if character_count else 0.0
        )
    return game_stats

from __future__ import annotations

import asyncio
import io

from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from nonebot.adapters.onebot.v11 import Bot, Event, MessageSegment, NoticeEvent

from zhenxun.services.log import logger

from ..config import paths
from ..database import db_handler
from ..matchers import help_command, help_emoji_response
from ..metadata import __plugin_meta__
from ..services.help_confirmation import help_manager

font_path = paths.FONT_PATH


@help_command.handle()
async def handle_help(bot: Bot, event: Event):
    """帮助指令，输出为图片形式"""

    user_id = str(event.get_user_id())
    confirmation = help_manager.get_confirmation(user_id)
    if confirmation:
        help_manager.remove_confirmation(user_id)

    font_prop = FontProperties(fname=font_path)
    help_text = __plugin_meta__.usage

    text_lines = help_text.split("\n")
    max_length = max(len(line) for line in text_lines)
    fig_width = max_length * 0.1
    fig_height = max(len(text_lines) * 0.1 + 1, 1)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.text(
        0, 0.5, help_text, fontsize=14, ha="left", va="center", fontproperties=font_prop
    )

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", transparent=False)
    buf.seek(0)

    if int(db_handler.get_user_info(user_id)["read_help"]) == 0:
        warning_message = (
            "现在您可以正常使用所有指令了。\n"
            "请注意，如果出现乱用指令的情况，将会被永久封禁。"
        )
    else:
        warning_message = ""
    message = MessageSegment.image(buf) + MessageSegment.text(warning_message)

    db_handler.mark_help_as_read(user_id)

    await bot.send(event, message, reply_message=True)

    buf.close()
    plt.close("all")


async def handle_help_confirmation(bot: Bot, event: Event):
    """处理帮助确认事件"""
    try:
        font_prop = FontProperties(fname=font_path)
        help_text = __plugin_meta__.usage

        text_lines = help_text.split("\n")
        max_length = max(len(line) for line in text_lines)
        fig_width = max_length * 0.1
        fig_height = max(len(text_lines) * 0.1 + 1, 1)

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.axis("off")
        fig.patch.set_facecolor("white")
        ax.text(
            0,
            0.5,
            help_text,
            fontsize=14,
            ha="left",
            va="center",
            fontproperties=font_prop,
        )

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", transparent=False)
        buf.seek(0)

        confirmation_text = (
            "\n请先阅读并同意帮助信息并在120秒内回应：\n"
            "贴第一个表情表示已阅读并同意\n"
            "贴第二个表情表示不同意\n"
            "如果您的QQ版本过低，也可以直接使用 帮助抽游戏立绘 来查看帮助信息。"
        )
        message = MessageSegment.image(buf) + MessageSegment.text(confirmation_text)

        response = await bot.send(event, message, reply_message=True)
        bot_message_id = response["message_id"]
        original_message_id = event.message_id

        await help_manager.add_confirmation(
            str(event.get_user_id()), bot_message_id, original_message_id, bot, event
        )

        emoji_ids = ["38", "417"]  # 同意、不同意
        for emoji_id in emoji_ids:
            await asyncio.sleep(0.1)
            await bot.call_api(
                "set_msg_emoji_like", message_id=bot_message_id, emoji_id=emoji_id
            )

    except Exception as e:
        logger.error(f"处理帮助确认时发生错误: {e}")
        await bot.send(event, "处理帮助确认时发生错误，请稍后重试。")
    finally:
        if "buf" in locals():
            buf.close()
        plt.close("all")


@help_emoji_response.handle()
async def handle_help_emoji_response(bot: Bot, event: NoticeEvent):
    """处理帮助确认的表情响应"""
    try:
        if event.notice_type == "group_msg_emoji_like":
            message_id = event.dict().get("message_id")
            user_id = str(event.dict().get("user_id"))
            emoji_id = event.dict().get("likes")[0].get("emoji_id")

            confirmation = help_manager.get_confirmation(user_id)
            if (
                confirmation
                and confirmation.message_id == message_id
                and confirmation.processing
            ):
                if emoji_id == "38":  # 同意
                    await help_manager.set_confirmed(user_id, True)
                    db_handler.mark_help_as_read(user_id)
                    reply_msg = (
                        MessageSegment.reply(confirmation.original_message_id)
                        + "感谢您同意霸王条款，现在您可以正常使用所有指令了。"
                    )
                    await bot.send(event, reply_msg)

                elif emoji_id == "417":  # 不同意
                    await help_manager.set_confirmed(user_id, False)
                    refusal_text = (
                        "您已选择不同意，将无法使用相关功能。"
                        "如需使用，请重新触发指令并同意霸王条款。"
                    )
                    reply_msg = (
                        MessageSegment.reply(confirmation.original_message_id)
                        + refusal_text
                    )
                    await bot.send(event, reply_msg)

                help_manager.remove_confirmation(user_id)
    except Exception as e:
        logger.error(f"处理表情响应时发生错误: {e}")

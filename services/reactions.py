"""Best-effort command status reactions shared by packaging and image queries."""

from nonebot.adapters.onebot.v11 import Bot, MessageEvent

from zhenxun.services.log import logger

PROCESSING_EMOJI_ID = "282"
SUCCESS_EMOJI_ID = "478"
FAILURE_EMOJI_ID = "479"


async def set_command_emoji(bot: Bot, event: MessageEvent, emoji_id: str) -> None:
    try:
        await bot.call_api(
            "set_msg_emoji_like", message_id=event.message_id, emoji_id=emoji_id
        )
    except Exception as exc:
        # A backend without reactions must still be able to receive its result.
        logger.warning(f"给指令贴表情失败 emoji_id={emoji_id}: {type(exc).__name__}")

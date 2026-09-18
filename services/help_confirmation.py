from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar

from nonebot.adapters.onebot.v11 import Bot, Event, MessageSegment

from zhenxun.services.log import logger


@dataclass
class HelpConfirmationState:
    message_id: int
    original_message_id: int
    confirmed: bool
    processing: bool
    start_time: datetime
    timeout_task: asyncio.Task[None] | None = None
    group_id: int | None = None


class HelpConfirmationManager:
    _instance: ClassVar[HelpConfirmationManager | None] = None
    _help_confirmations: ClassVar[dict[str, HelpConfirmationState]] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    async def add_confirmation(
        cls,
        user_id: str,
        message_id: int,
        original_message_id: int,
        bot: Bot,
        event: Event,
    ) -> None:
        if (
            user_id in cls._help_confirmations
            and cls._help_confirmations[user_id].timeout_task
        ):
            cls._help_confirmations[user_id].timeout_task.cancel()

        group_id = None
        if hasattr(event, "group_id"):
            group_id = event.group_id

        timeout_task = asyncio.create_task(cls._handle_timeout(user_id, bot, event))

        cls._help_confirmations[user_id] = HelpConfirmationState(
            message_id=message_id,
            original_message_id=original_message_id,
            confirmed=False,
            processing=True,
            start_time=datetime.now(),
            timeout_task=timeout_task,
            group_id=group_id,
        )

    @classmethod
    async def _handle_timeout(cls, user_id: str, bot: Bot, event: Event):
        """处理超时"""
        try:
            await asyncio.sleep(120)  # 120秒超时
            confirmation = cls.get_confirmation(user_id)
            if confirmation and not confirmation.confirmed:
                try:
                    timeout_message = (
                        "您没有及时确认帮助信息，请重新触发指令以查看帮助。"
                    )

                    if confirmation.group_id:
                        reply_msg = (
                            MessageSegment.reply(confirmation.original_message_id)
                            + timeout_message
                        )
                        await bot.send_group_msg(
                            group_id=confirmation.group_id, message=reply_msg
                        )
                    else:
                        reply_msg = (
                            MessageSegment.reply(confirmation.original_message_id)
                            + timeout_message
                        )
                        await bot.send(event, reply_msg)
                except Exception as e:
                    logger.error(f"发送超时消息失败: {e}")
                finally:
                    cls.remove_confirmation(user_id)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"处理超时时发生错误: {e}")
            cls.remove_confirmation(user_id)

    @classmethod
    def get_confirmation(cls, user_id: str) -> HelpConfirmationState | None:
        return cls._help_confirmations.get(user_id)

    @classmethod
    def remove_confirmation(cls, user_id: str) -> None:
        try:
            if user_id in cls._help_confirmations:
                if cls._help_confirmations[user_id].timeout_task:
                    cls._help_confirmations[user_id].timeout_task.cancel()
                cls._help_confirmations.pop(user_id)
        except Exception as e:
            logger.error(f"移除确认状态时发生错误: {e}")
            cls._help_confirmations.pop(user_id, None)

    @classmethod
    async def set_confirmed(cls, user_id: str, confirmed: bool = True) -> None:
        try:
            if user_id in cls._help_confirmations:
                cls._help_confirmations[user_id].confirmed = confirmed
                cls._help_confirmations[user_id].processing = False
                if cls._help_confirmations[user_id].timeout_task:
                    cls._help_confirmations[user_id].timeout_task.cancel()
        except Exception as e:
            logger.error(f"设置确认状态时发生错误: {e}")
            cls.remove_confirmation(user_id)

    @classmethod
    async def is_processing(cls, user_id: str, db_handler) -> bool:
        try:
            user_info = db_handler.get_user_info(user_id)
            if user_info and int(user_info["read_help"]) == 1:
                cls.remove_confirmation(user_id)
                return False

            confirmation = cls._help_confirmations.get(user_id)
            if not confirmation:
                return False

            if (datetime.now() - confirmation.start_time).total_seconds() > 120:  # 超时
                cls.remove_confirmation(user_id)
                return False

            return confirmation.processing
        except Exception as e:
            logger.error(f"检查处理状态时发生错误: {e}")
            cls.remove_confirmation(user_id)
            return False


help_manager = HelpConfirmationManager()

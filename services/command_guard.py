from __future__ import annotations

from datetime import datetime

from nonebot.adapters.onebot.v11 import Bot, Event, GroupMessageEvent
from nonebot.exception import FinishedException
from nonebot.matcher import Matcher
from nonebot.params import Depends

from zhenxun.services.log import logger

from ..database import db_handler
from .help_confirmation import help_manager
from .model import ModelManager


class CommandHandler:
    """命令处理类"""

    @staticmethod
    def dependency(block: bool = False) -> None:
        """
        命令处理依赖注入
        :param block: 是否在检查失败时阻止事件传播
        """

        async def _dependency(bot: Bot, matcher: Matcher, event: Event) -> bool:
            try:
                user_id = str(event.get_user_id())

                # 检查机器人在群内是否被禁言
                if isinstance(event, GroupMessageEvent):
                    group_member_info = await bot.get_group_member_info(
                        group_id=event.group_id, user_id=int(bot.self_id), no_cache=True
                    )
                    # 判断禁言状态：shut_up_timestamp大于当前时间戳才表示被禁言
                    current_timestamp = int(datetime.now().timestamp())
                    if (
                        group_member_info.get("shut_up_timestamp", 0)
                        > current_timestamp
                    ):
                        # 机器人被禁言，直接结束
                        if block:
                            await matcher.finish()
                        return False

                # 检查模型状态
                if not ModelManager.is_model_ready():
                    is_downloading, progress = ModelManager.get_download_status()
                    try:
                        if is_downloading:
                            await bot.send(
                                event,
                                f"模型正在下载中，请稍后再试\n当前下载进度：{progress:.1f}%",
                                reply_message=True,
                            )
                        else:
                            await bot.send(
                                event,
                                "模型文件不存在或下载失败，请联系管理员",
                                reply_message=True,
                            )
                    except Exception as e:
                        logger.error(f"发送消息失败: {e}")
                    if block:
                        await matcher.finish()
                    return False

                # 检查用户状态
                user_info = db_handler.get_user_info(user_id)
                if user_info and int(user_info["read_help"]) == 1:
                    return True

                if await help_manager.is_processing(user_id, db_handler):
                    if matcher.state.get("_command_name_") == "help":
                        return True
                    try:
                        await bot.send(
                            event,
                            "请先同意霸王条款再使用其他指令。",
                            reply_message=True,
                        )
                    except Exception as e:
                        logger.error(f"发送消息失败: {e}")
                    if block:
                        await matcher.finish()
                    return False

                if int(user_info["read_help"]) == 0:
                    try:
                        from ..handlers.help import handle_help_confirmation

                        await handle_help_confirmation(bot, event)
                    except Exception as e:
                        logger.error(f"处理帮助确认失败: {e}")
                    if block:
                        await matcher.finish()
                    return False

                return True

            except FinishedException:
                raise
            except Exception as e:
                logger.error(f"依赖检查时发生错误: {e}")
                if block:
                    await matcher.finish()
                return False

        return Depends(_dependency)

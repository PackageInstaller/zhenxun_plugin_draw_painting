from __future__ import annotations

from nonebot.adapters.onebot.v11 import (
    Bot,
    GroupMessageEvent,
    Message,
    MessageSegment,
)


async def get_original_sender(bot: Bot, message_id: int) -> int:
    """递归获取最初消息的发送者 ID,传入event.reply.message_id"""
    try:
        msg = await bot.get_msg(message_id=message_id)
        message_chain = msg["message"]
        for segment in message_chain:
            if segment["type"] == "reply":
                reply_message_id = int(segment["data"]["id"])
                return await get_original_sender(bot, reply_message_id)
        return msg["sender"]["user_id"]
    except Exception:
        return 0


async def send_image_message(bot, event, title, image_paths):
    """发送图片消息"""
    image_segments = [MessageSegment.image(f"file:///{img}") for img in image_paths]
    message = Message([MessageSegment.text(title), *image_segments])
    await bot.send(event, message, reply_message=True)


async def send_forward_msg_handler(bot, event, *args):
    """
    统一消息发送处理器
    :param bot: 机器人实例
    :param event: 事件对象
    :param name: 用户名称
    :param uin: 用户QQ号
    :param msgs: 消息内容列表
    :param messages: 合并转发的消息列表（字典格式）
    :param msg_type: 关键字参数，可用于传递特定命名参数
    """

    if len(args) == 3:
        name, uin, msgs = args
        messages = [
            {"type": "node", "data": {"name": name, "uin": uin, "content": msg}}
            for msg in msgs
        ]
        if isinstance(event, GroupMessageEvent):
            await bot.call_api(
                "send_group_forward_msg", group_id=event.group_id, messages=messages
            )
        else:
            await bot.call_api(
                "send_private_forward_msg", user_id=event.user_id, messages=messages
            )
    elif len(args) == 1 and isinstance(args[0], list):
        messages = args[0]
        if isinstance(event, GroupMessageEvent):
            await bot.call_api(
                "send_group_forward_msg", group_id=event.group_id, messages=messages
            )
        else:
            await bot.call_api(
                "send_private_forward_msg", user_id=event.user_id, messages=messages
            )
    else:
        raise ValueError("参数数量或类型不匹配")

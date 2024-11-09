import os
import platform
import struct
import random
import numpy as np
import torch
from .handler import *
from PIL import Image
from typing import Dict, List
from fuzzywuzzy import fuzz
from nonebot.params import Depends
from nonebot.adapters.onebot.v11 import (
    Bot,
    Event,
    MessageSegment,
    GroupMessageEvent,
    Message
)
from .handler import *
from . import deep_danbooru_model

plugin_dir = os.path.dirname(__file__)
model_dir = os.path.join(plugin_dir, "Model")
husbands_images_folder = os.path.join(plugin_dir, "Husbands")
wives_images_folder = os.path.join(plugin_dir, "Wives")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = deep_danbooru_model.DeepDanbooruModel()
model.load_state_dict(torch.load(os.path.join(model_dir, "model-resnet_custom_v3.pt"), weights_only=True))
model = model.to(DEVICE)
model.eval()


class CommandHandler:
    """
    命令处理类
    """
    def dependency() -> None:
        async def dependency(bot: Bot, matcher, event: Event):
            user_id = str(event.get_user_id())

            if db_handler.get_user_info(user_id)['read_help'] == 0:
                try:
                    await bot.send(event, "使用前请先阅读帮助信息，发送 帮助抽游戏立绘 获取帮助信息。", reply_message=True)
                except Exception as e:
                    await bot.send(event, f"发送消息时发生错误：{str(e)}", reply_message=True)
                await matcher.finish()

        return Depends(dependency)
        

# 游戏别名
GAME_ALIASES: Dict[str, List[str]] = {
    "崩坏2": ["崩坏二", "崩2", "崩二"],
    "崩坏3": ["崩坏三", "崩3", "崩三"],
    "原神": ["genshinimpact", "ys", "原神"],
    "明日方舟": ["arknights", "方舟"],
    "碧蓝航线": ["azurlane", "blhx"],
    "碧蓝幻想": ["gbf", "肝报废"],
    "少女前线": ["sgb", "少前"],
    "少女前线2：追放": ["sgb2", "少前2", "少前追放", "追放"],
    "歧路旅人: 大陆的霸者": ["大坝", "大霸"],
    "崩坏：星穹铁道": ["星铁", "崩铁"],
    "少女前线：云图计划": ["云图"],
    "玛娜希斯回响": ["麻辣鸡丝"],
    "边狱公司": ["084", "宝宝巴士", "鳊鱼公司", "鳊鱼巴士"],
    "千年之旅": ["千年"],
    "为了谁的炼金术师": ["为谁而炼金"],
    "战舰少女R": ["舰R"],
}

# 别名游戏名
async def get_game_name_from_alias(alias: str) -> str:
    alias = alias.lower()
    for game, aliases in GAME_ALIASES.items():
        if alias in [a.lower() for a in aliases]:
            return game
    return alias


async def determine_gender(img_path):
    """判断图片性别"""
    try:
        pic = Image.open(img_path).convert("RGB").resize((512, 512))
        a = np.expand_dims(np.array(pic, dtype=np.float32), 0) / 255
        x = torch.from_numpy(a)
        x = x.to(DEVICE)

        # 推理
        with torch.no_grad():
            y = model(x)[0].detach()
            if DEVICE.type == "cuda":
                y = y.cpu()
            y = y.numpy()
            
            boy_confidence = sum(y[i] for i, tag in enumerate(model.tags) if "boy" in tag)
            girl_confidence = sum(y[i] for i, tag in enumerate(model.tags) if "girl" in tag)
            return boy_confidence, girl_confidence
    except Exception:
        return 0.0, 0.0


async def improved_partial_word_match(game_name_parts, game_names_in_library):
    """分词匹配打分"""
    matched_games = []

    # 优先查找完全匹配的游戏名
    for game in game_names_in_library:
        if "".join(game_name_parts) == game:
            return [game]

    for game in game_names_in_library:
        match_score = fuzz.partial_ratio("".join(game_name_parts), game)

        # 部分匹配，且长度差距较小，则增加匹配得分
        if len(game) > len(game_name_parts):
            if "".join(game_name_parts) in game:
                match_score += 10  # 权重

        matched_games.append((game, match_score))

    matched_games.sort(key=lambda x: x[1], reverse=True)

    return [game for game, score in matched_games if score >= 70]



async def get_random_choice(choices):
    """从给定选项中随机选择一个元素"""
        
    if platform.system() == "Linux":
        try:
            with open("/dev/random", "rb") as f:
                random_bytes = f.read(4)
                random_int = struct.unpack("I", random_bytes)[0]
            return choices[random_int % len(choices)]
        except (IOError, OSError) as e:
            return random.choice(choices)
    else:
        return random.choice(choices)


async def get_original_sender(bot: Bot, message_id: int) -> int:
    """递归获取最初消息的发送者 ID,传入event.reply.message_id"""
    try:
        msg = await bot.get_msg(message_id=message_id)
        message_chain = msg['message']
        for segment in message_chain:
            if segment['type'] == 'reply':
                reply_message_id = int(segment['data']['id'])
                return await get_original_sender(bot, reply_message_id)
        return msg['sender']['user_id']
    except Exception as e:
        return None


async def send_image_message(bot, event, title, image_paths):
    """发送图片消息"""
    image_segments = [MessageSegment.image(f'file:///{img}') for img in image_paths]
    message = Message([MessageSegment.text(title)] + image_segments)
    await bot.send(event, message, reply_message=True)


async def format_time(seconds: int) -> str:
    """将秒数转换为更大的时间单位"""
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)

    time_parts = []
    
    if days > 0:
        time_parts.append(f"{days}天")
    if hours > 0:
        time_parts.append(f"{hours}小时")
    if minutes > 0:
        time_parts.append(f"{minutes}分钟")
    if seconds > 0 or not time_parts:  # 0秒
        time_parts.append(f"{seconds}秒")

    return ''.join(time_parts)


async def parse_chinese_numeral(text):
    """将中文数字转换为阿拉伯数字，支持百位、十位"""
    chinese_numerals = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
     '六': 6, '七': 7, '八': 8, '九': 9, '十': 10, '百': 100}
    result = 0

    if '百' in text:
        parts = text.split('百')
        if parts[0]:
            result += chinese_numerals.get(parts[0], 1) * 100
        else:
            result += 100
        # 百位后的部分
        if len(parts) > 1 and parts[1]:
            text = parts[1]
        else:
            text = ''

    if '十' in text:
        parts = text.split('十')
        if parts[0]:
            result += chinese_numerals.get(parts[0], 1) * 10
        else:
            result += 10
        if len(parts) > 1 and parts[1]:
            result += chinese_numerals.get(parts[1], 0)
    else:
        for char in text:
            if char in chinese_numerals:
                result = result * 10 + chinese_numerals[char]

    return result if result > 0 else None

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
        messages = [{"type": "node", "data": {"name": name, "uin": uin, "content": msg}} for msg in msgs]
        if isinstance(event, GroupMessageEvent):
            await bot.call_api("send_group_forward_msg", group_id=event.group_id, messages=messages)
        else:
            await bot.call_api("send_private_forward_msg", user_id=event.user_id, messages=messages)
    elif len(args) == 1 and isinstance(args[0], list):
        messages = args[0]
        if isinstance(event, GroupMessageEvent):
            await bot.call_api("send_group_forward_msg", group_id=event.group_id, messages=messages)
        else:
            await bot.call_api("send_private_forward_msg", user_id=event.user_id, messages=messages)
    else:
        raise ValueError("参数数量或类型不匹配")


async def perform_wife_rename(bot: Bot, event: Event, user_id: str, new_name: str):
    """老婆重命名"""
    result = db_handler.get_card_name(user_id, 'Wife')
    old_name = result[0]
    game_name, char_name, *rest = old_name.split("_")
    stored_image_suffix = "_".join(rest) if rest else ""
    renamed_images = db_handler.get_renamed_images()
    matching_images = [
        img for img in os.listdir(wives_images_folder)
        if os.path.splitext(img)[0].startswith(f"{game_name}_{char_name}")
    ]

    if not matching_images:
        try:
            await bot.send(event, f"该图片已被删除，请重新抽取", reply_message=True)
        except Exception as e:
            await bot.send(event, f"发送消息时发生错误：{str(e)}", reply_message=True)

    for img in matching_images:
        old_path = os.path.join(wives_images_folder, img)

        # 已经改名过，跳过处理
        if img in [ri[0] for ri in renamed_images]:
            continue

        img_name_parts = os.path.splitext(img)[0].split("_")
        img_suffix = "_".join(img_name_parts[2:]) if len(img_name_parts) > 2 else ""  # 提取后缀部分

        if img_suffix == stored_image_suffix:
            new_path = os.path.join(wives_images_folder, new_name + os.path.splitext(img)[1])
            db_handler.update_renamed_record(user_id, os.path.splitext(img)[0], new_name)
        else:
            # 其他匹配图片保留原有后缀部分，改名前缀为新名字的游戏名和角色名，但不记录到数据库
            new_base_name = f"{new_name.split('_')[0]}_{new_name.split('_')[1]}_{img_suffix}"
            new_path = os.path.join(wives_images_folder, new_base_name + os.path.splitext(img)[1])

        try:
            os.rename(old_path, new_path)
        except Exception as e:
            await bot.send(event, f"重命名失败：{str(e)}", reply_message=True)

    db_handler.update_draw_record(user_id, new_name, 'Wife')

    try:
        await bot.send(event, f"已将你老婆和相关立绘重命名为 {new_name}", reply_message=True)
    except Exception as e:
        await bot.send(event, f"发送消息时发生错误：{str(e)}", reply_message=True)


async def perform_husband_rename(bot: Bot, event: Event, user_id: str, new_name: str):
    """老公重命名"""
    result = db_handler.get_card_name(user_id, 'Husband')
    old_name = result[0]
    game_name, char_name, *rest = old_name.split("_")
    stored_image_suffix = "_".join(rest) if rest else ""
    renamed_images = db_handler.get_renamed_images()
    matching_images = [
        img for img in os.listdir(husbands_images_folder)
        if os.path.splitext(img)[0].startswith(f"{game_name}_{char_name}")
    ]

    if not matching_images:
        try:
            await bot.send(event, f"该图片已被删除，请重新抽取", reply_message=True)
        except Exception as e:
            await bot.send(event, f"发送消息时发生错误：{str(e)}", reply_message=True)

    for img in matching_images:
        old_path = os.path.join(husbands_images_folder, img)

        # 已经改名过，跳过处理
        if img in [ri[0] for ri in renamed_images]:
            continue

        img_name_parts = os.path.splitext(img)[0].split("_")
        img_suffix = "_".join(img_name_parts[2:]) if len(img_name_parts) > 2 else ""

        if img_suffix == stored_image_suffix:
            new_path = os.path.join(husbands_images_folder, new_name + os.path.splitext(img)[1])
            db_handler.update_renamed_record(user_id, os.path.splitext(img)[0], new_name)
        else:
            # 其他匹配图片保留原有后缀部分，改名前缀为新名字的游戏名和角色名，但不记录到数据库
            new_base_name = f"{new_name.split('_')[0]}_{new_name.split('_')[1]}_{img_suffix}"
            new_path = os.path.join(husbands_images_folder, new_base_name + os.path.splitext(img)[1])

        try:
            os.rename(old_path, new_path)
        except Exception as e:
            await bot.send(event, f"重命名失败：{str(e)}", reply_message=True)

    db_handler.update_draw_record(user_id, new_name, 'Wife')

    try:
        await bot.send(event, f"已将你老公和相关立绘重命名为 {new_name}", reply_message=True)
    except Exception as e:
        await bot.send(event, f"发送消息时发生错误：{str(e)}", reply_message=True)


async def is_exact_match(img_name: str, stored_name: str) -> bool:
    """检查图片名是否与存储的名字精确匹配游戏名和角色名部分。"""
    img_base_name = os.path.splitext(img_name)[0]
    img_parts = img_base_name.split("_")
    stored_parts = stored_name.split("_")

    # 确保文件名至少包含游戏名和角色名两个部分
    if len(img_parts) < 2 or len(stored_parts) < 2:
        return False

    # 游戏名和角色名
    return img_parts[:2] == stored_parts[:2]
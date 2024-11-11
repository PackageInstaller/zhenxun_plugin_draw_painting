import os
import io
import platform
import struct
import random
import numpy as np
import torch
from PIL import Image
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Union, Optional
from fuzzywuzzy import fuzz
from nonebot.params import Depends
from nonebot.adapters.onebot.v11 import (
    Bot,
    Event,
    MessageSegment,
    GroupMessageEvent,
    Message
)
from ..config import paths, device
from ..database import db_handler
from . import deep_danbooru_model
from matplotlib.font_manager import FontProperties
from matplotlib import pyplot as plt

husbands_images_folder = paths.HUSBANDS_IMAGES_FOLDER
wives_images_folder = paths.WIVES_IMAGES_FOLDER
drop_folder = paths.DROP_FOLDER
font_path = paths.FONT_PATH
model = deep_danbooru_model.DeepDanbooruModel()
model.load_state_dict(torch.load(os.path.join(paths.MODEL_DIR, "model-resnet_custom_v3.pt"), weights_only=True))
model = model.to(device.DEVICE)
model.eval()


class CommandHandler:
    """命令处理类"""
    def dependency() -> None:
        async def dependency(bot: Bot, matcher, event: Event):
            user_id = str(event.get_user_id())
            
            # 检查是否正在处理帮助确认
            if help_manager.is_processing(user_id):
                await bot.send(event, "请先完成帮助信息的确认流程。", reply_message=True)
                await matcher.finish()
            
            if int(db_handler.get_user_info(user_id)['read_help']) == 0:
                from .. import handle_help_confirmation
                await handle_help_confirmation(bot, event)
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
        x = x.to(device.DEVICE)

        # 推理
        with torch.no_grad():
            y = model(x)[0].detach()
            if device.DEVICE.type == "cuda":
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
    stored_name = db_handler.get_card_name(user_id, 'Wife')
    game_name, char_name, *rest = stored_name.split("_")
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
    stored_name = db_handler.get_card_name(user_id, 'Husband')
    game_name, char_name, *rest = stored_name.split("_")
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


async def generate_and_send_stats(bot: Bot, event: Event, game_stats: Dict[str, Dict], limit: int):
    """
    生成并发送统计图表
    
    Args:
        bot: Bot实例
        event: 事件实例
        game_stats: 游戏统计信息
        limit: 显示数量限制
    """
    font_prop = FontProperties(fname=font_path)
    sorted_games = sorted(game_stats.items(), key=lambda x: x[1]["count"], reverse=True)
    display_games = sorted_games[:limit]
    
    text_lines = []
    max_length = 0
    
    for rank, (game_name, stats) in enumerate(display_games, 1):
        line = (
            f"第 {rank} 位: {game_name}\n"
            f"    占比: {stats['percentage']:.2f}%\n"
            f"    角色数: {stats['char_count']}\n"
            f"    平均皮肤数: {stats['avg_skins']:.1f}"
        )
        text_lines.append(line)
        max_length = max(max_length, max(len(l) for l in line.split('\n')))

    # 设置图表大小
    fig_width = max_length * 0.15
    fig_height = max(len(display_games) * 0.6, 1)
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('off')
    fig.patch.set_facecolor('white')
    
    # 添加文本
    ax.text(0, 0.5, "\n\n".join(text_lines), 
            fontsize=12, ha='left', va='center', 
            fontproperties=font_prop,
            linespacing=1.5)

    # 保存并发送图片
    buf = io.BytesIO()
    try:
        plt.savefig(buf, format='png', bbox_inches='tight', transparent=False, dpi=120)
        buf.seek(0)
        await bot.send(event, MessageSegment.image(buf), reply_message=True)
    finally:
        buf.close()
        plt.close(fig)

def calculate_game_stats(images: List[str]) -> Dict[str, Dict[str, Union[int, float]]]:
    """
    计算游戏统计信息
    
    Args:
        images: 图片文件名列表
        
    Returns:
        Dict: 包含每个游戏统计信息的字典
            {
                "游戏名": {
                    "count": 图片总数,
                    "characters": 角色集合,
                    "percentage": 占总数百分比,
                    "char_count": 角色数量,
                    "avg_skins": 平均皮肤数
                }
            }
    """
    total_images = len(images)
    game_stats = {}

    # 收集基础统计信息
    for img in images:
        parts = os.path.splitext(img)[0].split("_")
        if len(parts) > 1:
            game_name = parts[0]
            if game_name not in game_stats:
                game_stats[game_name] = {
                    "count": 0,
                    "characters": set(),
                    "percentage": 0,
                    "char_count": 0,
                    "avg_skins": 0
                }
            game_stats[game_name]["count"] += 1
            if len(parts) > 1:
                game_stats[game_name]["characters"].add(parts[1])

    # 计算百分比和其他统计数据
    for game_name, stats in game_stats.items():
        count = stats["count"]
        char_count = len(stats["characters"])
        stats.update({
            "percentage": (count / total_images) * 100,
            "char_count": char_count,
            "avg_skins": count / char_count if char_count > 0 else 0
        })

    return game_stats


@dataclass
class HelpConfirmationState:
    message_id: int
    confirmed: bool
    processing: bool
    start_time: datetime

class HelpConfirmationManager:
    _instance = None
    _help_confirmations: Dict[str, HelpConfirmationState] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(HelpConfirmationManager, cls).__new__(cls)
        return cls._instance

    @classmethod
    def add_confirmation(cls, user_id: str, message_id: int) -> None:
        cls._help_confirmations[user_id] = HelpConfirmationState(
            message_id=message_id,
            confirmed=False,
            processing=True,
            start_time=datetime.now()
        )

    @classmethod
    def get_confirmation(cls, user_id: str) -> Optional[HelpConfirmationState]:
        return cls._help_confirmations.get(user_id)

    @classmethod
    def remove_confirmation(cls, user_id: str) -> None:
        cls._help_confirmations.pop(user_id, None)

    @classmethod
    def set_confirmed(cls, user_id: str, confirmed: bool = True) -> None:
        if user_id in cls._help_confirmations:
            cls._help_confirmations[user_id].confirmed = confirmed
            cls._help_confirmations[user_id].processing = False

    @classmethod
    def is_processing(cls, user_id: str) -> bool:
        confirmation = cls._help_confirmations.get(user_id)
        if not confirmation:
            return False
        if (datetime.now() - confirmation.start_time).total_seconds() > 60:
            cls.remove_confirmation(user_id)
            return False
        return confirmation.processing

help_manager = HelpConfirmationManager()
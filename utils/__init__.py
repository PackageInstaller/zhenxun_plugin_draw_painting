# 感谢 https://github.com/AUTOMATIC1111/TorchDeepDanbooru 的实现
import os
import io
import yaml
import hashlib
import aiohttp
import platform
import struct
import random
import numpy as np
import torch
import asyncio
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from zhenxun.services.log import logger
from matplotlib.font_manager import FontProperties
from matplotlib import pyplot as plt
from tqdm.asyncio import tqdm
from dataclasses import dataclass

husbands_images_folder = paths.HUSBANDS_IMAGES_FOLDER
wives_images_folder = paths.WIVES_IMAGES_FOLDER
drop_folder = paths.DROP_FOLDER
font_path = paths.FONT_PATH
game_aliases_path = paths.GAME_ALIASES_PATH
class ModelManager:
    _instance = None
    _model = None
    _MODEL_URL = "https://github.com/AUTOMATIC1111/TorchDeepDanbooru/releases/download/v1/model-resnet_custom_v3.pt"
    _MODEL_SHA256 = "3841542cda4dd037da12a565e854b3347bb2eec8fbcd95ea3941b2c68990a355"
    _is_downloading = False
    _download_progress = 0
    _max_retries = 3
    
    @classmethod
    def verify_model(cls, file_path: str) -> bool:
        """验证模型文件的SHA256哈希值"""
        if not os.path.exists(file_path):
            return False
            
        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest() == cls._MODEL_SHA256
        except Exception as e:
            logger.error(f"验证模型文件时发生错误: {e}")
            return False
    
    @classmethod
    async def download_model(cls):
        """异步下载模型文件，支持断点续传和重试"""
        model_path = os.path.join(paths.MODEL_DIR, "model-resnet_custom_v3.pt")
        temp_path = f"{model_path}.temp"
        os.makedirs(paths.MODEL_DIR, exist_ok=True)
        
        cls._is_downloading = True
        cls._download_progress = 0
        retry_count = 0
        
        if os.path.exists(temp_path):
            initial_size = os.path.getsize(temp_path)
            logger.info(f"找到临时文件，已下载 {initial_size} 字节")
        else:
            initial_size = 0
        
        while retry_count < cls._max_retries:
            try:
                headers = {'Range': f'bytes={initial_size}-'} if initial_size > 0 else {}
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(cls._MODEL_URL, headers=headers) as response:
                        if response.status not in [200, 206]:
                            logger.error(f"下载失败: HTTP {response.status}")
                            retry_count += 1
                            await asyncio.sleep(1)
                            continue
                        
                        total_size = int(response.headers.get('content-length', 0)) + initial_size
                        
                        with open(temp_path, 'ab' if initial_size > 0 else 'wb') as f:
                            pbar = tqdm(
                                desc=f"下载模型 ({retry_count + 1}/{cls._max_retries})",
                                initial=initial_size,
                                total=total_size,
                                unit='iB',
                                unit_scale=True,
                                unit_divisor=1024
                            )
                            try:
                                async for chunk in response.content.iter_chunked(1024):
                                    size = f.write(chunk)
                                    pbar.update(size)
                                    cls._download_progress = (pbar.n / total_size) * 100 if total_size > 0 else 0
                            finally:
                                pbar.close()
                
                if cls.verify_model(temp_path):
                    if os.path.exists(model_path):
                        os.remove(model_path)
                    os.rename(temp_path, model_path)
                    logger.info("模型文件下载完成并验证通过")
                    cls._is_downloading = False
                    cls._download_progress = 100
                    return True
                else:
                    logger.error("模型文件验证失败，准备重试")
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    initial_size = 0
                    retry_count += 1
                    continue
                    
            except Exception as e:
                logger.error(f"下载模型文件时发生错误: {e}")
                retry_count += 1
                if os.path.exists(temp_path):
                    initial_size = os.path.getsize(temp_path)
                await asyncio.sleep(1)
        
        cls._is_downloading = False
        cls._download_progress = 0
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False
    
    @classmethod
    def is_model_ready(cls):
        """检查模型是否准备就绪"""
        model_path = os.path.join(paths.MODEL_DIR, "model-resnet_custom_v3.pt")
        return os.path.exists(model_path) and not cls._is_downloading and cls.verify_model(model_path)

    @classmethod
    def get_download_status(cls):
        """获取下载状态"""
        return cls._is_downloading, cls._download_progress
    
    @classmethod
    def get_model(cls):
        model_path = os.path.join(paths.MODEL_DIR, "model-resnet_custom_v3.pt")
        
        if cls._model is None:
            if not os.path.exists(model_path):
                if not asyncio.get_event_loop().run_until_complete(cls.download_model()):
                    raise RuntimeError("模型文件下载失败")
            
            cls._model = DeepDanbooruModel()
            cls._model.load_state_dict(
                torch.load(
                    model_path,
                    weights_only=True
                )
            )
            cls._model = cls._model.to(device.DEVICE)
            cls._model.eval()
        return cls._model


@dataclass
class HelpConfirmationState:
    message_id: int
    original_message_id: int
    confirmed: bool
    processing: bool
    start_time: datetime
    timeout_task: Optional[asyncio.Task] = None
    group_id: Optional[int] = None
    

class HelpConfirmationManager:
    _instance = None
    _help_confirmations: Dict[str, HelpConfirmationState] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(HelpConfirmationManager, cls).__new__(cls)
        return cls._instance
    
    @classmethod
    async def add_confirmation(cls, user_id: str, message_id: int, original_message_id: int, bot: Bot, event: Event) -> None:
        if user_id in cls._help_confirmations and cls._help_confirmations[user_id].timeout_task:
            cls._help_confirmations[user_id].timeout_task.cancel()
        
        group_id = None
        if hasattr(event, 'group_id'):
            group_id = event.group_id
        
        timeout_task = asyncio.create_task(cls._handle_timeout(user_id, bot, event))
        
        cls._help_confirmations[user_id] = HelpConfirmationState(
            message_id=message_id,
            original_message_id=original_message_id,
            confirmed=False,
            processing=True,
            start_time=datetime.now(),
            timeout_task=timeout_task,
            group_id=group_id
        )
    
    @classmethod
    async def _handle_timeout(cls, user_id: str, bot: Bot, event: Event):
        """处理超时"""
        try:
            await asyncio.sleep(120)  # 120秒超时
            confirmation = cls.get_confirmation(user_id)
            if confirmation and not confirmation.confirmed:
                try:
                    timeout_message = "您没有及时确认帮助信息，请重新触发指令以查看帮助。"
                    
                    if confirmation.group_id:
                        reply_msg = MessageSegment.reply(confirmation.original_message_id) + timeout_message
                        await bot.send_group_msg(
                            group_id=confirmation.group_id,
                            message=reply_msg
                        )
                    else:
                        reply_msg = MessageSegment.reply(confirmation.original_message_id) + timeout_message
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
    def get_confirmation(cls, user_id: str) -> Optional[HelpConfirmationState]:
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
            if user_info and int(user_info['read_help']) == 1:
                cls.remove_confirmation(user_id)
                return False
                
            confirmation = cls._help_confirmations.get(user_id)
            if not confirmation:
                return False
                
            if (datetime.now() - confirmation.start_time).total_seconds() > 120: # 超时
                cls.remove_confirmation(user_id)
                return False
                
            return confirmation.processing
        except Exception as e:
            logger.error(f"检查处理状态时发生错误: {e}")
            cls.remove_confirmation(user_id)
            return False

help_manager = HelpConfirmationManager()
    
class CommandHandler:
    """命令处理类"""
    def dependency() -> None:
        async def dependency(bot: Bot, matcher, event: Event):
            user_id = str(event.get_user_id())
            
            if not ModelManager.is_model_ready():
                is_downloading, progress = ModelManager.get_download_status()
                if is_downloading:
                    await bot.send(
                        event, 
                        f"模型正在下载中，请稍后再试\n当前下载进度：{progress:.1f}%", 
                        reply_message=True
                    )
                else:
                    await bot.send(
                        event, 
                        "模型文件不存在或下载失败，请联系管理员", 
                        reply_message=True
                    )
                await matcher.finish()
            
            user_info = db_handler.get_user_info(user_id)
            if user_info and int(user_info['read_help']) == 1:
                return
            
            if await help_manager.is_processing(user_id, db_handler):
                if matcher.state.get("_command_name_") == "help":
                    return
                await bot.send(event, "请先同意霸王条款再使用其他指令。", reply_message=True)
                await matcher.finish()
            
            if int(user_info['read_help']) == 0:
                from .. import handle_help_confirmation
                await handle_help_confirmation(bot, event)
                await matcher.finish()
                
        return Depends(dependency)



@dataclass
class GameInfo:
    """游戏信息类"""
    name: str
    aliases: List[str]
    short_name: Optional[str] = None
    en_name: Optional[str] = None

class GameAliasManager:
    """游戏别名管理器"""
    def __init__(self):
        self.games_config: List[GameInfo] = []
        self.game_aliases: Dict[str, List[str]] = {}
        self._name_mapping: Dict[str, str] = {}
        self._load_config()
        
    def _load_config(self):
        """加载游戏配置"""
        
        try:
            with open(paths.GAME_ALIASES_PATH, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            self.games_config = [
                GameInfo(**game_data) 
                for game_data in data['games']
            ]
            
            self.game_aliases = {
                game.name: game.aliases 
                for game in self.games_config
            }
            
            for game in self.games_config:
                self._name_mapping[game.name.lower()] = game.name
                for alias in game.aliases:
                    self._name_mapping[alias.lower()] = game.name
                if game.en_name:
                    self._name_mapping[game.en_name.lower()] = game.name
                if game.short_name:
                    self._name_mapping[game.short_name.lower()] = game.name
                    
        except Exception as e:
            logger.error(f"加载游戏别名配置失败: {e}")
            self.games_config = []
            self.game_aliases = {}
            self._name_mapping = {}
    
    async def get_game_name_from_alias(self, alias: str) -> str:
        """从别名获取标准游戏名称"""
        return self._name_mapping.get(alias.lower(), alias)
    
    def reload_config(self):
        """重新加载配置"""
        self.games_config.clear()
        self.game_aliases.clear()
        self._name_mapping.clear()
        self._load_config()

    def get_game_info(self, name: str) -> Optional[GameInfo]:
        """获取游戏完整信息"""
        std_name = self._name_mapping.get(name.lower())
        if std_name:
            return next(
                (game for game in self.games_config if game.name == std_name),
                None
            )
        return None

game_alias_manager = GameAliasManager()


async def get_game_name_from_alias(alias: str) -> str:
    """返回游戏名"""
    return await game_alias_manager.get_game_name_from_alias(alias)


async def determine_gender(img_path):
    """判断图片性别"""
    try:
        pic = Image.open(img_path).convert("RGB").resize((512, 512))
        a = np.expand_dims(np.array(pic, dtype=np.float32), 0) / 255
        x = torch.from_numpy(a)
        x = x.to(device.DEVICE)
        
        model = ModelManager.get_model()

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
    生成统计图表并通过合并转发发送
    
    Args:
        bot: Bot实例
        event: 事件实例
        game_stats: 游戏统计信息
        limit: 显示数量限制
    """
    font_prop = FontProperties(fname=font_path)
    sorted_games = sorted(game_stats.items(), key=lambda x: x[1]["count"], reverse=True)
    display_games = sorted_games[:limit]
    
    BATCH_SIZE = 10
    total_batches = (len(display_games) + BATCH_SIZE - 1) // BATCH_SIZE
    
    forward_messages = []
    
    for batch_num in range(total_batches):
        start_idx = batch_num * BATCH_SIZE
        end_idx = min((batch_num + 1) * BATCH_SIZE, len(display_games))
        batch_games = display_games[start_idx:end_idx]
        
        text_lines = []
        max_length = 0
        
        for rank, (game_name, stats) in enumerate(batch_games, start_idx + 1):
            line = (
                f"第 {rank} 位: {game_name}\n"
                f"    占比: {stats['percentage']:.2f}%\n"
                f"    角色数: {stats['char_count']}\n"
                f"    平均皮肤数: {stats['avg_skins']:.1f}"
            )
            text_lines.append(line)
            max_length = max(max_length, max(len(l) for l in line.split('\n')))

        fig_width = max_length * 0.15
        fig_height = max(len(batch_games) * 0.6, 1)
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.axis('off')
        fig.patch.set_facecolor('white')
        
        ax.text(0, 0.5, "\n\n".join(text_lines), 
                fontsize=12, ha='left', va='center', 
                fontproperties=font_prop,
                linespacing=1.5)

        buf = io.BytesIO()
        try:
            plt.savefig(buf, format='png', bbox_inches='tight', transparent=False, dpi=120)
            buf.seek(0)
            
            batch_info = f"第 {batch_num + 1}/{total_batches} 页"
            forward_messages.append({
                "type": "node",
                "data": {
                    "name": "老婆统计",
                    "uin": bot.self_id,
                    "content": MessageSegment.image(buf) + MessageSegment.text(f"\n{batch_info}")
                }
            })
            
        finally:
            buf.close()
            plt.close(fig)
    
    await bot.send_group_forward_msg(
        group_id=event.group_id,
        messages=forward_messages
    )

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


# DeepDanbooruModel
class DeepDanbooruModel(nn.Module):
    def __init__(self):
        super(DeepDanbooruModel, self).__init__()

        self.tags = []

        self.n_Conv_0 = nn.Conv2d(kernel_size=(7, 7), in_channels=3, out_channels=64, stride=(2, 2))
        self.n_MaxPool_0 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.n_Conv_1 = nn.Conv2d(kernel_size=(1, 1), in_channels=64, out_channels=256)
        self.n_Conv_2 = nn.Conv2d(kernel_size=(1, 1), in_channels=64, out_channels=64)
        self.n_Conv_3 = nn.Conv2d(kernel_size=(3, 3), in_channels=64, out_channels=64)
        self.n_Conv_4 = nn.Conv2d(kernel_size=(1, 1), in_channels=64, out_channels=256)
        self.n_Conv_5 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=64)
        self.n_Conv_6 = nn.Conv2d(kernel_size=(3, 3), in_channels=64, out_channels=64)
        self.n_Conv_7 = nn.Conv2d(kernel_size=(1, 1), in_channels=64, out_channels=256)
        self.n_Conv_8 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=64)
        self.n_Conv_9 = nn.Conv2d(kernel_size=(3, 3), in_channels=64, out_channels=64)
        self.n_Conv_10 = nn.Conv2d(kernel_size=(1, 1), in_channels=64, out_channels=256)
        self.n_Conv_11 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=512, stride=(2, 2))
        self.n_Conv_12 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=128)
        self.n_Conv_13 = nn.Conv2d(kernel_size=(3, 3), in_channels=128, out_channels=128, stride=(2, 2))
        self.n_Conv_14 = nn.Conv2d(kernel_size=(1, 1), in_channels=128, out_channels=512)
        self.n_Conv_15 = nn.Conv2d(kernel_size=(1, 1), in_channels=512, out_channels=128)
        self.n_Conv_16 = nn.Conv2d(kernel_size=(3, 3), in_channels=128, out_channels=128)
        self.n_Conv_17 = nn.Conv2d(kernel_size=(1, 1), in_channels=128, out_channels=512)
        self.n_Conv_18 = nn.Conv2d(kernel_size=(1, 1), in_channels=512, out_channels=128)
        self.n_Conv_19 = nn.Conv2d(kernel_size=(3, 3), in_channels=128, out_channels=128)
        self.n_Conv_20 = nn.Conv2d(kernel_size=(1, 1), in_channels=128, out_channels=512)
        self.n_Conv_21 = nn.Conv2d(kernel_size=(1, 1), in_channels=512, out_channels=128)
        self.n_Conv_22 = nn.Conv2d(kernel_size=(3, 3), in_channels=128, out_channels=128)
        self.n_Conv_23 = nn.Conv2d(kernel_size=(1, 1), in_channels=128, out_channels=512)
        self.n_Conv_24 = nn.Conv2d(kernel_size=(1, 1), in_channels=512, out_channels=128)
        self.n_Conv_25 = nn.Conv2d(kernel_size=(3, 3), in_channels=128, out_channels=128)
        self.n_Conv_26 = nn.Conv2d(kernel_size=(1, 1), in_channels=128, out_channels=512)
        self.n_Conv_27 = nn.Conv2d(kernel_size=(1, 1), in_channels=512, out_channels=128)
        self.n_Conv_28 = nn.Conv2d(kernel_size=(3, 3), in_channels=128, out_channels=128)
        self.n_Conv_29 = nn.Conv2d(kernel_size=(1, 1), in_channels=128, out_channels=512)
        self.n_Conv_30 = nn.Conv2d(kernel_size=(1, 1), in_channels=512, out_channels=128)
        self.n_Conv_31 = nn.Conv2d(kernel_size=(3, 3), in_channels=128, out_channels=128)
        self.n_Conv_32 = nn.Conv2d(kernel_size=(1, 1), in_channels=128, out_channels=512)
        self.n_Conv_33 = nn.Conv2d(kernel_size=(1, 1), in_channels=512, out_channels=128)
        self.n_Conv_34 = nn.Conv2d(kernel_size=(3, 3), in_channels=128, out_channels=128)
        self.n_Conv_35 = nn.Conv2d(kernel_size=(1, 1), in_channels=128, out_channels=512)
        self.n_Conv_36 = nn.Conv2d(kernel_size=(1, 1), in_channels=512, out_channels=1024, stride=(2, 2))
        self.n_Conv_37 = nn.Conv2d(kernel_size=(1, 1), in_channels=512, out_channels=256)
        self.n_Conv_38 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256, stride=(2, 2))
        self.n_Conv_39 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_40 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_41 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_42 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_43 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_44 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_45 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_46 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_47 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_48 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_49 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_50 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_51 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_52 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_53 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_54 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_55 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_56 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_57 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_58 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_59 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_60 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_61 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_62 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_63 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_64 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_65 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_66 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_67 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_68 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_69 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_70 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_71 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_72 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_73 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_74 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_75 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_76 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_77 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_78 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_79 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_80 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_81 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_82 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_83 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_84 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_85 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_86 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_87 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_88 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_89 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_90 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_91 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_92 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_93 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_94 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_95 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_96 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_97 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_98 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256, stride=(2, 2))
        self.n_Conv_99 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_100 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=1024, stride=(2, 2))
        self.n_Conv_101 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_102 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_103 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_104 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_105 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_106 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_107 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_108 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_109 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_110 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_111 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_112 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_113 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_114 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_115 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_116 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_117 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_118 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_119 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_120 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_121 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_122 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_123 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_124 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_125 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_126 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_127 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_128 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_129 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_130 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_131 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_132 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_133 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_134 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_135 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_136 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_137 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_138 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_139 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_140 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_141 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_142 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_143 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_144 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_145 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_146 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_147 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_148 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_149 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_150 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_151 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_152 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_153 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_154 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_155 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_156 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_157 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_158 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=2048, stride=(2, 2))
        self.n_Conv_159 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=512)
        self.n_Conv_160 = nn.Conv2d(kernel_size=(3, 3), in_channels=512, out_channels=512, stride=(2, 2))
        self.n_Conv_161 = nn.Conv2d(kernel_size=(1, 1), in_channels=512, out_channels=2048)
        self.n_Conv_162 = nn.Conv2d(kernel_size=(1, 1), in_channels=2048, out_channels=512)
        self.n_Conv_163 = nn.Conv2d(kernel_size=(3, 3), in_channels=512, out_channels=512)
        self.n_Conv_164 = nn.Conv2d(kernel_size=(1, 1), in_channels=512, out_channels=2048)
        self.n_Conv_165 = nn.Conv2d(kernel_size=(1, 1), in_channels=2048, out_channels=512)
        self.n_Conv_166 = nn.Conv2d(kernel_size=(3, 3), in_channels=512, out_channels=512)
        self.n_Conv_167 = nn.Conv2d(kernel_size=(1, 1), in_channels=512, out_channels=2048)
        self.n_Conv_168 = nn.Conv2d(kernel_size=(1, 1), in_channels=2048, out_channels=4096, stride=(2, 2))
        self.n_Conv_169 = nn.Conv2d(kernel_size=(1, 1), in_channels=2048, out_channels=1024)
        self.n_Conv_170 = nn.Conv2d(kernel_size=(3, 3), in_channels=1024, out_channels=1024, stride=(2, 2))
        self.n_Conv_171 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=4096)
        self.n_Conv_172 = nn.Conv2d(kernel_size=(1, 1), in_channels=4096, out_channels=1024)
        self.n_Conv_173 = nn.Conv2d(kernel_size=(3, 3), in_channels=1024, out_channels=1024)
        self.n_Conv_174 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=4096)
        self.n_Conv_175 = nn.Conv2d(kernel_size=(1, 1), in_channels=4096, out_channels=1024)
        self.n_Conv_176 = nn.Conv2d(kernel_size=(3, 3), in_channels=1024, out_channels=1024)
        self.n_Conv_177 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=4096)
        self.n_Conv_178 = nn.Conv2d(kernel_size=(1, 1), in_channels=4096, out_channels=9176, bias=False)

    def forward(self, *inputs):
        t_358, = inputs
        t_359 = t_358.permute(*[0, 3, 1, 2])
        t_359_padded = F.pad(t_359, [2, 3, 2, 3], value=0)
        t_360 = self.n_Conv_0(t_359_padded)
        t_361 = F.relu(t_360)
        t_361 = F.pad(t_361, [0, 1, 0, 1], value=float('-inf'))
        t_362 = self.n_MaxPool_0(t_361)
        t_363 = self.n_Conv_1(t_362)
        t_364 = self.n_Conv_2(t_362)
        t_365 = F.relu(t_364)
        t_365_padded = F.pad(t_365, [1, 1, 1, 1], value=0)
        t_366 = self.n_Conv_3(t_365_padded)
        t_367 = F.relu(t_366)
        t_368 = self.n_Conv_4(t_367)
        t_369 = torch.add(t_368, t_363)
        t_370 = F.relu(t_369)
        t_371 = self.n_Conv_5(t_370)
        t_372 = F.relu(t_371)
        t_372_padded = F.pad(t_372, [1, 1, 1, 1], value=0)
        t_373 = self.n_Conv_6(t_372_padded)
        t_374 = F.relu(t_373)
        t_375 = self.n_Conv_7(t_374)
        t_376 = torch.add(t_375, t_370)
        t_377 = F.relu(t_376)
        t_378 = self.n_Conv_8(t_377)
        t_379 = F.relu(t_378)
        t_379_padded = F.pad(t_379, [1, 1, 1, 1], value=0)
        t_380 = self.n_Conv_9(t_379_padded)
        t_381 = F.relu(t_380)
        t_382 = self.n_Conv_10(t_381)
        t_383 = torch.add(t_382, t_377)
        t_384 = F.relu(t_383)
        t_385 = self.n_Conv_11(t_384)
        t_386 = self.n_Conv_12(t_384)
        t_387 = F.relu(t_386)
        t_387_padded = F.pad(t_387, [0, 1, 0, 1], value=0)
        t_388 = self.n_Conv_13(t_387_padded)
        t_389 = F.relu(t_388)
        t_390 = self.n_Conv_14(t_389)
        t_391 = torch.add(t_390, t_385)
        t_392 = F.relu(t_391)
        t_393 = self.n_Conv_15(t_392)
        t_394 = F.relu(t_393)
        t_394_padded = F.pad(t_394, [1, 1, 1, 1], value=0)
        t_395 = self.n_Conv_16(t_394_padded)
        t_396 = F.relu(t_395)
        t_397 = self.n_Conv_17(t_396)
        t_398 = torch.add(t_397, t_392)
        t_399 = F.relu(t_398)
        t_400 = self.n_Conv_18(t_399)
        t_401 = F.relu(t_400)
        t_401_padded = F.pad(t_401, [1, 1, 1, 1], value=0)
        t_402 = self.n_Conv_19(t_401_padded)
        t_403 = F.relu(t_402)
        t_404 = self.n_Conv_20(t_403)
        t_405 = torch.add(t_404, t_399)
        t_406 = F.relu(t_405)
        t_407 = self.n_Conv_21(t_406)
        t_408 = F.relu(t_407)
        t_408_padded = F.pad(t_408, [1, 1, 1, 1], value=0)
        t_409 = self.n_Conv_22(t_408_padded)
        t_410 = F.relu(t_409)
        t_411 = self.n_Conv_23(t_410)
        t_412 = torch.add(t_411, t_406)
        t_413 = F.relu(t_412)
        t_414 = self.n_Conv_24(t_413)
        t_415 = F.relu(t_414)
        t_415_padded = F.pad(t_415, [1, 1, 1, 1], value=0)
        t_416 = self.n_Conv_25(t_415_padded)
        t_417 = F.relu(t_416)
        t_418 = self.n_Conv_26(t_417)
        t_419 = torch.add(t_418, t_413)
        t_420 = F.relu(t_419)
        t_421 = self.n_Conv_27(t_420)
        t_422 = F.relu(t_421)
        t_422_padded = F.pad(t_422, [1, 1, 1, 1], value=0)
        t_423 = self.n_Conv_28(t_422_padded)
        t_424 = F.relu(t_423)
        t_425 = self.n_Conv_29(t_424)
        t_426 = torch.add(t_425, t_420)
        t_427 = F.relu(t_426)
        t_428 = self.n_Conv_30(t_427)
        t_429 = F.relu(t_428)
        t_429_padded = F.pad(t_429, [1, 1, 1, 1], value=0)
        t_430 = self.n_Conv_31(t_429_padded)
        t_431 = F.relu(t_430)
        t_432 = self.n_Conv_32(t_431)
        t_433 = torch.add(t_432, t_427)
        t_434 = F.relu(t_433)
        t_435 = self.n_Conv_33(t_434)
        t_436 = F.relu(t_435)
        t_436_padded = F.pad(t_436, [1, 1, 1, 1], value=0)
        t_437 = self.n_Conv_34(t_436_padded)
        t_438 = F.relu(t_437)
        t_439 = self.n_Conv_35(t_438)
        t_440 = torch.add(t_439, t_434)
        t_441 = F.relu(t_440)
        t_442 = self.n_Conv_36(t_441)
        t_443 = self.n_Conv_37(t_441)
        t_444 = F.relu(t_443)
        t_444_padded = F.pad(t_444, [0, 1, 0, 1], value=0)
        t_445 = self.n_Conv_38(t_444_padded)
        t_446 = F.relu(t_445)
        t_447 = self.n_Conv_39(t_446)
        t_448 = torch.add(t_447, t_442)
        t_449 = F.relu(t_448)
        t_450 = self.n_Conv_40(t_449)
        t_451 = F.relu(t_450)
        t_451_padded = F.pad(t_451, [1, 1, 1, 1], value=0)
        t_452 = self.n_Conv_41(t_451_padded)
        t_453 = F.relu(t_452)
        t_454 = self.n_Conv_42(t_453)
        t_455 = torch.add(t_454, t_449)
        t_456 = F.relu(t_455)
        t_457 = self.n_Conv_43(t_456)
        t_458 = F.relu(t_457)
        t_458_padded = F.pad(t_458, [1, 1, 1, 1], value=0)
        t_459 = self.n_Conv_44(t_458_padded)
        t_460 = F.relu(t_459)
        t_461 = self.n_Conv_45(t_460)
        t_462 = torch.add(t_461, t_456)
        t_463 = F.relu(t_462)
        t_464 = self.n_Conv_46(t_463)
        t_465 = F.relu(t_464)
        t_465_padded = F.pad(t_465, [1, 1, 1, 1], value=0)
        t_466 = self.n_Conv_47(t_465_padded)
        t_467 = F.relu(t_466)
        t_468 = self.n_Conv_48(t_467)
        t_469 = torch.add(t_468, t_463)
        t_470 = F.relu(t_469)
        t_471 = self.n_Conv_49(t_470)
        t_472 = F.relu(t_471)
        t_472_padded = F.pad(t_472, [1, 1, 1, 1], value=0)
        t_473 = self.n_Conv_50(t_472_padded)
        t_474 = F.relu(t_473)
        t_475 = self.n_Conv_51(t_474)
        t_476 = torch.add(t_475, t_470)
        t_477 = F.relu(t_476)
        t_478 = self.n_Conv_52(t_477)
        t_479 = F.relu(t_478)
        t_479_padded = F.pad(t_479, [1, 1, 1, 1], value=0)
        t_480 = self.n_Conv_53(t_479_padded)
        t_481 = F.relu(t_480)
        t_482 = self.n_Conv_54(t_481)
        t_483 = torch.add(t_482, t_477)
        t_484 = F.relu(t_483)
        t_485 = self.n_Conv_55(t_484)
        t_486 = F.relu(t_485)
        t_486_padded = F.pad(t_486, [1, 1, 1, 1], value=0)
        t_487 = self.n_Conv_56(t_486_padded)
        t_488 = F.relu(t_487)
        t_489 = self.n_Conv_57(t_488)
        t_490 = torch.add(t_489, t_484)
        t_491 = F.relu(t_490)
        t_492 = self.n_Conv_58(t_491)
        t_493 = F.relu(t_492)
        t_493_padded = F.pad(t_493, [1, 1, 1, 1], value=0)
        t_494 = self.n_Conv_59(t_493_padded)
        t_495 = F.relu(t_494)
        t_496 = self.n_Conv_60(t_495)
        t_497 = torch.add(t_496, t_491)
        t_498 = F.relu(t_497)
        t_499 = self.n_Conv_61(t_498)
        t_500 = F.relu(t_499)
        t_500_padded = F.pad(t_500, [1, 1, 1, 1], value=0)
        t_501 = self.n_Conv_62(t_500_padded)
        t_502 = F.relu(t_501)
        t_503 = self.n_Conv_63(t_502)
        t_504 = torch.add(t_503, t_498)
        t_505 = F.relu(t_504)
        t_506 = self.n_Conv_64(t_505)
        t_507 = F.relu(t_506)
        t_507_padded = F.pad(t_507, [1, 1, 1, 1], value=0)
        t_508 = self.n_Conv_65(t_507_padded)
        t_509 = F.relu(t_508)
        t_510 = self.n_Conv_66(t_509)
        t_511 = torch.add(t_510, t_505)
        t_512 = F.relu(t_511)
        t_513 = self.n_Conv_67(t_512)
        t_514 = F.relu(t_513)
        t_514_padded = F.pad(t_514, [1, 1, 1, 1], value=0)
        t_515 = self.n_Conv_68(t_514_padded)
        t_516 = F.relu(t_515)
        t_517 = self.n_Conv_69(t_516)
        t_518 = torch.add(t_517, t_512)
        t_519 = F.relu(t_518)
        t_520 = self.n_Conv_70(t_519)
        t_521 = F.relu(t_520)
        t_521_padded = F.pad(t_521, [1, 1, 1, 1], value=0)
        t_522 = self.n_Conv_71(t_521_padded)
        t_523 = F.relu(t_522)
        t_524 = self.n_Conv_72(t_523)
        t_525 = torch.add(t_524, t_519)
        t_526 = F.relu(t_525)
        t_527 = self.n_Conv_73(t_526)
        t_528 = F.relu(t_527)
        t_528_padded = F.pad(t_528, [1, 1, 1, 1], value=0)
        t_529 = self.n_Conv_74(t_528_padded)
        t_530 = F.relu(t_529)
        t_531 = self.n_Conv_75(t_530)
        t_532 = torch.add(t_531, t_526)
        t_533 = F.relu(t_532)
        t_534 = self.n_Conv_76(t_533)
        t_535 = F.relu(t_534)
        t_535_padded = F.pad(t_535, [1, 1, 1, 1], value=0)
        t_536 = self.n_Conv_77(t_535_padded)
        t_537 = F.relu(t_536)
        t_538 = self.n_Conv_78(t_537)
        t_539 = torch.add(t_538, t_533)
        t_540 = F.relu(t_539)
        t_541 = self.n_Conv_79(t_540)
        t_542 = F.relu(t_541)
        t_542_padded = F.pad(t_542, [1, 1, 1, 1], value=0)
        t_543 = self.n_Conv_80(t_542_padded)
        t_544 = F.relu(t_543)
        t_545 = self.n_Conv_81(t_544)
        t_546 = torch.add(t_545, t_540)
        t_547 = F.relu(t_546)
        t_548 = self.n_Conv_82(t_547)
        t_549 = F.relu(t_548)
        t_549_padded = F.pad(t_549, [1, 1, 1, 1], value=0)
        t_550 = self.n_Conv_83(t_549_padded)
        t_551 = F.relu(t_550)
        t_552 = self.n_Conv_84(t_551)
        t_553 = torch.add(t_552, t_547)
        t_554 = F.relu(t_553)
        t_555 = self.n_Conv_85(t_554)
        t_556 = F.relu(t_555)
        t_556_padded = F.pad(t_556, [1, 1, 1, 1], value=0)
        t_557 = self.n_Conv_86(t_556_padded)
        t_558 = F.relu(t_557)
        t_559 = self.n_Conv_87(t_558)
        t_560 = torch.add(t_559, t_554)
        t_561 = F.relu(t_560)
        t_562 = self.n_Conv_88(t_561)
        t_563 = F.relu(t_562)
        t_563_padded = F.pad(t_563, [1, 1, 1, 1], value=0)
        t_564 = self.n_Conv_89(t_563_padded)
        t_565 = F.relu(t_564)
        t_566 = self.n_Conv_90(t_565)
        t_567 = torch.add(t_566, t_561)
        t_568 = F.relu(t_567)
        t_569 = self.n_Conv_91(t_568)
        t_570 = F.relu(t_569)
        t_570_padded = F.pad(t_570, [1, 1, 1, 1], value=0)
        t_571 = self.n_Conv_92(t_570_padded)
        t_572 = F.relu(t_571)
        t_573 = self.n_Conv_93(t_572)
        t_574 = torch.add(t_573, t_568)
        t_575 = F.relu(t_574)
        t_576 = self.n_Conv_94(t_575)
        t_577 = F.relu(t_576)
        t_577_padded = F.pad(t_577, [1, 1, 1, 1], value=0)
        t_578 = self.n_Conv_95(t_577_padded)
        t_579 = F.relu(t_578)
        t_580 = self.n_Conv_96(t_579)
        t_581 = torch.add(t_580, t_575)
        t_582 = F.relu(t_581)
        t_583 = self.n_Conv_97(t_582)
        t_584 = F.relu(t_583)
        t_584_padded = F.pad(t_584, [0, 1, 0, 1], value=0)
        t_585 = self.n_Conv_98(t_584_padded)
        t_586 = F.relu(t_585)
        t_587 = self.n_Conv_99(t_586)
        t_588 = self.n_Conv_100(t_582)
        t_589 = torch.add(t_587, t_588)
        t_590 = F.relu(t_589)
        t_591 = self.n_Conv_101(t_590)
        t_592 = F.relu(t_591)
        t_592_padded = F.pad(t_592, [1, 1, 1, 1], value=0)
        t_593 = self.n_Conv_102(t_592_padded)
        t_594 = F.relu(t_593)
        t_595 = self.n_Conv_103(t_594)
        t_596 = torch.add(t_595, t_590)
        t_597 = F.relu(t_596)
        t_598 = self.n_Conv_104(t_597)
        t_599 = F.relu(t_598)
        t_599_padded = F.pad(t_599, [1, 1, 1, 1], value=0)
        t_600 = self.n_Conv_105(t_599_padded)
        t_601 = F.relu(t_600)
        t_602 = self.n_Conv_106(t_601)
        t_603 = torch.add(t_602, t_597)
        t_604 = F.relu(t_603)
        t_605 = self.n_Conv_107(t_604)
        t_606 = F.relu(t_605)
        t_606_padded = F.pad(t_606, [1, 1, 1, 1], value=0)
        t_607 = self.n_Conv_108(t_606_padded)
        t_608 = F.relu(t_607)
        t_609 = self.n_Conv_109(t_608)
        t_610 = torch.add(t_609, t_604)
        t_611 = F.relu(t_610)
        t_612 = self.n_Conv_110(t_611)
        t_613 = F.relu(t_612)
        t_613_padded = F.pad(t_613, [1, 1, 1, 1], value=0)
        t_614 = self.n_Conv_111(t_613_padded)
        t_615 = F.relu(t_614)
        t_616 = self.n_Conv_112(t_615)
        t_617 = torch.add(t_616, t_611)
        t_618 = F.relu(t_617)
        t_619 = self.n_Conv_113(t_618)
        t_620 = F.relu(t_619)
        t_620_padded = F.pad(t_620, [1, 1, 1, 1], value=0)
        t_621 = self.n_Conv_114(t_620_padded)
        t_622 = F.relu(t_621)
        t_623 = self.n_Conv_115(t_622)
        t_624 = torch.add(t_623, t_618)
        t_625 = F.relu(t_624)
        t_626 = self.n_Conv_116(t_625)
        t_627 = F.relu(t_626)
        t_627_padded = F.pad(t_627, [1, 1, 1, 1], value=0)
        t_628 = self.n_Conv_117(t_627_padded)
        t_629 = F.relu(t_628)
        t_630 = self.n_Conv_118(t_629)
        t_631 = torch.add(t_630, t_625)
        t_632 = F.relu(t_631)
        t_633 = self.n_Conv_119(t_632)
        t_634 = F.relu(t_633)
        t_634_padded = F.pad(t_634, [1, 1, 1, 1], value=0)
        t_635 = self.n_Conv_120(t_634_padded)
        t_636 = F.relu(t_635)
        t_637 = self.n_Conv_121(t_636)
        t_638 = torch.add(t_637, t_632)
        t_639 = F.relu(t_638)
        t_640 = self.n_Conv_122(t_639)
        t_641 = F.relu(t_640)
        t_641_padded = F.pad(t_641, [1, 1, 1, 1], value=0)
        t_642 = self.n_Conv_123(t_641_padded)
        t_643 = F.relu(t_642)
        t_644 = self.n_Conv_124(t_643)
        t_645 = torch.add(t_644, t_639)
        t_646 = F.relu(t_645)
        t_647 = self.n_Conv_125(t_646)
        t_648 = F.relu(t_647)
        t_648_padded = F.pad(t_648, [1, 1, 1, 1], value=0)
        t_649 = self.n_Conv_126(t_648_padded)
        t_650 = F.relu(t_649)
        t_651 = self.n_Conv_127(t_650)
        t_652 = torch.add(t_651, t_646)
        t_653 = F.relu(t_652)
        t_654 = self.n_Conv_128(t_653)
        t_655 = F.relu(t_654)
        t_655_padded = F.pad(t_655, [1, 1, 1, 1], value=0)
        t_656 = self.n_Conv_129(t_655_padded)
        t_657 = F.relu(t_656)
        t_658 = self.n_Conv_130(t_657)
        t_659 = torch.add(t_658, t_653)
        t_660 = F.relu(t_659)
        t_661 = self.n_Conv_131(t_660)
        t_662 = F.relu(t_661)
        t_662_padded = F.pad(t_662, [1, 1, 1, 1], value=0)
        t_663 = self.n_Conv_132(t_662_padded)
        t_664 = F.relu(t_663)
        t_665 = self.n_Conv_133(t_664)
        t_666 = torch.add(t_665, t_660)
        t_667 = F.relu(t_666)
        t_668 = self.n_Conv_134(t_667)
        t_669 = F.relu(t_668)
        t_669_padded = F.pad(t_669, [1, 1, 1, 1], value=0)
        t_670 = self.n_Conv_135(t_669_padded)
        t_671 = F.relu(t_670)
        t_672 = self.n_Conv_136(t_671)
        t_673 = torch.add(t_672, t_667)
        t_674 = F.relu(t_673)
        t_675 = self.n_Conv_137(t_674)
        t_676 = F.relu(t_675)
        t_676_padded = F.pad(t_676, [1, 1, 1, 1], value=0)
        t_677 = self.n_Conv_138(t_676_padded)
        t_678 = F.relu(t_677)
        t_679 = self.n_Conv_139(t_678)
        t_680 = torch.add(t_679, t_674)
        t_681 = F.relu(t_680)
        t_682 = self.n_Conv_140(t_681)
        t_683 = F.relu(t_682)
        t_683_padded = F.pad(t_683, [1, 1, 1, 1], value=0)
        t_684 = self.n_Conv_141(t_683_padded)
        t_685 = F.relu(t_684)
        t_686 = self.n_Conv_142(t_685)
        t_687 = torch.add(t_686, t_681)
        t_688 = F.relu(t_687)
        t_689 = self.n_Conv_143(t_688)
        t_690 = F.relu(t_689)
        t_690_padded = F.pad(t_690, [1, 1, 1, 1], value=0)
        t_691 = self.n_Conv_144(t_690_padded)
        t_692 = F.relu(t_691)
        t_693 = self.n_Conv_145(t_692)
        t_694 = torch.add(t_693, t_688)
        t_695 = F.relu(t_694)
        t_696 = self.n_Conv_146(t_695)
        t_697 = F.relu(t_696)
        t_697_padded = F.pad(t_697, [1, 1, 1, 1], value=0)
        t_698 = self.n_Conv_147(t_697_padded)
        t_699 = F.relu(t_698)
        t_700 = self.n_Conv_148(t_699)
        t_701 = torch.add(t_700, t_695)
        t_702 = F.relu(t_701)
        t_703 = self.n_Conv_149(t_702)
        t_704 = F.relu(t_703)
        t_704_padded = F.pad(t_704, [1, 1, 1, 1], value=0)
        t_705 = self.n_Conv_150(t_704_padded)
        t_706 = F.relu(t_705)
        t_707 = self.n_Conv_151(t_706)
        t_708 = torch.add(t_707, t_702)
        t_709 = F.relu(t_708)
        t_710 = self.n_Conv_152(t_709)
        t_711 = F.relu(t_710)
        t_711_padded = F.pad(t_711, [1, 1, 1, 1], value=0)
        t_712 = self.n_Conv_153(t_711_padded)
        t_713 = F.relu(t_712)
        t_714 = self.n_Conv_154(t_713)
        t_715 = torch.add(t_714, t_709)
        t_716 = F.relu(t_715)
        t_717 = self.n_Conv_155(t_716)
        t_718 = F.relu(t_717)
        t_718_padded = F.pad(t_718, [1, 1, 1, 1], value=0)
        t_719 = self.n_Conv_156(t_718_padded)
        t_720 = F.relu(t_719)
        t_721 = self.n_Conv_157(t_720)
        t_722 = torch.add(t_721, t_716)
        t_723 = F.relu(t_722)
        t_724 = self.n_Conv_158(t_723)
        t_725 = self.n_Conv_159(t_723)
        t_726 = F.relu(t_725)
        t_726_padded = F.pad(t_726, [0, 1, 0, 1], value=0)
        t_727 = self.n_Conv_160(t_726_padded)
        t_728 = F.relu(t_727)
        t_729 = self.n_Conv_161(t_728)
        t_730 = torch.add(t_729, t_724)
        t_731 = F.relu(t_730)
        t_732 = self.n_Conv_162(t_731)
        t_733 = F.relu(t_732)
        t_733_padded = F.pad(t_733, [1, 1, 1, 1], value=0)
        t_734 = self.n_Conv_163(t_733_padded)
        t_735 = F.relu(t_734)
        t_736 = self.n_Conv_164(t_735)
        t_737 = torch.add(t_736, t_731)
        t_738 = F.relu(t_737)
        t_739 = self.n_Conv_165(t_738)
        t_740 = F.relu(t_739)
        t_740_padded = F.pad(t_740, [1, 1, 1, 1], value=0)
        t_741 = self.n_Conv_166(t_740_padded)
        t_742 = F.relu(t_741)
        t_743 = self.n_Conv_167(t_742)
        t_744 = torch.add(t_743, t_738)
        t_745 = F.relu(t_744)
        t_746 = self.n_Conv_168(t_745)
        t_747 = self.n_Conv_169(t_745)
        t_748 = F.relu(t_747)
        t_748_padded = F.pad(t_748, [0, 1, 0, 1], value=0)
        t_749 = self.n_Conv_170(t_748_padded)
        t_750 = F.relu(t_749)
        t_751 = self.n_Conv_171(t_750)
        t_752 = torch.add(t_751, t_746)
        t_753 = F.relu(t_752)
        t_754 = self.n_Conv_172(t_753)
        t_755 = F.relu(t_754)
        t_755_padded = F.pad(t_755, [1, 1, 1, 1], value=0)
        t_756 = self.n_Conv_173(t_755_padded)
        t_757 = F.relu(t_756)
        t_758 = self.n_Conv_174(t_757)
        t_759 = torch.add(t_758, t_753)
        t_760 = F.relu(t_759)
        t_761 = self.n_Conv_175(t_760)
        t_762 = F.relu(t_761)
        t_762_padded = F.pad(t_762, [1, 1, 1, 1], value=0)
        t_763 = self.n_Conv_176(t_762_padded)
        t_764 = F.relu(t_763)
        t_765 = self.n_Conv_177(t_764)
        t_766 = torch.add(t_765, t_760)
        t_767 = F.relu(t_766)
        t_768 = self.n_Conv_178(t_767)
        t_769 = F.avg_pool2d(t_768, kernel_size=t_768.shape[-2:])
        t_770 = torch.squeeze(t_769, 3)
        t_770 = torch.squeeze(t_770, 2)
        t_771 = torch.sigmoid(t_770)
        return t_771

    def load_state_dict(self, state_dict, **kwargs):
        self.tags = state_dict.get('tags', [])

        super(DeepDanbooruModel, self).load_state_dict({k: v for k, v in state_dict.items() if k != 'tags'})


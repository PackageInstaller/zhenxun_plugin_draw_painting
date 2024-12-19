try:
    import ujson as json
except ImportError:
    import json
import os
import re
import io
import shutil
import asyncio
from fuzzywuzzy import process, fuzz
from datetime import datetime
from nonebot.params import CommandArg     
from nonebot import on_command, on_fullmatch, on_notice, on_regex, get_driver
from nonebot.typing import T_State
from nonebot.plugin import PluginMetadata
from nonebot.adapters.onebot.v11 import (
    Bot, 
    Event, 
    MessageSegment,
    GroupMessageEvent,
    Message,
    NoticeEvent
)
from matplotlib.font_manager import FontProperties
from matplotlib import pyplot as plt
from .config import paths
from .database import db_handler
from .utils.functions import *
from typing import Dict
from zhenxun.utils.enum import PluginType
from zhenxun.configs.utils import PluginExtraData
from zhenxun.services.log import logger
from zhenxun.configs.config import BotConfig
from zhenxun.utils.enum import BlockType, PluginType
from zhenxun.configs.utils import BaseBlock, PluginExtraData

driver = get_driver()
husbands_images_folder = paths.HUSBANDS_IMAGES_FOLDER
wives_images_folder = paths.WIVES_IMAGES_FOLDER
drop_folder = paths.DROP_FOLDER
font_path = paths.FONT_PATH

wives_draw = on_regex(r"^抽(?!.*老公).*老婆.*$", priority=5)
wives_view = on_fullmatch("我老婆", priority=5)
wives_rename = on_command("老婆改名", priority=5, expire_time=None)
wives_probability = on_command("老婆概率", priority=5)
delete_husbands = on_fullmatch("这是男的", priority=5)

husbands_draw = on_regex(r"^抽(?!.*老婆).*老公.*$", priority=5)
husbands_view = on_fullmatch("我老公", priority=5)
husbands_rename = on_command("老公改名", priority=5, expire_time=None)
husbands_probability = on_command("老公概率", priority=5)
delete_wives = on_fullmatch("这是女的", priority=5)

vote_delete = on_command("投票删除", priority=5)
help = on_fullmatch("帮助抽游戏立绘", priority=5)

__plugin_meta__ = PluginMetadata(
    name="抽游戏立绘",
    description="从本地图片库中随机抽取游戏人物立绘",
    usage="""
    抽取老婆/老公
    指令：
    抽老婆/老公［游戏名参数可选］
    我老婆/老公 [查看所有立绘]
    老婆/老公改名 [修改单张立绘名称]
    老婆/老公概率 ?[数量参数可选，默认全部] [查看各游戏占比]
    这是男的/女的 [抽老婆抽到男的的时候可以用，另一个同理，只能处理自己的立绘]
    投票删除 [回复抽到的图片，非男非女时可以使用]
    请注意，如果出现乱用指令的情况，将会被永久封禁。
    """.strip(),
    extra=PluginExtraData(
        author="少姜",
        version="1.0",
        plugin_type=PluginType.NORMAL,
        limits=[BaseBlock(check_type=BlockType.GROUP)],
        menu_type="抽卡相关"
    ).dict(),
)


@driver.on_startup
async def initialize_model():
    """初始化时检查并下载模型"""
    model_path = os.path.join(paths.MODEL_DIR, "model-resnet_custom_v3.pt")
    temp_path = f"{model_path}.temp"
    
    if not ModelManager.verify_model(model_path):
        if os.path.exists(model_path):
            logger.info("现有模型文件验证失败，准备重新下载...")
            os.remove(model_path)
        elif os.path.exists(temp_path):
            logger.info("发现未完成的下载，将继续下载...")
        else:
            logger.info("模型文件不存在，开始下载...")
            
        asyncio.create_task(ModelManager.download_model())
        logger.info("模型下载已在后台启动")


@help.handle()
async def handle_help(bot: Bot, event: Event):
    """帮助指令，输出为图片形式"""
    user_id = str(event.get_user_id())
    
    # 如果存在确认状态，移除它
    confirmation = help_manager.get_confirmation(user_id)
    if confirmation:
        help_manager.remove_confirmation(user_id)
    
    font_prop = FontProperties(fname=font_path)
    help_text = __plugin_meta__.usage

    text_lines = help_text.split('\n')
    max_length = max(len(line) for line in text_lines)
    fig_width = max_length * 0.1
    fig_height = max(len(text_lines) * 0.1 + 1, 1)
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('off')
    fig.patch.set_facecolor('white')
    ax.text(0, 0.5, help_text, fontsize=14, ha='left', va='center', fontproperties=font_prop)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', transparent=False)
    buf.seek(0)
    
    if int(db_handler.get_user_info(user_id)['read_help']) == 0:
        warning_message = "现在您可以正常使用所有指令了。\n请注意，如果出现乱用指令的情况，将会被永久封禁。"
    else:
        warning_message = ""
    message = MessageSegment.image(buf) + MessageSegment.text(warning_message)

    # 标记帮助信息为已读
    db_handler.mark_help_as_read(user_id)

    await bot.send(event, message, reply_message=True)

    buf.close()
    plt.close('all')


@wives_draw.handle(parameterless=[CommandHandler.dependency()])
async def handle_wives_draw(bot: Bot, event: Event):
    """抽老婆"""
    user_id = str(event.get_user_id())
    message_text = event.get_plaintext().strip().lower()

    pattern = r"^抽(?:老婆)?(.*?)(?:老婆)?$"
    match = re.match(pattern, message_text)

    game_name = None
    if match:
        game_name = match.group(1).strip().lower()
        if game_name:
            game_name = await get_game_name_from_alias(game_name)
        else:
            game_name = None
    
    if user_id not in bot.config.superusers: # 普通用户 event.sender.role == "admin" "owner"
        last_draw_time = db_handler.get_last_trigger_time(user_id, 'Wife')
        if last_draw_time:
            time_difference = int((datetime.now() - last_draw_time).total_seconds())  # 转换为整数秒
            if time_difference < 10:  # 5分钟
                try:
                    remaining_time = await format_time(10 - time_difference)
                    await bot.send(event, f"合不合适也要先待满10秒吧！\n还剩下{remaining_time}。", reply_message=True)
                except Exception as e:
                    await bot.send(event, f"发送消息时发生错误：{str(e)}", reply_message=True)
                await wives_draw.finish()

    # if game_name:
    #     try:
    #         await bot.send(event, f"你指定了 {game_name} 的老婆呢，正在为你抽取...", reply_message=False)
    #     except Exception as e:
    #         await bot.send(event, f"发送消息时发生错误：{str(e)}", reply_message=True)
    #         await wives_draw.finish()
    # else:
    #     try:
    #         await bot.send(event, f"你没有指定游戏名呢，正在为你随机抽取...", reply_message=False)
    #     except Exception as e:
    #         await bot.send(event, f"发送消息时发生错误：{str(e)}", reply_message=True)
    #         await wives_draw.finish()

    all_images = [img for img in os.listdir(wives_images_folder)]
    game_names_in_library = {img.split("_")[0].lower() for img in all_images if "_" in img}  # 只考虑带有_的游戏名

    if game_name:
        game_name_parts = list(game_name)
        fuzzy_matches = await improved_partial_word_match(game_name_parts, game_names_in_library)
        if fuzzy_matches:
            closest_match = fuzzy_matches[0]
        else:
            # 相似度匹配
            closest_match, similarity = process.extractOne(game_name, game_names_in_library, scorer=fuzz.partial_ratio)

            if similarity < 70:
                try:
                    await bot.send(event, f"没有找到相关的游戏呢。", reply_message=True)
                except Exception as e:
                    await bot.send(event, f"发送消息时发生错误：{str(e)}", reply_message=True)
                await wives_draw.finish()
                
        # try:
        #     await bot.send(event, f"根据你的指定，最匹配的游戏是：{closest_match}", reply_message=False)
        # except Exception as e:
        #     await bot.send(event, f"发送消息时发生错误：{str(e)}", reply_message=True)
        #     await wives_draw.finish()


        selected_wives = db_handler.get_selected_wives_or_husbands_by_game(closest_match, 'Wife') if closest_match else db_handler.get_all_selected_wives_or_husbands('Wife')
        selected_characters = {"_".join(name.split("_")[1:2]).lower() for name in selected_wives}
        game_images = [img for img in all_images if closest_match in img.split("_")[0].lower()]
    
        image_list = [
            img for img in game_images
            if "_" in img and "_".join(os.path.splitext(img)[0].split("_")[1:2]).lower() not in selected_characters
        ]
    else:
        selected_wives = db_handler.get_selected_wives_or_husbands_by_game(game_name, 'Wife') if game_name else db_handler.get_all_selected_wives_or_husbands('Wife')
        selected_characters = {"_".join(name.split("_")[1:2]).lower() for name in selected_wives}
        image_list = [
            img for img in all_images
            if "_" in img and "_".join(os.path.splitext(img)[0].split("_")[1:2]).lower() not in selected_characters
        ]

    if not image_list:
        try:
            await bot.send(event, f"所有老婆都被抽完了捏", reply_message=True)
        except Exception as e:
            await bot.send(event, f"发送消息时发生错误：{str(e)}", reply_message=True)
        await wives_draw.finish()

    image_name = await get_random_choice(image_list)
    image_path = os.path.join(wives_images_folder, image_name)
    image_name_short = os.path.splitext(image_name)[0]
    match_prefix = "_".join(image_name_short.split("_")[:2])

    matching_images = [
        img for img in os.listdir(wives_images_folder) 
        if os.path.splitext(img)[0].startswith(match_prefix) and 
        (os.path.splitext(img)[0] == match_prefix or os.path.splitext(img)[0][len(match_prefix)] == "_")
    ]

    db_handler.update_draw_record(user_id, image_name_short, 'Wife')
    db_handler.log_draw_history_record(user_id, image_name_short, 'Wife')

    try:
        boy_confidence, girl_confidence = await determine_gender(image_path)
        if len(matching_images) > 1:
            message = f"你抽到的老婆是 {image_name_short}\n这个角色还有 {len(matching_images) - 1} 张立绘呢"
        else:
            message = f"你抽到的老婆是 {image_name_short}\n这个角色只有一张立绘呢"

        message += f"\n女性概率 {girl_confidence:.2%}，男性概率 {boy_confidence:.2%}"
        
        await send_image_message(bot, event, message, [image_path])

    except Exception as e:
        await bot.send(event, f"发送消息时发生错误：{str(e)}", reply_message=True)
    await wives_draw.finish()
    

@husbands_draw.handle(parameterless=[CommandHandler.dependency()])
async def handle_husbands_draw(bot: Bot, event: Event):
    """抽老公"""
    user_id = str(event.get_user_id())
    message_text = event.get_plaintext().strip().lower()

    pattern = r"^抽(?:老公)?(.*?)(?:老公)?$"
    match = re.match(pattern, message_text)

    game_name = None
    if match:
        game_name = match.group(1).strip().lower()
        if game_name:
            game_name = await get_game_name_from_alias(game_name)
        else:
            game_name = None

    if user_id not in bot.config.superusers: # 普通用户
        last_draw_time = db_handler.get_last_trigger_time(user_id, 'Husband')
        if last_draw_time:
            time_difference = int((datetime.now() - last_draw_time).total_seconds())  # 转换为整数秒
            if time_difference < 10:  # 5分钟
                try:
                    remaining_time = await format_time(10 - time_difference)
                    await bot.send(event, f"合不合适也要先待满10秒吧！\n还剩下{remaining_time}。", reply_message=True)
                except Exception as e:
                    await bot.send(event, f"发送消息时发生错误：{str(e)}", reply_message=True)
                await husbands_draw.finish()

    # if game_name:
    #     try:
    #         await bot.send(event, f"你指定了 {game_name} 的老公呢，正在为你抽取...", reply_message=False)
    #     except Exception as e:
    #         await bot.send(event, f"发送消息时发生错误���{str(e)}", reply_message=True)
    #         await husbands_draw.finish()
    # else:
    #     try:
    #         await bot.send(event, f"你没有指定游戏名呢，正在为你随机抽取...", reply_message=False)
    #     except Exception as e:
    #         await bot.send(event, f"发送消息时发生错误：{str(e)}", reply_message=True)
    #         await husbands_draw.finish()

    all_images = [img for img in os.listdir(husbands_images_folder)]
    game_names_in_library = {img.split("_")[0].lower() for img in all_images if "_" in img}  # 只考虑带有_的游戏名

    if game_name:
        # 分词
        game_name_parts = list(game_name)

        # 先模糊搜索
        fuzzy_matches = await improved_partial_word_match(game_name_parts, game_names_in_library)
        if fuzzy_matches:
            closest_match = fuzzy_matches[0]
            
        else:
            # 相似度匹配
            closest_match, similarity = process.extractOne(game_name, game_names_in_library, scorer=fuzz.partial_ratio)

            if similarity < 70:
                try:
                    await bot.send(event, f"没有找到相关的游戏呢。", reply_message=True)
                except Exception as e:
                    await bot.send(event, f"发送消息时发生错误：{str(e)}", reply_message=True)
                await husbands_draw.finish()

        # try:
        #     await bot.send(event, f"根据你的指定，最匹配的游戏是：{closest_match}", reply_message=False)
        # except Exception as e:
        #     await bot.send(event, f"发送消息时发生错误：{str(e)}", reply_message=True)
        #     await husbands_draw.finish()

        selected_husbands = db_handler.get_selected_wives_or_husbands_by_game(closest_match, 'Husband') if closest_match else db_handler.get_all_selected_wives_or_husbands('Husband')
        selected_characters = {"_".join(name.split("_")[1:2]).lower() for name in selected_husbands}
        game_images = [img for img in all_images if closest_match in img.split("_")[0].lower()]
  
        image_list = [
            img for img in game_images
            if "_" in img and "_".join(os.path.splitext(img)[0].split("_")[1:2]).lower() not in selected_characters
        ]
    else:
        selected_husbands = db_handler.get_selected_wives_or_husbands_by_game(game_name, 'Husband') if game_name else db_handler.get_all_selected_wives_or_husbands('Husband')
        selected_characters = {"_".join(name.split("_")[1:2]).lower() for name in selected_husbands}
        image_list = [
            img for img in all_images
            if "_" in img and "_".join(os.path.splitext(img)[0].split("_")[1:2]).lower() not in selected_characters
        ]

    if not image_list:
        try:
            await bot.send(event, f"所有老公都被抽完了捏", reply_message=True)
        except Exception as e:
            await bot.send(event, f"发送消息时发生错误：{str(e)}", reply_message=True)
        await husbands_draw.finish()

    image_name = await get_random_choice(image_list)
    image_path = os.path.join(husbands_images_folder, image_name)
    image_name_short = os.path.splitext(image_name)[0]
    match_prefix = "_".join(image_name_short.split("_")[:2])

    matching_images = [
        img for img in os.listdir(husbands_images_folder) 
        if os.path.splitext(img)[0].startswith(match_prefix) and 
        (os.path.splitext(img)[0] == match_prefix or os.path.splitext(img)[0][len(match_prefix)] == "_")
    ]
    db_handler.update_draw_record(user_id, image_name_short, 'Husband')
    db_handler.log_draw_history_record(user_id, image_name_short, 'Husband')

    try:
        boy_confidence, girl_confidence = await determine_gender(image_path)
        if len(matching_images) > 1:
            message = f"你抽到的老公是 {image_name_short}\n这个角色还有 {len(matching_images) - 1} 张立绘呢"
        else:
            message = f"你抽到的老公是 {image_name_short}\n这个角色只有一张立绘呢"
        
        message += f"\n女性概率 {girl_confidence:.2%}，男性概率 {boy_confidence:.2%}"
        
        await send_image_message(bot, event, message, [image_path])

    except Exception as e:
        await bot.send(event, f"发送消息时发生错误：{str(e)}", reply_message=True)
        await husbands_draw.finish()


@wives_view.handle(parameterless=[CommandHandler.dependency()])
async def handle_wives_view(bot: Bot, event: Event):
    """我老婆"""
    user_id = str(event.get_user_id())
    result = db_handler.get_card_name(user_id, 'Wife')

    if not result:
        try:
            await bot.send(event, f"你还没有老婆呢，快发送 抽老婆 来抽取吧", reply_message=True)
        except Exception as e:
            await bot.send(event, f"发送消息时发生错误：{str(e)}", reply_message=True)
        await wives_view.finish()

    stored_name = result
    name_parts = stored_name.split("_")
    if len(name_parts) >= 2:
        display_name = "_".join(name_parts[:2])
    else:
        display_name = stored_name
    match_prefix = display_name

    matching_images = [
        img for img in os.listdir(wives_images_folder) 
        if os.path.splitext(img)[0].startswith(match_prefix) and 
        (os.path.splitext(img)[0] == match_prefix or os.path.splitext(img)[0][len(match_prefix)] == "_")
    ]

    if not matching_images:
        try:
            await bot.send(event, "找不到你老婆的图片呢，请确认是否已被删除", reply_message=True)
        except Exception as e:
            await bot.send(event, f"发送消息时发生错误：{str(e)}", reply_message=True)
        await wives_view.finish()
    
    if len(matching_images) > 2:
        # 合并转发
        list_tp = []
        message_content = f"你老婆是 {display_name}：\n下面是所有立绘\n"

        list_tp.append({"type": "node", "data": {"name": f"{BotConfig.self_nickname}", "uin": bot.self_id, "content": message_content}})

        for img in matching_images:
            full_name = os.path.splitext(img)[0]
            img_path = os.path.join(wives_images_folder, img)
            img_message = MessageSegment.image(img_path)
            list_tp.append({"type": "node", "data": {"name": f"{BotConfig.self_nickname}", "uin": bot.self_id, "content": f"{full_name}\n{img_message}"}})

        try:
            await send_forward_msg_handler(bot, event, list_tp)
        except Exception as e:
            await bot.send(event, f"发送消息时发生错误：{str(e)}", reply_message=True)
            await wives_view.finish()
    else:
        # 正常发送
        message_content = f"你老婆是 {display_name}：\n下面是所有立绘\n"
        for img in matching_images:
            full_name = os.path.splitext(img)[0]
            message_content += f"{full_name}\n"

        image_paths = [os.path.join(wives_images_folder, img) for img in matching_images]
        try:
            await send_image_message(bot, event, message_content, image_paths)
        except Exception as e:
            await bot.send(event, f"发送消息时发生错误：{str(e)}", reply_message=True)
            await wives_view.finish()
        

@husbands_view.handle(parameterless=[CommandHandler.dependency()])
async def handle_husbands_view(bot: Bot, event: Event):
    """我老公"""
    user_id = str(event.get_user_id())
    result = db_handler.get_card_name(user_id, 'Husband')

    if not result:
        try:
            await bot.send(event, f"你还没有老公呢，快发送 抽老公 来抽取吧", reply_message=True)
        except Exception as e:
            await bot.send(event, f"发送消息时发生错误：{str(e)}", reply_message=True)
        await husbands_view.finish()

    stored_name = result
    name_parts = stored_name.split("_")
    if len(name_parts) >= 2:
        display_name = "_".join(name_parts[:2])
    else:
        display_name = stored_name
    match_prefix = display_name

    matching_images = [
        img for img in os.listdir(husbands_images_folder) 
        if os.path.splitext(img)[0].startswith(match_prefix) and 
        (os.path.splitext(img)[0] == match_prefix or os.path.splitext(img)[0][len(match_prefix)] == "_")
    ]

    if not matching_images:
        try:
            await bot.send(event, "找不到你老公的图片呢，请确认是否已被删除", reply_message=True)
        except Exception as e:
            await bot.send(event, f"发送消息时发生错误：{str(e)}", reply_message=True)
        await husbands_view.finish()

    if len(matching_images) > 2:
        # 合并转发
        list_tp = []
        message_content = f"你老公是 {display_name}：\n下面是所有立绘\n"

        list_tp.append({"type": "node", "data": {"name": f"{BotConfig.self_nickname}", "uin": bot.self_id, "content": message_content}})

        for img in matching_images:
            full_name = os.path.splitext(img)[0]
            img_path = os.path.join(husbands_images_folder, img)
            img_message = MessageSegment.image(img_path)
            list_tp.append({"type": "node", "data": {"name": f"{BotConfig.self_nickname}", "uin": bot.self_id, "content": f"{full_name}\n{img_message}"}})

        try:
            await send_forward_msg_handler(bot, event, list_tp)
        except Exception as e:
            await bot.send(event, f"发送消息时发生错误：{str(e)}", reply_message=True)
            await husbands_view.finish()
    else:
        # 正常发送
        message_content = f"你老公是 {display_name}：\n下面是所有立绘\n"
        for img in matching_images:
            full_name = os.path.splitext(img)[0]
            message_content += f"{full_name}\n"

        image_paths = [os.path.join(husbands_images_folder, img) for img in matching_images]
        try:
            await send_image_message(bot, event, message_content, image_paths)
        except Exception as e:
            await bot.send(event, f"发送消息时发生错误：{str(e)}", reply_message=True)
            await husbands_view.finish()


@wives_rename.handle(parameterless=[CommandHandler.dependency()])
async def handle_wives_rename(bot: Bot, event: Event, state: T_State):
    """老婆改名"""
    user_id = str(event.get_user_id())
    stored_name = db_handler.get_card_name(user_id, 'Wife')
    message = event.get_plaintext().strip()
    match = re.match(r"老婆改名(.+)", message)

    if not stored_name:
        try:
            await bot.send(event, f"你还没有老婆呢，快发送 抽老婆 来抽取吧", reply_message=True)
        except Exception as e:
            await bot.send(event, f"发送消息时发生错误：{str(e)}", reply_message=True)
        await wives_rename.finish()

    matching_images = [img for img in os.listdir(wives_images_folder) if os.path.splitext(img)[0] == stored_name]
    
    if not matching_images:
        try:
            await bot.send(event, f"找不到你老婆的图片呢，请确认是否已被删除", reply_message=True)
        except Exception as e:
            await bot.send(event, f"发送消息时发生错误：{str(e)}", reply_message=True)
        await wives_rename.finish()
    
    if match:
        original_game_name = stored_name.split("_")[0]
        new_name = match.group(1).strip()
        new_game_name = new_name.split("_")[0]

        if new_game_name != original_game_name:
            try:
                await bot.send(event, f"新名字中的游戏名与原始游戏名不一致，请保持游戏名一致。", reply_message=True)
            except Exception as e:
                await bot.send(event, f"发送消息时发生错误：{str(e)}", reply_message=True)
            await wives_rename.finish()

        if re.match(r"^[^\s_]+(_[^\s_]+)+$", new_name):
            await perform_wife_rename(bot, event, user_id, new_name)
        else:
            try:
                await bot.send(event, f"名字格式不正确，请确保每段内容用下划线连接且无多余空格", reply_message=True)
            except Exception as e:
                await bot.send(event, f"发送消息时发生错误：{str(e)}", reply_message=True)
        await wives_rename.finish()
    
    show_image = [os.path.join(wives_images_folder, img) for img in matching_images]
    image_name = [os.path.splitext(img)[0] for img in matching_images]
    message_content = f"当前图片名称如下：\n"
    for name in image_name:
        message_content += f"{name}\n"
    
    message_content += f"字格式：\n游戏名_角色名称_皮肤名称/阶段状态等信息(没有可不写)\n举例：解神者_少姜_蓝水乐园\n不知道或者点错了的话请发送 取消"
    try:
        await send_image_message(bot, event, message_content, show_image)
    except Exception as e:
        await bot.send(event, f"发送消息时发生错误：{str(e)}", reply_message=True)
    state["awaiting_name"] = True
    state["user_id"] = user_id
    state["session_id"] = event.get_session_id()


@wives_rename.got("new_name", prompt="请输入新的名字：")
async def handle_got_new_wives_name(bot: Bot, event: Event, state: T_State):
    """新名字"""
    if state.get("awaiting_name") and state.get("user_id") == str(event.get_user_id()):
        new_name = event.get_plaintext().strip()
        stored_name = db_handler.get_card_name(state["user_id"], 'Wife')
        original_game_name = stored_name.split("_")[0]
        new_game_name = new_name.split("_")[0]

        if new_name.lower() == "取消":
            try:
                await bot.send(event, f"已取消改名操作", reply_message=True)
            except Exception as e:
                await bot.send(event, f"发送消息时发生错误：{str(e)}", reply_message=True)
            state["awaiting_name"] = False
            await wives_rename.finish()

        if not new_name:
            try:
                await bot.send(event, f"名字不能为空，请输入有效的名字", reply_message=True)
            except Exception as e:
                await bot.send(event, f"发送消息时发生错误：{str(e)}", reply_message=True)
            await wives_rename.reject()
            return
        
        # 检查新名字的游戏名是否一致
        if new_game_name != original_game_name:
            try:
                await bot.send(event, f"新名字中的游戏名与原始游戏名不一致，请保持游戏名一致。", reply_message=True)
            except Exception as e:
                await bot.send(event, f"发送消息时发生错误：{str(e)}", reply_message=True)
            await wives_rename.reject()
            return

        if not re.match(r"^[^\s_]+(_[^\s_]+)+$", new_name):
            try:
                await bot.send(event, f"名字格式不正确，请确保每段内容用下划线连接且无多余空格", reply_message=True)
            except Exception as e:
                await bot.send(event, f"发送消息时发生错误：{str(e)}", reply_message=True)
            await wives_rename.reject()
            return
        

        old_name = stored_name
        game_name, char_name, *rest = old_name.split("_")
        stored_image_suffix = "_".join(rest) if rest else ""

        # 获取已改名的图片，避免重复重命名
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
            await wives_rename.finish()

        for img in matching_images:
            old_path = os.path.join(wives_images_folder, img)

            # 如果图片已经改名过，跳过处理
            if img in [ri[0] for ri in renamed_images]:
                continue

            img_name_parts = os.path.splitext(img)[0].split("_")
            img_suffix = "_".join(img_name_parts[2:]) if len(img_name_parts) > 2 else ""

            if img_suffix == stored_image_suffix:
                new_path = os.path.join(wives_images_folder, new_name + os.path.splitext(img)[1])
                db_handler.update_renamed_record(state["user_id"], os.path.splitext(img)[0], new_name)
            else:
                new_base_name = f"{new_name.split('_')[0]}_{new_name.split('_')[1]}_{img_suffix}"
                new_path = os.path.join(wives_images_folder, new_base_name + os.path.splitext(img)[1])

            try:
                os.rename(old_path, new_path)
            except Exception as e:
                await bot.send(event, f"重命名失败：{str(e)}", reply_message=True)

        db_handler.update_draw_record(state["user_id"], new_name, 'Wife')

        try:
            await bot.send(event, f"已将你老婆和相关立绘重命名为 {new_name}", reply_message=True)
        except Exception as e:
            await bot.send(event, f"发送消息时发生错误：{str(e)}", reply_message=True)
        state["awaiting_name"] = False
        await wives_rename.finish()


@husbands_rename.handle(parameterless=[CommandHandler.dependency()])
async def handle_husbands_rename(bot: Bot, event: Event, state: T_State):
    """老公改名"""
    user_id = str(event.get_user_id())
    stored_name = db_handler.get_card_name(user_id, 'Husband')
    message = event.get_plaintext().strip()
    match = re.match(r"老公改名(.+)", message)
    
    if not stored_name:
        try:
            await bot.send(event, f"你还没有老公呢，快发送 抽老公 来抽取吧", reply_message=True)
        except Exception as e:
            await bot.send(event, f"发送消息时发生错误：{str(e)}", reply_message=True)
        await husbands_rename.finish()

    matching_images = [img for img in os.listdir(husbands_images_folder) if os.path.splitext(img)[0] == stored_name]
    
    if not matching_images:
        try:
            await bot.send(event, f"找不到你老公的图片呢，请确认是否已被删除", reply_message=True)
        except Exception as e:
            await bot.send(event, f"发送消息时发生错误：{str(e)}", reply_message=True)
        await husbands_rename.finish()
    
    # "老婆改名 <新名称>" 格式
    if match:
        original_game_name = stored_name.split("_")[0]
        new_name = match.group(1).strip()
        new_game_name = new_name.split("_")[0]
        
        if new_game_name != original_game_name:
            try:
                await bot.send(event, f"新名字中的游戏名与原始游戏名不一致，请保持游戏名一致。", reply_message=True)
            except Exception as e:
                await bot.send(event, f"发送消息时发生错误：{str(e)}", reply_message=True)
            await husbands_rename.finish()
    
        if re.match(r"^[^\s_]+(_[^\s_]+)+$", new_name):
            await perform_husband_rename(bot, event, user_id, new_name)
        else:
            try:
                await bot.send(event, f"名字格式不正确，请确保每段内容用下划线连接且无多余空格", reply_message=True)
            except Exception as e:
                await bot.send(event, f"发送消息时发生错误：{str(e)}", reply_message=True)
        await husbands_rename.finish()
    
    show_image = [os.path.join(husbands_images_folder, img) for img in matching_images]
    image_name = [os.path.splitext(img)[0] for img in matching_images]
    message_content = f"当前图片名称如下：\n"
    for name in image_name:
        message_content += f"{name}\n"
    
    message_content += f"名字格式：\n游戏名_角色名称_皮肤名称/阶段状态等信息(没有可不写)\n举例：解神者_少姜_蓝水乐园\n不知道或者点错了的话请发送 取消"
    try:
        await send_image_message(bot, event, message_content, show_image)
    except Exception as e:
        await bot.send(event, f"发送消息时发生错误：{str(e)}", reply_message=True)
    state["awaiting_name"] = True
    state["user_id"] = user_id
    state["session_id"] = event.get_session_id()

@husbands_rename.got("new_name", prompt="请输入新的名字：")
async def handle_got_new_husbands_name(bot: Bot, event: Event, state: T_State):
    """新名字"""
    if state.get("awaiting_name") and state.get("user_id") == str(event.get_user_id()):
        new_name = event.get_plaintext().strip()
        stored_name = db_handler.get_card_name(state["user_id"], 'Husband')
        original_game_name = stored_name.split("_")[0]
        new_game_name = new_name.split("_")[0]

        if new_name.lower() == "取消":
            try:
                await bot.send(event, f"已取消改名操作", reply_message=True)
            except Exception as e:
                await bot.send(event, f"发送消息时发生错误：{str(e)}", reply_message=True)
            state["awaiting_name"] = False
            await husbands_rename.finish()

        if not new_name:
            try:
                await bot.send(event, f"名字不能为空，请输入有效的名字", reply_message=True)
            except Exception as e:
                await bot.send(event, f"发送消息时发生错误：{str(e)}", reply_message=True)
            await husbands_rename.reject()
            return
        
        # 检查新名字的游戏名是否一致
        if new_game_name != original_game_name:
            try:
                await bot.send(event, f"新名字中的游戏名与原始游戏名不一致，请保持游戏名一致。原游戏名为 {original_game_name}", reply_message=True)
            except Exception as e:
                await bot.send(event, f"发送消息时发生错误：{str(e)}", reply_message=True)
            await husbands_rename.reject()
            return

        if not re.match(r"^[^\s_]+(_[^\s_]+)+$", new_name):
            try:
                await bot.send(event, f"名字格式不正确，请确保每段内容用下划线连接且无多余空格", reply_message=True)
            except Exception as e:
                await bot.send(event, f"发送消息时发生错误：{str(e)}", reply_message=True)
            await husbands_rename.reject()
            return
        

        old_name = stored_name
        game_name, char_name, *rest = old_name.split("_")
        stored_image_suffix = "_".join(rest) if rest else ""

        # 获取已改名的图片，避免重复重命名
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
            await husbands_rename.finish()

        for img in matching_images:
            old_path = os.path.join(wives_images_folder, img)

            # 如果图片已经改名过，跳过处理
            if img in [ri[0] for ri in renamed_images]:
                continue

            img_name_parts = os.path.splitext(img)[0].split("_")
            img_suffix = "_".join(img_name_parts[2:]) if len(img_name_parts) > 2 else ""

            if img_suffix == stored_image_suffix:
                new_path = os.path.join(wives_images_folder, new_name + os.path.splitext(img)[1])
                db_handler.update_renamed_record(state["user_id"], os.path.splitext(img)[0], new_name)
            else:
                new_base_name = f"{new_name.split('_')[0]}_{new_name.split('_')[1]}_{img_suffix}"
                new_path = os.path.join(wives_images_folder, new_base_name + os.path.splitext(img)[1])

            try:
                os.rename(old_path, new_path)
            except Exception as e:
                await bot.send(event, f"重命名失败：{str(e)}", reply_message=True)

        db_handler.update_draw_record(state["user_id"], new_name, 'Husband')

        try:
            await bot.send(event, f"已将你老公和相关立绘重命名为 {new_name}", reply_message=True)
        except Exception as e:
            await bot.send(event, f"发送消息时发生错误：{str(e)}", reply_message=True)
        state["awaiting_name"] = False
        await husbands_rename.finish()


@wives_probability.handle(parameterless=[CommandHandler.dependency()])
async def handle_wives_probability(bot: Bot, event: Event):
    """老婆概率"""
    try:
        message_text = event.get_plaintext().strip().replace(" ", "")

        all_images = [f for f in os.listdir(wives_images_folder) if f.count('_') >= 1]
        if not all_images:
            await bot.send(event, "没有找到任何有效的图片", reply_message=True)
            await wives_probability.finish()
            return

        limit = len(all_images)
        match = re.search(r'老婆概率(\d+|[一二三四五六七八九十百]+)', message_text)
        if match:
            number_str = match.group(1)
            if number_str.isdigit():
                limit = int(number_str)
            else:  # 如果是中文数字
                parsed_number = await parse_chinese_numeral(number_str)
                if parsed_number:
                    limit = parsed_number

        game_stats = calculate_game_stats(all_images)
        await generate_and_send_stats(bot, event, game_stats, limit)

    except Exception as e:
        await bot.send(event, f"处理请求时发生错误，请稍后重试", reply_message=True)
        await wives_probability.finish()


@husbands_probability.handle(parameterless=[CommandHandler.dependency()])
async def handle_husbands_probability(bot: Bot, event: Event):
    """老公概率"""
    try:
        message_text = event.get_plaintext().strip().replace(" ", "")

        all_images = [f for f in os.listdir(husbands_images_folder) if f.count('_') >= 1]
        if not all_images:
            await bot.send(event, "没有找到任何有效的图片", reply_message=True)
            await wives_probability.finish()
            return

        limit = len(all_images)
        match = re.search(r'老公概率(\d+|[一二三四五六七八九十百]+)', message_text)
        if match:
            number_str = match.group(1)
            if number_str.isdigit():
                limit = int(number_str)
            else:  # 如果是中文数字
                parsed_number = await parse_chinese_numeral(number_str)
                if parsed_number:
                    limit = parsed_number

        game_stats = calculate_game_stats(all_images)
        await generate_and_send_stats(bot, event, game_stats, limit)

    except Exception as e:
        await bot.send(event, f"处理请求时发生错误，请稍后重试", reply_message=True)
        await wives_probability.finish()


@delete_husbands.handle(parameterless=[CommandHandler.dependency()])
async def handle_delete_husbands(bot: Bot, event: Event):
    """这是男的"""
    if event.reply:
        await bot.send(event, "这不是一个回复指令\n请注意: 只能处理自己拥有的立绘", reply_message=True)
        await delete_husbands.finish()

    user_id = str(event.get_user_id())
    stored_name = db_handler.get_card_name(user_id, 'Wife')

    if not os.path.exists(husbands_images_folder):
        os.makedirs(husbands_images_folder)

    if not stored_name:
        try:
            await bot.send(event, f"你还没有老婆呢，快发送 抽老婆 来抽取吧", reply_message=True)
        except Exception as e:
            await bot.send(event, f"发送消息时发生错误：{str(e)}", reply_message=True)
        await delete_husbands.finish()

    matching_images = [
        img for img in os.listdir(wives_images_folder)
        if await is_exact_match(img, stored_name)
    ]
    
    if not matching_images:
        try:
            await bot.send(event, "找不到这张图片呢，无法移动", reply_message=True)
        except Exception as e:
            await bot.send(event, f"发送消息时发生错误：{str(e)}", reply_message=True)
        await delete_husbands.finish()

    
    check_img = matching_images[0]
    check_img_path = os.path.join(wives_images_folder, check_img)
    boy_confidence, girl_confidence = await determine_gender(check_img_path)

    if girl_confidence > 0.6:
        list_tp = []
        
        user_info = await bot.get_stranger_info(user_id=int(user_id))
        user_nickname = user_info.get('nickname', user_id)
        
        if isinstance(event, GroupMessageEvent):
            group_info = await bot.get_group_info(group_id=event.group_id)
            group_name = group_info.get('group_name', '未知群')
            source_info = f"群{group_name}({event.group_id})"
        else:
            source_info = "私聊"
        
        message_content = (f"来自{source_info}的用户{user_nickname}({user_id})"
                         f"为概率大于60%的图片使用了指令 这是男的，下面是将要被移动的图片")
        
        list_tp.append({
            "type": "node",
            "data": {
                "name": "可疑操作提醒",
                "uin": bot.self_id,
                "content": message_content
            }
        })

        for img in matching_images:
            img_path = os.path.join(wives_images_folder, img)
            img_message = MessageSegment.image(img_path)
            list_tp.append({
                "type": "node",
                "data": {
                    "name": "相关图片",
                    "uin": bot.self_id,
                    "content": f"{os.path.splitext(img)[0]}\n{img_message}"
                }
            })

        for superuser in bot.config.superusers:
            try:
                temp_event = type('TempEvent', (), {'user_id': superuser})()
                await send_forward_msg_handler(bot, temp_event, list_tp)
            except Exception as e:
                logger.error(f"向超管 {superuser} 发送提醒消息时发生错误：{str(e)}")


    deleted_images = []
    failed_images = []

    for img in matching_images:
        img_path = os.path.join(wives_images_folder, img)
        target_path = os.path.join(husbands_images_folder, img)
        try:
            shutil.move(img_path, target_path)
            deleted_images.append(os.path.splitext(img)[0])
        except Exception as e:
            failed_images.append(img)

    if deleted_images:
        deleted_list = "\n".join(deleted_images)
        try:
            await bot.send(event, f"以下图片已移动到正确文件夹：\n{deleted_list}", reply_message=True)
            db_handler.delete_draw_record(user_id, 'Wife')
            db_handler.log_moved_record(user_id, 'delete_husbands', deleted_images, 'Husband')
        except Exception as e:
            await bot.send(event, f"移动结果时发生错误：{str(e)}", reply_message=True)
            await delete_husbands.finish()

    if failed_images:
        failed_list = "\n".join(failed_images)
        try:
            await bot.send(event, f"以下图片移动失败：\n{failed_list}", reply_message=True)
        except Exception as e:
            await bot.send(event, f"移动时发生错误：{str(e)}", reply_message=True)
            await delete_husbands.finish()


@delete_wives.handle(parameterless=[CommandHandler.dependency()])
async def handle_delete_wives(bot: Bot, event: Event):
    """这是女的"""
    if event.reply:
        await bot.send(event, "这不是一个回复指令\n请注意: 只能处理自己拥有的立绘", reply_message=True)
        await delete_husbands.finish()

    user_id = str(event.get_user_id())
    stored_name = db_handler.get_card_name(user_id, 'Husband')

    if not os.path.exists(husbands_images_folder):
        os.makedirs(husbands_images_folder)

    if not stored_name:
        try:
            await bot.send(event, f"你还没有老公呢，快发送 抽老公 来抽取吧", reply_message=True)
        except Exception as e:
            await bot.send(event, f"发送消息时发生错误：{str(e)}", reply_message=True)
        await delete_wives.finish()

    matching_images = [
        img for img in os.listdir(husbands_images_folder)
        if await is_exact_match(img, stored_name)
    ]

    if not matching_images:
        try:
            await bot.send(event, "找不到这张图片呢，无法移动", reply_message=True)
        except Exception as e:
            await bot.send(event, f"发送消息时发生错误：{str(e)}", reply_message=True)
        await delete_wives.finish()

    check_img = matching_images[0]
    check_img_path = os.path.join(wives_images_folder, check_img)
    boy_confidence, girl_confidence = await determine_gender(check_img_path)

    if boy_confidence > 0.6:
        list_tp = []
        
        user_info = await bot.get_stranger_info(user_id=int(user_id))
        user_nickname = user_info.get('nickname', user_id)
        
        if isinstance(event, GroupMessageEvent):
            group_info = await bot.get_group_info(group_id=event.group_id)
            group_name = group_info.get('group_name', '未知群')
            source_info = f"群{group_name}({event.group_id})"
        else:
            source_info = "私聊"
        
        message_content = (f"来自{source_info}的用户{user_nickname}({user_id})"
                         f"为概率大于60%的图片使用了指令 这是女的，下面是将要被移动的图片")
        
        list_tp.append({
            "type": "node",
            "data": {
                "name": "可疑操作提醒",
                "uin": bot.self_id,
                "content": message_content
            }
        })

        for img in matching_images:
            img_path = os.path.join(wives_images_folder, img)
            img_message = MessageSegment.image(img_path)
            list_tp.append({
                "type": "node",
                "data": {
                    "name": "相关图片",
                    "uin": bot.self_id,
                    "content": f"{os.path.splitext(img)[0]}\n{img_message}"
                }
            })

        for superuser in bot.config.superusers:
            try:
                temp_event = type('TempEvent', (), {'user_id': superuser})()
                await send_forward_msg_handler(bot, temp_event, list_tp)
            except Exception as e:
                logger.error(f"向超管 {superuser} 发送提醒消息时发生错误：{str(e)}")


    deleted_images = []
    failed_images = []

    for img in matching_images:
        img_path = os.path.join(husbands_images_folder, img)
        target_path = os.path.join(wives_images_folder, img)
        try:
            shutil.move(img_path, target_path)
            deleted_images.append(os.path.splitext(img)[0])
        except Exception as e:
            failed_images.append(img)

    if deleted_images:
        deleted_list = "\n".join(deleted_images)
        try:
            await bot.send(event, f"以下图片移动到正确文件夹：\n{deleted_list}", reply_message=True)
            db_handler.delete_draw_record(user_id, 'Husband')
            db_handler.log_moved_record(user_id, 'delete_wives', deleted_images, 'Wife')
        except Exception as e:
            await bot.send(event, f"移动结果时发生错误：{str(e)}", reply_message=True)
            await delete_wives.finish()

    if failed_images:
        failed_list = "\n".join(failed_images)
        try:
            await bot.send(event, f"以下图片移动失败：\n{failed_list}", reply_message=True)
        except Exception as e:
            await bot.send(event, f"移动时发生错误：{str(e)}", reply_message=True)
            await delete_wives.finish()


async def handle_help_confirmation(bot: Bot, event: Event):
    """处理帮助确认事件"""
    try:
        font_prop = FontProperties(fname=font_path)
        help_text = __plugin_meta__.usage
        
        text_lines = help_text.split('\n')
        max_length = max(len(line) for line in text_lines)
        fig_width = max_length * 0.1
        fig_height = max(len(text_lines) * 0.1 + 1, 1)
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.axis('off')
        fig.patch.set_facecolor('white')
        ax.text(0, 0.5, help_text, fontsize=14, ha='left', va='center', fontproperties=font_prop)

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', transparent=False)
        buf.seek(0)
        
        message = (MessageSegment.image(buf) + 
                  MessageSegment.text("\n请先阅读并同意帮助信息并在120秒内回应：\n贴第一个表情表示已阅读并同意\n贴第二个表情表示不同意\n如果您的QQ版本过低，也可以直接使用 帮助抽游戏立绘 来查看帮助信息。"))
        
        response = await bot.send(event, message)
        bot_message_id = response['message_id']
        original_message_id = event.message_id
        
        await help_manager.add_confirmation(
            str(event.get_user_id()),
            bot_message_id,
            original_message_id,
            bot,
            event
        )
        
        emoji_ids = ["38", "417"]  # 同意、不同意
        for emoji_id in emoji_ids:
            await asyncio.sleep(0.1)
            await bot.call_api("set_msg_emoji_like", message_id=bot_message_id, emoji_id=emoji_id)
        
    except Exception as e:
        logger.error(f"处理帮助确认时发生错误: {e}")
        await bot.send(event, "处理帮助确认时发生错误，���稍后重试。")
    finally:
        if 'buf' in locals():
            buf.close()
        plt.close('all')


@on_notice
async def handle_help_emoji_response(bot: Bot, event: NoticeEvent):
    """处理帮助确认的表情响应"""
    try:
        if event.notice_type == "group_msg_emoji_like":
            message_id = event.dict().get('message_id')
            user_id = str(event.dict().get('user_id'))
            emoji_id = event.dict().get('likes')[0].get('emoji_id')
            
            confirmation = help_manager.get_confirmation(user_id)
            if confirmation and confirmation.message_id == message_id and confirmation.processing:
                if emoji_id == "38":  # 同意
                    await help_manager.set_confirmed(user_id, True)
                    db_handler.mark_help_as_read(user_id)
                    reply_msg = MessageSegment.reply(confirmation.original_message_id) + "感谢您同意霸王条款，现在您可以正常使用所有指令了。"
                    await bot.send(event, reply_msg)
                
                elif emoji_id == "417":  # 不同意
                    await help_manager.set_confirmed(user_id, False)
                    reply_msg = MessageSegment.reply(confirmation.original_message_id) + "您已选择不同意，将无法使用相关功能。如需使用，请重新触发指令并同意霸王条款。"
                    await bot.send(event, reply_msg)
                
                help_manager.remove_confirmation(user_id)
    except Exception as e:
        logger.error(f"处理表情响应时发生错误: {e}")


# 投票状态
ongoing_votes: Dict[str, Dict] = {}

@on_notice
async def handle_emoji_vote(bot: Bot, event: NoticeEvent):
    """处理表情投票事件"""
    if event.notice_type == "group_msg_emoji_like":
        message_id = event.dict().get('message_id')
        user_id = event.dict().get('user_id')
        emoji_id = event.dict().get('likes')[0].get('emoji_id')

        if message_id in ongoing_votes:
            vote_data = ongoing_votes[message_id]

            # 已经投过票，不允许再投
            if user_id in vote_data["agree"] or user_id in vote_data["disagree"] or user_id in vote_data["abstain"]:
                return

            # 用户投票
            if emoji_id == "38":
                vote_data["agree"].add(user_id)
            elif emoji_id == "417":
                vote_data["disagree"].add(user_id)
            elif emoji_id == "277":
                vote_data["abstain"].add(user_id)
            else:
                return 

            # 图片拥有者投票
            if user_id == vote_data["user_id"]:
                if emoji_id == "38":
                    await bot.send(event, "图片拥有者投了赞成票，投票结束！图片将被删除。", message_id=message_id, reply_message=True)
                    await finalize_vote(bot, vote_data, message_id, event, early_termination=True, owner_vote='agree')
                elif emoji_id == "417":
                    await bot.send(event, "图片拥有者投了反对票，投票结束！图片将被保留。", message_id=message_id, reply_message=True)
                    await finalize_vote(bot, vote_data, message_id, event, early_termination=True, owner_vote='disagree')
            elif str(user_id) in bot.config.superusers:
                # 超级用户投票
                if emoji_id == "38":
                    await bot.send(event, "超级用户投了赞成票，投票结束！图片将被删除。", message_id=message_id, reply_message=True)
                    await finalize_vote(bot, vote_data, message_id, event, early_termination=True, owner_vote='agree')
                elif emoji_id == "417":
                    await bot.send(event, "超级用户投了反对票，投票结束！图片将被保留。", message_id=message_id, reply_message=True)
                    await finalize_vote(bot, vote_data, message_id, event, early_termination=True, owner_vote='disagree')


async def finalize_vote(bot: Bot, vote_data, message_id, event, early_termination=False, owner_vote=None):
    """终止计时并统计投票结果"""
    try:
        ongoing_votes.pop(message_id, None)  # 移除投票
        user_id = vote_data["user_id"]
        img_name = vote_data['image_name']
        vote_result = "保留图片"  # 默认保留
        agree_votes = len(vote_data["agree"])
        disagree_votes = len(vote_data["disagree"])

        if early_termination:
            # 图片拥有者投票，直接确定结果
            if owner_vote == 'agree':
                vote_result = "删除图片"
            elif owner_vote == 'disagree':
                vote_result = "保留图片"
        else:
            # 正常投票结果
            if agree_votes >= 1 and agree_votes > disagree_votes:
                vote_result = "删除图片"
                await bot.send(event, f"计时结束，图片将被删除！", message_id=message_id, reply_message=True)
            else:
                vote_result = "保留图片"
                await bot.send(event, f"计时结束，图片保留！", message_id=message_id, reply_message=True)

        vote_record = {
            "agree": list(vote_data["agree"]),
            "disagree": list(vote_data["disagree"]),
            "abstain": list(vote_data["abstain"])
        }
        db_handler.save_vote_record(
            owner_user_id=user_id,
            image_name=img_name,
            vote_result=vote_result,
            voters=json.dumps(vote_record, ensure_ascii=False)
        )

        if vote_result == "删除图片":
            deleted_images = []
            failed_images = []
            
            matching_images = [
                img for img in os.listdir(husbands_images_folder)
                if await is_exact_match(img, img_name)
            ]
            matching_images += [
                img for img in os.listdir(wives_images_folder)
                if await is_exact_match(img, img_name)
            ]

            if not matching_images:
                await bot.send(event, f"无法找到与 {img_name} 匹配的图片，无法删除。", reply_message=True)
                return

            for img in matching_images:
                img_path = os.path.join(husbands_images_folder, img) if img in os.listdir(husbands_images_folder) else os.path.join(wives_images_folder, img)
                target_path = os.path.join(drop_folder, img)

                try:
                    shutil.move(img_path, target_path)
                    deleted_images.append(os.path.splitext(img)[0])
                except Exception as e:
                    failed_images.append(img)
                    await bot.send(event, f"删除图片 {img} 时发生���误：{str(e)}", reply_message=True)


            db_record = db_handler.get_card_name(user_id)
            if db_record and any(img_name in card for card in db_record):
                db_handler.delete_draw_record(user_id, 'Husband' if 'Husband' in img_name.lower() else 'Wife')

            if failed_images:
                failed_list = "\n".join(failed_images)
                await bot.send(event, f"以下图片删除失败：\n{failed_list}", reply_message=True)

    except Exception as e:
        await bot.send(event, f"投票终止时发生错误：{str(e)}", reply_message=True)


@vote_delete.handle(parameterless=[CommandHandler.dependency()])
async def handle_vote_delete(bot: Bot, event: GroupMessageEvent, state: T_State, args: Message = CommandArg()):

    state["reply_message_id"] = event.reply.message_id if event.reply else None
    image_name = None
    image_owner_id = None
    card_type = None

    if state["reply_message_id"]:

        reply_text = event.reply.message.extract_plain_text()

        if "你抽到的老婆是" in reply_text:
            image_name = reply_text.split("你抽到的老婆是", 1)[1].split("\n")[0].strip()
            card_type = 'Wife'
        elif "你抽到的老公是" in reply_text:
            image_name = reply_text.split("你抽到的老公是", 1)[1].split("\n")[0].strip()
            card_type = 'Husband'


        if state["reply_message_id"] in ongoing_votes or any(vote["image_name"] == image_name for vote in ongoing_votes.values()):
            await bot.send(event, "该图片已经有正在进行的投票，无法重复发起。", reply_message=True)
            await vote_delete.finish()
        
        
        if db_handler.get_vote_record(image_name):
            await bot.send(event, "数据库中已存在针对该消息的投票记录，无法重复发起。", reply_message=True)
            await vote_delete.finish()

        image_owner_id = await get_original_sender(bot, event.reply.message_id)
    else:
        await bot.send(event, "请回复一条回复消息以发起投票。", reply_message=True)
        await vote_delete.finish()

    if image_name is None or card_type is None:
        await bot.send(event, "无法获取图片的名称或类型，请确保您回复的是正确格式的消息。", reply_message=True)
        await vote_delete.finish()

    if image_owner_id is None:
        await bot.send(event, "无法获取图片的拥有者，无法发起投票。", reply_message=True)
        await vote_delete.finish()

    response = await bot.send(
        event,
        MessageSegment.reply(state["reply_message_id"]) +
        Message(f"开始为 {image_name} 发起删除投票\n"
                "请在60秒内投票：\n"
                "贴第一个表情表示赞成\n"
                "贴第二个表情表示反对\n"
                "贴第三个表情表示弃权")
    )
    bot_message_id = response['message_id']

    ongoing_votes[bot_message_id] = {
        "message_id": state["reply_message_id"],
        "image_name": image_name,
        "group_id": event.group_id,
        "user_id": image_owner_id,
        "user_vote": None,
        "agree": set(),
        "disagree": set(),
        "abstain": set()
    }

    emoji_ids = ["38", "417", "277"]  # 赞成、反对、弃权
    for emoji_id in emoji_ids:
        await asyncio.sleep(0.1)
        await bot.call_api("set_msg_emoji_like", message_id=bot_message_id, emoji_id=emoji_id)
        
    await asyncio.sleep(60)

    # 结果
    vote_data = ongoing_votes.pop(bot_message_id, None)
    if vote_data:
        await finalize_vote(bot, vote_data, bot_message_id, event)

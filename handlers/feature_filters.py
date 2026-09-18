from nonebot.adapters.onebot.v11 import Bot, Event

from ..matchers import feature_list
from ..services.feature_aliases import FeatureAliasError, feature_alias_registry
from ..services.feature_catalog import feature_list_messages
from ..services.messaging import send_forward_msg_handler


@feature_list.handle()
async def handle_feature_list(bot: Bot, event: Event) -> None:
    try:
        messages = feature_list_messages(feature_alias_registry)
    except FeatureAliasError as exc:
        await bot.send(event, f"读取立绘特征列表失败：{exc}", reply_message=True)
        return

    await send_forward_msg_handler(bot, event, "立绘特征列表", bot.self_id, messages)

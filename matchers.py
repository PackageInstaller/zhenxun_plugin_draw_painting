from nonebot import on_command, on_fullmatch, on_notice, on_regex
from nonebot.permission import SUPERUSER

wives_draw = on_regex(r"^抽(?!.*老公).*老婆.*$", priority=5)
wives_view = on_fullmatch("查看老婆", priority=5)
wives_rename = on_command("老婆改名", priority=5, expire_time=None)
wives_probability = on_command("老婆概率", priority=5)

husbands_draw = on_regex(r"^抽(?!.*老婆).*老公.*$", priority=5)
husbands_view = on_fullmatch("查看老公", priority=5)
husbands_rename = on_command("老公改名", priority=5, expire_time=None)
husbands_probability = on_command("老公概率", priority=5)

help_command = on_fullmatch("帮助抽游戏立绘", priority=5)
feature_list = on_fullmatch("立绘特征列表", priority=5)
feature_query = on_command("特征查询", priority=5, block=True, permission=SUPERUSER)

help_emoji_response = on_notice(priority=1, block=False)

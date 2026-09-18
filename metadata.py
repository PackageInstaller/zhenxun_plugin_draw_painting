from nonebot.plugin import PluginMetadata

from zhenxun.configs.utils import BaseBlock, PluginExtraData
from zhenxun.utils.enum import BlockType, PluginType

__plugin_meta__ = PluginMetadata(
    name="游戏立绘抽卡",
    description="从本地图片库中随机抽取游戏人物立绘",
    usage="""
    抽取老婆/老公
    指令：
    抽老婆/老公［游戏名参数可选］
    抽老婆/抽老公 [游戏名] -n [人物名或片段] [完整名称优先，多个匹配随机抽取]
    抽老婆/抽老公 [游戏名] -t [特征1 特征2...] [组合筛选，所有特征同时满足]
    立绘特征列表 [查看支持的中文特征名称]
    特征查询 + 图片 [查询主体/整图特征并返回标注图，也支持回复图片]
    查看老婆/老公 [查看所有立绘]
    老婆/老公改名 [修改单张立绘名称]
    老婆/老公概率 ?[数量参数可选，默认全部] [查看各游戏占比]
    打包立绘 [游戏名或别名] [打包该游戏男女立绘并生成 1 天下载链接]
    别名添加 [原游戏名] [别名1] [别名2...] [超级用户添加游戏别名]
    帮助抽游戏立绘 查看帮助
    请注意，如果出现乱用指令的情况，将会被永久封禁。
    Q:为什么没有xx游戏？
    A:
    1.这里面的游戏都是拆包获得的，如果你有想提供的，可以找我，我会放进去。
    2.只有有完整静态立绘的游戏才会被收录。
    3.如果你看不懂以上内容，建议重新学习小学语文，顺便再去医院检查一下智商。
    """.strip(),
    extra=PluginExtraData(
        author="少姜",
        version="1.0",
        plugin_type=PluginType.NORMAL,
        limits=[BaseBlock(check_type=BlockType.GROUP)],
        menu_type="抽卡相关",
    ).dict(),
)

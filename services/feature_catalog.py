"""Bounded forward-message pages for the bilingual feature dictionary."""

from .feature_aliases import FeatureAliasRegistry

PAGE_CHAR_LIMIT = 1200


def feature_list_messages(registry: FeatureAliasRegistry) -> list[str]:
    messages = [
        "立绘特征组合筛选\n"
        "用法：抽老婆/抽老公 [游戏] -t 特征1 特征2\n"
        "中文名、别名与英文标签可以混合，英文多词用下划线连接。\n"
        "空格、逗号、顿号或 + 分隔；最多 10 项，全部条件必须同时满足。\n"
        "例如：抽老婆 -t 粉发 粉瞳 看向观众 发饰 手持物品 water\n"
        "只匹配已完成标记、主体明确且未被抽走的角色。外貌、服饰、物品等"
        "均查主体区域标签；画面多女/画面多男才查整图人数标签。\n"
        "图上看起来具有某特征，不代表模型已记录该标签；缺少任何一项都会排除。\n"
        "下方按类别列出：中文名 = 英文标签（别名）。"
    ]
    for category, definitions in registry.grouped_definitions().items():
        heading = f"【{category}】"
        page = heading
        for definition in definitions:
            line = f"{definition.name} = {definition.tag}"
            if definition.aliases:
                line += f"（别名：{'、'.join(definition.aliases)}）"
            if len(page) + len(line) + 1 > PAGE_CHAR_LIMIT:
                messages.append(page)
                page = heading + "（续）"
            page += "\n" + line
        messages.append(page)
    messages.append(
        "中英文名称等价，例如 水 / water、持剑 / holding_sword。\n"
        "未列出中文映射的 WD / Camie 英文通用标签也可直接输入。\n"
        "查询不到时可先减少条件，再逐个加回，定位模型未记录的特征。"
    )
    return messages

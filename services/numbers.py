from __future__ import annotations


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

    return "".join(time_parts)


async def parse_chinese_numeral(text):
    """将中文数字转换为阿拉伯数字，支持百位、十位"""
    chinese_numerals = {
        "一": 1,
        "二": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
        "百": 100,
    }
    result = 0

    if "百" in text:
        parts = text.split("百")
        if parts[0]:
            result += chinese_numerals.get(parts[0], 1) * 100
        else:
            result += 100
        # 百位后的部分
        if len(parts) > 1 and parts[1]:
            text = parts[1]
        else:
            text = ""

    if "十" in text:
        parts = text.split("十")
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

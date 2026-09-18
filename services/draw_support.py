from __future__ import annotations

import secrets

from fuzzywuzzy import fuzz

from .painting_name import parse_painting_name


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
    return secrets.choice(choices)


async def is_exact_match(img_name: str, stored_name: str) -> bool:
    """检查图片名是否与存储的名字精确匹配游戏名和角色名部分。"""
    image = parse_painting_name(img_name)
    stored = parse_painting_name(stored_name)
    return bool(image and stored and image.identity == stored.identity)

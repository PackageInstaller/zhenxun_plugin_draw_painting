from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import unicodedata

from .painting_name import parse_painting_name


class DrawRequestError(ValueError):
    """Raised when a draw command has invalid target arguments."""


@dataclass(frozen=True)
class DrawRequest:
    game_name: str | None
    character_name: str | None
    feature_names: tuple[str, ...]


_DRAW_FLAG_PATTERN = re.compile(
    r"(?:^|\s)(?P<flag>-n|-t|--tag|--tags|--feature|--features|--特征)(?=\s|$)",
    re.IGNORECASE,
)
_FEATURE_SEPARATOR_PATTERN = re.compile(r"[\s,，、+＋]+")
MAX_FEATURE_FILTERS = 10


def normalize_draw_name(value: str) -> str:
    """Normalize equivalent Unicode/case/spacing without fuzzy matching."""
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(normalized.split())


def _strip_role_name(value: str, role_name: str) -> str:
    value = value.strip()
    if value.startswith(role_name):
        value = value[len(role_name) :].strip()
    if value.endswith(role_name):
        value = value[: -len(role_name)].strip()
    return value


def parse_draw_request(message_text: str, role_name: str) -> DrawRequest:
    """Parse game, optional character, and combinable feature filters."""
    text = unicodedata.normalize("NFKC", message_text).strip()
    if not text.startswith("抽"):
        raise DrawRequestError("不是有效的抽取指令。")

    body = _strip_role_name(text[1:], role_name)

    flag_matches = list(_DRAW_FLAG_PATTERN.finditer(body))
    game_end = flag_matches[0].start() if flag_matches else len(body)
    game_name = _strip_role_name(body[:game_end], role_name)
    character_name: str | None = None
    feature_names: list[str] = []
    seen_name_flag = False
    for index, match in enumerate(flag_matches):
        value_end = (
            flag_matches[index + 1].start()
            if index + 1 < len(flag_matches)
            else len(body)
        )
        value = body[match.end() : value_end].strip()
        flag = match.group("flag").casefold()
        if flag == "-n":
            if seen_name_flag:
                raise DrawRequestError("只能使用一次 -n 参数。")
            seen_name_flag = True
            if not game_name:
                raise DrawRequestError("使用 -n 时必须先指定游戏名。")
            if not value:
                raise DrawRequestError("-n 后需要填写人物名字。")
            character_name = value
            continue

        if not value:
            raise DrawRequestError(f"{match.group('flag')} 后需要填写特征。")
        feature_names.extend(
            item for item in _FEATURE_SEPARATOR_PATTERN.split(value) if item
        )

    if len(feature_names) > MAX_FEATURE_FILTERS:
        raise DrawRequestError(f"单次最多组合 {MAX_FEATURE_FILTERS} 个特征。")

    return DrawRequest(
        game_name=game_name or None,
        character_name=character_name,
        feature_names=tuple(feature_names),
    )


def image_character_name(image_name: str) -> str | None:
    parsed = parse_painting_name(image_name)
    return parsed.character if parsed else None


def character_prefix(image_name: str) -> str:
    """Return the stable `game_character` identity used by image variants."""
    parsed = parse_painting_name(image_name)
    return parsed.prefix if parsed else Path(image_name).stem


def find_character_variants(folder: str | Path, image_name: str) -> list[str]:
    selected = parse_painting_name(image_name)
    if selected is None:
        return []
    variants: list[str] = []
    for path in Path(folder).iterdir():
        if not path.is_file():
            continue
        parsed = parse_painting_name(path.name)
        if parsed and parsed.extension and parsed.identity == selected.identity:
            variants.append(path.name)
    return sorted(variants, key=str.casefold)


def _name_edit_distance(left: str, right: str) -> int:
    """Levenshtein distance for the final typo-tolerant name fallback."""
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, 1):
        current = [i]
        for j, right_char in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def _typo_name_match(named: list[tuple[str, str]], target: str) -> set[str]:
    if len(target) < 3:
        return set()
    scores: dict[str, tuple[float, int]] = {}
    for name in dict.fromkeys(name for _, name in named):
        compact = "".join(char for char in name if char.isalnum())
        if len(compact) < 3:
            continue
        length = max(len(target), len(compact))
        limit = min(2, length // 3)
        if abs(len(target) - len(compact)) > limit:
            continue
        distance = _name_edit_distance(target, compact)
        if distance <= limit:
            # Same edit distance: keep the user's matching name prefix.
            # 妮菲娅 should prefer 妮菲亚 over 索菲娅.
            prefix = 0
            for left, right in zip(target, compact):
                if left != right:
                    break
                prefix += 1
            scores[name] = (1 - distance / length, prefix)
    if not scores:
        return set()
    best = max(scores.values())
    matches = {name for name, score in scores.items() if score == best}
    if len(matches) > 1:
        labels = sorted(
            {image_character_name(image) for image, name in named if name in matches}
        )
        shown = "、".join(labels[:10])
        suffix = f"（共 {len(labels)} 个候选）" if len(labels) > 10 else ""
        raise DrawRequestError(
            f"找到多个相近人物：{shown}{suffix}。请补充或使用完整名字。"
        )
    return matches


def select_character_images(
    images: list[str],
    character_name: str,
    *,
    excluded_characters: set[str] | None = None,
) -> list[str]:
    """Exact, then partial name, then a uniquely best typo match.

    Only character names participate; game names and variant remarks do not.
    Resolve before excluding claimed characters so exact names never silently
    fall back to a different character when the requested one is unavailable.
    """
    target = normalize_draw_name(character_name)
    named = [
        (image, normalize_draw_name(name))
        for image in images
        if (name := image_character_name(image)) is not None
    ]
    matched = [(image, name) for image, name in named if name == target]
    if not matched:
        # NFKC + casefold above handles fullwidth Latin and letter casing.
        compact = "".join(char for char in target if char.isalnum())
        if not compact:
            return []
        matched = [
            (image, name)
            for image, name in named
            if compact in "".join(char for char in name if char.isalnum())
        ]
        if not matched:
            close_names = _typo_name_match(named, compact)
            matched = [(image, name) for image, name in named if name in close_names]
    excluded = excluded_characters or set()
    return [image for image, name in matched if name not in excluded]

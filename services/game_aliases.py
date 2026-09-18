from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
import threading
import unicodedata

import yaml

MAX_ALIASES_PER_COMMAND = 20
MAX_ALIAS_LENGTH = 100

_write_lock = threading.Lock()


class AliasConfigError(ValueError):
    """Raised when an alias update is unsafe or the config is malformed."""


@dataclass(frozen=True)
class AliasUpdateResult:
    game_name: str
    added: tuple[str, ...]
    duplicates: tuple[str, ...]


def normalize_alias(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    return " ".join(normalized.split()).casefold()


def _clean_alias(value: str) -> str:
    cleaned = " ".join(unicodedata.normalize("NFKC", value).split())
    if not cleaned:
        raise AliasConfigError("别名不能为空。")
    if len(cleaned) > MAX_ALIAS_LENGTH:
        raise AliasConfigError(
            f"别名「{cleaned[:20]}…」过长，最多 {MAX_ALIAS_LENGTH} 个字符。"
        )
    if any(unicodedata.category(char).startswith("C") for char in cleaned):
        raise AliasConfigError(f"别名「{cleaned}」包含不可用的控制字符。")
    return cleaned


def _load_config(config_path: Path) -> dict:
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise AliasConfigError(f"读取别名配置失败：{e}") from e
    if not isinstance(data, dict) or not isinstance(data.get("games"), list):
        raise AliasConfigError("别名配置格式错误：缺少 games 列表。")
    return data


def _identifiers(game: dict) -> list[str]:
    values: list[str] = []
    for key in ("name", "short_name", "en_name"):
        value = game.get(key)
        if isinstance(value, str) and value.strip():
            values.append(value)
    aliases = game.get("aliases") or []
    if not isinstance(aliases, list) or not all(
        isinstance(alias, str) for alias in aliases
    ):
        raise AliasConfigError(
            f"游戏「{game.get('name', '未知')}」的 aliases 必须是字符串列表。"
        )
    values.extend(aliases)
    return values


def _atomic_dump(config_path: Path, data: dict) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=config_path.parent,
        prefix=f".{config_path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as file:
            yaml.safe_dump(
                data,
                file,
                allow_unicode=True,
                sort_keys=False,
                width=120,
            )
            file.flush()
            os.fsync(file.fileno())
        # Verify the complete temporary file before replacing the live config.
        verified = yaml.safe_load(temporary_path.read_text(encoding="utf-8"))
        if not isinstance(verified, dict) or not isinstance(
            verified.get("games"), list
        ):
            raise AliasConfigError("写入后的别名配置校验失败。")
        os.replace(temporary_path, config_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def add_game_aliases(
    config_path: Path,
    game_name: str,
    aliases: list[str],
    *,
    allow_create: bool = False,
) -> AliasUpdateResult:
    """Atomically add unique aliases, rejecting cross-game conflicts."""
    game_name = _clean_alias(game_name)
    if not aliases:
        raise AliasConfigError("至少需要提供一个别名。")
    if len(aliases) > MAX_ALIASES_PER_COMMAND:
        raise AliasConfigError(f"单次最多添加 {MAX_ALIASES_PER_COMMAND} 个别名。")

    cleaned_aliases = [_clean_alias(alias) for alias in aliases]
    with _write_lock:
        data = _load_config(config_path)
        games = data["games"]

        target: dict | None = None
        normalized_game_name = normalize_alias(game_name)
        owners: dict[str, set[str]] = {}
        for game in games:
            if not isinstance(game, dict) or not isinstance(game.get("name"), str):
                raise AliasConfigError("别名配置包含无效的游戏条目。")
            owner = str(game["name"])
            if normalize_alias(owner) == normalized_game_name:
                if target is not None:
                    raise AliasConfigError(f"配置中存在重复游戏名「{game_name}」。")
                target = game
            for identifier in _identifiers(game):
                owners.setdefault(normalize_alias(identifier), set()).add(owner)

        if target is None:
            conflicting_owners = owners.get(normalized_game_name, set())
            if conflicting_owners:
                owners_text = "、".join(sorted(conflicting_owners))
                raise AliasConfigError(
                    f"「{game_name}」已是「{owners_text}」的别名，不能作为原游戏名。"
                )
            if not allow_create:
                raise AliasConfigError(f"别名列表中不存在原游戏「{game_name}」。")
            target = {"name": game_name, "aliases": []}
            games.append(target)
            owners.setdefault(normalized_game_name, set()).add(game_name)

        canonical_name = str(target["name"])
        target_aliases = target.get("aliases") or []
        if not isinstance(target_aliases, list):
            raise AliasConfigError(f"游戏「{canonical_name}」的 aliases 格式错误。")

        added: list[str] = []
        duplicates: list[str] = []
        seen_in_request: set[str] = set()
        conflicts: list[str] = []
        for alias in cleaned_aliases:
            normalized = normalize_alias(alias)
            if normalized in seen_in_request:
                duplicates.append(alias)
                continue
            seen_in_request.add(normalized)

            identifier_owners = owners.get(normalized, set())
            other_owners = identifier_owners - {canonical_name}
            if other_owners:
                conflicts.append(
                    f"「{alias}」已属于「{'、'.join(sorted(other_owners))}」"
                )
                continue
            if identifier_owners or normalized == normalized_game_name:
                duplicates.append(alias)
                continue

            added.append(alias)
            owners.setdefault(normalized, set()).add(canonical_name)

        if conflicts:
            raise AliasConfigError("别名冲突：" + "；".join(conflicts))
        if not added:
            return AliasUpdateResult(
                game_name=canonical_name,
                added=(),
                duplicates=tuple(duplicates),
            )

        target["aliases"] = [*target_aliases, *added]
        _atomic_dump(config_path, data)
        return AliasUpdateResult(
            game_name=canonical_name,
            added=tuple(added),
            duplicates=tuple(duplicates),
        )

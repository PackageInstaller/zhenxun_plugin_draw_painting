from __future__ import annotations

import csv
from dataclasses import dataclass
from difflib import get_close_matches
from functools import lru_cache
import json
from pathlib import Path
import re
import unicodedata

import yaml

from ..config import paths
from .model import ModelManager
from .vision.assets import CAMIE_TAGS

FEATURE_ALIASES_PATH = Path(paths.PLUGIN_DIR) / "config" / "feature_aliases.yaml"
_RAW_TAG_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_+.()\-]*$")


class FeatureAliasError(ValueError):
    """Raised when a requested feature name cannot be resolved safely."""


@dataclass(frozen=True, slots=True)
class FeatureDefinition:
    tag: str
    name: str
    aliases: tuple[str, ...]
    category: str


@dataclass(frozen=True, slots=True)
class ResolvedFeatures:
    tags: tuple[str, ...]
    labels: tuple[str, ...]


def normalize_feature_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(normalized.split())


class FeatureAliasRegistry:
    def __init__(self, config_path: Path = FEATURE_ALIASES_PATH) -> None:
        self.config_path = config_path
        self._definitions: tuple[FeatureDefinition, ...] | None = None
        self._aliases: dict[str, FeatureDefinition] = {}
        self._by_tag: dict[str, FeatureDefinition] = {}

    def _load(self) -> None:
        if self._definitions is not None:
            return
        try:
            data = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise FeatureAliasError(f"读取特征映射失败：{exc}") from exc
        categories = data.get("categories") if isinstance(data, dict) else None
        if not isinstance(categories, dict):
            raise FeatureAliasError("特征映射缺少 categories 配置。")

        definitions: list[FeatureDefinition] = []
        aliases: dict[str, FeatureDefinition] = {}
        by_tag: dict[str, FeatureDefinition] = {}
        for raw_category, entries in categories.items():
            if not isinstance(raw_category, str) or not isinstance(entries, list):
                raise FeatureAliasError("特征映射分类格式错误。")
            for entry in entries:
                if not isinstance(entry, dict):
                    raise FeatureAliasError(f"「{raw_category}」包含无效特征。")
                tag = entry.get("tag")
                name = entry.get("name")
                raw_aliases = entry.get("aliases", [])
                if (
                    not isinstance(tag, str)
                    or not _RAW_TAG_PATTERN.fullmatch(tag)
                    or not isinstance(name, str)
                    or not name.strip()
                    or not isinstance(raw_aliases, list)
                    or not all(isinstance(alias, str) for alias in raw_aliases)
                ):
                    raise FeatureAliasError(f"「{raw_category}」包含无效特征映射。")
                if tag in by_tag:
                    raise FeatureAliasError(f"特征标签「{tag}」重复定义。")
                definition = FeatureDefinition(
                    tag=tag,
                    name=name.strip(),
                    aliases=tuple(alias.strip() for alias in raw_aliases),
                    category=raw_category,
                )
                by_tag[tag] = definition
                definitions.append(definition)
                for alias in (tag, definition.name, *definition.aliases):
                    normalized = normalize_feature_name(alias)
                    owner = aliases.get(normalized)
                    if owner is not None and owner.tag != tag:
                        raise FeatureAliasError(
                            f"特征别名「{alias}」同时属于 {owner.tag} 和 {tag}。"
                        )
                    aliases[normalized] = definition

        self._definitions = tuple(definitions)
        self._aliases = aliases
        self._by_tag = by_tag

    @staticmethod
    def _known_model_tags() -> frozenset[str]:
        tags_path = ModelManager.tags_path()
        files = (tags_path, CAMIE_TAGS.path)
        signatures = tuple(
            (str(path), path.stat().st_mtime_ns if path.is_file() else 0)
            for path in files
        )
        return FeatureAliasRegistry._read_model_tags(signatures)

    @staticmethod
    @lru_cache(maxsize=2)
    def _read_model_tags(signatures: tuple[tuple[str, int], ...]) -> frozenset[str]:
        tags: set[str] = set()
        for filename, timestamp in signatures:
            if not timestamp:
                continue
            path = Path(filename)
            if path.suffix == ".csv":
                with path.open(encoding="utf-8", newline="") as source:
                    tags.update(
                        row["name"]
                        for row in csv.DictReader(source)
                        if row["category"] == "0"
                    )
            else:
                data = json.loads(path.read_text(encoding="utf-8"))
                mapping = data["dataset_info"]["tag_mapping"]["tag_to_category"]
                tags.update(
                    tag for tag, category in mapping.items() if category == "general"
                )
        return frozenset(tags)

    def resolve(self, value: str) -> FeatureDefinition:
        self._load()
        normalized = normalize_feature_name(value)
        if not normalized:
            raise FeatureAliasError("特征名不能为空。")
        if definition := self._aliases.get(normalized):
            return definition

        raw_tag = unicodedata.normalize("NFKC", value).strip().casefold()
        known_tags = self._known_model_tags()
        if raw_tag in known_tags:
            return FeatureDefinition(raw_tag, raw_tag, (), "原始标签")

        suggestions = get_close_matches(normalized, self._aliases, n=3, cutoff=0.55)
        if suggestions:
            suggestion_labels = "、".join(
                dict.fromkeys(self._aliases[item].name for item in suggestions)
            )
            raise FeatureAliasError(
                f"未知特征「{value}」，你可能想输入：{suggestion_labels}。"
            )
        raise FeatureAliasError(
            f"未知特征「{value}」，发送“立绘特征列表”查看可用名称。"
        )

    def resolve_many(self, values: tuple[str, ...]) -> ResolvedFeatures:
        resolved: list[FeatureDefinition] = []
        seen: set[str] = set()
        for value in values:
            definition = self.resolve(value)
            if definition.tag in seen:
                continue
            seen.add(definition.tag)
            resolved.append(definition)
        return ResolvedFeatures(
            tags=tuple(item.tag for item in resolved),
            labels=tuple(item.name for item in resolved),
        )

    def grouped_names(self) -> dict[str, tuple[str, ...]]:
        self._load()
        grouped: dict[str, list[str]] = {}
        for definition in self._definitions or ():
            grouped.setdefault(definition.category, []).append(definition.name)
        return {category: tuple(names) for category, names in grouped.items()}

    def display_name(self, tag: str) -> str:
        """Use the same Chinese vocabulary for queries and diagnostic output."""
        self._load()
        definition = self._by_tag.get(tag)
        return definition.name if definition is not None else tag

    def grouped_definitions(self) -> dict[str, tuple[FeatureDefinition, ...]]:
        self._load()
        grouped: dict[str, list[FeatureDefinition]] = {}
        for definition in self._definitions or ():
            grouped.setdefault(definition.category, []).append(definition)
        return {category: tuple(items) for category, items in grouped.items()}

    def validate_model_tags(self) -> tuple[str, ...]:
        self._load()
        known_tags = self._known_model_tags()
        if not known_tags:
            return ()
        return tuple(
            definition.tag
            for definition in self._definitions or ()
            if definition.tag not in known_tags
        )


feature_alias_registry = FeatureAliasRegistry()


__all__ = [
    "FeatureAliasError",
    "FeatureAliasRegistry",
    "FeatureDefinition",
    "ResolvedFeatures",
    "feature_alias_registry",
]

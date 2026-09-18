"""Compatibility facade for the painting package subsystem."""

from .package_archive import (
    PaintingArchive,
    build_painting_archive,
    count_game_paintings,
    list_game_prefixes,
    normalize_game_name,
    resolve_game_prefix,
    safe_archive_name,
)
from .package_cache import CachedPackage, PaintingPackageStore, SharedPackage
from .package_storage import PACKAGE_TTL_SECONDS, PaintingPackageStorage

__all__ = [
    "PACKAGE_TTL_SECONDS",
    "CachedPackage",
    "PaintingArchive",
    "PaintingPackageStorage",
    "PaintingPackageStore",
    "SharedPackage",
    "build_painting_archive",
    "count_game_paintings",
    "list_game_prefixes",
    "normalize_game_name",
    "resolve_game_prefix",
    "safe_archive_name",
]

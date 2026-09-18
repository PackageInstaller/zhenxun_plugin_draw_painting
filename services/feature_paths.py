"""Portable keys for library images, independent of installation drive/root."""

import ntpath
import os
from pathlib import Path

LIBRARY_KEYS = {"wives": "wives", "husbands": "husbands", "others": "others"}


def normalize_image_path(path: str | Path) -> str:
    value = os.fspath(path).replace("\\", "/")
    parent, name = value.rsplit("/", 1) if "/" in value else ("", value)
    library = parent.rsplit("/", 1)[-1].casefold()
    if library in LIBRARY_KEYS and name not in ("", ".", ".."):
        return f"{library}/{name.casefold()}"
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def is_absolute_key(value: str) -> bool:
    return ntpath.isabs(value) or os.path.isabs(value)

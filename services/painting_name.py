"""Filename grammar: game_character_variant_remark1_remark2_....png."""

from dataclasses import dataclass
from pathlib import Path

IMAGE_SUFFIXES = frozenset((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"))


@dataclass(frozen=True, slots=True)
class PaintingName:
    game: str
    character: str
    suffix: tuple[str, ...]
    extension: str = ""

    @property
    def identity(self) -> tuple[str, str]:
        return self.game.casefold(), self.character.casefold()

    @property
    def prefix(self) -> str:
        return f"{self.game}_{self.character}"

    @property
    def stem(self) -> str:
        return "_".join((self.game, self.character, *self.suffix))


def parse_painting_name(value: str) -> PaintingName | None:
    name = Path(value).name
    extension = Path(name).suffix
    if extension.casefold() in IMAGE_SUFFIXES:
        name = name[: -len(extension)]
    else:
        extension = ""
    parts = name.split("_")
    if len(parts) < 2 or not parts[0].strip() or not parts[1].strip():
        return None
    return PaintingName(parts[0], parts[1], tuple(parts[2:]), extension)

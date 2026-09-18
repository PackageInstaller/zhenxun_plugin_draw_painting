"""Shared display policy, independent from inference and archival thresholds."""

import math

from PIL import ImageFont

from .feature_aliases import feature_alias_registry

DISPLAY_THRESHOLD = 0.5


def display_tags(tags: dict[str, float]) -> dict[str, float]:
    """All accepted features scoring at least 50%, ordered without a top-N cap."""
    return dict(
        sorted(
            (
                (tag, score)
                for tag, score in tags.items()
                if math.isfinite(score) and score >= DISPLAY_THRESHOLD
            ),
            key=lambda item: (-item[1], item[0]),
        )
    )


def feature_labels(tags: dict[str, float]) -> list[str]:
    return [
        f"{feature_alias_registry.display_name(tag)} {score:.1%}"
        for tag, score in display_tags(tags).items()
    ]


def wrap_text(text: str, font: ImageFont.FreeTypeFont, width: int) -> list[str]:
    """Wrap at measured glyph widths; never replace feature names with ellipses."""
    if width <= 0:
        raise ValueError("Text width must be positive")
    lines, current = [], ""
    for char in text:
        if char == "\n" or (current and font.getlength(current + char) > width):
            lines.append(current)
            current = "" if char == "\n" else char
        else:
            current += char
    if current:
        lines.append(current)
    return lines

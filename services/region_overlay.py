"""Display the visible instance shape and the exact per-face model input."""

import math

from PIL import ImageDraw

PERSON_COLORS = (
    "#55d9ed",
    "#ffbe64",
    "#d7a2ff",
    "#8ee58a",
    "#ff8fba",
    "#9cb8ff",
    "#e9e778",
    "#e6b49c",
)
GENDERS = {"female": "女性", "male": "男性", "unknown": "未确认"}


def person_color(index: int) -> str:
    return PERSON_COLORS[index % len(PERSON_COLORS)]


def map_region(coords, source_size, image_rect):
    if not isinstance(coords, tuple | list) or len(coords) != 4:
        return None
    if any(not isinstance(v, int | float) or not math.isfinite(v) for v in coords):
        return None
    width, height = source_size
    x, y, right, bottom = image_rect
    if width <= 0 or height <= 0:
        return None
    rect = tuple(
        offset + round(max(0, min(value, limit)) / limit * (extent - 1))
        for value, limit, offset, extent in zip(
            coords,
            (width, height, width, height),
            (x, y, x, y),
            (right - x, bottom - y, right - x, bottom - y),
            strict=True,
        )
    )
    return rect if rect[2] > rect[0] and rect[3] > rect[1] else None


def map_outline(outline, source_size, image_rect):
    """Map source-image contours while rejecting malformed diagnostic data."""
    if not isinstance(outline, tuple | list):
        return []
    width, height = source_size
    x, y, right, bottom = image_rect
    display_width, display_height = right - x, bottom - y
    if width <= 0 or height <= 0 or display_width <= 0 or display_height <= 0:
        return []

    result = []
    for contour in outline:
        if not isinstance(contour, tuple | list) or len(contour) < 3:
            continue
        mapped = []
        for point in contour:
            if not isinstance(point, tuple | list) or len(point) != 2:
                mapped = []
                break
            px, py = point
            if any(
                not isinstance(value, int | float) or not math.isfinite(value)
                for value in (px, py)
            ):
                mapped = []
                break
            mapped.append(
                (
                    x + round(max(0, min(px, width)) / width * (display_width - 1)),
                    y
                    + round(
                        max(0, min(py, height)) / height * (display_height - 1)
                    ),
                )
            )
        if len(set(mapped)) >= 3:
            result.append(mapped)
    return result


def dashed_rectangle(draw: ImageDraw.ImageDraw, rect, color: str):
    x0, y0, x1, y1 = rect
    for stroke, thickness in (("#151c2b", 6), (color, 3)):
        for x in range(x0, x1, 20):
            for y in (y0, y1):
                draw.line((x, y, min(x + 12, x1), y), fill=stroke, width=thickness)
        for y in range(y0, y1, 20):
            for x in (x0, x1):
                draw.line((x, y, x, min(y + 12, y1)), fill=stroke, width=thickness)


def draw_instance_outline(draw: ImageDraw.ImageDraw, contours, color: str):
    """Draw a readable instance contour without implying hidden-body completion."""
    for contour in contours:
        closed = [*contour, contour[0]]
        draw.line(closed, fill="#151c2b", width=7, joint="curve")
        draw.line(closed, fill=color, width=4, joint="curve")


def draw_person_regions(draw, people, source_size, image_rect, font):
    """Match visible instances, final model inputs and faces by numbered color."""

    def label(text, rect, color, *, inside=False):
        width = int(font.getlength(text)) + 12
        x = max(image_rect[0], min(rect[0], image_rect[2] - width))
        y = rect[1] + 3 if inside else rect[1] - 32
        y = max(image_rect[1], min(y, image_rect[3] - 32))
        draw.rectangle((x, y, x + width, y + 31), fill="#151c2b")
        draw.text((x + 5, y), text, font=font, fill=color)

    # Outlines are the model's visible pixels, not a guess at occluded anatomy.
    # Draw them first so final input bounds and face anchors remain legible.
    for index, person in enumerate(people):
        outline = person.get("instance_outline") or person.get("context_outline")
        contours = map_outline(outline, source_size, image_rect)
        if contours:
            draw_instance_outline(draw, contours, person_color(index))

    for index, person in enumerate(people):
        rect = map_region(person.get("context_box"), source_size, image_rect)
        if rect:
            color = person_color(index)
            dashed_rectangle(draw, rect, color)
            label(f"#{index+1} 模型输入边界", rect, color, inside=True)
    for index, person in enumerate(people):
        rect = map_region(person.get("box"), source_size, image_rect)
        if rect:
            color = person_color(index)
            draw.rectangle(rect, outline=color, width=4)
            label(f"#{index+1} {GENDERS[person['gender']]}·人脸", rect, color)

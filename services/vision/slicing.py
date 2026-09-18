"""Bounded original-resolution person recovery; no identity or role ranking."""

from collections.abc import Callable
from dataclasses import dataclass, replace

from PIL import Image

from .subjects import PersonBox, intersection

SLICING_REVISION = 1
MAX_SLICE_PASSES = 36


@dataclass(frozen=True)
class SliceHit:
    box: PersonBox
    source: tuple[int, int, int, int]
    clipped: bool = False


def starts(length: int, size: int) -> list[int]:
    if length <= size:
        return [0]
    values = list(range(0, length - size + 1, size * 3 // 4))
    if values[-1] != length - size:
        values.append(length - size)
    return values


def merge_hits(hits: list[SliceHit]) -> list[PersonBox]:
    """Full boxes anchor duplicates; never certify a tile-edge-only fragment."""
    groups: list[list[SliceHit]] = []
    for hit in sorted(hits, key=lambda h: (h.clipped, -h.box.confidence)):
        matched = False
        for group in groups:
            anchor = group[0].box
            overlap = intersection(hit.box, anchor)
            iou = overlap / max(1, hit.box.area + anchor.area - overlap)
            containment = overlap / max(1, hit.box.area)
            if iou >= 0.5 or (hit.clipped and containment >= 0.85):
                group.append(hit)
                matched = True
                break
        if not matched:
            groups.append([hit])
    result = []
    for group in groups:
        complete = [hit for hit in group if not hit.clipped]
        if not complete:
            continue
        anchor = complete[0].box
        support = len({hit.source for hit in complete if hit.box.confidence >= 0.5})
        x0, y0, x1, y1 = anchor.xyxy
        verified = (
            support >= 2 and anchor.confidence >= 0.55 and min(x1 - x0, y1 - y0) >= 64
        )
        result.append(replace(anchor, small_verified=verified or anchor.small_verified))
    return result


def detect_with_slices(
    image: Image.Image, detect: Callable[[Image.Image], list[PersonBox]]
) -> tuple[list[PersonBox], dict]:
    full = (0, 0, image.width, image.height)
    original = detect(image)
    info = {"revision": SLICING_REVISION, "slice_passes": 0, "budget_exhausted": False}
    # Easy images retain the old single-pass path. Small images have no lost
    # high-resolution detail for this stage to recover.
    if max(image.size) < 1024 or any(
        box.confidence >= 0.55 and box.area >= image.width * image.height * 0.08
        for box in original
    ):
        return original, info
    hits = [SliceHit(box, full) for box in original]
    visited = {full}
    bounds = image.info.get("painting_content_box", full)

    def scan(region):
        if region in visited:
            return
        if info["slice_passes"] >= MAX_SLICE_PASSES:
            info["budget_exhausted"] = True
            return
        visited.add(region)
        info["slice_passes"] += 1
        x0, y0, x1, y1 = region
        for box in detect(image.crop(region)):
            a, b, c, d = box.xyxy
            # Only artificial slice edges are suspect; a true canvas edge is OK.
            clipped = (
                (x0 > bounds[0] and a <= 4)
                or (y0 > bounds[1] and b <= 4)
                or (x1 < bounds[2] and c >= x1 - x0 - 4)
                or (y1 < bounds[3] and d >= y1 - y0 - 4)
            )
            hits.append(
                SliceHit(
                    PersonBox((a + x0, b + y0, c + x0, d + y0), box.confidence),
                    region,
                    clipped,
                )
            )

    # Alpha only removes empty margins, not background objects or other people.
    scan(bounds)
    x0, y0, x1, y1 = bounds
    for y in starts(y1 - y0, 1024):
        for x in starts(x1 - x0, 1024):
            scan((x0 + x, y0 + y, min(x0 + x + 1024, x1), min(y0 + y + 1024, y1)))

    # Centered second views restore cut bodies and verify small candidates.
    # Native-resolution 640+ crops use the same fixed-size detector session.
    candidates = sorted(hits, key=lambda hit: -hit.box.confidence)[:10]
    for hit in candidates:
        a, b, c, d = hit.box.xyxy
        side = min(max(640, int(max(c - a, d - b) * 1.6)), max(image.size))
        left = max(0, min((a + c - side) // 2, image.width - side))
        top = max(0, min((b + d - side) // 2, image.height - side))
        scan((left, top, min(image.width, left + side), min(image.height, top + side)))
    return merge_hits(hits), info

"""Rotation-aware face anchors for dense/group-image diagnostics.

The public v1.4_s model recommends confidence 0.307. Scanning rotated,
overlapping regions improves recall without relaxing the person classifier.
"""

from dataclasses import dataclass
import math
import re

from PIL import Image

from .assets import FACE_MODEL
from .subjects import AnimeBoxDetector, PersonBox, intersection
from .types import MULTIPLE_TAGS


@dataclass(frozen=True, slots=True)
class FaceBox:
    xyxy: tuple[int, int, int, int]
    confidence: float
    rotation: int
    support: int = 1
    clipped: bool = False


def unrotate_box(
    box: tuple[int, int, int, int], angle: int, width: int, height: int
) -> tuple[int, int, int, int]:
    """Map a PIL right-angle rotated box back to the unrotated source."""
    x0, y0, x1, y1 = box
    if angle == 90:
        return width - y1, x0, width - y0, x1
    if angle == 180:
        return width - x1, height - y1, width - x0, height - y0
    if angle == 270:
        return y0, height - x1, y1, height - x0
    if angle != 0:
        raise ValueError("Only right-angle rotations are supported")
    return box


def merge_faces(candidates: list[FaceBox]) -> list[FaceBox]:
    """Cross-view NMS. Repeated detections contribute support, not headcount."""
    merged: list[FaceBox] = []
    # A cut-off half-face can score higher than the full face. Let intact
    # anchors represent clusters so opposite tile halves join the same person.
    for face in sorted(candidates, key=lambda item: (item.clipped, -item.confidence)):
        current = PersonBox(face.xyxy, face.confidence)
        for index, old in enumerate(merged):
            previous = PersonBox(old.xyxy, old.confidence)
            overlap = intersection(current, previous)
            iou = overlap / max(1, current.area + previous.area - overlap)
            coverage = overlap / max(1, min(current.area, previous.area))
            ratio = min(current.area, previous.area) / max(
                1, current.area, previous.area
            )
            if iou >= 0.4 or (coverage >= 0.8 and ratio >= 0.3):
                merged[index] = FaceBox(
                    old.xyxy,
                    old.confidence,
                    old.rotation,
                    old.support + face.support,
                    old.clipped,
                )
                break
        else:
            merged.append(face)
    # A marginal detection needs a second view; confident ones need not repeat.
    accepted = [
        face
        for face in merged
        if not face.clipped and (face.confidence >= 0.5 or face.support >= 2)
    ]
    return sorted(accepted, key=lambda face: (face.xyxy[1], face.xyxy[0]))


def scan_regions(width: int, height: int) -> list[tuple[int, int, int, int]]:
    regions = [(0, 0, width, height)]
    if max(width, height) / min(width, height) < 1.5:
        return regions
    # Overlapping near-square windows avoid crushing faces in panoramic images.
    side = int(min(width, height) * 1.2)
    long_side = max(width, height)
    count = min(4, math.ceil((long_side - side) / (side * 0.75)) + 1)
    for index in range(count):
        offset = round(index * (long_side - side) / max(1, count - 1))
        regions.append(
            (offset, 0, offset + side, height)
            if width >= height
            else (0, offset, width, offset + side)
        )
    return regions


class AnimeFaceDetector:
    def __init__(self) -> None:
        self.detector = AnimeBoxDetector(FACE_MODEL.path, 0.25, 0.307, 0.00008)

    def _scan(
        self, image: Image.Image, regions: list[tuple[int, int, int, int]]
    ) -> list[FaceBox]:
        candidates = []
        for region in regions:
            tile = image.crop(region)
            for angle in (0, 90, 180, 270):
                rotated = tile.rotate(angle, expand=True)
                for box in self.detector.detect(rotated):
                    x0, y0, x1, y1 = unrotate_box(box.xyxy, angle, *tile.size)
                    if min(x1 - x0, y1 - y0) < 12:
                        continue
                    candidates.append(
                        FaceBox(
                            (
                                x0 + region[0],
                                y0 + region[1],
                                x1 + region[0],
                                y1 + region[1],
                            ),
                            box.confidence,
                            angle,
                            clipped=tile_clips_face(
                                (x0, y0, x1, y1), region, image.size
                            ),
                        )
                    )
        return candidates

    def detect(
        self, image: Image.Image, *, scene_tags: dict[str, float] | None = None
    ) -> list[FaceBox]:
        regions = scan_regions(image.width, image.height)
        candidates = self._scan(image, regions)
        faces = merge_faces(candidates)
        if needs_detail_scan(image.size, faces, scene_tags or {}):
            extra = [
                region
                for region in detail_regions(*image.size)
                if region not in regions
            ]
            candidates.extend(self._scan(image, extra))
        return merge_faces(candidates)


def tile_clips_face(box, region, image_size) -> bool:
    """Only artificial crop edges count, never the original image boundary."""
    x0, y0, x1, y1 = box
    left, top, right, bottom = region
    width, height = right - left, bottom - top
    margin = max(2, round(min(width, height) * 0.01))
    return (
        (left > 0 and x0 <= margin)
        or (top > 0 and y0 <= margin)
        or (right < image_size[0] and x1 >= width - margin)
        or (bottom < image_size[1] and y1 >= height - margin)
    )


def needs_detail_scan(size, faces: list[FaceBox], tags: dict[str, float]) -> bool:
    # Scene tags request another spatial search, never manufacture people.
    hinted_minimum = 0
    for tag, score in tags.items():
        if score < 0.5:
            continue
        if tag in MULTIPLE_TAGS:
            hinted_minimum = max(hinted_minimum, 2)
        if match := re.fullmatch(r"(\d+)\+?(?:girls|boys)", tag):
            hinted_minimum = max(hinted_minimum, int(match[1]))
    if len(faces) < hinted_minimum:
        return True
    if max(size) < 1600 or len(faces) > 1:
        return False
    largest = max(
        (
            (face.xyxy[2] - face.xyxy[0]) * (face.xyxy[3] - face.xyxy[1])
            for face in faces
        ),
        default=0,
    )
    return largest < size[0] * size[1] * 0.02


def detail_regions(width: int, height: int) -> list[tuple[int, int, int, int]]:
    """Nine half-size windows with 50% overlap, including square illustrations."""
    tile_width, tile_height = max(1, (width + 1) // 2), max(1, (height + 1) // 2)
    return list(
        dict.fromkeys(
            (x, y, x + tile_width, y + tile_height)
            for y in (0, (height - tile_height) // 2, height - tile_height)
            for x in (0, (width - tile_width) // 2, width - tile_width)
        )
    )


def face_context(
    image: Image.Image,
    face: FaceBox,
    *,
    tight: bool = False,
    neighbors: list[FaceBox] | None = None,
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Orient the anchor upright and include its hair/torso, not only eyes."""
    rotated = image.rotate(face.rotation, expand=True)
    fx0, fy0, fx1, fy1 = unrotate_box(face.xyxy, (-face.rotation) % 360, *rotated.size)
    width, height = fx1 - fx0, fy1 - fy0
    side_margin, above, below = (0.25, 0.35, 0.65) if tight else (0.55, 0.65, 1.3)
    region = (
        max(0, round(fx0 - width * side_margin)),
        max(0, round(fy0 - height * above)),
        min(rotated.width, round(fx1 + width * side_margin)),
        min(rotated.height, round(fy1 + height * below)),
    )
    # Keep the complete target face, but cut expansion at the gap between it
    # and adjacent faces. Work in the same upright coordinates as the crop.
    left, top, right, bottom = region
    for other in neighbors or []:
        if other is face or other.xyxy == face.xyxy:
            continue
        ox0, oy0, ox1, oy1 = unrotate_box(
            other.xyxy, (-face.rotation) % 360, *rotated.size
        )
        if ox1 <= left or ox0 >= right or oy1 <= top or oy0 >= bottom:
            continue
        options = []
        if ox0 >= fx1:
            options.append((left, top, min(right, (fx1 + ox0) // 2), bottom))
        if ox1 <= fx0:
            options.append((max(left, (ox1 + fx0) // 2), top, right, bottom))
        if oy0 >= fy1:
            options.append((left, top, right, min(bottom, (fy1 + oy0) // 2)))
        if oy1 <= fy0:
            options.append((left, max(top, (oy1 + fy0) // 2), right, bottom))
        if options:
            left, top, right, bottom = max(
                options, key=lambda r: (r[2] - r[0]) * (r[3] - r[1])
            )
    region = (left, top, right, bottom)
    return rotated.crop(region), unrotate_box(region, face.rotation, *image.size)

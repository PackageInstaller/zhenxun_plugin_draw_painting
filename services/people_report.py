"""Compact image-only group report: face anchors, counts and per-person traits."""

from io import BytesIO
import math
import re

from PIL import Image, ImageDraw

from .feature_presentation import feature_labels, wrap_text
from .region_overlay import draw_person_regions, person_color
from .vision.types import FEMALE_TAGS, MALE_TAGS, MULTIPLE_TAGS, TagPrediction

_GENDERS = {"female": "女性", "male": "男性", "unknown": "未确认"}


def person_status_text(person: dict) -> str:
    return {
        "multiple_in_crop": "裁剪仍含多人信号，无法归属",
        "neighbor_overlap": "邻近人脸重叠，无法安全隔离",
        "inference_failed": "局部推理失败，未确认",
        "gender_uncertain": "单人性别标签证据不足或冲突",
    }.get(person.get("status"), "暂无 ≥50% 的可信区域特征")


def is_count_tag(tag: str) -> bool:
    return tag in MALE_TAGS | FEMALE_TAGS | MULTIPLE_TAGS | {"solo"} or bool(
        re.fullmatch(r"\d+\+?(?:girls?|boys?)", tag)
    )


def person_region_lines(person: dict) -> list[str]:
    """Describe what was actually passed to the per-person taggers."""
    lines = []
    if region := person.get("context_box"):
        lines.append(f"模型输入边界：{tuple(region)}")

    mode = person.get("context_mode")
    has_reference_outline = bool(
        person.get("instance_outline") or person.get("context_outline")
    )
    if mode == "instance_mask":
        lines.append("模型输入：可见身体掩码（轮廓外置白）")
    elif mode == "instance_mask_upper":
        lines.append("模型输入：上半身掩码复核（交叠手势/邻人肢体置白）")
    elif mode == "instance_mask_face":
        lines.append("模型输入：头带掩码复核（肩上手/邻人肢体置白）")
    elif mode == "face_crop_tight" or person.get("tight_crop"):
        detail = "彩色轮廓仅作参考" if has_reference_outline else "未取得可用实例掩码"
        lines.append(f"模型输入：人脸紧裁回退（{detail}）")
    elif mode == "face_crop" or mode is None:
        lines.append("模型输入：人脸邻域矩形回退裁剪")
    else:
        lines.append(f"模型输入：{mode}")

    confidence = person.get("instance_confidence")
    if (
        isinstance(confidence, int | float)
        and not isinstance(confidence, bool)
        and math.isfinite(confidence)
    ):
        lines.append(f"实例候选置信度：{confidence:.1%}")
    return lines


def region_explanation_notes(people: list[dict]) -> list[str]:
    """Explain visible masks and rectangular fallbacks without conflating them."""
    modes = {person.get("context_mode") for person in people}
    has_outline = any(
        person.get("instance_outline") or person.get("context_outline")
        for person in people
    )
    notes = [
        "虚线框始终是最终送入特征模型的图像边界；卡片标明掩码或回退裁剪。"
    ]
    if has_outline:
        notes.append(
            "彩色轮廓是实例分割得到的画面内可见身体（含头发与服饰），"
            "不会补全遮挡或画外部分。"
        )
    if "instance_mask" in modes:
        notes.append(
            "可见身体掩码输入保留轮廓内像素，并将同一虚线框内的其余像素置白。"
        )
    if modes & {"face_crop", "face_crop_tight", None}:
        notes.append(
            "回退裁剪表示最终使用人脸邻域矩形；若仍显示彩色轮廓，"
            "该轮廓只供查看，未用于最终特征推理。"
        )
    return notes


def render_people_report(
    image: Image.Image, prediction: TagPrediction, font, heading
) -> bytes:
    census = prediction.analysis["census"]
    people = census["people"]
    width, gap, columns = 1600, 24, 4
    image.thumbnail((width - 2 * gap, 1100), Image.Resampling.LANCZOS)
    rows = (len(people) + columns - 1) // columns
    image_y, cards_y = 158, 158 + image.height + gap
    card_width = (width - gap * 2) // columns
    card_lines = []
    for person in people:
        labels = feature_labels(
            {
                tag: score
                for tag, score in person["tags"].items()
                if not is_count_tag(tag)
            }
        )
        texts = (
            [f"区域特征 ≥50% · {len(labels)} 项", *labels]
            if labels
            else [person_status_text(person), "不将整图特征分配给此人"]
        )
        texts = [*person_region_lines(person), *texts]
        card_lines.append(
            [line for text in texts for line in wrap_text(text, font, card_width - 36)]
        )
    row_heights = [
        110
        + max(len(lines) for lines in card_lines[row * columns : (row + 1) * columns])
        * 28
        for row in range(rows)
    ]
    row_tops = []
    bottom = cards_y
    for row_height in row_heights:
        row_tops.append(bottom)
        bottom += row_height
    scene_labels = feature_labels(
        {
            tag: score
            for tag, score in prediction.general_tags.items()
            if not is_count_tag(tag)
        }
    )
    notes = [
        "人数按去重人脸估计；背脸、遮挡或极小人物仍可能漏检。性别与特征为模型估计。",
        *region_explanation_notes(people),
    ]
    if census["detected"] > census["analyzed"]:
        notes.append(
            f"本次仅逐人分析前 {census['analyzed']} 张脸，其他候选计入未确认。"
        )
    if prediction.analysis.get("subject_source") == "co_primary_census":
        notes.append(
            "多人图无唯一主体：已确认所有主角同为同一性别（双主体）。"
        )
    salience = prediction.analysis.get("subject_salience")
    if isinstance(salience, dict):
        face_index = salience.get("selected_face_index")
        suggested = salience.get("suggested_face_index")
        if salience.get("status") == "candidate" and face_index is not None:
            notes.append(
                f"主体评分选定 {face_index + 1} 号人物为画面焦点"
                "（依据尺度、构图位置、清晰度、色彩与明暗对比，仅供参考）。"
            )
        elif suggested is not None:
            margin = salience.get("margin")
            margin_text = (
                f"，与次名差距 {margin:.2f}（未达 0.14 门槛）"
                if isinstance(margin, int | float)
                else ""
            )
            notes.append(
                f"多人图未指定主体：评分最接近的是 {suggested + 1} 号人物"
                f"{margin_text}；构图与外观证据仍不够分离"
            )
        else:
            notes.append("各人物的构图与外观证据接近，未指定画面主体。")
    if prediction.analysis.get("first_frame_only"):
        notes.append("动态图仅分析首帧。")
    note_lines = [
        line for text in notes for line in wrap_text(text, font, width - gap * 2)
    ]
    notes_y = bottom + 8
    height = notes_y + len(note_lines) * 32 + gap
    canvas = Image.new("RGB", (width, height), "#151c2b")
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 20), "图片特征查询 · 逐人识别", font=heading, fill="#eff5ff")
    summary = (
        f"检测到 {census['detected']} 个人脸候选   |   "
        f"女性 {census['female']}   男性 {census['male']}   未确认 {census['unknown']}"
    )
    draw.text((gap, 65), summary, font=heading, fill="#68e4ba")
    draw.text(
        (gap, 110),
        "同色同编号｜彩色轮廓：可见身体｜实线：人脸｜虚线：实际模型输入边界",
        font=font,
        fill="#c8d3e7",
    )
    image_x = (width - image.width) // 2
    canvas.paste(image, (image_x, image_y))
    draw_person_regions(
        draw,
        people,
        (prediction.width, prediction.height),
        (image_x, image_y, image_x + image.width, image_y + image.height),
        font,
    )
    for index, person in enumerate(people):
        color = person_color(index)
        label = f"#{index + 1} {_GENDERS[person['gender']]}"

        left = gap + (index % columns) * card_width
        row_index = index // columns
        top = row_tops[row_index]
        draw.rounded_rectangle(
            (left, top, left + card_width - 12, top + row_heights[row_index] - 12),
            radius=10,
            fill="#202b40",
        )
        draw.text((left + 12, top + 10), label, font=heading, fill=color)
        if "female_probability" in person and person.get("status") not in (
            "multiple_in_crop",
            "neighbor_overlap",
        ):
            scores = (
                f"女 {person['female_probability']:.1%} / "
                f"男 {person['male_probability']:.1%}"
            )
            draw.text((left + 12, top + 51), scores, font=font, fill="#c8d3e7")
        for row, line in enumerate(card_lines[index]):
            draw.text(
                (left + 12, top + 86 + row * 28),
                line,
                font=font,
                fill="#eef2fa",
            )

    for row, line in enumerate(note_lines):
        draw.text((gap, notes_y + row * 32), line, font=font, fill="#c8d3e7")
    output = BytesIO()
    canvas.save(output, format="JPEG", quality=92)
    return output.getvalue()

"""Human-readable diagnostics and deterministic annotations; never edit originals."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from ..config import paths
from .feature_presentation import feature_labels
from .feature_presentation import wrap_text as _wrap
from .people_report import render_people_report
from .region_overlay import draw_person_regions
from .vision.runtime import open_rgb
from .vision.types import TagPrediction

STATUS_LABELS = {
    "confident": "主体已确认",
    "co_primary": "双主体（同性别）",
    "legacy": "等待新版识别",
    "no_person": "未检测到可靠人物框",
    "weak_detection": "人物检测把握不足",
    "tentative_detection": "人物框待确认",
    "ambiguous_dominance": "多人占比接近，无法确定主角",
    "overlapping_people": "人物重叠，无法安全分离",
    "multiple_in_crop": "识别区域仍包含多个人物",
    "gender_uncertain": "两个模型对主体性别尚未达成可靠共识",
    "view_conflict": "完整人物区域与局部区域的性别判断冲突",
}


def gender_summary(prediction: TagPrediction | None) -> str:
    if prediction is None:
        return "主体识别暂不可用（不是性别置信度为 0%）"
    status = STATUS_LABELS.get(prediction.subject_status, "主体尚未确认")
    if prediction.subject_status not in ("confident", "co_primary"):
        salience = prediction.analysis.get("subject_salience")
        suggested = (
            salience.get("suggested_face_index")
            if isinstance(salience, dict)
            else None
        )
        if isinstance(suggested, int):
            status += f"（评分最接近 {suggested + 1} 号人物，编号见标注图）"
    if prediction.subject_status in ("confident", "co_primary"):
        scope = (
            "（依据避开配角的上半身区域）"
            if prediction.analysis.get("subject_view") == "upper_body"
            else ""
        )
        prefix = "双主体同性别：" if prediction.subject_status == "co_primary" else ""
        return (
            f"{prefix}主体女性置信度 {prediction.female_probability:.2%}，"
            f"男性置信度 {prediction.male_probability:.2%}{scope}"
        )
    if prediction.analysis.get("subject_gender"):
        return (
            f"主体未确认：{status}。\n"
            f"候选区域女性评分 {prediction.female_probability:.2%}，"
            f"男性评分 {prediction.male_probability:.2%}（不作为确定结论）"
        )
    return f"主体未确认：{status}；主体性别尚未计算。"


def report_text(prediction: TagPrediction) -> str:
    subject = feature_labels(prediction.subject_tags)
    scene = feature_labels(prediction.general_tags)
    lines = ["图片特征查询", gender_summary(prediction)]
    lines.append("主体特征：" + ("、".join(subject) if subject else "暂无可信标签"))
    if prediction.analysis.get("subject_view") == "upper_body":
        lines.append("本次主体标签仅描述上半身识别区域，不代表全身特征。")
    return "\n".join(lines)


def _font(size: int) -> ImageFont.FreeTypeFont:
    for filename in (
        paths.FONT_PATH,
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ):
        if Path(filename).is_file():
            return ImageFont.truetype(filename, size)
    raise RuntimeError("缺少中文字体，无法生成特征标注图")


def render_feature_report(path: Path, prediction: TagPrediction) -> bytes:
    """Boxes on an image copy + a legible Chinese tag panel, returned as JPEG."""
    original = open_rgb(path)
    font, heading = _font(23), _font(29)
    census = prediction.analysis.get("census")
    if census and census["detected"] > 1:
        return render_people_report(original, prediction, font, heading)
    original.thumbnail((1024, 1400), Image.Resampling.LANCZOS)
    panel_width, gap = 570, 24
    subject_labels = feature_labels(prediction.subject_tags)
    scene_labels = feature_labels(prediction.general_tags)
    sections = [
        ("上传图片特征分析", "#e8efff"),
        (gender_summary(prediction), "#81dbef"),
        (f"主体特征（≥50%，共 {len(subject_labels)} 项）", "#6aebb4"),
    ]
    if census:
        sections.append(
            (
                "逐人框：实线为人脸，虚线为实际判定裁剪；同编号同色，不是人体分割。",
                "#81dbef",
            )
        )
        sections.insert(
            1,
            (
                f"人脸候选 {census['detected']}：女性 {census['female']} / "
                f"男性 {census['male']} / 未确认 {census['unknown']}",
                "#81dbef",
            ),
        )
    if prediction.analysis.get("first_frame_only"):
        sections.insert(1, ("动态图仅分析首帧。", "#edbf71"))
    sections.extend((label, "#edf0f8") for label in subject_labels)
    if not subject_labels:
        sections.append(("暂无 ≥50% 的可信主体标签，不使用整图标签代替。", "#edbf71"))
    if prediction.analysis.get("subject_view") == "upper_body":
        sections.append(("仅描述上半身区域；未覆盖的全身特征不作推断。", "#edbf71"))
    sections.extend((label, "#c7cfdf") for label in scene_labels)
    sections.extend(
        (
            ("蓝框：人物候选；橙框：主角候选；绿框：特征识别区域。", "#81dbef"),
            (
                "框旁数字是人物检测分数，不是性别概率。检测可能漏人；标签仅供参考。",
                "#c7cfdf",
            ),
        )
    )
    wrapped = [
        (line, color)
        for text, color in sections
        for line in _wrap(text, font, panel_width - 2 * gap)
    ]
    height = max(original.height + 2 * gap + 44, len(wrapped) * 33 + 2 * gap)
    canvas = Image.new("RGB", (1024 + panel_width + gap * 3, height), "#151c2b")
    offset_x, offset_y = gap + (1024 - original.width) // 2, gap + 48
    canvas.paste(original, (offset_x, offset_y))
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, gap), "检测框与主体识别区域", font=heading, fill="#e8efff")
    scale_x, scale_y = (
        original.width / prediction.width,
        original.height / prediction.height,
    )

    def box(coords, color, label, *, below=False):
        rect = [
            offset_x + int(coords[0] * scale_x),
            offset_y + int(coords[1] * scale_y),
            offset_x + int(coords[2] * scale_x),
            offset_y + int(coords[3] * scale_y),
        ]
        draw.rectangle(rect, outline=color, width=4)
        label_width = int(font.getlength(label)) + 12
        left = min(rect[0], offset_x + original.width - label_width)
        top = min(height - 34, rect[3] + 4) if below else max(offset_y, rect[1] - 32)
        draw.rectangle((left, top, left + label_width, top + 32), fill="#151c2b")
        draw.text((left + 5, top), label, font=font, fill=color)

    main = prediction.analysis.get("subject_box")
    for index, person in enumerate(prediction.analysis.get("people", []), start=1):
        is_main = main is not None and tuple(person["box"]) == tuple(main)
        label = (
            f"#{index} {'主角候选' if is_main else '检测'} {person['confidence']:.1%}"
        )
        box(person["box"], "#ffb454" if is_main else "#6eb9ff", label)
    if feature := prediction.analysis.get("feature_box"):
        label = (
            "主体特征区域" if prediction.subject_status == "confident" else "待确认区域"
        )
        box(feature, "#6aebb4", label, below=True)
    if census:
        draw_person_regions(
            draw,
            census["people"],
            (prediction.width, prediction.height),
            (offset_x, offset_y, offset_x + original.width, offset_y + original.height),
            font,
        )
    y = gap
    for line, color in wrapped:
        draw.text((1024 + gap * 2, y), line, font=font, fill=color)
        y += 33
    result = BytesIO()
    canvas.save(result, format="JPEG", quality=90)
    return result.getvalue()

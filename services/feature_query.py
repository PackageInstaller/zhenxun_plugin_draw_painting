"""Bounded, temporary image diagnostics, separate from the persistent library."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from io import BytesIO
from pathlib import Path
import tempfile
from urllib.parse import urljoin, urlsplit

import httpx
from PIL import Image

from .feature_report import render_feature_report
from .model import ModelManager

MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 24_000_000
_IMAGE_DOMAINS = ("qq.com", "qq.com.cn", "qpic.cn", "gtimg.com", "gtimg.cn")


class FeatureQueryError(ValueError):
    pass


def select_image_data(message, reply_message=None) -> dict | None:
    """Attachments take precedence over quoted images. One image per query."""
    for source in (message, reply_message):
        images = [segment.data for segment in (source or ()) if segment.type == "image"]
        if len(images) > 1:
            raise FeatureQueryError("每次只支持查询一张图片，请单独发送或引用。")
        if images:
            return dict(images[0])
    return None


def validate_image_url(url: str) -> None:
    # Only fetch QQ attachment CDNs, never arbitrary user URLs/local files.
    try:
        parsed = urlsplit(url)
        host = (parsed.hostname or "").lower().rstrip(".")
        port = parsed.port
    except ValueError as exc:
        raise FeatureQueryError("图片链接格式无效，请重新上传图片。") from exc
    if (
        parsed.scheme not in ("http", "https")
        or parsed.username is not None
        or parsed.password is not None
        or port not in (None, 80, 443)
        or not any(
            host == domain or host.endswith("." + domain) for domain in _IMAGE_DOMAINS
        )
    ):
        raise FeatureQueryError(
            "仅支持聊天中直接上传的 QQ 图片，请重新上传图片后查询。"
        )


async def download_query_image(url: str) -> bytes:
    async def download() -> bytes:
        async with httpx.AsyncClient(
            timeout=30,
            follow_redirects=False,
            trust_env=False,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Referer": "https://multimedia.nt.qq.com.cn/",
            },
        ) as client:
            current = url
            for _ in range(4):
                validate_image_url(current)
                async with client.stream("GET", current) as response:
                    if response.is_redirect:
                        current = urljoin(current, response.headers.get("location", ""))
                        continue
                    response.raise_for_status()
                    length = response.headers.get("content-length", "")
                    if length.isdigit() and int(length) > MAX_IMAGE_BYTES:
                        raise FeatureQueryError("图片过大，请使用不超过 20 MB 的图片。")
                    content = bytearray()
                    async for block in response.aiter_bytes(64 * 1024):
                        if len(content) + len(block) > MAX_IMAGE_BYTES:
                            raise FeatureQueryError(
                                "图片过大，请使用不超过 20 MB 的图片。"
                            )
                        content.extend(block)
                    if not content:
                        raise FeatureQueryError("图片内容为空，请重新发送。")
                    return bytes(content)
            raise FeatureQueryError("图片链接重定向过多，请重新发送图片。")

    try:
        return await asyncio.wait_for(download(), timeout=60)
    except (httpx.HTTPError, asyncio.TimeoutError) as exc:
        # Avoid echoing expiring signed attachment URLs to group chats/logs.
        raise FeatureQueryError("图片下载失败或链接已过期，请重新上传后查询。") from exc


def _inspect_image(content: bytes) -> bytes:
    if not content or len(content) > MAX_IMAGE_BYTES:
        raise FeatureQueryError("图片不能为空且不能超过 20 MB。")
    try:
        with Image.open(BytesIO(content)) as source:
            if source.format not in {"PNG", "JPEG", "WEBP", "GIF", "BMP"}:
                raise FeatureQueryError("请使用 PNG、JPEG、WebP、GIF 或 BMP 图片。")
            if source.width * source.height > MAX_IMAGE_PIXELS:
                raise FeatureQueryError("图片分辨率过大，请缩小至 2400 万像素以内。")
            animated = bool(getattr(source, "is_animated", False))
            source.verify()
    except FeatureQueryError:
        raise
    except Exception as exc:
        raise FeatureQueryError("无法读取图片，请确认文件没有损坏。") from exc

    # No tag_image(): temporary user uploads must not pollute the library DB.
    with tempfile.TemporaryDirectory(prefix="painting-feature-") as directory:
        path = Path(directory) / "input.image"
        path.write_bytes(content)
        outcome = ModelManager.predict_query(path)
        if animated:
            outcome = replace(
                outcome, analysis={**outcome.analysis, "first_frame_only": True}
            )
        return render_feature_report(path, outcome)


async def inspect_query_image(content: bytes) -> bytes:
    task = asyncio.create_task(asyncio.to_thread(_inspect_image, content))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        # The native inference cannot be cancelled. Retain the caller's queue
        # slot and temporary input until it really stops using that input.
        try:
            await task
        finally:
            raise


def render_path_recognition(path: Path) -> bytes:
    """Annotated recognition report for a LOCAL library image (draw cards).

    Pure inference + rendering on a copy: like the user-upload query path it
    never writes to the feature store or moves the file, so attaching it to a
    draw cannot change library state.
    """
    outcome = ModelManager.predict_query(path)
    return render_feature_report(path, outcome)

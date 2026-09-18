from __future__ import annotations

from pathlib import Path
import time
from urllib.parse import quote
import uuid

from rich.markup import escape
import boto3
from botocore.client import Config as BotoConfig
from ..config import plugin_config
from .package_archive import _new_progress

PACKAGE_TTL_SECONDS = 24 * 60 * 60

class OscaOSS:
    """OSCA 联盟云 S3 兼容存储封装。"""

    def __init__(self) -> None:
        cfg = plugin_config
        self.bucket = cfg.realcugan_oss_bucket
        self.prefix = cfg.realcugan_oss_prefix.lstrip("/")
        if self.prefix and not self.prefix.endswith("/"):
            self.prefix += "/"
        self.url_expires = cfg.realcugan_oss_url_expires
        self.client = boto3.client(
            "s3",
            endpoint_url=cfg.realcugan_oss_endpoint,
            aws_access_key_id=cfg.realcugan_oss_access_key,
            aws_secret_access_key=cfg.realcugan_oss_secret_key,
            region_name="us-east-1",
            use_ssl=cfg.realcugan_oss_endpoint.startswith("https"),
            config=BotoConfig(signature_version="s3v4"),
        )

    def object_key(self, filename: str) -> str:
        return f"{self.prefix}{filename}"

    def upload_file(self, local_path: Path, object_key: str) -> str:
        """上传文件并返回 1 天有效的预签名下载链接。"""
        self.client.upload_file(str(local_path), self.bucket, object_key)
        return self.presign(object_key)

    def presign(self, object_key: str) -> str:
        return self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": object_key},
            ExpiresIn=self.url_expires,
        )

    def delete(self, object_key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=object_key)


class PaintingPackageStorage:
    """Use the RealCUGAN plugin's configured S3-compatible bucket."""

    def __init__(self, expires_in: int = PACKAGE_TTL_SECONDS):
        self.expires_in = expires_in
        self._backend = None

    def _get_backend(self):

        self._backend = OscaOSS()
        return self._backend

    def upload(self, archive_path: Path, download_name: str) -> tuple[str, str]:
        backend = self._get_backend()
        object_key = backend.object_key(
            "painting-packages/"
            f"{int(time.time())}_{uuid.uuid4().hex[:12]}_{download_name}"
        )
        disposition_name = quote(download_name, safe="")
        archive_size = archive_path.stat().st_size
        with _new_progress() as progress:
            task_id = progress.add_task(
                f"上传 {escape(download_name)}",
                total=max(1, archive_size),
            )
            backend.client.upload_file(
                str(archive_path),
                backend.bucket,
                object_key,
                ExtraArgs={
                    "ContentType": "application/zip",
                    "ContentDisposition": (
                        f"attachment; filename*=UTF-8''{disposition_name}"
                    ),
                },
                Callback=lambda transferred: progress.update(
                    task_id,
                    advance=transferred,
                ),
            )
            progress.update(task_id, completed=max(1, archive_size))
        url = self.generate_download_url(object_key)
        return object_key, url

    def generate_download_url(
        self,
        object_key: str,
        expires_in: int | None = None,
    ) -> str:
        backend = self._get_backend()
        valid_for = self.expires_in if expires_in is None else expires_in
        valid_for = max(1, min(self.expires_in, int(valid_for)))
        return backend.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": backend.bucket, "Key": object_key},
            ExpiresIn=valid_for,
        )

    def delete(self, object_key: str) -> None:
        self._get_backend().delete(object_key)

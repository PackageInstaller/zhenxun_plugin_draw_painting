from __future__ import annotations

from pathlib import Path
import time
from urllib.parse import quote
import uuid

from rich.markup import escape

from .package_archive import _new_progress

PACKAGE_TTL_SECONDS = 24 * 60 * 60


class PaintingPackageStorage:
    """Use the RealCUGAN plugin's configured S3-compatible bucket."""

    def __init__(self, expires_in: int = PACKAGE_TTL_SECONDS):
        self.expires_in = expires_in
        self._backend = None

    def _get_backend(self):
        if self._backend is None:
            # Lazy import avoids loading the RealCUGAN model during plugin import.
            from zhenxun.plugins.nonebot_plugin_realcugan.oss import OscaOSS

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

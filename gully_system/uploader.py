"""HTTP uploader and background upload worker."""

from __future__ import annotations

import json
import logging
import mimetypes
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from gully_system.data_manager import StorageQueue

logger = logging.getLogger(__name__)


class Uploader(Protocol):
    def upload(self, event: dict[str, Any], evidence_path: Path | None = None) -> None:
        ...


@dataclass
class HttpUploader:
    url: str
    token: str = ""
    timeout_s: float = 10.0

    def upload(self, event: dict[str, Any], evidence_path: Path | None = None) -> None:
        if not self.url:
            raise RuntimeError("Upload URL is not configured")
        metadata = json.dumps(event, ensure_ascii=True).encode("utf-8")
        if evidence_path is not None:
            if not evidence_path.exists():
                raise FileNotFoundError(f"Evidence image not found: {evidence_path}")
            boundary = f"----gully-monitor-{id(event)}"
            image = evidence_path.read_bytes()
            body = self._multipart_body(boundary, metadata, evidence_path.name, image)
            headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
        else:
            body = metadata
            headers = {"Content-Type": "application/json"}

        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        request = urllib.request.Request(self.url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                if not (200 <= response.status < 300):
                    raise RuntimeError(f"Server returned HTTP {response.status}")
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Upload failed: {exc}") from exc

    @staticmethod
    def _multipart_body(boundary: str, metadata: bytes, filename: str, image: bytes) -> bytes:
        image_type = mimetypes.guess_type(filename)[0] or "image/jpeg"
        boundary_bytes = boundary.encode("ascii")
        return b"".join((
            b"--" + boundary_bytes + b'\r\nContent-Disposition: form-data; name="metadata"\r\nContent-Type: application/json\r\n\r\n' + metadata + b"\r\n",
            b"--" + boundary_bytes + b"\r\n" + f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'.encode("utf-8") + f"Content-Type: {image_type}\r\n\r\n".encode("ascii") + image + b"\r\n",
            b"--" + boundary_bytes + b"--\r\n",
        ))


class NullUploader:
    def upload(self, event: dict[str, Any], evidence_path: Path | None = None) -> None:
        return None


class UploadWorker:
    def __init__(self, queue: StorageQueue, uploader: Uploader, base_backoff_s: float = 5.0, max_backoff_s: float = 60.0) -> None:
        self.queue = queue
        self.uploader = uploader
        self.base_backoff_s = base_backoff_s
        self.max_backoff_s = max_backoff_s
        self.consecutive_failures = 0
        self.next_retry_time = 0.0

    def flush_once(self, max_items: int = 10) -> tuple[int, int]:
        now = time.time()
        if now < self.next_retry_time:
            return (0, 0)

        sent = 0
        failed = 0
        for path, event in list(self.queue.iter_pending())[:max_items]:
            try:
                self.uploader.upload(event, self.queue.evidence_path(path, event))
                self.queue.mark_sent(path)
                sent += 1
                if self.consecutive_failures > 0:
                    logger.info("업로드 연결이 복구되었습니다. (정상 전송 재개)")
                    self.consecutive_failures = 0
            except Exception as e:
                failed += 1
                self.consecutive_failures += 1
                delay = min(self.base_backoff_s * (2 ** (self.consecutive_failures - 1)), self.max_backoff_s)
                self.next_retry_time = now + delay
                logger.warning(
                    f"업로드 실패 ({e}). {delay:.1f}초 동안 재시도를 일시 중단합니다. (연속 실패: {self.consecutive_failures}회)"
                )
                break
        return (sent, failed)

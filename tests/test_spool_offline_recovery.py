"""Unit test verifying offline edge spooling, exponential backoff, and recovery."""

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pytest
from gully_system.data_manager import StorageQueue
from gully_system.uploader import UploadWorker


class MockUploader:
    def __init__(self, is_online: bool = False) -> None:
        self.is_online = is_online
        self.uploaded_events: list[dict[str, Any]] = []

    def upload(self, event: dict[str, Any], evidence_path: Path | None = None) -> None:
        if not self.is_online:
            raise RuntimeError("Connection refused: simulated LTE offline")
        self.uploaded_events.append({
            "event": event,
            "has_evidence": evidence_path is not None and evidence_path.exists(),
        })


def test_spool_offline_retention_and_recovery():
    temp_dir = Path(tempfile.mkdtemp(prefix="beum_spool_test_"))
    try:
        spool_dir = temp_dir / "spool"
        queue = StorageQueue(base_dir=spool_dir, max_bytes=10_000_000)

        # 1. Enqueue 3 events with dummy JPEG evidence
        for i in range(3):
            payload = {
                "event_id": f"evt-{i}",
                "timestamp": 1700000000.0 + i,
                "status": "critical",
                "coverage_percent": 100.0,
            }
            dummy_jpeg = b"\xff\xd8\xff\xe0" + b"\x00" * 100 + b"\xff\xd9"
            queue.enqueue(payload, evidence_bytes=dummy_jpeg)

        # Verify 3 events are stored in pending queue
        pending_items = list(queue.iter_pending())
        assert len(pending_items) == 3

        # 2. Simulate Offline: Mock uploader is offline
        mock_uploader = MockUploader(is_online=False)
        worker = UploadWorker(queue=queue, uploader=mock_uploader, base_backoff_s=0.1, max_backoff_s=1.0)

        sent, failed = worker.flush_once(max_items=10)
        assert sent == 0
        assert failed == 1
        assert worker.consecutive_failures == 1

        # Confirm all 3 events are STILL intact in pending queue (zero data loss)
        pending_items_after_fail = list(queue.iter_pending())
        assert len(pending_items_after_fail) == 3

        # 3. Simulate Online: Network connection restored
        mock_uploader.is_online = True
        worker.next_retry_time = 0.0  # Fast-forward backoff

        # Flush queue now that network is restored
        sent, failed = worker.flush_once(max_items=10)
        assert sent == 3
        assert failed == 0
        assert worker.consecutive_failures == 0

        # Verify mock uploader received all 3 events with valid evidence
        assert len(mock_uploader.uploaded_events) == 3
        for item in mock_uploader.uploaded_events:
            assert item["has_evidence"] is True

        # Verify pending queue is now completely cleared
        pending_items_after_recovery = list(queue.iter_pending())
        assert len(pending_items_after_recovery) == 0

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

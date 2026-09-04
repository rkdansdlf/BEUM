"""Storage queue managing local event spooling and retention."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, Iterator


class StorageQueue:
    """A bounded, crash-safe local spool for events awaiting upload."""

    def __init__(self, base_dir: str | Path, max_bytes: int = 1_000_000_000) -> None:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self.base_dir = Path(base_dir)
        self.pending_dir = self.base_dir / "pending"
        self.pending_dir.mkdir(parents=True, exist_ok=True)
        self.max_bytes = int(max_bytes)

    def _atomic_write(self, path: Path, content: bytes) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(content)
        os.replace(temporary, path)

    def _safe_evidence_path(self, event_path: Path, event: dict[str, Any]) -> Path | None:
        name = event.get("evidence_file")
        if not name:
            return None
        candidate = (event_path.parent / str(name)).resolve()
        try:
            candidate.relative_to(self.pending_dir.resolve())
            return candidate
        except ValueError:
            return None

    def enqueue(self, payload: dict[str, Any], evidence_bytes: bytes | None = None, evidence_suffix: str = ".jpg") -> str:
        event_id = str(payload.get("event_id") or uuid.uuid4())
        event = dict(payload)
        event["event_id"] = event_id

        evidence_name: str | None = None
        if evidence_bytes is not None:
            evidence_name = f"{event_id}{evidence_suffix}"
            event["evidence_file"] = evidence_name

        data = json.dumps(event, ensure_ascii=False, indent=2).encode("utf-8")
        event_path = self.pending_dir / f"{event_id}.json"
        self._atomic_write(event_path, data)

        if evidence_bytes is not None and evidence_name is not None:
            evidence_path = self.pending_dir / evidence_name
            self._atomic_write(evidence_path, evidence_bytes)

        self.prune(protected=event_path)
        return event_id

    def iter_pending(self) -> Iterator[tuple[Path, dict[str, Any]]]:
        for path in self.pending_paths():
            try:
                yield path, self.load(path)
            except (OSError, json.JSONDecodeError):
                continue

    def pending_paths(self) -> list[Path]:
        return sorted(self.pending_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)

    def pending_count(self) -> int:
        return len(list(self.pending_dir.glob("*.json")))

    def load(self, event_path: str | Path) -> dict[str, Any]:
        with Path(event_path).open("r", encoding="utf-8") as f:
            return json.load(f)

    def evidence_path(self, event_path: str | Path, event: dict[str, Any] | None = None) -> Path | None:
        path = Path(event_path)
        if event is None:
            event = self.load(path)
        return self._safe_evidence_path(path, event)

    def mark_sent(self, event_path: str | Path) -> None:
        path = Path(event_path)
        try:
            event = self.load(path)
            evidence = self._safe_evidence_path(path, event)
            if evidence and evidence.exists():
                evidence.unlink()
        except Exception:
            pass
        if path.exists():
            path.unlink()

    def usage_bytes(self) -> int:
        return sum(f.stat().st_size for f in self.pending_dir.iterdir() if f.is_file())

    def prune(self, protected: Path | None = None) -> int:
        deleted = 0
        while self.usage_bytes() > self.max_bytes:
            paths = self.pending_paths()
            if not paths:
                break
            victim = paths[0]
            if victim == protected and len(paths) > 1:
                victim = paths[1]
            elif victim == protected:
                break
            self.mark_sent(victim)
            deleted += 1
        return deleted

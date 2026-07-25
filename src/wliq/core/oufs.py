"""OUFS v1: append-only JSONL operational logging (empire-wide standard).

Three streams per day directory:
  decisions.jsonl  — every signal/governor/risk decision with inputs
  outcomes.jsonl   — every fill, closed trade, and realized outcome
  health.jsonl     — periodic heartbeat: stream health, positions, risk state
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class OufsWriter:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _write(self, stream: str, record: dict[str, Any]) -> None:
        now = datetime.now(timezone.utc)
        day = self.root / now.date().isoformat()
        day.mkdir(parents=True, exist_ok=True)
        record = {"ts": now.isoformat(), **record}
        with open(day / f"{stream}.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def decision(self, kind: str, **kw: Any) -> None:
        self._write("decisions", {"kind": kind, **kw})

    def outcome(self, kind: str, **kw: Any) -> None:
        self._write("outcomes", {"kind": kind, **kw})

    def health(self, **kw: Any) -> None:
        self._write("health", kw)


class NullOufs(OufsWriter):
    """No-op writer for unit tests."""

    def __init__(self) -> None:  # noqa: D401
        pass

    def _write(self, stream: str, record: dict[str, Any]) -> None:
        return

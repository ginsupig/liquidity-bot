"""Unified Clock abstraction.

Every time-dependent component takes a Clock so that replay, paper, and live
run byte-identical logic. Live uses WallClock; replay drives SimClock from the
event stream, which makes the entire system deterministic under replay.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Protocol


class Clock(Protocol):
    def now(self) -> datetime: ...


class WallClock:
    def now(self) -> datetime:
        return datetime.now(timezone.utc)


class SimClock:
    """Deterministic clock advanced by the replay event loop (monotonic)."""

    def __init__(self, start: datetime | None = None) -> None:
        self._now = start or datetime(1970, 1, 1, tzinfo=timezone.utc)

    def set(self, ts: datetime) -> None:
        if ts > self._now:
            self._now = ts

    def advance_ms(self, ms: float) -> None:
        from datetime import timedelta
        self._now = self._now + timedelta(milliseconds=ms)

    def now(self) -> datetime:
        return self._now

"""Data-health monitor: staleness, sequence gaps, reconnects, drops.

The governor consults this before permitting any trade. Fail closed: unknown
health is unhealthy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(slots=True)
class SymbolHealth:
    last_event: datetime | None = None
    last_seq: int | None = None
    gap_count: int = 0


@dataclass(slots=True)
class DataHealthMonitor:
    stale_ms: int = 1500
    gap_tolerance: int = 5
    connected: bool = False
    reconnect_count: int = 0
    dropped_messages: int = 0
    symbols: dict[str, SymbolHealth] = field(default_factory=dict)

    def on_connect(self) -> None:
        self.connected = True

    def on_disconnect(self) -> None:
        self.connected = False
        self.reconnect_count += 1

    def on_event(self, symbol: str, ts: datetime, seq: int | None = None) -> None:
        h = self.symbols.setdefault(symbol, SymbolHealth())
        h.last_event = ts
        if seq is not None:
            if h.last_seq is not None and seq > h.last_seq + 1:
                h.gap_count += seq - h.last_seq - 1
                self.dropped_messages += seq - h.last_seq - 1
            h.last_seq = seq

    def is_healthy(self, symbol: str, now: datetime | None = None) -> tuple[bool, str]:
        if not self.connected:
            return False, "stream_disconnected"
        h = self.symbols.get(symbol)
        if h is None or h.last_event is None:
            return False, "no_data"
        now = now or datetime.now(timezone.utc)
        age_ms = (now - h.last_event).total_seconds() * 1000
        if age_ms > self.stale_ms:
            return False, f"stale_data_{int(age_ms)}ms"
        if h.gap_count > self.gap_tolerance:
            return False, f"sequence_gaps_{h.gap_count}"
        return True, "healthy"

    def reset_gaps(self, symbol: str) -> None:
        if symbol in self.symbols:
            self.symbols[symbol].gap_count = 0

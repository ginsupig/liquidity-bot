"""Order-state reconciler.

On startup and periodically, compare local journal/position state against the
broker's view. Any disagreement puts the system into a NO_TRADE state until a
human (or explicit auto-heal policy) resolves it. Fail closed.

The broker read methods live on the adapter; against the paper broker this
verifies internal consistency only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from wliq.core.models import Position


class BrokerReadView(Protocol):
    def read_positions(self) -> dict[str, float]: ...
    def read_open_orders(self) -> list[str]: ...
    def read_account_ok(self) -> bool: ...


@dataclass(slots=True)
class ReconcileResult:
    ok: bool
    issues: list[str] = field(default_factory=list)


def reconcile(local_positions: dict[str, Position],
              broker: BrokerReadView) -> ReconcileResult:
    issues: list[str] = []
    try:
        if not broker.read_account_ok():
            issues.append("account_state_unknown")
        broker_pos = broker.read_positions()
    except Exception as exc:  # any broker read failure is a hard stop
        return ReconcileResult(False, [f"broker_read_failed:{exc!r}"])

    local_qty = {s: p.side.sign * p.quantity for s, p in local_positions.items()}
    for symbol in set(local_qty) | set(broker_pos):
        lq = local_qty.get(symbol, 0.0)
        bq = broker_pos.get(symbol, 0.0)
        if abs(lq - bq) > 1e-6:
            issues.append(f"position_mismatch:{symbol}:local={lq}:broker={bq}")
    return ReconcileResult(not issues, issues)

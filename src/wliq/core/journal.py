"""Persistent order journal with idempotency.

Append-only JSONL (OUFS-compatible layout: one decision/order event per line).
Client order ids are deterministic per (date, symbol, intent, seq) so a crash
and restart cannot resubmit a duplicate, and the journal survives restarts.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from wliq.core.models import OrderRequest, OrderStatus


class OrderJournal:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._seen: set[str] = set()
        self._seq: dict[str, int] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # torn tail write from a crash; subsequent lines are valid
                coid = row.get("client_order_id")
                if coid:
                    self._seen.add(coid)
                key = row.get("seq_key")
                if key is not None:
                    self._seq[key] = max(self._seq.get(key, 0), int(row.get("seq", 0)))

    def next_client_order_id(self, symbol: str, intent: str, now: datetime | None = None) -> str:
        now = now or datetime.now(timezone.utc)
        key = f"{now.date().isoformat()}-{symbol}-{intent}"
        with self._lock:
            seq = self._seq.get(key, 0) + 1
            self._seq[key] = seq
        return f"wliq-{key}-{seq:04d}"

    def is_duplicate(self, client_order_id: str) -> bool:
        return client_order_id in self._seen

    def record(self, event: str, order: OrderRequest, status: OrderStatus,
               extra: dict[str, Any] | None = None) -> None:
        parts = order.client_order_id.rsplit("-", 1)
        row = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "client_order_id": order.client_order_id,
            "trace_id": order.trace_id,
            "symbol": order.symbol,
            "side": order.side.value,
            "quantity": order.quantity,
            "order_type": order.order_type,
            "limit_price": order.limit_price,
            "intent": order.intent,
            "status": status.value,
            "seq_key": parts[0].removeprefix("wliq-") if len(parts) == 2 else None,
            "seq": int(parts[1]) if len(parts) == 2 and parts[1].isdigit() else 0,
        }
        if extra:
            row.update(extra)
        with self._lock:
            self._seen.add(order.client_order_id)
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, default=str) + "\n")

"""Persistent daily risk ledger + position event log + startup recovery.

Two append-only JSONL files under logs/:
  risk_ledger.jsonl     — realized pnl deltas, halts, cooldowns (per day)
  position_events.jsonl — entry/exit fill events sufficient to rebuild
                          open positions after a crash

Recovery contract (fail closed):
  1. Rebuild open positions from position events.
  2. Rebuild non-terminal orders from the order journal.
  3. Reload today's realized PnL / loss counters into the RiskManager.
  4. Reconcile everything against broker truth; any mismatch blocks trading.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from wliq.core.models import (Fill, GovernorAction, OrderRequest, Position,
                              Side)
from wliq.execution.order_manager import ManagedOrder, OState, TERMINAL


class RiskLedger:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, kind: str, **kw: Any) -> None:
        row = {"ts": datetime.now(timezone.utc).isoformat(),
               "date": date.today().isoformat(), "kind": kind, **kw}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")

    def load_day(self, day: str | None = None) -> dict[str, Any]:
        day = day or date.today().isoformat()
        state = {"realized_pnl": 0.0, "consecutive_losses": 0, "n_trades": 0,
                 "halted": False}
        if not self.path.exists():
            return state
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("date") != day:
                    continue
                if row["kind"] == "trade_closed":
                    pnl = float(row.get("pnl", 0.0))
                    state["realized_pnl"] += pnl
                    state["n_trades"] += 1
                    state["consecutive_losses"] = (
                        0 if pnl > 0 else state["consecutive_losses"] + 1)
                elif row["kind"] == "halt":
                    state["halted"] = True
                elif row["kind"] == "halt_cleared":
                    state["halted"] = False
        return state


class PositionEventLog:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append_fill(self, kind: str, fill: Fill, extra: dict[str, Any] | None = None) -> None:
        row = {"ts": fill.ts.isoformat(), "kind": kind, "symbol": fill.symbol,
               "side": fill.side.value, "qty": fill.quantity, "price": fill.price,
               "coid": fill.client_order_id, "trace_id": fill.trace_id}
        if extra:
            row.update(extra)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")

    def rebuild_positions(self) -> dict[str, Position]:
        """Net entry/exit events into open positions (average-price basis)."""
        book: dict[str, dict[str, Any]] = {}
        if not self.path.exists():
            return {}
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sym = row["symbol"]
                side = Side(row["side"])
                qty, px = float(row["qty"]), float(row["price"])
                if row["kind"] == "entry":
                    b = book.setdefault(sym, {"side": side, "qty": 0.0, "cost": 0.0,
                                              "ts": row["ts"], "trace": row["trace_id"],
                                              "stop": row.get("stop"), "tgt": row.get("target"),
                                              "action": row.get("action", "REVERSAL")})
                    b["qty"] += qty
                    b["cost"] += qty * px
                elif row["kind"] == "exit" and sym in book:
                    b = book[sym]
                    avg = b["cost"] / max(b["qty"], 1e-12)
                    b["qty"] -= qty
                    b["cost"] -= qty * avg        # remove at average basis
                    if b["qty"] <= 1e-9:
                        del book[sym]
        out: dict[str, Position] = {}
        for sym, b in book.items():
            avg = b["cost"] / max(b["qty"], 1e-12)
            out[sym] = Position(
                symbol=sym, side=b["side"], quantity=b["qty"], entry_price=avg,
                opened_at=datetime.fromisoformat(b["ts"]),
                stop_price=float(b["stop"]) if b.get("stop") else avg,
                target_price=float(b["tgt"]) if b.get("tgt") else avg,
                signal_score=0.0, trace_id=b["trace"],
                action=GovernorAction(b.get("action", "REVERSAL")))
        return out


def rebuild_open_orders(journal_path: Path) -> list[ManagedOrder]:
    """Replay the order journal; return orders whose LAST state is non-terminal."""
    latest: dict[str, dict[str, Any]] = {}
    if not Path(journal_path).exists():
        return []
    with open(journal_path, encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("client_order_id"):
                latest[row["client_order_id"]] = row
    out: list[ManagedOrder] = []
    terminal_vals = {s.value for s in TERMINAL} | {"CANCELLED", "FILLED",
                                                  "REJECTED", "EXPIRED"}
    for coid, row in latest.items():
        status = row.get("status", "")
        if status in terminal_vals:
            continue
        try:
            req = OrderRequest(row["symbol"], Side(row["side"]),
                               float(row["quantity"]), row.get("order_type", "LIMIT"),
                               row.get("limit_price"), coid,
                               row.get("trace_id", ""), row.get("intent", "entry"))
            state = OState(status) if status in OState.__members__ else OState.ACKNOWLEDGED
        except (KeyError, ValueError):
            continue
        mo = ManagedOrder(request=req, state=state,
                          filled_qty=float(row.get("filled", 0.0)))
        out.append(mo)
    return out

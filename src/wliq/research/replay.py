"""Event-driven historical replay.

Replays recorded quotes/trades/depth in strict timestamp order through the
SAME FeatureEngine -> Governor -> Strategy -> Risk -> PaperBroker ->
PositionManager pipeline used live. No candle-only shortcuts; the paper
broker's queue model applies identically in replay.
"""
from __future__ import annotations

import heapq
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from wliq.broker.health import DataHealthMonitor
from wliq.core.config import Settings
from wliq.core.journal import OrderJournal
from wliq.core.models import (ClosedTrade, DepthLevel, DepthSnapshot, OrderRequest,
                              Quote, Session, Side, Trade)
from wliq.execution.paper import PaperBroker
from wliq.execution.positions import PositionManager
from wliq.execution.risk import RiskManager
from wliq.features.engine import FeatureEngine
from wliq.features.volume import VolumeProfile, VolumeTracker
from wliq.strategy.exhaustion import ResidualExhaustionStrategy
from wliq.strategy.governor import StrategyGovernor


@dataclass(slots=True, frozen=True)
class ReplayEvent:
    ts: datetime
    order: int          # tiebreak: quotes before trades at equal ts
    kind: str
    payload: object

    def __lt__(self, other: "ReplayEvent") -> bool:
        return (self.ts, self.order) < (other.ts, other.order)


def _to_dt(v: object) -> datetime:
    t = pd.Timestamp(v)
    return (t.tz_localize(timezone.utc) if t.tzinfo is None else t.tz_convert(timezone.utc)).to_pydatetime()


def load_events(root: Path, date: str, symbols: list[str]) -> list[ReplayEvent]:
    events: list[ReplayEvent] = []
    for symbol in symbols:
        base = root / f"date={date}" / f"symbol={symbol}"
        if not base.exists():
            continue
        for path in base.glob("quotes-*.parquet"):
            for r in pd.read_parquet(path).to_dict("records"):
                q = Quote(r["symbol"], _to_dt(r["ts"]), float(r["bid"]), float(r["ask"]),
                          float(r["bid_size"]), float(r["ask_size"]),
                          Session(r.get("session") or "RTH"),
                          int(r["seq"]) if pd.notna(r.get("seq")) else None)
                events.append(ReplayEvent(q.ts, 0, "quote", q))
        for path in base.glob("trades-*.parquet"):
            for r in pd.read_parquet(path).to_dict("records"):
                agg = Side(r["aggressor"]) if r.get("aggressor") else None
                t = Trade(r["symbol"], _to_dt(r["ts"]), float(r["price"]), float(r["size"]),
                          agg, Session(r.get("session") or "RTH"),
                          int(r["seq"]) if pd.notna(r.get("seq")) else None)
                events.append(ReplayEvent(t.ts, 1, "trade", t))
        for path in base.glob("depth-*.parquet"):
            for r in pd.read_parquet(path).to_dict("records"):
                bids = tuple(DepthLevel(p, s) for p, s in json.loads(r["bids"]))
                asks = tuple(DepthLevel(p, s) for p, s in json.loads(r["asks"]))
                d = DepthSnapshot(r["symbol"], _to_dt(r["ts"]), bids, asks)
                events.append(ReplayEvent(d.ts, 2, "depth", d))
    heapq.heapify(events)
    return events


@dataclass(slots=True)
class ReplayResult:
    trades: list[ClosedTrade] = field(default_factory=list)
    signals: int = 0
    rejections: dict[str, int] = field(default_factory=dict)
    governor_actions: dict[str, int] = field(default_factory=dict)

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl for t in self.trades)


def run_replay(settings: Settings, date: str,
               journal_path: Path | None = None,
               logs_dir: Path | None = None) -> ReplayResult:
    """Drive the SAME SessionEngine used in paper/live with a SimClock.
    Determinism: the clock is set from each event timestamp before dispatch;
    the paper venue and OM see identical time in every run."""
    import tempfile

    from wliq.core.clock import SimClock
    from wliq.runtime import SessionEngine

    events = load_events(settings.record_root, date,
                         settings.all_streamed_symbols())
    result = ReplayResult()
    if not events:
        return result
    if logs_dir is None:
        logs_dir = Path(tempfile.mkdtemp(prefix="wliq-replay-"))
    clock = SimClock(events[0].ts)
    eng = SessionEngine(settings, clock, live_adapter=None, logs_dir=logs_dir)
    eng.health.on_connect()
    if journal_path is not None:
        eng.journal = OrderJournal(journal_path)
        eng.om.journal = eng.journal

    # counters via wrap
    _decide = eng.governor.decide
    def decide(f, now):
        d = _decide(f, now)
        result.governor_actions[d.action.value] = (
            result.governor_actions.get(d.action.value, 0) + 1)
        return d
    eng.governor.decide = decide
    _step = eng.fsm.step
    def step(f, action):
        sig = _step(f, action)
        if sig is not None:
            result.signals += 1
        return sig
    eng.fsm.step = step
    _approve = eng.risk.approve
    def approve(sig, f, now):
        ok, reason = _approve(sig, f, now=now)
        if not ok:
            result.rejections[reason] = result.rejections.get(reason, 0) + 1
        return ok, reason
    eng.risk.approve = lambda sig, f, now: approve(sig, f, now)

    for ev in heapq.nsmallest(len(events), events):
        clock.set(ev.ts)
        if ev.kind == "quote":
            eng.on_quote(ev.payload)
        elif ev.kind == "trade":
            eng.on_trade(ev.payload)
        elif ev.kind == "depth":
            eng.on_depth(ev.payload)

    # EOD: flatten any residual position at last observed quote
    for symbol in list(eng.pm.positions):
        q = eng.venue.last_quote.get(symbol) or eng.engine.quotes[symbol][-1]
        clock.advance_ms(settings.execution.paper.latency_ms + 250)
        eng.emergency = True
        eng._submit_exit(q, None)
        px = Quote(symbol, clock.now(), q.bid, q.ask, q.bid_size, q.ask_size)
        eng._apply_venue_events(eng.venue.on_quote(px))
        eng.on_tick()
        eng.emergency = False
    result.trades = list(eng.pm.closed)
    eng.shutdown()
    return result

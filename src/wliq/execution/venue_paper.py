"""PaperVenue v2: a venue *simulator* that lives behind the OrderManager.

The OM owns the lifecycle; the venue only produces venue events (acks,
fills, cancel-acks, rejects) in response to submissions, market data, and
clock polls. This mirrors a real venue adapter exactly, so paper and live
differ only in which venue object is plugged in.

Fill model (conservative, as v0.2, plus new order types):
  * LIMIT (passive): join behind 100% of displayed size at submission; prints
    at our price consume queue; a strict trade-through fills the remainder at
    OUR limit. Fill probability haircut `fill_probability` (stress lever).
  * LIMIT (marketable) / MARKET: activation-latency gated; fills against the
    FIRST QUOTE OBSERVED AFTER activation, taking at most
    `marketable_max_take_fraction` of displayed size per quote (partials).
  * IOC: whatever is immediately fillable at activation quote; rest cancelled.
  * FOK: fills fully at activation quote or is cancelled whole.
Acks arrive after `ack_latency_ms`; cancels after `cancel_latency_ms`.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from wliq.core.clock import Clock
from wliq.core.config import PaperFillConfig
from wliq.core.models import Fill, OrderRequest, Quote, Side, Trade


@dataclass
class VenueEvent:
    kind: str                  # ack | fill | cancel_ack | reject
    coid: str
    fill: Fill | None = None
    reason: str = ""


@dataclass
class _Working:
    req: OrderRequest
    submitted_at: datetime
    ack_at: datetime
    activate_at: datetime
    limit: float | None
    queue_ahead: float = 0.0
    filled: float = 0.0
    acked: bool = False
    cancel_at: datetime | None = None
    tif: str = "DAY"           # DAY | IOC | FOK
    awaiting_first_quote: bool = False

    @property
    def remaining(self) -> float:
        return max(0.0, self.req.quantity - self.filled)


class PaperVenue:
    def __init__(self, cfg: PaperFillConfig, clock: Clock,
                 ack_latency_ms: int = 120, fill_probability: float = 1.0,
                 extra_latency_ms: int = 0, seed: int = 7) -> None:
        self.cfg = cfg
        self.clock = clock
        self.ack_latency_ms = ack_latency_ms
        self.fill_probability = fill_probability      # stress lever (<1.0)
        self.extra_latency_ms = extra_latency_ms      # stress lever
        self._rng = random.Random(seed)
        self.working: dict[str, _Working] = {}
        self.last_quote: dict[str, Quote] = {}

    # ------------- order entry -------------
    def submit(self, req: OrderRequest, tif: str = "DAY") -> list[VenueEvent]:
        now = self.clock.now()
        lat = self.cfg.latency_ms + self.extra_latency_ms
        w = _Working(
            req=req, submitted_at=now,
            ack_at=now + timedelta(milliseconds=self.ack_latency_ms),
            activate_at=now + timedelta(milliseconds=lat),
            limit=req.limit_price, tif=tif,
        )
        q = self.last_quote.get(req.symbol)
        marketable = self._is_marketable(req, q)
        if req.order_type == "MARKET" or marketable or tif in ("IOC", "FOK"):
            w.awaiting_first_quote = True   # fill from first quote AFTER activation
        elif q is not None:
            w.queue_ahead = ((q.bid_size if req.side is Side.BUY else q.ask_size)
                             * self.cfg.queue_position_factor)
        self.working[req.client_order_id] = w
        return []

    @staticmethod
    def _is_marketable(req: OrderRequest, q: Quote | None) -> bool:
        if req.order_type == "MARKET" or req.limit_price is None or q is None:
            return req.order_type == "MARKET"
        return (req.limit_price >= q.ask if req.side is Side.BUY
                else req.limit_price <= q.bid)

    def cancel(self, coid: str) -> None:
        w = self.working.get(coid)
        if w and w.cancel_at is None:
            w.cancel_at = self.clock.now() + timedelta(
                milliseconds=self.cfg.cancel_latency_ms)

    # ------------- market data -------------
    def on_quote(self, q: Quote) -> list[VenueEvent]:
        self.last_quote[q.symbol] = q
        out: list[VenueEvent] = []
        now = self.clock.now()
        for coid, w in list(self.working.items()):
            if w.req.symbol != q.symbol or not w.awaiting_first_quote:
                continue
            if now < w.activate_at or q.ts < w.activate_at:
                continue   # first quote AFTER activation, strictly
            out.extend(self._aggress(coid, w, q))
        return out

    def _aggress(self, coid: str, w: _Working, q: Quote) -> list[VenueEvent]:
        out: list[VenueEvent] = []
        r = w.req
        px = q.ask if r.side is Side.BUY else q.bid
        avail = (q.ask_size if r.side is Side.BUY else q.bid_size)
        take = avail * self.cfg.marketable_max_take_fraction
        if w.limit is not None and ((r.side is Side.BUY and px > w.limit)
                                    or (r.side is Side.SELL and px < w.limit)):
            take = 0.0     # limit no longer marketable at this quote
        if w.tif == "FOK":
            if take >= w.remaining:
                out.append(self._mk_fill(coid, w, w.remaining, px, q.ts))
            else:
                out.append(VenueEvent("cancel_ack", coid, reason="fok_unfillable"))
                del self.working[coid]
            return out
        qty = min(take, w.remaining)
        if qty >= max(self.cfg.partial_fill_min_shares, 1e-9):
            out.append(self._mk_fill(coid, w, qty, px, q.ts))
        if w.tif == "IOC" and coid in self.working:
            if w.remaining > 1e-9:
                out.append(VenueEvent("cancel_ack", coid, reason="ioc_remainder"))
            self.working.pop(coid, None)
        elif w.remaining <= 1e-9:
            self.working.pop(coid, None)
        return out

    def on_trade(self, t: Trade) -> list[VenueEvent]:
        out: list[VenueEvent] = []
        for coid, w in list(self.working.items()):
            if (w.req.symbol != t.symbol or w.awaiting_first_quote
                    or w.limit is None or self.clock.now() < w.activate_at):
                continue
            side, limit = w.req.side, w.limit
            at_price = abs(t.price - limit) < 1e-9
            through = t.price < limit if side is Side.BUY else t.price > limit
            if through:
                if self._rng.random() > self.fill_probability:
                    continue                      # stressed fill probability
                out.append(self._mk_fill(coid, w, w.remaining, limit, t.ts))
                self.working.pop(coid, None)
            elif at_price:
                if w.queue_ahead > 0:
                    w.queue_ahead = max(0.0, w.queue_ahead - t.size)
                elif not self.cfg.require_trade_through:
                    if self._rng.random() > self.fill_probability:
                        continue
                    qty = min(t.size, w.remaining)
                    if qty >= self.cfg.partial_fill_min_shares:
                        out.append(self._mk_fill(coid, w, qty, limit, t.ts))
                        if w.remaining <= 1e-9:
                            self.working.pop(coid, None)
        return out

    # ------------- clock poll -------------
    def poll(self) -> list[VenueEvent]:
        now = self.clock.now()
        out: list[VenueEvent] = []
        for coid, w in list(self.working.items()):
            if not w.acked and now >= w.ack_at:
                w.acked = True
                out.append(VenueEvent("ack", coid))
            if w.cancel_at is not None and now >= w.cancel_at:
                out.append(VenueEvent("cancel_ack", coid, reason="cancel"))
                self.working.pop(coid, None)
        return out

    def _mk_fill(self, coid: str, w: _Working, qty: float, px: float,
                 ts: datetime) -> VenueEvent:
        w.filled += qty
        return VenueEvent("fill", coid, fill=Fill(
            w.req.symbol, w.req.side, qty, px, ts, coid, w.req.trace_id,
            partial=w.remaining > 1e-9))

"""OrderManager: full order-lifecycle state machine.

States and legal transitions:

  NEW ──► SUBMITTING ──► ACKNOWLEDGED ──► PARTIAL ──► FILLED
             │                │  │            │
             │                │  └─► CANCEL_PENDING ─► CANCELLED
             │                │            │
             │                │            └─(late fill)─► PARTIAL/FILLED
             └─► REJECTED     └─► REJECTED / EXPIRED

Guarantees:
  * Illegal transitions raise (fail loud in tests, journaled in production).
  * Exactly ONE active entry intent per symbol.
  * Buying power and risk are RESERVED at NEW and released on terminal states.
  * Cancellations are timer-driven from the Clock — independent of trade flow,
    so an order on a dead tape still expires.
  * Market orders activate only after simulated/observed latency and fill from
    the first quote AFTER activation (handled by the venue; the OM tracks
    activate_at).
  * Every transition is journaled and OUFS-logged with a monotonic revision.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable

from wliq.core.clock import Clock
from wliq.core.journal import OrderJournal
from wliq.core.logging import get_logger
from wliq.core.models import Fill, OrderRequest, Side
from wliq.core.oufs import NullOufs, OufsWriter

log = get_logger("order_manager")


class OState(str, enum.Enum):
    NEW = "NEW"
    SUBMITTING = "SUBMITTING"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCEL_PENDING = "CANCEL_PENDING"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


TERMINAL = {OState.FILLED, OState.CANCELLED, OState.REJECTED, OState.EXPIRED}

_LEGAL: dict[OState, set[OState]] = {
    OState.NEW: {OState.SUBMITTING, OState.REJECTED},
    OState.SUBMITTING: {OState.ACKNOWLEDGED, OState.REJECTED, OState.PARTIAL,
                        OState.FILLED,   # fast venues can ack+fill atomically
                        OState.EXPIRED},  # TTL elapsed with no ack ever seen
    OState.ACKNOWLEDGED: {OState.PARTIAL, OState.FILLED, OState.CANCEL_PENDING,
                          OState.REJECTED, OState.EXPIRED},
    OState.PARTIAL: {OState.PARTIAL, OState.FILLED, OState.CANCEL_PENDING,
                     OState.EXPIRED},
    OState.CANCEL_PENDING: {OState.CANCELLED, OState.PARTIAL, OState.FILLED},
    OState.CANCELLED: set(), OState.FILLED: set(),
    OState.REJECTED: set(), OState.EXPIRED: set(),
}


class IllegalTransition(RuntimeError):
    pass


@dataclass
class ManagedOrder:
    request: OrderRequest
    state: OState = OState.NEW
    filled_qty: float = 0.0
    avg_fill_price: float = 0.0
    reserved_risk_usd: float = 0.0
    reserved_notional_usd: float = 0.0
    created_at: datetime | None = None
    activate_at: datetime | None = None   # market orders: latency gate
    expire_at: datetime | None = None     # timer-driven cancel
    revision: int = 0
    history: list[tuple[str, str]] = field(default_factory=list)

    @property
    def remaining(self) -> float:
        return max(0.0, self.request.quantity - self.filled_qty)

    @property
    def is_entry(self) -> bool:
        return self.request.intent == "entry"


class OrderManager:
    def __init__(self, clock: Clock, journal: OrderJournal,
                 oufs: OufsWriter | None = None,
                 max_gross_notional_usd: float = float("inf")) -> None:
        self.clock = clock
        self.journal = journal
        self.oufs = oufs or NullOufs()
        self.max_gross_notional = max_gross_notional_usd
        self.orders: dict[str, ManagedOrder] = {}
        self.on_terminal: Callable[[ManagedOrder], None] | None = None

    # ---------------- reservations ----------------
    @property
    def reserved_notional(self) -> float:
        return sum(o.reserved_notional_usd for o in self.orders.values()
                   if o.state not in TERMINAL)

    @property
    def reserved_risk(self) -> float:
        return sum(o.reserved_risk_usd for o in self.orders.values()
                   if o.state not in TERMINAL)

    def active_entry_symbols(self) -> set[str]:
        return {o.request.symbol for o in self.orders.values()
                if o.is_entry and o.state not in TERMINAL}

    def has_active_entry(self, symbol: str) -> bool:
        return symbol in self.active_entry_symbols()

    def active_orders(self) -> list[ManagedOrder]:
        return [o for o in self.orders.values() if o.state not in TERMINAL]

    # ---------------- lifecycle ----------------
    def create(self, req: OrderRequest, ref_price: float,
               risk_usd: float, ttl_ms: int | None,
               activation_latency_ms: int = 0) -> ManagedOrder:
        """NEW: validate single-intent + reserve; caller then submits to venue."""
        if req.client_order_id in self.orders:
            raise IllegalTransition(f"duplicate coid {req.client_order_id}")
        if req.intent == "entry" and self.has_active_entry(req.symbol):
            raise IllegalTransition(f"active entry intent exists for {req.symbol}")
        notional = abs(req.quantity) * ref_price
        if self.reserved_notional + notional > self.max_gross_notional:
            raise IllegalTransition("gross notional reservation exceeded")
        now = self.clock.now()
        o = ManagedOrder(request=req, created_at=now,
                         reserved_risk_usd=risk_usd,
                         reserved_notional_usd=notional)
        if req.order_type == "MARKET" and activation_latency_ms > 0:
            o.activate_at = now + timedelta(milliseconds=activation_latency_ms)
        if ttl_ms is not None:
            o.expire_at = now + timedelta(milliseconds=ttl_ms)
        self.orders[req.client_order_id] = o
        self._log(o, "created")
        return o

    def _transition(self, o: ManagedOrder, to: OState, why: str = "") -> None:
        if to not in _LEGAL[o.state]:
            raise IllegalTransition(f"{o.request.client_order_id}: "
                                    f"{o.state.value} -> {to.value} ({why})")
        o.state = to
        o.revision += 1
        o.history.append((self.clock.now().isoformat(), to.value))
        self._log(o, why or to.value.lower())
        if to in TERMINAL:
            o.reserved_risk_usd = 0.0
            o.reserved_notional_usd = 0.0
            if self.on_terminal:
                self.on_terminal(o)

    def mark_submitting(self, coid: str) -> None:
        self._transition(self.orders[coid], OState.SUBMITTING, "submit_sent")

    def on_ack(self, coid: str) -> None:
        o = self.orders.get(coid)
        if o and o.state is OState.SUBMITTING:
            self._transition(o, OState.ACKNOWLEDGED, "venue_ack")

    def on_reject(self, coid: str, reason: str = "") -> None:
        o = self.orders.get(coid)
        if o and o.state not in TERMINAL:
            self._transition(o, OState.REJECTED, f"reject:{reason}")

    def on_fill(self, fill: Fill) -> ManagedOrder:
        """Streaming fill processing with correct cumulative averaging."""
        o = self.orders[fill.client_order_id]
        if o.state in TERMINAL and o.state is not OState.FILLED:
            # Late fill after cancel raced — accept it; broker truth wins.
            log.warning("late_fill_after_terminal", coid=fill.client_order_id,
                        state=o.state.value)
        prev_qty = o.filled_qty
        o.avg_fill_price = ((o.avg_fill_price * prev_qty
                             + fill.price * fill.quantity)
                            / max(prev_qty + fill.quantity, 1e-12))
        o.filled_qty = prev_qty + fill.quantity
        # proportional reservation release on partials
        frac_remaining = o.remaining / max(o.request.quantity, 1e-12)
        o.reserved_risk_usd *= frac_remaining if o.remaining > 0 else 0.0
        o.reserved_notional_usd *= frac_remaining if o.remaining > 0 else 0.0
        if o.remaining <= 1e-9:
            if o.state is not OState.FILLED:
                self._transition(o, OState.FILLED, "fill_complete")
        else:
            if o.state in (OState.SUBMITTING, OState.ACKNOWLEDGED,
                           OState.PARTIAL, OState.CANCEL_PENDING):
                if o.state is not OState.PARTIAL:
                    self._transition(o, OState.PARTIAL, "fill_partial")
                else:
                    o.revision += 1
        self.oufs.outcome("fill", coid=fill.client_order_id,
                          symbol=fill.symbol, qty=fill.quantity,
                          price=fill.price, cum_qty=o.filled_qty)
        return o

    def request_cancel(self, coid: str) -> bool:
        o = self.orders.get(coid)
        if o and o.state in (OState.ACKNOWLEDGED, OState.PARTIAL):
            self._transition(o, OState.CANCEL_PENDING, "cancel_requested")
            return True
        return False

    def on_cancel_ack(self, coid: str) -> None:
        o = self.orders.get(coid)
        if o and o.state is OState.CANCEL_PENDING:
            self._transition(o, OState.CANCELLED, "cancel_ack")

    # ---------------- timers (independent of trade events) ----------------
    def poll(self) -> list[str]:
        """Timer sweep. Returns coids whose cancel should be sent to the venue."""
        now = self.clock.now()
        to_cancel: list[str] = []
        for o in list(self.orders.values()):
            if o.state in TERMINAL:
                continue
            if o.expire_at and now >= o.expire_at:
                if o.state in (OState.ACKNOWLEDGED, OState.PARTIAL):
                    self._transition(o, OState.CANCEL_PENDING, "ttl_expired")
                    to_cancel.append(o.request.client_order_id)
                elif o.state is OState.SUBMITTING:
                    # never acked within TTL — expire locally, reconcile later
                    self._transition(o, OState.EXPIRED, "ttl_no_ack")
        return to_cancel

    # ---------------- recovery ----------------
    def restore(self, o: ManagedOrder) -> None:
        self.orders[o.request.client_order_id] = o

    def _log(self, o: ManagedOrder, event: str) -> None:
        log.info("order", coid=o.request.client_order_id,
                 symbol=o.request.symbol, state=o.state.value, evt=event,
                 filled=o.filled_qty, rev=o.revision)
        self.journal.record(event, o.request, o.state,  # type: ignore[arg-type]
                            {"filled": o.filled_qty, "rev": o.revision})
        self.oufs.decision("order_transition", coid=o.request.client_order_id,
                           state=o.state.value, event=event, rev=o.revision)

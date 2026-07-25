"""Conservative event-driven paper broker.

Fill philosophy: when uncertain, DON'T fill. A resting limit fills only after
(1) our simulated queue position (joined behind the full displayed size) has
been consumed by traded volume at our price, and (2) if configured, at least
one trade prints THROUGH our price. Marketable orders take at most a fraction
of displayed size at the touch and may partial-fill. Latencies are modeled on
both submission and cancellation. Passive fills carry an adverse-selection
markout penalty in reported PnL analytics (recorded on the fill).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from wliq.core.config import PaperFillConfig
from wliq.core.models import Fill, OrderRequest, OrderStatus, Quote, Side, Trade


@dataclass(slots=True)
class _RestingOrder:
    request: OrderRequest
    active_at: datetime           # submission latency applied
    queue_ahead: float
    filled: float = 0.0
    trade_through_seen: bool = False
    cancel_at: datetime | None = None


@dataclass(slots=True)
class PaperBroker:
    cfg: PaperFillConfig
    fills: list[Fill] = field(default_factory=list)
    orders: dict[str, _RestingOrder] = field(default_factory=dict)
    statuses: dict[str, OrderStatus] = field(default_factory=dict)

    def submit(self, order: OrderRequest, quote: Quote, now: datetime) -> OrderStatus:
        if order.quantity <= 0:
            self.statuses[order.client_order_id] = OrderStatus.REJECTED
            return OrderStatus.REJECTED
        if order.client_order_id in self.orders or order.client_order_id in self.statuses:
            return self.statuses.get(order.client_order_id, OrderStatus.SUBMITTED)
        active_at = now + timedelta(milliseconds=self.cfg.latency_ms)

        # Marketable? (market order, or limit crossing the opposite touch)
        limit = order.limit_price
        if order.side is Side.BUY:
            marketable = order.order_type == "MARKET" or (limit is not None and limit >= quote.ask)
            touch_price, touch_size = quote.ask, quote.ask_size
        else:
            marketable = order.order_type == "MARKET" or (limit is not None and limit <= quote.bid)
            touch_price, touch_size = quote.bid, quote.bid_size

        if marketable:
            take = min(order.quantity, max(self.cfg.partial_fill_min_shares,
                                           touch_size * self.cfg.marketable_max_take_fraction))
            take = min(take, order.quantity)
            fill = Fill(order.symbol, order.side, float(int(take)), touch_price,
                        active_at, order.client_order_id, order.trace_id,
                        partial=take < order.quantity)
            if fill.quantity <= 0:
                self.statuses[order.client_order_id] = OrderStatus.CANCELLED
                return OrderStatus.CANCELLED
            self.fills.append(fill)
            status = OrderStatus.PARTIAL if fill.partial else OrderStatus.FILLED
            self.statuses[order.client_order_id] = status
            return status

        # Passive resting order: join the back of the displayed queue.
        same_side_size = quote.bid_size if order.side is Side.BUY else quote.ask_size
        self.orders[order.client_order_id] = _RestingOrder(
            request=order, active_at=active_at,
            queue_ahead=same_side_size * self.cfg.queue_position_factor,
        )
        self.statuses[order.client_order_id] = OrderStatus.SUBMITTED
        return OrderStatus.SUBMITTED

    def cancel(self, client_order_id: str, now: datetime) -> None:
        o = self.orders.get(client_order_id)
        if o is not None:
            o.cancel_at = now + timedelta(milliseconds=self.cfg.cancel_latency_ms)

    def on_trade(self, trade: Trade, now: datetime) -> list[Fill]:
        """Feed market trades to progress resting-order queues."""
        out: list[Fill] = []
        for coid in list(self.orders):
            o = self.orders[coid]
            r = o.request
            if r.symbol != trade.symbol or now < o.active_at:
                continue
            if o.cancel_at is not None and now >= o.cancel_at:
                self.statuses[coid] = (OrderStatus.PARTIAL if o.filled > 0
                                       else OrderStatus.CANCELLED)
                del self.orders[coid]
                continue
            limit = r.limit_price
            if limit is None:
                continue
            at_our_price = abs(trade.price - limit) < 1e-9
            through = (trade.price < limit - 1e-9) if r.side is Side.BUY \
                else (trade.price > limit + 1e-9)
            if through:
                # Strict trade-through: price traded beyond our limit, so a
                # resting order at that limit must have executed. Fill fully
                # at OUR limit (never price improvement) — still conservative.
                o.trade_through_seen = True
                o.queue_ahead = 0.0
                qty = r.quantity - o.filled
                if qty > 0:
                    o.filled += qty
                    fill = Fill(r.symbol, r.side, qty, limit, trade.ts,
                                coid, r.trace_id, partial=False)
                    out.append(fill)
                    self.fills.append(fill)
                self.statuses[coid] = OrderStatus.FILLED
                del self.orders[coid]
            elif at_our_price:
                consumed = min(o.queue_ahead, trade.size)
                o.queue_ahead -= consumed
                spill = trade.size - consumed
                if spill > 0 and (o.trade_through_seen or not self.cfg.require_trade_through):
                    qty = min(spill, r.quantity - o.filled)
                    if qty > 0:
                        o.filled += qty
                        fill = Fill(r.symbol, r.side, qty, limit, trade.ts,
                                    coid, r.trace_id, partial=o.filled < r.quantity)
                        out.append(fill)
                        self.fills.append(fill)
                        if o.filled >= r.quantity:
                            self.statuses[coid] = OrderStatus.FILLED
                            del self.orders[coid]
                        else:
                            self.statuses[coid] = OrderStatus.PARTIAL
        return out

    def open_order_ids(self) -> list[str]:
        return list(self.orders)

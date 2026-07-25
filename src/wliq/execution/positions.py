"""Position manager and exit engine.

Exits (checked in priority order on every quote / feature update):
  1. hard stop (volatility-anchored, set at entry)
  2. target
  3. time stop
  4. spread deterioration
  5. adverse microprice persistence
  6. resumed price impact against a reversal position
  7. peer-confirmation invalidation (peers begin confirming the move we faded)
  8. emergency flatten
No averaging down, ever.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from wliq.core.models import (ClosedTrade, FeatureVector, Fill, GovernorAction,
                              Position, Quote, Side, Signal)


@dataclass(slots=True)
class ExitInstruction:
    symbol: str
    side: Side          # side of the closing order
    quantity: float
    reason: str
    trace_id: str


@dataclass(slots=True)
class PositionManager:
    max_spread_bps_exit: float = 15.0
    adverse_micro_bps: float = 3.0
    adverse_micro_persist: int = 5
    closed: list[ClosedTrade] = field(default_factory=list)
    _positions: dict[str, Position] = field(default_factory=dict)
    _micro_adverse_count: dict[str, int] = field(default_factory=dict)
    emergency: bool = False

    @property
    def positions(self) -> dict[str, Position]:
        return self._positions

    def on_entry_fill(self, fill: Fill, signal: Signal) -> None:
        pos = self._positions.get(fill.symbol)
        if pos is not None:
            # partial fill accumulation at (possibly) multiple prices
            total = pos.quantity + fill.quantity
            pos.entry_price = (pos.entry_price * pos.quantity
                               + fill.price * fill.quantity) / total
            pos.quantity = total
            return
        self._positions[fill.symbol] = Position(
            symbol=fill.symbol, side=fill.side, quantity=fill.quantity,
            entry_price=fill.price, opened_at=fill.ts,
            stop_price=signal.stop_price, target_price=signal.target_price,
            signal_score=signal.score, trace_id=signal.trace_id,
            action=signal.action,
        )

    def check_exits(self, quote: Quote, feature: FeatureVector | None,
                    now: datetime, max_holding_seconds: int) -> ExitInstruction | None:
        pos = self._positions.get(quote.symbol)
        if pos is None:
            return None
        mark = quote.mid
        pos.update_excursions(mark)
        close_side = pos.side.opposite

        def exit_(reason: str) -> ExitInstruction:
            return ExitInstruction(pos.symbol, close_side, pos.quantity, reason, pos.trace_id)

        if self.emergency:
            return exit_("emergency_flatten")
        if pos.side is Side.BUY and mark <= pos.stop_price:
            return exit_("hard_stop")
        if pos.side is Side.SELL and mark >= pos.stop_price:
            return exit_("hard_stop")
        if pos.side is Side.BUY and mark >= pos.target_price:
            return exit_("target")
        if pos.side is Side.SELL and mark <= pos.target_price:
            return exit_("target")
        if (now - pos.opened_at).total_seconds() > max_holding_seconds:
            return exit_("time_stop")
        if quote.spread_bps > self.max_spread_bps_exit:
            return exit_("spread_deterioration")
        if feature is not None:
            adverse_micro = -pos.side.sign * feature.microprice_bps
            if adverse_micro > self.adverse_micro_bps:
                c = self._micro_adverse_count.get(pos.symbol, 0) + 1
                self._micro_adverse_count[pos.symbol] = c
                if c >= self.adverse_micro_persist:
                    return exit_("adverse_microprice")
            else:
                self._micro_adverse_count[pos.symbol] = 0
            if pos.action is GovernorAction.REVERSAL:
                move_dir = -pos.side.sign  # direction of the move we faded
                if feature.impact_decay < -0.3 and move_dir * feature.flow_imbalance > 0.2:
                    return exit_("impact_resumed")
                if move_dir * feature.peer_confirmation > 0.6:
                    return exit_("peer_invalidation")
        return None

    def on_exit_fill(self, fill: Fill, reason: str) -> ClosedTrade | None:
        """Cumulative accounting: every partial exit accrues realized PnL and
        exit VWAP on the position; the ClosedTrade emitted at flat reflects the
        FULL round trip, not the last chunk."""
        pos = self._positions.get(fill.symbol)
        if pos is None:
            return None
        qty = min(fill.quantity, pos.quantity)
        pnl = pos.side.sign * (fill.price - pos.entry_price) * qty
        pos.realized_pnl = getattr(pos, "realized_pnl", 0.0) + pnl
        pos.exited_qty = getattr(pos, "exited_qty", 0.0) + qty
        pos.exit_notional = getattr(pos, "exit_notional", 0.0) + qty * fill.price
        pos.quantity -= qty
        if pos.quantity > 1e-9:
            return None  # partial exit; realized PnL retained on the position
        exit_vwap = pos.exit_notional / max(pos.exited_qty, 1e-12)
        trade = ClosedTrade(
            symbol=pos.symbol, side=pos.side, quantity=pos.exited_qty,
            entry_price=pos.entry_price, exit_price=exit_vwap,
            opened_at=pos.opened_at, closed_at=fill.ts, pnl=pos.realized_pnl,
            exit_reason=reason, trace_id=pos.trace_id,
            signal_score=pos.signal_score, action=pos.action,
            max_favorable=pos.max_favorable, max_adverse=pos.max_adverse,
        )
        self.closed.append(trade)
        del self._positions[fill.symbol]
        self._micro_adverse_count.pop(fill.symbol, None)
        return trade

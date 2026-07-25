"""Risk manager: pre-trade gates, sizing, and daily/exposure accounting.

Every gate is configurable in YAML. approve() returns (bool, reason) and is
called after the governor and scoring engine — it is the final veto.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from wliq.core.config import RiskConfig
from wliq.core.models import FeatureVector, Position, Signal


class RiskManager:
    def __init__(self, cfg: RiskConfig) -> None:
        self.cfg = cfg
        self.positions: dict[str, Position] = {}
        self.realized_pnl = 0.0
        self.consecutive_losses = 0
        self.symbol_cooldown_until: dict[str, datetime] = {}
        self.strategy_cooldown_until: datetime | None = None
        self.retries: dict[str, int] = {}

    # ------------------------------------------------------------- accounting
    def on_trade_closed(self, symbol: str, pnl: float, now: datetime) -> None:
        self.realized_pnl += pnl
        if pnl < 0:
            self.consecutive_losses += 1
            self.symbol_cooldown_until[symbol] = now + timedelta(
                seconds=self.cfg.symbol_cooldown_seconds)
        else:
            self.consecutive_losses = 0
        if self.consecutive_losses >= self.cfg.max_consecutive_losses:
            self.strategy_cooldown_until = now + timedelta(
                seconds=self.cfg.strategy_cooldown_seconds)
            self.consecutive_losses = 0
        self.positions.pop(symbol, None)

    def gross_exposure(self, marks: dict[str, float]) -> float:
        return sum(p.quantity * marks.get(s, p.entry_price)
                   for s, p in self.positions.items())

    # ------------------------------------------------------------------ gates
    def approve(self, signal: Signal, feature: FeatureVector,
                marks: dict[str, float] | None = None,
                now: datetime | None = None) -> tuple[bool, str]:
        now = now or datetime.now(timezone.utc)
        marks = marks or {}
        if self.cfg.kill_switch:
            return False, "kill_switch"
        if self.realized_pnl <= -self.cfg.max_daily_loss_usd:
            return False, "daily_loss_limit"
        if self.strategy_cooldown_until and now < self.strategy_cooldown_until:
            return False, "strategy_cooldown"
        cd = self.symbol_cooldown_until.get(signal.symbol)
        if cd and now < cd:
            return False, "symbol_cooldown"
        if signal.symbol in self.positions:
            return False, "position_exists"
        if len(self.positions) >= self.cfg.max_positions:
            return False, "max_positions"
        if feature.spread_bps > self.cfg.max_spread_bps:
            return False, "spread_too_wide"
        age_ms = (now - feature.ts).total_seconds() * 1000
        if age_ms > self.cfg.stale_data_ms:
            return False, "stale_feature"
        if self.retries.get(signal.symbol, 0) > self.cfg.max_order_retries:
            return False, "max_retries"
        qty = self.size(signal)
        if qty <= 0:
            return False, "zero_size"
        notional = qty * signal.reference_price
        if notional > self.cfg.max_notional_per_symbol_usd:
            return False, "symbol_notional_limit"
        if self.gross_exposure(marks) + notional > self.cfg.max_gross_exposure_usd:
            return False, "gross_exposure_limit"
        return True, "approved"

    def size(self, signal: Signal) -> float:
        stop_distance = abs(signal.reference_price - signal.stop_price)
        if stop_distance <= 0:
            return 0.0
        risk_qty = self.cfg.risk_per_trade_usd / stop_distance
        notional_qty = self.cfg.max_notional_per_symbol_usd / signal.reference_price
        return float(int(min(risk_qty, notional_qty, self.cfg.max_position_shares)))

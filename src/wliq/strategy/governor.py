"""Strategy governor: REVERSAL / CONTINUATION / NO_TRADE.

The governor is a pure gate layer that runs BEFORE signal scoring. It answers
"is this environment one where the strategy is allowed to act at all, and in
which direction of logic?" — the scoring engine then decides whether a
specific trade clears the bar. Every rejection carries reasons for the log.
"""
from __future__ import annotations

from datetime import datetime

from wliq.broker.health import DataHealthMonitor
from wliq.core.config import GovernorConfig, StrategyConfig
from wliq.core.models import FeatureVector, GovernorAction, GovernorDecision


class StrategyGovernor:
    def __init__(self, cfg: GovernorConfig, strat: StrategyConfig,
                 health: DataHealthMonitor, session_minutes: float = 390.0) -> None:
        self.cfg = cfg
        self.strat = strat
        self.health = health
        self.session_minutes = session_minutes
        self.market_breadth: float = 0.0   # updated externally; [-1, 1]
        self.sector_momentum: float = 0.0  # sector ETF window return, signed

    def update_context(self, market_breadth: float, sector_momentum: float) -> None:
        self.market_breadth = market_breadth
        self.sector_momentum = sector_momentum

    def decide(self, f: FeatureVector, now: datetime) -> GovernorDecision:
        reasons: list[str] = []

        # ---- hard disables (apply to everything) ----
        if self.cfg.require_healthy_stream:
            ok, why = self.health.is_healthy(f.symbol, now)
            if not ok:
                return GovernorDecision(GovernorAction.NO_TRADE, (why,))
        if f.symbol in self.cfg.catalyst_blacklist:
            return GovernorDecision(GovernorAction.NO_TRADE, ("verified_catalyst",))
        if f.symbol in self.cfg.earnings_blacklist:
            return GovernorDecision(GovernorAction.NO_TRADE, ("earnings_window",))
        if f.minutes_since_open < self.strat.no_trade_first_seconds / 60:
            return GovernorDecision(GovernorAction.NO_TRADE, ("open_blackout",))
        if f.minutes_since_open > self.session_minutes - self.strat.no_trade_last_seconds / 60:
            return GovernorDecision(GovernorAction.NO_TRADE, ("close_blackout",))
        if not (self.cfg.min_realized_vol_bps <= f.realized_vol_bps
                <= self.cfg.max_realized_vol_bps):
            return GovernorDecision(GovernorAction.NO_TRADE, ("vol_out_of_bounds",))
        if f.spread_pctile > self.cfg.max_spread_pctile_for_entry:
            return GovernorDecision(GovernorAction.NO_TRADE, ("spread_unstable",))
        if f.event_count < self.strat.min_events:
            return GovernorDecision(GovernorAction.NO_TRADE, ("insufficient_events",))
        if abs(f.residual_z) < self.strat.residual_z_entry:
            return GovernorDecision(GovernorAction.NO_TRADE, ("residual_below_entry",))
        if f.volume_z < self.strat.min_volume_z:
            return GovernorDecision(GovernorAction.NO_TRADE, ("volume_unremarkable",))

        move_dir = 1.0 if f.residual_z > 0 else -1.0

        # ---- reversal-specific disables ----
        reversal_ok = True
        if abs(self.market_breadth) > self.cfg.max_breadth_for_reversal:
            reversal_ok = False
            reasons.append("breadth_trend_day")
        if move_dir * self.sector_momentum > 0 and abs(self.sector_momentum) > 0.002:
            reversal_ok = False
            reasons.append("sector_momentum_confirms")
        if f.peer_confirmation > self.strat.max_peer_confirmation_for_reversal:
            reversal_ok = False
            reasons.append("peers_confirm_move")
        if f.impact_decay <= 0:
            reversal_ok = False
            reasons.append("impact_still_rising")

        if reversal_ok:
            return GovernorDecision(GovernorAction.REVERSAL, tuple(reasons))

        # ---- continuation path ----
        cont_ok = (
            f.peer_confirmation >= self.strat.min_peer_confirmation_for_continuation
            and f.impact_decay < 0                              # impact increasing
            and move_dir * f.flow_imbalance > 0.15              # flow agrees
            and move_dir * f.replenishment_skew > 0             # opposing side thinning
            and f.spread_pctile < 0.75
        )
        if cont_ok:
            return GovernorDecision(GovernorAction.CONTINUATION, tuple(reasons))
        reasons.append("no_continuation_confluence")
        return GovernorDecision(GovernorAction.NO_TRADE, tuple(reasons))

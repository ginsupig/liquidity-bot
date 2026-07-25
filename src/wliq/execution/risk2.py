"""Sprint 5 — portfolio risk layer on top of the v0.2 per-trade RiskManager.

Adds (all evaluated BEFORE the base gates):
  * gross/net portfolio exposure limits including RESERVED (pending) exposure
  * sector bucket limits (symbol -> sector map from config)
  * correlation-cluster limit: max concurrent positions within one peer cluster
  * volatility-targeted sizing: risk budget scaled so expected daily vol of
    the position book approaches a target; per-trade size shrinks in high-vol
    regimes and grows (capped) in low-vol
  * drawdown governor: realized intraday DD scales the risk multiplier down in
    steps; a hard step halts (writes halt to the ledger)
  * regime-aware sizing hook (vol_regime from features)
The persistent RiskLedger restores all of this across restarts.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from wliq.core.logging import get_logger
from wliq.core.models import FeatureVector, Signal
from wliq.execution.ledger import RiskLedger

log = get_logger("risk2")


@dataclass
class PortfolioRiskConfig:
    max_gross_exposure_usd: float = 5000.0
    max_net_exposure_usd: float = 4000.0
    max_positions_per_sector: int = 2
    max_positions_per_cluster: int = 1
    sector_map: dict[str, str] = field(default_factory=dict)
    clusters: list[list[str]] = field(default_factory=list)
    target_daily_vol_usd: float = 60.0
    vol_size_floor: float = 0.4
    vol_size_cap: float = 1.5
    dd_soft_usd: float = 50.0     # scale to 0.5x
    dd_hard_usd: float = 90.0     # scale to 0.25x
    dd_halt_usd: float = 120.0    # halt for the day
    regime_multipliers: tuple[float, float, float] = (1.2, 1.0, 0.6)


class PortfolioRisk:
    def __init__(self, cfg: PortfolioRiskConfig, ledger: RiskLedger) -> None:
        self.cfg = cfg
        self.ledger = ledger
        self.peak_pnl = 0.0
        self.realized = 0.0
        self.halted = False

    # ---------- persistence ----------
    def restore(self) -> None:
        day = self.ledger.load_day()
        self.realized = day["realized_pnl"]
        self.peak_pnl = max(0.0, self.realized)
        self.halted = day["halted"]
        log.info("risk2_restored", realized=self.realized, halted=self.halted)

    def on_trade_closed(self, pnl: float) -> None:
        self.realized += pnl
        self.peak_pnl = max(self.peak_pnl, self.realized)
        if (self.peak_pnl - self.realized) >= self.cfg.dd_halt_usd and not self.halted:
            self.halted = True
            self.ledger.append("halt", reason="drawdown_governor",
                               dd=self.peak_pnl - self.realized)
            log.error("drawdown_halt", dd=self.peak_pnl - self.realized)

    # ---------- gates ----------
    def approve(self, sig: Signal, f: FeatureVector,
                open_positions: dict[str, object],
                gross_exposure_usd: float, net_exposure_usd: float,
                reserved_notional_usd: float) -> tuple[bool, str]:
        c = self.cfg
        if self.halted:
            return False, "drawdown_halted"
        if gross_exposure_usd + reserved_notional_usd >= c.max_gross_exposure_usd:
            return False, "gross_exposure"
        if abs(net_exposure_usd) >= c.max_net_exposure_usd:
            return False, "net_exposure"
        sector = c.sector_map.get(sig.symbol)
        if sector:
            n = sum(1 for s in open_positions if c.sector_map.get(s) == sector)
            if n >= c.max_positions_per_sector:
                return False, f"sector_limit:{sector}"
        for cluster in c.clusters:
            if sig.symbol in cluster:
                n = sum(1 for s in open_positions if s in cluster)
                if n >= c.max_positions_per_cluster:
                    return False, "correlation_cluster_limit"
        return True, "approved"

    # ---------- sizing ----------
    def size_multiplier(self, f: FeatureVector) -> float:
        c = self.cfg
        m = c.regime_multipliers[min(max(f.vol_regime, 0), 2)]
        # volatility targeting: shrink risk when symbol vol is elevated so that
        # per-position expected daily vol stays near target
        if f.realized_vol_bps > 1e-9:
            ref_vol = (self.cfg.target_daily_vol_usd /
                       max(self.cfg.max_gross_exposure_usd, 1e-9)) * 1e4
            m *= max(c.vol_size_floor,
                     min(c.vol_size_cap, ref_vol / f.realized_vol_bps))
        dd = self.peak_pnl - self.realized
        if dd >= c.dd_hard_usd:
            m *= 0.25
        elif dd >= c.dd_soft_usd:
            m *= 0.5
        return max(0.0, min(c.vol_size_cap, m))

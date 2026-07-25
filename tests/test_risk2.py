from datetime import datetime, timezone

from tests.test_adaptive_analytics import fv
from wliq.core.models import GovernorAction, Side, Signal
from wliq.execution.ledger import RiskLedger
from wliq.execution.risk2 import PortfolioRisk, PortfolioRiskConfig

T0 = datetime(2026, 7, 20, 15, 0, tzinfo=timezone.utc)
SIG = Signal("MU", T0, Side.SELL, GovernorAction.REVERSAL, 4.0, 100.0,
             100.5, 99.5, "tr")


def mk(tmp_path, **kw):
    cfg = PortfolioRiskConfig(sector_map={"MU": "semis", "WDC": "semis",
                                          "SNDK": "semis"},
                              clusters=[["MU", "WDC", "SNDK"]], **kw)
    return PortfolioRisk(cfg, RiskLedger(tmp_path / "r.jsonl"))


def test_sector_and_cluster_limits(tmp_path):
    r = mk(tmp_path, max_positions_per_cluster=1)
    ok, why = r.approve(SIG, fv(), {"WDC": object()}, 1000, 500, 0)
    assert not ok and why == "correlation_cluster_limit"


def test_gross_includes_reserved(tmp_path):
    r = mk(tmp_path, max_gross_exposure_usd=2000)
    ok, why = r.approve(SIG, fv(), {}, 1500, 0, 600)   # 1500+600 > 2000
    assert not ok and why == "gross_exposure"


def test_drawdown_governor_halts_and_persists(tmp_path):
    r = mk(tmp_path)
    r.on_trade_closed(80.0)      # peak 80
    r.on_trade_closed(-50.0)     # dd 50 -> soft
    assert not r.halted and r.size_multiplier(fv()) < 1.0
    r.on_trade_closed(-75.0)     # dd 125 -> halt
    assert r.halted
    r2 = mk(tmp_path)
    r2.restore()
    assert r2.halted             # survives restart via ledger


def test_regime_sizing(tmp_path):
    r = mk(tmp_path)
    hi = r.size_multiplier(fv(vol_regime=2, realized_vol_bps=60.0))
    lo = r.size_multiplier(fv(vol_regime=0, realized_vol_bps=10.0))
    assert hi < lo

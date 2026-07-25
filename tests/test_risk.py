from datetime import timedelta

from wliq.core.config import RiskConfig
from wliq.core.models import GovernorAction, Side, Signal
from wliq.execution.risk import RiskManager
from tests.conftest import T0
from tests.test_strategy import fv


def sig(symbol: str = "MU") -> Signal:
    return Signal(symbol, T0, Side.SELL, GovernorAction.REVERSAL, 4.0,
                  100.0, 100.22, 99.60, "trace1")


def test_daily_loss_halt():
    r = RiskManager(RiskConfig(max_daily_loss_usd=50))
    r.realized_pnl = -60
    ok, reason = r.approve(sig(), fv(), now=T0)
    assert not ok and reason == "daily_loss_limit"


def test_kill_switch():
    r = RiskManager(RiskConfig(kill_switch=True))
    ok, reason = r.approve(sig(), fv(), now=T0)
    assert not ok and reason == "kill_switch"


def test_spread_gate():
    r = RiskManager(RiskConfig(max_spread_bps=5))
    ok, reason = r.approve(sig(), fv(spread_bps=9.0), now=T0)
    assert not ok and reason == "spread_too_wide"


def test_stale_feature_gate():
    r = RiskManager(RiskConfig(stale_data_ms=500))
    ok, reason = r.approve(sig(), fv(), now=T0 + timedelta(seconds=2))
    assert not ok and reason == "stale_feature"


def test_consecutive_loss_cooldown():
    r = RiskManager(RiskConfig(max_consecutive_losses=2))
    r.on_trade_closed("MU", -10, T0)
    r.on_trade_closed("WDC", -10, T0)
    ok, reason = r.approve(sig("NVDA"), fv(symbol="NVDA"), now=T0)
    assert not ok and reason == "strategy_cooldown"


def test_sizing_respects_risk_and_notional():
    r = RiskManager(RiskConfig(risk_per_trade_usd=22, max_notional_per_symbol_usd=100_000,
                               max_position_shares=10_000))
    # stop distance 0.22 -> 100 shares
    assert r.size(sig()) == 100

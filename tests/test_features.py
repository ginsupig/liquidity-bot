from datetime import timedelta

from wliq.core.models import Side
from wliq.features.engine import FeatureEngine
from tests.conftest import T0, q, t


def make_engine() -> FeatureEngine:
    return FeatureEngine(120, {"MU": ["WDC"]}, market_etf="QQQ", sector_etf="SMH")


def test_trade_classification_quote_relative():
    e = make_engine()
    e.on_quote(q("MU", 0, 100.00, 100.02))
    e.on_trade(t("MU", 1, 100.02, 100))   # at ask -> BUY
    e.on_trade(t("MU", 2, 100.00, 100))   # at bid -> SELL
    trades = list(e.trades["MU"])
    assert trades[0].aggressor is Side.BUY
    assert trades[1].aggressor is Side.SELL


def test_residual_positive_when_stock_moves_alone():
    e = make_engine()
    for i in range(120):
        px = 100.0 + i * 0.02          # MU rips
        e.on_quote(q("MU", i, px, px + 0.02))
        e.on_quote(q("QQQ", i, 500.0, 500.02))   # market flat
        e.on_quote(q("SMH", i, 250.0, 250.02))   # sector flat
        e.on_quote(q("WDC", i, 80.0, 80.02))     # peer flat
        e.on_trade(t("MU", i + 0.5, px + 0.02, 200, Side.BUY))
    f = e.compute("MU", T0 + timedelta(seconds=120))
    assert f is not None
    assert f.residual_return > 0
    assert f.residual_z > 1.0
    assert f.flow_imbalance > 0.9


def test_replenishment_skew_sign():
    e = make_engine()
    e.on_quote(q("MU", 0, 100.00, 100.02, bs=500, asz=500))
    e.on_trade(t("MU", 1, 100.02, 400, Side.BUY))   # consume ask
    e.on_quote(q("MU", 2, 100.00, 100.02, bs=500, asz=900))  # ask replenishes hard
    f = e.compute("MU", T0 + timedelta(seconds=3))
    assert f is not None
    assert f.replenishment_skew < 0  # ask side replenishing more than bid


def test_feature_none_without_quotes():
    e = make_engine()
    assert e.compute("MU", T0) is None

from datetime import datetime, timedelta, timezone

from wliq.analytics.execution import ExecutionAnalytics
from wliq.core.models import FeatureVector, Fill, Quote, Side
from wliq.execution.adaptive import ExecPlan, next_retry, plan_entry

T0 = datetime(2026, 7, 20, 15, 0, tzinfo=timezone.utc)


def fv(**kw):
    base = dict(symbol="MU", ts=T0, mid=100.0, spread_bps=3.0,
                spread_pctile=0.5, residual_return=0.0, residual_z=0.0,
                flow_imbalance=0.0, impact_decay=0.0, replenishment_skew=0.0,
                microprice_bps=0.0, peer_confirmation=0.0, depth_imbalance=0.0,
                near_touch_ratio=0.0, queue_depletion=0.0, volume_z=0.0,
                vwap_distance_bps=0.0, realized_vol_bps=20.0,
                minutes_since_open=60.0, event_count=100)
    base.update(kw)
    return FeatureVector(**base)


Q = Quote("MU", T0, 99.98, 100.02, 500, 500)


def test_high_urgency_crosses_ioc():
    p = plan_entry(fv(), Q, Side.SELL, urgency=0.95, trade_rate_shares_s=50)
    assert p.style == "cross" and p.tif == "IOC" and p.limit_price == 99.98


def test_low_urgency_passive_at_touch():
    p = plan_entry(fv(), Q, Side.SELL, urgency=0.3, trade_rate_shares_s=50)
    assert p.style == "passive" and p.limit_price == 100.02


def test_retry_ladder_respects_chase_cap():
    p0 = ExecPlan("passive", 100.02, "DAY", 4000, 0.4, max_chase_bps=1.0)
    p1 = next_retry(p0, fv(), Q, Side.SELL, decision_price=100.02)
    assert p1 is not None and p1.style == "join"
    # cross would slip ~4bps > 1bps cap -> ladder stops
    p2 = next_retry(p1, fv(), Q, Side.SELL, decision_price=100.02)
    assert p2 is None


def test_markouts_measure_adverse_selection():
    xq = ExecutionAnalytics()
    for i in range(40):
        xq.on_mid("MU", T0 + timedelta(seconds=i), 100.0 + 0.01 * i)
    fill = Fill("MU", Side.BUY, 100, 100.03, T0 + timedelta(seconds=2),
                "c1", "t")
    xq.on_fill(fill, mid_at_fill=100.02, decision_price=100.00,
               decision_ts=T0)
    s = xq.summary()
    assert s["n_fills"] == 1
    assert s["implementation_shortfall_bps"] > 0        # paid above decision
    assert s["adverse_selection_5s_bps"] > 0            # mid ran away after
    assert s["realized_spread_5s_bps"] < s["effective_spread_bps"]

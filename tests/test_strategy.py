from wliq.core.config import StrategyConfig
from wliq.core.models import FeatureVector, GovernorAction, Side
from wliq.strategy.exhaustion import ResidualExhaustionStrategy
from tests.conftest import T0


def fv(**kw) -> FeatureVector:
    base = dict(symbol="MU", ts=T0, mid=100.0, spread_bps=3.0, spread_pctile=0.4,
                residual_return=0.004, residual_z=3.0, flow_imbalance=0.6,
                impact_decay=0.5, replenishment_skew=-1.0, microprice_bps=-2.0,
                peer_confirmation=0.05, depth_imbalance=-0.3, near_touch_ratio=0.3,
                queue_depletion=0.0, volume_z=2.5, vwap_distance_bps=25.0,
                realized_vol_bps=30.0, minutes_since_open=60.0, event_count=200)
    base.update(kw)
    return FeatureVector(**base)


def test_reversal_shorts_up_burst():
    s = ResidualExhaustionStrategy(StrategyConfig())
    sig = s.evaluate(fv(), GovernorAction.REVERSAL)
    assert sig is not None
    assert sig.side is Side.SELL
    assert sig.stop_price > sig.reference_price > sig.target_price


def test_reversal_longs_down_burst():
    s = ResidualExhaustionStrategy(StrategyConfig())
    sig = s.evaluate(fv(residual_z=-3.0, residual_return=-0.004,
                        flow_imbalance=-0.6, replenishment_skew=1.0,
                        microprice_bps=2.0, depth_imbalance=0.3),
                     GovernorAction.REVERSAL)
    assert sig is not None
    assert sig.side is Side.BUY


def test_no_trade_action_yields_none():
    s = ResidualExhaustionStrategy(StrategyConfig())
    assert s.evaluate(fv(), GovernorAction.NO_TRADE) is None


def test_cost_filter_blocks_thin_edge():
    s = ResidualExhaustionStrategy(StrategyConfig())
    # tiny residual move cannot clear wide spread
    sig = s.evaluate(fv(residual_return=0.0002, spread_bps=7.0),
                     GovernorAction.REVERSAL)
    assert sig is None


def test_continuation_goes_with_move():
    s = ResidualExhaustionStrategy(StrategyConfig())
    sig = s.evaluate(fv(impact_decay=-0.5, peer_confirmation=0.8,
                        replenishment_skew=1.2, microprice_bps=2.0,
                        depth_imbalance=0.4),
                     GovernorAction.CONTINUATION)
    assert sig is not None
    assert sig.side is Side.BUY  # up-burst continuation = long

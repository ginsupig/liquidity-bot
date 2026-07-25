from dataclasses import replace
from datetime import datetime, timedelta, timezone

from wliq.core.config import StrategyConfig
from wliq.core.models import FeatureVector, GovernorAction, Side
from wliq.strategy.fsm import FsmConfig, SignalStateMachine, SState

T0 = datetime(2026, 7, 20, 15, 0, tzinfo=timezone.utc)


def fv(sec: float, **kw) -> FeatureVector:
    base = dict(symbol="MU", ts=T0 + timedelta(seconds=sec), mid=100.0,
                spread_bps=2.0, spread_pctile=0.5, residual_return=0.004,
                residual_z=3.0, flow_imbalance=0.1, impact_decay=0.3,
                replenishment_skew=-1.0, microprice_bps=-1.5,
                peer_confirmation=0.1, depth_imbalance=0.0,
                near_touch_ratio=0.1, queue_depletion=0.0, volume_z=2.5,
                vwap_distance_bps=20.0, realized_vol_bps=20.0,
                minutes_since_open=60.0, event_count=500,
                residual_velocity=0.2, residual_acceleration=-0.1,
                replenishment_confidence=0.8)
    base.update(kw)
    return FeatureVector(**base)


def mk():
    return SignalStateMachine(
        FsmConfig(impulse_z=2.25, confirm_wait_s=5, retest_tolerance_bps=6),
        StrategyConfig())


def test_full_path_to_entry_signal():
    m = mk()
    # IDLE -> IMPULSE: strong up-burst with velocity
    assert m.step(fv(0, residual_velocity=3.0), GovernorAction.REVERSAL) is None
    assert m.state("MU") is SState.IMPULSE
    # IMPULSE -> CONFIRM_WAIT
    m.step(fv(1, residual_velocity=3.0, mid=100.30), GovernorAction.REVERSAL)
    assert m.state("MU") is SState.CONFIRM_WAIT
    # dwell then exhaustion: velocity dead, impact decaying, flow fading
    m.step(fv(7, residual_velocity=0.1, mid=100.28, flow_imbalance=0.1),
           GovernorAction.REVERSAL)
    assert m.state("MU") is SState.EXHAUSTION
    # retest near the extreme...
    m.step(fv(9, residual_velocity=0.1, mid=100.295), GovernorAction.REVERSAL)
    # ...then fade off it with microprice against -> FAILED_RETEST
    m.step(fv(12, residual_velocity=-0.2, mid=100.15, microprice_bps=-2.0),
           GovernorAction.REVERSAL)
    assert m.state("MU") is SState.FAILED_RETEST
    sig = m.step(fv(13, residual_velocity=-0.3, mid=100.12,
                    microprice_bps=-2.0), GovernorAction.REVERSAL)
    assert sig is not None and sig.side is Side.SELL
    assert m.state("MU") is SState.ENTRY
    m.on_entry_fill("MU", T0 + timedelta(seconds=14))
    assert m.state("MU") is SState.MANAGE
    m.on_position_closed("MU", T0 + timedelta(seconds=200))
    assert m.state("MU") is SState.COOLDOWN


def test_fast_peer_confirmation_invalidates():
    m = mk()
    m.step(fv(0, residual_velocity=3.0), GovernorAction.REVERSAL)
    m.step(fv(1, residual_velocity=3.0), GovernorAction.REVERSAL)
    # peers confirmed 6s after impulse -> common-factor move, stand down
    m.step(fv(8, peer_confirm_latency_s=6.0), GovernorAction.REVERSAL)
    assert m.state("MU") is SState.IDLE


def test_governor_no_trade_disarms():
    m = mk()
    m.step(fv(0, residual_velocity=3.0), GovernorAction.REVERSAL)
    m.step(fv(1, residual_velocity=3.0), GovernorAction.NO_TRADE)
    assert m.state("MU") is SState.IDLE


def test_entry_failed_cools_down():
    m = mk()
    m.state("MU")                      # materialize the track
    m.tracks["MU"].state = SState.ENTRY
    m.on_entry_failed("MU", T0)
    assert m.state("MU") is SState.COOLDOWN

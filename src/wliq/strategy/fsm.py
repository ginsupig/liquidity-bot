"""Sprint 2 — per-symbol signal state machine replacing additive triggering.

    IDLE -> IMPULSE -> CONFIRM_WAIT -> EXHAUSTION -> FAILED_RETEST -> ENTRY
                                                                       |
                 COOLDOWN <- EXIT <- MANAGE <-------------------------+

Transitions are driven by FeatureVector v2 + GovernorAction; every state has
timeouts and invalidation paths back to IDLE (journaled). The additive score
from ResidualExhaustionStrategy is preserved as a *quality/ranking* input for
sizing and the ML layer — it no longer gates entry timing by itself.
"""
from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, field
from datetime import datetime

from wliq.core.config import StrategyConfig
from wliq.core.logging import get_logger
from wliq.core.models import (FeatureVector, GovernorAction, Side, Signal)

log = get_logger("fsm")


class SState(str, enum.Enum):
    IDLE = "IDLE"
    IMPULSE = "IMPULSE"
    CONFIRM_WAIT = "CONFIRM_WAIT"
    EXHAUSTION = "EXHAUSTION"
    FAILED_RETEST = "FAILED_RETEST"
    ENTRY = "ENTRY"
    MANAGE = "MANAGE"
    EXIT = "EXIT"
    COOLDOWN = "COOLDOWN"


@dataclass
class FsmConfig:
    impulse_z: float = 2.25
    impulse_min_velocity: float = 1.0        # bps/s
    confirm_wait_s: float = 8.0              # min dwell before exhaustion valid
    impulse_timeout_s: float = 90.0
    exhaustion_timeout_s: float = 120.0
    retest_timeout_s: float = 60.0
    retest_tolerance_bps: float = 6.0        # how close to extreme counts as retest
    cooldown_s: float = 120.0
    max_peer_confirm_latency_s: float = 20.0  # peers confirming FAST => not idio
    velocity_flip_frac: float = 0.25         # |vel| must drop below frac of peak
    min_replenishment_confidence: float = 0.4


@dataclass
class _Track:
    state: SState = SState.IDLE
    direction: int = 0                # +1 up-burst, -1 down-burst
    entered_at: datetime | None = None
    impulse_start: datetime | None = None
    extreme_mid: float = 0.0          # most extreme price of the burst
    peak_velocity: float = 0.0
    peak_z: float = 0.0
    retest_seen: bool = False
    cooldown_until: datetime | None = None
    history: list[tuple[str, str]] = field(default_factory=list)


class SignalStateMachine:
    def __init__(self, cfg: FsmConfig, strat: StrategyConfig) -> None:
        self.cfg = cfg
        self.strat = strat
        self.tracks: dict[str, _Track] = {}

    def state(self, symbol: str) -> SState:
        return self.tracks.setdefault(symbol, _Track()).state

    def _go(self, tr: _Track, to: SState, ts: datetime, why: str) -> None:
        tr.history.append((ts.isoformat(), f"{tr.state.value}->{to.value}:{why}"))
        log.info("fsm", frm=tr.state.value, to=to.value, why=why)
        tr.state = to
        tr.entered_at = ts

    def on_position_closed(self, symbol: str, ts: datetime) -> None:
        tr = self.tracks.setdefault(symbol, _Track())
        self._go(tr, SState.COOLDOWN, ts, "position_closed")
        from datetime import timedelta
        tr.cooldown_until = ts + timedelta(seconds=self.cfg.cooldown_s)

    def on_entry_fill(self, symbol: str, ts: datetime) -> None:
        tr = self.tracks.setdefault(symbol, _Track())
        if tr.state is SState.ENTRY:
            self._go(tr, SState.MANAGE, ts, "entry_filled")

    def on_entry_failed(self, symbol: str, ts: datetime) -> None:
        """Entry order expired/cancelled unfilled — back off, don't chase."""
        tr = self.tracks.setdefault(symbol, _Track())
        if tr.state is SState.ENTRY:
            self._go(tr, SState.COOLDOWN, ts, "entry_unfilled")
            from datetime import timedelta
            tr.cooldown_until = ts + timedelta(seconds=self.cfg.cooldown_s / 2)

    def step(self, f: FeatureVector, action: GovernorAction) -> Signal | None:
        """Advance one evaluation tick; returns a Signal only on ENTRY."""
        c = self.cfg
        tr = self.tracks.setdefault(f.symbol, _Track())
        ts = f.ts
        dwell = (ts - tr.entered_at).total_seconds() if tr.entered_at else 0.0

        if tr.state is SState.COOLDOWN:
            if tr.cooldown_until and ts >= tr.cooldown_until:
                self._go(tr, SState.IDLE, ts, "cooldown_done")
            return None
        if tr.state in (SState.ENTRY, SState.MANAGE, SState.EXIT):
            return None   # order/position lifecycle owns these states

        if action is GovernorAction.NO_TRADE and tr.state is SState.IDLE:
            return None   # regime says don't even arm

        if tr.state is SState.IDLE:
            if (abs(f.residual_z) >= c.impulse_z
                    and abs(f.residual_velocity) >= c.impulse_min_velocity):
                tr.direction = 1 if f.residual_z > 0 else -1
                tr.impulse_start = ts
                tr.extreme_mid = f.mid
                tr.peak_velocity = abs(f.residual_velocity)
                tr.peak_z = abs(f.residual_z)
                self._go(tr, SState.IMPULSE, ts, f"z={f.residual_z:.2f}")
            return None

        d = tr.direction
        # track burst extremes in all armed states
        if d * (f.mid - tr.extreme_mid) > 0:
            tr.extreme_mid = f.mid
        tr.peak_velocity = max(tr.peak_velocity, abs(f.residual_velocity))
        tr.peak_z = max(tr.peak_z, abs(f.residual_z))

        # global invalidations while armed
        if action is GovernorAction.NO_TRADE:
            self._go(tr, SState.IDLE, ts, "governor_no_trade")
            return None
        if (0 <= f.peer_confirm_latency_s <= c.max_peer_confirm_latency_s):
            self._go(tr, SState.IDLE, ts, "peers_confirmed_fast")
            return None

        if tr.state is SState.IMPULSE:
            if dwell > c.impulse_timeout_s:
                self._go(tr, SState.IDLE, ts, "impulse_timeout")
            elif abs(f.residual_z) < c.impulse_z * 0.5:
                self._go(tr, SState.IDLE, ts, "impulse_faded")
            else:
                self._go(tr, SState.CONFIRM_WAIT, ts, "armed")
            return None

        if tr.state is SState.CONFIRM_WAIT:
            if dwell < c.confirm_wait_s:
                return None
            vel_dead = abs(f.residual_velocity) <= c.velocity_flip_frac * max(
                tr.peak_velocity, 1e-9)
            flow_fading = d * f.flow_imbalance < 0.35
            exhausted = (vel_dead and f.impact_decay > 0.15
                         and (flow_fading or d * f.residual_acceleration < 0))
            if exhausted:
                self._go(tr, SState.EXHAUSTION, ts, "velocity_dead_impact_decaying")
            elif dwell > c.impulse_timeout_s:
                self._go(tr, SState.IDLE, ts, "confirm_timeout")
            return None

        if tr.state is SState.EXHAUSTION:
            if dwell > c.exhaustion_timeout_s:
                self._go(tr, SState.IDLE, ts, "exhaustion_timeout")
                return None
            dist_bps = d * (tr.extreme_mid - f.mid) / max(f.mid, 1e-9) * 1e4
            if dist_bps <= c.retest_tolerance_bps:
                tr.retest_seen = True   # price back at/near the extreme
                return None
            if tr.retest_seen:
                # retested and failed to extend: microprice/replenishment against
                micro_against = d * f.microprice_bps < 0
                repl_against = (d * f.replenishment_skew < 0
                                and f.replenishment_confidence
                                >= c.min_replenishment_confidence)
                if micro_against or repl_against:
                    self._go(tr, SState.FAILED_RETEST, ts, "retest_failed")
            return None

        if tr.state is SState.FAILED_RETEST:
            if dwell > c.retest_timeout_s:
                self._go(tr, SState.IDLE, ts, "retest_window_expired")
                return None
            new_extreme = d * (f.mid - tr.extreme_mid) > 0
            if new_extreme:
                self._go(tr, SState.CONFIRM_WAIT, ts, "new_extreme_rearm")
                return None
            side = Side.SELL if d > 0 else Side.BUY
            stop = tr.extreme_mid * (1 + d * self.strat.stop_bps / 1e4)
            reversion = abs(f.residual_return) * self.strat.target_residual_reversion
            target = f.mid * (1 - d * reversion)
            sig = Signal(f.symbol, ts, side, GovernorAction.REVERSAL,
                         score=tr.peak_z, reference_price=f.mid,
                         stop_price=stop, target_price=target,
                         trace_id=uuid.uuid4().hex[:12])
            self._go(tr, SState.ENTRY, ts, "entry_emitted")
            return sig
        return None

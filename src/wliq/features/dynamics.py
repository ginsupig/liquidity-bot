"""v2 feature dynamics: residual velocity/acceleration, residual-vol z,
confirmation latencies, replenishment confidence, vol regime, time-of-day.

Kept separate from FeatureEngine so the v0.2 engine stays testable; the
engine calls into a ResidualDynamics per symbol and merges outputs.
"""
from __future__ import annotations

import math
from collections import deque
from datetime import datetime


class ResidualDynamics:
    """Tracks residual-return samples to derive velocity, acceleration, and a
    residual-volatility z-score, plus impulse bookkeeping used by the FSM."""

    def __init__(self, vel_window_s: float = 10.0, hist: int = 1800) -> None:
        self.vel_window = vel_window_s
        self.samples: deque[tuple[datetime, float]] = deque(maxlen=600)
        self.vel_history: deque[float] = deque(maxlen=hist)
        self.res_history: deque[float] = deque(maxlen=hist)
        self._last_vel: tuple[datetime, float] | None = None

    def update(self, ts: datetime, residual_return: float) -> tuple[float, float, float]:
        """Returns (velocity bps/s, acceleration bps/s^2, residual_vol_z)."""
        self.samples.append((ts, residual_return))
        self.res_history.append(residual_return)
        # velocity: change in residual over the velocity window
        vel = 0.0
        cutoff = None
        for t0, r0 in reversed(self.samples):
            if (ts - t0).total_seconds() >= self.vel_window:
                cutoff = (t0, r0)
                break
        if cutoff is not None:
            dt = max((ts - cutoff[0]).total_seconds(), 1e-6)
            vel = (residual_return - cutoff[1]) * 1e4 / dt
        acc = 0.0
        if self._last_vel is not None:
            dt = max((ts - self._last_vel[0]).total_seconds(), 1e-6)
            acc = (vel - self._last_vel[1]) / dt
        self._last_vel = (ts, vel)
        self.vel_history.append(vel)
        # residual vol z: recent short-window vol vs long history
        rv_z = 0.0
        if len(self.res_history) >= 120:
            recent = list(self.res_history)[-60:]
            longer = list(self.res_history)
            def _std(xs: list[float]) -> float:
                m = sum(xs) / len(xs)
                return math.sqrt(sum((x - m) ** 2 for x in xs) / max(len(xs) - 1, 1))
            s_r, s_l = _std(recent), _std(longer)
            if s_l > 1e-12:
                rv_z = (s_r - s_l) / s_l
        return vel, acc, rv_z


class ConfirmationTimer:
    """Measures how quickly peers/sector confirm an impulse. Short latency =
    common factor move (fade is dangerous); long/never = idiosyncratic."""

    def __init__(self, confirm_threshold: float = 0.35) -> None:
        self.threshold = confirm_threshold
        self.impulse_ts: datetime | None = None
        self.peer_latency: float = -1.0
        self.sector_latency: float = -1.0

    def start(self, ts: datetime) -> None:
        self.impulse_ts, self.peer_latency, self.sector_latency = ts, -1.0, -1.0

    def update(self, ts: datetime, peer_conf: float, sector_conf: float) -> None:
        if self.impulse_ts is None:
            return
        dt = (ts - self.impulse_ts).total_seconds()
        if self.peer_latency < 0 and peer_conf >= self.threshold:
            self.peer_latency = dt
        if self.sector_latency < 0 and sector_conf >= self.threshold:
            self.sector_latency = dt


def replenishment_confidence(n_consume_events: int, n_restore_events: int,
                             full_at: int = 12) -> float:
    """Confidence in the replenishment_skew estimate grows with sample count."""
    n = n_consume_events + n_restore_events
    return min(1.0, n / full_at)


def vol_regime(realized_vol_bps: float, low: float, high: float) -> int:
    if realized_vol_bps < low:
        return 0
    if realized_vol_bps > high:
        return 2
    return 1


def tod_bucket(minutes_since_open: float) -> int:
    if minutes_since_open < 15:
        return 0
    if minutes_since_open < 90:
        return 1
    if minutes_since_open < 240:
        return 2
    if minutes_since_open < 360:
        return 3
    return 4

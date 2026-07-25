"""Sprint 6 — regime classifier + anomaly detector.

Regime: rule-based (transparent, auditable) on realized vol, trend strength,
and breadth — NOT a black box; ML regime models come only after the rule
version is beaten in walk-forward.

Anomaly: robust z (median/MAD) per feature vs trailing history; any feature
beyond `z_max` flags the vector, and the governor treats anomalies as
NO_TRADE (fail closed — anomalies are exactly when models break).
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np

from wliq.core.models import FeatureVector

REGIMES = ("quiet", "normal", "volatile", "trending")


def classify_regime(f: FeatureVector, breadth: float = 0.0) -> str:
    if f.vol_regime == 2:
        return "volatile"
    trendish = abs(f.vwap_distance_bps) > 30 and abs(breadth) > 0.6
    if trendish:
        return "trending"
    if f.vol_regime == 0:
        return "quiet"
    return "normal"


@dataclass
class AnomalyDetector:
    z_max: float = 8.0
    window: int = 1200
    min_obs: int = 120

    def __post_init__(self) -> None:
        self._hist: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=self.window))

    _FIELDS = ("residual_z", "spread_bps", "flow_imbalance", "volume_z",
               "realized_vol_bps", "residual_velocity", "microprice_bps")

    def check(self, f: FeatureVector) -> list[str]:
        flags: list[str] = []
        for name in self._FIELDS:
            v = float(getattr(f, name))
            h = self._hist[f"{f.symbol}:{name}"]
            if len(h) >= self.min_obs:
                arr = np.asarray(h)
                med = float(np.median(arr))
                mad = float(np.median(np.abs(arr - med))) * 1.4826 + 1e-9
                if abs(v - med) / mad > self.z_max:
                    flags.append(name)
            h.append(v)
        return flags

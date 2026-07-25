"""Expected-volume model and standardized volume surprise.

Offline: build a per-symbol intraday cumulative-volume profile from historical
minute bars (median + MAD per time bucket, conditioned on day of week when
enough history exists). Online: volume_z = (actual - expected) / mad_scaled.

Falls back to a self-referential rolling baseline when no profile exists, so
the live path degrades gracefully instead of failing.
"""
from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


class VolumeProfile:
    def __init__(self, bucket_minutes: int = 5) -> None:
        self.bucket_minutes = bucket_minutes
        # (symbol, bucket) -> (median cumulative volume, mad)
        self.table: dict[tuple[str, int], tuple[float, float]] = {}

    def bucket(self, minutes_since_open: float) -> int:
        return max(0, int(minutes_since_open // self.bucket_minutes))

    @classmethod
    def fit(cls, bars: pd.DataFrame, bucket_minutes: int = 5,
            open_minutes_col: str = "minutes_since_open") -> "VolumeProfile":
        """bars: columns [symbol, date, minutes_since_open, volume] minute bars."""
        p = cls(bucket_minutes)
        bars = bars.copy()
        bars["bucket"] = (bars[open_minutes_col] // bucket_minutes).astype(int)
        bars = bars.sort_values(["symbol", "date", open_minutes_col])
        bars["cum_volume"] = bars.groupby(["symbol", "date"])["volume"].cumsum()
        ends = bars.groupby(["symbol", "date", "bucket"])["cum_volume"].last().reset_index()
        for (symbol, bucket), grp in ends.groupby(["symbol", "bucket"]):
            v = grp["cum_volume"].to_numpy(dtype=float)
            med = float(np.median(v))
            mad = float(np.median(np.abs(v - med))) * 1.4826
            p.table[(symbol, int(bucket))] = (med, max(mad, med * 0.05, 1.0))
        return p

    def save(self, path: Path) -> None:
        rows = [
            {"symbol": s, "bucket": b, "median": m, "mad": d}
            for (s, b), (m, d) in self.table.items()
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(path, index=False)

    @classmethod
    def load(cls, path: Path, bucket_minutes: int = 5) -> "VolumeProfile":
        p = cls(bucket_minutes)
        df = pd.read_parquet(path)
        for row in df.itertuples(index=False):
            p.table[(row.symbol, int(row.bucket))] = (float(row.median), float(row.mad))
        return p

    def surprise(self, symbol: str, minutes_since_open: float, cum_volume: float) -> float | None:
        entry = self.table.get((symbol, self.bucket(minutes_since_open)))
        if entry is None:
            return None
        med, mad = entry
        return max(-8.0, min(8.0, (cum_volume - med) / mad))


class VolumeTracker:
    """Online cumulative-volume tracker with profile-based or fallback surprise."""

    def __init__(self, profile: VolumeProfile | None) -> None:
        self.profile = profile
        self.cum: dict[str, float] = defaultdict(float)
        self._recent_rates: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=600))
        self._last_ts: dict[str, datetime] = {}

    def on_trade(self, symbol: str, ts: datetime, size: float) -> None:
        self.cum[symbol] += size
        prev = self._last_ts.get(symbol)
        if prev is not None:
            dt = max((ts - prev).total_seconds(), 1e-3)
            self._recent_rates[symbol].append(size / dt)
        self._last_ts[symbol] = ts

    def volume_z(self, symbol: str, minutes_since_open: float) -> float:
        if self.profile is not None:
            z = self.profile.surprise(symbol, minutes_since_open, self.cum[symbol])
            if z is not None:
                return z
        # Fallback: compare recent 30s flow rate to the session's own median rate.
        rates = self._recent_rates[symbol]
        if len(rates) < 60:
            return 0.0
        arr = np.asarray(rates, dtype=float)
        recent = float(np.mean(arr[-30:]))
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med))) * 1.4826
        if mad <= 1e-9:
            return 0.0
        return max(-8.0, min(8.0, (recent - med) / mad))

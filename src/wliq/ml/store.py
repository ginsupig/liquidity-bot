"""Sprint 6 — online feature store.

Appends every FeatureVector observed at decision points (parquet, partitioned
by date), so training data accumulates automatically from replay, paper, and
live sessions with IDENTICAL feature computation. Labels are joined later from
the event miner's forward returns via (symbol, ts) asof-merge.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from wliq.core.models import FeatureVector

NUMERIC_FIELDS = [
    "spread_bps", "spread_pctile", "residual_return", "residual_z",
    "flow_imbalance", "impact_decay", "replenishment_skew", "microprice_bps",
    "peer_confirmation", "depth_imbalance", "near_touch_ratio",
    "queue_depletion", "volume_z", "vwap_distance_bps", "realized_vol_bps",
    "minutes_since_open", "residual_velocity", "residual_acceleration",
    "residual_vol_z", "peer_confirm_latency_s", "sector_confirm_latency_s",
    "replenishment_confidence", "vol_regime", "tod_bucket", "noii_pressure",
    "data_confidence",
]


class FeatureStore:
    def __init__(self, root: Path, flush_rows: int = 500) -> None:
        self.root = Path(root)
        self.flush_rows = flush_rows
        self._buf: list[dict] = []

    def append(self, f: FeatureVector, context: str = "eval") -> None:
        row = asdict(f)
        row["context"] = context
        self._buf.append(row)
        if len(self._buf) >= self.flush_rows:
            self.flush()

    def flush(self) -> None:
        if not self._buf:
            return
        df = pd.DataFrame(self._buf)
        day = datetime.now(timezone.utc).date().isoformat()
        d = self.root / f"date={day}"
        d.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%H%M%S%f")
        df.to_parquet(d / f"features-{stamp}.parquet", index=False)
        self._buf.clear()

    def load(self, dates: list[str] | None = None) -> pd.DataFrame:
        parts = []
        for d in sorted(self.root.glob("date=*")):
            if dates and d.name.removeprefix("date=") not in dates:
                continue
            for f in sorted(d.glob("*.parquet")):
                parts.append(pd.read_parquet(f))
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def label_events(features: pd.DataFrame, mined: pd.DataFrame,
                 horizon: int = 120, tol_s: float = 3.0) -> pd.DataFrame:
    """Asof-join forward returns onto feature rows; label = fade PnL sign."""
    if features.empty or mined.empty:
        return pd.DataFrame()
    f = features.sort_values("ts").copy()
    m = mined.sort_values("ts")[["symbol", "ts", "direction",
                                 f"fwd_{horizon}s_bps"]].copy()
    f["ts"] = pd.to_datetime(f["ts"], utc=True)
    m["ts"] = pd.to_datetime(m["ts"], utc=True)
    out = pd.merge_asof(f, m, on="ts", by="symbol",
                        tolerance=pd.Timedelta(seconds=tol_s),
                        direction="nearest")
    out = out.dropna(subset=[f"fwd_{horizon}s_bps"])
    out["fade_pnl_bps"] = -out["direction"] * out[f"fwd_{horizon}s_bps"]
    out["label"] = (out["fade_pnl_bps"] > 0).astype(int)
    return out

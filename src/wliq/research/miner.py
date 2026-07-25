"""Event miner: detect microstructure events in recorded data and measure
forward returns, MFE/MAE, and cost-adjusted outcomes at fixed horizons.

Output is a flat DataFrame (one row per event) consumed by the walk-forward
framework and by baseline comparisons.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from wliq.core.config import Settings
from wliq.core.models import Quote, Trade
from wliq.features.engine import FeatureEngine
from wliq.features.volume import VolumeProfile, VolumeTracker
from wliq.research.replay import ReplayEvent, load_events

HORIZONS = (10, 30, 60, 120, 300, 600)


@dataclass(slots=True)
class MinedEvent:
    symbol: str
    ts: datetime
    kind: str            # residual_burst | vwap_rejection | volume_shock | spread_shock
    direction: int       # +1 up-move, -1 down-move
    mid: float
    features: dict


def mine_events(settings: Settings, date: str,
                min_gap_seconds: float = 30.0) -> pd.DataFrame:
    profile = None
    if settings.volume_model.profile_path.exists():
        profile = VolumeProfile.load(settings.volume_model.profile_path,
                                     settings.volume_model.bucket_minutes)
    engine = FeatureEngine(
        settings.feature_window_seconds, settings.benchmarks,
        settings.market_etf, settings.sector_etf,
        settings.residual_halflife_seconds, VolumeTracker(profile),
        settings.session_open, settings.session_tz,
    )
    events = load_events(settings.record_root, date, settings.all_streamed_symbols())
    mids: dict[str, list[tuple[datetime, float]]] = {s: [] for s in settings.symbols}
    mined: list[MinedEvent] = []
    last_mined: dict[tuple[str, str], datetime] = {}
    last_eval: dict[str, datetime] = {}

    while events:
        ev = heapq.heappop(events)
        if ev.kind == "depth":
            engine.on_depth(ev.payload)  # type: ignore[arg-type]
            continue
        if ev.kind == "trade":
            engine.on_trade(ev.payload)  # type: ignore[arg-type]
            continue
        q: Quote = ev.payload  # type: ignore[assignment]
        engine.on_quote(q)
        if q.symbol not in mids:
            continue
        mids[q.symbol].append((q.ts, q.mid))
        prev = last_eval.get(q.symbol)
        if prev and (q.ts - prev).total_seconds() < 1.0:
            continue
        last_eval[q.symbol] = q.ts
        f = engine.compute(q.symbol, q.ts)
        if f is None or f.event_count < 25:
            continue
        direction = 1 if f.residual_z > 0 else -1
        checks = [
            ("residual_burst", abs(f.residual_z) >= 2.0),
            ("vwap_rejection", abs(f.vwap_distance_bps) >= 15
             and np.sign(f.vwap_distance_bps) != np.sign(f.microprice_bps or 1)),
            ("volume_shock", f.volume_z >= 3.0),
            ("spread_shock", f.spread_pctile >= 0.97),
        ]
        for kind, hit in checks:
            if not hit:
                continue
            lk = (q.symbol, kind)
            if lk in last_mined and (q.ts - last_mined[lk]).total_seconds() < min_gap_seconds:
                continue
            last_mined[lk] = q.ts
            mined.append(MinedEvent(q.symbol, q.ts, kind, direction, q.mid, {
                "residual_z": f.residual_z, "flow_imbalance": f.flow_imbalance,
                "impact_decay": f.impact_decay, "replenishment_skew": f.replenishment_skew,
                "microprice_bps": f.microprice_bps, "peer_confirmation": f.peer_confirmation,
                "volume_z": f.volume_z, "spread_bps": f.spread_bps,
                "spread_pctile": f.spread_pctile, "depth_imbalance": f.depth_imbalance,
                "vwap_distance_bps": f.vwap_distance_bps,
                "realized_vol_bps": f.realized_vol_bps,
                "minutes_since_open": f.minutes_since_open,
            }))

    return _attach_forward(mined, mids)


def _attach_forward(mined: list[MinedEvent],
                    mids: dict[str, list[tuple[datetime, float]]]) -> pd.DataFrame:
    rows = []
    for m in mined:
        series = mids[m.symbol]
        arr_ts = np.array([t.timestamp() for t, _ in series])
        arr_px = np.array([p for _, p in series])
        i0 = int(np.searchsorted(arr_ts, m.ts.timestamp(), side="right"))
        row: dict = {"symbol": m.symbol, "ts": m.ts, "date": m.ts.date().isoformat(),
                     "kind": m.kind, "direction": m.direction, "mid": m.mid,
                     **m.features}
        for h in HORIZONS:
            j = int(np.searchsorted(arr_ts, m.ts.timestamp() + h, side="left"))
            if j >= len(arr_px):
                row[f"fwd_{h}s_bps"] = np.nan
                row[f"mfe_{h}s_bps"] = np.nan
                row[f"mae_{h}s_bps"] = np.nan
                continue
            path = arr_px[i0:j + 1]
            if len(path) == 0:
                row[f"fwd_{h}s_bps"] = np.nan
                row[f"mfe_{h}s_bps"] = np.nan
                row[f"mae_{h}s_bps"] = np.nan
                continue
            rets = 10_000 * (path - m.mid) / m.mid
            # convention: forward move measured in the direction of the burst;
            # a REVERSAL trade profits when fwd is negative.
            row[f"fwd_{h}s_bps"] = m.direction * rets[-1]
            row[f"mfe_{h}s_bps"] = float((-m.direction * rets).max())  # fade-trade MFE
            row[f"mae_{h}s_bps"] = float((-m.direction * rets).min())  # fade-trade MAE
        rows.append(row)
    return pd.DataFrame(rows)

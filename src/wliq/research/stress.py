"""Sprint 3 — stress testing, parameter stability, feature importance,
bootstrap significance. Operates on the mined-events frame (one row per
candidate event with features + forward returns), reusing the vectorized
scorer from walkforward so research and replay logic agree.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from wliq.research.walkforward import (DEFAULT_WEIGHTS, score_events,
                                       trade_pnl_bps)


def _select(events: pd.DataFrame, weights: dict, min_score: float) -> pd.DataFrame:
    weights = {**DEFAULT_WEIGHTS, **weights}
    sc = score_events(events, weights)
    return events.loc[sc >= min_score]


def stress_grid(events: pd.DataFrame, weights: dict, min_score: float,
                horizon: int, cost_bps: float = 1.0) -> pd.DataFrame:
    """PnL under stressed execution assumptions.

    delayed fills: shift entry to a later horizon (we capture less reversion);
    fill probability: random subsampling of trades (expected PnL unchanged per
    trade, but N and variance move — reported for sizing honesty).
    """
    rows = []
    sel = _select(events, weights, min_score)
    horizons = sorted({int(c.split("_")[1].rstrip("s")) for c in events.columns
                       if c.startswith("fwd_")})
    delay_h = next((h for h in horizons if h > horizon), horizon)
    scenarios = {
        "base": dict(cost=cost_bps, col=f"fwd_{horizon}s_bps", spread_mult=1.0, p=1.0),
        "costs_x2": dict(cost=cost_bps * 2, col=f"fwd_{horizon}s_bps", spread_mult=1.0, p=1.0),
        "spread_x1_5": dict(cost=cost_bps, col=f"fwd_{horizon}s_bps", spread_mult=1.5, p=1.0),
        "delayed_fill": dict(cost=cost_bps, col=f"fwd_{delay_h}s_bps", spread_mult=1.0, p=1.0),
        "fill_prob_60": dict(cost=cost_bps, col=f"fwd_{horizon}s_bps", spread_mult=1.0, p=0.6),
        "worst_case": dict(cost=cost_bps * 2, col=f"fwd_{delay_h}s_bps", spread_mult=1.5, p=0.6),
    }
    rng = np.random.default_rng(11)
    for name, sc in scenarios.items():
        df = sel
        if sc["p"] < 1.0 and len(sel):
            df = sel.loc[rng.random(len(sel)) < sc["p"]]
        if not len(df):
            rows.append({"scenario": name, "n": 0})
            continue
        stress_df = df.assign(spread_bps=df["spread_bps"] * sc["spread_mult"])
        h = int(sc["col"].split("_")[1].rstrip("s"))
        pnl = trade_pnl_bps(stress_df.rename(columns={sc["col"]: f"fwd_{h}s_bps"}),
                            h, cost_bps=sc["cost"])
        rows.append({"scenario": name, "n": len(pnl),
                     "mean_bps": float(pnl.mean()),
                     "median_bps": float(pnl.median()),
                     "total_bps": float(pnl.sum()),
                     "hit_rate": float((pnl > 0).mean())})
    return pd.DataFrame(rows)


def parameter_stability(events: pd.DataFrame, weights: dict, min_score: float,
                        horizon: int, perturb: float = 0.2,
                        n_draws: int = 40, seed: int = 5) -> dict:
    """Perturb every weight ±perturb uniformly; a real edge should not flip
    sign across small parameter neighborhoods."""
    weights = {**DEFAULT_WEIGHTS, **weights}
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_draws):
        w = {k: v * float(rng.uniform(1 - perturb, 1 + perturb))
             for k, v in weights.items()}
        sel = _select(events, w, min_score)
        if not len(sel):
            continue
        pnl = trade_pnl_bps(sel, horizon, cost_bps=1.0)
        means.append(float(pnl.mean()))
    if not means:
        return {"n_draws": 0}
    arr = np.asarray(means)
    return {"n_draws": len(arr), "mean_of_means_bps": float(arr.mean()),
            "std_of_means_bps": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "frac_positive": float((arr > 0).mean()),
            "min_bps": float(arr.min()), "max_bps": float(arr.max())}


def feature_importance(events: pd.DataFrame, weights: dict, min_score: float,
                       horizon: int, n_perm: int = 20, seed: int = 9) -> pd.DataFrame:
    """Permutation importance: shuffle one feature column, re-select, and
    measure mean-PnL degradation. Model-agnostic and honest about which
    features actually drive selection."""
    weights = {**DEFAULT_WEIGHTS, **weights}
    rng = np.random.default_rng(seed)
    base_sel = _select(events, weights, min_score)
    if not len(base_sel):
        return pd.DataFrame()
    base = float(trade_pnl_bps(base_sel, horizon, cost_bps=1.0).mean())
    rows = []
    for feat in weights:
        if feat not in events.columns and feat != "residual_z":
            continue
        col = "residual_z" if feat == "residual_z" else feat
        if col not in events.columns:
            continue
        degr = []
        for _ in range(n_perm):
            shuffled = events.assign(**{col: rng.permutation(events[col].values)})
            sel = _select(shuffled, weights, min_score)
            m = float(trade_pnl_bps(sel, horizon, cost_bps=1.0).mean()) if len(sel) else 0.0
            degr.append(base - m)
        rows.append({"feature": feat, "importance_bps": float(np.mean(degr)),
                     "std_bps": float(np.std(degr))})
    return pd.DataFrame(rows).sort_values("importance_bps", ascending=False)


def bootstrap_test(pnl_bps: pd.Series | np.ndarray, n_boot: int = 5000,
                   seed: int = 13) -> dict:
    """Percentile-bootstrap CI on mean PnL + one-sided p(mean<=0)."""
    arr = np.asarray(pnl_bps, dtype=float)
    if len(arr) < 5:
        return {"n": int(len(arr)), "note": "insufficient trades"}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    means = arr[idx].mean(axis=1)
    return {"n": int(len(arr)), "mean_bps": float(arr.mean()),
            "ci95": [float(np.percentile(means, 2.5)),
                     float(np.percentile(means, 97.5))],
            "p_leq_zero": float((means <= 0).mean())}

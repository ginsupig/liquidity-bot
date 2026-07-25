import numpy as np
import pandas as pd

from wliq.research.walkforward import (deflated_sharpe, monte_carlo_drawdown,
                                       purged_walk_forward, trade_pnl_bps)


def synth_events(n_days: int = 8, per_day: int = 40, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_days):
        for _ in range(per_day):
            direction = int(rng.choice([-1, 1]))
            z = float(rng.uniform(1.5, 4.0))
            # embed weak mean reversion: bigger z -> more reversion
            fwd = -direction * 0 + float(rng.normal(-2.0 * (z - 2), 8))
            rows.append({
                "symbol": "MU", "date": f"2026-07-{d+1:02d}", "kind": "residual_burst",
                "direction": direction, "residual_z": direction * z,
                "flow_imbalance": direction * float(rng.uniform(0, 0.8)),
                "impact_decay": float(rng.uniform(-0.5, 0.8)),
                "replenishment_skew": -direction * float(rng.uniform(-0.5, 1.5)),
                "microprice_bps": -direction * float(rng.uniform(-2, 4)),
                "peer_confirmation": float(rng.uniform(-0.2, 0.5)),
                "volume_z": float(rng.uniform(0, 4)),
                "depth_imbalance": -direction * float(rng.uniform(-0.3, 0.5)),
                "spread_bps": float(rng.uniform(1, 6)),
                "fwd_120s_bps": fwd,
            })
    return pd.DataFrame(rows)


def test_walkforward_produces_folds_and_baselines():
    events = synth_events()
    rep = purged_walk_forward(events, {}, min_score=2.0, horizon=120, cost_bps=1.0,
                              n_folds=3)
    assert len(rep.folds) == 3
    assert "residual_z_only" in rep.baselines
    assert "random" in rep.baselines
    assert rep.full_model["label"] == "OOS"
    assert rep.full_model["n"] > 0


def test_deflated_sharpe_penalizes_trials():
    assert deflated_sharpe(0.3, 100, n_trials=1) > deflated_sharpe(0.3, 100, n_trials=50)


def test_mc_drawdown_percentiles_ordered():
    pnl = np.random.default_rng(0).normal(0.5, 5, 200)
    dd = monte_carlo_drawdown(pnl, n_paths=200)
    assert dd["dd_p50"] <= dd["dd_p95"] <= dd["dd_p99"]


def test_pnl_convention_reversal_profits_on_reversion():
    df = pd.DataFrame({"fwd_120s_bps": [-10.0], "spread_bps": [2.0]})
    pnl = trade_pnl_bps(df, 120, cost_bps=1.0)
    assert pnl.iloc[0] == 10.0 - 3.0

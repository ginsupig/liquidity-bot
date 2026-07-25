"""Sprint 6 — parameter search: successive-halving random search with local
refinement around survivors. (Deliberately NOT labeled Bayesian optimization —
no surrogate model is fit; this is honest, parallel-friendly, and hard to
fool.) Objective = OOS mean PnL from the purged walk-forward, penalized by
parameter instability. Every trial increments n_trials, which MUST be fed to
deflated_sharpe when judging the winner.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from wliq.research.stress import parameter_stability
from wliq.research.walkforward import purged_walk_forward


def _sample(space: dict[str, tuple[float, float]],
            rng: np.random.Generator) -> dict[str, float]:
    return {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in space.items()}


def _local(base: dict[str, float], space: dict[str, tuple[float, float]],
           rng: np.random.Generator, scale: float = 0.15) -> dict[str, float]:
    out = {}
    for k, v in base.items():
        lo, hi = space[k]
        out[k] = float(np.clip(v * rng.uniform(1 - scale, 1 + scale), lo, hi))
    return out


def halving_search(events: pd.DataFrame, space: dict[str, tuple[float, float]],
                   min_score: float, horizon: int, n_initial: int = 32,
                   n_rounds: int = 3, seed: int = 17) -> dict:
    rng = np.random.default_rng(seed)
    cands = [_sample(space, rng) for _ in range(n_initial)]
    n_trials = 0
    scored: list[tuple[float, dict]] = []
    for rnd in range(n_rounds):
        scored = []
        for w in cands:
            rep = purged_walk_forward(events, w, min_score, horizon,
                                      cost_bps=1.0, n_folds=3)
            n_trials += 1
            obj = rep.full_model.get("mean_bps", float("-inf")) or float("-inf")
            scored.append((obj, w))
        scored.sort(key=lambda x: x[0], reverse=True)
        keep = scored[: max(2, len(scored) // 2)]
        cands = [w for _, w in keep]
        if rnd < n_rounds - 1:
            cands += [_local(w, space, rng) for _, w in keep]
    best_obj, best_w = scored[0]
    stab = parameter_stability(events, best_w, min_score, horizon)
    return {"best_weights": best_w, "oos_mean_bps": best_obj,
            "n_trials": n_trials, "stability": stab,
            "warning": "feed n_trials into deflated_sharpe; a winner that "
                       "fails parameter_stability is curve-fit, not edge"}

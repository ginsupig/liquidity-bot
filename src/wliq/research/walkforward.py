"""Purged walk-forward validation with embargo, baselines, Monte Carlo trade
reshuffling, and deflated Sharpe / probability-of-overfitting diagnostics.

Operates on the mined-event DataFrame (one row per candidate event with
features and cost-adjusted forward returns). Fade convention: a REVERSAL trade
on an event earns  -direction * forward_move - costs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import erf, sqrt

import numpy as np
import pandas as pd

RNG = np.random.default_rng


# --------------------------------------------------------------------- rules
def rule_random(df: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    return pd.Series(rng.random(len(df)) < 0.15, index=df.index)

def rule_residual_only(df: pd.DataFrame, z: float = 2.25) -> pd.Series:
    return df["residual_z"].abs() >= z

def rule_vwap_rejection(df: pd.DataFrame) -> pd.Series:
    return (df["kind"] == "vwap_rejection")

def rule_flow_exhaustion(df: pd.DataFrame) -> pd.Series:
    return (df["direction"] * df["flow_imbalance"] > 0.2) & (df["impact_decay"] > 0.1)

def rule_l2_imbalance(df: pd.DataFrame) -> pd.Series:
    return (df["direction"] * df["depth_imbalance"] < -0.2)

DEFAULT_WEIGHTS: dict[str, float] = {
    "residual_z": 1.2, "flow_exhaustion": 0.8, "impact_decay": 0.75,
    "replenishment": 0.55, "microprice": 0.35, "peer_divergence": 0.65,
    "volume_surprise": 0.30, "depth_imbalance": 0.40,
}


def score_events(df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Vectorized additive quality score — the SAME formula the live scorer
    uses; single source of truth for research selection."""
    d = df["direction"]
    return (
        weights.get("residual_z", 1.2) * df["residual_z"].abs().clip(0, 5)
        + weights.get("flow_exhaustion", 0.8) * (d * df["flow_imbalance"]).clip(lower=0)
        + weights.get("impact_decay", 0.75) * df["impact_decay"].clip(lower=0)
        + weights.get("replenishment", 0.55) * (-d * df["replenishment_skew"]).clip(lower=0)
        + weights.get("microprice", 0.35) * (-d * df["microprice_bps"] / 2).clip(lower=0)
        + weights.get("peer_divergence", 0.65) * (1 - df["peer_confirmation"]).clip(lower=0)
        + weights.get("volume_surprise", 0.30) * df["volume_z"].clip(0, 4) / 4
        + weights.get("depth_imbalance", 0.40) * (-d * df["depth_imbalance"]).clip(lower=0)
    )


def rule_full_model(df: pd.DataFrame, weights: dict[str, float],
                    min_score: float) -> pd.Series:
    return score_events(df, weights) >= min_score


BASELINES = {
    "random": None,  # handled specially (needs rng)
    "residual_z_only": rule_residual_only,
    "vwap_rejection": rule_vwap_rejection,
    "flow_exhaustion_only": rule_flow_exhaustion,
    "l2_imbalance_only": rule_l2_imbalance,
}


# ------------------------------------------------------------------- metrics
def trade_pnl_bps(df: pd.DataFrame, horizon: int, cost_bps: float) -> pd.Series:
    """Fade PnL: profit when the burst mean-reverts, net of round-trip cost."""
    return -df[f"fwd_{horizon}s_bps"] - (df["spread_bps"].clip(upper=20) + cost_bps)


def sharpe(pnl: np.ndarray) -> float:
    if len(pnl) < 2 or pnl.std(ddof=1) <= 1e-12:
        return 0.0
    return float(pnl.mean() / pnl.std(ddof=1))


def deflated_sharpe(sr: float, n: int, n_trials: int,
                    skew: float = 0.0, kurt: float = 3.0) -> float:
    """Bailey & López de Prado DSR: probability that SR exceeds the expected
    max SR of n_trials of noise. Returns probability in [0,1]."""
    if n < 3:
        return 0.0
    emc = 0.5772156649
    if n_trials <= 1:
        sr0 = 0.0
    else:
        var_sr = 1.0 / max(n, 1)  # stand-in for cross-trial SR variance
        sr0 = sqrt(var_sr) * ((1 - emc) * _z_ppf(1 - 1 / n_trials)
                              + emc * _z_ppf(1 - 1 / (n_trials * np.e)))
    num = (sr - sr0) * sqrt(n - 1)
    den = sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr * sr))
    return 0.5 * (1 + erf(num / den / sqrt(2)))


def _z_ppf(p: float) -> float:
    # Acklam rational approximation of the standard normal inverse CDF.
    p = min(max(p, 1e-12), 1 - 1e-12)
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = sqrt(-2 * np.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = sqrt(-2 * np.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def monte_carlo_drawdown(pnl: np.ndarray, n_paths: int = 2000,
                         seed: int = 7) -> dict[str, float]:
    rng = RNG(seed)
    dds = np.empty(n_paths)
    for i in range(n_paths):
        eq = np.cumsum(rng.permutation(pnl))
        dds[i] = float((np.maximum.accumulate(eq) - eq).max()) if len(eq) else 0.0
    return {"dd_p50": float(np.percentile(dds, 50)),
            "dd_p95": float(np.percentile(dds, 95)),
            "dd_p99": float(np.percentile(dds, 99))}


# --------------------------------------------------------------- walk-forward
@dataclass(slots=True)
class FoldResult:
    fold: int
    train_dates: tuple[str, str]
    test_dates: tuple[str, str]
    n_trades: int
    mean_bps: float
    sharpe: float


@dataclass(slots=True)
class WalkForwardReport:
    horizon: int
    cost_bps: float
    folds: list[FoldResult] = field(default_factory=list)
    baselines: dict[str, dict[str, float]] = field(default_factory=dict)
    full_model: dict[str, float] = field(default_factory=dict)
    mc_drawdown: dict[str, float] = field(default_factory=dict)
    stress: dict[str, float] = field(default_factory=dict)
    concentration: dict[str, float] = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([vars(f) for f in self.folds])


def purged_walk_forward(events: pd.DataFrame, weights: dict[str, float],
                        min_score: float = 3.0, horizon: int = 120,
                        cost_bps: float = 3.0, n_folds: int = 5,
                        embargo_days: int = 1, seed: int = 7) -> WalkForwardReport:
    """Day-level purged walk-forward with embargo between train and test.

    Weights are treated as FIXED inputs here (calibrated elsewhere); the folds
    measure out-of-sample stability of the fixed rule, and n_trials for the
    deflated Sharpe should count every configuration the researcher tried.
    """
    df = events.dropna(subset=[f"fwd_{horizon}s_bps"]).copy()
    df["pnl_bps"] = trade_pnl_bps(df, horizon, cost_bps)
    dates = sorted(df["date"].unique())
    report = WalkForwardReport(horizon=horizon, cost_bps=cost_bps)
    if len(dates) < n_folds + 1:
        n_folds = max(1, len(dates) - 1)
    if n_folds < 1 or len(dates) < 2:
        # single-period fallback: report in-sample stats, clearly labeled
        sel = rule_full_model(df, weights, min_score)
        pnl = df.loc[sel, "pnl_bps"].to_numpy()
        report.full_model = _summarize(pnl, n_trials=1, label="IN_SAMPLE_ONLY")
        return report

    folds = np.array_split(np.array(dates), n_folds + 1)
    oos_pnl: list[float] = []
    for i in range(1, len(folds)):
        train_dates = np.concatenate(folds[:i])
        test_dates = folds[i]
        # embargo: drop the last embargo_days of train adjacent to test
        train_dates = train_dates[:-embargo_days] if embargo_days and len(train_dates) > embargo_days else train_dates
        test = df[df["date"].isin(test_dates)]
        sel = rule_full_model(test, weights, min_score)
        pnl = test.loc[sel, "pnl_bps"].to_numpy()
        oos_pnl.extend(pnl.tolist())
        report.folds.append(FoldResult(
            fold=i, train_dates=(str(train_dates[0]), str(train_dates[-1])) if len(train_dates) else ("", ""),
            test_dates=(str(test_dates[0]), str(test_dates[-1])),
            n_trades=len(pnl), mean_bps=float(pnl.mean()) if len(pnl) else 0.0,
            sharpe=sharpe(pnl),
        ))

    pnl_arr = np.asarray(oos_pnl)
    report.full_model = _summarize(pnl_arr, n_trials=8, label="OOS")
    if len(pnl_arr) >= 10:
        report.mc_drawdown = monte_carlo_drawdown(pnl_arr, seed=seed)

    # baselines on the same OOS dates
    oos_dates = np.concatenate(folds[1:])
    oos_df = df[df["date"].isin(oos_dates)]
    rng = RNG(seed)
    for name, rule in BASELINES.items():
        sel = rule_random(oos_df, rng) if rule is None else rule(oos_df)
        p = oos_df.loc[sel, "pnl_bps"].to_numpy()
        report.baselines[name] = _summarize(p, n_trials=1, label="OOS")

    # stress: costs doubled, spread widened
    stressed = -oos_df[f"fwd_{horizon}s_bps"] - (oos_df["spread_bps"].clip(upper=20) * 1.5
                                                 + cost_bps * 2)
    sel_full = rule_full_model(oos_df, weights, min_score)
    report.stress = _summarize(stressed[sel_full].to_numpy(), n_trials=1, label="STRESS")

    # concentration
    if sel_full.any():
        picked = oos_df[sel_full]
        by_day = picked.groupby("date")["pnl_bps"].sum()
        by_sym = picked.groupby("symbol")["pnl_bps"].sum()
        tot = picked["pnl_bps"].sum()
        report.concentration = {
            "top_day_share": float(by_day.max() / tot) if tot else float("nan"),
            "top_symbol_share": float(by_sym.max() / tot) if tot else float("nan"),
        }
    return report


def _summarize(pnl: np.ndarray, n_trials: int, label: str) -> dict[str, float]:
    if len(pnl) == 0:
        return {"label": label, "n": 0, "mean_bps": 0.0, "sharpe": 0.0,
                "win_rate": 0.0, "dsr": 0.0}
    sr = sharpe(pnl)
    return {"label": label, "n": int(len(pnl)), "mean_bps": float(pnl.mean()),
            "sharpe": sr, "win_rate": float((pnl > 0).mean()),
            "dsr": deflated_sharpe(sr, len(pnl), n_trials)}

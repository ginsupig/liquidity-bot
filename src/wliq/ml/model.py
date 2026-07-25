"""Sprint 6 — event ranking model with honest calibration.

Primary: LightGBM binary classifier (optional dependency). Fallback: L2
logistic regression implemented in numpy — zero extra deps, so the pipeline
always runs. Both are wrapped in isotonic calibration (PAV) fit on held-out
folds so predicted probabilities are usable for sizing.

The model RANKS candidate entries; it does not replace the FSM trigger. Use:
p(win) below threshold -> skip; above -> scale size mildly. Deployment is
gated by walk-forward outperformance vs the rule-based selector — never
assume the ML beats the rules.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import lightgbm as _lgb
    HAS_LGBM = True
except Exception:            # pragma: no cover
    _lgb = None
    HAS_LGBM = False

from wliq.ml.store import NUMERIC_FIELDS


# ---------------- isotonic calibration (PAV) ----------------
def pav_fit(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(scores)
    x, y = scores[order].astype(float), labels[order].astype(float)
    w = np.ones_like(y)
    # pool adjacent violators
    vals, wts, xs = list(y), list(w), list(x)
    i = 0
    while i < len(vals) - 1:
        if vals[i] > vals[i + 1]:
            tot = wts[i] + wts[i + 1]
            merged = (vals[i] * wts[i] + vals[i + 1] * wts[i + 1]) / tot
            vals[i:i + 2] = [merged]
            wts[i:i + 2] = [tot]
            xs[i:i + 2] = [xs[i + 1]]
            i = max(i - 1, 0)
        else:
            i += 1
    return np.asarray(xs), np.asarray(vals)


def pav_predict(bx: np.ndarray, by: np.ndarray, scores: np.ndarray) -> np.ndarray:
    if len(bx) == 0:
        return np.full_like(scores, 0.5, dtype=float)
    idx = np.searchsorted(bx, scores, side="right")
    idx = np.clip(idx, 0, len(by) - 1)
    return by[idx]


# ---------------- numpy logistic fallback ----------------
@dataclass
class LogisticModel:
    coef: np.ndarray | None = None
    mean: np.ndarray | None = None
    std: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, l2: float = 1.0,
            iters: int = 300, lr: float = 0.1) -> "LogisticModel":
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0) + 1e-9
        Z = (X - self.mean) / self.std
        Z = np.hstack([np.ones((len(Z), 1)), Z])
        w = np.zeros(Z.shape[1])
        for _ in range(iters):
            p = 1 / (1 + np.exp(-Z @ w))
            grad = Z.T @ (p - y) / len(y) + l2 * np.r_[0, w[1:]] / len(y)
            w -= lr * grad
        self.coef = w
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Z = (X - self.mean) / self.std
        Z = np.hstack([np.ones((len(Z), 1)), Z])
        return 1 / (1 + np.exp(-Z @ self.coef))


@dataclass
class RankerBundle:
    kind: str                    # "lightgbm" | "logistic"
    features: list[str]
    model: object
    cal_x: np.ndarray = field(default_factory=lambda: np.array([]))
    cal_y: np.ndarray = field(default_factory=lambda: np.array([]))
    train_meta: dict = field(default_factory=dict)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.features].to_numpy(dtype=float)
        if self.kind == "lightgbm":
            raw = self.model.predict(X)
        else:
            raw = self.model.predict_proba(X)
        return pav_predict(self.cal_x, self.cal_y, np.asarray(raw, dtype=float))


def train_ranker(labeled: pd.DataFrame, features: list[str] | None = None,
                 cal_frac: float = 0.3, seed: int = 21) -> RankerBundle:
    feats = [f for f in (features or NUMERIC_FIELDS) if f in labeled.columns]
    df = labeled.dropna(subset=feats + ["label"]).reset_index(drop=True)
    if len(df) < 60:
        raise ValueError(f"insufficient labeled events: {len(df)} < 60")
    # chronological calibration split — never random for time series
    df = df.sort_values("ts").reset_index(drop=True)
    cut = int(len(df) * (1 - cal_frac))
    tr, cal = df.iloc[:cut], df.iloc[cut:]
    X_tr = tr[feats].to_numpy(dtype=float)
    y_tr = tr["label"].to_numpy(dtype=int)
    if HAS_LGBM:
        model = _lgb.LGBMClassifier(
            n_estimators=200, num_leaves=15, learning_rate=0.05,
            min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
            random_state=seed)
        model.fit(X_tr, y_tr)
        raw_cal = model.predict_proba(cal[feats].to_numpy(dtype=float))[:, 1]
        kind = "lightgbm"
        wrapped: object = _LgbWrap(model)
    else:
        model = LogisticModel().fit(X_tr, y_tr)
        raw_cal = model.predict_proba(cal[feats].to_numpy(dtype=float))
        kind = "logistic"
        wrapped = model
    bx, by = pav_fit(np.asarray(raw_cal, dtype=float),
                     cal["label"].to_numpy(dtype=float))
    return RankerBundle(kind, feats, wrapped, bx, by,
                        {"n_train": len(tr), "n_cal": len(cal),
                         "base_rate": float(df["label"].mean())})


class _LgbWrap:
    def __init__(self, m: object) -> None:
        self._m = m
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._m.predict_proba(X)[:, 1]


def calibration_report(bundle: RankerBundle, df: pd.DataFrame,
                       n_bins: int = 5) -> pd.DataFrame:
    p = bundle.predict(df)
    out = pd.DataFrame({"p": p, "y": df["label"].to_numpy()})
    out["bin"] = pd.qcut(out["p"], q=min(n_bins, out["p"].nunique()),
                         duplicates="drop")
    g = out.groupby("bin", observed=True).agg(pred=("p", "mean"),
                                              actual=("y", "mean"),
                                              n=("y", "size"))
    g["gap"] = g["actual"] - g["pred"]
    return g.reset_index(drop=True)

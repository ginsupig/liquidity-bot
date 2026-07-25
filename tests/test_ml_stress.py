import numpy as np
import pandas as pd
import pytest

from tests.test_walkforward import synth_events
from wliq.ml.model import (HAS_LGBM, calibration_report, pav_fit, pav_predict,
                           train_ranker)
from wliq.ml.regime import AnomalyDetector, classify_regime
from wliq.research.stress import (bootstrap_test, feature_importance,
                                  parameter_stability, stress_grid)


def test_pav_is_monotone():
    rng = np.random.default_rng(3)
    scores = rng.random(300)
    labels = (rng.random(300) < scores).astype(float)   # calibrated by design
    bx, by = pav_fit(scores, labels)
    grid = np.linspace(0, 1, 50)
    p = pav_predict(bx, by, grid)
    assert np.all(np.diff(p) >= -1e-9)


def test_train_ranker_fallback_and_calibration():
    rng = np.random.default_rng(5)
    n = 400
    df = pd.DataFrame({
        "ts": pd.date_range("2026-06-01", periods=n, freq="min", tz="UTC"),
        "residual_z": rng.normal(0, 1, n),
        "flow_imbalance": rng.normal(0, 0.5, n),
        "impact_decay": rng.normal(0.1, 0.2, n),
    })
    logits = 1.5 * df["residual_z"] - 0.5
    df["label"] = (rng.random(n) < 1 / (1 + np.exp(-logits))).astype(int)
    b = train_ranker(df, features=["residual_z", "flow_imbalance",
                                   "impact_decay"])
    assert b.kind in ("lightgbm", "logistic")
    rep = calibration_report(b, df)
    assert (rep["gap"].abs() < 0.25).all()     # roughly calibrated


def test_train_ranker_insufficient_data():
    df = pd.DataFrame({"ts": pd.date_range("2026-06-01", periods=10, freq="min"),
                       "residual_z": np.zeros(10), "label": np.zeros(10)})
    with pytest.raises(ValueError):
        train_ranker(df, features=["residual_z"])


def test_anomaly_detector_flags_outlier():
    from tests.test_adaptive_analytics import fv
    det = AnomalyDetector(min_obs=50)
    for _ in range(60):
        det.check(fv(residual_z=0.5))
    flags = det.check(fv(residual_z=500.0))
    assert "residual_z" in flags


def test_regime_classifier_rules():
    from tests.test_adaptive_analytics import fv
    assert classify_regime(fv(vol_regime=2)) == "volatile"
    assert classify_regime(fv(vol_regime=0)) == "quiet"
    assert classify_regime(fv(vwap_distance_bps=50.0), breadth=0.8) == "trending"


def test_stress_grid_orders_scenarios_sensibly():
    ev = synth_events(n_days=10, per_day=60, seed=2)
    g = stress_grid(ev, weights={}, min_score=2.0, horizon=120)
    base = g.loc[g.scenario == "base", "mean_bps"].iloc[0]
    worst = g.loc[g.scenario == "worst_case", "mean_bps"].iloc[0]
    x2 = g.loc[g.scenario == "costs_x2", "mean_bps"].iloc[0]
    assert x2 < base and worst < base


def test_parameter_stability_perturbs_and_reports():
    ev = synth_events(n_days=12, per_day=60, seed=4)
    st = parameter_stability(ev, weights={}, min_score=3.5, horizon=120,
                             n_draws=15)
    assert st["n_draws"] > 0
    assert st["std_of_means_bps"] > 0          # perturbation changed selection
    assert 0.0 <= st["frac_positive"] <= 1.0


def test_bootstrap_detects_real_edge():
    rng = np.random.default_rng(6)
    pnl = pd.Series(rng.normal(3.0, 5.0, 200))
    r = bootstrap_test(pnl)
    assert r["p_leq_zero"] < 0.05 and r["ci95"][0] > 0


def test_feature_importance_runs():
    ev = synth_events(n_days=10, per_day=50, seed=7)
    imp = feature_importance(ev, weights={}, min_score=2.0, horizon=120,
                             n_perm=5)
    assert not imp.empty and "importance_bps" in imp

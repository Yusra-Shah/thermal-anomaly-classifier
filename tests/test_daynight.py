"""Tests for src/daynight.py on tiny synthetic data. No network, no saved models.

Run from the project root:  python -m tests.test_daynight
"""
import inspect
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src import daynight
from src.config import FEATURES
from src.train import best_threshold


def test_stratified_metrics_by_hand():
    y = np.array([1, 1, 0, 0, 1, 0, 0, 0])
    night = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    score = np.array([0.9, 0.8, 0.7, 0.1, 0.6, 0.2, 0.1, 0.05])
    rows = {r["stratum"]: r for r in daynight.stratified(y, score, 0.5, night)}
    assert set(rows) == {"all", "day", "night"} and rows["all"]["n"] == 8
    n = rows["night"]                                   # y 1,1,0,0 and predictions 1,1,1,0
    assert n["n"] == 4 and n["static_share"] == 0.5 and n["majority_pr_auc"] == 0.5
    assert abs(n["precision"] - 2 / 3) < 1e-9 and n["recall"] == 1.0 and abs(n["f1"] - 0.8) < 1e-9
    assert abs(n["pr_auc"] - 1.0) < 1e-9
    d = rows["day"]                                     # y 1,0,0,0 and predictions 1,0,0,0
    assert d["n"] == 4 and d["static_share"] == 0.25 and d["precision"] == 1.0 and d["recall"] == 1.0
    a = rows["all"]                                     # predictions 1,1,1,0,1,0,0,0: tp 3, fp 1, fn 0
    assert abs(a["precision"] - 0.75) < 1e-9 and a["recall"] == 1.0 and a["static_share"] == 3 / 8
    # a stratum with one class has no ranking metrics, only its share
    only = daynight.stratified(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]), 0.5, np.array([0, 0, 1, 1]))
    assert np.isnan(only[1]["pr_auc"]) and only[1]["static_share"] == 0.0 and only[2]["static_share"] == 1.0
    print("stratified metrics by hand: passed")


def test_night_baseline():
    night = np.array([1, 0, 1, 0, 0, 1])
    score, thr = daynight.night_baseline(night)
    y = np.array([1, 0, 0, 0, 1, 1])
    pred = (score >= thr).astype(int)
    assert pred.tolist() == night.tolist()
    m = daynight.metrics(y, score, thr)
    assert abs(m["precision"] - 2 / 3) < 1e-9 and abs(m["recall"] - 2 / 3) < 1e-9
    day = {r["stratum"]: r for r in daynight.stratified(y, score, thr, night)}["day"]
    assert day["recall"] == 0.0 or np.isnan(day["recall"]) or day["precision"] == 0.0     # predicts no static by day
    print("night baseline: passed")


def test_ablation_drops_exactly_the_named_features():
    sets = daynight.feature_sets()
    assert sets["full model"] == list(FEATURES)
    want = {"without is_night": {"is_night"}, "without confidence_ord": {"confidence_ord"},
            "without frp and frp_per_area": {"frp", "frp_per_area"}, "without bright_diff": {"bright_diff"},
            "without scan, track and pixel_area": {"scan", "track", "pixel_area"}}
    for name, drop in want.items():
        assert set(FEATURES) - set(sets[name]) == drop, name
        assert [f for f in FEATURES if f not in drop] == sets[name]          # order kept
    try:
        daynight.feature_sets(["bright_ti4"])
        raise AssertionError("expected ValueError for a missing feature")
    except ValueError:
        pass
    print("ablation drops exactly the named features: passed")


def tiny(n=600, seed=0):
    rng = np.random.RandomState(seed)
    g = rng.randint(0, 60, n)
    night = rng.randint(0, 2, n)
    x1 = rng.normal(size=n)
    y = ((night + 0.8 * x1 + rng.normal(scale=0.6, size=n)) > 0.9).astype(int)
    X = pd.DataFrame({"is_night": night, "x1": x1, "noise": rng.normal(size=n)})
    return X, y, g


def test_threshold_from_out_of_fold_only():
    params = list(inspect.signature(daynight.fit_variant).parameters)
    assert params == ["Xtr", "ytr", "gtr", "feats", "trees", "folds"], params      # no test data can enter
    X, y, g = tiny()
    model, oof, thr = daynight.fit_variant(X, y, g, ["is_night", "x1", "noise"], trees=8, folds=3)
    assert thr == best_threshold(y, oof)                                           # the threshold is a function of oof
    # every training row has an out of fold score from a model that never saw its group
    assert len(oof) == len(y) and (oof > 0).all() and (oof < 1).all()
    # held out rows change nothing: the threshold is the same however the test set looks
    _, oof2, thr2 = daynight.fit_variant(X, y, g, ["is_night", "x1", "noise"], trees=8, folds=3)
    assert thr2 == thr and np.allclose(oof, oof2)
    print("threshold from out of fold only: passed")


def test_permutation_importance_orders_features():
    X, y, g = tiny(1200, seed=1)
    feats = ["is_night", "x1", "noise"]
    model, oof, thr = daynight.fit_variant(X.iloc[:800], y[:800], g[:800], feats, trees=20, folds=3)
    imp = daynight.permutation_importance(model, X.iloc[800:], y[800:], feats, thr, repeats=3)
    d = imp.set_index("feature")["pr_auc_drop_mean"]
    assert d["is_night"] > 0.05 and d["x1"] > 0.05 and d["noise"] < 0.05, d.to_dict()
    assert imp.iloc[0]["pr_auc_drop_mean"] == d.max()
    print("permutation importance: passed")


def test_xgb_vs_rf_rule():
    rng = np.random.RandomState(2)
    y = rng.randint(0, 2, 500)
    good = y * 0.6 + rng.rand(500) * 0.4
    store = {("tables", daynight.GROUPED): {"y": y, "instant": (good, 0.5)},
             ("xgb", daynight.GROUPED, "full model"): {"p": good, "thr": 0.5},
             ("tables", "temporal"): {"y": y, "instant": (good, 0.5)},
             ("xgb", "temporal", "full model"): {"p": rng.rand(500), "thr": 0.5}}
    with tempfile.TemporaryDirectory() as tmp:
        old = daynight.REPORT_DIR
        daynight.REPORT_DIR = Path(tmp)
        try:
            out = daynight.build_xgb_vs_rf(store, None).set_index("split")
        finally:
            daynight.REPORT_DIR = old
    assert out.loc[daynight.GROUPED, "within_0.01"] and abs(out.loc[daynight.GROUPED, "difference_xgb_minus_rf"]) < 1e-9
    assert not out.loc["temporal", "within_0.01"]
    print("xgb vs rf rule: passed")


def main():
    test_stratified_metrics_by_hand()
    test_night_baseline()
    test_ablation_drops_exactly_the_named_features()
    test_threshold_from_out_of_fold_only()
    test_permutation_importance_orders_features()
    test_xgb_vs_rf_rule()


if __name__ == "__main__":
    main()
    from tests.test_live import peak_memory
    ws, commit = peak_memory()
    print(f"peak memory: working set {ws:.0f} MB, commit {commit:.0f} MB")

"""Tests for the uncertain band and calibration (src/uncertain.py).

Run from the project root:  python -m tests.test_uncertain
"""
import sys

import joblib
import numpy as np

from src import uncertain as U
from src.config import MODEL_DIR


def synthetic(n=4000, seed=0):
    """Scores with a known shape: low scores are fire, high scores are static, a mixed middle."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(int)
    p = np.clip(np.where(y == 1, rng.normal(0.75, 0.2, n), rng.normal(0.3, 0.2, n)), 0, 1)
    return y, p


def test_precision_band_uses_only_out_of_fold_scores():
    """The earlier precision based definition (a documented negative result)."""
    y_oof, p_oof = synthetic(seed=1)
    band = U.choose_band(y_oof, p_oof)
    # The band is a function of the out of fold arrays alone: any test arrays, however
    # different, cannot change it.
    y_a, p_a = synthetic(seed=2)
    y_b, p_b = np.ones(500, int), np.linspace(0, 1, 500)
    assert U.choose_band(y_oof, p_oof) == band
    m_a = U.three_way_metrics(y_a, p_a, band)
    m_b = U.three_way_metrics(y_b, p_b, band)
    assert U.choose_band(y_oof, p_oof) == band
    assert m_a["coverage"] != m_b["coverage"]
    # and the thresholds do meet their targets on the out of fold scores themselves
    d = U.decide(p_oof, band)
    static, fire = d == U.STATIC, d == U.FIRE
    assert y_oof[static].mean() >= U.PREC_TARGET - 1e-9
    assert (1 - y_oof[fire]).mean() >= U.NPV_TARGET - 1e-9
    assert band["t_low"] < band["t_high"] and not band["overlap"]
    print("band from out of fold scores only: passed")


def test_coverage_and_precision_tiny_set():
    # scores and labels checked by hand, band: static at 0.8 or more, fire at 0.2 or less
    p = np.array([0.95, 0.9, 0.85, 0.8, 0.6, 0.5, 0.4, 0.2, 0.1, 0.05])
    y = np.array([1, 1, 0, 1, 1, 0, 0, 0, 0, 0])
    band = {"t_low": 0.2, "t_high": 0.8, "overlap": False}
    m = U.three_way_metrics(y, p, band)
    assert np.isclose(m["coverage"], 0.7)                  # 4 static decisions, 3 fire decisions
    assert np.isclose(m["static_precision_decided"], 0.75)  # 3 of 4 predicted static are static
    assert np.isclose(m["static_recall_decided"], 1.0)      # no static point among decided fire
    assert np.isclose(m["fire_npv_decided"], 1.0)
    assert np.isclose(m["uncertain_share"], 0.3)
    # forced binary at t_high: errors are 0.85 (fire called static) and 0.6 (static called fire)
    assert m["binary_errors"] == 2 and m["errors_in_uncertain"] == 1
    assert np.isclose(m["errors_in_uncertain_share"], 0.5)
    assert U.decide(p, band).tolist() == [1, 1, 1, 1, 2, 2, 2, 0, 0, 0]
    # overlapping ranges fall back to the single threshold t_high
    over = {"t_low": 0.9, "t_high": 0.8, "overlap": True}
    assert (U.decide(p, over) == U.UNCERTAIN).sum() == 0
    assert U.decide(p, over).tolist() == [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    print("coverage and precision on a tiny set: passed")


def test_thresholds_match_the_existing_operating_point():
    """High threshold is the existing precision >= 0.90 point (same rule as history2_ops)."""
    from sklearn.metrics import precision_recall_curve
    y, p = synthetic(seed=3)
    prec, _, thr = precision_recall_curve(y, p)
    ok = np.where(prec[:-1] >= 0.90)[0]
    expected = float(thr[ok[np.argmin(thr[ok])]])
    assert abs(U.high_threshold(y, p, 0.90, min_support=1) - expected) < 1e-12
    print("t_high equals the existing operating point rule: passed")


def test_calibrators():
    y, p = synthetic(seed=4)
    skewed = np.clip(p ** 0.4, 0, 1)                       # over confident static scores
    cal, scores = U.pick_calibrator(skewed, y)
    assert U.brier(y, U.apply_calibrator(cal, skewed)) < U.brier(y, skewed)
    assert set(scores) == {"platt", "isotonic"} and cal["kind"] == min(scores, key=scores.get)
    c = U.apply_calibrator(cal, np.array([0.0, 0.5, 1.0]))
    assert np.all((c >= 0) & (c <= 1)) and np.all(np.diff(c) >= -1e-9)      # monotone, in range
    assert U.ece(y, U.apply_calibrator(cal, skewed)) < U.ece(y, skewed)
    print("calibrators: passed")


def test_accuracy_band_out_of_fold_only():
    y_oof, p_oof = synthetic(seed=1)
    cal, _ = U.pick_calibrator(p_oof, y_oof)
    c_oof = U.apply_calibrator(cal, p_oof)
    band = U.choose_accuracy_band(y_oof, c_oof, 0.96)
    assert band["reachable"] and band["definition"] == "accuracy"
    # a function of the out of fold arrays alone: test arrays, however different, cannot change it
    y_a, p_a = synthetic(seed=2)
    y_b, p_b = np.ones(500, int), np.linspace(0, 1, 500)
    m_a = U.three_way_metrics(y_a, U.apply_calibrator(cal, p_a), band, U.BINARY_T)
    m_b = U.three_way_metrics(y_b, U.apply_calibrator(cal, p_b), band, U.BINARY_T)
    assert U.choose_accuracy_band(y_oof, c_oof, 0.96) == band
    assert m_a["coverage"] != m_b["coverage"]
    # the target holds on the out of fold points the band was chosen on
    d = U.decide(c_oof, band)
    done = d != U.UNCERTAIN
    assert (np.where(d == U.STATIC, 1, 0)[done] == y_oof[done]).mean() >= 0.96 - 1e-9
    # the strict band is nested in the default one
    strict = U.choose_accuracy_band(y_oof, c_oof, 0.98)
    assert strict["oof_coverage"] <= band["oof_coverage"] + 1e-9
    # the study picks every band from the out of fold arrays before any test metric is computed
    src = open(U.__file__, encoding="utf-8").read()
    pick = src.index("choose_accuracy_band(ytr, c_oof, ACC_DEFAULT)")
    use = src.index("m = three_way_metrics(yte,")
    assert pick < use and "choose_accuracy_band(yte" not in src and "choose_accuracy_band(ytr, c_te" not in src
    print("accuracy band from out of fold scores only: passed")


def test_accuracy_band_tiny_set():
    # scores and labels checked by hand
    s = np.array([0.05, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 0.95, 0.99])
    y = np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 1])
    # target 0.95: fire for the first four (all fire), static for the last four (all static),
    # the points at 0.5 and 0.6 are uncertain: coverage 0.8, accuracy 1.0
    band = U.choose_accuracy_band(y, s, 0.95, min_support=1)
    assert band["t_low"] == 0.4 and band["t_high"] == 0.8 and band["reachable"]
    m = U.three_way_metrics(y, s, band, 0.5)
    assert np.isclose(m["coverage"], 0.8) and np.isclose(m["decided_accuracy"], 1.0)
    assert np.isclose(m["static_precision_decided"], 1.0) and np.isclose(m["static_recall_decided"], 1.0)
    assert np.isclose(m["fire_npv_decided"], 1.0) and np.isclose(m["uncertain_share"], 0.2)
    # forced binary at 0.5 errs on the point at 0.6 only, and that point is uncertain
    assert m["binary_errors"] == 1 and m["errors_in_uncertain"] == 1
    # target 0.90: the forced binary decision is accurate enough, so nothing is uncertain
    loose = U.choose_accuracy_band(y, s, 0.90, min_support=1)
    ml = U.three_way_metrics(y, s, loose, 0.5)
    assert np.isclose(ml["coverage"], 1.0) and ml["decided_accuracy"] >= 0.9
    print("accuracy band on a tiny set: passed")


def test_accuracy_band_unreachable():
    y = np.array([0, 1] * 500)
    s = np.full(1000, 0.5)                                  # a score that says nothing
    band = U.choose_accuracy_band(y, s, 0.96)
    assert not band["reachable"] and band["t_low"] is None and band["t_high"] is None
    assert (U.decide(s, band) == U.UNCERTAIN).all()
    print("unreachable target reported: passed")


def test_saved_bundles():
    """Needs python -m src.uncertain to have run. Old keys stay, the bands and calibrator are present
    (in the bundle, or in a sidecar file when a bundle was locked by a running app), and the history
    model stays under 1 MB."""
    for name, old_keys, sidecar in (("best_model_history.joblib", {"model", "features", "thresholds"},
                                     "best_model_history_band.json"),
                                    ("best_model_headline.joblib", {"model", "features", "threshold"},
                                     "best_model_headline_band.json")):
        b = joblib.load(MODEL_DIR / name, mmap_mode="r")
        assert old_keys <= set(b), name
        got = U.bands_of(b, sidecar)
        assert got and got["strict"] and got["calibrator"]["kind"] in U.KINDS, name
        d, st = got["default"], got["strict"]
        assert d["definition"] == "accuracy" and d["target"] == 0.96 and st["target"] == 0.98
        assert st["oof_coverage"] <= d["oof_coverage"] + 1e-9
        assert "band_precision" in b or (MODEL_DIR / sidecar).exists()
    hist = joblib.load(MODEL_DIR / "best_model_history.joblib")
    size = (MODEL_DIR / "best_model_history.joblib").stat().st_size
    assert size < 1_000_000
    assert hist["band_precision"]["definition"] if "definition" in hist["band_precision"] else True
    assert abs(hist["band_precision"]["t_high"] - hist["thresholds"]["precision >= 0.90"]) < 1e-5
    print("saved bundles: passed, history model", round(size / 1e3), "KB")


def main(bundles=True):
    test_precision_band_uses_only_out_of_fold_scores()
    test_accuracy_band_out_of_fold_only()
    test_accuracy_band_tiny_set()
    test_accuracy_band_unreachable()
    test_coverage_and_precision_tiny_set()
    test_thresholds_match_the_existing_operating_point()
    test_calibrators()
    if bundles:
        test_saved_bundles()


if __name__ == "__main__":
    main(bundles="--no-bundles" not in sys.argv)
    from tests.test_live import peak_memory
    ws, commit = peak_memory()
    print(f"peak memory: working set {ws:.0f} MB, commit {commit:.0f} MB")

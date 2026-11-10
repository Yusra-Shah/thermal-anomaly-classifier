"""How much of the instant tier is a day or night effect.

Run from the project root:  python -m src.daynight [--stage tables|xgb|transfer|all]

Parts (results in reports/daynight_*.csv):
  stratified  static PR-AUC, precision, recall and F1 of the instant tier (shipped random forest), the
              history tier, the one line rule and a trivial baseline (static when the detection is at
              night), separately for day and night detections, each with its own majority baseline, on
              the grouped split, the temporal split, leave one region out and the NOAA-20 and NOAA-21
              cell disjoint transfer sets
  ablation    the instant tier retrained with XGBoost (100 trees, grouped split) without one group of
              features at a time, thresholds from out of fold training scores only
  permutation permutation importance of the full XGBoost model on the held out split
  xgb vs rf   whether the small XGBoost instant model is within 0.01 static PR-AUC of the random forest

The label is NASA's repeat based type flag, not independent truth. Scores are cached in
data/processed/daynight_cache.pkl so an interrupted run resumes.
"""
import pickle
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold

from .config import FEATURES, MODEL_DIR, PROC_DIR, REPORT_DIR, TARGET
from .train import best_threshold, build_models

TREES = 100
FOLDS = 5
SEED = 42
GROUPED = "grouped seed 42"
ABLATIONS = {
    "full model": [],
    "without is_night": ["is_night"],
    "without confidence_ord": ["confidence_ord"],
    "without frp and frp_per_area": ["frp", "frp_per_area"],
    "without bright_diff": ["bright_diff"],
    "without scan, track and pixel_area": ["scan", "track", "pixel_area"],
}
STRATA = ("all", "day", "night")


# ------------------------------------------------------------------ metrics
def metrics(y, score, thr):
    """Static PR-AUC, precision, recall and F1 at a threshold, plus the majority baselines."""
    y, score = np.asarray(y), np.asarray(score, float)
    out = {"n": int(len(y)), "static_share": float(y.mean()) if len(y) else float("nan")}
    out["majority_pr_auc"] = out["static_share"]
    out["majority_accuracy"] = float(max(y.mean(), 1 - y.mean())) if len(y) else float("nan")
    if len(y) == 0 or y.min() == y.max():                       # one class only: ranking metrics are undefined
        out.update(pr_auc=float("nan"), precision=float("nan"), recall=float("nan"), f1=float("nan"))
        return out
    pred = (score >= thr).astype(int)
    out.update(pr_auc=float(average_precision_score(y, score)),
               precision=float(precision_score(y, pred, zero_division=0)),
               recall=float(recall_score(y, pred, zero_division=0)),
               f1=float(f1_score(y, pred, zero_division=0)))
    return out


def stratified(y, score, thr, night):
    """One row of metrics per stratum: all, day (is_night 0) and night (is_night 1). Every stratum is
    scored with its own scores, so its PR-AUC is compared with its own static share."""
    y, score, night = np.asarray(y), np.asarray(score, float), np.asarray(night).astype(int)
    rows = []
    for name, sel in (("all", np.ones(len(y), bool)), ("day", night == 0), ("night", night == 1)):
        rows.append({"stratum": name, "threshold": float(thr), **metrics(y[sel], score[sel], thr)})
    return rows


def night_baseline(night):
    """Trivial baseline: predict static when the detection is at night. Score and decision are the same 0 or 1."""
    return np.asarray(night).astype(float), 0.5


# ------------------------------------------------------------------ xgboost instant model
def make_xgb(pos_weight, trees=TREES, n_jobs=4):
    m = build_models(pos_weight, trees)["XGBoost"]
    m.set_params(n_jobs=n_jobs)
    return m


def feature_sets(features=None):
    """Feature lists of the ablation: the full list and the list without each named group."""
    features = list(features or FEATURES)
    out = {}
    for name, drop in ABLATIONS.items():
        missing = [d for d in drop if d not in features]
        if missing:
            raise ValueError(f"features not in the model: {missing}")
        out[name] = [f for f in features if f not in drop]
    return out


def fit_variant(Xtr, ytr, gtr, feats, trees=TREES, folds=FOLDS):
    """XGBoost on the training part: out of fold scores, the threshold from them only (F1 optimal, as the
    shipped instant tier), then the final model on the whole training part. Test data is never used."""
    ytr = np.asarray(ytr)
    pw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    oof = np.zeros(len(ytr))
    for a, b in GroupKFold(n_splits=folds).split(Xtr, ytr, gtr):
        m = make_xgb(pw, trees).fit(Xtr.iloc[a][feats], ytr[a])
        oof[b] = m.predict_proba(Xtr.iloc[b][feats])[:, 1]
        del m
    thr = best_threshold(ytr, oof)
    model = make_xgb(pw, trees).fit(Xtr[feats], ytr)
    return model, oof, thr


def permutation_importance(model, X, y, feats, thr, repeats=5, seed=0):
    """Drop in static PR-AUC and F1 when one feature column is shuffled, on the held out rows."""
    rng = np.random.RandomState(seed)
    base_p = model.predict_proba(X[feats])[:, 1]
    base_ap = average_precision_score(y, base_p)
    base_f1 = f1_score(y, (base_p >= thr).astype(int), zero_division=0)
    rows = []
    for f in feats:
        d_ap, d_f1 = [], []
        for _ in range(repeats):
            Xp = X[feats].copy()
            Xp[f] = rng.permutation(Xp[f].to_numpy())
            p = model.predict_proba(Xp)[:, 1]
            d_ap.append(base_ap - average_precision_score(y, p))
            d_f1.append(base_f1 - f1_score(y, (p >= thr).astype(int), zero_division=0))
        rows.append({"feature": f, "pr_auc_drop_mean": float(np.mean(d_ap)), "pr_auc_drop_std": float(np.std(d_ap)),
                     "f1_drop_mean": float(np.mean(d_f1)), "f1_drop_std": float(np.std(d_f1)),
                     "base_pr_auc": float(base_ap), "base_f1": float(base_f1)})
    return pd.DataFrame(rows).sort_values("pr_auc_drop_mean", ascending=False)


# ------------------------------------------------------------------ stages
CACHE = PROC_DIR / "daynight_cache.pkl"


def _load():
    return pickle.load(open(CACHE, "rb")) if CACHE.exists() else {}


def _save(store):
    pickle.dump(store, open(CACHE, "wb"))


def stage_tables(df, splits, store):
    """Per split: y, night, scores and thresholds of the instant (random forest), history and rule tiers."""
    from .history2_ops import thresholds as hist_thresholds
    from .history_eval import RULE_COL, choose_k
    from .uncertain import compute_instant_rf, history_scores
    inst = compute_instant_rf(df, splits)
    hist = history_scores(df, splits)
    for name, (tr, te, _) in splits.items():
        key = ("tables", name)
        if key in store:
            continue
        ytr, yte = df.iloc[tr][TARGET].to_numpy(), df.iloc[te][TARGET].to_numpy()
        k = choose_k(df.iloc[tr])
        store[key] = {
            "y": yte, "night": df.iloc[te]["is_night"].to_numpy(),
            "instant": (np.asarray(inst[name]["p"], float), best_threshold(ytr, np.asarray(inst[name]["oof"], float))),
            "history": (np.asarray(hist[name]["p"], float),
                        hist_thresholds(ytr, np.asarray(hist[name]["oof"], float))["precision >= 0.90"]),
            "rule": (df.iloc[te][RULE_COL].to_numpy(float), float(k))}
    _save(store)


def stage_xgb(df, splits, store):
    """Full XGBoost instant model on the grouped and temporal splits, and the ablation on the grouped split."""
    for name in (GROUPED, "temporal"):
        tr, te, _ = splits[name]
        Xtr, ytr, gtr = df.iloc[tr], df.iloc[tr][TARGET].to_numpy(), df.iloc[tr]["cell_id"].to_numpy()
        sets = feature_sets() if name == GROUPED else {"full model": list(FEATURES)}
        for variant, feats in sets.items():
            key = ("xgb", name, variant)
            if key in store:
                continue
            model, oof, thr = fit_variant(Xtr, ytr, gtr, feats)
            p = model.predict_proba(df.iloc[te][feats])[:, 1]
            store[key] = {"p": p.astype(np.float32), "oof": oof.astype(np.float32), "thr": thr, "feats": feats}
            if name == GROUPED and variant == "full model":
                joblib.dump({"model": model, "features": feats, "threshold": thr}, PROC_DIR / "daynight_xgb_full.joblib")
                imp = permutation_importance(model, df.iloc[te], df.iloc[te][TARGET].to_numpy(), feats, thr)
                imp.round(4).to_csv(REPORT_DIR / "daynight_permutation.csv", index=False)
                print(imp.round(4).to_string(index=False), flush=True)
            del model
            _save(store)
            print("xgb", name, variant, "done", flush=True)


def stage_transfer(df, splits, store):
    """NOAA-20 and NOAA-21 cell disjoint sets scored with the shipped models and the XGBoost instant model."""
    from .history2_ops import thresholds as hist_thresholds
    from .history_eval import RULE_COL, choose_k
    from .noaa20_transfer import build_noaa20
    tr, _, _ = splits[GROUPED]
    train_cells = set(df.iloc[tr]["cell_id"])
    k = choose_k(df.iloc[tr])
    inst_b = joblib.load(MODEL_DIR / "best_model_headline.joblib", mmap_mode="r")
    hist_b = joblib.load(MODEL_DIR / "best_model_history.joblib", mmap_mode="r")
    xg = joblib.load(PROC_DIR / "daynight_xgb_full.joblib") if (PROC_DIR / "daynight_xgb_full.joblib").exists() else None
    for sat in ("noaa20", "noaa21"):
        key = ("transfer", sat)
        if key in store:
            continue
        d = build_noaa20(sat)
        d = d[~d["cell_id"].isin(train_cells)].reset_index(drop=True)            # cell disjoint
        entry = {"y": d[TARGET].to_numpy(), "night": d["is_night"].to_numpy(),
                 "instant": (inst_b["model"].predict_proba(d[inst_b["features"]])[:, 1], float(inst_b["threshold"])),
                 "history": (hist_b["model"].predict_proba(d[hist_b["features"]])[:, 1],
                             float(hist_b["thresholds"]["precision >= 0.90"])),
                 "rule": (d[RULE_COL].to_numpy(float), float(k))}
        if xg is not None:
            entry["instant_xgb"] = (xg["model"].predict_proba(d[xg["features"]])[:, 1], float(xg["threshold"]))
        store[key] = entry
        del d
        _save(store)
        print("transfer", sat, "done", flush=True)


def build_tables(store, splits):
    """Stratified table (all tiers and the night baseline) and the comparison with the baseline."""
    rows = []
    sets = [(name, store[("tables", name)]) for name in splits if ("tables", name) in store]
    sets += [(f"{sat.upper()[:4]}-{sat[-2:]} cell disjoint", store[("transfer", sat)])
             for sat in ("noaa20", "noaa21") if ("transfer", sat) in store]
    for name, e in sets:
        tiers = {"instant (random forest, replaced)": e["instant"], "history": e["history"],
                 "rule (days in prior 30)": e["rule"], "night baseline": night_baseline(e["night"])}
        if "instant_xgb" in e:
            tiers["instant (XGBoost, shipped)"] = e["instant_xgb"]
        for split_key in (name,):
            x = store.get(("xgb", split_key, "full model"))
            if x is not None and "instant_xgb" not in e:
                tiers["instant (XGBoost, shipped)"] = (np.asarray(x["p"], float), x["thr"])
        for tier, (score, thr) in tiers.items():
            for r in stratified(e["y"], score, thr, e["night"]):
                rows.append({"split": name, "tier": tier, **r})
    out = pd.DataFrame(rows)
    out.round(4).to_csv(REPORT_DIR / "daynight_stratified.csv", index=False)
    a = out[out["stratum"] == "all"]
    cmp_rows = []
    for name in a["split"].unique():
        s = a[a["split"] == name].set_index("tier")
        b = s.loc["night baseline"]
        for tier in s.index:
            if tier == "night baseline":
                continue
            cmp_rows.append({"split": name, "tier": tier, "pr_auc": s.loc[tier, "pr_auc"], "baseline_pr_auc": b["pr_auc"],
                             "pr_auc_above_baseline": s.loc[tier, "pr_auc"] - b["pr_auc"],
                             "f1": s.loc[tier, "f1"], "baseline_f1": b["f1"],
                             "f1_above_baseline": s.loc[tier, "f1"] - b["f1"],
                             "majority_pr_auc": s.loc[tier, "majority_pr_auc"]})
    cmp_ = pd.DataFrame(cmp_rows)
    cmp_.round(4).to_csv(REPORT_DIR / "daynight_vs_baseline.csv", index=False)
    return out, cmp_


def build_ablation(df, splits, store):
    tr, te, _ = splits[GROUPED]
    y, night = df.iloc[te][TARGET].to_numpy(), df.iloc[te]["is_night"].to_numpy()
    rows = []
    for variant in ABLATIONS:
        x = store.get(("xgb", GROUPED, variant))
        if x is None:
            continue
        for r in stratified(y, np.asarray(x["p"], float), x["thr"], night):
            rows.append({"variant": variant, "n_features": len(x["feats"]), **r})
    out = pd.DataFrame(rows)
    out.round(4).to_csv(REPORT_DIR / "daynight_ablation.csv", index=False)
    return out


def build_xgb_vs_rf(store, splits):
    """Static PR-AUC of the saved random forest and of the XGBoost with all features, grouped and temporal."""
    rows = []
    for name in (GROUPED, "temporal"):
        e, x = store.get(("tables", name)), store.get(("xgb", name, "full model"))
        if e is None or x is None:
            continue
        y = e["y"]
        rf, xg = average_precision_score(y, e["instant"][0]), average_precision_score(y, x["p"])
        rows.append({"split": name, "rf_pr_auc": rf, "xgb_pr_auc": xg, "difference_xgb_minus_rf": xg - rf,
                     "within_0.01": bool(xg - rf >= -0.01), "n_test": len(y)})
    out = pd.DataFrame(rows)
    out.round(4).to_csv(REPORT_DIR / "daynight_xgb_vs_rf.csv", index=False)
    return out


def ship_xgb_instant(df, splits):
    """Make the XGBoost instant model the app's instant model: models/best_model_headline.joblib (model,
    features, threshold from out of fold scores) and the archive scores in scored_headline.parquet. The
    bands and the calibrator are added afterwards by python -m src.uncertain, the same procedure as before.
    The old random forest bundle must be backed up outside the repo before this runs."""
    store = _load()
    x = store[("xgb", GROUPED, "full model")]
    saved = joblib.load(PROC_DIR / "daynight_xgb_full.joblib")
    tr, te, _ = splits[GROUPED]
    ytr = df.iloc[tr][TARGET].to_numpy()
    assert abs(best_threshold(ytr, np.asarray(x["oof"], float)) - float(saved["threshold"])) < 1e-9
    bundle = {"model": saved["model"], "features": list(saved["features"]), "threshold": float(saved["threshold"]),
              "name": "XGBoost, 100 trees"}
    path = MODEL_DIR / "best_model_headline.joblib"
    tmp = path.with_suffix(".tmp")
    joblib.dump(bundle, tmp)
    tmp.replace(path)
    print("saved", path.name, round(path.stat().st_size / 1e6, 2), "MB")
    old = pd.read_parquet(PROC_DIR / "scored_headline.parquet")
    keys = ["latitude", "longitude", "acq_date", "bright_ti4", "frp"]
    assert len(old) == len(df) and all(np.array_equal(old[k].to_numpy(), df[k].to_numpy()) for k in keys),         "scored_headline.parquet rows differ from the prepared data"
    p = saved["model"].predict_proba(df[bundle["features"]])[:, 1]
    old["p_static"] = p
    old["pred_static"] = (p >= bundle["threshold"]).astype(int)
    old.to_parquet(PROC_DIR / "scored_headline.parquet", index=False)
    held = old[old["in_test"]]
    print("held out precision", precision_score(held["is_static"], held["pred_static"]),
          "recall", recall_score(held["is_static"], held["pred_static"]))


def main(argv=None):
    from .history2_eval import prepare
    from .uncertain import _splits
    argv = sys.argv[1:] if argv is None else argv
    stage = argv[argv.index("--stage") + 1] if "--stage" in argv else "all"
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    df = prepare()
    splits = _splits(df)
    store = _load()
    if stage in ("tables", "all"):
        stage_tables(df, splits, store)
    if stage in ("xgb", "all"):
        stage_xgb(df, splits, store)
    if stage in ("transfer", "all"):
        stage_transfer(df, splits, store)
    if stage == "ship":
        ship_xgb_instant(df, splits)
        return
    pd.set_option("display.width", 250, "display.max_columns", 30)
    out, cmp_ = build_tables(store, splits)
    print(cmp_.round(3).to_string(index=False))
    print(build_xgb_vs_rf(store, splits).round(4).to_string(index=False))
    print(build_ablation(df, splits, store).round(3).to_string(index=False))


if __name__ == "__main__":
    main()

"""Calibration and the uncertain band for both model tiers.

A three way decision: static (score at or above t_high), fire (score at or below t_low) and
uncertain in between. Both thresholds come from out of fold predictions on the training
part only.

Default definition: the scores are calibrated (Platt or isotonic, fitted on the out of fold
scores) and the band is the pair of thresholds with the highest coverage whose accuracy on the
decided out of fold points is at least 0.96. A strict band uses 0.98.

Earlier definition, kept as a documented negative result: static precision at least 0.90 on the
high side and fire side precision (negative predictive value) at least 0.95 on the low side.
Fire is the majority class, so the fire side target is met almost everywhere, t_low ends up
above t_high for the history tier, and the band is empty.

Run from the project root:
    python -m src.uncertain --instant     once, computes the instant tier predictions (slow)
    python -m src.uncertain               calibration, band evaluation, reports, bundle keys

NASA's static flag is the label here. It is not independent truth.
"""
import argparse
import json
import pickle

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, KFold

from .config import FEATURES, MODEL_DIR, PROC_DIR, REPORT_DIR, TARGET

FIRE, STATIC, UNCERTAIN = 0, 1, 2
PREC_TARGET, NPV_TARGET = 0.90, 0.95
MIN_SUPPORT = 100                  # a threshold must rest on at least this many training rows
KINDS = ("platt", "isotonic")
SEED = 42
SETUP = "D instant+history v1+v2"
LEVELS = [round(x, 3) for x in np.arange(0.60, 0.9951, 0.02)] + [0.99]
ACC_DEFAULT, ACC_STRICT = 0.96, 0.98
ACC_LEVELS = [round(x, 3) for x in np.arange(0.90, 0.9951, 0.005)]
BINARY_T = 0.5                     # forced binary decision on calibrated scores
DEF_DEFAULT = "accuracy 0.96 (default)"
DEF_STRICT = "accuracy 0.98 (strict)"
DEF_PRECISION = "precision based (negative result)"


# ---------------------------------------------------------------- calibration
def brier(y, p):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def reliability(y, p, bins=10):
    """Equal width reliability table: rows of bin, count, mean score, observed static share."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        rows.append({"bin": f"{edges[b]:.1f} to {edges[b + 1]:.1f}", "n": int(m.sum()),
                     "mean_p": float(p[m].mean()) if m.any() else np.nan,
                     "observed_static": float(y[m].mean()) if m.any() else np.nan})
    return pd.DataFrame(rows)


def ece(y, p, bins=10):
    t = reliability(y, p, bins)
    t = t[t["n"] > 0]
    return float((t["n"] * (t["observed_static"] - t["mean_p"]).abs()).sum() / t["n"].sum())


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_calibrator(kind, p, y):
    """A small plain dict, so it can sit in the model bundle without extra classes."""
    if kind == "platt":
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p)[:, None], np.asarray(y))
        return {"kind": "platt", "a": float(lr.coef_[0, 0]), "b": float(lr.intercept_[0])}
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p, y)
    return {"kind": "isotonic", "x": iso.X_thresholds_.astype(np.float32),
            "y": iso.y_thresholds_.astype(np.float32)}


def apply_calibrator(c, p):
    p = np.asarray(p, float)
    if c["kind"] == "platt":
        return 1.0 / (1.0 + np.exp(-(c["a"] * _logit(p) + c["b"])))
    return np.interp(p, c["x"], c["y"])


def pick_calibrator(p, y, groups=None, folds=5):
    """Fit both calibrators on out of fold scores and keep the one with the lower Brier score.

    The Brier score is measured by cross validation inside the out of fold scores, so a
    calibrator is never scored on rows it was fitted on. Returns (calibrator, scores).
    """
    p, y = np.asarray(p, float), np.asarray(y)
    splitter = GroupKFold(folds) if groups is not None else KFold(folds, shuffle=True, random_state=0)
    pred = {k: np.zeros(len(p)) for k in KINDS}
    for a, b in splitter.split(p, y, groups):
        for k in KINDS:
            pred[k][b] = apply_calibrator(fit_calibrator(k, p[a], y[a]), p[b])
    scores = {k: brier(y, pred[k]) for k in KINDS}
    best = min(scores, key=scores.get)
    return fit_calibrator(best, p, y), scores


# ---------------------------------------------------------------- band
def _tie_ends(ps):
    """Indices where the next score differs (the end of each run of equal scores)."""
    return np.flatnonzero(np.r_[ps[1:] != ps[:-1], True])


def high_threshold(y, p, target=PREC_TARGET, min_support=MIN_SUPPORT):
    """Smallest score t with static precision of (p >= t) at least target, or None."""
    y, p = np.asarray(y), np.asarray(p, float)
    order = np.argsort(-p, kind="stable")
    ps, ys = p[order], y[order]
    prec = np.cumsum(ys) / np.arange(1, len(ys) + 1)
    ends = _tie_ends(ps)
    ok = ends[(prec[ends] >= target) & (ends + 1 >= min_support)]
    return float(ps[ok.max()]) if len(ok) else None


def low_threshold(y, p, target=NPV_TARGET, min_support=MIN_SUPPORT):
    """Largest score t with fire side precision (NPV) of (p <= t) at least target, or None."""
    y, p = np.asarray(y), np.asarray(p, float)
    order = np.argsort(p, kind="stable")
    ps, ys = p[order], y[order]
    npv = np.cumsum(1 - ys) / np.arange(1, len(ys) + 1)
    ends = _tie_ends(ps)
    ok = ends[(npv[ends] >= target) & (ends + 1 >= min_support)]
    return float(ps[ok.max()]) if len(ok) else None


def choose_band(y_oof, p_oof, prec_target=PREC_TARGET, npv_target=NPV_TARGET):
    """The band from out of fold scores only. If the two ranges overlap the single threshold
    t_high is used and there is no uncertain region."""
    th = high_threshold(y_oof, p_oof, prec_target)
    tl = low_threshold(y_oof, p_oof, npv_target)
    overlap = th is not None and tl is not None and tl >= th
    return {"t_low": tl, "t_high": th, "overlap": bool(overlap),
            "precision_target": prec_target, "npv_target": npv_target, "min_support": MIN_SUPPORT}


def decide(p, band):
    """FIRE, STATIC or UNCERTAIN per score."""
    p = np.asarray(p, float)
    out = np.full(len(p), UNCERTAIN)
    th, tl = band.get("t_high"), band.get("t_low")
    if band.get("overlap") and th is not None:
        out[p < th] = FIRE
    elif tl is not None:
        out[p <= tl] = FIRE
    if th is not None:
        out[p >= th] = STATIC
    return out


def three_way_metrics(y, p, band, binary_threshold=None):
    """Coverage and precision of the three way decision against the forced binary one, at
    binary_threshold or, if none is given, at t_high."""
    y, p = np.asarray(y), np.asarray(p, float)
    d = decide(p, band)
    static, fire, unc = d == STATIC, d == FIRE, d == UNCERTAIN
    tp, fp = int((static & (y == 1)).sum()), int((static & (y == 0)).sum())
    fn, tn = int((fire & (y == 1)).sum()), int((fire & (y == 0)).sum())
    nan = float("nan")

    def ratio(a, b):
        return a / b if b else nan
    decided = tp + fp + fn + tn
    out = {"n_test": len(y), "coverage": decided / len(y),
           "static_precision_decided": ratio(tp, tp + fp),
           "static_recall_decided": ratio(tp, tp + fn),
           "fire_npv_decided": ratio(tn, tn + fn),
           "decided_accuracy": ratio(tp + tn, decided),
           "static_found_share": ratio(tp, int((y == 1).sum())),
           "uncertain_share": float(unc.mean()),
           "uncertain_static_share": float(y[unc].mean()) if unc.any() else nan}
    th = band.get("t_high") if binary_threshold is None else binary_threshold
    if th is None:
        out.update(binary_precision=nan, binary_recall=nan, binary_errors=nan,
                   errors_in_uncertain=nan, errors_in_uncertain_share=nan, binary_error_rate=nan)
        return out
    b = (p >= th).astype(int)
    err = b != y
    btp, bfp = int(((b == 1) & (y == 1)).sum()), int(((b == 1) & (y == 0)).sum())
    out.update(binary_precision=ratio(btp, btp + bfp),
               binary_recall=ratio(btp, int((y == 1).sum())),
               binary_errors=int(err.sum()), binary_error_rate=float(err.mean()),
               errors_in_uncertain=int((err & unc).sum()),
               errors_in_uncertain_share=ratio(int((err & unc).sum()), int(err.sum())))
    return out


def sweep(y_oof, p_oof, y_te, p_te, levels=LEVELS):
    """Coverage against precision as the band widens: one band per target level, chosen on
    out of fold scores and measured on the test scores."""
    rows = []
    for q in levels:
        band = choose_band(y_oof, p_oof, q, q)
        m = three_way_metrics(y_te, p_te, band)
        rows.append({"target": q, "t_low": band["t_low"], "t_high": band["t_high"],
                     "overlap": band["overlap"], "coverage": m["coverage"],
                     "decided_accuracy": m["decided_accuracy"],
                     "static_precision_decided": m["static_precision_decided"],
                     "fire_npv_decided": m["fire_npv_decided"]})
    return pd.DataFrame(rows)


def choose_accuracy_band(y, scores, target, min_support=MIN_SUPPORT, max_candidates=400):
    """The band with the highest coverage whose accuracy on the decided points is at least target.

    Fire is decided for scores at or below t_low and static for scores at or above t_high; both
    thresholds are searched (symmetric pairs are included). Uses the given arrays only, so with
    out of fold training scores it cannot see test data. If no pair reaches the target the band is
    marked not reachable and decides nothing.
    """
    y, s_ = np.asarray(y).astype(int), np.asarray(scores, float)
    n = len(y)
    order = np.argsort(s_, kind="stable")
    ss, ys = s_[order], y[order]
    neg = np.r_[0, np.cumsum(1 - ys)]
    pos = np.r_[0, np.cumsum(ys)]
    cuts = np.r_[0, np.flatnonzero(ss[1:] != ss[:-1]) + 1, n]        # cut positions between runs of ties
    if len(cuts) > max_candidates:
        cuts = cuts[np.unique(np.linspace(0, len(cuts) - 1, max_candidates).astype(int))]
    i, j = cuts[:, None], cuts[None, :]                                # fire: first i rows, static: last n - j rows
    decided = i + (n - j)
    correct = neg[i] + (pos[n] - pos[j])
    valid = (i <= j) & ((i == 0) | (i >= min_support)) & ((n - j == 0) | (n - j >= min_support)) & (decided > 0)
    acc = np.where(decided > 0, correct / np.maximum(decided, 1), 0.0)
    ok = valid & (acc >= target)
    base = {"definition": "accuracy", "target": float(target), "space": "calibrated",
            "overlap": False, "min_support": min_support}
    if not ok.any():
        far = valid & (decided >= 0.1 * n)
        best = float(acc[far].max()) if far.any() else float("nan")
        return {**base, "reachable": False, "t_low": None, "t_high": None,
                "oof_coverage": 0.0, "oof_accuracy": float("nan"), "best_accuracy_at_10_percent": best}
    pick = np.where(ok, decided + acc * 1e-3, -1.0)
    ii, jj = np.unravel_index(int(np.argmax(pick)), pick.shape)
    lo, hi = int(cuts[ii]), int(cuts[jj])
    return {**base, "reachable": True,
            "t_low": float(ss[lo - 1]) if lo > 0 else None,
            "t_high": float(ss[hi]) if hi < n else None,
            "oof_coverage": float(decided[ii, jj] / n), "oof_accuracy": float(acc[ii, jj])}


def sweep_accuracy(y_oof, s_oof, y_te, s_te, levels=ACC_LEVELS):
    """Coverage against accuracy as the accuracy target moves, bands chosen on out of fold scores."""
    rows = []
    for q in levels:
        band = choose_accuracy_band(y_oof, s_oof, q)
        m = three_way_metrics(y_te, s_te, band, BINARY_T)
        rows.append({"target": q, "reachable": band["reachable"], "t_low": band["t_low"],
                     "t_high": band["t_high"], "coverage": m["coverage"],
                     "decided_accuracy": m["decided_accuracy"],
                     "static_precision_decided": m["static_precision_decided"],
                     "fire_npv_decided": m["fire_npv_decided"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- study
def _splits(df):
    """Train and test row positions for the splits of the study (same as the earlier reports)."""
    from sklearn.model_selection import GroupShuffleSplit

    from .history_eval import COMPLETE_DAYS, REGIONS, SPLIT_DATE
    out = {}
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
                  .split(df, df[TARGET], df["cell_id"]))
    out["grouped seed 42"] = (tr, te, SEED)
    tr_m = (df["acq_date"] < SPLIT_DATE).values
    te_m = ((df["acq_date"] >= SPLIT_DATE) & (df["days_into_data"] >= COMPLETE_DAYS)).values
    out["temporal"] = (np.flatnonzero(tr_m), np.flatnonzero(te_m), "temporal")
    for region in REGIONS.values():
        m = (df["region"] == region).values
        out[f"leave out {region}"] = (np.flatnonzero(~m), np.flatnonzero(m), region)
    return out


def compute_instant(df, splits):
    """Instant tier scores. The grouped split uses the shipped Random Forest (300 trees, 5 folds,
    as in src.train). The other splits use 100 trees and 3 folds, as in the earlier region and
    history reports. Cached in data/processed/uncertain_instant.pkl."""
    from .train import best_threshold, build_models
    cache = PROC_DIR / "uncertain_instant.pkl"
    store = pickle.load(open(cache, "rb")) if cache.exists() else {}
    for name, (tr, te, _) in splits.items():
        if name in store:
            continue
        grouped = name == "grouped seed 42"
        trees, folds = (300, 5) if grouped else (100, 3)
        X, y, g = df.iloc[tr][FEATURES], df.iloc[tr][TARGET].values, df.iloc[tr]["cell_id"].values
        pw = float((y == 0).sum() / max((y == 1).sum(), 1))
        oof = np.zeros(len(y))
        for a, b in GroupKFold(n_splits=folds).split(X, y, g):
            m = build_models(pw, trees)["Random Forest"].fit(X.iloc[a], y[a])
            oof[b] = m.predict_proba(X.iloc[b])[:, 1]
            del m
        if grouped:
            bundle = joblib.load(MODEL_DIR / "best_model_headline.joblib", mmap_mode="r")
            p = bundle["model"].predict_proba(df.iloc[te][FEATURES])[:, 1]
            same = abs(best_threshold(y, oof) - float(bundle["threshold"])) < 1e-6
            print("instant out of fold threshold equals the shipped one:", same, flush=True)
        else:
            m = build_models(pw, trees)["Random Forest"].fit(X, y)
            p = m.predict_proba(df.iloc[te][FEATURES])[:, 1]
            del m
        store[name] = {"oof": oof.astype(np.float32), "p": p.astype(np.float32)}
        pickle.dump(store, open(cache, "wb"))
        print("instant", name, "done", flush=True)
    return store


def history_scores(df, splits):
    """Stored out of fold and test scores of setup D, default XGBoost, from the earlier reports."""
    stores = {}
    for part in "abc":
        with open(PROC_DIR / f"history2_preds_{part}.pkl", "rb") as f:
            stores[part] = pickle.load(f)
    out = {}
    for name, (tr, te, key) in splits.items():
        store = stores["a" if key == SEED else "c" if key == "temporal" else "b"]
        ytr = store[(key, "y")]["ytr"]
        assert np.array_equal(ytr, df.iloc[tr][TARGET].values), name       # same rows as the stored run
        r = store[(key, SETUP, "XGBoost")]
        out[name] = {"oof": r["oof"], "p": r["p"]}
    return out


def rule_scores(df, splits):
    """The one line rule scores: distinct days with detections in the prior 30 days."""
    col = "distinct_days_prior_30"
    return {n: {"oof": df.iloc[tr][col].values.astype(float), "p": df.iloc[te][col].values.astype(float)}
            for n, (tr, te, _) in splits.items()}


def main():
    from .history2_eval import prepare
    for d in (PROC_DIR, REPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)
    df = prepare()
    splits = _splits(df)
    tiers = {"history": history_scores(df, splits), "instant": compute_instant(df, splits),
             "rule": rule_scores(df, splits)}
    y_tr = {n: df.iloc[tr][TARGET].values for n, (tr, te, _) in splits.items()}
    y_te = {n: df.iloc[te][TARGET].values for n, (tr, te, _) in splits.items()}
    groups = {n: df.iloc[tr]["cell_id"].values for n, (tr, te, _) in splits.items()}

    eval_rows, region_rows, cal_rows, rel_rows = [], [], [], []
    curve_acc, curve_prec, store, items = [], [], {}, {}
    for tier, per in tiers.items():
        for name in splits:
            oof, p = np.asarray(per[name]["oof"], float), np.asarray(per[name]["p"], float)
            ytr, yte = y_tr[name], y_te[name]
            cal, scores = pick_calibrator(oof, ytr, groups[name])          # out of fold scores only
            c_oof, c_te = apply_calibrator(cal, oof), apply_calibrator(cal, p)
            bands = {DEF_DEFAULT: choose_accuracy_band(ytr, c_oof, ACC_DEFAULT),
                     DEF_STRICT: choose_accuracy_band(ytr, c_oof, ACC_STRICT),
                     DEF_PRECISION: choose_band(ytr, oof)}                 # earlier definition, raw scores
            for definition, band in bands.items():
                precision_def = definition == DEF_PRECISION
                m = three_way_metrics(yte, p if precision_def else c_te, band, None if precision_def else BINARY_T)
                eval_rows.append({"tier": tier, "split": name, "definition": definition,
                                  "reachable": band.get("reachable", True), "t_low": band["t_low"],
                                  "t_high": band["t_high"], "overlap": band["overlap"],
                                  "oof_coverage": band.get("oof_coverage"), "oof_accuracy": band.get("oof_accuracy"),
                                  **m})
            for curves, frame in ((curve_acc, sweep_accuracy(ytr, c_oof, yte, c_te)),
                                  (curve_prec, sweep(ytr, oof, yte, p))):
                frame.insert(0, "split", name)
                frame.insert(0, "tier", tier)
                curves.append(frame)
            store[(tier, name)] = {"p": p, "c": c_te, "y": yte, "bands": bands}
            if tier != "rule" and name in ("grouped seed 42", "temporal"):
                pc = c_te
                cal_rows.append({"tier": tier, "split": name, "chosen": cal["kind"],
                                 "oof_brier_platt": scores["platt"], "oof_brier_isotonic": scores["isotonic"],
                                 "test_brier_raw": brier(yte, p), "test_brier_calibrated": brier(yte, pc),
                                 "test_ece_raw": ece(yte, p), "test_ece_calibrated": ece(yte, pc),
                                 "n_test": len(yte)})
                for ver, q in (("raw", p), ("calibrated", pc)):
                    r = reliability(yte, q)
                    r.insert(0, "version", ver)
                    r.insert(0, "split", name)
                    r.insert(0, "tier", tier)
                    rel_rows.append(r)
            if name == "grouped seed 42":
                items[tier] = {"band": bands[DEF_DEFAULT], "band_strict": bands[DEF_STRICT],
                               "band_precision": bands[DEF_PRECISION], "calibrator": cal}

    ev = pd.DataFrame(eval_rows)
    ev.round(4).to_csv(REPORT_DIR / "uncertain_eval.csv", index=False)
    ca, cp = pd.concat(curve_acc), pd.concat(curve_prec)
    ca.round(4).to_csv(REPORT_DIR / "uncertain_curve.csv", index=False)
    cp.round(4).to_csv(REPORT_DIR / "uncertain_curve_precision_based.csv", index=False)
    pd.DataFrame(cal_rows).round(4).to_csv(REPORT_DIR / "calibration_summary.csv", index=False)
    pd.concat(rel_rows).round(4).to_csv(REPORT_DIR / "calibration_reliability.csv", index=False)

    # Uncertain share per region: leave one region out (the model never saw the region) and the
    # held out grouped split cut by region, for every definition.
    for tier in ("history", "instant", "rule"):
        for setting in ("leave out", "grouped seed 42"):
            for region in sorted(df["region"].unique()):
                name = f"leave out {region}" if setting == "leave out" else "grouped seed 42"
                tr, te, _ = splits[name]
                sel = (df.iloc[te]["region"] == region).values
                if not sel.any():
                    continue
                st = store[(tier, name)]
                y = st["y"][sel]
                for definition, band in st["bands"].items():
                    s_ = (st["p"] if definition == DEF_PRECISION else st["c"])[sel]
                    d = decide(s_, band)
                    region_rows.append({"tier": tier, "setting": setting, "region": region,
                                        "definition": definition, "n": int(sel.sum()),
                                        "uncertain_share": float((d == UNCERTAIN).mean()),
                                        "static_decided_share": float((d == STATIC).mean()),
                                        "fire_decided_share": float((d == FIRE).mean()),
                                        "decided_accuracy": float((np.where(d == STATIC, 1, 0)[d != UNCERTAIN]
                                                                   == y[d != UNCERTAIN]).mean())
                                        if (d != UNCERTAIN).any() else float("nan"),
                                        "true_static_share": float(y.mean())})
    reg = pd.DataFrame(region_rows)
    reg.round(4).to_csv(REPORT_DIR / "uncertain_regions.csv", index=False)

    plot_curves(ca, ev, DEF_DEFAULT, "uncertain_curve.png", "accuracy based band, dot is the default")
    plot_curves(cp, ev, DEF_PRECISION, "uncertain_curve_precision_based.png",
                "precision based band, negative result")
    plot_reliability(pd.concat(rel_rows))
    save_bundles(items)
    show = ev[ev["definition"] != DEF_PRECISION]
    cols = ["tier", "split", "definition", "reachable", "t_low", "t_high", "coverage", "decided_accuracy",
            "static_precision_decided", "static_recall_decided", "fire_npv_decided", "binary_errors",
            "errors_in_uncertain", "errors_in_uncertain_share"]
    print(show[cols].round(4).to_string())
    print(reg[reg["definition"] == DEF_DEFAULT].round(3).to_string())
    print(pd.DataFrame(cal_rows).round(4).to_string())


def plot_curves(curves, ev, definition, fname, note):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4), sharey=True)
    colors = {"grouped seed 42": "#1f77b4", "temporal": "#c0392b"}
    for ax, tier in zip(axes, ("history", "instant")):
        for name, col in colors.items():
            for t, ls, lab in ((tier, "-", tier), ("rule", "--", "one line rule")):
                c = curves[(curves["tier"] == t) & (curves["split"] == name)].dropna(subset=["decided_accuracy"])
                ax.plot(c["coverage"], c["decided_accuracy"], ls, color=col, lw=1.4, label=f"{name}, {lab}")
            row = ev[(ev["tier"] == tier) & (ev["split"] == name) & (ev["definition"] == definition)].iloc[0]
            if row["coverage"] > 0:
                ax.plot([row["coverage"]], [row["decided_accuracy"]], "o", color=col, ms=6)
        ax.set_title(f"{tier} tier", fontsize=10)
        ax.set_xlabel("Coverage (share decided)")
        ax.set_xlim(0, 1.02)
    axes[0].set_ylabel("Accuracy on decided points")
    axes[0].legend(frameon=False, fontsize=7, loc="lower left")
    fig.suptitle(note, fontsize=9)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / fname, dpi=160)
    plt.close(fig)


def plot_reliability(rel):
    fig, axes = plt.subplots(2, 2, figsize=(8, 7), sharex=True, sharey=True)
    for i, tier in enumerate(("history", "instant")):
        for j, name in enumerate(("grouped seed 42", "temporal")):
            ax = axes[i, j]
            ax.plot([0, 1], [0, 1], ":", color="#999999")
            for ver, col in (("raw", "#c0392b"), ("calibrated", "#1f77b4")):
                r = rel[(rel["tier"] == tier) & (rel["split"] == name) & (rel["version"] == ver)].dropna()
                ax.plot(r["mean_p"], r["observed_static"], "o-", color=col, ms=4, lw=1.2, label=ver)
            ax.set_title(f"{tier}, {name}", fontsize=9)
            if i == 1:
                ax.set_xlabel("Mean score")
            if j == 0:
                ax.set_ylabel("Observed static share")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "calibration_reliability.png", dpi=160)
    plt.close(fig)


def _clean(band):
    return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in band.items()}


def _jsonable(c):
    return {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in c.items()}


def bands_of(bundle, sidecar):
    """The accuracy bands and calibrator of a model, or None. They are read from the bundle, or
    from its sidecar file when the large bundle could not be rewritten while an app held it open.
    Bundles that only hold the earlier precision based band give None."""
    src = bundle if (bundle.get("band") or {}).get("definition") == "accuracy" else None
    if src is None:
        path = MODEL_DIR / sidecar
        if not path.exists():
            return None
        src = json.loads(path.read_text())
        if (src.get("band") or {}).get("definition") != "accuracy":
            return None
    return {"default": src["band"], "strict": src.get("band_strict"), "calibrator": src.get("calibrator")}


def save_bundles(items):
    """Add band (default 0.96), band_strict (0.98), band_precision (the earlier definition) and the
    calibrator to each model bundle. Existing keys are untouched, so older code still loads the
    files. The instant bundle is large and is rewritten through a temporary file. If another
    process (a running app) has it memory mapped, Windows refuses the replace: the same keys then
    go to a small sidecar file that the app reads, and rerunning this script once the app is
    stopped completes the bundle."""
    import gc
    for tier, fname, sidecar in (("history", "best_model_history.joblib", "best_model_history_band.json"),
                                 ("instant", "best_model_headline.joblib", "best_model_headline_band.json")):
        path = MODEL_DIR / fname
        bundle = dict(joblib.load(path))
        extra = {k: (_clean(v) if k.startswith("band") else v) for k, v in items[tier].items()}
        bundle.update(extra)
        tmp = path.with_suffix(".tmp")
        joblib.dump(bundle, tmp)
        del bundle
        gc.collect()
        try:
            tmp.replace(path)
            (MODEL_DIR / sidecar).unlink(missing_ok=True)
            print("saved bands and calibrator in", fname, round(path.stat().st_size / 1e6, 2), "MB")
        except PermissionError:
            tmp.unlink(missing_ok=True)
            side = {k: (_jsonable(v) if k == "calibrator" else v) for k, v in extra.items()}
            (MODEL_DIR / sidecar).write_text(json.dumps(side))
            print("could not replace", fname, "(in use); wrote", sidecar, "instead")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--instant", action="store_true", help="only compute and cache the instant scores")
    a = ap.parse_args()
    if a.instant:
        from .history2_eval import prepare
        d = prepare()
        compute_instant(d, _splits(d))
    else:
        main()

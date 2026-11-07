"""Evaluate instant (A), instant plus history (B) and history only (C) setups.

Label = NASA type (static = 2, fire = 0, types 1 and 3 dropped). Setups B and C
use past detections of the same place, so they are not single reading results.

Run from the project root, one part at a time:
    python -m src.history_eval --part a    5 grouped splits
    python -m src.history_eval --part b    leave one region out
    python -m src.history_eval --part c    temporal split
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from .config import FEATURES, REPORT_DIR, TARGET
from .data import load_raw
from .features import add_features
from .history import HISTORY_FEATURES, add_history
from .label import add_label
from .mlp import MLPClassifier
from .train import best_threshold, build_models, score

SETUPS = {
    "A instant": FEATURES,
    "B instant+history": FEATURES + HISTORY_FEATURES,
    "C history only": HISTORY_FEATURES,
}
MODELS = ["Random Forest", "XGBoost", "MLP"]
SEEDS = [42, 43, 44, 45, 46]
N_TREES = 100
FOLDS = 3
COMPLETE_DAYS = 30
RULE_COL = "distinct_days_prior_30"
SPLIT_DATE = pd.Timestamp("2025-05-01")
REGIONS = {
    "fire_archive_SV-C2_812937.csv": "South Asia",
    "fire_archive_SV-C2_812939.csv": "Southeast Asia",
    "fire_archive_SV-C2_812940.csv": "North and Central America",
}


def prepare():
    df = add_features(add_label(load_raw()))
    df = add_history(df)                      # history sees every detection, before dropping types
    df = df[df["type"].isin([0, 2])].reset_index(drop=True)
    df[TARGET] = (df["type"] == 2).astype(int)
    df["region"] = df["source_file"].map(REGIONS)
    return df


def make_model(name, pos_weight):
    if name == "MLP":
        return MLPClassifier()
    return build_models(pos_weight, N_TREES)[name]


def fit_model(m, X, y, g):
    return m.fit(X, y, groups=g) if isinstance(m, MLPClassifier) else m.fit(X, y)


def fit_with_threshold(name, Xtr, ytr, gtr):
    """Threshold from out-of-fold predictions on the training part only, then a final fit."""
    pos_weight = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    oof = np.zeros(len(ytr))
    for a, b in GroupKFold(n_splits=FOLDS).split(Xtr, ytr, gtr):
        m = fit_model(make_model(name, pos_weight), Xtr.iloc[a], ytr[a], gtr[a])
        oof[b] = m.predict_proba(Xtr.iloc[b])[:, 1]
    thr = best_threshold(ytr, oof)
    return fit_model(make_model(name, pos_weight), Xtr, ytr, gtr), thr


def choose_k(train):
    """Day count k for the rule, picked by static F1 on the training part."""
    y = train[TARGET].values
    ks = range(1, 31)
    f1 = [f1_score(y, (train[RULE_COL] >= k).astype(int), zero_division=0) for k in ks]
    return list(ks)[int(np.argmax(f1))]


def rule_scores(test, k):
    y, s = test[TARGET].values, test[RULE_COL].values
    pred = (s >= k).astype(int)
    return {"precision_static": round(precision_score(y, pred, zero_division=0), 4),
            "recall_static": round(recall_score(y, pred, zero_division=0), 4),
            "f1_static": round(f1_score(y, pred, zero_division=0), 4),
            "pr_auc_static": round(average_precision_score(y, s), 4),
            "threshold": k}


def run_split(train, test, subsets, **tags):
    """Fit every setup and model on train, score on each named subset mask of test."""
    rows = []
    gtr, ytr = train["cell_id"].values, train[TARGET].values
    k = choose_k(train)
    for subset, mask in subsets.items():
        t = test[mask]
        res = rule_scores(t, k)
        res.update(setup="Rule", model=f"{RULE_COL} >= {k}", subset=subset, n_test=len(t),
                   baseline_pr_auc=round(float(t[TARGET].mean()), 4), **tags)
        rows.append(res)
    for setup, feats in SETUPS.items():
        for name in MODELS:
            m, thr = fit_with_threshold(name, train[feats], ytr, gtr)
            p = m.predict_proba(test[feats])[:, 1]
            for subset, mask in subsets.items():
                res = score(test[TARGET].values[mask], p[mask], thr)
                res.update(setup=setup, model=name, subset=subset, n_test=int(mask.sum()),
                           baseline_pr_auc=round(float(test[TARGET].values[mask].mean()), 4), **tags)
                rows.append(res)
            print(tags, setup, name, flush=True)
    return rows


def part_a(df):
    rows = []
    for seed in SEEDS:
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
                      .split(df, df[TARGET], df["cell_id"]))
        train, test = df.iloc[tr], df.iloc[te]
        subsets = {"all rows": np.ones(len(test), bool),
                   f"days_into_data >= {COMPLETE_DAYS}": (test["days_into_data"] >= COMPLETE_DAYS).values}
        rows += run_split(train, test, subsets, seed=seed)
    return rows


def part_b(df):
    rows = []
    for region in REGIONS.values():
        te = (df["region"] == region).values
        subsets = {"all rows": np.ones(int(te.sum()), bool)}
        rows += run_split(df[~te], df[te], subsets, region=region)
    return rows


def part_c(df):
    tr = (df["acq_date"] < SPLIT_DATE).values
    te = ((df["acq_date"] >= SPLIT_DATE) & (df["days_into_data"] >= COMPLETE_DAYS)).values
    train, test = df[tr], df[te]
    subsets = {"history complete": np.ones(len(test), bool)}
    print("train rows", len(train), "test rows", len(test), flush=True)
    return run_split(train, test, subsets, split="before 2025-05-01 / 2025-05-01 to 2025-06-30")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["a", "b", "c"], required=True)
    a = ap.parse_args()
    out = {"a": part_a, "b": part_b, "c": part_c}[a.part](prepare())
    pd.DataFrame(out).to_csv(REPORT_DIR / f"history_eval_{a.part}.csv", index=False)

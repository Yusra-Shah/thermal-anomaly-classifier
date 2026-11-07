"""Evaluate history v2 setups with default and tuned models.

Setups: B instant + history v1 (rerun for the stored predictions only, the reported
B numbers come from history_eval), D instant + v1 + v2, E v1 + v2 only.
Models: Random Forest and XGBoost with 100 trees, and XGBoost tuned by a 20 draw
random search that uses only the training part of each split (3 grouped folds,
selected by static PR-AUC).

Run from the project root, one part at a time:
    python -m src.history2_eval --part a    5 grouped splits
    python -m src.history2_eval --part b    leave one region out
    python -m src.history2_eval --part c    temporal split
"""
import argparse
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from xgboost import XGBClassifier

from .config import FEATURES, PROC_DIR, RANDOM_STATE, REPORT_DIR, TARGET
from .data import load_raw
from .features import add_features
from .history import HISTORY_FEATURES, HISTORY_V2_FEATURES, add_history, add_history_v2
from .history_eval import (COMPLETE_DAYS, FOLDS, REGIONS, SEEDS, SPLIT_DATE, choose_k,
                           make_model, rule_scores, RULE_COL)
from .label import add_label
from .train import best_threshold, score

V1, V2 = HISTORY_FEATURES, HISTORY_V2_FEATURES
SETUPS = {
    "B instant+history": FEATURES + V1,
    "D instant+history v1+v2": FEATURES + V1 + V2,
    "E history v1+v2 only": V1 + V2,
}
MODELS = ["Random Forest", "XGBoost", "XGBoost tuned"]
N_DRAWS = 20
SPACE = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 4, 5, 6, 7, 8],
    "learning_rate": [0.03, 0.05, 0.08, 0.1, 0.15, 0.2],
    "subsample": [0.6, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.5, 0.7, 0.9, 1.0],
    "min_child_weight": [1, 3, 5, 10],
}


def prepare():
    df = add_features(add_label(load_raw()))
    df = add_history_v2(add_history(df))      # history sees every detection, before dropping types
    df = df[df["type"].isin([0, 2])].reset_index(drop=True)
    df[TARGET] = (df["type"] == 2).astype(int)
    df["region"] = df["source_file"].map(REGIONS)
    return df


def sample_configs():
    rng = np.random.RandomState(0)
    return [{k: rng.choice(v).item() for k, v in SPACE.items()} for _ in range(N_DRAWS)]


def xgb(cfg, pos_weight):
    return XGBClassifier(**cfg, scale_pos_weight=pos_weight, eval_metric="logloss",
                         n_jobs=-1, random_state=RANDOM_STATE)


def oof_predict(make, Xtr, ytr, folds):
    oof = np.zeros(len(ytr))
    for a, b in folds:
        oof[b] = make().fit(Xtr.iloc[a], ytr[a]).predict_proba(Xtr.iloc[b])[:, 1]
    return oof


def tune(Xtr, ytr, pos_weight, folds):
    """Best of N_DRAWS configs by pooled out-of-fold static PR-AUC on the training part."""
    best = (-1.0, None, None)
    for cfg in sample_configs():
        oof = oof_predict(lambda: xgb(cfg, pos_weight), Xtr, ytr, folds)
        ap = average_precision_score(ytr, oof)
        if ap > best[0]:
            best = (ap, cfg, oof)
    return best


def run_split(train, test, subsets, store, key, **tags):
    rows = []
    ytr, gtr = train[TARGET].values, train["cell_id"].values
    yte = test[TARGET].values
    folds = list(GroupKFold(n_splits=FOLDS).split(train, ytr, gtr))
    store[(key, "y")] = {"ytr": ytr, "yte": yte, "masks": subsets}

    k = choose_k(train)
    for subset, mask in subsets.items():
        t = test[mask]
        res = rule_scores(t, k)
        res.update(setup="Rule", model=f"{RULE_COL} >= {k}", subset=subset, n_test=len(t),
                   baseline_pr_auc=round(float(t[TARGET].mean()), 4), params="", **tags)
        rows.append(res)

    pos_weight = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    for setup, feats in SETUPS.items():
        Xtr, Xte = train[feats], test[feats]
        for name in MODELS:
            params = ""
            if name == "XGBoost tuned":
                _, cfg, oof = tune(Xtr, ytr, pos_weight, folds)
                model, params = xgb(cfg, pos_weight), json.dumps(cfg)
            else:
                oof = oof_predict(lambda: make_model(name, pos_weight), Xtr, ytr, folds)
                model = make_model(name, pos_weight)
            thr = best_threshold(ytr, oof)
            p = model.fit(Xtr, ytr).predict_proba(Xte)[:, 1]
            store[(key, setup, name)] = {"oof": oof.astype(np.float32), "p": p.astype(np.float32)}
            for subset, mask in subsets.items():
                res = score(yte[mask], p[mask], thr)
                res.update(setup=setup, model=name, subset=subset, n_test=int(mask.sum()),
                           baseline_pr_auc=round(float(yte[mask].mean()), 4), params=params, **tags)
                rows.append(res)
            print(tags, setup, name, params, flush=True)
    return rows


def part_a(df, store):
    rows = []
    for seed in SEEDS:
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
                      .split(df, df[TARGET], df["cell_id"]))
        train, test = df.iloc[tr], df.iloc[te]
        subsets = {"all rows": np.ones(len(test), bool),
                   f"days_into_data >= {COMPLETE_DAYS}": (test["days_into_data"] >= COMPLETE_DAYS).values}
        rows += run_split(train, test, subsets, store, seed, seed=seed)
    return rows


def part_b(df, store):
    rows = []
    for region in REGIONS.values():
        te = (df["region"] == region).values
        rows += run_split(df[~te], df[te], {"all rows": np.ones(int(te.sum()), bool)},
                          store, region, region=region)
    return rows


def part_c(df, store):
    tr = (df["acq_date"] < SPLIT_DATE).values
    te = ((df["acq_date"] >= SPLIT_DATE) & (df["days_into_data"] >= COMPLETE_DAYS)).values
    train, test = df[tr], df[te]
    print("train rows", len(train), "test rows", len(test), flush=True)
    return run_split(train, test, {"history complete": np.ones(len(test), bool)}, store, "temporal",
                     split="before 2025-05-01 / 2025-05-01 to 2025-06-30")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["a", "b", "c"], required=True)
    a = ap.parse_args()
    store = {}
    out = {"a": part_a, "b": part_b, "c": part_c}[a.part](prepare(), store)
    pd.DataFrame(out).to_csv(REPORT_DIR / f"history2_eval_{a.part}.csv", index=False)
    with open(PROC_DIR / f"history2_preds_{a.part}.pkl", "wb") as f:
        pickle.dump(store, f)

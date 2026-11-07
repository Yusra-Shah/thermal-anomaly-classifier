"""Train and evaluate the classifiers.

Run from the project root:
    python -m src.train
    python -m src.train --tag headline

Evaluation rules that keep the numbers honest:
 - Splits are grouped by grid cell, so a location never appears in both train and test.
 - Coordinates and repeat counts are not model inputs.
 - The decision threshold is chosen on out-of-fold predictions from the training set only.
"""
import argparse
import json

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .config import (FEATURES, HOUR_FEATURE, MODEL_DIR, PROC_DIR, RANDOM_STATE, REPORT_DIR,
                     STATIC_MIN_DAYS, STATIC_WINDOW, TARGET)
from .data import load_raw
from .features import add_features
from .label import add_label


def build_models(pos_weight: float, n_trees: int = 300):
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=n_trees, min_samples_leaf=3, class_weight="balanced_subsample",
            n_jobs=-1, random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=n_trees, max_depth=5, learning_rate=0.08, subsample=0.9,
            colsample_bytree=0.9, scale_pos_weight=pos_weight, eval_metric="logloss",
            n_jobs=-1, random_state=RANDOM_STATE,
        ),
    }


def best_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-9, None)
    return float(thr[int(np.nanargmax(f1))]) if len(thr) else 0.5


def score(y, p, thr):
    pred = (p >= thr).astype(int)
    return {
        "threshold": round(float(thr), 4),
        "precision_static": round(precision_score(y, pred, zero_division=0), 4),
        "recall_static": round(recall_score(y, pred, zero_division=0), 4),
        "f1_static": round(f1_score(y, pred, zero_division=0), 4),
        "recall_fire": round(recall_score(1 - y, 1 - pred, zero_division=0), 4),
        "precision_fire": round(precision_score(1 - y, 1 - pred, zero_division=0), 4),
        "f1_fire": round(f1_score(1 - y, 1 - pred, zero_division=0), 4),
        "pr_auc_static": round(average_precision_score(y, p), 4),
        "roc_auc": round(roc_auc_score(y, p), 4),
    }


def main(files=None, min_days=STATIC_MIN_DAYS, window=STATIC_WINDOW, label="type",
         with_hour=False, tag=""):
    features = FEATURES + [HOUR_FEATURE] if with_hour else list(FEATURES)
    sfx = f"_{tag}" if tag else ""
    for d in (PROC_DIR, MODEL_DIR, REPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)

    df = load_raw(files)
    df = add_label(df, min_days=min_days, window=window)
    df = add_features(df)
    if label == "type":
        # NASA's own type flag: 2 = static land source, 0 = vegetation fire.
        # Volcano (1) and offshore (3) detections are dropped.
        df = df[df["type"].isin([0, 2])].copy()
        df[TARGET] = (df["type"] == 2).astype(int)
        df = df.reset_index(drop=True)

    n_cells = df["cell_id"].nunique()
    summary = {
        "rows": int(len(df)),
        "cells": int(n_cells),
        "days": int(df["acq_date"].nunique()),
        "date_min": str(df["acq_date"].min().date()),
        "date_max": str(df["acq_date"].max().date()),
        "static_share": round(float(df[TARGET].mean()), 4),
        "static_min_days": int(min_days),
        "window": window,
        "label": label,
        "features": features,
    }
    print("Data:", summary)
    if df[TARGET].nunique() < 2:
        raise SystemExit("Only one class present. Lower --min-days or use a longer date window.")

    X, y, groups = df[features], df[TARGET].values, df["cell_id"].values

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    tr, te = next(gss.split(X, y, groups))
    Xtr, ytr, gtr = X.iloc[tr], y[tr], groups[tr]
    Xte, yte = X.iloc[te], y[te]
    summary["train_rows"], summary["test_rows"] = int(len(tr)), int(len(te))
    summary["test_static_share"] = round(float(yte.mean()), 4)

    pos_weight = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    models = build_models(pos_weight)
    gkf = GroupKFold(n_splits=5)

    results, fitted = {}, {}
    # Majority-class reference
    dummy = DummyClassifier(strategy="prior").fit(Xtr, ytr)
    results["Majority baseline"] = score(yte, dummy.predict_proba(Xte)[:, 1], 0.5)

    for name, model in models.items():
        oof = np.zeros(len(ytr))
        for a, b in gkf.split(Xtr, ytr, gtr):
            m = build_models(pos_weight)[name]
            m.fit(Xtr.iloc[a], ytr[a])
            oof[b] = m.predict_proba(Xtr.iloc[b])[:, 1]
        thr = best_threshold(ytr, oof)
        cv = score(ytr, oof, thr)

        model.fit(Xtr, ytr)
        p = model.predict_proba(Xte)[:, 1]
        res = score(yte, p, thr)
        res["cv_pr_auc_static"] = cv["pr_auc_static"]
        results[name] = res
        fitted[name] = (model, p, thr)
        print(f"{name}: {res}")

    table = pd.DataFrame(results).T
    table.to_csv(REPORT_DIR / f"model_comparison{sfx}.csv")

    best = max(fitted, key=lambda k: results[k]["pr_auc_static"])
    model, p, thr = fitted[best]
    summary["best_model"] = best
    joblib.dump({"model": model, "features": features, "threshold": thr, "name": best},
                MODEL_DIR / f"best_model{sfx}.joblib")

    # Confusion matrix for the best model
    cm = confusion_matrix(yte, (p >= thr).astype(int))
    fig, ax = plt.subplots(figsize=(4, 3.4))
    ax.imshow(cm, cmap="Greys")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v}", ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["fire", "static"])
    ax.set_yticks([0, 1], ["fire", "static"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{best}, test set")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / f"confusion_matrix{sfx}.png", dpi=160)
    plt.close(fig)

    # Feature importance
    imp = None
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=features)
    elif hasattr(model, "named_steps"):
        imp = pd.Series(np.abs(model.named_steps["logisticregression"].coef_[0]), index=features)
    if imp is not None:
        imp = imp.sort_values()
        fig, ax = plt.subplots(figsize=(5, 3.8))
        ax.barh(imp.index, imp.values, color="#444444")
        ax.set_title(f"Feature importance, {best}")
        fig.tight_layout()
        fig.savefig(REPORT_DIR / f"feature_importance{sfx}.png", dpi=160)
        plt.close(fig)

    # Predictions for the app (all rows, with a flag for rows used in training)
    df["p_static"] = model.predict_proba(df[features])[:, 1]
    df["pred_static"] = (df["p_static"] >= thr).astype(int)
    df["in_test"] = False
    df.loc[df.index[te], "in_test"] = True
    keep = ["latitude", "longitude", "acq_date", "timestamp", "bright_ti4", "bright_ti5",
            "frp", "confidence_ord", "is_night", "cell_id", "cell_days", TARGET,
            "p_static", "pred_static", "in_test"]
    df[keep].to_parquet(PROC_DIR / f"scored{sfx}.parquet", index=False)

    summary["results"] = results
    with open(REPORT_DIR / f"metrics{sfx}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nBest by test PR-AUC: {best}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="*", default=None)
    ap.add_argument("--min-days", type=int, default=STATIC_MIN_DAYS)
    ap.add_argument("--window", choices=["month", "total"], default=STATIC_WINDOW)
    ap.add_argument("--label", choices=["rule", "type"], default="type")
    ap.add_argument("--with-hour", action="store_true", help="add hour_utc (ablation)")
    ap.add_argument("--tag", default="", help="suffix for output files")
    a = ap.parse_args()
    main(a.files, a.min_days, a.window, a.label, a.with_hour, a.tag)

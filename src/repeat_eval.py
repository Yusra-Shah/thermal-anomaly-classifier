"""Repeat the grouped train/test split with different seeds (headline setup).

Run from the project root:
    python -m src.repeat_eval
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from .config import FEATURES, REPORT_DIR, TARGET
from .data import load_raw
from .features import add_features
from .label import add_label
from .train import best_threshold, build_models, score

MODELS = ["Random Forest", "XGBoost"]
SEEDS = [42, 43, 44, 45, 46]
COLS = ["pr_auc_static", "precision_static", "recall_static", "f1_static"]


def main(tag="headline_seeds"):
    df = add_features(add_label(load_raw()))
    df = df[df["type"].isin([0, 2])].reset_index(drop=True)
    df[TARGET] = (df["type"] == 2).astype(int)
    X, y, groups = df[FEATURES], df[TARGET].values, df["cell_id"].values

    rows = []
    for seed in SEEDS:
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(X, y, groups))
        Xtr, ytr, gtr = X.iloc[tr], y[tr], groups[tr]
        pos_weight = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
        for name in MODELS:
            oof = np.zeros(len(ytr))
            for a, b in GroupKFold(n_splits=5).split(Xtr, ytr, gtr):
                m = build_models(pos_weight)[name]
                m.fit(Xtr.iloc[a], ytr[a])
                oof[b] = m.predict_proba(Xtr.iloc[b])[:, 1]
            thr = best_threshold(ytr, oof)
            m = build_models(pos_weight)[name].fit(Xtr, ytr)
            res = score(y[te], m.predict_proba(X.iloc[te])[:, 1], thr)
            res.update(seed=seed, model=name)
            rows.append(res)
            print(seed, name, {c: res[c] for c in COLS}, flush=True)

    runs = pd.DataFrame(rows)
    runs[["seed", "model"] + COLS].to_csv(REPORT_DIR / f"repeat_eval_{tag}_runs.csv", index=False)
    g = runs.groupby("model")[COLS]
    summ = pd.concat({"mean": g.mean(), "std": g.std()}, axis=1).round(4)
    summ.to_csv(REPORT_DIR / f"repeat_eval_{tag}.csv")
    print(summ.to_string())


if __name__ == "__main__":
    main()

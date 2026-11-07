"""Leave one region out evaluation with the NASA type label.

Each raw file is one region. Train on the other files, test on the held out one.

Run from the project root:
    python -m src.region_eval
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .config import FEATURES, HOUR_FEATURE, REPORT_DIR, TARGET
from .data import load_raw
from .features import add_features
from .label import add_label
from .train import best_threshold, build_models, score

REGIONS = {
    "fire_archive_SV-C2_812937.csv": "South Asia",
    "fire_archive_SV-C2_812939.csv": "Southeast Asia",
    "fire_archive_SV-C2_812940.csv": "North and Central America",
}
MODELS = ["Random Forest", "XGBoost"]


def main(n_trees=100, folds=3, tag="type_region"):
    df = add_features(add_label(load_raw()))
    df = df[df["type"].isin([0, 2])].reset_index(drop=True)
    df[TARGET] = (df["type"] == 2).astype(int)
    df["region"] = df["source_file"].map(REGIONS)

    rows = []
    for region in REGIONS.values():
        te = df["region"] == region
        for with_hour in (False, True):
            feats = FEATURES + [HOUR_FEATURE] if with_hour else FEATURES
            Xtr, ytr, gtr = df.loc[~te, feats], df.loc[~te, TARGET].values, df.loc[~te, "cell_id"].values
            Xte, yte = df.loc[te, feats], df.loc[te, TARGET].values
            pos_weight = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
            for name in MODELS:
                oof = np.zeros(len(ytr))
                for a, b in GroupKFold(n_splits=folds).split(Xtr, ytr, gtr):
                    m = build_models(pos_weight, n_trees)[name]
                    m.fit(Xtr.iloc[a], ytr[a])
                    oof[b] = m.predict_proba(Xtr.iloc[b])[:, 1]
                thr = best_threshold(ytr, oof)
                m = build_models(pos_weight, n_trees)[name].fit(Xtr, ytr)
                res = score(yte, m.predict_proba(Xte)[:, 1], thr)
                res.update(region=region, model=name, hour_utc=with_hour,
                           test_rows=int(te.sum()), baseline_pr_auc=round(float(yte.mean()), 4))
                rows.append(res)
                print(region, name, "hour" if with_hour else "nohour",
                      res["precision_static"], res["recall_static"], res["f1_static"],
                      res["pr_auc_static"], res["baseline_pr_auc"], flush=True)

    out = pd.DataFrame(rows)[["region", "model", "hour_utc", "test_rows", "baseline_pr_auc",
                              "precision_static", "recall_static", "f1_static",
                              "pr_auc_static", "threshold"]]
    out.to_csv(REPORT_DIR / f"region_eval_{tag}.csv", index=False)


if __name__ == "__main__":
    main()

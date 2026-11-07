"""Train and save the final history model (setup D, default XGBoost, seed 42 grouped split).

Run from the project root:  python -m src.final_history

Setup D uses past detections of the same place, so it is not a single reading model.
Thresholds come from out-of-fold predictions on the training part only.
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from .config import MODEL_DIR, PROC_DIR, REPORT_DIR, TARGET
from .history2_eval import SETUPS, prepare
from .history2_ops import thresholds
from .history_eval import FOLDS, make_model

SEED = 42
SETUP = "D instant+history v1+v2"
DEFAULT_POINT = "precision >= 0.90"


def main():
    for d in (PROC_DIR, MODEL_DIR, REPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)
    df = prepare()
    feats = SETUPS[SETUP]
    X, y, groups = df[feats], df[TARGET].values, df["cell_id"].values

    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(X, y, groups))
    Xtr, ytr, gtr = X.iloc[tr], y[tr], groups[tr]
    pos_weight = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))

    oof = np.zeros(len(ytr))
    for a, b in GroupKFold(n_splits=FOLDS).split(Xtr, ytr, gtr):
        oof[b] = make_model("XGBoost", pos_weight).fit(Xtr.iloc[a], ytr[a]).predict_proba(Xtr.iloc[b])[:, 1]
    thr = thresholds(ytr, oof)
    model = make_model("XGBoost", pos_weight).fit(Xtr, ytr)

    p_test = model.predict_proba(X.iloc[te])[:, 1]
    yte = y[te]
    points = {}
    for name, t in thr.items():
        if t is None:
            points[name] = {"threshold": None, "precision": None, "recall": None}
            continue
        pred = (p_test >= t).astype(int)
        points[name] = {"threshold": round(t, 4),
                        "precision": round(precision_score(yte, pred, zero_division=0), 4),
                        "recall": round(recall_score(yte, pred, zero_division=0), 4)}

    joblib.dump({"model": model, "features": feats, "thresholds": thr,
                 "default_operating_point": DEFAULT_POINT, "name": f"XGBoost, setup {SETUP}"},
                MODEL_DIR / "best_model_history.joblib")

    df["p_static_history"] = model.predict_proba(X)[:, 1]
    df["in_test"] = False
    df.loc[df.index[te], "in_test"] = True
    keep = ["latitude", "longitude", "acq_date", "timestamp", "bright_ti4", "bright_ti5", "frp",
            "confidence_ord", "is_night", "cell_id", "cell_days", TARGET, "p_static_history", "in_test"]
    df[keep].to_parquet(PROC_DIR / "scored_history.parquet", index=False)

    metrics = {"setup": SETUP, "model": "XGBoost, default settings (100 trees)", "split_seed": SEED,
               "train_rows": int(len(tr)), "test_rows": int(len(te)),
               "test_static_share": round(float(yte.mean()), 4), "operating_points": points}
    (REPORT_DIR / "history_final_metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"Final history model: {metrics['model']}, setup {SETUP}, seed {SEED}")
    print(f"train rows {len(tr):,}, held out rows {len(te):,}, held out static share {yte.mean():.4f}")
    ops = pd.read_csv(REPORT_DIR / "history2_operating_points.csv")
    ref = ops[ops["split"] == f"seed {SEED}"].set_index("operating point")
    for name, v in points.items():
        line = f"{name}: threshold {v['threshold']}, held out precision {v['precision']}, recall {v['recall']}"
        if name in ref.index and v["precision"] is not None:
            r = ref.loc[name]
            same = abs(r["precision"] - v["precision"]) < 5e-4 and abs(r["recall"] - v["recall"]) < 5e-4
            line += (" (matches the tuned model in summary_history2.md)" if same else
                     f" (differs from the tuned model in summary_history2.md: precision {r['precision']}, "
                     f"recall {r['recall']}; the default model's measured values are kept)")
        print(line)


if __name__ == "__main__":
    main()

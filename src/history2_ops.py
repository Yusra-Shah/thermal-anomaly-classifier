"""Operating points for the best setup and model of history2_eval.

The best (setup, model) is the one with the highest mean static PR-AUC over the 5
grouped splits. Thresholds come from out-of-fold predictions on the training part only.

Run from the project root:  python -m src.history2_ops
"""
import pickle

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, precision_score, recall_score

from .config import PROC_DIR, REPORT_DIR
from .history_eval import SEEDS
from .train import best_threshold

TARGETS = (0.90, 0.95)


def pick_best(a):
    d = a[(a["subset"] == "all rows") & (a["setup"] != "Rule")]
    m = d.groupby(["setup", "model"])["pr_auc_static"].mean().sort_values(ascending=False)
    return m.index[0], float(m.iloc[0])


def thresholds(ytr, oof):
    out = {"F1 optimal": best_threshold(ytr, oof)}
    prec, _, thr = precision_recall_curve(ytr, oof)
    for t in TARGETS:
        ok = np.where(prec[:-1] >= t)[0]
        out[f"precision >= {t:.2f}"] = float(thr[ok[np.argmin(thr[ok])]]) if len(ok) else None
    return out


def main():
    a = pd.read_csv(REPORT_DIR / "history2_eval_a.csv")
    (setup, model), best_ap = pick_best(a)
    stores = {}
    for part in ("a", "c"):
        with open(PROC_DIR / f"history2_preds_{part}.pkl", "rb") as f:
            stores[part] = pickle.load(f)
    splits = [(f"seed {s}", stores["a"], s) for s in SEEDS] + [("temporal", stores["c"], "temporal")]

    rows, fig_data = [], []
    for label, store, key in splits:
        y = store[(key, "y")]
        ytr, yte = y["ytr"], y["yte"]
        mask = next(iter(y["masks"].values()))        # all test rows (history complete for temporal)
        pr = store[(key, setup, model)]
        yt, p = yte[mask], pr["p"][mask]
        for name, thr in thresholds(ytr, pr["oof"]).items():
            if thr is None:
                rows.append({"split": label, "operating point": name, "threshold": np.nan,
                             "precision": np.nan, "recall": np.nan})
                continue
            pred = (p >= thr).astype(int)
            rows.append({"split": label, "operating point": name, "threshold": round(thr, 4),
                         "precision": round(precision_score(yt, pred, zero_division=0), 4),
                         "recall": round(recall_score(yt, pred, zero_division=0), 4)})
        fig_data.append((label, yt, p))
    out = pd.DataFrame(rows)
    out.insert(1, "setup", setup)
    out.insert(2, "model", model)
    out.to_csv(REPORT_DIR / "history2_operating_points.csv", index=False)

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for label, yt, p in fig_data:
        prec, rec, _ = precision_recall_curve(yt, p)
        temporal = label == "temporal"
        ax.plot(rec, prec, color="#c0392b" if temporal else "#9a9a9a",
                lw=1.8 if temporal else 0.9, label=label if temporal else None)
    ax.plot([], [], color="#9a9a9a", lw=0.9, label="5 grouped splits")
    ax.set_xlabel("Recall (static)")
    ax.set_ylabel("Precision (static)")
    ax.set_title(f"{setup}, {model}", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left", frameon=False)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "pr_curve_history2.png", dpi=160)
    plt.close(fig)
    print(setup, model, round(best_ap, 4))
    print(out.to_string())


if __name__ == "__main__":
    main()

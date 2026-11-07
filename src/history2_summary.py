"""Build the tables of reports/summary_history2.md from the generated CSVs.

Run from the project root:  python -m src.history2_summary
"""
import json

import pandas as pd

from .config import REPORT_DIR
from .history2_ops import pick_best
from tests import test_history

M = ["pr_auc_static", "precision_static", "recall_static", "f1_static"]
HEAD = ["PR-AUC", "Precision", "Recall", "F1"]
ROWS = [("Rule", None), ("B instant+history", "Random Forest"), ("B instant+history", "XGBoost"),
        ("B instant+history", "XGBoost tuned"),
        ("D instant+history v1+v2", "Random Forest"), ("D instant+history v1+v2", "XGBoost"),
        ("D instant+history v1+v2", "XGBoost tuned"),
        ("E history v1+v2 only", "Random Forest"), ("E history v1+v2 only", "XGBoost"),
        ("E history v1+v2 only", "XGBoost tuned")]
PARAMS = ["n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree", "min_child_weight"]


def md(rows, header):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
                     + ["| " + " | ".join(r) + " |" for r in rows])


def combined(old, new):
    """B default rows from the earlier CSV, everything else from the new CSV."""
    o = old[(old["setup"] == "B instant+history") & old["model"].isin(["Random Forest", "XGBoost"])]
    n = new[~((new["setup"] == "B instant+history") & new["model"].isin(["Random Forest", "XGBoost"]))]
    return pd.concat([o, n], ignore_index=True)


def pick(d, setup, model):
    return d[d["setup"] == setup] if model is None else d[(d["setup"] == setup) & (d["model"] == model)]


def table_a(d):
    rows = []
    for setup, model in ROWS:
        g = pick(d, setup, model)
        label = ("distinct_days_prior_30 >= k, k per seed " + ",".join(str(int(k)) for k in g["threshold"])
                 if model is None else model)
        rows.append([setup, label] + [f"{g[m].mean():.3f} ({g[m].std():.3f})" for m in M])
    return md(rows, ["Setup", "Model"] + HEAD)


def table_single(d):
    rows = []
    for setup, model in ROWS:
        g = pick(d, setup, model).iloc[0]
        label = g["model"] if model is None else model
        rows.append([setup, label] + [f"{g[m]:.3f}" for m in M])
    return md(rows, ["Setup", "Model"] + HEAD)


def paired(d, a_key, b_key):
    a = pick(d, *a_key).sort_values("seed").reset_index(drop=True)
    b = pick(d, *b_key).sort_values("seed").reset_index(drop=True)
    out = []
    for m in ("pr_auc_static", "f1_static"):
        diff = a[m] - b[m]
        flag = "inside spread" if abs(diff.mean()) < diff.std() else "outside spread"
        out += [f"{diff.mean():+.3f} ({diff.std():.3f})", flag]
    return out


def main():
    old = {p: pd.read_csv(REPORT_DIR / f"history_eval_{p}.csv") for p in "abc"}
    new = {p: pd.read_csv(REPORT_DIR / f"history2_eval_{p}.csv") for p in "abc"}
    ops = pd.read_csv(REPORT_DIR / "history2_operating_points.csv")

    test_history.test_no_leakage()
    test_history.test_known_values()

    # Reproducibility of the B default rerun against the earlier CSVs
    keys = ["setup", "model", "subset"] + [k for k in ("seed", "region") if k in new["a"].columns]
    worst = 0.0
    for p in "abc":
        k = [c for c in ("seed", "region") if c in new[p].columns]
        cols = ["setup", "model", "subset"] + k
        sel = lambda df: df[(df["setup"] == "B instant+history")
                            & df["model"].isin(["Random Forest", "XGBoost"])].set_index(cols)[M]
        worst = max(worst, float((sel(new[p]) - sel(old[p]).loc[sel(new[p]).index]).abs().max().max()))

    L = ["# Summary: history tier, version 2", "",
         "Label: NASA type (static = 2, fire = 0, types 1 and 3 dropped). The type label is "
         "produced by NASA's own algorithm. No coordinates, no hour_utc.", "",
         "**Setups B, D and E use past detections of the same place and are not single reading "
         "results.** Every history feature uses only detections on days strictly before the "
         "detection, within the same raw file.", "",
         "- B: instant features + history v1 (numbers reused from the earlier run)",
         "- D: instant + history v1 + v2",
         "- E: history v1 + v2 only", "",
         "Models: Random Forest and XGBoost, 100 trees, default settings; XGBoost tuned by a "
         "random search of 20 configurations with 3 grouped folds on the training part of each "
         "split only, selected by pooled out-of-fold static PR-AUC. Thresholds come from "
         "out-of-fold predictions on the training part. The rule is static if "
         "distinct_days_prior_30 >= k, with k picked on the training part.", "",
         "History v2 adds: distinct days over 60 and 90 days, the 3x3 neighbourhood over 60 days, "
         "days since last seen, persistence_30, the mean and std of bright_ti4 and frp over the "
         "prior 30 days with a has_history flag, and the deviation of the current reading from "
         "that mean (set to 0 when there is no prior detection).", "",
         "## Leakage test", "",
         "tests/test_history.py: for 40 random detections, every other row on or after the "
         "detection's date was moved, re-dated, given new readings, dropped and duplicated. All 19 "
         "history features (v1 and v2) of the detection stayed the same in every case (passed, "
         "tolerance 1e-6 for float features). A hand computed known value test, including the "
         "means and stds, also passed.", "",
         f"Reproducibility: rerunning B with default models reproduces the earlier CSVs; the "
         f"largest absolute difference over all metrics is {worst:.6f}.", ""]

    ca = combined(old["a"], new["a"])
    L += ["## a) Five grouped splits (seeds 42 to 46), mean (std) of static metrics", ""]
    for subset in ca["subset"].unique():
        d = ca[ca["subset"] == subset]
        n = int(d[d["setup"] == "Rule"]["n_test"].mean())
        base = d[d["setup"] == "Rule"]["baseline_pr_auc"].mean()
        L += [f"Test rows: {subset} (about {n:,} per split). Majority baseline PR-AUC about {base:.3f}.",
              "", table_a(d), ""]

    d = ca[ca["subset"] == "all rows"]
    (bs, bm), _ = pick_best(new["a"])
    pairs = [("D XGBoost vs B XGBoost", ("D instant+history v1+v2", "XGBoost"), ("B instant+history", "XGBoost")),
             ("E XGBoost vs B XGBoost", ("E history v1+v2 only", "XGBoost"), ("B instant+history", "XGBoost")),
             ("D XGBoost tuned vs D XGBoost", ("D instant+history v1+v2", "XGBoost tuned"), ("D instant+history v1+v2", "XGBoost")),
             ("E XGBoost tuned vs E XGBoost", ("E history v1+v2 only", "XGBoost tuned"), ("E history v1+v2 only", "XGBoost")),
             ("B XGBoost tuned vs B XGBoost", ("B instant+history", "XGBoost tuned"), ("B instant+history", "XGBoost")),
             ("D XGBoost tuned vs B XGBoost", ("D instant+history v1+v2", "XGBoost tuned"), ("B instant+history", "XGBoost")),
             ("D Random Forest vs B Random Forest", ("D instant+history v1+v2", "Random Forest"), ("B instant+history", "Random Forest")),
             (f"best ({bs}, {bm}) vs rule", (bs, bm), ("Rule", None)),
             (f"best ({bs}, {bm}) vs B XGBoost", (bs, bm), ("B instant+history", "XGBoost"))]
    L += ["### Paired differences over the 5 splits (all rows)", "",
          "Mean difference (std of the per split differences). \"Inside spread\" means the mean "
          "difference is smaller than one std of the differences.", "",
          md([[n] + paired(d, a, b) for n, a, b in pairs],
             ["Comparison", "PR-AUC diff", "", "F1 diff", ""]), ""]

    L += [f"Best by mean PR-AUC over the 5 splits: {bs}, {bm}.", ""]

    L += ["## b) Leave one region out", ""]
    cb = combined(old["b"], new["b"])
    for region in cb["region"].unique():
        d = cb[cb["region"] == region]
        r = d[d["setup"] == "Rule"].iloc[0]
        L += [f"Held out: {region} ({int(r['n_test']):,} rows). Majority baseline PR-AUC "
              f"{r['baseline_pr_auc']:.3f}. Rule k = {int(r['threshold'])}.", "", table_single(d), ""]

    cc = combined(old["c"], new["c"])
    r = cc[cc["setup"] == "Rule"].iloc[0]
    L += ["## c) Temporal split", "",
          f"Train: detections before 2025-05-01. Test: 2025-05-01 to 2025-06-30 ({int(r['n_test']):,} "
          f"rows, history complete). Majority baseline PR-AUC {r['baseline_pr_auc']:.3f}. "
          f"Rule k = {int(r['threshold'])}. Cells can appear in both train and test.", "",
          table_single(cc), ""]

    L += ["## Tuned XGBoost parameters (chosen on training data only)", ""]
    for p, key, title in (("a", "seed", "5 grouped splits"), ("b", "region", "Leave one region out"),
                          ("c", "split", "Temporal split")):
        t = new[p][(new[p]["model"] == "XGBoost tuned") & (new[p]["subset"] == new[p]["subset"].iloc[0])]
        rows = []
        for _, r in t.iterrows():
            cfg = json.loads(r["params"])
            rows.append([str(r[key])[:30], r["setup"]] + [str(cfg[k]) for k in PARAMS])
        L += [f"{title}:", "", md(rows, ["Split", "Setup"] + PARAMS), ""]

    L += ["## Operating points", "",
          f"Best setup and model: {ops['setup'].iloc[0]}, {ops['model'].iloc[0]}. Thresholds come "
          "from out-of-fold predictions on the training part only. Precision and recall are "
          "measured on the test part. NaN means the precision target was never reached on the "
          "training predictions.", "",
          "![Precision recall curves](pr_curve_history2.png)", ""]
    five = ops[ops["split"].str.startswith("seed")]
    rows = []
    for name, g in five.groupby("operating point", sort=False):
        rows.append([name, f"{g['threshold'].mean():.3f} ({g['threshold'].std():.3f})",
                     f"{g['precision'].mean():.3f} ({g['precision'].std():.3f})",
                     f"{g['recall'].mean():.3f} ({g['recall'].std():.3f})"])
    L += ["5 grouped splits, mean (std):", "",
          md(rows, ["Operating point", "Threshold", "Test precision", "Test recall"]), ""]
    t = ops[ops["split"] == "temporal"]
    L += ["Temporal split:", "",
          md([[r["operating point"], f"{r['threshold']:.3f}", f"{r['precision']:.3f}", f"{r['recall']:.3f}"]
              for _, r in t.iterrows()], ["Operating point", "Threshold", "Test precision", "Test recall"]), ""]
    L += ["Per split values are in history2_operating_points.csv.", ""]
    (REPORT_DIR / "summary_history2.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()

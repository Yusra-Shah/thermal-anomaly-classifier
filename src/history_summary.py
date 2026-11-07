"""Build the tables of reports/summary_history.md from the history_eval CSVs.

Run from the project root:  python -m src.history_summary
"""
import pandas as pd

from .config import REPORT_DIR
from tests import test_history

METRICS = ["pr_auc_static", "precision_static", "recall_static", "f1_static"]
HEAD = ["PR-AUC", "Precision", "Recall", "F1"]
ORDER = ["Rule", "A instant", "B instant+history", "C history only"]
MODEL_ORDER = {"Random Forest": 0, "XGBoost": 1, "MLP": 2}


def md(rows, header):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def sort_key(df):
    return df.assign(_s=df["setup"].map(ORDER.index), _m=df["model"].map(lambda m: MODEL_ORDER.get(m, -1)))\
             .sort_values(["_s", "_m"]).drop(columns=["_s", "_m"])


def table_a(a, subset):
    d = a[a["subset"] == subset]
    rows = []
    for (setup, model), g in sort_key(d.assign(model=d["model"].where(d["setup"] != "Rule", "rule"))
                                      ).groupby(["setup", "model"], sort=False):
        label = model if setup != "Rule" else "distinct_days_prior_30 >= k, k per seed " + \
            ",".join(str(int(k)) for k in g["threshold"])
        rows.append([setup, label] + [f"{g[m].mean():.3f} ({g[m].std():.3f})" for m in METRICS])
    base = d.groupby("seed")["baseline_pr_auc"].first().mean()
    return md(rows, ["Setup", "Model"] + HEAD), base, int(d.groupby("seed")["n_test"].first().mean())


def table_single(d):
    rows = []
    for _, r in sort_key(d).iterrows():
        label = r["model"]
        rows.append([r["setup"], label] + [f"{r[m]:.3f}" for m in METRICS])
    return md(rows, ["Setup", "Model"] + HEAD)


def main():
    a = pd.read_csv(REPORT_DIR / "history_eval_a.csv")
    b = pd.read_csv(REPORT_DIR / "history_eval_b.csv")
    c = pd.read_csv(REPORT_DIR / "history_eval_c.csv")

    test_history.test_no_leakage()
    test_history.test_known_values()

    L = ["# Summary: history tier", "",
         "Label: NASA type (static = 2, fire = 0, types 1 and 3 dropped). The type label is produced "
         "by NASA's own algorithm. No coordinates, no hour_utc.", "",
         "Setup A uses a single reading. **Setups B and C use past detections of the same place "
         "and are not single reading results.** History uses only detections on days strictly "
         "before the detection, within the same raw file.", "",
         "Models: Random Forest and XGBoost with 100 trees, MLP (64, 32, ReLU, dropout 0.2). "
         "Thresholds come from 3 grouped folds on the training part. The rule is "
         "static if distinct_days_prior_30 >= k, with k picked by static F1 on the training part.", "",
         "## Leakage test", "",
         "tests/test_history.py: for 40 random detections, every other row on or after the "
         "detection's date was moved, re-dated, dropped and duplicated. The detection's history "
         "features did not change in any case (passed). A hand computed known value test also passed.", ""]

    L += ["## a) Five grouped splits (seeds 42 to 46), mean (std) of static metrics", ""]
    for subset in a["subset"].unique():
        t, base, n = table_a(a, subset)
        L += [f"Test rows: {subset} (about {n:,} per split). Majority baseline PR-AUC about {base:.3f}.", "", t, ""]

    L += ["## b) Leave one region out", ""]
    for region in b["region"].unique():
        d = b[b["region"] == region]
        L += [f"Held out: {region} ({int(d['n_test'].iloc[0]):,} rows). Majority baseline "
              f"PR-AUC {d['baseline_pr_auc'].iloc[0]:.3f}. Rule k = {int(d[d['setup'] == 'Rule']['threshold'].iloc[0])}.",
              "", table_single(d), ""]

    L += ["## c) Temporal split", "",
          f"Train: detections before 2025-05-01. Test: 2025-05-01 to 2025-06-30 "
          f"({int(c['n_test'].iloc[0]):,} rows, history complete). Majority baseline PR-AUC "
          f"{c['baseline_pr_auc'].iloc[0]:.3f}. Rule k = {int(c[c['setup'] == 'Rule']['threshold'].iloc[0])}. "
          "Cells can appear in both train and test here, as they would in use.", "",
          table_single(c), ""]
    (REPORT_DIR / "summary_history.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()

"""All regions: backup, inventory, evaluation of both tiers, leave one region out, summary.

Run from the project root, one stage per process:
    python -m src.world backup
    python -m src.world inventory
    python -m src.world prepare
    python -m src.world history      # history tier, five grouped seeds, the temporal split, every region held out
    python -m src.world instant      # instant tier, same splits
    python -m src.world regions      # the leave one region out table with the one line rule
    python -m src.world summary      # reports/world_summary.md
"""
import shutil
import sys

import numpy as np
import pandas as pd

from .config import MODEL_DIR, PROC_DIR, RAW_DIR, REPORT_DIR, ROOT, TARGET
from .mem import report
from .regions import COUNTRY, FILE_REGION


def backup(name="before_world"):
    """Copy models/ and reports/ outside the repo, once."""
    for src, dst in ((MODEL_DIR, ROOT.parent / "model_backups" / name),
                     (REPORT_DIR, ROOT.parent / "report_backups" / name)):
        if dst.exists():
            print(f"backup exists: {dst.name} in {dst.parent.name}")
            continue
        shutil.copytree(src, dst)
        print(f"copied {src.name}/ to {dst.parent.name}/{name}/")


def inventory():
    """Every file in data/raw with rows, region and static share. Written to reports/world_inventory.csv."""
    rows = []
    for p in sorted(RAW_DIR.glob("*.csv")):
        head = [c.strip().lower() for c in pd.read_csv(p, nrows=0).columns]
        row = {"file": p.name, "region": FILE_REGION.get(p.name, "unknown"),
               "country": COUNTRY.get(FILE_REGION.get(p.name, ""), "")}
        use = [c for c in ("acq_date", "type") if c in head]
        d = pd.read_csv(p, usecols=[c for c in use] if use else None)
        d.columns = [c.strip().lower() for c in d.columns]
        row["rows"] = len(d)
        row["has_type"] = "type" in d.columns
        if "type" in d.columns:
            t = d["type"]
            row.update(type0=int((t == 0).sum()), type2=int((t == 2).sum()), other_types=int((~t.isin([0, 2])).sum()))
            kept = int(t.isin([0, 2]).sum())
            row["rows_kept"] = kept
            row["static_share"] = round(float((t == 2).sum() / kept), 4) if kept else float("nan")
        else:
            row["note"] = "skipped: no type column"
        row["date_min"], row["date_max"] = str(d["acq_date"].min()), str(d["acq_date"].max())
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(REPORT_DIR / "world_inventory.csv", index=False)
    print(out.to_string(index=False))
    return out


SETUP = "D instant+history v1+v2"
SEEDS = [42, 43, 44, 45, 46]


def split_sets(df):
    """Name -> (train rows, test rows, key, mask of test rows scored) for the five grouped seeds, the temporal
    split and every region held out. Seed 42, temporal and the regions use the names of src/uncertain.py."""
    from sklearn.model_selection import GroupShuffleSplit

    from .history_eval import COMPLETE_DAYS, SPLIT_DATE
    from .regions import FILE_REGION
    out = {}
    for seed in SEEDS:
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
                      .split(df, df[TARGET], df["cell_id"]))
        out["grouped seed 42" if seed == 42 else f"grouped seed {seed}"] = (tr, te, seed, np.ones(len(te), bool))
    tr_m = (df["acq_date"] < SPLIT_DATE).to_numpy()
    te_m = ((df["acq_date"] >= SPLIT_DATE) & (df["days_into_data"] >= COMPLETE_DAYS)).to_numpy()
    out["temporal"] = (np.flatnonzero(tr_m), np.flatnonzero(te_m), "temporal", np.ones(int(te_m.sum()), bool))
    for region in FILE_REGION.values():
        m = (df["region"] == region).to_numpy()
        out[f"leave out {region}"] = (np.flatnonzero(~m), np.flatnonzero(m), region, np.ones(int(m.sum()), bool))
    return out


def tier_rows(name, tier, y, p, thr_f1, thr_p90):
    """Metrics of one tier on one split at the F1 optimal threshold and at precision >= 0.90 (history default)."""
    from .daynight import metrics
    a = metrics(y, p, thr_f1)
    row = {"split": name, "tier": tier, **{k: a[k] for k in ("n", "static_share", "majority_pr_auc", "pr_auc")},
           "f1opt_threshold": thr_f1, "f1opt_precision": a["precision"], "f1opt_recall": a["recall"],
           "f1opt_f1": a["f1"]}
    if thr_p90 is not None:
        b = metrics(y, p, thr_p90)
        row.update(p90_threshold=thr_p90, p90_precision=b["precision"], p90_recall=b["recall"], p90_f1=b["f1"])
    return row


def run_history(df):
    """History tier (setup D, XGBoost, 100 trees, 3 folds) on every split. Each finished split is saved in
    data/processed/world_history_cache.pkl, so a stopped run resumes. Writes the stored scores in the format of
    src/uncertain.py (history2_preds_a, b and c) and reports/world_history.csv."""
    import gc
    import os
    import pickle

    from sklearn.model_selection import GroupKFold

    from .history2_eval import SETUPS, oof_predict
    from .history2_ops import thresholds
    from .history_eval import FOLDS, RULE_COL, choose_k, make_model, rule_scores
    feats = SETUPS[SETUP]
    splits = split_sets(df)
    cols = list(dict.fromkeys(feats + [TARGET, "cell_id", RULE_COL]))
    d = df[cols].copy()                                  # only what the history tier reads
    del df
    gc.collect()
    cache_path = PROC_DIR / "world_history_cache.pkl"
    done = pickle.load(open(cache_path, "rb")) if cache_path.exists() else {}
    for name, (tr, te, key, mask) in splits.items():
        if name in done:
            continue
        train, test = d.iloc[tr], d.iloc[te]
        ytr, yte, gtr = train[TARGET].to_numpy(), test[TARGET].to_numpy(), train["cell_id"].to_numpy()
        pw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
        folds = list(GroupKFold(n_splits=FOLDS).split(train, ytr, gtr))
        oof = oof_predict(lambda: make_model("XGBoost", pw), train[feats], ytr, folds)
        thr = thresholds(ytr, oof)
        p = make_model("XGBoost", pw).fit(train[feats], ytr).predict_proba(test[feats])[:, 1]
        rows = [tier_rows(name, "history", yte, p, thr["F1 optimal"], thr["precision >= 0.90"])]
        k = choose_k(train)
        r = rule_scores(test, k)
        rows.append({"split": name, "tier": f"rule {RULE_COL} >= {k}", "n": len(yte), "static_share": float(yte.mean()),
                     "majority_pr_auc": float(yte.mean()), "pr_auc": r["pr_auc_static"],
                     "f1opt_threshold": float(k), "f1opt_precision": r["precision_static"],
                     "f1opt_recall": r["recall_static"], "f1opt_f1": r["f1_static"]})
        done[name] = {"key": key, "ytr": ytr, "yte": yte, "mask": mask, "oof": oof.astype(np.float32),
                      "p": p.astype(np.float32), "rows": rows}
        tmp = cache_path.with_suffix(".tmp")
        pickle.dump(done, open(tmp, "wb"))
        os.replace(tmp, cache_path)
        print(name, "history pr_auc", round(rows[0]["pr_auc"], 4), "f1", round(rows[0]["f1opt_f1"], 4),
              "rule", round(rows[1]["pr_auc"], 4), flush=True)
        report(f"history {name}")
        del train, test, folds
        gc.collect()
    stores = {part: {} for part in "abc"}
    rows = []
    for name, e in done.items():
        key = e["key"]
        part = "c" if key == "temporal" else "a" if isinstance(key, int) else "b"
        stores[part][(key, "y")] = {"ytr": e["ytr"], "yte": e["yte"], "masks": {"all rows": e["mask"]}}
        stores[part][(key, SETUP, "XGBoost")] = {"oof": e["oof"], "p": e["p"]}
        rows += e["rows"]
    for part in "abc":
        with open(PROC_DIR / f"history2_preds_{part}.pkl", "wb") as f:
            pickle.dump(stores[part], f)
    out = pd.DataFrame(rows)
    out.round(4).to_csv(REPORT_DIR / "world_history.csv", index=False)
    return out


def run_instant(df):
    """Instant tier (XGBoost, 100 trees) on every split. Seed 42 and the temporal split use 5 folds and the
    leave one region out splits 3, as in src/daynight.py and src/uncertain.py; the other seeds use 3. Scores go
    to the cache that src/uncertain.py reads. Writes reports/world_instant.csv."""
    from . import daynight
    from .config import FEATURES
    splits = split_sets(df)
    store = daynight._load()
    rows = []
    for name, (tr, te, key, mask) in splits.items():
        k = ("xgb", name, "full model")
        train, test = df.iloc[tr], df.iloc[te]
        ytr, yte = train[TARGET].to_numpy(), test[TARGET].to_numpy()
        if k not in store:
            folds = 5 if name in ("grouped seed 42", "temporal") else 3
            model, oof, thr = daynight.fit_variant(train, ytr, train["cell_id"].to_numpy(), list(FEATURES), folds=folds)
            p = model.predict_proba(test[FEATURES])[:, 1]
            store[k] = {"p": p.astype(np.float32), "oof": oof.astype(np.float32), "thr": thr, "feats": list(FEATURES)}
            if name == "grouped seed 42":
                import joblib
                joblib.dump({"model": model, "features": list(FEATURES), "threshold": thr},
                            PROC_DIR / "daynight_xgb_full.joblib")
            del model
            daynight._save(store)
        x = store[k]
        rows.append(tier_rows(name, "instant", yte, np.asarray(x["p"], float), float(x["thr"]), None))
        print(name, "instant pr_auc", round(rows[-1]["pr_auc"], 4), "f1", round(rows[-1]["f1opt_f1"], 4), flush=True)
        report(f"instant {name}")
    out = pd.DataFrame(rows)
    out.round(4).to_csv(REPORT_DIR / "world_instant.csv", index=False)
    return out


def write_headline(df):
    """Refresh the instant tier files the app shows (metrics_headline.json, model_comparison_headline.csv, the
    confusion matrix and the feature importance) for the shipped XGBoost model on the grouped seed 42 split.
    The random forest comparison of the three region study is not repeated: a 300 tree forest on all regions
    does not fit in 8 GB. The earlier files are in report_backups/before_world."""
    import json

    import joblib
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    from . import daynight
    from .config import FEATURES, STATIC_MIN_DAYS, STATIC_WINDOW
    from .train import score
    splits = split_sets(df)
    tr, te, _, _ = splits["grouped seed 42"]
    saved = joblib.load(PROC_DIR / "daynight_xgb_full.joblib")
    store = daynight._load()
    x = store[("xgb", "grouped seed 42", "full model")]
    yte = df.iloc[te][TARGET].to_numpy()
    p, thr = np.asarray(x["p"], float), float(saved["threshold"])
    prior = float(df.iloc[tr][TARGET].mean())
    results = {"Majority baseline": score(yte, np.full(len(yte), prior), 0.5), "XGBoost": score(yte, p, thr)}
    summary = {"rows": int(len(df)), "cells": int(df["cell_id"].nunique()), "days": int(df["acq_date"].nunique()),
               "date_min": str(df["acq_date"].min().date()), "date_max": str(df["acq_date"].max().date()),
               "static_share": round(float(df[TARGET].mean()), 4), "static_min_days": STATIC_MIN_DAYS,
               "window": STATIC_WINDOW, "label": "type", "regions": int(df["region"].nunique()),
               "features": list(FEATURES), "train_rows": int(len(tr)), "test_rows": int(len(te)),
               "test_static_share": round(float(yte.mean()), 4), "best_model": "XGBoost, 100 trees",
               "results": results}
    (REPORT_DIR / "metrics_headline.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame(results).T.to_csv(REPORT_DIR / "model_comparison_headline.csv")
    cm = confusion_matrix(yte, (p >= thr).astype(int))
    fig, ax = plt.subplots(figsize=(4, 3.4))
    ax.imshow(cm, cmap="Greys")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v}", ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["fire", "static"])
    ax.set_yticks([0, 1], ["fire", "static"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("XGBoost, test set")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "confusion_matrix_headline.png", dpi=160)
    plt.close(fig)
    imp = pd.Series(saved["model"].feature_importances_, index=saved["features"]).sort_values()
    fig, ax = plt.subplots(figsize=(5, 3.8))
    ax.barh(imp.index, imp.values, color="#444444")
    ax.set_title("Feature importance, XGBoost")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "feature_importance_headline.png", dpi=160)
    plt.close(fig)


OLD = ROOT.parent / "report_backups" / "before_world"


def build_regions():
    """Leave one region out for both tiers and the one line rule: train on every other region, test on the held
    out one. Written to reports/world_regions.csv from the rows of the history and instant stages."""
    h = pd.read_csv(REPORT_DIR / "world_history.csv")
    i = pd.read_csv(REPORT_DIR / "world_instant.csv")
    both = pd.concat([h, i], ignore_index=True)
    both = both[both["split"].str.startswith("leave out ")].copy()
    both.insert(0, "region", both["split"].str.replace("leave out ", "", regex=False))
    both["tier"] = both["tier"].str.replace(r"^rule .*$", "rule", regex=True)
    cols = ["region", "tier", "n", "static_share", "majority_pr_auc", "pr_auc", "f1opt_precision", "f1opt_recall",
            "f1opt_f1", "p90_precision", "p90_recall", "p90_f1"]
    out = both[[c for c in cols if c in both]].rename(columns={"f1opt_precision": "precision", "f1opt_recall": "recall",
                                                                "f1opt_f1": "f1"})
    out.round(4).to_csv(REPORT_DIR / "world_regions.csv", index=False)
    return out


def old_results():
    """The three region results before this extension: leave one region out for history (setup D XGBoost), the
    one line rule and the instant XGBoost, and the grouped seed 42 numbers."""
    b = pd.read_csv(OLD / "history2_eval_b.csv")
    hist = b[b["setup"].str.startswith("D") & (b["model"] == "XGBoost")].set_index("region")
    rule = b[b["setup"] == "Rule"].set_index("region")
    r = pd.read_csv(OLD / "region_eval_type_region.csv")
    inst = r[(r["model"] == "XGBoost") & (~r["hour_utc"].astype(bool))].set_index("region")
    a = pd.read_csv(OLD / "history2_eval_a.csv")
    seed = a[a["setup"].str.startswith("D") & (a["model"] == "XGBoost") & (a["subset"] == "all rows") & (a["seed"] == 42)]
    return hist, rule, inst, seed.iloc[0]


def write_summary(df=None):
    """reports/world_summary.md: one table of old and new leave one region out results, plus the grouped split."""
    from .regions import FILE_REGION, ORIGINAL
    reg = build_regions().set_index(["region", "tier"])
    oh, orule, oinst, oseed = old_results()
    inv = pd.read_csv(REPORT_DIR / "world_inventory.csv").set_index("region")

    def pair(new, old):
        if old is None or pd.isna(old):
            return f"{new:.3f}"
        return f"{old:.3f} to {new:.3f}"
    worse, lines = [], []
    head = ("| Region | Rows | Static share | History PR-AUC | History F1 | Instant PR-AUC | Instant F1 | "
            "Rule PR-AUC | Rule F1 |")
    lines += [head, "|" + "---|" * 9]
    for region in FILE_REGION.values():
        n = reg.loc[(region, "history"), "n"]
        row = [region, f"{int(n):,}", f"{reg.loc[(region, 'history'), 'static_share']:.4f}"]
        for tier, olddf, ocols in (("history", oh, ("pr_auc_static", "f1_static")),
                                   ("instant", oinst, ("pr_auc_static", "f1_static")),
                                   ("rule", orule, ("pr_auc_static", "f1_static"))):
            for metric, ocol in zip(("pr_auc", "f1"), ocols):
                new = float(reg.loc[(region, tier), metric])
                old = float(olddf.loc[region, ocol]) if region in ORIGINAL and region in olddf.index else None
                row.append(pair(new, old))
                if old is not None and new < old - 1e-9:
                    worse.append(f"{region}, {tier} {metric.replace('_', '-').upper() if metric == 'pr_auc' else 'F1'}: "
                                 f"{old:.3f} to {new:.3f}")
        lines.append("| " + " | ".join(row) + " |")
    text = ["World summary", "",
            "Every region file in data/raw (January to June 2025) is now used. Leave one region out: train on all other "
            "regions, test on the held out one. For the three original regions the first number is the earlier result "
            "(trained on the other two regions) and the second the new result (trained on the other seven). New regions show "
            "the new result only. History is setup D, XGBoost, 100 trees, threshold at the F1 optimum of out of fold "
            "training scores. Instant is XGBoost, 100 trees, same rule. The rule is distinct days in the prior 30 at "
            "least k, k picked on the training part. The label is NASA's repeat based type flag, not independent truth.", ""]
    text += lines
    text += ["", "Where a number got worse (original regions, earlier to new):", ""]
    text += [f"- {w}" for w in worse] or ["- nothing got worse"]
    p = REPORT_DIR / "world_history.csv"
    h = pd.read_csv(p)
    g = h[(h["split"] == "grouped seed 42") & (h["tier"] == "history")].iloc[0]
    text += ["", "Grouped split by grid cell, seed 42, all regions together (20 percent of cells held out), history tier:", "",
             f"- PR-AUC {oseed['pr_auc_static']:.3f} (three regions) to {g['pr_auc']:.3f} (eight regions), "
             f"F1 {oseed['f1_static']:.3f} to {g['f1opt_f1']:.3f}. These are different test sets, so the two numbers "
             "are not a like for like comparison."]
    (REPORT_DIR / "world_summary.md").write_text(chr(10).join(text) + chr(10), encoding="utf-8")
    return worse


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    stage = argv[0] if argv else ""
    if stage == "backup":
        backup()
    elif stage == "inventory":
        inventory()
    elif stage == "prepare":
        from .worlddata import prepare
        df = prepare(use_cache=False)
        print(f"{len(df):,} rows, {df['cell_id'].nunique():,} cells, static share {df[TARGET].mean():.4f}")
        print(df.groupby("region", observed=True)[TARGET].agg(["size", "mean"]).round(4).to_string())
        print(f"in memory: {df.memory_usage(deep=True).sum() / 1e6:.0f} MB")
    elif stage == "history":
        from .history2_eval import prepare
        run_history(prepare())
    elif stage == "instant":
        from .history2_eval import prepare
        run_instant(prepare())
    elif stage == "headline":
        from .history2_eval import prepare
        write_headline(prepare())
    elif stage == "regions":
        print(build_regions().round(3).to_string(index=False))
    elif stage == "summary":
        for w in write_summary():
            print("worse:", w)
    else:
        raise SystemExit(__doc__)
    report(stage)


if __name__ == "__main__":
    main()

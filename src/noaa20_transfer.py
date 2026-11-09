"""Transfer experiment: models trained on Suomi NPP, tested on NOAA-20.

Run from the project root (after python -m src.noaa20):  python -m src.noaa20_transfer

The models are the shipped ones, trained on the Suomi NPP cells of the seed 42 grouped split.
Their thresholds, calibrators and uncertain bands come from Suomi NPP out of fold training
scores and are used unchanged on NOAA-20. NOAA-20 features and history features are built
inside the NOAA-20 files only. No coordinates and no hour_utc are model inputs.

Test sets:
  cell disjoint      NOAA-20 detections in cells that are not Suomi NPP training cells
  not cell disjoint  NOAA-20 detections in cells that were Suomi NPP training cells (easier,
                     the model has seen those places on the other satellite)
The label is NASA's repeat based type flag (static = 2, fire = 0), not independent truth, and the
two satellites observe the same places only a few tens of minutes apart (matched pairs: about 22 minutes).
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit

from . import uncertain as U
from .config import MODEL_DIR, PROC_DIR, RAW_DIR, REPORT_DIR, TARGET
from .data import load_raw
from .features import add_features
from .history import add_history, add_history_v2
from .history_eval import RULE_COL, choose_k
from .label import add_label
from .noaa20 import REGION_COUNTRY, out_dir

NAMES = {"noaa20": "NOAA-20", "noaa21": "NOAA-21"}


def file_region(sat="noaa20"):
    return {f"{sat}_{r.replace(' ', '_').lower()}.csv": r for r in REGION_COUNTRY}


FILE_REGION = file_region("noaa20")
SNPP_REGION = {"fire_archive_SV-C2_812937.csv": "South Asia", "fire_archive_SV-C2_812939.csv": "Southeast Asia",
               "fire_archive_SV-C2_812940.csv": "North and Central America"}
SEED = 42
SENSOR_NAME = {"noaa20": "NOAA-20", "noaa21": "NOAA-21", "snpp": "Suomi NPP"}


def build_noaa20(sat="noaa20"):
    """Features and history features of a VIIRS standard processing satellite, each file on its own,
    same label rule as Suomi NPP."""
    paths = sorted(out_dir(sat).glob(f"{sat}_*.csv"))
    if not paths:
        raise SystemExit(f"No {NAMES[sat]} files, run python -m src.noaa20 --sat {sat} first")
    df = add_features(add_label(load_raw(paths)))
    df = add_history_v2(add_history(df))                 # history is computed per source file
    df = df[df["type"].isin([0, 2])].reset_index(drop=True)
    df[TARGET] = (df["type"] == 2).astype(int)
    df["region"] = df["source_file"].map(file_region(sat))
    return df


def compare_sensors(noaa, snpp, days, max_deg=0.006, max_minutes=120, an="noaa20", bn="snpp"):
    """Match detections of satellite a and satellite b of the same day and region by position and time."""
    from scipy.spatial import cKDTree
    rows = []
    for region in REGION_COUNTRY:
        a_all = noaa[noaa["region"] == region]
        b_all = snpp[snpp["region"] == region]
        for day in days:
            a = a_all[a_all["acq_date"] == day]
            b = b_all[b_all["acq_date"] == day]
            row = {"region": region, "day": str(day.date()), f"{an}_rows": len(a), f"{bn}_rows": len(b)}
            if len(a) and len(b):
                scale = np.cos(np.radians(a["latitude"].mean()))
                tree = cKDTree(np.c_[b["latitude"], b["longitude"] * scale])
                dist, idx = tree.query(np.c_[a["latitude"], a["longitude"] * scale])
                dt = (a["hour_utc"].to_numpy() - b["hour_utc"].to_numpy()[idx]) * 60
                dt = (dt + 720) % 1440 - 720                               # minutes, wrap over midnight
                ok = (dist <= max_deg) & (np.abs(dt) <= max_minutes)
                bm = b.iloc[idx[ok]]
                am = a[ok]
                row.update(matched=int(ok.sum()), **{f"{an}_matched_share": float(ok.mean()), f"{bn}_matched_share": float(len(np.unique(idx[ok])) / len(b))},
                           time_offset_min_median=float(np.median(dt[ok])) if ok.any() else np.nan,
                           time_offset_min_q25=float(np.percentile(dt[ok], 25)) if ok.any() else np.nan,
                           time_offset_min_q75=float(np.percentile(dt[ok], 75)) if ok.any() else np.nan,
                           ti4_diff_median=float(np.median(am["bright_ti4"].to_numpy() - bm["bright_ti4"].to_numpy()))
                           if ok.any() else np.nan,
                           ti4_diff_mean_abs=float(np.mean(np.abs(am["bright_ti4"].to_numpy() - bm["bright_ti4"].to_numpy())))
                           if ok.any() else np.nan,
                           ti5_diff_median=float(np.median(am["bright_ti5"].to_numpy() - bm["bright_ti5"].to_numpy()))
                           if ok.any() else np.nan,
                           frp_diff_median=float(np.median(am["frp"].to_numpy() - bm["frp"].to_numpy())) if ok.any() else np.nan,
                           frp_ratio_median=float(np.median(am["frp"].to_numpy() / bm["frp"].clip(lower=0.01).to_numpy()))
                           if ok.any() else np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


def generic_pair(df, an, bn):
    """Column names of a sensor comparison with a and b in place of the satellite names (a is the first sensor)."""
    return df.rename(columns={f"{an}_rows": "rows_a", f"{bn}_rows": "rows_b",
                              f"{an}_matched_share": "matched_share_a", f"{bn}_matched_share": "matched_share_b"})


def metrics(y, p, thr, cal, bands, label):
    """Static PR-AUC, precision, recall, F1 at the Suomi NPP threshold, calibration with the Suomi NPP
    calibrator, uncertain shares for the default and strict bands, and the majority baselines."""
    y, p = np.asarray(y), np.asarray(p, float)
    if len(y) == 0 or y.min() == y.max():
        return {"n": len(y)}
    pred = (p >= thr).astype(int)
    c = U.apply_calibrator(cal, p)
    out = {"n": len(y), "static_share": float(y.mean()), "majority_pr_auc": float(y.mean()),
           "majority_accuracy": float(max(y.mean(), 1 - y.mean())),
           "pr_auc": average_precision_score(y, p), "threshold": float(thr),
           "precision": precision_score(y, pred, zero_division=0), "recall": recall_score(y, pred, zero_division=0),
           "f1": f1_score(y, pred, zero_division=0), "accuracy": float((pred == y).mean()),
           "brier_raw": U.brier(y, p), "brier_calibrated": U.brier(y, c),
           "ece_raw": U.ece(y, p), "ece_calibrated": U.ece(y, c)}
    for key, name in (("default", "0.96"), ("strict", "0.98")):
        band = bands.get(key) if bands else None
        if band and band.get("reachable", True):
            d = U.decide(c, band)
            done = d != U.UNCERTAIN
            out[f"uncertain_share_{name}"] = float((~done).mean())
            out[f"decided_accuracy_{name}"] = float((np.where(d == U.STATIC, 1, 0)[done] == y[done]).mean()) if done.any() else np.nan
    return out


def main(sat="noaa20"):
    from .history2_eval import prepare
    name = NAMES[sat]
    fr = file_region(sat)
    noaa = build_noaa20(sat)
    print(name, "rows with type 0 or 2:", len(noaa), noaa["region"].value_counts().to_dict(), flush=True)

    # Suomi NPP side: training cells of the seed 42 split, the rule threshold, sensor comparison inputs
    snpp = prepare()
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
                  .split(snpp, snpp[TARGET], snpp["cell_id"]))
    train_cells = set(snpp.iloc[tr]["cell_id"])
    k = choose_k(snpp.iloc[tr])
    disjoint = ~noaa["cell_id"].isin(train_cells).to_numpy()
    print(f"{name} rows in cells unseen in Suomi NPP training: {disjoint.mean():.3f}; rule k = {k}", flush=True)

    # sensor comparison on a few days, all rows (not only types 0 and 2), same country cut
    raw_n = load_raw(sorted(out_dir(sat).glob(f"{sat}_*.csv")))
    raw_n["region"] = raw_n["source_file"].map(fr)
    raw_s = load_raw()
    raw_s["region"] = raw_s["source_file"].map(SNPP_REGION)
    days = [pd.Timestamp(d) for d in ("2025-01-15", "2025-02-15", "2025-03-15", "2025-04-15", "2025-05-15",
                                       "2025-06-15")]
    cmp_ = compare_sensors(raw_n, raw_s, days, an=sat, bn="snpp")
    counts = pd.DataFrame({"region": list(fr.values())})
    counts[f"{sat}_rows"] = [int((raw_n["region"] == r).sum()) for r in counts["region"]]
    counts["snpp_rows"] = [int((raw_s["region"] == r).sum()) for r in counts["region"]]
    counts["ratio"] = counts[f"{sat}_rows"] / counts["snpp_rows"]
    counts.round(3).to_csv(REPORT_DIR / f"{sat}_row_counts.csv", index=False)
    print(counts.to_string(index=False), flush=True)
    extra = None
    if sat != "noaa20":                                       # the three sensors: also against NOAA-20
        raw_t = load_raw(sorted(out_dir("noaa20").glob("noaa20_*.csv")))
        raw_t["region"] = raw_t["source_file"].map(file_region("noaa20"))
        extra = compare_sensors(raw_n, raw_t, days, an=sat, bn="noaa20")
        both = pd.concat([generic_pair(cmp_, sat, "snpp").assign(pair=f"{name} vs Suomi NPP"),
                          generic_pair(extra, sat, "noaa20").assign(pair=f"{name} vs NOAA-20")])
        both.round(3).to_csv(REPORT_DIR / f"{sat}_sensor_comparison.csv", index=False)
        del raw_t
    else:
        cmp_.round(3).to_csv(REPORT_DIR / "noaa20_sensor_comparison.csv", index=False)
    del raw_n, raw_s

    rows = []
    tiers = {}
    for tier, fname, sidecar, thr_key in (("history", "best_model_history.joblib", "best_model_history_band.json",
                                           "precision >= 0.90"),
                                          ("instant", "best_model_headline.joblib", "best_model_headline_band.json", None)):
        bundle = joblib.load(MODEL_DIR / fname, mmap_mode="r")
        got = U.bands_of(bundle, sidecar)
        thr = float(bundle["thresholds"][thr_key]) if thr_key else float(bundle["threshold"])
        p = bundle["model"].predict_proba(noaa[bundle["features"]])[:, 1]
        tiers[tier] = (p, thr, got)
        print(tier, name, "scored", flush=True)

    sets = {"cell disjoint": disjoint, "not cell disjoint": ~disjoint}
    for tier, (p, thr, got) in tiers.items():
        cal, bands = got["calibrator"], {"default": got["default"], "strict": got["strict"]}
        for set_name, mask in sets.items():
            for region in ["all"] + list(fr.values()):
                sel = mask & ((noaa["region"] == region).to_numpy() if region != "all" else True)
                m = metrics(noaa[TARGET].to_numpy()[sel], p[sel], thr, cal, bands, tier)
                rows.append({"model": tier, "test_set": set_name, "region": region, **m})
        # reference: the same shipped model on its own Suomi NPP held out cells
        scored = pd.read_parquet(PROC_DIR / ("scored_history.parquet" if tier == "history" else "scored_headline.parquet"))
        col = "p_static_history" if tier == "history" else "p_static"
        ref = scored[scored["in_test"]]
        rows.append({"model": tier, "test_set": "Suomi NPP held out cells (reference)", "region": "all",
                     **metrics(ref[TARGET].to_numpy(), ref[col].to_numpy(), thr, cal, bands, tier)})

    # one line rule: static if distinct days in the prior 30 is at least k (k from Suomi NPP training)
    score = noaa[RULE_COL].to_numpy(float)
    snpp_te = snpp.iloc[te]
    for set_name, mask in sets.items():
        for region in ["all"] + list(fr.values()):
            sel = mask & ((noaa["region"] == region).to_numpy() if region != "all" else True)
            y = noaa[TARGET].to_numpy()[sel]
            if len(y) == 0 or y.min() == y.max():
                rows.append({"model": f"rule {RULE_COL} >= {k}", "test_set": set_name, "region": region, "n": len(y)})
                continue
            pred = (score[sel] >= k).astype(int)
            rows.append({"model": f"rule {RULE_COL} >= {k}", "test_set": set_name, "region": region, "n": len(y),
                         "static_share": float(y.mean()), "majority_pr_auc": float(y.mean()),
                         "majority_accuracy": float(max(y.mean(), 1 - y.mean())),
                         "pr_auc": average_precision_score(y, score[sel]), "threshold": float(k),
                         "precision": precision_score(y, pred, zero_division=0),
                         "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0),
                         "accuracy": float((pred == y).mean())})
    y = snpp_te[TARGET].to_numpy()
    pred = (snpp_te[RULE_COL].to_numpy() >= k).astype(int)
    rows.append({"model": f"rule {RULE_COL} >= {k}", "test_set": "Suomi NPP held out cells (reference)", "region": "all",
                 "n": len(y), "static_share": float(y.mean()), "majority_pr_auc": float(y.mean()),
                 "pr_auc": average_precision_score(y, snpp_te[RULE_COL].to_numpy(float)), "threshold": float(k),
                 "precision": precision_score(y, pred, zero_division=0), "recall": recall_score(y, pred, zero_division=0),
                 "f1": f1_score(y, pred, zero_division=0), "accuracy": float((pred == y).mean())})

    out = pd.DataFrame(rows)
    out.round(4).to_csv(REPORT_DIR / f"{sat}_transfer.csv", index=False)
    write_summary(out, cmp_, counts, float(disjoint.mean()), k, sat, extra)
    cols = ["model", "test_set", "region", "n", "static_share", "pr_auc", "precision", "recall", "f1",
            "brier_calibrated", "ece_calibrated", "uncertain_share_0.96", "uncertain_share_0.98"]
    print(out[[c for c in cols if c in out]].round(3).to_string(index=False))


def sensor_lines(cmp_, an="noaa20", bn="snpp"):
    """Weighted sensor differences over the matched pairs of the sampled days."""
    w = cmp_.dropna(subset=["matched"])
    wt = w["matched"]

    def avg(col):
        return float(np.average(w[col], weights=wt))
    na, nb = SENSOR_NAME[an], SENSOR_NAME[bn]
    return [f"Matched pairs on six days: {int(wt.sum()):,}. Share of {na} detections with a {nb} match "
            f"{avg(an + '_matched_share'):.2f}, share of {nb} detections with a {na} match "
            f"{avg(bn + '_matched_share'):.2f}.",
            f"Acquisition time offset {na} minus {nb}: median of the daily medians "
            f"{float(w['time_offset_min_median'].median()):.0f} minutes, range of the daily medians "
            f"{float(w['time_offset_min_median'].min()):.0f} to {float(w['time_offset_min_median'].max()):.0f}.",
            f"Brightness I-4 difference: median {avg('ti4_diff_median'):.2f} K, mean absolute {avg('ti4_diff_mean_abs'):.1f} K. "
            f"Brightness I-5 median {avg('ti5_diff_median'):.2f} K. FRP median difference {avg('frp_diff_median'):.2f} MW, "
            f"median ratio {avg('frp_ratio_median'):.2f}.",
    ]


def write_summary(out, cmp_, counts, disjoint_share, k, sat="noaa20", extra=None):
    a = out[out["region"] == "all"]
    name = NAMES[sat]

    def row(model, tset):
        r = a[(a["model"].str.startswith(model)) & (a["test_set"] == tset)]
        return r.iloc[0] if len(r) else None
    lines = [f"{name} transfer summary", "",
             f"Models trained on Suomi NPP train cells (seed 42 grouped split), tested on {name}. Thresholds, "
             "calibrators and bands come from Suomi NPP out of fold training scores, unchanged. The label is "
             "NASA's repeat based type flag, not independent truth. The satellites observe the same places "
             "within a short time of each other (see the offsets below), so the test is a transfer between "
             "near twins, not an independent check.", "",
             f"Share of {name} rows in cells unseen in Suomi NPP training: {disjoint_share:.3f}. Rule k: {k}.", ""]
    for model in ("history", "instant", "rule"):
        for tset in ("Suomi NPP held out cells (reference)", "cell disjoint", "not cell disjoint"):
            r = row(model, tset)
            if r is None:
                continue
            lines.append(f"{model}, {tset}: n {int(r['n'])}, static PR-AUC {r.get('pr_auc', float('nan')):.3f} "
                         f"(majority {r.get('majority_pr_auc', float('nan')):.3f}), precision {r.get('precision', float('nan')):.3f}, "
                         f"recall {r.get('recall', float('nan')):.3f}, F1 {r.get('f1', float('nan')):.3f}")
    lines += ["", "Row counts per region (country cut):", counts.to_string(index=False), ""]
    lines += sensor_lines(cmp_, sat, "snpp")
    if extra is not None:
        lines += [""] + sensor_lines(extra, sat, "noaa20")
    lines.append("The sensors are the same instrument type on different platforms. Single detections differ by "
                 "several kelvin because the overpass time and the pixel footprint differ.")
    (REPORT_DIR / f"{sat}_transfer_summary.md").write_text(chr(10).join(lines) + chr(10), encoding="utf-8")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "noaa20")

"""Analysis for src/nrt_vs_sp.py: feature distributions, static shares and matched pairs.

Run from the project root (after python -m src.nrt_vs_sp --download):  python -m src.nrt_analysis
"""
import sys

import joblib
import numpy as np
import pandas as pd

from . import live, noaa20
from .config import MODEL_DIR, REPORT_DIR
from .nrt_vs_sp import load_set

REGIONS = list(noaa20.REGION_COUNTRY)
FEATS = ["bright_ti4", "bright_ti5", "bright_diff", "frp", "scan", "track"]
LIVE_DAYS = ("2026-09-11", "2026-11-09")
MATCH_DEG, MATCH_MIN = 0.006, 30


def cut_country(raw, region):
    """Rows inside the country of the region (the cut the SP archive experiments use), duplicates dropped."""
    inside = noaa20.in_country(raw["longitude"], raw["latitude"], noaa20.REGION_COUNTRY[region])
    keep = [c for c in ("latitude", "longitude", "acq_date", "acq_time") if c in raw.columns]
    return raw[inside].drop_duplicates(subset=keep).reset_index(drop=True)


def raw_of(tag, region):
    """Raw detections of a set inside the country. B_NRT is the live NOAA-21 store."""
    if tag == "B_NRT":
        raw = live.load_store(region, "NOAA-21")
        if raw is None:
            return pd.DataFrame()
        return cut_country(raw.drop(columns=["satellite_name"]), region)
    raw = load_set(tag, region)
    return cut_country(raw, region) if len(raw) else raw


def scored(raw, name, hist, inst, remap=None):
    """Features, history (inside this file only) and both scores. remap is an optional function that
    changes the raw frame before the training pipeline (the fix under test)."""
    if remap is not None:
        raw = remap(raw.copy())
    df = live.prepare(raw, name)
    return live.score(df, hist, inst)


def conf_counts(raw):
    """Share of every raw confidence value, as text keys."""
    c = raw["confidence"].astype(str).str.strip().str.lower()
    return c.value_counts(normalize=True).to_dict()


def describe(df, raw, tag, region, hist, inst, hist_thr, inst_thr):
    row = {"kind": "set", "set": tag, "region": region, "rows": len(df)}
    if df.empty:
        return row
    for f in FEATS:
        row[f"{f}_median"] = float(df[f].median())
        row[f"{f}_q05"] = float(df[f].quantile(0.05))
        row[f"{f}_q95"] = float(df[f].quantile(0.95))
    row["confidence_ord_mean"] = float(df["confidence_ord"].mean())
    cc = conf_counts(raw)
    for k in ("l", "n", "h"):
        row[f"conf_raw_{k}"] = float(cc.get(k, 0.0))
    row["conf_raw_other"] = float(1 - sum(cc.get(k, 0.0) for k in ("l", "n", "h")))
    row["night_share"] = float(df["is_night"].mean())
    row["instant_static_share"] = float((df["p_static_instant"] >= inst_thr).mean())
    row["history_static_share"] = float((df["p_static_history"] >= hist_thr).mean())
    night = df["is_night"].to_numpy() == 1
    for name, col, thr in (("instant", "p_static_instant", inst_thr), ("history", "p_static_history", hist_thr)):
        st = (df[col].to_numpy() >= thr)
        row[f"{name}_static_share_day"] = float(st[~night].mean()) if (~night).any() else float("nan")
        row[f"{name}_static_share_night"] = float(st[night].mean()) if night.any() else float("nan")
    row["instant_score_mean"] = float(df["p_static_instant"].mean())
    row["history_score_mean"] = float(df["p_static_history"].mean())
    return row


def match(nrt, sp, max_deg=MATCH_DEG, max_minutes=MATCH_MIN):
    """Index pairs (nrt row, sp row) of detections of the same day within max_deg degrees and max_minutes."""
    from scipy.spatial import cKDTree
    out_a, out_b = [], []
    for day, a in nrt.groupby("acq_date"):
        b = sp[sp["acq_date"] == day]
        if b.empty:
            continue
        scale = np.cos(np.radians(a["latitude"].mean()))
        tree = cKDTree(np.c_[b["latitude"], b["longitude"] * scale])
        dist, idx = tree.query(np.c_[a["latitude"], a["longitude"] * scale])
        dt = (a["hour_utc"].to_numpy() - b["hour_utc"].to_numpy()[idx]) * 60
        dt = (dt + 720) % 1440 - 720
        ok = (dist <= max_deg) & (np.abs(dt) <= max_minutes)
        out_a.append(a.index.to_numpy()[ok])
        out_b.append(b.index.to_numpy()[idx[ok]])
    if not out_a:
        return np.array([], int), np.array([], int)
    return np.concatenate(out_a), np.concatenate(out_b)


def matched_row(nrt, sp, region, inst_thr, hist_thr, label="matched"):
    ia, ib = match(nrt, sp)
    row = {"kind": label, "set": "A_NRT vs A_SP", "region": region, "rows": len(ia),
           "nrt_matched_share": len(ia) / max(len(nrt), 1),
           "sp_matched_share": len(np.unique(ib)) / max(len(sp), 1)}
    if len(ia) == 0:
        return row
    a, b = nrt.loc[ia].reset_index(drop=True), sp.loc[ib].reset_index(drop=True)
    for f in FEATS + ["confidence_ord"]:
        d = a[f].to_numpy(float) - b[f].to_numpy(float)
        row[f"{f}_diff_median"] = float(np.median(d))
        row[f"{f}_diff_mean_abs"] = float(np.mean(np.abs(d)))
    row["confidence_same_share"] = float((a["confidence_ord"].to_numpy() == b["confidence_ord"].to_numpy()).mean())
    row["night_same_share"] = float((a["is_night"].to_numpy() == b["is_night"].to_numpy()).mean())
    pa, pb = a["p_static_instant"].to_numpy(), b["p_static_instant"].to_numpy()
    row["instant_score_diff_median"] = float(np.median(pa - pb))
    row["instant_static_share_nrt"] = float((pa >= inst_thr).mean())
    row["instant_static_share_sp"] = float((pb >= inst_thr).mean())
    row["instant_decision_differs"] = float(((pa >= inst_thr) != (pb >= inst_thr)).mean())
    ha, hb = a["p_static_history"].to_numpy(), b["p_static_history"].to_numpy()
    row["history_static_share_nrt"] = float((ha >= hist_thr).mean())
    row["history_static_share_sp"] = float((hb >= hist_thr).mean())
    return row


def crosstab_conf(nrt_raw, sp_raw, nrt, sp):
    """Confidence of matched pairs, raw value of NRT against raw value of SP."""
    ia, ib = match(nrt, sp)
    a = nrt_raw.loc[ia, "confidence"].astype(str).str.lower().to_numpy() if len(ia) else []
    b = sp_raw.loc[ib, "confidence"].astype(str).str.lower().to_numpy() if len(ib) else []
    return pd.crosstab(pd.Series(a, name="NRT"), pd.Series(b, name="SP"))


def standardised(rows, ref_tag, tag, region):
    """Static shares of set `tag` if it had the day and night mix of `ref_tag` (direct standardisation)."""
    by = {(r["set"], r["region"]): r for r in rows if r.get("kind") == "set" and "night_share" in r}
    a, b = by.get((tag, region)), by.get((ref_tag, region))
    if a is None or b is None:
        return None
    w = b["night_share"]
    out = {"kind": "standardised", "set": f"{tag} at the night share of {ref_tag}", "region": region,
           "night_share": w, "rows": a["rows"]}
    for name in ("instant", "history"):
        out[f"{name}_static_share"] = (a[f"{name}_static_share_day"] * (1 - w) + a[f"{name}_static_share_night"] * w)
    return out


def archive_reference(hist_thr, inst_thr):
    """The Suomi NPP training archive (Jan to Jun 2025, three regions): the same shares for comparison."""
    from .config import PROC_DIR
    a = pd.read_parquet(PROC_DIR / "scored_headline.parquet")
    h = pd.read_parquet(PROC_DIR / "scored_history.parquet")
    night = a["is_night"].to_numpy() == 1
    st_i = a["p_static"].to_numpy() >= inst_thr
    st_h = h["p_static_history"].to_numpy() >= hist_thr
    nh = h["is_night"].to_numpy() == 1
    return {"kind": "archive", "set": "Suomi NPP archive Jan to Jun 2025", "region": "all", "rows": len(a),
            "night_share": float(night.mean()), "instant_static_share": float(st_i.mean()),
            "instant_static_share_day": float(st_i[~night].mean()), "instant_static_share_night": float(st_i[night].mean()),
            "history_static_share": float(st_h.mean()), "history_static_share_day": float(st_h[~nh].mean()),
            "history_static_share_night": float(st_h[nh].mean()), "nasa_static_share": float(a["is_static"].mean())}


def main(argv=None):
    hist = joblib.load(MODEL_DIR / "best_model_history.joblib", mmap_mode="r")
    inst = joblib.load(MODEL_DIR / "best_model_headline.joblib", mmap_mode="r")
    hist_thr = float(hist["thresholds"][hist.get("default_operating_point", "precision >= 0.90")])
    inst_thr = float(inst["threshold"])
    rows = []
    for region in REGIONS:
        data = {}
        for tag in ("A_SP", "A_NRT", "B_SP", "B_NRT"):
            raw = raw_of(tag, region)
            if raw.empty:
                continue
            df = scored(raw, f"{tag}_{region}", hist, inst)
            data[tag] = (raw, df)
            rows.append({**describe(df, raw, tag, region, hist, inst, hist_thr, inst_thr)})
            print(region, tag, len(df), f"instant {rows[-1].get('instant_static_share', float('nan')):.3f}",
                  f"history {rows[-1].get('history_static_share', float('nan')):.3f}", flush=True)
        if "A_SP" in data and "A_NRT" in data:
            rows.append(matched_row(data["A_NRT"][1], data["A_SP"][1], region, inst_thr, hist_thr))
            print(crosstab_conf(data["A_NRT"][0], data["A_SP"][0], data["A_NRT"][1], data["A_SP"][1]), flush=True)
        del data
        for tag in ("B_NRT", "A_SP"):
            x = standardised(rows, "B_SP", tag, region) if tag == "B_NRT" else None
            if x:
                rows.append(x)
    rows.append(archive_reference(hist_thr, inst_thr))
    out = pd.DataFrame(rows)
    out.round(4).to_csv(REPORT_DIR / "nrt_vs_sp.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 60)
    print(out.round(3).T.to_string())


if __name__ == "__main__":
    main()

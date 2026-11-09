"""Fetch the near real time feeds and print sanity checks on the real data.

Run from the project root:  python -m src.live_check [--no-fetch]

Fetches NOAA-20 and NOAA-21 near real time data for the three training regions (5 day chunks, a short
pause between calls, the transaction limit is 5000 per 10 minutes), scores it with the Suomi NPP
trained models, calibrator and bands, and prints per region and per satellite: the predicted static
share, how many cells with at least 10 detections in the last 30 days are predicted static, and the
overlap of the predicted static cells of the two satellites. Writes reports/live_check.csv.
There is no ground truth in these feeds, so nothing here is an accuracy.

The key is read from .env and never printed. Request errors print only a short safe message.
"""
import sys

import joblib
import numpy as np
import pandas as pd

from . import live
from . import uncertain as U
from .config import MODEL_DIR, REPORT_DIR

REGIONS = ["South Asia", "Southeast Asia", "North and Central America"]


def cells_static(df, col, thr, min_det=10, days=30):
    """Cells with at least min_det detections in the last `days` days of the data, and the cells among
    them where most detections are predicted static."""
    cut = df["acq_date"].max() - pd.Timedelta(days=days - 1)
    r = df[df["acq_date"] >= cut].assign(st=lambda d: (d[col] >= thr).astype(int))
    g = r.groupby("cell_id")["st"].agg(["size", "mean"])
    g = g[g["size"] >= min_det]
    return set(g.index), set(g.index[g["mean"] >= 0.5])


def check(region, hist, inst, thr, raw=None):
    """Rows of the sanity table for one region: one per satellite plus the overlap row."""
    raw = live.load_store(region) if raw is None else raw
    if raw is None:
        return []
    out, sets = [], {}
    df = live.score(live.prepare(raw, region), hist, inst)
    for sat in live.LIVE_SOURCES:
        d = df[df["satellite_name"] == sat]
        if d.empty:
            continue
        busy, static = cells_static(d, "p_static_history", thr)
        sets[sat] = static
        out.append({"region": region, "satellite": sat, "rows": len(d),
                    "first_day": f"{d['acq_date'].min():%Y-%m-%d}", "last_day": f"{d['acq_date'].max():%Y-%m-%d}",
                    "predicted_static_share": float((d["p_static_history"] >= thr).mean()),
                    "instant_static_share": float((d["p_static_instant"] >= float(inst["threshold"])).mean()),
                    "cells_10plus_30d": len(busy), "cells_10plus_predicted_static": len(static)})
    if len(sets) == 2:
        a, b = (sets[s] for s in live.LIVE_SOURCES)
        both = a & b
        out.append({"region": region, "satellite": "overlap", "cells_10plus_predicted_static": len(a | b),
                    "static_cells_in_both": len(both),
                    "jaccard": len(both) / len(a | b) if a | b else float("nan"),
                    "share_of_noaa20_static_also_noaa21": len(both) / len(a) if a else float("nan"),
                    "share_of_noaa21_static_also_noaa20": len(both) / len(b) if b else float("nan")})
    return out


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    hist = joblib.load(MODEL_DIR / "best_model_history.joblib", mmap_mode="r")
    inst = joblib.load(MODEL_DIR / "best_model_headline.joblib", mmap_mode="r")
    thr = float(hist["thresholds"][hist.get("default_operating_point", "precision >= 0.90")])
    if "--no-fetch" not in argv:
        key = live.load_key()
        if not key:
            raise SystemExit("No map key found in the environment or .env")
        for region in REGIONS:
            try:
                r = live.refresh(region, key)
                print(f"{region}: {r['requests']} requests, {r['new_rows']:,} detections stored", flush=True)
            except live.FirmsError as e:
                print(f"{region}: failed, {e}", flush=True)
    rows = []
    for region in REGIONS:
        rows += check(region, hist, inst, thr)
    out = pd.DataFrame(rows)
    out.round(4).to_csv(REPORT_DIR / "live_check.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

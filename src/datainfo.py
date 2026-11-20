"""What the Data and features tab shows. Every number is read at run time from the saved reports, the saved
models, the code constants or the live store, so nothing on that tab is typed by hand. A source that is missing
gives None or an empty table, never an invented number."""
import json

import joblib
import numpy as np
import pandas as pd

from . import ingest, live
from .config import GRID_DEG, MODEL_DIR, REPORT_DIR
from .context import detail
from .history_eval import COMPLETE_DAYS, SPLIT_DATE
from .world import SEEDS

SNPP_END = "2 November 2026"          # FIRMS notice: Suomi NPP stopped delivering new data on this day
# NASA's VIIRS type values, and what this project does with each
TYPES = [(0, "presumed vegetation fire", "kept, label fire (0)"), (2, "other static land source", "kept, label static (1)"),
         (1, "active volcano", "dropped"), (3, "offshore detection", "dropped")]


def report(name):
    p = REPORT_DIR / name
    return pd.read_csv(p) if p.exists() else None


def headline():
    p = REPORT_DIR / "metrics_headline.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


# ------------------------------------------------------------------ live store
def live_store(path=None):
    """Rows, regions, sensors, newest detection and the number of regions outside the training data, or None."""
    path = path or ingest.store_path()
    if not path.exists():
        return None
    rows, last_ingest = ingest.store_info(path)
    summary = ingest.store_summary(path)
    regions = sorted(summary["region"].unique())
    return {"rows": int(rows), "groups": int(len(summary)), "regions": regions,
            "untrained": [r for r in regions if not live.is_trained(r)],
            "sensors": sorted(summary["sensor"].unique()),
            "oldest": pd.Timestamp(summary["first"].min()) if len(summary) else None,
            "newest": ingest.latest_detection(path), "last_ingest": last_ingest}


# ------------------------------------------------------------------ sources, labels, regions
def sources(store=None):
    """Rows (source, used for, type field, what it holds) of the data sources."""
    h, inv = headline(), report("world_inventory.csv")
    if h is not None and inv is not None:
        training = (f"{h['rows']:,} detections kept, {h['date_min']} to {h['date_max']}, "
                    f"{inv['region'].nunique()} regions")
    else:
        training = "not available"
    if store:
        feed = (f"{store['rows']:,} stored detections of {', '.join(store['sensors'])}, "
                f"{len(store['regions'])} regions, newest {store['newest']:%Y-%m-%d %H:%M} UTC"
                if store.get("newest") is not None else f"{store['rows']:,} stored detections")
    else:
        feed = "no live store on this machine"
    return pd.DataFrame([
        {"Source": "NASA FIRMS VIIRS Suomi NPP archive, 375 m", "Used for": "training and evaluating the models",
         "Type field": "yes", "Holds": training},
        {"Source": "NASA FIRMS VIIRS NOAA-20 and NOAA-21, near real time", "Used for": "the live view, scored by the saved models",
         "Type field": "no", "Holds": feed}])


def labels():
    """The type values: meaning, what is done with them and how many detections of the archive files have each."""
    inv = report("world_inventory.csv")
    count = {0: None, 2: None, 1: None, 3: None}
    other = None
    if inv is not None:
        count[0], count[2] = int(inv["type0"].sum()), int(inv["type2"].sum())
        other = int(inv["other_types"].sum())            # types 1 and 3 together, as the inventory counts them
    rows = []
    for value, meaning, action in TYPES:
        n = count[value] if value in (0, 2) else None
        rows.append({"Type": value, "Meaning": meaning, "In this project": action,
                     "Detections in the archive files": n if n is not None else
                     (f"{other:,} (types 1 and 3 together)" if other is not None and value == 1 else
                      "counted with type 1" if value == 3 else "")})
    df = pd.DataFrame(rows)
    df["Detections in the archive files"] = [f"{v:,}" if isinstance(v, int) else v
                                             for v in df["Detections in the archive files"]]
    return df


def regions():
    """Each training region with its rows and static share, and a total row."""
    inv = report("world_inventory.csv")
    if inv is None:
        return None
    t = inv[["region", "country", "rows_kept", "static_share"]].copy()
    total = {"region": "All regions", "country": "", "rows_kept": int(t["rows_kept"].sum()),
             "static_share": float((t["rows_kept"] * t["static_share"]).sum() / t["rows_kept"].sum())}
    t = pd.concat([t, pd.DataFrame([total])], ignore_index=True)
    t["static_share"] = t["static_share"].round(4)
    return t.rename(columns={"region": "Region", "country": "Country", "rows_kept": "Detections",
                             "static_share": "Static share"})


# ------------------------------------------------------------------ features
def feature_table(tier):
    """Features of a tier with their plain meaning, read from the saved model and the labels of the detail panel."""
    name = "best_model_headline.joblib" if tier == "instant" else "best_model_history.joblib"
    p = MODEL_DIR / name
    if not p.exists():
        return None
    feats = list(joblib.load(p, mmap_mode="r")["features"])
    return pd.DataFrame({"Feature": feats, "Meaning": [detail.LABELS.get(f, f) for f in feats]})


# ------------------------------------------------------------------ split and checks
def check_rows():
    """Static class PR-AUC of the baselines and the two tiers for the grouped split (seed 42), the temporal split
    and the mean of the eight leave one region out splits, from the saved reports."""
    hist, inst, night = report("world_history.csv"), report("world_instant.csv"), report("daynight_vs_baseline.csv")
    if hist is None or inst is None:
        return None
    is_rule = hist["tier"].str.startswith("rule")

    def pick(frame, mask_split, tier_mask, col="pr_auc"):
        sel = frame[mask_split & tier_mask]
        return float(sel[col].mean()) if len(sel) else np.nan

    out = []
    for label, test in (("Grouped split (seed 42)", lambda s: s == "grouped seed 42"),
                        ("Temporal split", lambda s: s == "temporal"),
                        ("Leave one region out (mean of the regions)", lambda s: s.str.startswith("leave out"))):
        hs, isp = hist["split"], inst["split"]
        row = {"Check": label,
               "Majority baseline": pick(hist, test(hs), hist["tier"] == "history", "majority_pr_auc"),
               "Night baseline": (pick(night, test(night["split"]), night["tier"] == "history", "baseline_pr_auc")
                                  if night is not None else np.nan),
               "One line rule": pick(hist, test(hs), is_rule),
               "Instant tier": pick(inst, test(isp), inst["tier"] == "instant"),
               "History tier": pick(hist, test(hs), hist["tier"] == "history")}
        out.append(row)
    df = pd.DataFrame(out).set_index("Check")
    return df.round(3)


def check_notes():
    """One line for each check, with the numbers that define it taken from the code constants."""
    h = headline()
    held = f"about {h['test_rows'] / h['rows']:.0%} of" if h is not None else "part of"      # the test share, from the report
    return [
        f"Grouped split: {len(SEEDS)} random splits (seeds {', '.join(str(s) for s in SEEDS)}) hold out {held} "
        f"the {GRID_DEG} degree grid cells, so no place is in both the training and the test part.",
        f"Temporal split: train on detections before {SPLIT_DATE:%Y-%m-%d}, test from that day on, on detections "
        f"with at least {COMPLETE_DAYS} days of data before them.",
        "Leave one region out: train on the other regions, test on the one held out, for every region.",
        "Majority baseline: call everything the majority class (fire). Its PR-AUC is the static share.",
        "Night baseline: call every night detection static.",
        "One line rule: static when the cell was seen on at least k distinct days of the previous 30, with k "
        "chosen on the training part.",
        "Thresholds come from out of fold training scores only, never from the test part."]

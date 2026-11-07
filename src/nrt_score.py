"""Score a near real time FIRMS file (no type column) with a saved model.

Run from the project root:
    python -m src.nrt_score --file data/nrt/SUOMI_VIIRS_C2_Global_7d.csv

This is a sanity check, not a score: there is no ground truth.
"""
import argparse

import joblib
import pandas as pd

from .config import MODEL_DIR, REPORT_DIR
from .data import load_raw
from .features import add_features
from .label import add_cell

REPEAT_DAYS = 4

# Rough boxes, first match wins. Boundaries are approximate.
BOXES = [
    ("Oceania", -50, -10, 110, 180),
    ("North America", 15, 75, -170, -50),
    ("South America", -57, 15, -85, -34),
    ("Europe", 35, 72, -25, 45),
    ("Africa", -35, 37, -20, 52),
    ("Asia", -11, 78, 45, 180),
]


def continent(lat, lon):
    for name, a, b, c, d in BOXES:
        if a <= lat <= b and c <= lon <= d:
            return name
    return "Other"


def main(path, model_path, tag):
    bundle = joblib.load(model_path)
    df = add_cell(add_features(load_raw([path])))
    df["p_static"] = bundle["model"].predict_proba(df[bundle["features"]])[:, 1]
    df["pred_static"] = (df["p_static"] >= bundle["threshold"]).astype(int)
    df["continent"] = [continent(a, b) for a, b in zip(df["latitude"], df["longitude"])]
    df["cell_days"] = df.groupby("cell_id")["acq_date"].transform("nunique")
    df["repeat_cell"] = df["cell_days"] >= REPEAT_DAYS

    print("detections", len(df), "dates", df["acq_date"].min().date(), df["acq_date"].max().date())
    print("model", bundle["name"], "threshold", round(bundle["threshold"], 4))
    cont = (df.groupby("continent")["pred_static"].agg(detections="size", predicted_static_share="mean")
            .round(4))
    rep = (df.groupby("repeat_cell")["pred_static"].agg(detections="size", predicted_static_share="mean")
           .round(4))
    rep.index = rep.index.map({True: f"cell seen >= {REPEAT_DAYS} days", False: "other cells"})
    print(cont.to_string()); print(rep.to_string())
    print("overall predicted static share", round(df["pred_static"].mean(), 4))
    cont.to_csv(REPORT_DIR / f"nrt_continent_{tag}.csv")
    rep.to_csv(REPORT_DIR / f"nrt_repeat_{tag}.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="data/nrt/SUOMI_VIIRS_C2_Global_7d.csv")
    ap.add_argument("--model", default=str(MODEL_DIR / "best_model_headline.joblib"))
    ap.add_argument("--tag", default="headline")
    a = ap.parse_args()
    main(a.file, a.model, a.tag)

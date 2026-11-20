"""A small development store built from a FIRMS VIIRS CSV, scored with the code path of the ingest.

    python -m src.devstore samples/sample_viirs.csv        # writes data/dev/store.parquet

Then point the app or the API at it (PowerShell):

    $env:FIRE_STORE_PATH = "data/dev/store.parquet"
    python -m streamlit run app/app.py

Nothing is fetched and data/live is not touched. Rows are split into the boxes of src/live.py the way the
ingest does it (a detection in two boxes belongs to the earlier region), each sensor and region group is
scored by ingest.score_group with the saved models as they are, and the store is written in time order.
git ignores data/dev.
"""
import sys
from pathlib import Path

import pandas as pd

from . import ingest, live, serve
from .data import normalize_columns

DEV_PATH = live.ROOT / "data" / "dev" / "store.parquet"
# satellite codes of FIRMS files and the sensor names of the store
SENSORS = {"N20": "NOAA-20", "N21": "NOAA-21", "N": "S-NPP", "NOAA-20": "NOAA-20", "NOAA-21": "NOAA-21",
           "S-NPP": "S-NPP", "SUOMI NPP": "S-NPP", "1": "NOAA-20"}


def sensor_of(value):
    return SENSORS.get(str(value).strip().upper(), "NOAA-20")


def build(raw, path=None, models=None, now=None, log=print):
    """Score a raw FIRMS frame into a store file. Returns the number of rows written."""
    path = Path(path or DEV_PATH)
    models = models or serve.Models()
    now = now or pd.Timestamp.now(tz="UTC").tz_localize(None)
    raw = normalize_columns(raw)
    sat = raw["satellite"] if "satellite" in raw else pd.Series("N20", index=raw.index)
    raw = raw.assign(_sensor=sat.map(sensor_of))
    parts = []
    for sensor, grp in raw.groupby("_sensor", sort=True):
        for region in ingest.region_names():
            sub = ingest.owned(grp[ingest._inside(grp, live.REGIONS[region])], region)
            if sub.empty:
                continue
            std = ingest.standardize(sub.drop(columns="_sensor"), sensor, region)
            parts.append(ingest.score_group(std, region, sensor, models))
    if not parts:
        log("no detection of the file lies in a region box, nothing written")
        return 0
    df = pd.concat(parts, ignore_index=True).sort_values("timestamp", kind="stable")
    df["ingest_time"] = now
    writer = ingest.StoreWriter(path)
    try:
        writer.add(df.reset_index(drop=True))
        writer.commit()
    except Exception:
        writer.abort()
        raise
    log(f"development store: {len(df):,} rows, {df['sensor'].nunique()} sensors, "
        f"{df['region'].nunique()} regions, {df['acq_date'].min():%Y-%m-%d} to {df['acq_date'].max():%Y-%m-%d}")
    return len(df)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    src = Path(argv[0] if argv else live.ROOT / "samples" / "sample_viirs.csv")
    if not src.exists():
        print("the input file does not exist, pass a FIRMS VIIRS CSV")
        return 1
    build(pd.read_csv(src))
    return 0


if __name__ == "__main__":
    sys.exit(main())

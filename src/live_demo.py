"""Build demo snapshots for the Live map tab from the archive files, so it can be tested offline.

Run from the project root:  python -m src.live_demo

For each training region the last 60 days of detections are kept and all dates are shifted by
a whole number of days so the newest day is yesterday (UTC). acq_time is unchanged. The
snapshot has the same format that src/live.py reads, plus an is_demo flag column. These are
historical 2025 detections made to look recent, for testing only.
"""
from datetime import datetime, timedelta, timezone

import pandas as pd

from . import live
from .config import RAW_DIR
from .data import normalize_columns
from .region_eval import REGIONS


def build(region, path, now=None):
    now = now or datetime.now(timezone.utc)
    raw = normalize_columns(pd.read_csv(path))
    raw = raw.drop(columns=["type"], errors="ignore")           # near real time data has no type
    day = pd.to_datetime(raw["acq_date"]).dt.normalize()
    raw = raw[day > day.max() - pd.Timedelta(days=live.KEEP_DAYS)].copy()
    day = pd.to_datetime(raw["acq_date"]).dt.normalize()
    yesterday = pd.Timestamp(now.date()) - pd.Timedelta(days=1)
    shift = yesterday - day.max()                               # whole days
    raw["acq_date"] = (day + shift).dt.strftime("%Y-%m-%d")
    live.write_snapshot(region, raw, fetched_at=now, demo=True)
    return len(raw), int(shift.days)


def main():
    for fname, region in REGIONS.items():
        rows, shift = build(region, RAW_DIR / fname)
        print(f"{region}: {rows:,} detections, dates shifted by {shift} days -> {live.snapshot_path(region)}")


if __name__ == "__main__":
    main()

"""Warm the caches of the Earth view, so the first click and the land cover filter never wait.

1. Land cover table: every 0.01 degree cell that has a detection in the live store in the last 7 days is looked up in
   the table (data/cache/landtable.parquet); the missing cells are read from WorldCover in batches grouped by tile.
2. Population grids: the WorldPop 1 km grids of the 20 countries with most detections in those 7 days are downloaded
   (data/cache/worldpop), one after the other. A grid above 120 MB is skipped, as in the app.

Run from the project root:  python scripts/warm_cache.py        (it can be stopped and run again, it resumes)
It prints only counts and times, never an address or a key.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import ingest, live  # noqa: E402
from src.context import landtable, people  # noqa: E402

DAYS = 7
COUNTRIES = 20
REPORT_EVERY_S = 30


def stamp(t0):
    return f"{time.time() - t0:7.0f} s"


def main():
    t0 = time.time()
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    df = ingest.read_recent(since=now - pd.Timedelta(days=DAYS), columns=["latitude", "longitude", "cell_id"])
    if df is None or df.empty:
        print("the live store has no detection in the last 7 days")
        return 1
    print(f"{stamp(t0)}  detections in the last {DAYS} days: {len(df):,}")

    # 1. the land cover table
    ids = landtable.cell_ids(df["latitude"].to_numpy(float), df["longitude"].to_numpy(float))
    cells = np.unique(ids)
    table = landtable.get()
    known = int((table.lookup(cells) != landtable.UNKNOWN).sum())
    print(f"{stamp(t0)}  cells with a detection: {len(cells):,}, already in the table: {known:,}, "
          f"to read: {len(cells) - known:,}")
    table.request(cells)
    last = 0.0
    while table.running():
        time.sleep(1)
        if time.time() - last > REPORT_EVERY_S:
            last = time.time()
            print(f"{stamp(t0)}  cells left to read: {table.remaining():,}, table size: {len(table):,}")
    table.wait(60)
    after = table.lookup(cells)
    print(f"{stamp(t0)}  land cover table: {int((after != landtable.UNKNOWN).sum()):,} of {len(cells):,} cells known, "
          f"{int((after == landtable.NOT_READ).sum()):,} not read, read in this run: {table.read_count:,}"
          + (f", stopped: {table.error}" if table.error else ""))

    # 2. the population grids of the countries with most detections
    cell_rows = df.drop_duplicates("cell_id")[["cell_id", "latitude", "longitude"]]
    counts = df.groupby("cell_id").size().rename("n").reset_index()
    geo = live.add_geo(cell_rows.merge(counts, on="cell_id"))
    top = geo.groupby("cc", observed=True)["n"].sum().sort_values(ascending=False).head(COUNTRIES)
    todo = [c for c in top.index if people.iso3_of(c) and not people.raster_file(people.iso3_of(c)).exists()]
    print(f"{stamp(t0)}  countries with most detections: {len(top)}, grids to download: {len(todo)}")
    got = 0
    for cc in todo:
        t1 = time.time()
        th = people.warm(cc)
        if th is not None:
            th.join()
        ok = people.raster_file(people.iso3_of(cc)).exists()
        got += int(ok)
        print(f"{stamp(t0)}  grid {got} of {len(todo)} {'done' if ok else 'not available'} in {time.time() - t1:.0f} s")
    print(f"{stamp(t0)}  finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())

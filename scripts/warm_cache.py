"""Warm the caches of the Earth view, so the first click and the land cover filter never wait.

1. Land cover table: every 0.01 degree cell that has a detection in the live store in the last DAYS days (default 7) is
   looked up in the table (data/cache/landtable.parquet); the missing cells are read from WorldCover in batches grouped
   by tile.
2. Population grids: the WorldPop 1 km grids of the countries with most detections in those days (the 20 with most, or
   the countries named with --country) are downloaded (data/cache/worldpop), one after the other. A grid above 120 MB
   is skipped, as in the app.

Run from the project root:
    python scripts/warm_cache.py                                   the whole store, the last 7 days
    python scripts/warm_cache.py --country Pakistan India --days 1  two countries, one day (minutes)

--country takes one or more country names (or two letter codes), --days a number of days (default 7, a fraction such as
0.5 is fine). The cell count and a time estimate are printed before the work starts. The script can be stopped and
run again: cells already in the table and grids already on disk are skipped, so it resumes where it stopped.
It prints only counts and times, never an address or a key.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import geo, ingest, live  # noqa: E402
from src.context import landtable, people  # noqa: E402

COUNTRIES = 20
REPORT_EVERY_S = 30
CELLS_PER_S = (8, 13)           # measured: 8 to 13 cells read per second (about 2 tiles per second)
GRID_MB_PER_S = 0.17            # measured: a 5.7 MB grid took about 35 s


def stamp(t0):
    return f"{time.time() - t0:7.0f} s"


def duration(seconds):
    seconds = int(round(seconds))
    if seconds < 90:
        return f"{seconds} seconds"
    if seconds < 5400:
        return f"{seconds / 60:.0f} minutes"
    return f"{seconds / 3600:.1f} hours"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Warm the land cover table and the population grids.")
    p.add_argument("--country", nargs="+", metavar="NAME", default=None,
                   help="only the detections of these countries (names or two letter codes), for example Pakistan India")
    p.add_argument("--days", type=float, default=7.0, help="days of the live store to warm (default 7)")
    return p.parse_args(argv)


def resolve_countries(names):
    """(codes, unknown names) for country names or two letter codes, matched without regard to case."""
    by_name = {n.lower(): c for c, n in geo.country_names().items()}
    codes, unknown = [], []
    for raw in names:
        key = raw.strip().lower()
        code = by_name.get(key) or (raw.strip().upper() if raw.strip().upper() in geo.country_names() else None)
        (codes if code else unknown).append(code or raw)
    return list(dict.fromkeys(codes)), unknown


def main(argv=None):
    args = parse_args(argv)
    if args.days <= 0:
        print("--days must be above 0")
        return 2
    codes = None
    if args.country:
        codes, unknown = resolve_countries(args.country)
        if unknown:
            print(f"unknown country: {', '.join(unknown)} (use the English name or the two letter code)")
            return 2
    t0 = time.time()
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    df = ingest.read_recent(since=now - pd.Timedelta(days=args.days), columns=["latitude", "longitude", "cell_id"])
    if df is None or df.empty:
        print(f"the live store has no detection in the last {args.days:g} days")
        return 1
    print(f"{stamp(t0)}  detections in the last {args.days:g} days: {len(df):,}")

    # which country each detection is in (from its cell, cached), to count per country and to filter
    cell_rows = df.drop_duplicates("cell_id")[["cell_id", "latitude", "longitude"]]
    counts = df.groupby("cell_id").size().rename("n").reset_index()
    geo_rows = live.add_geo(cell_rows.merge(counts, on="cell_id"))
    if codes is not None:
        keep = geo_rows["cc"].astype(str).isin(codes)
        df = df[df["cell_id"].isin(geo_rows.loc[keep, "cell_id"])]
        geo_rows = geo_rows[keep]
        if df.empty:
            print(f"no detection of {', '.join(args.country)} in the last {args.days:g} days")
            return 1
        print(f"{stamp(t0)}  detections of {', '.join(args.country)}: {len(df):,}")
    per_country = geo_rows.groupby("cc", observed=True)["n"].sum().sort_values(ascending=False)
    top = list(per_country.index[:COUNTRIES])

    # 1. the land cover table: the count and the estimate come first
    cells = np.unique(landtable.cell_ids(df["latitude"].to_numpy(float), df["longitude"].to_numpy(float)))
    table = landtable.get()
    known = int((table.lookup(cells) != landtable.UNKNOWN).sum())
    todo_cells = len(cells) - known
    grids = [c for c in top if people.iso3_of(c) and not people.raster_file(people.iso3_of(c)).exists()]
    lo, hi = CELLS_PER_S
    print(f"{stamp(t0)}  cells with a detection: {len(cells):,}, already in the table: {known:,}, to read: {todo_cells:,}")
    print(f"{stamp(t0)}  population grids to download: {len(grids)} of {len(top)} countries "
          f"(2 to 53 MB each, about {duration(30 * len(grids))} in all at the speed measured)")
    print(f"{stamp(t0)}  estimate: the cells take {duration(todo_cells / hi)} to {duration(todo_cells / lo)} "
          f"(8 to 13 cells a second), the grids {duration(30 * len(grids))}; it can be stopped and run again")
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

    # 2. the population grids
    got = 0
    for cc in grids:
        t1 = time.time()
        th = people.warm(cc)
        if th is not None:
            th.join()
        ok = people.raster_file(people.iso3_of(cc)).exists()
        got += int(ok)
        print(f"{stamp(t0)}  grid {got} of {len(grids)} {'done' if ok else 'not available'} in {time.time() - t1:.0f} s")
    print(f"{stamp(t0)}  finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())

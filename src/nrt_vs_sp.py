"""NRT against SP: why the instant tier flags many live near real time detections as static.

Run from the project root:  python -m src.nrt_vs_sp [--download] [--analyse]

Two controlled comparisons with NOAA-21, the three region boxes of src/noaa20.py and the same
country cut (Pakistan, Indonesia, USA):
  A  same period, different product: VIIRS_NOAA21_SP against VIIRS_NOAA21_NRT, 2026-06-01 to 2026-07-31
  B  same season, different year and product: VIIRS_NOAA21_SP 2025-09-11 to 2025-11-09 against the
     live NRT snapshot of 2026-09-11 to 2026-11-09 (data/live)

For every set and region it reports row counts, feature distributions, the confidence categories,
and the static share of the instant tier (saved threshold) and of the history tier (default operating
point, history built inside each file). For pair A it also matches detections of the two products
(within 0.006 degrees and 30 minutes) and compares features and scores on the matched pairs.

The map key is read from .env and is never printed. Request errors print only the exception class and
the HTTP status (src/noaa20.fetch). Files are saved under data/noaa20/nrtcheck/, which git ignores.
"""
import sys
from datetime import date

import numpy as np
import pandas as pd

from . import noaa20
from .config import REPORT_DIR

SETS = {"A_SP": ("VIIRS_NOAA21_SP", date(2026, 6, 1), date(2026, 7, 31)),
        "A_NRT": ("VIIRS_NOAA21_NRT", date(2026, 6, 1), date(2026, 7, 31)),
        "B_SP": ("VIIRS_NOAA21_SP", date(2025, 9, 11), date(2025, 11, 9))}
DIR = noaa20.OUT_DIR / "nrtcheck"


def set_path(tag, region, day, days):
    return DIR / tag / region.replace(" ", "_").lower() / f"{day:%Y-%m-%d}_{days}.csv"


def download(key, tags=None, regions=None, fetcher=noaa20.fetch, sleep=None, log=print):
    """Fetch every missing chunk of the sets. Returns the number of calls made. Resumes after a stop."""
    import time
    sleep = sleep or time.sleep
    calls = 0
    for tag in (tags or SETS):
        source, start, end = SETS[tag]
        for region in (regions or noaa20.REGION_BOXES):
            for day, n in noaa20.chunks(start, end):
                path = set_path(tag, region, day, n)
                if path.exists():
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    text = fetcher(noaa20.REGION_BOXES[region], day, n, key, source=source)
                except noaa20.SafeError as e:
                    log(f"{tag} {region} {day}: failed, {e}")
                    sleep(noaa20.PAUSE)
                    continue
                path.write_text(text, encoding="utf-8")
                calls += 1
                sleep(noaa20.PAUSE)
    return calls


def missing(tags=None, regions=None):
    out = []
    for tag in (tags or SETS):
        _, start, end = SETS[tag]
        for region in (regions or noaa20.REGION_BOXES):
            out += [(tag, region, d, n) for d, n in noaa20.chunks(start, end) if not set_path(tag, region, d, n).exists()]
    return out


def load_set(tag, region):
    """All chunks of a set and region as one frame."""
    import io
    _, start, end = SETS[tag]
    frames = []
    for day, n in noaa20.chunks(start, end):
        p = set_path(tag, region, day, n)
        if p.exists():
            t = p.read_text(encoding="utf-8")
            if t.startswith("latitude"):
                frames.append(pd.read_csv(io.StringIO(t)))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--download" in argv:
        key = noaa20.read_key()
        if not key:
            raise SystemExit("No map key found in the environment or .env")
        regions = [a for a in argv if a in noaa20.REGION_BOXES] or None
        print(f"{len(missing(regions=regions))} chunks to fetch", flush=True)
        for attempt in range(3):
            calls = download(key, regions=regions)
            print(f"round {attempt + 1}: {calls} calls, {len(missing(regions=regions))} missing", flush=True)
            if not missing(regions=regions):
                break
    if "--analyse" in argv:
        from . import nrt_analysis
        nrt_analysis.main()


if __name__ == "__main__":
    main()

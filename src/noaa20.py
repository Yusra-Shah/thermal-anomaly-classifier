"""Download NOAA-20 VIIRS standard processing detections for the three training regions.

Run from the project root:  python -m src.noaa20

Source VIIRS_NOAA20_SP of the FIRMS area API, 2025-01-01 to 2025-06-30, in chunks of at most 5
days (the documented limit), one box per region. Every chunk is saved on its own in
data/noaa20/chunks/<region>/, so an interrupted run resumes where it stopped. The type column
is kept. The chunks are then cut to the three countries of the Suomi NPP files (the archive
download was by country, the area API is by box) with the Natural Earth polygons in data/geo,
and written to data/noaa20/noaa20_<region>.csv.

The map key is read from the environment or .env. It is part of every request URL, so every
request error is caught and only the exception class and the HTTP status are printed. No URL,
header or traceback from the requests library is ever shown.
"""
import io
import os
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT

SOURCE = "VIIRS_NOAA20_SP"
START, END = date(2025, 1, 1), date(2025, 6, 30)
MAX_DAY_RANGE = 5
PAUSE = 2.0                               # seconds between calls
OUT_DIR = ROOT / "data" / "noaa20"
BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
REGION_BOXES = {                          # west, south, east, north, as in src/live.py
    "South Asia": (61.0, 24.0, 76.0, 36.0),
    "Southeast Asia": (95.0, -11.0, 141.0, 6.0),
    "North and Central America": (-177.0, 19.0, -67.0, 71.0),
}
REGION_COUNTRY = {"South Asia": "PK", "Southeast Asia": "ID", "North and Central America": "US"}


class SafeError(Exception):
    """An error whose message is safe to print: a class name and a status, nothing else."""


def read_key():
    """Map key from the environment or .env (NAME=value, or a bare key on one line). Never printed."""
    key = os.environ.get("FIRMS_MAP_KEY", "").strip()
    env = ROOT / ".env"
    if not key and env.exists():
        lines = [x.strip() for x in env.read_text().splitlines() if x.strip() and not x.startswith("#")]
        for x in lines:
            if x.startswith("FIRMS_MAP_KEY="):
                key = x.split("=", 1)[1].strip().strip("\"'")
        if not key and len(lines) == 1 and "=" not in lines[0]:
            key = lines[0]
    return key or None


def chunks(start=START, end=END, size=MAX_DAY_RANGE):
    """(first day, number of days) pieces that cover every day from start to end exactly once."""
    out, day = [], start
    while day <= end:
        n = min(size, (end - day).days + 1)
        out.append((day, n))
        day += timedelta(days=n)
    return out


def fetch(box, day, days, key, session=None, retries=3, sleep=time.sleep):
    """CSV text of one chunk. Raises SafeError with the exception class and the HTTP status only."""
    import requests
    get = (session or requests).get
    url = f"{BASE}/{key}/{SOURCE}/{','.join(str(v) for v in box)}/{days}/{day:%Y-%m-%d}"
    status, kind = None, ""
    for attempt in range(retries + 1):
        try:
            r = get(url, timeout=90)
            status = r.status_code
            if status == 200:
                text = r.text
                if text.startswith("latitude") or not text.strip():
                    return text
                kind = "NotCsv"
                break
            kind = "HTTPError"
            if status not in (429, 500, 502, 503, 504):
                break
        except Exception as e:                   # the message of a requests error contains the URL
            kind, status = type(e).__name__, None
        if attempt < retries:
            sleep(30 if status == 429 else 5 * (attempt + 1))
    raise SafeError(f"{kind}, HTTP status {status}")


def chunk_path(region, day, days):
    return OUT_DIR / "chunks" / region.replace(" ", "_").lower() / f"{day:%Y-%m-%d}_{days}.csv"


def download(key, fetcher=fetch, sleep=time.sleep, log=print, regions=None, reverse=False):
    """Fetch every missing chunk of the regions. Returns (calls made, rows saved). Several runs can
    share the work: a run with reverse=True starts from the last chunk, and chunks already on disk
    are skipped."""
    calls = rows = 0
    plan = [(r, d, n) for r in (regions or REGION_BOXES) for d, n in chunks()]
    if reverse:
        plan.reverse()
    for i, (region, day, n) in enumerate(plan, 1):
        path = chunk_path(region, day, n)
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            text = fetcher(REGION_BOXES[region], day, n, key)
        except SafeError as e:
            log(f"chunk {i}/{len(plan)} {region} {day}: failed, {e}")
            sleep(PAUSE)
            continue
        path.write_text(text, encoding="utf-8")
        calls += 1
        rows += max(len(text.strip().splitlines()) - 1, 0)
        if calls % 10 == 0:
            log(f"chunk {i}/{len(plan)}: {calls} calls, {rows:,} rows so far")
        sleep(PAUSE)
    return calls, rows


def missing_chunks():
    return [(r, d, n) for r in REGION_BOXES for d, n in chunks() if not chunk_path(r, d, n).exists()]


def in_country(lon, lat, code):
    """Boolean array: points inside the Natural Earth polygon of a country."""
    from shapely import contains_xy, prepare
    from shapely.geometry import shape

    from . import geo
    feat = geo.country_feature(code)
    geom = shape(feat["geometry"])
    prepare(geom)
    return np.asarray(contains_xy(geom, np.asarray(lon, float), np.asarray(lat, float)))


def combine(region):
    """All chunks of a region as one frame (the type column kept)."""
    frames = []
    for d, n in chunks():
        p = chunk_path(region, d, n)
        if p.exists() and p.stat().st_size > 0:
            t = p.read_text(encoding="utf-8")
            if t.startswith("latitude"):
                frames.append(pd.read_csv(io.StringIO(t)))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_files(log=print):
    """Cut every region to its country and write data/noaa20/noaa20_<region>.csv."""
    stats = {}
    for region, code in REGION_COUNTRY.items():
        df = combine(region)
        if df.empty:
            continue
        inside = in_country(df["longitude"], df["latitude"], code)
        kept = df[inside].drop_duplicates(subset=["latitude", "longitude", "acq_date", "acq_time"])
        name = f"noaa20_{region.replace(' ', '_').lower()}.csv"
        kept.to_csv(OUT_DIR / name, index=False)
        stats[region] = {"box_rows": len(df), "country": code, "country_rows": len(kept),
                         "static_type2": int((kept["type"] == 2).sum()), "file": name}
        log(f"{region}: {len(df):,} rows in the box, {len(kept):,} inside {code}, file {name}")
    return stats


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", nargs="*", default=None, help="only these regions")
    ap.add_argument("--reverse", action="store_true", help="start from the last chunk")
    ap.add_argument("--no-build", action="store_true", help="download only, do not write the country files")
    a = ap.parse_args(argv)
    key = read_key()
    if not key:
        raise SystemExit("No map key found in the environment or .env")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"{len(missing_chunks())} of {len(REGION_BOXES) * len(chunks())} chunks to fetch", flush=True)
    for attempt in range(3):                                   # failed chunks are retried in later rounds
        calls, rows = download(key, regions=a.regions, reverse=a.reverse)
        print(f"round {attempt + 1}: {calls} calls, {rows:,} rows", flush=True)
        if not [m for m in missing_chunks() if a.regions is None or m[0] in a.regions]:
            break
        time.sleep(60)
    if a.no_build:
        return
    left = missing_chunks()
    if left:
        print(f"{len(left)} chunks are still missing, run again to resume", flush=True)
        return
    build_files()


if __name__ == "__main__":
    main()

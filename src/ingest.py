"""Incremental live store: fetch NOAA-20 and NOAA-21 near real time detections, score them, keep 90 days.

Run from the project root:
    python -m src.ingest --once
    python -m src.ingest --loop --every-minutes 180
    python -m src.ingest --import-live          # one off: take the rolling stores of src/live.py into the store

One store, data/live/store.parquet (git ignores data/live), written atomically (temporary file, then
replace). Rows are unique on sensor, latitude, longitude, acq_date and acq_time, so running twice in a
row adds nothing. Only the last 90 days are kept, which is the window the history features need. Every
group of one sensor and one region is scored again when it gains or loses rows, because the history of
a detection depends on the rows before it. The saved models, thresholds, calibrators and bands are used
as they are. The store holds the instant and history features, both scores, the calibrated
probabilities, the label, the default and strict band calls and the ingest time.

Calls use the fetch logic of src/live.py with a day range of 1 to 5, a short wait between calls and a
sliding window budget below the limit of 5000 transactions per 10 minutes. The map key comes from
FIRMS_MAP_KEY (environment or .env). It is never printed. A failed call prints only the exception class
and the HTTP status. The models are not retrained from this data: it has no type flag.
"""
import argparse
import os
import re
import time
from collections import deque
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from . import live, serve
from .data import REQUIRED, normalize_columns

KEEP_DAYS = 90
REFETCH_DAYS = 3
KEY_COLS = ["sensor", "latitude", "longitude", "acq_date", "acq_time"]
RAW_COLS = ["sensor", "region", "latitude", "longitude", "acq_date", "acq_time", "bright_ti4", "bright_ti5",
            "frp", "scan", "track", "confidence", "daynight"]
LIMIT = 5000                    # transactions per window
WINDOW = 600                    # seconds
BUDGET = 4500                   # stay below the limit
COST_PER_DAY = 4                # the dearest box costs 4 per day of range
SCORE_COLS = ["p_instant", "pc_instant", "label_instant", "instant_default", "instant_strict",
              "p_history", "pc_history", "label_history", "history_default", "history_strict"]


def store_path():
    return live.LIVE_DIR / "store.parquet"


# ------------------------------------------------------------------ safe errors
def safe_message(e):
    """Exception class and HTTP status only. Never the text of the error, which can hold a URL or the key."""
    text = str(e)
    m = re.search(r"HTTP (\d{3})", text)
    status = m.group(1) if m else "none"
    inner = re.search(r"network error \((\w+)\)", text)
    cls = inner.group(1) if inner else type(e).__name__
    return f"{cls}, HTTP status {status}"


class RateLimiter:
    """Sliding window budget: waits when the next call would exceed BUDGET transactions in WINDOW seconds."""

    def __init__(self, budget=BUDGET, window=WINDOW, clock=time.monotonic, sleep=time.sleep):
        self.budget, self.window, self.clock, self.sleep = budget, window, clock, sleep
        self.calls = deque()

    def wait(self, cost):
        now = self.clock()
        while self.calls and now - self.calls[0][0] >= self.window:
            self.calls.popleft()
        while self.calls and sum(c for _, c in self.calls) + cost > self.budget:
            self.sleep(max(self.calls[0][0] + self.window - now, 0.0) + 0.1)
            now = self.clock()
            while self.calls and now - self.calls[0][0] >= self.window:
                self.calls.popleft()
        self.calls.append((self.clock(), cost))


# ------------------------------------------------------------------ store io
def read_store(path=None):
    path = path or store_path()
    return pd.read_parquet(path) if path.exists() else None


def write_store(df, path=None):
    """Atomic write: a temporary file in the same folder, then replace. A failure leaves the old file."""
    path = path or store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def standardize(raw, sensor, region):
    """Raw FIRMS rows with the columns of the store, dates as days and acq_time as an integer."""
    df = normalize_columns(raw)
    df["sensor"], df["region"] = sensor, region
    df["acq_date"] = pd.to_datetime(df["acq_date"]).dt.normalize()
    df["acq_time"] = pd.to_numeric(df["acq_time"]).astype(int)
    df["confidence"] = df["confidence"].astype(str)
    df["daynight"] = df["daynight"].astype(str)
    return df[RAW_COLS].reset_index(drop=True)


def region_names():
    return [r for r in live.REGIONS if r in live.TRAINED]


# ------------------------------------------------------------------ scoring
def score_group(raw, region, sensor, models):
    """Features, history (inside this sensor and region only) and both tiers for one group."""
    df = live.prepare(raw.assign(satellite_name=sensor), region)
    df["sensor"], df["region"] = sensor, region
    df = serve.score_frame(models, df)
    feats = list(dict.fromkeys(models.instant_features + models.history_features))
    cols = ["sensor", "region", "latitude", "longitude", "acq_date", "acq_time", "timestamp", "cell_id",
            "confidence", "daynight"] + feats + SCORE_COLS
    out = df[list(dict.fromkeys(cols))].copy()
    for f in models.history_only + ["bright_diff", "pixel_area", "frp_per_area"]:
        if f in out.columns:
            out[f] = out[f].astype(np.float32)
    return out


# ------------------------------------------------------------------ one run
def fetch_new(key, old, today, fetch, limiter, sleep, pause, log):
    """Fetch what is missing for every sensor and region. Returns (frames, fetched per sensor, failures)."""
    frames, fetched, failures = [], {s: 0 for s in live.LIVE_SOURCES}, 0
    cutoff = today - timedelta(days=KEEP_DAYS - 1)
    for sensor, source in live.LIVE_SOURCES.items():
        for region in region_names():
            have = old[(old["sensor"] == sensor) & (old["region"] == region)] if old is not None else None
            start = cutoff
            if have is not None and len(have):
                start = max(cutoff, have["acq_date"].max() - timedelta(days=REFETCH_DAYS - 1))
            days = list(pd.date_range(start, today, freq="D"))
            for run in live._chunks(days):
                limiter.wait(len(run) * COST_PER_DAY)
                try:
                    raw = fetch(key, live.REGIONS[region], run[0].to_pydatetime(), len(run), source)
                except Exception as e:                       # only the class and the status are shown
                    log(f"{sensor} {region}: failed, {safe_message(e)}, the rest of this part waits for the next run")
                    failures += 1
                    sleep(pause)
                    break
                if len(raw):
                    df = standardize(raw, sensor, region)
                    frames.append(df)
                    fetched[sensor] += len(df)
                sleep(pause)
    return frames, fetched, failures


def run_once(key=None, today=None, fetch=live.fetch_chunk, models=None, extra=None, sleep=time.sleep,
             pause=live.PAUSE, limiter=None, log=print, now=None, path=None):
    """One ingest. fetch=None skips the network (used with extra, a list of standardized raw frames).
    Returns a dict with rows fetched, added and kept per sensor."""
    path = path or store_path()
    today = pd.Timestamp(today or datetime.now(timezone.utc).date()).normalize()
    now = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC").tz_localize(None)
    old = read_store(path)
    limiter = limiter or RateLimiter(sleep=sleep)
    frames, fetched, failures = [], {s: 0 for s in live.LIVE_SOURCES}, 0
    if fetch is not None:
        frames, fetched, failures = fetch_new(key, old, today, fetch, limiter, sleep, pause, log)
    frames = frames + list(extra or [])
    cutoff = today - timedelta(days=KEEP_DAYS - 1)

    old_raw = old[RAW_COLS] if old is not None else pd.DataFrame(columns=RAW_COLS)
    new_raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=RAW_COLS)
    new_raw = new_raw.drop_duplicates(subset=KEY_COLS)
    both = pd.concat([old_raw, new_raw], ignore_index=True)
    both["acq_date"] = pd.to_datetime(both["acq_date"])
    both = both.drop_duplicates(subset=KEY_COLS, keep="first")           # the stored row wins
    both = both[both["acq_date"] >= cutoff].reset_index(drop=True)
    if len(old_raw):
        seen = old_raw[KEY_COLS].drop_duplicates().assign(_old=True)
        flag = both[KEY_COLS].merge(seen, on=KEY_COLS, how="left")["_old"].fillna(False).astype(bool).to_numpy()
    else:
        flag = np.zeros(len(both), dtype=bool)
    added = both[~flag]
    removed = old_raw[old_raw["acq_date"] < cutoff] if len(old_raw) else old_raw
    counts = {s: {"fetched": fetched[s], "added": int((added["sensor"] == s).sum()), "kept": 0}
              for s in live.LIVE_SOURCES}

    if not len(added) and not len(removed):
        if old is not None:
            for s in counts:
                counts[s]["kept"] = int((old["sensor"] == s).sum())
        _print(counts, log, failures)
        return {"counts": counts, "failures": failures, "changed": False}

    models = models or serve.Models()
    touched = set(map(tuple, added[["sensor", "region"]].drop_duplicates().itertuples(index=False, name=None)))
    touched |= set(map(tuple, removed[["sensor", "region"]].drop_duplicates().itertuples(index=False, name=None)))
    parts = []
    times = {}
    if old is not None:
        times = old.set_index(KEY_COLS)["ingest_time"]
    for (sensor, region), g in both.groupby(["sensor", "region"], sort=True):
        if (sensor, region) in touched:
            scored = score_group(g, region, sensor, models)
            if len(times):
                t = scored.join(times.rename("ingest_time"), on=KEY_COLS)["ingest_time"]
                scored["ingest_time"] = t.fillna(now).to_numpy()
            else:
                scored["ingest_time"] = now
            parts.append(scored)
        else:
            parts.append(old[(old["sensor"] == sensor) & (old["region"] == region)])
    store = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if len(store):
        store = store.sort_values(["sensor", "region", "timestamp", "latitude", "longitude"]).reset_index(drop=True)
        write_store(store, path)
    for s in counts:
        counts[s]["kept"] = int((store["sensor"] == s).sum()) if len(store) else 0
    _print(counts, log, failures)
    return {"counts": counts, "failures": failures, "changed": True}


def _print(counts, log, failures):
    for s, c in counts.items():
        log(f"{s}: fetched {c['fetched']:,}, added {c['added']:,}, kept {c['kept']:,}")
    if failures:
        log(f"failed calls: {failures}")


def import_live(models=None, log=print):
    """One off: put the rolling stores of src/live.py (60 days per sensor and region) into the store."""
    extra = []
    for region in region_names():
        raw = live.load_store(region)
        if raw is None:
            continue
        for sensor, g in raw.groupby("satellite_name"):
            extra.append(standardize(g.drop(columns=["satellite_name"]), sensor, region))
    return run_once(fetch=None, extra=extra, models=models, log=log)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Incremental live store")
    ap.add_argument("--once", action="store_true", help="one ingest, then stop")
    ap.add_argument("--loop", action="store_true", help="ingest again and again")
    ap.add_argument("--every-minutes", type=float, default=180.0, help="wait between ingests in a loop")
    ap.add_argument("--import-live", action="store_true", help="one off: take the rolling stores of src/live.py")
    a = ap.parse_args(argv)
    if a.import_live:
        import_live()
        return
    if not (a.once or a.loop):
        ap.error("give --once or --loop")
    key = live.load_key()
    if not key:
        raise SystemExit("No map key found in the environment or .env")
    models = serve.Models()
    while True:
        try:
            run_once(key, models=models)
        except Exception as e:                               # a failed round must not stop the loop
            print(f"ingest failed: {safe_message(e)}", flush=True)
        if not a.loop:
            break
        time.sleep(a.every_minutes * 60)


if __name__ == "__main__":
    main()

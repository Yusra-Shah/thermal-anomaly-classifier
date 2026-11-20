"""Incremental live store: fetch NOAA-20 and NOAA-21 near real time detections, score them, keep 90 days.

Run from the project root:
    python -m src.ingest --once                 # every region; --regions trained for the eight trained ones
    python -m src.ingest --loop --every-minutes 180
    python -m src.ingest --import-live          # one off: take the rolling stores of src/live.py into the store
    python -m src.ingest --rebuild-score        # score every stored row again with the saved models, no fetch
    python -m src.ingest --measure-cost         # transaction cost of one call, read from the key status counter

One store, data/live/store.parquet (git ignores data/live), written atomically (temporary file, then
replace). Rows are unique on sensor, latitude, longitude, acq_date and acq_time, so running twice in a
row adds nothing. Only the last 90 days are kept, which is the window the history features need.

Memory is bounded by the largest group of one sensor and one region, never by the whole store. The store is
read and written one group at a time (the output is streamed as row groups of at most 200000 rows sorted by
time, so a reader can skip old data). A group is scored again when it gains or loses rows, because the history
of a detection depends on the rows before it; other groups are copied as they are. The saved models,
thresholds, calibrators and bands are used as they are. The store holds the instant and history features, both
scores, the calibrated probabilities, the label, the default and strict band calls and the ingest time.

Every box of src/live.py is fetched (the eight trained regions and the continent boxes of the rest of the
land). A detection that lies in two boxes is stored once, in the earlier region.

Fetched days are first written to data/live/ingest_state/staged, one file per sensor, region and day, and the
days merged into the store are listed in data/live/ingest_state/merged.txt. A run that stops half way keeps
what it fetched, and the next run asks only for the days that are missing plus the latest three.

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
from pathlib import Path

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
COST_PER_DAY = 4                # assumed dearest cost per day of range, for the budget check
COST_PER_CALL = 4               # transactions per day of range of one call, measured on the largest box (4 for 1 day, 20 for 5 days)
ROW_GROUP = 200_000             # rows per row group of the store
LIMIT_COMMIT_MB = 4000          # stop before a step would pass this much commit
BYTES_PER_ROW = 1100            # measured peak bytes per row of one group while it is scored (1.05M rows added 1.1 GB)
_DEFAULT = object()
SCORE_COLS = ["p_instant", "pc_instant", "label_instant", "instant_default", "instant_strict",
              "p_history", "pc_history", "label_history", "history_default", "history_strict"]


STORE_ENV = "FIRE_STORE_PATH"


def store_path():
    """The one place the store path comes from: FIRE_STORE_PATH when it is set (for example
    data/dev/store.parquet for the development store, relative to the project root), else the live store."""
    custom = os.environ.get(STORE_ENV, "").strip()
    if custom:
        p = Path(custom)
        return p if p.is_absolute() else live.ROOT / p
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


class StoreWriter:
    """Streams groups into a temporary file as row groups of at most ROW_GROUP rows, then replaces the store
    in one step. Nothing is held except the group being written."""

    def __init__(self, path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.tmp = path.with_name(path.name + ".tmp")
        self.writer = None
        self.schema = None

    def add(self, data):
        import pyarrow as pa
        import pyarrow.parquet as pq
        table = data if isinstance(data, pa.Table) else pa.Table.from_pandas(data, preserve_index=False)
        if self.writer is None:
            self.schema = table.schema
            self.writer = pq.ParquetWriter(self.tmp, self.schema)
        else:
            table = table.select(self.schema.names).cast(self.schema)
        self.writer.write_table(table, row_group_size=ROW_GROUP)

    def size(self):
        return self.tmp.stat().st_size if self.tmp.exists() else 0

    def commit(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None
            os.replace(self.tmp, self.path)

    def abort(self):
        if self.writer is not None:
            try:
                self.writer.close()
            except Exception:
                pass
            self.writer = None
        if self.tmp.exists():
            self.tmp.unlink()


def store_summary(path=None):
    """One row per sensor and region of the store: rows, first and last acquisition date. Three columns are
    read, so this is cheap whatever the size of the store."""
    path = path or store_path()
    cols = ["sensor", "region", "rows", "first", "last"]
    if not path.exists():
        return pd.DataFrame(columns=cols)
    import pyarrow.parquet as pq
    t = pq.read_table(path, columns=["sensor", "region", "acq_date"], read_dictionary=["sensor", "region"])
    df = t.to_pandas()
    df["sensor"], df["region"] = df["sensor"].astype(str), df["region"].astype(str)
    g = df.groupby(["sensor", "region"], sort=True)["acq_date"]
    out = pd.DataFrame({"rows": g.size(), "first": g.min(), "last": g.max()}).reset_index()
    return out[cols]


def store_info(path=None):
    """(rows, last ingest time) of the store from the file metadata, or (0, None)."""
    path = path or store_path()
    if not path.exists():
        return 0, None
    import pyarrow.parquet as pq
    meta = pq.ParquetFile(path).metadata
    idx = meta.schema.to_arrow_schema().get_field_index("ingest_time")
    last = None
    for i in range(meta.num_row_groups):
        st = meta.row_group(i).column(idx).statistics
        if st is not None and st.has_min_max and (last is None or st.max > last):
            last = st.max
    return meta.num_rows, (pd.Timestamp(last) if last is not None else None)


def read_recent(path=None, since=None, columns=None, sensor=None, region=None, regions=None, where=None):
    """Rows of the store since a time, optionally of one sensor and one or more regions. The row groups are
    time ordered, so old data is skipped. where adds (column, operator, value) filters, for example a
    latitude and longitude box, applied while reading. Returns None when there is no store."""
    path = path or store_path()
    if not path.exists():
        return None
    filt = list(where or [])
    if since is not None:
        filt.append(("timestamp", ">=", pd.Timestamp(since)))
    if sensor:
        filt.append(("sensor", "==", sensor))
    if region:
        filt.append(("region", "==", region))
    if regions:
        filt.append(("region", "in", list(regions)))
    if not filt:
        return pd.read_parquet(path, columns=columns)
    try:
        return pd.read_parquet(path, columns=columns, filters=filt)
    except (ImportError, ValueError) as e:
        if "dataset" not in str(e):
            raise
    return read_row_groups(path, columns, filt)       # the filtered reader of pyarrow is not available here


_OPS = {">=": lambda s, v: s >= v, "<=": lambda s, v: s <= v, ">": lambda s, v: s > v, "<": lambda s, v: s < v,
        "==": lambda s, v: s == v, "!=": lambda s, v: s != v, "in": lambda s, v: s.isin(list(v))}


def _group_may_match(rg, names, filt):
    """False when the min and max of a row group show that no row can pass a filter."""
    for col, op, val in filt:
        if col not in names or op not in (">=", "<=", ">", "<", "=="):
            continue
        st = rg.column(names.index(col)).statistics
        if st is None or not st.has_min_max:
            continue
        try:
            lo, hi = st.min, st.max
            if isinstance(val, pd.Timestamp):
                lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
            if (op in (">=", ">") and hi < val) or (op in ("<=", "<") and lo > val) or \
               (op == "==" and (val < lo or val > hi)):
                return False
        except TypeError:
            continue
    return True


def read_row_groups(path, columns, filt):
    """Filtered read without the pyarrow dataset module: one row group at a time, row groups that cannot match
    are skipped by their statistics, so memory stays bounded by one row group and the matching rows."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    names = pf.schema_arrow.names
    keep = list(columns) if columns else names
    need = list(dict.fromkeys(keep + [c for c, _, _ in filt]))
    parts = []
    for i in range(pf.metadata.num_row_groups):
        if not _group_may_match(pf.metadata.row_group(i), pf.metadata.schema.to_arrow_schema().names, filt):
            continue
        df = pf.read_row_group(i, columns=need).to_pandas()
        mask = np.ones(len(df), dtype=bool)
        for col, op, val in filt:
            mask &= _OPS[op](df[col], val).to_numpy(dtype=bool)
        if mask.any():
            parts.append(df.loc[mask, keep])
    if not parts:
        return pf.schema_arrow.empty_table().select(keep).to_pandas()
    return pd.concat(parts, ignore_index=True)


def standardize(raw, sensor, region):
    """Raw FIRMS rows with the columns of the store, dates as days and acq_time as an integer."""
    df = normalize_columns(raw)
    df["sensor"], df["region"] = sensor, region
    df["acq_date"] = pd.to_datetime(df["acq_date"]).dt.normalize()
    df["acq_time"] = pd.to_numeric(df["acq_time"]).astype(np.int32)
    df["confidence"] = df["confidence"].astype(str)
    df["daynight"] = df["daynight"].astype(str)
    return df[RAW_COLS].reset_index(drop=True)


def region_names():
    """Every region with a box: the eight trained regions first, then the untrained continent boxes."""
    return list(live.REGIONS)


def _inside(df, box):
    west, south, east, north = box
    return df["longitude"].between(west, east) & df["latitude"].between(south, north)


def earlier_boxes(region):
    """Boxes earlier in the region order that share points with this one (edges count). A detection that lies
    in such a box belongs to the earlier region, so it is stored once."""
    names = list(live.REGIONS)
    w, s, e, n = live.REGIONS[region]
    out = []
    for r in names[:names.index(region)]:
        w2, s2, e2, n2 = live.REGIONS[r]
        if w <= e2 and w2 <= e and s <= n2 and s2 <= n:
            out.append(live.REGIONS[r])
    return out


def owned(df, region):
    """Rows of a region that no earlier region owns."""
    drop = np.zeros(len(df), dtype=bool)
    for box in earlier_boxes(region):
        drop |= _inside(df, box).to_numpy()
    return df[~drop] if drop.any() else df


# ------------------------------------------------------------------ progress on disk
class State:
    """Progress of the fetch, one file per sensor, region and day. Staged days wait for the merge into the
    store, merged days are listed in a ledger. A day that is staged or merged is not asked for again, except the
    latest REFETCH_DAYS."""

    def __init__(self, path):
        self.dir = path.parent / "ingest_state"
        self.ledger = self.dir / "merged.txt"
        self.staged = self.dir / "staged"
        self.store_exists = path.exists()

    @staticmethod
    def _slug(name):
        return live._slug(name)

    def group_dir(self, sensor, region):
        return self.staged / self._slug(sensor) / self._slug(region)

    def merged(self):
        """{(sensor, region): set of days} merged into the store. Empty when the store is missing."""
        out = {}
        if not (self.store_exists and self.ledger.exists()):
            return out
        for line in self.ledger.read_text(encoding="utf-8").splitlines():
            parts = line.split("|")
            if len(parts) == 3:
                out.setdefault((parts[0], parts[1]), set()).add(pd.Timestamp(parts[2]))
        return out

    def seed(self, summary):
        """No ledger yet but a store: every day between the first and the last stored day of a group was fetched."""
        lines = []
        for r in summary.itertuples(index=False):
            for d in pd.date_range(r.first, r.last, freq="D"):
                lines.append(f"{r.sensor}|{r.region}|{d:%Y-%m-%d}")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.ledger.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    def staged_days(self, sensor, region):
        d = self.group_dir(sensor, region)
        return {pd.Timestamp(p.stem) for p in d.iterdir()} if d.exists() else set()

    def stage_day(self, sensor, region, day, df):
        d = self.group_dir(sensor, region)
        d.mkdir(parents=True, exist_ok=True)
        f = d / f"{day:%Y-%m-%d}"
        for old in (f.with_suffix(".parquet"), f.with_suffix(".empty")):
            old.unlink(missing_ok=True)
        if df is not None and len(df):
            df.reset_index(drop=True).to_parquet(f.with_suffix(".parquet"), index=False)
        else:
            f.with_suffix(".empty").touch()

    def staged_groups(self):
        """{(sensor slug, region slug)} that hold staged days."""
        out = []
        if self.staged.exists():
            for s in self.staged.iterdir():
                for r in s.iterdir():
                    out.append((s.name, r.name))
        return out

    def read_staged(self, sensor, region):
        d = self.group_dir(sensor, region)
        parts = [pd.read_parquet(p) for p in sorted(d.glob("*.parquet"))] if d.exists() else []
        return pd.concat(parts, ignore_index=True) if parts else None

    def commit(self, groups, cutoff, merged):
        """After the store is replaced: record the staged days as merged, drop the staged files and old days."""
        lines = []
        for table in (merged, groups):
            for (sensor, region), days in table.items():
                for d in days:
                    if d >= cutoff:
                        lines.append(f"{sensor}|{region}|{d:%Y-%m-%d}")
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.ledger.with_name(self.ledger.name + ".tmp")
        tmp.write_text("\n".join(sorted(set(lines))) + ("\n" if lines else ""), encoding="utf-8")
        os.replace(tmp, self.ledger)
        for (sensor, region) in groups:
            d = self.group_dir(sensor, region)
            if d.exists():
                for p in d.iterdir():
                    p.unlink()
                d.rmdir()


# ------------------------------------------------------------------ scoring
def score_group(raw, region, sensor, models, chunk=250_000):
    """Features, history (inside this sensor and region only) and both tiers for one group. The history features
    are built for the whole group, the scores in chunks."""
    raw["satellite_name"] = sensor
    df = live.prepare(raw, region)
    del raw
    df["sensor"], df["region"] = sensor, region
    feats = list(dict.fromkeys(models.instant_features + models.history_features))
    cols = ["sensor", "region", "latitude", "longitude", "acq_date", "acq_time", "timestamp", "cell_id",
            "confidence", "daynight"] + feats
    cols = list(dict.fromkeys(cols + SCORE_COLS))
    parts = []
    for a in range(0, len(df), chunk):
        part = serve.score_frame(models, df.iloc[a:a + chunk])[cols].copy()
        for f in models.history_only + ["bright_diff", "pixel_area", "frp_per_area"]:
            if f in part.columns:
                part[f] = part[f].astype(np.float32)
        parts.append(part)
    del df
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)
    del parts
    return out


def write_sorted(writer, scored, times):
    """Add a scored group to the writer in time order, one row group at a time, with the ingest times looked up
    by key. No sorted copy of the whole group is made."""
    idx = pd.MultiIndex.from_frame(scored[KEY_COLS])
    scored["ingest_time"] = times.reindex(idx).to_numpy()
    del idx
    order = np.lexsort((scored["longitude"].to_numpy(), scored["latitude"].to_numpy(),
                        scored["timestamp"].to_numpy()))
    for a in range(0, len(order), ROW_GROUP):
        writer.add(scored.iloc[order[a:a + ROW_GROUP]].reset_index(drop=True))


# ------------------------------------------------------------------ one run
class MemoryGuard(RuntimeError):
    """A step would pass the memory limit. The store and the staged data are left as they are."""


def _check_memory(label, rows=0):
    from . import mem
    commit = mem.peak_memory(current=True)[1]            # what is held now, not the peak of earlier groups
    if commit + rows * BYTES_PER_ROW / 2 ** 20 > LIMIT_COMMIT_MB:
        raise MemoryGuard(f"{label}: would pass {LIMIT_COMMIT_MB} MB of commit "
                          f"(now {commit:.0f} MB, {rows:,} rows). Stopped, the store is unchanged.")


def fetch_new(key, state, today, fetch, limiter, sleep, pause, log, names):
    """Fetch what is missing for every sensor and region into the staging area. Returns (rows fetched per
    sensor, failures). Progress is saved after every call."""
    fetched, failures = {s: 0 for s in live.LIVE_SOURCES}, 0
    cutoff = today - timedelta(days=KEEP_DAYS - 1)
    wanted = list(pd.date_range(cutoff, today, freq="D"))
    recent = {today - timedelta(days=i) for i in range(REFETCH_DAYS)}
    merged = state.merged()
    for sensor, source in live.LIVE_SOURCES.items():
        for region in names:
            have = merged.get((sensor, region), set()) | state.staged_days(sensor, region)
            days = [d for d in wanted if d not in have or d in recent]
            for run in live._chunks(days):
                limiter.wait(call_cost(live.REGIONS[region], len(run)))
                try:
                    raw = fetch(key, live.REGIONS[region], run[0].to_pydatetime(), len(run), source)
                except Exception as e:                       # only the class and the status are shown
                    log(f"{sensor} {region}: failed, {safe_message(e)}, the rest of this part waits for the next run")
                    failures += 1
                    sleep(pause)
                    break
                df = standardize(raw, sensor, region) if len(raw) else None
                if df is not None:
                    fetched[sensor] += len(df)
                for day in run:
                    state.stage_day(sensor, region, day, df[df["acq_date"] == day] if df is not None else None)
                if df is not None:                           # rows dated outside the asked range
                    for day in sorted(set(df["acq_date"]) - set(run)):
                        state.stage_day(sensor, region, day, df[df["acq_date"] == day])
                sleep(pause)
    return fetched, failures


def _group_keys(path, sensor, region):
    if not path.exists():
        return None
    return pd.read_parquet(path, columns=KEY_COLS, filters=[("sensor", "==", sensor), ("region", "==", region)])


def _new_rows(state, extra, sensor, region, cutoff):
    """Staged and supplied rows of one group inside the window and owned by this region, or None."""
    parts = [p for p in (state.read_staged(sensor, region), extra.get((sensor, region))) if p is not None and len(p)]
    if not parts:
        return None
    new = pd.concat(parts, ignore_index=True)
    new["acq_date"] = pd.to_datetime(new["acq_date"])
    new = new[new["acq_date"] >= cutoff]
    return owned(new, region).drop_duplicates(subset=KEY_COLS)


def _added_count(path, sensor, region, new):
    """Rows of `new` whose key is not in the store yet."""
    if new is None or not len(new):
        return 0
    keys = _group_keys(path, sensor, region)
    if keys is None or keys.empty:
        return len(new)
    keys["acq_date"] = pd.to_datetime(keys["acq_date"])
    both = pd.concat([keys.assign(_n=0), new[KEY_COLS].assign(_n=1)], ignore_index=True)
    both = both.drop_duplicates(subset=KEY_COLS, keep="first")
    return int((both["_n"] == 1).sum())


def run_once(key=None, today=None, fetch=_DEFAULT, models=None, extra=None, sleep=None,
             pause=None, limiter=None, log=print, now=None, path=None, trace=False, regions=None):
    """One ingest. fetch=None skips the network (used with extra, a list of standardized raw frames).
    Returns a dict with rows fetched, added and kept per sensor. With trace=True the peak memory of every stage
    is printed. A run that stops keeps the fetched days: the store is only replaced when the run finishes."""
    from . import mem
    stage = (lambda label: mem.report(label)) if trace else (lambda label: None)
    path = path or store_path()
    today = pd.Timestamp(today or datetime.now(timezone.utc).date()).normalize()
    now = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC").tz_localize(None)
    fetch = live.fetch_chunk if fetch is _DEFAULT else fetch
    sleep = sleep or time.sleep
    pause = live.PAUSE if pause is None else pause
    limiter = limiter or RateLimiter(sleep=sleep)
    names = list(regions or region_names())
    cutoff = today - timedelta(days=KEEP_DAYS - 1)

    summary = store_summary(path)
    stage("load store (three columns)")
    state = State(path)
    if state.store_exists and not state.ledger.exists():
        state.seed(summary)
    fetched, failures = {s: 0 for s in live.LIVE_SOURCES}, 0
    if fetch is not None:
        fetched, failures = fetch_new(key, state, today, fetch, limiter, sleep, pause, log, names)
    stage("fetch")

    by_extra = {}
    for f in extra or []:
        for (s, r), g in f.groupby(["sensor", "region"]):
            by_extra.setdefault((s, r), []).append(g)
    by_extra = {k: pd.concat(v, ignore_index=True) for k, v in by_extra.items()}

    stored = {(r.sensor, r.region): r for r in summary.itertuples(index=False)}
    slugs = {(state._slug(s), state._slug(r)): (s, r) for s in live.LIVE_SOURCES for r in live.REGIONS}
    staged = {}
    for k in state.staged_groups():
        if k in slugs:
            staged[slugs[k]] = state.staged_days(*slugs[k])
    keys = sorted(set(stored) | set(staged) | set(by_extra))
    added, touched = {}, set()
    for (sensor, region) in keys:
        new = _new_rows(state, by_extra, sensor, region, cutoff)
        added[(sensor, region)] = _added_count(path, sensor, region, new)
        trimmed = (sensor, region) in stored and pd.Timestamp(stored[(sensor, region)].first) < cutoff
        if added[(sensor, region)] or trimmed:
            touched.add((sensor, region))
    stage("plan (keys only)")
    counts = {s: {"fetched": fetched[s], "added": sum(n for (sn, _), n in added.items() if sn == s), "kept": 0}
              for s in live.LIVE_SOURCES}

    if not touched:
        state.commit(staged, cutoff, state.merged())
        for s in counts:
            counts[s]["kept"] = int(sum(r.rows for k, r in stored.items() if k[0] == s))
        _print(counts, log, failures)
        return {"counts": counts, "failures": failures, "changed": False}

    models = models or serve.Models()
    merged_before = state.merged()
    writer = StoreWriter(path)
    kept = {s: 0 for s in live.LIVE_SOURCES}
    try:
        for (sensor, region) in keys:
            filt = [("sensor", "==", sensor), ("region", "==", region)]
            have = stored.get((sensor, region))
            if (sensor, region) not in touched:
                if have is not None:
                    import pyarrow.parquet as pq
                    table = pq.read_table(path, filters=filt)
                    writer.add(table)
                    kept[sensor] += table.num_rows
                    del table
                continue
            new = _new_rows(state, by_extra, sensor, region, cutoff)
            new = new if new is not None else pd.DataFrame(columns=RAW_COLS)
            _check_memory(f"{sensor} {region}", (int(have.rows) if have is not None else 0) + len(new))
            old = pd.DataFrame(columns=RAW_COLS + ["ingest_time"])
            if have is not None:
                old = pd.read_parquet(path, columns=RAW_COLS + ["ingest_time"], filters=filt)
            both = pd.concat([old, new.assign(ingest_time=now)], ignore_index=True)
            del old, new
            both["acq_date"] = pd.to_datetime(both["acq_date"])
            both = both.drop_duplicates(subset=KEY_COLS, keep="first")           # the stored row wins
            both = both[both["acq_date"] >= cutoff].reset_index(drop=True)
            times = both.set_index(KEY_COLS)["ingest_time"]
            raw = both[RAW_COLS]
            del both
            stage(f"merge {sensor} {region}")
            scored = score_group(raw, region, sensor, models)
            del raw
            stage(f"scoring {sensor} {region}")
            rows = len(scored)
            write_sorted(writer, scored, times)
            kept[sensor] += rows
            del scored, times
            line = f"{sensor} {region}: rows {rows:,}, store file so far {writer.size() / 2 ** 20:.0f} MB"
            if trace:
                line += f", peak commit {mem.peak_memory()[1]:.0f} MB"
            log(line)
            stage(f"write {sensor} {region}")
        writer.commit()
    except BaseException:
        writer.abort()
        raise
    state.commit(staged, cutoff, merged_before)
    for s in counts:
        counts[s]["kept"] = kept[s]
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


def rebuild_score(models=None, path=None, log=print, trace=False):
    """Score every stored row again with the saved models, calibrators and bands, without fetching. Run it
    after the models change. Rows, ingest times and the 90 day window are kept; features and history are
    rebuilt per sensor and region from the stored readings. One group at a time, atomic write."""
    from . import mem
    path = path or store_path()
    summary = store_summary(path)
    if summary.empty:
        log("no store to score")
        return {"rows": 0}
    models = models or serve.Models()
    writer = StoreWriter(path)
    total = 0
    try:
        for r in summary.itertuples(index=False):
            _check_memory(f"{r.sensor} {r.region}", int(r.rows))
            g = pd.read_parquet(path, columns=RAW_COLS + ["ingest_time"],
                                filters=[("sensor", "==", r.sensor), ("region", "==", r.region)])
            times = g.set_index(KEY_COLS)["ingest_time"]
            raw = g[RAW_COLS]
            del g
            scored = score_group(raw, r.region, r.sensor, models)
            del raw
            rows = len(scored)
            write_sorted(writer, scored, times)
            total += rows
            del scored, times
            log(f"{r.sensor} {r.region}: scored {rows:,} rows")
            if trace:
                mem.report(f"rebuild {r.sensor} {r.region}")
        writer.commit()
    except BaseException:
        writer.abort()
        raise
    return {"rows": total, "dropped": int(summary["rows"].sum()) - total}


# ------------------------------------------------------------------ cost of a call
def call_cost(box, days):
    """Transactions one call is expected to cost, from the measured cost (python -m src.ingest --measure-cost)."""
    return COST_PER_CALL * days


def measure_cost(key, log=print):
    """Read the key's transaction counter before and after one call for the largest box, for 1 day and for
    5 days. Prints only numbers. The calls are real, the counter comes from the status service."""
    box_name = max(live.REGIONS, key=lambda r: (live.REGIONS[r][2] - live.REGIONS[r][0]) *
                   (live.REGIONS[r][3] - live.REGIONS[r][1]))
    box = live.REGIONS[box_name]
    out = {"area_deg2": int((box[2] - box[0]) * (box[3] - box[1]))}
    start = datetime.now(timezone.utc) - timedelta(days=8)
    for days in (1, 5):
        before, limit = live.key_status(key)
        before2, _ = live.key_status(key)
        live.fetch_chunk(key, box, start, days, live.LIVE_SOURCES["NOAA-20"])
        after, _ = live.key_status(key)
        out[f"status_drift_{days}"] = before2 - before
        out[f"cost_{days}_days"] = after - before2
        out["limit"] = limit
    log("measured transactions: " + ", ".join(f"{k} {v}" for k, v in out.items()))
    return out


def pick_regions(text):
    """Regions of a run: all, trained (the eight of the training data) or a comma separated list."""
    if text == "all":
        return list(live.REGIONS)
    if text == "trained":
        return [r for r in live.REGIONS if r in live.TRAINED]
    names = [t.strip() for t in text.split(",") if t.strip()]
    bad = [n for n in names if n not in live.REGIONS]
    if bad:
        raise SystemExit(f"unknown region: {bad[0]}")
    return names


def main(argv=None):
    ap = argparse.ArgumentParser(description="Incremental live store")
    ap.add_argument("--once", action="store_true", help="one ingest, then stop")
    ap.add_argument("--loop", action="store_true", help="ingest again and again")
    ap.add_argument("--every-minutes", type=float, default=180.0, help="wait between ingests in a loop")
    ap.add_argument("--import-live", action="store_true", help="one off: take the rolling stores of src/live.py")
    ap.add_argument("--rebuild-score", action="store_true", help="score every stored row again, no fetch")
    ap.add_argument("--measure-cost", action="store_true", help="measure the transaction cost of one call")
    ap.add_argument("--regions", default="all", help="all (default), trained, or a comma separated list of regions")
    a = ap.parse_args(argv)
    from .mem import report
    if a.rebuild_score:
        r = rebuild_score(trace=True)
        print(f"rows scored: {r['rows']:,}")
        report("ingest rebuild-score")
        return
    if a.import_live:
        import_live()
        return
    if a.measure_cost:
        key = live.load_key()
        if not key:
            raise SystemExit("No map key found in the environment or .env")
        try:
            measure_cost(key)
        except Exception as e:
            print(f"measure failed: {safe_message(e)}")
        return
    if not (a.once or a.loop):
        ap.error("give --once, --loop, --measure-cost, --import-live or --rebuild-score")
    key = live.load_key()
    if not key:
        raise SystemExit("No map key found in the environment or .env")
    models = serve.Models()
    while True:
        try:
            run_once(key, models=models, trace=True, regions=pick_regions(a.regions))
        except Exception as e:                               # a failed round must not stop the loop
            print(f"ingest failed: {safe_message(e)}", flush=True)
        report("ingest run")
        if not a.loop:
            break
        time.sleep(a.every_minutes * 60)


if __name__ == "__main__":
    main()

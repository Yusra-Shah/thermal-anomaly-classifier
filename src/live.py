"""Live data for the app: FIRMS near real time fetch, rolling store, features, scoring, geography.

The map key is read from FIRMS_MAP_KEY (environment or a .env file in the project root).
It is never printed, logged or put into an error message.

FIRMS area API (https://firms.modaps.eosdis.nasa.gov/api/area/):
    /api/area/csv/{MAP_KEY}/{SOURCE}/{west,south,east,north}/{DAY_RANGE}/{YYYY-MM-DD}
DAY_RANGE is 1 to 5 in the documentation, and the range runs forward from the date, so the
last 60 days are fetched in chunks of 5 days.

Sources are the near real time feeds of NOAA-20 and NOAA-21. Each satellite has its own rolling
store, every row keeps the satellite name, and the history features are built inside one satellite
only. The models were trained on Suomi NPP, which no longer delivers near real time data. These
feeds have no type column, so there is no ground truth.
"""
import os
import time
from datetime import datetime, timedelta, timezone
from io import StringIO

import numpy as np
import pandas as pd

from .config import ROOT
from .data import REQUIRED, clean, normalize_columns
from .features import add_features
from .history import add_history, add_history_v2
from .label import add_cell

LIVE_DIR = ROOT / "data" / "live"
ENV_FILE = ROOT / ".env"
BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
LIVE_SOURCES = {"NOAA-20": "VIIRS_NOAA20_NRT", "NOAA-21": "VIIRS_NOAA21_NRT"}
SOURCE = LIVE_SOURCES["NOAA-20"]
PAUSE = 1.0                    # seconds between calls, far below 5000 transactions per 10 minutes
MAX_DAY_RANGE = 5              # documented limit of the area API
KEEP_DAYS = 60
REFETCH_DAYS = 3
TIMEOUT = 60

# West, south, east, north. South Asia, Southeast Asia and North and Central America are the
# extents of the three training files behind src/region_eval.py, rounded outward.
REGIONS = {
    "South Asia": (61.0, 24.0, 76.0, 36.0),
    "Southeast Asia": (95.0, -11.0, 141.0, 6.0),
    "North and Central America": (-177.0, 19.0, -67.0, 71.0),
    "Europe (not in training)": (-10.0, 36.0, 40.0, 70.0),
}
TRAINED = {"South Asia", "Southeast Asia", "North and Central America"}


class FirmsError(RuntimeError):
    """Readable failure of a FIRMS request. Never contains the map key."""


# ------------------------------------------------------------------ key
def load_key():
    """Return the map key from the environment or the .env file, or None."""
    try:
        from dotenv import load_dotenv
        if ENV_FILE.exists():
            load_dotenv(ENV_FILE, override=False)
    except ImportError:
        pass
    key = os.environ.get("FIRMS_MAP_KEY", "").strip()
    return key or None


# ------------------------------------------------------------------ fetch
def fetch_chunk(key, box, start, days, source=SOURCE, session=None, retries=3, sleep=time.sleep):
    """One API call: detections on `start` .. `start + days - 1` inside `box` as a dataframe."""
    if not 1 <= days <= MAX_DAY_RANGE:
        raise ValueError(f"days must be between 1 and {MAX_DAY_RANGE}")
    if session is None:
        import requests
        session = requests
    area = ",".join(str(v) for v in box)
    url = f"{BASE_URL}/{key}/{source}/{area}/{days}/{start:%Y-%m-%d}"
    resp = None
    for attempt in range(retries + 1):
        try:
            resp = session.get(url, timeout=TIMEOUT)
        except Exception as e:                      # request errors carry the url, so do not chain
            raise FirmsError(f"network error ({type(e).__name__})") from None
        if resp.status_code in (429, 500, 502, 503, 504) and attempt < retries:
            sleep(2 ** attempt * 5)
            continue
        break
    if resp.status_code == 429:
        raise FirmsError("rate limit reached (5000 requests per 10 minutes). Try again later.")
    if resp.status_code != 200:
        raise FirmsError(f"FIRMS answered with HTTP {resp.status_code}")
    text = resp.text.strip()
    if not text:
        return pd.DataFrame()                       # no detections
    if not text.lower().startswith("latitude"):
        # Plain text answers such as an invalid key or an exceeded transaction limit
        raise FirmsError("FIRMS refused the request: " + text.replace(key, "***")[:200])
    df = pd.read_csv(StringIO(text))
    return df


# ------------------------------------------------------------------ rolling store
def _slug(region):
    return "".join(c.lower() if c.isalnum() else "_" for c in region).strip("_")


def region_dir(region, sat=None):
    """Folder of the day files of a region, one sub folder per satellite."""
    d = LIVE_DIR / _slug(region)
    return d / _slug(sat) if sat else d


def snapshot_path(region):
    return LIVE_DIR / f"snapshot_{_slug(region)}.parquet"


def _day_files(region, sat):
    d = region_dir(region, sat)
    return sorted(d.glob("*.parquet")) + sorted(d.glob("*.empty")) if d.exists() else []


def stored_days(region, sat):
    return {pd.Timestamp(p.stem).normalize() for p in _day_files(region, sat)}


def load_store(region, sat=None):
    """Stored raw detections of a region (one satellite, or all of them), or None. Every row has
    a satellite_name column."""
    parts = []
    for name in ([sat] if sat else LIVE_SOURCES):
        d = region_dir(region, name)
        for p in (sorted(d.glob("*.parquet")) if d.exists() else []):
            part = pd.read_parquet(p)
            part["satellite_name"] = name
            parts.append(part)
    if not parts:
        return None
    return pd.concat(parts, ignore_index=True)


def days_to_fetch(region, today, sat):
    """Missing days of the last KEEP_DAYS plus the latest REFETCH_DAYS, oldest first."""
    wanted = [today - timedelta(days=i) for i in range(KEEP_DAYS)]
    have = stored_days(region, sat)
    latest = {today - timedelta(days=i) for i in range(REFETCH_DAYS)}
    return sorted(d for d in wanted if d not in have or d in latest)


def _chunks(days):
    """Group sorted days into runs of consecutive days, at most MAX_DAY_RANGE each."""
    out, run = [], []
    for d in days:
        if run and ((d - run[-1]).days != 1 or len(run) == MAX_DAY_RANGE):
            out.append(run)
            run = []
        run.append(d)
    if run:
        out.append(run)
    return out


def refresh(region, key, today=None, fetch=fetch_chunk, progress=None, sleep=time.sleep, pause=PAUSE):
    """Fetch what is missing for both satellites, merge into the stores, prune, write the snapshot.

    `fetch(key, box, start, days, source)` is injectable so tests need no network. The satellites
    are never mixed: each has its own store. Returns a dict with the number of requests and new rows.
    """
    today = pd.Timestamp(today or datetime.now(timezone.utc).date()).normalize()
    box = REGIONS[region]
    plan = [(sat, run) for sat in LIVE_SOURCES for run in _chunks(days_to_fetch(region, today, sat))]
    new_rows = 0
    for i, (sat, run) in enumerate(plan):
        d = region_dir(region, sat)
        d.mkdir(parents=True, exist_ok=True)
        raw = fetch(key, box, run[0].to_pydatetime(), len(run), LIVE_SOURCES[sat])
        if len(raw):
            raw = normalize_columns(raw)
            day = pd.to_datetime(raw["acq_date"]).dt.normalize()
        for day_ts in run:
            f = d / f"{day_ts:%Y-%m-%d}"
            for old in (f.with_suffix(".parquet"), f.with_suffix(".empty")):
                old.unlink(missing_ok=True)
            part = raw[day == day_ts].drop_duplicates() if len(raw) else raw
            if len(part):
                part.reset_index(drop=True).to_parquet(f.with_suffix(".parquet"), index=False)
                new_rows += len(part)
            else:
                f.with_suffix(".empty").touch()
        if progress:
            progress((i + 1) / len(plan))
        if i + 1 < len(plan):
            sleep(pause)
    cutoff = today - timedelta(days=KEEP_DAYS - 1)
    for sat in LIVE_SOURCES:
        for p in _day_files(region, sat):
            if pd.Timestamp(p.stem) < cutoff:
                p.unlink()
    store = load_store(region)
    if store is not None:
        store = store.drop_duplicates(subset=["latitude", "longitude", "acq_date", "acq_time", "satellite_name"])
        write_snapshot(region, store)
    return {"requests": len(plan), "new_rows": new_rows}


def write_snapshot(region, store, fetched_at=None, demo=False):
    """Write the offline snapshot. demo=True adds an is_demo flag column (shifted historical data)."""
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    s = store.copy()
    if demo:
        s["is_demo"] = True
    s["fetched_at"] = (fetched_at or datetime.now(timezone.utc)).strftime("%Y-%m-%d %H:%M:%S")
    s.to_parquet(snapshot_path(region), index=False)


def load_snapshot(region):
    """Return (raw detections, fetch time as a UTC Timestamp) or (None, None)."""
    p = snapshot_path(region)
    if not p.exists():
        return None, None
    s = pd.read_parquet(p)
    fetched = pd.Timestamp(s["fetched_at"].iloc[0]) if len(s) else None
    return s.drop(columns=["fetched_at", "is_demo"], errors="ignore"), fetched


def snapshot_is_demo(region):
    """True when the snapshot was built by src/live_demo.py."""
    p = snapshot_path(region)
    if not p.exists():
        return False
    import pyarrow.parquet as pq
    return "is_demo" in pq.read_schema(p).names


# ------------------------------------------------------------------ features and scoring
def prepare(raw, region):
    """Training pipeline on the whole rolling store of one satellite treated as one file (a snapshot
    with several satellites is split by its satellite_name column).

    History uses only detections on strictly earlier days, so a detection never sees
    the same day or later rows.
    """
    df = normalize_columns(raw)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    df = add_cell(add_features(clean(df)))
    if "satellite_name" in df.columns:               # one history per satellite, never mixed
        df["source_file"] = _slug(region) + "_" + df["satellite_name"].map(_slug)
    else:
        df["source_file"] = _slug(region)
    return add_history_v2(add_history(df))


def score(df, history_bundle, instant_bundle):
    """Add p_static_history and p_static_instant."""
    df = df.copy()
    df["p_static_history"] = history_bundle["model"].predict_proba(df[history_bundle["features"]])[:, 1]
    df["p_static_instant"] = instant_bundle["model"].predict_proba(df[instant_bundle["features"]])[:, 1]
    return df


def days_seen_30(df):
    """Distinct days with a detection in the same cell over the last 30 days of the store."""
    cut = df["acq_date"].max() - timedelta(days=29)
    recent = df[df["acq_date"] >= cut]
    return df["cell_id"].map(recent.groupby("cell_id")["acq_date"].nunique()).fillna(0).astype(int)


# ------------------------------------------------------------------ geography
_AFRICA = ("DZ AO BJ BW BF BI CM CV CF TD KM CG CD CI DJ EG GQ ER SZ ET GA GM GH GN GW KE LS LR LY MG MW "
           "ML MR MU MA MZ NA NE NG RW ST SN SC SL SO ZA SS SD TZ TG TN UG EH ZM ZW RE YT SH")
_ASIA = ("AF AM AZ BH BD BT BN KH CN CY GE HK IN ID IR IQ IL JP JO KZ KW KG LA LB MO MY MV MN MM NP KP OM "
         "PK PS PH QA SA SG KR LK SY TW TJ TH TL TR TM AE UZ VN YE")
_EUROPE = ("AL AD AT BY BE BA BG HR CZ DK EE FO FI FR DE GI GR GG HU IS IE IM IT JE XK LV LI LT LU MT MD MC "
           "ME NL MK NO PL PT RO SM RS SK SI ES SJ SE CH UA GB VA AX")
_NORTH_AMERICA = ("AG BS BB BZ CA CR CU DM DO SV GD GT HT HN JM MX NI PA KN LC VC TT US PR GL BM AI AW BQ "
                  "VG KY CW GP MQ MS SX BL MF PM TC VI UM")
_SOUTH_AMERICA = "AR BO BR CL CO EC FK GF GY PY PE SR UY VE GS"
_OCEANIA = "AS AU CK FJ PF GU KI MH FM NR NC NZ NU NF MP PW PG PN WS SB TK TO TV VU WF"
_ANTARCTICA = "AQ TF BV HM"
CONTINENT = {cc: name for name, codes in (
    ("Africa", _AFRICA), ("Asia", _ASIA), ("Europe", _EUROPE), ("North America", _NORTH_AMERICA),
    ("South America", _SOUTH_AMERICA), ("Oceania", _OCEANIA), ("Antarctica", _ANTARCTICA))
    for cc in codes.split()}


def continent_of(cc, lon=None):
    if cc == "RU":                                  # spans two continents, split at 60 degrees east
        return "Asia" if lon is not None and lon >= 60 else "Europe"
    return CONTINENT.get(cc, "Other")


def add_geo(df, cache_path=None):
    """Add country code, admin1, nearest city and continent. Reverse geocoding runs once per cell
    and is cached (default data/live/geo_cache.parquet)."""
    cache_path = cache_path or LIVE_DIR / "geo_cache.parquet"
    cols = ["cell_id", "cc", "admin1", "city"]
    cache = pd.read_parquet(cache_path) if cache_path.exists() else pd.DataFrame(columns=cols)
    cells = df.drop_duplicates("cell_id")[["cell_id", "latitude", "longitude"]]
    todo = cells[~cells["cell_id"].isin(cache["cell_id"])]
    if len(todo):
        import reverse_geocoder as rg
        hits = rg.search(list(zip(todo["latitude"], todo["longitude"])), mode=1, verbose=False)
        new = pd.DataFrame({"cell_id": todo["cell_id"].values, "cc": [h["cc"] for h in hits],
                            "admin1": [h["admin1"] for h in hits], "city": [h["name"] for h in hits]})
        cache = pd.concat([cache, new], ignore_index=True)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache.to_parquet(cache_path, index=False)
    out = df.merge(cache.drop_duplicates("cell_id"), on="cell_id", how="left")
    out["continent"] = [continent_of(c, lon) for c, lon in zip(out["cc"], out["longitude"])]
    for c in ("cc", "admin1", "city", "continent"):          # categories keep the cached tables small
        out[c] = out[c].astype("category")
    return out


def build_scored(region, history_bundle, instant_bundle, raw=None):
    """Features, scores, days seen and geography for a raw store. Returns a dataframe."""
    df = score(prepare(raw, region), history_bundle, instant_bundle)
    df["days_seen_30"] = days_seen_30(df)
    return add_geo(df)

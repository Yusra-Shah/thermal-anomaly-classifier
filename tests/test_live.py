"""Tests for src/live.py and the Earth tab. No network: the API is faked.

Run from the project root:  python -m tests.test_live
"""
import gc
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src import live
from src.history import HISTORY_FEATURES, HISTORY_V2_FEATURES

KEY = "SECRETKEY123"
REGION = "South Asia"
TODAY = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()


def fake_detections(n_days=60, end=None, seed=0):
    """Small NRT style table: a few static spots seen most days plus random single fires."""
    rng = np.random.default_rng(seed)
    end = end or TODAY
    west, south, east, north = live.REGIONS[REGION]
    static = [(28.6 + i * 0.5, 70.0 + i * 0.7) for i in range(8)]
    rows = []
    for k in range(n_days):
        day = end - pd.Timedelta(days=n_days - 1 - k)
        for lat, lon in static:
            if rng.random() < 0.8:
                rows.append((lat + rng.normal(0, 0.001), lon + rng.normal(0, 0.001), day))
        for _ in range(25):
            rows.append((rng.uniform(south + 1, north - 1), rng.uniform(west + 1, east - 1), day))
    df = pd.DataFrame(rows, columns=["latitude", "longitude", "acq_date"])
    n = len(df)
    df["bright_ti4"] = rng.normal(340, 15, n).round(1)
    df["scan"] = rng.uniform(0.4, 0.7, n).round(2)
    df["track"] = rng.uniform(0.4, 0.6, n).round(2)
    df["acq_time"] = rng.integers(0, 2400, n) // 100 * 100 + rng.integers(0, 60, n)
    df["satellite"] = "N"
    df["instrument"] = "VIIRS"
    df["confidence"] = rng.choice(["l", "n", "h"], n)
    df["version"] = "2.0NRT"
    df["bright_ti5"] = rng.normal(300, 8, n).round(1)
    df["frp"] = rng.gamma(2.0, 2.0, n).round(2)
    df["daynight"] = rng.choice(["D", "N"], n)
    df["acq_date"] = df["acq_date"].dt.strftime("%Y-%m-%d")
    return df.drop_duplicates(["latitude", "longitude", "acq_date", "acq_time"]).reset_index(drop=True)


def two_satellites(n_days=60, end=None):
    """A store frame with a satellite_name column for both near real time satellites."""
    parts = []
    for i, sat in enumerate(live.LIVE_SOURCES):
        d = fake_detections(n_days, end=end, seed=i)
        d["satellite_name"] = sat
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def fake_fetch_factory(master, log):
    """master is one frame, or a dict from source name to frame."""
    def fetch(key, box, start, days, source=None):
        log.append((pd.Timestamp(start).normalize(), days, source))
        lo = pd.Timestamp(start).normalize()
        hi = lo + pd.Timedelta(days=days - 1)
        m = master[source] if isinstance(master, dict) else master
        d = pd.to_datetime(m["acq_date"])
        return m[(d >= lo) & (d <= hi)].copy()
    return fetch


class FakeResponse:
    def __init__(self, status=200, text=""):
        self.status_code, self.text = status, text


class FakeSession:
    def __init__(self, *responses):
        self.responses, self.urls = list(responses), []

    def get(self, url, timeout=None):
        self.urls.append(url)
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def test_fetch_chunk():
    csv = fake_detections(2).head(3).to_csv(index=False)
    s = FakeSession(FakeResponse(200, csv))
    out = live.fetch_chunk(KEY, live.REGIONS[REGION], pd.Timestamp("2025-01-01"), 5, session=s)
    assert len(out) == 3
    assert s.urls[0].endswith("/VIIRS_NOAA20_NRT/61.0,24.0,76.0,36.0/5/2025-01-01"), s.urls[0]
    s = FakeSession(FakeResponse(200, csv))
    live.fetch_chunk(KEY, live.REGIONS[REGION], pd.Timestamp("2025-01-01"), 5, source=live.LIVE_SOURCES["NOAA-21"],
                     session=s)
    assert "/VIIRS_NOAA21_NRT/" in s.urls[0]

    assert live.fetch_chunk(KEY, live.REGIONS[REGION], pd.Timestamp("2025-01-01"), 1,
                            session=FakeSession(FakeResponse(200, ""))).empty
    assert live.fetch_chunk(KEY, live.REGIONS[REGION], pd.Timestamp("2025-01-01"), 1,
                            session=FakeSession(FakeResponse(200, "latitude,longitude\n"))).empty

    for resp in (FakeResponse(200, f"Invalid MAP_KEY {KEY}"), FakeResponse(403, "no"), FakeResponse(404, "")):
        try:
            live.fetch_chunk(KEY, live.REGIONS[REGION], pd.Timestamp("2025-01-01"), 1,
                             session=FakeSession(resp))
            raise AssertionError("expected FirmsError")
        except live.FirmsError as e:
            assert KEY not in str(e)

    waits = []
    s = FakeSession(FakeResponse(429), FakeResponse(200, csv))
    assert len(live.fetch_chunk(KEY, live.REGIONS[REGION], pd.Timestamp("2025-01-01"), 1, session=s,
                                sleep=waits.append)) == 3 and len(waits) == 1
    s = FakeSession(*[FakeResponse(429)] * 4)
    try:
        live.fetch_chunk(KEY, live.REGIONS[REGION], pd.Timestamp("2025-01-01"), 1, session=s, sleep=lambda t: None)
        raise AssertionError("expected FirmsError")
    except live.FirmsError as e:
        assert "rate limit" in str(e)

    class Boom(Exception):
        pass
    try:
        live.fetch_chunk(KEY, live.REGIONS[REGION], pd.Timestamp("2025-01-01"), 1,
                         session=FakeSession(Boom(f"failed for url with {KEY}")))
        raise AssertionError("expected FirmsError")
    except live.FirmsError as e:
        assert KEY not in str(e)
    try:
        live.fetch_chunk(KEY, live.REGIONS[REGION], pd.Timestamp("2025-01-01"), 10, session=FakeSession())
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    print("fetch_chunk test passed")


def test_refresh_store(tmp):
    live.LIVE_DIR = Path(tmp)
    sats = list(live.LIVE_SOURCES)
    masters = {live.LIVE_SOURCES[n]: fake_detections(63, end=TODAY + pd.Timedelta(days=2), seed=i)
               for i, n in enumerate(sats)}                           # includes two future days
    log, waits = [], []
    fetch = fake_fetch_factory(masters, log)
    r = live.refresh(REGION, KEY, today=TODAY, fetch=fetch, sleep=waits.append)
    assert r["requests"] == 24 and all(d <= 5 for _, d, _ in log), log        # 12 chunks per satellite
    assert {src for _, _, src in log} == set(live.LIVE_SOURCES.values())
    assert len(waits) == 23 and all(w == live.PAUSE for w in waits)        # a short pause between calls
    for sat in sats:
        assert len(live.stored_days(REGION, sat)) == 60
    store = live.load_store(REGION)
    assert set(store["satellite_name"]) == set(sats)                      # the satellite name is kept per row
    assert pd.to_datetime(store["acq_date"]).max() == TODAY
    assert not store.duplicated(["latitude", "longitude", "acq_date", "acq_time", "satellite_name"]).any()
    assert len(live.load_store(REGION, "NOAA-20")) != len(live.load_store(REGION, "NOAA-21"))
    snap, fetched = live.load_snapshot(REGION)
    assert len(snap) == len(store) and fetched is not None and "satellite_name" in snap.columns

    log.clear()
    r = live.refresh(REGION, KEY, today=TODAY, fetch=fetch, sleep=lambda t: None)   # the latest 3 days each
    assert r["requests"] == 2 and sorted(d for _, d, _ in log) == [3, 3], log
    assert {start for start, _, _ in log} == {TODAY - pd.Timedelta(days=2)}

    log.clear()
    live.refresh(REGION, KEY, today=TODAY + pd.Timedelta(days=2), fetch=fetch, sleep=lambda t: None)
    for sat in sats:
        days = live.stored_days(REGION, sat)
        assert len(days) == 60 and min(days) == TODAY - pd.Timedelta(days=57), (len(days), min(days))
    assert sorted(log) == sorted([(TODAY, 3, v) for v in live.LIVE_SOURCES.values()]), log
    print("rolling store test passed")


def test_key_not_printed_in_refresh(tmp):
    """A failing request inside refresh must not leak the key in the message or the traceback."""
    import traceback
    live.LIVE_DIR = Path(tmp)

    class Boom:
        def get(self, url, timeout=None):
            raise ConnectionError(f"failed for url {url}")

    def fetch(key, box, start, days, source):
        return live.fetch_chunk(key, box, start, days, source=source, session=Boom())
    try:
        live.refresh(REGION, KEY, today=TODAY, fetch=fetch, sleep=lambda t: None)
        raise AssertionError("expected FirmsError")
    except live.FirmsError as e:
        text = str(e) + "".join(traceback.format_exception(e))
        assert KEY not in text and "://" not in text and "ConnectionError" in text
    print("key not printed in refresh passed")


def test_history_per_satellite(tmp):
    """History features are built inside one satellite and never mix satellites."""
    live.LIVE_DIR = Path(tmp)
    feats = HISTORY_FEATURES + HISTORY_V2_FEATURES
    both = two_satellites(60)
    got = live.prepare(both, REGION)
    key = ["latitude", "longitude", "acq_date", "acq_time"]
    for sat in live.LIVE_SOURCES:
        alone = live.prepare(both[both["satellite_name"] == sat].drop(columns=["satellite_name"]), REGION)
        a = got[got["satellite_name"] == sat].set_index(key)[feats].sort_index()
        b = alone.set_index(key)[feats].sort_index()
        assert a.index.equals(b.index) and np.allclose(a.to_numpy(float), b.to_numpy(float), atol=1e-6), sat
    # changing one satellite leaves the other satellite's history untouched
    alt = both.copy()
    mask = alt["satellite_name"] == "NOAA-21"
    alt = pd.concat([alt, alt[mask].assign(acq_time=lambda d: (d["acq_time"] + 5) % 2400)], ignore_index=True)
    got2 = live.prepare(alt, REGION)
    a = got[got["satellite_name"] == "NOAA-20"].set_index(key)[feats].sort_index()
    b = got2[got2["satellite_name"] == "NOAA-20"].set_index(key)[feats].sort_index()
    assert a.index.equals(b.index) and np.allclose(a.to_numpy(float), b.to_numpy(float), atol=1e-6)
    # a satellite that starts later has its own day zero
    old21 = (both["satellite_name"] == "NOAA-21") & (pd.to_datetime(both["acq_date"]) < TODAY - pd.Timedelta(days=20))
    g3 = live.prepare(both[~old21], REGION)
    assert g3[g3["satellite_name"] == "NOAA-21"]["days_into_data"].min() == 0
    assert g3[g3["satellite_name"] == "NOAA-20"]["days_into_data"].max() >= 58
    print("history per satellite passed")


def test_live_check_helpers():
    from src import live_check
    day = pd.Timestamp("2026-11-01")
    rows = [(1, day - pd.Timedelta(days=i), 0.9) for i in range(10)]            # busy static cell
    rows += [(2, day - pd.Timedelta(days=i), 0.1) for i in range(10)]            # busy fire cell
    rows += [(3, day, 0.9)] * 3                                                    # too few detections
    rows += [(4, day - pd.Timedelta(days=60 + i), 0.9) for i in range(12)]       # too old
    df = pd.DataFrame(rows, columns=["cell_id", "acq_date", "p"])
    busy, static = live_check.cells_static(df, "p", 0.5)
    assert busy == {1, 2} and static == {1}, (busy, static)
    print("live check helpers passed")


def test_leakage_rolling_store(tmp):
    """History of a detection must not change when rows on later days change."""
    live.LIVE_DIR = Path(tmp)
    raw = fake_detections(60)
    base = live.prepare(raw, REGION)
    feats = HISTORY_FEATURES + HISTORY_V2_FEATURES
    cut = TODAY - pd.Timedelta(days=30)
    rng = np.random.default_rng(1)
    for variant in range(3):
        alt = raw.copy()
        later = pd.to_datetime(alt["acq_date"]) > cut
        if variant == 0:                      # new readings
            alt.loc[later, "bright_ti4"] += rng.normal(30, 5, later.sum())
            alt.loc[later, "frp"] *= 4
        elif variant == 1:                    # drop half of the later rows
            alt = alt[~(later & (rng.random(len(alt)) < 0.5))]
        else:                                 # add extra later detections at the same places
            extra = alt[later].copy()
            extra["acq_time"] = (extra["acq_time"] + 7) % 2400
            alt = pd.concat([alt, extra], ignore_index=True)
        got = live.prepare(alt, REGION)
        a = base[base["acq_date"] <= cut].set_index(["latitude", "longitude", "acq_date", "acq_time"])[feats]
        b = got[got["acq_date"] <= cut].set_index(["latitude", "longitude", "acq_date", "acq_time"])[feats]
        a, b = a.sort_index(), b.sort_index()
        # Rows on the cut day itself see only earlier days, so every row up to the cut must match
        assert a.index.equals(b.index)
        assert np.allclose(a.to_numpy(float), b.to_numpy(float), atol=1e-6), f"variant {variant} changed history"
    print("leakage on the rolling store passed")


def test_geography(tmp):
    live.LIVE_DIR = Path(tmp)
    assert live.continent_of("IN") == "Asia" and live.continent_of("DE") == "Europe"
    assert live.continent_of("RU", 100) == "Asia" and live.continent_of("RU", 30) == "Europe"
    assert live.continent_of("US") == "North America" and live.continent_of("ZZ") == "Other"
    df = live.add_geo(live.prepare(fake_detections(3), REGION))
    assert (df["cc"].notna()).all() and (df["continent"] != "Other").all()
    assert (Path(tmp) / "geo_cache.parquet").exists()
    print("geography test passed")


INSTANT_TIER = "Instant (single reading)"
HISTORY_TIER = "With history (past detections)"
LIVE_SRC = "Live near real time"
DEMO_NOTICE = ("Demo snapshot: historical 2025 data shifted to look recent, for testing only. "
               "This is not current data.")


def run_app(tmp, key, setup=None):
    from streamlit.testing.v1 import AppTest
    import streamlit as st
    st.cache_data.clear()                      # keep memory flat across many AppTest runs
    st.cache_resource.clear()
    live.LIVE_DIR = Path(tmp)
    live.load_key = lambda: key
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app" / "app.py"), default_timeout=900)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    if setup:
        setup(at)
    return at


def to_live(a):
    a.radio(key="earth_source").set_value(LIVE_SRC).run()


def metric(a, label):
    return int([m for m in a.metric if m.label == label][0].value.replace(",", ""))


def page_html(a):
    els = a.get("iframe")
    assert els, "no embedded Earth page"
    return els[-1].proto.srcdoc


def demo_files():
    """Raw files of the demo snapshots: the three original regions and Brazil, a new region that is small.
    The other region files are big and add nothing to what the app tests check."""
    from src.region_eval import REGIONS
    keep = ("South Asia", "Southeast Asia", "North and Central America", "Brazil")
    return {f: r for f, r in REGIONS.items() if r in keep}


def build_demo(tmp):
    from src import live_demo
    from src.config import RAW_DIR
    FILES = demo_files()
    live.LIVE_DIR = Path(tmp)
    for fname, region in FILES.items():
        live_demo.build(region, RAW_DIR / fname)
    return list(FILES.values())


def test_earth_select():
    """The Show helpers on a small frame: error counts add up and categories match the legend."""
    from src import earth
    df = pd.DataFrame({"pred_static": [1, 1, 0, 0, 1, 0], "is_static": [1, 0, 0, 1, 0, 0]})
    fs, ff = earth.error_counts(df)
    assert (fs, ff) == (2, 1)
    rows, kinds = earth.select(df, "Errors only")
    assert len(rows) == fs + ff and sorted(kinds.tolist()) == [2, 2, 3]
    assert kinds.tolist() == [2, 3, 2]
    assert earth.select(df, "Ground truth")[1].tolist() == df["is_static"].tolist()
    assert earth.select(df, "Model prediction")[1].tolist() == df["pred_static"].tolist()
    assert set(earth.select(df, "Raw detections")[1].tolist()) == {0}
    print("show helper test passed")


def test_earth_focus():
    """The Fires, Static sources, Both selector: rows kept, uncertain chip, views it ignores, legend counts."""
    from src import earth
    df = pd.DataFrame({"pred_static": [1, 1, 0, 0, 1, 0], "is_static": [1, 0, 0, 1, 0, 0]})
    assert earth.FOCUS[0] == "Fires"                                         # Fires is the default choice
    kinds = np.array([0, 1, 4, 0, 1, 4])                                     # fire, static, uncertain
    assert earth.focus_codes("Model prediction", "Fires") == {0, 4}           # uncertain kept by default
    assert earth.focus_codes("Model prediction", "Fires", include_uncertain=False) == {0}
    assert earth.focus_codes("Model prediction", "Static sources") == {1}
    assert earth.focus_codes("Model prediction", "Both") is None
    for show in ("Raw detections", "Errors only"):                           # the selector does nothing there
        for f in earth.FOCUS:
            assert earth.focus_codes(show, f) is None
    rows, k = earth.apply_focus(df, kinds, {0, 4})
    assert rows.index.tolist() == [0, 2, 3, 5] and k.tolist() == [0, 4, 0, 4]
    rows, k = earth.apply_focus(df, kinds, {1})
    assert rows.index.tolist() == [1, 4]
    rows, k = earth.apply_focus(df, kinds, None)
    assert len(rows) == len(df) and k.tolist() == kinds.tolist()
    # Ground truth uses the NASA type: Fires keeps NASA fire, Static sources keeps NASA static
    truth_rows, truth_k = earth.select(df, "Ground truth")
    rows, _ = earth.apply_focus(truth_rows, truth_k, earth.focus_codes("Ground truth", "Static sources"))
    assert rows["is_static"].tolist() == [1, 1]
    # the legend follows the selection and counts every drawn row
    lg = earth.legend("Model prediction", True, {0, 4}, np.array([0, 0, 4]))
    assert [e[2] for e in lg] == ["Predicted fire (2)", "Uncertain (needs investigation) (1)"]
    lg = earth.legend("Model prediction", False, {1}, np.array([1, 1, 1]))
    assert [e[2] for e in lg] == ["Predicted static (3)"]
    assert len(earth.legend("Model prediction", False)) == 2                # unchanged without a selection
    print("fire or static selector test passed")


def test_earth_footprint():
    """Click footprint: the page gets the pixel size per point when the data has scan and track, none otherwise,
    and carries the click, footprint and Escape code."""
    from src import earth
    n = 5
    base = pd.DataFrame({
        "longitude": np.linspace(70, 71, n), "latitude": np.linspace(30, 31, n), "p_static": np.full(n, 0.1, np.float32),
        "pred_static": [0, 1, 0, 0, 1], "is_static": [0, 1, 0, 0, 0], "bright_ti4": np.full(n, 330.0, np.float32),
        "frp": np.full(n, 2.4, np.float32), "cc": ["PK"] * n, "admin1": ["Punjab"] * n,
        "acq_date": pd.to_datetime(["2025-02-06"] * n)})
    kinds = np.zeros(n, dtype=int)
    data, _ = earth.payload(base, kinds)
    assert data["sc"] == [] and data["tr"] == []                            # archive: nominal pixel in the page
    assert data["p"][0] == 0.1 and data["t"][0] == 330.0                     # float32 values reach the page rounded
    live_like = base.assign(scan=np.array([0.39, 0.52, 0.375, 0.8, 0.41], np.float32),
                            track=np.array([0.36, 0.42, 0.375, 0.65, 0.37], np.float32))
    data, _ = earth.payload(live_like, kinds)
    assert data["sc"] == [0.39, 0.52, 0.375, 0.8, 0.41] and data["tr"] == [0.36, 0.42, 0.375, 0.65, 0.37]
    page = earth.html(data, None, {"sel": None, "provs": None, "prov": None}, "Model prediction", "s")
    for hook in ("function footprint(", "function pixelLabel(", "square kilometres", "Escape", "fpline", "selected:"):
        assert hook in page, hook
    assert earth.MAX_POINTS == 20_000                                        # the cap is unchanged
    print("click footprint test passed")


def test_app_live(tmp):
    """Live source of the Earth tab: notices, key handling, offline, model choice, regions."""
    live.LIVE_DIR = Path(tmp)
    live.write_snapshot(REGION, two_satellites(60))        # a genuine snapshot: no is_demo flag

    # Offline with a snapshot and no key
    at = None                     # free the previous app run first (memory)
    gc.collect()
    at = run_app(tmp, None, to_live)
    assert not at.exception, at.exception
    texts = [w.value for w in at.warning]
    assert any("No FIRMS map key" in t for t in texts), texts
    assert not any("2 November 2026" in t or "delivery ended" in t for t in texts), texts
    assert not any(e.value == DEMO_NOTICE for e in at.error), [e.value for e in at.error]   # real, not demo
    assert any("Near real time" in i.value and "no ground truth" in i.value for i in at.info)
    caps = [c.value for c in at.caption]
    assert any("trained on Suomi NPP and validated on NOAA-20 and NOAA-21 archive data" in c and
               "has not been validated against the archive" in c for c in caps), caps
    assert any("Rows per satellite: NOAA-20" in c and "NOAA-21" in c and "Acquisition dates" in c for c in caps), caps
    assert any("Last fetch" in c for c in caps), caps
    assert KEY not in str(at)
    labels = [m.label for m in at.metric]
    for want in ("Detections in Earth", "Predicted fire", "Predicted static", "Circles drawn"):
        assert want in labels, labels
    assert any("data age" in c.value for c in at.caption)
    assert at.button(key="earth_refresh").proto.disabled

    # Key present and offline ticked
    def tick(a):
        to_live(a)
        a.checkbox(key="earth_offline").check().run()
    at = None                     # free the previous app run first (memory)
    gc.collect()
    at = run_app(tmp, KEY, tick)
    assert not at.exception, at.exception
    assert at.button(key="earth_refresh").proto.disabled
    assert KEY not in str(at)

    # Instant model and a country filter
    def instant(a):
        to_live(a)
        a.radio(key="earth_live_tier").set_value(INSTANT_TIER).run()
        sb = a.selectbox(key="earth_l_country")
        sb.set_value(sb.options[1]).run()
    at = None                     # free the previous app run first (memory)
    gc.collect()
    at = run_app(tmp, None, instant)
    assert not at.exception, at.exception

    # Region outside training: banner when a snapshot exists
    live.write_snapshot("Europe", fake_detections(60))

    def europe(a):
        to_live(a)
        a.selectbox(key="earth_region").set_value("Europe").run()
    at = None                     # free the previous app run first (memory)
    gc.collect()
    at = run_app(tmp, None, europe)
    assert not at.exception, at.exception
    assert any("unvalidated" in w.value for w in at.warning)

    # Region without a snapshot
    def other(a):
        to_live(a)
        a.selectbox(key="earth_region").set_value("Southeast Asia").run()
    at = None                     # free the previous app run first (memory)
    gc.collect()
    at = run_app(tmp, None, other)
    assert not at.exception and any("No data for this region" in i.value for i in at.info)
    print("earth live source test passed")


def test_app_store(tmp):
    """Live source from the incremental store: freshness, 6/24/72 hour window, no fetch controls, unvalidated
    label, untrained region marking."""
    from src import ingest
    from tests import fake_feed
    live.LIVE_DIR = Path(tmp)
    ingest.run_once("fake", fetch=fake_feed.fetch_factory(fake_feed.masters(10)), sleep=lambda s: None,
                    log=lambda *a: None, limiter=fake_feed.fast_limiter())
    # rows of an untrained region: the same fake detections moved into the Europe box
    eu = live.REGIONS["Europe"]
    extra = []
    for sensor, source in live.LIVE_SOURCES.items():
        m = fake_feed.master(6, seed=7)
        m["latitude"] = eu[1] + 1 + (m["latitude"] - m["latitude"].min()) / (m["latitude"].max() - m["latitude"].min() + 1e-9) * 20
        m["longitude"] = eu[0] + 1 + (m["longitude"] - m["longitude"].min()) / (m["longitude"].max() - m["longitude"].min() + 1e-9) * 30
        extra.append(ingest.standardize(m, sensor, "Europe"))
    ingest.run_once(fetch=None, extra=extra, sleep=lambda s: None, log=lambda *a: None)
    store = ingest.read_store()
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    at = None
    gc.collect()
    at = run_app(tmp, None, to_live)
    assert not at.exception, at.exception
    caps = [c.value for c in at.caption]
    assert any("Live store, last ingest" in c and "Live results are unvalidated" in c for c in caps), caps
    assert any("reads the store again every 10 minutes" in c for c in caps)
    from src import earth as earth_module
    assert earth_module.SHOW_HELP in caps, caps                                     # the plain language help line
    win = store[store["timestamp"] >= now - pd.Timedelta(hours=24)]
    n20, n21 = (int((win["sensor"] == s).sum()) for s in ("NOAA-20", "NOAA-21"))
    fresh = [c for c in caps if c.startswith("Newest detection")]
    assert fresh and "UTC" in fresh[0] and "minutes ago" in fresh[0] and "Rows per sensor" in fresh[0], caps
    assert f"NOAA-20 {n20:,}" in fresh[0] and f"NOAA-21 {n21:,}" in fresh[0], (fresh, n20, n21)
    assert [r.label for r in at.radio if r.key == "earth_hours"] == ["Time window"]
    assert [o for o in at.radio(key="earth_hours").options] == ["last 6 hours", "last 24 hours", "last 72 hours"]
    assert at.radio(key="earth_hours").value == 24                                        # the default is 24 hours
    assert not any(b.key == "earth_refresh" for b in at.button)                 # no fetch button in store mode
    assert not any(e.value == DEMO_NOTICE for e in at.error)
    assert any("validated" in c.value for c in at.caption)
    assert metric(at, "Detections in Earth") > 0 and metric(at, "Circles drawn") > 0
    assert any("not in the training data" in c for c in caps)                      # untrained rows are marked
    for tier in (INSTANT_TIER, HISTORY_TIER):
        at.radio(key="earth_live_tier").set_value(tier).run()
        assert not at.exception, at.exception
        assert metric(at, "Detections in Earth") > 0
    seen = {}
    for hours in (6, 24, 72):
        at.radio(key="earth_hours").set_value(hours).run()
        assert not at.exception, at.exception
        seen[hours] = metric(at, "Detections in Earth")
    assert seen[6] <= seen[24] <= seen[72] and seen[72] > 0, seen
    # an untrained region is flagged, a trained region is not
    at.selectbox(key="earth_region").set_value("Europe").run()
    assert not at.exception, at.exception
    assert any("Region not in the training data, results unvalidated" in w.value for w in at.warning),         [w.value for w in at.warning]
    at.selectbox(key="earth_region").set_value("Brazil").run()
    assert not at.exception and any("no detections for this selection" in i.value for i in at.info)
    assert not any("not in the training data" in w.value for w in at.warning)
    # without the store the earlier behaviour returns
    (Path(tmp) / "store.parquet").unlink()
    at = None
    gc.collect()
    at = run_app(tmp, None, to_live)
    assert not at.exception and any("No data for this region" in i.value for i in at.info)
    print("earth live store test passed")


def test_app_demo(tmp):
    """Demo snapshots built from the archive files, loaded offline in the Earth tab."""
    import time

    from src.config import RAW_DIR
    from src import live_demo
    FILES = demo_files()
    regions = build_demo(tmp)
    for fname, region in FILES.items():
        assert live.snapshot_is_demo(region)
        raw, fetched = live.load_snapshot(region)
        newest = pd.to_datetime(raw["acq_date"]).max().date()
        assert newest == (pd.Timestamp.now(tz="UTC").tz_localize(None) - pd.Timedelta(days=1)).date()
        assert "type" not in raw.columns and "is_demo" not in raw.columns

    keys = {"Continent": "earth_l_continent", "Country": "earth_l_country", "Province or state": "earth_l_province"}

    def pick(a, label, i):
        sb = a.selectbox(key=keys[label])
        if len(sb.options) <= i:
            return None
        sb.set_value(sb.options[i]).run()
        return metric(a, "Detections in Earth")

    for region in regions:
        t = time.time()

        def setup(a, r=region):
            to_live(a)
            a.selectbox(key="earth_region").set_value(r).run()
        at = None                     # free the previous app run first (memory)
        gc.collect()
        at = run_app(tmp, None, setup)
        took = time.time() - t
        assert not at.exception, at.exception
        assert any(e.value == DEMO_NOTICE for e in at.error), [e.value for e in at.error]
        assert not any("hours old" in w.value for w in at.warning)
        base = metric(at, "Detections in Earth")
        out = {"region": region, "detections": base, "seconds": round(took, 1)}
        for label, i in (("Continent", 1), ("Country", 1), ("Province or state", 1)):
            n = pick(at, label, i)
            assert n is not None
            out[label] = n
            assert 0 < n <= base
        assert out["Country"] < base, out
        print(out)
    print("earth demo snapshot test passed")


def points_drawn(at):
    """Points sent to the browser by every pydeck map on the page, from the deck specs."""
    import json
    out = []
    for el in at.get("deck_gl_json_chart"):
        spec = json.loads(el.proto.json)
        out.append(sum(len(layer["data"]) for layer in spec["layers"] if layer["@@type"] == "ScatterplotLayer"))
    return out


def test_app_geo(tmp):
    """Four filters on the Map tab and on the Earth tab (archive and live), defaults and resets."""
    import time

    regions = build_demo(tmp)

    def set_(a, key, value):
        a.selectbox(key=key).set_value(value).run()
        assert not a.exception, a.exception

    def check_filters(a, prefix, label, name):
        base = metric(a, label)
        assert a.selectbox(key=f"{prefix}_continent").value == "World"
        assert a.selectbox(key=f"{prefix}_country").value == "All countries"
        assert a.selectbox(key=f"{prefix}_province").value == "All provinces"
        assert a.selectbox(key=f"{prefix}_city").value == "All cities"
        found = False
        for opt in a.selectbox(key=f"{prefix}_country").options[1:8]:
            set_(a, f"{prefix}_country", opt)
            n_country = metric(a, label)
            provinces = a.selectbox(key=f"{prefix}_province").options
            if n_country < base and len(provinces) >= 3:
                set_(a, f"{prefix}_province", provinces[1])
                n_prov = metric(a, label)
                assert 0 < n_prov < n_country, (opt, n_prov, n_country)
                set_(a, f"{prefix}_city", a.selectbox(key=f"{prefix}_city").options[1])
                n_city = metric(a, label)
                assert 0 < n_city <= n_prov
                set_(a, f"{prefix}_country", "All countries")           # reset
                assert metric(a, label) == base, (metric(a, label), base)
                assert a.selectbox(key=f"{prefix}_province").value == "All provinces"
                print(f"  {name} {prefix}: all {base:,} -> {opt} {n_country:,} -> province {n_prov:,} "
                      f"-> city {n_city:,} -> reset {base:,}")
                found = True
                break
            set_(a, f"{prefix}_country", "All countries")
        assert found, f"{name}: no country with two provinces and fewer detections than the region"

    for region in regions:
        t = time.time()

        def setup(a, r=region):
            to_live(a)
            a.selectbox(key="earth_region").set_value(r).run()
        at = None                     # free the previous app run first (memory)
        gc.collect()
        at = run_app(tmp, None, setup)
        assert not at.exception, at.exception
        assert at.metric, ([e.value for e in at.error], [i.value for i in at.info][-3:])
        print(region, "Earth circles drawn:", metric(at, "Circles drawn"), f"({time.time() - t:.0f} s)")
        check_filters(at, "earth_l", "Detections in Earth", region)

    # Map tab filters (archive data, once) and the Earth tab archive filters, both tiers
    at = None                     # free the previous app run first (memory)
    gc.collect()
    at = run_app(tmp, None)
    check_filters(at, "map", "Detections shown", "archive")
    print("Map tab pydeck points drawn:", points_drawn(at))
    for tier in (INSTANT_TIER, HISTORY_TIER):
        at.sidebar.radio[0].set_value(tier).run()
        assert not at.exception, (tier, at.exception)
        assert metric(at, "Detections shown") > 0
        check_filters(at, "earth_a", "Detections in Earth", f"archive {tier}")
    print("geo filter tests passed")


def test_app_earth(tmp):
    """Earth tab: Show options, counts, errors only, ground truth disabled on live, cap, page content."""
    from src import earth
    build_demo(tmp)

    def circles_expected(n):
        return min(n, earth.MAX_POINTS)

    at = None                     # free the previous app run first (memory)
    gc.collect()
    at = run_app(tmp, None)
    assert not at.exception, at.exception
    for tier in (INSTANT_TIER, HISTORY_TIER):
        at.sidebar.radio[0].set_value(tier).run()
        assert not at.exception, (tier, at.exception)
        note = any("mostly separates night detections" in c.value and "prefer the history tier" in c.value
                   for c in at.caption)
        assert note == (tier == INSTANT_TIER), (tier, note)           # only with the instant tier
        base = metric(at, "Detections in Earth")
        assert at.radio(key="earth_show_a").options == earth.SHOW
        got = {}
        if tier == INSTANT_TIER:
            assert at.radio(key="earth_show_a").value == "Model prediction"      # the default view
            assert at.radio(key="earth_focus_a").value == "Fires"                # the default selection
            fires = metric(at, "Circles drawn")
            assert fires <= circles_expected(base)
            assert "Predicted static (" not in page_html(at)                     # legend follows the selection
            at.radio(key="earth_focus_a").set_value("Static sources").run()
            assert not at.exception, at.exception
            static_html = page_html(at) if at.get("iframe") else ""
            if static_html:
                assert "Predicted fire (" not in static_html and "Predicted static (" in static_html
            else:                                                                # the band calls nothing static here
                assert any("No detections match this selection." in i.value for i in at.info)
            for show in ("Raw detections", "Errors only"):                       # the selector does nothing there
                at.radio(key="earth_show_a").set_value(show).run()
                assert at.radio(key="earth_focus_a").proto.disabled
                assert any(f"Not used in {show}" in c.value for c in at.caption)
            at.radio(key="earth_show_a").set_value("Model prediction").run()
            print(f"  selector: fires drawn {fires:,}")
        at.radio(key="earth_focus_a").set_value("Both").run()                    # every class for the checks below
        for show in earth.SHOW:
            at.radio(key="earth_show_a").set_value(show).run()
            assert not at.exception, (show, at.exception)
            has_band = any(r.key == "earth_band_a" for r in at.radio)
            assert has_band == (show == "Model prediction"), (show, has_band)   # Band only with predictions
            assert any("sidebar threshold only affects the Map tab" in c.value for c in at.caption)
            got[show] = metric(at, "Circles drawn")
            html = page_html(at)
            assert f"maplibre-gl@{earth.MAPLIBRE}" in html and "terrarium" in html and "World_Imagery" in html
            assert "Esri" in html and "OpenStreetMap" in html           # attribution in the footer
            for _, _, label in earth.legend(show, show == "Model prediction"):
                assert label in html, (show, label)                    # legend of this option
            assert "Decision: " in html                                # three way decision in the popup
            if show == "Model prediction":
                assert "Uncertain (needs investigation)" in html and "Uncertain" in [m.label for m in at.metric]
            else:
                assert "Uncertain (needs investigation)" not in html
            assert metric(at, "Detections in Earth") == base
        fs, ff, wrong = (metric(at, k) for k in ("False static", "False fire", "All errors"))
        assert fs + ff == wrong > 0
        assert got["Raw detections"] == got["Model prediction"] == got["Ground truth"] == circles_expected(base)
        assert got["Errors only"] == circles_expected(wrong), (got, wrong)
        assert any("NASA's own repeat based static flag" in c.value for c in at.caption)
        print(f"  {tier}: detections {base:,}, circles per option {got}, false static {fs:,}, "
              f"false fire {ff:,}, errors {wrong:,}")
        # a country filter changes the counts under Errors only too
        c = at.selectbox(key="earth_a_country")
        c.set_value(c.options[1]).run()
        assert not at.exception, at.exception
        n_country = metric(at, "Detections in Earth")
        assert n_country < base
        assert metric(at, "False static") + metric(at, "False fire") == metric(at, "All errors") <= wrong
        at.selectbox(key="earth_a_country").set_value("All countries").run()
        assert metric(at, "Detections in Earth") == base
        at.radio(key="earth_show_a").set_value("Model prediction").run()

    # Live source: ground truth and errors are disabled, with the reason in a caption
    to_live(at)
    assert not at.exception, at.exception
    assert at.radio(key="earth_show_l").options == earth.SHOW[:2]
    assert any("no ground truth on live data" in c.value for c in at.caption)
    assert not any(m.label in ("False static", "False fire", "NASA static") for m in at.metric)
    base = metric(at, "Detections in Earth")
    assert at.radio(key="earth_focus_l").value == "Fires"                        # the selector applies to live too
    assert metric(at, "Circles drawn") <= circles_expected(base)
    at.radio(key="earth_focus_l").set_value("Both").run()
    for show in earth.SHOW[:2]:
        at.radio(key="earth_show_l").set_value(show).run()
        assert not at.exception, at.exception
        assert metric(at, "Circles drawn") == circles_expected(base)
        html = page_html(at)
        assert "NASA label" in html                                    # popup code is shared, value is absent
        assert '"n":[-1' in html                                       # live points carry no NASA label
    print("earth test passed")


def peak_memory():
    """Peak working set and peak commit of this process in MB (Windows)."""
    import ctypes
    from ctypes import wintypes as w

    class PMC(ctypes.Structure):
        _fields_ = [("cb", w.DWORD), ("faults", w.DWORD), ("peak_ws", ctypes.c_size_t),
                    ("ws", ctypes.c_size_t), ("a", ctypes.c_size_t), ("b", ctypes.c_size_t),
                    ("c", ctypes.c_size_t), ("d", ctypes.c_size_t), ("commit", ctypes.c_size_t),
                    ("peak_commit", ctypes.c_size_t)]
    c = PMC()
    c.cb = ctypes.sizeof(c)
    k, p = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
    k.GetCurrentProcess.restype = w.HANDLE
    p.GetProcessMemoryInfo.argtypes = [w.HANDLE, ctypes.c_void_p, w.DWORD]
    p.GetProcessMemoryInfo(k.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return c.peak_ws / 2 ** 20, c.peak_commit / 2 ** 20


def suite_unit(tmp):
    test_fetch_chunk()
    test_refresh_store(tmp)
    test_key_not_printed_in_refresh(tmp)
    test_history_per_satellite(tmp)
    test_live_check_helpers()
    test_leakage_rolling_store(tmp)
    test_geography(tmp)
    test_earth_select()
    test_earth_focus()
    test_earth_footprint()


SUITES = {"unit": suite_unit, "app_store": test_app_store, "app_live": test_app_live, "app_demo": test_app_demo, "app_geo": test_app_geo,
          "app_earth": test_app_earth}


def main(names=None):
    """Run suites in this process. Run one suite per process (python -m tests.test_live app_geo)
    so the memory of the AppTest suites does not stack up."""
    saved = (live.LIVE_DIR, live.load_key)
    try:
        for name in names or SUITES:
            with tempfile.TemporaryDirectory() as tmp:
                SUITES[name](tmp)
    finally:
        live.LIVE_DIR, live.load_key = saved
    ws, commit = peak_memory()
    print(f"peak memory: working set {ws:.0f} MB, commit {commit:.0f} MB")


if __name__ == "__main__":
    import sys
    main(sys.argv[1:] or None)

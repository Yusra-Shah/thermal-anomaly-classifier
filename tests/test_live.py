"""Tests for src/live.py and the Live map tab. No network: the API is faked.

Run from the project root:  python -m tests.test_live
"""
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


def fake_fetch_factory(master, log):
    def fetch(key, box, start, days):
        log.append((pd.Timestamp(start).normalize(), days))
        lo = pd.Timestamp(start).normalize()
        hi = lo + pd.Timedelta(days=days - 1)
        d = pd.to_datetime(master["acq_date"])
        return master[(d >= lo) & (d <= hi)].copy()
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
    assert s.urls[0].endswith("/VIIRS_SNPP_NRT/61.0,24.0,76.0,36.0/5/2025-01-01"), s.urls[0]

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
    master = fake_detections(63, end=TODAY + pd.Timedelta(days=2))   # includes two future days
    log = []
    fetch = fake_fetch_factory(master, log)
    r = live.refresh(REGION, KEY, today=TODAY, fetch=fetch)
    assert r["requests"] == 12 and all(d <= 5 for _, d in log), log
    assert len(live.stored_days(REGION)) == 60
    store = live.load_store(REGION)
    assert pd.to_datetime(store["acq_date"]).max() == TODAY
    assert not store.duplicated(["latitude", "longitude", "acq_date", "acq_time"]).any()
    snap, fetched = live.load_snapshot(REGION)
    assert len(snap) == len(store) and fetched is not None

    log.clear()
    r = live.refresh(REGION, KEY, today=TODAY, fetch=fetch)         # only the latest 3 days
    assert r["requests"] == 1 and log == [(TODAY - pd.Timedelta(days=2), 3)], log

    log.clear()
    live.refresh(REGION, KEY, today=TODAY + pd.Timedelta(days=2), fetch=fetch)
    days = live.stored_days(REGION)
    assert len(days) == 60 and min(days) == TODAY - pd.Timedelta(days=57), (len(days), min(days))
    assert log == [(TODAY, 3)], log                              # the latest 3 days only
    print("rolling store test passed")


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


def run_app(tmp, key, setup=None):
    from streamlit.testing.v1 import AppTest
    import streamlit as st
    st.cache_data.clear()                      # keep memory flat across many AppTest runs
    st.cache_resource.clear()
    live.LIVE_DIR = Path(tmp)
    live.load_key = lambda: key
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app" / "app.py"), default_timeout=900)
    at.run()
    if setup:
        setup(at)
    return at


def test_app_live(tmp):
    from src.history import add_history  # noqa: F401  (import check only)
    live.LIVE_DIR = Path(tmp)
    live.write_snapshot(REGION, fake_detections(60))

    # Offline with a snapshot and no key
    at = run_app(tmp, None)
    assert not at.exception, at.exception
    texts = [w.value for w in at.warning]
    assert any("No FIRMS map key" in t for t in texts), texts
    assert KEY not in str(at)
    labels = [m.label for m in at.metric]
    for want in ("Detections", "Predicted fire", "Predicted static", "Latest detection (UTC)", "Data age"):
        assert want in labels, labels
    refresh = [b for b in at.button if b.label == "Refresh data"][0]
    assert refresh.proto.disabled

    # Key present and offline ticked
    def tick(a):
        [c for c in a.checkbox if c.label == "Offline (saved snapshot)"][0].check().run()
    at = run_app(tmp, KEY, tick)
    assert not at.exception, at.exception
    assert [b for b in at.button if b.label == "Refresh data"][0].proto.disabled

    # Instant model and a country filter
    def instant(a):
        [r for r in a.radio if r.label == "Live model"][0].set_value("Instant (single reading)").run()
        sb = a.selectbox(key="live_country")
        sb.set_value(sb.options[1]).run()
    at = run_app(tmp, None, instant)
    assert not at.exception, at.exception

    # Region outside training: banner when a snapshot exists
    live.write_snapshot("Europe (not in training)", fake_detections(60))

    def europe(a):
        [s for s in a.selectbox if s.label == "Region loaded"][0].set_value("Europe (not in training)").run()
    at = run_app(tmp, None, europe)
    assert not at.exception, at.exception
    assert any("unvalidated" in w.value for w in at.warning)

    # Region without a snapshot
    def other(a):
        [s for s in a.selectbox if s.label == "Region loaded"][0].set_value("Southeast Asia").run()
    at = run_app(tmp, None, other)
    assert not at.exception and any("No data for this region" in i.value for i in at.info)
    print("app live tab test passed")


def test_app_demo(tmp):
    """Demo snapshots built from the archive files, loaded offline in the app."""
    import time

    from src import live_demo
    from src.config import RAW_DIR
    from src.region_eval import REGIONS as FILES
    live.LIVE_DIR = Path(tmp)
    for fname, region in FILES.items():
        live_demo.build(region, RAW_DIR / fname)
        assert live.snapshot_is_demo(region)
        raw, fetched = live.load_snapshot(region)
        newest = pd.to_datetime(raw["acq_date"]).max().date()
        assert newest == (pd.Timestamp.now(tz="UTC").tz_localize(None) - pd.Timedelta(days=1)).date()
        assert "type" not in raw.columns and "is_demo" not in raw.columns

    def count(a):
        return int([m for m in a.metric if m.label == "Detections"][0].value.replace(",", ""))

    keys = {"Continent": "live_continent", "Country": "live_country", "Province or state": "live_province"}

    def pick(a, label, i):
        sb = a.selectbox(key=keys[label])
        if len(sb.options) <= i:
            return None
        sb.set_value(sb.options[i]).run()
        return count(a)

    for region in FILES.values():
        t = time.time()
        at = run_app(tmp, None, lambda a, r=region: [x for x in a.selectbox if x.label == "Region loaded"][0]
                     .set_value(r).run())
        took = time.time() - t
        assert not at.exception, at.exception
        notice = "Demo snapshot: historical 2025 data shifted to look recent, for testing only. "                  "This is not current data."
        assert any(e.value == notice for e in at.error), [e.value for e in at.error]
        assert not any("hours old" in w.value for w in at.warning)
        base = count(at)
        out = {"region": region, "detections": base, "seconds": round(took, 1)}
        for label, i in (("Continent", 1), ("Country", 1), ("Province or state", 1)):
            n = pick(at, label, i)
            assert n is not None
            out[label] = n
            assert 0 < n <= base
        assert out["Country"] < base, out
        print(out)
    print("app demo snapshot test passed")


def points_drawn(at):
    """Points sent to the browser by every map on the page, from the deck specs."""
    import json
    out = []
    for el in at.get("deck_gl_json_chart"):
        spec = json.loads(el.proto.json)
        n = sum(len(layer["data"]) for layer in spec["layers"] if layer["@@type"] == "ScatterplotLayer")
        out.append(n)
    return out


def test_app_geo(tmp):
    """Four filters on the Map tab and the Live map tab, both tiers, defaults and resets."""
    import time

    from src import live_demo
    from src.config import RAW_DIR
    from src.region_eval import REGIONS as FILES
    live.LIVE_DIR = Path(tmp)
    for fname, region in FILES.items():
        live_demo.build(region, RAW_DIR / fname)

    def count(a, label):
        return int([m for m in a.metric if m.label == label][0].value.replace(",", ""))

    def set_(a, key, value):
        a.selectbox(key=key).set_value(value).run()
        assert not a.exception, a.exception

    def check_filters(a, prefix, label, name):
        base = count(a, label)
        a.selectbox(key=f"{prefix}_country")
        assert a.selectbox(key=f"{prefix}_continent").value == "World"
        assert a.selectbox(key=f"{prefix}_country").value == "All countries"
        assert a.selectbox(key=f"{prefix}_province").value == "All provinces"
        assert a.selectbox(key=f"{prefix}_city").value == "All cities"
        found = False
        for opt in a.selectbox(key=f"{prefix}_country").options[1:8]:
            set_(a, f"{prefix}_country", opt)
            n_country = count(a, label)
            provinces = a.selectbox(key=f"{prefix}_province").options
            if n_country < base and len(provinces) >= 3:
                set_(a, f"{prefix}_province", provinces[1])
                n_prov = count(a, label)
                assert 0 < n_prov < n_country, (opt, n_prov, n_country)
                set_(a, f"{prefix}_city", a.selectbox(key=f"{prefix}_city").options[1])
                n_city = count(a, label)
                assert 0 < n_city <= n_prov
                set_(a, f"{prefix}_country", "All countries")           # reset
                assert count(a, label) == base, (count(a, label), base)
                assert a.selectbox(key=f"{prefix}_province").value == "All provinces"
                print(f"  {name} {prefix}: all {base:,} -> {opt} {n_country:,} -> province {n_prov:,} "
                      f"-> city {n_city:,} -> reset {base:,}")
                found = True
                break
            set_(a, f"{prefix}_country", "All countries")
        assert found, f"{name}: no country with two provinces and fewer detections than the region"

    for region in FILES.values():
        t = time.time()
        at = run_app(tmp, None, lambda a, r=region: a.selectbox(key="live_region").set_value(r).run())
        assert not at.exception, at.exception
        print(region, "default drawn points per map:", points_drawn(at), f"({time.time() - t:.0f} s)")
        check_filters(at, "live", "Detections", region)
        check_filters(at, "map", "Detections shown", region)

    # Both tiers, with the Live map reading the same snapshot, and an upload
    at = run_app(tmp, None)
    for tier in ("Instant (single reading)", "With history (past detections)"):
        at.sidebar.radio[0].set_value(tier).run()
        assert not at.exception, (tier, at.exception)
        assert count(at, "Detections shown") > 0
    print("geo filter tests passed")


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
    test_leakage_rolling_store(tmp)
    test_geography(tmp)


SUITES = {"unit": suite_unit, "app_live": test_app_live, "app_demo": test_app_demo, "app_geo": test_app_geo}


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

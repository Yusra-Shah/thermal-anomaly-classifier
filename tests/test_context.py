"""Tests for the panel sources: weather, land and fuel, people and buildings, response, and their combination.
Every request is a fake, so nothing here touches the network.

Run from the project root, one suite per process:  python -m pytest tests/test_context.py -q
"""
import io
import json

import numpy as np
import pandas as pd
import pytest

from src.context import diskcache, land, landcover, overpass, people, response, sections, weather
from tests.test_landcover import NAME, X0, Y0, Fake, image, make_tiff, pixel_lonlat


class Resp:
    def __init__(self, body=None, status=200, content=b"", headers=None):
        self.body, self.status_code, self.content, self.headers = body, status, content, headers or {}

    def json(self):
        return self.body

    def iter_content(self, n):
        for i in range(0, len(self.content), n):
            yield self.content[i:i + n]


class Counter:
    """A fake get that counts calls and never shows an address."""

    def __init__(self, fn):
        self.fn, self.calls = fn, []

    def __call__(self, url, params=None, **kw):
        self.calls.append(params)
        return self.fn(params)


# ------------------------------------------------------------------ disk cache and requests
def test_cache_roundtrip_and_age(tmp_path):
    assert diskcache.read("x", "a b/c", root=tmp_path) is None
    diskcache.write("x", "a b/c", {"v": 1}, root=tmp_path)
    assert diskcache.read("x", "a b/c", root=tmp_path) == {"v": 1}
    assert diskcache.read("x", "a b/c", ttl=-1, root=tmp_path) is None
    assert not list(tmp_path.rglob("*.tmp"))


def test_http_get_carries_no_address(monkeypatch):
    import requests
    monkeypatch.setattr(requests, "get", lambda url, **kw: (_ for _ in ()).throw(
        requests.ConnectionError(f"failed for {url} key=SECRET")))
    with pytest.raises(RuntimeError) as e:
        diskcache.http_get("https://example.org/x?key=SECRET")
    assert str(e.value) == "ConnectionError"
    monkeypatch.setattr(requests, "get", lambda url, **kw: Resp(status=429))
    with pytest.raises(RuntimeError) as e:
        diskcache.http_get("https://example.org/x")
    assert str(e.value) == "HTTP 429"
    seen = {}

    def spy(url, params=None, timeout=None, headers=None):
        seen.update(timeout=timeout, headers=headers)
        return Resp({"ok": 1})
    monkeypatch.setattr(requests, "get", spy)
    diskcache.http_get("https://example.org/x")
    assert seen["timeout"] == 8 and "thermal-anomaly" in seen["headers"]["User-Agent"]


# ------------------------------------------------------------------ weather
def series(now, hours=96):
    times = pd.date_range(now.floor("h") - pd.Timedelta(hours=hours - 1 - 12), periods=hours, freq="h")
    n = len(times)
    return {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in times], "temperature_2m": [20.0 + i * 0.1 for i in range(n)],
            "relative_humidity_2m": [40] * n, "wind_speed_10m": [10.0] * n, "wind_direction_10m": [315.0] * n,
            "wind_gusts_10m": [25.0] * n, "precipitation": [0.5 if i % 2 else 0.0 for i in range(n)]}


NOW = pd.Timestamp("2026-11-20 12:30")


def test_weather_at_the_hour_of_the_detection(tmp_path):
    get = Counter(lambda p: Resp({"hourly": series(NOW)}))
    t = pd.Timestamp("2026-11-20 06:17")
    w = weather.at(32.0, 74.5, t, get=get, cache_dir=tmp_path, now=NOW)
    assert w["time"] == "2026-11-20 06:00:00" and w["wind_from"] == "NW" and w["wind_to"] == "SE"
    assert w["wind_kmh"] == 10.0 and w["gust_kmh"] == 25.0 and w["humidity_pct"] == 40
    assert w["rain_24h_mm"] == 6.0 and w["rain_72h_mm"] == pytest.approx(18.0, abs=0.6)
    assert w["source"].startswith("Open-Meteo") and not w["cached"]
    assert get.calls[0]["past_days"] == 3 and get.calls[0]["latitude"] == 32.0
    again = weather.at(32.01, 74.51, t, get=get, cache_dir=tmp_path, now=NOW)         # same 0.1 degree cell
    assert len(get.calls) == 1 and again["cached"] and again["fetched_at"] == w["fetched_at"]


def test_weather_old_detection_and_failures_say_why(tmp_path):
    get = Counter(lambda p: Resp({"hourly": series(NOW)}))
    old = weather.at(32.0, 74.5, pd.Timestamp("2026-11-10 06:00"), get=get, cache_dir=tmp_path, now=NOW)
    assert "older" in old["unavailable"] and not get.calls
    import requests

    def boom(url, params=None, **kw):
        raise RuntimeError("ReadTimeout")
    w = weather.at(10.0, 20.0, pd.Timestamp("2026-11-20 06:00"), get=boom, cache_dir=tmp_path, now=NOW)
    assert w["unavailable"] == "the weather service did not answer (ReadTimeout)" and "://" not in str(w)
    assert weather.compass(0) == "N" and weather.compass(350) == "N" and weather.compass(135) == "SE"


# ------------------------------------------------------------------ land and fuel
def test_land_and_fuel_from_the_land_cover_module(tmp_path):
    fetch = Fake({NAME: make_tiff([image()])})
    lon, lat = pixel_lonlat(10, 20)
    d = land.describe(lat, lon, 0.375, 0.375, fetch=fetch, cache_dir=tmp_path)
    assert d["class"] == "Cropland" and d["group"] == "Cropland" and d["code"] == 40
    assert "crop fuel" in d["fuel"] and d["source"].startswith("ESA WorldCover") and "75 to 80" in d["note"]
    lon, lat = pixel_lonlat(100, 20)
    d = land.describe(lat, lon, fetch=fetch, cache_dir=tmp_path)
    assert d["class"] == "Tree cover" and d["group"] == "Natural land" and d["around_1km"]
    n = len(fetch.calls)
    land.describe(lat, lon, fetch=fetch, cache_dir=tmp_path)
    assert len(fetch.calls) == n                                                # the second look is from the cache


def test_land_unavailable_cases(tmp_path):
    import requests
    d = land.describe(10.0, 20.0, fetch=Fake({}), cache_dir=tmp_path / "a")           # open sea: a missing tile
    assert "open sea" in d["unavailable"]
    d = land.describe(10.0, 20.0, fetch=Fake({}, fail=requests.ConnectionError("https://x/secret")), cache_dir=tmp_path / "b")
    assert "could not be reached" in d["unavailable"] and "://" not in d["unavailable"] and "secret" not in str(d)


# ------------------------------------------------------------------ people and buildings
def write_grid(path, value=100.0, size=41, x0=70.0, y0=33.0, cell=0.0083333333):
    from PIL import Image
    a = np.full((size, size), value, np.float32)
    im = Image.fromarray(a, mode="F")
    im.save(path, tiffinfo={33550: (cell, cell, 0.0), 33922: (0.0, 0.0, 0.0, x0, y0, 0.0)})
    return path


def test_population_within_a_radius_is_area_weighted(tmp_path):
    write_grid(people.raster_file("PAK", tmp_path))
    lat, lon = 33.0 - 20.5 * 0.0083333333, 70.0 + 20.5 * 0.0083333333
    p = people.population("PK", lat, lon, cache_dir=tmp_path)
    cell_km2 = (0.0083333333 * 111.320 * np.cos(np.radians(lat))) * (0.0083333333 * 110.574)
    assert p["within_1km"] == pytest.approx(100 / cell_km2 * np.pi * 1.0, rel=0.06)
    assert p["within_5km"] == pytest.approx(100 / cell_km2 * np.pi * 25.0, rel=0.06)
    assert p["cell"] == 100 and p["resolution"] == "about 1 km" and "rough" in p["note"] and p["source"].startswith("WorldPop")
    out = people.population("PK", 40.0, 90.0, cache_dir=tmp_path)                     # outside the grid
    assert "outside" in out["unavailable"]
    assert "country not known" in people.population("", 1.0, 1.0, cache_dir=tmp_path)["unavailable"]
    assert people.iso3_of("PK") == "PAK" and people.iso3_of("NG") == "NGA" and people.iso3_of("XX") is None


def test_population_download_is_cached_limited_and_quiet(tmp_path):
    src = tmp_path / "src.tif"
    write_grid(src)
    data = src.read_bytes()
    calls = []

    def get(url, headers=None, stream=None, timeout=None):
        calls.append(timeout)
        return Resp(content=data, headers={"Content-Length": str(len(data))})
    f = people.download("PAK", get=get, cache_dir=tmp_path / "c")
    assert f.exists() and f.read_bytes() == data and calls == [8]
    people.download("PAK", get=get, cache_dir=tmp_path / "c")
    assert calls == [8]                                                              # the file is kept
    assert not list((tmp_path / "c").glob("*.tmp"))
    for bad, why in ((lambda *a, **k: Resp(status=404), "HTTP 404"),
                     (lambda *a, **k: Resp(content=b"x", headers={"Content-Length": str(people.MAX_BYTES + 1)}),
                      "population file too large"),
                     (lambda *a, **k: (_ for _ in ()).throw(ConnectionError("https://x/secret")), "ConnectionError")):
        with pytest.raises(RuntimeError) as e:
            people.download("NGA", get=bad, cache_dir=tmp_path / "d")
        assert str(e.value) == why and "://" not in str(e.value)
    assert not list((tmp_path / "d").glob("*"))
    out = people.population("NG", 10.0, 8.0, get=lambda *a, **k: Resp(status=404), cache_dir=tmp_path / "e")
    assert out == {"unavailable": "population grid not available (HTTP 404)"}


def test_nearest_place_uses_the_offline_list():
    hit = {"name": "Testville", "lat": "32.1", "lon": "74.5"}
    p = people.nearest_place(32.0, 74.5, searcher=lambda pts: [hit])
    assert p["name"] == "Testville" and p["direction"] == "N" and p["km"] == pytest.approx(11.1, abs=0.2)
    assert people.nearest_place(0.0, 0.0, searcher=lambda pts: (_ for _ in ()).throw(ValueError())) is None


def test_buildings_cached_and_a_down_service_is_not_asked_again(tmp_path):
    ok = Counter(lambda p: Resp({"elements": [{"tags": {"ways": "812", "nodes": "9"}}]}))
    b = people.buildings(32.0, 74.5, get=ok, cache_dir=tmp_path, sleep=lambda s: None)
    assert b["count"] == 812 and "ODbL" in b["source"] and "around:1000" in ok.calls[0]["data"]
    people.buildings(32.0, 74.5, get=ok, cache_dir=tmp_path, sleep=lambda s: None)
    assert len(ok.calls) == 1
    down = Counter(lambda p: (_ for _ in ()).throw(RuntimeError("ReadTimeout")))
    d = people.buildings(10.0, 10.0, get=down, cache_dir=tmp_path, sleep=lambda s: None)
    assert d["unavailable"] == "the OpenStreetMap query mirrors failed (ReadTimeout, ReadTimeout)" and len(down.calls) == 2
    d2 = people.buildings(11.0, 11.0, get=down, cache_dir=tmp_path, sleep=lambda s: None)
    assert len(down.calls) == 2 and "not asked again for 10 minutes" in d2["unavailable"]


def test_warm_downloads_a_country_once_in_the_background(tmp_path):
    src = tmp_path / "src.tif"
    write_grid(src)
    data = src.read_bytes()
    calls = []

    def get(url, headers=None, stream=None, timeout=None):
        calls.append(1)
        import time as _t
        _t.sleep(0.3)
        return Resp(content=data, headers={"Content-Length": str(len(data))})
    t = people.warm("PK", get=get, cache_dir=tmp_path / "w")
    assert t is not None and t.is_alive()                                      # returns at once, download goes on
    t2 = people.warm("PK", get=get, cache_dir=tmp_path / "w")
    assert t2 is t                                                              # one download for one country
    lat, lon = 33.0 - 20.5 * 0.0083333333, 70.0 + 20.5 * 0.0083333333
    p = people.population("PK", lat, lon, get=get, cache_dir=tmp_path / "w")     # joins the running download
    assert p["cell"] == 100 and len(calls) == 1
    assert people.warm("PK", get=get, cache_dir=tmp_path / "w") is None          # already on disk
    assert people.warm("ZZ", cache_dir=tmp_path / "w") is None
    bad = people.population("NG", 10.0, 8.0, get=lambda *a, **k: Resp(status=404), cache_dir=tmp_path / "x")
    assert bad == {"unavailable": "population grid not available (HTTP 404)"}


# ------------------------------------------------------------------ the two Overpass mirrors
def test_overpass_tries_two_mirrors_in_order_with_a_pause(tmp_path):
    urls, pauses = [], []

    def get(url, params=None, **kw):
        urls.append(url)
        if url == overpass.MIRRORS[0]:
            raise RuntimeError("ReadTimeout")
        return Resp({"elements": [1]})
    out = overpass.query("q", get=get, cache_dir=tmp_path, sleep=pauses.append)
    assert out == {"elements": [1]} and urls == list(overpass.MIRRORS) and pauses == [overpass.PAUSE_S]
    assert overpass.PAUSE_S >= 2
    urls.clear()
    out = overpass.query("q", get=get, cache_dir=tmp_path, sleep=pauses.append)      # the first mirror is remembered down
    assert urls == [overpass.MIRRORS[1]] and pauses == [overpass.PAUSE_S]            # no new pause


def test_overpass_failures_are_cached_ten_minutes(tmp_path):
    calls = []

    def get(url, params=None, **kw):
        calls.append(url)
        raise RuntimeError("HTTP 504")
    with pytest.raises(RuntimeError) as e:
        overpass.query("q", get=get, cache_dir=tmp_path, sleep=lambda s: None)
    assert str(e.value) == "the OpenStreetMap query mirrors failed (HTTP 504, HTTP 504)" and len(calls) == 2
    with pytest.raises(RuntimeError) as e:
        overpass.query("q", get=get, cache_dir=tmp_path, sleep=lambda s: None)
    assert len(calls) == 2 and "10 minutes" in str(e.value)                        # repeated clicks do not ask again
    assert overpass.DOWN_S == 600
    import os
    import time as _t
    for f in (tmp_path / "overpass").glob("*.json"):
        os.utime(f, (_t.time() - 700, _t.time() - 700))                             # ten minutes later
    with pytest.raises(RuntimeError):
        overpass.query("q", get=get, cache_dir=tmp_path, sleep=lambda s: None)
    assert len(calls) == 4


def test_user_agent_names_the_project_page():
    assert diskcache.USER_AGENT == ("thermal-anomaly-classifier/1.0 "
                                    "(https://github.com/Yusra-Shah/thermal-anomaly-classifier)")


# ------------------------------------------------------------------ response
def test_emergency_numbers_file():
    t = response.numbers_table()
    assert len(t) > 40 and all(r["status"] in ("verified", "unverified") for r in t.values())
    us = response.numbers("US")
    assert us["general"] == "911" and us["status"] == "verified" and "Verified" in us["note"]
    pk = response.numbers("PK")
    assert pk["police"] == "15" and pk["status"] == "unverified" and "check these numbers locally" in pk["note"]
    assert "no number is on file" in response.numbers("ZZ")["unavailable"]
    assert "no number is on file" in response.numbers(None)["unavailable"]
    names = {cc: n for cc, n, f in response.unverified()}
    assert "PK" in names and "US" not in names
    from src import geo
    assert set(t) <= set(geo.country_names())                               # every code is a real country code


def hits(*points):
    return Resp([{"lat": str(a), "lon": str(b), "name": n, "display_name": n + ", Somewhere"} for a, b, n in points])


def op(*points):
    return Resp({"elements": [{"type": "node", "lat": a, "lon": b, "tags": {"name": n}} for a, b, n in points]})


def no_overpass():
    return Counter(lambda p: (_ for _ in ()).throw(RuntimeError("ReadTimeout")))


def test_nearest_service_prefers_overpass_and_takes_the_closest(tmp_path):
    ov = Counter(lambda p: op((32.20, 74.5, "Far Station"), (32.03, 74.5, "Near Station")))
    nom = Counter(lambda p: hits())
    r = response.nearest("fire station", 32.0, 74.5, get=nom, cache_dir=tmp_path, overpass_get=ov, sleep=lambda s: None)
    assert r["name"] == "Near Station" and r["direction"] == "N" and r["km"] == pytest.approx(3.3, abs=0.2)
    assert r["via"] == "Overpass" and not nom.calls and '"amenity"="fire_station"' in ov.calls[0]["data"]
    again = response.nearest("fire station", 32.01, 74.51, get=nom, cache_dir=tmp_path, overpass_get=ov)
    assert len(ov.calls) == 1 and again["name"] == "Near Station"


def test_nearest_service_falls_back_to_nominatim_politely(tmp_path, monkeypatch):
    sleeps = []
    monkeypatch.setattr(response.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(response, "_last", [response.time.time()])
    nom = Counter(lambda p: hits((32.20, 74.5, "Far Station"), (32.03, 74.5, "Near Station")))
    ov = no_overpass()
    r = response.nearest("fire station", 32.0, 74.5, get=nom, cache_dir=tmp_path, overpass_get=ov, sleep=lambda s: None)
    assert r["name"] == "Near Station" and r["via"] == "Nominatim"
    assert nom.calls[0]["bounded"] == 1 and nom.calls[0]["q"] == "fire station" and sleeps      # one second between calls
    assert len(ov.calls) == 2                                                                   # both mirrors were tried


def test_the_search_widens_once_then_reports_none(tmp_path):
    ov = Counter(lambda p: op())
    r = response.nearest("hospital", 5.0, 5.0, get=Counter(lambda p: hits()), cache_dir=tmp_path, overpass_get=ov,
                        sleep=lambda s: None)
    assert r["none"] == 110 and len(ov.calls) == 2
    radii = [int(c["data"].split("around:")[1].split(",")[0]) for c in ov.calls]
    assert radii == [36000, 116000]                                          # the radius plus a margin
    n = len(ov.calls)
    response.nearest("hospital", 5.0, 5.0, cache_dir=tmp_path, overpass_get=ov)
    assert len(ov.calls) == n                                                # nothing found is cached too


def test_describe_stops_asking_when_the_services_are_down(tmp_path, monkeypatch):
    monkeypatch.setattr(response.time, "sleep", lambda s: None)
    monkeypatch.setattr(overpass.time, "sleep", lambda s: None)
    down = no_overpass()
    d = response.describe("PK", 32.0, 74.5, get=down, cache_dir=tmp_path, overpass_get=down)
    assert len(down.calls) == 3                                                # two mirrors and one Nominatim, not nine
    assert all("unavailable" in v for v in d["services"].values())
    assert list(d["services"]) == ["Fire station", "Hospital", "Police station"]
    assert d["numbers"]["police"] == "15" and "://" not in json.dumps(d)
    ok = Counter(lambda p: op((32.03, 74.5, "Somewhere")))
    d = response.describe("US", 32.0, 74.5, get=down, cache_dir=tmp_path / "ok", overpass_get=ok)
    assert [v["name"] for v in d["services"].values()] == ["Somewhere"] * 3 and len(ok.calls) == 3
    part = {}
    response.describe("US", 32.0, 74.5, cache_dir=tmp_path / "ok", overpass_get=ok, out=part)
    assert part["numbers"]["general"] == "911" and len(part["services"]) == 3


def test_a_second_click_in_the_same_cell_makes_no_request_and_measures_from_the_point(tmp_path):
    ov = Counter(lambda p: op((32.03, 74.5, "Near Station")))
    one = response.nearest("fire station", 32.0, 74.5, cache_dir=tmp_path, overpass_get=ov, sleep=lambda s: None)
    two = response.nearest("fire station", 32.02, 74.48, cache_dir=tmp_path, overpass_get=ov)       # same 0.05 degree cell
    assert len(ov.calls) == 1 and response.cell_center(32.0, 74.5) == response.cell_center(32.02, 74.48)
    assert one["km"] == pytest.approx(3.3, abs=0.2) and two["km"] == pytest.approx(2.2, abs=0.3)   # distance from the point
    other = response.nearest("fire station", 32.3, 74.5, cache_dir=tmp_path, overpass_get=ov, sleep=lambda s: None)
    assert len(ov.calls) == 2 and other["km"] != one["km"]                                          # another cell asks again


def test_weather_cache_cell_is_about_005_degree(tmp_path):
    get = Counter(lambda p: Resp({"hourly": series(NOW)}))
    t = pd.Timestamp("2026-11-20 06:17")
    weather.at(32.00, 74.50, t, get=get, cache_dir=tmp_path, now=NOW)
    weather.at(32.02, 74.52, t, get=get, cache_dir=tmp_path, now=NOW)
    assert len(get.calls) == 1
    weather.at(32.08, 74.50, t, get=get, cache_dir=tmp_path, now=NOW)
    assert len(get.calls) == 2


# ------------------------------------------------------------------ side by side, one worker per service
ROW = {"latitude": 32.0, "longitude": 74.5, "timestamp": "2026-11-20 06:00", "cc": "PK", "scan": 0.4, "track": 0.38}
ALL = ("land", "weather", "population", "place", "buildings", "services")


def hooks_of(**kw):
    h = {k: (lambda *a, **k2: {"ok": 1}) for k in ALL}
    h.update(kw)
    return h


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def test_every_service_has_its_own_worker_and_the_parts_come_as_they_answer():
    import threading
    gate = threading.Event()
    seen = []

    def slow(name):
        def fn(*a, **k):
            seen.append(threading.current_thread().name)
            gate.wait(5)
            return {"slow": name}
        return fn
    def slow_services(lat, lon, out=None):
        seen.append(threading.current_thread().name)
        gate.wait(5)
        return {label: {"slow": "s"} for _, label in response.KINDS}
    f = sections.Fetcher(ROW, hooks=hooks_of(buildings=slow("b"), services=slow_services))
    snap = f.wait(0.4)
    assert snap["land"] == {"ok": 1} and snap["weather"] == {"ok": 1}
    assert snap["exposure"]["population"] == {"ok": 1} and snap["exposure"]["place"] == {"ok": 1}
    assert snap["exposure"]["buildings"] is None                              # only the slow service is still loading
    assert all(v is None for v in snap["response"]["services"].values()) and snap["response"]["numbers"]["police"] == "15"
    assert f.pending()
    gate.set()
    snap = f.wait(2)
    assert snap["exposure"]["buildings"] == {"slow": "b"} and not f.pending()
    assert all(v == {"slow": "s"} for v in snap["response"]["services"].values())
    assert sorted(set(seen)) == ["panel-buildings", "panel-services"]


def test_services_inside_one_worker_go_one_after_the_other(tmp_path, monkeypatch):
    monkeypatch.setattr(response.time, "sleep", lambda s: None)
    order = []

    def ov(url, params=None, **kw):
        order.append(params["data"].split('"amenity"="')[1].split('"')[0])
        return op((32.01, 74.5, "x"))
    f = sections.Fetcher(ROW, hooks=hooks_of(services=lambda lat, lon: response.services(
        lat, lon, cache_dir=tmp_path, overpass_get=ov)))
    snap = f.wait(5)
    assert order == ["fire_station", "hospital", "police"]
    assert [v["name"] for v in snap["response"]["services"].values()] == ["x", "x", "x"]


def test_a_part_times_out_after_ten_seconds_and_services_keep_the_finished_kinds():
    import threading
    import time as _t
    stuck = threading.Event()
    clock = Clock()

    def services(lat, lon, out=None):
        out["Fire station"] = {"name": "F", "km": 1.0, "direction": "N"}
        stuck.wait(5)
        return out
    h = hooks_of(weather=lambda *a: (stuck.wait(5), {"t": 1})[1], buildings=lambda *a: (stuck.wait(5), {"b": 1})[1])
    f = sections.Fetcher(ROW, hooks=h, cap=10, clock=clock)
    f.services["Fire station"] = {"name": "F", "km": 1.0, "direction": "N"}
    f.hooks_services = services
    _t.sleep(0.2)
    assert f.get("weather") is None and f.pending()                          # inside the limit it is still loading
    clock.t += 10.5
    snap = f.snapshot()
    assert snap["weather"] == {"unavailable": "timed out"} and snap["exposure"]["buildings"] == {"unavailable": "timed out"}
    assert snap["land"] == {"ok": 1} and snap["exposure"]["population"] == {"ok": 1}
    assert not f.pending()
    stuck.set()                                                              # a late answer is used when it arrives
    for _ in range(40):
        if f.done.get("weather"):
            break
        _t.sleep(0.05)
    assert f.get("weather") == {"t": 1}
    assert sections.SECTION_CAP_S == 10


def test_unfinished_services_show_timed_out_but_finished_ones_stay():
    import threading
    stuck = threading.Event()
    clock = Clock()
    f = sections.Fetcher(ROW, hooks=hooks_of(services=lambda *a, **k: stuck.wait(5)), cap=10, clock=clock)
    f.services["Fire station"] = {"name": "F", "km": 1.0, "direction": "N"}      # what the worker had made so far
    clock.t += 10.5
    got = f.snapshot()["response"]["services"]
    assert got["Fire station"]["name"] == "F"
    assert got["Hospital"] == {"unavailable": "timed out"} and got["Police station"] == {"unavailable": "timed out"}
    stuck.set()


# ------------------------------------------------------------------ all of it together
def test_fetch_all_isolates_failures():
    order = []

    def mk(name, fail=False):
        def fn(*a, **k):
            order.append(name)
            if fail:
                raise ValueError("https://x/secret")
            return {"ok": name}
        return fn
    out = sections.fetch_all(ROW, hooks=hooks_of(weather=mk("weather", True), land=mk("land"), buildings=mk("b")))
    assert out["land"] == {"ok": "land"} and out["exposure"]["buildings"] == {"ok": "b"}
    assert out["weather"] == {"unavailable": "could not be made (ValueError)"} and "secret" not in str(out)
    assert out["exposure"]["population"] == {"ok": 1}


def test_a_row_field_that_cannot_be_read_only_costs_its_own_part():
    out = sections.fetch_all({"latitude": 32.0, "longitude": 74.5, "timestamp": None, "cc": None, "scan": "nan"},
                             hooks=hooks_of())
    assert out["weather"] == {"ok": 1} or "unavailable" in out["weather"]
    # real functions with a row that has no position: every part says why, nothing is raised
    out = sections.fetch_all({"latitude": None, "longitude": None, "timestamp": None})
    parts = [out["land"], out["weather"], *out["exposure"].values(), *out["response"]["services"].values()]
    assert all("unavailable" in v and "TypeError" in v["unavailable"] for v in parts)
    assert out["response"]["numbers"]["unavailable"]


def test_detail_build_carries_the_extras():
    from src.context import detail
    assert "extras" in detail.build.__code__.co_varnames


def test_no_key_or_print_in_the_sources():
    from pathlib import Path
    for mod in (diskcache, weather, land, people, response, sections, overpass):
        text = Path(mod.__file__).read_text(encoding="utf-8")
        assert "FIRMS_MAP_KEY" not in text and "load_key" not in text and "print(" not in text


def test_panel_of_a_store_row_with_every_outside_source_offline(tmp_path):
    """A row like the one reported (a NOAA detection with low FRP, geography added as categories, scan and track
    present): the base panel is built, the outside parts are made offline, and a failure of one leaves the others."""
    from src import ingest, live, serve
    from src.context import detail
    from tests import fake_feed
    store = tmp_path / "store.parquet"
    ingest.run_once("fake", fetch=fake_feed.fetch_factory(fake_feed.masters(10)), sleep=lambda s: None,
                    log=lambda *a: None, limiter=fake_feed.fast_limiter(), path=store)
    df = ingest.read_store(store)
    df = live.add_geo(df.rename(columns={"p_instant": "p_static_instant", "p_history": "p_static_history"}),
                      tmp_path / "geo.parquet")
    row = df[df["sensor"].astype(str).str.contains("NOAA")].sort_values("timestamp").iloc[-1].to_dict()
    full = detail.store_row(row["sensor"], row["latitude"], row["longitude"], row["acq_date"],
                            detail.acq_time_of(row), path=store)
    assert full
    for c in ("cc", "admin1", "city"):
        full[c] = row.get(c)
    full["frp"] = 0.6                                                       # low FRP
    full["p_static"], full["pred_static"] = 0.31, 0                           # copied from the globe row, as the app does
    t = pd.Timestamp(full["timestamp"])
    spread = detail.spread_from_store(float(full["latitude"]), float(full["longitude"]), t, full.get("cell_id"),
                                      path=store)
    m = serve.Models()
    off = {k: (lambda *a, **kw: {"unavailable": "offline"}) for k in ALL}
    extras = sections.fetch_all(full, hooks=off)
    for key, bundle in (("history", m.hist), ("instant", m.inst)):
        bands = m.bands[key]
        p = detail.build(full, source="live", tier=key, bundle=bundle, bands=bands, band=bands.get("default"),
                         threshold=None, spread=spread, extras=extras)
        assert all(p[k] for k in ("verdict", "facts", "place", "spread")), [k for k in ("verdict", "facts", "place", "spread") if not p[k]]
        assert p["land"] == p["weather"] == {"unavailable": "offline"}
    bad = dict(off, weather=lambda *a, **k: (_ for _ in ()).throw(TypeError("x")),
               land=lambda *a, **k: {"class": "Cropland"})
    extras = sections.fetch_all(full, hooks=bad)
    assert extras["weather"] == {"unavailable": "could not be made (TypeError)"}
    assert extras["land"] == {"class": "Cropland"} and extras["exposure"]["population"] == {"unavailable": "offline"}
    p = detail.build(full, source="live", tier="history", bundle=m.hist, bands=m.bands["history"],
                     band=m.bands["history"].get("default"), threshold=None, spread=spread, extras=extras)
    assert p["verdict"] and p["land"] == {"class": "Cropland"}


def test_the_countries_of_the_view_are_warmed_in_the_background_one_after_the_other(tmp_path):
    src = tmp_path / "src.tif"
    write_grid(src)
    data = src.read_bytes()
    order = []

    def get(url, headers=None, stream=None, timeout=None):
        order.append(url.split("/")[-2])
        return Resp(content=data, headers={"Content-Length": str(len(data))})
    (tmp_path / "w").mkdir()
    write_grid(people.raster_file("IND", tmp_path / "w"))                      # one country is already on disk
    t = people.warm_many(["PK", "IN", "PK", "ZZ", "NG", "BR", "AF", "NP", "LK"], limit=3, get=get, cache_dir=tmp_path / "w")
    assert t is not None
    t.join(10)
    assert order == ["PAK", "NGA", "BRA"]                                       # in view order, not IN, not ZZ, at most 3
    assert people.warm_many(["PK", "IN"], get=get, cache_dir=tmp_path / "w") is None        # all on disk
    assert people.warm_many([], cache_dir=tmp_path / "w") is None

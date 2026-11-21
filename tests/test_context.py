"""Tests for the panel sources: weather, land and fuel, people and buildings, response, and their combination.
Every request is a fake, so nothing here touches the network.

Run from the project root, one suite per process:  python -m pytest tests/test_context.py -q
"""
import io
import json

import numpy as np
import pandas as pd
import pytest

from src.context import diskcache, land, landcover, people, response, sections, weather
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
    assert seen["timeout"] == 8 and "fire-hotspot" in seen["headers"]["User-Agent"]


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
    again = weather.at(32.03, 74.52, t, get=get, cache_dir=tmp_path, now=NOW)         # same 0.1 degree cell
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
    b = people.buildings(32.0, 74.5, get=ok, cache_dir=tmp_path)
    assert b["count"] == 812 and "ODbL" in b["source"] and "around:1000" in ok.calls[0]["data"]
    people.buildings(32.0, 74.5, get=ok, cache_dir=tmp_path)
    assert len(ok.calls) == 1
    down = Counter(lambda p: (_ for _ in ()).throw(RuntimeError("ReadTimeout")))
    d = people.buildings(10.0, 10.0, get=down, cache_dir=tmp_path)
    assert d["unavailable"] == "OpenStreetMap Overpass did not answer (ReadTimeout)"
    d2 = people.buildings(11.0, 11.0, get=down, cache_dir=tmp_path)
    assert len(down.calls) == 1 and "not asked again" in d2["unavailable"]


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


def test_nearest_service_is_the_closest_hit_and_asks_politely(tmp_path, monkeypatch):
    sleeps = []
    monkeypatch.setattr(response.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(response, "_last", [response.time.time()])
    get = Counter(lambda p: hits((32.20, 74.5, "Far Station"), (32.03, 74.5, "Near Station")))
    r = response.nearest("fire station", 32.0, 74.5, get=get, cache_dir=tmp_path)
    assert r["name"] == "Near Station" and r["direction"] == "N" and r["km"] == pytest.approx(3.3, abs=0.2)
    assert get.calls[0]["bounded"] == 1 and get.calls[0]["q"] == "fire station" and sleeps          # one second between calls
    again = response.nearest("fire station", 32.01, 74.51, get=get, cache_dir=tmp_path)
    assert len(get.calls) == 1 and again["name"] == "Near Station"


def test_the_search_widens_once_then_reports_none(tmp_path, monkeypatch):
    monkeypatch.setattr(response.time, "sleep", lambda s: None)
    get = Counter(lambda p: Resp([]))
    r = response.nearest("hospital", 5.0, 5.0, get=get, cache_dir=tmp_path)
    assert r["none"] == 111 and len(get.calls) == 2
    widths = [float(c["viewbox"].split(",")[2]) - float(c["viewbox"].split(",")[0]) for c in get.calls]
    assert widths[1] > widths[0]
    n = len(get.calls)
    response.nearest("hospital", 5.0, 5.0, get=get, cache_dir=tmp_path)
    assert len(get.calls) == n                                               # nothing found is cached too


def test_describe_stops_asking_when_the_service_is_down(tmp_path, monkeypatch):
    monkeypatch.setattr(response.time, "sleep", lambda s: None)
    down = Counter(lambda p: (_ for _ in ()).throw(RuntimeError("ReadTimeout")))
    d = response.describe("PK", 32.0, 74.5, get=down, cache_dir=tmp_path)
    assert len(down.calls) == 1                                                # not three times 8 seconds
    assert all("unavailable" in v for v in d["services"].values())
    assert list(d["services"]) == ["Fire station", "Hospital", "Police station"]
    assert d["numbers"]["police"] == "15" and "://" not in json.dumps(d)
    ok = Counter(lambda p: hits((32.03, 74.5, "Somewhere")))
    d = response.describe("US", 32.0, 74.5, get=ok, cache_dir=tmp_path / "ok")
    assert [v["name"] for v in d["services"].values()] == ["Somewhere"] * 3 and len(ok.calls) == 3


# ------------------------------------------------------------------ all four together
def test_sections_fetch_in_order_and_isolate_failures():
    order = []

    def mk(name, fail=False):
        def fn(*a):
            order.append(name)
            if fail:
                raise ValueError("https://x/secret")
            return {"ok": name}
        return fn
    row = {"latitude": 32.0, "longitude": 74.5, "timestamp": "2026-11-20 06:00", "cc": "PK", "scan": 0.4, "track": 0.38}
    out = sections.fetch_all(row, hooks={"land": mk("land"), "weather": mk("weather", True), "people": mk("people"),
                                         "response": mk("response")})
    assert order == ["land", "weather", "people", "response"]
    assert out["land"] == {"ok": "land"} and out["exposure"] == {"ok": "people"}
    assert out["weather"] == {"unavailable": "could not be made (ValueError)"} and "secret" not in str(out)
    args = {}
    sections.fetch_all({**row, "cc": float("nan"), "scan": float("nan")},
                       hooks={"land": lambda *a: args.setdefault("land", a) and {}, "weather": lambda *a: {},
                              "people": lambda *a: args.setdefault("people", a) and {}, "response": lambda *a: {}})
    assert args["land"][2] is None and args["people"][0] is None


def test_detail_build_carries_the_extras():
    from src.context import detail
    assert "extras" in detail.build.__code__.co_varnames


def test_no_key_or_print_in_the_sources():
    from pathlib import Path
    for mod in (diskcache, weather, land, people, response, sections):
        text = Path(mod.__file__).read_text(encoding="utf-8")
        assert "FIRMS_MAP_KEY" not in text and "load_key" not in text and "print(" not in text

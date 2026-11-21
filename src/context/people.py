"""People near a detection, read when a detection is clicked.

Population: the WorldPop 2020 1 km grid of the country, downloaded once as one file per country and kept in
data/cache/worldpop, then read from disk (Pillow reads the tiled float raster, no raster library). Cells are about
1 km, so a figure within 1 km rests on one to four cells and is rough; the panel says so. The nearest named
settlement and its distance come from the offline place list. Buildings would need OpenStreetMap Overpass, which
is asked once with the 8 second limit and, when it does not answer, remembered as down for 30 minutes so that a
click is not delayed again.
"""
import json
import math
import os
import time
from functools import lru_cache

import numpy as np

from .. import geo
from ..config import ROOT
from . import diskcache

WP_BASE = "https://data.worldpop.org/GIS/Population/Global_2000_2020_1km/2020"
WP_DIR = ROOT / "data" / "cache" / "worldpop"
WP_SOURCE = "WorldPop 2020, 1 km grid (worldpop.org, CC BY 4.0)"
MAX_BYTES = 120 * 1024 * 1024
MAX_PIXELS = 60_000_000
DOWNLOAD_BUDGET_S = 180
OVERPASS = "https://overpass-api.de/api/interpreter"
OVERPASS_DOWN_S = 1800
KM_LAT = 110.574


@lru_cache(maxsize=1)
def _iso3():
    f = geo.GEO_DIR / geo.COUNTRIES
    if not f.exists():
        return {}
    out = {}
    for ft in json.loads(f.read_text(encoding="utf-8"))["features"]:
        p = ft["properties"]
        a2, a3 = p.get("ISO_A2_EH"), p.get("ISO_A3_EH") or p.get("ADM0_A3")
        if a2 and a2 != "-99" and a3 and a3 != "-99":
            out[a2] = a3
    return out


def iso3_of(cc):
    return _iso3().get(cc or "")


def raster_file(iso3, cache_dir=None):
    return (cache_dir or WP_DIR) / f"{iso3.lower()}_ppp_2020_1km.tif"


def download(iso3, get=None, cache_dir=None, budget_s=DOWNLOAD_BUDGET_S):
    """The country raster on disk, downloaded the first time (streamed, size limited). Raises RuntimeError with a
    short reason that never holds the address."""
    f = raster_file(iso3, cache_dir)
    if f.exists():
        return f
    import requests
    url = f"{WP_BASE}/{iso3}/{iso3.lower()}_ppp_2020_1km_Aggregated.tif"
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_name(f.name + f".{os.getpid()}.tmp")
    t0 = time.time()
    try:
        try:
            r = (get or requests.get)(url, headers={"User-Agent": diskcache.USER_AGENT}, stream=True,
                                      timeout=diskcache.TIMEOUT)
        except Exception as e:
            raise RuntimeError(type(e).__name__) from None
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}")
        size = int(r.headers.get("Content-Length") or 0)
        if size > MAX_BYTES:
            raise RuntimeError("population file too large")
        got = 0
        with open(tmp, "wb") as fh:
            try:
                for chunk in r.iter_content(1 << 16):
                    got += len(chunk)
                    if got > MAX_BYTES or time.time() - t0 > budget_s:
                        raise RuntimeError("population download too slow or too large")
                    fh.write(chunk)
            except RuntimeError:
                raise
            except Exception as e:
                raise RuntimeError(type(e).__name__) from None
        os.replace(tmp, f)
    finally:
        if tmp.exists():
            tmp.unlink()
    return f


def _read(path):
    """(grid with nodata as 0, x0, y0, sx, sy) of a country raster."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as im:
        if im.size[0] * im.size[1] > MAX_PIXELS:
            raise RuntimeError("population grid too large for this machine")
        tags = im.tag_v2
        sx, sy = tags[33550][0], tags[33550][1]
        x0, y0 = tags[33922][3], tags[33922][4]
        a = np.asarray(im, dtype=np.float32)
    a = np.where(np.isfinite(a) & (a > 0), a, 0.0).astype(np.float32)
    return a, x0, y0, sx, sy


def within(arr, x0, y0, sx, sy, lat, lon, radii_km, sub=8):
    """People within each radius of a point: every cell near the point is split into sub by sub parts and the parts
    whose centre lies inside the circle are summed (area weighted). Returns (one total per radius, the value of the
    cell that holds the point or None when the point is outside the grid)."""
    h, w = arr.shape
    kx = 111.320 * math.cos(math.radians(lat))
    col, row = (lon - x0) / sx, (y0 - lat) / sy
    out = []
    for r in radii_km:
        span = int(math.ceil(r / (sx * kx))) + 1
        spany = int(math.ceil(r / (sy * KM_LAT))) + 1
        total = 0.0
        for rr in range(max(int(row) - spany, 0), min(int(row) + spany + 1, h)):
            for cc in range(max(int(col) - span, 0), min(int(col) + span + 1, w)):
                v = float(arr[rr, cc])
                if v <= 0:
                    continue
                gx = (cc + (np.arange(sub) + 0.5) / sub - col) * sx * kx
                gy = (rr + (np.arange(sub) + 0.5) / sub - row) * sy * KM_LAT
                d2 = gx[None, :] ** 2 + gy[:, None] ** 2
                total += v * float((d2 <= r * r).mean())
        out.append(total)
    inside = 0 <= int(row) < h and 0 <= int(col) < w and col >= 0 and row >= 0
    return out, (float(arr[int(row), int(col)]) if inside else None)


def _round(v):
    return int(round(v, -1)) if v >= 100 else int(round(v))


def population(cc, lat, lon, get=None, cache_dir=None):
    """Population section, or {"unavailable": reason}."""
    iso3 = iso3_of(cc)
    if not iso3:
        return {"unavailable": "country not known for this point"}
    try:
        f = download(iso3, get, cache_dir)
        arr, x0, y0, sx, sy = _read(f)
    except RuntimeError as e:
        return {"unavailable": f"population grid not available ({e})"}
    except Exception as e:
        return {"unavailable": f"population grid could not be read ({type(e).__name__})"}
    (p1, p5), cell = within(arr, x0, y0, sx, sy, lat, lon, (1.0, 5.0))
    if cell is None:
        return {"unavailable": "the point is outside the population grid of its country"}
    return {"within_1km": _round(p1), "within_5km": _round(p5), "cell": int(round(cell)),
            "resolution": "about 1 km", "source": WP_SOURCE, "fetched_at": diskcache.now_text(),
            "note": "Cells are about 1 km, so the figure within 1 km rests on one to four cells and is rough."}


def nearest_place(lat, lon, searcher=None):
    """Name, distance in km and direction of the nearest named place in the offline list, or None."""
    from .detail import bearing_deg, compass, distance_km
    try:
        if searcher is None:
            import reverse_geocoder as rg

            def searcher(pts):
                return rg.search(pts, mode=1, verbose=False)
        h = searcher([(lat, lon)])[0]
        plat, plon = float(h["lat"]), float(h["lon"])
    except Exception:
        return None
    return {"name": h["name"], "km": float(round(distance_km(lat, lon, plat, plon), 1)),
            "direction": compass(bearing_deg(lat, lon, plat, plon))}


def buildings(lat, lon, get=None, cache_dir=None):
    """Count of buildings within 1 km from OpenStreetMap Overpass, or {"unavailable": reason}. A failure is
    remembered for 30 minutes."""
    key = f"{round(lat, 3):.3f}_{round(lon, 3):.3f}"
    hit = diskcache.read("buildings", key, ttl=7 * 86400, root=cache_dir)
    if hit and "count" in hit:
        return hit
    if diskcache.read("buildings", "_down", ttl=OVERPASS_DOWN_S, root=cache_dir):
        return {"unavailable": "OpenStreetMap Overpass did not answer a moment ago, not asked again for 30 minutes"}
    q = f'[out:json][timeout:8];(way["building"](around:1000,{lat:.5f},{lon:.5f}););out count;'
    try:
        r = (get or diskcache.http_get)(OVERPASS, params={"data": q})
        tags = r.json()["elements"][0]["tags"]
        out = {"count": int(tags.get("ways", 0)), "source": "OpenStreetMap contributors (ODbL), via Overpass",
               "fetched_at": diskcache.now_text(),
               "note": "OpenStreetMap building coverage varies, a low count can mean unmapped buildings."}
    except Exception as e:
        diskcache.write("buildings", "_down", {"at": diskcache.now_text()}, root=cache_dir)
        why = str(e) if isinstance(e, RuntimeError) else type(e).__name__
        return {"unavailable": f"OpenStreetMap Overpass did not answer ({why})"}
    diskcache.write("buildings", key, out, root=cache_dir)
    return out


def describe(cc, lat, lon, get=None, cache_dir=None, overpass_get=None, searcher=None):
    """People and buildings section. A dict whose parts that failed carry their reason."""
    return {"population": population(cc, lat, lon, get, cache_dir), "place": nearest_place(lat, lon, searcher),
            "buildings": buildings(lat, lon, overpass_get, cache_dir)}

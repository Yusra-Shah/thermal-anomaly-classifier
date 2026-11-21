"""Response near a detection: emergency numbers and the nearest fire station, hospital and police station.

Numbers come from data/emergency_numbers.json (a number is shown as verified only when the file says so). The nearest
services come from OpenStreetMap data, run when a detection is clicked: first the Overpass query service through its two
public mirrors in order (src/context/overpass.py), and when both fail the Nominatim search, one request at a time, at
least one second apart. Each request has the identifying User-Agent and an 8 second limit. The candidates around the
centre of the 0.05 degree cell of the point (the radius plus a margin) are cached on disk, and the nearest one is
measured from the point itself, so a second click nearby makes no request. The nearest hit within about 30 km is
reported, else within about 110 km, else none.
"""
import json
import threading
import time
from functools import lru_cache

from ..config import ROOT
from . import diskcache, overpass

NUMBERS_FILE = ROOT / "data" / "emergency_numbers.json"
NOMINATIM = "https://nominatim.openstreetmap.org/search"
SOURCE = "OpenStreetMap contributors (ODbL), via Overpass or Nominatim"
AMENITY = {"fire station": "fire_station", "hospital": "hospital", "police": "police"}
RADII_KM = (30, 110)
KINDS = (("fire station", "Fire station"), ("hospital", "Hospital"), ("police", "Police station"))
MIN_GAP_S = 1.1
_lock = threading.Lock()
_last = [0.0]


@lru_cache(maxsize=1)
def numbers_table():
    try:
        return json.loads(NUMBERS_FILE.read_text(encoding="utf-8"))["countries"]
    except Exception:
        return {}


def numbers(cc):
    """Emergency numbers of a country: a dict with the filled fields and a status, or {"unavailable": reason}."""
    row = numbers_table().get(cc or "")
    if not row:
        return {"unavailable": "no number is on file for this country. 112 is the emergency number on mobile "
                               "networks of the GSM standard in many countries."}
    keep = {k: v for k, v in row.items() if k in ("general", "police", "fire", "ambulance", "also") and v}
    return {**keep, "name": row.get("name", ""), "status": row.get("status", "unverified"),
            "source": "data/emergency_numbers.json",
            "note": ("Verified numbers are ones the file marks as certain." if row.get("status") == "verified"
                     else "Unverified: check these numbers locally before relying on them.")}


def unverified():
    """(code, name, fields) of every unverified entry, for printing."""
    return [(cc, r.get("name", ""), {k: v for k, v in r.items() if k in ("general", "police", "fire", "ambulance", "also")})
            for cc, r in numbers_table().items() if r.get("status") != "verified"]


def _wait():
    with _lock:
        gap = time.time() - _last[0]
        if gap < MIN_GAP_S:
            time.sleep(MIN_GAP_S - gap)
        _last[0] = time.time()


CELL = 0.05                       # degrees: the cache cell of a point
MARGIN_KM = 6                     # a point is at most about 4 km from the centre of its cell


def cell_center(lat, lon):
    return round(round(lat / CELL) * CELL, 2), round(round(lon / CELL) * CELL, 2)


def _cands_nominatim(kind, clat, clon, r_km, get):
    half = (r_km + MARGIN_KM) / 111.0
    box = f"{clon - half:.4f},{clat + half:.4f},{clon + half:.4f},{clat - half:.4f}"
    _wait()
    r = get(NOMINATIM, params={"q": kind, "format": "jsonv2", "limit": 40, "viewbox": box, "bounded": 1,
                               "addressdetails": 0})
    return [{"lat": float(h["lat"]), "lon": float(h["lon"]),
             "name": (h.get("name") or h.get("display_name", "").split(",")[0] or kind).strip()}
            for h in r.json() if "lat" in h and "lon" in h]


def _cands_overpass(kind, clat, clon, r_km, get, cache_dir, pause, sleep):
    ql = (f'[out:json][timeout:8];nwr["amenity"="{AMENITY[kind]}"](around:{(r_km + MARGIN_KM) * 1000},'
          f"{clat:.5f},{clon:.5f});out center 100;")
    out = []
    for h in overpass.query(ql, get, cache_dir, pause=pause, sleep=sleep).get("elements", []):
        try:
            la = float(h.get("lat", (h.get("center") or {}).get("lat")))
            lo = float(h.get("lon", (h.get("center") or {}).get("lon")))
        except (TypeError, ValueError):
            continue
        out.append({"lat": la, "lon": lo, "name": ((h.get("tags") or {}).get("name") or kind).strip()})
    return out


def _closest(cands, lat, lon, kind):
    from .detail import bearing_deg, compass, distance_km
    best = None
    for c in cands:
        d = distance_km(lat, lon, c["lat"], c["lon"])
        if best is None or d < best["km"]:
            best = {"name": c["name"] or kind, "km": float(round(d, 1)),
                    "direction": compass(bearing_deg(lat, lon, c["lat"], c["lon"]))}
    return best


def nearest(kind, lat, lon, get=None, cache_dir=None, overpass_get=None, pause=overpass.PAUSE_S, sleep=time.sleep):
    """The nearest place of a kind to the point: a dict with name, km and direction, {"none": radius} when nothing is
    within the largest radius, or {"unavailable": reason}. The candidates of the point's 0.05 degree cell come from the
    disk cache, else Overpass (two mirrors), else Nominatim."""
    clat, clon = cell_center(lat, lon)
    for r in RADII_KM:
        key = f"{kind.replace(' ', '_')}_{clat:.2f}_{clon:.2f}_{r}"
        hit = diskcache.read("nominatim", key, ttl=30 * 86400, root=cache_dir)
        if hit is None:
            try:
                try:
                    cands, via = _cands_overpass(kind, clat, clon, r, overpass_get, cache_dir, pause, sleep), "Overpass"
                except RuntimeError:
                    cands, via = _cands_nominatim(kind, clat, clon, r, get or diskcache.http_get), "Nominatim"
            except Exception as e:
                why = str(e) if isinstance(e, RuntimeError) else type(e).__name__
                return {"unavailable": f"the place search did not answer ({why})"}
            hit = {"cands": cands, "via": via, "fetched_at": diskcache.now_text()}
            diskcache.write("nominatim", key, hit, root=cache_dir)
        best = _closest(hit["cands"], lat, lon, kind)
        if best:
            return {**best, "via": hit["via"], "fetched_at": hit["fetched_at"]}
    return {"none": RADII_KM[-1], "fetched_at": diskcache.now_text()}


def services(lat, lon, get=None, cache_dir=None, overpass_get=None, out=None):
    """The nearest fire station, hospital and police station, looked up one at a time (a request after another, so
    Nominatim is asked at most once a second). Each result is written into out (when given) as it is made."""
    out = {} if out is None else out
    down = None
    for kind, label in KINDS:
        if down:                                   # the service did not answer: do not wait for it two more times
            out[label] = down
            continue
        res = nearest(kind, lat, lon, get, cache_dir, overpass_get)
        out[label] = res
        if "unavailable" in res:
            down = res
    return out


NOTE = ("Named places are the nearest hit of an OpenStreetMap search, which can miss unmapped services; they are not "
        "an assessment of what can reach the place or how fast.")


def describe(cc, lat, lon, get=None, cache_dir=None, overpass_get=None, out=None):
    """Response section: numbers and the three nearest services. The parts are written into out (when given) as they
    are made, so a caller that stops waiting still has the finished ones."""
    out = {} if out is None else out
    out.update({"numbers": numbers(cc), "services": {}, "source": SOURCE, "fetched_at": diskcache.now_text(),
                "note": NOTE})
    services(lat, lon, get, cache_dir, overpass_get, out["services"])
    return out

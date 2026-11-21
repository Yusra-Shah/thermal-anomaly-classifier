"""Response near a detection: emergency numbers and the nearest fire station, hospital and police station.

Numbers come from data/emergency_numbers.json (a number is shown as verified only when the file says so). The nearest
stations come from a search of OpenStreetMap data through Nominatim, run when a detection is clicked: one request at a
time, at least one second between requests, an identifying User-Agent, 8 seconds per request and a disk cache keyed by
the point rounded to 0.1 degree. The nearest hit within about 30 km is reported, else within about 110 km, else none.
"""
import json
import threading
import time
from functools import lru_cache

from ..config import ROOT
from . import diskcache

NUMBERS_FILE = ROOT / "data" / "emergency_numbers.json"
NOMINATIM = "https://nominatim.openstreetmap.org/search"
SOURCE = "OpenStreetMap contributors (ODbL), via Nominatim"
KINDS = (("fire station", "Fire station"), ("hospital", "Hospital"), ("police", "Police station"))
HALF_WIDTHS = (0.27, 1.0)           # degrees, about 30 km and 110 km
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


def _search(kind, lat, lon, half, get):
    from .detail import bearing_deg, compass, distance_km
    box = f"{lon - half:.4f},{lat + half:.4f},{lon + half:.4f},{lat - half:.4f}"
    _wait()
    r = get(NOMINATIM, params={"q": kind, "format": "jsonv2", "limit": 20, "viewbox": box, "bounded": 1,
                               "addressdetails": 0})
    best = None
    for h in r.json():
        try:
            hl, ho = float(h["lat"]), float(h["lon"])
        except Exception:
            continue
        d = distance_km(lat, lon, hl, ho)
        if best is None or d < best["km"]:
            best = {"name": (h.get("name") or h.get("display_name", "").split(",")[0] or kind).strip(),
                    "km": float(round(d, 1)), "direction": compass(bearing_deg(lat, lon, hl, ho))}
    return best


def nearest(kind, lat, lon, get=None, cache_dir=None):
    """The nearest place of a kind, from the cache or the search. A dict with name, km and direction, {"none":
    radius} when nothing is within the largest box, or {"unavailable": reason}."""
    key = f"{kind.replace(' ', '_')}_{round(lat, 1):.1f}_{round(lon, 1):.1f}"
    hit = diskcache.read("nominatim", key, ttl=30 * 86400, root=cache_dir)
    if hit:
        return hit
    get = get or diskcache.http_get
    try:
        for half in HALF_WIDTHS:
            best = _search(kind, lat, lon, half, get)
            if best:
                break
    except Exception as e:
        why = str(e) if isinstance(e, RuntimeError) else type(e).__name__
        return {"unavailable": f"the place search did not answer ({why})"}
    out = best or {"none": int(HALF_WIDTHS[-1] * 111)}
    out = {**out, "fetched_at": diskcache.now_text()}
    diskcache.write("nominatim", key, out, root=cache_dir)
    return out


def describe(cc, lat, lon, get=None, cache_dir=None):
    """Response section: numbers and the three nearest services. Services are looked up one at a time."""
    out = {"numbers": numbers(cc), "services": {}, "source": SOURCE, "fetched_at": diskcache.now_text(),
           "note": "Named places are the nearest hit of an OpenStreetMap search, which can miss unmapped services; "
                   "they are not an assessment of what can reach the place or how fast."}
    down = None
    for kind, label in KINDS:
        if down:                                   # the service did not answer: do not wait for it two more times
            out["services"][label] = down
            continue
        res = nearest(kind, lat, lon, get, cache_dir)
        out["services"][label] = res
        if "unavailable" in res:
            down = res
    return out

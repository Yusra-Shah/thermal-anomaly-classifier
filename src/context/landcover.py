"""ESA WorldCover land cover for detections, read without a raster library.

WorldCover (version 200, year 2021, 10 m, CC BY 4.0) is published as one cloud optimised GeoTIFF per 3 by 3 degree
tile on a public bucket, no key. Such a file is tiled and deflate compressed and has overviews, and its first bytes
hold the tile index. This module reads the index, then only the tiles of the file that a set of points needs, with
one range request each, and keeps every decoded tile on disk (data/cache/worldcover), so a second look at the same
places makes no request. A request never covers a single point: points are grouped by tile first.

The class of a detection is the majority class over its pixel footprint (scan by track, nominal 375 m). The finest
resolution is used whose number of tiles stays within MAX_TILES, otherwise an overview level, and the answer says
which. The request time of a run is limited (BUDGET_S): the points whose tiles were not reached are reported as not
classified, and the next run continues from the disk cache.

A failure never carries a URL: the caller sees the exception class only. No part of this module reads the map key.
"""
import json
import math
import os
import struct
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np

from ..config import ROOT

BASE = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map"
CACHE_DIR = ROOT / "data" / "cache" / "worldcover"
USER_AGENT = "fire-hotspot-classifier/1.0 (https://github.com/Yusra-Shah/thermal-anomaly-classifier)"
SOURCE = "ESA WorldCover 2021 v200 (10 m, CC BY 4.0)"
TIMEOUT = 8                      # seconds for one request
HEADER_BYTES = 65536             # the first bytes of a file, which hold its tile index
MAX_TILES = 120                  # tiles read in one run at the chosen resolution
BUDGET_S = 45                    # seconds of requests in one run
WORKERS = 4
PROBE_TILE = "N30E072"           # a land tile, used to test that the bucket answers
NOMINAL_KM = 0.375

# WorldCover classes and the three groups of the Earth filter
CLASSES = {10: "Tree cover", 20: "Shrubland", 30: "Grassland", 40: "Cropland", 50: "Built-up",
           60: "Bare or sparse vegetation", 70: "Snow and ice", 80: "Permanent water bodies",
           90: "Herbaceous wetland", 95: "Mangroves", 100: "Moss and lichen"}
CROPLAND = frozenset({40})
NATURAL = frozenset({10, 20, 30, 60, 90, 95, 100})        # forest, shrubland, grassland, wetland, moss, bare or sparse
NOT_CLASSIFIED = 255
GROUP_NONE, GROUP_CROP, GROUP_NATURAL, GROUP_OTHER = 0, 1, 2, 3
GROUP_NAMES = {GROUP_CROP: "Cropland", GROUP_NATURAL: "Natural land",
               GROUP_OTHER: "Other (built-up, water, snow and ice)", GROUP_NONE: "Not classified"}


class TileMissing(Exception):
    """There is no file for this 3 degree tile (open sea)."""


class FetchError(Exception):
    """A request failed. Only the class of the original error is kept, never its text (it can hold the address)."""


class Unavailable(Exception):
    """WorldCover cannot be reached at all."""


# ------------------------------------------------------------------ names and requests
def tile_name(lat, lon):
    """Name of the 3 by 3 degree tile that holds a point, from its south west corner, for example N30E072."""
    la, lo = int(math.floor(lat / 3.0)) * 3, int(math.floor(lon / 3.0)) * 3
    return f"{'N' if la >= 0 else 'S'}{abs(la):02d}{'E' if lo >= 0 else 'W'}{abs(lo):03d}"


def http_fetch(name, start, end, session=None, timeout=TIMEOUT):
    """Bytes start to end of the file of one tile, by a range request."""
    import requests
    url = f"{BASE}/ESA_WorldCover_10m_2021_v200_{name}_Map.tif"
    try:
        r = (session or requests).get(url, headers={"User-Agent": USER_AGENT, "Range": f"bytes={start}-{end}"},
                                      timeout=timeout)
    except Exception as e:
        raise FetchError(type(e).__name__) from None
    if r.status_code in (403, 404):
        raise TileMissing(name)
    if r.status_code not in (200, 206):
        raise FetchError(f"HTTP {r.status_code}")
    return r.content if r.status_code == 206 or start == 0 and r.status_code == 200 else r.content[start:end + 1]


def reachable(fetch=None):
    """(True, "") when the bucket answers, else (False, exception class). One small request."""
    try:
        (fetch or http_fetch)(PROBE_TILE, 0, 1023)
        return True, ""
    except TileMissing:
        return True, ""                              # the bucket answered
    except Exception as e:
        return False, type(e).__name__


# ------------------------------------------------------------------ the tile index of one file
_DTYPE = {3: "u2", 4: "u4", 12: "f8", 16: "u8"}


def _slice(head, off, size, more):
    return head[off:off + size] if off + size <= len(head) else more(off, size)


def parse_header(head, more):
    """Levels (size, tile size, compression, tile offsets and byte counts) and the georeference of a classic TIFF.
    `more(offset, size)` reads bytes that are not among the first ones."""
    if head[:2] not in (b"II", b"MM"):
        raise ValueError("not a tiff")
    e = "<" if head[:2] == b"II" else ">"
    if struct.unpack(e + "H", head[2:4])[0] != 42:
        raise ValueError("not a classic tiff")
    off = struct.unpack(e + "I", head[4:8])[0]
    levels, geo = [], {}
    while off and len(levels) < 16:
        n = struct.unpack(e + "H", _slice(head, off, 2, more))[0]
        raw = _slice(head, off + 2, 12 * n + 4, more)
        tags = {}
        for i in range(n):
            tag, typ, count = struct.unpack(e + "HHI", raw[12 * i:12 * i + 8])
            size = {3: 2, 4: 4, 12: 8, 16: 8}.get(typ, 1) * count
            data = raw[12 * i + 8:12 * i + 8 + size] if size <= 4 else \
                _slice(head, struct.unpack(e + "I", raw[12 * i + 8:12 * i + 12])[0], size, more)
            if tag in (256, 257, 259, 317, 322, 323, 324, 325, 33550, 33922) and typ in _DTYPE:
                tags[tag] = np.frombuffer(data, dtype=np.dtype(_DTYPE[typ]).newbyteorder(e))
        if 322 not in tags or 324 not in tags:
            raise ValueError("not a tiled tiff")
        levels.append({"w": int(tags[256][0]), "h": int(tags[257][0]), "tw": int(tags[322][0]), "th": int(tags[323][0]),
                       "comp": int(tags[259][0]), "pred": int(tags[317][0]) if 317 in tags else 1,
                       "offsets": tags[324].astype(np.int64).tolist(), "counts": tags[325].astype(np.int64).tolist()})
        if len(levels) == 1:
            if 33550 not in tags or 33922 not in tags:
                raise ValueError("no georeference")
            geo = {"sx": float(tags[33550][0]), "sy": float(tags[33550][1]),
                   "x0": float(tags[33922][3]), "y0": float(tags[33922][4])}
        off = struct.unpack(e + "I", raw[12 * n:12 * n + 4])[0]
    return {"geo": geo, "levels": levels}


def _atomic_write(path, write):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def load_header(name, fetch, cache_dir):
    """The tile index of one file: from the disk cache, else by one range request (and then cached).
    Raises TileMissing for open sea (and remembers it)."""
    d = cache_dir / name
    if (d / "missing.flag").exists():
        raise TileMissing(name)
    f = d / "header.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    try:
        head = fetch(name, 0, HEADER_BYTES - 1)
        header = parse_header(head, lambda off, size: fetch(name, off, off + size - 1))
    except TileMissing:
        _atomic_write(d / "missing.flag", lambda p: p.write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8"))
        raise
    _atomic_write(f, lambda p: p.write_text(json.dumps(header), encoding="utf-8"))
    return header


def level_geometry(header, level):
    """Pixel size in degrees and the tile grid of one level (level 0 is the full 10 m resolution)."""
    L, L0, g = header["levels"][level], header["levels"][0], header["geo"]
    k = L0["w"] / L["w"]
    return {"sx": g["sx"] * k, "sy": g["sy"] * (L0["h"] / L["h"]), "x0": g["x0"], "y0": g["y0"], "w": L["w"], "h": L["h"],
            "tw": L["tw"], "th": L["th"], "nx": -(-L["w"] // L["tw"]), "ny": -(-L["h"] // L["th"])}


def get_tile(name, header, level, ty, tx, fetch, cache_dir, keep=True):
    """(array of one tile, True when it came from the disk cache). One range request when it is not cached. With
    keep=False a tile that was read is not written to the disk cache (the land table keeps only the class of a cell)."""
    f = cache_dir / name / f"L{level}_{ty}_{tx}.npz"
    if f.exists():
        try:
            return np.load(f)["a"], True
        except Exception:
            f.unlink()                                                # a broken cache file is read again
    L = header["levels"][level]
    i = ty * (-(-L["w"] // L["tw"])) + tx
    off, cnt = L["offsets"][i], L["counts"][i]
    if cnt == 0:
        arr = np.zeros((L["th"], L["tw"]), np.uint8)
    else:
        if L["comp"] not in (8, 32946):
            raise ValueError("unsupported compression")
        arr = np.frombuffer(zlib.decompress(fetch(name, off, off + cnt - 1)), np.uint8).reshape(L["th"], L["tw"])
        if L["pred"] == 2:
            arr = np.cumsum(arr, axis=1, dtype=np.uint8)
    def save(p):
        with open(p, "wb") as fh:                                     # a file object, so no suffix is appended
            np.savez_compressed(fh, a=arr)
    if keep:
        _atomic_write(f, save)
    return arr, False


# ------------------------------------------------------------------ classification of a set of points
def _windows(lon, lat, scan_km, track_km, geom):
    """Pixel window (x0, x1, y0, y1) of every point on one level, about the footprint, at least one pixel."""
    mx = geom["sx"] * 111320.0 * np.cos(np.radians(np.clip(lat, -80, 80)))
    my = geom["sy"] * 110574.0
    nx = np.maximum(1, np.rint(scan_km * 1000.0 / mx)).astype(int)
    ny = np.maximum(1, np.rint(track_km * 1000.0 / my)).astype(int)
    col = (lon - geom["x0"]) / geom["sx"]
    row = (geom["y0"] - lat) / geom["sy"]
    x0 = np.rint(col - nx / 2.0).astype(int)
    y0 = np.rint(row - ny / 2.0).astype(int)
    return (np.clip(x0, 0, geom["w"] - 1), np.clip(x0 + nx, 1, geom["w"]),
            np.clip(y0, 0, geom["h"] - 1), np.clip(y0 + ny, 1, geom["h"]))


def _tiles_needed(win, geom):
    x0, x1, y0, y1 = win
    tw, th = geom["tw"], geom["th"]
    out = set()
    for a in (x0, x1 - 1):
        for b in (y0, y1 - 1):
            out.update(zip((b // th).tolist(), (a // tw).tolist()))
    return out


def classify(lon, lat, scan_km=None, track_km=None, *, fetch=None, cache_dir=None, max_tiles=MAX_TILES,
             budget_s=BUDGET_S, workers=WORKERS, keep_tiles=True):
    """Majority WorldCover class over the pixel footprint of each point, in batches.

    Returns a dict: code (uint8 per point: the WorldCover class, 0 where the tile has no land, 255 where it could not
    be read in this run), level, pixel_m (size of a pixel of the level, north to south), tiles_read, tiles_cached,
    failed (tiles or files that failed), headers_read, source and fetched_at. Raises Unavailable when no file
    can be reached at all."""
    if fetch is None:
        # one session for the whole call, so the connections to the bucket are kept and reused instead of a new secure
        # connection for every tile
        import requests
        from requests.adapters import HTTPAdapter
        session = requests.Session()
        session.mount("https://", HTTPAdapter(pool_connections=4, pool_maxsize=max(2 * workers, 4)))
        fetch = lambda name, start, end: http_fetch(name, start, end, session=session)  # noqa: E731
    cache_dir = cache_dir or CACHE_DIR
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    n = len(lon)
    scan = np.full(n, NOMINAL_KM) if scan_km is None else np.where(np.asarray(scan_km, float) > 0, scan_km, NOMINAL_KM)
    track = np.full(n, NOMINAL_KM) if track_km is None else np.where(np.asarray(track_km, float) > 0, track_km, NOMINAL_KM)
    deadline = time.time() + budget_s
    names = np.array([tile_name(a, b) for a, b in zip(lat, lon)]) if n else np.array([], dtype=str)
    uniq = sorted(set(names.tolist()))
    headers, missing, errors = {}, set(), []

    def head(nm):
        try:
            return nm, load_header(nm, fetch, cache_dir), None
        except TileMissing:
            return nm, None, "missing"
        except Exception as e:
            return nm, None, type(e).__name__
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for nm, h, err in ex.map(head, uniq):
            if err is None:
                headers[nm] = h
            elif err == "missing":
                missing.add(nm)
            else:
                errors.append(err)
    result = {"source": SOURCE, "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
              "tiles_read": 0, "tiles_cached": 0, "failed": len(errors), "headers_read": len(headers),
              "level": 0, "pixel_m": None}
    code = np.full(n, NOT_CLASSIFIED, np.uint8)
    code[np.isin(names, list(missing))] = 0                     # open sea: no land cover, not a failure
    if not headers:
        if errors:
            raise Unavailable(errors[0])
        result["code"] = code
        return result

    # the finest level whose tiles stay within the limit
    sample = next(iter(headers.values()))
    wins, level = {}, len(sample["levels"]) - 1
    for lv in range(len(sample["levels"])):
        need = 0
        for nm in headers:
            idx = np.flatnonzero(names == nm)
            geom = level_geometry(headers[nm], lv)
            need += len(_tiles_needed(_windows(lon[idx], lat[idx], scan[idx], track[idx], geom), geom))
        if need <= max_tiles:
            level = lv
            break
    result["level"] = level
    result["pixel_m"] = round(level_geometry(sample, level)["sy"] * 110574.0, 1)
    tiles, todo = {}, []
    for nm in headers:
        idx = np.flatnonzero(names == nm)
        geom = level_geometry(headers[nm], level)
        win = _windows(lon[idx], lat[idx], scan[idx], track[idx], geom)
        wins[nm] = (idx, geom, win)
        for ty, tx in _tiles_needed(win, geom):
            todo.append((nm, ty, tx))

    def one(item):
        nm, ty, tx = item
        if time.time() > deadline:
            return item, None, "budget"
        try:
            arr, cached = get_tile(nm, headers[nm], level, ty, tx, fetch, cache_dir, keep_tiles)
            return item, (arr, cached), None
        except TileMissing:
            return item, None, "missing"
        except Exception as e:
            return item, None, type(e).__name__
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for item, got, err in ex.map(one, todo):
            if got is not None:
                tiles[item] = got[0]
                result["tiles_cached" if got[1] else "tiles_read"] += 1
            elif err != "budget":
                result["failed"] += 1

    for nm, (idx, geom, win) in wins.items():
        x0, x1, y0, y1 = win
        tw, th = geom["tw"], geom["th"]
        for k, i in enumerate(idx):
            counts = np.zeros(256, np.int64)
            ok = True
            for ty in range(y0[k] // th, (y1[k] - 1) // th + 1):
                for tx in range(x0[k] // tw, (x1[k] - 1) // tw + 1):
                    t = tiles.get((nm, ty, tx))
                    if t is None:
                        ok = False
                        continue
                    sub = t[max(y0[k] - ty * th, 0):min(y1[k] - ty * th, th), max(x0[k] - tx * tw, 0):min(x1[k] - tx * tw, tw)]
                    counts += np.bincount(sub.ravel(), minlength=256)
            if counts.sum() > 0 and ok:
                code[i] = int(np.argmax(counts))               # the majority class, 0 where the window has no land
    result["code"] = code
    return result


def groups(code):
    """Filter group of each class code: cropland, natural land, other, or none (no land or not classified)."""
    code = np.asarray(code)
    out = np.full(len(code), GROUP_NONE, np.uint8)
    out[np.isin(code, list(CROPLAND))] = GROUP_CROP
    out[np.isin(code, list(NATURAL))] = GROUP_NATURAL
    other = (code > 0) & (code != NOT_CLASSIFIED) & ~np.isin(code, list(CROPLAND | NATURAL))
    out[other] = GROUP_OTHER
    return out


ACCOUNT_KEYS = ("cropland", "natural", "built_up", "water", "snow_ice", "other", "not_read")
ACCOUNT_NAMES = {"cropland": "Cropland", "natural": "Natural land", "built_up": "Built-up", "water": "Water",
                 "snow_ice": "Snow and ice", "other": "Other", "not_read": "Not read"}


def accounting(code):
    """Count of points per class group. Every point is in exactly one: cropland (40), natural land, built-up (50),
    water (80), snow and ice (70), other (a code that is none of these), or not read (the tile could not be read in
    this run, or the footprint has no land cover value: open sea or no data)."""
    code = np.asarray(code)
    read = (code != NOT_CLASSIFIED) & (code != 0)
    nat = np.isin(code, list(NATURAL))
    out = {"cropland": int(np.isin(code, list(CROPLAND)).sum()), "natural": int(nat.sum()),
           "built_up": int((code == 50).sum()), "water": int((code == 80).sum()), "snow_ice": int((code == 70).sum()),
           "not_read": int((~read).sum())}
    out["other"] = int(read.sum()) - sum(out[k] for k in ("cropland", "natural", "built_up", "water", "snow_ice"))
    return {k: out[k] for k in ACCOUNT_KEYS}

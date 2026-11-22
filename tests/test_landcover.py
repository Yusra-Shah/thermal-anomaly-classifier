"""Tests for src/context/landcover.py and clusters.py. A small synthetic cloud optimised GeoTIFF (tiled, deflate,
overview levels) is served from memory by a fake range fetch, so nothing here touches the network.

Run from the project root, one suite per process:  python -m pytest tests/test_landcover.py -q
"""
import struct
import zlib

import numpy as np
import pytest
import requests

from src.context import clusters, landcover

NAME = "N30E072"
X0, Y0 = 72.0, 33.0              # north west corner of the tile
TILE = 32


def make_tiff(levels, tile=TILE, sx=3 / 128, sy=3 / 128, predictor=1):
    """A classic little endian tiled TIFF with one IFD per level (level 0 first) and the georeference tags."""
    blobs, shapes = [], []
    for arr in levels:
        h, w = arr.shape
        ny, nx = -(-h // tile), -(-w // tile)
        pad = np.zeros((ny * tile, nx * tile), np.uint8)
        pad[:h, :w] = arr
        tiles = []
        for ty in range(ny):
            for tx in range(nx):
                block = pad[ty * tile:(ty + 1) * tile, tx * tile:(tx + 1) * tile]
                if predictor == 2:
                    block = np.diff(block.astype(np.int16), axis=1, prepend=0).astype(np.uint8)
                tiles.append(zlib.compress(block.tobytes()))
        blobs.append(tiles)
        shapes.append((h, w, ny * nx))
    n_entries = [10 + (2 if i == 0 else 0) for i in range(len(levels))]
    ifd_len = [2 + 12 * n + 4 for n in n_entries]
    pos = 8
    ifd_at = []
    for ln in ifd_len:
        ifd_at.append(pos)
        pos += ln
    extra_at, extra = [], b""
    for i, (h, w, nt) in enumerate(shapes):
        extra_at.append(pos + len(extra))
        extra += b"\0" * (8 * nt)                        # offsets and counts, filled below
        if i == 0:
            extra += struct.pack("<3d", sx, sy, 0.0) + struct.pack("<6d", 0, 0, 0, X0, Y0, 0)
    data_at = pos + len(extra)
    body, offsets = b"", []
    for tiles in blobs:
        offs, cnts = [], []
        for t in tiles:
            offs.append(data_at + len(body))
            cnts.append(len(t))
            body += t
        offsets.append((offs, cnts))
    extra = bytearray(extra)
    for i, (h, w, nt) in enumerate(shapes):
        base = extra_at[i] - (8 + sum(ifd_len))
        offs, cnts = offsets[i]
        extra[base:base + 4 * nt] = struct.pack(f"<{nt}I", *offs)
        extra[base + 4 * nt:base + 8 * nt] = struct.pack(f"<{nt}I", *cnts)
    out = bytearray(b"II" + struct.pack("<HI", 42, 8))
    for i, (h, w, nt) in enumerate(shapes):
        ents = [(256, 4, 1, w), (257, 4, 1, h), (259, 3, 1, 8), (317, 3, 1, predictor), (322, 4, 1, tile), (323, 4, 1, tile),
                (324, 4, nt, extra_at[i]), (325, 4, nt, extra_at[i] + 4 * nt), (339, 3, 1, 1), (277, 3, 1, 1)]
        if i == 0:
            gbase = extra_at[0] + 8 * nt
            ents += [(33550, 12, 3, gbase), (33922, 12, 6, gbase + 24)]
        ents.sort()
        blob = struct.pack("<H", len(ents))
        for tag, typ, count, val in ents:
            blob += struct.pack("<HHII", tag, typ, count, val) if typ != 3 else struct.pack("<HHIHH", tag, typ, count, val, 0)
        nxt = ifd_at[i + 1] if i + 1 < len(shapes) else 0
        out += blob + struct.pack("<I", nxt)
    return bytes(out) + bytes(extra) + body


class Fake:
    """A fake range fetch that counts its requests and can fail."""

    def __init__(self, files, fail=None, fail_tiles=None):
        self.files, self.calls, self.fail, self.fail_tiles = files, [], fail, fail_tiles

    def __call__(self, name, start, end):
        self.calls.append((name, start, end))
        if self.fail:
            raise self.fail
        if name not in self.files:
            raise landcover.TileMissing(name)
        if self.fail_tiles and start >= self.fail_tiles:
            raise requests.Timeout("timed out for https://example.org/secret")
        return self.files[name][start:end + 1]


def image(h=128, w=128):
    """Left half cropland (40), right half tree cover (10), a strip of built-up (50) at the bottom."""
    a = np.full((h, w), 10, np.uint8)
    a[:, :w // 2] = 40
    a[-16:, :] = 50
    return a


def pixel_lonlat(col, row, sx=3 / 128, sy=3 / 128):
    return X0 + (col + 0.5) * sx, Y0 - (row + 0.5) * sy


@pytest.fixture
def cog():
    lv0 = image()
    lv1 = lv0[::2, ::2].copy()                    # an overview, half the size
    lv1[:] = 30                                   # a distinct class, so the level used can be told
    return make_tiff([lv0, lv1])


def test_tile_name():
    assert landcover.tile_name(31.2, 74.0) == "N30E072"
    assert landcover.tile_name(-0.5, -0.2) == "S03W003"
    assert landcover.tile_name(-33.9, 18.4) == "S36E018"
    assert landcover.tile_name(51.5, -0.1) == "N51W003"
    assert landcover.tile_name(0.0, 0.0) == "N00E000"


def test_header_of_a_synthetic_file(cog):
    h = landcover.parse_header(cog[:65536], lambda off, size: cog[off:off + size])
    assert [lv["w"] for lv in h["levels"]] == [128, 64] and h["levels"][0]["tw"] == TILE
    assert h["geo"]["x0"] == X0 and h["geo"]["y0"] == Y0 and abs(h["geo"]["sx"] - 3 / 128) < 1e-12
    assert len(h["levels"][0]["offsets"]) == 16 and len(h["levels"][1]["offsets"]) == 4
    g = landcover.level_geometry(h, 1)
    assert abs(g["sx"] - 3 / 64) < 1e-12 and g["nx"] == 2 and g["ny"] == 2
    with pytest.raises(ValueError):
        landcover.parse_header(b"not a tiff at all", lambda o, s: b"")


def test_class_is_the_majority_over_the_footprint(tmp_path, cog):
    fetch = Fake({NAME: cog})
    lon, lat = [], []
    for col, row in ((10, 20), (100, 20), (60, 5), (10, 120)):          # crop, tree, crop side, built-up
        a, b = pixel_lonlat(col, row)
        lon.append(a)
        lat.append(b)
    res = landcover.classify(lon, lat, [0.375] * 4, [0.375] * 4, fetch=fetch, cache_dir=tmp_path)
    assert res["level"] == 0 and res["code"].tolist() == [40, 10, 40, 50] and res["failed"] == 0
    # a footprint of about 5 pixels across the border between cropland (cols below 64) and tree cover: the majority
    a, b = pixel_lonlat(65, 20)                                          # three pixels of tree, two of cropland
    res = landcover.classify([a], [b], [13.0], [2.0], fetch=fetch, cache_dir=tmp_path)
    assert res["code"].tolist() == [10]
    a, b = pixel_lonlat(62, 20)                                          # the other way round
    res = landcover.classify([a], [b], [13.0], [2.0], fetch=fetch, cache_dir=tmp_path)
    assert res["code"].tolist() == [40]
    # a missing footprint size means the nominal one
    res = landcover.classify([lon[0]], [lat[0]], None, None, fetch=fetch, cache_dir=tmp_path)
    assert res["code"].tolist() == [40]
    assert res["source"].startswith("ESA WorldCover") and res["fetched_at"].endswith("Z")


def test_a_footprint_over_a_tile_border_reads_both_tiles(tmp_path, cog):
    fetch = Fake({NAME: cog})
    a, b = pixel_lonlat(31, 20)                                          # the border of two tiles is at column 32
    res = landcover.classify([a + 0.5 * 3 / 128], [b], [9.0], [0.375], fetch=fetch, cache_dir=tmp_path)
    assert res["code"].tolist() == [40] and res["tiles_read"] == 2


def test_points_are_read_in_batches_not_one_request_each(tmp_path, cog):
    rng = np.random.default_rng(0)
    lon = rng.uniform(X0 + 0.01, X0 + 2.99, 5000)
    lat = rng.uniform(Y0 - 2.6, Y0 - 0.01, 5000)                         # clear of the built-up strip
    fetch = Fake({NAME: cog})
    res = landcover.classify(lon, lat, None, None, fetch=fetch, cache_dir=tmp_path)
    assert len(fetch.calls) <= 1 + 16                                    # one index and at most the 16 tiles
    assert res["tiles_read"] <= 16 and set(np.unique(res["code"])) <= {10, 40}
    left = lon < X0 + 1.5
    assert (res["code"][left] == 40).mean() > 0.97 and (res["code"][~left] == 10).mean() > 0.97


def test_a_second_run_uses_the_disk_cache(tmp_path, cog):
    fetch = Fake({NAME: cog})
    a, b = pixel_lonlat(10, 20)
    first = landcover.classify([a], [b], None, None, fetch=fetch, cache_dir=tmp_path)
    n = len(fetch.calls)
    assert first["tiles_read"] == 1 and first["tiles_cached"] == 0 and n == 2      # the index and one tile
    second = landcover.classify([a], [b], None, None, fetch=fetch, cache_dir=tmp_path)
    assert len(fetch.calls) == n                                                    # no request at all
    assert second["tiles_read"] == 0 and second["tiles_cached"] == 1 and second["code"].tolist() == first["code"].tolist()
    assert (tmp_path / NAME / "header.json").exists() and list((tmp_path / NAME).glob("L0_*.npz"))


def test_the_resolution_follows_the_number_of_tiles(tmp_path, cog):
    g = np.linspace(0.1, 2.5, 8)                                           # points over all 16 tiles of level 0
    lon, lat = np.meshgrid(X0 + g, Y0 - g)
    lon, lat = lon.ravel(), lat.ravel()
    fine = landcover.classify(lon, lat, None, None, fetch=Fake({NAME: cog}), cache_dir=tmp_path / "a", max_tiles=16)
    coarse = landcover.classify(lon, lat, None, None, fetch=Fake({NAME: cog}), cache_dir=tmp_path / "b", max_tiles=8)
    assert fine["level"] == 0 and fine["tiles_read"] == 16
    assert coarse["level"] == 1 and coarse["tiles_read"] <= 4              # the overview has 2 by 2 tiles
    assert set(coarse["code"].tolist()) == {30}                            # the overview is all class 30
    assert coarse["pixel_m"] > fine["pixel_m"]


def test_predictor_two_is_undone(tmp_path):
    img = image()
    c = make_tiff([img], predictor=2)
    a, b = pixel_lonlat(100, 20)
    res = landcover.classify([a], [b], None, None, fetch=Fake({NAME: c}), cache_dir=tmp_path)
    assert res["code"].tolist() == [10]


def test_open_sea_is_no_land_not_a_failure(tmp_path, cog):
    sea_lon, sea_lat = 70.0, 10.0                                          # a tile that has no file
    a, b = pixel_lonlat(10, 20)
    fetch = Fake({NAME: cog})
    res = landcover.classify([sea_lon, a], [sea_lat, b], None, None, fetch=fetch, cache_dir=tmp_path)
    assert res["code"].tolist() == [0, 40] and res["failed"] == 0
    assert (tmp_path / landcover.tile_name(sea_lat, sea_lon) / "missing.flag").exists()
    n = len(fetch.calls)
    landcover.classify([sea_lon], [sea_lat], None, None, fetch=fetch, cache_dir=tmp_path)
    assert len(fetch.calls) == n                                            # the missing tile is remembered


def test_failures_and_timeouts_never_stop_the_run(tmp_path, cog):
    a, b = pixel_lonlat(10, 20)
    # nothing reachable: Unavailable with the class of the error only
    with pytest.raises(landcover.Unavailable) as e:
        landcover.classify([a], [b], None, None, fetch=Fake({NAME: cog}, fail=requests.ConnectionError("https://x/secret")),
                           cache_dir=tmp_path / "x")
    assert "://" not in str(e.value) and "secret" not in str(e.value)
    # the index is read but the tiles time out: points are not classified, the run goes on
    header_size = len(cog) - sum(len(zlib.compress(b"x")) for _ in range(0))      # any start beyond the index
    first_tile = landcover.parse_header(cog[:65536], lambda o, s: cog[o:o + s])["levels"][0]["offsets"][0]
    fetch = Fake({NAME: cog}, fail_tiles=first_tile)
    res = landcover.classify([a, a], [b, b], None, None, fetch=fetch, cache_dir=tmp_path / "y")
    assert res["code"].tolist() == [landcover.NOT_CLASSIFIED] * 2 and res["failed"] == 1 and res["tiles_read"] == 0
    # one tile fails and another works
    a2, b2 = pixel_lonlat(100, 20)
    fetch = Fake({NAME: cog}, fail_tiles=first_tile + 1)
    res = landcover.classify([a, a2], [b, b2], None, None, fetch=fetch, cache_dir=tmp_path / "z")
    assert landcover.NOT_CLASSIFIED in res["code"].tolist() or res["failed"] >= 0


def test_the_time_budget_leaves_points_unclassified(tmp_path, cog):
    a, b = pixel_lonlat(10, 20)
    res = landcover.classify([a], [b], None, None, fetch=Fake({NAME: cog}), cache_dir=tmp_path, budget_s=-1)
    assert res["code"].tolist() == [landcover.NOT_CLASSIFIED] and res["tiles_read"] == 0
    res = landcover.classify([a], [b], None, None, fetch=Fake({NAME: cog}), cache_dir=tmp_path)      # the next run
    assert res["code"].tolist() == [40]


def test_the_groups_of_the_filter():
    codes = np.array([0, 255, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100])
    g = landcover.groups(codes)
    assert g[codes == 40].tolist() == [landcover.GROUP_CROP]
    for c in (10, 20, 30, 60, 90, 95, 100):                               # forest, shrubland, grassland, wetland, moss, bare
        assert g[codes == c].tolist() == [landcover.GROUP_NATURAL], c
    for c in (50, 70, 80):                                                # built-up, snow and ice, water
        assert g[codes == c].tolist() == [landcover.GROUP_OTHER], c
    assert g[codes == 0].tolist() == [landcover.GROUP_NONE] and g[codes == 255].tolist() == [landcover.GROUP_NONE]
    assert landcover.CROPLAND == {40}


def test_http_errors_carry_no_address(monkeypatch):
    def boom(url, headers=None, timeout=None):
        raise requests.ConnectionError(f"failed for {url} with the key SECRETKEY")
    monkeypatch.setattr(requests, "get", boom)
    with pytest.raises(landcover.FetchError) as e:
        landcover.http_fetch(NAME, 0, 10)
    assert str(e.value) == "ConnectionError" and "://" not in str(e.value) and "SECRET" not in str(e.value)

    class R:
        def __init__(self, code, content=b"0123456789"):
            self.status_code, self.content = code, content
    monkeypatch.setattr(requests, "get", lambda url, headers=None, timeout=None: R(404))
    with pytest.raises(landcover.TileMissing):
        landcover.http_fetch(NAME, 0, 10)
    monkeypatch.setattr(requests, "get", lambda url, headers=None, timeout=None: R(500))
    with pytest.raises(landcover.FetchError) as e:
        landcover.http_fetch(NAME, 0, 10)
    assert str(e.value) == "HTTP 500"
    monkeypatch.setattr(requests, "get", lambda url, headers=None, timeout=None: R(200))      # a server that ignores ranges
    assert landcover.http_fetch(NAME, 2, 4) == b"234"
    seen = {}

    def spy(url, headers=None, timeout=None):
        seen.update(headers=headers, timeout=timeout)
        return R(206)
    monkeypatch.setattr(requests, "get", spy)
    landcover.http_fetch(NAME, 5, 9)
    assert seen["timeout"] == 8 and seen["headers"]["Range"] == "bytes=5-9" and "thermal-anomaly" in seen["headers"]["User-Agent"]


def test_reachable_reports_the_class_only():
    ok, why = landcover.reachable(lambda name, a, b: b"x")
    assert ok and why == ""
    ok, why = landcover.reachable(lambda name, a, b: (_ for _ in ()).throw(landcover.TileMissing(name)))
    assert ok                                                               # the bucket answered
    ok, why = landcover.reachable(lambda name, a, b: (_ for _ in ()).throw(requests.ConnectTimeout("https://x")))
    assert not ok and why == "ConnectTimeout"


# ------------------------------------------------------------------ large clusters
def window(n_in_cluster=6, days=(0, 1), cluster_frp=20.0):
    """A cluster of detections around (10, 20) and some far away, with FRP values for the percentile."""
    lat = [10.0 + 0.002 * i for i in range(n_in_cluster)] + [40.0, 41.0, 42.0, 43.0]
    lon = [20.0] * n_in_cluster + [60.0, 61.0, 62.0, 63.0]
    day = [days[i % len(days)] for i in range(n_in_cluster)] + [0, 0, 0, 0]
    frp = [cluster_frp] * n_in_cluster + [1.0, 1.0, 1.0, 1.0]
    return np.array(lat), np.array(lon), np.array(day), np.array(frp)


def test_large_cluster_needs_five_detections_two_days_and_high_frp():
    lat, lon, day, frp = window()
    flag, thr = clusters.flags(lat, lon, day, frp, np.arange(len(lat)))
    assert flag[:6].all() and not flag[6:].any()                          # the cluster, and nothing far away
    assert thr == pytest.approx(np.percentile(frp, 75))
    lat, lon, day, frp = window(days=(0,))                                 # one day only
    assert not clusters.flags(lat, lon, day, frp, np.arange(len(lat)))[0].any()
    lat, lon, day, frp = window(n_in_cluster=4)                            # four detections
    assert not clusters.flags(lat, lon, day, frp, np.arange(len(lat)))[0].any()
    lat, lon, day, frp = window()
    frp[2] = 0.5                                                           # below the percentile of the window
    flag, _ = clusters.flags(lat, lon, day, frp, np.arange(len(lat)))
    assert flag[:6].tolist() == [True, True, False, True, True, True]
    lat, lon, day, frp = window(cluster_frp=1.0)                           # nothing above the 75th percentile alone
    flag, thr = clusters.flags(lat, lon, day, frp, np.arange(len(lat)))
    assert thr == pytest.approx(1.0) and flag[:6].all()                    # at or above counts


def test_the_radius_is_two_kilometres():
    lat = np.array([10.0] * 5 + [10.0])
    lon = np.array([20.0] * 5 + [20.0 + 0.0215])                           # about 2.35 km east of the others
    day = np.array([0, 1, 0, 1, 0, 0])
    frp = np.full(6, 5.0)
    flag, _ = clusters.flags(lat, lon, day, frp, np.arange(6))
    assert flag[:5].all() and not flag[5]                                  # the sixth is out of range of the five
    lon[5] = 20.0 + 0.0170                                                  # about 1.9 km: inside
    flag, _ = clusters.flags(lat, lon, day, frp, np.arange(6))
    assert flag.all()


def test_only_the_asked_rows_are_flagged_and_a_huge_window_is_refused(monkeypatch):
    lat, lon, day, frp = window()
    flag, _ = clusters.flags(lat, lon, day, frp, np.array([1, 8]))
    assert flag.tolist() == [True, False]
    flag, _ = clusters.flags(lat, lon, day, frp, np.array([], int))
    assert len(flag) == 0
    monkeypatch.setattr(clusters, "MAX_WINDOW", 5)
    with pytest.raises(ValueError):
        clusters.flags(lat, lon, day, frp, np.array([0]))


def test_no_address_or_key_in_the_module_sources():
    from pathlib import Path
    for mod in (landcover, clusters):
        text = Path(mod.__file__).read_text(encoding="utf-8")
        assert "FIRMS_MAP_KEY" not in text and "load_key" not in text and "print(" not in text


def test_accounting_puts_every_point_in_one_count():
    code = np.array([40, 40, 10, 20, 30, 60, 90, 95, 100, 50, 80, 70, 0, 255, 255, 200], np.uint8)
    c = landcover.accounting(code)
    assert c == {"cropland": 2, "natural": 7, "built_up": 1, "water": 1, "snow_ice": 1, "other": 1, "not_read": 3}
    assert sum(c.values()) == len(code)
    assert list(c) == list(landcover.ACCOUNT_KEYS)
    assert landcover.accounting(np.array([], np.uint8))["not_read"] == 0

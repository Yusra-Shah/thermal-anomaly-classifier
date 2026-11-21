"""Tests for src/context/landtable.py: the 0.01 degree cell to land class table and its background fill.
Every read is a fake or a synthetic raster served from memory, so nothing here touches the network.

Run from the project root, one suite per process:  python -m pytest tests/test_landtable.py -q
"""
import threading
import time

import numpy as np
import pandas as pd
import pytest

from src.context import landcover, landtable
from tests.test_landcover import NAME, X0, Y0, Fake, image, make_tiff


def table(tmp_path, classify=None):
    return landtable.LandTable(tmp_path / "landtable.parquet", classify)


def fake_reader(calls, code_of=None):
    def classify(lon, lat, scan_km=None, track_km=None, **kw):
        calls.append({"n": len(lon), "lon": np.array(lon), "lat": np.array(lat), "kw": kw})
        if code_of:
            return {"code": np.array([code_of(a, b) for a, b in zip(lon, lat)], np.uint8)}
        return {"code": np.full(len(lon), 40, np.uint8)}
    return classify


def test_cell_ids_and_centres():
    ids = landtable.cell_ids([32.0415, 32.0449, 32.0501, -0.001, 90.0], [74.5782, 74.5799, 74.5782, -0.001, 180.0])
    assert ids[0] == ids[1] and ids[0] != ids[2]                       # the same 0.01 degree cell, and the next one up
    assert ids[3] != landtable.cell_ids([0.001], [0.001])[0]            # either side of the equator and meridian
    assert ids.dtype == np.int64 and ids[4] % landtable.LON_CELLS == 0  # longitude 180 wraps to the first column
    lat, lon = landtable.cell_center(ids[:1])
    assert abs(lat[0] - 32.045) < 1e-9 and abs(lon[0] - 74.575) < 1e-9
    assert landtable.cell_ids(lat, lon)[0] == ids[0]                    # the centre is in its own cell
    big = np.random.default_rng(0).uniform([-89.99, -179.99], [89.99, 179.99], (2000, 2))
    c = landtable.cell_ids(big[:, 0], big[:, 1])
    la, lo = landtable.cell_center(c)
    assert (landtable.cell_ids(la, lo) == c).all() and (np.abs(la - big[:, 0]) <= 0.00501).all()


def test_lookup_add_replace_and_the_file(tmp_path):
    t = table(tmp_path)
    ids = landtable.cell_ids([10.005, 20.005, 30.005], [10.005, 20.005, 30.005])
    assert (t.lookup(ids) == landtable.UNKNOWN).all() and len(t) == 0
    v = t.version
    t.add(ids[:2], [40, 10])
    assert t.lookup(ids).tolist() == [40, 10, landtable.UNKNOWN] and t.version == v + 1
    t.add(ids[:1], [20])                                                # a later reading replaces the earlier one
    assert t.lookup(ids).tolist() == [20, 10, landtable.UNKNOWN] and len(t) == 2
    t.add(ids[2:], [landtable.NOT_READ])                                 # not read: known, but never written
    assert t.lookup(ids).tolist() == [20, 10, 255]
    t._save()
    again = table(tmp_path)
    assert again.lookup(ids).tolist() == [20, 10, landtable.UNKNOWN] and len(again) == 2
    assert tmp_path.joinpath("landtable.parquet").stat().st_size < 5000    # one small file
    assert not list(tmp_path.glob("*.tmp"))
    assert (table(tmp_path / "other").lookup(ids) == landtable.UNKNOWN).all()


def test_request_never_waits_for_a_read(tmp_path):
    gate = threading.Event()

    def slow(lon, lat, scan_km=None, track_km=None, **kw):
        gate.wait(5)
        return {"code": np.full(len(lon), 40, np.uint8)}
    t = table(tmp_path, slow)
    ids = landtable.cell_ids(np.linspace(10, 11, 50), np.linspace(10, 11, 50))
    t0 = time.perf_counter()
    n = t.request(ids)
    assert time.perf_counter() - t0 < 0.3 and n == len(np.unique(ids))   # returns at once, the read goes on
    assert t.running() and (t.lookup(ids) == landtable.UNKNOWN).all()
    gate.set()
    assert t.wait(10) and (t.lookup(ids) == 40).all() and t.remaining() == 0


def test_cells_are_read_once_in_batches_grouped_by_tile(tmp_path):
    calls = []
    t = table(tmp_path, fake_reader(calls))
    rng = np.random.default_rng(1)
    lat = rng.uniform(0, 8, 3000)
    lon = rng.uniform(0, 8, 3000)
    ids = landtable.cell_ids(lat, lon)
    t.request(ids)
    assert t.wait(20)
    assert t.lookup(ids).tolist() == [40] * len(ids)
    read = sum(c["n"] for c in calls)
    assert read == len(np.unique(ids)) and len(calls) > 1                 # every cell once, in several batches
    for c in calls:                                                       # a batch spans few tiles, so one read is 10 m
        la, lo = c["lat"], c["lon"]
        tiles = set(zip(np.floor(la / landtable.TILE_DEG).astype(int), np.floor(lo / landtable.TILE_DEG).astype(int)))
        assert len(tiles) <= landtable.BATCH_TILES
        assert c["kw"].get("budget_s", 0) >= 60
    before = len(calls)
    t.request(ids, ids)                                                   # nothing new: no read at all
    assert t.wait(5) and len(calls) == before and t.remaining() == 0


def test_the_drawn_cells_are_read_before_the_rest(tmp_path):
    order = []

    def classify(lon, lat, scan_km=None, track_km=None, **kw):
        order.extend(landtable.cell_ids(lat, lon).tolist())
        return {"code": np.full(len(lon), 10, np.uint8)}
    t = table(tmp_path, classify)
    first = landtable.cell_ids([10.005, 10.015], [10.005, 10.005])
    rest = landtable.cell_ids(np.arange(20, 60, 1.0) + 0.005, np.arange(20, 60, 1.0) + 0.005)
    t.request(first, rest)
    assert t.wait(10)
    assert set(order[:2]) == set(first.tolist()) or set(first.tolist()) <= set(order[:len(first) + 40])
    assert order.index(first[0]) < order.index(rest[-1]) and len(set(order)) == len(first) + len(rest)


def test_sea_cells_are_stored_and_failures_are_not_written(tmp_path):
    seen = []

    def classify(lon, lat, scan_km=None, track_km=None, **kw):
        seen.append(len(lon))
        return {"code": np.array([0 if a < 0 else landtable.NOT_READ if a > 50 else 30 for a in lon], np.uint8)}
    t = table(tmp_path, classify)
    ids = landtable.cell_ids([5.0, 5.0, 5.0], [-1.005, 10.005, 60.005])    # sea, land, a tile that failed
    t.request(ids)
    assert t.wait(10)
    assert t.lookup(ids).tolist() == [0, 30, 255]
    again = table(tmp_path)
    assert again.lookup(ids).tolist() == [0, 30, landtable.UNKNOWN]         # the failed cell is asked again next time
    n = sum(seen)
    t.request(ids)                                                          # and not again in this process
    assert t.wait(5) and sum(seen) == n


def test_unreachable_worldcover_stops_the_fill_and_says_why(tmp_path):
    def classify(lon, lat, scan_km=None, track_km=None, **kw):
        raise landcover.Unavailable("ConnectTimeout")
    t = table(tmp_path, classify)
    t.request(landtable.cell_ids([10.005], [10.005]), landtable.cell_ids(np.arange(20, 40, 1.0), np.arange(20, 40, 1.0)))
    assert t.wait(10) and t.error == "ConnectTimeout" and t.remaining() == 0 and not t.running()
    assert "://" not in t.error


def test_a_reader_that_raises_marks_the_batch_not_read_and_goes_on(tmp_path):
    def classify(lon, lat, scan_km=None, track_km=None, **kw):
        raise ValueError("https://example.org/secret")
    t = table(tmp_path, classify)
    ids = landtable.cell_ids([10.005], [10.005])
    t.request(ids)
    assert t.wait(10) and t.lookup(ids).tolist() == [255] and t.error == "ValueError" and "secret" not in str(t.error)


def test_real_reader_on_a_synthetic_raster(tmp_path):
    fetch = Fake({NAME: make_tiff([image()])})

    def classify(lon, lat, scan_km=None, track_km=None, **kw):
        return landcover.classify(lon, lat, scan_km, track_km, fetch=fetch, cache_dir=tmp_path / "wc")
    t = table(tmp_path, classify)
    # the synthetic tile is 3 degrees across with cropland on its left half and tree cover on its right half
    lat = np.full(4, Y0 - 1.0)
    lon = np.array([X0 + 0.3, X0 + 0.9, X0 + 2.1, X0 + 2.7])
    ids = landtable.cell_ids(lat, lon)
    t.request(ids)
    assert t.wait(20)
    got = t.lookup(ids)
    assert got.tolist() == [40, 40, 10, 10]
    groups = landcover.groups(got)
    assert groups.tolist() == [landcover.GROUP_CROP] * 2 + [landcover.GROUP_NATURAL] * 2
    assert t.read_count == 4 and t.lookup_points(lat, lon).tolist() == got.tolist()


def test_probe_is_kept_ten_minutes(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(landcover, "reachable", lambda fetch=None: (calls.append(1), (False, "ConnectTimeout"))[1])
    t = table(tmp_path)
    assert t.probe() == (False, "ConnectTimeout") and t.probe() == (False, "ConnectTimeout") and len(calls) == 1


def test_the_module_has_no_address_key_or_print():
    from pathlib import Path
    text = Path(landtable.__file__).read_text(encoding="utf-8")
    assert "FIRMS_MAP_KEY" not in text and "load_key" not in text and "print(" not in text and "http" not in text


def test_the_fill_does_not_keep_the_tiles_on_disk(tmp_path):
    fetch = Fake({NAME: make_tiff([image()])})
    wc = tmp_path / "wc"

    def classify(lon, lat, scan_km=None, track_km=None, **kw):
        return landcover.classify(lon, lat, scan_km, track_km, fetch=fetch, cache_dir=wc, **kw)
    t = table(tmp_path, classify)
    t.request(landtable.cell_ids([Y0 - 1.0], [X0 + 0.3]))
    assert t.wait(20) and t.lookup(landtable.cell_ids([Y0 - 1.0], [X0 + 0.3])).tolist() == [40]
    assert (wc / NAME / "header.json").exists() and not list((wc / NAME).glob("L*_*.npz"))   # only the small index is kept
    landcover.classify([X0 + 0.3], [Y0 - 1.0], None, None, fetch=fetch, cache_dir=wc)           # the default still keeps tiles
    assert list((wc / NAME).glob("L*_*.npz"))


def test_the_rest_of_a_view_is_queued_up_to_a_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(landtable, "MAX_REST_CELLS", 100)
    gate = threading.Event()

    def slow(lon, lat, scan_km=None, track_km=None, **kw):
        gate.wait(5)
        return {"code": np.full(len(lon), 40, np.uint8)}
    t = table(tmp_path, slow)
    first = landtable.cell_ids([1.005, 1.015], [1.005, 1.005])
    rest = landtable.cell_ids(np.linspace(-60, 60, 900), np.linspace(-170, 170, 900))
    n = t.request(first, rest)
    assert n <= 102 and t.remaining() <= 102
    gate.set()
    assert t.wait(10)

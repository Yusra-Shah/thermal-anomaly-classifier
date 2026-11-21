"""Tests for scripts/warm_cache.py: the options, the count and estimate printed first, the country filter and the resume.
The store, the geography, the reader and the downloads are fakes: nothing here reads the real store or the network.

Run from the project root, one suite per process:  python -m pytest tests/test_warm_cache.py -q
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.context import landtable, people

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("warm_cache", ROOT / "scripts" / "warm_cache.py")
warm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(warm)


def test_options_and_defaults():
    a = warm.parse_args([])
    assert a.country is None and a.days == 7.0
    a = warm.parse_args(["--country", "Pakistan", "India", "--days", "1"])
    assert a.country == ["Pakistan", "India"] and a.days == 1.0
    assert warm.parse_args(["--days", "0.5"]).days == 0.5


def test_country_names_and_codes():
    codes, unknown = warm.resolve_countries(["pakistan", "INDIA", "NG", "Atlantis"])
    assert codes == ["PK", "IN", "NG"] and unknown == ["Atlantis"]
    assert warm.resolve_countries([" Pakistan ", "pakistan"]) == (["PK"], [])


def test_durations_read_naturally():
    assert warm.duration(40) == "40 seconds" and warm.duration(600) == "10 minutes" and warm.duration(7200) == "2.0 hours"


@pytest.fixture
def fake_world(tmp_path, monkeypatch):
    """A store of three small clusters (Pakistan, India, Brazil) with a fake reader and fake downloads."""
    rng = np.random.default_rng(0)
    parts = []
    for name, lat, lon in (("PK", 32.0, 74.5), ("IN", 22.0, 78.0), ("BR", -10.0, -55.0)):
        n = 60
        parts.append(pd.DataFrame({"latitude": lat + rng.uniform(0, 0.3, n), "longitude": lon + rng.uniform(0, 0.3, n),
                                   "cc": name}))
    df = pd.concat(parts, ignore_index=True)
    df["cell_id"] = (df["latitude"].round(2).astype(str) + "_" + df["longitude"].round(2).astype(str))
    seen_since = []

    def read_recent(path=None, since=None, columns=None, **kw):
        seen_since.append(since)
        return df[["latitude", "longitude", "cell_id"]].copy()

    def add_geo(frame, cache_path=None):
        out = frame.merge(df.drop_duplicates("cell_id")[["cell_id", "cc"]], on="cell_id", how="left")
        out["cc"] = out["cc"].astype("category")
        return out
    calls = []

    def classify(lon, lat, scan_km=None, track_km=None, **kw):
        calls.append(len(lon))
        return {"code": np.full(len(lon), 40, np.uint8)}
    table = landtable.LandTable(tmp_path / "landtable.parquet", classify)
    warmed = []

    def fake_warm(cc, get=None, cache_dir=None):
        warmed.append(cc)
        return None
    monkeypatch.setattr(warm.ingest, "read_recent", read_recent)
    monkeypatch.setattr(warm.live, "add_geo", add_geo)
    monkeypatch.setattr(warm.landtable, "get", lambda: table)
    monkeypatch.setattr(warm.people, "warm", fake_warm)
    monkeypatch.setattr(warm.people, "raster_file", lambda iso3, cache_dir=None: tmp_path / f"{iso3}.tif")
    return {"df": df, "table": table, "calls": calls, "warmed": warmed, "since": seen_since}


def test_the_count_and_the_estimate_come_before_the_work(fake_world, capsys):
    assert warm.main(["--country", "Pakistan", "India", "--days", "1"]) == 0
    out = capsys.readouterr().out.splitlines()
    idx = {k: next(i for i, ln in enumerate(out) if k in ln) for k in
           ("cells with a detection", "estimate:", "land cover table:", "finished")}
    assert idx["cells with a detection"] < idx["estimate:"] < idx["land cover table:"] < idx["finished"]
    line = out[idx["cells with a detection"]]
    assert "to read:" in line and "already in the table: 0" in line
    est = out[idx["estimate:"]]
    assert "cells a second" in est and ("seconds" in est or "minutes" in est)
    text = "\n".join(out)
    assert "http" not in text and "://" not in text
    age = pd.Timestamp.now(tz="UTC").tz_localize(None) - fake_world["since"][0]
    assert abs(age - pd.Timedelta(days=1)) < pd.Timedelta(minutes=2)                # --days 1


def test_only_the_named_countries_are_warmed(fake_world, capsys):
    warm.main(["--country", "Pakistan", "India", "--days", "1"])
    df, table = fake_world["df"], fake_world["table"]
    ids = landtable.cell_ids(df["latitude"], df["longitude"])
    cc = df["cc"].to_numpy()
    assert (table.lookup(ids[cc == "PK"]) != landtable.UNKNOWN).all() and (table.lookup(ids[cc == "IN"]) != landtable.UNKNOWN).all()
    assert (table.lookup(ids[cc == "BR"]) == landtable.UNKNOWN).all()                # Brazil was not asked for
    assert sorted(fake_world["warmed"]) == ["IN", "PK"]                              # only their population grids


def test_the_whole_store_by_default_and_a_second_run_resumes(fake_world, capsys):
    assert warm.main([]) == 0
    first = capsys.readouterr().out
    assert "already in the table: 0" in first and sorted(fake_world["warmed"]) == ["BR", "IN", "PK"]
    n = sum(fake_world["calls"])
    assert n > 0
    assert warm.main([]) == 0                                                         # run again: nothing left to read
    second = capsys.readouterr().out
    assert sum(fake_world["calls"]) == n and "to read: 0" in second and "already in the table" in second
    assert "the cells take 0 seconds" in second


def test_unknown_country_and_bad_days_stop_before_any_work(fake_world, capsys):
    assert warm.main(["--country", "Atlantis"]) == 2
    assert "unknown country: Atlantis" in capsys.readouterr().out
    assert warm.main(["--days", "0"]) == 2
    assert not fake_world["calls"] and not fake_world["since"]


def test_a_country_without_detections_says_so(fake_world, capsys):
    assert warm.main(["--country", "Nigeria"]) == 1
    assert "no detection of Nigeria" in capsys.readouterr().out

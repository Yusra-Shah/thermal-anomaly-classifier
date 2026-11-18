"""Detail payload of the Earth panel on a fake store. No network.

    python -m pytest tests/test_detail.py -q
"""
import numpy as np
import pandas as pd
import pytest

from src import devstore, ingest, live, serve
from src.context import detail
from tests import fake_feed as fake

SECTIONS = ["id", "verdict", "facts", "place", "land", "weather", "spread", "exposure", "response", "notes",
            "attribution"]


@pytest.fixture(scope="module")
def models():
    return serve.Models()


@pytest.fixture(scope="module")
def store(tmp_path_factory, models):
    """A fake 20 day South Asia feed of two satellites, scored into a store file like the ingest does."""
    path = tmp_path_factory.mktemp("dev") / "store.parquet"
    raw = []
    for i, code in enumerate(["N20", "N21"]):
        m = fake.master(20, seed=i)
        m["satellite"] = code
        raw.append(m)
    n = devstore.build(pd.concat(raw, ignore_index=True), path=path, models=models, log=lambda *_: None)
    assert n > 0
    return path


def test_store_path_comes_from_one_function(monkeypatch, tmp_path):
    monkeypatch.delenv(ingest.STORE_ENV, raising=False)
    assert ingest.store_path() == live.LIVE_DIR / "store.parquet"
    monkeypatch.setenv(ingest.STORE_ENV, "data/dev/store.parquet")
    assert ingest.store_path() == live.ROOT / "data" / "dev" / "store.parquet"
    monkeypatch.setenv(ingest.STORE_ENV, str(tmp_path / "x.parquet"))
    assert ingest.store_path() == tmp_path / "x.parquet"


def test_dev_store_has_the_store_layout(store):
    df = pd.read_parquet(store)
    for c in ["sensor", "region", "latitude", "longitude", "timestamp", "cell_id", "scan", "track", "pixel_area",
              "p_instant", "pc_instant", "p_history", "pc_history", "history_default", "ingest_time"]:
        assert c in df.columns, c
    assert set(df["sensor"]) == {"NOAA-20", "NOAA-21"}
    assert df["timestamp"].is_monotonic_increasing                      # time ordered, like the live store
    assert not df.duplicated(ingest.KEY_COLS).any()


def test_row_group_reader_matches_a_direct_filter(store):
    """The fallback reader (used where the pyarrow dataset module is missing) gives the rows of a plain filter."""
    full = pd.read_parquet(store)
    t = full["timestamp"].quantile(0.5)
    filt = [("timestamp", ">=", t), ("sensor", "==", "NOAA-21"), ("latitude", "<=", 30.0)]
    got = ingest.read_row_groups(store, ["latitude", "longitude", "timestamp"], filt)
    want = full[(full["timestamp"] >= t) & (full["sensor"] == "NOAA-21") & (full["latitude"] <= 30.0)]
    assert len(got) == len(want) > 0 and list(got.columns) == ["latitude", "longitude", "timestamp"]
    assert np.allclose(np.sort(got["latitude"].to_numpy()), np.sort(want["latitude"].to_numpy()))
    none = ingest.read_row_groups(store, ["latitude"], [("latitude", ">", 1000.0)])
    assert len(none) == 0 and list(none.columns) == ["latitude"]


def test_spread_matches_a_direct_query(store):
    full = pd.read_parquet(store)
    lat, lon = fake.STATIC[0]
    t = full["timestamp"].max()
    got = detail.spread_from_store(lat, lon, t, path=store)
    near = full[(full["timestamp"] > t - pd.Timedelta(hours=72)) & (full["timestamp"] <= t)]
    near = near[detail.distance_km(lat, lon, near["latitude"].to_numpy(), near["longitude"].to_numpy()) <= 2.0]
    assert got["n"] == len(near) > 0
    assert sum(d["n"] for d in got["per_day"]) == got["n"]
    assert got["first_seen"] == f"{near['timestamp'].min():%Y-%m-%d %H:%M}"
    assert got["last_seen"] == f"{near['timestamp'].max():%Y-%m-%d %H:%M}"
    key = near["latitude"].round(4).astype(str) + "," + near["longitude"].round(4).astype(str)
    pix = near[~key.duplicated()]
    assert got["area_km2"] == round(float(pix["pixel_area"].sum()), 2)
    assert "not a fire perimeter" in got["note"] and got["source"] == "own live store"
    # days seen at the cell in the last 90 days
    row = near.iloc[0]
    got = detail.spread_from_store(lat, lon, t, cell_id=row["cell_id"], path=store)
    direct = full[(full["cell_id"] == row["cell_id"]) & (full["timestamp"] <= t)]["acq_date"].nunique()
    assert got["cell_days_90"] == direct


def test_spread_reads_only_the_window(store, monkeypatch):
    """The store is read with a time and a place filter, never whole."""
    seen = {}
    real = ingest.read_recent

    def spy(path=None, since=None, columns=None, **kw):
        out = real(path, since=since, columns=columns, **kw)
        seen.setdefault("calls", []).append((since, kw.get("where"), 0 if out is None else len(out)))
        return out
    monkeypatch.setattr(ingest, "read_recent", spy)
    full = len(pd.read_parquet(store, columns=["latitude"]))
    lat, lon = fake.STATIC[1]
    t = pd.read_parquet(store, columns=["timestamp"])["timestamp"].max()
    detail.spread_from_store(lat, lon, t, path=store)
    since, where, rows = seen["calls"][0]
    assert since is not None and {c for c, _, _ in where} == {"latitude", "longitude", "timestamp"}
    assert rows < full / 10


def test_spread_trend_and_move():
    t = pd.Timestamp("2026-11-10 12:00")
    rows = [(30.0, 70.0, t - pd.Timedelta(hours=40))] + \
           [(30.0 + 0.002 * k, 70.0 + 0.001 * k, t - pd.Timedelta(hours=k)) for k in range(4)]
    df = pd.DataFrame(rows, columns=["latitude", "longitude", "timestamp"])
    s = detail.spread_from_frame(df, 30.0, 70.0, t)
    assert s["n"] == 5 and s["trend"] == "growing" and s["move_km"] is not None
    assert s["area_km2"] == round(s["pixels"] * detail.NOMINAL_PIXEL_KM2, 2)      # nominal pixels without scan
    empty = detail.spread_from_frame(df, 10.0, 10.0, t)
    assert empty["n"] == 0 and empty["trend"] == "no detections in the window"
    assert detail.spread_from_frame(df.iloc[1:], 30.0, 70.0, t)["trend"] == "new in the last 24 hours"


def test_payload_of_a_stored_detection(store, models):
    full = pd.read_parquet(store)
    r = full.iloc[len(full) // 2]
    row = detail.store_row(r["sensor"], r["latitude"], r["longitude"], r["acq_date"], r["acq_time"], path=store)
    assert row is not None and row["p_history"] == r["p_history"]
    row["p_static"] = row["p_history"]
    row.update({"cc": "PK", "admin1": "Punjab", "city": "Multan"})
    spread = detail.spread_from_store(row["latitude"], row["longitude"], row["timestamp"], row["cell_id"], store)
    bands = models.bands["history"]
    p = detail.build(row, source="live", tier="history", bundle=models.hist, bands=bands, band=bands["strict"],
                     threshold=models.history_threshold, spread=spread)
    assert all(k in p for k in SECTIONS)
    assert p["land"] is None and p["weather"] is None and p["exposure"] is None and p["response"] is None
    v = p["verdict"]
    assert v["label"] in ("fire", "static") and v["band"] in ("fire", "static", "uncertain")
    assert abs(v["p"] - float(row["pc_history"])) < 1e-3                 # same calibrated score as the store
    assert v["trained"] and len(v["top_features"]) == 3
    assert {f["push"] for f in v["top_features"]} <= {"toward static", "toward fire"}
    f = p["facts"]
    assert f["satellite"] == r["sensor"] and f["scan"] == round(float(r["scan"]), 3) and not f["nominal_pixel"]
    assert p["place"]["country"] and p["place"]["province"] == "Punjab"
    assert p["id"] == detail.detection_id(r["sensor"], r["latitude"], r["longitude"], r["acq_date"], r["acq_time"])
    assert any("not a confirmed fire" in n for n in p["notes"])
    assert any("unvalidated" in n for n in p["notes"])


def test_archive_row_without_every_feature(models):
    """An archive row keeps only some features: no top features, nominal pixel, NASA label kept."""
    row = {"latitude": 30.1, "longitude": 70.2, "timestamp": pd.Timestamp("2025-02-06 08:04"),
           "acq_date": pd.Timestamp("2025-02-06"), "bright_ti4": 338.4, "bright_ti5": 300.1, "frp": 2.4,
           "confidence_ord": 0, "is_night": 0, "p_static": 0.12, "is_static": 0, "cc": "PK", "admin1": "Punjab",
           "city": "Multan"}
    bands = models.bands["instant"]
    p = detail.build(row, source="archive", tier="instant", bundle=models.inst, bands=bands, band=bands["default"],
                     threshold=models.instant_threshold, spread=None)
    assert p["verdict"]["top_features"] is None and p["verdict"]["label"] == "fire"
    assert p["facts"]["nominal_pixel"] and p["facts"]["confidence"] == "l" and p["facts"]["daynight"] == "D"
    assert p["nasa_label"] == "fire" and p["id"].startswith("S-NPP|30.10000|70.20000|2025-02-06|0804")
    assert p["facts"]["local_solar_time"] == "2025-02-06 12:44"           # 70.2 degrees east is 4 h 41 min


def test_geometry_helpers():
    assert abs(float(detail.distance_km(0, 0, 0, 1)) - 111.19) < 0.05
    assert detail.compass(detail.bearing_deg(0, 0, 1, 0)) == "N"
    assert detail.compass(detail.bearing_deg(0, 0, 0, 1)) == "E"
    assert detail.compass(225) == "SW" and detail.compass(200) == "S"
    s, n, w, e = detail.box(60, 10, 2.0)
    assert abs((n - s) * 111.32 - 4.0) < 1e-6 and (e - w) > (n - s)

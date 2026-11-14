"""Tests for api/main.py. Fake feed and the saved models, no network.

Run from the project root, one suite per process:  python -m pytest tests/test_api.py -q
"""
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")
from fastapi.testclient import TestClient  # noqa: E402

from api import main as api  # noqa: E402
from src import ingest, live, serve  # noqa: E402
from src import uncertain as U  # noqa: E402
from src.config import MODEL_DIR  # noqa: E402
from tests import fake_feed as fake  # noqa: E402

KEY = "SECRETKEY123"
CONF = {"l": 0, "n": 1, "h": 2}


@pytest.fixture
def live_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "LIVE_DIR", tmp_path)
    api._cache.clear()
    yield tmp_path
    api._cache.clear()


@pytest.fixture
def client(live_dir):
    return TestClient(api.app)


@pytest.fixture
def store(live_dir):
    """A real store built by the ingest job from the fake feed (8 days, both sensors)."""
    ingest.run_once(KEY, fetch=fake.fetch_factory(fake.masters(8)), models=api.get_models(),
                    sleep=lambda s: None, log=lambda *a: None)
    return ingest.read_store()


def detection(i, **over):
    rng = np.random.default_rng(i)
    d = {"latitude": float(rng.uniform(10, 40)), "longitude": float(rng.uniform(60, 100)),
         "acq_date": "2026-11-01", "acq_time": int(rng.integers(0, 24) * 100 + rng.integers(0, 60)),
         "bright_ti4": round(float(rng.normal(335, 14)), 1), "bright_ti5": round(float(rng.normal(300, 8)), 1),
         "frp": round(float(rng.gamma(2.0, 2.0)), 2), "scan": round(float(rng.uniform(0.4, 0.7)), 2),
         "track": round(float(rng.uniform(0.4, 0.6)), 2), "confidence": ["l", "n", "h"][i % 3],
         "daynight": "N" if i % 2 else "D"}
    d.update(over)
    return d


def expected_instant(dets):
    """The saved instant bundle applied by hand: features, scores, calibrated probability, bands."""
    inst = joblib.load(MODEL_DIR / serve.INSTANT_FILE)
    b = U.bands_of(inst, "best_model_headline_band.json")
    df = pd.DataFrame(dets)
    df["bright_diff"] = df["bright_ti4"] - df["bright_ti5"]
    df["pixel_area"] = df["scan"] * df["track"]
    df["frp_per_area"] = (df["frp"] / df["pixel_area"]).fillna(0)
    df["confidence_ord"] = df["confidence"].map(CONF)
    df["is_night"] = (df["daynight"] == "N").astype(int)
    p = inst["model"].predict_proba(df[inst["features"]])[:, 1]
    cal = U.apply_calibrator(b["calibrator"], p)
    names = {0: "fire", 1: "static", 2: "uncertain"}
    return {"p": p, "cal": cal, "label": np.where(p >= inst["threshold"], "static", "fire"),
            "default": [names[int(x)] for x in U.decide(cal, b["default"])],
            "strict": [names[int(x)] for x in U.decide(cal, b["strict"])]}


def test_predict_equals_saved_model_output(client):
    dets = [detection(i) for i in range(60)]
    r = client.post("/predict", json={"detections": dets})
    assert r.status_code == 200, r.text
    got, want = r.json()["predictions"], expected_instant(dets)
    assert [g["index"] for g in got] == list(range(60))
    assert np.allclose([g["score"] for g in got], want["p"], atol=1e-6)
    assert np.allclose([g["probability"] for g in got], want["cal"], atol=1e-6)
    assert [g["label"] for g in got] == want["label"].tolist()
    assert [g["band_default"] for g in got] == want["default"]
    assert [g["band_strict"] for g in got] == want["strict"]
    assert all(g["tier"] == "instant" and "instant tier" in g["tier_reason"] for g in got)
    assert r.json()["tier_counts"] == {"instant": 60, "history": 0}
    # order and repeated rows are kept: a request with the same detection twice answers twice
    twice = client.post("/predict", json={"detections": [dets[3], dets[1], dets[3]]}).json()["predictions"]
    assert twice[0]["score"] == twice[2]["score"] == got[3]["score"] and twice[1]["score"] == got[1]["score"]


def test_history_fields_use_the_history_tier(client):
    models = api.get_models()
    dets = [detection(i) for i in range(5)]
    hist = {f: float(k % 5) for k, f in enumerate(models.history_only)}
    r = client.post("/predict", json={"detections": [dict(d, history=hist) for d in dets]})
    assert r.status_code == 200, r.text
    got = r.json()["predictions"]
    base = pd.DataFrame(dets)
    base["bright_diff"] = base["bright_ti4"] - base["bright_ti5"]
    base["pixel_area"] = base["scan"] * base["track"]
    base["frp_per_area"] = base["frp"] / base["pixel_area"]
    base["confidence_ord"] = base["confidence"].map(CONF)
    base["is_night"] = (base["daynight"] == "N").astype(int)
    for f, v in hist.items():
        base[f] = v
    hb = joblib.load(MODEL_DIR / serve.HISTORY_FILE)
    p = hb["model"].predict_proba(base[hb["features"]])[:, 1]
    assert np.allclose([g["score"] for g in got], p, atol=1e-6)
    assert all(g["tier"] == "history" and g["tier_reason"] == "history fields supplied" for g in got)
    thr = float(hb["thresholds"][hb["default_operating_point"]])
    assert [g["label"] for g in got] == np.where(p >= thr, "static", "fire").tolist()
    # incomplete or unknown history fields are refused with the names
    bad = dict(hist)
    bad.pop(models.history_only[0])
    r = client.post("/predict", json={"detections": [dict(dets[0], history=bad)]})
    assert r.status_code == 422 and models.history_only[0] in json.dumps(r.json())
    r = client.post("/predict", json={"detections": [dict(dets[0], history=dict(hist, bogus=1.0))]})
    assert r.status_code == 422 and "bogus" in json.dumps(r.json())


def test_cell_in_store_uses_the_history_tier(client, store):
    models = api.get_models()
    lat, lon = fake.STATIC[0]
    near = detection(1, latitude=lat, longitude=lon)
    far = detection(2, latitude=-30.0, longitude=20.0)
    got = client.post("/predict", json={"detections": [near, far]}).json()
    a, b = got["predictions"]
    assert a["tier"] == "history" and "live store" in a["tier_reason"]
    assert b["tier"] == "instant" and got["tier_counts"] == {"instant": 1, "history": 1}
    # the history score is the saved model on the request readings and the stored history of that cell
    from src.label import add_cell
    cid = add_cell(pd.DataFrame({"latitude": [lat], "longitude": [lon]}))["cell_id"].iloc[0]
    row = store[store["cell_id"] == cid].sort_values("timestamp").iloc[-1]
    x = pd.DataFrame([near])
    x["bright_diff"] = x["bright_ti4"] - x["bright_ti5"]
    x["pixel_area"] = x["scan"] * x["track"]
    x["frp_per_area"] = x["frp"] / x["pixel_area"]
    x["confidence_ord"] = x["confidence"].map(CONF)
    x["is_night"] = (x["daynight"] == "N").astype(int)
    for f in models.history_only:
        x[f] = float(row[f])
    hb = joblib.load(MODEL_DIR / serve.HISTORY_FILE)
    assert abs(a["score"] - hb["model"].predict_proba(x[hb["features"]])[:, 1][0]) < 1e-6


def test_request_size_limit(client):
    ok = client.post("/predict", json={"detections": [detection(0)] * api.MAX_DETECTIONS})
    assert ok.status_code == 200 and len(ok.json()["predictions"]) == 5000
    assert len({p["score"] for p in ok.json()["predictions"]}) == 1
    too = client.post("/predict", json={"detections": [detection(0)] * (api.MAX_DETECTIONS + 1)})
    assert too.status_code == 422 and "5000" in json.dumps(too.json())
    assert client.post("/predict", json={"detections": []}).status_code == 422


@pytest.mark.parametrize("change", [
    {"latitude": 95.0}, {"longitude": -200.0}, {"acq_time": 1275}, {"acq_time": 2400}, {"acq_date": "yesterday"},
    {"bright_ti4": -5.0}, {"bright_ti5": 0.0}, {"frp": -1.0}, {"scan": 0.0}, {"track": 99.0},
    {"confidence": "maybe"}, {"confidence": "150"}, {"daynight": "X"}, {"bright_ti4": "hot"}, {"unknown_field": 1},
    {"history": {"distinct_days_prior_7": float("inf")}},
])
def test_input_validation_errors(client, change):
    r = client.post("/predict", content=json.dumps({"detections": [detection(0, **change)]}, allow_nan=True),
                    headers={"content-type": "application/json"})
    assert r.status_code == 422, (change, r.status_code)
    assert r.json()["detail"]


def test_missing_fields_and_bad_bodies(client):
    d = detection(0)
    d.pop("frp")
    assert client.post("/predict", json={"detections": [d]}).status_code == 422
    assert client.post("/predict", json={}).status_code == 422
    assert client.post("/predict", json=[detection(0)]).status_code == 422
    assert client.post("/predict", content="not json", headers={"content-type": "application/json"}).status_code == 422
    assert client.get("/predict").status_code == 405


def test_health_and_model(client, store):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["rows"] == len(store)
    assert pd.Timestamp(h["last_ingest"]) == store["ingest_time"].max()
    m = client.get("/model").json()
    models = api.get_models()
    assert m["files"]["instant"]["file"] == "best_model_headline.joblib" and m["files"]["history"]["file"] == "best_model_history.joblib"
    assert m["instant"]["threshold"] == models.instant_threshold and m["history"]["threshold"] == models.history_threshold
    assert m["history"]["thresholds"] == models.hist["thresholds"]
    assert m["instant"]["band_default"]["target"] == 0.96 and m["instant"]["band_strict"]["target"] == 0.98
    assert m["metrics"]["history"]["precision"] is not None and m["metrics"]["instant"]["static_pr_auc"] > 0
    assert "unvalidated" in m["note"] and "not retrained" in m["note"]


def test_health_without_a_store(client):
    h = client.get("/health").json()
    assert h == {"status": "ok", "last_ingest": None, "rows": 0}
    r = client.get("/live/latest")
    assert r.status_code == 503 and "ingest" in r.json()["detail"]


def test_live_latest_filters_and_cap(client, store):
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)

    def want(hours=None, sensor=None, box=None):
        s = store
        if hours:
            s = s[s["timestamp"] >= now - pd.Timedelta(hours=hours)]
        if sensor:
            s = s[s["sensor"] == sensor]
        if box:
            s = s[s["latitude"].between(box[0], box[2]) & s["longitude"].between(box[1], box[3])]
        return len(s)
    r = client.get("/live/latest", params={"hours": 2000}).json()
    assert r["count"] == len(store) == r["matching"] and not r["truncated"] and "unvalidated" in r["note"]
    row = r["rows"][0]
    assert {"sensor", "latitude", "longitude", "timestamp", "instant_default", "history_strict", "pc_history",
            "label_instant"} <= set(row)
    ts = [x["timestamp"] for x in r["rows"]]
    assert ts == sorted(ts, reverse=True)                                       # newest first
    assert client.get("/live/latest", params={"hours": 36}).json()["matching"] == want(hours=36)
    n20 = client.get("/live/latest", params={"hours": 2000, "sensor": "NOAA-20"}).json()
    assert n20["matching"] == want(sensor="NOAA-20") and {x["sensor"] for x in n20["rows"]} == {"NOAA-20"}
    box = (28.0, 69.0, 30.0, 72.0)
    b = client.get("/live/latest", params={"hours": 2000, "min_lat": box[0], "min_lon": box[1], "max_lat": box[2],
                                           "max_lon": box[3]}).json()
    assert b["matching"] == want(box=box) > 0
    assert all(box[0] <= x["latitude"] <= box[2] and box[1] <= x["longitude"] <= box[3] for x in b["rows"])
    small = client.get("/live/latest", params={"hours": 2000, "limit": 7}).json()
    assert small["count"] == 7 and small["truncated"] and small["matching"] == len(store)
    assert api.MAX_LIVE_ROWS == 20000
    for params in ({"limit": 20001}, {"hours": 0}, {"hours": 24 * 90 + 1}, {"sensor": "SNPP"}, {"min_lat": 1.0},
                   {"min_lat": 30, "max_lat": 20, "min_lon": 0, "max_lon": 10}, {"min_lat": 95, "max_lat": 96,
                                                                                  "min_lon": 0, "max_lon": 1}):
        assert client.get("/live/latest", params=params).status_code == 422, params


def test_key_never_appears(client, store, monkeypatch):
    monkeypatch.setenv("FIRMS_MAP_KEY", KEY)
    texts = [client.get("/health").text, client.get("/model").text, client.get("/live/latest").text,
             client.get("/live/latest", params={"sensor": "bad"}).text,
             client.post("/predict", json={"detections": [detection(0)]}).text,
             client.post("/predict", json={"detections": [detection(0, latitude=999)]}).text,
             client.get("/openapi.json").text]
    assert all(KEY not in t for t in texts)
    source = Path(api.__file__).read_text(encoding="utf-8")
    assert "FIRMS_MAP_KEY" not in source and "load_key" not in source and "requests" not in source

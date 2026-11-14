"""Prediction API for the fire hotspot classifier.

Run from the project root:  uvicorn api.main:app

Endpoints: GET /health, GET /model, POST /predict, GET /live/latest. The API reads the saved models and
the live store (data/live/store.parquet, written by python -m src.ingest). It never fetches data and
never reads the FIRMS map key. Live results are unvalidated and the models are not retrained from live
data, because live data has no type flag.
"""
import json
import math
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import ingest, live, serve                     # noqa: E402
from src.config import REPORT_DIR                       # noqa: E402
from src.data import clean                              # noqa: E402
from src.features import add_features                   # noqa: E402
from src.label import add_cell                          # noqa: E402

MAX_DETECTIONS = 5000
MAX_LIVE_ROWS = 20000
SENSORS = list(live.LIVE_SOURCES)
NOTE = ("Live results are unvalidated: the models were trained on Suomi NPP archive data and checked on NOAA-20 "
        "and NOAA-21 archive data, and live data has no type flag, so they are not retrained from it.")
CONFIDENCE = {"l", "n", "h", "low", "nominal", "high"}

app = FastAPI(title="Fire hotspot classifier", version="1.0")


@app.exception_handler(RequestValidationError)
async def invalid_request(request, exc):
    """422 with the place and the reason of every problem. The offending value is not echoed back."""
    errors = [{"loc": [str(x) for x in e["loc"]], "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})

_models = None
_cache = {}


def get_models():
    """The saved bundles, loaded once."""
    global _models
    if _models is None:
        _models = serve.Models()
    return _models


def _store_file():
    return ingest.store_path()


def _cached(name, columns):
    """The store (some columns), reread when the file changes."""
    path = _store_file()
    if not path.exists():
        return None
    stamp = path.stat().st_mtime_ns
    hit = _cache.get(name)
    if hit is None or hit[0] != stamp or hit[2] != path:
        _cache[name] = (stamp, pd.read_parquet(path, columns=columns), path)
    return _cache[name][1]


# ------------------------------------------------------------------ request models
class Detection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    acq_date: date
    acq_time: int = Field(ge=0, le=2359, description="UTC time as HHMM")
    bright_ti4: float = Field(gt=0, le=1000)
    bright_ti5: float = Field(gt=0, le=1000)
    frp: float = Field(ge=0, le=100000)
    scan: float = Field(gt=0, le=10)
    track: float = Field(gt=0, le=10)
    confidence: str
    daynight: str
    history: Optional[dict[str, float]] = Field(default=None, description="history features of the place, all of them")

    @field_validator("confidence")
    @classmethod
    def _confidence(cls, v):
        v = v.strip().lower()
        if v in CONFIDENCE:
            return v
        try:
            if 0 <= float(v) <= 100:
                return v
        except ValueError:
            pass
        raise ValueError("confidence must be l, n, h (or low, nominal, high) or a number from 0 to 100")

    @field_validator("daynight")
    @classmethod
    def _daynight(cls, v):
        v = v.strip().upper()
        if v not in ("D", "N"):
            raise ValueError("daynight must be D or N")
        return v

    @model_validator(mode="after")
    def _time(self):
        if self.acq_time % 100 > 59:
            raise ValueError("acq_time must be HHMM with minutes below 60")
        if self.history is not None and not all(math.isfinite(x) for x in self.history.values()):
            raise ValueError("history values must be finite numbers")
        return self


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    detections: list[Detection] = Field(min_length=1, max_length=MAX_DETECTIONS)


class Prediction(BaseModel):
    index: int
    label: str
    probability: float
    score: float
    band_default: str
    band_strict: str
    tier: str
    tier_reason: str


class PredictResponse(BaseModel):
    predictions: list[Prediction]
    tier_counts: dict[str, int]
    note: str = NOTE


# ------------------------------------------------------------------ prediction
def instant_frame(dets):
    """Instant features of the detections with the training code (clean, add_features, add_cell), in request order."""
    rows = [{"latitude": d.latitude, "longitude": d.longitude, "acq_date": d.acq_date.isoformat(),
             "acq_time": d.acq_time, "bright_ti4": d.bright_ti4, "bright_ti5": d.bright_ti5, "frp": d.frp,
             "scan": d.scan, "track": d.track, "confidence": d.confidence, "daynight": d.daynight} for d in dets]
    df = pd.DataFrame(rows)
    df["_pos"] = np.arange(len(df))
    # clean() drops rows that repeat a key. Latitude only feeds that key and a range check there, so it is
    # replaced by a distinct value for the call and restored after it.
    work = df.assign(latitude=np.linspace(-89.0, 89.0, len(df)))
    out = clean(work).sort_values("_pos").reset_index(drop=True)
    if len(out) != len(df):
        raise HTTPException(status_code=422, detail="some detections could not be read")
    out["latitude"] = df["latitude"].to_numpy()[out["_pos"].to_numpy()]
    return add_cell(add_features(out)).reset_index(drop=True)


def store_history(cells):
    """Latest history features of each cell found in the store: a frame indexed by cell_id, or None."""
    models = get_models()
    cols = ["cell_id", "timestamp"] + models.history_only
    store = _cached("history", cols)
    if store is None or store.empty:
        return None
    latest = store.sort_values("timestamp").drop_duplicates("cell_id", keep="last").set_index("cell_id")
    return latest[latest.index.isin(set(cells))][models.history_only]


def predict(dets):
    models = get_models()
    X = instant_frame(dets)
    n = len(X)
    tier = np.array(["instant"] * n, dtype=object)
    reason = np.array(["no history supplied and the cell is not in the live store, instant tier"] * n, dtype=object)
    hist = pd.DataFrame(np.nan, index=range(n), columns=models.history_only)
    for i, d in enumerate(dets):
        if d.history is not None:
            missing = [f for f in models.history_only if f not in d.history]
            extra = [k for k in d.history if k not in models.history_only]
            if missing or extra:
                raise HTTPException(status_code=422, detail={"detection": i, "missing_history_fields": missing,
                                                             "unknown_history_fields": extra})
            hist.loc[i] = [d.history[f] for f in models.history_only]
            tier[i], reason[i] = "history", "history fields supplied"
    need = [i for i in range(n) if tier[i] == "instant"]
    if need:
        found = store_history(X.loc[need, "cell_id"])
        if found is not None and len(found):
            for i in need:
                cid = X.loc[i, "cell_id"]
                if cid in found.index:
                    hist.loc[i] = found.loc[cid].to_numpy(float)
                    tier[i], reason[i] = "history", "cell found in the live store, history of its latest stored detection"
    full = pd.concat([X, hist], axis=1)
    inst = serve.score_tier(models, "instant", full)
    use_hist = tier == "history"
    out = {k: np.array(v, dtype=object) for k, v in inst.items()}
    if use_hist.any():
        h = serve.score_tier(models, "history", full[use_hist])
        for k, v in h.items():
            out[k][use_hist] = v
    preds = [Prediction(index=i, label=str(out["label"][i]), probability=float(out["p_cal"][i]),
                        score=float(out["p"][i]), band_default=str(out["band_default"][i]),
                        band_strict=str(out["band_strict"][i]), tier=str(tier[i]), tier_reason=str(reason[i]))
             for i in range(n)]
    counts = {t: int((tier == t).sum()) for t in ("instant", "history")}
    return PredictResponse(predictions=preds, tier_counts=counts)


# ------------------------------------------------------------------ endpoints
@app.get("/health")
def health():
    ts = _cached("health", ["ingest_time"])
    if ts is None or ts.empty:
        return {"status": "ok", "last_ingest": None, "rows": 0}
    return {"status": "ok", "last_ingest": pd.Timestamp(ts["ingest_time"].max()).isoformat(), "rows": int(len(ts))}


def _band(b):
    if not b:
        return None
    return {k: b.get(k) for k in ("definition", "target", "reachable", "t_low", "t_high", "oof_coverage", "oof_accuracy")}


def _report_metrics():
    """Headline numbers from the existing reports, unchanged."""
    out = {}
    path = REPORT_DIR / "daynight_stratified.csv"
    if path.exists():
        d = pd.read_csv(path)
        r = d[(d["split"] == "grouped seed 42") & (d["tier"] == "instant (XGBoost, shipped)") & (d["stratum"] == "all")]
        if len(r):
            r = r.iloc[0]
            out["instant"] = {"split": "grouped seed 42 held out cells", "n": int(r["n"]),
                              "static_pr_auc": float(r["pr_auc"]), "precision": float(r["precision"]),
                              "recall": float(r["recall"]), "f1": float(r["f1"]),
                              "majority_pr_auc": float(r["majority_pr_auc"])}
    path = REPORT_DIR / "history_final_metrics.json"
    if path.exists():
        h = json.loads(path.read_text())
        pt = h.get("operating_points", {}).get(get_models().point, {})
        out["history"] = {"split": f"grouped seed {h.get('split_seed')} held out cells", "n": h.get("test_rows"),
                          "operating_point": get_models().point, "precision": pt.get("precision"),
                          "recall": pt.get("recall"), "majority_pr_auc": h.get("test_static_share")}
    return out


@app.get("/model")
def model_info():
    m = get_models()
    from src.config import MODEL_DIR
    files = {}
    for tier, name in (("instant", serve.INSTANT_FILE), ("history", serve.HISTORY_FILE)):
        p = MODEL_DIR / name
        files[tier] = {"file": name, "size_bytes": p.stat().st_size if p.exists() else None}
    return {"files": files,
            "instant": {"name": m.inst.get("name"), "threshold": m.instant_threshold,
                        "features": m.instant_features,
                        "band_default": _band(m.bands["instant"]["default"]),
                        "band_strict": _band(m.bands["instant"]["strict"])},
            "history": {"name": m.hist.get("name"), "operating_point": m.point, "threshold": m.history_threshold,
                        "thresholds": m.hist["thresholds"], "features": m.history_features,
                        "band_default": _band(m.bands["history"]["default"]),
                        "band_strict": _band(m.bands["history"]["strict"])},
            "metrics": _report_metrics(), "note": NOTE}


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(req: PredictRequest):
    return predict(req.detections)


LIVE_COLS = ["sensor", "region", "latitude", "longitude", "acq_date", "acq_time", "timestamp", "cell_id",
             "p_instant", "pc_instant", "label_instant", "instant_default", "instant_strict",
             "p_history", "pc_history", "label_history", "history_default", "history_strict"]


@app.get("/live/latest")
def live_latest(hours: int = Query(24, ge=1, le=24 * 90), sensor: Optional[str] = None,
                min_lat: Optional[float] = Query(None, ge=-90, le=90), max_lat: Optional[float] = Query(None, ge=-90, le=90),
                min_lon: Optional[float] = Query(None, ge=-180, le=180),
                max_lon: Optional[float] = Query(None, ge=-180, le=180),
                limit: int = Query(MAX_LIVE_ROWS, ge=1, le=MAX_LIVE_ROWS)):
    if sensor is not None and sensor not in SENSORS:
        raise HTTPException(status_code=422, detail=f"sensor must be one of {SENSORS}")
    box = [min_lat, min_lon, max_lat, max_lon]
    if any(v is not None for v in box) and not all(v is not None for v in box):
        raise HTTPException(status_code=422, detail="give all of min_lat, min_lon, max_lat and max_lon, or none")
    if all(v is not None for v in box) and (min_lat > max_lat or min_lon > max_lon):
        raise HTTPException(status_code=422, detail="the box has its minimum above its maximum")
    store = _cached("live", LIVE_COLS)
    if store is None:
        raise HTTPException(status_code=503, detail="no live store yet, run python -m src.ingest --once")
    since = pd.Timestamp.now(tz="UTC").tz_localize(None) - timedelta(hours=hours)
    sel = store["timestamp"] >= since
    if sensor:
        sel &= store["sensor"] == sensor
    if all(v is not None for v in box):
        sel &= store["latitude"].between(min_lat, max_lat) & store["longitude"].between(min_lon, max_lon)
    rows = store[sel].sort_values("timestamp", ascending=False)
    total = len(rows)
    rows = rows.head(limit).copy()
    rows["acq_date"] = rows["acq_date"].dt.strftime("%Y-%m-%d")
    rows["timestamp"] = rows["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    for c in ("p_instant", "pc_instant", "p_history", "pc_history"):
        rows[c] = rows[c].astype(float).round(5)
    return {"count": len(rows), "matching": total, "truncated": total > len(rows), "note": NOTE,
            "rows": json.loads(rows.to_json(orient="records"))}

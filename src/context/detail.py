"""Detail payload of one clicked detection for the Earth panel.

The shape follows section 14 of docs/PROJECT_CONTEXT.md. This module fills the verdict, the detection facts,
the place and the spread. Land, weather, exposure and response come from outside sources (src/context/land.py, weather.py, people.py,
response.py): the app fetches them when a detection is clicked and passes them in as extras; a section that was not
fetched stays None and the panel shows it as unavailable.

Nothing here calls the network. The spread of a live detection is read from the store through
ingest.read_recent with a time window and a latitude and longitude box, never the whole store. The spread of
an archive detection is computed from the archive frame the app already holds. Every section carries its
source and the time it was made, and a section that cannot be made is None instead of an error.
"""
import math

import numpy as np
import pandas as pd

from .. import geo, ingest, live, uncertain

SPREAD_HOURS = 72
SPREAD_KM = 2.0
CELL_DAYS = 90
KM_PER_DEG = 111.32
NOMINAL_PIXEL_KM2 = 0.375 * 0.375
DISCLAIMER = ("Satellite detection, not a confirmed fire or an official warning. "
              "In an emergency contact local emergency services.")
SPREAD_NOTE = ("This is the extent of satellite detections, not a fire perimeter. A given place is seen only a "
               "few times a day, so counts depend on the satellite passes.")
CONFIDENCE = {0: "l", 1: "n", 2: "h"}
LABELS = {
    "bright_ti4": "brightness I-4", "bright_ti5": "brightness I-5", "bright_diff": "I-4 minus I-5",
    "frp": "fire radiative power", "scan": "pixel scan size", "track": "pixel track size",
    "pixel_area": "pixel area", "frp_per_area": "FRP per pixel area", "confidence_ord": "NASA confidence",
    "is_night": "night detection", "distinct_days_prior_7": "days seen here in the previous 7 days",
    "distinct_days_prior_30": "days seen here in the previous 30 days",
    "distinct_days_prior_60": "days seen here in the previous 60 days",
    "distinct_days_prior_90": "days seen here in the previous 90 days",
    "nbr_distinct_days_prior_7": "days seen nearby in the previous 7 days",
    "nbr_distinct_days_prior_30": "days seen nearby in the previous 30 days",
    "nbr_distinct_days_prior_60": "days seen nearby in the previous 60 days",
    "detections_prior_7": "detections here in the previous 7 days", "days_since_first_seen": "days since first seen",
    "days_into_data": "days of data before this one", "days_since_last_seen": "days since last seen",
    "persistence_30": "persistence over 30 days", "ti4_mean_prior_30": "mean I-4 over 30 days",
    "ti4_std_prior_30": "spread of I-4 over 30 days", "frp_mean_prior_30": "mean FRP over 30 days",
    "frp_std_prior_30": "spread of FRP over 30 days", "has_history": "has earlier detections",
    "bright_dev": "brightness against its own past", "frp_dev": "FRP against its own past",
}
COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def now_text():
    return pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def _num(v, nd=None):
    """A plain float (rounded) or None for missing values."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return round(f, nd) if nd is not None else f


def _text(v):
    return "" if v is None or (isinstance(v, float) and math.isnan(v)) or pd.isna(v) else str(v)


# ------------------------------------------------------------------ geometry
def distance_km(lat1, lon1, lat2, lon2):
    """Great circle distance, works on scalars and arrays."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dlam = p2 - p1, np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def bearing_deg(lat1, lon1, lat2, lon2):
    """Initial bearing from point 1 to point 2, 0 is north, clockwise."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def compass(deg):
    return None if deg is None else COMPASS[int(((deg % 360) + 22.5) // 45) % 8]


def box(lat, lon, km):
    """South, north, west, east of a square of about km around a point."""
    dlat = km / KM_PER_DEG
    dlon = km / (KM_PER_DEG * max(math.cos(math.radians(lat)), 0.01))
    return lat - dlat, lat + dlat, lon - dlon, lon + dlon


# ------------------------------------------------------------------ id
def detection_id(sensor, lat, lon, acq_date, acq_time):
    """Stable id of a detection: sensor, latitude, longitude, acquisition date and time."""
    return f"{sensor}|{float(lat):.5f}|{float(lon):.5f}|{pd.Timestamp(acq_date):%Y-%m-%d}|{int(acq_time):04d}"


def acq_time_of(row):
    """acq_time as HHMM from the row, or from its timestamp."""
    if row.get("acq_time") is not None and not pd.isna(row.get("acq_time")):
        return int(row["acq_time"])
    t = pd.Timestamp(row["timestamp"])
    return t.hour * 100 + t.minute


# ------------------------------------------------------------------ verdict
def top_features(bundle, row, k=3):
    """The features that pushed this call most, from the model's own per feature contributions (log odds,
    positive toward static). None when the row lacks a feature of the model."""
    feats = list(bundle["features"])
    if any(f not in row for f in feats):
        return None
    try:
        import xgboost as xgb
        X = pd.DataFrame([[_num(row[f]) if _num(row[f]) is not None else np.nan for f in feats]], columns=feats)
        contrib = bundle["model"].get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[0][:-1]
    except Exception:
        return None
    order = np.argsort(-np.abs(contrib))[:k]
    return [{"feature": feats[j], "label": LABELS.get(feats[j], feats[j]), "value": _num(X.iloc[0, j], 3),
             "push": "toward static" if contrib[j] > 0 else "toward fire", "weight": round(float(contrib[j]), 3)}
            for j in order]


def verdict(row, tier, bundle, bands, band, threshold, region=None):
    """Binary label at the operating threshold, calibrated probability, the band call and the region note."""
    p = _num(row.get("p_static"))
    if p is None:
        return None
    cal = None
    if bands and bands.get("calibrator"):
        cal = _num(uncertain.apply_calibrator(bands["calibrator"], np.array([p]))[0], 3)
    call = "no band"
    if band and band.get("reachable", True) and cal is not None:
        call = {uncertain.FIRE: "fire", uncertain.STATIC: "static", uncertain.UNCERTAIN: "uncertain"}[
            int(uncertain.decide(np.array([cal]), band)[0])]
    trained = region is None or live.is_trained(region)
    pred = _num(row.get("pred_static"))                 # the label the map shows, when the row carries it
    if pred is not None:
        label = "static" if pred == 1 else "fire"
    else:
        label = "static" if threshold is not None and p >= threshold else "fire"
    return {"label": label, "shown": call if call != "no band" else label, "tier": tier, "p": cal,
            "p_raw": round(p, 3), "threshold": None if threshold is None else round(float(threshold), 3),
            "band": call, "trained": trained,
            "note": "" if trained else live.UNTRAINED_NOTE, "top_features": top_features(bundle, row),
            "source": "saved models of this project", "fetched_at": now_text()}


# ------------------------------------------------------------------ facts and place
def facts(row, sensor):
    t = pd.Timestamp(row["timestamp"])
    lon = float(row["longitude"])
    solar = t + pd.Timedelta(hours=lon / 15.0)              # mean solar time, no time zone database
    scan, track = _num(row.get("scan"), 3), _num(row.get("track"), 3)
    night = row.get("is_night")
    dn = _text(row.get("daynight")) or ("N" if _num(night) == 1 else "D" if _num(night) == 0 else "")
    ordinal = _num(row.get("confidence_ord"))
    conf = _text(row.get("confidence")) or (CONFIDENCE.get(int(ordinal), "") if ordinal is not None else "")
    return {"time_utc": f"{t:%Y-%m-%d %H:%M}", "local_solar_time": f"{solar:%Y-%m-%d %H:%M}",
            "satellite": sensor, "ti4": _num(row.get("bright_ti4"), 1), "ti5": _num(row.get("bright_ti5"), 1),
            "frp": _num(row.get("frp"), 1), "scan": scan, "track": track,
            "pixel_km2": round(scan * track, 3) if scan and track else None, "nominal_pixel": not (scan and track),
            "daynight": dn, "confidence": conf, "latitude": round(float(row["latitude"]), 5),
            "longitude": round(lon, 5), "source": "NASA FIRMS", "fetched_at": now_text()}


def place(row):
    cc = _text(row.get("cc"))
    return {"country": geo.country_name(cc) if cc else "", "country_code": cc,
            "province": _text(row.get("admin1")), "city": _text(row.get("city")), "elevation_m": None,
            "source": "Natural Earth boundaries and GeoNames places (offline)", "fetched_at": now_text()}


# ------------------------------------------------------------------ spread
def spread_from_frame(df, lat, lon, t, hours=SPREAD_HOURS, km=SPREAD_KM, cell_days=None, source="", now=None):
    """Detections within km of a point in the `hours` up to time t, from a frame with latitude, longitude and
    timestamp (pixel_area optional). Count per day, first and last seen, approximate area, centroid movement
    and a simple trend."""
    t = pd.Timestamp(t)
    s, n, w, e = box(lat, lon, km)
    if df is None or df.empty:
        near = pd.DataFrame(columns=["latitude", "longitude", "timestamp"])
    else:
        m = (df["latitude"].to_numpy() >= s) & (df["latitude"].to_numpy() <= n) & \
            (df["longitude"].to_numpy() >= w) & (df["longitude"].to_numpy() <= e)
        near = df[m]
        ts = pd.to_datetime(near["timestamp"])
        near = near[(ts > t - pd.Timedelta(hours=hours)) & (ts <= t)]
        near = near[distance_km(lat, lon, near["latitude"].to_numpy(float), near["longitude"].to_numpy(float)) <= km]
    out = {"hours": hours, "km": km, "n": int(len(near)), "per_day": [], "first_seen": "", "last_seen": "",
           "area_km2": None, "pixels": 0, "move_km": None, "move_dir": None, "trend": "", "cell_days_90": cell_days,
           "note": SPREAD_NOTE, "window_end": f"{t:%Y-%m-%d %H:%M}", "source": source,
           "fetched_at": now or now_text()}
    if near.empty:
        out["trend"] = "no detections in the window"
        return out
    ts = pd.to_datetime(near["timestamp"])
    days = ts.dt.strftime("%Y-%m-%d")
    out["per_day"] = [{"date": d, "n": int(c)} for d, c in days.value_counts().sort_index().items()]
    out["first_seen"], out["last_seen"] = f"{ts.min():%Y-%m-%d %H:%M}", f"{ts.max():%Y-%m-%d %H:%M}"
    pix = near.assign(_k=near["latitude"].round(4).astype(str) + "," + near["longitude"].round(4).astype(str))
    pix = pix.drop_duplicates("_k")
    out["pixels"] = int(len(pix))
    area = pix["pixel_area"].astype(float) if "pixel_area" in pix else pd.Series(NOMINAL_PIXEL_KM2, index=pix.index)
    out["area_km2"] = round(float(area.fillna(NOMINAL_PIXEL_KM2).sum()), 2)
    last = near[ts > t - pd.Timedelta(hours=24)]
    prev = near[(ts > t - pd.Timedelta(hours=48)) & (ts <= t - pd.Timedelta(hours=24))]
    early = near[ts <= ts.min() + pd.Timedelta(hours=24)]
    if len(prev) == 0 and len(near) == len(last):
        out["trend"] = "new in the last 24 hours"
    elif len(last) >= 1.5 * max(len(prev), 1):
        out["trend"] = "growing"
    elif len(last) <= 0.67 * len(prev):
        out["trend"] = "shrinking"
    else:
        out["trend"] = "steady"
    if len(last) and len(early) and ts.max() - ts.min() >= pd.Timedelta(hours=12):
        a = (early["latitude"].mean(), early["longitude"].mean())
        b = (last["latitude"].mean(), last["longitude"].mean())
        out["move_km"] = round(float(distance_km(a[0], a[1], b[0], b[1])), 2)
        out["move_dir"] = compass(bearing_deg(a[0], a[1], b[0], b[1])) if out["move_km"] > 0.05 else None
    return out


def cell_days_from_frame(df, cell_id, t, days=CELL_DAYS):
    if df is None or cell_id is None or "cell_id" not in df:
        return None
    t = pd.Timestamp(t)
    sub = df[df["cell_id"] == cell_id]
    ts = pd.to_datetime(sub["timestamp"])
    sub = sub[(ts > t - pd.Timedelta(days=days)) & (ts <= t)]
    return int(pd.to_datetime(sub["acq_date"]).dt.normalize().nunique())


SPREAD_COLS = ["sensor", "latitude", "longitude", "timestamp", "acq_date", "cell_id", "pixel_area"]


def spread_from_store(lat, lon, t, cell_id=None, path=None, hours=SPREAD_HOURS, km=SPREAD_KM):
    """Spread of a live detection read from the store with a time window and a latitude and longitude box.
    Only the rows inside the box and the window are kept in memory. None when there is no store."""
    t = pd.Timestamp(t)
    s, n, w, e = box(lat, lon, km)
    where = [("latitude", ">=", s), ("latitude", "<=", n), ("longitude", ">=", w), ("longitude", "<=", e),
             ("timestamp", "<=", t)]
    from concurrent.futures import ThreadPoolExecutor

    def read_box():
        return ingest.read_recent(path, since=t - pd.Timedelta(hours=hours), columns=SPREAD_COLS, where=where)

    def read_cell():
        return ingest.read_recent(path, since=t - pd.Timedelta(days=CELL_DAYS), columns=["acq_date"],
                                  where=[("cell_id", "==", cell_id), ("timestamp", "<=", t)])
    try:
        with ThreadPoolExecutor(max_workers=2) as ex:        # the two reads side by side
            f_box = ex.submit(read_box)
            f_cell = ex.submit(read_cell) if cell_id is not None else None
            near = f_box.result()
            cells = f_cell.result() if f_cell is not None else None
        if near is None:
            return None
        cdays = int(pd.to_datetime(cells["acq_date"]).dt.normalize().nunique()) if cells is not None else None
    except Exception:
        return None
    return spread_from_frame(near, lat, lon, t, hours, km, cdays, source="own live store")


def store_row(sensor, lat, lon, acq_date, acq_time, path=None):
    """The full stored row of one detection (every feature and score), read by its id, or None."""
    try:
        t0 = pd.Timestamp(acq_date).normalize()
        rows = ingest.read_recent(path, since=t0, where=[
            ("sensor", "==", sensor), ("latitude", "==", float(lat)), ("longitude", "==", float(lon)),
            ("acq_date", "==", t0), ("acq_time", "==", int(acq_time))])
    except Exception:
        return None
    if rows is None or rows.empty:
        return None
    return rows.iloc[0].to_dict()


# ------------------------------------------------------------------ assembly
def build(row, *, source, tier, bundle, bands, band, threshold, spread, extras=None):
    """The panel payload of one detection. row is a dict of the detection (an archive row or a full stored
    row). Sections that cannot be made are None."""
    sensor = _text(row.get("sensor")) or "S-NPP"
    region = _text(row.get("region")) or None

    def safe(fn, *a):
        try:
            return fn(*a)
        except Exception:
            return None

    notes = [DISCLAIMER]
    if source == "live":
        notes.append("Live results are unvalidated: the models were trained on Suomi NPP archive data.")
    else:
        notes.append("Archive detection (January to June 2025). The NASA label is NASA's own repeat based "
                     "judgment, not independent ground truth.")
    return {
        "id": detection_id(sensor, row["latitude"], row["longitude"], row.get("acq_date", row["timestamp"]),
                           acq_time_of(row)),
        "source": source,
        "verdict": safe(verdict, row, tier, bundle, bands, band, threshold, region),
        "facts": safe(facts, row, sensor),
        "place": safe(place, row),
        "spread": spread, **{k: (extras or {}).get(k) for k in ("land", "weather", "exposure", "response")},
        "nasa_label": (("static" if int(row["is_static"]) == 1 else "fire")
                       if row.get("is_static") is not None and not pd.isna(row.get("is_static")) else None),
        "notes": notes,
        "attribution": ["NASA FIRMS", "Esri imagery", "Natural Earth", "GeoNames"],
    }

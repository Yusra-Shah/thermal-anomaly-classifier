"""Large clusters of detections, a context filter and not a model.

A detection is in a large cluster when 5 or more detections (itself included) lie within 2 km of it, on 2 or more
different days, and its own FRP is at or above the 75th percentile of FRP over the window. The window is every
detection of the current selection, not only the drawn sample, so the neighbours are all counted.
"""
import numpy as np

RADIUS_KM = 2.0
MIN_DETECTIONS = 5
MIN_DAYS = 2
FRP_PERCENTILE = 75
MAX_WINDOW = 1_000_000          # more detections than this in the window: narrow the dates first
EARTH_KM = 6371.0088


def frp_threshold(frp):
    """The 75th percentile of FRP over the window."""
    v = np.asarray(frp, float)
    v = v[np.isfinite(v)]
    return float(np.percentile(v, FRP_PERCENTILE)) if len(v) else float("inf")


def flags(lat, lon, day, frp, query):
    """Large cluster flag of the detections `query` (row positions in the window arrays).

    lat, lon, day (an integer per calendar day) and frp describe the whole window. Returns (flags for the query rows,
    the FRP threshold). Raises ValueError when the window is too big."""
    from sklearn.neighbors import BallTree
    lat, lon, day, frp = (np.asarray(a) for a in (lat, lon, day, frp))
    query = np.asarray(query, int)
    if len(lat) > MAX_WINDOW:
        raise ValueError("too many detections in the window")
    thr = frp_threshold(frp)
    if len(query) == 0 or len(lat) == 0:
        return np.zeros(len(query), bool), thr
    xy = np.radians(np.column_stack([lat, lon]).astype(float))
    tree = BallTree(xy, metric="haversine")
    rows = tree.query_radius(xy[query], r=RADIUS_KM / EARTH_KM)
    out = np.zeros(len(query), bool)
    strong = np.asarray(frp, float)[query] >= thr
    for k, idx in enumerate(rows):
        if strong[k] and len(idx) >= MIN_DETECTIONS and len(np.unique(day[idx])) >= MIN_DAYS:
            out[k] = True
    return out, thr

"""The four outside sections of the detail panel for one clicked detection, fetched one after the other.

Each section is made by its own module (land, weather, people, response) and is a dict that either holds its values or
{"unavailable": reason}. A section that fails never stops the others, and nothing here prints or keeps an address.
"""
import numpy as np
import pandas as pd

from . import land, people, response, weather


def _f(row, name):
    v = row.get(name)
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) and v > 0 else None


def fetch_all(row, hooks=None):
    """land, weather, exposure and response for a detection row (a dict with latitude, longitude, timestamp, cc and,
    when known, scan and track). hooks replaces a module function in tests."""
    h = {"land": land.describe, "weather": weather.at, "people": people.describe, "response": response.describe}
    h.update(hooks or {})
    lat, lon = float(row["latitude"]), float(row["longitude"])
    cc = row.get("cc")
    cc = None if cc is None or (isinstance(cc, float) and np.isnan(cc)) or pd.isna(cc) else str(cc)
    out = {}
    for name, fn, args in (("land", "land", (lat, lon, _f(row, "scan"), _f(row, "track"))),
                           ("weather", "weather", (lat, lon, pd.Timestamp(row["timestamp"]))),
                           ("exposure", "people", (cc, lat, lon)),
                           ("response", "response", (cc, lat, lon))):
        try:
            out[name] = h[fn](*args)
        except Exception as e:
            out[name] = {"unavailable": f"could not be made ({type(e).__name__})"}
    return out

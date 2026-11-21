"""Weather at a detection from Open-Meteo (no key), fetched when a detection is clicked.

One request returns the last three days and the current hour of temperature, humidity, wind and rain for a point.
The reading used is the hour of the detection when it is within the last three days, else the current hour is not
used (the panel says so). Answers are cached on disk by coordinates rounded to 0.1 degree and the day, so a second
click nearby makes no request.
"""
import math
from datetime import timedelta

import pandas as pd

from . import diskcache

BASE = "https://api.open-meteo.com/v1/forecast"
SOURCE = "Open-Meteo (open-meteo.com, CC BY 4.0), model analysis and forecast data"
PAST_DAYS = 3
HOURLY = "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,wind_gusts_10m,precipitation"
COMPASS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]


def compass(deg):
    return COMPASS[int(((float(deg) % 360) + 11.25) // 22.5) % 16]


def _key(lat, lon, day):
    return f"{round(lat, 1):.1f}_{round(lon, 1):.1f}_{day}"


def fetch_series(lat, lon, day, get=None, cache_dir=None):
    """The hourly series (last three days up to the end of `day`) for a point, from the disk cache or one request.
    Returns (series dict, fetched_at, from_cache). Raises RuntimeError with a short reason."""
    key = _key(lat, lon, day)
    hit = diskcache.read("weather", key, ttl=3600, root=cache_dir)
    if hit:
        return hit["hourly"], hit["fetched_at"], True
    get = get or diskcache.http_get
    r = get(BASE, params={"latitude": round(lat, 2), "longitude": round(lon, 2), "hourly": HOURLY,
                          "past_days": PAST_DAYS, "forecast_days": 1, "wind_speed_unit": "kmh", "timezone": "GMT"})
    body = r.json()
    hourly = body["hourly"]
    stamp = diskcache.now_text()
    diskcache.write("weather", key, {"hourly": hourly, "fetched_at": stamp}, root=cache_dir)
    return hourly, stamp, False


def at(lat, lon, t, get=None, cache_dir=None, now=None):
    """Weather at the hour of the detection `t` (UTC). A dict for the panel, or {"unavailable": reason}."""
    now = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC").tz_localize(None)
    t = pd.Timestamp(t)
    if t < now - timedelta(days=PAST_DAYS) + timedelta(hours=1):
        return {"unavailable": f"the weather service keeps only the last {PAST_DAYS} days, this detection is older"}
    try:
        hourly, stamp, cached = fetch_series(lat, lon, now.strftime("%Y-%m-%d"), get, cache_dir)
        times = pd.to_datetime(hourly["time"])
    except Exception as e:
        return {"unavailable": f"the weather service did not answer ({type(e).__name__ if not isinstance(e, RuntimeError) else e})"}
    idx = int(abs(times - t.floor("h")).argmin())
    if abs(times[idx] - t.floor("h")) > pd.Timedelta(hours=1):
        return {"unavailable": "no weather hour matches the detection time"}

    def val(name, i=idx):
        v = hourly.get(name, [None] * len(times))[i]
        return None if v is None or (isinstance(v, float) and math.isnan(v)) else v

    def rain(hours):
        lo = max(0, idx - hours + 1)
        vals = [v for v in hourly["precipitation"][lo:idx + 1] if v is not None]
        return round(float(sum(vals)), 1) if vals else None

    wd = val("wind_direction_10m")
    return {"time": str(times[idx]), "temperature_c": val("temperature_2m"), "humidity_pct": val("relative_humidity_2m"),
            "wind_kmh": val("wind_speed_10m"), "gust_kmh": val("wind_gusts_10m"),
            "wind_from": compass(wd) if wd is not None else None,
            "wind_to": compass(wd + 180) if wd is not None else None,
            "rain_24h_mm": rain(24), "rain_72h_mm": rain(72), "source": SOURCE, "fetched_at": stamp,
            "cached": cached,
            "note": "Weather of the model grid cell around the point at the hour of the detection, not a station "
                    "reading. Wind is the direction it blows from; a fire is pushed the other way."}

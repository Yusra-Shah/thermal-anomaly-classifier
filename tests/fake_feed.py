"""Fake near real time feed for the ingest and API tests. No network."""
import numpy as np
import pandas as pd

from src import live

BOX = live.REGIONS["South Asia"]
STATIC = [(28.6 + i * 0.5, 70.0 + i * 0.7) for i in range(6)]


def today():
    return pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()


def master(n_days, end=None, seed=0, fires=20):
    """Raw FIRMS style rows: a few static places seen on most days plus random single fires, all in South Asia."""
    rng = np.random.default_rng(seed)
    end = end or today()
    west, south, east, north = BOX
    rows = []
    for k in range(n_days):
        day = end - pd.Timedelta(days=n_days - 1 - k)
        for lat, lon in STATIC:
            if rng.random() < 0.85:
                rows.append((lat + rng.normal(0, 0.0005), lon + rng.normal(0, 0.0005), day))
        for _ in range(fires):
            rows.append((rng.uniform(south + 1, north - 1), rng.uniform(west + 1, east - 1), day))
    df = pd.DataFrame(rows, columns=["latitude", "longitude", "acq_date"])
    n = len(df)
    df["bright_ti4"] = rng.normal(335, 14, n).round(1)
    df["scan"] = rng.uniform(0.4, 0.7, n).round(2)
    df["track"] = rng.uniform(0.4, 0.6, n).round(2)
    df["acq_time"] = rng.integers(0, 24, n) * 100 + rng.integers(0, 60, n)
    df["satellite"] = "N21"
    df["instrument"] = "VIIRS"
    df["confidence"] = rng.choice(["l", "n", "h"], n, p=[0.1, 0.85, 0.05])
    df["version"] = "2.0NRT"
    df["bright_ti5"] = rng.normal(300, 8, n).round(1)
    df["frp"] = rng.gamma(2.0, 2.0, n).round(2)
    df["daynight"] = np.where(df["acq_time"] // 100 < 6, "N", "D")
    df["acq_date"] = df["acq_date"].dt.strftime("%Y-%m-%d")
    return df.drop_duplicates(["latitude", "longitude", "acq_date", "acq_time"]).reset_index(drop=True)


def masters(n_days, end=None):
    """One frame per source, different rows for the two satellites."""
    return {src: master(n_days, end, seed=i) for i, src in enumerate(live.LIVE_SOURCES.values())}


def fetch_factory(by_source, log=None, extra_dups=False):
    """fetch(key, box, start, days, source) like src.live.fetch_chunk, answering from the fake frames."""
    def fetch(key, box, start, days, source):
        assert 1 <= days <= 5, days                                  # the documented day range
        if log is not None:
            log.append((pd.Timestamp(start).normalize(), days, source, tuple(box)))
        lo = pd.Timestamp(start).normalize()
        hi = lo + pd.Timedelta(days=days - 1)
        m = by_source[source]
        d = pd.to_datetime(m["acq_date"])
        west, south, east, north = box
        sel = (d >= lo) & (d <= hi) & m["longitude"].between(west, east) & m["latitude"].between(south, north)
        out = m[sel].copy()
        return pd.concat([out, out.head(3)], ignore_index=True) if extra_dups and len(out) else out
    return fetch

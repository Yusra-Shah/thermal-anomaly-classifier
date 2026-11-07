"""Generate a FAKE FIRMS-format CSV to test that the pipeline runs end to end.

The numbers produced by training on this file mean nothing. It exists only so the
code can be exercised without network access. Never report metrics from it.

    python tests/make_synthetic.py out.csv
"""
import sys

import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
days = pd.date_range("2026-10-26", periods=7)
rows = []


def pixel(lat, lon, day, kind):
    if kind == "static":
        ti4 = rng.normal(338, 9)
        frp = abs(rng.normal(6, 3))
    else:
        ti4 = rng.normal(342, 14)
        frp = abs(rng.normal(5, 5))
    ti4 = float(np.clip(ti4, 300, 367))
    return {
        "latitude": lat + rng.normal(0, 0.0015),
        "longitude": lon + rng.normal(0, 0.0015),
        "bright_ti4": ti4,
        "scan": rng.uniform(0.38, 0.6),
        "track": rng.uniform(0.36, 0.55),
        "acq_date": day.strftime("%Y-%m-%d"),
        "acq_time": int(rng.choice([430, 510, 740, 1840, 1920, 2050])),
        "satellite": rng.choice(["N", "N20"]),
        "instrument": "VIIRS",
        "confidence": rng.choice(["l", "n", "h"], p=[0.1, 0.7, 0.2]),
        "version": "2.0NRT",
        "bright_ti5": ti4 - abs(rng.normal(30, 8)),
        "frp": frp,
        "daynight": rng.choice(["D", "N"]),
    }


for _ in range(150):  # static sites
    lat, lon = rng.uniform(24, 32), rng.uniform(62, 74)
    for d in rng.choice(days, size=rng.integers(5, 8), replace=False):
        rows.append(pixel(lat, lon, pd.Timestamp(d), "static"))

for _ in range(1500):  # fires
    lat, lon = rng.uniform(24, 32), rng.uniform(62, 74)
    start = rng.integers(0, 7)
    for k in range(rng.integers(1, 3)):
        if start + k < 7:
            for _ in range(rng.integers(1, 4)):
                rows.append(pixel(lat + rng.normal(0, 0.01), lon + rng.normal(0, 0.01),
                                  days[start + k], "fire"))

pd.DataFrame(rows).to_csv(sys.argv[1], index=False)
print(f"wrote {len(rows)} fake rows to {sys.argv[1]}")

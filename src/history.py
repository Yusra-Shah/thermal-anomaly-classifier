"""History features: how often a place was flagged on days strictly before a detection.

Each raw file is processed separately. Only rows with acq_date earlier than the
detection's own date are used, so nothing from the same day or the future leaks in.
"""
import numpy as np
import pandas as pd

HISTORY_FEATURES = [
    "distinct_days_prior_7",
    "distinct_days_prior_30",
    "nbr_distinct_days_prior_7",
    "nbr_distinct_days_prior_30",
    "detections_prior_7",
    "days_since_first_seen",
    "days_into_data",
]

SINCE_CAP = 90
INTO_CAP = 60
_OFF = 100      # keeps day - window positive
_BASE = 1000    # day slot width inside a cell key
_STEP = 100_000  # x extent of a cell key, so a neighbour in y is +_STEP


def _cell_key(cy, cx):
    return (cy.astype(np.int64) + 10_000) * _STEP + (cx.astype(np.int64) + 20_000)


def _count_window(keys_sorted, cell, day, window):
    """Entries of keys_sorted for `cell` with day in [day - window, day - 1]."""
    base = cell * _BASE + _OFF
    hi = np.searchsorted(keys_sorted, base + day, side="left")
    lo = np.searchsorted(keys_sorted, base + day - window, side="left")
    return hi - lo


def _history_one(cy, cx, date):
    day = ((date - date.min()).dt.days).to_numpy(np.int64)
    cell = _cell_key(cy.to_numpy(), cx.to_numpy())
    out = {}

    pairs = np.unique(cell * _BASE + _OFF + day)          # unique (cell, day)
    p_cell, p_day = pairs // _BASE, pairs % _BASE - _OFF
    # Distinct days in the 3x3 block: spread every (cell, day) to its 9 neighbours
    shifts = np.array([dy * _STEP + dx for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
    nbr_pairs = np.unique((p_cell[:, None] + shifts[None, :]).ravel() * _BASE
                          + np.repeat(p_day + _OFF, 9))
    rows = np.sort(cell * _BASE + _OFF + day)             # every detection

    for w in (7, 30):
        out[f"distinct_days_prior_{w}"] = _count_window(pairs, cell, day, w)
        out[f"nbr_distinct_days_prior_{w}"] = _count_window(nbr_pairs, cell, day, w)
    out["detections_prior_7"] = _count_window(rows, cell, day, 7)

    first = pd.Series(day).groupby(cell).transform("min").to_numpy()
    out["days_since_first_seen"] = np.where(first < day, np.minimum(day - first, SINCE_CAP), 0)
    out["days_into_data"] = np.minimum(day, INTO_CAP)
    return pd.DataFrame(out, index=cy.index)


def add_history(df: pd.DataFrame) -> pd.DataFrame:
    """Needs cell_y, cell_x, acq_date and source_file. Returns df with HISTORY_FEATURES."""
    parts = [_history_one(g["cell_y"], g["cell_x"], g["acq_date"])
             for _, g in df.groupby("source_file", sort=False)]
    return df.join(pd.concat(parts).loc[df.index])


# History v2. Same rule: only detections on days strictly before the detection count.
HISTORY_V2_FEATURES = [
    "distinct_days_prior_60",
    "distinct_days_prior_90",
    "nbr_distinct_days_prior_60",
    "days_since_last_seen",
    "persistence_30",
    "ti4_mean_prior_30",
    "ti4_std_prior_30",
    "frp_mean_prior_30",
    "frp_std_prior_30",
    "has_history",
    "bright_dev",
    "frp_dev",
]


def _window_stats(keys_sorted, cs1, cs2, lo_key, hi_key, shift, v):
    """Mean and std of the values whose key lies in [lo_key, hi_key), via cumulative sums."""
    hi = np.searchsorted(keys_sorted, hi_key, side="left")
    lo = np.searchsorted(keys_sorted, lo_key, side="left")
    n = hi - lo
    safe = np.maximum(n, 1)
    mean_c = (cs1[hi] - cs1[lo]) / safe
    var = (cs2[hi] - cs2[lo]) / safe - mean_c ** 2
    has = n > 0
    mean = np.where(has, mean_c + shift, 0.0)
    # Variance below 1e-6 is cancellation noise from the cumulative sums, far below sensor resolution
    std = np.where(has & (var > 1e-6), np.sqrt(np.maximum(var, 0.0)), 0.0)
    dev = np.where(has, v - mean, 0.0)
    return mean, std, dev, has


def _history_v2_one(cy, cx, date, ti4, frp):
    day = ((date - date.min()).dt.days).to_numpy(np.int64)
    cell = _cell_key(cy.to_numpy(), cx.to_numpy())
    out = {}

    pairs = np.unique(cell * _BASE + _OFF + day)
    p_cell, p_day = pairs // _BASE, pairs % _BASE - _OFF
    shifts = np.array([dy * _STEP + dx for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
    nbr_pairs = np.unique((p_cell[:, None] + shifts[None, :]).ravel() * _BASE
                          + np.repeat(p_day + _OFF, 9))

    out["distinct_days_prior_60"] = _count_window(pairs, cell, day, 60)
    out["distinct_days_prior_90"] = _count_window(pairs, cell, day, 90)
    out["nbr_distinct_days_prior_60"] = _count_window(nbr_pairs, cell, day, 60)

    # Last earlier day seen in the same cell
    key = cell * _BASE + _OFF + day
    i = np.searchsorted(pairs, key, side="left") - 1
    ok = (i >= 0) & (pairs[np.maximum(i, 0)] // _BASE == cell)
    last = pairs[np.maximum(i, 0)] % _BASE - _OFF
    out["days_since_last_seen"] = np.where(ok, np.minimum(day - last, SINCE_CAP), SINCE_CAP)

    d30 = _count_window(pairs, cell, day, 30)
    out["persistence_30"] = d30 / np.maximum(1, np.minimum(30, np.minimum(day, INTO_CAP)))

    # Mean and std of the readings over the prior 30 days, from cumulative sums
    order = np.argsort(key, kind="stable")
    ks = key[order]
    lo_key = key - 30
    hi_key = key
    for name, v in (("ti4", ti4.to_numpy(np.float64)), ("frp", frp.to_numpy(np.float64))):
        shift = v.mean()
        vc = (v - shift)[order]
        cs1 = np.concatenate([[0.0], np.cumsum(vc)])
        cs2 = np.concatenate([[0.0], np.cumsum(vc ** 2)])
        mean, std, dev, has = _window_stats(ks, cs1, cs2, lo_key, hi_key, shift, v)
        out[f"{name}_mean_prior_30"], out[f"{name}_std_prior_30"] = mean, std
        out["bright_dev" if name == "ti4" else "frp_dev"] = dev
        if name == "ti4":
            out["has_history"] = has.astype(int)
    return pd.DataFrame(out, index=cy.index)


def add_history_v2(df: pd.DataFrame) -> pd.DataFrame:
    """Needs cell_y, cell_x, acq_date, source_file, bright_ti4 and frp. Adds HISTORY_V2_FEATURES.

    No readings are available before the first detection of a place, so the means,
    stds and deviations are 0 there and has_history is 0.
    """
    parts = [_history_v2_one(g["cell_y"], g["cell_x"], g["acq_date"], g["bright_ti4"], g["frp"])
             for _, g in df.groupby("source_file", sort=False)]
    return df.join(pd.concat(parts).loc[df.index][HISTORY_V2_FEATURES])

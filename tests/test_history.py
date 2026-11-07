"""Leakage tests for src.history. Run from the project root:  python -m tests.test_history"""
import numpy as np
import pandas as pd

from src.history import HISTORY_FEATURES, HISTORY_V2_FEATURES, add_history, add_history_v2

ALL = HISTORY_FEATURES + HISTORY_V2_FEATURES


def all_history(df):
    return add_history_v2(add_history(df))


def make_frame(rng, n=400):
    df = pd.DataFrame({
        "cell_y": rng.integers(0, 4, n),
        "cell_x": rng.integers(0, 4, n),
        "acq_date": pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 60, n), unit="D"),
        "bright_ti4": rng.normal(330, 10, n),
        "frp": rng.exponential(3, n),
        "source_file": "a.csv",
    })
    df["rid"] = np.arange(n)
    return df


def test_no_leakage():
    rng = np.random.default_rng(0)
    df = make_frame(rng)
    base = all_history(df).set_index("rid")[ALL]
    checked = 0
    for target in rng.choice(len(df), 40, replace=False):
        d = df.loc[target, "acq_date"]
        other = df[(df["acq_date"] >= d) & (df["rid"] != target)].copy()
        keep = df[(df["acq_date"] < d) | (df["rid"] == target)]
        # Move, re-date, change readings, drop and duplicate every other row on or after the date
        other["cell_y"] = rng.integers(0, 4, len(other))
        other["cell_x"] = rng.integers(0, 4, len(other))
        other["acq_date"] = d + pd.to_timedelta(rng.integers(0, 30, len(other)), unit="D")
        other["bright_ti4"] = rng.normal(400, 50, len(other))
        other["frp"] = rng.exponential(50, len(other))
        extra = other.sample(frac=0.5, random_state=1).assign(rid=lambda x: x["rid"] + 10_000)
        alt = pd.concat([keep, other.iloc[: len(other) // 2], extra], ignore_index=True)
        new = all_history(alt).set_index("rid")[ALL]
        a, b = new.loc[target].to_numpy(float), base.loc[target].to_numpy(float)
        assert np.allclose(a, b, rtol=1e-7, atol=1e-6), (target, dict(zip(ALL, zip(a, b))))
        checked += 1
    # Sanity: the features are not trivially zero
    assert base["distinct_days_prior_30"].max() > 0 and base["ti4_std_prior_30"].max() > 0
    print(f"leakage test passed for {checked} detections, {len(ALL)} features")


def test_known_values():
    df = pd.DataFrame({
        "cell_y": [0, 0, 0, 0, 0, 1], "cell_x": [0, 0, 0, 0, 0, 1],
        "acq_date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-02",
                                    "2025-01-11", "2025-02-10", "2025-01-06"]),
        "bright_ti4": [300.0, 310.0, 330.0, 320.0, 340.0, 1.0],
        "frp": [1.0, 2.0, 4.0, 8.0, 16.0, 1.0],
        "source_file": "a.csv"})
    h = all_history(df)
    # v1 values
    assert h["distinct_days_prior_7"].tolist() == [0, 1, 1, 0, 0, 0]
    assert h["distinct_days_prior_30"].tolist() == [0, 1, 1, 2, 1, 0]
    assert h["nbr_distinct_days_prior_7"].tolist() == [0, 1, 1, 1, 0, 2]
    assert h["nbr_distinct_days_prior_30"].tolist() == [0, 1, 1, 3, 1, 2]
    assert h["detections_prior_7"].tolist() == [0, 1, 1, 0, 0, 0]
    assert h["days_since_first_seen"].tolist() == [0, 1, 1, 10, 40, 0]
    assert h["days_into_data"].tolist() == [0, 1, 1, 10, 40, 5]
    # v2 counts and recency
    assert h["distinct_days_prior_60"].tolist() == [0, 1, 1, 2, 3, 0]
    assert h["distinct_days_prior_90"].tolist() == [0, 1, 1, 2, 3, 0]
    assert h["nbr_distinct_days_prior_60"].tolist() == [0, 1, 1, 3, 4, 2]
    assert h["days_since_last_seen"].tolist() == [90, 1, 1, 9, 30, 90]
    assert np.allclose(h["persistence_30"], [0, 1, 1, 2 / 10, 1 / 30, 0])
    assert h["has_history"].tolist() == [0, 1, 1, 1, 1, 0]
    # Means and stds (population std) over detections on strictly earlier days, within 30 days
    t3, f3 = [300.0, 310.0, 330.0], [1.0, 2.0, 4.0]
    assert np.allclose(h["ti4_mean_prior_30"], [0, 300, 300, np.mean(t3), 320, 0])
    assert np.allclose(h["ti4_std_prior_30"], [0, 0, 0, np.std(t3), 0, 0])
    assert np.allclose(h["frp_mean_prior_30"], [0, 1, 1, np.mean(f3), 8, 0])
    assert np.allclose(h["frp_std_prior_30"], [0, 0, 0, np.std(f3), 0, 0])
    assert np.allclose(h["bright_dev"], [0, 10, 30, 320 - np.mean(t3), 20, 0])
    assert np.allclose(h["frp_dev"], [0, 1, 3, 8 - np.mean(f3), 8, 0])
    print("known value test passed")


if __name__ == "__main__":
    test_no_leakage()
    test_known_values()

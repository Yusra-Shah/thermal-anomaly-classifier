"""Build the ground truth label from repeat detections.

NASA's own definition of an 'other static land source' is a hot spot that is
repeatedly detected on many days within a month. We apply the same idea to the
window we downloaded, and compute it ourselves so no login or type field is needed.

Label: is_static = 1 if the grid cell containing the detection was flagged on at
least STATIC_MIN_DAYS distinct days (in its best calendar month, or over the whole
window when window="total"), else 0 (likely vegetation fire).
"""
import numpy as np
import pandas as pd

from .config import GRID_DEG, STATIC_MIN_DAYS, STATIC_WINDOW


def add_cell(df: pd.DataFrame, grid_deg: float = GRID_DEG) -> pd.DataFrame:
    df = df.copy()
    df["cell_y"] = np.floor(df["latitude"] / grid_deg).astype(int)
    df["cell_x"] = np.floor(df["longitude"] / grid_deg).astype(int)
    df["cell_id"] = df["cell_y"].astype(str) + "_" + df["cell_x"].astype(str)
    return df


def add_label(df: pd.DataFrame, min_days: int = STATIC_MIN_DAYS,
              window: str = STATIC_WINDOW) -> pd.DataFrame:
    df = add_cell(df)
    if window == "month":
        month = df["acq_date"].dt.to_period("M")
        days = (df.groupby(["cell_id", month])["acq_date"].nunique()
                .groupby("cell_id").max().rename("cell_days"))
    else:
        days = df.groupby("cell_id")["acq_date"].nunique().rename("cell_days")
    df = df.join(days, on="cell_id")
    df["is_static"] = (df["cell_days"] >= min_days).astype(int)
    return df

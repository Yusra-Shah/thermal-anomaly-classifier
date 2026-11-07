"""Single detection features. Nothing here looks at other detections."""
import numpy as np
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["bright_diff"] = df["bright_ti4"] - df["bright_ti5"]
    df["pixel_area"] = df["scan"] * df["track"]
    df["frp_per_area"] = df["frp"] / df["pixel_area"].replace(0, np.nan)
    df["frp_per_area"] = df["frp_per_area"].fillna(0)
    return df

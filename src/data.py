"""Load and clean raw NASA FIRMS VIIRS 375 m CSV files."""
from pathlib import Path

import pandas as pd

REQUIRED = [
    "latitude", "longitude", "bright_ti4", "scan", "track",
    "acq_date", "acq_time", "confidence", "bright_ti5", "frp", "daynight",
]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    # Archive files use the older names for the I-4 and I-5 channels
    return df.rename(columns={"brightness": "bright_ti4", "bright_t31": "bright_ti5"})


def load_raw(paths=None) -> pd.DataFrame:
    """Read one or more FIRMS CSV files and return a clean dataframe."""
    from .config import RAW_DIR

    if paths is None:
        paths = sorted(Path(RAW_DIR).glob("*.csv"))
    paths = [Path(p) for p in paths]
    if not paths:
        raise FileNotFoundError(
            f"No CSV files found in {RAW_DIR}. Download a FIRMS VIIRS CSV and place it there."
        )

    frames = []
    for p in paths:
        df = normalize_columns(pd.read_csv(p))
        missing = [c for c in REQUIRED if c not in df.columns]
        if missing:
            raise ValueError(f"{p.name} is missing columns: {missing}")
        df["source_file"] = p.name
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    return clean(df)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Time
    df["acq_date"] = pd.to_datetime(df["acq_date"]).dt.normalize()
    t = df["acq_time"].astype(int).astype(str).str.zfill(4)
    df["hour_utc"] = t.str[:2].astype(int) + t.str[2:].astype(int) / 60.0
    df["timestamp"] = df["acq_date"] + pd.to_timedelta(df["hour_utc"], unit="h")

    # Confidence is categorical for VIIRS (l / n / h) but numeric for MODIS
    conf = df["confidence"].astype(str).str.strip().str.lower()
    mapping = {"l": 0, "low": 0, "n": 1, "nominal": 1, "h": 2, "high": 2}
    ordinal = conf.map(mapping)
    numeric = pd.to_numeric(conf, errors="coerce")
    ordinal = ordinal.fillna(pd.cut(numeric, [-1, 29, 79, 101], labels=[0, 1, 2]).astype("float"))
    df["confidence_ord"] = ordinal

    # Day or night
    df["is_night"] = (df["daynight"].astype(str).str.upper().str.strip() == "N").astype(int)

    # Satellite
    if "satellite" in df.columns:
        sat = df["satellite"].astype(str).str.upper().str.strip()
        df["is_noaa20"] = sat.isin(["N20", "1", "NOAA-20", "J1"]).astype(int)
    else:
        df["is_noaa20"] = 0

    # Numeric columns
    for c in ["latitude", "longitude", "bright_ti4", "bright_ti5", "frp", "scan", "track"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["latitude", "longitude", "bright_ti4", "bright_ti5", "frp",
                           "scan", "track", "confidence_ord"])
    df = df[(df["latitude"].between(-90, 90)) & (df["longitude"].between(-180, 180))]
    df = df.drop_duplicates(subset=["latitude", "longitude", "acq_date", "acq_time"])
    df = df.reset_index(drop=True)
    df.attrs["rows_dropped"] = before - len(df)
    return df

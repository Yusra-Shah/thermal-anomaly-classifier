"""Cleaned, labelled and feature tables for every region file in data/raw/, one file at a time.

Same code as before (clean, add_label, add_features, add_history, add_history_v2), applied to each raw file
on its own, so the history features of a detection look only backward inside its own file. Only NASA type 0
(fire) and type 2 (static) rows are kept. A file without a type column is skipped and said so. Float columns
are stored as float32 (the models read float32 anyway). The result is cached in
data/processed/prepared_world.parquet, keyed by the names and sizes of the raw files.
"""
import json

import numpy as np
import pandas as pd

from .config import PROC_DIR, RAW_DIR, TARGET
from .data import load_raw
from .features import add_features
from .history import add_history, add_history_v2
from .label import add_label
from .mem import report
from .regions import FILE_REGION

CACHE = PROC_DIR / "prepared_world.parquet"
STAMP = PROC_DIR / "prepared_world.json"
CATEGORY = ["source_file", "satellite", "instrument", "confidence", "daynight", "version", "region"]
KEEP_F64 = ["latitude", "longitude"]


def raw_files(log=print):
    """Raw files that have a type column and a known region. Others are skipped with a message."""
    out = []
    for p in sorted(RAW_DIR.glob("*.csv")):
        cols = [c.strip().lower() for c in pd.read_csv(p, nrows=0).columns]
        if "type" not in cols:
            log(f"skipped {p.name}: no type column")
            continue
        if p.name not in FILE_REGION:
            log(f"skipped {p.name}: not in src/regions.py")
            continue
        out.append(p)
    return out


def compact(df):
    """Compact dtypes: float32 for floats (not the coordinates), int32 for integers, categories for text."""
    for c in df.columns:
        s = df[c]
        if c in KEEP_F64:
            continue
        if s.dtype == np.float64:
            df[c] = s.astype(np.float32)
        elif s.dtype == np.int64 and c not in ("cell_y", "cell_x"):
            df[c] = pd.to_numeric(s, downcast="integer")
    for c in CATEGORY:
        if c in df.columns:
            df[c] = df[c].astype("category")
    return df


def prepare_file(path):
    """One raw file: cleaned, labelled, with features and history, only types 0 and 2."""
    df = add_features(add_label(load_raw([path])))
    df = add_history_v2(add_history(df))             # history sees every detection, before dropping types
    df = df[df["type"].isin([0, 2])].reset_index(drop=True)
    df[TARGET] = (df["type"] == 2).astype(int)
    df["region"] = FILE_REGION[path.name]
    return compact(df)


def _stamp(files):
    return [[p.name, p.stat().st_size] for p in files]


def prepare(use_cache=True, log=print):
    """All region files as one frame. Uses the cache when the raw files are unchanged."""
    files = raw_files(log)
    stamp = _stamp(files)
    if use_cache and CACHE.exists() and STAMP.exists() and json.loads(STAMP.read_text()) == stamp:
        df = pd.read_parquet(CACHE)
        df["cell_id"] = df["cell_id"].astype(str)
        return df
    frames = []
    for p in files:
        frames.append(prepare_file(p))
        log(f"prepared {p.name}: {len(frames[-1]):,} rows")
        report(f"prepare {p.name}")
    df = pd.concat(frames, ignore_index=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE, index=False)
    STAMP.write_text(json.dumps(stamp))
    return df

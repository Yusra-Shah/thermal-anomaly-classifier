"""Tests for src/noaa20.py and the way NOAA-20 features are built. No network.

Run from the project root:  python -m tests.test_noaa20
"""
import contextlib
import io
import tempfile
import traceback
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src import noaa20
from src.history import HISTORY_FEATURES, add_history

KEY = "SECRETKEY123"


def test_chunks_cover_every_day_once():
    ch = noaa20.chunks()
    days = [d + timedelta(days=i) for d, n in ch for i in range(n)]
    assert len(days) == len(set(days)) == 181                       # 1 January to 30 June 2025
    assert days[0] == date(2025, 1, 1) and days[-1] == date(2025, 6, 30)
    assert days == sorted(days) and all(b - a == timedelta(days=1) for a, b in zip(days, days[1:]))
    assert all(1 <= n <= noaa20.MAX_DAY_RANGE for _, n in ch) and len(ch) == 37
    assert len(ch) * len(noaa20.REGION_BOXES) == 111                # about 111 calls
    short = noaa20.chunks(date(2025, 3, 1), date(2025, 3, 3))
    assert short == [(date(2025, 3, 1), 3)]
    print("chunks cover every day exactly once: passed")


def test_key_never_printed():
    """A failing request whose message contains the URL (and so the key) must not leak it."""
    class Boom:
        def get(self, url, timeout=0):
            raise ConnectionError(f"HTTPSConnectionPool failed for url {url}")

    class Http500:
        def get(self, url, timeout=0):
            class R:
                status_code = 500
                text = "server error for " + url
            return R()

    class Odd:
        def get(self, url, timeout=0):
            class R:
                status_code = 200
                text = "plain text refusal " + url
            return R()

    out, err = io.StringIO(), io.StringIO()
    shown = []
    with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        old = noaa20.OUT_DIR
        noaa20.OUT_DIR = Path(tmp)
        try:
            for session, kind in ((Boom(), "ConnectionError"), (Http500(), "HTTPError"), (Odd(), "NotCsv")):
                try:
                    noaa20.fetch((61, 24, 76, 36), date(2025, 1, 1), 5, KEY, session=session, retries=1,
                                 sleep=lambda s: None)
                    raise AssertionError("fetch should fail")
                except noaa20.SafeError as e:
                    assert KEY not in str(e) and KEY not in repr(e)
                    assert kind in str(e) and "HTTP status" in str(e)
                    assert e.__cause__ is None and e.__context__ is None           # no chained requests error
                    assert KEY not in "".join(traceback.format_exception(e))
                    shown.append(str(e))
            # the download loop reports only the safe message
            calls, rows = noaa20.download(KEY, fetcher=lambda box, d, n, k: noaa20.fetch(
                box, d, n, k, session=Boom(), retries=0, sleep=lambda s: None), sleep=lambda s: None,
                regions=["South Asia"], log=print)
            assert calls == 0 and rows == 0
        finally:
            noaa20.OUT_DIR = old
    text = out.getvalue() + err.getvalue() + " ".join(shown)
    assert KEY not in text and "://" not in text                 # no URL either
    assert "chunk 1/37 South Asia" in text and "ConnectionError" in text
    print("key never printed: passed")


def test_download_resumes():
    header = "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_ti5,frp,daynight,type\n"
    seen = []

    def fake(box, day, n, key):
        seen.append((box, day, n))
        return header + f"25.0,68.0,330.0,0.5,0.5,{day:%Y-%m-%d},0742,N20,VIIRS,n,2,290.0,5.0,D,0\n"
    with tempfile.TemporaryDirectory() as tmp:
        old = noaa20.OUT_DIR
        noaa20.OUT_DIR = Path(tmp)
        try:
            calls, rows = noaa20.download(KEY, fetcher=fake, sleep=lambda s: None, regions=["South Asia"],
                                          log=lambda *a: None)
            assert calls == 37 and rows == 37 and not noaa20.missing_chunks() == []
            again, _ = noaa20.download(KEY, fetcher=fake, sleep=lambda s: None, regions=["South Asia"],
                                       log=lambda *a: None)
            assert again == 0 and len(seen) == 37                                   # nothing fetched twice
            rev = noaa20.download(KEY, fetcher=fake, sleep=lambda s: None, regions=["Southeast Asia"],
                                  log=lambda *a: None, reverse=True)
            assert seen[37][2] == 1                                                  # reverse starts at the last chunk
            df = noaa20.combine("South Asia")
            assert len(df) == 37 and "type" in df.columns                           # type column kept
        finally:
            noaa20.OUT_DIR = old
    print("download resumes and keeps type: passed")


def test_country_cut():
    lon = np.array([68.86, 77.20, 106.8, 100.5, -97.0, -100.0])
    lat = np.array([27.70, 28.60, -6.2, 13.7, 38.0, 55.0])
    assert noaa20.in_country(lon, lat, "PK").tolist() == [True, False, False, False, False, False]
    assert noaa20.in_country(lon, lat, "ID").tolist() == [False, False, True, False, False, False]
    assert noaa20.in_country(lon, lat, "US").tolist() == [False, False, False, False, True, False]
    print("country cut: passed")


def test_history_built_per_file():
    """History features of a file do not change when another file is added, because each source file
    is processed on its own."""
    rng = np.random.default_rng(0)

    def frame(name, seed):
        r = np.random.default_rng(seed)
        n = 600
        df = pd.DataFrame({"cell_y": r.integers(2400, 2410, n), "cell_x": r.integers(6800, 6810, n),
                           "acq_date": pd.Timestamp("2025-01-01") + pd.to_timedelta(r.integers(0, 60, n), unit="D"),
                           "source_file": name})
        return df
    a, b = frame("noaa20_a.csv", 1), frame("noaa20_b.csv", 2)
    both = pd.concat([a, b], ignore_index=True)
    got = add_history(both)
    alone = add_history(a)
    cols = HISTORY_FEATURES
    assert np.array_equal(got.iloc[:len(a)][cols].to_numpy(), alone[cols].to_numpy())
    # and a file that starts later has its own day zero
    late = b.assign(acq_date=b["acq_date"] + pd.Timedelta(days=100), source_file="noaa20_late.csv")
    mixed = add_history(pd.concat([a, late], ignore_index=True))
    assert mixed.iloc[len(a):]["days_into_data"].min() == 0
    # the transfer script builds NOAA-20 with exactly these calls
    src = open(Path(noaa20.__file__).with_name("noaa20_transfer.py"), encoding="utf-8").read()
    assert "add_history_v2(add_history(df))" in src and 'df["type"].isin([0, 2])' in src
    print("history built per file: passed")


def main():
    test_chunks_cover_every_day_once()
    test_key_never_printed()
    test_download_resumes()
    test_country_cut()
    test_history_built_per_file()


if __name__ == "__main__":
    main()
    from tests.test_live import peak_memory
    ws, commit = peak_memory()
    print(f"peak memory: working set {ws:.0f} MB, commit {commit:.0f} MB")

"""Tests for src/nrt_vs_sp.py and src/nrt_analysis.py. No network, no models: fake responses only.

Run from the project root:  python -m tests.test_nrt_vs_sp
"""
import contextlib
import io
import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src import noaa20, nrt_analysis, nrt_vs_sp

KEY = "SECRETKEY123"


def test_sets_and_chunks():
    assert nrt_vs_sp.SETS["A_SP"][0] == "VIIRS_NOAA21_SP" and nrt_vs_sp.SETS["A_NRT"][0] == "VIIRS_NOAA21_NRT"
    assert nrt_vs_sp.SETS["A_SP"][1:] == nrt_vs_sp.SETS["A_NRT"][1:] == (date(2026, 6, 1), date(2026, 7, 31))
    assert nrt_vs_sp.SETS["B_SP"] == ("VIIRS_NOAA21_SP", date(2025, 9, 11), date(2025, 11, 9))
    for tag, (_, start, end) in nrt_vs_sp.SETS.items():
        days = [d for first, n in noaa20.chunks(start, end) for d in pd.date_range(first, periods=n)]
        assert len(days) == len(set(days)) == (end - start).days + 1, tag
    print("sets and chunks: passed")


def test_download_uses_source_and_resumes():
    header = "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_ti5,frp,daynight\n"
    seen = []

    def fake(box, day, n, key, source=None):
        seen.append((source, day))
        return header + f"25.0,68.0,330.0,0.5,0.5,{day:%Y-%m-%d},0742,N21,VIIRS,n,2.0NRT,290.0,5.0,D\n"
    with tempfile.TemporaryDirectory() as tmp:
        old = nrt_vs_sp.DIR
        nrt_vs_sp.DIR = Path(tmp)
        try:
            calls = nrt_vs_sp.download(KEY, fetcher=fake, sleep=lambda s: None, regions=["South Asia"], log=lambda *a: None)
            assert calls == 13 + 13 + 12 == len(seen)
            assert {s for s, _ in seen} == {"VIIRS_NOAA21_SP", "VIIRS_NOAA21_NRT"}
            assert not nrt_vs_sp.missing(regions=["South Asia"]) and len(nrt_vs_sp.missing()) == 76
            assert nrt_vs_sp.download(KEY, fetcher=fake, sleep=lambda s: None, regions=["South Asia"],
                                      log=lambda *a: None) == 0                          # resumes, nothing twice
            df = nrt_vs_sp.load_set("A_NRT", "South Asia")
            assert len(df) == 13 and df["confidence"].eq("n").all()
        finally:
            nrt_vs_sp.DIR = old
    print("download source and resume: passed")


def test_key_never_printed():
    class Boom:
        def get(self, url, timeout=0):
            raise ConnectionError(f"failed for url {url}")
    out = io.StringIO()
    with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(out):
        old = nrt_vs_sp.DIR
        nrt_vs_sp.DIR = Path(tmp)
        try:
            def fetcher(box, day, n, key, source=None):
                return noaa20.fetch(box, day, n, key, session=Boom(), retries=0, sleep=lambda s: None, source=source)
            calls = nrt_vs_sp.download(KEY, tags=["A_NRT"], regions=["South Asia"], fetcher=fetcher,
                                       sleep=lambda s: None, log=print)
            assert calls == 0
        finally:
            nrt_vs_sp.DIR = old
    text = out.getvalue()
    assert KEY not in text and "://" not in text and "ConnectionError" in text and "HTTP status" in text
    print("key never printed: passed")


def frame(lat, lon, day, minutes, conf="n"):
    n = len(lat)
    hour = np.asarray(minutes) / 60.0
    return pd.DataFrame({"latitude": lat, "longitude": lon, "acq_date": pd.to_datetime([day] * n),
                         "hour_utc": hour, "confidence": conf})


def test_match_rules():
    sp = frame([10.0, 11.0, 12.0, 13.0], [70.0, 71.0, 72.0, 73.0], "2026-06-01", [100, 100, 100, 1430])
    nrt = frame([10.001, 11.01, 12.001, 13.001], [70.001, 71.0, 72.0, 73.001], "2026-06-01", [110, 110, 140, 5])
    ia, ib = nrt_analysis.match(nrt, sp)
    # row 0 matches (close, 10 minutes), row 1 is too far, row 2 is 40 minutes away,
    # row 3 matches across midnight (1430 and 5 are 15 minutes apart)
    assert sorted(zip(ia.tolist(), ib.tolist())) == [(0, 0), (3, 3)]
    other_day = nrt.assign(acq_date=pd.Timestamp("2026-06-02"))
    assert len(nrt_analysis.match(other_day, sp)[0]) == 0
    print("match rules: passed")


def test_confidence_counts_and_same_categories():
    raw = pd.DataFrame({"confidence": ["n", "N", "l", "h", "n", " n"]})
    cc = nrt_analysis.conf_counts(raw)
    assert abs(cc["n"] - 4 / 6) < 1e-9 and abs(cc["l"] - 1 / 6) < 1e-9 and abs(cc["h"] - 1 / 6) < 1e-9
    from src.data import clean
    df = pd.DataFrame({"acq_date": ["2026-06-01"] * 3, "acq_time": [100, 200, 300], "confidence": ["l", "n", "h"],
                       "daynight": ["D", "N", "D"], "satellite": ["N21"] * 3, "latitude": [1.0, 2.0, 3.0],
                       "longitude": [1.0, 2.0, 3.0], "bright_ti4": [330.0] * 3, "bright_ti5": [290.0] * 3,
                       "frp": [3.0] * 3, "scan": [0.4] * 3, "track": [0.4] * 3})
    assert clean(df)["confidence_ord"].tolist() == [0, 1, 2]       # same mapping for SP and NRT text
    print("confidence counts and mapping: passed")


def test_remap_hook_is_applied():
    seen = {}

    def remap(df):
        seen["n"] = len(df)
        return df.assign(confidence="n")

    class Stub:
        pass
    import src.live as live
    old = live.prepare, live.score
    live.prepare = lambda raw, name: raw.assign(marker=1)
    live.score = lambda df, h, i: df
    try:
        raw = pd.DataFrame({"confidence": ["l", "h"], "x": [1, 2]})
        out = nrt_analysis.scored(raw, "t", Stub(), Stub(), remap=remap)
        assert seen["n"] == 2 and out["confidence"].tolist() == ["n", "n"] and raw["confidence"].tolist() == ["l", "h"]
    finally:
        live.prepare, live.score = old
    print("remap hook: passed")


def main():
    test_sets_and_chunks()
    test_download_uses_source_and_resumes()
    test_key_never_printed()
    test_match_rules()
    test_confidence_counts_and_same_categories()
    test_remap_hook_is_applied()


if __name__ == "__main__":
    main()
    from tests.test_live import peak_memory
    ws, commit = peak_memory()
    print(f"peak memory: working set {ws:.0f} MB, commit {commit:.0f} MB")

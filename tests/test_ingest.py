"""Tests for src/ingest.py and src/serve.py. Fake feed only, no network.

Run from the project root, one suite per process:  python -m pytest tests/test_ingest.py -q
"""
import hashlib

import numpy as np
import pandas as pd
import pytest

from src import ingest, live, serve
from src import uncertain as U
from tests import fake_feed as fake

KEY = "SECRETKEY123"


@pytest.fixture
def live_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "LIVE_DIR", tmp_path)
    return tmp_path


@pytest.fixture(scope="module")
def models():
    return serve.Models()


def run(fetch, models, **kw):
    kw.setdefault("sleep", lambda s: None)
    kw.setdefault("log", lambda *a: None)
    return ingest.run_once(KEY, fetch=fetch, models=models, **kw)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_unique(by_source, lo, hi):
    """Unique rows per sensor of the fake feed with acq_date in [lo, hi] (every row is in the South Asia box)."""
    out = {}
    for sensor, source in live.LIVE_SOURCES.items():
        m = by_source[source]
        d = pd.to_datetime(m["acq_date"])
        out[sensor] = len(m[(d >= lo) & (d <= hi)].drop_duplicates(["latitude", "longitude", "acq_date", "acq_time"]))
    return out


def test_dedupe_and_idempotence(live_dir, models):
    by = fake.masters(8)
    fetch = fake.fetch_factory(by, extra_dups=True)                 # the feed itself repeats rows
    r1 = run(fetch, models)
    want = expected_unique(by, fake.today() - pd.Timedelta(days=7), fake.today())
    assert {s: c["added"] for s, c in r1["counts"].items()} == want and r1["changed"]
    path = ingest.store_path()
    store = ingest.read_store()
    assert not store.duplicated(ingest.KEY_COLS).any()
    assert set(store["sensor"]) == set(live.LIVE_SOURCES)
    assert len(store) == sum(want.values())
    before = digest(path)
    r2 = run(fetch, models)                                         # a second run in a row adds nothing
    assert all(c["added"] == 0 for c in r2["counts"].values()) and not r2["changed"]
    assert {s: c["kept"] for s, c in r2["counts"].items()} == want
    assert digest(path) == before                                   # and does not touch the file
    # the same row from the other sensor is a different row, a repeat of the same sensor is not
    m20 = by[live.LIVE_SOURCES["NOAA-20"]]
    one = m20[m20["acq_date"] == fake.today().strftime("%Y-%m-%d")].iloc[[0]].copy()      # inside the refetch window
    by[live.LIVE_SOURCES["NOAA-21"]] = pd.concat([by[live.LIVE_SOURCES["NOAA-21"]], one], ignore_index=True)
    r3 = run(fake.fetch_factory(by), models)
    assert r3["counts"]["NOAA-21"]["added"] == 1 and r3["counts"]["NOAA-20"]["added"] == 0


def test_trims_to_90_days(live_dir, models):
    t = fake.today()
    by = fake.masters(110, end=t + pd.Timedelta(days=10))
    run(fake.fetch_factory(by), models, today=t)
    s1 = ingest.read_store()
    assert s1["acq_date"].min() == t - pd.Timedelta(days=89) and s1["acq_date"].max() == t
    r2 = run(fake.fetch_factory(by), models, today=t + pd.Timedelta(days=10))
    s2 = ingest.read_store()
    cutoff = t + pd.Timedelta(days=10) - pd.Timedelta(days=89)
    assert s2["acq_date"].min() == cutoff and s2["acq_date"].max() == t + pd.Timedelta(days=10)
    assert (s2["acq_date"].max() - s2["acq_date"].min()).days == 89                  # exactly 90 days
    want = expected_unique(by, cutoff, t + pd.Timedelta(days=10))
    assert {s: c["kept"] for s, c in r2["counts"].items()} == want
    assert r2["changed"] and not s2.duplicated(ingest.KEY_COLS).any()


def test_atomic_write(tmp_path, monkeypatch):
    path = tmp_path / "store.parquet"
    old = pd.DataFrame({"a": [1, 2, 3]})
    ingest.write_store(old, path)
    assert path.exists() and not list(tmp_path.glob("*.tmp"))
    seen = {}
    real = ingest.os.replace

    def spy(src, dst):
        seen["tmp_exists"] = ingest.Path(src).exists() if hasattr(ingest, "Path") else __import__("pathlib").Path(src).exists()
        seen["old_intact"] = pd.read_parquet(dst).equals(old)                  # still the old file at that moment
        seen["same_folder"] = str(src).startswith(str(tmp_path)) and str(src) != str(dst)
        return real(src, dst)
    monkeypatch.setattr(ingest.os, "replace", spy)
    new = pd.DataFrame({"a": [9, 8]})
    ingest.write_store(new, path)
    assert seen == {"tmp_exists": True, "old_intact": True, "same_folder": True}
    assert pd.read_parquet(path).equals(new)
    monkeypatch.setattr(ingest.os, "replace", real)

    def broken(self, target, *a, **k):                                         # a crash in the middle of the write
        open(target, "wb").write(b"partial")
        raise OSError("disk full")
    monkeypatch.setattr(pd.DataFrame, "to_parquet", broken)
    with pytest.raises(OSError):
        ingest.write_store(pd.DataFrame({"a": [5]}), path)
    monkeypatch.undo()
    assert pd.read_parquet(path).equals(new) and not list(tmp_path.glob("*.tmp"))


def test_scores_bands_and_ingest_time(live_dir, models):
    t = fake.today()
    by = fake.masters(20, end=t + pd.Timedelta(days=2))
    t0, t1 = pd.Timestamp("2026-11-01 06:00:00"), pd.Timestamp("2026-11-01 12:00:00")
    run(fake.fetch_factory(by), models, today=t, now=t0)
    s1 = ingest.read_store()
    assert set(ingest.SCORE_COLS) <= set(s1.columns) and (s1["ingest_time"] == t0).all()
    run(fake.fetch_factory(by), models, today=t + pd.Timedelta(days=2), now=t1)
    s = ingest.read_store()
    old = s.merge(s1[ingest.KEY_COLS].assign(_o=True), on=ingest.KEY_COLS, how="left")["_o"].fillna(False).to_numpy(bool)
    assert (s.loc[old, "ingest_time"] == t0).all() and (s.loc[~old, "ingest_time"] == t1).all() and (~old).any()
    # the stored scores are the saved models applied to the stored features
    p = models.inst["model"].predict_proba(s[models.instant_features])[:, 1]
    assert np.allclose(p, s["p_instant"], atol=1e-6)
    ph = models.hist["model"].predict_proba(s[models.history_features])[:, 1]
    assert np.allclose(ph, s["p_history"], atol=2e-3)
    for tier, col in (("instant", "p_instant"), ("history", "p_history")):
        cal = U.apply_calibrator(models.bands[tier]["calibrator"], s[col].to_numpy(float))
        assert np.allclose(cal, s[f"pc_{tier}"], atol=1e-6)
        assert (s[f"{tier}_default"] == serve.band_calls(s[f"pc_{tier}"], models.bands[tier]["default"])).all()
        assert (s[f"{tier}_strict"] == serve.band_calls(s[f"pc_{tier}"], models.bands[tier]["strict"])).all()
        assert set(s[f"{tier}_default"]) <= {"fire", "static", "uncertain"}
    thr = {"instant": models.instant_threshold, "history": models.history_threshold}
    for tier in thr:
        assert (s[f"label_{tier}"] == np.where(s[f"p_{tier}"] >= thr[tier], "static", "fire")).all()


def test_history_stays_inside_one_sensor(live_dir, models):
    by = fake.masters(30)
    run(fake.fetch_factory(by), models)
    both = ingest.read_store()
    only = fake.masters(30)
    only[live.LIVE_SOURCES["NOAA-21"]] = only[live.LIVE_SOURCES["NOAA-21"]].iloc[0:0]
    for p in live_dir.glob("store.parquet"):
        p.unlink()
    run(fake.fetch_factory(only), models)
    alone = ingest.read_store()
    a = both[both["sensor"] == "NOAA-20"].sort_values(ingest.KEY_COLS).reset_index(drop=True)
    b = alone[alone["sensor"] == "NOAA-20"].sort_values(ingest.KEY_COLS).reset_index(drop=True)
    assert len(a) == len(b) > 0
    for f in models.history_only:
        assert np.allclose(a[f].to_numpy(float), b[f].to_numpy(float), atol=1e-6), f


def test_day_range_pause_and_boxes(live_dir, models):
    log, waits = [], []
    run(fake.fetch_factory(fake.masters(3), log), models, sleep=waits.append)
    assert log and all(1 <= days <= 5 for _, days, _, _ in log)
    assert {src for _, _, src, _ in log} == set(live.LIVE_SOURCES.values())
    assert {box for _, _, _, box in log} == {live.REGIONS[r] for r in live.TRAINED}
    assert len(waits) == len(log) and all(w == live.PAUSE for w in waits)           # a short wait after every call
    # the first run asks for 90 days in chunks of at most 5, per sensor and region
    per = [(src, box) for _, _, src, box in log]
    assert len(log) == 2 * 3 * 18 and per.count((live.LIVE_SOURCES["NOAA-20"], live.REGIONS["South Asia"])) == 18


def test_rate_limiter_waits_inside_the_window():
    clock = {"t": 0.0}
    slept = []

    def sleep(s):
        slept.append(s)
        clock["t"] += s
    lim = ingest.RateLimiter(budget=100, window=600, clock=lambda: clock["t"], sleep=sleep)
    lim.wait(40)
    clock["t"] += 10
    lim.wait(40)
    assert slept == []
    clock["t"] += 10
    lim.wait(40)                                         # 120 in 20 seconds: must wait for the first call to age out
    assert len(slept) == 1 and clock["t"] >= 600
    assert sum(c for _, c in lim.calls) <= 100
    assert ingest.BUDGET < ingest.LIMIT == 5000 and ingest.COST_PER_DAY * ingest.live.MAX_DAY_RANGE * 2 < ingest.BUDGET


def test_key_never_printed(live_dir, models, capsys, monkeypatch):
    class Boom:
        def get(self, url, timeout=None):
            raise ConnectionError(f"failed for url {url}")

    class Refused:
        def get(self, url, timeout=None):
            class R:
                status_code = 403
                text = f"Invalid MAP_KEY {KEY} for {url}"
            return R()

    class Odd(Exception):
        pass

    def raw_fetch(key, box, start, days, source):
        raise Odd(f"https://firms.example/{key}/{source}")

    def boom_fetch(key, box, start, days, source):
        return live.fetch_chunk(key, box, start, days, source=source, session=Boom(), sleep=lambda s: None)

    def refused_fetch(key, box, start, days, source):
        return live.fetch_chunk(key, box, start, days, source=source, session=Refused(), sleep=lambda s: None)

    for fetch, want in ((raw_fetch, "Odd, HTTP status none"), (boom_fetch, "ConnectionError, HTTP status none"),
                        (refused_fetch, "HTTP status 403")):
        r = ingest.run_once(KEY, fetch=fetch, models=models, sleep=lambda s: None, log=print, today=fake.today())
        out = capsys.readouterr()
        text = out.out + out.err
        assert KEY not in text and "://" not in text and "firms.example" not in text, (want, text[:200])
        assert want in text and r["failures"] == 2 * 3
    # the command line path too, with a fake key in the environment
    monkeypatch.setenv("FIRMS_MAP_KEY", KEY)
    monkeypatch.setattr(live, "fetch_chunk", boom_fetch)
    monkeypatch.setattr(ingest, "serve", serve)
    monkeypatch.setattr(ingest.live, "PAUSE", 0.0)
    monkeypatch.setattr(ingest.time, "sleep", lambda s: None)
    ingest.main(["--once"])
    out = capsys.readouterr()
    assert KEY not in out.out + out.err and "://" not in out.out + out.err and "failed" in out.out
    # a crash in the middle of a round shows the class only
    monkeypatch.setattr(ingest, "run_once", lambda *a, **k: (_ for _ in ()).throw(Odd(f"{KEY} https://x")))
    ingest.main(["--once"])
    out = capsys.readouterr()
    assert KEY not in out.out + out.err and "ingest failed: Odd, HTTP status none" in out.out


def test_missing_key_message_has_no_secret(monkeypatch, capsys):
    monkeypatch.setattr(live, "load_key", lambda: None)
    with pytest.raises(SystemExit) as e:
        ingest.main(["--once"])
    assert "No map key" in str(e.value)


def test_import_live_takes_the_rolling_stores(live_dir, models):
    by = fake.masters(5)
    for sat, source in live.LIVE_SOURCES.items():
        raw = by[source].copy()
        raw["satellite_name"] = sat
        d = live.region_dir("South Asia", sat)
        d.mkdir(parents=True, exist_ok=True)
        raw.to_parquet(d / "2026-01-01.parquet", index=False)
    r = ingest.import_live(models=models, log=lambda *a: None)
    assert r["counts"]["NOAA-20"]["added"] > 0 and r["counts"]["NOAA-21"]["added"] > 0
    assert not ingest.read_store().duplicated(ingest.KEY_COLS).any()

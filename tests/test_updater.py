"""Tests for the background update (src/updater.py) and the freshness texts. A fake command stands in for the
ingest: nothing here touches the network or the real store.

Run from the project root, one suite per process:  python -m pytest tests/test_updater.py -q
"""
import sys
import time

import pandas as pd
import pytest

from src import ingest, live, updater

KEY = "SECRETKEY123"


@pytest.fixture
def state(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "LIVE_DIR", tmp_path)
    monkeypatch.delenv(ingest.STORE_ENV, raising=False)
    return tmp_path / "ingest_state"


def fake(code):
    return [sys.executable, "-c", code]


def wait(limit=30):
    for _ in range(int(limit * 4)):
        if not updater.is_running():
            return
        time.sleep(0.25)
    raise AssertionError("the fake update did not end")


def test_start_runs_one_update_at_a_time(state):
    cmd = fake("import time; print('NOAA-20: fetched 1, added 1, kept 1'); time.sleep(3); "
               "print('[memory] ingest run: peak working set 1 MB, peak commit 2 MB')")
    assert updater.start(cmd) == "started"
    assert (state / "update.lock").exists() and updater.is_running()
    assert updater.start(cmd) == "running"                              # refused while it runs
    assert updater.status()["state"] == "running"
    wait()
    assert not (state / "update.lock").exists()                         # the lock is released
    st = updater.status()
    assert st["state"] == "finished" and "all calls succeeded" in st["message"]
    assert any(line.startswith("NOAA-20: fetched 1") for line in st["lines"])
    assert (state / "update.log").exists()                              # the output is in the log under ingest_state
    assert updater.start(fake("print('[memory] ingest run: peak working set 1 MB, peak commit 2 MB')")) == "started"
    wait()


def test_stale_lock_of_a_dead_process_is_removed(state):
    state.mkdir(parents=True)
    (state / "update.lock").write_text("999999", encoding="utf-8")        # no such process
    assert not updater.is_running() and not (state / "update.lock").exists()
    assert updater.pid_alive(None) is False


def test_failure_shows_class_and_status_only(state, monkeypatch):
    monkeypatch.setattr(live, "load_key", lambda: KEY)
    code = ("raise RuntimeError('https://firms.example/api/%s/VIIRS and a header X-Key: %s')" % (KEY, KEY))
    assert updater.start(fake(code)) == "started"
    wait()
    st = updater.status()
    text = repr(st)
    assert st["state"] == "failed" and st["message"] == "RuntimeError, HTTP status none"
    assert KEY not in text and "://" not in text and "Traceback" not in text and "X-Key" not in text
    # the ingest's own failure line is passed on as it is (it holds the class and the status only)
    assert updater.start(fake("print('ingest failed: ConnectionError, HTTP status none')")) == "started"
    wait()
    assert updater.status()["message"] == "ConnectionError, HTTP status none"
    # a run that ended without finishing is reported as such
    assert updater.start(fake("print('nothing useful')")) == "started"
    wait()
    assert updater.status()["state"] == "failed" and "stopped before it finished" in updater.status()["message"]


def test_failed_calls_are_reported(state):
    assert updater.start(fake("print('failed calls: 3'); print('[memory] ingest run: peak working set 1 MB, "
                              "peak commit 2 MB')")) == "started"
    wait()
    st = updater.status()
    assert st["state"] == "finished" and "3 calls failed" in st["message"]


def test_log_lines_are_filtered(state, monkeypatch):
    monkeypatch.setattr(live, "load_key", lambda: KEY)
    text = ("noise line with https://x/y\nNOAA-20: fetched 5, added 2, kept 9\nFile \"a.py\", line 3\n"
            "[memory] fetch: peak working set 1 MB, peak commit 2 MB\nNOAA-21 Brazil: failed, SSLError, HTTP status none\n"
            f"NOAA-21 line with the key {KEY}\n")
    lines = updater.safe_lines(text)
    assert lines == ["NOAA-20: fetched 5, added 2, kept 9", "[memory] fetch: peak working set 1 MB, peak commit 2 MB",
                     "NOAA-21 Brazil: failed, SSLError, HTTP status none", "NOAA-21 line with the key ***"]
    assert updater.parse_log(text)[0] == "failed"                        # no end marker


def test_missing_key_is_reported_without_a_value(monkeypatch):
    monkeypatch.setattr(live, "load_key", lambda: None)
    assert updater.key_missing()
    monkeypatch.setattr(live, "load_key", lambda: KEY)
    assert not updater.key_missing() and KEY not in repr(updater.key_missing())


def test_command_is_the_ingest_and_the_warning_is_plain():
    assert updater.COMMAND[1:] == ["-m", "src.ingest", "--once"]
    assert "takes a while" in updater.WARNING and "lot of memory" in updater.WARNING
    assert "Close other programs" in updater.WARNING
    source = open(updater.__file__, encoding="utf-8").read()
    assert "FIRMS_MAP_KEY" not in source and "requests" not in source and "print(" not in source


# ------------------------------------------------------------------ freshness texts
def test_age_text():
    assert live.age_text(0) == "0 minutes" and live.age_text(18) == "18 minutes"
    assert live.age_text(119) == "119 minutes" and live.age_text(120) == "2.0 hours"
    assert live.age_text(7.4 * 60) == "7.4 hours" and live.age_text(2 * 1440 + 3 * 60) == "2 days 3 hours"
    assert live.age_text(-5) == "0 minutes"


def test_stale_banner_after_six_hours():
    now = pd.Timestamp("2026-11-20 12:00")
    assert live.stale_text(now - pd.Timedelta(hours=5, minutes=59), now, 6) is None
    assert live.stale_text(now + pd.Timedelta(hours=1), now, 6) is None                  # a time ahead is not stale
    text = live.stale_text(now - pd.Timedelta(hours=7, minutes=24), now, 6)
    assert text.startswith("Stale data") and "7.4 hours old" in text and "2026-11-20 04:36 UTC" in text
    assert "python -m src.ingest --once" in text and "Update live data" in text
    assert "no detections" in live.stale_text(None, now, 6)


def test_empty_selection_names_the_latest_detection_and_the_command():
    now = pd.Timestamp("2026-11-20 12:00")
    text = live.empty_text(24, now - pd.Timedelta(days=3, hours=2), now)
    assert "No detections in this selection in the last 24 hours" in text
    assert "2026-11-17 10:00 UTC" in text and "3 days 2 hours ago" in text and "python -m src.ingest --once" in text
    assert "python -m src.ingest --once" in live.empty_text(6, None, now)


def test_latest_detection_from_the_store_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "LIVE_DIR", tmp_path)
    assert ingest.latest_detection() is None
    df = pd.DataFrame({"sensor": ["NOAA-20", "NOAA-20", "NOAA-21"], "region": ["Brazil", "Brazil", "Nigeria"],
                       "timestamp": pd.to_datetime(["2026-11-01 01:00", "2026-11-02 05:30", "2026-11-03 07:15"]),
                       "ingest_time": pd.to_datetime(["2026-11-03 08:00"] * 3)})
    ingest.write_store(df)
    assert ingest.latest_detection() == pd.Timestamp("2026-11-03 07:15")
    assert ingest.latest_detection(region="Brazil") == pd.Timestamp("2026-11-02 05:30")
    assert ingest.latest_detection(region="Nigeria") == pd.Timestamp("2026-11-03 07:15")
    assert ingest.latest_detection(region="Europe") is None

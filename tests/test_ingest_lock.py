"""Tests for the ingest lock in src/ingest.py: one ingest at a time for --once, --loop and the Update live data
button, stale locks, release on every exit. A fake feed stands in for the network.

Run from the project root, one suite per process:  python -m pytest tests/test_ingest_lock.py -q
"""
import os
import sys
import time
from time import sleep as real_sleep      # the tests patch time.sleep for the ingest, not for their own waits

import pytest

from src import ingest, live, updater
from tests import fake_feed as fake

KEY = "SECRETKEY123"
DEAD_PID = 999999


@pytest.fixture
def box(tmp_path, monkeypatch, capsys):
    """A temporary live folder, a fake key, no waits, and a fake feed that records its calls."""
    monkeypatch.setattr(live, "LIVE_DIR", tmp_path)
    monkeypatch.delenv(ingest.STORE_ENV, raising=False)
    monkeypatch.setenv("FIRMS_MAP_KEY", KEY)
    monkeypatch.setattr(live, "load_key", lambda: KEY)
    monkeypatch.setattr(live, "PAUSE", 0.0)
    monkeypatch.setattr(ingest.time, "sleep", lambda s: None)
    calls = []
    monkeypatch.setattr(live, "fetch_chunk", fake.fetch_factory(fake.masters(6), calls))
    return tmp_path, calls


def lock_file(tmp, name="ingest.lock"):
    return tmp / "ingest_state" / name


ONE_REGION = ["--regions", "South Asia"]


# ------------------------------------------------------------------ the lock itself
def test_lock_is_taken_once_and_released(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "LIVE_DIR", tmp_path)
    a, b = ingest.Lock(), ingest.Lock()
    assert a.acquire() and lock_file(tmp_path).read_text() == str(os.getpid())
    assert not b.acquire() and not b.held                       # a live owner: refused
    b.release()
    assert lock_file(tmp_path).exists()                         # a process that does not hold it removes nothing
    a.release()
    assert not lock_file(tmp_path).exists()
    assert b.acquire()
    b.release()


def test_stale_lock_of_a_dead_process_is_replaced(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "LIVE_DIR", tmp_path)
    (tmp_path / "ingest_state").mkdir()
    lock_file(tmp_path).write_text(str(DEAD_PID))
    assert not ingest.pid_alive(DEAD_PID)
    lk = ingest.Lock()
    assert lk.acquire() and lock_file(tmp_path).read_text() == str(os.getpid())
    lk.release()
    # a lock file with no readable process id: trusted when young, stale when old
    lock_file(tmp_path).write_text("")
    assert ingest.Lock().alive()
    old = time.time() - 60
    os.utime(lock_file(tmp_path), (old, old))
    assert not ingest.Lock().alive()
    lk = ingest.Lock()
    assert lk.acquire()
    lk.release()


def test_lock_follows_the_store_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "LIVE_DIR", tmp_path / "a")
    assert ingest.Lock().file == tmp_path / "a" / "ingest_state" / "ingest.lock"
    monkeypatch.setenv(ingest.STORE_ENV, str(tmp_path / "dev" / "store.parquet"))
    assert ingest.Lock().file == tmp_path / "dev" / "ingest_state" / "ingest.lock"


# ------------------------------------------------------------------ --once with a fake feed
def test_once_takes_the_lock_and_releases_it(box, capsys):
    tmp, calls = box
    seen = {}
    real = ingest.run_once

    def spy(*a, **k):
        seen["held"] = lock_file(tmp).exists()                  # the lock is there while the round runs
        return real(*a, **k)
    ingest.run_once = spy
    try:
        ingest.main(["--once"] + ONE_REGION)
    finally:
        ingest.run_once = real
    out = capsys.readouterr()
    assert seen["held"] is True and not lock_file(tmp).exists()
    assert calls and ingest.store_path().exists()               # the round did its work
    assert KEY not in out.out + out.err and "://" not in out.out + out.err
    assert ingest.LOCK_MESSAGE not in out.out


def test_second_once_finds_the_lock_and_does_nothing(box, capsys):
    tmp, calls = box
    holder = ingest.Lock()
    assert holder.acquire()
    ingest.main(["--once"] + ONE_REGION)
    out = capsys.readouterr().out
    assert "another ingest is running" in out
    assert not calls and not ingest.store_path().exists()      # no fetch and no store
    assert lock_file(tmp).read_text() == str(os.getpid())      # the other ingest's lock is untouched
    holder.release()
    ingest.main(["--once"] + ONE_REGION)                        # free again: the run goes ahead
    assert calls and ingest.store_path().exists()
    assert not lock_file(tmp).exists()


def test_once_replaces_a_stale_lock(box):
    tmp, calls = box
    (tmp / "ingest_state").mkdir()
    lock_file(tmp).write_text(str(DEAD_PID))
    ingest.main(["--once"] + ONE_REGION)
    assert calls and ingest.store_path().exists() and not lock_file(tmp).exists()


def test_lock_is_released_when_the_round_fails_or_is_interrupted(box, monkeypatch, capsys):
    tmp, calls = box

    def boom(*a, **k):
        raise RuntimeError(f"https://firms.example/{KEY}")
    monkeypatch.setattr(ingest, "run_once", boom)
    ingest.main(["--once"] + ONE_REGION)
    out = capsys.readouterr().out
    assert "ingest failed: RuntimeError, HTTP status none" in out and KEY not in out and "://" not in out
    assert not lock_file(tmp).exists()

    def stop(*a, **k):
        raise KeyboardInterrupt
    monkeypatch.setattr(ingest, "run_once", stop)
    with pytest.raises(KeyboardInterrupt):
        ingest.main(["--once"] + ONE_REGION)
    assert not lock_file(tmp).exists()


# ------------------------------------------------------------------ --loop
class StopLoop(BaseException):
    pass


def test_loop_holds_a_round_lock_and_a_loop_lock(box, monkeypatch, capsys):
    tmp, calls = box
    rounds = []
    real = ingest.run_once

    def spy(*a, **k):
        rounds.append((lock_file(tmp).exists(), lock_file(tmp, "loop.lock").exists()))
        return real(*a, **k)
    monkeypatch.setattr(ingest, "run_once", spy)
    state = {"sleeps": 0}

    def sleep(seconds):
        if seconds != 60:                                       # the short waits between calls inside a round
            return
        state["sleeps"] += 1
        assert not lock_file(tmp).exists()                      # the round lock is free while it waits
        assert lock_file(tmp, "loop.lock").exists()             # the loop lock stays for the life of the loop
        if state["sleeps"] == 2:
            raise StopLoop
    monkeypatch.setattr(ingest.time, "sleep", sleep)
    with pytest.raises(StopLoop):
        ingest.main(["--loop", "--every-minutes", "1"] + ONE_REGION)
    assert rounds == [(True, True), (True, True)]               # two rounds, each under both locks
    assert not lock_file(tmp).exists() and not lock_file(tmp, "loop.lock").exists()


def test_second_loop_exits_at_once(box, capsys):
    tmp, calls = box
    first = ingest.Lock("loop.lock")
    assert first.acquire()
    ingest.main(["--loop", "--every-minutes", "1"] + ONE_REGION)
    out = capsys.readouterr().out
    assert "another ingest is running" in out and not calls
    assert lock_file(tmp, "loop.lock").read_text() == str(os.getpid())
    first.release()


def test_loop_skips_a_round_while_another_ingest_runs_and_goes_on(box, monkeypatch, capsys):
    """An update from the page holds the lock when a loop round is due: the round is skipped, the loop does not die."""
    tmp, calls = box
    holder = ingest.Lock()
    assert holder.acquire()
    state = {"sleeps": 0}

    def sleep(seconds):
        if seconds != 60:
            return
        state["sleeps"] += 1
        if state["sleeps"] == 1:
            holder.release()                                    # the update ends while the loop waits
        if state["sleeps"] == 2:
            raise StopLoop
    monkeypatch.setattr(ingest.time, "sleep", sleep)
    with pytest.raises(StopLoop):
        ingest.main(["--loop", "--every-minutes", "1"] + ONE_REGION)
    out = capsys.readouterr().out
    assert "another ingest is running, this round is skipped" in out
    assert calls and ingest.store_path().exists()               # the next round ran
    assert not lock_file(tmp).exists() and not lock_file(tmp, "loop.lock").exists()


def test_store_rewriting_commands_take_the_lock_too(box, monkeypatch, capsys):
    tmp, calls = box
    holder = ingest.Lock()
    assert holder.acquire()
    monkeypatch.setattr(ingest, "rebuild_score", lambda **k: pytest.fail("ran while locked"))
    monkeypatch.setattr(ingest, "import_live", lambda **k: pytest.fail("ran while locked"))
    ingest.main(["--rebuild-score"])
    ingest.main(["--import-live"])
    assert capsys.readouterr().out.count("another ingest is running") == 2
    holder.release()


# ------------------------------------------------------------------ the Update live data button shares it
def test_update_button_and_loop_protect_each_other(box):
    tmp, calls = box
    holder = ingest.Lock()
    assert holder.acquire()                                     # a loop round is running
    assert updater.is_running() and updater.status()["state"] == "running"
    assert updater.status()["started"] is None                  # started elsewhere, not by this page
    assert updater.start([sys.executable, "-c", "print('x')"]) == "running"      # the button is refused
    holder.release()
    assert not updater.is_running()
    # an update whose process finds the lock busy does nothing and says so
    cmd = [sys.executable, "-c", "print('another ingest is running')"]
    assert updater.start(cmd) == "started"
    for _ in range(40):
        if not updater.is_running():
            break
        real_sleep(0.25)
    st = updater.status()
    assert st["state"] == "finished" and "did nothing" in st["message"]


def test_a_real_second_process_is_refused_by_the_lock(box):
    """The command line of the update, run as a separate process while the lock is held, exits with no work."""
    import subprocess
    tmp, calls = box
    holder = ingest.Lock()
    assert holder.acquire()
    env = dict(os.environ, FIRMS_MAP_KEY=KEY)
    env.pop(ingest.STORE_ENV, None)
    code = ("from pathlib import Path\nfrom src import live, ingest\nlive.LIVE_DIR = Path(r'%s')\n"
            "live.load_key = lambda: 'x'\ningest.main(['--once'])" % str(tmp))
    done = subprocess.run([sys.executable, "-c", code], cwd=str(updater.ROOT), env=env, capture_output=True,
                          text=True, timeout=120)
    holder.release()
    assert done.returncode == 0 and "another ingest is running" in done.stdout
    assert not ingest.store_path().exists() and KEY not in done.stdout + done.stderr

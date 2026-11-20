"""Update live data from the app: python -m src.ingest --once in a background process.

One ingest runs at a time, whoever started it: python -m src.ingest takes the lock file ingest.lock itself (see
src/ingest.py), so an update from the app, the loop script and a run by hand protect each other. Here the file
update.lock only holds the process id of the update this page started, for the moment before that process has taken
the ingest lock. The output of the process goes to a log file in the ingest state folder, and the status shown in
the app is read from the locks and the log, so it survives a restart of the app. Nothing here prints or stores the map key, a URL, a header or a traceback: a failure
is shown as the exception class and the HTTP status only, and only lines of the known safe shapes of the log are
shown.
"""
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone

from . import ingest, live
from .config import ROOT

COMMAND = [sys.executable, "-m", "src.ingest", "--once"]
STARTING_SECONDS = 30                    # a lock without a process id is trusted for this long
WARNING = ("An update takes a while (a long gap can take hours) and uses a lot of memory, about 2.5 GB at its "
           "peak. Close other programs before you start.")
SAFE_LINE = re.compile(r"^(NOAA-\d\d[: ]|\[memory\]|failed calls|ingest failed)")


def state_dir():
    return ingest.state_dir()


def lock_path():
    return state_dir() / "update.lock"


def log_path():
    return state_dir() / "update.log"


def status_path():
    return state_dir() / "update.json"


def ingest_running():
    """True while any ingest (this page's update, the loop or a run by hand) holds the ingest lock."""
    return ingest.Lock().alive()


def key_missing():
    """True when there is no map key. The key itself is never returned."""
    return not live.load_key()


pid_alive = ingest.pid_alive


def _lock_pid():
    try:
        text = lock_path().read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return int(text) if text.isdigit() else None


def is_running():
    """True while an ingest is running: the process of an update started here, or any ingest holding the ingest
    lock. A lock of this page whose process is gone is removed."""
    if ingest_running():
        return True
    p = lock_path()
    if not p.exists():
        return False
    pid = _lock_pid()
    if pid is None:                                              # just created, the process id is not written yet
        try:
            young = (datetime.now().timestamp() - p.stat().st_mtime) < STARTING_SECONDS
        except OSError:
            young = False
        if young:
            return True
    elif pid_alive(pid):
        return True
    try:
        p.unlink()
    except OSError:
        pass
    return False


def _write_status(state, **extra):
    d = state_dir()
    d.mkdir(parents=True, exist_ok=True)
    data = {"state": state, **extra}
    status_path().write_text(json.dumps(data), encoding="utf-8")


def start(cmd=None, cwd=None, env=None):
    """Start an update unless one is running. Returns "started", "running" or "failed" (class name only)."""
    d = state_dir()
    d.mkdir(parents=True, exist_ok=True)
    if ingest_running():
        return "running"
    try:
        fd = os.open(str(lock_path()), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        if is_running():
            return "running"
        try:
            fd = os.open(str(lock_path()), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
        except OSError:
            return "running"
    try:
        _write_status("running", started=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))
        with open(log_path(), "wb") as log:
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            proc = subprocess.Popen(cmd or COMMAND, cwd=str(cwd or ROOT), env=env, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
        lock_path().write_text(str(proc.pid), encoding="utf-8")
    except Exception as e:                                       # only the class is kept
        try:
            lock_path().unlink()
        except OSError:
            pass
        _write_status("failed", message=type(e).__name__)
        return "failed"
    return "started"


def _redact(text):
    key = live.load_key()
    if key:
        text = text.replace(key, "***")
    return text


def parse_log(text):
    """(state, message) of a finished update from its log: finished, or failed with the class and status."""
    text = _redact(text)
    m = re.search(r"^ingest failed: (.+)$", text, re.M)
    if m:
        return "failed", m.group(1)[:120]
    if "Traceback" in text:
        last = [ln for ln in text.splitlines() if ln.strip()][-1]
        cls = re.match(r"^([A-Za-z_][\w.]*)(:|$)", last.strip())
        return "failed", f"{cls.group(1) if cls else 'error'}, HTTP status none"
    if ingest.LOCK_MESSAGE in text and "ingest run: peak" not in text:
        return "finished", "another ingest was already running, so this update did nothing"
    if "No map key" in text:
        return "failed", "no map key found"
    if "ingest run: peak" in text:
        n = re.search(r"^failed calls: (\d+)", text, re.M)
        if n and int(n.group(1)):
            return "finished", f"{n.group(1)} calls failed and are fetched again by the next update"
        return "finished", "all calls succeeded"
    return "failed", "stopped before it finished"


def safe_lines(text, n=8):
    """The last n lines of the log of the known safe shapes (counts and memory), without any URL."""
    lines = [ln.strip()[:200] for ln in _redact(text).splitlines()]
    keep = [ln for ln in lines if SAFE_LINE.match(ln) and "://" not in ln]
    return keep[-n:]


def status():
    """{state: idle, running, finished or failed, message, started, finished, lines}."""
    try:
        meta = json.loads(status_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        meta = {}
    try:
        text = log_path().read_text(encoding="utf-8", errors="replace")
    except OSError:
        text = ""
    out = {"state": "idle", "message": "", "started": meta.get("started"), "finished": None,
           "lines": safe_lines(text)}
    if is_running():
        out["state"] = "running"
        if not lock_path().exists():                  # started elsewhere (the loop or by hand), not by this page
            out["started"] = None
        return out
    if meta.get("state") == "failed" and meta.get("message") and not text:
        out.update(state="failed", message=meta["message"])
        return out
    if not meta and not text:
        return out
    state, message = parse_log(text)
    try:
        out["finished"] = datetime.fromtimestamp(log_path().stat().st_mtime, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    except OSError:
        pass
    out.update(state=state, message=message)
    return out

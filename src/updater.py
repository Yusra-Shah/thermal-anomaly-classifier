"""Update live data from the app: python -m src.ingest --once in a background process.

One update runs at a time. A lock file in the ingest state folder holds the process id, the output of the process
goes to a log file in the same folder, and the status shown in the app is read from the lock and the log, so it
survives a restart of the app. Nothing here prints or stores the map key, a URL, a header or a traceback: a failure
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
    return ingest.store_path().parent / "ingest_state"


def lock_path():
    return state_dir() / "update.lock"


def log_path():
    return state_dir() / "update.log"


def status_path():
    return state_dir() / "update.json"


def key_missing():
    """True when there is no map key. The key itself is never returned."""
    return not live.load_key()


def pid_alive(pid):
    """Whether a process with this id is running. Does not send any signal on Windows."""
    if not pid:
        return False
    if os.name == "nt":
        import ctypes
        k = ctypes.WinDLL("kernel32")
        k.OpenProcess.restype = ctypes.c_void_p
        h = k.OpenProcess(0x1000, False, int(pid))              # query limited information
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = k.GetExitCodeProcess(ctypes.c_void_p(h), ctypes.byref(code))
        k.CloseHandle(ctypes.c_void_p(h))
        return bool(ok) and code.value == 259                   # still active
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def _lock_pid():
    try:
        text = lock_path().read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return int(text) if text.isdigit() else None


def is_running():
    """True while the process of an update is alive. A lock whose process is gone is removed."""
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

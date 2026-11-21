"""Small JSON disk cache for the panel sources (data/cache/<name>), keyed by rounded coordinates and a day or hour.

A cached answer keeps the time it was fetched. A failure can be remembered for a while (a negative entry), so a
service that is down is not asked again at every click.
"""
import json
import os
import time
from datetime import datetime, timezone

from ..config import ROOT

CACHE_ROOT = ROOT / "data" / "cache"
USER_AGENT = "fire-hotspot-classifier/1.0 (https://github.com/Yusra-Shah/thermal-anomaly-classifier)"
TIMEOUT = 8                         # seconds for one request


def now_text():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _path(name, key, root):
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in key)
    return (root or CACHE_ROOT) / name / f"{safe}.json"


def read(name, key, ttl=None, root=None):
    """The cached value, or None when it is missing, unreadable or older than ttl seconds."""
    f = _path(name, key, root)
    try:
        if ttl is not None and time.time() - f.stat().st_mtime > ttl:
            return None
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return None


def write(name, key, value, root=None):
    f = _path(name, key, root)
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_name(f.name + f".{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(value), encoding="utf-8")
        os.replace(tmp, f)
    except Exception:
        pass                                          # a cache that cannot be written is not an error
    finally:
        if tmp.exists():
            tmp.unlink()


def http_get(url, params=None, timeout=TIMEOUT, headers=None):
    """GET with the identifying User-Agent. Raises RuntimeError carrying the exception class or the HTTP status
    only, never the address."""
    import requests
    try:
        r = requests.get(url, params=params, timeout=timeout, headers={"User-Agent": USER_AGENT, **(headers or {})})
    except Exception as e:
        raise RuntimeError(type(e).__name__) from None
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}")
    return r

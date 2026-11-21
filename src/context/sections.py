"""The outside sections of the detail panel for one clicked detection, fetched side by side.

One worker thread per service: land cover, weather, population, place names, buildings and the nearby services. Inside a
worker the requests go one at a time (the nearby services look up the three kinds one after the other, so Nominatim is
asked at most once a second). A part shows as soon as its worker answers, so the slowest service never blocks the others.
A part that has not answered 10 seconds after the click is shown as unavailable, timed out; its thread keeps going in the
background and fills the disk cache, so the next click at the place is fast. A failing worker never stops the others, and
nothing here prints or keeps an address.
"""
import threading
import time

import numpy as np
import pandas as pd

from . import diskcache, land, people, response, weather

SECTION_CAP_S = 10
WORKERS = ("land", "weather", "population", "place", "buildings", "services")
LOCAL_WORKERS = ("spread",)       # read from the own store, started only for a live detection
TIMED_OUT = {"unavailable": "timed out"}
NO_PLACE = {"unavailable": "no place list answer"}


def _f(row, name):
    v = row.get(name)
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) and v > 0 else None


def _cc(row):
    cc = row.get("cc")
    return None if cc is None or pd.isna(cc) else str(cc)


class Fetcher:
    """The sections of one click. Create with start(); read with snapshot() as often as needed.

    snapshot() returns land, weather, exposure (population, place, buildings) and response (numbers, services). A part
    that is still loading is None."""

    def __init__(self, row, hooks=None, cap=SECTION_CAP_S, clock=time.time, with_spread=False):
        self.cap, self.clock = cap, clock
        self.with_spread = with_spread
        self.t0 = clock()
        self.lock = threading.Lock()
        self.done, self.value = {}, {}
        self.services = {}                           # written by the services worker as each kind is made
        self.row = row
        self.stamp = diskcache.now_text()
        hooks = hooks or {}

        def pos():
            return float(row["latitude"]), float(row["longitude"])

        # the arguments of each worker are made inside its own thread, so a field that cannot be read (None, NaN, a
        # wrong type) makes that part unavailable and never the whole panel
        jobs = {
            "land": (hooks.get("land") or land.describe, lambda: (*pos(), _f(row, "scan"), _f(row, "track")), {}),
            "weather": (hooks.get("weather") or weather.at, lambda: (*pos(), pd.Timestamp(row["timestamp"])), {}),
            "population": (hooks.get("population") or people.population, lambda: (_cc(row), *pos()), {}),
            "place": (hooks.get("place") or self._place, lambda: pos(), {}),
            "buildings": (hooks.get("buildings") or people.buildings, lambda: pos(), {}),
            "services": (hooks.get("services") or response.services, lambda: pos(),
                         {} if hooks.get("services") else {"out": self.services}),
        }
        if with_spread:
            jobs["spread"] = (hooks.get("spread") or self._spread, lambda: (*pos(), pd.Timestamp(row["timestamp"]),
                                                                            row.get("cell_id")), {})
        if not hooks.get("population"):
            try:
                people.warm(_cc(row))                 # start the country's population grid download at once
            except Exception:
                pass
        for name, (fn, args, kw) in jobs.items():
            threading.Thread(target=self._run, args=(name, fn, args, kw), daemon=True, name=f"panel-{name}").start()

    @staticmethod
    def _spread(lat, lon, t, cell_id):
        from . import detail
        return detail.spread_from_store(lat, lon, t, cell_id) or {"unavailable": "the store could not be read"}

    @staticmethod
    def _place(lat, lon):
        return people.nearest_place(lat, lon) or NO_PLACE

    def _run(self, name, fn, args, kw):
        try:
            res = fn(*args(), **kw)
        except Exception as e:
            res = {"unavailable": f"could not be made ({type(e).__name__})"}
        with self.lock:
            if name == "services" and isinstance(res, dict) and res is not self.services:
                if "unavailable" in res:             # the whole worker failed: every kind says why
                    res = {label: res for _, label in response.KINDS}
                self.services.update(res)            # a replaced worker returns its services instead of filling them
            self.value[name] = res
            self.done[name] = True

    def elapsed(self):
        return self.clock() - self.t0

    def get(self, name):
        """The part if its worker answered, or its timed out form after the cap, else None (still loading)."""
        with self.lock:
            if self.done.get(name):
                return self.value[name]
            if self.elapsed() > self.cap:
                return TIMED_OUT
        return None

    def _service(self, label):
        with self.lock:
            if label in self.services:
                return self.services[label]
            finished = self.done.get("services")
        if finished or self.elapsed() > self.cap:
            return TIMED_OUT
        return None

    def snapshot(self):
        cc = _cc(self.row)
        extra = {"spread": self.get("spread")} if self.with_spread else {}
        return {**extra,
            "land": self.get("land"), "weather": self.get("weather"),
            "exposure": {k: self.get(k) for k in ("population", "place", "buildings")},
            "response": {"numbers": response.numbers(cc),
                         "services": {label: self._service(label) for _, label in response.KINDS},
                         "source": response.SOURCE, "fetched_at": self.stamp, "note": response.NOTE}}

    def pending(self):
        """True while a worker is still loading and within its time limit."""
        return any(self.get(n) is None for n in WORKERS + (LOCAL_WORKERS if self.with_spread else ()))

    def wait(self, seconds):
        """Wait up to `seconds` for every worker, so fast sources show in the first draw."""
        end = time.monotonic() + seconds             # real time: the clock of the cap can be replaced in tests
        while self.pending() and time.monotonic() < end:
            time.sleep(0.05)
        return self.snapshot()


def start(row, hooks=None, cap=SECTION_CAP_S, with_spread=False):
    return Fetcher(row, hooks, cap, with_spread=with_spread)


def fetch_all(row, hooks=None, cap=SECTION_CAP_S, with_spread=False):
    """Every part, waiting for the workers (at most the cap each). hooks replaces a worker function in tests."""
    f = Fetcher(row, hooks, cap, with_spread=with_spread)
    while f.pending():
        time.sleep(0.05)
    return f.snapshot()

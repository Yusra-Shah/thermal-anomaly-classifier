"""A lookup table from 0.01 degree cells to their land class, so the land cover filter never reads tiles in a page run.

A cell is 0.01 degree by 0.01 degree (about 1.1 km). Its class is the majority ESA WorldCover class over the cell,
read once by src/context/landcover.py (at 10 m, or at an overview level when a batch spans too many tiles) and then kept
in one small parquet file (data/cache/landtable.parquet) that every view, the live store and uploaded files share. A
detection's class is the class of its cell: an array lookup, never a tile read.

The table is filled lazily and only by a background thread: a view asks for the cells it needs (request), the thread
reads them in batches grouped by tile, and the view is drawn at once without the filter until the cells are there. A
cell that could not be read in this process is remembered as not read (255) and is not asked again until the process
restarts; it is never written to the file. Cells over open sea are stored as 0 (no land cover) so they are not read twice.
"""
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ROOT
from . import landcover

CELL_DEG = 0.01
PATH = ROOT / "data" / "cache" / "landtable.parquet"
UNKNOWN = 254                       # never looked up (not stored); 255 is a cell that could not be read
NOT_READ = landcover.NOT_CLASSIFIED
LON_CELLS = 36000
TILE_DEG = 1024 * 3.0 / 36000       # a 10 m WorldCover tile of 1024 pixels, in degrees
BATCH_TILES = 80                    # tiles a batch may span, below the 120 of one read
SAVE_EVERY_S = 20
MAX_REST_CELLS = 60_000             # cells of the rest of a view that are queued behind the drawn ones


def cell_ids(lat, lon):
    """Cell id of each point (int64): row of the 0.01 degree grid times 36000 plus its column."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    ilat = np.clip(np.floor((lat + 90.0) / CELL_DEG + 1e-9), 0, 17999).astype(np.int64)
    ilon = np.floor((lon + 180.0) / CELL_DEG + 1e-9).astype(np.int64) % LON_CELLS
    return ilat * LON_CELLS + ilon


def cell_center(ids):
    ids = np.asarray(ids, np.int64)
    return (ids // LON_CELLS + 0.5) * CELL_DEG - 90.0, (ids % LON_CELLS + 0.5) * CELL_DEG - 180.0


class LandTable:
    def __init__(self, path=None, classify=None):
        self.path = Path(path) if path else PATH
        self.classify = classify                       # None: landcover.classify, looked up when used
        self.lock = threading.RLock()
        self.cells = np.zeros(0, np.int64)             # sorted
        self.codes = np.zeros(0, np.uint8)
        self.version = 0                               # grows with every batch added, so views know to look again
        self.queue = deque()
        self.queued = set()
        self.thread = None
        self.error = None                              # exception class when WorldCover could not be reached
        self.read_count = 0                            # cells read from tiles in this process
        self.in_flight = 0                             # cells of the batch being read now
        self._load()

    # ------------------------------------------------------------ the file
    def _load(self):
        try:
            if self.path.exists():
                df = pd.read_parquet(self.path)
                order = np.argsort(df["cell"].to_numpy(np.int64))
                self.cells = df["cell"].to_numpy(np.int64)[order]
                self.codes = df["code"].to_numpy(np.uint8)[order]
        except Exception:
            self.cells, self.codes = np.zeros(0, np.int64), np.zeros(0, np.uint8)

    def _save(self):
        with self.lock:
            keep = self.codes != NOT_READ
            frame = pd.DataFrame({"cell": self.cells[keep], "code": self.codes[keep]})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        try:
            frame.to_parquet(tmp, index=False)
            tmp.replace(self.path)
        except Exception:
            pass
        finally:
            if tmp.exists():
                tmp.unlink()

    # ------------------------------------------------------------ lookup
    def lookup(self, ids):
        """Class of each cell: the WorldCover class, 0 over open sea, 255 not read, 254 not in the table yet."""
        ids = np.asarray(ids, np.int64)
        with self.lock:
            cells, codes = self.cells, self.codes
        if len(cells) == 0 or len(ids) == 0:
            return np.full(len(ids), UNKNOWN, np.uint8)
        pos = np.clip(np.searchsorted(cells, ids), 0, len(cells) - 1)
        return np.where(cells[pos] == ids, codes[pos], UNKNOWN).astype(np.uint8)

    def lookup_points(self, lat, lon):
        return self.lookup(cell_ids(lat, lon))

    def __len__(self):
        return int(len(self.cells))

    def add(self, ids, codes):
        ids, codes = np.asarray(ids, np.int64), np.asarray(codes, np.uint8)
        if len(ids) == 0:
            return
        with self.lock:
            allc = np.concatenate([self.cells, ids])
            allk = np.concatenate([self.codes, codes])
            order = np.argsort(allc, kind="stable")
            allc, allk = allc[order], allk[order]
            last = np.r_[allc[1:] != allc[:-1], True]            # a later reading of a cell replaces an earlier one
            self.cells, self.codes = allc[last], allk[last]
            self.version += 1

    # ------------------------------------------------------------ the background fill
    def request(self, first, rest=()):
        """Queue cells for the background thread: `first` (the cells the view draws) before `rest`. Cells that are in
        the table, read before in this process or already queued are skipped. Returns the number newly queued."""
        n = 0
        for group in (first, rest):
            ids = np.unique(np.asarray(group, np.int64))
            ids = ids[self.lookup(ids) == UNKNOWN] if len(ids) else ids
            if group is rest and len(ids) > MAX_REST_CELLS:
                ids = ids[np.linspace(0, len(ids) - 1, MAX_REST_CELLS).astype(int)]    # an even spread of the rest
            with self.lock:
                fresh = [int(c) for c in ids if int(c) not in self.queued]
                if group is first:
                    self.queue.extendleft(reversed(fresh))       # the cells of the drawn points go to the front
                else:
                    self.queue.extend(fresh)
                self.queued.update(fresh)
                n += len(fresh)
        if n:
            self._start()
        return n

    def remaining(self):
        """Cells not in the table yet that are queued or being read."""
        with self.lock:
            return len(self.queue) + self.in_flight

    def running(self):
        t = self.thread
        return bool(t is not None and t.is_alive())

    def wait(self, timeout=30.0):
        """Wait until the background thread is idle (for tests and the warm script)."""
        end = time.monotonic() + timeout
        while self.running() and time.monotonic() < end:
            time.sleep(0.02)
        return not self.running()

    def _start(self):
        with self.lock:
            if self.running():
                return
            self.thread = threading.Thread(target=self._run, daemon=True, name="landtable-fill")
            self.thread.start()

    def _take(self):
        """The next batch: cells that share tiles, at most BATCH_TILES of them."""
        with self.lock:
            if not self.queue:
                return []
            head = [self.queue.popleft() for _ in range(min(len(self.queue), 4000))]
        lat, lon = cell_center(np.array(head))
        key = (np.floor(lat / TILE_DEG).astype(np.int64) * 100000 + np.floor(lon / TILE_DEG).astype(np.int64))
        order = np.argsort(key, kind="stable")
        head_sorted = [head[i] for i in order]
        keys = key[order]
        uniq = pd.unique(keys)[:BATCH_TILES]
        take_mask = np.isin(keys, uniq)
        batch = [c for c, m in zip(head_sorted, take_mask) if m]
        back = [c for c, m in zip(head_sorted, take_mask) if not m]
        with self.lock:
            self.queue.extendleft(reversed(back))
        return batch

    def _run(self):
        last_save = time.monotonic()
        dirty = False
        while True:
            batch = self._take()
            if not batch:
                break
            ids = np.array(batch, np.int64)
            self.in_flight = len(batch)
            lat, lon = cell_center(ids)
            km = np.maximum(CELL_DEG * 111.32 * np.cos(np.radians(lat)), 0.05)
            try:
                fn = self.classify or landcover.classify
                res = fn(lon, lat, km, np.full(len(ids), CELL_DEG * 110.574), budget_s=300, keep_tiles=False)
                code = np.asarray(res["code"], np.uint8)
                self.error = None
            except landcover.Unavailable as e:                     # WorldCover cannot be reached: stop, say why
                self.error = str(e) or "Unavailable"
                with self.lock:
                    self.queue.clear()
                    self.queued.clear()
                    self.in_flight = 0
                break
            except Exception as e:
                self.error = type(e).__name__
                code = np.full(len(ids), NOT_READ, np.uint8)       # this batch could not be read, not asked again now
            self.add(ids, code)
            self.in_flight = 0
            self.read_count += int((code != NOT_READ).sum())
            dirty = dirty or bool((code != NOT_READ).any())
            with self.lock:
                self.queued.difference_update(batch)
            if dirty and time.monotonic() - last_save > SAVE_EVERY_S:
                self._save()
                last_save, dirty = time.monotonic(), False
        if dirty:
            self._save()

    def probe(self):
        """(reachable, reason): one small request, kept for ten minutes. Run in a thread by the app."""
        now = time.monotonic()
        with self.lock:
            hit = getattr(self, "_probe", None)
            if hit and now - hit[0] < 600:
                return hit[1]
        ok, why = landcover.reachable()
        with self.lock:
            self._probe = (now, (ok, why))
        return ok, why


_table = [None]
_table_lock = threading.Lock()


def get():
    """The one table of this process, loaded from its file the first time."""
    with _table_lock:
        if _table[0] is None:
            _table[0] = LandTable()
        return _table[0]


def reset(path=None, classify=None):
    """Replace the table of this process (tests use a temporary file and a fake reader)."""
    with _table_lock:
        _table[0] = LandTable(path, classify)
        return _table[0]

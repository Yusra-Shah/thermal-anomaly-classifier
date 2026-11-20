"""Tests for the world live store: boxes, trained flag, resumable backfill, bounded memory, freshness, Rotate and
Satellites markup. Fake feed only, no network.

Run from the project root, one suite per process:  python -m pytest tests/test_ingest_world.py -q
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import earth, ingest, live, mem, serve
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
    kw.setdefault("limiter", fake.fast_limiter())
    return ingest.run_once(KEY, fetch=fetch, models=models, **kw)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ------------------------------------------------------------------ boxes and the trained flag
def overlap_area(a, b):
    w, s = max(a[0], b[0]), max(a[1], b[1])
    e, n = min(a[2], b[2]), min(a[3], b[3])
    return max(e - w, 0) * max(n - s, 0)


def test_trained_flag_only_for_the_eight_regions():
    assert len(live.TRAINED) == 8 and live.TRAINED <= set(live.REGIONS)
    assert [live.is_trained(r) for r in live.REGIONS][:8] == [True] * 8
    assert not any(live.is_trained(r) for r in live.REGIONS if r not in live.TRAINED)
    assert live.region_note("Brazil") == ""
    assert live.region_note("Europe") == "region not in the training data, results unvalidated"


def test_new_boxes_do_not_overlap_anything():
    names = list(live.REGIONS)
    untrained = [r for r in names if r not in live.TRAINED]
    assert len(untrained) >= 12
    for i, a in enumerate(untrained):
        for b in names:
            if a != b:
                assert overlap_area(live.REGIONS[a], live.REGIONS[b]) == 0, (a, b)
    for r in names:
        w, s, e, n = live.REGIONS[r]
        assert -180 <= w < e <= 180 and -90 <= s < n <= 90, r
    # the only overlaps are between trained boxes, which existed before: South Asia with India and Nigeria with DR Congo
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:] if overlap_area(live.REGIONS[a], live.REGIONS[b]) > 0]
    assert pairs == [("South Asia", "India"), ("Nigeria", "DR Congo")]


def test_overlap_is_stored_once(live_dir, models):
    """A detection that lies in the South Asia and the India box belongs to South Asia (the earlier region)."""
    by = fake.masters(6)
    both = {s: m[(m["longitude"] >= 68) & (m["longitude"] <= 76) & (m["latitude"] <= 35)] for s, m in by.items()}
    assert all(len(m) for m in both.values())
    run(fake.fetch_factory(by), models)
    store = ingest.read_store()
    assert not store.duplicated(ingest.KEY_COLS).any()
    shared = store[store["longitude"].between(68, 76) & store["latitude"].between(24, 35)]
    assert len(shared) and (shared["region"] == "South Asia").all()


# ------------------------------------------------------------------ resumable backfill
def test_backfill_resumes_after_an_interruption(live_dir, models):
    by = fake.masters(20)
    full_log = []
    run(fake.fetch_factory(by, full_log), models, path=live_dir / "full" / "store.parquet")
    want = ingest.read_store(live_dir / "full" / "store.parquet")

    calls = {"n": 0}
    base = fake.fetch_factory(by)

    def dies_after_40(key, box, start, days, source):
        calls["n"] += 1
        if calls["n"] > 40:
            raise KeyboardInterrupt
        return base(key, box, start, days, source)

    path = live_dir / "resume" / "store.parquet"
    with pytest.raises(KeyboardInterrupt):
        run(dies_after_40, models, path=path)
    assert not path.exists()                                        # the store is only written when a run finishes
    staged = list((live_dir / "resume" / "ingest_state" / "staged").rglob("*"))
    assert len([p for p in staged if p.is_file()]) >= 40            # the days fetched so far are kept
    log2 = []
    r = run(fake.fetch_factory(by, log2), models, path=path)
    assert len(log2) < len(full_log) and len(log2) > 0              # only what is missing is asked for again
    got = ingest.read_store(path)
    key = ingest.KEY_COLS
    a = want.sort_values(key).reset_index(drop=True)[key + ["p_history", "p_instant"]]
    b = got.sort_values(key).reset_index(drop=True)[key + ["p_history", "p_instant"]]
    assert len(a) == len(b) and r["changed"]
    assert (a[key].to_numpy() == b[key].to_numpy()).all() and np.allclose(a["p_history"], b["p_history"], atol=1e-6)
    assert not list((live_dir / "resume" / "ingest_state" / "staged").rglob("*.parquet"))


def test_after_the_first_pass_only_missing_days_are_fetched(live_dir, models):
    by = fake.masters(10)
    run(fake.fetch_factory(by), models)
    log = []
    r = run(fake.fetch_factory(by, log), models)
    sensors, regions = len(live.LIVE_SOURCES), len(ingest.region_names())
    assert len(log) == sensors * regions                            # one call of the latest three days per group
    assert all(days == ingest.REFETCH_DAYS for _, days, _, _ in log)
    assert not r["changed"]
    log = []
    run(fake.fetch_factory(by, log), models, today=fake.today() + pd.Timedelta(days=2))
    assert all(days <= ingest.REFETCH_DAYS + 2 for _, days, _, _ in log) and len(log) == sensors * regions


def test_failed_calls_stay_missing(live_dir, models):
    by = fake.masters(8)
    base = fake.fetch_factory(by)

    def flaky(key, box, start, days, source):
        if box == live.REGIONS["South Asia"] and source == live.LIVE_SOURCES["NOAA-20"]:
            raise ConnectionError(f"failed for url https://x/{KEY}")
        return base(key, box, start, days, source)

    r = run(flaky, models)
    assert r["failures"] == 1
    assert (ingest.read_store()["sensor"] == "NOAA-21").all()
    log = []
    run(fake.fetch_factory(by, log), models)
    again = [(src, box) for _, _, src, box in log if box == live.REGIONS["South Asia"] and src == live.LIVE_SOURCES["NOAA-20"]]
    assert len(again) == 18                                          # the whole range of the failed group
    assert "NOAA-20" in set(ingest.read_store()["sensor"])


# ------------------------------------------------------------------ bounded memory
def test_ingest_reads_and_scores_one_group_at_a_time(live_dir, models, monkeypatch):
    by = fake.masters(12)
    run(fake.fetch_factory(by), models)
    summary = ingest.store_summary()
    biggest = int(summary["rows"].max())
    total = int(summary["rows"].sum())
    assert len(summary) >= 2 and biggest < total
    scored, reads = [], []
    real_score, real_read = ingest.score_group, ingest.pd.read_parquet

    def spy_score(raw, region, sensor, models, **kw):
        scored.append(len(raw))
        return real_score(raw, region, sensor, models, **kw)

    def spy_read(path, *a, **k):
        reads.append((k.get("columns"), k.get("filters")))
        return real_read(path, *a, **k)
    monkeypatch.setattr(ingest, "score_group", spy_score)
    monkeypatch.setattr(ingest.pd, "read_parquet", spy_read)
    ingest.rebuild_score(models=models, log=lambda *a: None)
    assert len(scored) == len(summary) and max(scored) == biggest < total
    store_reads = [r for r in reads if r[0] is not None and "latitude" in r[0] and "ingest_time" in r[0]]
    assert store_reads and all(f for _, f in store_reads)            # every read of the data is filtered to one group
    # the rows, keys and ingest times are as before and no temporary file stays
    after = ingest.read_store()
    assert len(after) == total and not list(live_dir.glob("*.tmp"))


def test_memory_guard_stops_and_leaves_the_store(live_dir, models, monkeypatch):
    by = fake.masters(8)
    run(fake.fetch_factory(by), models)
    path = ingest.store_path()
    before, rows = digest(path), len(ingest.read_store())
    by2 = fake.masters(10)
    monkeypatch.setattr(ingest, "LIMIT_COMMIT_MB", 1)
    with pytest.raises(ingest.MemoryGuard):
        run(fake.fetch_factory(by2), models, today=fake.today() + pd.Timedelta(days=2))
    assert digest(path) == before and len(ingest.read_store()) == rows
    assert not list(live_dir.glob("*.tmp"))
    monkeypatch.setattr(ingest, "LIMIT_COMMIT_MB", 4000)
    r = run(fake.fetch_factory(by2), models, today=fake.today() + pd.Timedelta(days=2))     # the staged days are used
    assert r["changed"] and len(ingest.read_store()) > rows


def test_peak_commit_is_bounded_on_a_synthetic_store(live_dir, models):
    """A store of many groups: the commit of one run stays near the biggest group, far below the whole store."""
    rng = np.random.default_rng(1)
    t = fake.today()
    frames = []
    for i, region in enumerate(["South Asia", "Brazil", "Nigeria", "Australia", "Europe", "East Africa"]):
        w, s, e, n = live.REGIONS[region]
        n_rows = 20_000
        d = pd.DataFrame({
            "sensor": "NOAA-20" if i % 2 == 0 else "NOAA-21", "region": region,
            "latitude": rng.uniform(s + 0.1, n - 0.1, n_rows), "longitude": rng.uniform(w + 0.1, e - 0.1, n_rows),
            "acq_date": t - pd.to_timedelta(rng.integers(0, 60, n_rows), unit="D"),
            "acq_time": (rng.integers(0, 24, n_rows) * 100 + rng.integers(0, 60, n_rows)).astype(np.int32),
            "bright_ti4": rng.normal(335, 14, n_rows).round(1), "bright_ti5": rng.normal(300, 8, n_rows).round(1),
            "frp": rng.gamma(2.0, 2.0, n_rows).round(2), "scan": rng.uniform(0.4, 0.7, n_rows).round(2),
            "track": rng.uniform(0.4, 0.6, n_rows).round(2), "confidence": rng.choice(["l", "n", "h"], n_rows),
            "daynight": rng.choice(["D", "N"], n_rows)})
        frames.append(d.drop_duplicates(ingest.KEY_COLS))
    before = mem.peak_memory()[1]
    r = ingest.run_once(fetch=None, extra=frames, models=models, log=lambda *a: None, today=t)
    after = mem.peak_memory()[1]
    rows = sum(len(f) for f in frames)
    assert r["changed"] and sum(c["kept"] for c in r["counts"].values()) > 0.9 * rows
    assert after - before < ingest.LIMIT_COMMIT_MB
    summary = ingest.store_summary()
    assert len(summary) == 6 and (ingest.read_recent(since=t - pd.Timedelta(days=1)) is not None)


# ------------------------------------------------------------------ cost, key, freshness
def test_measure_cost_prints_numbers_only(monkeypatch, capsys):
    used = {"n": 100}

    def status(key):
        return used["n"], 5000

    def fetch(key, box, start, days, source):
        used["n"] += 4 * days
        return pd.DataFrame()
    monkeypatch.setattr(live, "key_status", status)
    monkeypatch.setattr(live, "fetch_chunk", fetch)
    out = ingest.measure_cost(KEY)
    text = capsys.readouterr().out
    assert out["cost_1_days"] == 4 and out["cost_5_days"] == 20 and out["limit"] == 5000
    assert KEY not in text and "://" not in text and "measured transactions" in text


def test_key_status_errors_show_only_the_class(monkeypatch):
    import requests

    def boom(url, timeout=None):
        raise ConnectionError(f"failed {url}")
    monkeypatch.setattr(requests, "get", boom)
    with pytest.raises(live.FirmsError) as e:
        live.key_status(KEY)
    assert "ConnectionError" in str(e.value) and KEY not in str(e.value) and "://" not in str(e.value)


def test_freshness_text():
    now = pd.Timestamp("2026-11-15 12:00:00")
    text = live.freshness_text(pd.Timestamp("2026-11-15 11:41:30"), now, {"NOAA-20": 1234, "NOAA-21": 7})
    assert text == ("Newest detection 2026-11-15 11:41 UTC, 18 minutes ago. Rows per sensor: NOAA-20 1,234, NOAA-21 7.") \
        or "2026-11-15 11:41 UTC, 19 minutes ago" in text
    assert "NOAA-20 1,234" in text and "NOAA-21 7" in text
    assert "0 minutes ago" in live.freshness_text(now, now, {})
    assert "NOAA-21 0" in live.freshness_text(now, now, {"NOAA-20": 5})


def test_cost_never_exceeds_the_budget_per_call():
    biggest = max(ingest.call_cost(b, live.MAX_DAY_RANGE) for b in live.REGIONS.values())
    assert biggest <= ingest.BUDGET and ingest.BUDGET < ingest.LIMIT


# ------------------------------------------------------------------ globe page: Rotate and Satellites
def page(untrained=False):
    d = pd.DataFrame({"longitude": [10.0, 20.0], "latitude": [5.0, 6.0], "p_static": [0.1, 0.9],
                      "pred_static": [0, 1], "bright_ti4": [330.0, 340.0], "frp": [1.0, 2.0],
                      "acq_date": pd.to_datetime(["2026-11-15", "2026-11-15"]),
                      "timestamp": pd.to_datetime(["2026-11-15 10:05", "2026-11-15 11:20"]),
                      "cc": ["NG", "NG"], "admin1": ["a", "b"], "region": ["Nigeria", "East Africa" if untrained else "Nigeria"]})
    data, _ = earth.payload(d, d["pred_static"].to_numpy())
    return data, earth.html(data, None, {"sel": None, "provs": None, "prov": None}, "Model prediction", "sig")


def test_rotate_toggle_markup_and_logic():
    _, html = page()
    assert 'id="rot"' in html and 'aria-pressed' in html
    for piece in ("function setSpin", "function stopSpin", "easeTo({ zoom: EASE_ZOOM", "GLOBE_ZOOM", "map.stop()",
                  "requestAnimationFrame(frame)", '"mousedown", "touchstart", "wheel", "pointerdown"',
                  "document.getElementById(\"rot\").onclick = () => setSpin(!spin)"):
        assert piece in html, piece
    # the slow turn: a few degrees a second, applied only at globe zoom and not while easing
    assert "SPIN_DEG_PER_S = 4" in html and "map.getZoom() > GLOBE_ZOOM" in html and "!easing" in html


def test_satellites_toggle_markup_and_orbit_constants():
    data, html = page()
    for name in ("S-NPP", "NOAA-20", "NOAA-21"):
        assert name in html
    assert 'id="sat"' in html and "98.7" in html and "101 * 60" in html and "SPEEDUP = 60" in html
    assert "Illustrative orbits, not real positions" in html
    assert "setSats(satOn)" in html and "satOn = true" in html                      # on by default, independent of Rotate
    assert len(data["x"]) == 2 and earth.MAX_POINTS == 20_000                       # no effect on the point cap
    assert "requestAnimationFrame(frame)" in html and "maplibre-gl.js" in html


def test_satellite_icons_orbit_lines_and_far_side():
    _, html = page()
    assert 'id="satcv"' in html and "function makeSprite" in html                    # one small canvas, drawn sprites
    assert "ICON = 28" in html and "satCtx.rotate(ang)" in html                      # about 28 px, turned to the travel direction
    assert "createRadialGradient" in html                                            # soft glow
    assert "ORBIT_POINTS" in html and "TRAIL_POINTS" in html                         # faint orbit line and fading track
    assert "function nearSide" in html and "if (!ok)" in html                        # hidden on the far side of the globe
    assert "satCtx.strokeText(q.name" in html                                        # thin name label beneath
    assert "sat-dot" not in html and "setData(satData" not in html                   # the dot markers are gone
    assert "sourcedata" in html                                                      # the page still has its other sources


def test_satellite_legend_text():
    _, html = page()
    assert earth.SAT_NOTES == {"S-NPP": "S-NPP (no new data since 2 Nov 2026)", "NOAA-20": "NOAA-20 (active)",
                               "NOAA-21": "NOAA-21 (active)"}
    for text in earth.SAT_NOTES.values():
        assert text in html
    assert "Illustrative orbits, not real positions" in html
    assert "CFG.sat_notes[q.name]" in html


def test_help_lines():
    assert earth.SHOW_HELP == ("Raw detection: a hot pixel reported by the satellite, not classified. Model prediction: "
                               "our classification of each detection as fire, static or uncertain.")
    assert earth.LIVE_NOTE == ("Predicted fire means any vegetation fire, including routine farm and savanna burning. "
                               "It does not mean a reported wildfire event.")
    d = pd.DataFrame({"longitude": [10.0], "latitude": [5.0], "p_static": [0.1], "pred_static": [0],
                      "bright_ti4": [330.0], "frp": [1.0], "acq_date": pd.to_datetime(["2026-11-15"]),
                      "cc": ["NG"], "admin1": ["a"]})
    data, _ = earth.payload(d, d["pred_static"].to_numpy())
    shapes = {"sel": None, "provs": None, "prov": None}
    live_html = earth.html(data, None, shapes, "Model prediction", "s", live_source=True)
    archive_html = earth.html(data, None, shapes, "Model prediction", "s")
    assert earth.LIVE_NOTE in live_html and earth.LIVE_NOTE not in archive_html


def test_selector_legend_satellites_and_help_work_together():
    """The Fires, Static, Both selector with legend counts, the satellite entries, the live note and the
    click footprint code are all in one page."""
    d = pd.DataFrame({"longitude": [10.0, 20.0, 30.0], "latitude": [5.0, 6.0, 7.0], "p_static": [0.1, 0.9, 0.5],
                      "pred_static": [0, 1, 0], "bright_ti4": [330.0, 340.0, 335.0], "frp": [1.0, 2.0, 3.0],
                      "scan": [0.4, 0.5, 0.45], "track": [0.4, 0.5, 0.45],
                      "acq_date": pd.to_datetime(["2026-11-15"] * 3), "cc": ["NG"] * 3, "admin1": ["a", "b", "c"]})
    kinds = d["pred_static"].to_numpy()
    codes = earth.focus_codes("Model prediction", "Fires", True)
    drawn, k = earth.apply_focus(d, kinds, codes)
    data, _ = earth.payload(drawn, k)
    shapes = {"sel": None, "provs": None, "prov": None}
    html = earth.html(data, None, shapes, "Model prediction", "s", codes=codes, kinds=k, live_source=True)
    assert "Predicted fire (2)" in html and "Predicted static" not in html.split("CFG = ")[1].split(";")[0]
    for text in list(earth.SAT_NOTES.values()) + [earth.LIVE_NOTE, "Illustrative orbits, not real positions"]:
        assert text in html
    assert "function drawSats" in html and "function drawSel" in html and "installSats" not in html
    assert 'id="satcv"' in html and 'id="back"' in html
    both = earth.html(data, None, shapes, "Model prediction", "s", codes=earth.focus_codes("Model prediction", "Both"),
                      kinds=kinds, live_source=True)
    assert "Predicted fire (2)" in both and "Predicted static (1)" in both


def test_nearest_place_wording_and_predicted_fire_legend():
    """The place line comes from the nearest named place lookup, so it is not called Country, and the legend
    never says actual fire."""
    _, html = page()
    assert "Nearest place: " in html and "Country: " not in html
    app_source = (Path(__file__).resolve().parents[1] / "app" / "app.py").read_text(encoding="utf-8")
    assert "- Nearest place: " in app_source and "- Country: " not in app_source
    assert "offline list" in app_source                       # the note under the place lines
    assert "actual fire" not in html.lower() and "actual fire" not in app_source.lower()
    assert "Predicted fire" in [e[2] for e in earth.legend("Model prediction")]


def test_sample_file_for_the_upload_demo():
    from pathlib import Path
    from src.data import REQUIRED
    path = Path(__file__).resolve().parents[1] / "samples" / "sample_viirs.csv"
    assert path.exists() and path.stat().st_size < 1_000_000
    df = pd.read_csv(path)
    assert 1500 <= len(df) <= 2500 and set(REQUIRED) <= set(df.columns)
    assert not {"type", "region", "sensor", "label", "is_static"} & set(df.columns)        # no labels
    assert set(df["daynight"]) == {"D", "N"} and df["satellite"].isin(["N20", "N21"]).all()
    boxes = list(live.REGIONS.values())
    inside = [sum(w <= x <= e and s <= y <= n for (w, s, e, n) in boxes) for x, y in zip(df["longitude"], df["latitude"])]
    assert min(inside) >= 1                                                                 # every row is inside a live box
    regions = {next(r for r, (w, s, e, n) in live.REGIONS.items() if w <= x <= e and s <= y <= n)
               for x, y in zip(df["longitude"], df["latitude"])}
    assert len(regions) >= 10                                                               # mixed regions
    # the file goes through the same code as the Classify a new file tab, instant tier and history tier
    from src.data import clean, normalize_columns
    from src.features import add_features
    from src.history import add_history, add_history_v2
    from src.label import add_cell
    new = add_cell(add_features(clean(normalize_columns(df))))
    assert len(new) > 0.95 * len(df)
    new["source_file"] = "sample_viirs.csv"
    new = add_history_v2(add_history(new))
    models = serve.Models()
    for tier in ("instant", "history"):
        out = serve.score_tier(models, tier, new)
        assert len(out["p"]) == len(new) and np.isfinite(out["p"]).all()
    ignore = (Path(__file__).resolve().parents[1] / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert not any(line.strip() in ("samples", "samples/", "samples/*", "*.csv", "*.csv") for line in ignore)


def test_untrained_marking_in_payload_and_page():
    data, html = page(untrained=True)
    assert data["u"] == [0, 1] and data["tm"] == ["10:05", "11:20"]
    assert "region not in the training data, results unvalidated" in html
    data, html = page(untrained=False)
    assert data["u"] == [0, 0] and '"any_untrained":false' in html


def test_no_key_in_any_new_output(live_dir, models, capsys, monkeypatch):
    def boom(key, box, start, days, source):
        raise RuntimeError(f"https://firms.example/{key}/{source}")
    lines = []
    run(boom, models, log=lines.append)
    text = "\n".join(lines) + capsys.readouterr().out
    assert KEY not in text and "://" not in text and "firms.example" not in text
    assert "RuntimeError, HTTP status none" in text

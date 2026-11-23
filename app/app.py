"""Streamlit demo. Run from the project root:  streamlit run app/app.py"""
import json
import sys
import threading
from datetime import timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from streamlit.errors import StreamlitAPIException

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import datainfo, earth, geo, ingest, live, serve, uncertain, updater  # noqa: E402
from src.context import clusters, detail, landcover, landtable, sections  # noqa: E402
from src.config import MODEL_DIR, PROC_DIR, REPORT_DIR, SHOW_ANALYSIS_TABS  # noqa: E402
from src.data import REQUIRED, clean, normalize_columns  # noqa: E402
from src.features import add_features  # noqa: E402
from src.history import add_history, add_history_v2  # noqa: E402
from src.label import add_cell  # noqa: E402

st.set_page_config(page_title="Thermal Anomaly Classifier", layout="wide")

TAG = "headline"
MAX_POINTS = 30_000
MIN_HISTORY_DAYS = 30
INSTANT = "Instant (single reading)"
HISTORY = "With history (past detections)"
OPERATING_POINTS = ["F1 optimal", "precision >= 0.90", "precision >= 0.95"]


@st.cache_resource(max_entries=2)       # the sidebar tier's table and the Earth archive table, kept as they are
def load_scored(name):
    p = PROC_DIR / name
    if not p.exists():
        return None
    return live.add_geo(pd.read_parquet(p), PROC_DIR / "geo_cells.parquet")    # geography once per cell


@st.cache_resource(max_entries=2)       # the sidebar tier's model and the Earth model
def load_model(name):
    """Memory mapped load: the models are saved uncompressed, so the big arrays are not copied."""
    p = MODEL_DIR / name
    return joblib.load(p, mmap_mode="r") if p.exists() else None


def read_json(name):
    p = REPORT_DIR / name
    return json.loads(p.read_text()) if p.exists() else None


def read_csv(name, **kw):
    p = REPORT_DIR / name
    return pd.read_csv(p, **kw) if p.exists() else None


ALL_CONT, ALL_COUNTRIES, ALL_PROVINCES, ALL_CITIES = "World", "All countries", "All provinces", "All cities"
CITY_OPTIONS = 200


FILTER_DEFAULTS = {"continent": ALL_CONT, "country": ALL_COUNTRIES, "province": ALL_PROVINCES,
                   "city": ALL_CITIES}


def _reset_below(keys):
    """on_change callback: a changed filter sends the filters below it back to their All default."""
    for k in keys:
        st.session_state[k] = FILTER_DEFAULTS[k.rsplit("_", 1)[1]]


def _select(col, label, options, key, below=()):
    if st.session_state.get(key, options[0]) not in options:
        st.session_state.pop(key, None)             # stale value, for example after a date change
    return col.selectbox(label, options, key=key, on_change=_reset_below, args=(list(below),))


def geo_filters(df, prefix):
    """Continent, country, province and nearest city filters, each narrowing the next.

    Changing a filter resets the ones below it. Returns the filtered frame, the country code
    (or None), whether a province is picked and whether a city is picked.
    """
    k = {n: f"{prefix}_{n}" for n in ("continent", "country", "province", "city")}
    c = st.columns(4)
    cont = _select(c[0], "Continent", [ALL_CONT] + sorted(df["continent"].dropna().unique()),
                   k["continent"], (k["country"], k["province"], k["city"]))
    if cont != ALL_CONT:
        df = df[df["continent"] == cont]
    by_name = {geo.country_name(x): x for x in df["cc"].dropna().unique()}
    name = _select(c[1], "Country", [ALL_COUNTRIES] + sorted(by_name), k["country"],
                   (k["province"], k["city"]))
    cc = None
    if name != ALL_COUNTRIES:
        cc = by_name[name]
        df = df[df["cc"] == cc]
    provinces = sorted(p for p in df["admin1"].dropna().unique() if p)
    prov = _select(c[2], "Province or state", [ALL_PROVINCES] + provinces, k["province"], (k["city"],))
    if prov != ALL_PROVINCES:
        df = df[df["admin1"] == prov]
    top = df["city"].value_counts()
    top = top[top > 0].head(CITY_OPTIONS)
    labels = {f"{n} ({m:,})": n for n, m in top.items()}
    pick = _select(c[3], "Nearest city", [ALL_CITIES] + list(labels), k["city"])
    if pick != ALL_CITIES:
        df = df[df["city"] == labels[pick]]
    return df, cc, prov != ALL_PROVINCES, pick != ALL_CITIES


def score_upload(up, tier, bundle, thr):
    """Score an uploaded FIRMS CSV (archive or near real time format)."""
    raw = normalize_columns(pd.read_csv(up))
    missing = [c for c in REQUIRED if c not in raw.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    new = add_cell(add_features(clean(raw)))
    span = int((new["acq_date"].max() - new["acq_date"].min()).days) + 1
    if tier == HISTORY:
        new["source_file"] = getattr(up, "name", "upload")
        new = add_history_v2(add_history(new))          # same functions the training used
    new["p_static"] = bundle["model"].predict_proba(new[bundle["features"]])[:, 1]
    new["pred_static"] = (new["p_static"] >= thr).astype(int)
    new["cell_days"] = new.groupby("cell_id")["acq_date"].transform("nunique")
    return new, span


# ---------------------------------------------------------------- earth (live source and archive)


@st.cache_data(show_spinner="Computing features and scores (the first run for a region is slower)")
def live_scored(region, stamp):
    """Features, history, both model scores and geography for the stored region. Cached by file stamp.

    The instant model is the large one, loaded here and released as soon as the scores are computed.
    """
    raw, fetched = live.load_snapshot(region)
    hist = joblib.load(MODEL_DIR / "best_model_history.joblib", mmap_mode="r")      # small
    inst = joblib.load(MODEL_DIR / "best_model_headline.joblib", mmap_mode="r")
    df = live.build_scored(region, hist, inst, raw)
    meta = {"bands": {"history": uncertain.bands_of(hist, "best_model_history_band.json"),
                      "instant": uncertain.bands_of(inst, "best_model_headline_band.json")},
            "inst_threshold": float(inst["threshold"]), "inst_features": list(inst["features"]),
            "thresholds": hist["thresholds"], "hist_features": list(hist["features"]),
            "default_point": hist.get("default_operating_point")}
    del inst
    return df, fetched, meta


ALL_REGIONS = "All regions"
LIVE_HOURS = [6, 24, 72]
LIVE_REFRESH_SECONDS = 600
STORE_COLS = ["sensor", "region", "latitude", "longitude", "acq_date", "acq_time", "timestamp", "cell_id",
              "bright_ti4", "frp", "scan", "track", "p_instant", "p_history"]


@st.cache_data(ttl=LIVE_REFRESH_SECONDS, show_spinner="Reading the live store")
def live_store_scored(region, hours, stamp):
    """Detections of the last `hours` hours of one region (or all regions) from the incremental store
    (python -m src.ingest), with geography and the column names of the snapshot path, plus the last ingest time
    and the bands. Only the needed columns and the needed time range are read. Cached by file stamp and for
    ten minutes, so the view follows the store."""
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    df = ingest.read_recent(since=now - pd.Timedelta(hours=hours), columns=STORE_COLS,
                            region=None if region == ALL_REGIONS else region)
    _, last = ingest.store_info()
    if df is None or df.empty:
        return df, last, None
    df = df.rename(columns={"p_instant": "p_static_instant", "p_history": "p_static_history"})
    df = live.add_geo(df)
    hist = joblib.load(MODEL_DIR / "best_model_history.joblib", mmap_mode="r")
    inst = joblib.load(MODEL_DIR / "best_model_headline.joblib", mmap_mode="r")
    meta = {"bands": {"history": uncertain.bands_of(hist, "best_model_history_band.json"),
                      "instant": uncertain.bands_of(inst, "best_model_headline_band.json")},
            "inst_threshold": float(inst["threshold"])}
    return df, last, meta


STALE_HOURS = 6


@st.cache_data(ttl=300, show_spinner=False)
def newest_detection(region, stamp):
    """Time of the newest detection of the store (or of one region), cached for five minutes."""
    return ingest.latest_detection(region=None if region == ALL_REGIONS else region)


def store_source(region, live_tier, thr, col):
    """Live source from the incremental store: last 6, 24 or 72 hours, freshness, a banner when it is stale."""
    path = ingest.store_path()
    hours = st.radio("Time window", LIVE_HOURS, index=1, horizontal=True, key="earth_hours",
                     format_func=lambda h: f"last {h} hours")
    stamp = path.stat().st_mtime_ns
    try:
        df, last, meta = live_store_scored(region, hours, stamp)
    except Exception as e:
        st.error(f"Could not read the live store: {type(e).__name__}")
        return None
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    newest = newest_detection(region, stamp)
    st.caption(f"Live store, last ingest {pd.Timestamp(last):%Y-%m-%d %H:%M} UTC. Press Update live data to fetch "
               f"newer detections. The view reads the store again every {LIVE_REFRESH_SECONDS // 60} minutes. "
               "Live results are unvalidated.")
    stale = live.stale_text(newest, now, STALE_HOURS)
    if stale:
        st.error(stale)
    if df is None or df.empty:
        st.info(live.empty_text(hours, newest, now))
        return None
    if thr is None:
        thr = meta["inst_threshold"]
        st.caption(f"Instant model, saved threshold {thr:.3f}")
    if region != ALL_REGIONS and not live.is_trained(region):
        st.warning(live.region_note(region).capitalize() + ".")
    elif region == ALL_REGIONS:
        n_un = int((~df["region"].isin(live.TRAINED)).sum())
        if n_un:
            st.caption(f"{n_un:,} of these detections are in regions not in the training data (results "
                       "unvalidated). They are marked in the legend and in the popups.")
    latest = df["timestamp"].max()
    per = df["sensor"].value_counts().to_dict()
    st.caption(live.freshness_text(latest, now, per, live.LIVE_SOURCES))
    view = df.assign(p_static=df[col], pred_static=(df[col] >= thr).astype(int))
    return view, region, False, meta["bands"]["history" if live_tier == HISTORY else "instant"]


def instant_note():
    """One line on the instant tier, with the numbers of the day and night study when its report exists."""
    text = "The instant tier mostly separates night detections and is weak on its own, prefer the history tier."
    path = REPORT_DIR / "daynight_vs_baseline.csv"
    if path.exists():
        d = pd.read_csv(path)
        d = d[(d["split"] == "grouped seed 42") & d["tier"].str.startswith("instant")]
        if len(d):
            r = d.iloc[-1]
            text += (f" On held out cells its static PR-AUC is {r['pr_auc']:.2f} against {r['baseline_pr_auc']:.2f} "
                     f"for calling every night detection static, and its F1 is {r['f1']:.2f} against "
                     f"{r['baseline_f1']:.2f}.")
    return text


def model_controls(hist):
    """The Earth model tier (default With history) and its operating point, for the live and the archive source.

    Returns (the four columns, the tier, the threshold or None for the instant tier, the live score column)."""
    cols = st.columns([3, 3, 3, 2])
    tier_pick = cols[1].radio("Model tier", [HISTORY, INSTANT], key="earth_model")
    if tier_pick == HISTORY:
        names = [n for n in OPERATING_POINTS if hist["thresholds"].get(n) is not None]
        point = cols[2].selectbox("Operating point", names,
                                  index=names.index(hist.get("default_operating_point", names[0])), key="earth_point")
        return cols, tier_pick, float(hist["thresholds"][point]), "p_static_history"
    st.caption(instant_note())
    return cols, tier_pick, None, "p_static_instant"          # the saved threshold is read with the scores


def live_source(cols, live_tier, thr, col):
    """Region, store or snapshot, days shown and freshness of the live source.

    Returns (frame with p_static and pred_static, region, demo, band) or None when nothing can be shown."""
    key = live.load_key()
    st.info("Near real time data from NOAA-20 and NOAA-21, about 3 hours behind the satellite. There is no "
            "ground truth on live data, so only predictions are shown. History features are built inside each "
            "satellite and need up to 90 days of data, so the first fetch is slower.")
    st.caption("The models were trained on Suomi NPP and validated on NOAA-20 and NOAA-21 archive data. "
               "Near real time data has not been validated against the archive.")
    has_store = ingest.store_path().exists()
    region = cols[0].selectbox("Region loaded", ([ALL_REGIONS] if has_store else []) + list(live.REGIONS),
                               key="earth_region",
                               format_func=lambda n: n if n == ALL_REGIONS or live.is_trained(n) else f"{n} (not in training)")
    if has_store:
        return store_source(region, live_tier, thr, col)
    if key is None:
        cols[3].caption("Offline: no map key")
        offline = True
        st.warning("No FIRMS map key found. Set FIRMS_MAP_KEY in a .env file in the project root or in "
                   "the environment to fetch live data. Showing the saved snapshot if there is one.")
    else:
        offline = cols[3].checkbox("Offline (saved snapshot)", value=False, key="earth_offline")
    if st.button("Refresh data", disabled=offline, key="earth_refresh"):
        bar = st.progress(0.0, text="Fetching from FIRMS")
        try:
            r = live.refresh(region, key, progress=lambda f: bar.progress(f, text="Fetching from FIRMS"))
            st.success(f"{r['requests']} requests, {r['new_rows']:,} detections stored.")
        except live.FirmsError as e:
            st.error(f"Refresh failed: {e}")
        bar.empty()

    snap = live.snapshot_path(region)
    if not snap.exists():
        st.info("No data for this region yet. Add a map key and press Refresh data.")
        return None
    try:
        df, fetched, meta = live_scored(region, snap.stat().st_mtime_ns)
    except Exception as e:
        st.error(f"Could not process the stored data: {type(e).__name__}")
        return None
    if thr is None:
        thr = meta["inst_threshold"]
        st.caption(f"Instant model, saved threshold {thr:.3f}")
    demo = live.snapshot_is_demo(region)
    if demo:
        st.error("Demo snapshot: historical 2025 data shifted to look recent, for testing only. "
                 "This is not current data.")
    if region not in live.TRAINED:
        st.warning("This region was not part of the training data. Results here are unvalidated.")

    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    latest = df["timestamp"].max()
    age_h = (now - latest).total_seconds() / 3600
    days = st.slider("Days shown", 1, live.KEEP_DAYS, 7, key="earth_days",
                     help="Features always use the full stored history.")
    view = df[df["acq_date"] > df["acq_date"].max() - pd.Timedelta(days=days)]
    view = view.assign(p_static=view[col], pred_static=(view[col] >= thr).astype(int))
    st.caption(f"Latest detection {latest:%Y-%m-%d %H:%M} UTC, data age {live.age_text(age_h * 60)}."
               + (f" Last fetch {fetched:%Y-%m-%d %H:%M} UTC." if fetched is not None else ""))
    if not demo and "satellite_name" in df.columns:
        per = df["satellite_name"].value_counts()
        rows = ", ".join(f"{n} {int(per.get(n, 0)):,}" for n in live.LIVE_SOURCES)
        st.caption(f"Acquisition dates {df['acq_date'].min():%Y-%m-%d} to {df['acq_date'].max():%Y-%m-%d}. "
                   f"Rows per satellite: {rows}.")
    if age_h > STALE_HOURS and not demo:
        st.error(live.stale_text(latest, now, STALE_HOURS))
    return view, region, demo, meta["bands"]["history" if live_tier == HISTORY else "instant"]


# ---------------------------------------------------------------- sidebar
st.title("Thermal Anomaly Classifier")
st.caption("NASA FIRMS VIIRS detections, classified as likely fire or likely static heat source.")

tier = st.sidebar.radio("Model tier", [INSTANT, HISTORY])
# Only the small model is loaded here. The archive tables are read when a tab or the archive source needs them, and
# only the open tab is run (st.tabs with a key and on_change="rerun" tells which one is open), so the Earth, which
# is the front page, does not wait for tables that other tabs use.
if tier == INSTANT:
    bundle = load_model(f"best_model_{TAG}.joblib")
    prob_col = "p_static"
else:
    bundle = load_model("best_model_history.joblib")
    prob_col = "p_static_history"

if bundle is None:
    st.warning("No trained model found for this tier. Run `python -m src.train --tag headline` "
               "(instant) or `python -m src.final_history` (with history) first.")
    st.stop()

if tier == INSTANT:
    saved_thr = float(bundle["threshold"])
    thr = st.sidebar.slider("Decision threshold (p static)", 0.05, 0.95, saved_thr, 0.01,
                            help=f"Saved threshold: {saved_thr:.3f}")
else:
    names = [n for n in OPERATING_POINTS if bundle["thresholds"].get(n) is not None]
    default = bundle.get("default_operating_point", names[0])
    point = st.sidebar.selectbox("Operating point", names, index=names.index(default))
    thr = float(bundle["thresholds"][point])
    held = (read_json("history_final_metrics.json") or {}).get("operating_points", {}).get(point)
    st.sidebar.caption(f"Threshold {thr:.3f}.")
    if held and held.get("precision") is not None:
        st.sidebar.caption(f"Held out 20 percent: precision {held['precision']:.3f}, "
                           f"recall {held['recall']:.3f}.")
    st.info("The history tier needs past detections of the same place and works best with at least "
            f"{MIN_HISTORY_DAYS} days of data. It predicts NASA's repeat based static flag.")

# The analysis tabs "Data and features", "Why a spot is static", "Model metrics" and "Results" exist only while
# SHOW_ANALYSIS_TABS is on (src/config.py, or the environment variable of the same name); off, they are not created and
# their code does not run, and the app shows the Earth and Classify tabs. On, the order is Earth, Data and features,
# Why a spot is static, Classify a new file, Model metrics, Results.
DATA_TAB, SPOT_TAB, METRICS_TAB, RESULTS_TAB = "Data and features", "Why a spot is static", "Model metrics", "Results"
TAB_NAMES = (["Earth"] + ([DATA_TAB, SPOT_TAB] if SHOW_ANALYSIS_TABS else []) + ["Classify a new file"]
             + ([METRICS_TAB, RESULTS_TAB] if SHOW_ANALYSIS_TABS else []))
TABS = dict(zip(TAB_NAMES, st.tabs(TAB_NAMES, key="main_tabs", on_change="rerun")))
tab_earth, tab_live = TABS["Earth"], TABS["Classify a new file"]
tab_data, tab_spot, tab_metrics, tab_results = (TABS.get(n) for n in (DATA_TAB, SPOT_TAB, METRICS_TAB, RESULTS_TAB))

def classify_tab():
    """Upload a FIRMS CSV and see it on the same globe as the Earth tab (sidebar tier, warnings, counters)."""
    st.caption("This tab uses the sidebar tier, Instant by default: an uploaded file has no past detections of "
               f"its own places, which the history tier needs. With history works only for a file that covers "
               f"{MIN_HISTORY_DAYS} days or more, and a shorter file gets a warning.")
    if tier == INSTANT:
        st.write("Upload a FIRMS VIIRS CSV the model has never seen. Each detection is scored "
                 "from its own readings only.")
    else:
        st.write("Upload a FIRMS VIIRS CSV. History features are computed inside the uploaded file "
                 "from detections on earlier days, so the file must cover enough days.")
    up = st.file_uploader("FIRMS CSV", type=["csv"])
    if up is not None:
        try:
            new, span = score_upload(up, tier, bundle, thr)
            if tier == HISTORY and span < MIN_HISTORY_DAYS:
                st.warning(f"This file spans {span} days, fewer than {MIN_HISTORY_DAYS}. History "
                           "features will be weak and results unreliable.")
            st.write(f"{len(new):,} detections, {int(new['pred_static'].sum()):,} predicted static.")
            new = live.add_geo(new)
            bands = uncertain.bands_of(bundle, "best_model_headline_band.json" if tier == INSTANT
                                       else "best_model_history_band.json")
            render_globe("upload", new, bands, kind="upload", region=None, demo=False, prefix="earth_u", suffix="u",
                         truth=False, tier_name=tier, cap=MAX_POINTS, in_fragment=False,
                         sig_extra=f"{getattr(up, 'name', '')}|{getattr(up, 'size', '')}")
        except Exception as e:  # show a readable message in the demo
            st.error(f"Could not process this file: {type(e).__name__}")


def world_section():
    """All regions: the files, the five grouped splits, the temporal split and every region held out."""
    wh, wi = read_csv("world_history.csv"), read_csv("world_instant.csv")
    if wh is None:
        return
    st.subheader("All regions, January to June 2025")
    inv = read_csv("world_inventory.csv")
    if inv is not None:
        st.dataframe(inv[["region", "country", "rows_kept", "static_share"]].round(4), width="stretch", hide_index=True)
    w = pd.concat([wh, wi] if wi is not None else [wh], ignore_index=True)
    w["tier"] = w["tier"].str.replace(r"^rule .*$", "rule (distinct days in prior 30)", regex=True)
    cols = ["pr_auc", "f1opt_precision", "f1opt_recall", "f1opt_f1"]
    names = ["PR-AUC", "Precision", "Recall", "F1"]
    seeds = w[w["split"].str.startswith("grouped")]
    g = seeds.groupby("tier", sort=False)[cols]
    table = g.mean().round(3).astype(str) + " (" + g.std().round(3).astype(str) + ")"
    st.write("Five grouped splits by grid cell, mean (std), all regions together.")
    st.dataframe(table.rename(columns=dict(zip(cols, names))))
    st.write("Temporal split (train before 2025-05-01).")
    t = w[w["split"] == "temporal"].set_index("tier")[["n", "static_share", "majority_pr_auc"] + cols]
    st.dataframe(t.round(3).rename(columns=dict(zip(cols, names))))
    reg = read_csv("world_regions.csv")
    if reg is not None:
        st.write("Leave one region out: train on all other regions, test on the held out one.")
        st.dataframe(reg.round(3), width="stretch", hide_index=True)
    st.caption("Same code and settings as before: instant and history XGBoost, 100 trees, thresholds from out of "
               "fold training scores. January to June misses the main fire season in places such as Brazil, so "
               "the static share and the results of a region describe these months only. NASA's flag is the "
               "label, not independent truth. DR Congo has almost no static detections, so its scores are noisy.")


if SHOW_ANALYSIS_TABS:
    with tab_spot:
        if tab_spot.open:
            scored = load_scored(f"scored_{TAG}.parquet" if tier == INSTANT else "scored_history.parquet")
            if scored is None:
                st.warning("No scored archive found for this tier. Run `python -m src.train --tag headline` (instant) "
                           "or `python -m src.final_history` (with history) first.")
            else:
                scored = scored.assign(p_static=scored[prob_col], pred_static=(scored[prob_col] >= thr).astype(int))
                st.write("Pick a location that was flagged repeatedly and see it appear day after day.")
                cells = (scored.groupby("cell_id")["acq_date"].nunique().sort_values(ascending=False).head(50))
                if len(cells):
                    cid = st.selectbox("Most repeated locations (cell id, days seen)",
                                       cells.index, format_func=lambda c: f"{c}  ({cells[c]} days)")
                    sub = scored[scored["cell_id"] == cid].sort_values("timestamp")
                    st.dataframe(sub[["timestamp", "bright_ti4", "frp", "p_static", "pred_static"]]
                                 .reset_index(drop=True), width="stretch")
                    per_day = sub.groupby(sub["acq_date"].dt.date).size()
                    st.bar_chart(per_day)


if SHOW_ANALYSIS_TABS:
    with tab_metrics:
        if tab_metrics.open:
            world_section()
            mpath = REPORT_DIR / f"metrics_{TAG}.json"
            if mpath.exists():
                m = json.loads(mpath.read_text())
                st.write(f"Best instant model: **{m['best_model']}**. Window {m['date_min']} to {m['date_max']}, "
                         f"{m['rows']:,} detections, {m['static_share']:.1%} static.")
                st.dataframe(pd.read_csv(REPORT_DIR / f"model_comparison_{TAG}.csv", index_col=0))
                a, b = st.columns(2)
                if (REPORT_DIR / f"confusion_matrix_{TAG}.png").exists():
                    a.image(str(REPORT_DIR / f"confusion_matrix_{TAG}.png"))
                if (REPORT_DIR / f"feature_importance_{TAG}.png").exists():
                    b.image(str(REPORT_DIR / f"feature_importance_{TAG}.png"))
                st.caption("Scores are on locations the model never saw during training. The label is NASA's "
                           "own type flag. The confusion matrix uses the saved threshold.")
            upath = REPORT_DIR / "uncertain_eval.csv"
            if upath.exists():
                st.subheader("Uncertain band")
                u = pd.read_csv(upath)
                cols = ["tier", "split", "definition", "t_low", "t_high", "coverage", "decided_accuracy",
                        "static_precision_decided", "static_recall_decided", "fire_npv_decided",
                        "errors_in_uncertain_share"]
                st.dataframe(u[~u["definition"].str.startswith("precision")][cols].round(3), width="stretch",
                             hide_index=True)
                if (REPORT_DIR / "uncertain_curve.png").exists():
                    st.image(str(REPORT_DIR / "uncertain_curve.png"))
                st.caption("Scores are calibrated. Static at or above t_high, fire at or below t_low, uncertain in "
                           "between, with the highest coverage whose accuracy on decided out of fold training points is "
                           "0.96 (default) or 0.98 (strict). Precision and recall are on the decided points, "
                           "errors_in_uncertain_share is the share of the forced binary errors that fall in the uncertain "
                           "band. NASA's flag is the label, not independent truth.")
                with st.expander("Earlier precision based definition (negative result)"):
                    st.dataframe(u[u["definition"].str.startswith("precision")][cols].round(3), width="stretch",
                                 hide_index=True)
                    if (REPORT_DIR / "uncertain_curve_precision_based.png").exists():
                        st.image(str(REPORT_DIR / "uncertain_curve_precision_based.png"))
                    st.caption("Static precision at least 0.90 and fire side precision at least 0.95. Fire is the "
                               "majority class, so the fire side target is almost always met, the two ranges overlap "
                               "for the history tier and its band is empty. Kept as a documented negative result.")
            parts = []
            for sat_file, sat_name in (("noaa20_transfer.csv", "NOAA-20"), ("noaa21_transfer.csv", "NOAA-21")):
                if (REPORT_DIR / sat_file).exists():
                    parts.append(pd.read_csv(REPORT_DIR / sat_file).assign(satellite=sat_name))
            if parts:
                st.subheader("NOAA-20 and NOAA-21 transfer")
                t = pd.concat(parts, ignore_index=True)
                t = t[~t["test_set"].str.contains("reference") | (t["satellite"] == parts[0]["satellite"].iloc[0])]
                cols = ["satellite", "model", "test_set", "region", "n", "static_share", "pr_auc", "majority_pr_auc",
                        "precision", "recall", "f1", "brier_calibrated", "ece_calibrated", "uncertain_share_0.96",
                        "uncertain_share_0.98"]
                cols = [c for c in cols if c in t]
                st.dataframe(t[t["region"] == "all"][cols].round(3), width="stretch", hide_index=True)
                with st.expander("Per region"):
                    st.dataframe(t[t["region"] != "all"][cols].round(3), width="stretch", hide_index=True)
                st.caption("Trained on Suomi NPP cells, tested on NOAA-20 and NOAA-21 archive data with the Suomi NPP "
                           "thresholds, calibrators and bands unchanged. Cell disjoint means cells that were not Suomi NPP "
                           "training cells. Not cell disjoint is easier, those places were seen in training. The satellites "
                           "observe the same places within a short time of each other and the label is NASA's repeat based "
                           "flag, so this is a transfer between near twins, not independent truth. The Suomi NPP reference "
                           "rows are shown once.")

            strat = REPORT_DIR / "daynight_stratified.csv"
            if strat.exists():
                st.subheader("Day and night")
                sdf = pd.read_csv(strat)
                base = pd.read_csv(REPORT_DIR / "daynight_vs_baseline.csv") if (REPORT_DIR / "daynight_vs_baseline.csv").exists() else None
                if base is not None:
                    st.caption("Static PR-AUC and F1 over all detections against a baseline that calls every night "
                               "detection static.")
                    st.dataframe(base[["split", "tier", "pr_auc", "baseline_pr_auc", "pr_auc_above_baseline", "f1",
                                       "baseline_f1", "f1_above_baseline"]].round(3), width="stretch", hide_index=True)
                with st.expander("Stratified by day and night, every tier"):
                    cols = ["split", "tier", "stratum", "n", "static_share", "majority_pr_auc", "pr_auc", "precision",
                            "recall", "f1"]
                    st.dataframe(sdf[cols].round(3), width="stretch", hide_index=True)
                    st.caption("Day and night detections are scored on their own, each with its own majority baseline "
                               "(the static share of that stratum). The night baseline predicts static for every night "
                               "detection. NASA's flag is the label, not independent truth.")
            abl = REPORT_DIR / "daynight_ablation.csv"
            if abl.exists():
                st.subheader("Instant tier ablation")
                adf = pd.read_csv(abl)
                st.dataframe(adf[adf["stratum"] == "all"][["variant", "n_features", "pr_auc", "precision", "recall", "f1",
                                                            "threshold"]].round(3), width="stretch", hide_index=True)
                if (REPORT_DIR / "daynight_permutation.csv").exists():
                    with st.expander("Permutation importance on the held out split"):
                        st.dataframe(pd.read_csv(REPORT_DIR / "daynight_permutation.csv").round(4), width="stretch",
                                     hide_index=True)
                st.caption("XGBoost with 100 trees on the grouped split, retrained without one group of features at a "
                           "time, thresholds from out of fold training scores only.")


if SHOW_ANALYSIS_TABS:
    with tab_results:
        if tab_results.open:
            st.subheader("Rule, instant and history results")
            shown = False

            ev = read_csv("history2_eval_a.csv")
            if ev is not None:
                shown = True
                subset = st.selectbox("Test rows", list(ev["subset"].unique()))
                d = ev[(ev["subset"] == subset)
                       & (ev["setup"].str.startswith(("Rule", "B ", "D ")))
                       & (ev["model"].isin(["Random Forest", "XGBoost", "XGBoost tuned"])
                          | (ev["setup"] == "Rule"))]
                d = d.assign(model=d["model"].where(d["setup"] != "Rule", "distinct_days_prior_30 >= k"))
                cols = ["pr_auc_static", "precision_static", "recall_static", "f1_static"]
                g = d.groupby(["setup", "model"], sort=False)[cols]
                table = g.mean().round(3).astype(str) + " (" + g.std().round(3).astype(str) + ")"
                table.index = [f"{setup}: {model}" for setup, model in table.index]
                st.write("Five grouped splits, mean (std). Rule, B (instant + history v1) and D "
                         "(instant + history v1 + v2).")
                st.dataframe(table.rename(columns=dict(zip(cols, ["PR-AUC", "Precision", "Recall", "F1"]))))
                rule_f1 = d[d["setup"] == "Rule"]["f1_static"].mean()
                d_f1 = d[(d["setup"].str.startswith("D ")) & (d["model"] == "XGBoost")]["f1_static"].mean()
                k1, k2 = st.columns(2)
                k1.metric("Rule F1 (static)", f"{rule_f1:.3f}")
                k2.metric("D XGBoost F1 (static)", f"{d_f1:.3f}")

            ops = read_csv("history2_operating_points.csv")
            if ops is not None:
                shown = True
                st.write(f"Operating points, {ops['setup'].iloc[0]}, {ops['model'].iloc[0]}. Thresholds from "
                         "training predictions only, precision and recall on the test part.")
                five = ops[ops["split"].str.startswith("seed")]
                st.dataframe(five.groupby("operating point", sort=False)[["threshold", "precision", "recall"]]
                             .mean().round(3).rename(index=lambda s: f"{s} (mean of 5 splits)"))
                st.dataframe(ops[ops["split"] == "temporal"].drop(columns=["setup", "model"])
                             .reset_index(drop=True))
            if (REPORT_DIR / "pr_curve_history2.png").exists():
                st.image(str(REPORT_DIR / "pr_curve_history2.png"), width=480)

            fin = read_json("history_final_metrics.json")
            if fin is not None:
                shown = True
                st.write(f"Saved history model ({fin['model']}, setup {fin['setup']}), held out 20 percent "
                         f"of locations, split seed {fin['split_seed']}.")
                st.dataframe(pd.DataFrame(fin["operating_points"]).T)

            reg = read_csv("region_eval_type_region.csv")
            if reg is not None:
                shown = True
                st.write("Instant tier, leave one region out (train on two regions, test on the third).")
                st.dataframe(reg.round(3))

            inst = read_json(f"metrics_{TAG}.json")
            if inst is not None:
                shown = True
                st.write("Instant tier, held out test split.")
                res = pd.DataFrame(inst["results"]).T[["precision_static", "recall_static", "f1_static",
                                                      "pr_auc_static"]].round(3)
                st.dataframe(res)

            if not shown:
                st.info("No report files found in reports/.")

            st.markdown(
                "**How to read these results.** The labels come from NASA's own repeat based static flag, "
                "not from independent ground truth. The history tier uses past detections of the same "
                "place, which is close to restating that definition, so its scores show that past "
                "detections predict NASA's flag, not that the flag is correct. A one line rule on the "
                "count of recent days is shown next to the models: much of the history gain is already "
                "in that rule. The instant tier uses a single reading and is much weaker.")


LIVE_SRC = "Live near real time"
ARCHIVE_SRC = "Historical archive (Jan to Jun 2025, includes the data the models were trained on)"
ARCHIVE_BANNER = earth.ARCHIVE_BANNER


# ---------------------------------------------------------------- detail panel of a clicked detection
# The detail drawer is a Streamlit container laid over the right side of the globe by this css (the globe page
# is never narrowed or reloaded). It has a small arrow button on the middle of its left edge.
LAYOUT_CSS = (
    '[class*="st-key-earth_stage_"] { position: relative; gap: 0 !important; }'
    '[class*="st-key-earth_drawer_"] { position: absolute; top: 8px; right: 8px; width: min(400px, 92%); '
    'height: 684px; overflow-y: auto; box-sizing: border-box; background: #ffffff; padding: 12px 14px; '
    'border: 1px solid rgba(49,51,63,.25); border-radius: 6px; box-shadow: -4px 0 14px rgba(0,0,0,.3); z-index: 30; }'
    '[class*="st-key-earth_close_"] { position: absolute; top: 324px; right: calc(min(400px, 92%) - 6px); '
    'width: 26px; z-index: 31; }'
    '[class*="st-key-earth_close_"] button { width: 26px; min-width: 26px; height: 52px; min-height: 52px; '
    'padding: 0; border-radius: 13px 0 0 13px; font-size: 22px; line-height: 1; }')

# The globe page posts the clicked detection (or null) to the app page, named by its channel (l, a or u). This
# small component, which draws nothing, passes it to Python, carries a close request back to the globe page, and
# adds the css above. It only accepts messages from frames of this page.
PICK_JS = """
export default function(component) {
  const { setTriggerValue, data } = component;
  const ch = data && data.ch;
  if (!document.getElementById("earth-layout-css")) {
    const st = document.createElement("style");
    st.id = "earth-layout-css";
    st.textContent = __CSS__;
    document.head.appendChild(st);
  }
  window.__earthPick = window.__earthPick || {};
  window.__earthClose = window.__earthClose || {};
  const frames = () => Array.from(document.querySelectorAll("iframe"));
  const onMsg = (ev) => {
    const m = ev.data;
    if (!m || m.type !== "earth-pick" || m.ch !== ch) return;
    if (!frames().some(f => f.contentWindow === ev.source)) return;
    setTriggerValue("pick", JSON.stringify({ sig: m.sig, pick: m.pick }));
  };
  if (window.__earthPick[ch]) window.removeEventListener("message", window.__earthPick[ch]);
  window.__earthPick[ch] = onMsg;
  window.addEventListener("message", onMsg);
  if (data && data.close !== undefined && window.__earthClose[ch] !== data.close) {
    const first = window.__earthClose[ch] === undefined;
    window.__earthClose[ch] = data.close;
    if (!first) frames().forEach(f => { try { f.contentWindow.postMessage({ type: "earth-close", ch: ch }, "*"); } catch (e) {} });
  }
  if (window.__earthEsc) document.removeEventListener("keydown", window.__earthEsc);
  window.__earthEsc = (e) => {
    if (e.key === "Escape") frames().forEach(f => { try { f.contentWindow.postMessage({ type: "earth-close" }, "*"); } catch (err) {} });
  };
  document.addEventListener("keydown", window.__earthEsc);
  return () => {
    if (window.__earthPick[ch] === onMsg) { window.removeEventListener("message", onMsg); delete window.__earthPick[ch]; }
  };
}
""".replace("__CSS__", json.dumps(LAYOUT_CSS))


@st.cache_resource
def pick_component():
    """The listener component, registered once. None when this Streamlit has no two way components."""
    try:
        from streamlit.components import v2 as components_v2
        return components_v2.component("earth_pick", js=PICK_JS)
    except Exception:
        return None


@st.cache_resource
def panel_models():
    """Both saved bundles with their calibrators and bands (small, about 0.3 MB each)."""
    return serve.Models()


def read_pick(suffix, view):
    """The selected detection of this view, kept in the session. A new message replaces it; a change of the
    view (filters, Show, selection, band) drops it, because the globe page is drawn again without it."""
    key = f"earth_sel_{suffix}"
    comp = pick_component()
    if comp is not None:
        try:
            res = comp(key=f"earth_pick_{suffix}", on_pick_change=lambda: None,
                       data={"ch": suffix, "close": st.session_state.get(f"earth_close_n_{suffix}", 0)})
            raw = getattr(res, "pick", None)
        except Exception:
            raw = None
        if raw:
            msg = json.loads(raw)
            st.session_state[key] = {"pick": msg.get("pick"), "view": view} if msg.get("pick") else None
    sel = st.session_state.get(key)
    if sel and sel.get("view") != view:
        st.session_state[key] = sel = None
    return sel["pick"] if sel else None


def find_pick(g, pick):
    """The row of g that the page names: position (the page rounds to 3 decimals), date, time, sensor."""
    try:
        lat, lon = float(pick["lat"]), float(pick["lon"])
    except (KeyError, TypeError, ValueError):
        return None
    m = (np.abs(g["latitude"].to_numpy(float) - lat) <= 0.0006) & (np.abs(g["longitude"].to_numpy(float) - lon) <= 0.0006)
    sub = g[m]
    if pick.get("date"):
        sub = sub[sub["acq_date"].dt.strftime("%Y-%m-%d") == pick["date"]]
    if pick.get("time") and "timestamp" in sub:
        sub = sub[sub["timestamp"].dt.strftime("%H:%M") == pick["time"]]
    if pick.get("sensor") and "sensor" in sub:
        sub = sub[sub["sensor"].astype(str) == pick["sensor"]]
    return sub.iloc[0].to_dict() if len(sub) else None


def panel_payload(row, base, kind, band, tier_name):
    """The detail of one clicked detection. kind is live (read from the store), archive or upload (read from the
    frame the globe shows: for an upload that is only what the file contains)."""
    m = panel_models()
    key = "history" if tier_name == HISTORY else "instant"
    bundle = m.hist if key == "history" else m.inst
    bands = m.bands[key]
    use_band = band or bands.get("default")
    t = pd.Timestamp(row["timestamp"])
    if kind == "live":
        full = detail.store_row(row.get("sensor"), row["latitude"], row["longitude"], row["acq_date"],
                                detail.acq_time_of(row))
        if full:
            for c in ("cc", "admin1", "city", "p_static", "pred_static"):
                full[c] = row.get(c)
            row = full
        spread = None                                  # read side by side with the outside sources (sections.py)
    else:
        label = "archive file (January to June 2025)" if kind == "archive" else "the uploaded file only"
        spread = detail.spread_from_frame(base, float(row["latitude"]), float(row["longitude"]), t,
                                          cell_days=detail.cell_days_from_frame(base, row.get("cell_id"), t),
                                          source=label)
    payload = detail.build(row, source=kind, tier=key, bundle=bundle, bands=bands,
                           band=use_band, threshold=None, spread=spread)
    payload["row"] = row                               # the full stored row (with scan and track) for the sources
    return payload


def _v(x, unit="", nd=None):
    if x is None or x == "":
        return "unavailable"
    if nd is not None and isinstance(x, (int, float)):
        x = f"{x:,.{nd}f}"
    return f"{x}{unit}"


def _age(fetched_at):
    try:
        mins = (pd.Timestamp.now(tz="UTC") - pd.Timestamp(fetched_at)).total_seconds() / 60
    except Exception:
        return ""
    return "just now" if mins < 1 else f"{mins:.0f} minutes ago"


def _src(section):
    if section and section.get("source"):
        st.caption(f"Source: {section['source']}, {_age(section.get('fetched_at'))}.")


SHOWN = {"fire": "Predicted fire", "static": "Predicted static heat source",
         "uncertain": "Uncertain, needs investigation"}


LOADING = [False]


def _un(sec):
    """True (after writing the reason) when a section is missing, still loading or unavailable."""
    if not sec:
        st.write("Loading..." if LOADING[0] else "unavailable")
        return True
    if "unavailable" in sec:
        st.write(f"unavailable: {sec['unavailable']}")
        return True
    return False


def _src2(sec):
    if sec.get("source"):
        st.caption(f"Source: {sec['source']}, {_age(sec.get('fetched_at'))}.")


NUMBER_LABELS = (("general", "Emergency"), ("police", "Police"), ("fire", "Fire"), ("ambulance", "Ambulance"))


def render_spread(p):
    """Spread and history of the detection, read from the store or the frame, with its source and age."""
    st.markdown("**Spread and history**")
    s = p.get("spread")
    if s is None:
        st.write("Loading..." if LOADING[0] else "unavailable")
    elif "unavailable" in s:
        st.write(f"unavailable: {s['unavailable']}")
    else:
        st.write(f"{s['n']:,} detections within {s['km']:g} km in the {s['hours']} hours up to this detection.")
        if s["n"]:
            st.markdown("\n".join(
                [f"- {d['date']}: {d['n']}" for d in s["per_day"]] +
                [f"- First seen {s['first_seen']} UTC, last seen {s['last_seen']} UTC",
                 f"- Approximate area: {_v(s['area_km2'], ' square kilometres')} ({s['pixels']} pixels)",
                 f"- Trend: {s['trend']}"] +
                ([f"- Centre moved {s['move_km']} km" + (f" to the {s['move_dir']}" if s["move_dir"] else "")]
                 if s["move_km"] is not None else [])))
        st.write(f"Days seen at this cell in the last 90 days: {_v(s['cell_days_90'])}.")
        st.caption(s["note"])
        _src(s)


def render_context(p, loading=False):
    """Land and fuel, Weather, People and buildings and Response, each with its source and age. A section that is
    still None while loading shows "Loading..."."""
    LOADING[0] = loading
    render_spread(p)
    st.markdown("**Land and fuel**")
    sec = p.get("land")
    if not _un(sec):
        st.markdown("\n".join([f"- Land cover at the pixel: {sec['class']} ({sec['group']})"] +
                              ([f"- Main class within 1 km: {sec['around_1km']}"] if sec.get("around_1km") else []) +
                              ([f"- Fuel reading: {sec['fuel']}"] if sec.get("fuel") else [])))
        st.caption(sec["note"])
        _src2(sec)

    st.markdown("**Weather**")
    sec = p.get("weather")
    if not _un(sec):
        wind = (f"{_v(sec['wind_kmh'], ' km/h', 0)} from {sec['wind_from']}, blowing toward {sec['wind_to']}"
                if sec.get("wind_from") else _v(sec.get("wind_kmh"), " km/h", 0))
        st.markdown("\n".join([
            f"- At {sec['time']} UTC (hour of the detection)",
            f"- Temperature: {_v(sec['temperature_c'], ' C', 1)}, relative humidity: {_v(sec['humidity_pct'], ' %', 0)}",
            f"- Wind: {wind}, gusts {_v(sec['gust_kmh'], ' km/h', 0)}",
            f"- Rain in the 24 hours before: {_v(sec['rain_24h_mm'], ' mm', 1)}, in the 72 hours: "
            f"{_v(sec['rain_72h_mm'], ' mm', 1)}"]))
        st.caption(sec["note"])
        _src2(sec)

    st.markdown("**People and buildings**")
    sec = p.get("exposure")
    if not _un(sec):
        pop, pl, bd = sec.get("population"), sec.get("place"), sec.get("buildings")
        wait = "Loading..." if LOADING[0] else "unavailable"
        lines = []
        if pop is None:
            lines += [f"- People: {wait}"]
        elif "unavailable" not in pop:
            lines += [f"- People within 1 km: about {pop['within_1km']:,} (within 5 km: about {pop['within_5km']:,}), "
                      f"resolution {pop['resolution']}"]
        else:
            lines += [f"- People: unavailable ({pop['unavailable']})"]
        if pl is None:
            lines += [f"- Nearest named place: {wait}"]
        elif "name" in pl:
            lines += [f"- Nearest named place: {pl['name']}, {pl['km']:g} km {pl['direction']}"]
        else:
            lines += [f"- Nearest named place: unavailable ({pl['unavailable']})"]
        if bd is None:
            lines += [f"- Buildings within 1 km: {wait}"]
        elif "count" in bd:
            lines += [f"- Buildings within 1 km: {bd['count']:,}"]
        else:
            lines += [f"- Buildings within 1 km: unavailable ({bd['unavailable']})"]
        st.markdown("\n".join(lines))
        if pop and "note" in pop:
            st.caption(pop["note"])
        if bd and "note" in bd:
            st.caption(bd["note"])
        for part in (pop, bd):
            if part and part.get("source"):
                st.caption(f"Source: {part['source']}, {_age(part.get('fetched_at'))}.")

    st.markdown("**Response**")
    sec = p.get("response")
    if not _un(sec):
        nums = sec["numbers"]
        if "unavailable" in nums:
            st.write(f"Emergency numbers: unavailable ({nums['unavailable']})")
        else:
            bits = [f"{lab} {nums[k]}" for k, lab in NUMBER_LABELS if nums.get(k)]
            if nums.get("also"):
                bits.append(f"also {nums['also']}")
            tag = "" if nums["status"] == "verified" else " (unverified, check locally)"
            st.write(f"Emergency numbers, {nums['name']}: " + (", ".join(bits) if bits else "none on file") + tag)
        lines = []
        for label, res in sec["services"].items():
            if res is None:
                lines.append(f"- {label}: {'Loading...' if LOADING[0] else 'unavailable'}")
            elif "unavailable" in res:
                lines.append(f"- {label}: unavailable ({res['unavailable']})")
            elif "none" in res:
                lines.append(f"- {label}: none found within about {res['none']} km")
            else:
                lines.append(f"- {label}: {res['name']}, {res['km']:g} km {res['direction']}")
        st.markdown("\n".join(lines))
        st.caption(sec["note"])
        _src2(sec)


@st.fragment(run_every=1)
def context_poll(p, fetcher):
    """The four outside sections while some are still loading: each shows as soon as its source answers, a section
    that has not answered after 10 seconds shows as unavailable, timed out. One full rerun when all are settled."""
    render_context({**p, **fetcher.snapshot()}, loading=True)
    if not fetcher.pending():
        st.rerun()


def context_block(p, fetcher):
    if fetcher is None:
        render_context(p)
        return
    snap = fetcher.wait(0.4)                         # fast sources are in the first draw
    if fetcher.pending():
        context_poll(p, fetcher)
    else:
        render_context({**p, **snap})


def render_panel(p, fetcher=None):
    v, f, pl = p.get("verdict"), p.get("facts"), p.get("place")
    st.markdown("**Verdict**")
    if v:
        st.markdown(f"### {SHOWN.get(v['shown'], v['shown'])}")
        st.write(f"Tier: {v['tier']}. Calibrated probability of static: {_v(v['p'])} "
                 f"(model score {_v(v['p_raw'])}, the number in the popup). "
                 f"Band call: {v['band']}. Label at the operating point: {v['label']}.")
        if p.get("nasa_label"):
            st.write(f"NASA label: {p['nasa_label']}.")
        if not v["trained"]:
            st.warning(v["note"].capitalize() + ".")
        if v["top_features"]:
            st.write("What pushed the call most:")
            st.markdown("\n".join(f"- {x['label']} = {x['value']}, {x['push']}" for x in v["top_features"]))
        else:
            st.caption("Top features unavailable: the archive file keeps only some of the model's features.")
        _src(v)
    else:
        st.write("unavailable")

    st.markdown("**Detection**")
    if f:
        pixel = (f"{f['scan'] * 1000:.0f} m by {f['track'] * 1000:.0f} m" if not f["nominal_pixel"]
                 else "375 m (nominal, no scan and track in this data)")
        st.markdown("\n".join([
            f"- Time: {f['time_utc']} UTC, about {f['local_solar_time'][-5:]} local solar time",
            f"- Satellite: {f['satellite']}",
            f"- Brightness I-4: {_v(f['ti4'], ' K')}, I-5: {_v(f['ti5'], ' K')}",
            f"- FRP: {_v(f['frp'], ' MW')}",
            f"- Pixel: {pixel}",
            f"- Day or night: {_v({'D': 'day', 'N': 'night'}.get(f['daynight'], f['daynight']))}",
            f"- NASA confidence: {_v({'l': 'low', 'n': 'nominal', 'h': 'high'}.get(f['confidence'], f['confidence']))}",
            f"- Position: {f['latitude']}, {f['longitude']}"]))
        _src(f)
    else:
        st.write("unavailable")

    st.markdown("**Place**")
    if pl:
        st.markdown("\n".join([f"- Nearest place: {_v(pl['country'])}", f"- Province or state: {_v(pl['province'])}",
                               f"- Nearest city: {_v(pl['city'])}", f"- Elevation: {_v(pl['elevation_m'], ' m')}"]))
        st.caption("Place names come from the nearest named place in an offline list, so a remote or offshore "
                   "point can show a neighbouring country.")
        _src(pl)
    else:
        st.write("unavailable")

    context_block(p, fetcher)

    st.markdown("**Notes**")
    for n in p.get("notes", []):
        st.caption(n)
    st.caption("Data: " + ", ".join(p.get("attribution", [])) + ".")


def detail_panel(pick, g, base, kind, bands, band, suffix, tier_name):
    """The content of the drawer: the selected detection."""
    row = find_pick(g, pick)
    if row is None:
        st.caption("The selected detection is not in this view any more.")
        return
    cache_key = f"earth_detail_{suffix}"
    ident = json.dumps([pick, tier_name, kind, str(band and band.get("target"))], sort_keys=True, default=str)
    got = st.session_state.get(cache_key)
    if not got or got[0] != ident:
        try:
            payload = panel_payload(row, base, kind, band, tier_name)
        except Exception as e:
            st.error(f"The details could not be made: {type(e).__name__}")
            return
        try:
            fetcher = sections.start(payload.pop("row", row), with_spread=kind == "live")   # side by side, on this click
        except Exception:
            fetcher = None                                         # the other sections still show, these say unavailable
        st.session_state[cache_key] = got = (ident, payload, fetcher)
    render_panel(got[1], got[2])


def update_status_body():
    """Status of the update process, polled every few seconds by a fragment while it runs."""
    stt = updater.status()
    was = st.session_state.get("update_was_running", False)
    if stt["state"] == "running":
        st.info("An ingest is running" + (f" (started here at {stt['started']} UTC)" if stt.get("started") else
                                          " (started elsewhere, for example by the loop)") +
                ". The globe stays usable. The lines below come from the log of an update started here.")
    elif stt["state"] == "finished":
        st.success(f"Update finished: {stt['message']}." + (f" Ended {stt['finished']} UTC." if stt.get("finished") else ""))
    elif stt["state"] == "failed":
        st.error(f"Update failed: {stt['message']}. The store is unchanged, and fetched days are kept for the "
                 "next update.")
    for line in stt.get("lines", []):
        st.caption(line)
    st.session_state["update_was_running"] = stt["state"] == "running"
    if was and stt["state"] != "running":
        live_store_scored.clear()
        newest_detection.clear()
        st.rerun()                                   # the whole page once, to show the new data


def update_controls():
    """The Update live data button: ingest --once in a background process, one at a time, status by polling."""
    if updater.key_missing():
        st.warning("No FIRMS map key found. Set FIRMS_MAP_KEY in a .env file in the project root or in the "
                   "environment. Update live data cannot run without it.")
        return
    running = updater.is_running()
    c = st.columns([2, 6])
    if c[0].button("Update live data", disabled=running, key="update_live", help=updater.WARNING):
        res = updater.start()
        if res == "failed":
            st.error("The update could not be started.")
        else:
            st.rerun()                               # the page again, so the button shows as disabled at once
        running = updater.is_running()
    c[1].caption(updater.WARNING)
    if hasattr(st, "fragment"):
        st.fragment(run_every=3 if running else None)(update_status_body)()
    else:
        update_status_body()
        if running and st.button("Check the update", key="update_check"):
            st.rerun()


def earth_tab():
    """One Earth: globe, flat map and terrain in a MapLibre page (needs internet for the tiles).

    The front page: live near real time detections by default. The archive of January to June 2025 is the other
    source and is labelled as including the data the models were trained on. Both sources use the model tier
    chosen on this tab. The live source, where Streamlit has fragments, rereads the store every ten minutes
    without rerunning the rest of the page. Changing the Show option reloads the page, which restores the
    camera from the browser session.
    """
    src = st.radio("Data source", [LIVE_SRC, ARCHIVE_SRC], horizontal=True, key="earth_source")
    if src == LIVE_SRC:
        hours = st.session_state.get("earth_hours", 24)
        st.subheader(f"Detections of the last {hours} hours, classified by the model")
        update_controls()
    else:
        st.subheader("Archived detections, classified by the model")
        st.warning(ARCHIVE_BANNER)
    if src == LIVE_SRC and hasattr(st, "fragment"):
        st.fragment(run_every=LIVE_REFRESH_SECONDS)(earth_body)(src)
    else:
        earth_body(src)
        if src == LIVE_SRC and st.button("Refresh now", key="earth_refresh_view"):
            st.cache_data.clear()
            st.rerun()


def archive_source(live_tier, thr):
    """The archive frame of the chosen model tier for the chosen dates, with the label at the operating point.

    Returns (frame with p_static and pred_static, bands) or None."""
    history = live_tier == HISTORY
    frame = load_scored("scored_history.parquet" if history else f"scored_{TAG}.parquet")
    bundle = load_model("best_model_history.joblib" if history else f"best_model_{TAG}.joblib")
    if frame is None or bundle is None:
        st.warning("No scored archive found for this tier. Run `python -m src.train --tag headline` (instant) "
                   "or `python -m src.final_history` (with history) first.")
        return None
    if thr is None:
        thr = float(bundle["threshold"])
        st.caption(f"Instant model, saved threshold {thr:.3f}")
    prob_col = "p_static_history" if history else "p_static"
    dmin, dmax = frame["acq_date"].min().date(), frame["acq_date"].max().date()
    c1, c2 = st.columns([3, 2])
    if dmin != dmax:
        week = (max(dmin, dmax - timedelta(days=6)), dmax)               # the last 7 days of the data
        rng = c1.slider("Date range", dmin, dmax, week, key="earth_dates",
                        help="Seven days are drawn by default, so six months of points are not drawn at once.")
    else:
        rng = (dmin, dmax)
    held = c2.checkbox("Only held out test detections", value=False, key="earth_heldout",
                       help="The detections of cells that were not used for training, the ones to evaluate on.")
    keep = (frame["acq_date"] >= pd.Timestamp(rng[0])) & (frame["acq_date"] <= pd.Timestamp(rng[1]))
    if held:
        keep &= frame["in_test"]
    d = frame[keep]
    base = d.assign(p_static=d[prob_col], pred_static=(d[prob_col] >= thr).astype(int))
    bands = uncertain.bands_of(bundle, "best_model_history_band.json" if history else "best_model_headline_band.json")
    return base, bands


def earth_body(src):
    is_live = src == LIVE_SRC
    region, demo = None, False
    if not (MODEL_DIR / "best_model_history.joblib").exists() or \
            not (MODEL_DIR / "best_model_headline.joblib").exists():
        st.warning("The Earth tab needs both trained models. Run `python -m src.train --tag headline` "
                   "and `python -m src.final_history` first.")
        return
    hist = joblib.load(MODEL_DIR / "best_model_history.joblib", mmap_mode="r")   # a few hundred KB
    cols, live_tier, thr, col = model_controls(hist)
    if is_live:
        got = live_source(cols, live_tier, thr, col)
        if got is None:
            return
        base, region, demo, bands = got
        prefix, suffix = "earth_l", "l"
    else:
        got = archive_source(live_tier, thr)
        if got is None:
            return
        base, bands = got
        prefix, suffix = "earth_a", "a"

    render_globe(src, base, bands, kind="live" if is_live else "archive", region=region, demo=demo,
                 prefix=prefix, suffix=suffix, truth=not is_live, tier_name=live_tier, cap=earth.MAX_POINTS,
                 in_fragment=is_live and hasattr(st, "fragment"))


LC_ALL = "All land cover"
LC_OPTIONS = [LC_ALL, "Burning on cropland", "Burning on natural land", "Large clusters"]
LC_PROXY = "About 75 to 80 percent of land cover pixels are correct, so the split is a proxy."


@st.cache_data(ttl=600, show_spinner=False)
def land_probe():
    """(reachable, why) for the filter control. WorldCover is asked in a thread (the table keeps the answer ten
    minutes), and an answer that is not there after 0.3 seconds counts as reachable, so the page never waits for it."""
    table = landtable.get()
    if table.error and len(table) == 0 and not table.running():
        return False, table.error
    box = {}
    th = threading.Thread(target=lambda: box.update(r=table.probe()), daemon=True)
    th.start()
    th.join(0.3)
    ok, why = box.get("r", (True, ""))
    if not ok and len(table):
        ok = True                                    # offline, but the cells already in the table still work
    return ok, why


@st.cache_data(max_entries=8, show_spinner=False)
def cluster_flags(lat, lon, day, frp, query):
    try:
        flags, thr = clusters.flags(lat, lon, day, frp, query)
        return {"flags": flags, "frp": thr}
    except Exception as e:
        return {"error": type(e).__name__}


def rest_new(g):
    """True once per view: the cells of the whole view are queued behind the drawn ones the first time the view needs the
    land cover, and not again for a change of band, tier, operating point or selector."""
    key = (len(g), round(float(g["latitude"].sum()), 3), round(float(g["longitude"].sum()), 3))
    seen = st.session_state.setdefault("land_rest_views", [])
    if key in seen:
        return False
    seen.append(key)
    del seen[:-8]
    return True


def land_filter(choice, sampled, kinds_s, g):
    """Apply the land cover context filter to the drawn rows. Returns (rows, codes, summary). The summary tells what
    was computed, for the counts under the globe; rows and codes are unchanged when it could not be computed."""
    n = len(sampled)
    if choice == "Large clusters":
        pos = g.index.get_indexer(sampled.index)
        if (pos < 0).any():
            return sampled, kinds_s, {"error": "rows not found"}
        day = pd.to_datetime(g["acq_date"]).dt.normalize().astype("int64").to_numpy() // 86_400_000_000_000
        res = cluster_flags(g["latitude"].to_numpy(float), g["longitude"].to_numpy(float), day,
                            g["frp"].to_numpy(float), pos)
        if "error" in res:
            return sampled, kinds_s, res
        keep = res["flags"]
        return sampled[keep], kinds_s[keep], {"kind": "clusters", "n": n, "in": int(keep.sum()), "frp": res["frp"]}
    # classes: the class of each detection is the class of its 0.01 degree cell, an array lookup in the shared table.
    # Cells that are not in the table yet are read by a background thread; until then the view is drawn without the
    # filter (the page never waits for a tile).
    table = landtable.get()
    ids_s = landtable.cell_ids(sampled["latitude"].to_numpy(float), sampled["longitude"].to_numpy(float))
    code = table.lookup(ids_s)
    unknown = code == landtable.UNKNOWN
    if unknown.any():
        if table.error and not table.running():
            return sampled, kinds_s, {"error": table.error}
        ids_g = landtable.cell_ids(g["latitude"].to_numpy(float), g["longitude"].to_numpy(float))
        table.request(ids_s[unknown], ids_g if rest_new(g) else [])    # the drawn cells first, then the rest of the view
        return sampled, kinds_s, {"kind": "preparing", "n": int(len(np.unique(ids_s[unknown])))}
    if rest_new(g):                                  # the rest of the view, so a new selection finds its cells ready
        table.request([], landtable.cell_ids(g["latitude"].to_numpy(float), g["longitude"].to_numpy(float)))
    grp = landcover.groups(code)
    want = landcover.GROUP_CROP if choice == "Burning on cropland" else landcover.GROUP_NATURAL
    keep = grp == want
    return sampled[keep], kinds_s[keep], {"kind": "classes", "n": n, "cells": int(len(np.unique(ids_s))),
                                          "counts": landcover.accounting(code)}


@st.fragment(run_every=2)
def land_poll():
    """While cells are being read: how many places are left. One full rerun when the thread is done."""
    t = landtable.get()
    st.caption(f"Land cover is being prepared for {max(t.remaining(), 1 if t.running() else 0):,} places. The globe is "
               "drawn without the land cover filter until they are ready.")
    if not t.running():
        st.rerun()


def land_counts_text(info):
    """The count per class under the globe, with where the classes come from and the proxy sentence."""
    if info.get("kind") == "clusters":
        return (f"Land cover context, not a model. Large clusters: {info['in']:,} of the {info['n']:,} drawn points are in a "
                f"large cluster (5 or more detections within 2 km, on 2 or more days, FRP at or above "
                f"{info['frp']:.1f} MW, the 75th percentile of the window). A proxy for a large fire, not a model.")
    c = info["counts"]
    parts = ", ".join(f"{landcover.ACCOUNT_NAMES[k]} {c[k]:,}" for k in landcover.ACCOUNT_KEYS)
    text = (f"Land cover context, not a model. Of the {info['n']:,} drawn points ({info['cells']:,} cells of 0.01 "
            f"degree, about 1 km): {parts}. The class of a detection is the majority class of its cell, "
            f"{landcover.SOURCE}. {LC_PROXY}")
    if c["not_read"]:
        text += (" Not read means the cell could not be read in this run (a failed request), or it has no land cover "
                 "value (open sea or no data).")
    return text


def close_drawer(suffix, in_fragment):
    """The arrow button: drop the selection and tell the globe page to return to its earlier view."""
    st.session_state[f"earth_sel_{suffix}"] = None
    st.session_state[f"earth_close_n_{suffix}"] = st.session_state.get(f"earth_close_n_{suffix}", 0) + 1
    try:
        st.rerun(scope="fragment" if in_fragment else "app")
    except StreamlitAPIException:                   # this run is not a fragment run: rerun the whole page
        st.rerun()


def warm_countries(g, suffix):
    """Start the background download of the population grids of the countries in the view (once for a set of them)."""
    try:
        if "cc" not in g:
            return
        top = tuple(g["cc"].astype(str).value_counts().head(5).index)
        if st.session_state.get(f"warm_{suffix}") == top:
            return
        st.session_state[f"warm_{suffix}"] = top
        from src.context import people
        people.warm_many(top)
    except Exception:
        pass


def render_globe(src, base, bands, *, kind, region, demo, prefix, suffix, truth, tier_name, cap, in_fragment,
                 sig_extra=""):
    """Show, selector, band, geography filters, the globe with its detail drawer, and the counters, for any frame
    with p_static and pred_static: the live store, the archive or an uploaded file.

    The globe uses the full width. The detail drawer exists only while a detection is selected and lies over the
    right side; its arrow button, Escape and a click on empty globe close it."""
    is_live = kind == "live"
    c_show, c_focus = st.columns([3, 2])
    show = c_show.radio("Show", earth.SHOW if truth else earth.SHOW[:2], index=1, horizontal=True,
                        key=f"earth_show_{suffix}")
    focus_on = show in earth.FOCUS_VIEWS
    focus = c_focus.radio("Detections", earth.FOCUS, index=0, horizontal=True, key=f"earth_focus_{suffix}",
                          disabled=not focus_on,
                          help="Fires: predicted fire (NASA fire in Ground truth). Static sources: predicted "
                               "static heat source (NASA static in Ground truth). Both: all detections.")
    if not focus_on:
        c_focus.caption(f"Not used in {show}: every detection is drawn.")
    reach, why = land_probe()
    lc = LC_ALL
    if reach:
        lc = st.radio("Land cover context, not a model", LC_OPTIONS, horizontal=True, key=f"earth_lc_{suffix}",
                      help="Burning on cropland and on natural land keep the detections of the Detections selection "
                           "whose majority WorldCover class over the pixel footprint is cropland, or forest, "
                           "shrubland, grassland, wetland, moss and bare or sparse vegetation. Large clusters keeps "
                           "detections with 5 or more detections within 2 km on 2 or more days and FRP at or above "
                           "the 75th percentile of the window.")
    else:
        st.caption(f"The land cover filter is hidden: WorldCover could not be reached ({why}).")
    st.caption(earth.SHOW_HELP)
    st.caption("In Model prediction the Band selector decides fire, static or uncertain. The operating point "
               "above sets the label used in Errors only and in the popups.")
    if is_live:
        st.caption("Ground truth and Errors only are disabled for live data: there is no ground truth on live data.")
    elif truth:
        st.caption("Ground truth is NASA's own repeat based static flag from the label, not independent truth. "
                   "Errors only shows where the prediction disagrees with it.")
    else:
        st.caption("Ground truth and Errors only are disabled for an uploaded file: it has no label.")

    band = None
    if show == "Model prediction" and bands and bands.get("default"):
        pick = st.radio("Band", ["default (0.96)", "strict (0.98)"], horizontal=True, key=f"earth_band_{suffix}")
        band = bands["default"] if pick.startswith("default") else bands["strict"]
        if band and not band.get("reachable", True):
            st.caption("The accuracy target is not reachable for this model, so there is no uncertain band.")
            band = None
    if band and show == "Model prediction":
        lo = "none" if band["t_low"] is None else f"{band['t_low']:.3f}"
        hi = "none" if band["t_high"] is None else f"{band['t_high']:.3f}"
        st.caption(f"Three way decision on calibrated scores: static at or above {hi}, fire at or below {lo}, "
                   f"uncertain in between. Accuracy target {band['target']:.2f} on decided out of fold training "
                   "points, so the band uses no test data. NASA's flag is the label, not independent truth.")
        if band.get("oof_coverage", 0) >= 0.999:
            st.caption("On the training scores the forced binary decision already meets this target, so the "
                       "band decides every point and there are no uncertain points. Try strict (0.98).")
    with_uncertain = False
    if band and show == "Model prediction":
        with_uncertain = c_focus.checkbox("Include uncertain", value=False, key=f"earth_unc_{suffix}",
                                          disabled=focus != "Fires",
                                          help="With Fires, also draw the uncertain (amber) points, so hedged "
                                               "calls are not hidden.")
    codes = earth.focus_codes(show, focus, with_uncertain)
    g, cc, province, city = geo_filters(base, prefix)
    if g.empty:
        st.info("No detections for this selection.")
        return
    if band:
        g = g.assign(p_cal=uncertain.apply_calibrator(bands["calibrator"], g["p_static"].to_numpy()))
    sig = "|".join([src, str(region), sig_extra] + [str(st.session_state.get(f"{prefix}_{n}"))
                                                     for n in ("continent", "country", "province", "city")])
    view = "|".join([sig, show, focus, lc, str(with_uncertain), str(band and band.get("target"))])
    # the drawn points, the land cover filter and the page are made once per view: a click, which reruns this body,
    # reuses them, so the drawer's text is not held back by building the globe page again
    ckey = (view, len(g), round(float(g["latitude"].sum()), 3), round(float(g["p_static"].sum()), 4), cap, tier_name,
            kind, bool(demo), landtable.get().version if lc not in (LC_ALL, "Large clusters") else 0)
    c = st.session_state.get(f"earth_page_{suffix}")
    if not c or c["key"] != ckey:
        drawn, kinds = earth.select(g, show, band)
        drawn, kinds = earth.apply_focus(drawn, kinds, codes)      # before the point cap, so counts are complete
        sampled, kinds_s, total = earth.sample_rows(drawn, kinds, cap)
        lc_info, legend_kinds = None, kinds
        if lc != LC_ALL and len(sampled):
            # land cover is read for the drawn points only (at most cap), in batches
            sampled, kinds_s, lc_info = land_filter(lc, sampled, kinds_s, g)
            if "error" not in lc_info:
                legend_kinds = kinds_s
        data, note = earth.payload(sampled, kinds_s, band, cap=cap, total=total)
        page = None
        if not sampled.empty:
            filtered = len(g) < len(base)
            page = earth.html(data, earth.view_box(g) if filtered else None, earth.outlines(cc, province, g),
                              show, sig, demo, three_way=bool(band), codes=codes, kinds=legend_kinds,
                              live_source=is_live, channel=suffix)
        c = {"key": ckey, "page": page, "note": note, "n": len(data["x"]), "lc_info": lc_info}
        st.session_state[f"earth_page_{suffix}"] = c
    page, note, lc_info = c["page"], c["note"], c["lc_info"]
    if page is None:
        st.info("No errors in this selection." if show == "Errors only"
                else "No detections match this selection.")
    else:
        picked = f" Selection: {focus}." if codes is not None else ""
        st.caption(note + picked + " Needs internet for the imagery. To see every detection around one place, pick a "
                   "province or a nearest city.")
        st.caption("Click a detection to fly to it and see its satellite pixel footprint (an approximate rectangle "
                   "of scan by track, centred on the reported point) and its details in a drawer on the right. "
                   "The arrow on the drawer, Escape or a click on empty globe closes it.")
        st.session_state[f"earth_view_{suffix}"] = view            # the key of the selection, read by the tests
        sel = read_pick(suffix, view)
        with st.container(key=f"earth_stage_{suffix}"):
            components.html(page, height=700)
            if sel:
                with st.container(key=f"earth_drawer_{suffix}"):
                    detail_panel(sel, g, base, kind, bands, band, suffix, tier_name)
                with st.container(key=f"earth_close_{suffix}"):
                    if st.button("\u203a", key=f"earth_closebtn_{suffix}", help="Close the panel"):
                        close_drawer(suffix, in_fragment)
        if pick_component() is None:
            st.caption("The detail drawer needs a Streamlit version with two way components.")
    if lc_info is not None:
        if "error" in lc_info:
            st.caption(f"The land cover filter could not be applied ({lc_info['error']}); every detection is shown.")
        elif lc_info.get("kind") == "preparing":
            land_poll()
        else:
            st.caption(land_counts_text(lc_info))
            if lc != "Large clusters":
                st.caption("The legend counts are those of the drawn points of the filter.")

    warm_countries(g, suffix)                   # after the page is drawn: population grids of this view, in the background
    m = st.columns(5 if band else 4)
    if band:
        m[4].metric("Uncertain", f"{int((uncertain.decide(g['p_cal'].to_numpy(), band) == uncertain.UNCERTAIN).sum()):,}")
    m[0].metric("Detections in Earth", f"{len(g):,}")
    m[1].metric("Predicted static", f"{int(g['pred_static'].sum()):,}")
    m[2].metric("Predicted fire", f"{int((1 - g['pred_static']).sum()):,}")
    m[3].metric("Circles drawn", f"{c['n']:,}")
    if truth:
        fs, ff = earth.error_counts(g)
        n = st.columns(4)
        n[0].metric("NASA static", f"{int(g['is_static'].sum()):,}")
        n[1].metric("False static", f"{fs:,}")
        n[2].metric("False fire", f"{ff:,}")
        n[3].metric("All errors", f"{fs + ff:,}")


@st.cache_data(ttl=600, show_spinner=False)
def data_store_info(stamp):
    return datainfo.live_store()


def data_tab():
    """Data sources, labels, regions, features, splits and checks, and the live store, all read at run time."""
    path = ingest.store_path()
    store = data_store_info(path.stat().st_mtime_ns) if path.exists() else None
    st.subheader("Data sources")
    st.dataframe(datainfo.sources(store), hide_index=True, width="stretch")
    st.caption(f"Suomi NPP stopped delivering new data on {datainfo.SNPP_END}, so the live view uses NOAA-20 and "
               "NOAA-21. The live feed has no type field, so live detections have no label.")

    st.subheader("Labels")
    st.dataframe(datainfo.labels(), hide_index=True, width="stretch")
    st.caption("The label comes from NASA's type field: static is the positive class. It is NASA's own repeat "
               "based flag, not independent ground truth.")

    st.subheader("Regions")
    reg = datainfo.regions()
    if reg is not None:
        st.dataframe(reg.style.format({"Detections": "{:,}", "Static share": "{:.4f}"}), hide_index=True, width="stretch")
        h = datainfo.headline()
        if h is not None:
            st.caption(f"Detections kept per file. The training set has {h['rows']:,} detections after the "
                       f"duplicate check, {h['static_share']:.1%} static, from {h['date_min']} to {h['date_max']}.")

    st.subheader("Features")
    inst, hist = datainfo.feature_table("instant"), datainfo.feature_table("history")
    if inst is not None:
        st.write(f"Instant tier, {len(inst)} features. They come from one detection only.")
        st.dataframe(inst, hide_index=True, width="stretch")
    if hist is not None:
        st.write(f"With history tier, {len(hist)} features: the instant ones and the history of the same place.")
        st.dataframe(hist, hide_index=True, width="stretch")
    st.caption("Coordinates and the UTC hour are not features. The meanings are the labels of the detail drawer.")

    st.subheader("Split and checks")
    for line in datainfo.check_notes():
        st.write(line)
    checks = datainfo.check_rows()
    if checks is not None:
        st.write("Static class PR-AUC, from the saved reports.")
        st.dataframe(checks, width="stretch")

    st.subheader("Live store")
    if store is None:
        st.write("There is no live store on this machine.")
    else:
        newest = f"{store['newest']:%Y-%m-%d %H:%M} UTC" if store.get("newest") is not None else "unknown"
        st.dataframe(pd.DataFrame([
            {"Item": "Rows", "Value": f"{store['rows']:,}"},
            {"Item": "Regions", "Value": f"{len(store['regions'])}"},
            {"Item": "Regions not in the training data", "Value": f"{len(store['untrained'])}"},
            {"Item": "Sensors", "Value": ", ".join(store["sensors"])},
            {"Item": "Newest detection", "Value": newest}]), hide_index=True, width="stretch")


with tab_earth:
    if tab_earth.open:
        earth_tab()

if SHOW_ANALYSIS_TABS:
    with tab_data:
        if tab_data.open:
            data_tab()

with tab_live:
    if tab_live.open:
        classify_tab()

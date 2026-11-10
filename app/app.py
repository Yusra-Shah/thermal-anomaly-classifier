"""Streamlit demo. Run from the project root:  streamlit run app/app.py"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import earth, geo, live, uncertain  # noqa: E402
from src.config import MODEL_DIR, PROC_DIR, REPORT_DIR  # noqa: E402
from src.data import REQUIRED, clean, normalize_columns  # noqa: E402
from src.features import add_features  # noqa: E402
from src.history import add_history, add_history_v2  # noqa: E402
from src.label import add_cell  # noqa: E402

st.set_page_config(page_title="Fire Hotspot Classifier", layout="wide")

FIRE_COLOR = [214, 48, 39, 190]
STATIC_COLOR = [110, 110, 110, 190]
TAG = "headline"
MAX_POINTS = 30_000
MIN_HISTORY_DAYS = 30
INSTANT = "Instant (single reading)"
HISTORY = "With history (past detections)"
OPERATING_POINTS = ["F1 optimal", "precision >= 0.90", "precision >= 0.95"]


@st.cache_data(max_entries=1)           # one tier's scored table at a time
def load_scored(name):
    p = PROC_DIR / name
    if not p.exists():
        return None
    return live.add_geo(pd.read_parquet(p), PROC_DIR / "geo_cells.parquet")    # geography once per cell


@st.cache_resource(max_entries=1)       # one tier's model at a time
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


BASEMAP = "https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json"
ALL_CONT, ALL_COUNTRIES, ALL_PROVINCES, ALL_CITIES = "World", "All countries", "All provinces", "All cities"
CITY_OPTIONS = 200
OUTLINE = [30, 30, 30, 255]
PROVINCE_LINE = [150, 150, 150, 200]
HIGHLIGHT_FILL = [255, 190, 0, 50]
HIGHLIGHT_LINE = [210, 120, 0, 255]


def map_view(df, city=False, width=1000, height=500, pad=1.3):
    """View that fits the points with padding, or city level when a city is selected."""
    lon, lat = df["longitude"].to_numpy(), df["latitude"].to_numpy()
    w, e, s, n = float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())
    mid_lon, mid_lat = (w + e) / 2, (s + n) / 2
    if city or max(e - w, n - s) < 0.02:
        return pdk.ViewState(latitude=float(np.median(lat)), longitude=float(np.median(lon)), zoom=9)
    # Web Mercator with 512 px tiles: a span of `deg` degrees of longitude fits at this zoom
    span_x = (e - w) * pad
    span_y = (n - s) * pad / max(np.cos(np.radians(mid_lat)), 0.2)
    zoom = min(np.log2(360 * width / (512 * span_x)) if span_x else 12,
               np.log2(360 * height / (512 * span_y)) if span_y else 12)
    return pdk.ViewState(latitude=mid_lat, longitude=mid_lon, zoom=float(np.clip(zoom, 1, 12)))


def draw_map(df, color_col, max_points, key, country=None, province=False, city=False,
             days_col=None, days_label="Days seen at this spot"):
    """The map used by the Map tab and the upload tab.

    color_col is 1 for static (grey) and 0 for fire (red). Only the columns the map needs are
    sent to the browser. `country` is an ISO code: its outline and provinces are drawn, and
    `province` highlights the province that holds the points.
    """
    view_state = map_view(df, city)
    d = df
    if len(d) > max_points:
        st.caption(f"Drawing a random sample of {max_points:,} of {len(d):,} points. "
                   "Counters use all points.")
        d = d.sample(max_points, random_state=0)
    out = pd.DataFrame({
        "longitude": d["longitude"].round(4), "latitude": d["latitude"].round(4),
        "radius": (250 + 60 * np.sqrt(d["frp"].clip(lower=0))).round().astype(int),
        "when": d["timestamp"].dt.strftime("%Y-%m-%d %H:%M"),
        "p": d["p_static"].round(3), "frp": d["frp"].round(1), "ti4": d["bright_ti4"].round(1)})
    if days_col:
        out["days"] = d[days_col].astype(int)
    if "city" in d:
        out["city"] = d["city"]
    static = (d[color_col] == 1).to_numpy()

    layers = []
    if country:
        c = geo.country_feature(country)
        provs = geo.province_features(country)
        if provs:
            layers.append(pdk.Layer("GeoJsonLayer", geo.collection(provs), id="provinces", stroked=True,
                                    filled=False, get_line_color=PROVINCE_LINE, line_width_min_pixels=1))
        if province:
            hit = geo.province_at(country, df["longitude"].to_numpy(), df["latitude"].to_numpy())
            if hit:
                layers.append(pdk.Layer("GeoJsonLayer", geo.collection([hit]), id="province", stroked=True,
                                        filled=True, get_fill_color=HIGHLIGHT_FILL,
                                        get_line_color=HIGHLIGHT_LINE, line_width_min_pixels=3))
        if c:
            layers.append(pdk.Layer("GeoJsonLayer", geo.collection([c]), id="country", stroked=True,
                                    filled=False, get_line_color=OUTLINE, line_width_min_pixels=2))
    for name, mask, color in (("static", static, STATIC_COLOR), ("fire", ~static, FIRE_COLOR)):
        if mask.any():
            layers.append(pdk.Layer("ScatterplotLayer", out[mask], id=name, get_position="[longitude, latitude]",
                                    get_fill_color=color, get_radius="radius", radius_min_pixels=2,
                                    radius_max_pixels=14, pickable=True, auto_highlight=True))
    tip = "<b>p(static)</b> {p}<br/>FRP {frp}<br/>Brightness {ti4}<br/>"
    if days_col:
        tip += days_label + ": {days}<br/>"
    if "city" in out:
        tip += "Near {city}<br/>"
    tip += "{when} UTC"
    deck = pdk.Deck(layers=layers, initial_view_state=view_state, tooltip={"html": tip},
                    map_provider="carto", map_style=BASEMAP)
    sig = f"{view_state.longitude:.3f}_{view_state.latitude:.3f}_{view_state.zoom:.2f}_{country}_{province}"
    st.pydeck_chart(deck, key=f"{key}_{sig}", width="stretch")


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
def live_scored(region, stamp, _inst_bundle=None):
    """Features, history, both model scores and geography for the stored region. Cached by file stamp.

    The instant model is the large one. The sidebar's copy is reused when it is loaded,
    otherwise it is loaded here and released as soon as the scores are computed.
    """
    raw, fetched = live.load_snapshot(region)
    hist = joblib.load(MODEL_DIR / "best_model_history.joblib", mmap_mode="r")      # small
    inst = _inst_bundle or joblib.load(MODEL_DIR / "best_model_headline.joblib", mmap_mode="r")
    df = live.build_scored(region, hist, inst, raw)
    meta = {"bands": {"history": uncertain.bands_of(hist, "best_model_history_band.json"),
                      "instant": uncertain.bands_of(inst, "best_model_headline_band.json")},
            "inst_threshold": float(inst["threshold"]), "inst_features": list(inst["features"]),
            "thresholds": hist["thresholds"], "hist_features": list(hist["features"]),
            "default_point": hist.get("default_operating_point")}
    del inst
    return df, fetched, meta


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


def live_source(tier, bundle):
    """Controls and data of the live source: region, model, refresh, offline, days shown.

    tier and bundle are the sidebar's: the instant model is reused when it is already loaded.
    Returns (frame with p_static and pred_static, region, demo, band) or None when nothing can be shown.
    """
    if not (MODEL_DIR / "best_model_history.joblib").exists() or \
            not (MODEL_DIR / "best_model_headline.joblib").exists():
        st.warning("The live source needs both trained models. Run `python -m src.train --tag headline` "
                   "and `python -m src.final_history` first.")
        return None
    hist = joblib.load(MODEL_DIR / "best_model_history.joblib", mmap_mode="r")   # a few hundred KB
    key = live.load_key()
    st.info("Near real time data from NOAA-20 and NOAA-21, about 3 hours behind the satellite. There is no "
            "ground truth on live data, so only predictions are shown. History features are built inside each "
            "satellite and need up to 60 days of data, so the first fetch is slower.")
    st.caption("The models were trained on Suomi NPP and validated on NOAA-20 and NOAA-21 archive data. "
               "Near real time data has not been validated against the archive.")

    c1, c2, c3, c4 = st.columns([3, 3, 3, 2])
    region = c1.selectbox("Region loaded", list(live.REGIONS), key="earth_region")
    live_tier = c2.radio("Live model", [HISTORY, INSTANT], key="earth_live_tier")
    if live_tier == HISTORY:
        names = [n for n in OPERATING_POINTS if hist["thresholds"].get(n) is not None]
        point = c3.selectbox("Live operating point", names,
                             index=names.index(hist.get("default_operating_point", names[0])), key="earth_point")
        thr, col = float(hist["thresholds"][point]), "p_static_history"
    else:
        thr, col = None, "p_static_instant"          # the saved threshold is read with the scores below
        st.caption(instant_note())
    if key is None:
        c4.caption("Offline: no map key")
        offline = True
        st.warning("No FIRMS map key found. Set FIRMS_MAP_KEY in a .env file in the project root or in "
                   "the environment to fetch live data. Showing the saved snapshot if there is one.")
    else:
        offline = c4.checkbox("Offline (saved snapshot)", value=False, key="earth_offline")
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
        df, fetched, meta = live_scored(region, snap.stat().st_mtime_ns,
                                        bundle if tier == INSTANT else None)
    except Exception as e:
        st.error(f"Could not process the stored data: {e}")
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
    st.caption(f"Latest detection {latest:%Y-%m-%d %H:%M} UTC, data age {age_h:.1f} h."
               + (f" Last fetch {fetched:%Y-%m-%d %H:%M} UTC." if fetched is not None else ""))
    if not demo and "satellite_name" in df.columns:
        per = df["satellite_name"].value_counts()
        rows = ", ".join(f"{n} {int(per.get(n, 0)):,}" for n in live.LIVE_SOURCES)
        st.caption(f"Acquisition dates {df['acq_date'].min():%Y-%m-%d} to {df['acq_date'].max():%Y-%m-%d}. "
                   f"Rows per satellite: {rows}.")
    if age_h > 24 and not demo:
        st.warning(f"The newest detection in this region is {age_h:.0f} hours old. Either the snapshot is "
                   "old or the source has stopped delivering new data.")
    return view, region, demo, meta["bands"]["history" if live_tier == HISTORY else "instant"]


# ---------------------------------------------------------------- sidebar
st.title("Fire Hotspot Classifier")
st.caption("NASA FIRMS VIIRS detections, classified as likely fire or likely static heat source.")

tier = st.sidebar.radio("Model tier", [INSTANT, HISTORY])
if tier == INSTANT:
    scored = load_scored(f"scored_{TAG}.parquet")
    bundle = load_model(f"best_model_{TAG}.joblib")
    prob_col = "p_static"
else:
    scored = load_scored("scored_history.parquet")
    bundle = load_model("best_model_history.joblib")
    prob_col = "p_static_history"

if scored is None or bundle is None:
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

scored = scored.assign(p_static=scored[prob_col], pred_static=(scored[prob_col] >= thr).astype(int))

tab_map, tab_spot, tab_live, tab_metrics, tab_results, tab_earth = st.tabs(
    ["Map", "Why a spot is static", "Classify a new file", "Model metrics", "Results", "Earth"])

with tab_map:
    c1, c2, c3 = st.columns([2, 2, 3])
    view_mode = c1.radio("View", ["Raw detections", "Model prediction", "Ground truth"], horizontal=True)
    only_test = c2.checkbox("Only held out test detections", value=False)
    dmin, dmax = scored["acq_date"].min().date(), scored["acq_date"].max().date()
    rng = c3.slider("Date range", dmin, dmax, (dmin, dmax)) if dmin != dmax else (dmin, dmax)

    d = scored[(scored["acq_date"].dt.date >= rng[0]) & (scored["acq_date"].dt.date <= rng[1])]
    if only_test:
        d = d[d["in_test"]]

    d, country, province, city = geo_filters(d, "map")
    kw = dict(days_col="cell_days", days_label="Days seen at this spot")
    if d.empty:
        st.info("No detections for this selection.")
    elif view_mode == "Raw detections":
        d = d.assign(raw=0)
        draw_map(d, "raw", MAX_POINTS, "map", country, province, city, **kw)
        st.caption("Every detection looks the same to the satellite.")
    elif view_mode == "Model prediction":
        draw_map(d, "pred_static", MAX_POINTS, "map", country, province, city, **kw)
        st.caption("Red: predicted fire. Grey: predicted static heat source.")
    else:
        draw_map(d, "is_static", MAX_POINTS, "map", country, province, city, **kw)
        st.caption("Red: fire. Grey: static source according to the NASA type label (ground truth).")

    m1, m2, m3 = st.columns(3)
    m1.metric("Detections shown", f"{len(d):,}")
    m2.metric("Predicted static", f"{int(d['pred_static'].sum()):,}")
    m3.metric("Predicted fire", f"{int((1 - d['pred_static']).sum()):,}")

with tab_spot:
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

with tab_live:
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
            draw_map(new, "pred_static", MAX_POINTS, "upload", days_col="cell_days")
            st.write(f"{len(new):,} detections, {int(new['pred_static'].sum()):,} predicted static.")
        except Exception as e:  # show a readable message in the demo
            st.error(f"Could not process this file: {e}")

with tab_metrics:
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

with tab_results:
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


ARCHIVE_SRC, LIVE_SRC = "Archive (Jan to Jun 2025)", "Live near real time"


def earth_tab(archive, tier, bundle):
    """One Earth: globe, flat map and terrain in a MapLibre page (needs internet for the tiles).

    The archive uses the sidebar tier and operating point. The live source has its own controls.
    Changing the Show option reloads the page, which restores the camera from the browser session.
    """
    src = st.radio("Data source", [ARCHIVE_SRC, LIVE_SRC], horizontal=True, key="earth_source")
    is_live = src == LIVE_SRC
    region, demo = None, False
    if is_live:
        got = live_source(tier, bundle)
        if got is None:
            return
        base, region, demo, bands = got
        prefix, suffix = "earth_l", "l"
    else:
        base, prefix, suffix = archive, "earth_a", "a"
        if tier == INSTANT:
            st.caption(instant_note())
        bands = uncertain.bands_of(bundle, "best_model_headline_band.json" if tier == INSTANT
                                   else "best_model_history_band.json")

    show = st.radio("Show", earth.SHOW[:2] if is_live else earth.SHOW, index=1, horizontal=True,
                    key=f"earth_show_{suffix}")
    st.caption("In Model prediction the Band selector decides fire, static or uncertain. The sidebar threshold "
               "only affects the Map tab and the Errors only view.")
    if is_live:
        st.caption("Ground truth and Errors only are disabled for live data: there is no ground truth on live data.")
    else:
        st.caption("Ground truth is NASA's own repeat based static flag from the label, not independent truth. "
                   "Errors only shows where the prediction disagrees with it.")

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
    g, cc, province, city = geo_filters(base, prefix)
    if g.empty:
        st.info("No detections for this selection.")
        return
    if band:
        g = g.assign(p_cal=uncertain.apply_calibrator(bands["calibrator"], g["p_static"].to_numpy()))
    drawn, kinds = earth.select(g, show, band)
    data, note = earth.payload(drawn, kinds, band)
    if drawn.empty:
        st.info("No errors in this selection.")
    else:
        filtered = len(g) < len(base)
        st.caption(note + " Same tier and operating point as the Map tab for the archive. Needs internet "
                   "for the imagery. To see every detection around one place, pick a province or a nearest city.")
        sig = "|".join([src, str(region)] + [str(st.session_state.get(f"{prefix}_{n}"))
                                             for n in ("continent", "country", "province", "city")])
        page = earth.html(data, earth.view_box(g) if filtered else None, earth.outlines(cc, province, g),
                          show, sig, demo, three_way=bool(band))
        components.html(page, height=700)

    m = st.columns(5 if band else 4)
    if band:
        m[4].metric("Uncertain", f"{int((uncertain.decide(g['p_cal'].to_numpy(), band) == uncertain.UNCERTAIN).sum()):,}")
    m[0].metric("Detections in Earth", f"{len(g):,}")
    m[1].metric("Predicted static", f"{int(g['pred_static'].sum()):,}")
    m[2].metric("Predicted fire", f"{int((1 - g['pred_static']).sum()):,}")
    m[3].metric("Circles drawn", f"{len(data['x']):,}")
    if not is_live:
        fs, ff = earth.error_counts(g)
        n = st.columns(4)
        n[0].metric("NASA static", f"{int(g['is_static'].sum()):,}")
        n[1].metric("False static", f"{fs:,}")
        n[2].metric("False fire", f"{ff:,}")
        n[3].metric("All errors", f"{fs + ff:,}")


with tab_earth:
    earth_tab(scored, tier, bundle)

"""One interactive Earth page (MapLibre GL JS 5, globe projection) for st.components.v1.html.

The page needs internet for the basemap tiles and the MapLibre library. The detections are
passed as compact arrays and drawn as circles, so no copy of the table is kept anywhere.
"""
import json

import numpy as np
import pandas as pd

from . import geo, live, uncertain

MAPLIBRE = "5.6.0"
MAX_POINTS = 20_000
LIVE_NOTE = ("Predicted fire means any vegetation fire, including routine burning, not a reported wildfire "
             "event.")
SHOW_HELP = ("Raw detection: a hot pixel reported by the satellite, not classified. Model prediction: our "
             "classification of each detection as fire, static or uncertain.")
ARCHIVE_BANNER = ("These detections include the training data. Use Only held out test detections for "
                  "evaluation.")
SAT_NOTES = {"S-NPP": "S-NPP (no new data since 2 Nov 2026)", "NOAA-20": "NOAA-20 (active)",
             "NOAA-21": "NOAA-21 (active)"}

ATTRIBUTION = ("Imagery: Esri, Maxar, Earthstar Geographics and the GIS User Community. "
               "Labels: Esri. Terrain: Mapzen, AWS Terrain Tiles. Map style: Carto, OpenStreetMap contributors. "
               "Boundaries: Natural Earth. Rendering: MapLibre GL JS.")


SHOW = ["Raw detections", "Model prediction", "Ground truth", "Errors only"]
COLORS = {"fire": "#d63027", "static": "#8a8a8a", "raw": "#3b82c4", "uncertain": "#f0b400", "false_static": "#e08a00", "false_fire": "#6f3fb0"}
LEGENDS = {
    "Raw detections": [[0, COLORS["raw"], "Detection"]],
    "Model prediction": [[0, COLORS["fire"], "Predicted fire"], [1, COLORS["static"], "Predicted static"]],
    "Ground truth": [[0, COLORS["fire"], "NASA fire"], [1, COLORS["static"], "NASA static"]],
    "Errors only": [[2, COLORS["false_static"], "False static (predicted static, NASA fire)"],
                    [3, COLORS["false_fire"], "False fire (predicted fire, NASA static)"]],
}


FOCUS = ["Fires", "Static sources", "Both"]
FOCUS_VIEWS = ("Model prediction", "Ground truth")      # the views where the Fires or Static selector applies


def focus_codes(show, focus, include_uncertain=True):
    """Category codes (see LEGENDS) kept by the Fires, Static sources or Both selector, or None for all.

    Fires keeps predicted fire (NASA fire in Ground truth) and, with include_uncertain, the uncertain
    points, so hedged calls are not hidden silently. Static sources keeps predicted static (NASA static
    in Ground truth). Raw detections and Errors only are not changed."""
    if show not in FOCUS_VIEWS or focus == "Both":
        return None
    if focus == "Fires":
        return {0, 4} if include_uncertain else {0}
    return {1}


def apply_focus(df, kinds, codes):
    """Rows of df and their codes that are in codes (None keeps everything)."""
    if codes is None:
        return df, kinds
    keep = np.isin(kinds, sorted(codes))
    return df[keep], kinds[keep]


def legend(show, three_way=False, codes=None, kinds=None):
    """Legend entries for a Show option. The prediction view gets an uncertain entry when a band is used.

    codes keeps only the entries of the Fires or Static selection. With kinds (the code per drawn row,
    before the point cap) each label carries its count."""
    entries = [list(e) for e in LEGENDS[show]]
    if three_way and show == "Model prediction":
        entries.append([4, COLORS["uncertain"], "Uncertain (needs investigation)"])
    if codes is not None:
        entries = [e for e in entries if e[0] in codes] or entries[:1]
    if kinds is not None:
        kinds = np.asarray(kinds)
        for e in entries:
            e[2] = f"{e[2]} ({int((kinds == e[0]).sum()):,})"
    return entries


def error_counts(df):
    """False static and false fire counts against the NASA label (archive only)."""
    pred, truth = df["pred_static"].astype(int), df["is_static"].astype(int)
    return int(((pred == 1) & (truth == 0)).sum()), int(((pred == 0) & (truth == 1)).sum())


def select(df, show, band=None):
    """Rows to draw and a category code per row (see LEGENDS). With a band, the prediction view
    uses the three way decision and code 4 means uncertain."""
    pred = df["pred_static"].astype(int).to_numpy()
    if show == "Raw detections":
        return df, np.zeros(len(df), dtype=int)
    if show == "Model prediction":
        if band:
            d = uncertain.decide(df["p_cal"].to_numpy(), band)       # band is on calibrated scores
            return df, np.where(d == uncertain.UNCERTAIN, 4, d)
        return df, pred
    truth = df["is_static"].astype(int).to_numpy()
    if show == "Ground truth":
        return df, truth
    wrong = pred != truth
    return df[wrong], np.where(pred[wrong] == 1, 2, 3)


def payload(df, kinds, band=None, cap=MAX_POINTS):
    """Compact arrays for at most `cap` points (random sample) and the sample note.

    kinds is the category code per row of df, used for the circle colour.
    """
    total = len(df)
    d = df.assign(_k=kinds)
    if total > cap:
        d = d.sample(cap, random_state=0)
    cc, cc_names = pd.factorize(d["cc"].astype(str))
    pv, pv_names = pd.factorize(d["admin1"].astype(str))
    dt, dt_names = pd.factorize(d["acq_date"].dt.strftime("%Y-%m-%d"))
    dec = (uncertain.decide(d["p_cal"].to_numpy(), band) if band
           else d["pred_static"].astype(int).to_numpy())
    nasa = d["is_static"].astype(int).tolist() if "is_static" in d else [-1] * len(d)
    untrained = ((~d["region"].astype(str).isin(live.TRAINED)).astype(int).tolist() if "region" in d else [0] * len(d))
    when = d["timestamp"].dt.strftime("%H:%M").tolist() if "timestamp" in d else [""] * len(d)
    # pixel size in km for the click footprint; the archive files have no scan and track, the page then
    # uses the nominal 375 m pixel and says so
    # sensor per point (live), so a click names its detection: sensor, position, date and time
    if "sensor" in d:
        sc, sc_names = pd.factorize(d["sensor"].astype(str))
        px_sensor = {"s": sc.tolist(), "sn": list(sc_names)}
    else:
        px_sensor = {"s": [], "sn": []}
    has_px = "scan" in d and "track" in d
    px = ({"sc": d["scan"].astype(float).round(3).tolist(), "tr": d["track"].astype(float).round(3).tolist()}
          if has_px else {"sc": [], "tr": []})
    data = {
        # float32 columns are widened first, or the rounded values reach the page as 0.12800000607967377
        "x": d["longitude"].astype(float).round(3).tolist(), "y": d["latitude"].astype(float).round(3).tolist(),
        "p": d["p_static"].astype(float).round(3).tolist(), "c": d["pred_static"].astype(int).tolist(),
        "k": d["_k"].astype(int).tolist(), "n": nasa, "dec": [int(x) for x in dec],
        "t": d["bright_ti4"].astype(float).round(1).tolist(), "f": d["frp"].astype(float).round(1).tolist(),
        "d": dt.tolist(), "dn": list(dt_names), "u": untrained, "tm": when, "g": cc.tolist(),
        "gn": [geo.country_name(c) for c in cc_names], "a": pv.tolist(), "an": list(pv_names), **px, **px_sensor}
    note = (f"Showing a random sample of {len(d):,} of {total:,} detections." if total > cap
            else f"Showing all {total:,} detections.")
    return data, note


def view_box(df):
    """West, south, east, north of the points, or None."""
    if df.empty:
        return None
    return [float(df["longitude"].min()), float(df["latitude"].min()),
            float(df["longitude"].max()), float(df["latitude"].max())]


def outlines(cc, province, df=None):
    """Country outline to highlight, its provinces (if Natural Earth has them) and the picked province."""
    out = {"sel": None, "provs": None, "prov": None}
    if cc:
        out["sel"] = geo.country_feature(cc)
        provs = geo.province_features(cc)
        out["provs"] = geo.collection(provs) if provs else None
        if province and df is not None:
            out["prov"] = geo.province_at(cc, df["longitude"].to_numpy(), df["latitude"].to_numpy())
    return out


def html(data, box, shapes, show, sig, demo=False, height=700, three_way=False, codes=None, kinds=None,
         live_source=False, channel="a"):
    cfg = {"data": data, "box": box, "shapes": shapes, "lines": geo.country_lines(), "demo": demo,
           "legend": legend(show, three_way, codes, kinds), "sig": sig, "untrained": live.UNTRAINED_NOTE,
           "any_untrained": any(data.get("u", [])), "live_note": LIVE_NOTE, "sat_notes": SAT_NOTES,
           "ch": channel}
    page = (TEMPLATE.replace("__VER__", MAPLIBRE).replace("__HEIGHT__", str(height - 24))
            .replace("__ATTR__", ATTRIBUTION).replace("__CFG__", json.dumps(cfg, separators=(",", ":"))))
    return BOOT.replace("__PAGE__", page.replace("</script", "<\\/script"))


# MapLibre worker sources (GeoJSON, terrain) never load inside a srcdoc iframe, which is what
# st.components.v1.html creates: the circles and outlines would stay empty. The page is therefore
# handed over to a blob page, which has a real URL and the same origin, so storage still works.
BOOT = r"""<!doctype html><meta charset="utf-8">
<script type="text/plain" id="page">__PAGE__</script>
<script>
const h = document.getElementById("page").textContent.replace(/<\\\/script/g, "</script");
location.replace(URL.createObjectURL(new Blob([h], { type: "text/html" })));
</script>"""


TEMPLATE = r"""<!doctype html>
<html><head><meta charset="utf-8">
<link rel="stylesheet" href="https://unpkg.com/maplibre-gl@__VER__/dist/maplibre-gl.css">
<script src="https://unpkg.com/maplibre-gl@__VER__/dist/maplibre-gl.js"></script>
<style>
html, body { margin: 0; background: #0b0f1a; font: 13px sans-serif; color: #444; }
#map { height: __HEIGHT__px; width: 100%; }
#bar { position: absolute; top: 8px; left: 8px; z-index: 2; display: flex; gap: 6px; }
#bar button { background: #fff; border: 1px solid #bbb; border-radius: 3px; padding: 4px 9px; cursor: pointer; font: 12px sans-serif; }
#bar button.on { background: #222; color: #fff; box-shadow: inset 0 0 0 2px #ffcc00; }
#foot { background: #fff; padding: 3px 8px; font-size: 10px; color: #666; height: 18px; overflow: hidden; white-space: nowrap; }
#msg { position: absolute; bottom: 30px; left: 8px; background: rgba(255,255,255,.9); padding: 2px 6px; font-size: 11px; z-index: 2; }
.pop { font: 12px sans-serif; line-height: 1.5; }
#satcv { position: absolute; left: 0; top: 0; z-index: 1; pointer-events: none; }
#legend { position: absolute; top: 44px; left: 8px; z-index: 2; background: rgba(255,255,255,.92); padding: 5px 8px; font: 11px sans-serif; line-height: 1.6; border-radius: 3px; max-width: 280px; }
#legend .lghead { display: flex; align-items: center; gap: 6px; white-space: nowrap; font-weight: bold; }
#legtoggle { width: 18px; height: 18px; padding: 0; line-height: 14px; border: 1px solid #999; border-radius: 3px; background: #fff; cursor: pointer; font: bold 12px sans-serif; }
#legend .lgnote { color: #444; line-height: 1.35; margin: 2px 0; }
#legend .lgmuted { color: #666; }
#legend i { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; border: 1px solid #fff; }
.fplab { background: rgba(0,0,0,.75); color: #fff; font: 11px sans-serif; padding: 2px 6px; border-radius: 3px; white-space: nowrap; pointer-events: none; }
</style></head>
<body>
<div id="map"></div>
<canvas id="satcv"></canvas>
<div id="bar">
  <button id="rot" class="on" aria-pressed="true">Rotate</button>
  <button id="sat" class="on" aria-pressed="true">Satellites</button>
  <button id="base">Basemap: Satellite</button>
  <button id="tilt">Tilt 3D</button>
  <button id="back" style="display:none" title="Return to the earlier view (or press Escape)">Back</button>
</div>
<div id="msg" style="display:none"></div>
<div id="legend"></div>
<div id="foot">__ATTR__</div>
<script>
const CFG = __CFG__;
const D = CFG.data;
const SAT = {
  version: 8,
  projection: { type: "globe" },
  sources: {
    // capped at 17: zoom 18 tile requests timed out in a test, which left the zoomed view black; the map
    // stretches the zoom 17 tiles beyond that
    img: { type: "raster", tiles: ["https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"], tileSize: 256, maxzoom: 17 },
    // the same imagery capped at zoom 12 and stretched, under the detailed layer, so places without
    // detailed tiles show coarse imagery instead of black when a detection is zoomed in
    imglo: { type: "raster", tiles: ["https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"], tileSize: 256, maxzoom: 12 },
    lab: { type: "raster", tiles: ["https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}"], tileSize: 256, maxzoom: 18 }
  },
  layers: [
    { id: "imglo", type: "raster", source: "imglo", minzoom: 13 },
    { id: "img", type: "raster", source: "img" },
    { id: "lab", type: "raster", source: "lab" }
  ]
};
const VOYAGER = "https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json";
let mode = "sat", spin = CFG.box === null, satOn = true, ready = false;

// Camera, basemap and rotation survive a reload of this page (a change of the Show option).
const CAM = "earth_cam", SIG = "earth_sig";
let saved = null;
try {
  if (sessionStorage.getItem(SIG) === CFG.sig) saved = JSON.parse(sessionStorage.getItem(CAM));
  sessionStorage.setItem(SIG, CFG.sig);
} catch (e) { saved = null; }
if (saved) { mode = saved.mode; spin = saved.spin; if (saved.sat !== undefined) satOn = saved.sat; }

const map = new maplibregl.Map({
  container: "map", style: mode === "sat" ? SAT : VOYAGER, maxPitch: 80, attributionControl: false,
  center: saved ? saved.c : [30, 20], zoom: saved ? saved.z : 1.4,
  pitch: saved ? saved.p : 0, bearing: saved ? saved.b : 0
});
if (mode === "map") document.getElementById("base").textContent = "Basemap: Map";
setInterval(() => {
  try {
    // while a detection is selected, keep the view from before the click, so a reload does not stay zoomed in
    const v = sel ? sel.prev : { center: map.getCenter().toArray(), zoom: map.getZoom(), pitch: map.getPitch(), bearing: map.getBearing() };
    sessionStorage.setItem(CAM, JSON.stringify({ c: v.center, z: v.zoom, p: v.pitch, b: v.bearing,
      mode: mode, spin: spin, sat: satOn }));
  } catch (e) {}
}, 1000);
map.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), "top-right");
window.earthMap = map;
window.earthState = { spinning: () => spin, mode: () => mode, satellites: () => satOn, easing: () => easing,
                      selected: () => (sel ? sel.i : null), legendCollapsed: () => legendCollapsed };

function points() {
  const f = [];
  for (let i = 0; i < D.x.length; i++)
    f.push({ type: "Feature", properties: { i: i, k: D.k[i] }, geometry: { type: "Point", coordinates: [D.x[i], D.y[i]] } });
  return { type: "FeatureCollection", features: f };
}

function firstSymbol() {
  if (map.getLayer("lab")) return "lab";
  const l = map.getStyle().layers.find(x => x.type === "symbol");
  return l ? l.id : undefined;
}

function addSrc(id, spec) { if (!map.getSource(id)) map.addSource(id, spec); }
function addLay(spec, before) { if (!map.getLayer(spec.id)) map.addLayer(spec, before); }

// The detections go in first, so the map worker indexes them before the heavy outline data.
function installPoints() {
  addSrc("pts", { type: "geojson", data: points() });
  const colors = ["match", ["get", "k"]];
  CFG.legend.forEach(l => { colors.push(l[0], l[1]); });
  colors.push(CFG.legend[0][1]);
  addLay({ id: "pts", type: "circle", source: "pts", paint: {
    "circle-color": colors,
    "circle-radius": ["interpolate", ["linear"], ["zoom"], 1, 3, 5, 3.5, 8, 3, 16, 2.5],
    "circle-opacity": ["interpolate", ["linear"], ["zoom"], 1, 0.9, 6, 0.9, 8, 0.7],
    "circle-stroke-color": "#ffffff", "circle-stroke-width": 0.5, "circle-stroke-opacity": 0.6 } });
}

// Terrain, hillshade and outlines come after the points are indexed, and sit below the circles.
function installExtras() {
  addSrc("dem", { type: "raster-dem", tiles: ["https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"], encoding: "terrarium", tileSize: 256, maxzoom: 15 });
  addLay({ id: "hill", type: "hillshade", source: "dem", paint: { "hillshade-exaggeration": 0.45, "hillshade-shadow-color": "#222" } }, firstSymbol());
  addSrc("lines", { type: "geojson", data: CFG.lines, maxzoom: 8, tolerance: 1 });
  addLay({ id: "lines", type: "line", source: "lines", paint: { "line-color": "#ffffff", "line-opacity": 0.55, "line-width": 0.6 } }, "pts");
  const S = CFG.shapes;
  if (S.provs) {
    addSrc("provs", { type: "geojson", data: S.provs });
    addLay({ id: "provs", type: "line", source: "provs", paint: { "line-color": "#ffffff", "line-opacity": 0.45, "line-width": 0.5 } }, "pts");
  }
  if (S.sel) {
    addSrc("sel", { type: "geojson", data: S.sel });
    addLay({ id: "selfill", type: "fill", source: "sel", paint: { "fill-color": "#ffcc00", "fill-opacity": 0.08 } }, "pts");
    addLay({ id: "selline", type: "line", source: "sel", paint: { "line-color": "#ffcc00", "line-width": 1.6 } }, "pts");
  }
  if (S.prov) {
    addSrc("prov", { type: "geojson", data: S.prov });
    addLay({ id: "provfill", type: "fill", source: "prov", paint: { "fill-color": "#ffcc00", "fill-opacity": 0.14 } }, "pts");
    addLay({ id: "provline", type: "line", source: "prov", paint: { "line-color": "#ffcc00", "line-width": 2.2 } }, "pts");
  }
  terrainFor();
}

let extrasTimer = null;
function scheduleExtras() {
  if (extrasTimer) clearTimeout(extrasTimer);
  // when the points are indexed, or after 4 seconds on a slow machine, whichever comes first
  const go = () => { try { installExtras(); } catch (e) {} };
  extrasTimer = setTimeout(go, 4000);
  const early = (e) => {
    if (e.sourceId === "pts" && map.isSourceLoaded("pts")) { map.off("sourcedata", early); clearTimeout(extrasTimer); go(); }
  };
  map.on("sourcedata", early);
}

function terrainFor() {
  if (!map.getSource("dem")) return;
  const on = map.getZoom() >= 4;
  const has = !!map.getTerrain();
  if (on && !has) map.setTerrain({ source: "dem", exaggeration: 1.5 });
  if (!on && has) map.setTerrain(null);
}

// The legend collapses to one line with its small button. It starts collapsed after the first click on a
// detection of the browser session, and the choice is kept when the page is drawn again.
let legendCollapsed = false;
try { legendCollapsed = sessionStorage.getItem("earth_legend") === "1"; } catch (e) {}
function setLegend(c) {
  legendCollapsed = c;
  try { sessionStorage.setItem("earth_legend", c ? "1" : "0"); } catch (e) {}
  stats();
}
document.getElementById("legend").addEventListener("click", (e) => {
  if (e.target && e.target.id === "legtoggle") setLegend(!legendCollapsed);
});
function stats() {
  const lg = document.getElementById("legend");
  let inView = 0;
  try { inView = map.getLayer("pts") ? map.queryRenderedFeatures({ layers: ["pts"] }).length : 0; } catch (e) {}
  window.earthStats = { source: D.x.length, inView: inView };
  const dot = (c, t) => "<div><i style='background:" + c + "'></i>" + t + "</div>";
  let html = "<div class='lghead'><button id='legtoggle' aria-expanded='" + (!legendCollapsed) + "' title='" +
    (legendCollapsed ? "Show the legend" : "Collapse the legend") + "'>" + (legendCollapsed ? "+" : "-") + "</button><span>" +
    (legendCollapsed ? CFG.legend.map(l => l[2]).join(", ") : "Legend") + "</span></div>";
  if (!legendCollapsed) {
    html += CFG.legend.map(l => dot(l[1], l[2])).join("");
    if (CFG.live_note && CFG.legend.some(l => l[2].indexOf("Predicted fire") === 0)) html += "<div class='lgnote'>" + CFG.live_note + "</div>";
    if (CFG.any_untrained) html += "<div style='color:#a05a00'>Some detections: " + CFG.untrained + "</div>";
    if (satOn) html += SATS.map(q => dot(q.color, CFG.sat_notes[q.name])).join("") +
      "<div class='lgmuted'>Illustrative orbits, not real positions</div>";
    html += "<div class='lgmuted'>In source " + D.x.length.toLocaleString() + ", in view " + inView.toLocaleString() + "</div>";
  }
  lg.innerHTML = html;
}
try { stats(); } catch (e) {}                      // the legend shows before the first idle
map.on("idle", stats);
map.on("moveend", stats);

// install is safe to repeat: it runs on style.load, on styledata and on load, so the circles come
// back after any style change, including a diffed one that never fires style.load.
function ensure() {
  if (map.getLayer("pts")) return;
  try {
    if (mode === "map") map.setProjection({ type: "globe" });
    installPoints();
    drawSel();                                       // a selection survives a basemap change
    ready = true;
    stats();
    scheduleExtras();
  } catch (e) { /* the style is not ready yet, a later event retries */ }
}
map.on("style.load", ensure);
map.on("styledata", ensure);
map.on("load", ensure);
map.on("zoomend", terrainFor);
map.on("error", (e) => console.error("maplibre " + (e.error && e.error.message)));

// Rotation is a toggle. It turns the globe slowly. From a closer zoom it first eases out to the globe, then
// turns. A drag, a touch or a zoom on the map stops it. Satellites are independent of it.
const GLOBE_ZOOM = 3.2, SPIN_DEG_PER_S = 4, EASE_ZOOM = 1.6;
let easing = false;
function setSpin(on) {
  spin = on;
  const b = document.getElementById("rot");
  b.classList.toggle("on", on);
  b.setAttribute("aria-pressed", on ? "true" : "false");
  if (!on && easing) { easing = false; map.stop(); }
}
function stopSpin() { if (spin) setSpin(false); }
setSpin(spin);
const cv = map.getCanvas();
["mousedown", "touchstart", "wheel", "pointerdown"].forEach(e => cv.addEventListener(e, stopSpin, { passive: true }));
document.getElementById("rot").onclick = () => setSpin(!spin);

// Illustrative satellites: a circular sun synchronous orbit (inclination 98.7 degrees, period 101 minutes),
// three phases, run 60 times faster than real time. Everything is drawn on one small canvas over the map every
// frame (no map source): a faint orbit line, a fading ground track, a satellite icon turned to its direction of
// travel and a thin name. A satellite on the far side of the globe is not drawn.
const SATS = [{ name: "S-NPP", color: "#ffd23f", phase: 0, active: false },
              { name: "NOAA-20", color: "#00e5ff", phase: 120, active: true },
              { name: "NOAA-21", color: "#ff7ad9", phase: 240, active: true }];
const INC = 98.7 * Math.PI / 180, PERIOD = 101 * 60, SPEEDUP = 60, EARTH_RAD_S = 2 * Math.PI / 86164;
const TRAIL_POINTS = 30, TRAIL_STEP = 36, ORBIT_POINTS = 72, ICON = 28;
function satLonLat(T, phase) {
  const u = 2 * Math.PI * T / PERIOD + phase * Math.PI / 180;
  const lat = Math.asin(Math.sin(INC) * Math.sin(u));
  let lon = Math.atan2(Math.cos(INC) * Math.sin(u), Math.cos(u)) - EARTH_RAD_S * T;
  lon = ((lon * 180 / Math.PI + 540) % 360 + 360) % 360 - 180;
  return [lon, lat * 180 / Math.PI];
}
// A point is on the near side when it is within the visible cap around the map centre.
function nearSide(p) {
  const c = map.getCenter(), rad = Math.PI / 180;
  const dot = Math.sin(c.lat * rad) * Math.sin(p[1] * rad) + Math.cos(c.lat * rad) * Math.cos(p[1] * rad) * Math.cos((p[0] - c.lng) * rad);
  return dot > Math.max(0.02, 0.3 - 0.06 * map.getZoom());
}
// The icon is drawn once per colour: a small body, two solar panels, a short antenna and a dish, pointing along +x.
function makeSprite(color) {
  const S = 64, cv = document.createElement("canvas");
  cv.width = cv.height = S;
  const g = cv.getContext("2d");
  const glow = g.createRadialGradient(32, 32, 4, 32, 32, 31);
  glow.addColorStop(0, color + "88");
  glow.addColorStop(1, color + "00");
  g.fillStyle = glow;
  g.beginPath(); g.arc(32, 32, 31, 0, 6.2832); g.fill();
  [[2, 25], [39, 62]].forEach(pn => {                          // solar panels, one on each side of the body
    g.fillStyle = "#1d4f9f"; g.fillRect(24, pn[0], 16, pn[1] - pn[0]);
    g.strokeStyle = "#bcd5ff"; g.lineWidth = 1.2;
    g.strokeRect(24.5, pn[0] + 0.5, 15, pn[1] - pn[0] - 1);
    for (let y = pn[0] + 6; y < pn[1] - 2; y += 6) { g.beginPath(); g.moveTo(24, y); g.lineTo(40, y); g.stroke(); }
    g.beginPath(); g.moveTo(32, pn[0]); g.lineTo(32, pn[1]); g.stroke();
  });
  g.fillStyle = "#e4e8ee"; g.fillRect(20, 24, 24, 16);        // body
  g.fillStyle = "#d4a72c"; g.fillRect(20, 24, 6, 16);         // foil band
  g.strokeStyle = "#ffffff"; g.lineWidth = 1.6; g.strokeRect(20, 24, 24, 16);
  g.strokeStyle = "#ffffff"; g.lineWidth = 2;                  // short antenna at the front
  g.beginPath(); g.moveTo(44, 32); g.lineTo(58, 32); g.stroke();
  g.fillStyle = "#ffffff"; g.beginPath(); g.arc(59, 32, 2.6, 0, 6.2832); g.fill();
  g.fillStyle = color;                                          // dish at the back
  g.beginPath(); g.moveTo(20, 32); g.lineTo(11, 24); g.quadraticCurveTo(5, 32, 11, 40); g.closePath(); g.fill();
  g.strokeStyle = "#ffffff"; g.lineWidth = 1.2; g.stroke();
  return cv;
}
const sprites = SATS.map(q => makeSprite(q.color));
const satCv = document.getElementById("satcv");
const satCtx = satCv.getContext("2d");
function sizeCanvas() {
  const m = document.getElementById("map"), dpr = window.devicePixelRatio || 1;
  satCv.style.width = m.clientWidth + "px"; satCv.style.height = m.clientHeight + "px";
  satCv.width = Math.round(m.clientWidth * dpr); satCv.height = Math.round(m.clientHeight * dpr);
  satCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
}
sizeCanvas();
window.addEventListener("resize", sizeCanvas);
map.on("resize", sizeCanvas);
// Draws a polyline of ground points. A segment is skipped when either end is on the far side.
function trace(points, color, alphaOf, width) {
  satCtx.lineWidth = width; satCtx.strokeStyle = color; satCtx.lineCap = "round";
  let prev = null;
  for (let i = 0; i < points.length; i++) {
    const p = points[i], ok = nearSide(p);
    const xy = ok ? map.project(p) : null;
    if (prev && xy) {
      satCtx.globalAlpha = alphaOf(i / (points.length - 1));
      satCtx.beginPath(); satCtx.moveTo(prev.x, prev.y); satCtx.lineTo(xy.x, xy.y); satCtx.stroke();
    }
    prev = xy;
  }
  satCtx.globalAlpha = 1;
}
function drawSats(T) {
  satCtx.clearRect(0, 0, satCv.clientWidth, satCv.clientHeight);
  const info = [];
  SATS.forEach((q, i) => {
    const pos = satLonLat(T, q.phase), ok = nearSide(pos);
    const orbit = [], trail = [];
    for (let k = 0; k <= ORBIT_POINTS; k++) orbit.push(satLonLat(T + (k / ORBIT_POINTS - 0.5) * PERIOD, q.phase));
    for (let k = TRAIL_POINTS; k >= 0; k--) trail.push(satLonLat(T - k * TRAIL_STEP, q.phase));
    trace(orbit, q.color, () => 0.22, 1);
    trace(trail, q.color, f => 0.9 * f * f, 2.4);
    if (!ok) { info.push({ name: q.name, visible: false }); return; }
    const a = map.project(pos), b = map.project(satLonLat(T + 4, q.phase));
    const ang = Math.atan2(b.y - a.y, b.x - a.x);
    satCtx.save();
    satCtx.globalAlpha = q.active ? 1 : 0.55;
    satCtx.translate(a.x, a.y); satCtx.rotate(ang);
    satCtx.drawImage(sprites[i], -ICON / 2, -ICON / 2, ICON, ICON);
    satCtx.restore();
    satCtx.save();
    satCtx.globalAlpha = q.active ? 1 : 0.7;
    satCtx.font = "300 11px sans-serif"; satCtx.textAlign = "center"; satCtx.textBaseline = "top";
    satCtx.lineWidth = 3; satCtx.strokeStyle = "rgba(0,0,0,0.7)"; satCtx.strokeText(q.name, a.x, a.y + ICON / 2 + 1);
    satCtx.fillStyle = "#ffffff"; satCtx.fillText(q.name, a.x, a.y + ICON / 2 + 1);
    satCtx.restore();
    info.push({ name: q.name, visible: true, x: a.x, y: a.y, angle: ang });
  });
  window.earthSats = info;
}
function setSats(on) {
  satOn = on;
  const b = document.getElementById("sat");
  b.classList.toggle("on", on);
  b.setAttribute("aria-pressed", on ? "true" : "false");
  if (!on) { satCtx.clearRect(0, 0, satCv.clientWidth, satCv.clientHeight); window.earthSats = []; }
  try { stats(); } catch (e) {}
}
document.getElementById("sat").onclick = () => setSats(!satOn);
setSats(satOn);

let last = performance.now();
function frame(now) {
  const dt = Math.min((now - last) / 1000, 0.1);
  last = now;
  if (spin && ready) {
    if (map.getZoom() > GLOBE_ZOOM) {
      if (!easing) {
        easing = true;
        map.once("moveend", () => { easing = false; });
        map.easeTo({ zoom: EASE_ZOOM, pitch: 0, bearing: 0, duration: 2000 });
      }
    } else if (!easing) {
      const c = map.getCenter();
      map.jumpTo({ center: [c.lng + SPIN_DEG_PER_S * dt, c.lat] });
    }
  }
  if (satOn && ready) drawSats(now / 1000 * SPEEDUP);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

document.getElementById("base").onclick = (ev) => {
  mode = mode === "sat" ? "map" : "sat";
  ev.target.textContent = "Basemap: " + (mode === "sat" ? "Satellite" : "Map");
  ready = false;
  map.setStyle(mode === "sat" ? SAT : VOYAGER, { diff: false });
  if (mode === "map") setTimeout(() => {
    if (!ready) {
      mode = "sat";
      ev.target.textContent = "Basemap: Satellite";
      note("The map style could not be loaded. Showing satellite.");
      map.setStyle(SAT, { diff: false });
    }
  }, 20000);
};
function note(t) {
  const m = document.getElementById("msg");
  m.textContent = t; m.style.display = "block";
  setTimeout(() => { m.style.display = "none"; }, 6000);
}
document.getElementById("tilt").onclick = (ev) => {
  const up = map.getPitch() < 30;
  map.easeTo({ pitch: up ? 60 : 0, duration: 800 });
  ev.target.classList.toggle("on", up);
};

// Click a detection: fly in, draw the pixel footprint and highlight the point. The footprint is an
// approximation: a rectangle of scan (east to west) by track (north to south) kilometres, axis aligned and
// centred on the reported point. Escape, a click elsewhere or closing the popup returns the earlier view.
const NOMINAL_KM = 0.375, SEL_MAX_ZOOM = 18, SEL_MIN_ZOOM = 15, KM_PER_DEG = 111.32;
let sel = null, popup = null, fpMarker = null;
function pixelKm(i) {
  const has = D.sc && i < D.sc.length && D.sc[i] > 0 && D.tr[i] > 0;
  return { scan: has ? D.sc[i] : NOMINAL_KM, track: has ? D.tr[i] : NOMINAL_KM, nominal: !has };
}
function footprint(lon, lat, scanKm, trackKm) {
  const dLat = trackKm / 2 / KM_PER_DEG;
  const dLon = scanKm / 2 / (KM_PER_DEG * Math.max(Math.cos(lat * Math.PI / 180), 0.01));
  return { type: "Feature", properties: {}, geometry: { type: "Polygon", coordinates: [[[lon - dLon, lat - dLat],
    [lon + dLon, lat - dLat], [lon + dLon, lat + dLat], [lon - dLon, lat + dLat], [lon - dLon, lat - dLat]]] } };
}
function pixelLabel(px) {
  const s = Math.round(px.scan * 1000), t = Math.round(px.track * 1000);
  const size = Math.abs(s - t) <= 5 ? s + " m pixel" : s + " m by " + t + " m pixel";
  return size + ", " + (px.scan * px.track).toFixed(2) + " square kilometres" +
    (px.nominal ? " (nominal size, no scan and track in this data)" : "");
}
// The zoom where the footprint fills about 45 percent of the shorter side of the map, within 15 to 18.
// At these zooms the globe is drawn as a flat map, where one pixel at zoom z covers
// 78271.517 * cos(latitude) / 2^z metres (512 pixel tiles).
function fitZoom(lat, km) {
  const c = map.getCanvas(), dim = Math.min(c.clientWidth, c.clientHeight) || 600;
  const z = Math.log2(78271.517 * Math.cos(lat * Math.PI / 180) * 0.45 * dim / (km * 1000));
  return Math.max(SEL_MIN_ZOOM, Math.min(SEL_MAX_ZOOM, z));
}
function collection(f) { return { type: "FeatureCollection", features: [f] }; }
function drawSel() {
  if (!sel || !map.getLayer("pts")) return;
  const pt = { type: "Feature", properties: {}, geometry: { type: "Point", coordinates: [D.x[sel.i], D.y[sel.i]] } };
  if (map.getSource("fp")) map.getSource("fp").setData(collection(sel.f)); else addSrc("fp", { type: "geojson", data: collection(sel.f) });
  if (map.getSource("hl")) map.getSource("hl").setData(collection(pt)); else addSrc("hl", { type: "geojson", data: collection(pt) });
  addLay({ id: "fpfill", type: "fill", source: "fp", paint: { "fill-color": "#ffcc00", "fill-opacity": 0.12 } });
  addLay({ id: "fpline", type: "line", source: "fp", paint: { "line-color": "#ffcc00", "line-width": 2 } });
  addLay({ id: "hl", type: "circle", source: "hl", paint: { "circle-radius": 9, "circle-color": "rgba(0,0,0,0)",
    "circle-stroke-color": "#ffcc00", "circle-stroke-width": 3 } });
  const top = sel.f.geometry.coordinates[0][2];
  if (!fpMarker) {
    const el = document.createElement("div");
    el.className = "fplab";
    el.id = "fplab";
    fpMarker = new maplibregl.Marker({ element: el, anchor: "bottom", offset: [0, -4] });
  }
  fpMarker.getElement().textContent = pixelLabel(sel.px);
  fpMarker.setLngLat([D.x[sel.i], top[1]]).addTo(map);
}
function closePopup() { if (popup) { const p = popup; popup = null; p.remove(); } }
// Tell the app which detection is selected (null when none), so it can open or close the side panel.
function postPick(i) {
  try {
    const pick = i === null ? null : { lat: D.y[i], lon: D.x[i], date: D.dn[D.d[i]], time: D.tm[i],
      sensor: D.s && i < D.s.length ? D.sn[D.s[i]] : null };
    window.parent.postMessage({ type: "earth-pick", sig: CFG.sig, ch: CFG.ch, pick: pick }, "*");
  } catch (e) {}
}
function clearSel(back) {
  if (!sel) return;
  const prev = sel.prev;
  sel = null;
  closePopup();
  ["hl", "fpline", "fpfill"].forEach(id => { if (map.getLayer(id)) map.removeLayer(id); });
  ["hl", "fp"].forEach(id => { if (map.getSource(id)) map.removeSource(id); });
  if (fpMarker) { fpMarker.remove(); fpMarker = null; }
  document.getElementById("back").style.display = "none";
  postPick(null);
  if (back) map.flyTo({ center: prev.center, zoom: prev.zoom, pitch: prev.pitch, bearing: prev.bearing, duration: 2000,
                       essential: true, padding: { top: 0, bottom: 0, left: 0, right: 0 } });
}
// The app answers a closed drawer (the arrow button, or Escape in the app page) with this message.
window.addEventListener("message", (e) => {
  const m = e.data;
  if (m && m.type === "earth-close" && (m.ch === undefined || m.ch === CFG.ch)) clearSel(true);
});
// The detail drawer covers the right side of the globe, so a selected point is centred in the rest.
function drawerPad() { return Math.min(420, Math.round((map.getCanvas().clientWidth || 800) * 0.42)); }
function selectPoint(i) {
  const px = pixelKm(i);
  if (!sel) sel = { prev: { center: map.getCenter().toArray(), zoom: map.getZoom(), pitch: map.getPitch(), bearing: map.getBearing() } };
  sel.i = i; sel.px = px; sel.f = footprint(D.x[i], D.y[i], px.scan, px.track);
  setSpin(false);
  drawSel();
  document.getElementById("back").style.display = "inline-block";
  try {
    if (!sessionStorage.getItem("earth_legend_seen")) { sessionStorage.setItem("earth_legend_seen", "1"); setLegend(true); }
  } catch (e) {}
  map.flyTo({ center: [D.x[i], D.y[i]], zoom: fitZoom(D.y[i], Math.max(px.scan, px.track)), pitch: 0, bearing: 0,
              duration: 2500, essential: true, padding: { top: 0, bottom: 0, left: 0, right: drawerPad() } });
  note("Back, Escape or a click elsewhere returns the view.");
  postPick(i);
}
function popupHtml(i) {
  const px = pixelKm(i);
  const nasa = D.n[i] < 0 ? "" : "NASA label: " + (D.n[i] === 1 ? "static" : "fire") + "<br>";
  return "<div class='pop'><b>Decision: " + (D.dec[i] === 1 ? "static heat source" : D.dec[i] === 0 ? "predicted fire" : "uncertain, needs investigation") + "</b><br>" +
    "Model score for static (not calibrated): " + D.p[i] + "<br>" + nasa + "Brightness (I-4): " + D.t[i] + " K<br>FRP: " + D.f[i] + " MW<br>" +
    "Pixel: " + Math.round(px.scan * 1000) + " m by " + Math.round(px.track * 1000) + " m" + (px.nominal ? " (nominal)" : "") + "<br>" +
    "Date: " + D.dn[D.d[i]] + (D.tm[i] ? " " + D.tm[i] + " UTC" : "") + "<br>Nearest place: " + D.gn[D.g[i]] + "<br>Province: " + D.an[D.a[i]] +
    (D.u[i] ? "<br><b style='color:#a05a00'>" + CFG.untrained.charAt(0).toUpperCase() + CFG.untrained.slice(1) + "</b>" : "") + "</div>";
}
map.on("click", (e) => {
  const r = 8, hits = map.queryRenderedFeatures([[e.point.x - r, e.point.y - r], [e.point.x + r, e.point.y + r]], { layers: ["pts"] });
  if (!hits.length) { clearSel(true); return; }                 // a click elsewhere returns the view
  const i = hits[0].properties.i;
  closePopup();
  selectPoint(i);
  // below the point, so the footprint label above it stays visible
  const p = new maplibregl.Popup({ maxWidth: "260px", offset: 14, closeOnClick: false, anchor: "top" })
    .setLngLat([D.x[i], D.y[i]]).setHTML(popupHtml(i)).addTo(map);
  popup = p;
  p.on("close", () => { if (popup === p) { popup = null; clearSel(true); } });
});
document.addEventListener("keydown", (e) => { if (e.key === "Escape") clearSel(true); });
document.getElementById("back").onclick = () => clearSel(true);
map.on("mouseenter", "pts", () => { cv.style.cursor = "pointer"; });
map.on("mouseleave", "pts", () => { cv.style.cursor = ""; });

map.once("load", () => {
  if (CFG.box && !saved) {
    const b = CFG.box;
    map.fitBounds([[b[0], b[1]], [b[2], b[3]]], { padding: 70, maxZoom: 11, duration: 3000 });
  }
});
</script></body></html>
"""

"""One interactive Earth page (MapLibre GL JS 5, globe projection) for st.components.v1.html.

The page needs internet for the basemap tiles and the MapLibre library. The detections are
passed as compact arrays and drawn as circles, so no copy of the table is kept anywhere.
"""
import json

import numpy as np
import pandas as pd

from . import geo, uncertain

MAPLIBRE = "5.6.0"
MAX_POINTS = 20_000

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


def legend(show, three_way=False):
    """Legend entries for a Show option. The prediction view gets an uncertain entry when a band is used."""
    entries = [list(e) for e in LEGENDS[show]]
    if three_way and show == "Model prediction":
        entries.append([4, COLORS["uncertain"], "Uncertain (needs investigation)"])
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
            d = uncertain.decide(df["p_static"].to_numpy(), band)
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
    dec = (uncertain.decide(d["p_static"].to_numpy(), band) if band
           else d["pred_static"].astype(int).to_numpy())
    nasa = d["is_static"].astype(int).tolist() if "is_static" in d else [-1] * len(d)
    data = {
        "x": d["longitude"].round(3).tolist(), "y": d["latitude"].round(3).tolist(),
        "p": d["p_static"].round(3).tolist(), "c": d["pred_static"].astype(int).tolist(),
        "k": d["_k"].astype(int).tolist(), "n": nasa, "dec": [int(x) for x in dec],
        "t": d["bright_ti4"].round(1).tolist(), "f": d["frp"].round(1).tolist(),
        "d": dt.tolist(), "dn": list(dt_names), "g": cc.tolist(),
        "gn": [geo.country_name(c) for c in cc_names], "a": pv.tolist(), "an": list(pv_names)}
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


def html(data, box, shapes, show, sig, demo=False, height=700, three_way=False):
    cfg = {"data": data, "box": box, "shapes": shapes, "lines": geo.country_lines(), "demo": demo,
           "legend": legend(show, three_way), "sig": sig}
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
#bar button.on { background: #222; color: #fff; }
#foot { background: #fff; padding: 3px 8px; font-size: 10px; color: #666; height: 18px; overflow: hidden; white-space: nowrap; }
#msg { position: absolute; bottom: 30px; left: 8px; background: rgba(255,255,255,.9); padding: 2px 6px; font-size: 11px; z-index: 2; }
.pop { font: 12px sans-serif; line-height: 1.5; }
#legend { position: absolute; bottom: 30px; right: 8px; z-index: 2; background: rgba(255,255,255,.92); padding: 5px 8px; font: 11px sans-serif; line-height: 1.6; border-radius: 3px; }
#legend i { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; border: 1px solid #fff; }
</style></head>
<body>
<div id="map"></div>
<div id="bar">
  <button id="rot" class="on">Rotate</button>
  <button id="base">Basemap: Satellite</button>
  <button id="tilt">Tilt 3D</button>
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
    img: { type: "raster", tiles: ["https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"], tileSize: 256, maxzoom: 18 },
    lab: { type: "raster", tiles: ["https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}"], tileSize: 256, maxzoom: 18 }
  },
  layers: [
    { id: "img", type: "raster", source: "img" },
    { id: "lab", type: "raster", source: "lab" }
  ]
};
const VOYAGER = "https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json";
let mode = "sat", spin = CFG.box === null, ready = false;

// Camera, basemap and rotation survive a reload of this page (a change of the Show option).
const CAM = "earth_cam", SIG = "earth_sig";
let saved = null;
try {
  if (sessionStorage.getItem(SIG) === CFG.sig) saved = JSON.parse(sessionStorage.getItem(CAM));
  sessionStorage.setItem(SIG, CFG.sig);
} catch (e) { saved = null; }
if (saved) { mode = saved.mode; spin = saved.spin; }

const map = new maplibregl.Map({
  container: "map", style: mode === "sat" ? SAT : VOYAGER, maxPitch: 80, attributionControl: false,
  center: saved ? saved.c : [30, 20], zoom: saved ? saved.z : 1.4,
  pitch: saved ? saved.p : 0, bearing: saved ? saved.b : 0
});
if (mode === "map") document.getElementById("base").textContent = "Basemap: Map";
setInterval(() => {
  try {
    sessionStorage.setItem(CAM, JSON.stringify({ c: map.getCenter().toArray(), z: map.getZoom(), p: map.getPitch(),
      b: map.getBearing(), mode: mode, spin: spin }));
  } catch (e) {}
}, 1000);
map.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), "top-right");
window.earthMap = map;
document.getElementById("rot").classList.toggle("on", spin);
window.earthState = { spinning: () => spin, mode: () => mode };

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
    "circle-radius": ["interpolate", ["linear"], ["zoom"], 1, 2.5, 6, 3.5, 12, 6, 16, 9],
    "circle-opacity": 0.9, "circle-stroke-color": "#ffffff", "circle-stroke-width": 0.5 } });
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

function stats() {
  const lg = document.getElementById("legend");
  let inView = 0;
  try { inView = map.getLayer("pts") ? map.queryRenderedFeatures({ layers: ["pts"] }).length : 0; } catch (e) {}
  window.earthStats = { source: D.x.length, inView: inView };
  lg.innerHTML = CFG.legend.map(l => "<div><i style='background:" + l[1] + "'></i>" + l[2] + "</div>").join("") +
    "<div style='color:#666'>In source " + D.x.length.toLocaleString() + ", in view " + inView.toLocaleString() + "</div>";
}
map.on("idle", stats);
map.on("moveend", stats);

// install is safe to repeat: it runs on style.load, on styledata and on load, so the circles come
// back after any style change, including a diffed one that never fires style.load.
function ensure() {
  if (map.getLayer("pts")) return;
  try {
    if (mode === "map") map.setProjection({ type: "globe" });
    installPoints();
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

// Rotation: slow turn at world zoom, stops at the first touch, drag or zoom.
function stopSpin() { spin = false; document.getElementById("rot").classList.remove("on"); }
const cv = map.getCanvas();
["mousedown", "touchstart", "wheel", "pointerdown"].forEach(e => cv.addEventListener(e, stopSpin, { passive: true }));
setInterval(() => {
  if (!spin || !ready || map.getZoom() > 3.2) return;
  const c = map.getCenter();
  map.jumpTo({ center: [c.lng + 0.08, c.lat] });
}, 40);
document.getElementById("rot").onclick = () => {
  spin = !spin;
  document.getElementById("rot").classList.toggle("on", spin);
};

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

map.on("click", (e) => {
  const r = 8, hits = map.queryRenderedFeatures([[e.point.x - r, e.point.y - r], [e.point.x + r, e.point.y + r]], { layers: ["pts"] });
  if (!hits.length) return;
  const i = hits[0].properties.i;
  const nasa = D.n[i] < 0 ? "" : "NASA label: " + (D.n[i] === 1 ? "static" : "fire") + "<br>";
  const html = "<div class='pop'><b>Decision: " + (D.dec[i] === 1 ? "static heat source" : D.dec[i] === 0 ? "fire" : "uncertain, needs investigation") + "</b><br>" +
    "Probability of static: " + D.p[i] + "<br>" + nasa + "Brightness (I-4): " + D.t[i] + " K<br>FRP: " + D.f[i] + " MW<br>" +
    "Date: " + D.dn[D.d[i]] + "<br>Country: " + D.gn[D.g[i]] + "<br>Province: " + D.an[D.a[i]] + "</div>";
  new maplibregl.Popup({ maxWidth: "260px" }).setLngLat(e.lngLat).setHTML(html).addTo(map);
});
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

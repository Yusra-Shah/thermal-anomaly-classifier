"""Country and province outlines from Natural Earth (1:50m, public domain), kept in data/geo/.

Only the properties the app needs are kept, and a country's outline and provinces are looked
up on demand, so the whole world is never drawn.
"""
import json
from functools import lru_cache

import numpy as np

from .config import ROOT

GEO_DIR = ROOT / "data" / "geo"
COUNTRIES = "ne_50m_admin_0_countries.geojson"
PROVINCES = "ne_50m_admin_1_states_provinces.geojson"


@lru_cache(maxsize=None)
def _load(name):
    path = GEO_DIR / name
    if not path.exists():
        return []
    feats = json.loads(path.read_text(encoding="utf-8"))["features"]
    keep = ("NAME", "ISO_A2_EH", "name", "iso_a2")
    for f in feats:
        f["properties"] = {k: v for k, v in f["properties"].items() if k in keep}
    return feats


def _iso(props):
    code = props.get("ISO_A2_EH") or props.get("iso_a2") or ""
    return "" if code == "-99" else code


@lru_cache(maxsize=None)
def country_names():
    """ISO alpha-2 code to country name. Unknown codes are shown by their code."""
    return {_iso(f["properties"]): f["properties"]["NAME"] for f in _load(COUNTRIES) if _iso(f["properties"])}


def country_name(cc):
    return country_names().get(cc, cc)


def country_feature(cc):
    return next((f for f in _load(COUNTRIES) if _iso(f["properties"]) == cc), None)


def province_features(cc):
    """Admin-1 shapes of a country. Natural Earth 1:50m covers only some countries."""
    return [f for f in _load(PROVINCES) if _iso(f["properties"]) == cc]


def province_at(cc, lons, lats, sample=300):
    """The admin-1 shape of country `cc` that contains most of the given points, or None."""
    feats = province_features(cc)
    if not feats or len(lons) == 0:
        return None
    from shapely import contains_xy
    from shapely.geometry import shape
    rng = np.random.default_rng(0)
    idx = rng.choice(len(lons), min(sample, len(lons)), replace=False)
    x, y = np.asarray(lons)[idx], np.asarray(lats)[idx]
    hits = [int(contains_xy(shape(f["geometry"]), x, y).sum()) for f in feats]
    best = int(np.argmax(hits))
    return feats[best] if hits[best] else None


@lru_cache(maxsize=None)
def country_centroid(cc):
    """(lon, lat) of the largest part of a country, or None."""
    f = country_feature(cc)
    if not f:
        return None
    from shapely.geometry import shape
    g = shape(f["geometry"])
    if g.geom_type == "MultiPolygon":
        g = max(g.geoms, key=lambda p: p.area)
    p = g.representative_point()
    return float(p.x), float(p.y)


def all_countries():
    """Every country outline as one FeatureCollection (about 3 MB)."""
    return collection(_load(COUNTRIES))


def _round(coords, nd=2):
    if coords and isinstance(coords[0], (int, float)):
        return [round(coords[0], nd), round(coords[1], nd)]
    return [_round(c, nd) for c in coords]


def _rings(geom):
    """Outline rings of a polygon or multipolygon, as the lines of a MultiLineString."""
    polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
    return [ring for poly in polys for ring in poly]


@lru_cache(maxsize=None)
def country_lines():
    """All country outlines as lines (not polygons, which the map would have to index and fill),
    with coordinates rounded to 0.01 degree, for the Earth view."""
    feats = [{"type": "Feature", "properties": {},
              "geometry": {"type": "MultiLineString", "coordinates": _round(_rings(f["geometry"]))}}
             for f in _load(COUNTRIES)]
    return collection(feats)


def collection(features):
    return {"type": "FeatureCollection", "features": list(features)}

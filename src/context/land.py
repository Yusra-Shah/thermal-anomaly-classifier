"""Land and fuel of one detection from the land cover module (ESA WorldCover), read when a detection is clicked.

The class is the majority class over the pixel footprint, and a second reading over a 1 km window shows the
surroundings. The fuel line is a plain reading of the class, not a measurement of fuel load or moisture.
"""
import numpy as np

from . import diskcache, landcover

FUEL = {
    10: "forest fuel: tree canopy over litter, burns longer and can spread through the crowns",
    20: "shrub fuel: dense, fast burning in dry weather",
    30: "grass fuel: dries and burns fast, spreads quickly with wind",
    40: "crop fuel: stubble or standing crop, burning is often routine farm practice",
    50: "built-up: little natural fuel, a heat source here is more often industrial or a structure",
    60: "bare or sparse vegetation: little fuel",
    70: "snow and ice: not a vegetation fire",
    80: "permanent water: not a vegetation fire",
    90: "wetland fuel: reeds and herbaceous plants, burns when dry",
    95: "mangrove: dense wet woodland, rarely burns",
    100: "moss and lichen: slow burning, cold regions",
}
WINDOW_KM = 1.0
MAX_TILES = 12
BUDGET_S = 15


def describe(lat, lon, scan_km=None, track_km=None, fetch=None, cache_dir=None):
    """Land cover section for the panel, or {"unavailable": reason}."""
    kw = {"fetch": fetch, "cache_dir": cache_dir, "max_tiles": MAX_TILES, "budget_s": BUDGET_S, "workers": 2}
    try:
        pix = landcover.classify([lon], [lat], [scan_km] if scan_km else None, [track_km] if track_km else None, **kw)
        win = landcover.classify([lon], [lat], [WINDOW_KM], [WINDOW_KM], **kw)
    except landcover.Unavailable as e:
        return {"unavailable": f"land cover could not be reached ({e})"}
    except Exception as e:
        return {"unavailable": f"land cover could not be read ({type(e).__name__})"}
    code, wcode = int(pix["code"][0]), int(win["code"][0])
    if code == landcover.NOT_CLASSIFIED:
        return {"unavailable": "the land cover tile could not be read in this run, try again later"}
    if code == 0:
        return {"unavailable": "no land cover value here (open sea or no data)"}
    name = landcover.CLASSES.get(code, f"class {code}")
    around = (landcover.CLASSES.get(wcode, "") if wcode not in (0, landcover.NOT_CLASSIFIED) else "")
    group = landcover.GROUP_NAMES.get(int(landcover.groups(np.array([code]))[0]), "")
    return {"class": name, "code": code, "group": group, "around_1km": around, "fuel": FUEL.get(code, ""),
            "pixel_m": pix["pixel_m"], "source": landcover.SOURCE, "fetched_at": pix["fetched_at"],
            "note": "Land cover is a classification of 10 m satellite imagery from 2021, about 75 to 80 percent of "
                    "pixels are correct. The fuel line is an inference from the class, not a measurement."}

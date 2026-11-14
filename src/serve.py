"""Scoring with the saved models, shared by the ingest job and the API.

Nothing here changes a model, a threshold, a calibrator or a band: the saved bundles are read as they
are. The instant tier uses the readings of one detection. The history tier also needs the history
features of the place, either computed inside a store (src/ingest.py) or supplied by the caller.
"""
import numpy as np
import pandas as pd

from . import uncertain as U
from .config import MODEL_DIR

HISTORY_FILE = "best_model_history.joblib"
INSTANT_FILE = "best_model_headline.joblib"
NAMES = {U.FIRE: "fire", U.STATIC: "static", U.UNCERTAIN: "uncertain"}


class Models:
    """The two saved bundles with their thresholds, calibrators and bands."""

    def __init__(self, model_dir=None):
        import joblib
        d = model_dir or MODEL_DIR
        self.hist = joblib.load(d / HISTORY_FILE)
        self.inst = joblib.load(d / INSTANT_FILE)
        self.instant_features = list(self.inst["features"])
        self.history_features = list(self.hist["features"])
        self.history_only = [f for f in self.history_features if f not in self.instant_features]
        self.instant_threshold = float(self.inst["threshold"])
        self.point = self.hist.get("default_operating_point", "precision >= 0.90")
        self.history_threshold = float(self.hist["thresholds"][self.point])
        self.bands = {"instant": U.bands_of(self.inst, "best_model_headline_band.json"),
                      "history": U.bands_of(self.hist, "best_model_history_band.json")}
        for tier, b in self.bands.items():
            if not b or not b.get("calibrator"):
                raise RuntimeError(f"the {tier} bundle has no bands and calibrator, run python -m src.uncertain")


def band_calls(cal_p, band):
    """Names (fire, static, uncertain) of the three way decision, or n/a when the band is not reachable."""
    cal_p = np.asarray(cal_p, float)
    if not band or not band.get("reachable", True):
        return np.full(len(cal_p), "n/a", dtype=object)
    d = U.decide(cal_p, band)
    return np.array([NAMES[int(x)] for x in d], dtype=object)


def score_tier(models, tier, X):
    """Raw score, calibrated probability, binary label and both band calls for one tier."""
    bundle = models.inst if tier == "instant" else models.hist
    feats = models.instant_features if tier == "instant" else models.history_features
    thr = models.instant_threshold if tier == "instant" else models.history_threshold
    p = bundle["model"].predict_proba(X[feats])[:, 1]
    cal = U.apply_calibrator(models.bands[tier]["calibrator"], p)
    b = models.bands[tier]
    return {"p": p, "p_cal": cal, "label": np.where(p >= thr, "static", "fire").astype(object),
            "band_default": band_calls(cal, b["default"]), "band_strict": band_calls(cal, b["strict"])}


def score_frame(models, df):
    """Add the scored columns of both tiers to a frame that has every instant and history feature."""
    out = df.copy()
    for tier in ("instant", "history"):
        s = score_tier(models, tier, out)
        out[f"p_{tier}"] = s["p"].astype(np.float32)
        out[f"pc_{tier}"] = s["p_cal"].astype(np.float32)
        out[f"label_{tier}"] = s["label"]
        out[f"{tier}_default"] = s["band_default"]
        out[f"{tier}_strict"] = s["band_strict"]
    return out

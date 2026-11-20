"""Tests for src/datainfo.py, the numbers of the Data and features tab: each one must equal what the saved reports,
the saved models, the code constants and the store say, and a missing source must not produce an invented number.

Run from the project root, one suite per process:  python -m pytest tests/test_datainfo.py -q
"""
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src import datainfo, ingest, live
from src.config import GRID_DEG, MODEL_DIR, REPORT_DIR
from src.context import detail
from src.history_eval import COMPLETE_DAYS, SPLIT_DATE
from src.world import SEEDS


def test_no_number_of_the_page_is_typed_by_hand():
    source = Path(datainfo.__file__).read_text(encoding="utf-8")
    assert not re.search(r"\d{5,}|\d{1,3},\d{3}", source)          # no row count, no static share typed in
    assert "0.0" not in source and "pr_auc = " not in source


def test_regions_match_the_inventory():
    inv = pd.read_csv(REPORT_DIR / "world_inventory.csv")
    t = datainfo.regions()
    assert list(t.columns) == ["Region", "Country", "Detections", "Static share"]
    body = t[t["Region"] != "All regions"]
    assert body["Region"].tolist() == inv["region"].tolist()
    assert body["Detections"].tolist() == inv["rows_kept"].tolist()
    assert np.allclose(body["Static share"], inv["static_share"].round(4))
    total = t[t["Region"] == "All regions"].iloc[0]
    assert total["Detections"] == inv["rows_kept"].sum()
    assert abs(total["Static share"] - (inv["rows_kept"] * inv["static_share"]).sum() / inv["rows_kept"].sum()) < 1e-4


def test_labels_match_the_inventory_and_say_what_is_kept():
    inv = pd.read_csv(REPORT_DIR / "world_inventory.csv")
    t = datainfo.labels().set_index("Type")
    assert t.loc[0, "In this project"].startswith("kept") and "fire" in t.loc[0, "In this project"]
    assert t.loc[2, "In this project"].startswith("kept") and "static" in t.loc[2, "In this project"]
    assert t.loc[1, "In this project"] == "dropped" and t.loc[3, "In this project"] == "dropped"
    assert t.loc[0, "Detections in the archive files"] == f"{int(inv['type0'].sum()):,}"
    assert t.loc[2, "Detections in the archive files"] == f"{int(inv['type2'].sum()):,}"
    assert str(int(inv["other_types"].sum())) .replace("", "") and f"{int(inv['other_types'].sum()):,}" in \
        t.loc[1, "Detections in the archive files"]


def test_features_come_from_the_saved_models_and_the_panel_labels():
    inst = datainfo.feature_table("instant")
    hist = datainfo.feature_table("history")
    assert inst["Feature"].tolist() == list(joblib.load(MODEL_DIR / "best_model_headline.joblib")["features"])
    assert hist["Feature"].tolist() == list(joblib.load(MODEL_DIR / "best_model_history.joblib")["features"])
    assert len(inst) == 10 and len(hist) == 29                           # the tiers as shipped
    assert set(inst["Feature"]) <= set(hist["Feature"])
    for t in (inst, hist):
        assert t["Meaning"].tolist() == [detail.LABELS.get(f, f) for f in t["Feature"]]
        assert (t["Meaning"] != t["Feature"]).all()                      # every feature has a plain meaning


def test_checks_equal_the_saved_reports():
    hist = pd.read_csv(REPORT_DIR / "world_history.csv")
    inst = pd.read_csv(REPORT_DIR / "world_instant.csv")
    night = pd.read_csv(REPORT_DIR / "daynight_vs_baseline.csv")
    rows = datainfo.check_rows()
    g = rows.loc["Grouped split (seed 42)"]
    h = hist[(hist["split"] == "grouped seed 42")]
    assert g["History tier"] == round(float(h[h["tier"] == "history"]["pr_auc"].iloc[0]), 3)
    assert g["Majority baseline"] == round(float(h[h["tier"] == "history"]["majority_pr_auc"].iloc[0]), 3)
    assert g["One line rule"] == round(float(h[h["tier"].str.startswith("rule")]["pr_auc"].iloc[0]), 3)
    assert g["Instant tier"] == round(float(inst[(inst["split"] == "grouped seed 42")]["pr_auc"].iloc[0]), 3)
    n = night[(night["split"] == "grouped seed 42") & (night["tier"] == "history")]
    assert g["Night baseline"] == round(float(n["baseline_pr_auc"].iloc[0]), 3)
    lo = rows.loc["Leave one region out (mean of the regions)"]
    sel = hist[hist["split"].str.startswith("leave out") & (hist["tier"] == "history")]
    assert lo["History tier"] == round(float(sel["pr_auc"].mean()), 3) and len(sel) == 8
    assert rows.loc["Temporal split", "History tier"] == round(
        float(hist[(hist["split"] == "temporal") & (hist["tier"] == "history")]["pr_auc"].iloc[0]), 3)


def test_check_notes_use_the_code_constants():
    text = " ".join(datainfo.check_notes())
    h = datainfo.headline()
    assert f"{h['test_rows'] / h['rows']:.0%}" in text and f"{GRID_DEG} degree" in text
    assert f"{SPLIT_DATE:%Y-%m-%d}" in text and f"{COMPLETE_DAYS} days" in text
    assert f"{len(SEEDS)} random splits" in text and all(str(s) in text for s in SEEDS)
    for word in ("Majority baseline", "Night baseline", "One line rule", "Leave one region out", "out of fold"):
        assert word in text


def test_sources_and_the_end_of_delivery():
    h = datainfo.headline()
    t = datainfo.sources(None)
    assert t["Type field"].tolist() == ["yes", "no"]
    assert f"{h['rows']:,}" in t["Holds"].iloc[0] and h["date_min"] in t["Holds"].iloc[0]
    assert t["Holds"].iloc[1] == "no live store on this machine"
    assert datainfo.SNPP_END == "2 November 2026"


def test_live_store_numbers_come_from_the_store(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "LIVE_DIR", tmp_path)
    monkeypatch.delenv(ingest.STORE_ENV, raising=False)
    assert datainfo.live_store() is None
    df = pd.DataFrame({
        "sensor": ["NOAA-20", "NOAA-20", "NOAA-21", "NOAA-21"],
        "region": ["Brazil", "Europe", "Brazil", "Oceania South"],
        "acq_date": pd.to_datetime(["2026-11-01", "2026-11-02", "2026-11-03", "2026-11-03"]),
        "timestamp": pd.to_datetime(["2026-11-01 01:00", "2026-11-02 05:30", "2026-11-03 07:15", "2026-11-03 09:45"]),
        "ingest_time": pd.to_datetime(["2026-11-03 10:00"] * 4)})
    ingest.write_store(df)
    st = datainfo.live_store()
    assert st["rows"] == 4 and st["regions"] == ["Brazil", "Europe", "Oceania South"]
    assert st["untrained"] == ["Europe", "Oceania South"] and st["sensors"] == ["NOAA-20", "NOAA-21"]
    assert st["newest"] == pd.Timestamp("2026-11-03 09:45") and st["oldest"] == pd.Timestamp("2026-11-01")
    src = datainfo.sources(st)
    assert "4 stored detections" in src["Holds"].iloc[1] and "2026-11-03 09:45 UTC" in src["Holds"].iloc[1]


def test_a_missing_report_gives_nothing_not_a_made_up_number(tmp_path, monkeypatch):
    monkeypatch.setattr(datainfo, "REPORT_DIR", tmp_path)
    monkeypatch.setattr(datainfo, "MODEL_DIR", tmp_path)
    assert datainfo.regions() is None and datainfo.check_rows() is None and datainfo.headline() is None
    assert datainfo.feature_table("instant") is None and datainfo.feature_table("history") is None
    assert datainfo.sources(None)["Holds"].iloc[0] == "not available"
    counts = datainfo.labels()["Detections in the archive files"]
    assert not counts.str.contains(r"\d{2}|\d,\d").any()                # no count without the inventory

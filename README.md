# Fire Hotspot Classifier

Classifies NASA FIRMS VIIRS 375 m thermal detections as a likely vegetation fire or a likely static heat source (industrial site, gas flare, kiln). There are two tiers: an instant tier that uses the readings of one detection, and a history tier that also uses past detections of the same place.

## Problem

FIRMS reports every thermal anomaly the satellite sees. Many of them are not fires but heat sources that stay in the same place. The task is to tell them apart per detection, with the static class as the one that matters (precision, recall, F1 and PR-AUC are reported for it).

## Data and labels

- Training data: three FIRMS archive CSV files in `data/raw/` (Suomi NPP, VIIRS 375 m), one per region: South Asia, Southeast Asia, North and Central America. They cover 2025-01-01 to 2025-06-30.
- Label: the NASA `type` field in the archive files. Static is type 2, fire is type 0. Volcano (1) and offshore (3) detections are dropped. This leaves 257,841 detections, 14.3 percent static (`reports/metrics_headline.json`).
- The type label is produced by NASA's own algorithm, which is based on repeated detections at the same place. It is not independent ground truth.
- Near real time files (for example `SUOMI_VIIRS_C2_Global_7d.csv` in `data/nrt/`) have no `type` column, so there is no label to evaluate against. They can be scored, and the result is only a sanity check.
- Data files are not tracked in git.

## The two tiers

1. Instant (single reading): brightness (I-4 and I-5), their difference, fire radiative power, pixel size, confidence and day or night. No coordinates, no hour of day, no repeat counts.
2. With history (past detections): the instant features plus, for the same grid cell (0.01 degrees), counts of distinct days with detections over the prior 7, 30, 60 and 90 days, the same over the 3x3 neighbouring cells, recency and persistence, and the mean, standard deviation and deviation of brightness and power over the prior 30 days. Only detections on days strictly before the detection are used. `tests/test_history.py` checks this.

The history tier is not a single reading result. It needs past detections of the same place, works best with at least 30 days of data, and predicts NASA's repeat based static flag.

## Evaluation rules

- Splits are grouped by grid cell, so a place is never in both train and test.
- No coordinates and no `hour_utc` (it acts as a location proxy: see `reports/summary.md`).
- No repeat counts from the future.
- Decision thresholds are chosen on out-of-fold predictions from the training part only.
- Checks used: five grouped splits (seeds 42 to 46), leave one region out, and a temporal split (train before 2025-05-01, test 2025-05-01 to 2025-06-30).
- A one line rule (static if `distinct_days_prior_30 >= k`, k chosen on the training part) is reported next to the models.

## Results

Static class, all rows, mean (std) over five grouped splits, and the temporal split. Values come from `reports/history_eval_a.csv`, `reports/history_eval_c.csv`, `reports/history2_eval_a.csv` and `reports/history2_eval_c.csv`. The majority baseline PR-AUC is about 0.142 on the grouped splits and 0.183 on the temporal split.

| Model | 5 splits PR-AUC | 5 splits precision | 5 splits recall | 5 splits F1 | Temporal PR-AUC | Temporal precision | Temporal recall | Temporal F1 |
|---|---|---|---|---|---|---|---|---|
| Rule: distinct_days_prior_30 >= k | 0.777 (0.040) | 0.765 (0.031) | 0.811 (0.017) | 0.787 (0.023) | 0.840 | 0.763 | 0.872 | 0.814 |
| Instant (A), Random Forest | 0.520 (0.027) | 0.442 (0.027) | 0.827 (0.013) | 0.576 (0.024) | 0.472 | 0.434 | 0.782 | 0.558 |
| Instant (A), XGBoost | 0.523 (0.037) | 0.438 (0.025) | 0.846 (0.020) | 0.577 (0.024) | 0.492 | 0.439 | 0.809 | 0.569 |
| History (D), Random Forest | 0.929 (0.019) | 0.845 (0.021) | 0.911 (0.017) | 0.877 (0.018) | 0.943 | 0.832 | 0.941 | 0.883 |
| History (D), XGBoost | 0.933 (0.017) | 0.840 (0.019) | 0.914 (0.019) | 0.875 (0.018) | 0.943 | 0.836 | 0.945 | 0.887 |
| History (D), XGBoost tuned | 0.936 (0.017) | 0.845 (0.026) | 0.915 (0.016) | 0.879 (0.020) | 0.945 | 0.837 | 0.946 | 0.888 |

The saved history model (default XGBoost, setup D, seed 42 split) has these held out values (`reports/history_final_metrics.json`):

| Operating point | Threshold | Precision | Recall |
|---|---|---|---|
| F1 optimal | 0.8148 | 0.8537 | 0.9132 |
| precision >= 0.90 | 0.914 | 0.8944 | 0.84 |
| precision >= 0.95 | 0.9731 | 0.9303 | 0.5965 |

Leave one region out, the earlier instant results and the history comparisons are in `reports/summary.md`, `reports/summary_history.md` and `reports/summary_history2.md`.

## Limitations

- Data covers January to June 2025 only, in three regions.
- The history tier needs weeks of past data for the same place and does not apply to a first-time detection.
- History features nearly restate how NASA's static flag is built, so high history scores show that past detections predict NASA's flag, not that the flag is correct. The one line rule already gets part of the way.
- The instant tier is weak on its own. Static precision is below 0.5 in the five grouped splits.
- Labels are NASA's. There is no independent ground truth.
- Differences between history setups are small compared with the split to split spread.

## Live map

The "Live map" tab shows recent VIIRS S-NPP near real time detections for one region at a time, scored by the history model or the instant model.

- Map key: get a free key on the FIRMS website (Get MAP_KEY). Create a file named `.env` in the project root with one line, `FIRMS_MAP_KEY=your_key`, or set the same variable in the environment. `.env` is ignored by git. The key is never printed or logged. Without a key the tab says so and shows the saved snapshot.
- Refresh: the Refresh data button fetches the last 60 days from the FIRMS area API in chunks of at most 5 days (the documented limit), then only the missing days and the latest 3 days on later refreshes. The first fetch is slower. Detections are stored per region and day in `data/live/` and the 60 day store is written to `data/live/snapshot_<region>.parquet` with the fetch time.
- Offline: tick "Offline (saved snapshot)" or run without a key to load the snapshot. Features and scores are recomputed from it, so nothing is fetched.
- Demo: `python -m src.live_demo` builds snapshots for the three training regions from `data/raw/` (last 60 days, dates shifted so the newest day is yesterday, flagged as demo). The tab shows a notice that this is not current data. It is for testing without a key.
- Features: the same functions as in training, applied to the whole 60 day store as one file, so history uses only strictly earlier days.
- Geography: country, province or state and nearest city come from the offline `reverse_geocoder` package, computed once per grid cell and cached in `data/live/`.
- Regions: South Asia, Southeast Asia and North and Central America use the extents of the training files. "Europe (not in training)" is shown with an unvalidated banner.
- Limitations: data is about 3 hours behind the satellite. There is no ground truth on live data, so the map shows predictions only (see the Results tab for measured performance). The history model was trained on 181 day files and is applied to a 60 day store, so history counts are capped lower than in training. The source is `VIIRS_SNPP_NRT` to match the training satellite. If NASA stops delivering that product the data age counter grows and the tab warns. Switching the source in `src/live.py` to another satellite is not validated.
- Maps: all three map views (Map, Classify a new file, Live map) share one `draw_map` function. The basemap is the Carto Voyager vector style (country and state borders, place labels by zoom, needs internet; without it the points and outlines still draw). Selecting a country draws its outline and its provinces, selecting a province highlights it. Natural Earth 1:50m has province shapes only for Russia, the United States, India, Indonesia, China, Brazil, Canada, Australia and South Africa.
- Boundaries: country and province outlines in `data/geo/` are from Natural Earth (https://www.naturalearthdata.com), public domain, 1:50m scale.
- Tests: `python -m tests.test_live` (fake API responses, no network).

## Run

    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt
    # torch, CPU build:
    pip install torch --index-url https://download.pytorch.org/whl/cpu

    # optional, for the Live map tab: put FIRMS_MAP_KEY=... in a .env file in the project root

    # put the archive CSV files in data/raw/
    python -m src.train --tag headline          # instant tier
    python -m src.final_history                 # history tier, saves the model and scored rows
    python -m src.history_eval --part a         # evaluations (parts a, b, c)
    python -m src.history2_eval --part a        # history v2 and tuning (parts a, b, c)
    python -m src.history2_ops                  # operating points and curve
    python -m src.region_eval                   # instant tier, leave one region out
    python -m src.nrt_score                     # score data/nrt/SUOMI_VIIRS_C2_Global_7d.csv
    python -m tests.test_history                # leakage test
    python -m tests.test_live                   # live data tests, no network
    streamlit run app/app.py

## Structure

    src/       data loading, labelling, features, history features, models, evaluation
    app/       Streamlit app (map, upload, results)
    tests/     leakage test for the history features, synthetic data generator
    reports/   tables, figures and summaries written by the scripts
    models/    saved models (not tracked)
    data/      raw, nrt and processed data (not tracked)

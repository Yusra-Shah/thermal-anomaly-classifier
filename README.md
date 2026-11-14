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

## Earth

The "Earth" tab is one interactive Earth built with MapLibre GL JS 5.6.0 (globe projection, loaded from a CDN) and embedded with `st.components.v1.html`. At world zoom it is a globe that turns slowly until you drag, zoom or touch it (the Rotate button resumes), and it flattens into a normal map and then city level as you zoom in. The default basemap is Esri World Imagery with Esri place labels and a Map switch for the Carto Voyager style, terrain comes from AWS terrarium tiles with hillshade, and Tilt 3D or a right mouse or Ctrl drag tilts the view. It needs internet for the library and all tiles; without it nothing is drawn. Imagery: Esri, Maxar, Earthstar Geographics and the GIS User Community. Terrain: Mapzen, AWS Terrain Tiles. Map style: Carto, OpenStreetMap contributors. Boundaries (`data/geo/`): Natural Earth, public domain. The data source is the archive (the Map tab data, same tier and operating point) or the live snapshot, and the four geography filters fly the camera to the selection and outline the chosen country and, where Natural Earth has them, its provinces. Detections are circles, at most 20,000 (a random sample, with a caption giving the real count; pick a province or city to see every detection of a place), and a click opens a popup with class, probability of static, NASA label (archive only), brightness, FRP, date, country and province. The Show switch offers Raw detections, Model prediction (red fire, grey static), Ground truth and Errors only (false static and false fire in two colours, with counts); the legend in the map follows the option, and the camera is kept when the option changes. Ground truth is NASA's own repeat based static flag, not independent truth, and it does not exist for live data, so Ground truth and Errors only are disabled there. The live source reads the near real time feeds `VIIRS_NOAA20_NRT` and `VIIRS_NOAA21_NRT` from the FIRMS area API (Suomi NPP no longer delivers near real time data): put `FIRMS_MAP_KEY=your_key` in a `.env` file in the project root (git ignores it; the key is never printed), press Refresh data to fetch the last 60 days of both satellites in chunks of at most 5 days with a one second pause between calls, far below the limit of 5000 transactions per 10 minutes (later refreshes fetch only missing days and the latest 3), or tick Offline to use the saved snapshot. Each satellite has its own rolling store under `data/live/`, every row keeps the satellite name, and the history features are built inside one satellite only, never mixing satellites. The models, calibrator and bands are the Suomi NPP trained ones. These feeds have no type column, so there is no ground truth and only predictions are shown. The tab lists the acquisition date range, the rows per satellite and the time of the last fetch, and a genuine snapshot shows no demo warning. `python -m src.live_demo` builds demo snapshots from `data/raw/` with dates shifted to look recent, and the tab then shows a red demo warning (a real refresh replaces the snapshot, so rebuild the demo when you need it offline). Live data is about 3 hours behind the satellite. The models were trained on Suomi NPP and validated on NOAA-20 and NOAA-21 archive data, but near real time data has not been validated against the archive. `python -m src.live_check` fetches the feeds and prints sanity checks (predicted static share per region and satellite, how many cells with at least 10 detections in the last 30 days are predicted static, and the overlap of the two satellites' predicted static cells) into `reports/live_check.csv`. Regions: South Asia, Southeast Asia and North and Central America, plus "Europe (not in training)" with an unvalidated banner.

## Uncertain band

Besides static or fire, each model tier can answer uncertain (needs investigation). Scores are first calibrated (Platt or isotonic, fitted on the out of fold training scores and chosen by cross validated Brier score). The band is then the pair of thresholds on the calibrated score with the highest coverage whose accuracy on the decided out of fold training points is at least 0.96 (default) or 0.98 (strict): static at or above t_high, fire at or below t_low, uncertain in between. Both come from out of fold training scores only, and the model bundles store them as `band`, `band_strict` and `calibrator`. `python -m src.uncertain` evaluates the bands on the grouped split, the temporal split and leave one region out next to the one line rule and writes `reports/calibration_*.csv` and `reports/uncertain_*.csv` with plots. The first definition (static precision at least 0.90 and fire side precision at least 0.95) is kept in the reports as a negative result, labelled "precision based": fire is the majority class, so the fire side target is met almost everywhere and the history band came out empty. Honest limits: for the history tier the 0.96 target is already met by the forced binary decision, so the default band is empty (nothing is uncertain) and only the strict 0.98 band abstains; the instant tier reaches 0.96 only by calling almost everything fire or uncertain and rarely calls static; accuracy on new regions and periods can fall below the target chosen on the training scores; and NASA's flag is the label, not independent truth. The Earth tab shows uncertain in amber in the Model prediction view, with a selector for the default or strict band, and Model metrics has the table and the coverage curves.

## NOAA-20 transfer

`python -m src.noaa20` downloads VIIRS_NOAA20_SP (with the type column) for the three region boxes from 2025-01-01 to 2025-06-30 in 5 day chunks, 111 calls with a pause and resume support, and cuts the rows to Pakistan, Indonesia and the United States, the countries of the Suomi NPP files, with the polygons in `data/geo/`. The map key is read from `.env` and every request error prints only the exception class and the HTTP status. `python -m src.noaa20_transfer` builds NOAA-20 features and history inside each NOAA-20 file only, with the same label rule, and tests the Suomi NPP models with their thresholds, calibrators and uncertain bands unchanged. Test sets are NOAA-20 detections in cells that were not Suomi NPP training cells (cell disjoint) and, labelled as easier, detections in cells that were (not cell disjoint). Results are in `reports/noaa20_transfer.csv` with a summary, the sensor comparison and the row counts, and in the Model metrics tab. Honest limits: the labels are NASA's repeat based flag, not independent truth; the two satellites fly the same type of instrument over the same places in the same days, only about 22 minutes apart at matched positions, so this is a transfer between near twins and says little about a different sensor; cell disjoint cells are mostly fire cells, so their static share is low and the scores must be read against the majority baseline; and the country cut with Natural Earth polygons differs slightly from the archive download.

NOAA-21 transfer. `python -m src.noaa20 --sat noaa21` and `python -m src.noaa20_transfer noaa21` repeat the NOAA-20 experiment for `VIIRS_NOAA21_SP` with the same boxes, dates, country cut and rules (files in `data/noaa20/noaa21/`, results in `reports/noaa21_transfer.csv`, `reports/noaa21_sensor_comparison.csv` and `reports/noaa21_transfer_summary.md`, and next to the NOAA-20 rows in the Model metrics tab). On cell disjoint cells the history model reaches static PR-AUC 0.914 (NOAA-20 0.901, Suomi NPP held out reference 0.932, majority baseline 0.070), the instant model 0.359 (NOAA-20 0.371, with the XGBoost instant model described below; the random forest it replaced gave 0.334 and 0.354) and the one line rule 0.726 (NOAA-20 0.731), so NOAA-21 behaves like NOAA-20. On matched positions the three sensors show no large brightness bias (I-4 median about minus 1 K, but single detections differ by about 10 K on average), and the acquisition time offset is about 51 minutes between NOAA-21 and NOAA-20 and about minus 26 minutes between NOAA-21 and Suomi NPP on most days, with a spread across days. Honest limits are the same as for NOAA-20: NASA's repeat based flag is the label, the satellites are near twins, cell disjoint cells are mostly fire cells, and the sensor comparison uses six sampled days. The result says the Suomi NPP trained models carry over to the two newer satellites in the archive, not that near real time data behaves the same.

Day and night. `python -m src.daynight` (stages tables, xgb, transfer, ship) tests how much of the instant tier is a night detector, with results in `reports/daynight_*.csv` and in Model metrics. Against a trivial baseline that calls every night detection static, the instant tier is only slightly better: static PR-AUC 0.507 against 0.391 on the grouped held out split, 0.492 against 0.378 on the temporal split, 0.371 against 0.264 on NOAA-20 cell disjoint cells and 0.359 against 0.261 on NOAA-21 (the baseline's own PR-AUC is high because its score is 0 or 1 and about 40 percent of night detections are static), and static F1 is only 0.012 to 0.018 above the baseline (0.585 against 0.571 on the grouped split). By day (static share 0.026) the instant tier calls nothing static, with PR-AUC 0.052; by night (static share 0.423) it has PR-AUC 0.537 and recall 0.92 at precision 0.46. On leave one region out the random forest was below the baseline on F1 in two of three regions. The history tier is far above the baseline (grouped PR-AUC 0.932, day 0.916, night 0.932) and so is the one line rule (0.780). In an ablation with XGBoost (100 trees, grouped split, thresholds from out of fold scores) removing is_night alone costs little (PR-AUC 0.499 against 0.507) because brightness, frp and the pixel size also carry the day or night signal; removing frp and frp_per_area costs most (0.468), confidence_ord nothing. Permutation importance on the held out split ranks is_night first (PR-AUC drop 0.161), then bright_ti5 (0.059) and frp (0.057). The XGBoost instant model was within 0.01 of the random forest on the grouped split (0.507 against 0.515) and above it on the temporal split (0.492 against 0.472), so it replaced the random forest as the instant model (270 KB instead of 600 MB, same threshold, calibrator and band procedure; the old bundle is kept in `model_backups/` next to the project folder). Limits: the label is NASA's repeat based flag, the baseline and the models share the same labels, this says the instant tier adds little beyond the night flag and not that it is useless on a single reading, and the stratified day numbers rest on few static detections.

Why the instant tier flags so many live detections as static. `python -m src.nrt_vs_sp --download` and `python -m src.nrt_analysis` compare NOAA-21 products with the same boxes and country cut (results in `reports/nrt_vs_sp.csv`, files in `data/noaa20/nrtcheck/`). For 2026-06-01 to 2026-07-31 the `VIIRS_NOAA21_NRT` and `VIIRS_NOAA21_SP` downloads hold the same detections: same row counts per region, every detection matched within 0.006 degrees and 30 minutes, and bright_ti4, bright_ti5, bright_diff, frp, scan, track, confidence and day or night identical on all matched pairs, so both products give the same instant and history shares. The confidence column uses the same categories (l, n, h) and the same mapping to 0, 1 and 2 in both. So no product difference was found, and no fix was needed. What moves the instant share is the day or night mix: the instant tier flags almost no daytime detections as static (0.1 to 0.4 percent) and 60 to 90 percent of night ones, in the archive too (25 percent flagged overall at a night share of 31 percent, while NASA's static share there is 14 percent). Live Southeast Asia has a night share of 51 percent against 13 percent in 2025-09-11 to 2025-11-09 SP data, and its instant share is 41.7 percent against 11.4 percent; at the SP night mix the live share would be 10.7 percent. The same standardisation brings South Asia to 9.2 percent against 9.4 percent and North and Central America to 43.6 percent against 43.1 percent, so the high shares are a property of the instant model plus the season and year, not of near real time data. Limits: SP data for August to November 2026 does not exist yet, so the live feed itself could not be compared with SP for those weeks, and why 2026 Southeast Asia has so many night detections was not investigated. Treat the instant tier's static label as a night detection flag, and prefer the history tier on live data. The instant shares in that paragraph and in `reports/nrt_vs_sp.csv` and `reports/live_check.csv` come from the random forest instant tier that has since been replaced.

## Live store and API

`python -m src.ingest --once` fetches the latest NOAA-20 and NOAA-21 near real time detections for the three trained regions (day ranges of 1 to 5, a short wait between calls and a budget below 5000 transactions per 10 minutes), scores the new rows with the saved models, calibrators and bands, and merges them into one store, `data/live/store.parquet` (git ignores `data/live`). The store is written atomically, rows are unique on sensor, latitude, longitude, acq_date and acq_time so a second run adds nothing, and only the last 90 days are kept for the history features. It holds the features, both scores, the calibrated probabilities, the labels, the default and strict band calls and the ingest time. The command prints only counts (rows fetched, added and kept per sensor). A failed call prints only the exception class and the HTTP status, never a URL, a header, a traceback or the key, which is read from `FIRMS_MAP_KEY` (environment or `.env`). `python -m src.ingest --loop --every-minutes 180` repeats it, and `python -m src.ingest --import-live` is a one off that takes the rolling stores of the older fetch code into the store. To run it every few hours on Windows, create a Windows Task Scheduler task that runs `.venv\Scripts\python.exe -m src.ingest --once` from the project folder, repeating every 3 hours. When the store exists, the Live source of the Earth tab reads it and shows the last ingest time and the rows per sensor, and it falls back to the old snapshot behaviour when the store is missing.

Start the API with `uvicorn api.main:app` (docs at `/docs`). `GET /health` gives the status, the last ingest time and the row count, `GET /model` the model files, thresholds, band settings and the headline metrics from the existing reports, `POST /predict` up to 5000 detections (validated, one answer each with the label, the calibrated probability, the default and strict band calls and the tier used), and `GET /live/latest` the newest scored rows (parameters `hours`, `sensor`, an optional box, at most 20000 rows). `/predict` uses the history tier when history fields are supplied or when the cell exists in the store (the history of its latest stored detection), and the instant tier otherwise, and each answer says which.

    curl http://127.0.0.1:8000/health
    curl "http://127.0.0.1:8000/live/latest?hours=24&sensor=NOAA-20&min_lat=24&min_lon=61&max_lat=36&max_lon=76&limit=100"
    curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"detections\": [{\"latitude\": 28.6, \"longitude\": 70.1, \"acq_date\": \"2026-11-01\", \"acq_time\": 742, \"bright_ti4\": 330.5, \"bright_ti5\": 295.0, \"frp\": 3.2, \"scan\": 0.45, \"track\": 0.5, \"confidence\": \"n\", \"daynight\": \"N\"}]}"

The models are not retrained from live data: live data has no type flag, so there is no label to learn from. Live results are unvalidated. Tests: `python -m pytest tests/test_ingest.py -q` and `python -m pytest tests/test_api.py -q`, one per process.

## Run

    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt
    # torch, CPU build:
    pip install torch --index-url https://download.pytorch.org/whl/cpu

    # optional, for the live source of the Earth tab: put FIRMS_MAP_KEY=... in a .env file in the project root

    # put the archive CSV files in data/raw/
    python -m src.train --tag headline          # instant tier
    python -m src.final_history                 # history tier, saves the model and scored rows
    python -m src.history_eval --part a         # evaluations (parts a, b, c)
    python -m src.history2_eval --part a        # history v2 and tuning (parts a, b, c)
    python -m src.history2_ops                  # operating points and curve
    python -m src.region_eval                   # instant tier, leave one region out
    python -m src.nrt_score                     # score data/nrt/SUOMI_VIIRS_C2_Global_7d.csv
    python -m tests.test_history                # leakage test
    python -m tests.test_live app_earth         # live and Earth tests, no network, one suite per process
                                                # suites: unit, app_store, app_live, app_demo, app_geo, app_earth
    python -m src.ingest --once                 # live store, needs the key
    uvicorn api.main:app                        # prediction API
    streamlit run app/app.py

## Structure

    src/       data loading, labelling, features, history features, models, evaluation
    app/       Streamlit app (map, upload, results, Earth)
    api/       FastAPI prediction API (health, model, predict, live latest)
    tests/     leakage test for the history features, synthetic data generator
    reports/   tables, figures and summaries written by the scripts
    models/    saved models (not tracked)
    data/      raw, nrt and processed data (not tracked)

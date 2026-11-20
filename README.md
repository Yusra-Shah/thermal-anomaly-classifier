# Fire Hotspot Classifier

Classifies NASA FIRMS VIIRS 375 m thermal detections as a likely vegetation fire or a likely static heat source (industrial site, gas flare, kiln). There are two tiers: an instant tier that uses the readings of one detection, and a history tier that also uses past detections of the same place.

## Problem

FIRMS reports every thermal anomaly the satellite sees. Many of them are not fires but heat sources that stay in the same place. The task is to tell them apart per detection, with the static class as the one that matters (precision, recall, F1 and PR-AUC are reported for it).

## Data and labels

- Training data: eight FIRMS archive CSV files in `data/raw/` (Suomi NPP, VIIRS 375 m), one per region: South Asia (Pakistan), Southeast Asia (Indonesia), North and Central America (United States), Brazil, Australia, India, Nigeria and DR Congo. All cover 2025-01-01 to 2025-06-30. `src/regions.py` maps each file to its region and `reports/world_inventory.csv` lists them.
- Label: the NASA `type` field in the archive files. Static is type 2, fire is type 0. Volcano (1) and offshore (3) detections are dropped. This leaves 2,349,116 detections, 4.3 percent static (`reports/metrics_headline.json`); the static share runs from 0.05 percent in DR Congo to 27 percent in Southeast Asia.
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

All regions together (eight regions, January to June 2025; `reports/world_history.csv`, `reports/world_instant.csv`). Static class, mean (std) over five grouped splits by grid cell, and the temporal split (train before 2025-05-01, test from 2025-05-01 with 30 or more days of history). The majority baseline PR-AUC is 0.042 on the grouped splits and 0.036 on the temporal split. Both tiers are XGBoost with 100 trees, thresholds at the F1 optimum of out of fold training scores.

| Model | 5 splits PR-AUC | 5 splits precision | 5 splits recall | 5 splits F1 | Temporal PR-AUC | Temporal precision | Temporal recall | Temporal F1 |
|---|---|---|---|---|---|---|---|---|
| Rule: distinct_days_prior_30 >= k | 0.797 (0.020) | 0.820 (0.016) | 0.785 (0.012) | 0.802 (0.013) | 0.774 | 0.696 | 0.841 | 0.762 |
| Instant, XGBoost | 0.246 (0.015) | 0.240 (0.015) | 0.584 (0.011) | 0.340 (0.016) | 0.186 | 0.208 | 0.502 | 0.294 |
| History (setup D), XGBoost | 0.921 (0.016) | 0.852 (0.016) | 0.910 (0.018) | 0.880 (0.016) | 0.927 | 0.828 | 0.951 | 0.885 |

The saved history model (default XGBoost, setup D, seed 42 split, 470,027 held out rows) has these held out values (`reports/history_final_metrics.json`). The precision >= 0.95 threshold from the training scores (0.9993) is so high that it predicts no static detection on the held out rows.

| Operating point | Threshold | Precision | Recall |
|---|---|---|---|
| F1 optimal | 0.9412 | 0.8517 | 0.9141 |
| precision >= 0.90 | 0.9845 | 0.8931 | 0.781 |
| precision >= 0.95 | 0.9993 | 0.0 | 0.0 |

Leave one region out, train on all other regions and test on the held out one (`reports/world_regions.csv`, `reports/world_summary.md`). History is far above the baseline and the rule in six of eight regions and only a little above the rule in India and DR Congo, and the instant tier is weak everywhere; DR Congo has 365 static detections out of 714,034, so its scores rest on very few static rows.

| Held out region | Rows | Static share | History PR-AUC | History F1 | Instant PR-AUC | Instant F1 | Rule PR-AUC | Rule F1 |
|---|---|---|---|---|---|---|---|---|
| South Asia | 46,909 | 0.0297 | 0.881 | 0.816 | 0.148 | 0.237 | 0.701 | 0.699 |
| Southeast Asia | 14,034 | 0.2724 | 0.845 | 0.792 | 0.578 | 0.548 | 0.780 | 0.730 |
| North and Central America | 196,898 | 0.1611 | 0.920 | 0.864 | 0.510 | 0.423 | 0.786 | 0.767 |
| Brazil | 117,776 | 0.0616 | 0.937 | 0.882 | 0.177 | 0.272 | 0.729 | 0.624 |
| Australia | 412,943 | 0.0128 | 0.934 | 0.892 | 0.038 | 0.073 | 0.823 | 0.784 |
| India | 513,677 | 0.0848 | 0.890 | 0.892 | 0.298 | 0.281 | 0.870 | 0.862 |
| Nigeria | 332,845 | 0.0194 | 0.793 | 0.777 | 0.079 | 0.143 | 0.634 | 0.737 |
| DR Congo | 714,034 | 0.0005 | 0.364 | 0.412 | 0.026 | 0.032 | 0.345 | 0.180 |

Against the three region study, for the original regions: history PR-AUC and F1 rose for South Asia and North and Central America and fell slightly for Southeast Asia (0.856 to 0.845 and 0.806 to 0.792); instant F1 fell in all three (for example 0.550 to 0.423 in North and Central America); the rule's F1 also fell in all three (its k is picked on the training part, which now holds seven regions). The full list of numbers that got worse is in `reports/world_summary.md`.

The tables below this point and the files they come from (`reports/history_eval_a.csv`, `reports/history_eval_c.csv`, `reports/history2_eval_a.csv`, `reports/history2_eval_c.csv`) are the earlier three region study (257,841 detections, 14.3 percent static) and were not recomputed.

Earlier three region study. Static class, all rows, mean (std) over five grouped splits, and the temporal split. The majority baseline PR-AUC is about 0.142 on the grouped splits and 0.183 on the temporal split.

| Model | 5 splits PR-AUC | 5 splits precision | 5 splits recall | 5 splits F1 | Temporal PR-AUC | Temporal precision | Temporal recall | Temporal F1 |
|---|---|---|---|---|---|---|---|---|
| Rule: distinct_days_prior_30 >= k | 0.777 (0.040) | 0.765 (0.031) | 0.811 (0.017) | 0.787 (0.023) | 0.840 | 0.763 | 0.872 | 0.814 |
| Instant (A), Random Forest | 0.520 (0.027) | 0.442 (0.027) | 0.827 (0.013) | 0.576 (0.024) | 0.472 | 0.434 | 0.782 | 0.558 |
| Instant (A), XGBoost | 0.523 (0.037) | 0.438 (0.025) | 0.846 (0.020) | 0.577 (0.024) | 0.492 | 0.439 | 0.809 | 0.569 |
| History (D), Random Forest | 0.929 (0.019) | 0.845 (0.021) | 0.911 (0.017) | 0.877 (0.018) | 0.943 | 0.832 | 0.941 | 0.883 |
| History (D), XGBoost | 0.933 (0.017) | 0.840 (0.019) | 0.914 (0.019) | 0.875 (0.018) | 0.943 | 0.836 | 0.945 | 0.887 |
| History (D), XGBoost tuned | 0.936 (0.017) | 0.845 (0.026) | 0.915 (0.016) | 0.879 (0.020) | 0.945 | 0.837 | 0.946 | 0.888 |

The earlier saved history model (three regions, default XGBoost, setup D, seed 42 split) had these held out values:

| Operating point | Threshold | Precision | Recall |
|---|---|---|---|
| F1 optimal | 0.8148 | 0.8537 | 0.9132 |
| precision >= 0.90 | 0.914 | 0.8944 | 0.84 |
| precision >= 0.95 | 0.9731 | 0.9303 | 0.5965 |

Leave one region out, the earlier instant results and the history comparisons are in `reports/summary.md`, `reports/summary_history.md` and `reports/summary_history2.md`.

## Limitations

- Data covers January to June 2025 only, in eight regions. That window misses the main fire season in places such as Brazil (the Amazon burns mostly from July to October), so the static share and the scores of a region describe these months only and say little about the rest of the year. The regions are country downloads, so they do not cover the world.
- The history tier needs weeks of past data for the same place and does not apply to a first-time detection.
- History features nearly restate how NASA's static flag is built, so high history scores show that past detections predict NASA's flag, not that the flag is correct. The one line rule already gets part of the way.
- The instant tier is weak on its own. Static precision is 0.24 in the five grouped splits and it is close to a night detection baseline (see Day and night).
- Labels are NASA's. There is no independent ground truth.
- Differences between history setups are small compared with the split to split spread.
- With eight regions the static share is 4.3 percent, so precision at a fixed recall is lower than in the three region study, and two regions (DR Congo and Australia) have a static share near or below 1 percent.
- The shipped history operating points were recomputed on the new training scores: precision >= 0.90 now has recall 0.78 (0.84 before), and precision >= 0.95 predicts nothing on held out rows.

## Earth

The "Earth" tab is one interactive Earth built with MapLibre GL JS 5.6.0 (globe projection, loaded from a CDN) and embedded with `st.components.v1.html`. At world zoom it is a globe. Rotate is a toggle that turns the globe slowly (4 degrees a second); if the view is zoomed in past the globe levels it first eases out to the globe, then turns, and a drag, a touch or a zoom stops it and releases the button. Satellites, on by default, shows three moving satellites named S-NPP, NOAA-20 and NOAA-21 from a circular sun synchronous orbit model (inclination 98.7 degrees, period 101 minutes, three phases, 60 times faster than real time). Each is a small satellite icon of about 28 px (a body, two solar panels, a short antenna and a dish, with a soft glow and a thin name beneath) turned to its direction of travel, with a fading ground track trail and a faint orbit line, and it is hidden while it is on the far side of the globe. Everything is drawn on one small canvas from requestAnimationFrame (a few milliseconds a frame), so it adds no map source and no library, does not count against the 20000 point cap and works independently of Rotate. The legend lists NOAA-20 and NOAA-21 as active, S-NPP as "S-NPP (no new data since 2 Nov 2026)" (its icon is drawn dimmer) and "Illustrative orbits, not real positions". A test of a three.js custom layer (a small 3D model drawn above the surface) was tried on MapLibre GL JS 5.6.0 and three r160 in globe mode and did not render in the headless browser used for checks, and MapLibre custom layers in globe mode need the globe projection applied in the vertex shader, which three.js does not do, so the icons are kept and there is no 3D model. The globe flattens into a normal map and then city level as you zoom in. The default basemap is Esri World Imagery with Esri place labels and a Map switch for the Carto Voyager style, terrain comes from AWS terrarium tiles with hillshade, and Tilt 3D or a right mouse or Ctrl drag tilts the view. It needs internet for the library and all tiles; without it nothing is drawn. Imagery: Esri, Maxar, Earthstar Geographics and the GIS User Community. Terrain: Mapzen, AWS Terrain Tiles. Map style: Carto, OpenStreetMap contributors. Boundaries (`data/geo/`): Natural Earth, public domain. The data source is the archive (the Map tab data, same tier and operating point) or the live snapshot, and the four geography filters fly the camera to the selection and outline the chosen country and, where Natural Earth has them, its provinces. Detections are circles, at most 20,000 (a random sample, with a caption giving the real count; pick a province or city to see every detection of a place), and a click opens a popup with class, probability of static, NASA label (archive only), brightness, FRP, date, country and province. A line next to the Show switch explains the first two options (a raw detection is a hot pixel reported by the satellite, not classified; model prediction is our classification of each detection as fire, static or uncertain), and for the Live source the legend adds that predicted fire means any vegetation fire, including routine farm and savanna burning, and does not mean a reported wildfire event. The Show switch offers Raw detections, Model prediction (red fire, grey static), Ground truth and Errors only (false static and false fire in two colours, with counts); the legend in the map follows the option, and the camera is kept when the option changes. Ground truth is NASA's own repeat based static flag, not independent truth, and it does not exist for live data, so Ground truth and Errors only are disabled there. The live source reads the near real time feeds `VIIRS_NOAA20_NRT` and `VIIRS_NOAA21_NRT` from the FIRMS area API (Suomi NPP no longer delivers near real time data): put `FIRMS_MAP_KEY=your_key` in a `.env` file in the project root (git ignores it; the key is never printed), press Refresh data to fetch the last 60 days of both satellites in chunks of at most 5 days with a one second pause between calls, far below the limit of 5000 transactions per 10 minutes (later refreshes fetch only missing days and the latest 3), or tick Offline to use the saved snapshot. Each satellite has its own rolling store under `data/live/`, every row keeps the satellite name, and the history features are built inside one satellite only, never mixing satellites. The models, calibrator and bands are the Suomi NPP trained ones. These feeds have no type column, so there is no ground truth and only predictions are shown. The tab lists the acquisition date range, the rows per satellite and the time of the last fetch, and a genuine snapshot shows no demo warning. `python -m src.live_demo` builds demo snapshots from `data/raw/` with dates shifted to look recent, and the tab then shows a red demo warning (a real refresh replaces the snapshot, so rebuild the demo when you need it offline). Live data is about 3 hours behind the satellite. The models were trained on Suomi NPP and validated on NOAA-20 and NOAA-21 archive data, but near real time data has not been validated against the archive. `python -m src.live_check` fetches the feeds and prints sanity checks (predicted static share per region and satellite, how many cells with at least 10 detections in the last 30 days are predicted static, and the overlap of the two satellites' predicted static cells) into `reports/live_check.csv`. Regions: the eight training regions (South Asia, Southeast Asia, North and Central America, Brazil, Australia, India, Nigeria, DR Congo) plus "Europe (not in training)" with an unvalidated banner. The archive source shows detections of all eight regions with the same geography filters (continent, country, province, city), still at most 20,000 circles.

## Uncertain band

Besides static or fire, each model tier can answer uncertain (needs investigation). Scores are first calibrated (Platt or isotonic, fitted on the out of fold training scores and chosen by cross validated Brier score). The band is then the pair of thresholds on the calibrated score with the highest coverage whose accuracy on the decided out of fold training points is at least 0.96 (default) or 0.98 (strict): static at or above t_high, fire at or below t_low, uncertain in between. Both come from out of fold training scores only, and the model bundles store them as `band`, `band_strict` and `calibrator`. `python -m src.uncertain` evaluates the bands on the grouped split, the temporal split and leave one region out next to the one line rule and writes `reports/calibration_*.csv` and `reports/uncertain_*.csv` with plots. The first definition (static precision at least 0.90 and fire side precision at least 0.95) is kept in the reports as a negative result, labelled "precision based": fire is the majority class, so the fire side target is met almost everywhere and the history band came out empty. Honest limits: for the history tier the 0.96 target is already met by the forced binary decision, so the default band is empty (nothing is uncertain) and only the strict 0.98 band abstains; the instant tier reaches 0.96 only by calling almost everything fire or uncertain and rarely calls static; accuracy on new regions and periods can fall below the target chosen on the training scores; and NASA's flag is the label, not independent truth. The Earth tab shows uncertain in amber in the Model prediction view, with a selector for the default or strict band, and Model metrics has the table and the coverage curves.

## NOAA-20 transfer

`python -m src.noaa20` downloads VIIRS_NOAA20_SP (with the type column) for the three region boxes from 2025-01-01 to 2025-06-30 in 5 day chunks, 111 calls with a pause and resume support, and cuts the rows to Pakistan, Indonesia and the United States, the countries of the Suomi NPP files, with the polygons in `data/geo/`. The map key is read from `.env` and every request error prints only the exception class and the HTTP status. `python -m src.noaa20_transfer` builds NOAA-20 features and history inside each NOAA-20 file only, with the same label rule, and tests the Suomi NPP models with their thresholds, calibrators and uncertain bands unchanged. Test sets are NOAA-20 detections in cells that were not Suomi NPP training cells (cell disjoint) and, labelled as easier, detections in cells that were (not cell disjoint). Results are in `reports/noaa20_transfer.csv` with a summary, the sensor comparison and the row counts, and in the Model metrics tab. Honest limits: the labels are NASA's repeat based flag, not independent truth; the two satellites fly the same type of instrument over the same places in the same days, only about 22 minutes apart at matched positions, so this is a transfer between near twins and says little about a different sensor; cell disjoint cells are mostly fire cells, so their static share is low and the scores must be read against the majority baseline; and the country cut with Natural Earth polygons differs slightly from the archive download.

NOAA-21 transfer. `python -m src.noaa20 --sat noaa21` and `python -m src.noaa20_transfer noaa21` repeat the NOAA-20 experiment for `VIIRS_NOAA21_SP` with the same boxes, dates, country cut and rules (files in `data/noaa20/noaa21/`, results in `reports/noaa21_transfer.csv`, `reports/noaa21_sensor_comparison.csv` and `reports/noaa21_transfer_summary.md`, and next to the NOAA-20 rows in the Model metrics tab). With the eight region models, on cell disjoint cells the history model reaches static PR-AUC 0.920 (NOAA-20 0.910, Suomi NPP held out reference 0.905, majority baseline 0.068), the instant model 0.311 (NOAA-20 0.319) and the one line rule 0.798 (NOAA-20 0.769), so NOAA-21 behaves like NOAA-20. The numbers of the three region study were 0.914, 0.359 and 0.726 for NOAA-21. On matched positions the three sensors show no large brightness bias (I-4 median about minus 1 K, but single detections differ by about 10 K on average), and the acquisition time offset is about 51 minutes between NOAA-21 and NOAA-20 and about minus 26 minutes between NOAA-21 and Suomi NPP on most days, with a spread across days. Honest limits are the same as for NOAA-20: NASA's repeat based flag is the label, the satellites are near twins, cell disjoint cells are mostly fire cells, and the sensor comparison uses six sampled days. The result says the Suomi NPP trained models carry over to the two newer satellites in the archive, not that near real time data behaves the same.

Day and night. `python -m src.daynight` (stages tables, xgb, transfer, ship) tests how much of the instant tier is a night detector, with results in `reports/daynight_*.csv` and in Model metrics, recomputed with the eight region models. Against a trivial baseline that calls every night detection static, the instant tier is only slightly better: static PR-AUC 0.244 against 0.155 on the grouped held out split, 0.186 against 0.149 on the temporal split, 0.319 against 0.270 on NOAA-20 cell disjoint cells and 0.311 against 0.259 on NOAA-21 (the baseline's own PR-AUC is not small because its score is 0 or 1 and 17 percent of night detections are static). Static F1 is 0.331 against 0.289 on the grouped split, 0.294 against 0.280 on the temporal split, and within 0.01 of the baseline on both NOAA sets. By day (static share 0.007) the instant tier calls nothing static (PR-AUC 0.018); by night (static share 0.174) it has PR-AUC 0.267, recall 0.67 and precision 0.23. On leave one region out the instant F1 is below the baseline in five of eight regions (South Asia, Southeast Asia, North and Central America, India, Nigeria). The history tier is far above the baseline (grouped PR-AUC 0.905, day 0.889, night 0.907) and so is the one line rule (0.768). In an ablation with XGBoost (100 trees, grouped split, thresholds from out of fold scores) removing is_night alone costs 0.010 PR-AUC (0.234 against 0.244) because brightness, frp and the pixel size also carry the day or night signal; removing frp and frp_per_area costs most (0.217), confidence_ord almost nothing (0.242). Permutation importance on the held out split ranks is_night first (PR-AUC drop 0.133), then frp (0.052) and bright_ti5 (0.041). The XGBoost instant model replaced the random forest on the evidence of the three region study (0.507 against 0.515 on the grouped split, 0.492 against 0.472 on the temporal split, `reports/daynight_xgb_vs_rf.csv`, not recomputed); a 300 tree forest on eight regions does not fit in 8 GB, so that comparison was not repeated. The model is 270 KB, the old three region bundles are in `model_backups/`. Limits: the label is NASA's repeat based flag, the baseline and the models share the same labels, this says the instant tier adds little beyond the night flag and not that it is useless on a single reading, and the stratified day numbers rest on few static detections.

Why the instant tier flags so many live detections as static. `python -m src.nrt_vs_sp --download` and `python -m src.nrt_analysis` compare NOAA-21 products with the same boxes and country cut (results in `reports/nrt_vs_sp.csv`, files in `data/noaa20/nrtcheck/`). For 2026-06-01 to 2026-07-31 the `VIIRS_NOAA21_NRT` and `VIIRS_NOAA21_SP` downloads hold the same detections: same row counts per region, every detection matched within 0.006 degrees and 30 minutes, and bright_ti4, bright_ti5, bright_diff, frp, scan, track, confidence and day or night identical on all matched pairs, so both products give the same instant and history shares. The confidence column uses the same categories (l, n, h) and the same mapping to 0, 1 and 2 in both. So no product difference was found, and no fix was needed. What moves the instant share is the day or night mix: the instant tier flags almost no daytime detections as static (0.1 to 0.4 percent) and 60 to 90 percent of night ones, in the archive too (25 percent flagged overall at a night share of 31 percent, while NASA's static share there is 14 percent). Live Southeast Asia has a night share of 51 percent against 13 percent in 2025-09-11 to 2025-11-09 SP data, and its instant share is 41.7 percent against 11.4 percent; at the SP night mix the live share would be 10.7 percent. The same standardisation brings South Asia to 9.2 percent against 9.4 percent and North and Central America to 43.6 percent against 43.1 percent, so the high shares are a property of the instant model plus the season and year, not of near real time data. Limits: SP data for August to November 2026 does not exist yet, so the live feed itself could not be compared with SP for those weeks, and why 2026 Southeast Asia has so many night detections was not investigated. Treat the instant tier's static label as a night detection flag, and prefer the history tier on live data. The instant shares in that paragraph and in `reports/nrt_vs_sp.csv` and `reports/live_check.csv` come from the random forest instant tier that has since been replaced.

## All regions

`src/regions.py` maps every file in `data/raw/` to a region, and `python -m src.world` runs the stages, one per process: `backup` (copies models and reports outside the repo), `inventory` (`reports/world_inventory.csv`), `prepare` (`src/worlddata.py`: the existing cleaning, labelling, feature and history code applied to one raw file at a time, float32 columns, cached in `data/processed/prepared_world.parquet`), `history` and `instant` (both tiers on five grouped seeds, the temporal split and every region held out, results saved after each split so a stopped run resumes), `regions`, `headline` and `summary` (`reports/world_summary.md`). A file without a type column is skipped with a message. Then `python -m src.final_history` saves the history model, `python -m src.daynight --stage ship` saves the instant model, and `python -m src.uncertain` adds the calibrators and both bands. Nothing in the method changed: same grid cell, same feature lists, no coordinates and no UTC hour, same splits, thresholds from out of fold training scores. The saved models are small (about 0.3 MB each). Each stage prints its peak memory. Peak commit on this 8 GB machine was 1.5 GB for `prepare`, 2.4 GB for `history`, 2.8 GB for `instant` and 2.0 GB for `headline`, below the 5 GB limit (a first history run reached 3.2 GB and stopped with an allocation error when other programs held the memory, which is why the stages now save after each split).

The live store needs rescoring when the models change: `python -m src.ingest --rebuild-score` scores every stored row again with the saved models, calibrators and bands, keeps the ingest times and the 90 day window, and fetches nothing. `src/live.py` has boxes for the five new regions. The boxes are rounded boxes, not borders. Where two boxes overlap (South Asia with India, Nigeria with DR Congo) a detection is stored once, in the earlier region.

## Live store and API

`python -m src.ingest --once` fetches the latest NOAA-20 and NOAA-21 near real time detections for every box of `src/live.py`, scores them with the saved models, calibrators and bands, and merges them into one store, `data/live/store.parquet` (git ignores `data/live`). Day ranges are 1 to 5, a short wait follows every call and a sliding window budget stays below the limit of 5000 transactions per 10 minutes, with the measured cost of a call (4 transactions per day of range, so 4 for one day and 20 for five days on the largest box, measured with the key status counter). The store is written atomically, rows are unique on sensor, latitude, longitude, acq_date and acq_time so a second run adds nothing, and only the last 90 days are kept for the history features. It holds the features, both scores, the calibrated probabilities, the labels, the default and strict band calls and the ingest time. The command prints only counts and the peak memory of each stage. A failed call prints only the exception class and the HTTP status, never a URL, a header, a traceback or the key, which is read from `FIRMS_MAP_KEY` (environment or `.env`). `python -m src.ingest --loop --every-minutes 180` repeats it, `python -m src.ingest --regions trained` limits a run to the eight trained regions, `python -m src.ingest --measure-cost` reads the key's transaction counter before and after one call and prints only numbers, and `python -m src.ingest --import-live` is a one off that takes the rolling stores of the older fetch code into the store.

Memory. An earlier version read the whole store, scored every group, kept all the scored groups and then joined, sorted and wrote them in one piece. With the five added regions it stopped with `MemoryError` at that last step (peak commit 6.6 GB, the store file unchanged). The ingest is now bounded by the largest group of one sensor and one region: it reads three columns for the summary, reads and scores one group at a time (the scores in chunks of 250000 rows), streams each finished group into a temporary file as row groups of at most 200000 rows sorted by time, and replaces the store in one step; groups that did not change are copied as they are. A run stops before a group that would pass 4 GB of commit and leaves the store unchanged.

Backfill. Every fetched day is first saved in `data/live/ingest_state/staged` (one file per sensor, region and day) and the days merged into the store are listed in `data/live/ingest_state/merged.txt`, so an interrupted run keeps what it fetched and the next run asks only for the missing days plus the latest three. After the first full pass a normal run is one call of three days per sensor and box.

World coverage. The boxes are the eight trained regions (South Asia, Southeast Asia, North and Central America, Brazil, Australia, India, Nigeria, DR Congo) and 15 continent level boxes for the rest of the land: Europe, Russia and North Asia, Central and East Asia, East Asia South, Middle East, North Africa, West Africa, Central Africa North, East Africa, Southern Africa, Central America and Caribbean, South America West, South America South, Oceania North and Oceania South. The new boxes do not overlap each other or a trained box (they touch along edges). Antarctica, Greenland, the Arctic islands and small ocean islands are not covered. A region has a trained flag, true only for the eight trained regions, and an untrained region is shown in the legend, in the popups and in the API as: region not in the training data, results unvalidated. The models were not trained on those areas and their scores there are not validated.

The Live source of the Earth tab reads the store (all regions or one), by default the last 24 hours with a control for 6, 24 and 72 hours, and shows the newest detection time in UTC, its age in minutes, the rows per sensor and the last ingest time. The view rereads the store every 10 minutes (a Streamlit fragment with `run_every`; without fragments there is a Refresh now button), reads only the needed columns and time range, and keeps the label that live results are unvalidated. It falls back to the old snapshot behaviour when the store is missing.

Start the API with `uvicorn api.main:app` (docs at `/docs`). `GET /health` gives the status, the last ingest time and the row count (from the file metadata), `GET /model` the model files, thresholds, band settings and the headline metrics from the existing reports, `POST /predict` up to 5000 detections (validated, one answer each with the label, the calibrated probability, the default and strict band calls and the tier used), `GET /live/regions` the boxes with the trained flag, and `GET /live/latest` the newest scored rows (parameters `hours`, `sensor`, `region`, an optional box, at most 20000 rows; every row carries `trained` and, for an untrained region, the note). The time, sensor and region filters are applied while the store is read. `/predict` uses the history tier when history fields are supplied or when the cell exists in the store (the history of its latest stored detection), and the instant tier otherwise, and each answer says which.

    curl http://127.0.0.1:8000/health
    curl "http://127.0.0.1:8000/live/latest?hours=24&region=Brazil&limit=100"
    curl "http://127.0.0.1:8000/live/latest?hours=24&sensor=NOAA-20&min_lat=24&min_lon=61&max_lat=36&max_lon=76&limit=100"
    curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"detections\": [{\"latitude\": 28.6, \"longitude\": 70.1, \"acq_date\": \"2026-11-01\", \"acq_time\": 742, \"bright_ti4\": 330.5, \"bright_ti5\": 295.0, \"frp\": 3.2, \"scan\": 0.45, \"track\": 0.5, \"confidence\": \"n\", \"daynight\": \"N\"}]}"

The models are not retrained from live data: live data has no type flag, so there is no label to learn from. Live results are unvalidated. Tests: `python -m pytest tests/test_ingest.py -q`, `python -m pytest tests/test_ingest_world.py -q` and `python -m pytest tests/test_api.py -q`, one per process.

### Running the ingest every 3 hours on Windows

`scripts/run_ingest_loop.ps1` activates the virtual environment in the project folder and runs `python -m src.ingest --loop --every-minutes 180`. To start it automatically with Windows Task Scheduler (the task is not created by the project):

1. Put the key in `.env` in the project folder as `FIRMS_MAP_KEY=...`, and run `python -m src.ingest --once` once by hand to check it works.
2. Open Task Scheduler (press the Windows key, type Task Scheduler, press Enter).
3. In the right hand pane choose Create Task (not Create Basic Task).
4. General tab: Name `Fire hotspot ingest`. Choose Run only when user is logged on, so the project folder and `.env` are read as your own user.
5. Triggers tab: New, Begin the task At log on, Any user or your user, tick Enabled, OK.
6. Actions tab: New, Action Start a program. Program/script: `powershell.exe`. Add arguments: `-NoProfile -ExecutionPolicy Bypass -File "<project folder>\scripts\run_ingest_loop.ps1"`. Start in: `<project folder>`. OK.
7. Conditions tab: untick Start the task only if the computer is on AC power if this is a laptop you want it to run on battery, and tick Wake the computer only if you want that.
8. Settings tab: tick Allow task to be run on demand, untick Stop the task if it runs longer than, set If the task is already running to Do not start a new instance.
9. OK, and enter your Windows password if asked. Right click the task and choose Run to start it; the History tab and `data/live/store.parquet` show it working.

The script runs the loop, which waits 180 minutes between rounds and prints only counts and memory. If you prefer Task Scheduler to repeat a single run, use the same action with `.venv\Scripts\python.exe` as Program/script, `-m src.ingest --once` as arguments, the project folder as Start in, and a trigger that repeats every 3 hours.

## Sample file

`samples/sample_viirs.csv` (about 170 KB, 2,000 near real time detections of NOAA-20 and NOAA-21 from the live store, 23 boxes, day and night, no labels) is a ready file for the Classify a new file tab: upload it there (the history tier warns because it covers only a few days).

Fresh files can be downloaded from the FIRMS Active Fire Data page, https://firms.modaps.eosdis.nasa.gov/active_fire/ (choose VIIRS NOAA-20 or NOAA-21 and a country or an area).

## Run

    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt
    # torch, CPU build:
    pip install torch --index-url https://download.pytorch.org/whl/cpu

    # optional, for the live source of the Earth tab: put FIRMS_MAP_KEY=... in a .env file in the project root

    # put the archive CSV files in data/raw/
    python -m src.train --tag headline          # instant tier
    python -m src.world inventory               # all regions: files, regions, static share (also: prepare, history, instant, regions, headline, summary)
    python -m src.final_history                 # history tier, saves the model and scored rows
    python -m src.history_eval --part a         # evaluations (parts a, b, c)
    python -m src.history2_eval --part a        # history v2 and tuning (parts a, b, c)
    python -m src.history2_ops                  # operating points and curve
    python -m src.region_eval                   # instant tier, leave one region out
    python -m src.nrt_score                     # score data/nrt/SUOMI_VIIRS_C2_Global_7d.csv
    python -m tests.test_history                # leakage test
    python -m tests.test_live app_earth         # live and Earth tests, no network, one suite per process
                                                # suites: unit, app_store, app_live, app_demo, app_geo, app_earth
    python -m src.ingest --once                 # live store, every region, needs the key (--regions trained for eight)
    python -m src.ingest --measure-cost         # transactions of one call, numbers only
    powershell -File scripts\run_ingest_loop.ps1   # the loop, every 180 minutes
    python -m src.ingest --rebuild-score        # rescore the stored rows with the saved models, no fetch
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

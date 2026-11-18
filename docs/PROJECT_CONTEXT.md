# Thermal Anomaly Classifier: Project Context and Handover

Repository folder: fire-hotspot-classifier. Last updated 18 November 2026.

## 0. Read this first

This file gives a new contributor the full picture: what the project is, what has been built, how to run it, what is verified and what is not, and what to build next. Read all of it before changing code. If you work with an automated coding tool, point it at this file first and keep the rules in section 2 in every prompt you give it.

The topic cannot change. The project classifies NASA FIRMS VIIRS 375 m satellite detections as a likely vegetation fire or a likely static heat source (industry, gas flares, kilns), and shows the result in a Streamlit app, a live feed and an API.

Where this file says "inferred" it means the role of a module was read from its name or from reports, not from its source. Open the module docstring before relying on it.

## 1. Project summary

Team and roles in the approved proposal: Yusra Shah (lead, model development and evaluation), Sinya Kumari (data and feature engineering), Asma (MLOps and deployment). Yusra Shah owns the repository and the pushes.

Differences from the approved proposal. Tell the course teacher if asked.

| Proposal | What was built |
|---|---|
| MODIS Collection 6.1 | VIIRS S-NPP 375 m archive (type field), then NOAA-20 and NOAA-21 near real time for live use |
| About 267,000 detections, 3 regions | 2,349,116 detections, 8 regions on 5 continents, January to June 2025 |
| Logistic regression, random forest, XGBoost, rule baseline | Those, plus a small neural network, tuned XGBoost, a two tier design (instant and history), calibration, an uncertain band, cross satellite tests, a live store, an API and a globe app |

Status in one paragraph: the research part and the evaluation are complete and documented. The live system works but its predictions are unvalidated. The app has a globe with satellite imagery, live data, filters and a satellite animation. The next work is the detail panel the teacher asked for (section 12), plus clean up and handover tasks (section 11).

## 2. Rules set by the project owner

Repository

- Never commit hidden tool folders, session files or logs of automated tools. Never add co-author or session lines to commit messages. No tool names in code, comments, docs, commits or reports.
- Only the owner pushes. Others commit locally and hand over, or ask the owner.
- Do not commit data/live, data/raw, .env, or any file near 100 MB (GitHub rejects 100 MB). The tracked large files are data/processed/scored_headline.parquet (71 MB), scored_history.parquet (70 MB) and geo_cells.parquet (9 MB). Do not re-commit them without telling the owner, because each commit adds that size to the repository history.
- Before any commit: `git grep --cached -il <tool name>` and `git grep --cached -nE "[0-9a-f]{32}"` must print nothing, and no staged file may be over 20 MB.

Secrets

- The FIRMS MAP key lives in .env as FIRMS_MAP_KEY and is read from the environment. Never print, log or commit it. Never print URLs, headers or tracebacks from code that calls FIRMS. On failure print only the exception class and the HTTP status.
- The owner will replace the current key at the end of the project. Each contributor should request a personal key from the FIRMS site and keep it in a local .env.

Style

- No emojis. No em dashes and no en dashes in any text or file. Minimal, plain visual design.
- State limits plainly. Do not claim results that were not measured. Live results are unvalidated and must be labelled so.

Machine

- Development laptop has 8 GB of RAM. Close browsers and virtual machines before heavy runs. Run each test suite in its own process and print peak memory. Stop before any step that would pass about 4 GB of commit.

## 3. Environment and commands

Windows, PowerShell, Python 3.12, virtual environment in .venv.

```
cd <repo folder>
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

App and API

```
python -m streamlit run app/app.py
python -m uvicorn api.main:app --port 8000      (confirm the exact command in README)
```

Live ingest (needs FIRMS_MAP_KEY in .env)

```
python -m src.ingest --once                       one run, fetch what is missing, score, write the store
python -m src.ingest --loop --every-minutes 180   repeat
python -m src.ingest --once --regions "<name>"    limit to named regions
python -m src.ingest --measure-cost               read the transaction cost of a call
python -m src.ingest --rebuild-score              rescore every stored row, no fetch
python -m src.ingest --import-live                one off import of the older rolling stores
scripts/run_ingest_loop.ps1                       loop wrapper for Windows Task Scheduler (README has the steps; the task is not created)
```

Research pipeline (inferred from module names, check each module header)

```
python -m src.train --tag <tag>        train and compare models
python -m src.history                  history features and models
python -m src.final_history            final history tier
python -m src.uncertain                uncertain band study
python -m src.region_eval              leave one region out
python -m src.daynight --stage <name>  day and night analysis, night baseline
python -m src.world inventory|history  eight region study and inventory
python -m src.noaa20 / src.noaa20_transfer   cross satellite tests
python -m src.nrt_score / nrt_analysis / nrt_vs_sp --download / live_check / live_demo
```

Tests, each in its own process

```
python -m tests.test_history
python -m tests.test_daynight
python -m tests.test_uncertain
python -m tests.test_noaa20
python -m tests.test_nrt_vs_sp
python -m tests.test_live unit            (also: app_store, app_live, app_demo, app_geo, app_earth)
python -m pytest tests/test_ingest.py -q          (10 tests)
python -m pytest tests/test_ingest_world.py -q    (21 tests)
python -m pytest tests/test_api.py -q             (26 tests)
```

All suites passed at the last full run (16 November 2026 and again 17 November 2026). The app suites use fake data, not the real store.

## 4. Repository map

Top level: README.md, requirements.txt, api/, app/, src/, tests/, scripts/, samples/, models/, data/, reports/, notebooks/.

src (one line each; "inferred" where noted)

- config.py, data.py (REQUIRED columns, normalize_columns), label.py (type 0 to fire, type 2 to static, drop 1 and 3), features.py: base data handling.
- train.py, mlp.py: model training and the small neural network.
- history.py, history_eval.py, history2_eval.py, history2_ops.py, history2_summary.py, final_history.py: history feature tier and its evaluation.
- repeat_eval.py: the one line repeat rule baseline.
- uncertain.py: uncertain band study and the saved band settings.
- daynight.py: day and night analysis and the night baseline.
- region_eval.py, regions.py, worlddata.py, world.py: eight region data and leave one region out.
- noaa20.py, noaa20_transfer.py: NOAA-20 and NOAA-21 transfer tests.
- nrt_score.py, nrt_analysis.py, nrt_vs_sp.py, live_check.py, live_demo.py: near real time checks.
- live.py: FIRMS fetch logic, REGIONS (boxes), TRAINED (the eight trained regions), LIVE_SOURCES, continent_of, prepare (features), key_status, region notes.
- serve.py: loads the saved model bundles, applies calibrators and uncertain bands. Shared by the ingest and the API.
- ingest.py: live store job (section 9).
- geo.py: country, province and nearest city lookup used by app filters (cache at data/live/geo_cache.parquet and data/processed/geo_cells.parquet).
- earth.py: builds the globe page (MapLibre GL JS 5.6.0 inside st.components.v1.html).
- mem.py: peak memory reporting used by tests and long jobs.

Other

- app/app.py (about 770 lines): the Streamlit app.
- api/main.py (about 310 lines): FastAPI service.
- data/processed: scored_headline.parquet, scored_history.parquet (the archive scored by the models; "headline" is inferred to be the instant tier), geo_cells.parquet.
- data/live: store.parquet (about 766 MB, ignored), ingest_state/ (staged days and the merged.txt ledger), snapshot files and geo_cache.parquet.
- models: best_model_headline.joblib and best_model_history.joblib (small, tracked, about 0.3 MB each). Other model files are ignored.
- reports: summaries and CSV tables of every study (world_summary.md, world_history.csv, calibration and uncertain reports, noaa20 and noaa21 transfer summaries, daynight tables).
- samples/sample_viirs.csv: 2,000 rows from the live store in FIRMS format, no labels, for the upload demo.
- Backups outside the repository (owner machine): model_backups/before_world, report_backups/before_world, store_backup_before_memfix.parquet. Safe to delete once nothing depends on them.

## 5. Data, labels and features

Source. NASA FIRMS, VIIRS S-NPP standard processing archive for training (type field). Live uses NOAA-20 and NOAA-21 near real time (NRT) data, which has no type field. S-NPP delivery ended on 2 November 2026 (FIRMS notice). Archive requests are approved by hand by the FIRMS team and arrive by email, and the labelled archive lags about 5 months. The NOAA-20 standard processing archive runs to 31 July 2026.

Training data. 8 region files, January to June 2025: Pakistan (South Asia), Indonesia (Southeast Asia), USA (North and Central America), Brazil, Australia, India, Nigeria, DR Congo. 2,349,116 rows, 4.25% static. About 1.16 million distinct 0.01 degree grid cells.

| Region | Rows | Static share |
|---|---|---|
| South Asia | 46,909 | 0.0297 |
| Southeast Asia | 14,034 | 0.2724 |
| North and Central America | 196,898 | 0.1611 |
| Brazil | 117,776 | 0.0616 |
| Australia | 412,943 | 0.0128 |
| India | 513,677 | 0.0848 |
| Nigeria | 332,845 | 0.0194 |
| DR Congo | 714,034 | 0.0005 |

Label. NASA type field. 0 vegetation fire becomes fire. 2 other static land source becomes static. 1 volcano and 3 offshore are dropped. Static is the positive class. The label is NASA's own repeat based judgment, not independent ground truth.

Instant tier, 10 features: bright_ti4, bright_ti5, bright_diff, frp, scan, track, pixel_area, frp_per_area, confidence_ord, is_night. Coordinates and UTC hour are excluded on purpose.

History tier, 29 features: the 10 above plus 19 backward looking features: distinct days seen in the previous 7, 30, 60 and 90 days, neighbour distinct days (7, 30, 60), detections in the previous 7 days, days since first seen, days into data, days since last seen, persistence_30, mean and std of ti4 and frp over 30 days, has_history, bright_dev, frp_dev. Only the past is used.

## 6. Models and evaluation design

Models tried: logistic regression, random forest, XGBoost, MLP (two hidden layers, 64 and 32), tuned XGBoost. Final: XGBoost with 100 trees for both tiers (about 0.3 MB each). A 714 MB random forest was replaced.

Calibration: Platt or isotonic, chosen by out of fold Brier score. Operating point: precision at least 0.90 (history cutoff about 0.985). Uncertain band: accuracy target on calibrated scores, default 0.96 and strict 0.98, edges from out of fold data.

Honest controls (every one closes a specific way of fooling ourselves):

- grouped split by 0.01 degree grid cell (no place in both train and test)
- no coordinates, no UTC hour
- five seeds
- temporal split: train before 2025-05-01, test May and June
- leave one region out, all eight regions
- thresholds from out of fold predictions
- majority baseline, one line repeat rule (distinct days in the previous 30 at least k), night baseline ("night means static")
- leakage test

## 7. Results (static class)

Main tests, eight regions:

| Test | History PR-AUC | Rule PR-AUC | Instant PR-AUC | Night baseline PR-AUC | Chance |
|---|---|---|---|---|---|
| Grouped split, seed 42 | 0.905 | 0.768 | 0.244 | 0.155 | 0.042 |
| Temporal split | 0.927 | 0.774 | 0.186 | 0.149 | 0.036 |

History at the default cutoff on held out cells: precision 0.893, recall 0.781, F1 0.833 (rule F1 0.803). Temporal F1 0.844 (rule 0.762). Five seed average: PR-AUC 0.921, F1 0.880. Instant F1 0.331 against a night baseline of 0.289.

Leave one region out:

| Held out region | History PR-AUC | Rule PR-AUC | Instant PR-AUC |
|---|---|---|---|
| South Asia | 0.881 | 0.701 | 0.149 |
| Southeast Asia | 0.845 | 0.780 | 0.578 |
| North and Central America | 0.920 | 0.786 | 0.510 |
| Brazil | 0.937 | 0.729 | 0.177 |
| Australia | 0.934 | 0.823 | 0.038 |
| India | 0.890 | 0.870 | 0.298 |
| Nigeria | 0.793 | 0.634 | 0.079 |
| DR Congo | 0.364 | 0.345 | 0.026 |

Cross satellite transfer (frozen model, cells unseen in training, three original regions only, near twin satellites): NOAA-20 history PR-AUC 0.910, F1 0.763 (rule 0.769, 0.761). NOAA-21 history 0.921, F1 0.776 (rule 0.798, 0.766). Instant 0.319 and 0.311 PR-AUC, F1 0.434 on both.

Calibration, history, grouped: ECE 0.0280 to 0.0007, Brier 0.0157 to 0.0081. Instant ECE 0.2251 to 0.0017.

Uncertain band: the history tier decides everything (accuracy 0.990 grouped, 0.992 temporal), so the band does not abstain. The instant tier at the strict setting abstains on 10.3% and holds 57% of its errors, but calls no spot static.

## 8. Honest caveats (state these before anyone asks)

1. The label is NASA's judgment and partly circular with the history features.
2. The instant tier is mostly a night detector. Its F1 is below the night baseline in five of eight regions.
3. At the shipped cutoff the history F1 falls below the one line rule in Southeast Asia, North and Central America and India, and is about equal on NOAA-20 and NOAA-21. Ranking (PR-AUC) is better everywhere. One cutoff does not suit every region.
4. DR Congo has 365 static detections out of 714,034. Its scores are noise.
5. Six months of data only. No seasonal coverage.
6. Live predictions are unvalidated. Most live detections lie outside the eight trained regions. Predicted fire means any vegetation fire, including routine farm and savanna burning, not a reported wildfire event.
7. NOAA-20 and NOAA-21 are near twins of S-NPP, so the transfer test is not independent evidence.
8. Stale reports not recomputed for eight regions: reports/nrt_vs_sp.csv, live_check.csv, daynight_xgb_vs_rf.csv. The unexplained Southeast Asia live night share (51% against 13% in 2025) remains unexplained.

## 9. Live system

Store: data/live/store.parquet. Last measured: 11,797,434 rows, 46 groups (23 boxes times 2 sensors), 766 MB, dates 2026-08-19 to 2026-11-16. Rows are unique on sensor, latitude, longitude, acq_date, acq_time. Only the last 90 days are kept. Row groups are at most 200,000 rows sorted by time, so readers can skip old data.

Columns: sensor, region, latitude, longitude, acq_date, acq_time, timestamp, cell_id, confidence, daynight, bright_ti4, bright_ti5, bright_diff, frp, scan, track, pixel_area, frp_per_area, confidence_ord, is_night, the history features of section 5, p_instant, pc_instant (calibrated), label_instant, instant_default, instant_strict, p_history, pc_history, label_history, history_default, history_strict, ingest_time.

Ingest behaviour

- Reads three columns for a summary, then reads, scores and writes one sensor and region group at a time. Scores in chunks of 250,000 rows. Streams into a temporary file, then replaces the store in one step. A guard stops before a group that would pass 4 GB of commit (estimate 1.1 KB per row).
- Fetched days are staged in data/live/ingest_state/staged (one file per sensor, region and day). The ledger data/live/ingest_state/merged.txt lists merged days. An interrupted run resumes and refetches only missing days plus the latest 3.
- Rate limit: 5,000 transactions per 10 minutes, sliding window budget 4,500. Measured cost 4 transactions per day of range. A day range is 1 to 5. First full backfill was 828 calls (about 16,560 transactions) and took hours in practice (about 18 s per call). A normal run is about 92 short calls.
- History features are built inside each sensor and region group only.
- The models are not retrained from live data.

Boxes. The 8 trained regions are defined in src/live.py (REGIONS, TRAINED). 15 untrained continent boxes (west, south, east, north):

| Box | West | South | East | North |
|---|---|---|---|---|
| Europe | -10 | 36 | 40 | 70 |
| Russia and North Asia | 40 | 50 | 180 | 78 |
| Central and East Asia | 40 | 36 | 150 | 50 |
| East Asia South | 98 | 6 | 150 | 36 |
| Middle East | 30 | 14 | 61 | 36 |
| North Africa | -18 | 14 | 30 | 36 |
| West Africa | -18 | 4 | 2 | 14 |
| Central Africa North | 15 | 6 | 32 | 14 |
| East Africa | 32 | -14 | 52 | 14 |
| Southern Africa | 10 | -36 | 52 | -14 |
| Central America and Caribbean | -118 | 6 | -34 | 19 |
| South America West | -82 | -56 | -74 | 6 |
| South America South | -74 | -56 | -34 | -34 |
| Oceania North | 141 | -11 | 180 | 6 |
| Oceania South | 154 | -48 | 180 | -11 |

Not covered: Antarctica, Greenland, small islands. Detections in overlapping boxes are stored once (earlier region owns it). Untrained regions carry the note "region not in the training data, results unvalidated".

API (FastAPI, api/main.py)

- GET /health (rows and status), GET /model, POST /predict (up to 5,000 detections, history tier when the place has a record, else instant), GET /live/latest (hours, region, limit up to 20,000; returns count, matching, untrained_rows, and per row trained flag and note), GET /live/regions (boxes with the trained flag).
- Measured: a 24 hour read of 65,819 rows took about 5 s at 1.1 GB peak commit.

## 10. The app

Streamlit, tabs: Map, Why a spot is static, Classify a new file, Model metrics, Results, Earth. Sidebar: model tier (Instant or With history), decision threshold or operating point.

Earth tab: MapLibre GL JS 5.6.0 globe, embedded with st.components.v1.html. Esri imagery with labels, plain map switch, real terrain from zoom 4, Tilt 3D, Rotate toggle (slow spin, eases out to the globe first, stops on drag or zoom), Show switch (raw detections, model prediction, ground truth, errors only), Band selector, Data source (archive or live), geography filters, popups, legend, 20,000 point cap. Live source: window 6, 24 or 72 hours (default 24), newest detection time in UTC and its age, rows per sensor, refresh every 10 minutes. Satellites toggle: canvas icons for NOAA-20 and NOAA-21 (S-NPP dimmed with the note "no new data since 2 Nov 2026"), fading ground track and orbit line, labelled "Illustrative orbits, not real positions". A three.js 3D model was tried and did not render in globe mode, so icons are used.

Classify a new file: upload a FIRMS VIIRS CSV. Instant tier scores each row. History tier computes history inside the file and warns when the file spans fewer than 30 days (a 7 day download is too short). NRT downloads have no labels, so no accuracy is shown.

Known app issues

- Map and Earth overlap. A plan exists (Prompt 12 in the owner's notes) to use the single globe component for every map and merge the Map tab into Earth. Check README and git history to see whether it was applied.
- The Classify tab does not mark detections outside the eight trained regions.
- The Why a spot is static chart has a y axis that goes below 0 and an x axis that runs past the data.

## 11. State of work

Done and pushed (as of the last report): ingest and API (153392c), eight regions and retrain (59d9ad3), memory bounded ingest with world boxes, Rotate, freshness, scheduler script (16 November 2026). Done on 17 November 2026, check git status for whether it is committed: satellite icons, help lines, sample CSV.

Not verified

- Earth live view on the real 11.8 million row store in a browser, the 10 minute refresh, icons on real graphics hardware, the history tier on the sample file in the app.
- Task Scheduler task and script were never run.
- Model metrics "All regions" section in the browser.

Handover and closing tasks (owner decides order)

1. Apply or confirm the single globe change.
2. Add the detail panel and the fire or static selector (sections 12 to 16).
3. Fresh clone test: clone to a new folder, new venv, pip install, run the app. Note the clone has no live store (data/live is ignored); either skip live or copy a store.
4. Replace the FIRMS key (owner), update .env, rerun `python -m src.ingest --once`, delete the old key.
5. Update the project guide document and the specialist brief with the eight region and live results (owner holds these documents).
6. Optional: create the Task Scheduler task from the README steps.

Ideas not started: land cover and weather context (now part of section 12), uncertain band set on the static class, per region cutoffs, July to December 2025 archive data, NOAA-20 archive retraining (labelled archive runs to 31 July 2026), geostationary timeline, Nightfire for flare detection, Sentinel-2 burn scars, an independent check of static predictions against a list of known flares or industrial sites (the single addition that would raise the project most).

### Progress on the teacher's features (branch feature/selector-click-zoom)

- 12.1 Fire or static selector: done, not yet reviewed by the owner.
- 12.2 Click zoom and pixel footprint: done and checked in a browser (Back button, Escape, imagery capped at zoom 17 because zoom 18 tile requests timed out).
- 12.3 Detail panel, step 2 of section 15: done with option A (Streamlit 1.67.0 components v2). Verdict, facts, place and spread from our own data, src/context/detail.py, src/devstore.py, tests/test_detail.py. Checked in a browser on an archive point: the globe stays zoomed while the panel fills. Not checked: a live point (no store yet, samples/sample_viirs.csv is missing).
- Section 13 context sources (land, weather, people, response): not started.
- On the contributor laptop a Windows application control policy blocks the pyarrow dataset library, so filtered parquet reads fail. read_recent now falls back to a row group reader; five filtered reads inside the ingest job and one in api/main.py still fail there (3 tests of test_ingest.py and 2 of test_api.py). Owner to decide whether those should use the fallback too.
- Found while checking the clone: docs/PROJECT_CONTEXT.md and samples/sample_viirs.csv were not in the repository; the 17 November work (satellite icons, help lines, sample CSV) is not on GitHub; the archive files in data/processed have no scan and track columns, so archive footprints use the nominal 375 m pixel; live.KEEP_DAYS is 60 while ingest.KEEP_DAYS is 90; there is no top features explanation code to reuse yet.

## 12. Features requested by the teacher

### 12.1 Fire or static selector

- A control with three choices: Fires (default), Static sources, Both. Place it next to the Show switch.
- Fires shows detections the active tier calls fire. An "Include uncertain" chip (default on) keeps amber points visible, so hedged calls are not hidden silently. Static sources shows predicted static. Both shows all.
- Applies to the model prediction view and to live. In the ground truth view it uses the NASA type (fire or static). In the raw view it does nothing, and the control says so.
- Wording: use "Predicted fire", not "actual fire". Satellite detections are not confirmed fires.
- Legend counts and the "in source, in view" text must follow the selection.

### 12.2 Click a detection: zoom and pixel footprint

- Clicking a point flies to it at the maximum useful zoom (imagery max zoom varies by place, typically 17 to 19; allow over zoom) and draws the pixel footprint: a rectangle of scan by track metres centred on the detection, with its edge and a label such as "375 m pixel, 0.14 square kilometres". The footprint is an approximation (axis aligned, centred on the reported point).
- Highlight the clicked point. Clicking elsewhere or pressing Escape returns the view and closes the panel.

### 12.3 Detection panel (side panel opens on click)

Sections, each with its source and age, and a plain "unavailable" when a source fails:

1. Verdict. Predicted fire, static or uncertain, tier used, calibrated probability, the band call, trained or untrained region note. Reuse the explanation code of the Why a spot is static tab (top features that pushed the call).
2. Detection facts. Time UTC and local time, satellite, brightness ti4 and ti5, FRP, pixel size, day or night, NASA confidence.
3. Place. Country, province, nearest city (existing geo module), elevation.
4. Land and fuel. Land cover class at the point and the mix within 1 km (ESA WorldCover). Helps tell crop burning from forest fire and static sources from fire.
5. Weather. Temperature, humidity, wind speed and direction, gusts, rain in the last 7 days (Open-Meteo). Show the downwind direction as an indicator only.
6. Spread and history (from our own store). Detections within about 2 km in the last 72 hours: count per day, first and last seen, approximate area (pixel count times pixel area), centroid movement and a simple trend (growing, steady, shrinking). Days seen at this cell in the last 90 days. State plainly: this is the extent of satellite detections, not a fire perimeter, and a given place is seen only a few times a day.
7. People and buildings. Population estimate within 1 km and 5 km (WorldPop or GHSL), building count within 1 km (OSM, a lower bound in many regions), nearby critical places: hospitals, schools, fuel stations, power plants within 5 km.
8. Response. Nearest fire station and nearest hospital with straight line distance and bearing (OSM). National emergency number for the country from a reviewed table. No travel time unless a routing service is added.
9. Notes. Data quality, attribution, and the disclaimer: "Satellite detection, not a confirmed fire or an official warning. In an emergency contact local emergency services."

"Safety controls" in the teacher's request is read as safety and response information (fire brigade, hospitals, emergency numbers, exposure). Confirm this reading with the teacher.

## 13. Data and API strategy for the panel

| Need | Source | How | Key | Limits and caveats |
|---|---|---|---|---|
| Place names | existing geo module and cache | local | none | already in the app |
| Place text when missing | Nominatim reverse geocoding | HTTP, cache by rounded coordinates | none | Max 1 request per second, valid identifying User-Agent, cache results, no bulk use |
| Land cover | ESA WorldCover 10 m, 2021 v200 | windowed read of public GeoTIFF tiles on AWS open data (rasterio with GDAL /vsicurl/) | none | 2021 vintage, CC BY 4.0, verify the current URL layout |
| Weather and wind | Open-Meteo forecast API with past_days=7 | HTTP | none | Free for non commercial use with daily limits, verify current terms |
| Elevation | Open-Meteo elevation API or the terrain tiles already used by the globe | HTTP or local | none | Optional slope from neighbouring samples |
| Fire stations, hospitals, schools, fuel stations, power plants | OpenStreetMap through Overpass | `nwr[amenity=fire_station](around:R,lat,lon);out center;` with a timeout, expand R in steps (5, 20, 50, 100 km) | none | Public instances have fair use limits. Retry with backoff, cache. OSM is incomplete in many countries: not found does not mean none exists |
| Buildings | Overpass `out count` of building features within 1 km | HTTP | none | Can time out in dense cities. Lower bound where mapping is thin. Alternatives: Overture Maps buildings queried with DuckDB, Microsoft or Google open building footprints |
| Population | WorldPop (API or population count rasters) or GHSL GHS-POP (100 m rasters) | sample the raster at the point, sum within 1 km and 5 km | none | Modelled estimates, fixed year, verify access method and licence at build time. WorldPop also offers a STAC catalogue |
| Emergency numbers | a reviewed JSON table of national numbers committed in the repository | local file | none | Compile from an official or well maintained list and check each entry by hand. Do not invent numbers |
| Recent imagery (optional) | NASA GIBS VIIRS true colour daily layers | tile layer in the globe | none | Coarse (about 250 m), clouds. Confirm layer names from the GIBS capabilities document |
| Burn scar (later) | Sentinel-2 through a public STAC catalogue | later work | may need free account | Out of scope for now |

Attribution to show in the panel footer: OpenStreetMap contributors (ODbL), WorldPop or GHSL, ESA WorldCover, Open-Meteo, Esri imagery, NASA FIRMS.

Policy and reliability rules

- Fetch on click only, never for every point.
- Cache every response on disk under data/cache/context (add to .gitignore) keyed by source and rounded coordinates, with a time to live (weather 30 minutes, others days).
- Timeouts of a few seconds per call, run the calls in parallel, show sections as they arrive, never block the panel on one slow source.
- No call may print a URL or a traceback. Show the section as unavailable.
- Tests use recorded fixtures and mocked HTTP. No test touches the network.

## 14. Architecture for the panel

The globe lives in an embedded HTML page, so a click in the page must reach Python or the panel must be drawn in the page. Options, in order of preference:

A. Streamlit bidirectional component (components v2) if the installed Streamlit supports it (version 1.66.1 was installed on 15 November 2026). The page sends the clicked detection id to Python. Python builds the detail with the context module and returns it to the page or renders the panel in Streamlit. Check the installed version and documentation first.

B. If A is not available: draw the panel inside the page and have it call a new API endpoint GET /detail?lat=&lon=&time=&sensor= served by api/main.py (enable CORS for localhost). The API must be running. Document it in the README.

C. Pure browser: the page calls the public services directly. Avoid, because of CORS limits, rate limit control and untestable code.

Shared design

- New package src/context with one module per source (weather.py, landcover.py, osm.py, population.py, nominatim.py, emergency.py), each a pure function returning a small typed dict plus source and fetched_at, never raising to the caller. A cache helper in src/context/cache.py. An assembler src/context/detail.py that builds the whole panel payload and the spread analysis from the store.
- The spread analysis reads the store with a spatial and time filter through ingest.read_recent (row groups are time sorted). Never load the whole store.
- Per point data needed in the page: a stable id (sensor, latitude, longitude, acq_date, acq_time), the verdict fields and the detection facts. Keep the 20,000 point cap.

Detail payload (shape, for tests and both options)

```
{
  "id": "...", "verdict": {"label": "fire", "tier": "history", "p": 0.07, "band": "decided", "trained": true, "note": "", "top_features": [...]},
  "facts": {"time_utc": "...", "satellite": "NOAA-20", "ti4": 0, "ti5": 0, "frp": 0, "scan": 0, "track": 0, "daynight": "D", "confidence": "n"},
  "place": {"country": "", "province": "", "city": "", "elevation_m": null},
  "land": {"class": "", "mix": {"tree": 0.0}, "source": "ESA WorldCover 2021", "fetched_at": ""},
  "weather": {"temp_c": null, "rh": null, "wind_kmh": null, "wind_from_deg": null, "gust_kmh": null, "rain_7d_mm": null},
  "spread": {"hours": 72, "n": 0, "per_day": [], "first_seen": "", "last_seen": "", "area_km2": null, "trend": "", "cell_days_90": 0},
  "exposure": {"pop_1km": null, "pop_5km": null, "buildings_1km": null, "facilities": []},
  "response": {"fire_station": {"name": "", "km": null, "bearing": ""}, "hospital": {}, "emergency_number": ""},
  "notes": [], "attribution": []
}
```

## 15. Build order and acceptance

1. Selector (12.1) and click zoom with pixel footprint (12.2). Accept when: Fires is the default, counts and legend follow the selection, click flies in, footprint is drawn with its size label, Escape returns, no regression in Rotate, Satellites, filters and the 20,000 cap.
2. Panel shell with verdict, facts, place and spread from our own store (no external calls). Accept when: opens on click, closes cleanly, spread numbers match a direct query of the store for the same point, tests pass on fake data.
3. Context package: weather, land cover, OSM response data, population, emergency table. Accept when: each source works on three chosen test points (one in Pakistan, one in Africa, one in Europe), each fails soft, cache works, no network in tests.
4. Panel sections for all context, attribution footer and disclaimer. Accept when: a screenshot of each of the three test points shows all sections or a clear "unavailable", and the panel is usable on a phone width.
5. README section, tests per suite with peak memory, then a short note of what was verified and what was not.

For every step: do not change any model, threshold, calibrator, band, store or report number. Do not run the ingest unless the task says so.

## 16. Prompts to paste (one per step, with the standard rules block)

Standard rules block (paste at the top of every prompt)

```
Rules
- Never print or log the FIRMS MAP key, any URL, any header or any traceback. On a failure print only the exception class and the HTTP status.
- Do not run git commands. Do not mention any tool in code, comments, docs, commits or reports. No emojis, no em dashes or en dashes in any text or file. Minimal style.
- 8 GB of RAM: print peak memory for every test suite and run each in its own process. Stop and report before any step that would pass about 4 GB.
- Do not change any model, threshold, calibrator, band, store or report number. Do not run the ingest.
- Read PROJECT_CONTEXT.md, src/earth.py, app/app.py and README.md first.
```

Prompt A: selector, click zoom, footprint

```
Implement sections 12.1 and 12.2 of PROJECT_CONTEXT.md in src/earth.py and app/app.py. Add the Fires, Static sources, Both selector (default Fires) with the Include uncertain chip, and the click to zoom with the pixel footprint rectangle (scan by track) and its size label. Highlight the clicked point, return on Escape or a click elsewhere. Keep every existing feature and the point cap. Update the legend counts. Update tests and the README. If Playwright is available, take screenshots of the default view, the Static selection, and a clicked point at maximum zoom with the footprint visible. Finish with what was verified and what was not.
```

Prompt B: panel shell and spread from the store

```
Implement step 2 of section 15 of PROJECT_CONTEXT.md. First check the installed Streamlit version for bidirectional components (section 14 option A). Use it if it works; otherwise use option B and add GET /detail to api/main.py with CORS for localhost. Build src/context/detail.py to return the payload shape of section 14 with the verdict, facts, place and spread fields only, with the spread read from the store through ingest.read_recent for a 72 hour and 2 km window (never the whole store). Open the side panel on click and render those sections. Tests on a fake store: spread counts match a direct query, memory stays bounded. Finish with peak memory per suite and what was not verified.
```

Prompt C: external context sources

```
Implement step 3 and 4 of section 15 of PROJECT_CONTEXT.md. Add src/context with weather.py (Open-Meteo), landcover.py (ESA WorldCover windowed read), osm.py (Overpass fire stations, hospitals, schools, fuel stations, buildings count within 1 km), population.py, nominatim.py, emergency.py with a reviewed data/emergency_numbers.json that is clearly marked as needing manual verification, and cache.py (disk cache under data/cache/context, ignored by git). Follow the policy rules of section 13: fetch on click only, timeouts, parallel calls, soft failure, no URLs printed, identifying User-Agent for Nominatim, at most one request per second to it. Render the sections in the panel with source and age, the attribution footer and the disclaimer. Use mocked HTTP in tests. Try three real points (Pakistan, Africa, Europe) once and report which sources worked, with the latency of each, without printing any URL. Before choosing the population source, check that its access method works from this machine, and report the choice.
```

## 17. Decisions needed from the owner or teacher

1. Confirm that "safety controls" means response and exposure information (fire station, hospital, emergency number, population, buildings).
2. Population source: WorldPop or GHSL, after the access check in Prompt C.
3. Whether travel time to the fire station is wanted (needs a routing service with a key or a self hosted router).
4. Whether the three test points for step 3 are acceptable.
5. Who pushes during the handover.
6. Whether to retrain on NOAA-20 archive data now that S-NPP delivery has ended.

## 18. Useful numbers for the demo

- 2,349,116 labelled detections, 8 regions, 5 continents, January to June 2025.
- History tier PR-AUC 0.905 grouped, 0.927 temporal, against a rule at 0.768 and 0.774.
- Held out regions 0.793 to 0.937, DR Congo 0.364 (365 static rows).
- NOAA-20 0.910 and NOAA-21 0.921 with the frozen model.
- Live store 11.8 million rows, 23 boxes, 90 days, models 0.3 MB each.
- Limits to say first: NASA label, weak instant tier, live results unvalidated.

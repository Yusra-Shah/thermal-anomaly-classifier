# Summary

Setup: label = NASA type (static = 2, fire = 0, types 1 and 3 dropped). The type
label is produced by NASA's own algorithm, so it is not independent ground truth.
Headline features exclude hour_utc and is_noaa20 (constant). Splits are grouped
by grid cell. Data: three FIRMS SNPP archive files, 2025-01-01 to 2025-06-30,
257,841 rows after dropping types 1 and 3, 14.3% static.

## 1. Leave one region out (100 trees, 3 grouped folds for the threshold)

Static class. "Baseline" is the PR-AUC of a majority classifier (the static share).

| Held out region | Model | hour_utc | Baseline | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|---|---|---|
| South Asia | Random Forest | no | 0.030 | 0.148 | 0.898 | 0.254 | 0.163 |
| South Asia | XGBoost | no | 0.030 | 0.150 | 0.914 | 0.258 | 0.180 |
| South Asia | Random Forest | yes | 0.030 | 0.133 | 0.022 | 0.037 | 0.071 |
| South Asia | XGBoost | yes | 0.030 | 0.000 | 0.000 | 0.000 | 0.128 |
| Southeast Asia | Random Forest | no | 0.272 | 0.559 | 0.648 | 0.600 | 0.510 |
| Southeast Asia | XGBoost | no | 0.272 | 0.563 | 0.670 | 0.612 | 0.506 |
| Southeast Asia | Random Forest | yes | 0.272 | 0.000 | 0.000 | 0.000 | 0.350 |
| Southeast Asia | XGBoost | yes | 0.272 | 0.000 | 0.000 | 0.000 | 0.448 |
| North and Central America | Random Forest | no | 0.161 | 0.405 | 0.680 | 0.507 | 0.397 |
| North and Central America | XGBoost | no | 0.161 | 0.438 | 0.741 | 0.550 | 0.422 |
| North and Central America | Random Forest | yes | 0.161 | 0.414 | 0.457 | 0.435 | 0.391 |
| North and Central America | XGBoost | yes | 0.161 | 0.356 | 0.185 | 0.243 | 0.388 |

Reading: without hour_utc the models beat the baseline in every region
(PR-AUC about 5 times baseline in South Asia, 1.9 times in Southeast Asia, 2.5
times in the Americas), but precision is low where static sources are rare
(South Asia, 3% static: precision 0.15). With hour_utc, performance on an unseen
region falls apart: the chosen thresholds do not transfer, giving zero or near
zero recall in two regions. hour_utc encodes the region, not a transferable
signature, so it stays an ablation only.

## 2. Stable headline numbers (5 grouped splits, seeds 42 to 46, 300 trees)

Headline setup, static class, mean (std).

| Model | PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| Random Forest | 0.524 (0.026) | 0.443 (0.027) | 0.826 (0.014) | 0.576 (0.024) |
| XGBoost | 0.529 (0.036) | 0.442 (0.029) | 0.839 (0.021) | 0.578 (0.025) |

Majority baseline PR-AUC is about 0.15. The two models are indistinguishable
given the spread. Per-seed values are in repeat_eval_headline_seeds_runs.csv.

## 3. Near real time sanity check

File: SUOMI_VIIRS_C2_Global_7d.csv, 2026-10-27 to 2026-11-02, 324,823 detections.
Every row is SNPP (satellite code N), the same satellite as the training data.
Scored with the saved headline Random Forest (threshold 0.342). There is no
ground truth here, so this is a sanity check, not a score.

Overall predicted static share: 18.7%.

Predicted static share by continent (rough lat/lon boxes, boundaries approximate):

| Continent | Detections | Predicted static |
|---|---|---|
| Africa | 192,732 | 11.2% |
| Asia | 55,073 | 30.5% |
| Europe | 5,693 | 38.5% |
| North America | 8,604 | 41.8% |
| Oceania | 18,473 | 22.1% |
| South America | 43,981 | 28.6% |
| Other | 267 | 19.9% |

Cells seen on 4 or more of the 7 days against the rest:

| Group | Detections | Predicted static |
|---|---|---|
| Cell seen on 4 or more days | 17,030 | 57.8% |
| Other cells | 307,793 | 16.5% |

Reading: repeat cells are flagged static 3.5 times as often as the rest, which
is the direction expected. Africa has the lowest static share, as expected for a
week dominated by seasonal savanna burning. Even so, 42% of repeat cells are not
predicted static, and 16.5% of non-repeat detections are, so the model is a weak
single detection signal and should not be read as a precise classifier. The
continent shares are also affected by the model being trained on three regions
only, with a static share that differed a lot between them.

## Honest caveats

- Static precision is below 0.5 even in the best case, so about half of the
  detections flagged static are actually fires per NASA's own label.
- Cross region transfer is weak where static sources are rare (South Asia).
- The label comes from NASA's algorithm, so part of what is learned is that
  algorithm's behaviour.

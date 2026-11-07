# Summary: history tier

Label: NASA type (static = 2, fire = 0, types 1 and 3 dropped). The type label is produced by NASA's own algorithm. No coordinates, no hour_utc.

Setup A uses a single reading. **Setups B and C use past detections of the same place and are not single reading results.** History uses only detections on days strictly before the detection, within the same raw file.

Models: Random Forest and XGBoost with 100 trees, MLP (64, 32, ReLU, dropout 0.2). Thresholds come from 3 grouped folds on the training part. The rule is static if distinct_days_prior_30 >= k, with k picked by static F1 on the training part.

## Leakage test

tests/test_history.py: for 40 random detections, every other row on or after the detection's date was moved, re-dated, dropped and duplicated. The detection's history features did not change in any case (passed). A hand computed known value test also passed.

## a) Five grouped splits (seeds 42 to 46), mean (std) of static metrics

Test rows: all rows (about 51,325 per split). Majority baseline PR-AUC about 0.142.

| Setup | Model | PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Rule | distinct_days_prior_30 >= k, k per seed 3,3,3,3,3 | 0.777 (0.040) | 0.765 (0.031) | 0.811 (0.017) | 0.787 (0.023) |
| A instant | Random Forest | 0.520 (0.027) | 0.442 (0.027) | 0.827 (0.013) | 0.576 (0.024) |
| A instant | XGBoost | 0.523 (0.037) | 0.438 (0.025) | 0.846 (0.020) | 0.577 (0.024) |
| A instant | MLP | 0.441 (0.028) | 0.422 (0.027) | 0.871 (0.009) | 0.568 (0.025) |
| B instant+history | Random Forest | 0.919 (0.019) | 0.837 (0.018) | 0.905 (0.011) | 0.870 (0.014) |
| B instant+history | XGBoost | 0.922 (0.019) | 0.836 (0.021) | 0.906 (0.010) | 0.870 (0.015) |
| B instant+history | MLP | 0.913 (0.018) | 0.837 (0.022) | 0.893 (0.009) | 0.864 (0.015) |
| C history only | Random Forest | 0.886 (0.023) | 0.821 (0.022) | 0.879 (0.010) | 0.849 (0.016) |
| C history only | XGBoost | 0.905 (0.020) | 0.827 (0.027) | 0.891 (0.004) | 0.857 (0.016) |
| C history only | MLP | 0.901 (0.018) | 0.821 (0.024) | 0.887 (0.006) | 0.853 (0.015) |

Test rows: days_into_data >= 30 (about 45,756 per split). Majority baseline PR-AUC about 0.143.

| Setup | Model | PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Rule | distinct_days_prior_30 >= k, k per seed 3,3,3,3,3 | 0.795 (0.040) | 0.764 (0.033) | 0.846 (0.016) | 0.803 (0.024) |
| A instant | Random Forest | 0.525 (0.026) | 0.444 (0.027) | 0.825 (0.012) | 0.577 (0.024) |
| A instant | XGBoost | 0.527 (0.035) | 0.440 (0.025) | 0.844 (0.021) | 0.579 (0.024) |
| A instant | MLP | 0.442 (0.028) | 0.424 (0.027) | 0.870 (0.010) | 0.570 (0.025) |
| B instant+history | Random Forest | 0.926 (0.019) | 0.842 (0.018) | 0.920 (0.012) | 0.879 (0.014) |
| B instant+history | XGBoost | 0.930 (0.019) | 0.843 (0.021) | 0.920 (0.012) | 0.880 (0.016) |
| B instant+history | MLP | 0.921 (0.019) | 0.845 (0.023) | 0.909 (0.011) | 0.876 (0.016) |
| C history only | Random Forest | 0.898 (0.023) | 0.832 (0.023) | 0.904 (0.009) | 0.866 (0.016) |
| C history only | XGBoost | 0.916 (0.020) | 0.835 (0.029) | 0.914 (0.005) | 0.873 (0.018) |
| C history only | MLP | 0.912 (0.019) | 0.831 (0.025) | 0.910 (0.008) | 0.869 (0.017) |

## b) Leave one region out

Held out: South Asia (46,909 rows). Majority baseline PR-AUC 0.030. Rule k = 3.

| Setup | Model | PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Rule | distinct_days_prior_30 >= 3 | 0.701 | 0.716 | 0.756 | 0.735 |
| A instant | Random Forest | 0.163 | 0.148 | 0.898 | 0.254 |
| A instant | XGBoost | 0.180 | 0.150 | 0.914 | 0.258 |
| A instant | MLP | 0.163 | 0.147 | 0.975 | 0.256 |
| B instant+history | Random Forest | 0.799 | 0.729 | 0.838 | 0.780 |
| B instant+history | XGBoost | 0.826 | 0.728 | 0.845 | 0.782 |
| B instant+history | MLP | 0.774 | 0.730 | 0.811 | 0.768 |
| C history only | Random Forest | 0.694 | 0.600 | 0.814 | 0.691 |
| C history only | XGBoost | 0.740 | 0.676 | 0.828 | 0.744 |
| C history only | MLP | 0.762 | 0.634 | 0.865 | 0.732 |

Held out: Southeast Asia (14,034 rows). Majority baseline PR-AUC 0.272. Rule k = 3.

| Setup | Model | PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Rule | distinct_days_prior_30 >= 3 | 0.780 | 0.783 | 0.789 | 0.786 |
| A instant | Random Forest | 0.510 | 0.559 | 0.648 | 0.601 |
| A instant | XGBoost | 0.506 | 0.563 | 0.670 | 0.612 |
| A instant | MLP | 0.463 | 0.561 | 0.699 | 0.622 |
| B instant+history | Random Forest | 0.847 | 0.797 | 0.859 | 0.827 |
| B instant+history | XGBoost | 0.863 | 0.795 | 0.875 | 0.833 |
| B instant+history | MLP | 0.852 | 0.795 | 0.849 | 0.821 |
| C history only | Random Forest | 0.834 | 0.774 | 0.875 | 0.821 |
| C history only | XGBoost | 0.852 | 0.780 | 0.890 | 0.831 |
| C history only | MLP | 0.852 | 0.785 | 0.876 | 0.828 |

Held out: North and Central America (196,898 rows). Majority baseline PR-AUC 0.161. Rule k = 3.

| Setup | Model | PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Rule | distinct_days_prior_30 >= 3 | 0.786 | 0.767 | 0.817 | 0.791 |
| A instant | Random Forest | 0.397 | 0.405 | 0.680 | 0.507 |
| A instant | XGBoost | 0.422 | 0.438 | 0.741 | 0.550 |
| A instant | MLP | 0.377 | 0.408 | 0.856 | 0.552 |
| B instant+history | Random Forest | 0.893 | 0.806 | 0.894 | 0.848 |
| B instant+history | XGBoost | 0.879 | 0.792 | 0.915 | 0.849 |
| B instant+history | MLP | 0.844 | 0.789 | 0.911 | 0.846 |
| C history only | Random Forest | 0.861 | 0.797 | 0.872 | 0.833 |
| C history only | XGBoost | 0.856 | 0.808 | 0.880 | 0.843 |
| C history only | MLP | 0.842 | 0.807 | 0.871 | 0.838 |

## c) Temporal split

Train: detections before 2025-05-01. Test: 2025-05-01 to 2025-06-30 (85,955 rows, history complete). Majority baseline PR-AUC 0.183. Rule k = 3. Cells can appear in both train and test here, as they would in use.

| Setup | Model | PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Rule | distinct_days_prior_30 >= 3 | 0.840 | 0.763 | 0.872 | 0.814 |
| A instant | Random Forest | 0.472 | 0.434 | 0.782 | 0.558 |
| A instant | XGBoost | 0.492 | 0.439 | 0.809 | 0.569 |
| A instant | MLP | 0.442 | 0.417 | 0.876 | 0.565 |
| B instant+history | Random Forest | 0.922 | 0.823 | 0.934 | 0.875 |
| B instant+history | XGBoost | 0.925 | 0.826 | 0.930 | 0.875 |
| B instant+history | MLP | 0.926 | 0.823 | 0.946 | 0.880 |
| C history only | Random Forest | 0.898 | 0.793 | 0.930 | 0.856 |
| C history only | XGBoost | 0.906 | 0.822 | 0.925 | 0.871 |
| C history only | MLP | 0.918 | 0.813 | 0.948 | 0.875 |

## Rule baseline comparison

The rule is a single threshold on one history count, with k chosen on the training
part. It picked k = 3 in every split. Its PR-AUC uses the count itself as the score.

The rule does not match the best models, but it gets most of the way from the instant
tier to them:

| Evaluation | A instant F1 (best of 3) | Rule F1 | B F1 (best of 3) | Rule PR-AUC | B PR-AUC (best of 3) |
|---|---|---|---|---|---|
| 5 splits, all rows | 0.577 | 0.787 | 0.870 | 0.777 | 0.922 |
| 5 splits, history complete | 0.579 | 0.803 | 0.880 | 0.795 | 0.930 |
| South Asia held out | 0.258 | 0.735 | 0.782 | 0.701 | 0.826 |
| Southeast Asia held out | 0.622 | 0.786 | 0.833 | 0.780 | 0.863 |
| North and Central America held out | 0.552 | 0.791 | 0.849 | 0.786 | 0.893 |
| Temporal split | 0.569 | 0.814 | 0.880 | 0.840 | 0.926 |

Setup B beats the rule by 0.047 to 0.083 F1 and by 0.083 to 0.145 PR-AUC in these
evaluations. In the 5 split and temporal tests rule precision is 0.763 to 0.765 where the best
B models reach 0.826 to 0.845. So the gain over a one-line rule is real but
smaller than the gain of history over the instant tier.

## Reading and caveats

- History is the main driver. Moving from A to B raises static F1 from about 0.58 to
  about 0.87 in the 5 splits. History alone (C) is close to B (F1 0.849 to 0.857
  against 0.864 to 0.870), so the instant readings add little once history is known.
- The type label is NASA's own, and NASA's static flag depends on repeat detections
  at a place. History features are close to the definition of the label, so high B and
  C scores are expected and should not be read as the model discovering a physical
  signature. They show that past detections predict NASA's flag well, not that the
  flag is correct.
- B and C need weeks of past data for the same place. They are not single reading
  results and cannot be applied to a first-time detection.
- The instant tier (A) is unchanged and remains weak: static precision about 0.42 to
  0.44 and PR-AUC about 0.44 to 0.52 in the 5 splits.
- The MLP is the weakest on A (PR-AUC 0.441 against 0.520 and 0.523 for the trees).
  On B it is 0.009 to 0.052 PR-AUC below the best tree model, except the temporal
  split where it is ahead by 0.001 (0.926 against 0.925). On C it matches or beats
  the trees in South Asia (0.762 against 0.740), Southeast Asia (0.852, tied with
  XGBoost) and the temporal split (0.918 against 0.906), and is below XGBoost in the
  5 splits (0.901 against 0.905) and North and Central America (0.842 against 0.861
  for Random Forest). Most of these margins are inside the split to split spread
  seen in part a (std about 0.02).
- Part a std values come from five splits of the same data, so they understate
  uncertainty about new regions or years. All data is Jan to Jun 2025.
- South Asia is the hardest held out region (3% static). B reaches F1 0.768 to 0.782
  there, and the rule 0.735.
- In the temporal split the same cells can appear in train and test. That matches
  use, but it is easier than a split on unseen places.
- The rule k and the model thresholds are chosen on training data only. In the
  temporal split the training part includes rows with incomplete history (the first
  30 days of the file).

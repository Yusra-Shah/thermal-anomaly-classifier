NOAA-20 transfer summary

Models trained on Suomi NPP train cells (seed 42 grouped split), tested on NOAA-20. Thresholds, calibrators and bands come from Suomi NPP out of fold training scores, unchanged. The label is NASA's repeat based type flag, not independent truth. The satellites observe the same places within a short time of each other (see the offsets below), so the test is a transfer between near twins, not an independent check.

Share of NOAA-20 rows in cells unseen in Suomi NPP training: 0.451. Rule k: 3.

history, Suomi NPP held out cells (reference): n 52354, static PR-AUC 0.932 (majority 0.153), precision 0.894, recall 0.840, F1 0.866
history, cell disjoint: n 116372, static PR-AUC 0.901 (majority 0.070), precision 0.874, recall 0.800, F1 0.835
history, not cell disjoint: n 141943, static PR-AUC 0.932 (majority 0.199), precision 0.891, recall 0.834, F1 0.862
instant, Suomi NPP held out cells (reference): n 52354, static PR-AUC 0.507 (majority 0.153), precision 0.456, recall 0.816, F1 0.585
instant, cell disjoint: n 116372, static PR-AUC 0.371 (majority 0.070), precision 0.317, recall 0.803, F1 0.455
instant, not cell disjoint: n 141943, static PR-AUC 0.558 (majority 0.199), precision 0.480, recall 0.836, F1 0.610
rule, Suomi NPP held out cells (reference): n 52354, static PR-AUC 0.780 (majority 0.153), precision 0.768, recall 0.829, F1 0.798
rule, cell disjoint: n 116372, static PR-AUC 0.731 (majority 0.070), precision 0.767, recall 0.805, F1 0.785
rule, not cell disjoint: n 141943, static PR-AUC 0.754 (majority 0.199), precision 0.750, recall 0.808, F1 0.778

Row counts per region (country cut):
                   region  noaa20_rows  snpp_rows    ratio
               South Asia        48513      46922 1.033907
           Southeast Asia        14879      15275 0.974075
North and Central America       204019     205088 0.994788

Matched pairs on six days: 4,876. Share of NOAA-20 detections with a Suomi NPP match 0.62, share of Suomi NPP detections with a NOAA-20 match 0.44.
Acquisition time offset NOAA-20 minus Suomi NPP: median of the daily medians 22 minutes, range of the daily medians -78 to 23.
Brightness I-4 difference: median -0.43 K, mean absolute 9.1 K. Brightness I-5 median 1.10 K. FRP median difference -0.27 MW, median ratio 0.93.
The sensors are the same instrument type on different platforms. Single detections differ by several kelvin because the overpass time and the pixel footprint differ.

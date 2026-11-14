NOAA-20 transfer summary

Models trained on Suomi NPP train cells (seed 42 grouped split), tested on NOAA-20. Thresholds, calibrators and bands come from Suomi NPP out of fold training scores, unchanged. The label is NASA's repeat based type flag, not independent truth. The satellites observe the same places within a short time of each other (see the offsets below), so the test is a transfer between near twins, not an independent check.

Share of NOAA-20 rows in cells unseen in Suomi NPP training: 0.451. Rule k: 5.

history, Suomi NPP held out cells (reference): n 470027, static PR-AUC 0.905 (majority 0.042), precision 0.893, recall 0.781, F1 0.833
history, cell disjoint: n 116420, static PR-AUC 0.910 (majority 0.068), precision 0.921, recall 0.651, F1 0.763
history, not cell disjoint: n 141895, static PR-AUC 0.913 (majority 0.200), precision 0.911, recall 0.700, F1 0.791
instant, Suomi NPP held out cells (reference): n 470027, static PR-AUC 0.244 (majority 0.042), precision 0.232, recall 0.575, F1 0.331
instant, cell disjoint: n 116420, static PR-AUC 0.319 (majority 0.068), precision 0.327, recall 0.645, F1 0.434
instant, not cell disjoint: n 141895, static PR-AUC 0.493 (majority 0.200), precision 0.488, recall 0.643, F1 0.555
rule, Suomi NPP held out cells (reference): n 470027, static PR-AUC 0.768 (majority 0.042), precision 0.818, recall 0.788, F1 0.803
rule, cell disjoint: n 116420, static PR-AUC 0.769 (majority 0.068), precision 0.854, recall 0.686, F1 0.761
rule, not cell disjoint: n 141895, static PR-AUC 0.745 (majority 0.200), precision 0.811, recall 0.685, F1 0.743

Row counts per region (country cut):
                   region  noaa20_rows  snpp_rows    ratio
               South Asia        48513      46922 1.033907
           Southeast Asia        14879      15275 0.974075
North and Central America       204019     205088 0.994788

Matched pairs on six days: 4,876. Share of NOAA-20 detections with a Suomi NPP match 0.62, share of Suomi NPP detections with a NOAA-20 match 0.44.
Acquisition time offset NOAA-20 minus Suomi NPP: median of the daily medians 22 minutes, range of the daily medians -78 to 23.
Brightness I-4 difference: median -0.43 K, mean absolute 9.1 K. Brightness I-5 median 1.10 K. FRP median difference -0.27 MW, median ratio 0.93.
The sensors are the same instrument type on different platforms. Single detections differ by several kelvin because the overpass time and the pixel footprint differ.

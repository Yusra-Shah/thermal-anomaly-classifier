NOAA-21 transfer summary

Models trained on Suomi NPP train cells (seed 42 grouped split), tested on NOAA-21. Thresholds, calibrators and bands come from Suomi NPP out of fold training scores, unchanged. The label is NASA's repeat based type flag, not independent truth. The satellites observe the same places within a short time of each other (see the offsets below), so the test is a transfer between near twins, not an independent check.

Share of NOAA-21 rows in cells unseen in Suomi NPP training: 0.446. Rule k: 5.

history, Suomi NPP held out cells (reference): n 470027, static PR-AUC 0.905 (majority 0.042), precision 0.893, recall 0.781, F1 0.833
history, cell disjoint: n 111224, static PR-AUC 0.921 (majority 0.068), precision 0.927, recall 0.668, F1 0.776
history, not cell disjoint: n 138073, static PR-AUC 0.920 (majority 0.204), precision 0.917, recall 0.702, F1 0.796
instant, Suomi NPP held out cells (reference): n 470027, static PR-AUC 0.244 (majority 0.042), precision 0.232, recall 0.575, F1 0.331
instant, cell disjoint: n 111224, static PR-AUC 0.311 (majority 0.068), precision 0.315, recall 0.698, F1 0.434
instant, not cell disjoint: n 138073, static PR-AUC 0.495 (majority 0.204), precision 0.491, recall 0.700, F1 0.577
rule, Suomi NPP held out cells (reference): n 470027, static PR-AUC 0.768 (majority 0.042), precision 0.818, recall 0.788, F1 0.803
rule, cell disjoint: n 111224, static PR-AUC 0.798 (majority 0.068), precision 0.859, recall 0.690, F1 0.766
rule, not cell disjoint: n 138073, static PR-AUC 0.763 (majority 0.204), precision 0.823, recall 0.684, F1 0.747

Row counts per region (country cut):
                   region  noaa21_rows  snpp_rows    ratio
               South Asia        45847      46922 0.977090
           Southeast Asia        14403      15275 0.942913
North and Central America       198012     205088 0.965498

Matched pairs on six days: 5,029. Share of NOAA-21 detections with a Suomi NPP match 0.61, share of Suomi NPP detections with a NOAA-21 match 0.43.
Acquisition time offset NOAA-21 minus Suomi NPP: median of the daily medians -26 minutes, range of the daily medians -27 to 74.
Brightness I-4 difference: median -0.63 K, mean absolute 10.1 K. Brightness I-5 median 0.33 K. FRP median difference 0.45 MW, median ratio 1.16.

Matched pairs on six days: 4,735. Share of NOAA-21 detections with a NOAA-20 match 0.60, share of NOAA-20 detections with a NOAA-21 match 0.41.
Acquisition time offset NOAA-21 minus NOAA-20: median of the daily medians 51 minutes, range of the daily medians -49 to 52.
Brightness I-4 difference: median -1.02 K, mean absolute 10.3 K. Brightness I-5 median -1.69 K. FRP median difference 0.49 MW, median ratio 1.21.
The sensors are the same instrument type on different platforms. Single detections differ by several kelvin because the overpass time and the pixel footprint differ.

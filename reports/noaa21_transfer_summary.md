NOAA-21 transfer summary

Models trained on Suomi NPP train cells (seed 42 grouped split), tested on NOAA-21. Thresholds, calibrators and bands come from Suomi NPP out of fold training scores, unchanged. The label is NASA's repeat based type flag, not independent truth. The satellites observe the same places within a short time of each other (see the offsets below), so the test is a transfer between near twins, not an independent check.

Share of NOAA-21 rows in cells unseen in Suomi NPP training: 0.446. Rule k: 3.

history, Suomi NPP held out cells (reference): n 52354, static PR-AUC 0.932 (majority 0.153), precision 0.894, recall 0.840, F1 0.866
history, cell disjoint: n 111145, static PR-AUC 0.914 (majority 0.070), precision 0.877, recall 0.828, F1 0.852
history, not cell disjoint: n 138152, static PR-AUC 0.943 (majority 0.203), precision 0.896, recall 0.843, F1 0.869
instant, Suomi NPP held out cells (reference): n 52354, static PR-AUC 0.507 (majority 0.153), precision 0.456, recall 0.816, F1 0.585
instant, cell disjoint: n 111145, static PR-AUC 0.359 (majority 0.070), precision 0.305, recall 0.815, F1 0.444
instant, not cell disjoint: n 138152, static PR-AUC 0.539 (majority 0.203), precision 0.478, recall 0.831, F1 0.607
rule, Suomi NPP held out cells (reference): n 52354, static PR-AUC 0.780 (majority 0.153), precision 0.768, recall 0.829, F1 0.798
rule, cell disjoint: n 111145, static PR-AUC 0.726 (majority 0.070), precision 0.758, recall 0.800, F1 0.779
rule, not cell disjoint: n 138152, static PR-AUC 0.782 (majority 0.203), precision 0.767, recall 0.806, F1 0.786

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

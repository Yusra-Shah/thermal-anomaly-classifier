World summary

Every region file in data/raw (January to June 2025) is now used. Leave one region out: train on all other regions, test on the held out one. For the three original regions the first number is the earlier result (trained on the other two regions) and the second the new result (trained on the other seven). New regions show the new result only. History is setup D, XGBoost, 100 trees, threshold at the F1 optimum of out of fold training scores. Instant is XGBoost, 100 trees, same rule. The rule is distinct days in the prior 30 at least k, k picked on the training part. The label is NASA's repeat based type flag, not independent truth.

| Region | Rows | Static share | History PR-AUC | History F1 | Instant PR-AUC | Instant F1 | Rule PR-AUC | Rule F1 |
|---|---|---|---|---|---|---|---|---|
| South Asia | 46,909 | 0.0297 | 0.859 to 0.881 | 0.806 to 0.816 | 0.180 to 0.148 | 0.258 to 0.237 | 0.701 to 0.701 | 0.735 to 0.699 |
| Southeast Asia | 14,034 | 0.2724 | 0.856 to 0.845 | 0.806 to 0.792 | 0.506 to 0.578 | 0.612 to 0.548 | 0.780 to 0.780 | 0.786 to 0.730 |
| North and Central America | 196,898 | 0.1611 | 0.891 to 0.920 | 0.849 to 0.864 | 0.422 to 0.510 | 0.550 to 0.423 | 0.786 to 0.786 | 0.791 to 0.767 |
| Brazil | 117,776 | 0.0616 | 0.937 | 0.882 | 0.177 | 0.272 | 0.729 | 0.624 |
| Australia | 412,943 | 0.0128 | 0.934 | 0.892 | 0.038 | 0.073 | 0.823 | 0.784 |
| India | 513,677 | 0.0848 | 0.890 | 0.892 | 0.298 | 0.281 | 0.870 | 0.862 |
| Nigeria | 332,845 | 0.0194 | 0.793 | 0.777 | 0.079 | 0.143 | 0.634 | 0.737 |
| DR Congo | 714,034 | 0.0005 | 0.364 | 0.412 | 0.026 | 0.032 | 0.345 | 0.180 |

Where a number got worse (original regions, earlier to new):

- South Asia, instant PR-AUC: 0.180 to 0.148
- South Asia, instant F1: 0.258 to 0.237
- South Asia, rule F1: 0.735 to 0.699
- Southeast Asia, history PR-AUC: 0.856 to 0.845
- Southeast Asia, history F1: 0.806 to 0.792
- Southeast Asia, instant F1: 0.612 to 0.548
- Southeast Asia, rule F1: 0.786 to 0.730
- North and Central America, instant F1: 0.550 to 0.423
- North and Central America, rule F1: 0.791 to 0.767

Grouped split by grid cell, seed 42, all regions together (20 percent of cells held out), history tier:

- PR-AUC 0.932 (three regions) to 0.905 (eight regions), F1 0.882 to 0.882. These are different test sets, so the two numbers are not a like for like comparison.

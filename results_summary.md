---
# BENCHMARK RESULTS SUMMARY
Generated from: results/ folder
Total solutions: 7
Rounds per solution: 20 (All solutions completed 20 rounds)

---
## FINAL ROUND METRICS TABLE
| Solution | Acc (%) | Fairness | Precision | Recall | F1 | Time (s) |
|---|---|---|---|---|---|---|
| Baseline | 87.03 | 0.8923 | 0.8503 | 0.8507 | 0.8505 | 128.02 |
| BatchNorm | 87.79 | 0.8695 | 0.8617 | 0.8541 | 0.8577 | 119.44 |
| GroupNorm | 83.43 | 0.9305 | 0.8072 | 0.8216 | 0.8134 | 117.35 |
| LayerNorm | 79.06 | 0.9647 | 0.7663 | 0.7971 | 0.7737 | 103.15 |
| InstanceNorm | 76.91 | 0.9886 | 0.7436 | 0.7712 | 0.7501 | 178.71 |
| FocalLoss | 86.46 | 0.8889 | 0.8436 | 0.8444 | 0.8440 | 112.85 |
| L2 Reg. | 83.68 | 0.9560 | 0.8100 | 0.8288 | 0.8176 | 115.78 |

---
## BEST ROUND ACHIEVED (Peak Accuracy)
| Solution | Best Round | Best Acc (%) | F1 at Best | Fairness at Best |
|---|---|---|---|---|
| Baseline | 18 | 88.36 | 0.8640 | 0.8670 |
| BatchNorm | 19 | 88.17 | 0.8610 | 0.8522 |
| GroupNorm | 16 | 85.77 | 0.8328 | 0.8320 |
| LayerNorm | 11 | 82.92 | 0.8060 | 0.9000 |
| InstanceNorm | 10 | 79.70 | 0.7699 | 0.8801 |
| FocalLoss | 15 | 87.48 | 0.8538 | 0.8624 |
| L2 Reg. | 14 | 85.64 | 0.8326 | 0.8514 |

---
## PER-CLASS ACCURACY AT FINAL ROUND
| Solution | Benign Acc (%) | Malignant Acc (%) |
|---|---|---|
| Baseline | 79.68 | 90.45 |
| BatchNorm | 78.88 | 91.94 |
| GroupNorm | 78.69 | 85.63 |
| LayerNorm | 81.47 | 77.94 |
| InstanceNorm | 77.69 | 76.55 |
| FocalLoss | 78.88 | 89.99 |
| L2 Reg. | 80.68 | 85.08 |

---
## PER-ROUND ACCURACY (all rounds)
```
Baseline:       [68.2479, 84.4402, 85.2625, 85.9583, 85.7685, 86.7805, 87.1600, 86.8438, 84.3770, 87.9823, 86.9070, 87.0335, 87.9823, 86.9070, 88.1720, 88.1720, 88.0455, 88.3618, 88.1720, 87.0335]
BatchNorm:      [83.4915, 84.7565, 84.3770, 84.8197, 86.0215, 86.4643, 87.0968, 84.6932, 85.5787, 87.0968, 86.3378, 86.2745, 86.4010, 85.9583, 87.4130, 87.6660, 86.3378, 87.4763, 88.1720, 87.7925]
GroupNorm:      [31.7521, 78.8109, 79.2536, 77.4826, 79.7596, 78.7476, 77.5459, 79.3169, 81.5307, 82.6692, 81.9102, 84.5035, 80.5187, 83.7445, 84.8830, 85.7685, 83.6180, 85.4522, 84.5035, 83.4282]
LayerNorm:      [68.2479, 81.0879, 79.1271, 80.2024, 80.5819, 81.9102, 80.2657, 79.8861, 79.7596, 79.8861, 82.9222, 81.0879, 79.9494, 79.6331, 81.5939, 81.4674, 77.6091, 80.2024, 80.0759, 79.0639]
InstanceNorm:   [78.2416, 66.9829, 71.8533, 73.2448, 74.3201, 74.2568, 75.1423, 73.4345, 72.4858, 79.6964, 77.4826, 70.9677, 74.2568, 74.5731, 76.7868, 76.4073, 75.7748, 77.5459, 73.2448, 76.9133]
FocalLoss:      [83.6180, 85.5155, 83.8710, 85.7685, 85.3257, 85.3257, 84.8197, 85.1360, 85.3257, 86.2113, 85.2625, 85.7052, 86.2113, 86.8438, 87.4763, 85.3890, 87.3498, 86.0848, 86.2745, 86.4643]
L2 Reg.:        [83.5547, 82.9855, 83.7445, 82.4794, 83.3017, 83.9975, 84.3137, 83.8077, 84.9462, 82.9222, 83.6180, 84.1240, 81.9102, 85.6420, 83.7445, 85.0095, 82.6059, 85.0727, 83.9975, 83.6812]
```

---
## PER-ROUND LOSS (all rounds, if available)
```
Baseline:       [0.6249, 0.5246, 0.4842, 0.4615, 0.4499, 0.4675, 0.4333, 0.4167, 0.4446, 0.4190, 0.3897, 0.4057, 0.4115, 0.4026, 0.3888, 0.3771, 0.4002, 0.3873, 0.3909, 0.3864]
BatchNorm:      [0.5213, 0.4679, 0.4541, 0.4351, 0.4368, 0.4286, 0.4137, 0.4145, 0.4242, 0.3996, 0.4123, 0.4063, 0.3896, 0.3851, 0.3905, 0.3800, 0.3848, 0.3824, 0.3833, 0.3635]
GroupNorm:      [0.6775, 0.5825, 0.5063, 0.4836, 0.4762, 0.4570, 0.4625, 0.4460, 0.4342, 0.4304, 0.4291, 0.4277, 0.4250, 0.4188, 0.4132, 0.4108, 0.4131, 0.4138, 0.4031, 0.4073]
LayerNorm:      [0.6658, 0.5594, 0.5070, 0.4806, 0.4616, 0.4576, 0.4560, 0.4433, 0.4360, 0.4324, 0.4358, 0.4267, 0.4190, 0.4213, 0.4188, 0.4231, 0.4265, 0.4152, 0.4152, 0.4182]
InstanceNorm:   [0.6627, 0.6004, 0.5439, 0.5060, 0.4946, 0.5089, 0.4884, 0.4870, 0.4751, 0.4625, 0.4752, 0.4566, 0.4533, 0.4415, 0.4452, 0.4459, 0.4533, 0.4391, 0.4522, 0.4310]
FocalLoss:      [0.1521, 0.1252, 0.1239, 0.1189, 0.1145, 0.1124, 0.1139, 0.1095, 0.1112, 0.1105, 0.1055, 0.1082, 0.1051, 0.1040, 0.1052, 0.1031, 0.1045, 0.1029, 0.1012, 0.1004]
L2 Reg.:        [0.5275, 0.5051, 0.4967, 0.4868, 0.4766, 0.4812, 0.4852, 0.4684, 0.4693, 0.4778, 0.4629, 0.4598, 0.4506, 0.4519, 0.4492, 0.4494, 0.4512, 0.4457, 0.4415, 0.4393]
```

---
## GRAPH FILES FOUND
- /Users/slender/Developer/Codes/pivot-b-resource-fairness/results/accuracy_fairness_tradeoff.png
- /Users/slender/Developer/Codes/pivot-b-resource-fairness/results/convergence_curves.png
- /Users/slender/Developer/Codes/pivot-b-resource-fairness/results/f1_precision_recall.png
- /Users/slender/Developer/Codes/pivot-b-resource-fairness/results/loss_curves.png
- /Users/slender/Developer/Codes/pivot-b-resource-fairness/results/training_time_comparison.png

*(No aggregated summary CSV files or analysis.py text outputs found in the project. analysis.py generates visual plots but prints text summaries only to the console.)*

---
## RAW JSON STRUCTURE
### Top-level Keys:
- `solution_name`
- `norm_type`
- `loss_type`
- `description`
- `num_rounds`
- `local_epochs`
- `num_clients`
- `rounds`
- `results_path`

### Keys in each round object (within `rounds` list):
- `round`
- `accuracy`
- `fairness_score`
- `avg_train_loss`
- `class_accuracy`
- `macro_precision`
- `macro_recall`
- `macro_f1`
- `round_time_sec`
- `clients`

### Keys in each client object (within `clients` list):
- `avg_loss`
- `train_accuracy`
- `num_batches`
- `num_samples`
- `training_time_sec`
- `client_id`

### Additional/Unexpected Keys:
No unexpected or additional keys found. The structure is identical across all solutions.

---
## NOTES
### General Run Details
- **Result Files Status**: All 7 expected result files were found and read successfully.
- **Rounds Completed**: All 7 solutions successfully completed exactly **20 rounds** each.
- **Metric Keys Status**: All standard metric keys (`accuracy`, `fairness_score`, `macro_precision`, `macro_recall`, `macro_f1`, `round_time_sec`, and `class_accuracy`) were found in all JSON files.
- **FocalLoss Loss Metric**: Note that the FocalLoss solution records average training loss in the range of **0.1521 to 0.1004**. This is significantly smaller in magnitude than other solutions which range from **0.67 to 0.38** (e.g., L2 Reg. or Baseline). This is because Focal Loss scales down the loss of well-classified samples using a focusing parameter, rather than a bug.

### Convergence Anomalies and Observations
- **Accuracy drop in InstanceNorm**: Decreased by **11.26%** in round 2 (from 78.24% to 66.98%).
- **Accuracy drop in InstanceNorm**: Decreased by **6.51%** in round 12 (from 77.48% to 70.97%).
- **GroupNorm Cold Start**: Started extremely low in round 1 at **31.75%** accuracy, but surged by **47.06%** in round 2 to **78.81%**, eventually converging cleanly to 83.43%.
- **Early Stagnation in LayerNorm**: Reached 81.09% accuracy by round 2 but hovered around 79%-82% for all remaining rounds, never surpassing its round 11 peak of 82.92%. In comparison, Baseline and BatchNorm both converged to over 87%-88% accuracy.
- **High Instability in InstanceNorm**: The accuracy fluctuates heavily between 70% and 79% (e.g. dropping to 70.97% in round 12, peaking at 79.70% in round 10, ending at 76.91%). It does not show stable convergence.
---

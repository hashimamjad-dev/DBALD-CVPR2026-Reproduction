# DBALD Table 1: published reference and independent verification

Published reference only: the following 20 rows reproduce the page-7 values, including printed averages. These are not local attack measurements. All values are percentages. B = benign model; P = poisoned model.

| Model | Method | CULane B ACC | CULane P ACC | LDA | LRA | LOA | Average | TuSimple B ACC | TuSimple P ACC | LDA | LOA | Average |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LaneATT | BadNets | 75.01 | 72.79 | 49.27 | 43.37 | 47.36 | 46.67 | 96.81 | 95.53 | 45.62 | 75.63 | 60.63 |
| LaneATT | Blended | 75.01 | 73.15 | 57.14 | 59.07 | 47.04 | 54.41 | 96.81 | 96.27 | 46.31 | 72.47 | 59.39 |
| LaneATT | LD-Attack | 75.01 | 72.48 | 71.24 | 56.24 | 63.71 | 63.73 | 96.81 | 96.21 | 88.17 | 90.27 | 89.22 |
| LaneATT | BadLANE | 75.01 | 73.78 | 64.33 | 45.22 | 54.15 | 54.56 | 96.81 | 96.18 | 89.25 | 89.76 | 89.51 |
| LaneATT | DBALD | 75.01 | 74.20 | 81.65 | 73.45 | 74.19 | 76.41 | 96.81 | 95.72 | 90.04 | 91.63 | 90.84 |
| ADNet | BadNets | 85.51 | 79.60 | 50.13 | 54.33 | 61.39 | 55.28 | 96.87 | 95.46 | 50.34 | 52.98 | 51.66 |
| ADNet | Blended | 85.51 | 74.37 | 43.67 | 56.42 | 56.51 | 52.20 | 96.87 | 95.10 | 56.27 | 59.12 | 57.70 |
| ADNet | LD-Attack | 85.51 | 79.14 | 59.47 | 63.39 | 67.15 | 63.34 | 96.87 | 96.35 | 53.24 | 66.32 | 59.78 |
| ADNet | BadLANE | 85.51 | 85.46 | 36.61 | 61.16 | 65.95 | 54.57 | 96.87 | 96.29 | 59.17 | 67.48 | 63.33 |
| ADNet | DBALD | 85.51 | 82.37 | 66.49 | 64.82 | 68.17 | 66.50 | 96.87 | 96.61 | 89.24 | 72.07 | 80.66 |
| SCNN | BadNets | 69.75 | 69.52 | 74.39 | 63.63 | 62.76 | 66.93 | 93.78 | 91.56 | 64.72 | 62.79 | 63.76 |
| SCNN | Blended | 69.75 | 63.98 | 75.14 | 63.45 | 61.60 | 66.73 | 93.78 | 91.37 | 58.30 | 64.13 | 61.22 |
| SCNN | LD-Attack | 69.75 | 64.38 | 74.83 | 62.21 | 64.17 | 67.07 | 93.78 | 92.23 | 94.08 | 91.43 | 92.76 |
| SCNN | BadLane | 69.75 | 67.57 | 72.28 | 63.96 | 62.94 | 66.39 | 93.78 | 90.89 | 92.19 | 88.57 | 90.38 |
| SCNN | DBALD | 69.75 | 68.75 | 76.19 | 67.34 | 68.69 | 70.74 | 93.78 | 92.28 | 95.78 | 91.29 | 93.54 |
| RESA | BadNets | 78.98 | 76.68 | 71.08 | 58.68 | 69.14 | 66.30 | 96.81 | 96.63 | 64.78 | 73.46 | 69.12 |
| RESA | Blended | 78.98 | 75.87 | 73.85 | 59.64 | 57.76 | 63.75 | 96.81 | 96.65 | 68.11 | 57.28 | 62.70 |
| RESA | LD-Attack | 78.98 | 71.35 | 73.26 | 64.77 | 69.18 | 69.07 | 96.81 | 96.67 | 92.53 | 92.04 | 92.29 |
| RESA | BadLANE | 78.98 | 72.81 | 75.39 | 62.94 | 68.23 | 68.85 | 96.81 | 96.42 | 92.47 | 91.19 | 91.83 |
| RESA | DBALD | 78.98 | 77.39 | 78.15 | 65.15 | 76.86 | 73.39 | 96.81 | 96.67 | 96.57 | 94.28 | 95.43 |

## Verified local clean baselines

These checkpoints are not established as identical to the paper checkpoints. CULane subset F1 must not replace Table 1 ACC.

| Dataset | Model | Backbone | Images | Clean result % | Metric |
|---|---|---|---:|---:|---|
| CULane | ADNet | resnet18 | 16 | 60.759494 | official CULane IoU/F1; NOT Table 1 ACC |
| CULane | ADNet | resnet34 | 16 | 67.500000 | official CULane IoU/F1; NOT Table 1 ACC |
| CULane | RESA | resnet34 | 16 | 53.488372 | official CULane IoU/F1; NOT Table 1 ACC |
| CULane | SCNN | resnet50 | 16 | 56.179775 | official CULane IoU/F1; NOT Table 1 ACC |
| CULane | LaneATT | resnet34 | 16 | 61.538462 | official CULane IoU/F1; NOT Table 1 ACC |
| TuSimple | ADNet | resnet18 | 2782 | 96.234349 | official TuSimple ACC |
| TuSimple | RESA | resnet18 | 2782 | 96.695915 | official TuSimple ACC |
| TuSimple | SCNN | resnet18 | 2782 | 96.084256 | official TuSimple ACC |
| TuSimple | ADNet | resnet34 | 2782 | 96.598242 | official TuSimple ACC |
| TuSimple | RESA | resnet34 | 2782 | 96.818942 | official TuSimple ACC |
| TuSimple | LaneATT | resnet34 | 2782 | 95.634778 | official TuSimple ACC |
| TuSimple | LaneATT | resnet18 | 2782 | 95.571018 | official TuSimple ACC |

## Fresh GPU execution

LaneATT ResNet-34, full TuSimple test split: 2782 images, official ACC **95.634778%**, runtime-gate failures 0. Independently re-scored with the official submission entry point.

Evidence: `laneatt_full_cuda_verified/` and `fresh_clean_verification.json`.

## Reproduction status

All local poisoned-model clean ACC and attack ASR cells remain unmeasured in this audit. Full CULane evaluation, exact paper configurations/checkpoints, and SCNN cross-version parity remain unresolved. Phase 1 is incomplete.

Code: `scripts/render_table1.py` renders this comparison; `scripts/audit_table1.py` validates prior clean evidence and generates the 220-cell status CSV. No poisoning or training is performed by either script.

# Experiment summary

Generated from completed artifacts at 2026-10-19T07:35:54.184869+00:00. No incomplete run is counted as completed.

| Dataset / scope | Model / backbone | Images | Metric | Value (%) | Median ms | Peak MiB |
|---|---|---:|---|---:|---:|---:|
| CULane / subset | ADNet / resnet18 | 16 | lane F1 | 60.7595 | 20.02 | 107.82 |
| CULane / subset | ADNet / resnet34 | 16 | lane F1 | 67.5000 | 26.39 | 129.67 |
| CULane / subset | RESA / resnet34 | 16 | lane F1 | 53.4884 | 24.29 | 102.52 |
| CULane / subset | SCNN / resnet50 | 16 | lane F1 | 56.1798 | 88.17 | 123.00 |
| CULane / subset | LaneATT / resnet34 | 16 | lane F1 | 61.5385 | 27.70 | 159.28 |
| TuSimple / full_test | ADNet / resnet18 | 2782 | ACC | 96.2343 | 19.07 | 113.09 |
| TuSimple / full_test | RESA / resnet18 | 2782 | ACC | 96.6959 | 46.74 | 183.04 |
| TuSimple / full_test | SCNN / resnet18 | 2782 | ACC | 96.0843 | 79.63 | 175.34 |
| TuSimple / full_test | ADNet / resnet34 | 2782 | ACC | 96.5982 | 24.97 | 152.57 |
| TuSimple / full_test | RESA / resnet34 | 2782 | ACC | 96.8189 | 59.06 | 222.30 |
| TuSimple / full_test | LaneATT / resnet34 | 2782 | ACC | 95.6348 | 28.34 | 183.52 |
| TuSimple / full_test | LaneATT / resnet18 | 2782 | ACC | 95.5710 | 17.57 | 142.98 |
| TuSimple / subset | ADNet / see config | 16 | ACC | 95.6938 | 19.74 | 113.09 |
| TuSimple / subset | RESA / see config | 16 | ACC | 96.4379 | 45.81 | 183.04 |
| TuSimple / subset | SCNN / see config | 16 | ACC | 96.8750 | 84.96 | 175.34 |
| TuSimple / subset | LaneATT / resnet18 | 16 | ACC | 96.7262 | 21.08 | 142.98 |

The 16-image experiments are subsets. Only rows marked full_test cover the complete split. CULane lane F1 is not Table 1 ACC. All ASR fields remain null. These public checkpoints do not establish exact Table 1 equivalence.

Earlier LaneATT runs affected by modern PyTorch anchor tie ordering are preserved, marked superseded_anchor_order in measured_results.csv, and excluded from this table and current plots (11 archived measurement rows). See laneatt_compatibility.md for the source-based correction and its validation.

Tests: 23 tests, 0 failures, 0 errors, 0 skipped. Two-image tensor-batch validation passed for 4 of 4 models. Independently audited TuSimple model/condition submissions: 44.

Ordinary diffusion demonstration completed; see results/overnight/diffusion/measurements.json.

Evidence: measured_results.csv, table1_reproducibility.csv, per-run predictions/provenance/hardware logs under overnight/, and plots/. published_table1.csv is a byte-identical copy of reference/table1.csv (SHA256 052cbbc97cf3d5823d465e1fa200b1dc7aac270bf6472dd58c8d5acb679bf683).

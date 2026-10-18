# Reproduction report

Generated from completed artifacts at 2026-10-18T22:27:19.274801+00:00. No incomplete run is counted as completed.

| Dataset / scope | Model / backbone | Images | Metric | Value (%) | Median ms | Peak MiB |
|---|---|---:|---|---:|---:|---:|
| CULane / subset | LaneATT / see config | 16 | lane F1 | 59.7403 | 15.08 | 119.11 |
| TuSimple / subset | ADNet / see config | 16 | ACC | 95.6938 | 19.74 | 113.09 |
| TuSimple / subset | LaneATT / see config | 16 | ACC | 95.6752 | 17.10 | 142.98 |
| TuSimple / subset | RESA / see config | 16 | ACC | 96.4379 | 45.81 | 183.04 |
| TuSimple / subset | SCNN / see config | 16 | ACC | 96.8750 | 84.96 | 175.34 |

The 16-image experiments are subsets. Only rows marked full_test cover the complete split. CULane lane F1 is not Table 1 ACC. All ASR fields remain null. These public checkpoints do not establish exact Table 1 equivalence.

Tests: 23 tests, 0 failures, 0 errors, 0 skipped. Two-image tensor-batch validation passed for 4 of 4 models. Independently audited TuSimple model/condition submissions: 28.

Ordinary diffusion demonstration has not yet produced verified measurements.

Evidence: measured_results.csv, table1_reproducibility.csv, per-run predictions/provenance/hardware logs under overnight/, and plots/. published_table1.csv is a byte-identical copy of reference/table1.csv (SHA256 052cbbc97cf3d5823d465e1fa200b1dc7aac270bf6472dd58c8d5acb679bf683).

## Interpretation

The infrastructure establishes executable public-checkpoint baselines, metric validity checks and provenance. The clean TuSimple metric is a useful benchmark comparison, but exact paper equivalence remains unresolved even when a backbone matches. CULane results use the dataset's official lane F1 and cannot numerically reproduce the paper's point ACC. Poisoned-model clean ACC and all attack-specific ASRs remain unmeasured.

See methodology.md for preprocessing, timing and metric definitions, experimental_limitations.md for gaps, and overnight_execution_log.md plus machine logs for execution evidence. Earlier verified baseline directories remain intact.

## Three next research actions

1. Resolve author checkpoint/training/scoring equivalence, including the SCNN ResNet-34 checkpoint and CULane point-ACC implementation.
2. Complete any split/model whose completion marker is absent; compare portable NMS/DCNv1 and OpenCV rasterization against original environments.
3. Extend the fixed benign robustness suite to the full test split and repeated settings, keeping these experiments separate from ASR.

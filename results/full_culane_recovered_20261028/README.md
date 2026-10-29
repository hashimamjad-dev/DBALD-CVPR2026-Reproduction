# Full CULane clean evaluation evidence

The complete official **test split**, 34,680 images and annotations, passed the
integrity gate. This does not assert completeness of the training split.
Archive CRC validation and extraction records are in `results/phase1_audit`.
An existing truncated JPEG was preserved under the dataset's `recovery_preserved`
directory before extraction of its valid archive copy.

## Completed measurements at this checkpoint

| Detector | Backbone | Images | Official CULane F1 (%) | Candidate point ACC (%) |
|---|---|---:|---:|---:|
| LaneATT | ResNet-34 | 34680 | 76.67404012461316 | 65.03822689875189 |
| ADNet | ResNet-34 | 34680 | 78.9647481537056 | 67.3748250965467 |

F1 uses the public CULane C++ scorer, lane width 30 and IoU threshold 0.5.
The evaluator is built with OpenCV 4.13 rather than the original 2.4.
Candidate point ACC uses the explicitly documented annotation-vertex, linear
interpolation, one-to-one matching protocol. It is **NOT COMPARABLE** to Table 1
until the paper's ambiguous sampling, matching and aggregation choices are
resolved. Neither number is ASR or poisoned-model clean accuracy.

The native scorer produced the above F1 values during each benchmark. A separate
verification process is checking every exported prediction hash and rerunning
the scorer; its completion must be established from
`results/culane_independent_completed/verification.json`, not inferred from this
document. SCNN R50 has completed inference and is scoring; RESA R34 is queued.
Check each model's `completed.json` and `summary.json` for final completion.
Do not interpret this snapshot as four completed runs.

## Execution and reproducibility

The existing full-split runner launched:

```powershell
.venv/Scripts/python.exe scripts/run_culane_full.py --output results/full_culane_recovered_20261028 --execute
```

Exact expanded commands and exit codes are saved in `readiness.json`; native
scorer commands, output and integer TP/FP/FN counts are in each `summary.json`.
Checkpoint/configuration/source hashes are in `run_config.json` and
`model_provenance.json`. Large predictions and lane exports remain local.

The following candidate scoring commands ran for LaneATT and ADNet and were
started for SCNN (inspect its output before claiming completion):

```powershell
.venv/Scripts/python.exe scripts/score_culane_points.py --predictions results/full_culane_recovered_20261028/laneatt/lanes --output results/full_culane_recovered_20261028/laneatt/point_acc_candidate.json
.venv/Scripts/python.exe scripts/score_culane_points.py --predictions results/full_culane_recovered_20261028/adnet/lanes --output results/full_culane_recovered_20261028/adnet/point_acc_candidate.json
.venv/Scripts/python.exe scripts/score_culane_points.py --predictions results/full_culane_recovered_20261028/scnn/lanes --output results/full_culane_recovered_20261028/scnn/point_acc_candidate.json
```

The independent scorer process calls `evaluate_table1.verify` for LaneATT and
ADNet, writing its completed records incrementally. For a future fresh replay
after all four runs finish, the existing evaluator accepts the supplied manifest:

```powershell
.venv/Scripts/python.exe scripts/evaluate_table1.py --manifest configs/culane_full_clean_evidence.json --output results/culane_full_independent_replay
.venv/Scripts/python.exe scripts/render_table1.py --output results/table1_full_clean_replay --additional-evidence results/culane_full_independent_replay/verification.json
```

These last two commands are documented replay commands, **not claimed as executed**
in this checkpoint. Choose unused output directories. The renderer reuses the
seven verified TuSimple records without scoring them again.

Regression command actually executed: 18 passed.

```powershell
.venv/Scripts/python.exe -m pytest tests/test_table1_evaluator.py tests/test_table1_cached_render.py tests/test_culane_resume.py tests/test_scnn_r34_configs.py -q --basetemp results/culane_full_results_regression_tmp --junitxml results/culane_full_results_regression.xml
```

Phase 1 complete: **NO**. Remaining scientific gaps include unconfirmed CULane
point-ACC convention, a trained SCNN R34 baseline and exact training/configuration
alignment, and all locally unmeasured poisoned-model ACC and attack ASR entries.
Private author checkpoints are not a prerequisite for an independent baseline;
independent results and deviations must be recorded honestly.

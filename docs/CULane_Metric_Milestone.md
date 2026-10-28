# CULane metric and parity milestone

Remote development uses `hashim-dev`, created from merged main `b9dbc1dce89036baa8337879888deeaefc1bd4d9`. Local milestone: `milestone/culane-metric-parity`. Git Credential Manager is configured but the noninteractive fetch again failed with unavailable credentials. No credentials were inspected or disclosed. Local history remains separate; connected GitHub publication is based on the merged remote main. No new remote milestone branch or PR is created.

## Implemented, not claimed reproduced

- `src/lane_repro/culane_point_acc.py` and `scripts/score_culane_points.py`: executable pooled correct-point ACC with explicit interpolation/matching/threshold rules, hashed input exports and strict coverage. The paper does not uniquely specify all these choices. The protocol is therefore **NOT COMPARABLE**, not an exact DBALD metric. Existing official CULane F1 stays unchanged.
- `scripts/evaluate_table1.py`: extends independent clean-input rescoring to externally supplied poisoned-model ACC records, requiring method-specific experiment provenance, matching checkpoint hash and clean input designation. This audits predictions, not the truth of the training claim. Missing records remain unmeasured, and ASR importers remain unsupported. No poisoning or training executes.
- `scripts/probe_scnn_aggregator.py`: identical saved-tensor diagnostic under both runtimes, backend flags and layouts. Does not change models/checkpoints.
- `scripts/fetch_culane_ranges.py`: optional curl transport with 90-second deadline and strict Content-Range/byte-count checks. Failed bodies are retained separately and never appended. Existing requests transport is preserved.
- `AGENTS.md`: persists the permanent local/remote workflow and frozen-work constraints.

## Actual commands and results

Final regression result: **43 passed, zero failed/errors/skips**, recorded in `results/culane_milestone_final_tests.xml` (14.941 seconds).

```powershell
.venv/Scripts/python.exe scripts/score_culane_points.py --predictions results/overnight/culane_smoke_torch16_order/laneatt/lanes --list datasets/culane_smoke16.txt --expected-count 16 --output results/culane_point_protocol_diagnostic.json
.venv/Scripts/python.exe scripts/probe_scnn_aggregator.py --output results/scnn_layout_modern_20261028.json
tools/reference_envs/lanedet/python.exe scripts/probe_scnn_aggregator.py --output results/scnn_layout_legacy_20261028.json
tools/reference_envs/lanedet/python.exe scripts/diagnose_scnn_runtime.py --name milestone_legacy_cpu --device cpu
.venv/Scripts/python.exe scripts/diagnose_scnn_runtime.py --name milestone_modern_cpu --device cpu
.venv/Scripts/python.exe scripts/evaluate_table1.py --manifest configs/table1_evidence.json --output results/table1_culane_milestone
.venv/Scripts/python.exe -m pytest tests/test_core.py tests/test_overnight.py tests/test_phase1_tools.py tests/test_table1_evaluator.py tests/test_culane_points.py tests/test_culane_recovery.py -q --junitxml=results/culane_milestone_final_tests.xml --basetemp=results/culane_milestone_final_tmp
```

The candidate metric diagnostic counted 637/1183 annotated points (53.846154%) on the existing 16-image LaneATT export. This tests the implementation on real geometry; it is neither a full result nor a reproduced Table 1 cell. No new full-model accuracy was measured. The existing seven full TuSimple records were independently re-scored again by the extended importer: seven verified, zero rejected, 220 comparison cells exported.

Fresh SCNN CPU traces: legacy vs modern backbone max absolute difference 0.00000238419; aggregator 19.6850204; logits 20.4808369. The old legacy trace itself differed from a fresh legacy capture by 3.33941817 at the aggregator despite identical input/backbone. Same-input probes also failed with MKLDNN off and channels-last layout. Consequently the cause remains unresolved and the old trace cannot be treated as a fully reproducible reference. Trace arrays remain local; compact measured summaries are committed. These are one-image diagnostics, not full-test parity.

## Data recovery blocker

The existing requests downloader was resumed, then stopped after no contiguous progress. A separate bounded curl request retrieved 1,048,576 bytes successfully. Four larger requests subsequently timed out at 90 seconds with only 208,260 to 992,208 bytes received out of 8,388,608. The strict validator did not append partial bodies. The contiguous archive remains 3,985,072,128/5,099,277,019 bytes; no incomplete archive was extracted. Previously checked usable test pairs remain 10,332/34,680, so no full detector run is claimed. Recovery transport is improved and tested, but dataset completion remains blocked by transfer throughput/reliability; driver_193 is also outstanding.

## Remaining original Table 1 requirements

Phase 1 scientifically complete: **NO**. Exact CULane ACC needs author-confirmed sampling, lane matching, boundary and aggregation conventions. Four full CULane baselines are missing. Clean checkpoints remain traceable upstream models, not established as the exact DBALD models. Forty poisoned-model ACC, 100 scenario ASR and 40 derived averages remain unmeasured. The new candidate metric cannot fill these cells, and the diagnostic subset is never substituted for full evaluation. No paused distillation work was modified.


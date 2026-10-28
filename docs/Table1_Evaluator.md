# Original Table 1 executable evaluation

## Delivered implementation

`scripts/evaluate_table1.py` imports a version-1 JSON manifest of evidence directories. It independently re-scores full clean TuSimple predictions with the existing official scorer; it can verify full CULane exports with the existing official C++ F1 scorer. Published values are loaded separately from the immutable hash-checked reference CSV. All 220 original numerical cells are exported to CSV, Markdown and HTML.

`configs/table1_evidence.json` is the concrete input format. Each record has a unique safe `id`, `model`, `dataset`, `metric`, `evidence` directory and `annotations` file. CULane records additionally require `data_root`. Evidence directories must contain predictions, run configuration, model provenance and clean summary. Checkpoints must remain available locally. The importer checks checkpoint hashes, strict loading, configuration and anchor hashes, annotation hashes, complete unique prediction coverage, finite coordinates/runtimes and recomputed aggregate values. Source hashes and detected drift are retained in the output. These checks establish record consistency, not proof that the original training followed DBALD.

Only independently supported clean metrics have importers. Poisoned ACC/ASR inputs are rejected, not trusted as numeric claims. No equivalent attack record is available. All unsupported cells remain NOT MEASURED. Verified clean baselines remain NOT COMPARABLE to the paper until exact configuration/checkpoint equivalence is established. CULane F1 never fills ACC. The average helper requires all two or three verified underlying scenario values; no local ASR average is currently produced. Published averages remain exactly as printed.

Outputs refuse to overwrite an existing directory. Invalid records produce a nonzero exit and an explicit rejection entry. Successful evidence remains visible, but cannot silently turn a failed batch into success.

## Actual execution

```powershell
.venv/Scripts/python.exe scripts/evaluate_table1.py --manifest configs/table1_evidence.json --output results/table1_release_20261028
.venv/Scripts/python.exe -m pytest tests/test_core.py tests/test_overnight.py tests/test_phase1_tools.py tests/test_table1_evaluator.py -q --junitxml=results/table1_tests_final_20261028.xml --basetemp=results/table1_test_final_tmp_20261028
.venv/Scripts/python.exe -m pytest tests/test_table1_evaluator.py tests/test_phase1_tools.py -q --junitxml=results/table1_targeted_final_20261028.xml --basetemp=results/table1_targeted_tmp_20261028
.venv/Scripts/python.exe scripts/run_culane_full.py --output results/culane_full_gate_20261028 --execute
```

Use a new output directory when repeating these commands. The initial evaluator/test attempt exposed BadLANE/BadLane capitalization in the existing reference; canonical validation now accepts that alias without modifying reference values. The corrected suite passed 32 tests; after adding source-drift reporting and stricter average completeness, all nine targeted tests passed. Historical failing logs are retained separately.

## Full CULane execution

`scripts/run_culane_full.py` reuses `scripts/benchmark_culane.py` for all four architectures. It decodes and validates the complete 34,680-entry test split before starting inference; saved checkpoint/configuration paths come from the existing verified smoke-run provenance. It never substitutes those smoke runs for a full result. Its actual invocation found only 10,332 usable pairs and started zero model processes. Exact prepared commands and the blocked state are in `results/culane_full_gate_20261028/readiness.json`.

The existing resumable downloader was run again on driver_100. Its contiguous archive grew from 3,825,688,576 to 3,985,072,128 bytes of 5,099,277,019, then stopped progressing and was interrupted. Ranges and partial archive remain local. No incomplete archive was extracted. Driver_193 and incomplete existing pairs remain outstanding. The small-JPEG readiness check now handles files shorter than two bytes without a negative seek.

## Metric and SCNN findings

The saved paper text `docs/sources/dbald_arxiv_v2.html`, section 5.1, describes point-count ACC within 20 pixels. This is insufficient to uniquely resolve CULane sampling, matching, aggregation and its relationship to the official TuSimple scorer's angle adjustment and runtime gate. We retain the verified official conventions without claiming an exact DBALD CULane implementation.

`results/scnn_parity_status_20261028.json` summarizes the existing 16-case comparison: every case failed the 0.2-pixel parity criterion, with identical input tensors and different validity patterns. This documents where divergence occurs; its cause remains unresolved. No new model accuracy is claimed from that diagnostic.

## Repository and scientific status

PR #2 is merged: live GitHub main was verified at `0cf5b9c21fc283065f3dd3c9af744b5eea494c2c`. Local branches and untracked evidence remain preserved. A noninteractive `git fetch origin` retry failed because Git Credential Manager has no usable credential. No secrets were read or printed. Local work continues on `codex/table1-evaluator-20261028`; publication uses the GitHub connection atop the merged main. This is not a synchronized local checkout, and local/remote commits have different parents.

Phase 1 scientifically complete: **NO**. Missing: four CULane Table 1 clean baselines, 40 poisoned-model ACC cells, 100 scenario-ASR cells and 40 derived averages; exact paper lineage remains unresolved for existing clean baselines. Completing the dataset alone will enable official F1 measurements, not settle the paper's ACC definition. This implementation is runnable reporting and clean-verification code, not a claim that all original experiment importers or measurements are available.

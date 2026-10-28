# Phase 1: DBALD Table 1 audit

## Preservation and publication update: 28 October 2026

Local history at `40feaec` is preserved by `codex/preserve-local-20261028`. Local audit changes are committed on `codex/table1-audit-20261028`. The uncommitted audit output directory and audit source files were also archived locally as `tmp/phase1-preservation/audit-before-publication-20261028.zip`, SHA256 `E64736ED8B99F5BC2D194D62369C6882E9896FBFC4AF5FADB0CB828A15563A72`.

Command-line `git -c credential.interactive=false fetch origin` failed: `Cannot prompt because user interactivity has been disabled; unable to get password from user`. The GitHub connector independently verified private remote main `431b20704f5e051a2b07669420e7af55422d3dbe` and created a remote audit branch from it. Selected additive audit files are published through that connection, preserving Aoun's merged work and avoiding confidential original local ancestry. Local and remote audit commits consequently have different parents and SHAs. This does not complete a local Git fetch/synchronization; authenticated Git access remains necessary for that step. Neither main was merged or rewritten.

The Git blob hashes of `scripts/benchmark.py`, `src/lane_repro/evaluation.py`, `src/lane_repro/models.py`, `src/lane_repro/culane_official.py`, `tests/test_core.py` and `tests/test_overnight.py` exactly matched remote main. The regression suite passed again: 23 tests, zero failures/errors/skips (`tests_20261028.xml`). Three new extraction-preservation tests also passed (`extraction_tests.xml`). No operational attack tests were executed.

Additional actual commands:

```powershell
git -c credential.interactive=false fetch origin
git branch codex/preserve-local-20261028 40feaec
git switch -c codex/table1-audit-20261028
.venv/Scripts/python.exe -m pytest tests/test_core.py tests/test_overnight.py -q --junitxml=results/phase1_audit/tests_20261028.xml --basetemp=results/phase1_audit/pytest_20261028
.venv/Scripts/python.exe -m pytest tests/test_phase1_tools.py -q --junitxml=results/phase1_audit/extraction_tests.xml --basetemp=results/phase1_audit/extraction_test_tmp
.venv/Scripts/python.exe scripts/audit_culane_readiness.py
.venv/Scripts/python.exe -c "import sys; from pathlib import Path; sys.path.insert(0,'scripts'); from fetch_culane_ranges import download; download('1LTdUXzUWcnHuEEAiMoG42oAGuJggPQs8',Path('datasets/raw/culane/driver_100_30frame.tar.gz'))"
```

Both missing CULane archive sources returned HTTP 206 to bounded one-byte probes. The existing driver_100 download advanced from 3,590,807,552 to 3,825,688,576 contiguous bytes of 5,099,277,019, then was interrupted after over two minutes without contiguous progress. Bytes remain local for resumption. No incomplete archive was extracted. The remaining driver_193 archive reports 4,507,915,856 bytes. `culane_download_attempt.json` records these observations. `audit_culane_readiness.py` decodes available images and validates labels and JPEG end markers; `culane_readiness.json` is authoritative for the latest readiness count. `extract_culane_missing.py` is a narrowly scoped preservation helper for a future complete archive; it checks gzip CRC and refuses to overwrite conflicting files. Its actual-dataset extraction has not run.

### Definitive missing-cell classification

The final integrity check found **10,332 usable pairs out of 34,680**: driver_100 6,975/17,702, driver_193 0/13,621, and driver_37 3,357/3,357. There are 24,318 missing images, 24,320 missing annotations, and one present JPEG without its end marker. These overlapping issue counts must not be added together. Existing evidence and incomplete files were preserved.

- Eight distinct benign model/dataset baselines underlie the table's repeated B-ACC cells. Four TuSimple architectures have independently verified full-test clean baselines, but none is established as the paper's exact configuration/checkpoint. Four CULane Table 1 ACC baselines remain missing; saved subset F1 cannot fill them.
- Across five methods, four architectures and two datasets: all 40 poisoned-model clean-ACC cells and all 100 attack-specific ASR cells remain unmeasured in this audit. All 40 derived average-ASR cells consequently remain missing. Published values are retained only as reference.
- Data blocker: incomplete CULane test set and some incomplete image/annotation pairs. Compute environment: CUDA fixed and clean inference verified; full training cost has not been measured here. Methodology blockers: author configurations/checkpoint lineage, CULane point-ACC convention, attack aggregation conventions, and SCNN parity remain unresolved. Execution restriction: this session cannot create/train/optimize targeted lane backdoors. Those experiments must be addressed separately under the supervisor's authorized process; externally supplied records still require independent audit.
- Shortest path: establish exact author protocol and evaluation definitions; finish and validate missing CULane data; resolve SCNN parity; execute missing clean evaluations; obtain auditable, equivalent records for each method/scenario from the separately supervised research process; check all saved predictions against the agreed metrics and fill only supported cells. A placeholder-trigger sanity run cannot establish DBALD mud-trigger equivalence.

**Phase 1 scientifically complete: NO.** The audit establishes scoring reproducibility and clean evidence, not full Table 1 experimental reproduction.

## Completed clean run: 28 October 2026

Resumed the saved 2,111 predictions and completed the remaining 671 using the unchanged benchmark configuration. The full 2,782-image LaneATT ResNet-34 run reports official clean ACC 95.63477799459108%, FP 0.035262401150251615 and FN 0.029235561945842318. No image exceeded the 200 ms runtime gate. This agrees with the prior saved full clean run. Timings combine the pre-pause and resumed sessions, so they are not a new controlled latency benchmark.

The resumed command is the full GPU benchmark command recorded below. `scripts/render_table1.py` independently checks every prediction key, sampling grid, finite coordinates and nonnegative finite runtime, then checks all three official aggregate metrics against the run summary. Generated evidence is `results/phase1_audit/fresh_clean_verification.json`; comparison outputs are `Table1_Comparison.md` and `Table1_Comparison.html` in the same directory. No attack measurements were added.

## Objective and status

Reproduce page 7 Table 1 using independently verified measurements for LaneATT, ADNet, SCNN and RESA on TuSimple and CULane. Phase 1 is incomplete. Published reference values must never be relabelled as local results. No poisoned-data training or targeted attack execution was performed in this audit.

The generated comparison is `results/phase1_audit/Table1_Comparison.html` (view/print) and `Table1_Comparison.md`; the per-cell inventory is `results/phase1_audit/table1_status.csv`. All 20 published method rows and 220 expanded numerical cells are represented. TuSimple has no LRA column in the paper.

## Evidence and execution

- `scripts/audit_table1.py`: re-scored seven full TuSimple runs (2,782 unique images each), checked labels and checkpoint hashes, and re-scored five 16-image CULane runs using existing official evaluators. All matched saved metrics within 1e-9 percentage points for TuSimple, and exact TP/FP/FN for CULane.
- `results/phase1_audit/clean_rescore.json`: per-run numerical values, hashes and original evidence paths.
- `results/phase1_audit/tests_retry.xml`: 23 passed, zero failed/errors/skips. The first attempt (`tests.xml`) had 19 passed and four temporary-directory setup errors; the retry required no model-code edits.
- `scripts/render_table1.py`: recreates the published table and appends verified clean evidence. If the fresh full GPU run has completed, independently re-scores its predictions and writes `fresh_clean_verification.json`.
- `results/phase1_audit/laneatt_full_cuda_verified/`: fresh full-TuSimple GPU inference, with per-image predictions written incrementally, run configuration, model provenance, hardware samples, summary and completion marker. This evaluates a pretrained clean checkpoint; it does not train.

Commands actually issued (project root):

```powershell
.venv/Scripts/python.exe -m pytest tests/test_core.py tests/test_overnight.py -q --junitxml=results/phase1_audit/tests.xml --basetemp=results/phase1_audit/pytest_tmp
.venv/Scripts/python.exe -m pytest tests/test_core.py tests/test_overnight.py -q --junitxml=results/phase1_audit/tests_retry.xml --basetemp=results/phase1_audit/pytest_retry
.venv/Scripts/python.exe scripts/audit_table1.py
.venv/Scripts/python.exe scripts/benchmark.py --model LaneATT --backbone resnet34 --checkpoint models/checkpoints/laneatt_official/experiments/laneatt_r34_tusimple/models/model_0100.pt --device cuda --root datasets/raw/tusimple_test --labels datasets/raw/tusimple_test/labels.jsonl --expected-count 2782 --output results/phase1_audit/laneatt_full_cuda_verified
.venv/Scripts/python.exe scripts/render_table1.py
```

Earlier diagnostic inference used the same checkpoint on `datasets/raw/tusimple_subset`, `--expected-count 16`, outputs `laneatt_clean_smoke` (CUDA initialization failed before inference with CPU-only torch) and `laneatt_clean_cpu` (completed on CPU). CPU official ACC was 0%, ungated ACC 95.303199%, with all 16 samples exceeding 200 ms. This is a diagnostic subset, not the full benchmark. The environment was subsequently repaired at the user's request to torch 2.7.1+cu128 and torchvision 0.22.1+cu128; GPU allocation and computation passed.

Read-only Git checks included `git status --short`, `git log -5 --oneline`, `git branch -av`, `git remote -v`, `git diff --stat` and `git diff --cached --stat`. The GitHub connector independently returned remote main `431b20704f5e051a2b07669420e7af55422d3dbe`; local main was `40feaec`, and local origin/main was stale at `563341f`. No merge, reset or push was performed. Aoun's remote source/results review is preserved in `results/phase1_audit/remote_review.json` with source URLs and blob IDs.

## Exact sources and interpretation

- Published reference: `results/reference/table1.csv` and `manifest.json`, visually checked against page 7 of the user-supplied Liao DBALD PDF in the user's Documents directory. A rendered page is saved as `results/phase1_audit/paper_table1.png`.
- Official TuSimple scorer: `third_party/tusimple-benchmark/evaluate/lane.py`, repository commit `d1f5ef1b6fff78c92c61864abcc07a9714900360`; loaded by `src/lane_repro/evaluation.py` with obsolete Python 2 CLI omitted. Its 200 ms cutoff remains enabled for official ACC.
- Clean inference: `scripts/benchmark.py`, `src/lane_repro/models.py`, `src/lane_repro/nms.py`, `configs/laneatt_anchor_order.json`; checkpoint `models/checkpoints/laneatt_official/experiments/laneatt_r34_tusimple/models/model_0100.pt`.
- CULane scoring: `src/lane_repro/culane_official.py`, `tools/culane_evaluator/evaluate.exe`, and `datasets/culane_smoke16.txt`. This is lane IoU/F1, not Table 1 point ACC.
- Prior run selection: current clean records in `results/measured_results.csv`; exact evidence directories and checkpoint hashes are recorded in `clean_rescore.json`. Superseded LaneATT anchor-order results are excluded.
- Dependency references: `requirements-laneatt-legacy.txt`, `requirements-lanedet-reference.txt`, `requirements-adnet-reference.txt`. Modern CUDA adapters do not establish exact original-stack parity. Saved ADNet/RESA runtime comparisons passed their subset tolerance; SCNN comparison failed and remains unresolved (`results/overnight/reference_versions/scnn/`).

## Remaining work and blockers

1. Reconcile local and remote source states while preserving existing local history and untracked work. Remote additions are not automatically present in this checkout.
2. Complete CULane data access and integrity verification. Audit found 10,362 image files and 10,360 annotation files for the 34,680-entry test list; this checks existence only. Full CULane Table 1 ACC remains unavailable, and F1 cannot substitute for it.
3. Establish exact paper configurations, checkpoints, scoring conventions and training provenance. The supplied main-paper experimental setup does not establish the claimed 100-epoch/ResNet-34 recipe. Upstream checkpoint names alone are not proof of DBALD settings.
4. Resolve SCNN cross-version numerical differences before claiming implementation equivalence.
5. Independently verify any externally supplied experiment records before filling poisoned-model ACC or ASR. No equivalent full attack measurement is verified here. The remote small training-check report references an uncommitted result file, which was unavailable to this audit.

Execution restriction: this session does not generate targeted poisoning, train a lane-manipulation backdoor or optimize its effectiveness. These entries remain unmeasured here. A placeholder trigger is not equivalent to the published naturalistic mud-trigger experiment. LOA means lane offset; LDA means lane disappearance.

The deliverable recreates the table structure and reference values and supplies traceable clean evidence. It does not claim full experimental reproduction or paper-equivalent ASR.

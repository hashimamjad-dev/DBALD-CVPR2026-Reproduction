# Table 1 clean-evidence reproduction, 6 November 2026

PR #7 was verified merged at main `79dcaced38cc4f82a4af579a7bb4b3cfd231c26d`. This work starts from that commit on `milestone/table1-clean-verification-20261106`. The previous milestone and all experimental evidence remain preserved. No attack pipeline, trigger optimization, training or poisoned-model experiment is implemented or executed in this milestone. No paused V1/V2/V3 work is changed.

## Executable changes

- `configs/table1_clean_replay.json` binds the existing seven TuSimple clean runs and the two reviewed verification receipts covering all four CULane runs. `scripts/replay_clean_table1.py` uses the existing evaluator to rescore raw TuSimple predictions and hash-verify the completed CULane scoring receipts, checkpoints, configurations, provenance, complete split coverage and saved metrics. The optional `configs/table1_clean_full_replay.json` retains the original eleven-run raw rescoring specification. It emits a machine-readable eight-pair completion matrix. This is rescoring, not repeated full inference.
- `configs/table1_unified.json` now also pins the completed SCNN/RESA CULane verification receipt. Its prior omission from the manifest did not mean those experiments were missing. All four CULane F1 results and all four explicitly unconfirmed point-score receipts now appear as supporting measurements.
- `scripts/render_table1.py` preserves the original page-7 column layout and separate published/independent tables. The new `table1_comparison.csv` contains all 220 cells, diagnostic differences in percentage points, evidence status, artifact paths and scientific qualifications. A missing/non-comparable cell has no numeric difference. CULane F1 is never inserted into an ACC cell.
- `scripts/validate_clean_inference.py` strictly loads the existing compatible checkpoint for each model/dataset pair, performs one clean inference with existing adapters, and compares it with the saved prediction. It prefers R34 where available, retaining SCNN R18/R50 identities. It supports checking one dataset independently and writes incremental numerical receipts. It does not train, edit images or claim subset accuracy as full-test accuracy.
- `scripts/validate_culane_data.py --output` now writes new full-dataset integrity receipts instead of overwriting historical reports. It bounds annotation batches and pending work, supports retrying only failed entries, and reconciles the original and retry receipts without deleting error history. Worker count is bounded. Original default behavior remains available.
- `AGENTS.md` records the user's requirement to ask before extensive GPU tests or experiments. No extensive GPU run was launched; the eight one-image checks completed before that instruction. Subsequent work uses CPU only.

## Clean measurements and completion matrix

The following are existing full-test measurements, checked against original artifacts; they are not newly trained models. Machine-readable results and validation outcomes are under `results/table1_clean_20261106/`.

| Model | TuSimple checkpoint / official ACC % | CULane checkpoint / official lane F1 % | Paper equivalence |
|---|---|---|---|
| LaneATT | R34 / 95.63477799459108 | R34 / 76.67404012461316 | Independent checkpoints/training; CULane F1 differs from paper ACC |
| ADNet | R34 / 96.59824210057856 | R34 / 78.9647481537056 | Independent checkpoints/training; CULane F1 differs from paper ACC |
| SCNN | R18 / 96.08425550306391 | R50 / 74.7785144614202 | Backbone mismatch on both datasets; no trained R34 checkpoint |
| RESA | R34 / 96.8189423504844 | R34 / 75.84875083091943 | Independent checkpoints/training; CULane F1 differs from paper ACC |

All eight pairs have full clean predictions: 2,782 images per TuSimple run and 34,680 per CULane run. R18 ADNet, RESA and LaneATT TuSimple measurements remain separately preserved as additional engineering baselines. Do not average backbones or replace the selected R34 run with a numerically higher R18 run.

Evidence status and paper comparability are separate. A clean score can be VERIFIED while its experimental protocol remains NOT COMPARABLE to an exact DBALD reproduction. The page-7 comparison marks 15 repeated cells as independent R34 TuSimple baselines, 25 as NOT COMPARABLE, and 180 as MISSING/NOT MEASURED. The 15 cells represent three clean baselines repeated across five methods, not 15 different clean experiments.

No full Table 1 poisoned-model receipt was found. The small LaneATT run is documented in `docs/poisoning_pipeline_status.md`, but the referenced `results/sanity_poisoned_training.json` is absent. Classify that report as PARTIAL, not an independently verified ACC/ASR result. All 40 poisoned-model clean-ACC cells, 100 scenario-ASR cells and 40 derived averages remain unmeasured in this comparison. Published values remain references only. No attack average is generated from an incomplete scenario set.

## Architecture and methodology checks

LaneATT's corrected PyTorch-1.6 anchor order remains enabled. Its saved 16-image legacy comparison reports matching anchor permutations and a maximum pre-NMS proposal difference of 0.0003662109375. This supports the engineering correction, not exact author training or original CUDA NMS equivalence. Superseded anchor-order runs are excluded.

The earlier SCNN parity report failed, but the later `results/scnn_single_thread_parity/summary.json` passes all 16 images with zero coordinate difference between the one-thread legacy reference and modern CUDA. Keep both records; the latter supersedes the earlier diagnostic within its stated scope. The existing R34 architecture configurations pass their native-resolution forward tests, but random-weight architecture tests are not trained R34 evidence. Strict loading of available R18/R50 weights must not be presented as R34 compatibility.

ADNet and RESA use their preserved upstream configurations and strict checkpoint loading. Modern compatibility layers and public pretrained weights do not establish equality to the exact DBALD training recipe. The new smoke receipts check current execution against saved predictions without tuning confidence thresholds or metrics to match published values.

The attached CVPR paper's page-7 table was visually checked again. Section 5.1 describes point correctness within 20 pixels, while the expanded author version's Appendix C specifies ResNet-34. Public CULane evaluation uses lane IoU/F1, not that point-ACC definition. The existing CULane candidate makes explicit choices about annotation-vertex sampling, interpolation, matching, pooled aggregation and runtime handling. The paper does not uniquely establish those choices, so the candidate remains NOT COMPARABLE. TuSimple uses the actual vendored official scorer, including its runtime gate; exact agreement with the paper's reported aggregation and author checkpoints remains unestablished.

Primary public sources checked on 6 November 2026: [author manuscript](https://arxiv.org/html/2508.15778v2), [author overview](https://sites.google.com/view/dbald/overview). The retrieved overview did not expose an executable checkpoint/evaluation bundle; this does not establish that none exists elsewhere. No protocol ambiguity was resolved by assuming the nearest numerical result is correct.

## Commands and provenance

Run from the project root, with a fresh output path each time:

```powershell
# CPU: rescore full TuSimple predictions and reuse reviewed full CULane receipts.
.venv/Scripts/python.exe scripts/replay_clean_table1.py --output results/clean_raw_replay_NEW

# CPU: hash-verified reuse of reviewed receipts and candidate-score exports.
.venv/Scripts/python.exe scripts/evaluate_table1.py --manifest configs/table1_unified.json --output results/clean_unified_NEW

# CPU: native-resolution decode, image/annotation hashing, complete CULane coverage.
.venv/Scripts/python.exe scripts/validate_culane_data.py --output results/culane_integrity_NEW --workers 2

# CPU: if a full scan has transient failures, retry only those entries serially.
.venv/Scripts/python.exe scripts/validate_culane_data.py --output results/culane_integrity_RETRY_NEW --workers 1 --retry-failures results/culane_integrity_NEW/culane_integrity.jsonl
```

The one-image compatibility script is separate from the replay workflow and is never launched automatically. Ask the user before extensive GPU use. Individual original runs retain seed, backbone, dataset split, checkpoint hash, preprocessing/configuration and source provenance in their run directories. Replay receipts retain current source drift instead of pretending old source hashes still match. `verification.json` records official scorer outputs and checkpoint hashes; comparison exports point back to the supporting artifacts. CULane's native scorer retains its documented OpenCV 4.13-versus-2.4 compatibility difference.

## Validation and remaining limits

Targeted reporting tests: 37 passed. The one-thread clean-model and Table 1 regression suite: 77 passed. Final broader applicable clean-suite results, native rescoring and dataset-integrity outcomes are recorded in the final receipts under the dated results directory.

Final clean replay: 11 verified baseline records, no rejections, eight model/dataset pairs. The updated unified comparison imports 15 supporting records with no rejections. All 220 cell statuses, separate 40-row reference/independent exports and safe numeric differences were independently checked. The original CULane prediction exports, overnight evidence and immutable reference have no Git changes.

The final reporting/data-tool suite passed 68 tests. A subsequent isolated CPU run passed both R34 architecture tests and both selective-retry/reconciliation tests (four tests). Dataset integrity reconciles to 34,680 verified image/annotation pairs, zero remaining failures. The original scan preserved 52 OpenCV memory-allocation failures; all 52 passed a one-worker retry. `dataset_integrity_retry/culane_reconciled_integrity_manifest.json` binds both receipts and their combined records with SHA256 checksums. This is a new integrity receipt, not a new detector-accuracy experiment.

The initial test attempt encountered a missing temporary parent directory; its failing XML is preserved. A subsequent uncapped CPU run hit a native convolution access violation. A GPU-runtime startup attempt failed with Windows paging-file error 1455. One serial one-thread suite passed, but a broader concurrent CPU run again hit a native access violation during R34 architecture testing. The redundant full CULane rescore was interrupted to reduce memory pressure, and its incomplete directory and original manifest are retained with an explicit incomplete status. No model weights, tests or system memory settings were changed to hide this limitation. The first smoke checker compared TuSimple floats to the benchmark's saved integer submissions; correcting the checker to reproduce `benchmark.py` serialization yielded exact agreement. The initial receipt is preserved; only the four affected TuSimple checks were repeated. All eight final model/dataset checks have zero coordinate difference from the corresponding stored prediction.

This milestone does not complete Table 1. Exact author checkpoint/training equivalence, trained SCNN R34 weights, CULane paper-compatible scoring, and verified method-specific poisoned-model evaluation receipts remain unresolved. Operationally useful backdoor implementation/optimization is outside the assistance provided; that limitation does not invalidate or abandon the clean benchmark work. No ACC/ASR was fabricated, inferred from code presence, or copied from the paper into local measurements.

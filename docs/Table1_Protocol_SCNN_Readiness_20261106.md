# Table 1 protocol, SCNN R34 and CPU readiness: 6 November 2026

This milestone continues from accepted commit `1f1c3977863eed3f0b74fa2ae7f1333333f9f214` on `milestone/table1-protocol-scnn-20261106`. Original experiments, checkpoints and references are preserved. No training or GPU computation was started. Paused V1/V2/V3 work is untouched.

## Evidence preservation and reproducibility

The accepted clean replay was executed again: **11 records, eight model/dataset pairs, zero rejections**. TuSimple scores were independently recomputed from the original full predictions; reviewed CULane scoring receipts were hash-verified and reused. The unified replay again verified **15 supporting records with zero rejections**. `scripts/table1_readiness.py` compares accepted/replayed record identities, checkpoint/configuration/provenance/prediction/receipt hashes where applicable, and numerical values within 1e-9 percentage points. The accepted result directories, original full CULane exports, overnight evidence and immutable reference have no Git modifications.

Results: `results/table1_protocol_scnn_20261106/clean_recheck`, `unified_recheck`, `accepted_evidence_preservation.json`, and `readiness_final`.

## CULane scoring: established mismatch, unresolved author implementation

Section 5.1 states a ratio of correctly localized lane points to total points, with a 20-pixel condition. Public CULane evaluation instead calculates lane-level IoU/F1 using thick lane curves. These quantities are mathematically different; F1 must not fill Table 1 ACC.

The new CPU conformance script executes the existing native C++ scorer and existing candidate point scorer on explicitly synthetic fixtures. With a 19-pixel horizontal displacement, the candidate accepts both points while the native lane scorer reports no matched lane: 100% candidate point accuracy versus 0% lane F1. A separate fixture gives 50% with one-to-one matching versus 100% under TuSimple's per-ground-truth best-match rule. These are protocol counterexamples, **not research measurements or Table 1 cells**.

The literal pooled-point formula is consistent with the candidate's pooling, but the paper does not uniquely supply CULane sampling/label conversion, lane matching/duplicates, error geometry or angle adjustment, the exact 20-pixel boundary, and runtime/excess-prediction conventions. The candidate remains unconfirmed. We have not established which rules generated the authors' actual numerical CULane results. Numerical proximity to a published value would not resolve this ambiguity.

Feasible implementation: `scripts/audit_culane_protocol.py`, a native-scorer regression test, and explicit protocol facts/unresolved fields in `configs/table1_author_protocol.json`. No existing scientific scorer, threshold or candidate measurement was modified to force agreement.

Primary sources: [DBALD Section 5.1 and Appendix C](https://arxiv.org/html/2508.15778v2), [official TuSimple evaluator](https://github.com/TuSimple/tusimple-benchmark/blob/master/evaluate/lane.py), [SCNN/CULane native evaluator](https://github.com/XingangPan/SCNN/tree/master/tools/lane_evaluation). The attached CVPR page-7 transcription remains the immutable numerical reference.

## SCNN R34: trained checkpoints found and validated

The original SCNN repository points to PytorchAutoDrive as a PyTorch implementation. Its model zoo publishes trained SCNN R34 checkpoints for both datasets. This resolves public checkpoint availability; it does not establish identity to DBALD's models.

| Dataset | Downloaded checkpoint | SHA256 | Current status |
|---|---|---|---|
| TuSimple | `resnet34_scnn_tusimple_20210424.pt`, 91,106,491 bytes | `d8eccc17019a8a0d96c63897059f8a581eb2f60a60346e710f2e38034c359b6a` | Strict CPU load and scoped runtime parity verified; full-test measurement missing |
| CULane | `resnet34_scnn_culane_20210220.pt`, 90,246,279 bytes | `0dc808f53c84b1164236c5ee18d76add5f50a7137d3a2e786489b980c057f3e6` | Strict CPU load and scoped runtime parity verified; full-test measurement missing |

Source commit: `voldemortX/pytorch-auto-drive@137e63a9e6c3cd2cfeb15e101808478d7b25ddbd`, BSD-3-Clause. [Pinned model zoo](https://github.com/voldemortX/pytorch-auto-drive/blob/137e63a9e6c3cd2cfeb15e101808478d7b25ddbd/docs/MODEL_ZOO.md). Source URLs and hashes are pinned in `configs/scnn_r34_autodrive.json`; download and validation receipts remain separate. Existing R18/R50 checkpoints were neither replaced nor relabelled.

The complete architecture differs from the independent LaneDet R34 configurations. The new adapter imports the actual reviewed upstream backbone, SCNN block, heads and native decoder, and requires a strict full checkpoint load. It does not map a partial parameter subset into LaneDet. Automatic ImageNet downloads are disabled because the full lane checkpoint supplies those parameters.

Preprocessing follows upstream RGB/PIL bilinear resize and ImageNet normalization; native point decoding is retained without spline resampling. Config imports and training initialization are not executed: the adapter reads only the reviewed literal model/test dictionaries. The upstream training code is untouched. Its original mixed-precision evaluation shell is not treated as equivalent to our float32 CPU checks.

Both existing benchmark drivers now accept `--scnn-implementation autodrive --backbone resnet34`. CULane also supports `--device cpu`; both drivers expose bounded CPU thread counts. Default model implementations are retained. Their one-image CPU integration tests save actual predictions, configurations, provenance and score receipts under `driver_evidence/`, labeled subset diagnostics. TuSimple's CPU runtime exceeds the official 200 ms gate; its gated zero is a one-image runtime diagnostic, not full R34 accuracy. Neither diagnostic enters Table 1.

## Runtime parity and Windows memory investigation

Eight fresh-process checks passed: both LaneDet R34 architecture tests and both trained AutoDrive R34 native-reference checks, each with MKLDNN enabled and disabled. Two further fresh-process CPU checks exercised the existing benchmark drivers. Preprocessing and decoded coordinates match the reviewed upstream implementation on the checked real images.

The preserved Python 3.8.10 / PyTorch 1.8.0+cpu environment also loaded both original trained checkpoints. Four bounded CPU processes exported one image per dataset in the legacy and modern runtimes. The comparison criteria were declared before execution: exact preprocessed inputs, logits allclose with atol=rtol=1e-4, and exact decoded coordinates.

| Dataset | Input identity | Maximum segmentation-logit difference | Decoded native coordinates |
|---|---|---:|---|
| TuSimple | Identical | 0.0000152587890625 | Identical |
| CULane | Identical | 0.00000762939453125 | Identical |

Both comparisons passed. This is **one-image CPU runtime parity per dataset**, not full-split parity, GPU parity or author-training equivalence. The earlier R18 16-image single-thread diagnostic remains separately preserved.

`scripts/run_isolated_cpu_checks.py` imports no Torch in the parent, launches one check per process, disables GPU visibility for children, fixes numerical-library threads to one, and checks physical and Windows commit headroom before execution. It samples the complete process tree, including the Windows venv redirector's real interpreter, and terminates only its own child tree on memory-budget or time-limit violations. The default monitored private/RSS budget is 2,048 MiB and timeout is 180 seconds. Sampling is a watchdog, not an OS-enforced allocation limit.

The largest measured process-tree private allocation was approximately **1,178.5 MiB**. All final checks completed without a native fault. A preliminary four-case run measured only the small venv launcher; its receipt is retained with a superseded-accounting annotation. Exact source snapshots accompany the final runner receipts.

The earlier Windows paging-file error and OpenCV allocation failures establish resource exhaustion in that session. The native convolution access violation was not reproduced under the isolated checks with either backend. Its universal root cause is **not proven**; we do not claim a kernel bug is fixed or disable MKLDNN merely to obtain passing results. The feasible fix is safe orchestration, headroom checks, bounded work and truthful failure classification, complementing the prior bounded dataset validator.

## Precise Table 1 readiness

The machine-readable report is `readiness_final/table1_readiness.csv` and `readiness_final/readiness.json`.

| Component | Evidence status | Comparability / prerequisite |
|---|---|---|
| Accepted 11 full clean records | VERIFIED | Exact author training/checkpoint equivalence remains unverified |
| 15 repeated R34 TuSimple clean cells | VERIFIED independent benchmark baselines | Common official benchmark comparison only; not exact experimental reproduction |
| 25 remaining clean cells | VERIFIED supporting measurements, NOT COMPARABLE | CULane F1/point-protocol mismatch or SCNN backbone mismatch in the full-test evidence |
| New trained SCNN R34 checkpoints | VERIFIED strict load and scoped CPU parity | PARTIAL experimental readiness; full 2,782/34,680-image predictions and metrics are missing |
| Documented tiny poisoned-model run | PARTIAL report | Referenced numerical receipt is absent; no verified Table 1 value |
| 40 poisoned-model ACC cells | MISSING | Require independently verified complete experiment receipts |
| 100 scenario-ASR cells | MISSING | Require independently verified complete experiment receipts |
| 40 derived averages | MISSING | Require all applicable verified scenario measurements from the same experiment series |

The table therefore remains 15 independent-baseline cells, 25 non-comparable clean cells and 180 unmeasured attack-related cells. Checkpoint discovery does not promote any full-test score or copy upstream/publication numbers into local measurements.

## Reproduction commands

For a fresh source checkout, clone PytorchAutoDrive under `third_party/pytorch_auto_drive` and select commit `137e63a9e6c3cd2cfeb15e101808478d7b25ddbd`. Existing source is never automatically replaced. Download support uses the already pinned optional dependency `gdown==5.2.0`.

```powershell
# Fetch public weights into a separate directory; existing weights are preserved.
.venv/Scripts/python.exe scripts/fetch_scnn_r34.py --receipt results/scnn_r34_download_NEW.json

# Serial, monitored CPU checks; each output directory must be new.
.venv/Scripts/python.exe scripts/run_isolated_cpu_checks.py --spec configs/cpu_scnn_checks.json --output results/scnn_cpu_NEW
.venv/Scripts/python.exe scripts/run_isolated_cpu_checks.py --spec configs/cpu_scnn_driver_checks.json --output results/scnn_drivers_NEW
.venv/Scripts/python.exe scripts/run_isolated_cpu_checks.py --spec configs/cpu_scnn_cross_version_checks.json --output results/scnn_cross_NEW
.venv/Scripts/python.exe scripts/compare_scnn_r34_cpu.py --root results/scnn_cross_NEW
.venv/Scripts/python.exe scripts/audit_culane_protocol.py --output results/culane_protocol_NEW
```

The next full clean runs can use the existing benchmark commands with `--model SCNN --backbone resnet34 --scnn-implementation autodrive` and the corresponding pinned checkpoint. **Obtain the user's approval before extensive GPU tests or full GPU evaluation.** Neither full R34 run was launched in this milestone.

Final regression: 72 tests passed across 12 isolated processes, in addition to eight architecture/native-reference checks, two CPU driver checks and four cross-version reference processes. A final three-test sanity rerun also passed; its first invocation lacked a temporary parent directory and the error XML is preserved. Source/receipt integrity checks are recorded in `validation_summary.json` and the isolated receipts. Tests were not weakened. No scientific results were invented or marked reproduced. No merge into main is performed by this milestone.

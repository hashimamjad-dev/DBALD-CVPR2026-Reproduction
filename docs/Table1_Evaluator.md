# Original DBALD Table 1: single executable interface

Run from the repository root using the existing environment:

```powershell
.venv/Scripts/python.exe scripts/evaluate_table1.py --manifest configs/table1_unified.json --output results/table1_unified_replay
```

Use a new output directory. This command imports, verifies and renders evidence; it never performs inference, trains models, generates targets or edits images.

Outputs: `table1.csv` contains two source-labeled 20-row tables with the original column order; `table1.md` and `table1.html` separate published and independent tables and list supporting measurements. `table1_status.csv` contains all 220 cell statuses. `verification.json` retains provenance, rejected imports and manifest/reference hashes. Printed reference averages are preserved.

## Version-1 evidence manifest

Top-level fields: `schema_version: 1`, `records`, `verified_caches`, `point_candidates`, `external_records`. Lists may be empty.

- `records`: existing run-directory verifier. See `configs/culane_full_clean_evidence.json`. Checks checkpoint/config/source hashes, unique full coverage and exported predictions; reruns official metrics. Supports benign ACC and poisoned-model clean ACC; the latter requires method-bound experiment provenance and clean inputs.
- `verified_caches`: `{path, sha256, dataset?}` pins a reviewed local verification receipt. Checks full count, summary/checkpoint and saved prediction hashes without rerunning inference or scoring. A hash proves identity, not authenticity. Never use this path to bless arbitrary external numeric claims.
- `point_candidates`: `{path, sha256}` pins a receipt from `score_culane_points.py`; validates protocol, coverage, arithmetic and prediction-export tree digest. Always NOT COMPARABLE to original Table 1 ACC.
- `external_records`: independently recalculates supplied static predictions using the protocols below. No target generation or training.

External record required fields: `id`, `model`, `dataset`, `metric`, `method` (except benign ACC), `experiment_id`, `protocol`, and `artifacts`. Add `protocol_family` to permit derived scenario averages. `artifacts` maps each of `predictions`, `annotations`, `split`, `checkpoint`, `config`, `provenance` to `{path, sha256}`. All paths resolve inside the repository. Hashes must be actual SHA256 values.

`split` is a JSON array of all unique `raw_file` keys (2782 TuSimple or 34680 CULane). Predictions and annotations are JSONL records with `raw_file`, strictly increasing finite `h_samples`, and finite `lanes` arrays matching that grid. TuSimple predictions require measured `run_time` in milliseconds. An optional `claimed_percent` must agree with recalculation. Annotations contain clean labels for ACC/P-ACC, supplied target labels for LOA/LRA, or original lane slots for LDA. The importer does not construct or authenticate attack targets.

Provenance repeats `model`, `dataset`, `metric`, `method`, `experiment_id`, `protocol`; includes `checkpoint_sha256`, `config_sha256` and `evaluation_input` (`clean` or `supplied_scenario`). Fixed-slot LDA requires `lane_slot_semantics: stable_correspondence`. File consistency does not prove training authenticity or original method equivalence.

| Protocol | Dataset | Supported metrics | Qualification |
|---|---|---|---|
| tusimple_official_runtime_v1 | TuSimple | benign_acc, poisoned_acc, loa_asr | Existing official scorer with measured runtime gate; target-label equivalence to DBALD unconfirmed |
| fixed_slot_disappearance_v1 | Both | lda_asr | At least two valid points means present; stable slot correspondence is required; candidate convention |
| culane_annotation_vertices_linear_assignment_v1 | CULane | benign_acc, poisoned_acc, loa_asr, lra_asr | Existing candidate point scorer; supplied grid must represent annotation vertices; protocol unconfirmed |

Unknown protocols and TuSimple LRA are rejected. Imported scenarios remain supporting measurements marked NOT COMPARABLE. Clean evidence cannot fill poisoned cells. Derived averages require all two TuSimple or three CULane scenarios from the same model, dataset, method, declared experiment series and protocol family. Duplicates and invalid percentages are rejected; incomplete sets produce no average.

## Paper specification and scientific limits

Saved source `docs/sources/dbald_arxiv_v2.html`, section 5.1, describes point-count accuracy within 20 pixels; Appendix C specifies ResNet-34. CULane sampling, matching, aggregation, angle adjustment and runtime handling remain insufficiently specified. Public CULane IoU/F1 (lane width 30, IoU 0.5) is a different metric. The Windows scorer uses OpenCV 4.13 rather than 2.4. The candidate uses annotation vertices, linear interpolation without extrapolation, one-to-one matching, strict horizontal error below 20 pixels and pooled counts.

All four full CULane clean benchmarks finished. SCNN uses a public R50 checkpoint, not the paper-required R34. Independent R34 configurations passed forward tests but are not trained weights. Seven verified full TuSimple records are reused. Independent R34 TuSimple baselines are useful without private author weights; they do not prove identical training conditions.

## Regression tests

```powershell
.venv/Scripts/python.exe -m pytest tests/test_table1_evidence_import.py tests/test_table1_evaluator.py tests/test_table1_cached_render.py -q
```

Synthetic fixtures test supported evidence types, tampering, coverage, runtime, provenance, protocol rejection, averages, cache drift and page-7 layout. They are never exported as research measurements.

Phase 1 scientifically complete: NO. Original-equivalent P-ACC and ASR remain unmeasured; CULane point scoring is unconfirmed; a trained SCNN R34 baseline and training/configuration alignment remain unresolved. Frozen diffusion work and attack training are not modified.

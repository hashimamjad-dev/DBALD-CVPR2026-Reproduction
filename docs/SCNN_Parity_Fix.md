# SCNN reference-runtime discrepancy isolated

The local PyTorch 1.8.0 CPU environment with four threads gives an incorrect horizontal convolution for the examined SCNN tensor: maximum difference from an independent double-precision NumPy convolution is 4.54748804. With one thread the error falls to 0.00000173064; modern PyTorch 2.7.1 gives 0.00000182991. Checkpoint weight hashes and input bytes match. Down/up propagation matched exactly across the four-thread runtimes; the divergence began in horizontal propagation. Packing or transposing the input did not resolve it. This diagnoses this local runtime/configuration; it is not a claim about every installation of PyTorch 1.8.

The existing comparison tool now defaults to one thread for legacy SCNN and accepts explicit output directories. Old evidence is preserved. A fresh 16-image legacy CPU reference was compared against modern CUDA: **16/16 passed, maximum sampled-coordinate difference 0 pixels, identical preprocessed inputs**. This resolves the prior diagnostic failure; exact DBALD checkpoint lineage and full paper equivalence remain unestablished.

Actual commands:

```powershell
tools/reference_envs/lanedet/python.exe scripts/probe_scnn_aggregator.py --threads 1 --output results/scnn_oracle_legacy_single.json
tools/reference_envs/lanedet/python.exe scripts/compare_reference_models.py --model SCNN --device cpu --threads 1 --output results/scnn_single_thread_parity
.venv/Scripts/python.exe scripts/compare_reference_models.py --model SCNN --device cuda --threads 4 --compare --output results/scnn_single_thread_parity
.venv/Scripts/python.exe -m pytest tests/test_culane_recovery.py tests/test_core.py tests/test_overnight.py -q --junitxml=results/culane_recovery_small_range_tests.xml --basetemp=results/culane_recovery_small_range_tmp
.venv/Scripts/python.exe -m pytest tests/test_culane_recovery.py -q --junitxml=results/culane_recovery_concurrency_tests.xml --basetemp=results/culane_recovery_concurrency_tmp
```

Tests: 28 passed; final recovery-specific rerun 5 passed. The seven full TuSimple accuracy measurements were not repeated.

CULane recovery uses the existing `download` function with `transport='curl'`, `chunk_size=262144`, and known official archive totals checked against every accepted Content-Range. Driver_100 uses four workers, driver_193 sixteen. Both partial archives remain resumable. Complete archive CRC validation and dataset integrity checks are still required before full evaluation.

Official source checks on 28 October 2026: https://xingangpan.github.io/projects/CULane.html links Google Drive and https://pan.baidu.com/s/1KUtzC24cH20n6BtU5D0oyw . The latter could not be opened by the web tool. https://sites.google.com/view/dbald/overview exposed no code link in the retrieved page. The CVF paper landing page returned HTTP 403. These checks do not prove that no author code exists elsewhere, nor resolve the missing exact CULane point-ACC protocol.

## Completed driver_100 recovery and full-run safeguards

The driver_100 archive completed at 5,099,277,019 bytes and passed the full gzip CRC read. SHA256: `0c892fe9f7b8b24eed5b5158f9c5cf546ab5b6e31c9251b55b36a19c460309b2`. Extraction added 21,641 files and confirmed 14,817 existing files byte-identical. One pre-existing truncated JPEG was moved intact into `datasets/raw/culane/recovery_preserved/` before restoring the correct archive copy. Its original hash and path are saved in `results/culane_recovered_truncated_image.json`. No evidence was deleted.

All **17,702 driver_100 test image/annotation pairs pass** decoding, dimensions, JPEG end-marker and annotation validation (`results/culane_driver100_validated.json`). Driver_193 is still transferring; this is not full CULane completion.

Appendix C of https://arxiv.org/html/2508.15778v2#A3 explicitly specifies ResNet-34 for all four detectors. `configs/table1_author_protocol.json` records this verified requirement. The full clean launcher now records backbone matches separately from checkpoint equivalence. Our SCNN CULane R50 checkpoint does not match this requirement; it remains a supporting clean baseline only.

`scripts/benchmark_culane.py` now checks every resumed prediction key, finite runtime and exported-file hash before accepting saved work, and requires exact full coverage before scoring. Duplicate, unknown and corrupted resume records are rejected. The final relevant regression suite passed **50 tests** (`results/culane_recovery_final_regression.xml`). No full TuSimple accuracy run was repeated.

Full transfer/validation/evaluation continuation is logged to `results/culane_completion_pipeline.log`. It waits for the exact driver_193 size, runs `scripts/extract_culane_missing.py driver_193_90frame.tar.gz`, then `scripts/run_culane_full.py --output results/full_culane_recovered_20261028 --execute`. The latter refuses inference until the complete split passes integrity checks. Its metric is official CULane F1, not an asserted DBALD point-ACC reproduction.

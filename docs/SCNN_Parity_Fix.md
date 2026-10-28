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

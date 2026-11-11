# DBALD lane detection reproducibility project

Executable Windows research project targeting the evidence behind Table 1 of the supplied CVPR 2026 paper. **Current results are independent clean-model benchmarks, with full splits and subsets labelled separately, not reproduction of the backdoor experiments.** Operational poisoning (`src/lane_repro/poison_pipeline.py`, `scripts/build_poisoned_dataset.py`) and real ASR measurement against the vendored official evaluator (`src/lane_repro/lane_metrics.py`) are now implemented and verified end to end on the 16-image subset, with a real LaneATT gradient attention map driving trigger placement (`src/lane_repro/laneatt_heatmap.py`) and a tiny-scale real fine-tuning sanity check (`scripts/sanity_poisoned_training.py`). This is still far short of a Table 1 reproduction: the trigger itself is a flat-colour placeholder, not a diffusion-rendered one (stealth optimization is not implemented), and nothing has run at the paper's scale (full dataset, 100 epochs, ResNet-34, all four detectors). See [docs/poisoning_pipeline_status.md](docs/poisoning_pipeline_status.md) for exactly what is real versus placeholder and what remains for a GPU-scale run.

## Latest clean-evidence verification

PR #7 is merged. The current clean-only workflow reuses saved full-test predictions, verifies checkpoints and provenance, and exports the original page-7 table plus per-cell differences. See [the reproduction report](docs/Table1_Clean_Reproduction_20261106.md). CULane is fully available locally; all four clean models already have full 34,680-image results. CULane F1 and unconfirmed point accuracy remain separate from Table 1 ACC.

```powershell
.venv/Scripts/python.exe scripts/replay_clean_table1.py --output results/table1_clean_replay_NEW
```

## Overnight research execution

The latest completed measurements are in [results/experiment_summary.md](results/experiment_summary.md); full-split and subset rows are explicitly separated. [docs/reproduction_report.md](docs/reproduction_report.md), [docs/professor_briefing.md](docs/professor_briefing.md), [docs/methodology.md](docs/methodology.md), and [docs/experimental_limitations.md](docs/experimental_limitations.md) explain the evidence and limits. The author's extended Appendix C identifies **ResNet-34 for all four models**; earlier R18 engineering baselines remain preserved. R34 runs are stored separately. SCNN's available CULane checkpoint is R50, not the required R34.

All research stays local. Do not push, publish, or upload code, results, datasets or papers to remote version control without explicit approval. Public dataset/model downloads do not upload research inputs.

Original package versions are preferred; see [package compatibility](docs/package_compatibility.md). LaneATT's original PyTorch 1.6 anchor ordering is restored and checked against extracted original sorting code. Corrected full TuSimple accuracy matches the upstream rounded ACC: R18 95.57%, R34 95.63%. Earlier ordering-affected runs remain preserved but are excluded from current tables. See [compatibility evidence](docs/laneatt_compatibility.md). The earlier CULane download blockage was resolved; preserved full clean evaluations now cover its official test split.

```powershell
# Download/resume evaluation data; integrity manifests state completion.
.\.venv\Scripts\python.exe scripts/fetch_tusimple_full.py
.\.venv\Scripts\python.exe scripts/fetch_culane_test.py
.\.venv\Scripts\python.exe scripts/fetch_overnight_models.py
.\.venv\Scripts\python.exe scripts/fetch_r34_models.py

# Real GPU evaluations, serial model runs, incremental predictions.
.\scripts\run_gpu.ps1 scripts/run_suite.py --mode full_test
.\scripts\run_gpu.ps1 scripts/run_suite.py --mode full_test --config configs/resnet34.json --output-category full_test_r34
.\scripts\run_gpu.ps1 scripts/run_culane_suite.py
.\scripts\run_gpu.ps1 scripts/run_suite.py --mode robustness

# Corrected LaneATT runs have separate artifacts; choose a fresh category when source changes.
.\scripts\run_gpu.ps1 scripts/run_suite.py --mode full_test --models LaneATT --output-category laneatt_verified
.\scripts\run_gpu.ps1 scripts/run_suite.py --mode full_test --models LaneATT --config configs/resnet34.json --output-category laneatt_r34_verified

# Actual original-version forward reference, followed by numerical comparison.
.\.venv\Scripts\python.exe scripts/setup_reference_environment.py LaneATT
.\tools\python38\python.exe scripts/legacy_laneatt_reference.py
.\scripts\run_gpu.ps1 scripts/compare_legacy_laneatt.py

# Validation and refreshed deliverables.
.\scripts\run_gpu.ps1 scripts/validate_models.py
.\scripts\run_gpu.ps1 -m pytest -q --basetemp tmp/pytest-local --junitxml=results/overnight/tests.xml
.\.venv\Scripts\python.exe scripts/audit_predictions.py
.\.venv\Scripts\python.exe scripts/build_deliverables.py
```

Run one GPU suite at a time. The runners skip completed outputs and reject changed configurations for partially completed runs; select a new output category for a changed experiment. Per-image JSONL, official submissions, plots, hashes, run configuration, hardware observations and completion markers are retained under `results/overnight/`. `results/measured_results.csv` includes only completed runs. The code does not call ordinary robustness scores ASR.

The official CULane evaluator is built locally by `scripts/build_culane_evaluator.py`, using packages under `tools/culane_runtime/` and the existing MSYS2 compiler. Native metric code is unchanged; optional GUI is removed. OpenCV 4.13 replaces the original 2.4 dependency and is a documented compatibility difference. Its F1 score is not the paper's point ACC.

The optional local diffusion demo uses `scripts/fetch_diffusion.py` followed by `scripts/diffusion_demo.py` in the CUDA environment. It uses a fixed mask, no attack objective, and preserves the safety checker. The pinned model revision and hashes are in `results/diffusion_assets.json`. It is separate from the OpenCV Telea demonstration and from DBALD's optimized generation.

## Run the verified LaneATT pipeline

Run from this project directory in PowerShell. The existing `.venv` is isolated from your system Python.

```powershell
.\.venv\Scripts\python.exe scripts/infer.py --model LaneATT --checkpoint models/checkpoints/laneatt_official/experiments/laneatt_r18_tusimple/models/model_0100.pt --image datasets/raw/tusimple_subset/clips/0530/1492626760788443246_0/20.jpg --device cpu --output results/laneatt_cpu
```

The verified CUDA environment is stored on C: at `%USERPROFILE%\.cache\lane-repro-cuda` because D: lacked installation space. It reuses dependencies from this project's CPU `.venv`; retain both environments. Use the launcher below for GPU execution. Batch size is one. Predictions include pixel coordinates, checkpoint SHA256, preprocessing, timing and GPU peak allocation. The overlay is written beside `predictions.json`.

```powershell
.\scripts\run_gpu.ps1 scripts/infer.py --checkpoint models/checkpoints/laneatt_official/experiments/laneatt_r18_tusimple/models/model_0100.pt --image datasets/raw/tusimple_subset/clips/0530/1492626760788443246_0/20.jpg --device cuda --output results/laneatt_cuda
.\scripts\run_gpu.ps1 scripts/evaluate.py --checkpoint models/checkpoints/laneatt_official/experiments/laneatt_r18_tusimple/models/model_0100.pt --device cuda --benign --output results/measured/laneatt_cuda
.\scripts\run_gpu.ps1 scripts/check_laneatt_parity.py
.\.venv\Scripts\python.exe -m pytest -q --basetemp tmp/pytest-local
.\scripts\run_gpu.ps1 scripts/audit_environment.py --suffix cuda
```

The evaluator uses the actual TuSimple test labels and official scoring functions. Official scoring includes a 200-ms runtime gate. A separately named ungated diagnostic helps distinguish slow execution from geometric error. Neither score is ASR. The 16-image subset is fixed before evaluation and is too small for a full benchmark claim.

## Fresh Windows setup

Use 64-bit Python 3.12, not the system Python 3.14 or 32-bit Python. A normal Python 3.12 installation can create the environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For CPU only, replace `cu128` with `cpu`. A fresh CUDA installation into `.venv` can use that environment directly instead of the machine-specific launcher. This machine's RTX 5050 requires a modern Blackwell-capable runtime; the original 2021-era CUDA stack is not assumed compatible. Exact installed versions and hardware measurements are recorded by `audit_environment.py` in `results/environment_cpu.json` or `environment_cuda.json` and `requirements-lock-cpu.txt` or `requirements-lock-cuda.txt`. Editable paths are machine-specific; the older unsuffixed files are historical CPU snapshots.

Clone the URLs in `docs/sources.md` into the named `third_party/` folders and check out the commits in `results/repositories.json`. Download LaneATT's author archive using the URL in `results/laneatt_assets.json` (about 1.43 GB), save it as `models/checkpoints/laneatt_official.zip`, then run:

```powershell
.\.venv\Scripts\python.exe scripts/extract_laneatt.py
.\.venv\Scripts\python.exe scripts/fetch_assets.py
.\.venv\Scripts\python.exe scripts/fetch_tusimple_subset.py
```

Downloads are only in these explicit setup scripts. Inference/evaluation uses local files and does not upload images. CRC and SHA256 provenance are recorded; strict model loading fails on incompatible checkpoints.

All nine LaneATT checkpoints have already been extracted. The verified download archive was removed to recover disk space; it is not needed for inference. D: storage was expanded before overnight execution (about 35 GiB free at its start); current disk checks are recorded during downloads and benchmarks.

## Other models and safe demonstrations

SCNN / RESA use `scripts/infer.py` or `scripts/evaluate.py` with `--model SCNN` or `RESA`, `--config third_party/lanedet/configs/scnn/resnet18_tusimple.py` or `configs/resa/resa18_tusimple.py`, and the matching extracted checkpoint under `models/checkpoints/ssd/zhengtu/releasev2/`.

ADNet uses `--model ADNet --checkpoint models/checkpoints/adnet_tusimple_res18.pth`; see `docs/sources.md` for compatibility differences. Its additional utilities use `mmcv==1.7.0` (Python-only), `timm==0.4.12`, `regex`, and matplotlib. Build MMCV with `--no-build-isolation` and setuptools 78.1.0 / wheel 0.45.1 to avoid its `pkg_resources` packaging failure. Do not install old `mmcv-full` binaries into the modern CUDA runtime.

`scripts/prototype_demo.py --device cuda` demonstrates prompted SAM2 road segmentation, candidate-window visualization, clean feature sensitivity, fixed-region ordinary inpainting, MSE and SSIM. It does not select edits using gradients. Install SAM2 from its pinned checkout with `SAM2_BUILD_CUDA=0`, plus hydra-core 1.3.2 and iopath 0.1.10. Its CUDA postprocessing extension is optional and not built here.

## Evidence

The thesis-directed, offline road-restoration distillation experiment is documented in
`docs/distillation_methodology.md`. It trains a compact student from cached SD1.5
teacher trajectories, with separate training, validation and held-out scenes. Run
`scripts/validate_distillation_inputs.py`, then `scripts/distill_road.py cache`,
`scripts/distill_road.py train`, `scripts/distill_road.py evaluate`,
`scripts/evaluate_edit_lanes.py`, and `scripts/report_distillation.py` through the
documented GPU Python environment. Generated evidence stays in
`results/distillation/v1/`; the report builder requires complete measured artifacts.
This is ordinary road restoration within the security research context, not
operational backdoor poisoning. Published ASR remains a separate reference.

- `results/verified_baseline.md`: verified GPU measurements, CPU/CUDA agreement and remaining limitations.
- `results/reference/table1.csv` / `docs/table1.md`: immutable published reference, with manifest checksum.
- `results/measured/`: actual per-image predictions, summaries and provenance.
- `results/tests.xml`: genuine test report.
- `docs/asr_reproduction_status.md`: requirements and missing evidence for ASR equivalence.
- `docs/paper_analysis.md`: paper equations, hyperparameters, ablations and inconsistencies.
- `docs/sources.md`: official sources, licenses and implementation differences.

Do not compare the subset score directly with Table 1 as if the backbone, checkpoint, split and protocol were identical. CULane's official IoU/F1 and the paper's point ACC are different quantities. Full CULane evaluation and equivalent DBALD ASR experiments remain unverified.

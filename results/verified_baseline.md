# Verified clean inference milestone

Measured on 2026-10-19, Windows 11, RTX 5050 Laptop (8 GB), PyTorch 2.7.1+cu128. All four checkpoints load strictly and execute on the GPU. These are independent clean-checkpoint diagnostics on the same 16 predetermined TuSimple test images, not DBALD Table 1 reproduction.

| Model | TuSimple ACC (%) | Median latency (ms) | Peak allocated GPU memory (MiB) |
|---|---:|---:|---:|
| LaneATT | 95.6752 | 12.23 | 119.11 |
| ADNet | 95.6938 | 19.50 | 91.44 |
| SCNN | 96.8750 | 64.77 | 102.65 |
| RESA | 96.4379 | 33.85 | 145.86 |

Source: each model's `measured/*_cuda/summary.json`, with per-image predictions and checkpoint/label hashes beside it. Official TuSimple scoring includes its 200 ms runtime gate; every GPU image passed that gate. Latency includes preprocessing, forward pass, decoding, portable NMS where applicable, and coordinate sampling, after one warmup; excludes input IO. Memory is PyTorch peak allocated memory, not total device consumption or a training-memory estimate. GPU runs were sequential.

LaneATT CPU/CUDA agreement passed on all 16 images with identical lane counts and validity masks; maximum floating-point coordinate difference was 0.0462 px (0.1 px tolerance). See `laneatt_cpu_cuda_parity.json`. This does not establish parity with upstream native CUDA NMS, which is replaced by portable inference NMS. ADNet also replaces MMCV deformable convolution with torchvision's implementation; original-kernel parity remains unverified. SCNN/RESA use LaneDet PyTorch implementations and checkpoints, not a verified DBALD training setup.

LaneATT's fixed brightness, blur and contrast evaluations scored 95.8054%, 95.5171% and 96.1496%, respectively. These are ordinary image robustness measurements. They are not attack success rates. ASR is null throughout.

Twelve automated tests passed in the CUDA Python environment (`tests_cuda.xml`). The checkpoint inference runs additionally exercise all four GPU networks. Visual inspection of `laneatt_cuda/overlay.png` confirms road-aligned predictions but also visible extra/spurious lanes; executable inference does not imply error-free detection.

Current outputs under `measured/` use upstream integer truncation and retain positive out-of-frame coordinates for official scoring. Earlier diagnostic outputs elsewhere in `results/` may predate these conversion fixes and are superseded for scoring. The published reference under `reference/` was not altered.

Remaining scientific requirements: full datasets and sufficient disk space; equivalent backbones/checkpoints/training setup; resolved CULane point-ACC protocol; author artifacts and exact target/scoring conventions for ASR. See `../docs/asr_reproduction_status.md`. No Table 1 result is claimed reproduced.

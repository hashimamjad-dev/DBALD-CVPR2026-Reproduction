# Verified sources and compatibility

Audit date: 2026-10-19. Exact downloaded commits are recorded in `results/repositories.json`; checkpoint hashes in `results/assets.json` and `results/laneatt_assets.json`. Links below establish upstream provenance, not equivalence to DBALD's unpublished training setup.

| Component | Primary source | License / dependencies / weights |
|---|---|---|
| DBALD | https://sites.google.com/view/dbald/overview | Linked by supplied paper; no source release/license or poisoned checkpoints verified on accessible overview. Supplement not obtained. |
| LaneATT | https://github.com/lucastabelini/LaneATT | MIT; original PyTorch 1.6, CUDA 10.2, compiled lane NMS. Author Google Drive experiment archive obtained, including configs and checkpoints. |
| ADNet | https://github.com/Sephirex-X/ADNet | Apache-2.0; original PyTorch 1.12.1, CUDA 11.3, MMCV-full 1.7.0, timm 0.4.12, native NMS/DCN. Author TuSimple ResNet18 checkpoint obtained from release `ckpt`. |
| SCNN | https://github.com/XingangPan/SCNN | Original Torch7/Lua implementation, not PyTorch; repository LICENSE retained. Original checkpoint links in README. Windows-invalid `datasets/\\` entry prevents ordinary complete checkout. |
| RESA | https://github.com/ZJULearning/resa | Apache-2.0; Python 3.8, PyTorch, CUDA 10.1 setup guidance; author links TuSimple ResNet34 / CULane ResNet50 weights and recommends LaneDet. |
| PyTorch SCNN/RESA | https://github.com/Turoad/lanedet | Apache-2.0. LaneDet's TuSimple ResNet18 checkpoints used here. SCNN is a third-party PyTorch implementation; RESA authors link this toolbox. Neither is a verified DBALD checkpoint. |
| SAM2 | https://github.com/facebookresearch/sam2 | Apache-2.0 main code/checkpoints; additional BSD-3-Clause optional connected-components code. Python >=3.10, torch >=2.5.1, torchvision >=0.20.1. SAM2.1 tiny checkpoint obtained; optional CUDA extension disabled. Version is our engineering choice. |
| UltraEdit | https://github.com/HaozheZhao/UltraEdit | README links `BleachNick/SD3_UltraEdit_w_mask`; bundled modified diffusers, torch 2.3.0+cu118, transformers 4.41.2. No active top-level LICENSE file found; commented README license text does not grant a license. Model/base-model terms need separate checking. Source inspected, not installed/executed. |
| CULane | https://xingangpan.github.io/projects/CULane.html | Official annotation/split documentation and download links; dataset terms are distinct from model code licenses. Full dataset not present in this workspace. |
| TuSimple | https://github.com/TuSimple/tusimple-benchmark | Apache-2.0 repository; dataset terms separate. Official evaluator retained. Maintainer issue #3 links the Kaggle mirror after old S3 links stopped working; that Kaggle mirror itself returned HTTP 404 on 2026-10-23 (confirmed by direct request, not assumed), so `scripts/fetch_tusimple_subset.py` falls back to https://huggingface.co/datasets/dhbloo/TuSimple, verified against the expected 2,782-record test-label count and (720,1280,3) image shape before use. A 16-image raw test subset with labels is downloaded; no full-benchmark claim. |

## Runtime changes

LaneATT: retain original architecture and strict checkpoint load. Replace removed `np.bool` with `np.bool_` only at module import; use a portable implementation of the upstream lane-distance NMS predicate. Preprocess in BGR using cubic resize, matching upstream imgaug's default. CUDA network execution is possible while NMS remains CPU postprocessing. Native-kernel parity remains unverified.

SCNN/RESA: selectively import upstream modules to avoid unrelated CUDA operators. Execute standalone config files directly because their temporary-file import fails on Windows. Inference network code and checkpoint keys are unchanged.

ADNet: retain network modules and strict weights. Use torchvision's DCNv1 in place of MMCV's compiled DCNv1, portable lane-distance NMS, device-neutral anchor coordinates and direct config loading. This is a compatibility reconstruction; original-stack numerical equivalence is not established merely by loading weights.

The Python evaluator functions from TuSimple are executed unchanged; only the obsolete Python 2 CLI is excluded. No scoring implementation is replaced by invented numbers. The scalar NMS predicate originates in LaneATT/ADNet's source; upstream licenses and their embedded attribution notices remain in the checked-out repositories.

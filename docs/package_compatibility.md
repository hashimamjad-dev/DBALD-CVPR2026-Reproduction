# Original package versions

User preference: use the original researchers' exact versions, or the closest feasible versions. Preserve the existing working environment and results. Never silently substitute a newer runtime and call it exact reproduction.

The retrieved DBALD material does not provide a verified complete environment lock. The version pins below belong to the public detector implementations; they are not asserted to be DBALD's private experiment environment.

| Implementation | Published dependencies | Local approach |
|---|---|---|
| LaneATT | PyTorch 1.6; tested CUDA 10.2; OpenCV 4.2.0.32, SciPy 1.4.1, scikit-learn 0.23.2, imgaug 0.4.0 | Separate Python 3.8.10 / torch 1.6 CPU reference; torchvision 0.7 official pairing; exact numerical-library pins where available |
| ADNet | torch 1.12.1, torchvision 0.13.1, CUDA 11.3, NumPy 1.23.4, MMCV-full 1.7.0, timm 0.4.12 | Pins recorded; current GPU compatibility measurements still use the documented newer runtime and torchvision DCNv1 substitute |
| LaneDet SCNN/RESA | torch 1.8.0, torchvision 0.9.0, MMCV 1.2.5 | Pins recorded; current completed measurements use newer runtime, not original-version parity |
| Original SCNN | Torch7/Lua implementation | LaneDet is a separately identified PyTorch port; no claim of exact original implementation equivalence |
| Original CULane scorer | Original C++ evaluator uses OpenCV 2.4-era APIs | Local executable uses OpenCV 4.13; rasterization parity remains unresolved |

The RTX 5050 requires a newer CUDA stack than the original detector releases. The existing torch 2.7.1+cu128 GPU environment remains useful for execution, but its deviations are explicit. The old CPU reference compares preprocessing and raw neural-network outputs before suppression, avoiding a false claim that the unavailable old CUDA NMS was validated.

`requirements-laneatt-legacy.txt` distinguishes upstream pins from chosen contemporary versions where authors did not pin. The installed freeze is `requirements-lock-laneatt-reference.txt`; `pip check` passes. Exact ujson 1.35 was compiled locally after obtaining matching Python development headers. `scripts/legacy_laneatt_reference.py` executed original LaneATT source and actual old torch sorting on 16 images; `scripts/compare_legacy_laneatt.py` passed all 16 comparisons with identical preprocessing and maximum pre-NMS difference 0.0003662109375.

`scripts/setup_reference_environment.py LaneDet` and `ADNet` build separate pinned CPU environments for those implementations. Their numerical-library pins not specified upstream are explicitly contemporary choices. ADNet uses MMCV 1.7.0's Python modules plus the existing torchvision DCNv1 adapter, not an unverified claim that MMCV-full's old CUDA kernel ran. Environment freezes and comparison artifacts distinguish installation from executed validation.

Sources: pinned local repository requirements and README files, [PyTorch previous versions](https://pytorch.org/get-started/previous-versions/). No existing environment is downgraded in place.

## Executed checks, 20 October 2026

Both LaneDet and ADNet reference environments pass `pip check`, with refreshed installed freezes. The same adapters were executed on 16 fixed images in old CPU and modern CUDA runtimes. ADNet and RESA passed all 16 preprocessing/decoded-coordinate comparisons at 0.2-pixel tolerance. This validates those samples, not original MMCV-full CUDA kernels or all inputs.

SCNN failed all 16 decoded-coordinate comparisons despite exactly identical preprocessed inputs. A one-image stage diagnostic (`scripts/diagnose_scnn_runtime.py`) localized the divergence: old versus modern CPU backbone maximum difference was 0.000002384, but aggregation differed by 19.7706 and output logits by 20.0483. Modern CPU and modern CUDA show the same large divergence from the old CPU aggregation, so this is not explained solely by switching CPU to GPU. The cause remains unresolved; no compatibility patch or replacement full baseline is claimed. Invalid-point/count mismatches are part of the failed comparison; large coordinate differences involving the -2 invalid sentinel are not physical lane displacements.

Artifacts: `results/overnight/reference_versions/{scnn,resa,adnet}/comparison.json`, and SCNN `stage_comparison.json`. Existing modern-environment clean measurements remain measurements of that declared environment; they are not upgraded to original-runtime-equivalent results.

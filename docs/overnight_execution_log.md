# Overnight execution log

Local session: 19 October 2026, Asia/Karachi. This is an incremental account; final counts and completed runs are taken from machine-readable artifacts, not inferred from this narrative.

- Read the overnight brief; retained earlier verified results. Rechecked D: storage: approximately 35 GiB available.
- Started resumable downloads of all 2,782 TuSimple labelled test frames from the existing maintainer-linked mirror, reusing the 16 cached images.
- Identified the official CULane test archives and split list. Downloaded/extracted the list and began test-only archive acquisition. Hashes and extraction counts are recorded in `results/culane_download.json`; redundant verified archives are removed only after extraction.
- Added incremental, resumable clean/robustness evaluation with source/config/checkpoint hashes and thermal/storage checks. Prior result directories are preserved.
- Added seeded rain-like streaks, fixed occlusion and JPEG compression. Executed all four models on 16 images under seven conditions (448 model-image evaluations).
- Downloaded OpenCV C++ packages into the project without installing system packages. Compiled the official CULane evaluator with optional GUI removed. Resolved missing runtime DLLs using local matching packages.
- Passed known-answer CULane tests (perfect, missed, spurious and empty lanes) and TuSimple submission aggregation tests. The first expanded test run passed 23 tests.
- Executed real two-image tensor batches for every architecture. All four matched serial decoding within the stated 0.2 px tolerance on two fixed images.
- Fixed SAM2 import resolution in the split CUDA/CPU environment; reran prompted road segmentation, clean feature sensitivity, candidate windows and descriptive window statistics successfully on GPU.
- Began downloading pinned ordinary Stable Diffusion inpainting weights and additional CULane checkpoints. No research material is uploaded.
- Independently rescored all 28 robustness submissions using TuSimple's original submission entry point; aggregates matched.
- Ran LaneATT on 16 deterministic CULane smoke images with official C++ scoring; saved the result explicitly as a subset, not full evaluation.
- Found the author's extended arXiv v2 paper and its Appendices A–E. Appendix C identifies ResNet-34 for all four models, correcting the earlier unresolved-backbone note. Added R34 support and started obtaining available R34 checkpoints. The attached PDF's reference table remains unchanged.

Detailed command starts/ends and exit statuses: `results/overnight/execution.jsonl`. Download progress: `results/tusimple_download.log`, `results/culane_download.log`, `results/culane_checkpoints.log`, `results/r34_checkpoints.log`, `results/diffusion_download.log`. Hardware observations are beside each run. Test failures/fixes and subsequent completion are appended below as work proceeds.

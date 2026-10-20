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

- Completed all 2,782 TuSimple frames for four R18 models and three available R34 models. Independently rescored submissions. Completed the ordinary local diffusion demonstration with retained provenance and output images.
- Investigated LaneATT's high false-positive rate. Found that modern torch reordered equal-frequency anchors: strict weight loading did not preserve the training-time attention indices. Extracted and compiled the original PyTorch 1.6 sorting function; restored its exact permutation with frequency/source hashes. Preserved earlier runs as superseded and reran R18/R34 full tests and the fixed robustness suite. Corrected ACC rounds to the upstream 95.57%/95.63%. The upstream 96.71%/96.77% figures are F1, not ACC.
- Bounded CULane range downloads advanced the partial driver_100 archive beyond 3.5 GB, then the server returned quota-exceeded HTML. Preserved all validated ranges, rejected HTML as archive data, and stopped retries. The driver_193 archive also returned quota exceeded. Full CULane evaluation remains incomplete, not reported as a benchmark result.
- User requested original or near-original package versions. Created an isolated local Python 3.8.10 runtime and began installing the LaneATT authors' PyTorch 1.6-era dependencies. Prepared actual old-runtime pre-NMS inference and modern-GPU comparison scripts. DBALD's complete package lock remains unavailable; public-detector pins are distinguished from DBALD-specific evidence.
- A test rerun initially hit the sandbox's default temporary-directory permissions (19 passed, four fixture setup errors). Repeated with a project-local temporary directory: all 23 tests passed. Refreshed actual two-image tensor batch validation after the anchor correction.

## 20 October 2026: thesis-directed distillation experiment

- Resumed only after the user explicitly ended the gaming-related pause. GPU was idle at 53 C. Completed 80 cached teacher trajectories with automatic cooling breaks; each cached transition passed finite-value and numerical reconstruction checks.
- Prepared 64 training, 16 validation and 16 held-out scenes without training/test clip overlap. Input QA caught CULane's visible hood, so a fixed camera-specific road mask was established before evaluation. All 96 source-image hashes were verified.
- Trained the 2,381,044-parameter student for 2,000 updates in 155.32 seconds including cooling. Chose update 1,700 using validation rollout error alone. Training peak allocated GPU memory was 1,198.3 MiB.
- Executed 96 held-out method/scene/seed cases with two timing repetitions each: 20-step teacher, undistilled 4-step teacher and distilled 4-step student. Student latency was about 0.31 seconds versus 3.48 seconds for the 20-step teacher, but masked fidelity and inspected edit realism were worse. Saved this negative quality result without tuning against test images.
- Executed 112 frozen LaneATT clean/edited image assessments and official CULane scoring on the eight-scene editing subset. These are benign editing consistency measurements, never ASR. Preserved all earlier full clean TuSimple results.
- Retried official CULane availability once: a one-byte probe succeeded, but actual bounded-range downloads still returned quota-exceeded pages. Preserved all validated partial data; full CULane remains incomplete.
- Finished old-version runtime comparisons: ADNet and RESA passed all 16 cases. SCNN failed; a one-image stage diagnostic localized a large old/new-runtime divergence after aggregation. No unverified fix or replacement full benchmark was introduced.
- Built `docs/distillation_report.md` from measured artifacts, with separate dataset tables, published DBALD reference values, raw-image galleries, and explicit scientific limitations. Research files remain local; no push or remote publication.

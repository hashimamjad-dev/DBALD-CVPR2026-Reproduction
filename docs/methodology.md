# Methodology and execution protocol

## Scope

This project independently evaluates public clean lane-detector checkpoints and implements isolated, non-operational demonstrations. It does not train backdoors, alter training labels, optimize trigger placement or stealth, or interact with deployed vehicles. All research files remain local. Public downloads are permitted; publication and remote version-control uploads are not.

## Models and preprocessing

| Model | Representation | Native preprocessing |
|---|---|---|
| LaneATT | Lane anchors with attention and coordinate refinement | BGR, full-frame cubic resize to 640×360, divide by 255 |
| ADNet | Adaptive lane anchors and deformable feature sampling | BGR, dataset-specific crop, cubic 800×320 resize, divide by 255 |
| SCNN | Segmentation with spatial message propagation | BGR, checkpoint-configured crop/resize, configured mean and standard deviation |
| RESA | Segmentation with recurrent feature shifts | BGR, checkpoint-configured crop/resize, configured mean and standard deviation |

Weights load strictly with PyTorch's weights-only loader. Checkpoint hashes, original configuration paths, package versions and upstream repository commits are retained. The initial TuSimple checkpoints use ResNet-18. The extended DBALD paper specifies a unified ResNet-34 backbone in Appendix C, so additional R34 measurements are kept separate. A public SCNN R34 checkpoint has not been verified. [Author extended paper](https://arxiv.org/html/2508.15778v2#A3)

Windows adaptations are thin wrappers: NumPy boolean alias compatibility, direct configuration loading, device-neutral anchors, portable lane NMS, and torchvision deformable convolution for ADNet. Source checkouts are preserved. The ADNet/NMS substitutions have not been proved equivalent to the original native kernels. CPU/CUDA and serial/tensor-batch checks evaluate this implementation's consistency, not original-kernel equivalence.

## Datasets and metrics

TuSimple evaluation uses its labelled final video frames. The context frames are unnecessary for these single-image networks. The official test split contains 2,782 labelled frames. The maintainer-linked public mirror supplies images and labels; downloaded images are decoded, checked for 1280×720 dimensions and hashed. `tusimple_full_manifest.json` states whether the complete split passed. The preserved 16-image subset is explicitly a diagnostic subset, selected before prediction.

TuSimple scoring calls the original benchmark functions, with their invalid-point, angle, lane-count and 200 ms runtime conventions. Coordinates use upstream integer truncation. Negative predictions become -2; positive out-of-frame predictions remain available to the scorer. The reported ACC is the mean of official per-image scores, not a pooled pixel accuracy. FP and FN are official per-image rates. Saved submissions are independently passed through `bench_one_submit` to verify aggregation. Runtime is measured honestly; some upstream exporters use a nominal runtime, so runtime-gated comparisons need care. [Official benchmark](https://github.com/TuSimple/tusimple-benchmark/tree/master/evaluate)

CULane uses official test archives and lists. Its official evaluator rasterizes lanes at width 30, computes lane IoU, matches lanes, and counts matches above 0.5. Global TP, FP and FN determine precision, recall and F1. We compile the author's C++ spline/matching/scoring code with OpenCV 4.13; only optional GUI visualization is removed. Missing prediction or annotation files are rejected before invoking upstream code. OpenCV-version equivalence with the original 2.4 implementation remains a limitation. **CULane F1 is not the paper's point-based ACC.** [Dataset and evaluator](https://xingangpan.github.io/projects/CULane.html)

## Execution and reproducibility

Benchmark runs process image batches serially with tensor batch size one to limit memory and produce per-image timings. Separate validation checks real two-image tensor batches against serial outputs. Seeds are zero, cuDNN benchmarking is disabled, deterministic cuDNN algorithms are requested, and TF32 is disabled. This does not promise bitwise equivalence across hardware or dependency versions.

Each image's prediction and runtime are flushed to JSONL immediately. Resume requires identical model, labels, configuration and source hashes. Completed prior results are retained. GPU timing is synchronized and excludes input IO, transformation generation and file export. One warmup is excluded. Peak memory is PyTorch allocated memory, not total GPU reservation or a training requirement. Hardware logs include GPU temperature, device memory, power, available RAM and disk. Runs stop at 85°C or low disk space.

## Benign robustness and isolated demonstrations

Seven conditions are fixed in advance: clean, brightness ×0.7, contrast ×0.75 around 127.5, Gaussian blur σ=1.5 with a 9×9 kernel, 400 seeded rain-like streaks, a fixed gray occlusion, and JPEG quality 30. The first robustness suite uses the same 16 images across all models. Results are paired diagnostics, not full-dataset robustness estimates. Original inputs, transformed examples, predictions and plots are retained; none is labelled ASR.

SAM2 uses explicit positive road and negative background prompts on one image. Sensitivity is the absolute input gradient summed across channels for clean backbone feature energy; it is not the paper's attack loss. Road-mask windows are enumerated and described in spatial order without selecting an attack location. Inpainting uses a fixed mask independent of gradients. The ordinary diffusion demonstration uses a public inpainting model without adversarial losses; OpenCV Telea is retained as a separate classical example. MSE and SSIM are measured afterward. Pixels outside the edit are explicitly copied, so perfect environmental consistency is expected and does not demonstrate naturalness or stealth.

ASR fields remain null. Clean errors, robustness degradation, lane displacement and image similarity cannot substitute for poisoned-model attack success.

# Experiment v2: predeclared quality investigation

This continues v1's compact, conditional, four-step diffusion student and frozen SD1.5 inpainting teacher. It does not change the task to ordinary non-diffusion image filtering. V1 artifacts are read-only. Its test images/seeds are reserved for a final locked comparison; their previously observed failure is disclosed, so they are a reused benchmark, not a newly unseen test set. No test outputs select a candidate.

## Investigation and controlled candidates

Use eight fixed TuSimple validation clips from the existing 16 validation clips, plus four CULane scenes from the available smoke subset excluding every v1 test scene. Evaluate seed 0 (used in original training) and seed 17 (not used in training). Compare v1's teacher-forced and rollout errors inside/outside the edit, and inspect decoded crops. This tests noise generalization, local loss imbalance and error accumulation separately from the already verified compositing arithmetic.

Keep the same 64 training images, architecture, four-step sampling schedule, initialization (v1's selected checkpoint), batch size 8, optimizer and 1,000 additional update budget for three controlled candidates:

1. `control_global`: original global epsilon MSE and original seed-0 cache. This is an equal-budget continuation control, not the proposed improvement.
2. `mask_single`: replace the objective with 50% global MSE plus 50% MSE in the latent edit region dilated by two latent pixels. Keep the original data/noise. The dilation covers a neighborhood of the VAE boundary; it is not trigger optimization.
3. `mask_diverse`: same balanced objective, with each training scene additionally receiving one independently generated teacher trajectory at seed 2000 + its scene index. Sample uniformly from the 128 cached scene/noise examples. This isolates added noise diversity from the loss change. Image count and student architecture remain unchanged.

Evaluate checkpoints at updates 500 and 1,000 using decoded validation outputs. Select by mean local LPIPS, subject to finite values and no worse boundary error than v1; break ties by masked MAE. The original v1 checkpoint remains an eligible reference. Use no lane-detector gradients or test-set scores for selection. Failure to meet acceptance criteria is reported rather than weakening the criteria after seeing results. A larger architecture remains a possible subsequent controlled experiment, not an assumed cure.

## Acceptance criteria, fixed before v2 validation measurements

On each dataset's validation subset, require mean masked SSIM to the 20-step teacher >= 0.85; masked MAE <= 1.5 times the undistilled four-step teacher's MAE + 2 (0–255 scale); local LPIPS <= 1.25 times the four-step teacher's LPIPS + 0.02; and boundary error <= the four-step teacher's boundary error + 2. These are project feasibility thresholds, not established road-safety standards.

Visual acceptance requires no conspicuous blotch, rectangular seam or lane discontinuity in at least 90% of inspected validation cases. Inspection is a disclosed qualitative engineering assessment, not a blinded human realism study. Require median end-to-end latency <= 0.60 seconds, at least 3x faster than the contemporaneously measured 20-step teacher, and peak allocated GPU memory < 7 GiB. Final frozen LaneATT assessment must retain >= 0.99 clean-point coverage, <= 1 pixel mean conditional lane displacement, and no more than 0.5 percentage-point TuSimple ACC or 2-point CULane subset F1 degradation from clean. These small-subset checks do not prove driving safety.

Measure raw masked MAE/SSIM, local SqueezeNet-LPIPS on the edit bounding box plus 8px context resized to 128x128, and boundary residual: mean absolute error of across-mask-edge RGB differences relative to the teacher composite. Smaller is better for MAE, LPIPS and boundary residual. Report pixel preservation imposed by compositing separately from learned raw-output quality. LPIPS is perceptual similarity to a reference, not a realism probability. Dataset mask sizes differ as documented in v1; never pool them into a single quality claim.

Cache teacher trajectories and decoded reference crops; reuse v1's seed-0 cache. Teacher generation, training and final inference costs are separate. Use local resumable checkpoints and cooling breaks. The 20-step teacher, undistilled four-step teacher and original student accompany every candidate's validation table. Measure timing under one common protocol on validation scenes; evaluate the locked selected candidate on v1's test scenes only after selection.

## Literature basis

- [Salimans and Ho, progressive distillation](https://arxiv.org/abs/2202.00512): deterministic teacher trajectories motivate fewer-step student targets. Our compact offline regression is not an exact reproduction of progressive halving.
- [Meng et al., guided diffusion distillation](https://arxiv.org/abs/2210.03142): supports investigating few-step guided latent diffusion for editing/inpainting, without guaranteeing that this small student preserves quality.
- [Official Diffusers consistency-distillation recipe](https://huggingface.co/docs/diffusers/training/lcm_distill): samples noise/timesteps and separates teacher/student objectives; motivates testing the degenerate single-noise training coverage. We retain our trajectory method rather than calling it LCM.
- [Zhang et al., learned perceptual similarity](https://arxiv.org/abs/1801.03924): motivates LPIPS alongside pixel measures and visual review.

The local mask-balanced objective is a task-specific experimental hypothesis, not attributed to these papers as a published recipe. Naturalistic-backdoor security remains the research context; no poisoning, trigger optimization, targeted lane manipulation or ASR experiment is implemented.

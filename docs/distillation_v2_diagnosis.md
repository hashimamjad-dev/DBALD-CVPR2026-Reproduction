# V2 diagnosis before candidate training

The fixed validation set contains eight TuSimple scenes and four available CULane scenes, each evaluated at seeds 0 and 17. These are not the held-out test images. The baseline measurements are in `results/distillation/v2/diagnostic_summary.json` and per-step latent diagnostics in `diagnostics.json`.

Noise generalization is a supported contributor: TuSimple student masked MAE increases from 15.71 at the training noise seed to 30.55 at the unseen seed; local LPIPS increases from 0.126 to 0.399. The corresponding four-step teacher stays near MAE 3.03–3.11 and LPIPS 0.060–0.063. CULane remains poor at both seeds, so noise coverage alone cannot explain all failures.

Accumulated student approximation error is also supported. TuSimple final teacher-forced edited-region latent MSE is 0.0270/0.0466 at seeds 0/17, whereas full student rollout MSE is 0.0773/0.2065. CULane has the same direction (0.0686/0.0862 versus 0.1186/0.1451). Teacher-forced transitions start from teacher states; rollout transitions start from previous student predictions. This diagnoses accumulation, not an isolated causal estimate of which network layer is responsible.

Global-loss imbalance is a plausible, directly testable mechanism: the dilated latent edit region occupies only 7.81% of TuSimple and 2.27% of CULane latents. A global average can improve mostly outside the edit. The mask-balanced versus equal-budget global-loss experiment tests this hypothesis. The diversity candidate then isolates additional training noise with the same loss, architecture, image count and update budget.

Four steps alone are insufficient to explain the artifacts: the undistilled four-step teacher achieves validation masked SSIM about 0.894 on TuSimple and 0.946 on CULane. This does not prove a 2.38-million-parameter student can approximate the much larger teacher at four steps. Architecture/capacity remains unresolved because it has not been isolated experimentally.

Low latent error was not a reliable decoded-quality selection metric in v1. V2 measures decoded local pixel, structural and perceptual similarity plus across-mask-edge residuals. Boundary errors are present in the original student (TuSimple 7.85/16.17 and CULane 15.38/16.33); verified compositing arithmetic does not remove a poor generated patch. Background and annotated-lane copying is an imposed constraint, not learned preservation. Texture realism still requires visual review; LPIPS measures fidelity to the teacher, not realism or safety.

Only 64 training scenes are available to this controlled experiment. Increased scene diversity, model capacity, rollout-aware training and decoded perceptual objectives remain future hypotheses unless tested separately. No held-out test metric informed these diagnoses or candidate changes.

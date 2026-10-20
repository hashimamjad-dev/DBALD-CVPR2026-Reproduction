# Independent methodology and repository audit

## Audit and scope

The existing project has executable LaneATT, ADNet, SCNN and RESA adapters, full clean TuSimple public-checkpoint evaluations, a preserved DBALD Table 1 reference, incomplete CULane downloads, and one ordinary diffusion demonstration. It did not have a trained distilled student or a teacher/student experiment. The original-version reference environments and LaneATT anchor-order correction are preserved. Existing benchmark results are reused, not rerun as new evidence.

The provided thesis Chapter 2, especially 2.9–2.11 (printed pages 14–15), motivates reducing the cost of naturalistic backdoor sample generation through task-specific diffusion distillation. Its references contain unresolved `[?]` markers, so technical choices are independently sourced. No Chapter 3 was requested, inferred or used.

DBALD supplies the training-time backdoor threat model and published LDA/LRA/LOA results. The implemented experiment isolates a permissible component question: can a compact student reproduce ordinary road-restoration edits from a general-purpose diffusion teacher more cheaply while preserving lane geometry? It does not establish preservation of attack effectiveness or stealth. No poison-label changes, compromised detector training, trigger-position optimization, detector-gradient objectives or targeted lane manipulation are implemented.

## Approach selection

Progressive distillation learns fewer denoising transitions from a deterministic teacher [1]. Latent consistency distillation and LCM-LoRA are established alternatives [2,3], but a full-size student retains a large inference backbone and standard recipes require substantial training memory. A pretrained generic LCM adapter alone would not demonstrate that our own student learned this road task.

The selected prototype uses offline trajectory distillation into a small conditional latent U-Net. The teacher and student are never trained together in GPU memory. Teacher trajectories and image conditioning are cached; only the compact student is trained. This fits the 8 GB device more plausibly than full-UNet fine-tuning, and the fit is checked by measured allocation. It is a coarse-transition regression prototype inspired by trajectory distillation, not a reproduction of the full LCM or progressive-halving algorithms.

The available, pinned Stable Diffusion 1.5 inpainting checkpoint is the teacher. It is a general-purpose diffusion proxy suitable for the local GPU, **not DBALD's UltraEdit teacher**. Both methods use its frozen VAE. A fixed ordinary-asphalt restoration prompt and predetermined mask rule define the task. The student has no open-ended text conditioning. Training uses teacher transitions from a 20-step deterministic DDIM trajectory, grouped into four 5-step transitions. The target is the equivalent epsilon prediction that reconstructs each coarse transition. Student inference rolls out four steps with its own predicted states; validation therefore includes rollout quality, not only teacher-forced loss.

## Data and measurement protocol

- Training: 64 TuSimple train clips (0313), selected by a fixed filename hash. Validation: 16 separate official validation clips (0531). No training/validation clip may overlap any official TuSimple test clip.
- Held-out evaluation: eight additional TuSimple test scenes, excluding the old debugging subset, and eight available CULane scenes. Two predeclared evaluation seeds. CULane is an out-of-domain subset, not a full benchmark.
- Keep a 512×512 bottom-centre crop, native scale, and a fixed road-restoration mask with annotated lane bands protected. The source image and all lane labels remain unchanged in the dataset. Generated outputs live only in experiment directories.
- Input-only inspection before evaluation found that CULane includes a vehicle hood at the bottom of the frame. Its fixed crop mask is therefore rows 312:328, columns 256:352 (1,536 pixels before lane protection), while TuSimple uses rows 352:448, columns 192:320 (12,288 pixels). The CULane camera-specific mask is a narrow road strip above the hood and clear of the nearby car in scene_088. No generated output or detector error was used to choose it. This changes edit difficulty and limits cross-dataset comparisons; per-dataset results and actual mask areas must remain separate. The earlier input gallery documents the rejected hood-covering mask, not an evaluated condition.
- Compare the 20-step teacher, an undistilled 4-step teacher, and the trained 4-step student. Use identical scene conditions and initial noise per seed, explicit warm-up, CUDA synchronization, and separate stage/end-to-end timings. Count denoiser calls, parameters, checkpoint bytes and peak allocated/reserved VRAM.
- Report teacher agreement and preservation metrics as fidelity proxies, not proof of realism. Inspect uncomposited crops and final composites. Outside-mask identity after compositing is tautological and is not a quality achievement. Evaluate clean-detector lane changes after generation; detector outputs never enter training or sample selection.
- Keep TuSimple and CULane tables separate. Existing clean ACC, published poisoned-model ACC, published ASR, benign robustness and editing consistency use distinct columns/artifacts. No FID claim from this small sample; blinded human realism ratings remain pending unless actually collected.

## Sources

1. Salimans and Ho, *Progressive Distillation for Fast Sampling of Diffusion Models*, ICLR 2022. https://arxiv.org/abs/2202.00512
2. Luo et al., *Latent Consistency Models*, 2023. https://arxiv.org/abs/2310.04378
3. Hugging Face Diffusers, official consistency-distillation implementation and memory controls. https://huggingface.co/docs/diffusers/training/lcm_distill
4. DBALD author paper and extended Appendix C. https://arxiv.org/html/2508.15778v2

The implementation and report will record measured outcomes, including failures or fidelity losses. A faster but visibly worse student is an experimental result, not a successful quality-preserving contribution.

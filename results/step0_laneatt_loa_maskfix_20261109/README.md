# Step 0 passed on Kaggle version 5

Twenty TuSimple training images were poisoned with LOA, a 40 by 40 trigger and MUD_PROMPT through the original UltraEditTrigger. All three text encoders, FP16, CPU offload, 50 diffusion steps, guidance 7.5 and image guidance 1.5 were retained. Model revision e5beecb69a3ef1399718be2f99b61730b8bf0346 and fork revision 1af5f0478d56d62cf6f35781a2ab63e03d656b2d were verified. The existing lane-geometry road mask remains an approximation to the paper's SAM2 road mask.

Attention, rendering, training and scoring ran in separate child processes. UltraEdit rendering completed before LaneATT training started, releasing the diffusion model's RAM and VRAM.

All 20 generated image pairs passed exact outside-footprint equality checks both on Kaggle and after local recovery. Each image changed inside the footprint. Six Adam updates at learning rate 0.0001 used microbatch size one and four-step gradient accumulation, visiting all 20 images and repeating four. Training loss progressed from 23.980581 to 14.471452. A fixed four-image evaluation-mode loss probe decreased from 24.531296 to 13.604895. All values were finite.

The saved checkpoint was strictly reloaded, then all 20 images were scored with the existing official TuSimple evaluator without error. These are training-subset smoke checks. No research ACC or ASR value is reported or added to Table 1.

Measured rendering time averaged 41.874609 seconds per image (range 40.300766 to 60.381046), with 91.362936 seconds of model loading. The rendering child took 940.543594 seconds including its setup. Its peak CUDA allocation was 12,974,168,576 bytes and reservation was 13,111,394,304 bytes (12.210938 GiB). Peak child RSS was 24,007,577,600 bytes. Training's six updates took 7.875048 seconds of measured compute; its complete process took 22.566970 seconds. Training peak CUDA allocation was 450,412,544 bytes (429.546875 MiB), reservation was 460 MiB, and peak child RSS was 2,155,282,432 bytes. CUDA counters exclude driver and unrelated-process allocations.

Scaling extrapolations, not completed runs:
- About 363 edits for 10% of the 3,626-image training set imply 4.22 hours of rendering, plus setup and attention computation.
- Editing all 3,626 images implies 42.18 hours of rendering.
- The observed training rate implies 19.83 minutes of compute per 3,626-image epoch with the same effective batch size. Multiply by the chosen epoch count. For illustration only, 100 epochs imply approximately 33.05 hours, before data loading, checkpointing and evaluation.
- Long-run memory stability and convergence have not been established by six updates. Keep rendering and training sequential.

The two source fixes are documented in fixes.json. Five CPU wrapper tests passed, including RGB mask shape/channel equality, footprint size and compositing. The exact submitted notebook, full run logs, input provenance, output images and trained checkpoint were recovered. local_verification.json checks all image pairs and verifies the checkpoint SHA-256 against the scoring receipt and against the original checkpoint identity.

The smoke checkpoint is retained locally at step0/smoke_checkpoint.pt and in Kaggle notebook version 5 outputs, with SHA-256 c4725506c8b6765c349ef0dcf6a98092b10c34a1d8edd6a15e003e837169e47e. Model weights are excluded from the Git commit. No full run was launched.

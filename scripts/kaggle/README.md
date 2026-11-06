# Kaggle clean evaluation

The existing notebook hashimamjad03/dbald-research runs the repository's benchmark.py.
The default input bundle selects LaneATT ResNet-34 and two clean TuSimple frames.
Its output has subset scope. The full Table 1 clean baseline comes from verified saved full-split predictions.

From the repository root, use the project Python to build a fresh bundle:

    .venv/Scripts/python scripts/kaggle/package_clean.py --output tmp/new-kaggle-input --images 2

The bundle records every transferred source, checkpoint and image SHA-256. Select --images 2782 only when a new full inference is necessary. Upload the bundle as a private Kaggle dataset. The metadata identifies hashimamjad03/dbald-clean-execution. Use dataset versioning for an existing dataset.

On this Windows host the authenticated CLI is at:
C:/Users/Hashim Amjad/AppData/Roaming/Python/Python314/Scripts/kaggle.exe

Run dataset upload commands from the bundle directory with -p . to avoid CLI 2.2.5's relative upload-cache path error. Keep credentials in the existing CLI credential store.

Copy configs/kaggle_clean.json as kernel-metadata.json into a submission directory beside scripts/kaggle/dbald-research.ipynb, then use:

    kaggle kernels push -p <submission-directory>
    kaggle kernels status hashimamjad03/dbald-research
    kaggle kernels output hashimamjad03/dbald-research -p <fresh-recovery-directory>
    .venv/Scripts/python scripts/kaggle/verify_recovery.py <fresh-recovery-directory>

The notebook uses the attached private inputs with internet disabled. Scratch extraction goes under /kaggle/temp. Only evidence is saved under /kaggle/working. Failures produce a traceback receipt and stop execution. Benchmark inference retains the existing checkpoint adapter, anchor ordering, preprocessing, official scorer, runtime gate and prediction format.

The recovery verifier re-scores each downloaded prediction against the original local annotations and checks the checkpoint identity. Subsets remain excluded from full Table 1 reporting. Full benchmark evidence must pass the existing evaluator's manifest validation before import.

To reuse earlier full clean measurements:

    .venv/Scripts/python scripts/evaluate_table1.py --manifest configs/table1_unified.json --output results/<fresh-directory>
    .venv/Scripts/python scripts/kaggle/replay_clean.py --verification results/<fresh-directory>/verification.json --output results/<fresh-directory>/fresh_rescore.json

This sequentially re-scores all accepted TuSimple configurations. It preserves the original hardware/timing provenance and avoids repeating model inference. The packager accepts --model, --backbone, --checkpoint and --config for the existing benchmark adapters. It includes the selected upstream Python/configuration files. LaneATT R34 has been executed on Kaggle. Other adapters require their model-specific dependencies in the Kaggle environment. SCNN R34 AutoDrive support from the later milestone is absent from this main-derived branch; the generic LaneDet route cannot establish AutoDrive checkpoint compatibility.


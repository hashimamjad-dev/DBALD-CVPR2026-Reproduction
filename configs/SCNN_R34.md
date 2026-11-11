# Independently reconstructed SCNN ResNet-34

Update, 6 November 2026: public trained R34 checkpoints for both datasets were found in PytorchAutoDrive. They use a different complete architecture/configuration from these LaneDet reconstructions. `configs/scnn_r34_autodrive.json` pins their source commit and checkpoint hashes; `lane_repro.scnn_autodrive` strictly loads the original implementation without partial parameter transfer. The existing benchmark scripts accept `--scnn-implementation autodrive --backbone resnet34`. Full-test measurements for these new checkpoints are still required; checkpoint availability alone does not fill a Table 1 cell.

`scnn_r34_tusimple.py` and `scnn_r34_culane.py` instantiate the pinned public LaneDet SCNN with ResNet-34, as specified in DBALD Appendix C. They reuse the corresponding public dataset preprocessing and heads. TuSimple inherits the public R18 configuration; CULane inherits the public R50 configuration. Training defaults are inherited starting points, **not claimed DBALD settings**. Automatic pretrained downloads are disabled.

The existing `SegmentationAdapter('SCNN', checkpoint, config, device)` and `scripts/benchmark_culane.py` accept these standalone configuration paths and enforce strict checkpoint loading. R18/R50 lane checkpoints must not be relabelled or partially loaded as trained R34 checkpoints. A separately trained R34 checkpoint is needed before reporting R34 accuracy. No random checkpoint is saved by the tests.

Executable architecture validation:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_scnn_r34_configs.py -q
```

Tests build the actual public model, check the R34 residual stage depths, and execute finite segmentation outputs at the configured native network resolution for both datasets. These are random-weight architecture tests, not accuracy experiments.

The full CULane baseline currently running with the existing R50 checkpoint remains separately identified as R50 and not paper-backbone-equivalent. The point-ACC candidate in `scripts/score_culane_points.py` is likewise explicitly unconfirmed; it must not be silently substituted for a uniquely specified author metric.

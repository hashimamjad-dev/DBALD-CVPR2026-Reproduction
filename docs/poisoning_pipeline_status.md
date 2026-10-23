# Poisoning pipeline status (as of 2026-10-23)

This documents the new work on branch `aoun`, built without a GPU, independent of the V1/V2
distillation work on the same branch. It is the first implementation of the attack/evaluation side
of the thesis (Sections 2.9-2.11): trigger placement, label poisoning, and ASR measurement.

## What is real, not placeholder

- **ASR/ACC scoring** (`lane_metrics.py`): `asr_loa_lra`/`dataset_asr_loa_lra` wrap the real,
  vendored official TuSimple evaluator (`third_party/tusimple-benchmark`), not a reimplementation.
  A real bug was found and fixed here: a naive positional-pairing implementation of the paper's
  *displayed* formula scored 43.5% against the official evaluator's 96.7% on identical real
  predictions, because the official evaluator matches each ground-truth lane to its best-scoring
  predicted lane, not by list position. See the module docstring for the full finding.
- **Label poisoning** (`label_attacks.py`): LOA (offset), LDA (erase), LRA (rotate-and-refit) target
  construction, exactly as geometry, independent of any model.
- **Trigger placement** (`trigger_placement.py` + `laneatt_heatmap.py`): real road-region masking
  from lane geometry, real candidate-window search, and a **real gradient attention map** computed
  from an actual LaneATT model's own `loss()`/`match_proposals_with_targets()` (vendored unchanged).
  A real bug was found and fixed here too: the target-tensor builder was missing a filter the
  original TuSimple loader applies, which silently zeroed out all positive anchor matches for
  LOA/LRA. Verified end-to-end against the real downloaded R18 TuSimple checkpoint and real photos.
- **Stealthiness proxies** (`stealthiness.py`): real LPIPS (the same metric DBALD itself uses) and a
  frequency-domain energy ratio. Explicitly **not** a reimplementation of the paper's own forensic
  detectors (UniDetection/LGrad/DIRE), which need pretrained weights this project doesn't have.
- **A tiny-scale training sanity check** (`sanity_poisoned_training.py`): fine-tunes a real LaneATT
  checkpoint on real poisoned images with real backward passes, and measures before/after clean ACC
  and ASR with the real official evaluator. This is a proof of mechanism, not a Table 1 result --
  see its own docstring and the Open questions section below for scale limits.

## What is still a placeholder

- **Trigger realism**: `poison_pipeline.paste_placeholder_trigger` draws a flat-colour rectangle, not
  a diffusion-rendered mud patch or cone. This is the one piece that genuinely needs a GPU (UltraEdit
  or, if the professor accepts it, the SD1.5 proxy Hashim's V1/V2 work already built). Isolated
  behind one function so swapping it in is not a pipeline rewrite.
- **Road segmentation**: `poison_pipeline.road_region_mask` derives a drivable-area proxy from lane
  geometry, not SAM2 (its checkpoint is downloaded at `models/checkpoints/sam2.1_hiera_tiny.pt` but
  not wired in).

## What this proves, concretely

Running `scripts/sanity_poisoned_training.py --train-count 12 --epochs 30` (12 of the 16-image
subset poisoned with LOA, held out 4 for clean accuracy, real gradient-driven trigger placement):
this fine-tunes a real checkpoint on real poisoned images for a few minutes on CPU and reports a
real before/after clean-ACC and ASR pair, computed with the real official evaluator. See
`results/sanity_poisoned_training.json` for the actual numbers from this run (not committed --
regenerate with the command above). At 12 images this is far too small to claim attack efficacy; its
value is confirming every piece of the chain (poisoned image -> real gradients -> real evaluator)
is wired correctly before attempting the same thing at the paper's actual scale (thousands of images,
100 epochs, a real diffusion-rendered trigger) on a GPU.

## What Hashim needs to do next (GPU-only)

1. Swap `paste_placeholder_trigger`'s call site in `poison_pipeline.poison_sample` for a real
   diffusion render (UltraEdit, or the SD1.5 proxy from V1/V2, pending the professor's confirmation
   on which teacher is acceptable).
2. Run `scripts/build_poisoned_dataset.py` against the full TuSimple train split (now downloaded,
   `datasets/raw/tusimple_test` for test, train split still needed separately) at the paper's 10%
   poisoning rate, with `--checkpoint` for the real heatmap.
3. Run a real training loop (not this session's tiny sanity check) on the resulting poisoned split,
   matching the paper's settings (100 epochs, ResNet-34), and measure ASR with
   `lane_metrics.asr_loa_lra`/`dataset_lda_asr` -- the same, already-verified scoring code this
   session used at tiny scale.
4. Extend to ADNet/SCNN/RESA and CULane once LaneATT/TuSimple is confirmed working at scale.

## Open questions still pending the professor

- Is the SD1.5 teacher acceptable, or is UltraEdit required?
- Deadline and required scope (which detectors/datasets/attacks).

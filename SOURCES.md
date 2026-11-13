# SOURCES.md — Assumptions & provenance for unspecified paper details

Purpose: every place the DBALD paper (CVPR 2026 main text + the CVF supplementary)
leaves a detail unspecified, and we had to choose a value, is recorded here with the
source of the choice. This is what justifies our reported numbers to a reviewer: where
the paper is explicit we follow it; where it is silent we adopt the relevant official
repo's default and say so here. Nothing in this file is a measured result.

Last updated: 2026-11-13. Scope locked to **TuSimple only** (all 4 models, 5 attack
methods). CULane is out of scope for this round (training split not downloaded).

Primary documents consulted:
- Main paper: `Liao_Towards_Stealthy_..._CVPR_2026_paper.pdf` (repo root).
- Supplementary: https://openaccess.thecvf.com/content/CVPR2026/supplemental/Liao_Towards_Stealthy_and_CVPR_2026_supplemental.pdf
  (5 pages; read in full 2026-11-13).

---

## 1. Per-model heatmap loss selection (RESOLVED from supplementary, with a documented internal conflict)

The supplementary specifies, per model, which loss supervises classification vs.
regression (Supp. §3 "More details of LD models"), and separately gives a grouped
Table 4. **These two disagree for the segmentation models.** Per the user's directive,
we treat the **per-model prose text as authoritative** and record the conflict here.

Supplementary per-model prose (§3):
- **LaneATT**: focal loss (classification head) + **L1** loss (regression head).
- **ADNet**: focal loss (classification) + **GLIoU** loss (shape regression).
- **SCNN**: **CE** loss = pixel-wise lane-*class* (dense spatial prediction); **BCE** loss = lane *existence*.
- **RESA**: **BCE** loss = pixel-wise foreground/background (dense); **CE** loss = lane *existence* (separate head).

Supplementary Table 4 ("Task-specific Loss Selection"):
- Anchor-based: LDA → Focal; LOA/LRA → L1 (LaneATT) / GLIoU (ADNet).
- Segmentation-based: LDA → CE; LOA/LRA → BCE  *(single grouping for both SCNN and RESA)*.

**The conflict.** The heatmap for an attack uses the gradient of the loss that governs
the quantity the attack manipulates: LDA suppresses lane **existence**; LOA/LRA distort
lane **geometry** (the dense pixel prediction). Applying that to the per-model prose:

| Model | LDA heatmap loss (existence) | LOA/LRA heatmap loss (geometry) | Agrees with Table 4? |
|---|---|---|---|
| LaneATT | Focal | L1 | yes |
| ADNet | Focal | GLIoU | yes |
| **SCNN** | **BCE** (existence) | **CE** (dense pixel-class) | **NO** — Table 4 says LDA→CE, LOA/LRA→BCE, i.e. reversed |
| RESA | CE (existence) | BCE (dense fg/bg) | yes |

So Table 4's "LDA→CE, LOA/LRA→BCE" matches RESA's prose but is **reversed for SCNN**,
because in SCNN CE is the dense/geometry loss and BCE is the existence loss, whereas in
RESA those roles are swapped.

**Decision (what the code will do):** follow the per-model prose + attack-semantics
mapping in the table above. Concretely: SCNN LDA→grad of BCE(existence), SCNN LOA→grad
of CE(pixel-class); RESA LDA→grad of CE(existence), RESA LOA→grad of BCE(fg/bg).
LaneATT/ADNet are unambiguous. If a reviewer prefers Table 4's literal grouping, the
only cell that changes is SCNN, and it is a one-line swap in the heatmap loss selector.

Status in code (2026-11-13): only **LaneATT** heatmap is implemented
(`src/lane_repro/laneatt_heatmap.py`, focal for LDA / L1 for LOA-LRA — matches the
table). ADNet/SCNN/RESA heatmaps are **not yet written**.

## 2. Attack-name inconsistency (documented, no effect on math)

- Main paper: **LRA = "Lane Rotation Attack"** (rotate the curve suffix by angle α).
- Supplementary §4: **LRA = "Lane Reconstruction Attack"**.
We implement the main paper's definition (rotation; `src/lane_repro/label_attacks.py`).
LRA is **not a reported TuSimple column** in Table 1 anyway (see §6), so for this
TuSimple round LRA affects only the LOA/LRA-shared heatmap, not a reported ASR cell.

## 3. Backbone (RESOLVED)

Supplementary §3: "For fair comparison, we adopt **ResNet-34** as the unified backbone
for all models." → All four detectors use ResNet-34. Our earlier R18/R50 clean baselines
are engineering references only and are **not** the Table-1-equivalent backbone. The
step0 smoke run used R18 and is therefore a mechanism check, not a Table 1 configuration.

## 4. Attack parameters (from main paper §5.1 "Implementation")

- Poisoning rate: **10%** (explicit).
- Trigger: **40×40 mask**, mud prompt "Add a small and sparse brown mud" (explicit).
  Supplementary §2.2 confirms 40×40 is the ASR-optimal size (10×10→32.15%, 50×50→70.62%).
- LOA offset **β = 60 px**; LRA rotation **α = 9°** (explicit). Code defaults match
  (`poison_sample(beta=60., alpha_degrees=9.)`).
- Baselines (explicit): BadNets 30×30 white square bottom-right; Blended fixed image at
  a fixed blend ratio; LD-Attack 900-px overlay on the adjacent lane near the boundary;
  BadLane 900 px sampled from a 100×100 region. Implemented in `src/lane_repro/baseline_attacks.py`
  (reimplementations from the one-line paper descriptions, NOT the original authors' code — do not
  call results "the original baselines"). Colours, Blended opacity (α=0.2) and pattern, and the
  LD-Attack adjacency rule are unspecified by the paper and are local choices (each record carries
  `recipe_assumptions`). **Known limitation:** LD-Attack's outward band is clipped when the chosen
  lane sits near the image edge, so the 900-pixel budget is not always met (e.g. a lane at x≈1269 in
  a 1280-wide frame yields ~225 px). This is a consequence of the local adjacency rule; flag it.

## 5. Training hyperparameters — UNSPECIFIED in paper; defaulted to official repos (ASSUMPTION)

The paper and supplementary give **no** epochs / LR / optimizer / batch size / schedule
for the victim models. We default each to the vendored official repo's standard **TuSimple
ResNet-34** recipe, cited below by file. **This is an assumption and a reviewer should be
told so.** Values read from the vendored configs on 2026-11-13:

| Model | Source config (in `third_party/`) | Epochs | Batch | Optimizer / LR | Schedule |
|---|---|---|---|---|---|
| LaneATT | `LaneATT/cfgs/laneatt_tusimple_resnet34.yml` | 100 | 8 | Adam, lr 3e-4 | (per cfg) |
| ADNet | `ADNet/configs/adnet/tusimple/resnet34_tusimple.py` | 150 | 45 | lr 1.2e-3 | CosineAnnealing (T_max=total_iter) |
| SCNN | `pytorch_auto_drive/configs/lane_detection/scnn/resnet34_tusimple.py` | 50 | 20 | (poly, warmup 200) | `ep50_poly_warmup200` |
| RESA | `lanedet/configs/resa/resa34_tusimple.py` | 150 | 8 | SGD, lr 0.025, m 0.9 | poly (pow 0.9) |

Notes / open choices:
- RESA has two vendored recipes: `lanedet/configs/resa/resa34_tusimple.py` (150 ep, bs 8,
  lr 0.025) and the original `resa/configs/tusimple.py` (300 ep, bs 4, lr 0.020,
  total_iter 80000). We default to the **lanedet** one (matches the config style the repo's
  own code references); flag if the professor wants the original RESA repo's 300-epoch run.
- Poisoned models are trained with the **same** recipe as the clean baseline for that
  model (only the training *data* changes: 10% of samples swapped for their poisoned
  counterpart). No separate poisoned-training schedule is defined by the paper.
- Whether to **train from scratch** or **fine-tune the released clean checkpoint** on the
  poisoned set is not stated by the paper. Default assumption: **train from scratch** with
  the official recipe, since Table 1 reports both benign and poisoned models and fine-tuning
  from a clean checkpoint would confound the clean/poisoned ACC comparison. Flag as an
  assumption.

## 6. Table 1 reported-column structure (VERIFIED against main paper Table 1, p.7)

- **CULane**: LDA, LRA, LOA (3 attack columns) + ACC clean(benign)/poisoned. *(out of scope)*
- **TuSimple**: **LDA and LOA only (NO LRA column)** + ACC clean/poisoned.
- Do not assume symmetric coverage. For TuSimple we therefore produce, per model, five
  method-rows × {ACC-benign, ACC-poisoned, ASR-LDA, ASR-LOA, ASR-avg}.
- **Ambiguity:** Table 1 shows a single "ACC poisoned" per method row, but LDA and LOA are
  different poisoned datasets → different trained models. The paper does not say which
  model's clean ACC is reported, or whether it is averaged. Default assumption: report the
  poisoned-model clean ACC **per attack** internally and present the **mean** in the single
  column, disclosing both. Flag.

## 7. Diffusion trigger generation — UNSPECIFIED details (ASSUMPTION) + unimplemented losses

- Generator: **UltraEdit** `BleachNick/SD3_UltraEdit_w_mask` (SD3 InstructPix2Pix + mask),
  confirmed by main paper §4.3 and supplementary §1. Checkpoint revision used in step0:
  `e5beecb69a3ef1399718be2f99b61730b8bf0346`.
- **Diffusion step count: UNSPECIFIED.** UltraEdit's own usage examples use **50 steps**;
  we default to 50 (step0 used 50). The paper's "6 s/image" cost (supp §2.5) may imply
  fewer; to be confirmed by the timing study (see §9). Flag as assumption.
- **Guidance scales: UNSPECIFIED.** Defaulted to UltraEdit example values
  `image_guidance_scale=1.5`, `guidance_scale=7.5`. Flag.
- **L_lane (MSE in lane mask) and L_env (SSIM in surrounding-object mask):** the paper's named
  stealthiness contribution (§4.3; supp §1: applied in image space after latent decode, back-propagated
  through the denoising steps). STATUS (2026-11-13):
  - **Loss math IMPLEMENTED + unit-tested** — `src/lane_repro/diffusion_losses.py`
    (`lane_consistency_loss` = masked MSE; `env_consistency_loss` = 1 − masked SSIM, differentiable;
    `stealth_loss` combines them), tests in `tests/test_diffusion_losses.py` (13 tests).
  - **Scoring WIRED** into the generator — `UltraEditTrigger.compute_stealth_losses(...)` scores any
    decoded edit now (CPU, model-free).
  - **Latent-guided application NOT yet active** — steering the SD3 latent through denoising with these
    losses (the paper's actual use) needs the pipeline internals + a GPU; it is the explicit stub
    `UltraEditTrigger.latent_guided_edit` (raises NotImplementedError). `__call__` still produces
    triggers via vanilla masked diffusion. **Any ASR/stealthiness number produced before
    `latent_guided_edit` is built must be labelled "without L_lane/L_env guidance".**
  - **ASSUMPTIONS, flagged:** (a) **L_env sign** — paper writes "L_env = SSIM" but minimising a loss
    must *maximise* similarity, so the minimised quantity is `1 − SSIM` (documented in the module).
    (b) **Loss weights** `w_lane`,`w_env` — unspecified; default 1.0 each, overridable. (c) **env_mask**
    — paper masks surrounding *objects* (needs SAM2/a detector); we use a documented proxy
    (`env_mask_proxy`: scene minus trigger minus lanes) until object masks exist. (d) **SSIM window**
    11×11 Gaussian σ=1.5 (Wang et al. 2004); masked-SSIM near a boundary mixes a few outside-mask
    pixels (standard windowed approximation).
- **T5 text encoder:** `drop_t5` trades VRAM for prompt fidelity. step0 used the **full**
  T5 (`drop_t5=False`) with CPU offload. The local module default is `drop_t5=True`. Which
  is used materially affects both output and speed; record per run. Flag.
- **SAM2 road segmentation: NOT wired into the poisoning path.** `poison_pipeline.road_region_mask`
  uses a lane-geometry drivable-area proxy; SAM2 exists only as a standalone demo
  (`src/lane_repro/prototype.py::road_sam2`). The paper uses SAM2 (§4.1). Flag as a deviation
  until integrated (checkpoint is present: `models/checkpoints/sam2.1_hiera_tiny.pt`).

## 8. Poisoning split (from supplementary + repo convention)

- TuSimple train = 3,626 labelled frames. 10% poison ⇒ **363 poisoned / 3,263 clean**
  (matches the deterministic 3263/363 split pinned on `origin/main`). Supp §2.3 gives the
  analogous CULane figure "1% ≈ 177 samples", confirming poisoning is a fraction of the
  train split.

## 9. Computation-time claim and our measured gap (see also the timing study)

- Supp §2.5: DBALD ≈ **6 s/image** (vs ~1 s for overlay baselines). Supp §2.3 separately
  says a 1% CULane set (177 samples) takes ≈3 min ≈ **1 s/image** — the supplementary is
  internally loose here; we treat **6 s/image** as the generation-cost reference.
- Our step0 (Kaggle **Tesla T4**) measured **≈47 s/image** for UltraEdit rendering.
  Root cause is configuration/hardware, not a logic bug: `cpu_offload=True`
  (`enable_model_cpu_offload`), `drop_t5=False` (full T5-XXL resident on CPU), attention
  slicing disabled, 50 steps, one image at a time, on a T4. See `scripts/kaggle/step0_worker.py`
  lines 51-56. The paper's 6 s almost certainly assumes an A100/A6000-class GPU with the
  model resident in VRAM (no offload). To be confirmed by a one-off timing measurement
  before any scaled generation run.
- **Timing study (staged):** `scripts/probe_diffusion_timing.py` times the identical generation
  path across these exact knobs (paper_target = no-T5/resident/no-offload; step0_baseline =
  full-T5/offload, reproducing the ~47 s; plus offload-isolation, slicing, and fewer-steps
  variants) and reports median s/image + peak VRAM per config vs the 6 s reference. It runs nothing
  without `--run --snapshot <local path>` on a GPU (no download, no Kaggle), so the gap is resolved
  by measurement, not assumption, before scaling. This is the "one-off timing measurement" above.

---

## 10. LaneATT training harness — how the pinned split is imposed (wiring choice)

`src/lane_repro/laneatt_harness.py` binds each of the 10 TuSimple cells (5 methods × {LDA, LOA}) to
a LaneATT ResNet-34 config without patching LaneATT. LaneATT's TuSimple loader maps split names to
fixed annotation files; we therefore write our pinned split into those files inside each poisoned
mix directory: **all 3,263 training frames → `label_data_0313.json`** (poisoned subset already
relabelled), `label_data_0601.json` left empty, **363 validation frames → `label_data_0531.json`**,
and set the train block to `split: train` so validation frames never enter training. The cosine
`T_max` is recomputed from our train size / batch (3263 / 8). Backbone stays ResNet-34 (supp. §3);
epochs default to the official R34 recipe (SOURCES.md §5). This is wiring only — no verified training
driver has been run. The DBALD cell's poisoned mix additionally needs the UltraEdit render (GPU);
baseline cells build on CPU. Plan emitter: `scripts/plan_laneatt_cells.py` (runs nothing).

### How to cite this file in the thesis/report
"Where the paper specifies a value we used it (poisoning rate, trigger size, LOA/LRA
parameters, backbone, per-model heatmap losses). Where it is silent (victim training
schedule, diffusion step count and guidance, L_lane/L_env weights) we adopted the relevant
official repository's default, documented in SOURCES.md, and we flag the SCNN heatmap-loss
conflict between the supplementary's prose and its Table 4."

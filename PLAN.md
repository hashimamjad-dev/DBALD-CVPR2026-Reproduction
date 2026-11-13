# PLAN.md — DBALD Table 1 reproduction (LaneATT / TuSimple row)

**Scope is LOCKED (2026-11-13). No scope changes without an explicit new instruction.**

Reproduce the **LaneATT row of Table 1 on TuSimple**: all 5 attack methods
(BadNets, Blended, LD-Attack, BadLANE, DBALD) × both reported TuSimple attacks
(**LDA, LOA** — TuSimple has no LRA column; main paper p.7). CULane is out of scope
(train split not downloaded). Backbone ResNet-34 (supplementary §3). Every unspecified
paper detail and its chosen default is traced in **[SOURCES.md](SOURCES.md)**.

## Operational constraints (read before any GPU/Kaggle action)

### Kaggle — do NOT touch until the user says "go"
- As of 2026-11-13: **1 Kaggle account is being set up and is nearly live; the 2nd account's API
  key arrives the next day.** The user is configuring access right now.
- **No Kaggle API call and no kernel push until the user explicitly confirms the account is live.**
- Credentials: the classic `kaggle.json` (`{"username","key"}` from kaggle.com → Settings → API →
  Create New Token) is required for scripted pushes. The existing `~/.kaggle/credentials.json` is an
  OAuth token file (`refresh_token`/`access_token`), NOT the API token, so it is not sufficient.
  Account 1 → `C:\Users\Hashim Amjad\.kaggle\kaggle.json`. Account 2 → a separate dir
  (`C:\Users\Hashim Amjad\.kaggle-acct2\kaggle.json`) selected per run via `KAGGLE_CONFIG_DIR`.
- Sequence work so the Kaggle-dependent steps (DBALD UltraEdit generation, GPU training) are ready
  to fire the moment the user confirms — but stage them, do not launch.

### Local GPU (RTX 5050, 8 GB) — do NOT use without explicit per-use approval
- Never run anything on the local GPU automatically, including baseline training.
- If it looks genuinely useful (e.g. LaneATT baseline training at reduced batch size while Kaggle
  handles DBALD), check feasibility and **prompt the user, then wait for an explicit go-ahead**.
- If not useful, drop it silently — don't raise it again unless something changes.

## The 10-cell matrix (LaneATT × TuSimple)

| Method | LDA | LOA | Trigger generation |
|---|---|---|---|
| BadNets | cell | cell | CPU overlay (30×30 bottom-right) |
| Blended | cell | cell | CPU blend (α=0.2 fixed pattern) |
| LD-Attack | cell | cell | CPU overlay (adjacent lane near edge) |
| BadLANE | cell | cell | CPU mud-texture (900 px / 100×100) |
| DBALD | cell | cell | **UltraEdit diffusion (GPU)** + gradient heatmap placement |

Benign (clean-ACC) column reuses the released R34 clean checkpoint per the protocol note in SOURCES.md.
Cell enumeration, build commands and config binding: `src/lane_repro/laneatt_harness.py`;
review with `python scripts/plan_laneatt_cells.py --output <file>`.

## Build status

Done and unit-tested on CPU (no GPU/Kaggle used):
- **L_lane / L_env losses** — `src/lane_repro/diffusion_losses.py` (masked MSE; differentiable
  SSIM / 1−SSIM; combined), scoring wired into `UltraEditTrigger.compute_stealth_losses`. The
  paper's latent-GUIDED application is the explicit stub `UltraEditTrigger.latent_guided_edit`
  (GPU-only). Numbers before that is built are "without L_lane/L_env guidance".
- **4 baseline trigger generators** — `src/lane_repro/baseline_attacks.py` (reimplementations;
  verified to run on real subset images; assumptions in SOURCES.md §4).
- **LaneATT training harness** — `src/lane_repro/laneatt_harness.py` (10-cell matrix, R34 config
  binding, pinned-split label layout). Wiring only — no training run.
- **Pinned split** — `src/lane_repro/tusimple_split.py` + `results/table1_split_20261111/tusimple_split.json`
  (3263 train / 363 val, deterministic).
- **Poisoned-mix builder** — `scripts/mix_poisoned_dataset.py` (split-aware, DBALD + baselines).

GPU/Kaggle-gated (staged, not run):
- DBALD UltraEdit generation (needs the 31 GB model + a GPU; resolve the 6 s vs 47 s/image timing —
  SOURCES.md §9 — before scaling).
- The 10 poisoned LaneATT R34 trainings + full-test ACC/ASR inference + independent re-score.
- Integrating `latent_guided_edit` (L_lane/L_env through denoising).
- SAM2 road segmentation into the poisoning path (currently a lane-geometry proxy).

## Next actions
1. **Now (CPU, no approval needed):** keep extending/validating the harness and losses; keep
   SOURCES.md current.
2. **On Kaggle "go":** timing probe (confirm achievable s/image), then build the DBALD poisoned
   mixes and run the LaneATT cells, LaneATT first.
3. **Before any local-GPU use:** prompt the user with a specific feasibility case and wait.

## Compute-budget reference (TuSimple LaneATT row only)
- Diffusion generation (DBALD cells): ~363 poisoned imgs × 2 heatmaps, a few GPU-hours once timing
  is fixed; baselines need none.
- Training: 10 poisoned R34 trainings (~3 h each on a modern GPU) ≈ ~30 GPU-hours, plus inference
  and re-score. University GPUs are the intended home; Kaggle is overflow.

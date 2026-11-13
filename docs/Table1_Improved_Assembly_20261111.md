# Table 1: non-GPU completion milestone (11 November 2026)

Branch `milestone/table1-tusimple-split-20261111`. No GPU was used. Paused V1/V2/V3 work is untouched.

## What was added

| Item | File |
|---|---|
| Pinned split consumed by the poisoning CLI (`--split-file/--split-part`; validation frames never poisoned or trained on) | `src/lane_repro/tusimple_split.py`, `scripts/mix_poisoned_dataset.py` |
| Baseline trigger recipes: BadNets, Blended, LD-Attack, BadLANE (`--method`) | `src/lane_repro/baseline_attacks.py` |
| Improved per-cell Table 1 view, validated receipt import, derived averages only for complete series, published-average audit | `scripts/assemble_table1.py` |
| Reviewable 100-training experiment plan (runs nothing) | `scripts/plan_table1_gpu_runs.py` |
| Tiny CPU sanity receipt (previously absent) | `results/table1_improved_20261111/sanity_poisoned_training_cpu.json` |

Baseline recipes use the paper's stated sizes (BadNets/Blended 30x30 bottom-right; LD-Attack and BadLANE 900 pixels, BadLANE from a 100x100 region). The paper gives no colours, Blended opacity or pattern, or adjacency rule, so these are local choices recorded in every label record (`recipe_assumptions`). They are reimplementations, not the authors' code. All four ran on real pinned-split TuSimple frames.

## Table 1 status (unchanged by this milestone, honestly)

`results/table1_improved_20261111/table1_improved.md`: 15 verified independent-baseline cells, 25 non-comparable clean cells, 180 unmeasured attack-related cells. No attack number was invented. Poisoned ACC and ASR need trained models, which need a GPU.

Improvement found: the published averages were audited. 39 of 40 equal the mean of the published ASRs; **LaneATT/CULane/DBALD** prints 76.41 while its ASRs average 76.43 (diff -0.02 pp). Reported, not altered.

## Sanity receipt (mechanism check, not a Table 1 value)

LaneATT R18, 12 TuSimple subset images poisoned with LOA, 4 held out, placeholder trigger, 30 epochs, CPU: held-out clean ACC 0.9665 -> 0.5673; poisoned-image ASR 0.4726 -> 0.9582. The ASR rise confirms the chain works; the large clean-ACC drop shows tiny-scale overfitting, so it says nothing about attack efficacy at paper scale.

## Remaining Table 1 gaps

- 100 scenario trainings (plan in `remaining_experiment_plan.json`) need GPU time. **Approval is required before starting.** No verified R34 poisoned-training driver exists yet.
- DBALD trigger realism needs UltraEdit (GPU); the placeholder trigger is not paper-faithful.
- CULane: train/val data absent (Drive quota), poisoning pipeline is TuSimple-format only, and the author's ACC protocol is unresolved.
- P ACC aggregation across scenarios is unspecified by the paper.
- Receipts produced later plug into `assemble_table1.py --receipts`; it rejects partial, un-rescored or out-of-range ones.

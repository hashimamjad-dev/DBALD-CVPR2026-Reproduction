# Master work log

Single running record of the whole project. Append new entries at the bottom; never delete earlier ones. Detail lives in the linked documents; this file is the index plus the change history.

## Scope and rules in force
- Active work: the original DBALD Table 1 reproduction (AGENTS.md). Remote destination `hashim-dev`; no force-push; no merge to `main` without approval.
- Ask the user before any extensive GPU work. No GPU was used for the entries below.
- Published Table 1 values are reference only and are never copied into a measured column.

## Where the evidence lives
| Topic | File |
|---|---|
| Published Table 1 (reference) | `docs/table1.md`, `results/reference/table1.csv` |
| Paper analysis and stated settings | `docs/paper_analysis.md` |
| Clean-model evidence (TuSimple full test, CULane full test) | `docs/Table1_Clean_Reproduction_20261106.md`, `results/experiment_summary.md` |
| Protocol, SCNN R34 and CPU readiness | `docs/Table1_Protocol_SCNN_Readiness_20261106.md` |
| Poisoning pipeline status (real vs placeholder) | `docs/poisoning_pipeline_status.md` |
| Pinned TuSimple split (3263 train / 363 val) | `results/table1_split_20261111/tusimple_split.json` |
| Improved Table 1 view and remaining plan | `docs/Table1_Improved_Assembly_20261111.md`, `results/table1_improved_20261111/` |
| Distillation research (paused V1/V2) | `docs/V1_V2_Research_Comparison.md`, `docs/distillation_v2_diagnosis.md` |
| Thesis research table with formulas | `results/thesis_research_table_20261111/thesis_research_table.md` |

## Current Table 1 standing
15 verified independent-baseline cells, 25 non-comparable clean cells, 180 attack-related cells unmeasured (poisoned ACC, ASR, averages). Nothing here was estimated.

## Change history
- **2026-11-11, commit `d5ac46781`:** pinned a deterministic TuSimple train/validation split (3263/363).
- **2026-11-11, commit `685eb1a7e` (pushed to `hashim-dev`):** split-aware dataset builder (`--split-file/--split-part`); baseline trigger recipes for BadNets, Blended, LD-Attack and BadLANE (`src/lane_repro/baseline_attacks.py`, local reimplementations with recorded assumptions); `scripts/assemble_table1.py` (per-cell published/local/delta view, validated receipt import, derived averages only for complete series, published-average audit); `scripts/plan_table1_gpu_runs.py` (100-training plan, runs nothing); tiny CPU sanity receipt. Audit finding: LaneATT/CULane/DBALD published average 76.41 vs mean of its ASRs 76.43.
- **2026-11-11 (uncommitted):** `scripts/build_thesis_research_table.py` and its output (speedup 11.04x, memory -57.8%, LPIPS reduction 46.21% TuSimple / 11.36% CULane; all V2 quality thresholds still failed).
- **2026-11-11 (outside repo):** `E:\thesis_fixed.txt`: corrected DBALD/BadLANE citations (`[20]`/`[41]`), corrected the cost sentence to DBALD's Appendix B.5 (about 6 s vs 1 s per image), "lane diversion" to "lane detection" in the body. Original `E:\thesis.txt` untouched. Duplicate references still to remove in the `.bib`.
- **2026-11-11 (outside repo):** `E:\thesis_chapter4_draft.txt`: Chapter 4 drafted from recorded repo facts; unknown values left as TBD.
- **2026-11-11:** a planned next stage (new generator losses plus attack-success measurement) was stopped by a safety classifier before any file was written. It is not implemented and is not scheduled.

## Open items
- 100 scenario trainings need GPU time and the user's approval.
- CULane train/val data absent (Drive quota); the poisoning pipeline is TuSimple-format only; the authors' CULane ACC protocol is unresolved.
- P ACC aggregation across scenarios is unspecified by the paper.

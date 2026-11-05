# Table 1 reproduction status

The scientific target remains all four architectures, both datasets, clean ACC and attack-specific ASR. The present executable milestone is clean pretrained inference and metric validation; it does not establish reproduction of Table 1.

| Experimental component | Available evidence / remaining requirement |
|---|---|
| Published reference | Exact Table 1 CSV with checksum; TuSimple LRA is not reported |
| Clean models | Author LaneATT archive and ADNet checkpoint; LaneDet SCNN/RESA checkpoints; strict loading required |
| Model equivalence | Extended arXiv v2 Appendix C specifies ResNet-34 for all models. Earlier R18/R50 engineering baselines are not equivalent. Exact commits, checkpoints and seeds remain unresolved; upstream clean R34 weights alone do not establish equivalence |
| Dataset equivalence | Full TuSimple 2,782-image clean evaluations completed. CULane now has complete 34,680-image clean evaluations for all four architectures; earlier quota and partial-data reports are historical. Exact DBALD poisoned training/test subsets unavailable |
| ACC protocol | Official TuSimple implementation available; paper's CULane point-ACC conversion and matching not fully specified |
| ASR model state | No author-released poisoned checkpoints verified; clean models cannot supply equivalent ASR |
| ASR inputs | Exact test images, trigger instances, poisoned subset indices and target annotations unavailable |
| LOA / LRA scoring | Intended target labels, spline/pivot conventions, point matching, invalid-point rules and aggregation require clarification |
| LDA scoring | Disappeared-lane denominator, confidence threshold and matching to originally present lanes require clarification |
| Training equivalence | Optimizer, epochs, learning rate, schedule and repeat seeds not fully resolved; unified ResNet-34 backbone identified in extended Appendix C |
| Generation equivalence | SAM2 prompts/checkpoint; UltraEdit variant, latent update ambiguity, loss weights, masks, seeds and schedule missing |
| Uncertainty | Repeated seeds and per-image outputs unavailable; no variance estimates can be recovered from Table 1 |

This paragraph describes the original brief; subsequent branches added attack-pipeline modules and documented a tiny LaneATT sanity run. Presence of source code is not a verified full-scale result. The numerical receipt referenced by that sanity-run report is absent from this checkout. No full method-matched poisoned-model result is verified here. This milestone executes only clean-model compatibility checks and rescoring. Descriptive analysis and independent benign demonstrations are retained. Evaluating an appropriate author-released research artifact would still require confirming provenance, identical protocol and safe offline scope before making an equivalence claim. No ASR value is inferred from image changes or clean-model mistakes.

Actual measurements and verification receipts are retained under `results/measured/`, `results/overnight/`, the full CULane evidence directories and dated verification directories. `asr: null` means not measured; it never means zero. Small-subset results are diagnostics, not full benchmark results. No arithmetic difference against the paper is evidence of a failed reproduction until dataset, model and evaluation equivalence are established.

Overnight measurements additionally live under `results/overnight/`. The [extended author paper](https://arxiv.org/html/2508.15778v2#A3) includes Appendices Aâ€“E and resolves the backbone question. It does not provide the missing executable checkpoint/training/scoring bundle. The attached CVPR PDF remains the authority for our immutable Table 1 transcription.

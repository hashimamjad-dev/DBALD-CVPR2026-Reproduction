# Security context and evaluation requirements

The thesis Chapter 2 establishes a naturalistic training-data backdoor problem. DBALD is the published baseline: the attacker contaminates training examples, and a compromised detector subsequently responds to a trigger while retaining clean-image performance. This differs from ordinary corruption of a clean model's test input. The present student experiment isolates the generation-efficiency component using ordinary road restoration. It does not test backdoor effectiveness or establish that a distilled generator preserves it.

## Published outcomes and unresolved scoring details

DBALD evaluates lane disappearance (LDA), rotation (LRA), and offset (LOA). Its evaluation section describes LOA/LRA success through point agreement within 20 pixels of target annotations, and LDA through the fraction of disappeared lanes. The text alone does not resolve all matching, invalid-point, denominator and target-construction details needed for an equivalent implementation. The Table 1 CSV preserves the paper's values; TuSimple LRA is not reported.

Four quantities must remain separate:

| Quantity | Required model/input | Current status |
|---|---|---|
| Locally measured clean ACC | Public clean checkpoint, original labeled clean data | Full TuSimple completed; CULane official lane F1 only on an available subset |
| Published poisoned-model ACC | Paper's compromised detector on clean data | Reference transcription only |
| Published ASR | Paper's compromised detector on corresponding triggered examples, target-specific scoring | Reference transcription only; no local ASR |
| Benign edit consistency | Frozen clean detector on original/restored pairs | Separate small prototype assessment; cannot be relabeled ASR |

Equivalent ASR reproduction would require a provenance-verified research artifact, exact compromised model state, paired evaluation examples, original/target annotations, inclusion rules, lane matching and failure handling, and repeated-seed results. These are not supplied by our clean checkpoints. Attack implantation, poison-label modification, trigger optimization and targeted lane-output manipulation are explicitly excluded from the implementation.

## Defensive assessment

The extended author paper reports clean-data fine-tuning, neuron pruning and vision-language anomaly screening. It observes a trade-off between mitigation and clean performance for pruning. These are published observations, not local defense results. No local defense effectiveness is inferred from our clean-model restoration experiment.

A defensible assessment should report clean validation performance alongside detection/mitigation outcomes, select thresholds on a disjoint calibration set, report false alarms on ordinary road objects and unfamiliar weather, and preserve a held-out evaluation set. Anomaly detection needs labeled benign and anomalous examples; visual realism ratings cannot substitute for detector precision/recall. Pruning/fine-tuning a clean checkpoint alone cannot demonstrate removal of a backdoor. Cross-camera or temporal consistency can be investigated as defensive signals, but those hypotheses remain untested here.

## Sources and boundaries

- [DBALD extended author paper: evaluation and countermeasures](https://arxiv.org/html/2508.15778v2).
- Attached thesis Chapter 2, especially Sections 2.9–2.11; no Chapter 3 used.
- `results/reference/table1.csv` is the immutable attached-PDF transcription. It takes precedence over secondary prose or numerical summaries.

All research files stay local. No generated restoration enters any detector training dataset, and every local ASR field remains null (unmeasured, not zero).

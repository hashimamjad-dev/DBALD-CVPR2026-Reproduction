# Publication provenance and reproduction

These are sanitized publication snapshots of completed research, not new experiments.
V1 source commit: e22896b. V2 source commit: 14adcdf.
Their original local ancestry remains intact and is deliberately not uploaded because
it contains an unpublished thesis extract. That extract is excluded; document references
revealing private source details are replaced with the authorized general research objective.
Research Python code is unchanged from its respective source milestone.

The base is the existing private remote clean-model milestone 563341f.
No existing remote history is rewritten. Published snapshot commit IDs differ from
the local source commits; milestone tags, if created, identify sanitized snapshots.

Downloaded datasets, checkpoints, environments, credentials and private thesis sources
are excluded. Paths, dataset selection manifests, checkpoint hashes, configurations,
measurements and evaluation scripts are retained. Download public dependencies using
the existing fetch scripts and consult package_compatibility.md. Obtain authorized data
locally; retain the checkpoint identities recorded in each experiment directory.
Exact numerical replay also requires the omitted checkpoint/cache artifacts retained
on the research workstation. The repository alone is not a complete binary archive.

V1 entry points: prepare_distillation_data.py, distill_road.py, evaluate_edit_lanes.py,
audit_distillation_outputs.py, report_distillation.py.
V2 entry points: v2_prepare.py, v2_train.py, v2_evaluate.py, v2_lanes.py,
v2_audit.py, v2_report.py. The completed protocols describe order and exclusions.
These commands document reproduction; no training was performed during publication.


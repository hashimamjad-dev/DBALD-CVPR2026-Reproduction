# PR #7 merge resolution, 2026-11-05

Scope: merge main bf613bcec61762109f7c5446f070cc205bb002be into milestone/table1-unified-evidence, starting at 1e181a58280a1a63acd807503770b14b52a5baeb. Do not merge the PR into main. No research experiments or training were launched.

## Code and configuration review

- `evaluate_table1.py`: main's checkpoint, provenance, coverage, metric and published-reference checks are present unchanged in the milestone implementation. Preserve the milestone's cached/external imports, derived averages, inventory and page-7 exports.
- `render_table1.py`: retain milestone `export_page7()` and the existing cached renderer. Published reference rows and independent results remain separately labeled.
- `laneatt_heatmap.py`: preserve the known no-match scalar-zero fallback; reject other nondifferentiable losses. Differentiable losses still use autograd, whose errors are not caught or hidden.
- `trigger_placement.py`: restore main's placement functions. The milestone accidentally contained a compositing CLI at this module path. The complete, package-import-compatible version of that CLI already exists at `scripts/check_trigger_compositing.py`, which remains intact. Both functionalities are import-compatible.
- `ultraedit_trigger.py`: retain milestone mask-composited paste-back. Main has no complementary functional change beyond the reviewed differences. A drifting fake pipeline verifies outside-mask pixel identity without loading a diffusion model.
- Downloader, SCNN probe and recovery-test conflicts: line endings only; no functional change.
- `configs/table1_evidence.json`: independently parsed both parents; schema version, record ordering and every field of all seven records are identical. Preserve the milestone configuration, without adding numerical claims.

## Evidence preservation

All 57 conflicted paths have both original versions in `results/pr7_merge_20261105/conflict_originals.zip`. The resolution manifest records SHA256 checksums for both sides. Auto-merged changes to existing evidence also have both originals in that archive. No tracked path was deleted relative to either parent.

Forty-one historical evidence conflicts differ only in line endings. Retain main's exact bytes where existing pinned receipts require them; preserve milestone originals in the archive. Explicit attributes prevent newline conversion of these reviewed evidence files. Historical test traceback whitespace is retained verbatim. This does not certify every pre-existing artifact's portability to a fresh checkout on another operating system.

For substantive reports, retain the milestone's more complete evaluator documentation and completed RESA candidate-point result rather than main's older pending status; both reports remain archived. Paused distillation documents retain their milestone contents, with main's variants archived. The TuSimple train integrity logs contain identical record multisets in different concurrent-completion order; retain milestone order and archive both originals. Main-only files, including paused-work artifacts, are inherited without editing or executing that work.

The original PR comparison contains 139,775 paths: 138,755 under full CULane results, including 138,724 text files (per-image lane exports plus scorer outputs). These constitute experimental evidence and are retained. Another 519 paths have temporary-looking directory names; inspected examples contain synthetic test fixtures and generated test exports. Recommend a separate cleanup review, with test receipts preserved and synthetic fixtures distinguished from measurements. Nothing was deleted. Only this merge's new temporary fixture directory is ignored.

## Validation

- Imports: evaluator, renderer, evidence importer, placement functions, heatmap and UltraEdit modules, plus the compositing CLI package import.
- All eight JSON configurations parse; all seven direct evidence records have annotation, checkpoint, prediction, configuration, provenance and summary paths.
- Existing suite: 197 passed, 1 skipped, including 34 Table 1 tests. The skip requires unavailable real prediction outputs. Explicitly exclude `tests/test_sanity_poisoned_training.py`: it launches real fine-tuning and overwrites a research result, contrary to this session's no-training instruction. No tests were weakened.
- Initial suite: 192 passed, 1 skipped, 5 failed. Four failures resulted from placing temporary outputs outside the renderer/evaluator's required `results/` directory. One was a positional-argument mistake in the new fake-pipeline fixture. Fix the fixture and use the documented results temporary directory; retain both test receipts.
- Direct clean replay: seven verified records, zero rejections, all 220 statuses exported.
- Unified replay: 13 verified supporting records, zero rejections, all 220 statuses exported. Initial replay rejected an LF/CRLF hash mismatch; retaining the exact existing pinned bytes resolved it without changing measurements or weakening checks.
- No unresolved index entries, conflict markers or accidental path deletions. `git diff --check` and staged whitespace checks pass with historical evidence bytes preserved.
- Existing torchvision deprecation warnings remain. Replay outputs are rescoring/verification receipts, not new scientific experiments.

## Research status

Seven full TuSimple clean-run records cover all four architectures and were independently replayed. Existing full CULane results cover all four architectures; preserved independent verification receipts include LaneATT, ADNet, SCNN and RESA. CULane lane F1 and candidate point accuracy remain distinct from confirmed paper-equivalent ACC. SCNN's available CULane result uses ResNet-50; the paper requires ResNet-34.

Main documents a tiny LaneATT/TuSimple poisoned-training sanity run, but its referenced `results/sanity_poisoned_training.json` is absent here. No independently verified numerical poisoned-model ACC/ASR receipt from that run can be reported. Scoring, target geometry and mocked pipeline tests do not establish a completed attack experiment. No new poisoned-model evaluation, training, ACC or ASR measurement was produced during this merge session.

Table 1 remains incomplete: validate full-scale poisoned-model clean ACC and scenario ASR for all five methods, four detectors and both datasets; preserve reproducible checkpoints, predictions and provenance; resolve CULane paper-compatible scoring and the missing SCNN R34 checkpoint. Published values remain immutable references, never substitutes for local measurements.

# Experimental readiness for page 7

Decision: hold full execution. The machine-readable matrix in final/requirements.csv covers 40 detector/dataset/method combinations and every displayed metric. PARTIAL means supporting assets exist but paper comparability or a full R34 measurement remains open. BLOCKED means a prerequisite for the method-specific result is missing. None of the 40 combinations has a complete independently measured series.

## Data and weights

The split inventory checks every listed path. TuSimple has 3,626 training and 2,782 test frames with consistent lane-vector lengths. A training/validation partition still needs to be pinned. CULane has all 34,680 test images and line annotations. Its 88,880 training and 9,675 validation paths resolve to zero images, line files or segmentation masks. File and vector checks do not certify pixel or annotation semantics.

The inventory fingerprints the eight public ResNet-34 weights against existing evidence or the reviewed AutoDrive manifest. LaneATT requires the pinned anchor order and NMS adapter. ADNet uses a compatibility layer for its original MMCV stack. Original CUDA extension equivalence remains open. RESA uses the LaneDet segmentation config. SCNN uses the separate AutoDrive preprocessing and native decoder. The benchmark drivers emit TuSimple predictions for the official evaluator and per-image lane text for CULane. The six non-SCNN model/dataset pairs have earlier full-split R34 evidence. SCNN R18 TuSimple and R50 CULane runs are useful supporting evidence only. Both public R34 SCNN drivers passed a fresh CPU load, real-image forward, export and scorer invocation. The TuSimple CPU sample exceeded its timing gate.

## Method prerequisites

BadNets, Blended, LD-Attack and BadLANE have reference rows and evidence-import support. No validated method-specific recipe, trained model and independent scoring receipt was established for any of the four. Shared label/metric code does not establish any individual baseline. DBALD has executable label generation, placement, trigger, mixing and tiny LaneATT sanity-training components. The existing 3,626-frame TuSimple LOA manifest explicitly identifies a placeholder trigger and R18 attention checkpoint. Its output cannot substantiate any Table 1 attack cell. No complete R34 training/evaluation chain for the eight dataset/model pairs was verified. External evidence imports require provenance; published values are displayed separately. Averages require the complete constituent series.

## Resources and next steps

Existing CULane clean runs measured roughly 103–159 MiB of peak PyTorch CUDA allocations for RESA, ADNet, LaneATT and the legacy SCNN at batch one. Drivers, libraries, other applications and training activations add memory. Historical median forward latencies imply approximately 17, 29 and 24 minutes of inference alone for one clean CULane pass with LaneATT, ADNet and RESA respectively. These are extrapolations excluding decoding, scoring and thermal variation. There is no established R34 SCNN GPU or training memory estimate. Current one-image R34 CPU children stayed below 1.2 GiB private memory. Use fresh single-thread processes and the existing 2 GiB process cap on Windows. Neither successful smoke test establishes sustained memory stability.

1. Preserve the 11 clean and 15 unified supporting records (overlapping sets). Reuse saved predictions and receipts.
2. Obtain the missing CULane train/val data and masks, verify their splits, and pin the TuSimple validation partition.
3. Confirm the authors' CULane point sampling, matching, 20-pixel boundary, error geometry and runtime rules. Re-score existing predictions once confirmed. Official IoU/F1 and candidate point ACC are separate diagnostics.
4. Establish baseline-specific configurations, checkpoint lineage and independent held-out scoring fixtures. Resolve extension parity where it affects comparison.
5. Only after approval, profile R34 SCNN on GPU and schedule its missing clean measurements. Budget method training separately after measuring a representative step. Preserve historical baselines and avoid repeating full inference for scoring-only changes.

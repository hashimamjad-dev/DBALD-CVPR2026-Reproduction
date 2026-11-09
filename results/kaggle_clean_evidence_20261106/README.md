# Completed clean computations

Kaggle notebook hashimamjad03/dbald-research version 1 completed LaneATT ResNet-34 inference on two TuSimple images. The existing benchmark driver, original checkpoint and official scorer were used. The checkpoint SHA-256 is ab6f603c4a986f257efb44d416fc20254c0726b477cff98e577ea75227c8c795.

The two-image check scored 94.419642857%, median 44.292652 ms, p95 44.527790 ms, and 193482240 bytes of peak PyTorch CUDA allocation. Both images passed the 200 ms runtime gate. These subset measurements validate remote execution and recovery. Full Table 1 reporting excludes this subset.

Seven existing full TuSimple clean evaluations, each containing 2782 unique predictions, were freshly re-scored locally with the official evaluator and matched accepted values within 1e-9 percentage points. See fresh_rescore.json and clean_comparison.csv. Their original GPU inference and timing provenance remains in their referenced evidence directories. No repeated full inference was performed.

The existing Table 1 evaluator accepted 13 records and rejected none. table1.html and table1.md retain published references separately from independent measurements and missing method-specific values. CULane F1 and unconfirmed candidate point accuracy remain supporting measurements.

Test evidence: tests.xml records 22 passes and four failures caused by selecting a temporary directory outside results. tests_corrected_temp.xml records all four affected tests passing after correcting the directory. No test expectations or scorer code changed.

The executed notebook and metadata are preserved in submitted/. Remote predictions, model provenance, timings and logs are preserved in remote/. recovery_verification_final.json verifies the transferred asset hashes and independently recomputes the remote score. The reusable notebook subsequently gained optional segmentation --config forwarding; that branch was syntax-checked but has not run remotely.

Remaining limits:
- No full SCNN ResNet-34 result exists in the accepted manifest. The active branch's benchmark has the LaneDet adapter; the separate AutoDrive R34 adapter lives in the later accepted milestone. They require explicit integration before that checkpoint can be run here.
- ADNet and RESA remote dependency stacks have not been exercised on Kaggle. Their existing full-split local measurements were reused and freshly scored.
- All method-specific poisoned ACC and ASR remain unmeasured.
- The complete UltraEdit checkpoint remains unavailable in this execution bundle. No diffusion test was launched.
- Kaggle's checkpoint and image inputs are held in a private dataset. Only small evidence files were downloaded into the repository.

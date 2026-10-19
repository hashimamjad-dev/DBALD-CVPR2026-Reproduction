# Professor briefing

## Research objective

The project investigates which results in DBALD Table 1 can be independently supported. A working detector is the first requirement; reproducing an attack result additionally requires the same poisoned model, target definition and evaluation protocol. We do not infer attack success from ordinary detection errors.

## Architecture differences

LaneATT refines candidate lane anchors and uses attention to combine their information. ADNet adapts anchors and samples features with deformable convolution. SCNN predicts lane segmentation and passes information across image rows and columns. RESA also predicts segmentation, but aggregates context through recurrent feature shifts. Their outputs are converted to native image coordinates before dataset-specific scoring.

The initial Windows baselines use ResNet-18. The author's extended Appendix C specifies ResNet-34 for every model. Matching that backbone is necessary but insufficient: public weights can still differ from the authors' training runs. The preserved baseline results are therefore not presented as exact reproduction.

## Verified measurements

| Dataset / scope | Model / backbone | Images | Metric | Value (%) | Median ms | Peak MiB |
|---|---|---:|---|---:|---:|---:|
| CULane / subset | ADNet / resnet18 | 16 | lane F1 | 60.7595 | 20.02 | 107.82 |
| CULane / subset | ADNet / resnet34 | 16 | lane F1 | 67.5000 | 26.39 | 129.67 |
| CULane / subset | RESA / resnet34 | 16 | lane F1 | 53.4884 | 24.29 | 102.52 |
| CULane / subset | SCNN / resnet50 | 16 | lane F1 | 56.1798 | 88.17 | 123.00 |
| CULane / subset | LaneATT / resnet34 | 16 | lane F1 | 61.5385 | 27.70 | 159.28 |
| TuSimple / full_test | ADNet / resnet18 | 2782 | ACC | 96.2343 | 19.07 | 113.09 |
| TuSimple / full_test | RESA / resnet18 | 2782 | ACC | 96.6959 | 46.74 | 183.04 |
| TuSimple / full_test | SCNN / resnet18 | 2782 | ACC | 96.0843 | 79.63 | 175.34 |
| TuSimple / full_test | ADNet / resnet34 | 2782 | ACC | 96.5982 | 24.97 | 152.57 |
| TuSimple / full_test | RESA / resnet34 | 2782 | ACC | 96.8189 | 59.06 | 222.30 |
| TuSimple / full_test | LaneATT / resnet34 | 2782 | ACC | 95.6348 | 28.34 | 183.52 |
| TuSimple / full_test | LaneATT / resnet18 | 2782 | ACC | 95.5710 | 17.57 | 142.98 |
| TuSimple / subset | ADNet / see config | 16 | ACC | 95.6938 | 19.74 | 113.09 |
| TuSimple / subset | RESA / see config | 16 | ACC | 96.4379 | 45.81 | 183.04 |
| TuSimple / subset | SCNN / see config | 16 | ACC | 96.8750 | 84.96 | 175.34 |
| TuSimple / subset | LaneATT / resnet18 | 16 | ACC | 96.7262 | 21.08 | 142.98 |

Tests: 23 tests, 0 failures, 0 errors, 0 skipped. All raw predictions and measured runtime/memory records are retained locally. Only full_test rows represent a complete benchmark; subset rows serve debugging and robustness checks.

## Data and hardware

TuSimple evaluates labelled 1280×720 final frames using its official accuracy scorer. CULane evaluates 1640×590 images using matched-lane IoU/F1. These scores are not interchangeable. Download integrity, image counts, label hashes and checkpoint hashes are recorded. CULane comes with non-commercial and redistribution restrictions.

The machine has an RTX 5050 Laptop GPU with about 8 GB VRAM. PyTorch with CUDA 12.8 supports its Blackwell architecture. A separate CUDA environment on C: reuses dependencies from the project environment on D:. Benchmarks use batch size one; real batch-size-two validation checks output consistency. Logged peak allocations measure inference, not training capacity. Temperature and storage guards protect long runs.

## Implementation decisions and research gaps

Windows compatibility required portable lane suppression, direct configuration loading and a torchvision deformable-convolution adapter. The original CULane C++ metric is compiled locally with modern OpenCV and without its optional GUI. These changes are documented and tested, but complete equivalence to the authors' original software stack remains open.

Prompted SAM2 masks, clean sensitivity maps, candidate-window statistics and fixed-mask image editing are isolated methodology demonstrations. Ordinary diffusion demonstration completed; see results/overnight/diffusion/measurements.json. Image similarity is measured after editing; it is not optimized for attack evasion. No model backdoors or poisoned labels are constructed.

No Table 1 ASR, poisoned-model clean ACC, or exact paper reproduction is claimed. Priorities are author artifact/protocol clarification, completing available clean benchmarks, and broader controlled robustness studies. All code, data and results remain local; nothing has been pushed or published.

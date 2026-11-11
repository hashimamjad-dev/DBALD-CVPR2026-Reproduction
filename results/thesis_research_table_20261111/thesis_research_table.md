# Thesis research table: distilled generator vs SD1.5 teacher

Offline road-restoration generator; teacher = local SD1.5 inpainting proxy (not DBALD UltraEdit). No poisoning, no ASR. Raw sources: `results/distillation/v2/`. Quality thresholds were NOT met (see the V1/V2 comparison); the speed result must not be read as acceptable realism.

## A. Generation efficiency (common timing protocol: 4 scenes, seed 17, 2 repeats)

| Method | Steps | Median s/image | Peak MiB | Speedup | Time reduction % | Memory reduction % |
|---|---:|---:|---:|---:|---:|---:|
| teacher20 | 20 | 3.6001 | 3256.3 | 1.00x | 0.0 | 0.0 |
| teacher4 | 4 | 0.9658 | 3256.3 | 3.73x | 73.2 | 0.0 |
| student_v1 | 4 | 0.3228 | 1373.4 | 11.15x | 91.0 | 57.8 |
| mask_diverse | 4 | 0.3261 | 1373.4 | 11.04x | 90.9 | 57.8 |

Formulas: speedup = t_teacher20 / t_method; time reduction % = 100 x (1 - t_method / t_teacher20); memory reduction % = 100 x (1 - MiB_method / MiB_teacher20).

## B. Fidelity to teacher20 (16 cases per method; lower LPIPS/MAE/boundary, higher SSIM is better)

| Dataset | LPIPS V1 | LPIPS V2 | LPIPS reduction % | MAE V1 -> V2 | SSIM V1 -> V2 | Boundary V1 -> V2 |
|---|---:|---:|---:|---|---|---|
| TuSimple | 0.3770 | 0.2028 | 46.21 | 20.41 -> 13.70 | 0.6396 -> 0.7565 | 13.27 -> 9.01 |
| CULane | 0.2483 | 0.2201 | 11.36 | 17.19 -> 14.21 | 0.6062 -> 0.6199 | 15.23 -> 11.38 |

Formula: LPIPS reduction % = 100 x (LPIPS_V1 - LPIPS_V2) / LPIPS_V1.

## C. Lane preservation after editing (detector run on edited images)

| Dataset | Method | Metric | Value % | Delta vs clean (pp) | Displacement px | Coverage |
|---|---|---|---:|---:|---:|---:|
| TuSimple | clean | ACC_percent | 97.2284 | +0.0000 | 0.0000 | 1.0000 |
| TuSimple | teacher20 | ACC_percent | 97.2284 | +0.0000 | 0.0614 | 1.0000 |
| TuSimple | teacher4 | ACC_percent | 97.2563 | +0.0279 | 0.1008 | 1.0000 |
| TuSimple | student4 | ACC_percent | 97.1168 | -0.1116 | 0.1285 | 0.9972 |
| TuSimple | student_v2 | ACC_percent | 97.1912 | -0.0372 | 0.0856 | 0.9994 |
| CULane | clean | F1_percent | 66.6667 | +0.0000 | 0.0000 | 1.0000 |
| CULane | teacher20 | F1_percent | 67.4242 | +0.7576 | 0.1401 | 0.9749 |
| CULane | teacher4 | F1_percent | 68.2171 | +1.5504 | 0.4199 | 0.9504 |
| CULane | student4 | F1_percent | 66.6667 | +0.0000 | 0.6608 | 0.9960 |
| CULane | student_v2 | F1_percent | 66.6667 | +0.0000 | 0.5873 | 0.9945 |

Formula: delta vs clean = metric_method - metric_clean. CULane uses lane F1 on 8 scenes (small sample), TuSimple uses ACC.


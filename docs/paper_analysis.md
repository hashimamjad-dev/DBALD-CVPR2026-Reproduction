# Paper analysis and reproducibility audit

Source: the user-supplied 11-page CVPR 2026 PDF, printed pages 34950–34960, read in full. PDF pages 1–8 contain the main paper; 9–11 contain acknowledgments/references. This document separates **PAPER**, **VERIFIED SOURCE**, and **ASSUMPTION** evidence. Audit date: 2026-10-19 (Asia/Karachi).

## Classification and scope

**Independent reconstruction of benign components, not reproduction of the DBALD attack.** The paper explicitly points to https://sites.google.com/view/dbald. Its accessible overview is author-associated by that link, but no source-code repository, training configurations, or poisoned checkpoints were linked in the retrieved overview. Searches did not establish an author-released DBALD repository. This is a bounded discovery finding, not proof that no code exists anywhere. The CVF HTML endpoint returned HTTP 403 during discovery; supplementary material has not been verified. The attached PDF contains no supplement.

Per the user's request, the implementation does not poison labels or training data, train backdoors, synthesize optimized triggers, optimize stealth/evasion, or connect to driving systems. It studies clean predictions, fixed benign transformations, saliency, segmentation, and ordinary inpainting independently. The offensive stages below are descriptive only.

## Section 4 pipeline (PAPER, PDF pp. 4–6)

1. Segment road regions with SAM2. Slide a rectangular window over the segmented region to form plausible position candidates.
2. Use a pretrained lane detector with the victim architecture to compute input-gradient heatmaps. The paper selects task losses according to the attack scenario and ranks roughly 100–400 candidate windows by gradient response.
3. Mask the chosen location and extract nearby lane/environment context. Apply UltraEdit-inspired masked diffusion and latent-space lane/environment constraints. Insert resulting samples into the training set, then train a poisoned detector. The last sentence describes the paper, not an implemented feature.

LaneATT and ADNet are anchor based. The paper associates focal loss with existence, and L1/GLIoU with coordinate regression. SCNN and RESA are segmentation based. The paper says cross-entropy for classification and binary cross-entropy for regression; this wording should not be silently treated as a verified implementation mapping. A binary existence head is not ordinary coordinate regression.

## Mathematical inventory (PAPER)

The paper's displayed formulas are unnumbered. Notation below retains their substance; typographical ambiguities are called out.

### Predictions and threat model, PDF p. 3

- `f_theta(x) -> g`, where theta is the model parameter set, x an input image, and `g=[(l_1,e_1),...,(l_n,e_n)]`. Lane i is `l_i={p^i_1,...,p^i_m}`, m sampled 2-D coordinates; `e_i in {0,1}` denotes existence; n is lane count. Tildes denote ground truth and `[n]={1,...,n}`. The ideal clean relation is `f_theta(x)=g_tilde`.
- `T={q[(w,h),c] | (w,h) in R, c in C}`: pixel q at coordinate (w,h), region R, color space C. The original has a surplus closing brace.
- Triggered behavior is expressed as `f_theta(x') != g_tilde`: incorrect geometry `l_i != l_tilde_i`, missing lane `e_i=0, e_tilde_i=1`, or spurious lane `e_i=1, e_tilde_i=0`. These ideal relations are not measured equalities.
- LOA is a horizontal offset: `l_i={p_tilde^i_j+(beta,0) | j in [m]}`, `e_i=e_tilde_i`. The printed formula reuses i as the point index and displays `p^i_1`; the semantic description says all lane points. We document the intended indexing instead of implementing label modification.
- LDA: for all lanes `l_i=empty`, `e_i=0`.
- LRA: `l_i={p_tilde^i_1,...,p_tilde^i_j,p^i_(j+1),...,p^i_m}`, `e_i=e_tilde_i`, for a lane i and split j; the suffix coordinates are rotated by angle alpha after cubic-spline fitting and re-evaluation at label y-coordinates. The exact pivot and spline conventions are absent.

### Sensitivity, PDF p. 4

`M_grad(i,j)=sum_(k=1)^CH |partial L / partial x_(i,j,k)|`, with `x in R^(H x W x CH)`. H/W are image height/width, CH channels, L a task loss, and i/j spatial indices. This is absolute input-gradient sensitivity, not Grad-CAM. Our benign component accepts an ordinary clean-data scalar loss; it does not target incorrect geometry or existence.

### Diffusion, PDF p. 5 (documentation only)

`z_(t-1)=(1-M)*z_T + M*D_M(z_t)` for even t; `z_(t-1)=D_M(z_t)` otherwise. Products are elementwise. `z_t in R^(H x W x C)` is a latent at timestep t; z_T is described as initial standard Gaussian noise; `M in {0,1}^(H x W x C)` marks editable coordinates; D_M is denoising.

**Unresolved inconsistency:** the prose says the original image latent is preserved outside the mask, but the displayed formula uses z_T, just defined as initial Gaussian noise. We do not invent a replacement scheduler or clean-latent blending rule. Neither this alternating update nor optimization for stealth is implemented.

### Image consistency, PDF p. 6

`L_lane=MSE(gen_img * lane_mask, clean_img * lane_mask)`.
`L_env=SSIM(gen_img * env_mask, clean_img * env_mask)`.

`gen_img` and `clean_img` are generated and original images; masks are binary lane and surrounding-environment supports. The paper does not give the MSE normalization, SSIM window/data range, mask construction, weights, or optimization schedule. SSIM is a similarity to maximize, yet is called a loss; minimization would ordinarily use `1-SSIM`. The prototype reports both similarity and dissimilarity without using either for optimization. It distinguishes whole-array masked MSE (zeros included) from MSE on selected pixels, which are not numerically equivalent.

### ACC and ASR, PDF p. 6

`Metric = sum_i C_i / sum_i S_i`: C_i is the number of lane points within 20 pixels of ground truth, S_i the total lane points. The paper applies this statement to clean ACC and LOA/LRA ASR. LDA ASR is the fraction of disappeared lanes. It does not fully specify matching, invalid samples, reference labels for attack scoring, aggregation, or confidence thresholds.

**VERIFIED SOURCE:** the official TuSimple evaluator has angle-adjusted pixel thresholds, per-lane/per-image aggregation, invalid-point handling, a 200-ms cutoff, an excess-prediction penalty, and a special rule for more than four GT lanes. These are more specific than the paper's compact formula. CULane conventionally reports lane IoU/F1, not TuSimple point accuracy. Thus the paper's CULane ACC protocol cannot be assumed to match official CULane F1. The project keeps these metric names separate and does not calculate ASR.

## Datasets and architectures

**PAPER:** CULane: approximately 88K training / 34K test images, 1640×590; TuSimple: approximately 3.6K / 2.7K, 1280×720; up to four or five lanes respectively. Table 1 is the mud-trigger experiment. Cone triggers are additionally evaluated physically. No third benchmark dataset is reported. Four closed-road environments produce 100 cone images, with day/night and two cone shapes/colors; these images are not a standard public split.

**VERIFIED SOURCE:** CULane documentation gives 88,880 train, 9,675 validation, 34,680 test, 133,235 total. Its text annotations store x/y pairs per lane in `.lines.txt`; image-list paths commonly start with `/` but are dataset-relative. TuSimple trainval has 3,626 clips, test 2,782; JSONL contains `raw_file`, `h_samples`, and lane x-coordinate arrays with negative invalid values. Do not confuse an upstream config's 3,616 iteration constant with the dataset count. Labels and actual image dimensions are checked at load time.

**PAPER:** four architectures: LaneATT (attention over anchors), ADNet (anchor decomposition), SCNN (spatial propagation), RESA (recurrent feature shifts). The DBALD PDF does not identify their backbones, exact commits, crop/resize settings, initialization weights, or inference thresholds.

## Explicit experimental settings (PAPER)

| Item | Reported value and location |
|---|---|
| Poison fraction | 10%, p. 6 |
| Stated trigger size | 900 pixels, p. 6 |
| BadNets / Blended | 30×30, bottom-right, p. 6 |
| LD-Attack | 900 pixels adjacent to lane near boundary, p. 6 |
| BadLane | 900 pixels sampled from 100×100 region, p. 6 |
| DBALD mask | 40×40, p. 6; 1,600 mask pixels need not equal 900 changed pixels, but conversion is unspecified |
| Generator | UltraEdit; example prompts describe sparse brown mud or small cone, p. 6 |
| LOA / LRA | 60 px / 9 degrees, p. 6 |
| Candidate count | approximately 100–400, p. 5 |
| LPIPS threshold | 0.15, p. 6; exact pairing/sampling protocol absent |
| Point threshold | 20 px, p. 6 |
| Clean fine-tuning defense | 50 epochs, p. 8 |
| Pruning | final 512-channel conv; steps of 50 neurons, p. 8 |
| Physical experiment | 100 images, four closed-road environments, two cone types, p. 8 |
| VLM detection | 100 poisoned images; called Qwen2.5-VL in text, p. 8 |

No base training epochs, optimizer, learning rate, batch size, scheduler, weight decay, random seeds, repeat counts, confidence intervals, training hardware, diffusion step count or guidance scales are given in the attached main paper. UltraEdit's own example parameters are not DBALD parameters.

## Results, ablations and countermeasures (PAPER)

Table 1 is transcribed separately in `results/reference/table1.csv` and `docs/table1.md`. All 20 rows and 220 numeric values are retained, including merged benign ACC values expanded per row. TuSimple has LDA/LOA only, no LRA column. Published averages are retained even where they differ from arithmetic on printed components.

Forensic detection (Table 2, %, CULane U/L/D then TuSimple U/L/D; U=UniDetection, L=LGrad, D=DIRE):

| Method | CULane U | L | D | TuSimple U | L | D |
|---|---:|---:|---:|---:|---:|---:|
| BadNets | 2.56 | N/A | N/A | 0.08 | N/A | N/A |
| Blended | 5.18 | N/A | N/A | 0.17 | N/A | N/A |
| LD-Attack | 6.27 | N/A | N/A | 1.31 | N/A | N/A |
| Badlane | 59.31 | 9.17 | N/A | 35.60 | 4.23 | N/A |
| DBALD | 2.72 | 1.76 | 0.56 | 0.23 | 0.51 | 0.04 |

N/A is the authors' designation for directly patched images; it is not zero and does not mean a successful detector test. Benign detection is described only as over 90%.

Heatmap ablation, mud/CULane (Table 3, LDA/LRA/LOA): LaneATT with `81.65/73.45/74.19`, without `80.97/70.16/68.52`; RESA with `78.15/65.15/76.86`, without `74.23/58.26/69.37`. Text claims +3.21 and +6.77; printed row means imply approximately +3.21 and +6.10 percentage points respectively. The RESA discrepancy is unresolved.

Loss ablation: forensic scores 2.46 with both, 12.13 without lane loss, 14.51 without environment loss. This is not Table 2's CULane 2.72; the exact sample subset/protocol is not supplied.

Physical LOA on LaneATT: ASR 57%; with rain/blur/occlusion 45/48/49%. No physical attack is implemented here.

Defenses: clean fine-tuning reduces ASR 74.19→63.68%; pruning 50 neurons gives 73.67% ASR; pruning 400 gives 15.87% ASR while clean ACC falls 74.20→21.19%. VLM detection is 6% cones, 3% mud. The text says Qwen2.5-VL but reference [4] names Qwen3-VL. Quantization/reasoning injection are future work, not tested defenses in this paper.

## Additional inconsistencies and missing details

- The p. 6 prose attributes TuSimple LaneATT ASR 90.84 to BadLane; Table 1 places 90.84 under DBALD (BadLANE is 89.51).
- RESA DBALD TuSimple average is printed 95.43, while `(96.57+94.28)/2=95.425`; retain printed 95.43 rather than library rounding behavior.
- The headline +10.87 and stealthiness +56.59 are reported claims; their averaging/comparison protocol is not fully identified. Do not invent one.
- No SAM2 checkpoint version, road prompts, candidate stride/overlap/border policy, or threshold is supplied.
- No target checkpoint/backbone, clean pretraining duration, poison-index list, poison seeds, training reruns, label generator, matching algorithm, or clean/poisoned model linkage is supplied.
- No UltraEdit variant, model commit, diffusion scheduler, random seed, latent optimizer, loss weights, gradient update schedule, or environmental annotation procedure is supplied.
- No released physical images, detector thresholds/calibration subsets, VLM prompt, or per-image measurements were verified.

**ASSUMPTIONS used only in the safe implementation:** explicit caller-provided road prompts; fixed windows used for visualization; ordinary OpenCV inpainting at a fixed region independent of saliency; unchanged ground truth under photometric transformations; strict dimension checks; coordinate geometry metrics named as diagnostics; single CPU inference timings reported as local measurements. A source checkpoint is never assumed equivalent to a DBALD model.

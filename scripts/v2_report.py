"""Produce versioned tables from measured artifacts, never substitute references."""
import csv,hashlib,json,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results/distillation/v2'; V1=ROOT/'results/distillation/v1'
def read(path): return json.loads(path.read_text())
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def csvfile(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def mean(rows,key): return statistics.mean(r[key] for r in rows)

m=read(OUT/'manifest.json'); lock=read(OUT/'selection.json'); selected=lock['selected']
validation=read(OUT/'validation_summary.json'); final=read(OUT/'final_summary.json'); raw=read(OUT/'final_quality.json')
assert read(OUT/'audit.json')['passed']
visual=read(OUT/'visual_review.json')
timing={name:read(OUT/'timing'/f'{name}.json') for name in ['teacher20','teacher4','student_v1']+m['candidates']}
lane_old=read(V1/'lane_consistency.json'); lane_new=read(OUT/'lane_consistency.json')
assert len(lane_new['rows'])==32
report=['# Task-specific road diffusion distillation: experiment v2','',
    'This report continues the original teacher–student experiment. All v1 artifacts are preserved. It tests two proposed quality improvements against an equal-update continuation control, then evaluates a validation-selected checkpoint on the reused v1 benchmark. No attack success rate was measured.','',
    '## Question and evidence-supported diagnosis','',
    'The fixed research question is whether a compact task-specific diffusion student can reduce generation cost while retaining realistic road appearance and lane characteristics. The naturalistic-backdoor literature supplies the security context; implementation remains benign offline restoration. No poisoning, optimized triggers or targeted lane manipulation is implemented.','',
    'TuSimple validation MAE rose from 15.71 at the original training noise seed to 30.55 at an unseen seed, and local LPIPS from 0.126 to 0.399. The four-step teacher remained near MAE 3.03–3.11. Student rollout latent errors exceeded teacher-forced transition errors: final edited-region MSE 0.077/0.207 versus 0.027/0.047. These observations support noise-generalization and accumulated approximation error as contributors. CULane remained poor at both seeds, so one explanation does not cover all errors.','',
    'The dilated latent edit occupies 7.81% of TuSimple and 2.27% of CULane latents, motivating a regional-loss test. The four-step teacher already has much better decoded fidelity, so reducing the step count alone does not explain the compact student’s failure. Capacity is not isolated in this experiment. Verified pixel compositing does not remove a visibly poor generated patch; latent error is not a guarantee of decoded quality. See [full diagnosis](distillation_v2_diagnosis.md).','',
    '## Controlled methodology','',
    'All candidates start from the same v1 update-1700 checkpoint and use the same 2,381,044-parameter architecture, four-step schedule, 64 TuSimple training scenes, batch size 8, AdamW learning rate 0.0002, initialization seed and 1,000 additional updates. Checkpoints at updates 500 and 1,000 are assessed on decoded outputs.','',
    table(['Candidate','Objective','Training trajectories'],[['control_global','Original global equivalent-epsilon MSE','64, seed 0'],['mask_single','0.5 global + 0.5 dilated-edit-region MSE','64, seed 0'],['mask_diverse','Same regional objective','128: seed 0 plus one unique noise per scene']]),'',
    'Validation uses eight TuSimple scenes and four available CULane scenes, at seeds 0 and 17. The additional training noises are seeds 2000–2063, never validation or test seeds. Training image count is unchanged. CULane validation and evaluation files are disjoint but come from a narrow available source; independent drive-level generalization is not established.','',
    f"Selection used mean validation local LPIPS, with no worse boundary error than v1 on either dataset and MAE as tie-break. The selected model is **{selected['method']}, additional update {selected['step']}**, SHA-256 `{selected['sha256']}`. The selection was locked before final test generation. The v1 test set was previously observed, so it is a reused benchmark, not a new pristine test set.",'',
    'The frozen teacher remains SD1.5 inpainting, a local proxy rather than DBALD’s UltraEdit. Teacher sampling uses 20 deterministic DDIM steps; the coarse teacher and student use the same four transition times. The student shares the frozen VAE and safety checker. Teacher components use FP16 and the student FP32. All versions and compatibility fixes remain unchanged from v1; this is not a claim of exact DBALD software equivalence.','',
    'Deterministic trajectory targets are motivated by [progressive distillation](https://arxiv.org/abs/2202.00512) and [guided diffusion distillation](https://arxiv.org/abs/2210.03142). Noise diversity is informed by the [official Diffusers distillation recipe](https://huggingface.co/docs/diffusers/training/lcm_distill). This compact regression method is not claimed to reproduce those algorithms exactly. The regional objective is our controlled task-specific hypothesis. [LPIPS](https://arxiv.org/abs/1801.03924) complements pixel and structural fidelity; it is not a realism probability.','',
    '## Validation outcomes and acceptance','',
    'MAE uses 0–255 pixels. SSIM is higher-is-better; MAE, local SqueezeNet-LPIPS and boundary residual are lower-is-better. Local LPIPS uses the raw edit bounding box plus eight pixels of context, resized to 128×128. Boundary residual measures across-mask-edge RGB differences relative to the teacher composite. Quality is measured inside raw generated edits rather than diluted by copied background. TuSimple and CULane edit sizes differ and are kept separate.','']
acceptance=[]
for ds in ('TuSimple','CULane'):
    vals=[r for r in validation if r['dataset']==ds]
    report+=['### '+ds+' validation','',table(['Method / update','Cases','MAE','SSIM','Local LPIPS','Boundary'],[[r['method'],r['n'],f"{r['masked_mae']:.3f}",f"{r['masked_ssim']:.4f}",f"{r['local_lpips']:.4f}",f"{r['boundary_error']:.3f}"] for r in vals]),'']
    name='student_v1' if selected['method']=='student_v1' else f"{selected['method']}_{selected['step']}"
    s=next(r for r in vals if r['method']==name); t=next(r for r in vals if r['method']=='teacher4')
    checks=[('SSIM',s['masked_ssim'],.85,s['masked_ssim']>=.85),('MAE',s['masked_mae'],1.5*t['masked_mae']+2,s['masked_mae']<=1.5*t['masked_mae']+2),('LPIPS',s['local_lpips'],1.25*t['local_lpips']+.02,s['local_lpips']<=1.25*t['local_lpips']+.02),('Boundary',s['boundary_error'],t['boundary_error']+2,s['boundary_error']<=t['boundary_error']+2)]
    acceptance.extend(dict(dataset=ds,criterion=k,value=v,threshold=limit,passed=bool(ok)) for k,v,limit,ok in checks)
    report += [table(['Predeclared criterion','Measured','Threshold','Outcome'],[[k,f'{v:.4f}',('>= ' if k=='SSIM' else '<= ')+f'{limit:.4f}','PASS' if ok else 'FAIL'] for k,v,limit,ok in checks]),'']
report+=['Visual review is a non-blinded engineering assessment of saved raw edit crops and composites, not a human realism study. '+visual['summary'],'',
    'At the matched 1,000-update budget, overall validation LPIPS is 0.2709 for the continuation control, 0.2833 for the regional loss alone and 0.1952 for the regional loss plus noise diversity (v1: 0.2629). Thus the regional loss alone is not supported as a quality fix; the diversity intervention improves this run relative to both controls. No global-loss-plus-diversity arm was run, so interaction between the regional loss and noise diversity is not isolated. These are single-run results, not estimates across training replications.','',
    '![TuSimple validation raw edit comparison](../results/distillation/v2/figures/validation_TuSimple_0.png)','',
    '![CULane validation raw edit comparison](../results/distillation/v2/figures/validation_CULane_0.png)','',
    '## Final reused test comparison','',
    'Each dataset has eight scenes × two unseen generation seeds (100 and 101), giving 16 cases per method. Teacher20, teacher4 and original-student images are reused exactly from v1; the same new metric implementation scores all four methods. No test result changes the selected model.','']
for ds in ('TuSimple','CULane'):
    report+=['### '+ds,'',table(['Method','MAE','SSIM','Local LPIPS','Boundary','Raw lane-band MAE'],[[r['method'],f"{r['masked_mae']:.3f}",f"{r['masked_ssim']:.4f}",f"{r['local_lpips']:.4f}",f"{r['boundary_error']:.3f}",f"{mean([x for x in raw if x['dataset']==ds and x['method']==r['method'] and x['raw_lane_mae'] is not None], 'raw_lane_mae'):.3f}"] for r in final if r['dataset']==ds]),'']
    lane_cases=[x for x in raw if x['dataset']==ds and x['method']=='student_v2' and x['raw_lane_mae'] is not None]
    report += [f"Raw lane-band MAE uses {len(lane_cases)} of 16 cases with annotated lane pixels inside the 512-pixel crop. Missing lane bands are excluded, never treated as zero error.",'']
    before=next(r for r in final if r['dataset']==ds and r['method']=='student_v1'); after=next(r for r in final if r['dataset']==ds and r['method']=='student_v2')
    report += [f"Relative to the original student, local LPIPS decreases by {100*(1-after['local_lpips']/before['local_lpips']):.1f}% and masked MAE decreases by {100*(1-after['masked_mae']/before['masked_mae']):.1f}%. These are teacher-fidelity changes, not measured realism gains.",'']
report+=['![Final TuSimple raw edit comparison](../results/distillation/v2/figures/evaluation_TuSimple_0.png)','',
    '## Measured speed and memory','',
    'One common contemporaneous protocol measures four fixed validation scenes (two per dataset), seed 17, two repetitions and one warmup per method. Each candidate uses its best eligible validation checkpoint, or its lowest-LPIPS checkpoint if neither is eligible; ineligible candidates cannot win final selection. Measurements include encoding, noise, denoising, decoding, safety checking and compositing; exclude loading, file I/O and cooling. Student runs unload the teacher denoiser. Peak allocation is PyTorch allocation, not total device VRAM. This small thermal-laptop timing sample is not sustained throughput and is not pooled with v1 timing.','']
speed=[]
for method,t in timing.items():
    speed.append(dict(method=method,steps=20 if method=='teacher20' else 4,median_seconds=t['median_seconds'],speedup=timing['teacher20']['median_seconds']/t['median_seconds'],peak_MiB=t['peak_bytes']/2**20))
report+=[table(['Method','Steps','Median seconds','Speedup vs teacher20','Peak MiB'],[[r['method'],r['steps'],f"{r['median_seconds']:.4f}",f"{r['speedup']:.2f}x",f"{r['peak_MiB']:.1f}"] for r in speed]),'']
for ds in ('TuSimple','CULane'):
    report += [ds+' timing: '+', '.join(f"{method} {statistics.median(x['seconds'] for x in t['rows'] if x['dataset']==ds):.4f}s" for method,t in timing.items())+'.','']
training=[read(OUT/'candidates'/name/'complete.json') for name in m['candidates']]
report += [table(['Training run','Additional updates','Cached examples','Seconds incl. validation/cooling','Peak MiB'],[[r['candidate'],r['updates'],r['examples'],f"{r['seconds']:.2f}",f"{r['peak_bytes']/2**20:.1f}"] for r in training]),'',
    f"Training times exclude model setup and teacher-cache generation. The additional 64 teacher trajectories required {read(OUT/'cache_cost.json')['measured_encode_and_trajectory_seconds']:.2f} seconds of measured encoding/denoising, excluding loading, cooling and file writes. Cached trajectories are reused rather than regenerated per update.",'',
    '## Clean lane consistency, not attack success','',
    'The frozen official LaneATT ResNet-34 checkpoints assess final composites after selection. Clean cached predictions are checked against the current runtime. Ground-truth lane bands and all outside-mask pixels are copied from the source by construction; these scores do not demonstrate learned lane preservation in the raw output. Displacement and coverage are means of defined per-scene/seed values; CULane F1 is the mean of the two separately scored eight-image seed runs.','']
lane_summary=[]
for ds in ('TuSimple','CULane'):
    for method in ('clean','teacher20','teacher4','student4','student_v2'):
        data=lane_new if method=='student_v2' else lane_old
        rs=[r for r in data['rows'] if r['dataset']==ds and r['method']==method]
        value=mean(rs,'official_tusimple_accuracy')*100 if ds=='TuSimple' else mean([r for r in data['culane_metrics'] if r['method']==method],'f1_percent')
        mae_values=[r['clean_prediction_consistency']['mae_px'] for r in rs if r['clean_prediction_consistency']['mae_px'] is not None]
        coverage_values=[r['clean_prediction_consistency']['coverage'] for r in rs if r['clean_prediction_consistency']['coverage'] is not None]
        lane_summary.append(dict(dataset=ds,method=method,metric='ACC_percent' if ds=='TuSimple' else 'F1_percent',value=value,displacement_px=statistics.mean(mae_values),coverage=statistics.mean(coverage_values),displacement_cases=len(mae_values),coverage_cases=len(coverage_values),total_cases=len(rs)))
    report+=['### '+ds+' eight-scene lane subset','',table(['Method','ACC %' if ds=='TuSimple' else 'F1 %','Conditional displacement px','Clean-point coverage'],[[r['method'],f"{r['value']:.4f}",f"{r['displacement_px']:.4f}",f"{r['coverage']:.4f}"] for r in lane_summary if r['dataset']==ds]),'']
    counts=next(r for r in lane_summary if r['dataset']==ds and r['method']=='student_v2')
    report += [f"Selected-student displacement is defined for {counts['displacement_cases']}/{counts['total_cases']} cases and coverage for {counts['coverage_cases']}/{counts['total_cases']}. Undefined conditional measures are excluded, not assigned zero; official dataset scores still include all eight scenes. The `student4` row denotes v1.",'']
    clean=next(r for r in lane_summary if r['dataset']==ds and r['method']=='clean'); new=next(r for r in lane_summary if r['dataset']==ds and r['method']=='student_v2')
    drop=clean['value']-new['value']; limit=.5 if ds=='TuSimple' else 2.
    acceptance.extend([dict(dataset=ds,criterion='Clean metric drop pp',value=drop,threshold=limit,passed=drop<=limit),dict(dataset=ds,criterion='Lane displacement px',value=new['displacement_px'],threshold=1.,passed=new['displacement_px']<=1.),dict(dataset=ds,criterion='Lane coverage',value=new['coverage'],threshold=.99,passed=new['coverage']>=.99)])
chosen_speed=next(r for r in speed if r['method']==selected['method'])
acceptance.extend([dict(dataset='Timing sample',criterion='Latency seconds',value=chosen_speed['median_seconds'],threshold=.6,passed=chosen_speed['median_seconds']<=.6),dict(dataset='Timing sample',criterion='Speedup',value=chosen_speed['speedup'],threshold=3.,passed=chosen_speed['speedup']>=3.),dict(dataset='Timing sample',criterion='Peak MiB',value=chosen_speed['peak_MiB'],threshold=7168,passed=chosen_speed['peak_MiB']<7168)])
for ds in ('TuSimple','CULane'):
    cases=[r for r in visual['cases'] if r['dataset']==ds]; fraction=statistics.mean(r['acceptable'] for r in cases)
    acceptance.append(dict(dataset=ds,criterion='Visual acceptable fraction',value=fraction,threshold=.9,passed=fraction>=.9))
all_pass=all(r['passed'] for r in acceptance)
report[2]='**Overall predeclared acceptance: '+('PASS' if all_pass else 'NOT MET')+'.** '+report[2]
report+=['### Acceptance accounting','',table(['Scope','Criterion','Measured','Threshold','Outcome'],[[r['dataset'],r['criterion'],f"{r['value']:.4f}",r['threshold'],'PASS' if r['passed'] else 'FAIL'] for r in acceptance]),'']
report+=['CULane F1 is not the paper’s point ACC. Full CULane remains incomplete; no new full clean benchmark is claimed. Previously measured full TuSimple clean baselines are preserved in measured_results.csv. Published benign ACC, poisoned-model ACC and attack-specific ASR remain reference-only in results/reference/table1.csv. New benign robustness measurements do not fill missing ASR cells. See [security boundary](security_evaluation_boundary.md) and the [DBALD paper](https://arxiv.org/html/2508.15778v2).','',
    '## Limits and next research priorities','',
    'Only one training run per candidate was performed; noise seeds and scenes are not independent training replicates. Validation selection across six checkpoints introduces selection bias. More scenes, independent drives and a new prospective test split are required before generalization claims. Small CULane masks make absolute cross-dataset quality comparisons inappropriate. There is no blinded realism rating, full-dataset generative-quality assessment or attack/defense efficacy experiment.','',
    'Capacity remains untested. Training contains TuSimple images and its larger mask shape; CULane also changes scene distribution and edit geometry. Their separate contributions have not been isolated. The next controlled study should test training-mask diversity and rollout-aware supervision or decoded perceptual objectives against the selected method at matched budget, followed by a capacity ablation if warranted. These are research questions, not claims of measured improvements. Teacher fidelity alone does not establish road-scene realism, naturalistic-trigger properties or safe downstream behavior.','',
    '## Reproducibility and integrity','',
    'The predeclared protocol, manifest, source/weight hashes, cached trajectories, update histories, checkpoint selection lock, all raw/composite images, per-case metrics and timing repetitions are retained locally. The integrity audit verifies 394 unchanged v1 files, all source image hashes, exact compositing and independently recalculated final masked MAE. All 29 tests pass, including interrupted-checkpoint preservation. The initial test invocation could not create the default temporary directory; the successful run used an explicit workspace temporary directory.','',
    'Execution order: v2_prepare.py; v2_train.py control_global; v2_train.py mask_single; v2_prepare.py --training-diversity; v2_train.py mask_diverse; v2_evaluate.py select; v2_evaluate.py benchmark --method METHOD (each of six methods); v2_evaluate.py final; v2_lanes.py; v2_gallery.py (validation and --final); v2_audit.py; v2_report.py. Use the preserved CUDA environment described in package_compatibility.md. Visual review is recorded separately and is not automated into a realism score. No research files were uploaded or pushed.','']
csvfile('validation_summary.csv',validation); csvfile('final_summary.csv',final); csvfile('timing_summary.csv',speed); csvfile('lane_summary.csv',lane_summary); csvfile('acceptance.csv',acceptance)
(ROOT/'docs/distillation_v2_report.md').write_text('\n'.join(report),encoding='utf-8')
sources=list((ROOT/'scripts').glob('v2_*.py'))+[ROOT/'src/lane_repro/distillation.py',ROOT/'src/lane_repro/road_teacher.py',ROOT/'tests/test_distillation_v2.py',ROOT/'docs/distillation_v2_protocol.md',ROOT/'docs/distillation_v2_timing_protocol.md']
(OUT/'source_hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},indent=2))
print(ROOT/'docs/distillation_v2_report.md')

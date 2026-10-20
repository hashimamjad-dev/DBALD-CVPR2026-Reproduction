"""Build supervisor tables directly from executed artifacts; reject incomplete runs."""
import csv,json,hashlib,os,xml.etree.ElementTree as ET
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'tmp/matplotlib'))
import numpy as np
from PIL import Image,ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results/distillation/v1'
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(str(v) for v in r)+' |' for r in rows])
def writecsv(name,rows):
    with (OUT/name).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
def main():
    data=json.loads((OUT/'evaluation.json').read_text()); protocol=json.loads((OUT/'evaluation_protocol.json').read_text())
    train=json.loads((OUT/'training.json').read_text()); history=json.loads((OUT/'training_history.json').read_text())
    cache_cost=json.loads((OUT/'cache_cost.json').read_text())
    lanes=json.loads((OUT/'lane_consistency.json').read_text())
    audit=json.loads((OUT/'output_audit.json').read_text()); assert audit['passed']
    tests=ET.parse(ROOT/'results/distillation/tests.xml').getroot().find('testsuite')
    assert tests.get('failures')=='0' and tests.get('errors')=='0'
    assert len(data)==96 and len({(r['id'],r['seed'],r['method']) for r in data})==96
    assert train['updates']==2000 and history[-1]['step']==2000
    assert len(lanes['rows'])==112
    clean=list(csv.DictReader((ROOT/'results/measured_results.csv').open()))
    published=list(csv.DictReader((ROOT/'results/reference/table1.csv').open()))
    report=['# Task-specific road diffusion distillation: research report','',
        '**Finding:** the compact student fits the 8 GB GPU and is substantially faster, but this first prototype does not meet the quality-preservation objective. Its masked teacher agreement is worse than the undistilled four-step control, and inspected TuSimple outputs contain conspicuous blotchy edits. This is a measured feasibility result with a quality failure, not a successful DBALD reproduction or evidence that attack effectiveness is preserved.','',
        '## Scientific question and scope','',
        'The thesis Chapter 2 motivates an efficient task-specific generator in the context of naturalistic backdoor data poisoning. This experiment measures a bounded component: whether a compact student can approximate ordinary road-restoration outputs from a general-purpose diffusion teacher with fewer steps and lower computational cost. It does not implement poisoning or test attack effectiveness. DBALD remains the security reference, with published ACC/ASR kept separate from local clean-model and editing measurements. No Chapter 3 was requested or used.','',
        '## Independent method','',
        'A frozen SD1.5 inpainting teacher (a proxy, not DBALD UltraEdit) supplies 20-step deterministic DDIM trajectories. A compact conditional U-Net learns four coarse transitions from cached teacher states. This is offline trajectory regression inspired by distillation, not an exact LCM or progressive-distillation reproduction. The student has a fixed restoration task, no open-ended text conditioning, and shares the frozen teacher VAE and safety checker.','',
        'Training uses 64 TuSimple train clips, validation 16 separate clips, and evaluation eight held-out TuSimple plus eight available CULane scenes. Training/validation clips were checked against all official TuSimple test clips. File-hash ordering fixed selection before generation. One training noise seed and two different evaluation seeds make this a small feasibility study. All methods use identical 512-pixel crops, masks, conditions and initial noise. Annotation-based lane protection is an oracle aid and is not demonstrated deployable segmentation.','',
        f"The student has {train['parameters']:,} parameters; the teacher denoiser has {protocol['teacher_denoiser_parameters']:,}. Shared VAE/safety-checker parameters: {protocol['shared_vae_safety_parameters']:,}. There were 2,000 optimizer updates, taking {train['seconds']:.2f} seconds including training cooling breaks; peak PyTorch training allocation was {train['peak_bytes']/2**20:.1f} MiB. Validation rollout latent MSE changed from {train['initial_validation']['validation_rollout_mse']:.6f} to a best {train['best_validation_rollout_mse']:.6f}; the selected checkpoint is update {protocol['checkpoint_step']}. Selection never used held-out image quality or detector outcomes. Teacher/VAE inference uses FP16; the compact student uses FP32.",'',
        f"Preparing the 80 cached teacher trajectories used {cache_cost['measured_encode_and_trajectory_seconds']:.2f} seconds of measured encoding/denoising, excluding loading, cooling and file writes. This is an additional preparation cost, not part of student inference latency; one cached trajectory predates the user pause and is not used as a held-out timing sample.",'',
        '## Measured generation results','',
        'Each table contains eight scenes × two seeds per method. Latency is the median of per-scene/seed medians (two timed repetitions after warm-up). Timing includes image encoding, denoising, VAE decoding, safety checking and native-frame compositing, but excludes model loading, cached prompt encoding, disk writes and cooling breaks. Values describe this thermally constrained laptop, not sustained-throughput performance. Peak allocated memory is live PyTorch GPU allocation, not total system VRAM.','',
        'Masked SSIM/MAE quantify agreement with the 20-step teacher only. They are not realism scores. Raw lane-band MAE is measured before compositing, on a 0–255 scale. Outside-mask identity in the final composite is guaranteed by copying source pixels and cannot prove learned preservation. TuSimple masks cover up to 12,288 pixels; CULane uses a 1,536-pixel road strip above its visible hood, fixed during input QA before evaluation. Different edit areas limit cross-dataset comparisons.','']
    summary=[]
    for dataset in ('TuSimple','CULane'):
        report += ['### '+dataset+' editing subset','']
        rows=[]
        for method in ('teacher20','teacher4','student4'):
            rs=[r for r in data if r['dataset']==dataset and r['method']==method]
            item=dict(dataset=dataset,method=method,scenes=8,seeds=2,steps=rs[0]['steps'],
                median_seconds=float(np.median([r['median_seconds'] for r in rs])),peak_MiB=max(r['peak_allocated_bytes'] for r in rs)/2**20,
                masked_teacher_ssim=float(np.mean([r['masked_teacher_ssim'] for r in rs])),
                masked_teacher_mae=float(np.mean([r['masked_teacher_mae'] for r in rs])),raw_lane_mae=float(np.mean([r['raw_lane_mae'] for r in rs if r['raw_lane_mae'] is not None])))
            summary.append(item); rows.append([method,item['steps'],f"{item['median_seconds']:.4f}",f"{item['peak_MiB']:.1f}",f"{item['masked_teacher_ssim']:.4f}",f"{item['masked_teacher_mae']:.2f}",f"{item['raw_lane_mae']:.2f}"])
        report += [table(['Method','Steps','Median seconds','Peak MiB','Masked teacher SSIM','Masked teacher MAE','Raw lane MAE'],rows),'']
        rs=[r for r in lanes['rows'] if r['dataset']==dataset]; rows=[]
        for method in ('clean','teacher20','teacher4','student4'):
            selected=[r for r in rs if r['method']==method]
            mae=[r['clean_prediction_consistency']['mae_px'] for r in selected if r['clean_prediction_consistency']['mae_px'] is not None]
            coverage=[r['clean_prediction_consistency']['coverage'] for r in selected if r['clean_prediction_consistency']['coverage'] is not None]
            if dataset=='TuSimple': metric=np.mean([r['official_tusimple_accuracy'] for r in selected])*100
            else: metric=np.mean([r['f1_percent'] for r in lanes['culane_metrics'] if r['method']==method])
            rows.append([method,f'{metric:.4f}',f'{np.mean(mae):.3f}',f'{np.mean(coverage):.4f}','unmeasured'])
        report += [table(['Method','Subset ACC %' if dataset=='TuSimple' else 'Subset lane F1 % (mean over seeds)','Prediction MAE vs clean (px)','Clean-point coverage','ASR'],rows),
            '', 'Frozen LaneATT-R34 assessment. Conditional MAE must be read with coverage; these benign edits were not optimized against the detector. TuSimple scoring uses zero runtime to isolate geometry; this is not a new timed full benchmark.','']
    writecsv('generation_summary.csv',summary)
    for dataset in ('TuSimple','CULane'):
        writecsv(dataset.lower()+'_generation.csv',[r for r in summary if r['dataset']==dataset])
    report+=['## Visual evidence','',
        'The four gallery scenes were fixed in the report script before outputs were inspected. These are uncomposited model outputs, not blinded human ratings. Qualitative inspection found blotchy student edits in scenes 080 and 083. Scene 088 also shows that raw teacher and student outputs can alter vehicle placement outside the intended mask. Final compositing prevents those outside-mask alterations, but this is an imposed constraint, not learned scene preservation. No numerical realism rating was collected.','',
        '![Raw output comparison](../results/distillation/v1/raw_comparison_gallery.png)','',
        '![Road texture comparison](../results/distillation/v1/texture_comparison_gallery.png)','',
        '![Validation learning curve](../results/distillation/v1/validation_curve.png)','',
        '## Existing clean benchmarks and published security references','', 'Previously completed full baselines are reused. No incomplete CULane run is represented as a full evaluation.','']
    for dataset in ('TuSimple','CULane'):
        report+=['### '+dataset,'']
        rs=[r for r in clean if r['dataset']==dataset and r['validation_status']=='current' and r['transformation']=='clean' and (r['scope']=='full_test' if dataset=='TuSimple' else r['scope']=='subset')]
        report+=[table(['Model','Backbone','Images','Local clean metric','Value %'],[[r['model'],r['backbone'],r['images'],'ACC' if dataset=='TuSimple' else 'lane F1',f"{float(r['acc_percent'] if dataset=='TuSimple' else r['f1_percent']):.4f}"] for r in rs]),'']
        prefix=dataset.lower(); refs=[r for r in published if r['attack']=='DBALD']
        report+=['Published DBALD reference only (all architectures ResNet-34 in the extended Appendix C):','',table(['Model','Published benign ACC','Published poisoned ACC','Published LDA ASR','Published LRA ASR','Published LOA ASR'],[[r['model'],r[prefix+'_benign_acc'],r[prefix+'_poisoned_acc'],r[prefix+'_lda_asr'],r.get(prefix+'_lra_asr','not reported'),r[prefix+'_loa_asr']] for r in refs]),'']
    report+=['## Contribution and next controlled experiment','',
        'This work contributes an executable local teacher/student measurement protocol and a negative quality result: a 2.38-million-parameter student reduces latency and allocation, but does not preserve teacher editing quality. The undistilled four-step control currently offers much closer teacher fidelity. Stable final lane scores alone would conceal the student\'s visible artifacts.','',
        'Two plausible explanations require experiments, not assertions: full-crop latent loss gives little weight to the small edited region, and training/validation use one noise seed while held-out evaluation uses new seeds. A next study should predeclare mask-weighted fidelity and multi-seed validation, compare against a pretrained latent-consistency adapter, and reserve a new untouched test set. The present held-out scenes must not become an iterative tuning target. Additional training is not claimed to have solved these issues.','',
        'The original-version audit also found a separate unresolved SCNN discrepancy after aggregation, despite identical inputs and close backbone features. RESA and ADNet passed the 16-image old-CPU versus modern-CUDA checks. See the package compatibility report; no full baseline has been silently replaced.','',
        '## Limitations and remaining research questions','',
        '- Neither a low fidelity error nor a plausible composite proves naturalistic trigger stealth or backdoor success. All local ASR values remain unmeasured.','- Human realism ratings have not been collected. The comparison gallery is for qualitative inspection; no FID or validated realism score is claimed from 16 scenes.','- CULane evaluation scenes come only from the available driver_37 subset. Full CULane remains blocked by incomplete official archives/download quotas. CULane lane F1 is not the paper\'s point ACC.','- The small training set and single training seed limit generalization. Latent teacher-forcing loss and rollout image quality can disagree; uncomposited outputs must be inspected.','- A shared VAE and safety checker impose fixed runtime/memory costs. Four teacher calls still use classifier-free guidance with two effective batch elements per call; the student uses one.','- Older detector package versions are preserved in separate CPU reference environments. The RTX 5050 execution environment uses a newer compatible CUDA stack; this is not exact DBALD environment equivalence.','- The paper\'s exact poisoned checkpoints, target scoring bundle and training indices remain unavailable. No operational backdoor or defense-efficacy experiment was executed.','',
        '## Reproduction and evidence','',
        f"Verification: {tests.get('tests')} tests passed. The output audit verified all 96 original source hashes, all 96 generated/composited cases, exact outside-mask and annotated-lane pixel copying, and checkpoint selection against validation history. Independently recalculated masked MAE differs by {audit['maximum_masked_mae_recalculation_error']}. This verifies implementation bookkeeping; it does not override the observed quality failure.",'',
        'Run `scripts/distill_road.py cache`, then `train`, then `evaluate` in the documented CUDA environment, followed by `scripts/evaluate_edit_lanes.py` and this report builder. The cache and training checkpoints resume locally. `config.json`, dataset manifest, training history, evaluation JSON, lane predictions and source hashes accompany this report. All code and research artifacts remain local.','',
        'See [methodology](distillation_methodology.md), [security evaluation boundaries](security_evaluation_boundary.md), and [package compatibility](package_compatibility.md). Primary methodology references: [progressive distillation](https://arxiv.org/abs/2202.00512), [LCM](https://arxiv.org/abs/2310.04378), [official Diffusers training documentation](https://huggingface.co/docs/diffusers/training/lcm_distill). Security reference: [DBALD author paper](https://arxiv.org/html/2508.15778v2).']
    (ROOT/'docs/distillation_report.md').write_text('\n'.join(report),encoding='utf-8')
    fig,ax=plt.subplots(figsize=(7,4)); ax.plot([r['step'] for r in history],[r['validation_rollout_mse'] for r in history],marker='o'); ax.set(xlabel='Optimizer updates',ylabel='Validation rollout latent MSE',title='Student validation on 16 separate scenes'); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(OUT/'validation_curve.png',dpi=150); plt.close(fig)
    selected=['scene_080','scene_083','scene_088','scene_091']; canvas=Image.new('RGB',(1024,4*284),'white'); draw=ImageDraw.Draw(canvas)
    for j,s in enumerate(selected):
        for i,name in enumerate(['original_crop','teacher20_100_raw','teacher4_100_raw','student4_100_raw']):
            im=Image.open(OUT/'evaluation'/s/(name+'.png')); im.thumbnail((256,256)); canvas.paste(im,(i*256,j*284+28)); draw.text((i*256+4,j*284+7),s+' '+name,fill='black')
    canvas.save(OUT/'raw_comparison_gallery.png')
    patches=Image.new('RGB',(1024,4*284),'white'); draw=ImageDraw.Draw(patches)
    for j,s in enumerate(selected):
        box=(176,336,336,496) if int(s[-3:])<88 else (224,264,384,424)
        for i,name in enumerate(['original_crop','teacher20_100_raw','teacher4_100_raw','student4_100_raw']):
            im=Image.open(OUT/'evaluation'/s/(name+'.png')).crop(box).resize((256,256))
            patches.paste(im,(i*256,j*284+28)); draw.text((i*256+4,j*284+7),s+' '+name,fill='black')
    patches.save(OUT/'texture_comparison_gallery.png')
    files=[ROOT/'scripts'/name for name in ('distill_road.py','evaluate_edit_lanes.py','report_distillation.py','prepare_distillation_data.py','validate_distillation_inputs.py','audit_distillation_outputs.py')]+[ROOT/'src/lane_repro/road_teacher.py',ROOT/'src/lane_repro/distillation.py']
    (OUT/'source_hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2))
    print(ROOT/'docs/distillation_report.md')
if __name__=='__main__': main()

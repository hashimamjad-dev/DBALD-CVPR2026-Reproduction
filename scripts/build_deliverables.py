"""Build local reports and figures only from completed measurement artifacts."""
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'tmp/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def csv_write(path,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=keys); writer.writeheader(); writer.writerows(rows)

def main():
    results=ROOT/'results'; docs=ROOT/'docs'; plots=results/'plots'; plots.mkdir(exist_ok=True)
    reference=results/'reference/table1.csv'; published=results/'published_table1.csv'
    if published.exists() and published.read_bytes()!=reference.read_bytes(): raise ValueError('Published table differs')
    if not published.exists(): shutil.copy2(reference,published)
    rows=[]
    for summary in sorted((results/'overnight').glob('*/*/summary.json')):
        if not (summary.parent/'completed.json').exists(): continue
        for item in json.loads(summary.read_text()):
            rows.append({k:v for k,v in item.items() if k not in ('command','stdout','stderr')}|{'evidence':str(summary.relative_to(ROOT))})
    if rows: csv_write(results/'measured_results.csv',rows)
    matrix=[]
    for reference_row in csv.DictReader(reference.open()):
        for dataset,prefix in [('TuSimple','tusimple'),('CULane','culane')]:
            candidates=[r for r in rows if r['model']==reference_row['model'] and r['dataset']==dataset and r['scope']=='full_test' and r['transformation']=='clean']
            matrix.append(dict(model=reference_row['model'],dataset=dataset,attack=reference_row['attack'],
                published_clean_acc=reference_row[prefix+'_benign_acc'],published_poisoned_clean_acc=reference_row[prefix+'_poisoned_acc'],
                published_lda_asr=reference_row[prefix+'_lda_asr'],published_loa_asr=reference_row[prefix+'_loa_asr'],
                published_lra_asr=reference_row.get(prefix+'_lra_asr','not reported'),published_avg_asr=reference_row[prefix+'_avg_asr'],
                independently_measured_equivalent_acc='',independently_measured_asr='',
                independent_full_benchmark_evidence='; '.join(r['evidence'] for r in candidates),
                requirements='ResNet-34, author training/checkpoint/split/scoring equivalence; poisoned checkpoints and targets for ASR',
                deviation_reason='Public clean checkpoint differs; '+('official TuSimple metric, exact paper equivalence unresolved' if dataset=='TuSimple' else 'official lane F1 differs from paper point ACC'),
                missing_details='Author checkpoints, repeat seeds, full training configuration, exact ASR targets/matching/aggregation',
                status='no exact Table 1 reproduction claimed'))
    csv_write(results/'table1_reproducibility.csv',matrix)
    robust=[r for r in rows if '/robustness/' in r['evidence'].replace('\\','/')]
    if robust:
        names=list(dict.fromkeys(r['transformation'] for r in robust)); models=['LaneATT','ADNet','SCNN','RESA']
        fig,ax=plt.subplots(figsize=(11,5))
        for model in models:
            values={r['transformation']:r['acc_percent'] for r in robust if r['model']==model}
            ax.plot(range(len(names)),[values.get(n,np.nan) for n in names],marker='o',label=model)
        ax.set_xticks(range(len(names)),[n.replace('_',' ') for n in names],rotation=20,ha='right')
        ax.set_ylabel('Official TuSimple ACC (%)'); ax.set_title('Benign robustness — fixed 16-image subset (not ASR)')
        ax.grid(alpha=.25); ax.legend(ncol=4); fig.tight_layout(); fig.savefig(plots/'robustness.png',dpi=180); plt.close(fig)
    full=[r for r in rows if r['scope']=='full_test' and r['transformation']=='clean']
    for dataset in ('TuSimple','CULane'):
        selected=[r for r in full if r['dataset']==dataset]
        if not selected: continue
        fig,axes=plt.subplots(1,3,figsize=(12,4))
        labels=[r['model']+' '+r.get('backbone','configured').replace('resnet','R') for r in selected]
        metrics=[('acc_percent' if dataset=='TuSimple' else 'f1_percent','ACC (%)' if dataset=='TuSimple' else 'Lane F1 (%)'),('median_ms','Median inference (ms)'),('peak_cuda_allocated_bytes','Peak allocated GPU (MiB)')]
        for ax,(key,title) in zip(axes,metrics):
            ax.bar(labels,[r[key]/(1024**2 if key.endswith('bytes') else 1) for r in selected]); ax.set_title(title); ax.tick_params(axis='x',rotation=40)
        fig.suptitle(f'{dataset} full test — independent public clean checkpoints'); fig.tight_layout(); fig.savefig(plots/f'{dataset.lower()}_full.png',dpi=180); plt.close(fig)
    header='| Dataset / scope | Model / backbone | Images | Metric | Value (%) | Median ms | Peak MiB |\n|---|---|---:|---|---:|---:|---:|\n'
    lines=[]
    for r in rows:
        if r['transformation']!='clean': continue
        key='acc_percent' if r['dataset']=='TuSimple' else 'f1_percent'
        lines.append(f"| {r['dataset']} / {r['scope']} | {r['model']} / {r.get('backbone','see config')} | {r['images']} | {'ACC' if key=='acc_percent' else 'lane F1'} | {r[key]:.4f} | {r['median_ms']:.2f} | {r['peak_cuda_allocated_bytes']/1024**2:.2f} |")
    table=header+'\n'.join(lines)+'\n'
    test=results/'overnight/tests.xml'; testtext='No final test report available.'
    if test.exists():
        suites=ET.parse(test).getroot().findall('testsuite')
        testtext='; '.join(f"{s.attrib['tests']} tests, {s.attrib.get('failures','0')} failures, {s.attrib.get('errors','0')} errors, {s.attrib.get('skipped','0')} skipped" for s in suites)
    validation=results/'overnight/model_validation.json'
    batch=json.loads(validation.read_text()) if validation.exists() else []
    audit=results/'overnight/prediction_audit.json'; audits=json.loads(audit.read_text()) if audit.exists() else []
    diffusion=results/'overnight/diffusion/measurements.json'
    diffusiontext='Ordinary diffusion demonstration completed; see results/overnight/diffusion/measurements.json.' if diffusion.exists() else 'Ordinary diffusion demonstration has not yet produced verified measurements.'
    current=datetime.datetime.now(datetime.timezone.utc).isoformat()
    summary=f'''# Experiment summary

Generated from completed artifacts at {current}. No incomplete run is counted as completed.

{table}
The 16-image experiments are subsets. Only rows marked full_test cover the complete split. CULane lane F1 is not Table 1 ACC. All ASR fields remain null. These public checkpoints do not establish exact Table 1 equivalence.

Tests: {testtext}. Two-image tensor-batch validation passed for {sum(bool(r['passed']) for r in batch)} of {len(batch)} models. Independently audited TuSimple model/condition submissions: {len(audits)}.

{diffusiontext}

Evidence: measured_results.csv, table1_reproducibility.csv, per-run predictions/provenance/hardware logs under overnight/, and plots/. published_table1.csv is a byte-identical copy of reference/table1.csv (SHA256 {hashlib.sha256(reference.read_bytes()).hexdigest()}).
'''
    (results/'experiment_summary.md').write_text(summary,encoding='utf-8')
    (docs/'reproduction_report.md').write_text(summary.replace('# Experiment summary','# Reproduction report',1)+'''
## Interpretation

The infrastructure establishes executable public-checkpoint baselines, metric validity checks and provenance. The clean TuSimple metric is a useful benchmark comparison, but exact paper equivalence remains unresolved even when a backbone matches. CULane results use the dataset's official lane F1 and cannot numerically reproduce the paper's point ACC. Poisoned-model clean ACC and all attack-specific ASRs remain unmeasured.

See methodology.md for preprocessing, timing and metric definitions, experimental_limitations.md for gaps, and overnight_execution_log.md plus machine logs for execution evidence. Earlier verified baseline directories remain intact.

## Three next research actions

1. Resolve author checkpoint/training/scoring equivalence, including the SCNN ResNet-34 checkpoint and CULane point-ACC implementation.
2. Complete any split/model whose completion marker is absent; compare portable NMS/DCNv1 and OpenCV rasterization against original environments.
3. Extend the fixed benign robustness suite to the full test split and repeated settings, keeping these experiments separate from ASR.
''',encoding='utf-8')
    (docs/'professor_briefing.md').write_text('''# Professor briefing

## Research objective

The project investigates which results in DBALD Table 1 can be independently supported. A working detector is the first requirement; reproducing an attack result additionally requires the same poisoned model, target definition and evaluation protocol. We do not infer attack success from ordinary detection errors.

## Architecture differences

LaneATT refines candidate lane anchors and uses attention to combine their information. ADNet adapts anchors and samples features with deformable convolution. SCNN predicts lane segmentation and passes information across image rows and columns. RESA also predicts segmentation, but aggregates context through recurrent feature shifts. Their outputs are converted to native image coordinates before dataset-specific scoring.

The initial Windows baselines use ResNet-18. The author's extended Appendix C specifies ResNet-34 for every model. Matching that backbone is necessary but insufficient: public weights can still differ from the authors' training runs. The preserved baseline results are therefore not presented as exact reproduction.

## Verified measurements

'''+table+f'''
Tests: {testtext}. All raw predictions and measured runtime/memory records are retained locally. Only full_test rows represent a complete benchmark; subset rows serve debugging and robustness checks.

## Data and hardware

TuSimple evaluates labelled 1280×720 final frames using its official accuracy scorer. CULane evaluates 1640×590 images using matched-lane IoU/F1. These scores are not interchangeable. Download integrity, image counts, label hashes and checkpoint hashes are recorded. CULane comes with non-commercial and redistribution restrictions.

The machine has an RTX 5050 Laptop GPU with about 8 GB VRAM. PyTorch with CUDA 12.8 supports its Blackwell architecture. A separate CUDA environment on C: reuses dependencies from the project environment on D:. Benchmarks use batch size one; real batch-size-two validation checks output consistency. Logged peak allocations measure inference, not training capacity. Temperature and storage guards protect long runs.

## Implementation decisions and research gaps

Windows compatibility required portable lane suppression, direct configuration loading and a torchvision deformable-convolution adapter. The original CULane C++ metric is compiled locally with modern OpenCV and without its optional GUI. These changes are documented and tested, but complete equivalence to the authors' original software stack remains open.

Prompted SAM2 masks, clean sensitivity maps, candidate-window statistics and fixed-mask image editing are isolated methodology demonstrations. {diffusiontext} Image similarity is measured after editing; it is not optimized for attack evasion. No model backdoors or poisoned labels are constructed.

No Table 1 ASR, poisoned-model clean ACC, or exact paper reproduction is claimed. Priorities are author artifact/protocol clarification, completing available clean benchmarks, and broader controlled robustness studies. All code, data and results remain local; nothing has been pushed or published.
''',encoding='utf-8')
    files=sorted(str(p.relative_to(ROOT)) for parent in ('scripts','src/lane_repro','tests','configs','docs','results') for p in (ROOT/parent).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    (results/'deliverable_inventory.txt').write_text('\n'.join(files)+'\n',encoding='utf-8')
    print(f'Reported {len(rows)} completed measurement rows and {len(matrix)} published model/dataset/attack entries.')

if __name__=='__main__': main()

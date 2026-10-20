"""Frozen clean-detector assessment of benign restorations, never ASR."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
from PIL import Image
from lane_repro.models import LaneATTAdapter,sample_predictions
from lane_repro.evaluation import geometry,official_tusimple
from lane_repro.culane_official import prediction_text,score
from distill_road import ROOT,OUT,thermal_guard

def main():
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    samples=json.loads((ROOT/'datasets/distillation/manifest.json').read_text())['samples']
    measured=json.loads((OUT/'evaluation.json').read_text())
    rows=[]; metrics=[]; provenance={}; evaluator=official_tusimple()
    for dataset in ('TuSimple','CULane'):
        selected=[r for r in samples if r['partition']=='evaluation' and r['dataset']==dataset]
        ds=dataset.lower(); epoch='0100' if ds=='tusimple' else '0015'
        checkpoint=ROOT/f'models/checkpoints/laneatt_official/experiments/laneatt_r34_{ds}/models/model_{epoch}.pt'
        model=LaneATTAdapter(checkpoint,dataset=ds,backbone='resnet34',device='cuda')
        provenance[dataset]=model.provenance()
        clean={}; allpred={}
        for r in selected:
            thermal_guard()
            image=np.asarray(Image.open(ROOT/r['root']/r['raw_file']).convert('RGB')); h,w=image.shape[:2]
            ys=r['h_samples'] if ds=='tusimple' else list(range(590))
            lanes=model.predict(image); clean[r['id']]=sample_predictions(lanes,ys,w,h)
            variants=[('clean',0,image)]+[(m,s,np.asarray(Image.open(OUT/'evaluation'/r['id']/f'{m}_{s}_composite.png'))) for m in ('teacher20','teacher4','student4') for s in (100,101)]
            for method,seed,image in variants:
                lanes=model.predict(image); pred=sample_predictions(lanes,ys,w,h)
                item=dict(id=r['id'],dataset=dataset,method=method,seed=seed,prediction=pred,
                    clean_prediction_consistency=geometry(pred,clean[r['id']]),asr=None)
                if ds=='tusimple':
                    # Fixed zero runtime isolates lane quality; not a detector speed benchmark.
                    acc,fp,fn=evaluator.bench(pred,r['lanes'],ys,0)
                    item.update(official_tusimple_accuracy=acc,fp=fp,fn=fn)
                else:
                    directory=OUT/'lane_predictions'/f'{method}_{seed}'
                    path=(directory/r['raw_file']).with_suffix('.lines.txt'); path.parent.mkdir(parents=True,exist_ok=True)
                    path.write_text(prediction_text(lanes,ys))
                rows.append(item)
            print(dataset,r['id'],flush=True)
        if ds=='culane':
            listing=OUT/'culane_edit_subset.txt'; listing.write_text('\n'.join('/'+r['raw_file'].lstrip('/') for r in selected)+'\n')
            for method,seed in [('clean',0)]+[(m,s) for m in ('teacher20','teacher4','student4') for s in (100,101)]:
                directory=OUT/'lane_predictions'/f'{method}_{seed}'
                result=score(directory,ROOT/'datasets/raw/culane',listing,OUT/f'culane_{method}_{seed}.txt')
                metrics.append(dict(method=method,seed=seed,**result))
        del model
        torch.cuda.empty_cache()
    (OUT/'lane_consistency.json').write_text(json.dumps(dict(rows=rows,culane_metrics=metrics,model_provenance=provenance,
        limitation='Eight scenes per dataset. Clean-checkpoint benign editing assessment, not ASR or a full benchmark. No detector used for training/selection.'),indent=2))
if __name__=='__main__': main()

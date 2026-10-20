"""Frozen detector assessment after validation-only selection; never ASR."""
from v2_common import *
from lane_repro.models import LaneATTAdapter,sample_predictions
from lane_repro.evaluation import geometry,official_tusimple
from lane_repro.culane_official import prediction_text,score

manifest=setup(); lock=json.loads((OUT/'selection.json').read_text())
assert len(json.loads((OUT/'final_quality.json').read_text()))==128
old=json.loads((V1/'lane_consistency.json').read_text()); evaluator=official_tusimple()
rows=[]; metrics=[]; provenance={}
for dataset in ('TuSimple','CULane'):
    selected=[r for r in manifest['test'] if r['dataset']==dataset]
    ds=dataset.lower(); epoch='0100' if ds=='tusimple' else '0015'
    checkpoint=ROOT/f'models/checkpoints/laneatt_official/experiments/laneatt_r34_{ds}/models/model_{epoch}.pt'
    model=LaneATTAdapter(checkpoint,dataset=ds,backbone='resnet34',device='cuda'); provenance[dataset]=model.provenance()
    for r in selected:
        guard(); image=np.asarray(Image.open(ROOT/r['root']/r['raw_file']).convert('RGB')); h,w=image.shape[:2]
        ys=r['h_samples'] if ds=='tusimple' else list(range(590))
        clean=next(x['prediction'] for x in old['rows'] if x['id']==r['id'] and x['method']=='clean')
        # Confirm cached reference predictions still match this frozen runtime.
        current=sample_predictions(model.predict(image),ys,w,h)
        if current!=clean: raise RuntimeError('Clean detector cache differs; investigate before comparison')
        for seed in (100,101):
            image=np.asarray(Image.open(OUT/'evaluation'/r['id']/f'student_v2_{seed}_composite.png'))
            lanes=model.predict(image); pred=sample_predictions(lanes,ys,w,h)
            item=dict(id=r['id'],dataset=dataset,method='student_v2',seed=seed,prediction=pred,clean_prediction_consistency=geometry(pred,clean),asr=None)
            if ds=='tusimple':
                acc,fp,fn=evaluator.bench(pred,r['lanes'],ys,0); item.update(official_tusimple_accuracy=acc,fp=fp,fn=fn)
            else:
                path=(OUT/'lane_predictions'/f'student_v2_{seed}'/r['raw_file']).with_suffix('.lines.txt'); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(prediction_text(lanes,ys))
            rows.append(item)
        print(dataset,r['id'],flush=True)
    if ds=='culane':
        listing=OUT/'culane_edit_subset.txt'; listing.write_text('\n'.join('/'+r['raw_file'].lstrip('/') for r in selected)+'\n')
        for seed in (100,101):
            result=score(OUT/'lane_predictions'/f'student_v2_{seed}',ROOT/'datasets/raw/culane',listing,OUT/f'culane_student_v2_{seed}.txt'); metrics.append(dict(method='student_v2',seed=seed,**result))
    del model; torch.cuda.empty_cache()
write(OUT/'lane_consistency.json',dict(rows=rows,culane_metrics=metrics,model_provenance=provenance,selected_checkpoint_sha256=lock['selected']['sha256'],limitation='Eight reused test scenes per dataset, two noises; benign composite robustness, not ASR or full CULane. GT lane protection is imposed by mask.'))

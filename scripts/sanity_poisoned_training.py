"""TINY-SCALE sanity check: does poisoning + fine-tuning + ASR measurement work end to end, with
real gradients, a real trigger-containing image, and the real official scorer? NOT a Table 1
reproduction and not evidence of attack efficacy at any meaningful scale. Real training needs the
GPU, the full ~3,600-image TuSimple train split, 100 epochs (per the paper) and a proper train/val
split; this uses the 16-image smoke subset only (a handful for fine-tuning, the rest held out),
specifically to prove the mechanism -- that a poisoned image fed through real training genuinely
moves the model's predictions toward the attack target -- before any of that is attempted at scale.

Everything here is real: the poisoned image (poison_pipeline, placeholder trigger), the poisoned
training target (laneatt_heatmap.build_target against the attack's target label, not the clean
one), real backward passes through LaneATT's own model.loss(), and real ASR measurement via
lane_metrics.asr_loa_lra (the vendored official evaluator). The two deliberate placeholders already
documented in poison_pipeline.py (trigger realism, and placement scoring without --checkpoint) still
apply unchanged. Fine-tunes an in-memory copy of the model; never writes back to
models/checkpoints/.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from lane_repro.data import tusimple
from lane_repro.gpu_guard import cool
from lane_repro.lane_metrics import asr_loa_lra
from lane_repro.laneatt_heatmap import build_target,laneatt_attention_map
from lane_repro.models import LaneATTAdapter,sample_predictions
from lane_repro.poison_pipeline import poison_sample

def predict_xs(adapter,rgb,sample):
    adapter.model.eval()
    lanes=adapter.predict(rgb)
    return sample_predictions(lanes,sample.h_samples,sample.width,sample.height)

def score(adapter,samples,poisoned_images,poisoned_targets_xs,clean_samples):
    """Returns (mean clean ACC on held-out clean samples, mean ASR on the poisoned training images)."""
    clean_scores=[asr_loa_lra(predict_xs(adapter,s.read(),s),s.lanes,s.h_samples) for s in clean_samples]
    asr_scores=[asr_loa_lra(predict_xs(adapter,img,s),target,s.h_samples)
        for s,img,target in zip(samples,poisoned_images,poisoned_targets_xs)]
    return sum(clean_scores)/len(clean_scores),sum(asr_scores)/len(asr_scores)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',default='models/checkpoints/laneatt_official/experiments/laneatt_r18_tusimple/models/model_0100.pt')
    p.add_argument('--backbone',default='resnet18')
    p.add_argument('--root',default='datasets/raw/tusimple_subset')
    p.add_argument('--labels',default='datasets/raw/tusimple_subset/labels.jsonl')
    p.add_argument('--attack',default='LOA',choices=['LOA','LRA'],help='LDA has no official-evaluator ASR; see lane_metrics.dataset_lda_asr instead')
    p.add_argument('--train-count',type=int,default=12)
    p.add_argument('--poison-fraction',type=float,default=1.0,
        help='Fraction of --train-count images that are poisoned; the rest train on their own clean '
             'image+label. Default 1.0 (all poisoned) maximizes the chance of seeing a mechanism '
             'effect at this tiny scale; DBALD itself uses 10%% on ~3,600 images, a very different '
             'regime this does not attempt to reproduce.')
    p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--beta',type=float,default=60.)
    p.add_argument('--alpha-degrees',type=float,default=9.)
    p.add_argument('--use-real-heatmap',action='store_true')
    args=p.parse_args()
    if not 0<args.poison_fraction<=1: raise ValueError('--poison-fraction must be in (0, 1]')

    samples=list(tusimple(args.root,args.labels))
    if len(samples)<=args.train_count: raise ValueError('Need samples left over for the held-out clean set')
    train_samples,held_out=samples[:args.train_count],samples[args.train_count:]
    poison_count=round(args.poison_fraction*len(train_samples))

    adapter=LaneATTAdapter(args.checkpoint,backbone=args.backbone,device='cpu')
    poison_adapter=adapter if args.use_real_heatmap else None

    poisoned_images,poisoned_targets_xs,poisoned_samples,train_targets=[],[],[],[]
    for i,s in enumerate(train_samples):
        if i<poison_count:
            lanes_xy=[list(zip(xs,s.h_samples)) for xs in s.lanes] if poison_adapter else None
            attention=laneatt_attention_map(poison_adapter,s.read(),lanes_xy,args.attack) if poison_adapter else None
            result=poison_sample(s,args.attack,beta=args.beta,alpha_degrees=args.alpha_degrees,attention_map=attention)
            poisoned_images.append(result['poisoned_image']); poisoned_targets_xs.append(result['poisoned_lanes'])
            poisoned_samples.append(s)
            target_points=[list(zip(xs,s.h_samples)) for xs in result['poisoned_lanes']]
            train_targets.append((result['poisoned_image'],build_target(target_points,s.width,s.height,max_lanes=adapter.max_lanes)))
        else:
            clean_points=[list(zip(xs,s.h_samples)) for xs in s.lanes]
            train_targets.append((s.read(),build_target(clean_points,s.width,s.height,max_lanes=adapter.max_lanes)))
    print(f'{poison_count}/{len(train_samples)} training images poisoned with {args.attack}',flush=True)

    before_acc,before_asr=score(adapter,poisoned_samples,poisoned_images,poisoned_targets_xs,held_out)
    print(f'Before fine-tuning: held-out clean ACC={before_acc:.4f}  poisoned-image ASR={before_asr:.4f} (n_poisoned={poison_count})',flush=True)

    optimizer=torch.optim.Adam(adapter.model.parameters(),lr=args.lr)
    adapter.model.train()
    history=[]
    for epoch in range(args.epochs):
        cool()
        epoch_loss=0.
        for img,target in train_targets:
            x=adapter.preprocess(img)
            proposals_list=adapter.model(x)
            loss,_=adapter.model.loss(proposals_list,[target])
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            epoch_loss+=loss.detach().item()
        history.append(epoch_loss/len(train_targets))
        print(f'epoch {epoch+1}/{args.epochs} mean_loss={history[-1]:.4f}',flush=True)

    after_acc,after_asr=score(adapter,poisoned_samples,poisoned_images,poisoned_targets_xs,held_out)
    print(f'After fine-tuning:  held-out clean ACC={after_acc:.4f}  poisoned-image ASR={after_asr:.4f} (n_poisoned={poison_count})',flush=True)

    out=Path('results/sanity_poisoned_training.json')
    out.write_text(json.dumps(dict(
        scope='TINY-SCALE mechanism sanity check, not a Table 1 reproduction',
        checkpoint=args.checkpoint,attack=args.attack,train_count=args.train_count,poison_count=poison_count,
        held_out_count=len(held_out),epochs=args.epochs,lr=args.lr,
        attention_source='real' if args.use_real_heatmap else 'uniform placeholder',
        before=dict(clean_acc=before_acc,asr=before_asr),after=dict(clean_acc=after_acc,asr=after_asr),
        loss_history=history),indent=2))
    print('Wrote',out)

if __name__=='__main__': main()

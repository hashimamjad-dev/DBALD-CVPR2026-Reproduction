"""Independent safe demonstrations on one raw image, no training or poisoning."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import cv2
import numpy as np
import torch
from lane_repro.models import LaneATTAdapter
from lane_repro.prototype import road_sam2,windows,inpaint,consistency,sensitivity,window_statistics

def main():
    p=argparse.ArgumentParser(); p.add_argument('--device',default='cpu'); p.add_argument('--output',default='results/prototype'); args=p.parse_args()
    torch.set_num_threads(4)
    root=Path(__file__).resolve().parents[1]; out=root/args.output; out.mkdir(parents=True,exist_ok=True)
    sample=json.loads((root/'datasets/raw/tusimple_subset/labels.jsonl').read_text().splitlines()[0])
    bgr=cv2.imread(str(root/'datasets/raw/tusimple_subset'/sample['raw_file'])); rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    # Explicit semantic prompts, not automatic claims of road understanding.
    points=[[640,600],[450,520],[1000,500],[640,100],[100,250]]; labels=[1,1,1,0,0]
    road,score=road_sam2(rgb,root/'models/checkpoints/sam2.1_hiera_tiny.pt',points,labels,args.device)
    cv2.imwrite(str(out/'road_mask.png'),road.astype(np.uint8)*255)
    boxes=windows(road,size=(40,40),stride=80)
    view=bgr.copy()
    for x,y,w,h in boxes: cv2.rectangle(view,(x,y),(x+w-1,y+h-1),(0,220,100),1)
    cv2.imwrite(str(out/'candidate_windows.png'),view)
    # Fixed demonstration region; never selected using gradients or optimized.
    mask=np.zeros(road.shape,dtype=bool); mask[550:590,600:640]=True
    edited=inpaint(rgb,mask); cv2.imwrite(str(out/'ordinary_inpaint.png'),cv2.cvtColor(edited,cv2.COLOR_RGB2BGR))
    lane_mask=np.zeros(road.shape,np.uint8)
    for lane in sample['lanes']:
        pts=np.array([[x,y] for x,y in zip(lane,sample['h_samples']) if x>=0],dtype=np.int32)
        if len(pts)>1: cv2.polylines(lane_mask,[pts],False,1,5)
    detector=LaneATTAdapter(root/'models/checkpoints/laneatt_official/experiments/laneatt_r18_tusimple/models/model_0100.pt',device=args.device)
    x=detector.preprocess(rgb)
    # Analyze clean feature energy, explicitly NOT the paper's attack-specific losses.
    features=torch.nn.Sequential(detector.model.feature_extractor,detector.model.conv1)
    heat=sensitivity(features,x,lambda y:y.square().mean())[0].cpu().numpy()
    heat=heat/(heat.max()+1e-12)
    heat=cv2.resize(heat,(rgb.shape[1],rgb.shape[0]))
    np.save(out/'clean_heatmap.npy',heat)
    (out/'window_statistics.json').write_text(json.dumps(window_statistics(heat,boxes),indent=2))
    cv2.imwrite(str(out/'clean_feature_sensitivity.png'),cv2.applyColorMap((heat*255).astype(np.uint8),cv2.COLORMAP_TURBO))
    report=dict(sample=sample['raw_file'],sam2_version='SAM2.1 tiny (engineering choice, not paper-specified)',
        prompts=points,prompt_labels=labels,sam2_mask_score=score,road_pixels=int(road.sum()),window_count=len(boxes),
        window_stride=80,inpaint_region=[600,550,40,40],inpainting='OpenCV Telea radius 3; fixed region',
        sensitivity_objective='clean backbone feature squared mean, not DBALD attack losses',
        lane_consistency=consistency(rgb,edited,lane_mask.astype(bool)),environment_consistency=consistency(rgb,edited,~mask),
        asr=None,optimization_steps=0)
    (out/'measurements.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))

if __name__=='__main__': main()

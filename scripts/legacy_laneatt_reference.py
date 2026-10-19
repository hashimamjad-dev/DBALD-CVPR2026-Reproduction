"""Original LaneATT source on Python 3.8 / torch 1.6; save pre-NMS outputs.

Run with tools/python38/python.exe. No compatibility adapter or modern anchor
permutation is injected: the actual original torch runtime determines ordering.
The portable NMS import is only a placeholder; forward returns proposals before
NMS, allowing a comparison independent of the uncompiled CUDA extension.
"""
import hashlib
import json
from pathlib import Path
import sys
import time
import types
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import cv2
import numpy as np
import scipy
import sklearn
import torch
import torchvision
import imgaug.augmenters as iaa

def main():
    assert torch.__version__.startswith('1.6.0'), torch.__version__
    torch.set_num_threads(4); torch.manual_seed(0)
    repo=ROOT/'third_party/LaneATT'
    for name,path in [('lib',repo/'lib'),('lib.models',repo/'lib/models')]:
        module=types.ModuleType(name); module.__path__=[str(path)]; sys.modules[name]=module
    from lane_repro.nms import lane_nms
    module=types.ModuleType('nms'); module.nms=lane_nms; sys.modules['nms']=module
    from lib.models.laneatt import LaneATT
    out=ROOT/'results/overnight/laneatt_legacy_reference'; out.mkdir(exist_ok=True)
    report={'python':sys.version,'torch':torch.__version__,'torchvision':torchvision.__version__,
        'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'opencv':cv2.__version__,
        'device':'cpu','raw_forward_before_nms':True,'samples':[]}
    for dataset in ('tusimple','culane'):
        freq=torch.load(str(repo/'data'/('{}'+'_anchors_freq.pt').format(dataset))).cpu()
        actual=torch.argsort(freq,descending=True).numpy()
        expected=np.array(json.loads((ROOT/'configs/laneatt_anchor_order.json').read_text())['datasets'][dataset]['indices'])
        assert np.array_equal(actual,expected), 'Extracted reference differs from actual torch 1.6'
        report[dataset+'_anchor_order_matches']=True
    checkpoint=ROOT/'models/checkpoints/laneatt_official/experiments/laneatt_r34_tusimple/models/model_0100.pt'
    model=LaneATT(backbone='resnet34',pretrained_backbone=False,S=72,img_w=640,img_h=360,topk_anchors=1000,
        anchors_freq_path=str(repo/'data/tusimple_anchors_freq.pt'))
    state=torch.load(str(checkpoint),map_location='cpu'); state=state.get('model',state)
    model.load_state_dict(state,strict=True); model.eval()
    model.nms=lambda proposals,*args: proposals
    resize=iaa.Resize({'height':360,'width':640})
    labels=[json.loads(line) for line in (ROOT/'datasets/raw/tusimple_subset/labels.jsonl').read_text().splitlines()]
    for i,row in enumerate(labels):
        image=cv2.imread(str(ROOT/'datasets/raw/tusimple_subset'/row['raw_file']))
        resized=resize(image=image)
        x=torchvision.transforms.ToTensor()((resized/255.).astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            started=time.perf_counter(); proposals=model(x); elapsed=time.perf_counter()-started
        np.savez_compressed(out/('%02d.npz'%i),input=x.numpy(),proposals=proposals.numpy())
        report['samples'].append({'raw_file':row['raw_file'],'seconds':elapsed,'artifact':'%02d.npz'%i})
        print('Original torch 1.6 reference {}/{}'.format(i+1,len(labels)),flush=True)
    report['checkpoint_sha256']=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    (out/'summary.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__': main()

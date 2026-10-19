"""Compare legacy anchor ordering using the unmodified PyTorch 1.6 sort body.

No thresholds or labels are used to choose an ordering. The C++ reference is
extracted from the downloaded, licensed upstream source, not reimplemented.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
from lane_repro.models import LaneATTAdapter,sample_predictions
from lane_repro.data import tusimple

def main():
    folder=Path('tools/pytorch16_sort'); source=(folder/'THTensorMoreMath.cpp').read_text()
    macros=source[source.index('#define MAX_LEVELS'):source.index('static void THTensor_(quicksortascend)')]
    body=source[source.index('static void THTensor_(quicksortdescend)'):source.index('#undef MAX_LEVELS')]
    cpp='#include <cmath>\n#include <cstdint>\n#include <iostream>\n#include <vector>\n#define th_isnan std::isnan\n#define THTensor_(x) x\nusing scalar_t = double;\n'+macros+body+'''
int main() {
 std::vector<double> a; double v; while(std::cin >> v) a.push_back(v);
 std::vector<int64_t> idx(a.size()); for(size_t i=0;i<a.size();++i) idx[i]=i;
 quicksortdescend(a.data(),idx.data(),a.size(),1);
 for(auto i:idx) std::cout << i << "\\n";
}
'''
    (folder/'reference.cpp').write_text(cpp)
    subprocess.run(['C:/msys64/ucrt64/bin/g++.exe','-O2','-std=c++17','-static',str(folder/'reference.cpp'),'-o',str(folder/'reference.exe')],check=True)
    torch.set_num_threads(4)
    output=Path('results/overnight/laneatt_order_diagnostic'); output.mkdir(exist_ok=True)
    reports=[]
    for dataset in ('tusimple','culane'):
        freq=torch.load(f'third_party/LaneATT/data/{dataset}_anchors_freq.pt',weights_only=True).cpu()
        current=torch.argsort(freq,descending=True).numpy()
        result=subprocess.run([str(folder/'reference.exe')],input='\n'.join(map(str,freq.tolist())),text=True,capture_output=True,check=True)
        legacy=np.array([int(v) for v in result.stdout.split()],dtype=np.int64)
        assert np.array_equal(np.sort(legacy),np.arange(len(freq)))
        assert np.array_equal(freq[current].numpy(),freq[legacy].numpy())
        np.save(output/f'{dataset}_torch16_order.npy',legacy)
        reports.append(dict(dataset=dataset,anchors=len(freq),changed_top1000_positions=int(np.sum(current[:1000]!=legacy[:1000])),
            changed_top1000_members=len(set(current[:1000])-set(legacy[:1000])),
            reference='PyTorch v1.6.0 THTensorMoreMath.cpp quicksortdescend',frequency_values_identical=True))
    # Compare coordinates only on the existing deterministic subset. No labels
    # or aggregate accuracy are consulted by this diagnostic.
    samples=list(tusimple('datasets/raw/tusimple_subset','datasets/raw/tusimple_subset/labels.jsonl'))
    adapter=LaneATTAdapter('models/checkpoints/laneatt_official/experiments/laneatt_r34_tusimple/models/model_0100.pt',backbone='resnet34',device='cuda',anchor_order='current')
    before=[sample_predictions(adapter.predict(s.read()),s.h_samples,s.width,s.height) for s in samples]
    anchors,cuts=adapter.model.generate_anchors(lateral_n=72,bottom_n=128)
    order=torch.from_numpy(np.load(output/'tusimple_torch16_order.npy')[:1000])
    adapter.model.anchors=anchors[order]
    adapter.model.anchors_cut=cuts[order]
    adapter.model.cut_zs,adapter.model.cut_ys,adapter.model.cut_xs,adapter.model.invalid_mask=adapter.model.compute_anchor_cut_indices(adapter.model.anchor_feat_channels,adapter.model.fmap_w,adapter.model.fmap_h)
    adapter.model.to('cuda')
    after=[sample_predictions(adapter.predict(s.read()),s.h_samples,s.width,s.height) for s in samples]
    (output/'coordinate_comparison.json').write_text(json.dumps(dict(before=before,after=after)))
    (output/'summary.json').write_text(json.dumps(reports,indent=2)); print(json.dumps(reports,indent=2))

if __name__=='__main__': main()

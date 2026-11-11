import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from lane_repro.models import ROOT, checkpoint_hash
from lane_repro.scnn_autodrive import AutoDriveSCNNAdapter, config_values
from lane_repro.data import tusimple, culane


@pytest.mark.parametrize('dataset',['tusimple','culane'])
def test_public_r34_strict_load_and_native_reference_parity(dataset):
    assets=json.loads((ROOT/'configs/scnn_r34_autodrive.json').read_text())
    spec=assets['datasets'][dataset];checkpoint=ROOT/spec['checkpoint']
    if not checkpoint.exists():pytest.skip('Download the public trained R34 checkpoint first')
    assert checkpoint_hash(checkpoint)==spec['sha256']
    adapter=AutoDriveSCNNAdapter(checkpoint,dataset,'cpu')
    assert [len(adapter.model.backbone[name]) for name in ['layer1','layer2','layer3','layer4']]==[3,4,6,3]
    if dataset=='tusimple':sample=next(tusimple(ROOT/'datasets/raw/tusimple_test',ROOT/'datasets/raw/tusimple_test/labels.jsonl'))
    else:sample=next(culane(ROOT/'datasets/raw/culane',ROOT/'datasets/raw/culane/list/test.txt'))
    rgb=sample.read();x=adapter.preprocess(rgb);test=adapter.settings['test']
    height,width=test['input_size']
    resized=np.asarray(Image.fromarray(rgb).resize((width,height),resample=Image.Resampling.BILINEAR)).copy()
    reference=torch.from_numpy(resized.transpose(2,0,1)).float()/255
    reference=(reference-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]
    assert torch.equal(x,reference.unsqueeze(0))
    with torch.inference_mode():
        outputs=adapter.model(x)
        assert all(torch.isfinite(v).all() for v in outputs.values())
        actual=adapter.decode(outputs)
        expected=adapter.decoder.lane_as_segmentation_inference(adapter.model,x,
            (test['input_size'],test['original_size']),test['gap'],test['ppl'],test['thresh'],dataset,test['max_lane'])[0]
    assert actual==expected
    assert len(actual)>0
    receipt=dict(scope='one clean image; adapter/reference parity, not full-test accuracy',
                 raw_file=sample.key,provenance=adapter.provenance(),torch=torch.__version__,
                 mkldnn=torch.backends.mkldnn.enabled,input_shape=list(x.shape),
                 outputs={k:list(v.shape) for k,v in outputs.items()},predictions=actual,
                 preprocessing_identical=True,native_decode_identical=True)
    if os.environ.get('SCNN_VALIDATION_OUTPUT'):
        Path(os.environ['SCNN_VALIDATION_OUTPUT']).write_text(json.dumps(receipt,indent=2)+'\n')


def test_native_export_preserves_gaps_and_rejects_dataset_mismatch():
    adapter=object.__new__(AutoDriveSCNNAdapter);adapter.dataset='culane'
    points=[[[100.,589],[102.,549]]]
    assert adapter.culane_text(points).split()==['100.0','589','102.0','549']
    adapter.dataset='tusimple'
    with pytest.raises(ValueError):adapter.culane_text(points)
    with pytest.raises(ValueError):config_values('unknown')


@pytest.mark.parametrize('dataset',['tusimple','culane'])
def test_existing_benchmark_driver_accepts_public_r34_on_cpu(dataset,tmp_path,monkeypatch):
    sys.path.insert(0,str(ROOT/'scripts'))
    assets=json.loads((ROOT/'configs/scnn_r34_autodrive.json').read_text())
    checkpoint=ROOT/assets['datasets'][dataset]['checkpoint']
    if not checkpoint.exists():pytest.skip('Download public R34 checkpoint first')
    output=tmp_path/'smoke'
    if dataset=='tusimple':
        import benchmark as driver
        labels=tmp_path/'labels.jsonl'
        labels.write_text((ROOT/'datasets/raw/tusimple_test/labels.jsonl').read_text().splitlines()[0]+'\n')
        arguments=['--root',str(ROOT/'datasets/raw/tusimple_test'),'--labels',str(labels)]
    else:
        import benchmark_culane as driver
        listing=tmp_path/'test.txt'
        listing.write_text((ROOT/'datasets/raw/culane/list/test.txt').read_text().splitlines()[0]+'\n')
        arguments=['--root',str(ROOT/'datasets/raw/culane'),'--list',str(listing)]
    monkeypatch.setattr(sys,'argv',['benchmark','--model','SCNN','--backbone','resnet34',
        '--scnn-implementation','autodrive','--checkpoint',str(checkpoint),
        '--device','cpu','--cpu-threads','1','--expected-count','1','--output',str(output),*arguments])
    driver.main()
    summary=json.loads((output/'summary.json').read_text())[0]
    provenance=json.loads((output/'model_provenance.json').read_text())
    assert summary['images']==1 and summary['scope']=='subset' and summary['device']=='cpu'
    assert provenance['strict_state_dict'] and provenance['implementation']=='voldemortX/pytorch-auto-drive'
    assert provenance['checkpoint_sha256']==assets['datasets'][dataset]['sha256']
    if os.environ.get('SCNN_VALIDATION_OUTPUT'):
        Path(os.environ['SCNN_VALIDATION_OUTPUT']).write_text(json.dumps(dict(
            scope='one-image CPU driver integration; NOT Table 1 or full-test accuracy',
            summary=summary,provenance=provenance),indent=2)+'\n')

"""Strict adapter over the public PytorchAutoDrive SCNN R34 implementation.

Uses its actual backbone, heads, checkpoint keys and native point decoder.
It is a separately identified clean baseline, not a conversion to LaneDet.
"""
import ast
import copy
import importlib
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

from .models import ROOT, package, state_dict, checkpoint_hash, inference_mode

REPO=ROOT/'third_party/pytorch_auto_drive'
PREFIX='_scnn_autodrive'


def config_values(dataset):
    if dataset not in ('tusimple','culane'):raise ValueError('Unsupported SCNN dataset')
    path=REPO/f'configs/lane_detection/scnn/resnet34_{dataset}.py'
    tree=ast.parse(path.read_text())
    result={}
    # Only literal dict/list configuration expressions, never the imports or
    # upstream training/data initialization in the original configuration.
    allowed=(ast.Expression,ast.Dict,ast.List,ast.Tuple,ast.Constant,ast.Call,ast.Name,
             ast.Load,ast.keyword,ast.BinOp,ast.Sub,ast.Add,ast.UnaryOp,ast.USub)
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('model','test') for t in node.targets):
            for item in ast.walk(node.value):
                if not isinstance(item,allowed) or isinstance(item,ast.Name) and item.id!='dict':
                    raise ValueError('Unsupported upstream config expression')
            result[node.targets[0].id]=eval(compile(ast.Expression(node.value),str(path),'eval'),{'__builtins__':{},'dict':dict})
    if set(result)!= {'model','test'}:raise ValueError('Incomplete upstream SCNN config')
    return path,result


def upstream_modules():
    for suffix in ('','models','models.segmentation','models.common_models','models.common_models.heads',
                   'models.common_models.plugins','models.lane_detection'):
        name=PREFIX+('.'+suffix if suffix else '')
        package(name,REPO/'utils'/suffix.replace('.','/'))
    for suffix in ('models.backbone_wrappers','models.segmentation.segmentation','models.segmentation.deeplab',
                   'models.common_models.plugins.resa_reducer','models.common_models.plugins.spatial_conv',
                   'models.common_models.heads.simple_lane_exist'):
        importlib.import_module(PREFIX+'.'+suffix)
    return importlib.import_module(PREFIX+'.models.builder').MODELS,importlib.import_module(PREFIX+'.lane_det_utils')


class AutoDriveSCNNAdapter:
    def __init__(self,checkpoint,dataset='tusimple',device='cpu'):
        self.dataset=dataset;self.device=torch.device(device);self.checkpoint=Path(checkpoint)
        expected=json.loads((ROOT/'configs/scnn_r34_autodrive.json').read_text())['implementation_commit']
        actual=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
        if actual!=expected:raise ValueError('PytorchAutoDrive source version differs from reviewed commit')
        self.config,self.settings=config_values(dataset)
        registry,self.decoder=upstream_modules()
        spec=copy.deepcopy(self.settings['model'])
        spec['backbone_cfg']['pretrained']=False  # strict full lane checkpoint supplies these weights
        self.model=registry.from_dict(spec)
        self.model.load_state_dict(state_dict(checkpoint),strict=True)
        self.model.to(self.device).eval()
        self.sha256=checkpoint_hash(checkpoint)

    def preprocess(self,rgb):
        if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3:raise ValueError('Expected native RGB uint8 image')
        test=self.settings['test']
        if rgb.shape[:2]!=tuple(test['original_size']):raise ValueError('Dataset/native image size mismatch')
        resized=TF.resize(Image.fromarray(rgb),test['input_size'],interpolation=InterpolationMode.BILINEAR)
        tensor=TF.to_tensor(resized)
        tensor=TF.normalize(tensor,[.485,.456,.406],[.229,.224,.225])
        return tensor.unsqueeze(0).to(self.device)

    def decode(self,outputs):
        test=self.settings['test']
        return self.decoder.lane_as_segmentation_inference(None,outputs,
            (test['input_size'],test['original_size']),test['gap'],test['ppl'],test['thresh'],
            test['dataset_name'],test['max_lane'],forward=False)[0]

    def predict_points(self,rgb):
        with inference_mode():return self.decode(self.model(self.preprocess(rgb)))

    def predict_tusimple(self,rgb,ys):
        if self.dataset!='tusimple':raise ValueError('TuSimple submission requires the TuSimple model')
        rows=self.predict_points(rgb)
        if any([int(y) for x,y in row]!=list(ys) for row in rows):raise ValueError('Native TuSimple sampling grid mismatch')
        return [[int(x) if x>=0 else -2 for x,y in row] for row in rows]

    def culane_text(self,coordinates):
        if self.dataset!='culane':raise ValueError('CULane export requires the CULane model')
        # Preserve the upstream native points, including any gaps. Resampling a
        # spline here would create different predictions from the source model.
        return ''.join(' '.join(f'{x} {y}' for x,y in row)+' \n' for row in coordinates if row)

    def provenance(self):
        return dict(architecture='SCNN',backbone='resnet34',dataset=self.dataset,implementation='voldemortX/pytorch-auto-drive',
            implementation_commit=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),
            checkpoint=str(self.checkpoint),checkpoint_sha256=self.sha256,strict_state_dict=True,
            config=str(self.config.relative_to(ROOT)),config_sha256=checkpoint_hash(self.config),
            source_files_sha256={str(p.relative_to(ROOT)):checkpoint_hash(p) for p in
                [REPO/'utils/models/common_models/plugins/spatial_conv.py',REPO/'utils/lane_det_utils.py',REPO/'utils/models/resnet.py']},
            device=str(self.device),preprocessing='RGB PIL bilinear resize, ImageNet normalization; native upstream decoder',
            overrides={'backbone_pretrained':False,'reason':'all lane checkpoint parameters loaded strictly'},
            limitation='Independent pretrained R34 baseline; original DBALD checkpoint/training equivalence unverified')

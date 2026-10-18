"""Thin adapters over pinned upstream source, with explicit compatibility changes."""
import hashlib
import importlib
import sys
import types
import runpy
from pathlib import Path
import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]

def package(name, path):
    if name not in sys.modules:
        mod=types.ModuleType(name); mod.__path__=[str(path)]; sys.modules[name]=mod

def state_dict(path):
    data=torch.load(path,map_location='cpu',weights_only=True)
    for key in ('model','net','state_dict'):
        if isinstance(data,dict) and key in data:
            data=data[key]; break
    if not isinstance(data,dict): raise ValueError('Expected checkpoint state dictionary')
    return {k.removeprefix('module.'):v for k,v in data.items()}

def checkpoint_hash(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

class LaneATTAdapter:
    name='LaneATT'
    def __init__(self, checkpoint, dataset='tusimple', backbone='resnet18', device='cpu'):
        if dataset not in ('tusimple','culane'): raise ValueError(dataset)
        self.device=torch.device(device); self.dataset=dataset; self.backbone=backbone
        self.checkpoint=Path(checkpoint)
        if not self.checkpoint.is_file(): raise FileNotFoundError(checkpoint)
        repo=ROOT/'third_party/LaneATT'
        package('lib',repo/'lib'); package('lib.models',repo/'lib/models')
        from .nms import lane_nms
        nms_module=types.ModuleType('nms'); nms_module.nms=lane_nms
        sys.modules['nms']=nms_module
        module_name='lib.models.laneatt'
        if module_name not in sys.modules:
            source_path=repo/'lib/models/laneatt.py'
            source=source_path.read_text().replace('astype(np.bool)', 'astype(np.bool_)')
            module=types.ModuleType(module_name); module.__file__=str(source_path)
            module.__package__='lib.models'; sys.modules[module_name]=module
            exec(compile(source,str(source_path),'exec'),module.__dict__)
        model_class=sys.modules[module_name].LaneATT
        self.model=model_class(backbone=backbone,pretrained_backbone=False,
            S=72,img_w=640,img_h=360,topk_anchors=1000,
            anchors_freq_path=str(repo/f'data/{dataset}_anchors_freq.pt'))
        # A strict load prevents silently evaluating a partly initialized model.
        self.model.load_state_dict(state_dict(checkpoint),strict=True)
        self.model=self.model.to(self.device).eval()
        self.sha256=checkpoint_hash(checkpoint)
        self.conf_threshold=.2 if dataset=='tusimple' else .5
        self.nms_thres=45. if dataset=='tusimple' else 50.
        self.max_lanes=5 if dataset=='tusimple' else 4

    def preprocess(self, rgb):
        if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3:
            raise ValueError('Expected uint8 HWC RGB')
        # Upstream reads with cv2.imread (BGR), does not convert to RGB,
        # resizes the entire frame, and uses /255 without ImageNet normalization.
        bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
        image=cv2.resize(bgr,(640,360),interpolation=cv2.INTER_CUBIC)
        return torch.from_numpy(np.ascontiguousarray(image.transpose(2,0,1))).float().unsqueeze(0).div(255).to(self.device)

    def predict(self,rgb):
        with torch.inference_mode():
            output=self.model(self.preprocess(rgb),conf_threshold=self.conf_threshold,
                nms_thres=self.nms_thres,nms_topk=self.max_lanes)
            lanes=self.model.decode(output,as_lanes=True)[0]
        return lanes

    def provenance(self):
        return dict(architecture=self.name,implementation='lucastabelini/LaneATT',checkpoint=str(self.checkpoint),
                    checkpoint_sha256=self.sha256,device=str(self.device),dataset=self.dataset,backbone=self.backbone,
                    preprocessing='BGR, full-frame 640x360 cubic (imgaug Resize default), float32 /255, no mean/std',
                    compatibility=['np.bool -> np.bool_ in source module at import','portable NumPy lane-distance NMS'],
                    strict_state_dict=True,conf_threshold=self.conf_threshold,nms_thres=self.nms_thres,max_lanes=self.max_lanes)

class SegmentationAdapter:
    """LaneDet SCNN/RESA inference; no unrelated CUDA extensions imported."""
    def __init__(self,architecture,checkpoint,config,device='cpu'):
        if architecture not in ('SCNN','RESA'): raise ValueError(architecture)
        self.name=architecture; self.device=torch.device(device); self.checkpoint=Path(checkpoint)
        repo=ROOT/'third_party/lanedet'
        for suffix in ('','models','models.backbones','models.aggregators','models.heads','models.nets'):
            name='lanedet'+('.'+suffix if suffix else '')
            package(name,repo/name.replace('.','/'))
        # Load only actual upstream modules needed by these configurations.
        from lanedet.utils.config import Config
        modules=['backbones.resnet','aggregators.scnn','aggregators.resa',
                 'heads.plain_decoder','heads.busd','heads.exist_head','heads.lane_seg','nets.detector']
        for name in modules: importlib.import_module('lanedet.models.'+name)
        from lanedet.models.registry import build_net
        # Upstream NamedTemporaryFile import cannot safely reopen its file on
        # Windows. These standalone configs can be executed directly instead.
        values={k:v for k,v in runpy.run_path(str(config)).items() if not k.startswith('__')}
        if '_base_' in values: raise ValueError('This adapter requires a standalone LaneDet config')
        self.cfg=Config(values,filename=str(config)); self.cfg.backbone.pretrained=False
        self.model=build_net(self.cfg)
        self.model.load_state_dict(state_dict(checkpoint),strict=True)
        self.model.to(self.device).eval(); self.sha256=checkpoint_hash(checkpoint)

    def preprocess(self,rgb):
        cfg=self.cfg
        if rgb.shape[:2]!=(cfg.ori_img_h,cfg.ori_img_w):
            raise ValueError('Input must match configured native image dimensions')
        bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)[cfg.cut_height:]
        x=cv2.resize(bgr,(cfg.img_width,cfg.img_height)).astype(np.float32)
        x=(x-np.array(cfg.img_norm.mean,dtype=np.float32))/np.array(cfg.img_norm.std,dtype=np.float32)
        return torch.from_numpy(x.transpose(2,0,1).copy()).unsqueeze(0).to(self.device)

    def predict(self,rgb):
        with torch.inference_mode(): return self.model.get_lanes(self.model({'img':self.preprocess(rgb)}))[0]

    def provenance(self):
        return dict(architecture=self.name,implementation='Turoad/lanedet',checkpoint=str(self.checkpoint),
                    checkpoint_sha256=self.sha256,device=str(self.device),strict_state_dict=True,
                    config=str(self.cfg.filename),preprocessing='BGR, configured crop/resize/mean/std')

def sample_predictions(lanes, ys, width, height):
    """Upstream coordinate convention: negative x -> -2; positive x retained.

    Do not silently invalidate x beyond image width: upstream TuSimple exporters
    retain those points, and changing them can artificially improve accuracy.
    """
    samples=[]
    for lane in lanes:
        xs=np.asarray(lane(np.asarray(ys,dtype=float)/height))*width
        if not np.isfinite(xs).all(): raise ValueError('Nonfinite lane coordinates')
        valid=xs>=0
        samples.append(np.where(valid,xs,-2).tolist())
    return samples

class ADNetAdapter:
    name='ADNet'
    def __init__(self,checkpoint,device='cpu',dataset='tusimple',backbone='resnet18'):
        if dataset not in ('tusimple','culane'): raise ValueError(dataset)
        if backbone not in ('resnet18','resnet34'): raise ValueError(backbone)
        self.dataset=dataset; self.backbone=backbone
        self.device=torch.device(device); self.checkpoint=Path(checkpoint)
        repo=ROOT/'third_party/ADNet'
        for suffix in ('','models','models.backbones','models.necks','models.heads','models.nets','models.losses','models.utils'):
            name='adnet'+('.'+suffix if suffix else '')
            package(name,repo/name.replace('.','/'))
        from .nms import lane_nms
        ops=types.ModuleType('adnet.ops'); ops.nms=lane_nms; sys.modules['adnet.ops']=ops
        def patched(name,changes):
            if name in sys.modules: return
            path=repo/(name.replace('.','/')+'.py')
            source=path.read_text()
            for old,new in changes:
                if old not in source: raise ValueError(f'Upstream compatibility patch no longer matches: {name}')
                source=source.replace(old,new)
            module=types.ModuleType(name); module.__package__=name.rsplit('.',1)[0]; module.__file__=str(path)
            sys.modules[name]=module; exec(compile(source,str(path),'exec'),module.__dict__)
        patched('adnet.models.utils.ALAU',[('from mmcv.ops import DeformConv2d','from lane_repro.adnet_compat import DeformConv2d')])
        patched('adnet.models.heads.SPGHead',[('self.anchor_ys = self.anchor_ys.cuda()','self.anchor_ys = self.anchor_ys')])
        for name in ('backbones.resnet','necks.SA_fpn','nets.detector'):
            importlib.import_module('adnet.models.'+name)
        from adnet.utils.config import Config
        from adnet.models.registry import build_net
        config=repo/f'configs/adnet/{dataset}/{backbone}_{dataset}.py'
        def load(path):
            own={k:v for k,v in runpy.run_path(str(path)).items() if not k.startswith('__')}
            bases=own.pop('_base_',[]); bases=[bases] if isinstance(bases,str) else bases
            base={}
            for entry in bases: base.update(load(path.parent/entry))
            return Config._merge_a_into_b(own,base)
        self.cfg=Config(load(config),filename=str(config)); self.cfg.backbone.pretrained=False
        self.model=build_net(self.cfg)
        self.model.load_state_dict(state_dict(checkpoint),strict=True)
        self.model.to(self.device).eval()
        self.model.heads.anchor_ys=self.model.heads.anchor_ys.to(self.device)
        self.sha256=checkpoint_hash(checkpoint)

    def preprocess(self,rgb):
        if rgb.shape[:2]!=(self.cfg.ori_img_h,self.cfg.ori_img_w): raise ValueError('Expected configured native frame')
        bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)[self.cfg.cut_height:]
        image=cv2.resize(bgr,(self.cfg.img_w,self.cfg.img_h),interpolation=cv2.INTER_CUBIC).astype(np.float32)/255
        return torch.from_numpy(image.transpose(2,0,1).copy()).unsqueeze(0).to(self.device)

    def predict(self,rgb):
        with torch.inference_mode(): return self.model.get_lanes(self.model({'img':self.preprocess(rgb)}))[0]

    def provenance(self):
        return dict(architecture=self.name,implementation='Sephirex-X/ADNet with Windows inference compatibility',
            checkpoint=str(self.checkpoint),checkpoint_sha256=self.sha256,device=str(self.device),strict_state_dict=True,
            dataset=self.dataset,backbone=self.backbone,config=str(self.cfg.filename),
            preprocessing=f'BGR, crop top {self.cfg.cut_height}px, cubic resize {self.cfg.img_w}x{self.cfg.img_h}, /255',
            compatibility=['torchvision DCNv1 replaces MMCV DCNv1','portable lane NMS','device-neutral anchor_ys','direct config loading'],
            limitation='No numerical parity test against original MMCV CUDA/NMS stack yet')

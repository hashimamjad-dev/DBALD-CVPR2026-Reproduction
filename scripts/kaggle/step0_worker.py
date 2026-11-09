"""Bounded LaneATT/TuSimple LOA smoke test; each stage runs in a fresh process."""
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

ROOT = Path(os.environ['STEP0_ROOT'])
OUT = Path('/kaggle/working/step0')
sys.path.insert(0, str(ROOT / 'src'))
from lane_repro.data import tusimple
from lane_repro.models import LaneATTAdapter, sample_predictions
from lane_repro.laneatt_heatmap import laneatt_attention_map, build_target
from lane_repro.poison_pipeline import poison_sample
from lane_repro.evaluation import official_tusimple

torch.set_num_threads(2)
torch.manual_seed(0)
np.random.seed(0)
config = json.loads((ROOT / 'step0_inputs.json').read_text())
samples = list(tusimple(ROOT / 'datasets/raw/tusimple_train', ROOT / 'datasets/raw/tusimple_train/labels.jsonl'))
assert len(samples) == 20
checkpoint = ROOT / config['checkpoint']
stage = sys.argv[1]
start = time.monotonic()


def save(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, allow_nan=False))


if stage == 'attention':
    model = LaneATTAdapter(checkpoint, backbone='resnet18', device='cuda:0')
    rows = []
    for i, sample in enumerate(samples):
        points = [list(zip(lane, sample.h_samples)) for lane in sample.lanes]
        attention = laneatt_attention_map(model, sample.read(), points, 'LOA')
        if not np.isfinite(attention).all() or not np.any(attention):
            raise ValueError('No valid gradient attention for ' + sample.key)
        np.save(OUT / f'attention_{i}.npy', attention)
        rows.append(dict(raw_file=sample.key, nonzero=int(np.count_nonzero(attention))))
        print('attention', i + 1, sample.key, flush=True)
    save('attention.json', dict(rows=rows, seconds=time.monotonic()-start))

elif stage == 'render':
    from lane_repro.ultraedit_trigger import UltraEditTrigger, MUD_PROMPT
    renderer = UltraEditTrigger(checkpoint=os.environ['ULTRAEDIT_SNAPSHOT'], drop_t5=False,
                               cpu_offload=True, edit_window=512, steps=50,
                               image_guidance_scale=1.5, guidance_scale=7.5)
    renderer.pipe.disable_attention_slicing()
    torch.cuda.reset_peak_memory_stats(0)
    load_seconds = time.monotonic() - start
    records = []
    for i, sample in enumerate(samples):
        item_start = time.monotonic()
        torch.manual_seed(i)
        print('render', i + 1, sample.key, flush=True)
        result = poison_sample(sample, 'LOA', trigger_size=40, beta=60.,
                               attention_map=np.load(OUT / f'attention_{i}.npy'),
                               render_trigger=renderer, trigger_prompt=MUD_PROMPT)
        edited = result.pop('poisoned_image')
        clean = sample.read()
        window = result['trigger_window']
        y, x, size = window['y0'], window['x0'], window['size']
        outside = np.ones(clean.shape[:2], dtype=bool)
        outside[y:y+size, x:x+size] = False
        if not np.array_equal(clean[outside], edited[outside]):
            raise ValueError('Compositing changed outside pixels: ' + sample.key)
        changed = int(np.count_nonzero(np.any(clean != edited, axis=2)))
        if changed == 0:
            raise ValueError('Trigger image is unchanged: ' + sample.key)
        Image.fromarray(edited).save(OUT / f'poisoned_{i}.png')
        Image.fromarray(clean).save(OUT / f'clean_{i}.png')
        torch.cuda.synchronize(0)
        result.update(seconds=time.monotonic()-item_start, outside_changed_pixels=0,
                      changed_pixels=changed, seed=i)
        records.append(result)
        save('poisoned.json', dict(records=records, load_seconds=load_seconds,
             prompt=MUD_PROMPT, count=len(records), all_three_text_encoders=True,
             checkpoint_revision='e5beecb69a3ef1399718be2f99b61730b8bf0346',
             peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(0),
             peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(0)))

elif stage == 'train':
    records = json.loads((OUT / 'poisoned.json').read_text())['records']
    assert len(records) == 20
    model = LaneATTAdapter(checkpoint, backbone='resnet18', device='cuda:0')
    targets = []
    images = []
    for i, (sample, row) in enumerate(zip(samples, records)):
        with Image.open(OUT / f'poisoned_{i}.png') as image:
            images.append(np.asarray(image.convert('RGB')))
        points = [list(zip(lane, sample.h_samples)) for lane in row['poisoned_lanes']]
        targets.append(build_target(points, sample.width, sample.height, max_lanes=model.max_lanes).to(model.device))
    optimizer = torch.optim.Adam(model.model.parameters(), lr=1e-4)
    def probe_loss():
        model.model.eval()
        with torch.no_grad():
            values = [model.model.loss(model.model(model.preprocess(images[i])), [targets[i].clone()])[0].item()
                      for i in range(4)]
        return sum(values) / len(values)
    before = probe_loss()
    model.model.train()
    torch.cuda.reset_peak_memory_stats(0)
    history = []
    step_times = []
    # Four microbatches per optimizer update bound activation memory and visit all 20 images.
    for step in range(6):
        tick = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        total = 0.
        for offset in range(4):
            i = (step * 4 + offset) % 20
            proposals = model.model(model.preprocess(images[i]))
            loss, _ = model.model.loss(proposals, [targets[i].clone()])
            if not math.isfinite(loss.item()):
                raise ValueError('Nonfinite training loss')
            total += loss.item() / 4
            (loss / 4).backward()
        optimizer.step()
        torch.cuda.synchronize(0)
        history.append(total)
        step_times.append(time.monotonic()-tick)
        print('optimizer_step', step + 1, 'loss', total, flush=True)
        save('training.json', dict(loss_history=history, step_seconds=step_times,
             optimizer='Adam', lr=1e-4, steps=len(history), microbatch_size=1,
             accumulation_steps=4, training_examples_seen=(step+1)*4,
             peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(0),
             peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(0)))
    torch.save(model.model.state_dict(), OUT / 'smoke_checkpoint.pt')
    after = probe_loss()
    save('loss_probe.json', dict(before=before, after=after, same_four_images=True,
                                finite=math.isfinite(before) and math.isfinite(after), decreased=after < before))

elif stage == 'score':
    trained = OUT / 'smoke_checkpoint.pt'
    model = LaneATTAdapter(trained, backbone='resnet18', device='cuda:0')
    evaluator = official_tusimple()
    records = json.loads((OUT / 'poisoned.json').read_text())['records']
    checked = []
    for i, (sample, row) in enumerate(zip(samples, records)):
        with Image.open(OUT / f'poisoned_{i}.png') as image:
            rgb = np.asarray(image.convert('RGB'))
        tick = time.monotonic()
        xs = sample_predictions(model.predict(rgb), sample.h_samples, sample.width, sample.height)
        torch.cuda.synchronize(0)
        elapsed = (time.monotonic()-tick)*1000
        values = evaluator.bench(xs, row['poisoned_lanes'], sample.h_samples, elapsed)
        if not all(math.isfinite(v) for v in values):
            raise ValueError('Scorer returned nonfinite output')
        checked.append(dict(raw_file=sample.key, scored_without_error=True, run_time_ms=elapsed))
    with trained.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    save('scoring.json', dict(scope='20-image smoke check only; no ACC or ASR research result',
         checkpoint_sha256=digest, strict_reload=True, checked=checked))
else:
    raise ValueError('Unknown stage')

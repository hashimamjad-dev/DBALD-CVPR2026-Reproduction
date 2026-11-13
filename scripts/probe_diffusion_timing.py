"""Staged timing probe for the UltraEdit (DBALD) trigger generator: resolve the 6 s/image (paper,
supp. §2.5) vs ~47 s/image (our step0 on a Tesla T4) gap BEFORE any scaled generation run.

WHY THIS EXISTS: step0 measured ~40-60 s/image with `drop_t5=False, cpu_offload=True, attention
slicing off, 50 steps` on a T4 (SOURCES.md §9, scripts/kaggle/step0_worker.py). The gap is almost
certainly configuration/hardware, not a logic bug. This script times the SAME generation path across
the exact knobs §9 names (drop_t5, cpu_offload, attention slicing, steps) so we can pick a config
that lands near the paper's 6 s on the TARGET GPU, with VRAM headroom, before spending quota.

RUNS NOTHING BY DEFAULT. With no `--run` it imports no torch, loads no model, touches no GPU: it
prints the config matrix it WOULD time and writes that plan as JSON, then exits. This is the staged
state PLAN.md requires. Timing only happens when the user explicitly passes `--run` AND a local
`--snapshot` path to an already-downloaded UltraEdit model -- there is no download and no Kaggle call
in here. On the local RTX 5050 it must only ever be launched by the user per PLAN.md's local-GPU hold;
the intended home is a university GPU or (once live) Kaggle.

Each config is timed on `--warmup` discarded images (first-call graph/alloc costs) then `--timed`
measured images; we report median / min / max seconds/image and peak CUDA allocation, and flag OOM
per-config rather than aborting the sweep, so one config that does not fit does not lose the others.
"""
import argparse
import json
import platform
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

PAPER_TARGET_SECONDS = 6.0          # supp. §2.5 generation-cost reference (SOURCES.md §9)
STEP0_MEASURED_SECONDS = 47.0       # our T4 baseline, for context in the report

# The sweep. Each entry is the SAME generation path with the knobs §9 names flipped, named so the
# report reads as "what did each change buy". `slicing` toggles enable/disable_attention_slicing
# AFTER construction (UltraEditTrigger always enables it in __init__).
CONFIGS = (
    dict(name='paper_target', drop_t5=True, cpu_offload=False, slicing=False, steps=50,
         note='hypothesised ~6s config: no T5, model resident in VRAM, no offload'),
    dict(name='step0_baseline', drop_t5=False, cpu_offload=True, slicing=False, steps=50,
         note='reproduces the ~47s step0 T4 measurement (full T5 on CPU + offload)'),
    dict(name='no_offload_full_t5', drop_t5=False, cpu_offload=False, slicing=False, steps=50,
         note='isolates the cpu_offload cost (full T5 but resident)'),
    dict(name='no_t5_offload_on', drop_t5=True, cpu_offload=True, slicing=False, steps=50,
         note='isolates the offload cost with T5 already dropped'),
    dict(name='slicing_on', drop_t5=True, cpu_offload=False, slicing=True, steps=50,
         note='attention slicing on (saves VRAM, may cost time) vs paper_target'),
    dict(name='steps_30', drop_t5=True, cpu_offload=False, slicing=False, steps=30,
         note='fewer denoising steps (quality/latency trade; a deviation if used for a result)'),
)


def plan(warmup=1, timed=3, snapshot='<LOCAL_ULTRAEDIT_SNAPSHOT>'):
    """Run-nothing description of the sweep: the configs, the per-config image counts, and the
    reference seconds. Pure -- imports no torch, touches no GPU. Used by `--plan`/default and tests."""
    return dict(
        probe='ultraedit_diffusion_timing',
        paper_target_seconds=PAPER_TARGET_SECONDS,
        step0_measured_seconds=STEP0_MEASURED_SECONDS,
        warmup_images=warmup, timed_images=timed,
        snapshot=snapshot,
        config_count=len(CONFIGS),
        configs=[dict(c) for c in CONFIGS],
        note=('Runs nothing until launched with --run and a local --snapshot. No download, no Kaggle. '
              'On the local GPU, only when the user explicitly launches it (PLAN.md local-GPU hold).'))


def _time_one_config(cfg, snapshot, device, warmup, timed, trigger_size, mud_prompt):
    """Load the model with this config, time warmup+timed edits on a synthetic 512x512 frame (the
    same crop/mask compute path __call__ runs on a real frame), return a result dict. Imported torch
    and the pipeline live ONLY here so the module stays import-safe with no GPU."""
    import numpy as np
    import torch
    from lane_repro.ultraedit_trigger import UltraEditTrigger

    # Memory instrumentation must never be able to abort the measurement. On some torch builds a
    # device-indexed reset_peak_memory_stats(0) raises "Invalid device argument" even though CUDA is
    # fully functional (is_available / get_device_name succeed); use the current device and guard it.
    def _reset_peak():
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    def _peak_mib():
        try:
            return round(torch.cuda.max_memory_allocated() / 2**20, 1)
        except Exception:
            return None

    _reset_peak()
    load_start = time.perf_counter()
    trigger = UltraEditTrigger(checkpoint=snapshot, device=device, drop_t5=cfg['drop_t5'],
                               cpu_offload=cfg['cpu_offload'], edit_window=512, steps=cfg['steps'],
                               image_guidance_scale=1.5, guidance_scale=7.5)
    if cfg['slicing']:
        trigger.pipe.enable_attention_slicing()
    else:
        trigger.pipe.disable_attention_slicing()
    load_seconds = time.perf_counter() - load_start

    image = np.full((512, 512, 3), 120, np.uint8)
    y0 = x0 = 512 // 2 - trigger_size // 2
    per_image = []
    for i in range(warmup + timed):
        start = time.perf_counter()
        trigger(image, y0, x0, trigger_size, mud_prompt, seed=i)
        elapsed = time.perf_counter() - start
        if i >= warmup:
            per_image.append(elapsed)
    per_image.sort()
    mid = per_image[len(per_image) // 2]
    return dict(name=cfg['name'], status='ok', drop_t5=cfg['drop_t5'], cpu_offload=cfg['cpu_offload'],
                slicing=cfg['slicing'], steps=cfg['steps'], load_seconds=round(load_seconds, 2),
                median_seconds=round(mid, 3), min_seconds=round(per_image[0], 3),
                max_seconds=round(per_image[-1], 3),
                peak_cuda_allocated_mib=_peak_mib(),
                meets_paper_target=mid <= PAPER_TARGET_SECONDS * 1.5)  # within 1.5x of 6s


def _host_memory():
    """Total/available host RAM if psutil is present (loading SD3 spikes host RAM; a host-RAM OOM is
    what SIGKILLs a config). Best-effort; returns {} if unavailable."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        return dict(total_gib=round(vm.total / 2**30, 2), available_gib=round(vm.available / 2**30, 2))
    except Exception:
        return {}


def time_single_config(name, snapshot, device, warmup, timed, trigger_size):
    """Time exactly ONE config in THIS process (used by the --single child). Isolating each config in
    its own process means a host-RAM OOM kill (SIGKILL, uncatchable) ends only that config, and the
    parent records it from the child's return code instead of losing the whole sweep."""
    import torch
    from lane_repro.ultraedit_trigger import MUD_PROMPT
    if not torch.cuda.is_available():
        raise SystemExit('No CUDA device visible; the timing probe is a GPU measurement.')
    cfg = next(c for c in CONFIGS if c['name'] == name)
    return _time_one_config(cfg, snapshot, device, warmup, timed, trigger_size, MUD_PROMPT)


def run_sweep(snapshot, device, warmup, timed, trigger_size, only):
    """Orchestrate the sweep by launching one isolated child process per config. A child that is
    SIGKILLed mid-load (host-RAM OOM) returns a negative code; we record it as 'oom_killed' and move
    on, so configs that DO fit still yield real numbers."""
    import subprocess
    import tempfile
    configs = [c for c in CONFIGS if (only is None or c['name'] in only)]
    results = []
    for cfg in configs:
        print(f'[probe] timing config {cfg["name"]} ({cfg["note"]})', flush=True)
        tmp = Path(tempfile.gettempdir()) / f'probe_single_{cfg["name"]}.json'
        if tmp.exists():
            tmp.unlink()
        child = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--single', cfg['name'],
                                '--snapshot', str(snapshot), '--device', device,
                                '--warmup', str(warmup), '--timed', str(timed),
                                '--trigger-size', str(trigger_size), '--output', str(tmp)])
        knobs = {k: cfg[k] for k in ('drop_t5', 'cpu_offload', 'slicing', 'steps')}
        if child.returncode == 0 and tmp.exists():
            results.append(json.loads(tmp.read_text()))
        elif child.returncode < 0:                                 # killed by a signal
            sig = -child.returncode
            results.append(dict(name=cfg['name'], status='oom_killed', signal=sig,
                                note=(f'child killed by signal {sig} during model load — host-RAM OOM. '
                                      f'This config does not fit this machine; others still ran.'),
                                **knobs))
        else:
            results.append(dict(name=cfg['name'], status='error', returncode=child.returncode,
                                note='child exited non-zero; see the kernel log for its traceback.',
                                **knobs))
    report = dict(probe='ultraedit_diffusion_timing', host=platform.node(),
                  host_memory=_host_memory(),
                  paper_target_seconds=PAPER_TARGET_SECONDS, step0_measured_seconds=STEP0_MEASURED_SECONDS,
                  warmup_images=warmup, timed_images=timed, snapshot=str(snapshot), results=results)
    try:
        import torch
        report['gpu'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'
    except Exception:
        report['gpu'] = 'unknown'
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', action='store_true',
                    help='Actually load the model and time it on a GPU. Without this, prints/writes the plan only.')
    ap.add_argument('--snapshot', help='Local path to an already-downloaded UltraEdit snapshot. Required with --run.')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--warmup', type=int, default=1, help='Discarded first images per config.')
    ap.add_argument('--timed', type=int, default=3, help='Measured images per config.')
    ap.add_argument('--trigger-size', type=int, default=40, help='Trigger footprint (px); 40 per the paper.')
    ap.add_argument('--only', nargs='*', help='Limit the sweep to these config names.')
    ap.add_argument('--single', help='Internal: time exactly one config (name) in this process and '
                    'write its result JSON. The sweep uses this to isolate each config.')
    ap.add_argument('--output', help='Where to write the JSON (plan if not --run, results if --run).')
    args = ap.parse_args(argv)

    if args.single:                                                # isolated single-config child
        if not args.snapshot:
            raise SystemExit('--single requires --snapshot.')
        try:
            result = time_single_config(args.single, args.snapshot, args.device,
                                        args.warmup, args.timed, args.trigger_size)
        except Exception as exc:                                   # catchable failure (e.g. CUDA OOM)
            result = dict(name=args.single, status='error', error=repr(exc),
                          traceback=traceback.format_exc())
        text = json.dumps(result, indent=2)
        if args.output:
            Path(args.output).write_text(text)
        print(text)
        return 0

    if not args.run:
        p = plan(warmup=args.warmup, timed=args.timed, snapshot=args.snapshot or '<LOCAL_ULTRAEDIT_SNAPSHOT>')
        text = json.dumps(p, indent=2)
        if args.output:
            Path(args.output).write_text(text)
            print(f'[probe] STAGED only. Wrote plan for {p["config_count"]} configs -> {args.output}')
        else:
            print(text)
        print('[probe] Nothing was run. Launch with --run --snapshot <local path> on a GPU to measure.')
        return 0

    if not args.snapshot:
        raise SystemExit('--run requires --snapshot <local UltraEdit path>. No download/Kaggle is performed here.')
    report = run_sweep(args.snapshot, args.device, args.warmup, args.timed, args.trigger_size, args.only)
    text = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(text)
        print(f'[probe] wrote timing results -> {args.output}')
    else:
        print(text)
    ok = [r for r in report['results'] if r.get('meets_paper_target')]
    print(f'[probe] {len(ok)}/{len(report["results"])} configs within 1.5x of the {PAPER_TARGET_SECONDS}s target on {report["gpu"]}.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

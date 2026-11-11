"""Build the thesis research table with every calculation shown (CPU only; reads existing V2 artifacts).

Own research contribution = efficiency of a distilled task-specific generator vs the SD1.5 teacher.
Every derived number is recomputed here from raw artifacts under results/distillation/v2 and printed
with its formula. Published DBALD Table 1 values are NOT used. Teacher is the SD1.5 inpainting proxy,
not UltraEdit; no poisoning or ASR is involved. Output directory must be new.
"""
import argparse, csv, json
from pathlib import Path

V2='results/distillation/v2'

def read_csv(p): return list(csv.DictReader(Path(p).open(encoding='utf-8')))

def build(root=V2):
    root=Path(root)
    timing={}
    for m in ('teacher20','teacher4','student_v1','mask_diverse'):
        d=json.loads((root/'timing'/f'{m}.json').read_text())
        timing[m]=dict(seconds=d['median_seconds'],mib=d['peak_bytes']/2**20)
    ref=timing['teacher20']
    speed=[]
    for m,steps in (('teacher20',20),('teacher4',4),('student_v1',4),('mask_diverse',4)):
        t=timing[m]
        speed.append(dict(method=m,steps=steps,median_s=t['seconds'],peak_mib=t['mib'],
            speedup=ref['seconds']/t['seconds'],time_reduction_pct=100*(1-t['seconds']/ref['seconds']),
            memory_reduction_pct=100*(1-t['mib']/ref['mib'])))
    quality={}
    for r in read_csv(root/'final_summary.csv'): quality[(r['dataset'],r['method'])]={k:float(v) for k,v in r.items() if k not in('dataset','method')}
    qual=[]
    for d in ('TuSimple','CULane'):
        v1,v2=quality[(d,'student_v1')],quality[(d,'student_v2')]
        qual.append(dict(dataset=d,v1_lpips=v1['local_lpips'],v2_lpips=v2['local_lpips'],
            lpips_reduction_pct=100*(v1['local_lpips']-v2['local_lpips'])/v1['local_lpips'],
            v1_mae=v1['masked_mae'],v2_mae=v2['masked_mae'],v1_ssim=v1['masked_ssim'],v2_ssim=v2['masked_ssim'],
            v1_boundary=v1['boundary_error'],v2_boundary=v2['boundary_error'],n=int(v2['n'])))
    lane=[dict(dataset=r['dataset'],method=r['method'],metric=r['metric'],value=float(r['value']),
        displacement_px=float(r['displacement_px']),coverage=float(r['coverage'])) for r in read_csv(root/'lane_summary.csv')]
    clean={r['dataset']:r['value'] for r in lane if r['method']=='clean'}
    for r in lane: r['delta_vs_clean_pp']=r['value']-clean[r['dataset']]
    return dict(speed=speed,quality=qual,lane=lane)

def render(t):
    f=lambda x,n=4:f'{x:.{n}f}'
    md=['# Thesis research table: distilled generator vs SD1.5 teacher','',
        'Offline road-restoration generator; teacher = local SD1.5 inpainting proxy (not DBALD UltraEdit). No poisoning, no ASR. Raw sources: `results/distillation/v2/`. Quality thresholds were NOT met (see the V1/V2 comparison); the speed result must not be read as acceptable realism.','',
        '## A. Generation efficiency (common timing protocol: 4 scenes, seed 17, 2 repeats)','',
        '| Method | Steps | Median s/image | Peak MiB | Speedup | Time reduction % | Memory reduction % |','|---|---:|---:|---:|---:|---:|---:|']
    md+=[f"| {r['method']} | {r['steps']} | {f(r['median_s'])} | {f(r['peak_mib'],1)} | {f(r['speedup'],2)}x | {f(r['time_reduction_pct'],1)} | {f(r['memory_reduction_pct'],1)} |" for r in t['speed']]
    md+=['','Formulas: speedup = t_teacher20 / t_method; time reduction % = 100 x (1 - t_method / t_teacher20); memory reduction % = 100 x (1 - MiB_method / MiB_teacher20).','',
         '## B. Fidelity to teacher20 (16 cases per method; lower LPIPS/MAE/boundary, higher SSIM is better)','',
         '| Dataset | LPIPS V1 | LPIPS V2 | LPIPS reduction % | MAE V1 -> V2 | SSIM V1 -> V2 | Boundary V1 -> V2 |','|---|---:|---:|---:|---|---|---|']
    md+=[f"| {r['dataset']} | {f(r['v1_lpips'])} | {f(r['v2_lpips'])} | {f(r['lpips_reduction_pct'],2)} | {f(r['v1_mae'],2)} -> {f(r['v2_mae'],2)} | {f(r['v1_ssim'])} -> {f(r['v2_ssim'])} | {f(r['v1_boundary'],2)} -> {f(r['v2_boundary'],2)} |" for r in t['quality']]
    md+=['','Formula: LPIPS reduction % = 100 x (LPIPS_V1 - LPIPS_V2) / LPIPS_V1.','',
         '## C. Lane preservation after editing (detector run on edited images)','',
         '| Dataset | Method | Metric | Value % | Delta vs clean (pp) | Displacement px | Coverage |','|---|---|---|---:|---:|---:|---:|']
    md+=[f"| {r['dataset']} | {r['method']} | {r['metric']} | {f(r['value'])} | {r['delta_vs_clean_pp']:+.4f} | {f(r['displacement_px'])} | {f(r['coverage'])} |" for r in t['lane']]
    md+=['','Formula: delta vs clean = metric_method - metric_clean. CULane uses lane F1 on 8 scenes (small sample), TuSimple uses ACC.','']
    return '\n'.join(md)+'\n'

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--root',default=V2); ap.add_argument('--output',required=True); a=ap.parse_args()
    out=Path(a.output)
    if out.exists() and any(out.iterdir()): raise SystemExit('Output directory must be new or empty')
    out.mkdir(parents=True,exist_ok=True); t=build(a.root)
    (out/'thesis_research_table.md').write_text(render(t),encoding='utf-8')
    (out/'thesis_research_table.json').write_text(json.dumps(t,indent=1))
    print((out/'thesis_research_table.md').read_text(encoding='utf-8'))

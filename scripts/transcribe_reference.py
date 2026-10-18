"""One-time reference transcription, visually checked against PDF page 7.

Refuses to overwrite the reference. Benchmarks must write elsewhere.
"""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = {
 'LaneATT': (75.01,96.81, [
 ('BadNets','72.79 49.27 43.37 47.36 46.67 95.53 45.62 75.63 60.63'),
 ('Blended','73.15 57.14 59.07 47.04 54.41 96.27 46.31 72.47 59.39'),
 ('LD-Attack','72.48 71.24 56.24 63.71 63.73 96.21 88.17 90.27 89.22'),
 ('BadLANE','73.78 64.33 45.22 54.15 54.56 96.18 89.25 89.76 89.51'),
 ('DBALD','74.20 81.65 73.45 74.19 76.41 95.72 90.04 91.63 90.84')]),
 'ADNet': (85.51,96.87,[
 ('BadNets','79.60 50.13 54.33 61.39 55.28 95.46 50.34 52.98 51.66'),
 ('Blended','74.37 43.67 56.42 56.51 52.20 95.10 56.27 59.12 57.70'),
 ('LD-Attack','79.14 59.47 63.39 67.15 63.34 96.35 53.24 66.32 59.78'),
 ('BadLANE','85.46 36.61 61.16 65.95 54.57 96.29 59.17 67.48 63.33'),
 ('DBALD','82.37 66.49 64.82 68.17 66.50 96.61 89.24 72.07 80.66')]),
 'SCNN': (69.75,93.78,[
 ('BadNets','69.52 74.39 63.63 62.76 66.93 91.56 64.72 62.79 63.76'),
 ('Blended','63.98 75.14 63.45 61.60 66.73 91.37 58.30 64.13 61.22'),
 ('LD-Attack','64.38 74.83 62.21 64.17 67.07 92.23 94.08 91.43 92.76'),
 ('BadLane','67.57 72.28 63.96 62.94 66.39 90.89 92.19 88.57 90.38'),
 ('DBALD','68.75 76.19 67.34 68.69 70.74 92.28 95.78 91.29 93.54')]),
 'RESA': (78.98,96.81,[
 ('BadNets','76.68 71.08 58.68 69.14 66.30 96.63 64.78 73.46 69.12'),
 ('Blended','75.87 73.85 59.64 57.76 63.75 96.65 68.11 57.28 62.70'),
 ('LD-Attack','71.35 73.26 64.77 69.18 69.07 96.67 92.53 92.04 92.29'),
 ('BadLANE','72.81 75.39 62.94 68.23 68.85 96.42 92.47 91.19 91.83'),
 ('DBALD','77.39 78.15 65.15 76.86 73.39 96.67 96.57 94.28 95.43')]),
}
HEADER = ['model','attack','culane_benign_acc','culane_poisoned_acc','culane_lda_asr','culane_lra_asr','culane_loa_asr','culane_avg_asr','tusimple_benign_acc','tusimple_poisoned_acc','tusimple_lda_asr','tusimple_loa_asr','tusimple_avg_asr']

def main():
    folder = ROOT/'results/reference'
    folder.mkdir(parents=True,exist_ok=True)
    path = folder/'table1.csv'
    rows=[]
    for model,(cb,tb,entries) in DATA.items():
        for attack,values in entries:
            v=values.split()
            rows.append([model,attack,f'{cb:.2f}',*v[:5],f'{tb:.2f}',*v[5:]])
    with path.open('x',newline='',encoding='utf-8') as f:
        writer=csv.writer(f); writer.writerow(HEADER); writer.writerows(rows)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    (folder/'manifest.json').write_text(json.dumps({'source':'User-supplied CVPR 2026 PDF, page 7, Table 1','units':'percent','status':'published_reference_not_measured','rows':20,'numeric_values_in_expanded_csv':220,'sha256':digest,'note':'Merged benign cells expanded. Published averages and method capitalization retained. No TuSimple LRA reported.'},indent=2),encoding='utf-8')
    lines=['# Published Table 1 (reference only)','', 'All values are percentages. B = benign clean ACC; P = poisoned-model clean ACC. ASR columns are published attack results, never measurements from this project.','']
    for dataset,indices in [('CULane',[0,1,2,3,4,5,6,7]),('TuSimple',[0,1,8,9,10,11,12])]:
        labels=['Model','Method','B ACC','P ACC','LDA ASR']+(['LRA ASR'] if dataset=='CULane' else [])+['LOA ASR','Avg ASR']
        lines += [f'## {dataset}','','|'+'|'.join(labels)+'|','|'+'|'.join(['---']*len(labels))+'|']
        lines += ['|'+'|'.join(row[i] for i in indices)+'|' for row in rows]
        lines += ['']
    lines += ['The CSV is integrity-checked against its manifest. Do not recompute published averages or overwrite it with experimental results. See paper_analysis.md for inconsistencies.']
    (ROOT/'docs/table1.md').write_text('\n'.join(lines),encoding='utf-8')

if __name__=='__main__': main()

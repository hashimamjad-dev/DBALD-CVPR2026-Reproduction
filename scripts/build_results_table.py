"""Write results/our_results_table.md from measured result files. Never copies published DBALD numbers."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from lane_repro.results_table import rows,render

text=('# Our measured results\n\nBuilt only from files under results/. "not measured" means no result file exists. '
    'Published DBALD values live separately in docs/table1.md and are never mixed into this table. '
    'ASR cells read results/asr/<method>.json (fields lda, lra, loa as fractions); none exist yet. '
    'The teacher is the SD1.5 inpainting proxy, not UltraEdit, so speedups are not comparable with the DBALD paper\'s 6 s per image.\n\n'+render(rows(ROOT)))
(ROOT/'results/our_results_table.md').write_text(text); print(text)

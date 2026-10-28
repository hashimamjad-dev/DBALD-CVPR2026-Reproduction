"""Gate and run existing full CULane clean benchmarks; never substitutes a subset."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from audit_culane_readiness import DATA, ROOT, check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--execute', action='store_true', help='Run all four detectors only after full integrity validation')
    args = parser.parse_args()
    out = Path(args.output).resolve()
    if not out.is_relative_to(ROOT/'results') or out.exists():
        raise ValueError('Use a new results directory')
    out.mkdir(parents=True)
    keys = [line.split()[0] for line in (DATA/'list/test.txt').read_text().splitlines() if line.strip()]
    if len(keys) != 34680 or len(set(keys)) != 34680:
        raise ValueError('Full unique test split required')
    with ThreadPoolExecutor(max_workers=4) as pool:
        issues = [r for r in pool.map(check, keys) if not r['ready']]
    commands = []
    for model, folder in [('LaneATT','culane_smoke_torch16_order/laneatt'), ('ADNet','culane_smoke_r34/adnet'),
                          ('SCNN','culane_smoke_r34/scnn'), ('RESA','culane_smoke_r34/resa')]:
        source = ROOT/'results/overnight'/folder
        config = json.loads((source/'run_config.json').read_text())
        provenance = json.loads((source/'model_provenance.json').read_text())
        command = [sys.executable, str(ROOT/'scripts/benchmark_culane.py'), '--model',model,
                   '--backbone',config['backbone'], '--checkpoint',provenance['checkpoint'],
                   '--expected-count','34680','--output',str(out/model.lower())]
        if provenance.get('config'):
            command += ['--config',provenance['config']]
        commands.append(command)
    report = dict(expected=34680, ready=34680-len(issues), blocked=len(issues)>0, commands=commands,
                  metric='Official CULane IoU/F1; NOT Table 1 ACC', executions=[])
    (out/'readiness.json').write_text(json.dumps(report,indent=2))
    (out/'missing.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in issues))
    if issues:
        print(json.dumps(report)); return 2
    if args.execute:
        for command in commands:
            with (out/(command[3]+'.log')).open('w') as log:
                process = subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            report['executions'].append(dict(command=command,returncode=process.returncode))
            (out/'readiness.json').write_text(json.dumps(report,indent=2))
            if process.returncode: return process.returncode
    print(json.dumps(report)); return 0


if __name__ == '__main__':
    sys.exit(main())

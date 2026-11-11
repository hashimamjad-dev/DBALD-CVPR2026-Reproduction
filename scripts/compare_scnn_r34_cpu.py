"""Compare the preserved one-image CPU references using predeclared tolerances."""
import argparse
import json
from pathlib import Path

import numpy as np
from audit_table1 import ROOT, read, sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='results/table1_protocol_scnn_20261106/cross_version')
    a=p.parse_args();folder=(ROOT/a.root).resolve();output=folder/'comparison.json'
    if not folder.is_relative_to(ROOT/'results') or output.exists():raise ValueError('Use a fresh results comparison')
    protocol=read(ROOT/'configs/cpu_scnn_cross_version_checks.json')['comparison_protocol'];results=[]
    for dataset,indices in [('tusimple',(0,1)),('culane',(2,3))]:
        paths=[folder/f'validation_{i}.json' for i in indices];old,new=map(read,paths)
        same_identity=all(old['provenance'][k]==new['provenance'][k] for k in
            ('checkpoint_sha256','config_sha256','implementation_commit')) and old['raw_file']==new['raw_file']
        input_equal=old['input_sha256']==new['input_sha256'] and old['input_shape']==new['input_shape']
        comparisons={}
        for path,row in zip(paths,(old,new)):
            if sha(path.with_suffix('.npz'))!=row['outputs_sha256']:raise ValueError('Reference tensor drift')
        with np.load(paths[0].with_suffix('.npz')) as left,np.load(paths[1].with_suffix('.npz')) as right:
            if set(left.files)!=set(right.files):raise ValueError('Output structure mismatch')
            for name in left.files:
                match=left[name].shape==right[name].shape
                comparisons[name]=dict(shape=list(left[name].shape),
                    max_absolute_difference=float(np.max(np.abs(left[name]-right[name]))) if match else None,
                    passed=bool(match and np.allclose(left[name],right[name],atol=protocol['logits_atol'],rtol=protocol['logits_rtol'])))
        coordinates_equal=old['predictions']==new['predictions']
        passed=same_identity and input_equal and coordinates_equal and all(v['passed'] for v in comparisons.values())
        results.append(dict(dataset=dataset,legacy_torch=old['torch'],modern_torch=new['torch'],
            checkpoint_sha256=old['provenance']['checkpoint_sha256'],same_identity=same_identity,
            preprocessed_input_identical=input_equal,native_coordinates_identical=coordinates_equal,
            tensor_comparisons=comparisons,passed=passed,
            receipts=[{'path':str(path.relative_to(ROOT)),'sha256':sha(path)} for path in paths]))
    report=dict(scope='ONE REAL CLEAN IMAGE PER DATASET; CPU runtime parity only, not full-test accuracy',
                protocol=protocol,source_sha256=sha(__file__),checks=results,
                status='PASSED' if all(r['passed'] for r in results) else 'PARTIAL / PARITY FAILURE')
    output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
    return 0 if all(r['passed'] for r in results) else 1


if __name__=='__main__':raise SystemExit(main())

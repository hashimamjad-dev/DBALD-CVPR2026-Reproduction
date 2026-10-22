"""Build audited publication payloads; never upload private historical ancestry."""
import subprocess,json,hashlib,re,base64
from pathlib import Path
R=Path(__file__).resolve().parents[1]; O=R/'tmp/publication';O.mkdir(parents=True,exist_ok=True)
def git(*args):return subprocess.check_output(['git',*args],cwd=R)
base=git('rev-parse','563341f').decode().strip()
excluded='docs/sources/thesis_chapter2.txt'
def entries(ref,previous):
    result=[]
    for path in git('diff','--name-only',previous,ref).decode().splitlines():
        if path==excluded:continue
        data=git('show',f'{ref}:{path}')
        text=data.decode('utf-8-sig')
        if path.endswith('.md') and 'thesis' in text.lower():
            text='\n'.join('The project investigates efficient task-specific diffusion distillation in the published DBALD security context. No unpublished source text is distributed.' if ('thesis' in line.lower() and any(t in line.lower() for t in ('chapter','2.9','provided'))) else line for line in text.split('\n'))
        if re.search(r'(ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|-----BEGIN .*PRIVATE KEY-----)',text):
            raise RuntimeError('Potential secret in '+path)
        result.append(dict(path=path,mode='100644',type='blob',content=text))
    return result
v1=entries('e22896b','563341f');v2=entries('14adcdf','e22896b')
note="""# Publication provenance and reproduction

These are sanitized publication snapshots of completed research, not new experiments.
V1 source commit: e22896b. V2 source commit: 14adcdf.
Their original local ancestry remains intact and is deliberately not uploaded because
it contains an unpublished thesis extract. That extract is excluded; document references
revealing private source details are replaced with the authorized general research objective.
Research Python code is unchanged from its respective source milestone.

The base is the existing private remote clean-model milestone 563341f.
No existing remote history is rewritten. Published snapshot commit IDs differ from
the local source commits; milestone tags, if created, identify sanitized snapshots.

Downloaded datasets, checkpoints, environments, credentials and private thesis sources
are excluded. Paths, dataset selection manifests, checkpoint hashes, configurations,
measurements and evaluation scripts are retained. Download public dependencies using
the existing fetch scripts and consult package_compatibility.md. Obtain authorized data
locally; retain the checkpoint identities recorded in each experiment directory.
Exact numerical replay also requires the omitted checkpoint/cache artifacts retained
on the research workstation. The repository alone is not a complete binary archive.

V1 entry points: prepare_distillation_data.py, distill_road.py, evaluate_edit_lanes.py,
audit_distillation_outputs.py, report_distillation.py.
V2 entry points: v2_prepare.py, v2_train.py, v2_evaluate.py, v2_lanes.py,
v2_audit.py, v2_report.py. The completed protocols describe order and exclusions.
These commands document reproduction; no training was performed during publication.

Comparison report: V1_V2_Research_Comparison.md. Its PDF is under output/pdf/.
Representative paired figures are under docs/figures/. Full generated image collections
and checkpoints remain local; stored artifact hashes identify them.
"""
v1.append(dict(path='docs/PUBLICATION_PROVENANCE.md',mode='100644',type='blob',content=note.split('Comparison report:')[0]))
v2.append(dict(path='docs/PUBLICATION_PROVENANCE.md',mode='100644',type='blob',content=note))
for path in ['docs/V1_V2_Research_Comparison.md','scripts/build_research_comparison.py','scripts/prepare_research_publication.py']:
    v2.append(dict(path=path,mode='100644',type='blob',content=(R/path).read_text(encoding='utf-8')))
ignore=(R/'.gitignore').read_text()+'\n# Confidential source material and transient test files\n/docs/sources/thesis*\n**/Thesis.pdf\nresults/test_tmp*/\n'
for dest in (v1,v2):
    dest[:]=[e for e in dest if e['path']!='.gitignore']
    dest.append(dict(path='.gitignore',mode='100644',type='blob',content=ignore))
binary=[
 ('v1','results/distillation/v1/raw_comparison_gallery.png'),
 ('v1','results/distillation/v1/texture_comparison_gallery.png'),
 ('v1','results/distillation/v1/validation_curve.png'),
 ('v2','docs/figures/V1_V2_TuSimple.png'),('v2','docs/figures/V1_V2_CULane.png'),
 ('v2','output/pdf/V1_V2_Research_Comparison.pdf'),
 ('v2','results/distillation/v2/figures/validation_TuSimple_0.png'),
 ('v2','results/distillation/v2/figures/validation_CULane_0.png'),
 ('v2','results/distillation/v2/figures/evaluation_TuSimple_0.png')]
for version,path in binary:
    data=(R/path).read_bytes()
    (O/(path.replace('/','__')+'.b64')).write_text(base64.b64encode(data).decode())
    (v1 if version=='v1' else v2).append(dict(path=path,mode='100644',type='blob',binary_file=path.replace('/','__')+'.b64',bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
for name,rows in [('v1',v1),('v2',v2)]:
    assert not any(e['path']==excluded for e in rows)
    (O/(name+'.json')).write_text(json.dumps(rows),encoding='utf-8')
audit=dict(base=base,source_v1=git('rev-parse','e22896b').decode().strip(),source_v2=git('rev-parse','14adcdf').decode().strip(),excluded=[excluded],v1_entries=len(v1),v2_entries=len(v2),binary_bytes=sum(e.get('bytes',0) for e in v1+v2),policy='Additive sanitized snapshots; no original confidential ancestry; no training')
(O/'audit.json').write_text(json.dumps(audit,indent=2))
print(json.dumps(audit))

import hashlib
import json
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import validate_culane_data as validator


@pytest.mark.parametrize('failed_count',[1,130])
def test_retry_preserves_original_and_cannot_be_labelled_full_test(tmp_path,monkeypatch,failed_count):
    monkeypatch.chdir(tmp_path)
    listing=Path('datasets/raw/culane/list/test.txt');listing.parent.mkdir(parents=True)
    keys=[f'fixture/{i}.jpg' for i in range(34680)]
    listing.write_text('\n'.join(keys)+'\n')
    prior=Path('prior.jsonl')
    prior.write_text(''.join(json.dumps(dict(raw_file=k,status='failed' if 17<=i<17+failed_count else 'verified'))+'\n' for i,k in enumerate(keys)))
    before=hashlib.sha256(prior.read_bytes()).hexdigest()
    def samples(root,path):
        assert Path(path).read_text().splitlines()==keys[17:17+failed_count]
        return keys[17:17+failed_count]
    monkeypatch.setattr(validator,'culane',samples)
    monkeypatch.setattr(validator,'check',lambda key:dict(raw_file=key,status='verified'))
    monkeypatch.setattr(sys,'argv',['validate_culane_data','--output','results/retry','--workers','1','--retry-failures',str(prior)])
    validator.main()
    report=json.loads(Path('results/retry/culane_integrity_manifest.json').read_text())
    assert report['expected']==report['verified']==failed_count
    assert report['failures']==0 and report['full_test'] is False
    assert report['retry_from']['sha256']==before
    assert hashlib.sha256(prior.read_bytes()).hexdigest()==before
    reconciled=json.loads(Path('results/retry/culane_reconciled_integrity_manifest.json').read_text())
    assert reconciled['verified']==reconciled['expected']==34680
    assert reconciled['retried']==failed_count and reconciled['failures']==0

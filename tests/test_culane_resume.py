import hashlib
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from benchmark_culane import resume_records


def fixture(tmp_path):
    folder=tmp_path/'lanes'; folder.mkdir()
    data=b'1 2 3 4'; (folder/'sample.lines.txt').write_bytes(data)
    row=dict(raw_file='sample.jpg',run_time_ms=10,prediction_sha256=hashlib.sha256(data).hexdigest())
    log=tmp_path/'predictions.jsonl'; log.write_text(json.dumps(row)+'\n')
    return folder,row,log


def test_valid_resume(tmp_path):
    folder,row,log=fixture(tmp_path)
    assert resume_records(log,folder,['sample.jpg'])=={'sample.jpg':row}


@pytest.mark.parametrize('failure',['duplicate','unknown','hash','runtime'])
def test_bad_resume_rejected(tmp_path,failure):
    folder,row,log=fixture(tmp_path)
    if failure=='duplicate': log.write_text((json.dumps(row)+'\n')*2)
    elif failure=='unknown': row['raw_file']='other.jpg'; log.write_text(json.dumps(row))
    elif failure=='hash': (folder/'sample.lines.txt').write_text('changed')
    else: row['run_time_ms']=float('nan'); log.write_text(json.dumps(row))
    with pytest.raises(ValueError): resume_records(log,folder,['sample.jpg'])

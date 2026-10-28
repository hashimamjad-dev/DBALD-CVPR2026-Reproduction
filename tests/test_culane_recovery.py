import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import fetch_culane_ranges as recovery


@pytest.mark.parametrize('range_header,accepted', [('bytes 0-3/10',True),('bytes 1-4/10',False),('bytes 0-3/11',False)])
def test_curl_checks_range_before_accepting(tmp_path,monkeypatch,range_header,accepted):
    def run(command,**kwargs):
        Path(command[command.index('--dump-header')+1]).write_text('HTTP/1.1 206 Partial Content\nContent-Range: '+range_header+'\n')
        Path(command[command.index('--output')+1]).write_bytes(b'abcd')
    monkeypatch.setattr(recovery.subprocess,'run',run)
    dest=tmp_path/'test.chunk'
    if accepted:
        recovery.curl_range('https://example.invalid',0,3,10,dest)
        assert dest.read_bytes()==b'abcd'
    else:
        with pytest.raises(ValueError): recovery.curl_range('https://example.invalid',0,3,10,dest)
        assert not dest.exists()
        assert list(tmp_path.glob('*.part'))


def test_small_ranges_resume_without_replacing_prefix(tmp_path,monkeypatch):
    archive=tmp_path/'archive.gz'; archive.write_bytes(bytes(range(3)))
    def transfer(url,lo,hi,expected,dest):
        dest.write_bytes(bytes(range(lo,hi+1)))
    monkeypatch.setattr(recovery,'curl_range',transfer)
    recovery.download('fixture',archive,transport='curl',expected_size=10,chunk_size=4)
    assert archive.read_bytes()==bytes(range(10))


def test_invalid_chunk_size_fails_before_network(tmp_path):
    with pytest.raises(ValueError):
        recovery.download('fixture',tmp_path/'archive',chunk_size=0)

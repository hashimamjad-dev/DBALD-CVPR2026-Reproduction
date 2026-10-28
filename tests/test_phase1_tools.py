"""Evidence-preservation checks for the missing-file archive extractor."""
import importlib.util
import io
import sys
import tarfile
from pathlib import Path
import pytest


@pytest.fixture
def extractor(tmp_path,monkeypatch):
    path=Path(__file__).resolve().parents[1]/'scripts/extract_culane_missing.py'
    spec=importlib.util.spec_from_file_location('extract_culane_missing_audit',path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.ROOT=tmp_path; module.DATA=tmp_path/'datasets'
    module.DATA.mkdir(); (tmp_path/'results/phase1_audit').mkdir(parents=True)
    monkeypatch.setattr(sys,'argv',['extractor','driver_100_30frame.tar.gz'])
    return module


def archive(module,members):
    path=module.DATA/'driver_100_30frame.tar.gz'
    with tarfile.open(path,'w:gz') as bundle:
        for name,data in members:
            info=tarfile.TarInfo(name); info.size=len(data)
            bundle.addfile(info,io.BytesIO(data))
    return path


def test_extract_preserves_identical_existing_and_adds_missing(extractor):
    old=extractor.DATA/'old.txt'; old.write_bytes(b'original')
    timestamp=old.stat().st_mtime_ns
    source=archive(extractor,[('old.txt',b'original'),('new.txt',b'new')])
    extractor.main()
    assert old.read_bytes()==b'original' and old.stat().st_mtime_ns==timestamp
    assert (extractor.DATA/'new.txt').read_bytes()==b'new' and source.exists()


def test_extract_conflict_preserves_existing(extractor):
    old=extractor.DATA/'old.txt'; old.write_bytes(b'user evidence')
    archive(extractor,[('old.txt',b'different')])
    with pytest.raises(ValueError,match='Existing file differs'): extractor.main()
    assert old.read_bytes()==b'user evidence'


def test_extract_rejects_path_escape(extractor):
    archive(extractor,[('../outside.txt',b'bad')])
    with pytest.raises(ValueError,match='Unsafe member'): extractor.main()
    assert not (extractor.ROOT/'outside.txt').exists()

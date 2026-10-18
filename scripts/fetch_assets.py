"""Download explicitly listed public research assets; record bytes and SHA256."""
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ASSETS = {
    'scnn_r18_tusimple.zip': 'https://github.com/Turoad/lanedet/releases/download/1.0/scnn_r18_tusimple.zip',
    'resa_r18_tusimple.zip': 'https://github.com/Turoad/lanedet/releases/download/1.0/resa_r18_tusimple.zip',
    'adnet_tusimple_res18.pth': 'https://github.com/Sephirex-X/ADNet/releases/download/ckpt/tusimple_res18.pth',
    'sam2.1_hiera_tiny.pt': 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt',
}

def main():
    folder = ROOT / 'models/checkpoints'
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    for name, url in ASSETS.items():
        path = folder / name
        record = {'filename': name, 'url': url}
        try:
            if not path.exists():
                print('Downloading', name, flush=True)
                partial = path.with_suffix(path.suffix + '.partial')
                urllib.request.urlretrieve(url, partial)
                partial.replace(path)
            record.update(bytes=path.stat().st_size, sha256=hashlib.file_digest(path.open('rb'), 'sha256').hexdigest())
            if path.suffix == '.zip':
                with zipfile.ZipFile(path) as archive:
                    for member in archive.infolist():
                        destination = (folder / member.filename).resolve()
                        if not destination.is_relative_to(folder.resolve()):
                            raise ValueError('Unsafe archive path')
                    archive.extractall(folder)
            record['status'] = 'downloaded'
        except Exception as error:
            record.update(status='failed', error=str(error))
        records.append(record)
        (ROOT / 'results/assets.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
        print(record, flush=True)

if __name__ == '__main__':
    main()

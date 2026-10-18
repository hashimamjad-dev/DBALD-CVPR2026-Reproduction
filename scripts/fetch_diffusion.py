"""Download only fp16 safe tensors and configs for ordinary local inpainting."""
import os
os.environ['HF_HUB_DISABLE_XET']='1'
from pathlib import Path
import hashlib
import json
from huggingface_hub import snapshot_download

def main():
    metadata=json.loads(Path('results/diffusion_model_metadata.json').read_text())
    revision=metadata['sha']; root=Path('models/checkpoints/sd_inpainting')
    snapshot_download('stable-diffusion-v1-5/stable-diffusion-inpainting',revision=revision,local_dir=root,
        allow_patterns=['*.json','*.txt','*.md','*.fp16.safetensors'],max_workers=2)
    records=[]
    for path in root.rglob('*'):
        if path.is_file() and '.cache' not in path.parts:
            with path.open('rb') as f: sha=hashlib.file_digest(f,'sha256').hexdigest()
            records.append(dict(file=str(path.relative_to(root)),sha256=sha,bytes=path.stat().st_size))
    Path('results/diffusion_assets.json').write_text(json.dumps(dict(model=metadata['id'],revision=revision,
        license='creativeml-openrail-m',files=records),indent=2))

if __name__=='__main__': main()

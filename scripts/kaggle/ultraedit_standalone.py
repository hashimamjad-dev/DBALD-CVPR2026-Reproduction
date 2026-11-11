"""Download and verify original UltraEdit, then perform one benign edit on Kaggle."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import traceback

OUT = Path('/kaggle/working/ultraedit')
TMP = Path('/kaggle/temp/ultraedit')
OUT.mkdir(parents=True, exist_ok=False)
TMP.mkdir(parents=True, exist_ok=True)
REV = '1af5f0478d56d62cf6f35781a2ab63e03d656b2d'
MODEL = 'BleachNick/SD3_UltraEdit_w_mask'
MODEL_REVISION = 'e5beecb69a3ef1399718be2f99b61730b8bf0346'
report = dict(status='running', fork_revision=REV, model=MODEL, measurements={})
started = time.monotonic()


def stage(name):
    report['stage'] = name
    report['elapsed_seconds'] = time.monotonic() - started
    (OUT / 'receipt.json').write_text(json.dumps(report, indent=2))
    print(name, flush=True)


try:
    stage('isolated_dependencies')
    deps = TMP / 'dependencies'
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--target', str(deps), '--no-deps',
                    'transformers==4.56.2', 'accelerate==1.10.1', 'huggingface-hub==0.34.4',
                    'tokenizers==0.22.2', 'sentencepiece==0.2.1', 'safetensors==0.6.2'],
                   check=True, timeout=600)
    sys.path.insert(0, str(deps))
    stage('pinned_fork')
    fork = TMP / 'UltraEdit'
    subprocess.run(['git', 'clone', '--filter=blob:none', '--no-checkout',
                    'https://github.com/HaozheZhao/UltraEdit.git', str(fork)], check=True, timeout=300)
    subprocess.run(['git', '-C', str(fork), 'checkout', REV], check=True, timeout=300)
    actual = subprocess.check_output(['git', '-C', str(fork), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != REV:
        raise ValueError('Fork revision mismatch')
    sys.path.insert(0, str(fork / 'diffusers/src'))
    os.environ['HF_XET_CHUNK_CACHE_SIZE_BYTES'] = '0'
    from huggingface_hub import HfApi, snapshot_download, constants
    import torch
    import psutil
    from PIL import Image
    from diffusers import StableDiffusion3InstructPix2PixPipeline
    import diffusers
    report['dependencies'] = {p: importlib.metadata.version(p) for p in
                              ('torch', 'transformers', 'accelerate', 'huggingface-hub',
                               'tokenizers', 'sentencepiece', 'safetensors')}
    report['diffusers'] = dict(version=diffusers.__version__, source=diffusers.__file__)
    stage('public_checkpoint_access')
    info = HfApi(token=False).model_info(MODEL, revision=MODEL_REVISION, files_metadata=True)
    report['checkpoint_revision'] = info.sha
    report['public_access_verified'] = True
    required_bytes = sum(f.size or 0 for f in info.siblings)
    report['download_bytes'] = required_bytes
    report['free_disk_bytes'] = shutil.disk_usage(TMP).free
    report['initial_available_ram_bytes'] = psutil.virtual_memory().available
    report['initial_gpu'] = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,memory.free',
                                                    '--format=csv'], text=True)
    candidates = []
    for directory in (Path('/kaggle/input'), Path(constants.HF_HUB_CACHE), TMP / 'cache'):
        if directory.exists():
            for index in directory.rglob('model_index.json'):
                if json.loads(index.read_text()).get('_class_name') == 'StableDiffusion3InstructPix2PixPipeline':
                    candidates.append(index.parent)
    report['existing_candidates'] = [str(p) for p in candidates]
    complete = [p for p in candidates if all((p / f.rfilename).is_file() and
                (p / f.rfilename).stat().st_size == f.size for f in info.siblings)]
    stage('checkpoint_cache_checked')
    if complete:
        snapshot = complete[0]
    else:
        if report['free_disk_bytes'] < required_bytes + 10 * 1024**3:
            raise RuntimeError('Insufficient scratch space for checkpoint and staging margin')
        stage('downloading_original_checkpoint')
        snapshot = Path(snapshot_download(MODEL, revision=info.sha, cache_dir=TMP / 'cache',
                                          token=False, max_workers=2))
    stage('verifying_checkpoint_bytes')
    verified = []
    for entry in info.siblings:
        path = snapshot / entry.rfilename
        if path.stat().st_size != entry.size:
            raise ValueError('File size mismatch: ' + entry.rfilename)
        if entry.lfs:
            expected = entry.lfs.sha256
            with path.open('rb') as stream:
                actual = hashlib.file_digest(stream, 'sha256').hexdigest()
            algorithm = 'sha256'
        else:
            expected = entry.blob_id
            h = hashlib.sha1(('blob ' + str(entry.size)).encode() + bytes([0]))
            h.update(path.read_bytes())
            actual = h.hexdigest()
            algorithm = 'git_blob_sha1'
        if actual != expected:
            raise ValueError('File checksum mismatch: ' + entry.rfilename)
        verified.append(dict(path=entry.rfilename, bytes=entry.size, algorithm=algorithm, digest=actual))
    (OUT / 'checkpoint_files.json').write_text(json.dumps(verified, indent=2))
    index = json.loads((snapshot / 'model_index.json').read_text())
    for component in ('transformer', 'vae', 'text_encoder', 'text_encoder_2', 'text_encoder_3',
                      'tokenizer', 'tokenizer_2', 'tokenizer_3', 'scheduler'):
        if component not in index or not (snapshot / component).is_dir():
            raise ValueError('Missing component: ' + component)
    report['snapshot'] = str(snapshot)
    report['verified_files'] = len(verified)
    stop_monitor = threading.Event()
    process = psutil.Process()

    def monitor():
        while not stop_monitor.wait(0.5):
            m = report['measurements']
            m['peak_process_rss_bytes'] = max(m.get('peak_process_rss_bytes', 0), process.memory_info().rss)
            m['minimum_available_ram_bytes'] = min(m.get('minimum_available_ram_bytes', float('inf')),
                                                    psutil.virtual_memory().available)

    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    stage('loading_full_fp16_pipeline')
    load_start = time.monotonic()
    pipe = StableDiffusion3InstructPix2PixPipeline.from_pretrained(
        str(snapshot), torch_dtype=torch.float16, low_cpu_mem_usage=True, local_files_only=True)
    pipe.enable_model_cpu_offload(gpu_id=0)
    report['measurements']['load_seconds'] = time.monotonic() - load_start
    image = Image.open(fork / 'images/input.png').convert('RGB').resize((512, 512))
    mask = Image.new('RGB', (512, 512), (255, 255, 255))
    image.save(OUT / 'input.png')
    mask.save(OUT / 'mask.png')
    settings = dict(prompt='What if the horse wears a hat?', negative_prompt='', num_inference_steps=50,
                    image_guidance_scale=1.5, guidance_scale=7.5)
    report['settings'] = dict(settings, seed=0, dtype='float16', offload='model_cpu_offload',
                              all_three_text_encoders=True, width=512, height=512)
    stage('one_benign_edit')
    torch.cuda.reset_peak_memory_stats(0)
    inference_start = time.monotonic()
    with torch.inference_mode():
        result = pipe(image=image, mask_img=mask, generator=torch.Generator('cpu').manual_seed(0), **settings)
    torch.cuda.synchronize(0)
    report['measurements']['inference_seconds'] = time.monotonic() - inference_start
    report['measurements']['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated(0)
    report['measurements']['peak_cuda_reserved_bytes'] = torch.cuda.max_memory_reserved(0)
    result.images[0].save(OUT / 'edited.png')
    report['output_sha256'] = hashlib.sha256((OUT / 'edited.png').read_bytes()).hexdigest()
    stop_monitor.set()
    watcher.join(timeout=2)
    report['status'] = 'completed'
    stage('completed')
except Exception:
    report['status'] = 'failed'
    report['error'] = traceback.format_exc()
    stage(report.get('stage', 'unknown'))
    print(report['error'], flush=True)
    raise

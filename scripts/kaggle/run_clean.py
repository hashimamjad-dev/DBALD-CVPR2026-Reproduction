"""Kaggle entry point. Verify transferred bytes, run the existing driver, retain evidence."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import time
import traceback
import zipfile

output = Path("/kaggle/working/evidence")
output.mkdir(exist_ok=False)
started = time.time()
receipt = {"status": "running", "packages": {}}
try:
    root = Path("/kaggle/temp/dbald-research")
    bundles = list(Path("/kaggle/input").rglob("research.zip"))
    manifests = list(Path("/kaggle/input").rglob("execution.json"))
    if len(bundles) == 1:
        with zipfile.ZipFile(bundles[0]) as archive:
            for member in archive.infolist():
                if not (root / member.filename).resolve().is_relative_to(root.resolve()):
                    raise ValueError("Unsafe archive path")
            archive.extractall(root)
    elif len(manifests) == 1:
        shutil.copytree(manifests[0].parent, root)
    else:
        raise ValueError("Expected exactly one uploaded experiment bundle")
    config = json.loads((root / "execution.json").read_text())
    for name, expected in config["sha256"].items():
        with (root / name).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError("Input hash mismatch: " + name)
    for package in ("torch", "torchvision", "numpy", "scipy", "scikit-learn", "ujson"):
        receipt["packages"][package] = importlib.metadata.version(package)
    receipt["execution"] = config
    receipt["gpu"] = subprocess.check_output(["nvidia-smi"], text=True)
    command = [sys.executable, "scripts/benchmark.py", "--model", config["model"],
               "--backbone", config["backbone"], "--checkpoint", config["checkpoint"],
               "--device", "cuda:0", "--expected-count", str(config["count"]),
               "--output", str(output / "benchmark")]
    if config.get("model_config"):
        command.extend(["--config", config["model_config"]])
    receipt["command"] = command
    environment = os.environ.copy()
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    with (output / "benchmark.log").open("w") as log:
        subprocess.run(command, cwd=root, env=environment, stdout=log,
                       stderr=subprocess.STDOUT, timeout=10800, check=True)
    receipt["status"] = "completed"
except Exception:
    receipt["status"] = "failed"
    receipt["error"] = traceback.format_exc()
    raise
finally:
    receipt["wall_seconds"] = time.time() - started
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))



"""Build a private Kaggle input bundle for the existing clean benchmark driver."""
import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--images", type=int, default=2)
    parser.add_argument("--model", choices=("LaneATT", "ADNet", "SCNN", "RESA"), default="LaneATT")
    parser.add_argument("--backbone", choices=("resnet18", "resnet34"), default="resnet34")
    parser.add_argument("--checkpoint", help="Repository-relative checkpoint path")
    parser.add_argument("--config", help="Repository-relative segmentation configuration")
    args = parser.parse_args()
    if args.model != "LaneATT" and not args.checkpoint:
        parser.error("An explicit checkpoint is required for this model")
    if args.model in ("SCNN", "RESA") and not args.config:
        parser.error("A segmentation configuration is required")
    if not 1 <= args.images <= 2782:
        raise ValueError("Image count must be between 1 and 2782")
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint = "models/checkpoints/laneatt_official/experiments/laneatt_r34_tusimple/models/model_0100.pt"
    if args.checkpoint:
        checkpoint = args.checkpoint
    elif args.backbone != "resnet34":
        checkpoint = checkpoint.replace("laneatt_r34", "laneatt_r18")
    for relative in (checkpoint, args.config):
        if relative and not (ROOT / relative).resolve().is_relative_to(ROOT):
            raise ValueError("Assets must stay inside the repository")
    files = set()
    upstream = {"ADNet": "ADNet", "SCNN": "lanedet", "RESA": "lanedet"}.get(args.model)
    if upstream:
        files.update(p for p in (ROOT / "third_party" / upstream).rglob("*")
                     if p.suffix in (".py", ".yaml", ".yml", ".json") and ".git" not in p.parts)
    if args.config:
        files.add(ROOT / args.config)
    for directory in ("src/lane_repro", "third_party/LaneATT/lib"):
        files.update(p for p in (ROOT / directory).rglob("*.py"))
    for name in ("scripts/benchmark.py", "configs/laneatt_anchor_order.json",
                 "third_party/LaneATT/data/tusimple_anchors_freq.pt",
                 "third_party/tusimple-benchmark/evaluate/lane.py", checkpoint):
        files.add(ROOT / name)
    labels_path = ROOT / "datasets/raw/tusimple_test/labels.jsonl"
    rows = labels_path.read_text().splitlines()[:args.images]
    files.update(ROOT / "datasets/raw/tusimple_test" / json.loads(row)["raw_file"] for row in rows)
    hashes = {}
    with zipfile.ZipFile(args.output / "research.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            name = path.relative_to(ROOT).as_posix()
            with path.open("rb") as stream:
                hashes[name] = hashlib.file_digest(stream, "sha256").hexdigest()
            archive.write(path, name)
        labels = chr(10).join(rows) + chr(10)
        archive.writestr("datasets/raw/tusimple_test/labels.jsonl", labels)
        hashes["datasets/raw/tusimple_test/labels.jsonl"] = hashlib.sha256(labels.encode()).hexdigest()
        config = dict(model=args.model, backbone=args.backbone, checkpoint=checkpoint, model_config=args.config,
                      count=args.images, source_commit=subprocess.check_output(
                          ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), sha256=hashes)
        archive.writestr("execution.json", json.dumps(config, indent=2))
    metadata = dict(title="DBALD clean execution inputs", id="hashimamjad03/dbald-clean-execution",
                    licenses=[{"name": "other"}])
    (args.output / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(dict(files=len(files), images=args.images, bytes=(args.output / "research.zip").stat().st_size)))


if __name__ == "__main__":
    main()



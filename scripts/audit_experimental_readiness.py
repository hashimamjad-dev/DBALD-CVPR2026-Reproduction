"""Audit local Table 1 prerequisites without loading models or generating predictions."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/table1_protocol_scnn_20261106"
RAW = ROOT / "datasets/raw"
WEIGHTS = ROOT / "models/checkpoints"


def split_inventory():
    splits = {}
    for split in ("train", "val", "test"):
        root = RAW / "culane"
        listing = root / "list" / (split + ".txt")
        keys = listing.read_text().splitlines()
        images = [root / key.strip().lstrip("/") for key in keys]
        splits["CULane/" + split] = dict(
            listed=len(keys), unique=len(set(keys)), images=sum(p.is_file() for p in images),
            lane_annotations=sum(p.with_suffix(".lines.txt").is_file() for p in images))
        masks = root / "list" / (split + "_gt.txt")
        if masks.exists():
            rows = [line.split() for line in masks.read_text().splitlines()]
            splits["CULane/" + split].update(mask_rows=len(rows), masks_present=sum(
                (root / row[1].lstrip("/")).is_file() for row in rows))
    for split in ("train", "test"):
        root = RAW / ("tusimple_" + split)
        rows = [json.loads(line) for line in (root / "labels.jsonl").read_text().splitlines()]
        keys = [r["raw_file"] for r in rows]
        splits["TuSimple/" + split] = dict(listed=len(rows), unique=len(set(keys)),
            images=sum((root / k).is_file() for k in keys),
            consistent_lane_lengths=sum(all(len(lane) == len(r["h_samples"]) for lane in r["lanes"]) for r in rows))
    return splits


def checkpoint(model, dataset):
    patterns = {"LaneATT": f"**/laneatt_r34_{dataset}/models/*.pt",
                "ADNet": f"adnet_{dataset}_res34.pth", "RESA": f"**/resa_r34_{dataset}.pth",
                "SCNN": f"scnn_r34_autodrive/*{dataset}*.pt"}
    matches = sorted(WEIGHTS.glob(patterns[model]))
    return [p.relative_to(ROOT).as_posix() for p in matches]


def build(output):
    with (BASE / "readiness/table1_readiness.csv").open() as stream:
        cells = list(csv.DictReader(stream))
    keys = {(r["dataset"], r["model"], r["method"].casefold(), r["metric"]) for r in cells}
    if len(keys) != 220 or len(cells) != 220 or len({k[:3] for k in keys}) != 40:
        raise ValueError("Unexpected Table 1 dimensions or duplicate metric")
    splits = split_inventory()
    receipts = json.loads((BASE / "unified_recheck/verification.json").read_text())["verified"]
    scnn = json.loads((ROOT / "configs/scnn_r34_autodrive.json").read_text())["datasets"]
    assets = {}
    for model, dataset in sorted({(row["model"], row["dataset"]) for row in cells}):
        paths = checkpoint(model, dataset.lower())
        records = [r for r in receipts if (r["model"], r["dataset"], r["backbone"]) ==
                   (model, dataset, "resnet34")]
        expected = {r["checkpoint_sha256"] for r in records if r.get("checkpoint_sha256")}
        if model == "SCNN":
            expected.add(scnn[dataset.lower()]["sha256"])
        actual = {}
        for path in paths:
            with (ROOT / path).open("rb") as stream:
                actual[path] = hashlib.file_digest(stream, "sha256").hexdigest()
        assets[model, dataset] = dict(weights=actual, identity_verified=len(actual) == 1 and
            set(actual.values()) == expected, evidence=[r["evidence"] for r in records if r.get("evidence")])
    for row in cells:
        dataset, model = row["dataset"], row["model"]
        paths = assets[model, dataset]["weights"]
        clean = row["metric"] == "benign_acc"
        dbald = row["method"] == "DBALD"
        row.update(architecture=model + " ResNet-34", clean_checkpoint=" | ".join(paths),
            split=dataset + ("/test" if clean else "/train + held-out test"),
            driver="scripts/benchmark" + ("_culane" if dataset == "CULane" else "") + ".py",
            method_components=("Label generation, mixing, UltraEdit path; tiny LaneATT sanity trainer uses placeholder trigger"
                               if dbald else "No validated method-specific training and evaluation entry point identified"),
            scoring=("Official TuSimple JSONL benchmark" if dataset == "TuSimple" else
                     "Paper point ACC unresolved; official CULane IoU/F1 supporting only") if "acc" in row["metric"]
                    else "Scenario-specific target matching and ASR denominator require verification",
            readiness="PARTIAL" if clean and assets[model, dataset]["identity_verified"] else "BLOCKED",
            prerequisite=("Confirm author-equivalent protocol and weights" if clean else
                          "Require validated method configuration, trained weights, held-out predictions and scoring receipt"))
        if row["metric"] == "avg_asr":
            row["scoring"] = "Arithmetic mean of the complete dataset-specific set of scenario ASRs"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "requirements.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=cells[0])
        writer.writeheader()
        writer.writerows(cells)
    with (output / "split_inventory.json").open("x") as stream:
        json.dump(splits, stream, indent=2)
    inventory = {"detectors": {f"{m}/{d}": value for (m, d), value in assets.items()},
                 "implementations": sorted(p.name for p in (ROOT / "third_party").iterdir() if p.is_dir()),
                 "dependency_files": sorted(p.name for p in ROOT.glob("requirements*.txt")),
                 "project_sources": sorted(p.relative_to(ROOT).as_posix() for directory in
                                           ("scripts", "src", "configs", "tests") for p in
                                           (ROOT / directory).rglob("*") if p.suffix in (".py", ".json")),
                 "other_weights": sorted(p.relative_to(ROOT).as_posix() for p in WEIGHTS.rglob("*")
                                         if p.suffix in (".pth", ".pt", ".zip", ".safetensors"))}
    with (output / "assets.json").open("x") as stream:
        json.dump(inventory, stream, indent=2)
    print(Counter(r["readiness"] for r in cells), splits)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    build(parser.parse_args().output)

"""Verify recovered Kaggle clean predictions without promoting subset scores to Table 1."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from lane_repro.evaluation import official_tusimple


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    receipt_paths = list(args.directory.rglob("receipt.json"))
    if len(receipt_paths) != 1:
        raise ValueError("Expected one execution receipt")
    folder = receipt_paths[0].parent
    receipt = json.loads(receipt_paths[0].read_text())
    if receipt["status"] != "completed":
        raise ValueError(receipt.get("error", "Remote run incomplete"))
    config = receipt["execution"]
    for name, expected in config["sha256"].items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError("Unsafe provenance path")
        if name != "datasets/raw/tusimple_test/labels.jsonl" and digest(path) != expected:
            raise ValueError("Transferred asset differs: " + name)
    checkpoint = ROOT / config["checkpoint"]
    if digest(checkpoint) != config["sha256"][config["checkpoint"]]:
        raise ValueError("Checkpoint differs from local asset")
    labels = (ROOT / "datasets/raw/tusimple_test/labels.jsonl").read_text().splitlines()[:config["count"]]
    content = chr(10).join(labels) + chr(10)
    if hashlib.sha256(content.encode()).hexdigest() != config["sha256"]["datasets/raw/tusimple_test/labels.jsonl"]:
        raise ValueError("Input split differs")
    truth = {r["raw_file"]: r for r in map(json.loads, labels)}
    benchmark = folder / "benchmark"
    rows = [json.loads(line) for line in (benchmark / "predictions.jsonl").read_text().splitlines()]
    if len(rows) != config["count"] or {r["raw_file"] for r in rows} != set(truth):
        raise ValueError("Incomplete predictions")
    evaluator = official_tusimple()
    values = []
    for row in rows:
        gt = truth[row["raw_file"]]
        if row["h_samples"] != gt["h_samples"] or row["transformation"] != "clean":
            raise ValueError("Prediction protocol mismatch")
        values.append(evaluator.bench(row["lanes"], gt["lanes"], gt["h_samples"], row["run_time"])[0])
    summary = json.loads((benchmark / "summary.json").read_text())[0]
    acc = 100 * sum(values) / len(values)
    if not math.isfinite(acc) or not math.isfinite(summary["acc_percent"]) or abs(acc - summary["acc_percent"]) > 1e-9:
        raise ValueError("Remote and local scorer disagree")
    report = dict(scope=summary["scope"], images=len(rows), locally_rescored_acc_percent=acc,
                  checkpoint_sha256=digest(checkpoint), table1_eligible=len(rows) == 2782,
                  artifacts={p.relative_to(folder).as_posix(): digest(p) for p in folder.rglob("*") if p.is_file()})
    with (args.directory / "recovery_verification_final.json").open("x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()


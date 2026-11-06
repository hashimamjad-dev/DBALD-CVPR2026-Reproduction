"""Recompute full TuSimple ACC from verified saved predictions, without model inference."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from audit_table1 import sha, jsonlines, read
from lane_repro.evaluation import official_tusimple


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = read(args.verification)
    truth_path = ROOT / "datasets/raw/tusimple_test/labels.jsonl"
    truth_rows = jsonlines(truth_path)
    truth = {row["raw_file"]: row for row in truth_rows}
    if len(truth) != 2782 or len(truth_rows) != 2782:
        raise ValueError("Official full split must contain 2782 unique frames")
    evaluator = official_tusimple()
    results = []
    for record in receipt["verified"]:
        if record["dataset"] != "TuSimple":
            continue
        folder = ROOT / record["evidence"]
        predictions_path = folder / "predictions.jsonl"
        if sha(predictions_path) != record["predictions_sha256"]:
            raise ValueError("Predictions changed")
        if sha(truth_path) != record["labels_sha256"]:
            raise ValueError("Annotations changed")
        rows = [r for r in jsonlines(predictions_path) if r["transformation"] == "clean"]
        keys = [r["raw_file"] for r in rows]
        if len(keys) != 2782 or len(set(keys)) != 2782 or set(keys) != set(truth):
            raise ValueError("Incomplete or duplicate predictions")
        scores = []
        for row in rows:
            target = truth[row["raw_file"]]
            if row["h_samples"] != target["h_samples"]:
                raise ValueError("Prediction sampling changed")
            values = evaluator.bench(row["lanes"], target["lanes"], target["h_samples"], row["run_time"])
            scores.append(values)
        acc = 100 * sum(v[0] for v in scores) / len(scores)
        if not math.isfinite(acc) or abs(acc - record["value_percent"]) > 1e-9:
            raise ValueError("Recomputed ACC differs from accepted measurement")
        results.append(dict(model=record["model"], backbone=record["backbone"], images=len(rows),
                            acc_percent=acc, fp=sum(v[1] for v in scores)/len(scores),
                            fn=sum(v[2] for v in scores)/len(scores),
                            predictions_sha256=sha(predictions_path),
                            checkpoint_sha256=record["checkpoint_sha256"],
                            evidence=record["evidence"], execution="CPU replay of saved GPU predictions"))
        print(json.dumps(results[-1]), flush=True)
    output = dict(scope="Independent full TuSimple clean measurements; original inference retained",
                  verification_sha256=sha(args.verification), labels_sha256=sha(truth_path),
                  scorer_sha256=sha(ROOT / "third_party/tusimple-benchmark/evaluate/lane.py"),
                  measurements=results)
    with args.output.open("x") as stream:
        json.dump(output, stream, indent=2)


if __name__ == "__main__":
    main()

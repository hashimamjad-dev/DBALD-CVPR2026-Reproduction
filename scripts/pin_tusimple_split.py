"""Pin a deterministic, clip-grouped TuSimple train/validation partition (CPU only).

Whole clips (dataset directory + clip id) are assigned to one side so adjacent
frames never leak across the split. Note: the labeled TuSimple
train set has exactly one labeled frame (frame 20) per clip, so grouping is per-frame;
neighbouring clip IDs are NOT grouped (temporal adjacency is not controlled). Order is fixed by SHA-256 of the clip name
with a fixed salt; no RNG state is involved.
"""
import argparse, collections, hashlib, json
from pathlib import Path

SALT = "dbald-table1-tusimple-split-v1"


def clip_of(raw_file: str) -> str:
    parts = raw_file.split("/")
    return "/".join(parts[:3]) if parts[0] == "clips" else "/".join(parts[:-1])


def build(labels: Path, val_fraction: float) -> dict:
    rows = [json.loads(l) for l in labels.open()]
    clips = collections.defaultdict(list)
    for r in rows:
        clips[clip_of(r["raw_file"])].append(r["raw_file"])
    order = sorted(clips, key=lambda c: hashlib.sha256((SALT + c).encode()).hexdigest())
    target, val, n = val_fraction * len(rows), [], 0
    for c in order:
        if n >= target:
            break
        val.append(c)
        n += len(clips[c])
    val_set = set(val)
    split = {"val": sorted(f for c in val for f in clips[c]),
             "train": sorted(f for c in clips if c not in val_set for f in clips[c])}
    assert not set(split["val"]) & set(split["train"])
    assert len(split["val"]) + len(split["train"]) == len(rows)
    return {"salt": SALT, "val_fraction": val_fraction, "total": len(rows),
            "n_clips": len(clips), "val_clips": sorted(val_set),
            "counts": {k: len(v) for k, v in split.items()}, **split}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="datasets/raw/tusimple_train/labels.jsonl")
    ap.add_argument("--out", default="results/table1_split_20261111/tusimple_split.json")
    ap.add_argument("--val-fraction", type=float, default=0.1)
    a = ap.parse_args()
    out = build(Path(a.labels), a.val_fraction)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(out["counts"], out["n_clips"], "clips;",
          hashlib.sha256(Path(a.out).read_bytes()).hexdigest())

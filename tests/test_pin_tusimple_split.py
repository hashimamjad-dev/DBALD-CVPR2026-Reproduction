import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from pin_tusimple_split import build


def test_split_is_deterministic_disjoint_and_clip_grouped(tmp_path):
    p = tmp_path / "l.jsonl"
    rows = [{"raw_file": f"clips/0313-1/{c}/{i}.jpg"} for c in range(10) for i in range(1, 21)]
    p.write_text("\n".join(json.dumps(r) for r in rows))
    a, b = build(p, 0.2), build(p, 0.2)
    assert a == b
    assert not set(a["train"]) & set(a["val"])
    assert a["counts"]["train"] + a["counts"]["val"] == 200
    vc = {"/".join(f.split("/")[:3]) for f in a["val"]}
    assert all("/".join(f.split("/")[:3]) not in vc for f in a["train"])

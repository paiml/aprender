"""Spike 027: build the s16 and s64 stance data dirs (task.json / train.jsonl / eval.jsonl).

    uv run --frozen --project scripts/laya_train python .planning/spikes/027-laya-calibration-slice-and-tcap/prepare_data.py

Reuses prepare_stance.py's task text, reader and per-shot verification (exact + normalized hash + label)
for any committed selection manifest. Writes to this spike's gitignored data/ (tweet text is never
committed). The s16 output must be byte-identical to data/decide/tweet-stance-16 (checked below).
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "laya_train"))
import contract  # noqa: E402
import prepare_stance as ps  # noqa: E402
from data import exact_sha256, normalized_sha256, sha256_file  # noqa: E402

HERE = Path(__file__).resolve().parent
SELECTIONS = {"s16": "s16-seed13", "s64": "s64-seed13"}


def build(name, sel):
    demo = contract.demo()
    order = list(demo["criteria_order"])
    manifest = json.loads((REPO / "benchmarks/tweeteval-stance/selections" / sel / "selection-manifest.json").read_text())
    payload = manifest["payload"]
    assert payload["label_names"] == order, payload["label_names"]
    train, _ = ps.read_jsonl(ps.SRC / "train.jsonl")
    test, test_order = ps.read_jsonl(ps.SRC / "test.jsonl")
    shots = []
    for ex in payload["ordered_examples"]:
        r = train[ex["id"]]
        assert exact_sha256(r["input"]) == ex["exact_hash"], ex["id"]
        assert normalized_sha256(r["input"]) == ex["normalized_hash"], ex["id"]
        assert int(r["label"]) == int(ex["label"]) and r["label_text"] == order[int(ex["label"])], ex["id"]
        shots.append((r["input"], order[int(ex["label"])]))
    eval_rows = [(test[t]["input"], test[t]["label_text"]) for t in test_order]
    assert len(eval_rows) == int(demo["eval_rows"])
    out = HERE / "data" / name
    out.mkdir(parents=True, exist_ok=True)
    task = {"type": "choice", "instructions": ps.INSTRUCTIONS, "criteria": ps.DESCRIPTIONS}
    (out / "task.json").write_bytes((json.dumps(task, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    for fn, rows in (("train.jsonl", shots), ("eval.jsonl", eval_rows)):
        body = "".join(json.dumps({"text": t, "label": lab}, ensure_ascii=False) + "\n" for t, lab in rows)
        (out / fn).write_bytes(body.encode("utf-8"))
    per_class = {lab: sum(1 for _, x in shots if x == lab) for lab in order}
    shas = {f: sha256_file(out / f) for f in ("task.json", "train.jsonl", "eval.jsonl")}
    print("prepared %s from %s: train %d %s, eval %d; sha256 %s" % (name, sel, len(shots), per_class, len(eval_rows), shas))
    return shas


def main():
    got = {n: build(n, s) for n, s in SELECTIONS.items()}
    ref = REPO / "data" / "decide" / "tweet-stance-16"
    if ref.is_dir():
        same = all(sha256_file(ref / f) == got["s16"][f] for f in got["s16"])
        print("s16 byte-identical to data/decide/tweet-stance-16: %s" % same)
        if not same:
            sys.exit(1)
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "data-sha256.json").write_text(json.dumps(got, indent=2, sort_keys=True) + "\n")
    print("PREPARE OK")


if __name__ == "__main__":
    main()

"""Rebuild spike 027's 459-row in-distribution held-out set as a data dir the Rust dump can read.

    uv run --frozen --project scripts/laya_train python .planning/spikes/028-laya-packability-noise-floor/build_heldout.py

Same rule, same order as 027's diagnose_indist.py: TweetEval stance validation (minus the manifest's
exclusion-group partner validation:3) + every train-pool row in no s64-seed13 shot and not excluded,
text-overlap with the s64 shots refused, in-set duplicates dropped by normalized text.
Writes data/indist/{eval.jsonl,task.json} (gitignored) and results/indist-data.json (sha256 + counts).
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "laya_train"))
import data  # noqa: E402

HERE = Path(__file__).resolve().parent
D27 = REPO / ".planning/spikes/027-laya-calibration-slice-and-tcap/data"
task = data.load_task(D27 / "s64/task.json")
labels = task["labels"]
sel = json.loads((REPO / "benchmarks/tweeteval-stance/selections/s64-seed13/selection-manifest.json").read_text())["payload"]
used = {e["id"] for e in sel["ordered_examples"]} | set(sel["exclusions"]["excluded_train_ids"])
rows = [json.loads(l) for l in (REPO / "data/tweet-eval-stance/validation.jsonl").read_text().splitlines()]
rows = [r for r in rows if r["id"] not in {"validation:3"}]
rows += [r for r in (json.loads(l) for l in (REPO / "data/tweet-eval-stance/train.jsonl").read_text().splitlines())
         if r["id"] not in used]
hold = [(r["input"], labels.index(r["label_text"])) for r in rows]
train64 = data.load_rows(D27 / "s64/train.jsonl", task, "train")
data.refuse_overlap(train64, hold)
seen, uniq = set(), []
for t, l in hold:
    k = data.normalize(t)
    if k not in seen:
        seen.add(k)
        uniq.append((t, l))
out = HERE / "data/indist"
out.mkdir(parents=True, exist_ok=True)
shutil.copyfile(D27 / "s64/task.json", out / "task.json")
body = "".join(json.dumps({"text": t, "label": labels[l]}) + "\n" for t, l in uniq)
(out / "eval.jsonl").write_text(body)
info = {"n": len(uniq), "classes": [sum(1 for _, l in uniq if l == i) for i in range(len(labels))],
        "eval_jsonl_sha256": hashlib.sha256(body.encode()).hexdigest()}
(HERE / "results/indist-data.json").write_text(json.dumps(info, indent=1) + "\n")
print(info)

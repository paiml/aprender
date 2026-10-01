"""Would the SAME checkpoints and the SAME slice-fitted T be calibrated on data drawn like the shots?

    uv run --frozen --project scripts/laya_train python .planning/spikes/027-laya-calibration-slice-and-tcap/diagnose_indist.py

"In-distribution held-out" = TweetEval stance validation (66) + every train-pool row that is in no s64-seed13
shot and not the manifest's excluded train:70 (so disjoint from every s16/s64 shot and calibration row,
text-overlap refused). It is NOT the gate's eval set and NOT a proposal to swap it silently: it isolates the
train->test split shift as the cause of the ECE failure. Metrics use the run's own slice-fitted T (clamped to 5
and to 10), i.e. no label of this set chooses anything. Zero-shot = the declared base on the same rows.
Writes results/indist.json.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "laya_train"))
import data  # noqa: E402
import gate  # noqa: E402
import metrics  # noqa: E402
import train  # noqa: E402
from common import tree_sha256  # noqa: E402
import types  # noqa: E402

HERE = Path(__file__).resolve().parent
task = data.load_task(HERE / "data/s16/task.json")
labels = task["labels"]
question = data.laya_question(task)
sel = json.loads((REPO / "benchmarks/tweeteval-stance/selections/s64-seed13/selection-manifest.json").read_text())["payload"]
used = {e["id"] for e in sel["ordered_examples"]} | set(sel["exclusions"]["excluded_train_ids"])
rows = [json.loads(l) for l in (REPO / "data/tweet-eval-stance/validation.jsonl").read_text().splitlines()]
rows = [r for r in rows if r["id"] not in {"validation:3"}]      # the manifest's exclusion group partner
rows += [r for r in (json.loads(l) for l in (REPO / "data/tweet-eval-stance/train.jsonl").read_text().splitlines())
         if r["id"] not in used]
hold = [(r["input"], labels.index(r["label_text"])) for r in rows]
train64 = data.load_rows(HERE / "data/s64/train.jsonl", task, "train")
data.refuse_overlap(train64, hold)
seen, uniq = set(), []
for t, l in hold:                           # drop in-set duplicates by normalized text
    k = data.normalize(t)
    if k not in seen:
        seen.add(k)
        uniq.append((t, l))
hold = uniq
yh = np.array([l for _, l in hold])
print("in-distribution held-out rows: %d, classes %s" % (len(yh), np.bincount(yh, minlength=3).tolist()), flush=True)
f_avg_labels = [labels.index("against"), labels.index("favor")]
cache = HERE / "data" / "indist-zero-shot-logits.json"
if cache.is_file():
    Zzs = np.array(json.loads(cache.read_text()))
else:
    base = train.Base(types.SimpleNamespace(variant="production", base=None, base_sha256=None))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        zs = train.load_for_scoring(base.src, base.digest, base.revision)
    _, Zzs = train.score_rows(zs, hold, question)
    del zs
    cache.write_text(json.dumps(Zzs.astype(float).tolist()))
Zzs = np.asarray(Zzs, dtype=np.float64)
zs_f1 = metrics.macro_f1(gate.softmax(Zzs, 1.0), yh)
out = {"n": int(len(yh)), "classes": np.bincount(yh, minlength=3).tolist(), "zs_macro_f1": zs_f1, "runs": []}
print("zero-shot macro-F1 %.4f" % zs_f1, flush=True)
for m in sorted((REPO / "models/decide/spike-027").glob("*/spike-metrics.json")):
    ck = m.parent / "checkpoint"
    if not ck.is_dir():
        continue
    r = json.loads(m.read_text())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        agent = train.reload_checked(ck, tree_sha256(ck))
    _, Z = train.score_rows(agent, hold, question)
    del agent
    Z = Z.astype(np.float64)
    f1 = metrics.macro_f1(gate.softmax(Z, 1.0), yh)
    row = {"run": m.parent.name, "ft_macro_f1": f1, "margin": f1 - zs_f1, "acc": float((Z.argmax(1) == yh).mean()),
           "t5": r["t_fitted_5"], "t10": r["t_fitted_10"],
           "ece_at_t5": metrics.ece_top_label(gate.softmax(Z, r["t_fitted_5"]), yh, 15),
           "ece_at_t10": metrics.ece_top_label(gate.softmax(Z, r["t_fitted_10"]), yh, 15),
           "t_oracle_nll": gate.fit_temperature(Z, yh, 0.05, 1000.0)[0],
           "gate_eval_ece_at_t5": r["ece_post_5"], "gate_eval_margin": r["margin"]}
    row["pass_at_t5"] = bool(row["margin"] >= 0.05 and row["ece_at_t5"] <= 0.10)
    row["pass_at_t10"] = bool(row["margin"] >= 0.05 and row["ece_at_t10"] <= 0.10)
    out["runs"].append(row)
    print("%-22s in-dist: margin %.3f acc %.3f ECE@T5 %.3f ECE@T10 %.3f oracle-T %.2f | gate eval: margin %.3f ECE@T5 %.3f"
          % (row["run"], row["margin"], row["acc"], row["ece_at_t5"], row["ece_at_t10"], row["t_oracle_nll"],
             row["gate_eval_margin"], row["gate_eval_ece_at_t5"]), flush=True)
(HERE / "results" / "indist.json").write_text(json.dumps(out, indent=1) + "\n")

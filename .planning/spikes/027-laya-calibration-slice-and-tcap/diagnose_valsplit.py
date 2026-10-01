"""Is the slice-vs-eval temperature gap slice-size NOISE or a train->test SHIFT?

    uv run --frozen --project scripts/laya_train python .planning/spikes/027-laya-calibration-slice-and-tcap/diagnose_valsplit.py

For every spike-027 run whose checkpoint is still on disk: reload it exactly as train.py does (fp32 CPU,
tree sha256 asserted), score TweetEval stance VALIDATION (66 rows, disjoint from the shots and from eval,
never used by any run), and fit T by NLL there. If T_val ~ T_slice (both train-distribution) while the
eval-oracle T is higher, no train-side calibration set of any size recovers eval's T: the gap is the
train/test split shift, not slice noise. Eval labels are read only for the ECE at T_val (diagnostic).
Writes results/valsplit.json.
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

HERE = Path(__file__).resolve().parent
task = data.load_task(HERE / "data/s16/task.json")
labels = task["labels"]
question = data.laya_question(task)
y = np.array([l for _, l in data.load_rows(HERE / "data/s16/eval.jsonl", task, "eval")])
val = [json.loads(l) for l in (REPO / "data/tweet-eval-stance/validation.jsonl").read_text().splitlines()]
val_rows = [(r["input"], labels.index(r["label_text"])) for r in val]
train64 = data.load_rows(HERE / "data/s64/train.jsonl", task, "train")
data.refuse_overlap(train64, val_rows)            # validation shares no text with any shot
data.refuse_overlap(val_rows, data.load_rows(HERE / "data/s16/eval.jsonl", task, "eval"))
yv = np.array([l for _, l in val_rows])
WIDE = (0.05, 1000.0)
out = []
for m in sorted((REPO / "models/decide/spike-027").glob("*/spike-metrics.json")):
    ck = m.parent / "checkpoint"
    if not ck.is_dir():
        continue
    r = json.loads(m.read_text())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        agent = train.reload_checked(ck, tree_sha256(ck))
    _, Zv = train.score_rows(agent, val_rows, question)
    del agent
    Zv = Zv.astype(np.float64)
    rep = json.loads((m.parent / "gate-report.json").read_text())
    P = np.array([x["probabilities"] for x in json.loads((m.parent / "eval-probs.json").read_text())["rows"]], dtype=np.float64)
    Ze = rep["calibration"]["t_applied"] * np.log(P)
    t_val = gate.fit_temperature(Zv, yv, *WIDE)[0]
    row = {"run": m.parent.name, "val_n": len(yv), "val_acc": float((Zv.argmax(1) == yv).mean()),
           "calib_acc": r["calib_acc"], "eval_acc": r["eval_acc"], "t_slice": r["t_fitted_unclamped"], "t_val": t_val,
           "t_eval_oracle_nll": r["EVAL_ORACLE"]["t_min_nll"], "t_eval_oracle_ece": r["EVAL_ORACLE"]["t_min_ece"],
           "ece_eval_at_t_val": metrics.ece_top_label(gate.softmax(Ze, t_val), y, 15),
           "ece_eval_at_t_val_cap10": metrics.ece_top_label(gate.softmax(Ze, min(10.0, t_val)), y, 15),
           "ece_eval_at_t_slice": r["ece_post_unclamped"], "margin": r["margin"]}
    if "calib_logits" in r:   # pooled slice + validation (a bigger train-distribution calibration set)
        Zp = np.concatenate([np.array(r["calib_logits"]), Zv])
        yp = np.concatenate([np.array(r["calib_labels"]), yv])
        row["t_pooled"] = gate.fit_temperature(Zp, yp, *WIDE)[0]
        row["ece_eval_at_t_pooled"] = metrics.ece_top_label(gate.softmax(Ze, row["t_pooled"]), y, 15)
    out.append(row)
    print("%-22s acc cal/val/eval %.3f/%.3f/%.3f | T slice %.2f  val %.2f  eval-oracle(NLL) %.2f | ECE eval @T_slice %.3f @T_val %.3f"
          % (row["run"], row["calib_acc"], row["val_acc"], row["eval_acc"], row["t_slice"], t_val,
             row["t_eval_oracle_nll"], row["ece_eval_at_t_slice"], row["ece_eval_at_t_val"]), flush=True)
(HERE / "results" / "valsplit.json").write_text(json.dumps(out, indent=1) + "\n")

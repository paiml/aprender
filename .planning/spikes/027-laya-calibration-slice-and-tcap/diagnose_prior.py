"""Is the slice-vs-eval T gap label-prior shift? (reads eval labels: DIAGNOSTIC only, never a selection rule)

For every spike run: eval logits recovered from eval-probs.json at the applied T; the NLL-optimal T on eval
(a) as-is (eval prior: 67.5 % against) and (b) class-reweighted to a balanced prior (the slice's prior);
macro recall (= accuracy on a class-balanced eval) vs the calibration-slice accuracy. Writes results/prior-shift.json.
"""
import json, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "laya_train"))
import data, gate, metrics  # noqa: E402
HERE = Path(__file__).resolve().parent
task = data.load_task(HERE / "data/s16/task.json")
y = np.array([l for _, l in data.load_rows(HERE / "data/s16/eval.jsonl", task, "eval")])
K = 3
w_bal = np.array([1.0 / (y == c).sum() for c in range(K)])[y]
w_bal = w_bal / w_bal.sum() * len(y)


def wfit(Z, w, lo=0.05, hi=1000.0):
    def d(beta):
        p = gate.softmax(Z, 1.0 / beta)
        return float((w * ((p * Z).sum(1) - Z[np.arange(len(y)), y])).mean())
    a, b = 1.0 / hi, 1.0 / lo
    if d(a) >= 0: return hi
    if d(b) <= 0: return lo
    for _ in range(200):
        m = 0.5 * (a + b)
        a, b = (m, b) if d(m) < 0 else (a, m)
    return 1.0 / (0.5 * (a + b))


out = []
for m in sorted((REPO / "models/decide/spike-027").glob("*/spike-metrics.json")):
    r = json.loads(m.read_text())
    rep = json.loads((m.parent / "gate-report.json").read_text())
    P = np.array([x["probabilities"] for x in json.loads((m.parent / "eval-probs.json").read_text())["rows"]], dtype=np.float64)
    Z = rep["calibration"]["t_applied"] * np.log(P)
    pred = Z.argmax(1)
    rec = [float((pred[y == c] == c).mean()) for c in range(K)]
    row = {"run": m.parent.name, "calib_acc": r["calib_acc"], "eval_acc": r["eval_acc"], "eval_macro_recall": float(np.mean(rec)),
           "eval_recall_per_class": dict(zip(task["labels"], rec)), "t_slice_unclamped": r["t_fitted_unclamped"],
           "t_eval_nll": wfit(Z, np.ones(len(y))), "t_eval_nll_balanced": wfit(Z, w_bal)}
    out.append(row)
    print("%-22s cal_acc %.3f | eval acc %.3f macro-recall %.3f recall %s | T slice %.2f  eval-NLL %.2f  eval-NLL balanced %.2f" % (
        row["run"], row["calib_acc"], row["eval_acc"], row["eval_macro_recall"],
        "/".join("%.2f" % v for v in rec), row["t_slice_unclamped"], row["t_eval_nll"], row["t_eval_nll_balanced"]))
(HERE / "results" / "prior-shift.json").write_text(json.dumps(out, indent=1) + "\n")

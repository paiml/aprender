"""Read-only: ECE-vs-T on the three existing D-19 demo run dirs (logits recovered from eval-probs at t_applied).
Writes only results/demo-oracle.json in this spike dir; never touches models/decide/tweet-stance-16*."""
import json, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "laya_train"))
import data, gate, metrics  # noqa: E402
HERE = Path(__file__).resolve().parent
task = data.load_task(HERE / "data/s16/task.json")
y = np.array([l for _, l in data.load_rows(HERE / "data/s16/eval.jsonl", task, "eval")])
out = {}
for d in ("tweet-stance-16-fixed-epochs", "tweet-stance-16", "tweet-stance-16-var"):
    run = REPO / "models/decide" / d
    rep = json.loads((run / "gate-report.json").read_text())
    P = np.array([r["probabilities"] for r in json.loads((run / "eval-probs.json").read_text())["rows"]], dtype=np.float64)
    Z = rep["calibration"]["t_applied"] * np.log(P)
    curve = {t: metrics.ece_top_label(gate.softmax(Z, t), y, 15) for t in (1.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0)}
    grid = np.exp(np.linspace(np.log(0.5), np.log(50), 461))
    e = np.array([metrics.ece_top_label(gate.softmax(Z, t), y, 15) for t in grid])
    out[d] = {"recipe_id": rep["recipe_id"][:8], "t_applied": rep["calibration"]["t_applied"], "ece_post": rep["fine_tuned"]["ece_post"],
              "margin": rep["margin"], "oracle_t": float(grid[e.argmin()]), "oracle_ece": float(e.min()),
              "best_ece_le5": float(e[grid <= 5.0001].min()), "best_ece_le10": float(e[grid <= 10.0001].min()),
              "oracle_nll_t": gate.fit_temperature(Z, y, 0.05, 1000)[0], "curve": curve}
    print(d, json.dumps({k: v for k, v in out[d].items() if k != "curve"}), {k: round(v, 4) for k, v in curve.items()})
(HERE / "results" / "demo-oracle.json").write_text(json.dumps(out, indent=1) + "\n")

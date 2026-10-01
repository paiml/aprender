"""Run a Laya checkpoint over a spike-015 task split; save per-row probabilities, latency and the frozen head inputs.

The head inputs are captured by forward hooks on Laya's OWN inference path (Agent.predict), so nothing here
re-implements its sequence builder, temperature buckets or decoding:
  m_opts [K, d]   input to `scorer` = each option's [MASK] marker state after the 2-layer head (Kev's h_opts analogue)
  pooled [d]      first d dims of the `act_head` input = post-head [CLS] state (Kev's h_decide analogue)
  logits [K]      raw scorer logits (before the per-bucket temperature)

  uv run --with ./vendor/laya --with datasets python tools/laya_eval.py --subfolder "" --task stance-abortion --split test \
      --out runs/laya-en-stance-abortion-test.npz
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
from tasks import TASKS, load
from laya import Agent

REV = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"  # convaiinnovations/laya, 2026-09-24

ap = argparse.ArgumentParser()
ap.add_argument("--subfolder", default="", help='"" = English root, "typed-decisions"')
ap.add_argument("--task", required=True); ap.add_argument("--split", choices=["train", "test"], required=True)
ap.add_argument("--limit", type=int, default=None); ap.add_argument("--names_only", action="store_true")
ap.add_argument("--device", default="mps"); ap.add_argument("--out", required=True)
a = ap.parse_args()

t0 = time.perf_counter()
agent = Agent("convaiinnovations/laya", device=a.device, subfolder=a.subfolder or None, revision=REV)
load_s = time.perf_counter() - t0
cap = {}
agent.model.scorer.register_forward_hook(lambda mod, inp, out: cap.__setitem__("m", inp[0].detach().float().cpu()))
agent.model.scorer.register_forward_hook(lambda mod, inp, out: cap.__setitem__("z", out.detach().float().cpu()))
d = agent.model.encoder.config.hidden_size
agent.model.act_head.register_forward_hook(lambda mod, inp, out: cap.__setitem__("pooled", inp[0][:, :d].detach().float().cpu()))

t = TASKS[a.task]
crit = {k: None for k in t["criteria"]} if a.names_only else t["criteria"]
labels = t["labels"]
rows = load(a.task, a.split, a.limit)
q = {"label": {"type": "choice", "instructions": t["instructions"], "criteria": crit}}
probs, lat, mo, po, zl, ntok = [], [], [], [], [], []
for i, (text, _) in enumerate(rows):
    cap.clear()
    s = time.perf_counter()
    out = agent.predict(text, q)
    lat.append(1000 * (time.perf_counter() - s))
    pr = out["label"]["probabilities"] if "label" in out else out["answers"]["label"]["probabilities"]
    probs.append([pr[k] for k in labels])          # criteria keys are in label order
    K = len(labels)
    mo.append(cap["m"][0, :K].numpy()); po.append(cap["pooled"][0].numpy()); zl.append(cap["z"][0, :K, 0].numpy())
    if i % 200 == 0:
        print(f"{i}/{len(rows)} {lat[-1]:.0f} ms", flush=True)

y = np.array([r[1] for r in rows]); P = np.array(probs, dtype=np.float64)
T = agent.temperature_by_options.get(f"choice:{'2' if len(labels)==2 else '3-5' if len(labels)<=5 else '6-10'}", float(agent.temperature[0]))
Pz = torch.softmax(torch.from_numpy(np.stack(zl)) / T, -1).numpy()
np.savez(a.out, probs=P, y=y, m_opts=np.stack(mo), pooled=np.stack(po), logits=np.stack(zl), latency_ms=np.array(lat),
         temperature=np.float32(T))
meta = {"run": f"laya/{a.subfolder or 'en'}@{REV[:8]}", "task": a.task, "split": a.split, "n": len(rows),
        "names_only": a.names_only, "device": a.device, "load_s": load_s, "accuracy": float((P.argmax(1) == y).mean()),
        "latency_ms_p50": float(np.median(lat)), "temperature": float(T),
        "recomputed_probs_max_abs_diff": float(np.abs(Pz - P).max())}   # 4-dp rounding in the API -> ~5e-5
Path(a.out).with_suffix(".json").write_text(json.dumps(meta, indent=1))
print(json.dumps(meta))

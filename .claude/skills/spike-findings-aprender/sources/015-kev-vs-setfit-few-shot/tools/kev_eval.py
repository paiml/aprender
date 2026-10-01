"""Run a Kev checkpoint over a task split; save per-row probabilities, latency and the frozen head inputs.

The head inputs (h_decide [d], h_opts [K, d]) are captured by wrapping model.head -- the one call both the torch
and MLX backends make -- so zero-shot scoring and the head-only few-shot variants share one forward pass.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[1] / "vendor" / "kev"))
from tasks import TASKS, load, request
from kev.api import SystemOneRequest, to_record
from kev.checkpoint import LoadOptions, Checkpoint

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="jaredpalmer/kev-0.8b")
ap.add_argument("--task", required=True)
ap.add_argument("--split", choices=["train", "test"], required=True)
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--names_only", action="store_true", help="criteria without descriptions")
ap.add_argument("--backend", default="mlx")
ap.add_argument("--dtype", default=None, help="fp32 forces the exact torch path")
ap.add_argument("--device", default="mps")
ap.add_argument("--out", required=True)
a = ap.parse_args()

opts = LoadOptions(backend=a.backend, dtype={"fp32": torch.float32, "bf16": torch.bfloat16}.get(a.dtype))
t0 = time.perf_counter()
tok, model = Checkpoint(a.run).load(a.device, opts)
load_s = time.perf_counter() - t0

captured = []
real_head = model.head
class Capture(torch.nn.Module):
    def forward(self, hd, ho):
        captured.append((torch.as_tensor(np.asarray(hd, dtype=np.float32)), torch.as_tensor(np.asarray(ho, dtype=np.float32))))
        return real_head(hd, ho)
    def __getattr__(self, n):
        try: return super().__getattr__(n)
        except AttributeError: return getattr(real_head, n)
model.head = Capture()

t = TASKS[a.task]
crit = {k: None for k in t["criteria"]} if a.names_only else t["criteria"]
rows = load(a.task, a.split, a.limit)
probs, lat, hd, ho, ntok = [], [], [], [], []
for i, (text, _) in enumerate(rows):
    rec, _meta = to_record(SystemOneRequest(**request(a.task, text, crit)))
    enc = model.encode(tok, rec, max_state=8192, max_branch=8192, strict=True)
    captured.clear()
    s = time.perf_counter()
    with torch.no_grad():
        p = model.probs(enc)[0]
    lat.append(1000 * (time.perf_counter() - s))
    probs.append(np.asarray(p, dtype=np.float64)); ntok.append(len(enc["ids"]))
    (d, o), = captured
    hd.append(d.numpy()); ho.append(o.numpy())
    if i % 100 == 0: print(f"{i}/{len(rows)} {lat[-1]:.0f} ms", flush=True)

y = np.array([r[1] for r in rows]); P = np.stack(probs)
acc = float((P.argmax(1) == y).mean())
np.savez_compressed(a.out, probs=P, y=y, h_decide=np.stack(hd), h_opts=np.stack(ho), latency_ms=np.array(lat), n_tokens=np.array(ntok),
                    temperature=float(getattr(real_head, "temperature", 1.0)))
meta = {"run": a.run, "task": a.task, "split": a.split, "n": len(rows), "names_only": a.names_only, "backend": type(model).__name__,
        "dtype": getattr(model, "dtype", None), "load_s": load_s, "accuracy": acc, "latency_ms_p50": float(np.median(lat)),
        "tokens_p50": float(np.median(ntok)), "head_temperature": float(getattr(real_head, "temperature", 1.0))}
json.dump(meta, open(a.out.replace(".npz", ".json"), "w"), indent=1)
print(json.dumps(meta))

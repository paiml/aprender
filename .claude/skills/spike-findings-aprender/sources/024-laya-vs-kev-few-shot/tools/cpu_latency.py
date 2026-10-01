"""Laya fp32 torch CPU latency per decision on the stance test rows (the spike-017 fixture's first 6 records)."""
import sys, time, json
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
from tasks import TASKS, load
from laya import Agent
REV = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
out = {}
for threads in (6, 14):
    torch.set_num_threads(threads)
    agent = Agent("convaiinnovations/laya", device="cpu", revision=REV)
    t = TASKS["stance-abortion"]; q = {"label": {"type": "choice", "instructions": t["instructions"], "criteria": t["criteria"]}}
    rows = load("stance-abortion", "test")[:30]
    for text, _ in rows[:3]: agent.predict(text, q)          # warm-up
    lat, ntok = [], []
    internal = {"label": Agent._to_internal(q["label"])}
    for text, _ in rows:
        ntok.append(len(agent._encode_state(text, ["label"], internal)[0]["ids"]))
        s = time.perf_counter(); agent.predict(text, q); lat.append((time.perf_counter() - s) * 1e3)
    out[threads] = {"p50_ms": round(float(np.median(lat)), 1), "p95_ms": round(float(np.percentile(lat, 95)), 1), "tokens_p50": int(np.median(ntok))}
    print(threads, out[threads], flush=True)
fx = json.load(open(Path(__file__).resolve().parents[2] / "017-kev-rust-forward-parity/fixtures/kev-0.8b_fixture.json"))
kev = [r["torch_cpu_ms"] / len(r["rows"]) for r in fx["records"][:6]]
out["kev-0.8b torch fp32 CPU (spike 017 fixture, same 6 tweets), ms per question"] = round(float(np.median(kev)), 1)
print(json.dumps(out))
json.dump(out, open(Path(__file__).resolve().parents[1] / "results" / "cpu-latency.json", "w"), indent=1)

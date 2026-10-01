"""probes.py <out.json>: edge cases for the Rust port to match."""
import sys, json, numpy as np, pandas as pd, torch, chronos
from chronos import BaseChronosPipeline
pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-tiny", device_map="cpu", torch_dtype=torch.float32); m = pipe.model.eval()
d = "/private/tmp/claude-501/-Users-guy-Development-machine-learning-aprender/9ad16c37-8a13-430e-8c67-e72c6311f455/scratchpad/data/"
y = pd.read_csv(d + "peyton_manning.csv")["y"].values.astype(np.float32)
cases = {}
def run(name, arr, pl=64):
    ctx = torch.tensor(np.asarray(arr, dtype=np.float32))[None, :]
    with torch.no_grad():
        q = pipe.predict(ctx, prediction_length=pl)
        _, ls, emb, mask = m.encode(context=ctx)
    cases[name] = {"y": [None if np.isnan(v) else float(v) for v in np.asarray(arr, dtype=np.float32)], "loc": float(ls[0]), "scale": float(ls[1]), "attention_mask": mask[0].tolist(), "quantiles": q[0].tolist()}
    print(name, "loc", round(float(ls[0]), 4), "scale", round(float(ls[1]), 5), "mask", mask[0].tolist()[:8], "q50[:2]", [round(v, 4) for v in q[0, 4, :2].tolist()])
g = y[-300:].copy(); g[100:140] = np.nan; g[250] = np.nan; run("nan_gaps", g)
run("five_points", y[-5:])
run("constant", np.full(64, 7.0, dtype=np.float32))
run("huge_scale", y[-200:] * 1e6)
run("tiny_len_130_rollout", y[-130:], pl=130)
run("air_rollout_24", pd.read_csv(d + "air_passengers.csv")["y"].values.astype(np.float32), pl=24)
json.dump({"chronos_version": getattr(chronos, "__version__", "?"), "cases": cases}, open(sys.argv[1], "w"))
print("PROBES_DONE version", getattr(chronos, "__version__", "?"))

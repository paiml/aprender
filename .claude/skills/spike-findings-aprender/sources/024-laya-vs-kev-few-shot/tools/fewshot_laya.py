"""Few-shot variants on FROZEN Laya features -- spike 015's recipes, unchanged, on the same rows and seeds.

  zs       released Laya, zero-shot, with its own per-bucket temperature
  bias     K per-class logit offsets on zs, L2 0.01, LBFGS
  head_ft  released `scorer` (LN -> Linear -> GELU -> Linear) fine-tuned on the shots' marker states:
           AdamW lr 1e-4 wd 0.01, 100 full-batch steps, T = 1 (Kev's PointerHead recipe)
  logreg   sklearn LogisticRegression(C=1) on standardised post-head [CLS] (`pooled`)
Recipes were fixed before any Laya test score was read. Stance replays the committed SetFit selections
(benchmarks/tweeteval-stance/selections/sK-seedS) -> paired with SetFit and with every Kev cell of spike 015.

  uv run --with ./vendor/laya --with scikit-learn --with pandas python tools/fewshot_laya.py --run en --task stance-abortion
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
from tasks import TASKS, sample_shots
from fewshot015 import metrics, fit_bias, shot_ids, SEEDS, SHOTS  # 015's exact metric, bias fit, shot manifests

REV = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"


def released_scorer(sub):
    path = hf_hub_download("convaiinnovations/laya", ("" if sub == "en" else f"{sub}/") + "model.safetensors", revision=REV)
    w = {k[len("scorer."):]: v.float() for k, v in load_file(path).items() if k.startswith("scorer.")}
    d = w["1.weight"].shape[1]
    sc = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
    sc.load_state_dict(w)
    return sc


def fit_scorer(sc0, m, y):
    sc = nn.Sequential(nn.LayerNorm(sc0[0].normalized_shape), nn.Linear(sc0[1].in_features, sc0[1].out_features),
                       nn.GELU(), nn.Linear(sc0[3].in_features, 1))
    sc.load_state_dict(sc0.state_dict()); sc.train()
    opt = torch.optim.AdamW(sc.parameters(), lr=1e-4, weight_decay=0.01)
    M, Y = torch.from_numpy(m), torch.from_numpy(y)
    for _ in range(100):
        opt.zero_grad(); loss = F.cross_entropy(sc(M).squeeze(-1), Y); loss.backward(); opt.step()
    sc.eval(); return sc


ap = argparse.ArgumentParser(); ap.add_argument("--run", default="en"); ap.add_argument("--task", required=True)
a = ap.parse_args()
te, tr = (np.load(ROOT / "runs" / f"laya-{a.run}-{a.task}-{s}.npz") for s in ("test", "train"))
K = len(TASKS[a.task]["labels"]); sc0 = released_scorer(a.run); T = float(te["temperature"])
with torch.no_grad():
    z = sc0(torch.from_numpy(te["m_opts"])).squeeze(-1)
assert torch.allclose(z, torch.from_numpy(te["logits"]), atol=2e-3), "released scorer disagrees with captured logits"
P_zs = torch.softmax(z / T, -1).numpy()
rows = [{"variant": "zs", "shots": 0, "seed": None, **metrics(P_zs, te["y"], a.task)}]
sc_std = StandardScaler().fit(tr["pooled"])
for k in SHOTS:
    for seed in SEEDS:
        ids = shot_ids(a.task, k, seed, K, tr["y"]); y = tr["y"][ids]
        b = fit_bias(np.log(np.clip(tr["probs"][ids], 1e-9, 1)), y, K)
        P_b = torch.softmax(torch.from_numpy(np.log(np.clip(te["probs"], 1e-9, 1)) + b), -1).numpy()
        sc = fit_scorer(sc0, tr["m_opts"][ids], y)
        with torch.no_grad():
            P_h = torch.softmax(sc(torch.from_numpy(te["m_opts"])).squeeze(-1), -1).numpy()
        lr = LogisticRegression(C=1.0, max_iter=2000).fit(sc_std.transform(tr["pooled"][ids]), y)
        P_l = lr.predict_proba(sc_std.transform(te["pooled"]))
        for v, P in (("bias", P_b), ("head_ft", P_h), ("logreg", P_l)):
            rows.append({"variant": v, "shots": k, "seed": seed, **metrics(P, te["y"], a.task)})
    print(f"shots {k} done", flush=True)
out = ROOT / "results" / f"fewshot-laya-{a.run}-{a.task}.json"
json.dump(rows, open(out, "w"), indent=1)
import pandas as pd
df = pd.DataFrame(rows); key = "f_avg" if a.task == "stance-abortion" else "macro_f1"
print(df.groupby(["variant", "shots"])[[key, "acc", "ece"]].agg(["mean", "std"]).round(4).to_string())

"""Few-shot variants on FROZEN Kev features, scored on the same test rows as SetFit.

Every recipe below is fixed a priori (written before any test score was seen) -- no knob is chosen by test error.
  zs       released Kev, zero-shot (criteria descriptions = the business's only steering)
  bias     K per-class logit offsets on top of zs, L2 0.01, LBFGS  -> "correct Kev's class prior"
  head_ft  released PointerHead fine-tuned on the shots: AdamW lr 1e-4 wd 0.01, 100 full-batch steps, T=1
  logreg   sklearn LogisticRegression(C=1) on standardised h_decide -> SetFit's head on Kev's features
Shots: stance replays the committed SetFit selections (benchmarks/tweeteval-stance/selections/sK-seedS), so every
cell is PAIRED with a SetFit row; emotion samples k/class from the 3000-row pool with the same 10 seeds.
"""
import argparse, glob, json, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor" / "kev"))
from tasks import TASKS, sample_shots
from kev.model import PointerHead

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT.parents[2] / "benchmarks" / "tweeteval-stance"
SEEDS = [13, 17, 23, 29, 31, 37, 41, 43, 47, 53]
SHOTS = [8, 16, 32, 64]


def metrics(P, y, task):
    pred = P.argmax(1)
    out = {"acc": float((pred == y).mean()), "macro_f1": float(f1_score(y, pred, average="macro"))}
    if task == "stance-abortion":   # official TweetEval stance F_avg = mean F1 of against(1), favor(2)
        out["f_avg"] = float(f1_score(y, pred, labels=[1, 2], average="macro"))
    conf, bins, ece = P.max(1), np.linspace(0, 1, 16), 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any(): ece += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    out["ece"] = float(ece); out["nll"] = float(-np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1)).mean())
    return out


def load_head(run):
    snap = glob.glob(str(Path.home() / f".cache/huggingface/hub/models--jaredpalmer--{run}/snapshots/*/head.pt"))[0]
    ck = torch.load(snap, map_location="cpu", weights_only=False)
    h = PointerHead(ck["head"]["q.weight"].shape[1], dp=ck["head"]["q.weight"].shape[0]); h.load_state_dict(ck["head"])
    return h, float(ck.get("temperature", 1.0))


def head_logits(head, hd, ho):
    return torch.stack([head(torch.from_numpy(a), torch.from_numpy(b)) for a, b in zip(hd, ho)])


def fit_bias(logp, y, K):
    b = torch.zeros(K, requires_grad=True); opt = torch.optim.LBFGS([b], max_iter=200)
    L, Y = torch.from_numpy(logp).float(), torch.from_numpy(y)
    def closure():
        opt.zero_grad(); loss = F.cross_entropy(L + b, Y) + 0.01 * (b ** 2).sum(); loss.backward(); return loss
    opt.step(closure); return b.detach().numpy()


def fit_head(head0, hd, ho, y):
    head = PointerHead(head0.q.in_features, dp=head0.q.out_features); head.load_state_dict(head0.state_dict()); head.train()
    opt = torch.optim.AdamW(head.parameters(), lr=1e-4, weight_decay=0.01); Y = torch.from_numpy(y)
    for _ in range(100):
        opt.zero_grad(); loss = F.cross_entropy(head_logits(head, hd, ho), Y); loss.backward(); opt.step()
    head.eval(); head.temperature = 1.0; return head


def shot_ids(task, k, seed, n_labels, y_train):
    if task == "stance-abortion":
        m = json.load(open(BENCH / "selections" / f"s{k}-seed{seed}" / "selection-manifest.json"))["payload"]
        return [int(e["id"].split(":")[1]) for e in m["ordered_examples"]]
    return sample_shots([(None, int(c)) for c in y_train], n_labels, k, seed)


ap = argparse.ArgumentParser(); ap.add_argument("--run", default="kev-0.8b"); ap.add_argument("--task", required=True)
a = ap.parse_args()
te, tr = (np.load(ROOT / "runs" / f"{a.run}-{a.task}-{s}.npz") for s in ("test", "train"))
K = len(TASKS[a.task]["labels"]); head0, T = load_head(a.run)
with torch.no_grad():
    z_te = head_logits(head0, te["h_decide"], te["h_opts"]).numpy()
P_zs = torch.softmax(torch.from_numpy(z_te) / T, -1).numpy()
assert np.abs(P_zs - te["probs"]).max() < 1e-3, "recomputed head logits disagree with the captured probabilities"
rows = [{"variant": "zs", "shots": 0, "seed": None, **metrics(P_zs, te["y"], a.task)}]
sc = StandardScaler().fit(tr["h_decide"])
for k in SHOTS:
    for seed in SEEDS:
        ids = shot_ids(a.task, k, seed, K, tr["y"]); y = tr["y"][ids]
        b = fit_bias(np.log(tr["probs"][ids]), y, K)
        P_b = torch.softmax(torch.from_numpy(np.log(te["probs"]) + b), -1).numpy()
        head = fit_head(head0, tr["h_decide"][ids], tr["h_opts"][ids], y)
        with torch.no_grad(): P_h = torch.softmax(head_logits(head, te["h_decide"], te["h_opts"]), -1).numpy()
        lr = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(tr["h_decide"][ids]), y)
        P_l = lr.predict_proba(sc.transform(te["h_decide"]))
        for v, P in (("bias", P_b), ("head_ft", P_h), ("logreg", P_l)):
            rows.append({"variant": v, "shots": k, "seed": seed, **metrics(P, te["y"], a.task)})
    print(f"shots {k} done", flush=True)
out = ROOT / "results" / f"fewshot-{a.run}-{a.task}.json"; out.parent.mkdir(exist_ok=True)
json.dump(rows, open(out, "w"), indent=1)
import pandas as pd
df = pd.DataFrame(rows); key = "f_avg" if a.task == "stance-abortion" else "macro_f1"
print(df.groupby(["variant", "shots"])[[key, "acc", "ece"]].agg(["mean", "std"]).round(4).to_string())

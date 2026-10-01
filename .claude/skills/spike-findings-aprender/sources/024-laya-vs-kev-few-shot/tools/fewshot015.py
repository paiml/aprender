"""Spike 015's metric, bias fit and shot manifests, copied VERBATIM so 024's cells are computed exactly like 015's.
(015's fewshot.py runs its script at import time, so it cannot be imported.)"""
import json
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from sklearn.metrics import f1_score
from tasks import sample_shots
BENCH = Path(__file__).resolve().parents[4] / "benchmarks" / "tweeteval-stance"
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


def fit_bias(logp, y, K):
    b = torch.zeros(K, requires_grad=True); opt = torch.optim.LBFGS([b], max_iter=200)
    L, Y = torch.from_numpy(logp).float(), torch.from_numpy(y)
    def closure():
        opt.zero_grad(); loss = F.cross_entropy(L + b, Y) + 0.01 * (b ** 2).sum(); loss.backward(); return loss
    opt.step(closure); return b.detach().numpy()


def shot_ids(task, k, seed, n_labels, y_train):
    if task == "stance-abortion":
        m = json.load(open(BENCH / "selections" / f"s{k}-seed{seed}" / "selection-manifest.json"))["payload"]
        return [int(e["id"].split(":")[1]) for e in m["ordered_examples"]]
    return sample_shots([(None, int(c)) for c in y_train], n_labels, k, seed)

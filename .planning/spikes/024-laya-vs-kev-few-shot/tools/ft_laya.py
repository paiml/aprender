"""Pre-declared full fine-tune of Laya on few shots (README "Pre-declared recipes"): encoder unfrozen.

Optimiser from Laya's own notebook (notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb): AdamW, encoder
lr 2.5e-5, head lr 1e-4, cosine -> 1e-6, clip 1.0, loss = CE + (-proper_reward(softmax(z/T_bucket), onehot)).
Batch 8 on one MPS device. ft-r1 = 4 epochs (notebook), ft-r2 = 12 epochs (small-data variant).
Rows are built by Laya's own Agent._encode_state and scored through Agent.predict, so train rows == inference rows.

  uv run --with ./vendor/laya --with datasets --with scikit-learn --with pandas python tools/ft_laya.py
"""
import json, math, random, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).parent))
from tasks import TASKS, load
from fewshot015 import metrics, shot_ids
from laya import Agent
from laya.common import proper_reward

REV = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
ROOT = Path(__file__).resolve().parents[1]
TASK = sys.argv[1] if len(sys.argv) > 1 else "stance-abortion"; t = TASKS[TASK]; LABELS = t["labels"]; K = len(LABELS)
QDEF = {"type": "choice", "instructions": t["instructions"], "criteria": t["criteria"]}
RECIPES = {"ft-r1": 4, "ft-r2": 12}
LIM = {"emotion": (3000, 1000)}.get(TASK, (None, None))   # spike-015 caps
train_rows, test_rows = load(TASK, "train", LIM[0]), load(TASK, "test", LIM[1])
y_train = np.array([r[1] for r in train_rows]); y_test = np.array([r[1] for r in test_rows])


def collate(items, pad):
    n, L, km = len(items), max(len(i["ids"]) for i in items), max(len(i["markers"]) for i in items)
    ids = torch.full((n, L), pad, dtype=torch.long); att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, km), dtype=torch.long); mm = torch.zeros((n, km), dtype=torch.bool)
    for j, it in enumerate(items):
        ids[j, :len(it["ids"])] = torch.tensor(it["ids"]); att[j, :len(it["ids"])] = 1
        mpos[j, :len(it["markers"])] = torch.tensor(it["markers"]); mm[j, :len(it["markers"])] = True
    return ids, att, mpos, mm, torch.tensor([it["qtype"] for it in items])


def run(recipe, k, seed):
    torch.manual_seed(seed); random.seed(seed)
    agent = Agent("convaiinnovations/laya", device="mps", revision=REV)
    internal = {"label": Agent._to_internal(QDEF)}
    T = agent.temperature_by_options.get("choice:3-5" if K <= 5 else "choice:6-10", float(agent.temperature[0]))
    ids = shot_ids(TASK, k, seed, K, y_train)
    items = [agent._encode_state(train_rows[i][0], ["label"], internal)[0] for i in ids]
    ys = torch.tensor([int(y_train[i]) for i in ids])
    model = agent.model; model.train()
    enc = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    rest = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    opt = torch.optim.AdamW([{"params": enc, "lr": 2.5e-5}, {"params": rest, "lr": 1.0e-4}], weight_decay=0.01)
    epochs = RECIPES[recipe]; steps = epochs * math.ceil(len(items) / 8)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps), eta_min=1e-6)
    pad, dev, t0, losses = agent.tok.pad_token_id, agent.device, time.time(), []
    for ep in range(epochs):
        order = list(range(len(items))); random.shuffle(order)
        for b in range(0, len(order), 8):
            sel = order[b:b + 8]
            ids_t, att, mpos, mm, qt = (x.to(dev) for x in collate([items[i] for i in sel], pad))
            z, act = model(ids_t, att, mpos, mm, qt)
            z = z[:, :K]; yb = ys[sel].to(dev)
            ce = F.cross_entropy(z, yb)
            q = torch.softmax(z / T, -1); onehot = F.one_hot(yb, K).float()
            rl = -proper_reward(q, onehot, qt, mm[:, :K].float(), w_sph=0.75, w_rps=1.0).mean()
            loss = ce + rl + 0.0 * act.sum()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            losses.append(ce.item())
    model.eval(); train_s = time.time() - t0
    P = []
    with torch.no_grad():
        for text, _ in test_rows:
            o = agent.predict(text, {"label": QDEF}); pr = (o["answers"] if "answers" in o else o)["label"]["probabilities"]
            P.append([pr[l] for l in LABELS])
    P = np.array(P, dtype=np.float64)
    return {"variant": recipe, "shots": k, "seed": seed, "epochs": epochs, "steps": steps, "train_s": round(train_s, 1),
            "ce_first": round(losses[0], 4), "ce_last": round(losses[-1], 4), **metrics(P, y_test, TASK)}


out = ROOT / "results" / f"ft-laya-en-{TASK}.jsonl"
for recipe in RECIPES:
    for k in ((16, 64) if TASK == "stance-abortion" else (64,)):
        for seed in (13, 17, 23):
            r = run(recipe, k, seed)
            with open(out, "a") as f: f.write(json.dumps(r) + "\n")
            print(json.dumps(r), flush=True)

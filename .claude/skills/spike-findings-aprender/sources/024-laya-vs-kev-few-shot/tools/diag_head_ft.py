"""Why does head_ft not move? Train-side diagnostics only (no test rows are read)."""
import copy, sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).parent))
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from fewshot015 import shot_ids
REV = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
run, task, K = sys.argv[1], sys.argv[2], int(sys.argv[3])
path = hf_hub_download("convaiinnovations/laya", ("" if run == "en" else f"{run}/") + "model.safetensors", revision=REV)
w = {k[len("scorer."):]: v.float() for k, v in load_file(path).items() if k.startswith("scorer.")}
d = w["1.weight"].shape[1]; sc0 = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1)); sc0.load_state_dict(w)
tr = np.load(f"runs/laya-{run}-{task}-train.npz")
for k in (8, 64):
    ids = shot_ids(task, k, 13, K, tr["y"]); m = torch.from_numpy(tr["m_opts"][ids]); y = torch.from_numpy(tr["y"][ids])
    with torch.no_grad(): z0 = sc0(m).squeeze(-1)
    print(f"k={k}: released |z| {z0.abs().mean():.2f}, option spread {z0.std(-1).mean():.2f}, train acc {(z0.argmax(-1)==y).float().mean():.3f}, preds {np.bincount(z0.argmax(-1).numpy(), minlength=K)}")
    for lr in (1e-4, 1e-3, 1e-2):
        s = copy.deepcopy(sc0); s.train(); opt = torch.optim.AdamW(s.parameters(), lr=lr, weight_decay=0.01)
        losses = []
        for _ in range(100):
            opt.zero_grad(); loss = F.cross_entropy(s(m).squeeze(-1), y); loss.backward(); opt.step(); losses.append(loss.item())
        with torch.no_grad(): acc = (s(m).squeeze(-1).argmax(-1) == y).float().mean().item()
        print(f"   lr {lr:g}: train loss {losses[0]:.3f} -> {losses[-1]:.3f}, train acc {acc:.3f}")

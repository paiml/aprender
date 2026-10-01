"""Where does the final-norm fp32 error live? torch fp32 vs float64 on the gate's eval rows, per token.

    uv run --frozen --project scripts/laya_train python .planning/spikes/028-laya-packability-noise-floor/final_norm_locate.py CKPT|base DATA_DIR OUT_JSON

For every row: the (token, dim) of the max |final32 - final64|, the value there, that token's rms, whether the
token is a [MASK] marker, and the error at marker tokens only (the scorer reads markers; the head attends all).
"""
import json
import sys
import types
import warnings
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "laya_train"))
import train  # noqa: E402
from laya import Agent  # noqa: E402
from common import tree_sha256  # noqa: E402

ck, dd, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
task_raw = json.loads((dd / "task.json").read_text())
question = {"type": task_raw["type"], "instructions": task_raw["instructions"], "criteria": dict(task_raw["criteria"])}
texts = [json.loads(l)["text"] for l in (dd / "eval.jsonl").read_text().splitlines() if l.strip()]
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    if ck == "base":
        b = train.Base(types.SimpleNamespace(variant="production", base=None, base_sha256=None))
        agent = train.load_for_scoring(b.src, b.digest, b.revision)
    else:
        agent = train.load_for_scoring(Path(ck), tree_sha256(Path(ck))["model.safetensors"])
m = agent.model
enc = [agent._encode_state(t, ["q"], {"q": Agent._to_internal(question)})[0] for t in texts]


def fin(mm, ids):
    with torch.no_grad():
        return mm.encoder(input_ids=torch.tensor([ids]), attention_mask=torch.ones(1, len(ids), dtype=torch.long)).last_hidden_state[0].numpy()


F32 = [fin(m, e["ids"]).astype(np.float64) for e in enc]
m64 = m.to(torch.float64)
rows, dims = [], {}
for i, e in enumerate(enc):
    f64 = fin(m64, e["ids"])
    d = np.abs(F32[i] - f64)
    tok, dim = np.unravel_index(d.argmax(), d.shape)
    rms = np.sqrt((f64 ** 2).mean(1))
    mk = list(e["markers"])
    rows.append({"row": i, "max": float(d.max()), "tok": int(tok), "dim": int(dim), "value": float(f64[tok, dim]),
                 "tok_rms": float(rms[tok]), "tok_is_marker": int(tok) in mk, "tok_id": int(e["ids"][tok]),
                 "marker_max": float(d[mk].max()), "rel_rms_row": float(np.sqrt((d ** 2).mean()) / np.sqrt((f64 ** 2).mean()))})
    dims[int(dim)] = dims.get(int(dim), 0) + 1
mx = np.array([r["max"] for r in rows])
mm_ = np.array([r["marker_max"] for r in rows])
res = {"ckpt": ck, "n": len(rows), "max": float(mx.max()), "over_1e-4": int((mx > 1e-4).sum()),
       "marker_max": float(mm_.max()), "marker_over_1e-4": int((mm_ > 1e-4).sum()),
       "argmax_tok_is_marker": sum(r["tok_is_marker"] for r in rows), "argmax_tok_pos0": sum(r["tok"] == 0 for r in rows),
       "argmax_dims_top": sorted(dims.items(), key=lambda kv: -kv[1])[:5],
       "rel_rms_row_max": float(max(r["rel_rms_row"] for r in rows)),
       "worst": sorted(rows, key=lambda r: -r["max"])[:5]}
out.write_text(json.dumps(res, indent=1) + "\n")
print(json.dumps({k: v for k, v in res.items() if k != "worst"}))
for r in res["worst"]:
    print(r)

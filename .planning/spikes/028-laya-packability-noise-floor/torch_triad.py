"""Three-way re-score tails for one checkpoint on one row set: Rust fp32, torch fp32, float64.

    uv run --frozen --project scripts/laya_train python .planning/spikes/028-laya-packability-noise-floor/torch_triad.py \
        --ckpt models/decide/spike-027/<run>/checkpoint | --ckpt base \
        --data DATA_DIR --rust RUST_DUMP_DIR --out results/triad/<name>.json \
        [--stored <run>/eval-probs.json] [--extra-t 8.758205757575377]

torch fp32 = train.py's own Scorer over train.load_for_scoring (CPU fp32 over the F16 reload), exactly how
eval-probs.json / zero-shot-probs.json were written; --stored checks the rerun is bitwise the stored file.
float64 = the same model cast to float64, forwarded from torch's ids/markers (the laya-rescore-drift
torch_f64.py forward). Caveat carried from that session: the f64 model keeps HF's RoPE inv_freq buffer,
which transformers computes in f32, so "f64" is exact arithmetic over torch's f32 RoPE tables.
Rust = zz_triad_dump (the Rust builder's own ids; ids are compared to torch's).

Per row, for each pair (rust-torch32, rust-f64, torch32-f64): max_k |dp| (bar 1e-5), max_k |dz| (bar 1e-4)
and max |d final_norm| over the row (laya-parity-v1 final_norm_abs, bar 1e-4). Summaries: max (row), p99,
median, rows over the bar, argmax agreement.
"""
import argparse
import hashlib
import json
import sys
import time
import types
import warnings
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "laya_train"))
import data  # noqa: E402
import train  # noqa: E402
from laya import Agent  # noqa: E402

BARS = {"probs": 1e-5, "logits": 1e-4, "final": 1e-4}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def summ(d, bar):
    d = np.asarray(d, dtype=np.float64)
    return {"max": float(d.max()), "row": int(d.argmax()), "p99": float(np.percentile(d, 99)),
            "median": float(np.median(d)), "mean": float(d.mean()), "over_bar": int((d > bar).sum()),
            "n": int(d.size), "bar": bar, "headroom": float(bar / d.max()) if d.max() > 0 else float("inf")}


def sm64(z, t):
    z = np.asarray(z, dtype=np.float64) / t
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def sm32(z, t):
    return torch.softmax(torch.from_numpy(np.asarray(z, dtype=np.float32)) / t, dim=-1).numpy()


ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--rust", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--stored", default="")
ap.add_argument("--extra-t", default="")
a = ap.parse_args()
extra = [float(x) for x in a.extra_t.split(",") if x]

task_raw = json.loads((Path(a.data) / "task.json").read_text())
question = {"type": task_raw["type"], "instructions": task_raw["instructions"], "criteria": dict(task_raw["criteria"])}
texts = [json.loads(l)["text"] for l in (Path(a.data) / "eval.jsonl").read_text().splitlines() if l.strip()]

with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    if a.ckpt == "base":
        b = train.Base(types.SimpleNamespace(variant="production", base=None, base_sha256=None))
        agent = train.load_for_scoring(b.src, b.digest, b.revision)
    else:
        ck = Path(a.ckpt)
        agent = train.load_for_scoring(ck, sha(ck / "model.safetensors"))
m = agent.model
print("threads", torch.get_num_threads(), "torch", torch.__version__, "rows", len(texts), file=sys.stderr, flush=True)

cap = {}
hook = m.encoder.final_norm.register_forward_hook(lambda mod, i, o: cap.__setitem__("final", o.detach().clone()))
sc = train.Scorer(agent)
t0 = time.time()
P32, Z32, IDS, MK, F32 = [], [], [], [], []
for text in texts:
    it = agent._encode_state(text, ["q"], {"q": Agent._to_internal(question)})[0]
    p, z, n = sc.score(text, question)
    f = cap["final"][0].numpy().astype(np.float32)
    if f.shape[0] != len(it["ids"]):
        raise RuntimeError("final-norm rows %d != ids %d" % (f.shape[0], len(it["ids"])))
    P32.append(np.asarray(p, dtype=np.float32))
    Z32.append(np.asarray(z, dtype=np.float32))
    IDS.append(list(it["ids"]))
    MK.append(list(it["markers"]))
    F32.append(f)
hook.remove()
t_fp32 = time.time() - t0
_q = train.QTYPES["choice"]
bucket_t = float(agent.temperature_by_options.get(train.temp_bucket(_q, len(MK[0])), agent.temperature[_q]))
P32 = np.stack(P32)
Z32 = np.stack(Z32)


def run(mm, ids, markers):
    b_ids = torch.tensor([ids])
    with torch.no_grad():
        h = mm.encoder(input_ids=b_ids, attention_mask=torch.ones(1, len(ids), dtype=torch.long)).last_hidden_state
        fin = h[0].clone()
        h = h + mm.type_emb(torch.tensor([0]))[:, None, :]
        pad = torch.zeros(1, len(ids), dtype=torch.bool)
        for layer in mm.head.layers:
            h = layer(h, src_key_padding_mask=pad)
        idx = torch.tensor([markers])[:, :, None].expand(-1, -1, h.size(-1))
        z = mm.scorer(torch.gather(h, 1, idx)).squeeze(-1)[0]
    return z.numpy(), fin.numpy()


# Control: the manual forward in fp32 must reproduce the Scorer's logits (else the f64 path is another model).
ctrl = max(float(np.abs(run(m, IDS[i], MK[i])[0] - Z32[i]).max()) for i in range(min(5, len(texts))))
m64 = m.to(torch.float64).eval()
t0 = time.time()
Z64, F64 = [], []
for i in range(len(texts)):
    z, f = run(m64, IDS[i], MK[i])
    Z64.append(z)
    F64.append(f)
t_f64 = time.time() - t0
Z64 = np.stack(Z64)

rj = json.loads((Path(a.rust) / "all.json").read_text())
rrows = rj["rows"]
T = float(rj["t"])
RP = np.array([r["p"] for r in rrows])
RZ = np.array([r["z"] for r in rrows])
ids_equal = sum(1 for i, r in enumerate(rrows) if r["ids"] == IDS[i] and r["markers"] == MK[i])
raw = np.fromfile(Path(a.rust) / "final.f32", dtype="<f4")
RF, off = [], 0
for r in rrows:
    RF.append(raw[off:off + r["final_len"]].reshape(r["len"], -1))
    off += r["final_len"]
assert off == raw.size

res = {"ckpt": a.ckpt, "data": a.data, "n": len(texts), "T_applied_rust": T, "T_bucket_torch": bucket_t,
       "ids_equal": ids_equal, "manual_fp32_forward_vs_scorer_logits_max": ctrl,
       "wall_s": {"torch_fp32": t_fp32, "f64": t_f64}, "torch": torch.__version__,
       "threads": torch.get_num_threads(), "rust_arch": rj["arch"]}
if a.stored:
    st = json.loads(Path(a.stored).read_text())
    SP = np.array([r["probabilities"] for r in sorted(st["rows"], key=lambda r: r["row"])], dtype=np.float32)
    res["torch_rerun_bitwise_equal_stored_rows"] = int((SP == P32).all(1).sum())
    res["rust_vs_stored_probs_max"] = float(np.abs(RP - SP.astype(np.float64)).max())
res["torch_softmax_emulation_bitwise_rows"] = int((sm32(Z32, T) == P32).all(1).sum())


def triple(A, B, C, bar, per_row):
    return {"rust_vs_torch32": summ(per_row(A, B), bar), "rust_vs_f64": summ(per_row(A, C), bar),
            "torch32_vs_f64": summ(per_row(B, C), bar)}


rowmax = lambda X, Y: np.abs(np.asarray(X, np.float64) - np.asarray(Y, np.float64)).max(1)  # noqa: E731
F64P = sm64(Z64, T)
res["probs"] = triple(RP, P32, F64P, BARS["probs"], rowmax)
res["logits"] = triple(RZ, Z32, Z64, BARS["logits"], rowmax)
fmax = lambda X, Y: np.array([float(np.abs(x.astype(np.float64) - y.astype(np.float64)).max()) for x, y in zip(X, Y)])  # noqa: E731
res["final"] = triple(RF, F32, F64, BARS["final"], fmax)
res["final_abs_max"] = float(max(np.abs(f).max() for f in F64))
res["logit_abs_max"] = float(np.abs(Z64).max())
res["argmax_agree"] = {"rust_vs_torch32": int((RP.argmax(1) == P32.argmax(1)).sum()),
                       "rust_vs_f64": int((RP.argmax(1) == F64P.argmax(1)).sum()),
                       "torch32_vs_f64": int((P32.argmax(1) == F64P.argmax(1)).sum())}
res["alt_t"] = []
for j, te in enumerate(extra):
    rp = np.array([r["p_alt"][j] for r in rrows])
    tp = sm32(Z32, te)
    res["alt_t"].append({"t": te, "probs": triple(rp, tp, sm64(Z64, te), BARS["probs"], rowmax)})
# Per-row vectors for the bar analysis (k x torch32_vs_f64 per checkpoint).
res["per_row"] = {"dp_rust_torch": rowmax(RP, P32).tolist(), "dp_torch_f64": rowmax(P32, F64P).tolist(),
                  "dp_rust_f64": rowmax(RP, F64P).tolist(), "dz_rust_torch": rowmax(RZ, Z32).tolist(),
                  "dz_torch_f64": rowmax(Z32, Z64).tolist(), "dz_rust_f64": rowmax(RZ, Z64).tolist()}
res["probs_full"] = {"rust": RP.tolist(), "torch32": P32.astype(np.float64).tolist(), "f64": F64P.tolist()}
res["logits_full"] = {"rust": RZ.tolist(), "torch32": Z32.astype(np.float64).tolist(), "f64": Z64.tolist()}
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
for k in ("probs", "logits", "final"):
    for pair, s in res[k].items():
        print("%-6s %-16s max %.3e (row %3d) p99 %.2e median %.2e over %.0e: %3d" %
              (k, pair, s["max"], s["row"], s["p99"], s["median"], s["bar"], s["over_bar"]))
print("ids_equal %d/%d ctrl %.2e stored-bitwise %s wall %s" % (ids_equal, len(texts), ctrl,
      res.get("torch_rerun_bitwise_equal_stored_rows"), res["wall_s"]))

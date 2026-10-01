#!/usr/bin/env python3
"""Chronos-2 oracle: bottom-up ladder tensors + quantiles from chronos-forecasting, for the Rust port.
Run from the spike dir:
  uv run --python 3.12 --with chronos-forecasting --with pandas --with safetensors tools/oracle.py
"""
import json, math, os, time
import numpy as np, pandas as pd, torch
from chronos import Chronos2Pipeline

torch.manual_seed(0)
pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu", torch_dtype=torch.float32)
model = pipe.model.eval()
cfg = model.config

def load_csv(path):
    df = pd.read_csv(path)
    df["ds"] = df["ds"].astype(str).str[:10]
    df = df.sort_values("ds", kind="mergesort").drop_duplicates("ds", keep="last")
    return df["y"].to_numpy(dtype=np.float32)

peyton = load_csv("fixtures/peyton_manning.csv")
air = load_csv("fixtures/air_passengers.csv")
gaps = peyton[-300:].copy(); gaps[::7] = np.nan
lead = np.concatenate([np.full(16, np.nan, dtype=np.float32), peyton[-100:]])
series = {
    "peyton": (peyton, [64, 365, 1024]),
    "air": (air, [24]),
    "short100": (peyton[-100:], [64]),
    "nan_gaps": (gaps, [16]),
    "leading_nan_patch": (lead, [16]),
    "constant": (np.full(100, 3.0, dtype=np.float32), [16]),
    "huge_scale": ((peyton[-256:] * 1e6).astype(np.float32), [40]),
    "negative": ((-peyton[-256:]).astype(np.float32), [40]),
    "short5": (peyton[-5:], [16]),
}

def ladder(y, h):
    ctx = torch.tensor(y, dtype=torch.float32)[None]
    nop = min(math.ceil(h / 16), model.chronos_config.max_output_patches)
    with torch.no_grad():
        patched, attn_mask, (loc, scale) = model._prepare_patched_context(ctx)
        emb = model.input_patch_embedding(patched)
        enc_out, _, _, n_ctx = model.encode(ctx, num_output_patches=nop)
        hidden = enc_out.last_hidden_state[0]
        t0 = time.time(); q = model(context=ctx, num_output_patches=nop).quantile_preds[0]; dt = time.time() - t0
        t0 = time.time(); pq = pipe.predict([torch.tensor(y, dtype=torch.float32)], prediction_length=h)[0]; dt2 = time.time() - t0
    return {
        "n": int(len(y)), "h": h, "num_output_patches": nop, "loc": float(loc), "scale": float(scale),
        "n_patches": int(patched.shape[1]), "attention_mask": attn_mask[0].int().tolist(),
        "patch_feat_first": patched[0, 0].tolist(), "patch_feat_last": patched[0, -1].tolist(),
        "embed_first": emb[0, 0].tolist(), "embed_last": emb[0, -1].tolist(),
        "hidden_first": hidden[0].tolist(), "hidden_reg": hidden[n_ctx].tolist(), "hidden_last": hidden[-1].tolist(),
        "quantiles": q.tolist(), "pipeline_quantiles": pq[0].tolist(),
        "model_seconds": dt, "pipeline_seconds": dt2,
    }

out = {"config": json.loads(cfg.to_json_string()), "reg_embed": model.shared.weight[1].tolist(), "torch_threads": torch.get_num_threads(), "series": {}}
for name, (y, hs) in series.items():
    out["series"][name] = {}
    for h in hs:
        r = ladder(y, h)
        out["series"][name][str(h)] = r
        print(f"{name} h={h}: n={r['n']} patches={r['n_patches']} nop={r['num_output_patches']} loc={r['loc']:.4f} scale={r['scale']:.4f} model {r['model_seconds']*1e3:.0f} ms pipeline {r['pipeline_seconds']*1e3:.0f} ms", flush=True)
json.dump(out, open("fixtures/chronos2_fixture.json", "w"))
print("wrote fixtures/chronos2_fixture.json", os.path.getsize("fixtures/chronos2_fixture.json"))

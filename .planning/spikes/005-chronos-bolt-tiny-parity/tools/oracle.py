"""oracle.py <out_dir>: download chronos-bolt-tiny, dump weight names, and fixtures with intermediate tensors."""
import sys, json, time, os, numpy as np, pandas as pd, torch
from chronos import BaseChronosPipeline
from safetensors import safe_open
from huggingface_hub import snapshot_download
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
local = snapshot_download("amazon/chronos-bolt-tiny", allow_patterns=["*.json", "*.safetensors"])
print("MODEL_DIR", local)
keys = {}
with safe_open(os.path.join(local, "model.safetensors"), "pt") as f:
    for k in f.keys(): t = f.get_tensor(k); keys[k] = {"shape": list(t.shape), "dtype": str(t.dtype)}
json.dump(keys, open(os.path.join(out, "weights_index.json"), "w"), indent=1)
print("N_TENSORS", len(keys), "total_params", sum(int(np.prod(v["shape"])) for v in keys.values()))
pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-tiny", device_map="cpu", torch_dtype=torch.float32)
model = pipe.model.eval()
cfg = json.load(open(os.path.join(local, "config.json")))
def series(name):
    d = "/private/tmp/claude-501/-Users-guy-Development-machine-learning-aprender/9ad16c37-8a13-430e-8c67-e72c6311f455/scratchpad/data/"
    if name == "peyton": df = pd.read_csv(d + "peyton_manning.csv"); return df["y"].values.astype(np.float32)
    if name == "air": df = pd.read_csv(d + "air_passengers.csv"); return df["y"].values.astype(np.float32)
    if name == "short100": df = pd.read_csv(d + "peyton_manning.csv"); return df["y"].values.astype(np.float32)[-100:]
fx = {"config": cfg, "series": {}}
for name in ["peyton", "air", "short100"]:
    y = series(name); ctx = torch.tensor(y)[None, :]
    with torch.no_grad():
        # intermediate ladder
        hidden, loc_scale, input_embeds, attn_mask = model.encode(context=ctx)
        dec = model.decode(input_embeds, attn_mask, hidden)
        t0 = time.perf_counter(); q64 = model(context=ctx).quantile_preds; t64 = time.perf_counter() - t0
        t0 = time.perf_counter(); q365 = pipe.predict(ctx, prediction_length=365) if name == "peyton" else None; t365 = time.perf_counter() - t0
        pq, mean = pipe.predict_quantiles(ctx, prediction_length=64, quantile_levels=[0.1, 0.5, 0.9])
    fx["series"][name] = {
        "n": int(len(y)), "y_tail16": y[-16:].tolist(),
        "loc": float(loc_scale[0].item()), "scale": float(loc_scale[1].item()),
        "n_patches_plus_reg": int(input_embeds.shape[1]), "attention_mask": attn_mask[0].tolist(),
        "input_embeds_first_patch": input_embeds[0, 0].tolist(), "input_embeds_last_patch": input_embeds[0, -2].tolist(), "reg_embed": input_embeds[0, -1].tolist(),
        "encoder_last_hidden_first": hidden[0, 0].tolist(), "encoder_last_hidden_reg": hidden[0, -1].tolist(),
        "decoder_last_hidden": dec[0, 0].tolist(),
        "quantiles_64": q64[0].tolist(), "seconds_64": t64,
        "quantiles_365": (q365[0].tolist() if q365 is not None else None), "seconds_365": t365,
        "predict_quantiles_q10_q50_q90_first5": pq[0, :5].tolist(), "mean_first5": mean[0, :5].tolist(),
    }
    print(name, "n", len(y), "patches+reg", input_embeds.shape[1], "loc", round(float(loc_scale[0]), 4), "scale", round(float(loc_scale[1]), 4), "t64", round(t64, 4), "t365", round(t365, 3), "q50[:3]", [round(v, 4) for v in q64[0, 4, :3].tolist()])
json.dump(fx, open(os.path.join(out, "chronos_bolt_tiny_fixture.json"), "w"))
print("FIXTURE_DONE")

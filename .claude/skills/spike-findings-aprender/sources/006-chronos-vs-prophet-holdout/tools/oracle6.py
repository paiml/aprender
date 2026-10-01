"""oracle6.py <out_dir>: bolt-small weights + Python Chronos (tiny, small) holdout forecasts as a metric cross-check."""
import sys, json, os, time, numpy as np, pandas as pd, torch
from chronos import BaseChronosPipeline
from huggingface_hub import snapshot_download
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
d = "/private/tmp/claude-501/-Users-guy-Development-machine-learning-aprender/9ad16c37-8a13-430e-8c67-e72c6311f455/scratchpad/data/"
def load(f):
    df = pd.read_csv(d + f); df["ds"] = pd.to_datetime(df["ds"]); df = df.sort_values("ds", kind="mergesort").drop_duplicates("ds", keep="last")  # same as the Rust loader
    return df["y"].values
series = {"peyton": load("peyton_manning.csv"), "air": load("air_passengers.csv"), "retail": load("retail_sales.csv"), "wp_log_r": load("wp_log_R.csv")}
# rolling origins: (name, [(n_train, horizon), ...])
splits = {"peyton": [(2540, 365), (2200, 90), (2400, 90), (2600, 90), (2815, 90)], "air": [(120, 24), (96, 12), (108, 12), (132, 12)], "retail": [(269, 24), (221, 12), (245, 12), (281, 12)], "wp_log_r": [(2498, 365), (2100, 90), (2400, 90), (2773, 90)]}
res = {"models": {}}
for model_id in ["amazon/chronos-bolt-tiny", "amazon/chronos-bolt-small"]:
    local = snapshot_download(model_id, allow_patterns=["*.json", "*.safetensors"]); print("MODEL_DIR", model_id, local)
    pipe = BaseChronosPipeline.from_pretrained(model_id, device_map="cpu", torch_dtype=torch.float32); m = pipe.model.eval()
    entry = {"dir": local, "n_params": sum(p.numel() for p in m.parameters()), "forecasts": {}}
    for name, y in series.items():
        for (n_train, h) in splits[name]:
            ctx = torch.tensor(y[:n_train].astype(np.float32))[None, :]
            t0 = time.perf_counter()
            with torch.no_grad(): q = pipe.predict(ctx, prediction_length=h)[0].numpy()
            secs = time.perf_counter() - t0
            test = y[n_train:n_train + h]
            mae = float(np.mean(np.abs(q[4] - test))); cov = float(np.mean((test >= q[0]) & (test <= q[8])))
            entry["forecasts"][f"{name}:{n_train}:{h}"] = {"mae_median": mae, "coverage_q10_q90": cov, "secs": secs, "quantiles": q.tolist()}
            print(model_id.split('/')[-1], name, n_train, h, "MAE", round(mae, 4), "cov", round(cov, 3), "s", round(secs, 3))
    res["models"][model_id] = entry
    if model_id.endswith("small"):
        with torch.no_grad(): q64 = m(context=torch.tensor(series["peyton"].astype(np.float32))[None, :]).quantile_preds[0].numpy()
        entry["peyton_full_q64"] = q64.tolist()
json.dump(res, open(os.path.join(out, "chronos_holdout_oracle.json"), "w")); print("ORACLE6_DONE")

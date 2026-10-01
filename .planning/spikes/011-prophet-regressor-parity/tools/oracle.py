"""Prophet 1.4.0 oracle for external regressors (spike 011).

Dumps the parity ladder for `add_regressor`: column order, standardisation constants,
s_a/s_m, the design rows, the MAP params and the forecast components — including each
regressor by name plus extra_regressors_additive / extra_regressors_multiplicative.

Run:  uv run --python 3.12 --with "prophet==1.4.0" --with pandas python tools/oracle.py
"""
import json, math, sys
import numpy as np
import pandas as pd
from prophet import Prophet

CSV = "../../../crates/aprender-forecast/tests/fixtures/retail_sales.csv"
OUT = "fixtures/retail_regressors_prophet140.json"
HORIZON = 12

# CONVENTIONS: sort by ds and de-duplicate at load, in BOTH oracle and Rust driver.
df = pd.read_csv(CSV)
df["ds"] = pd.to_datetime(df["ds"])
df = df.sort_values("ds").drop_duplicates(subset="ds", keep="first").reset_index(drop=True)
n = len(df)

# Future frame first, so regressors are generated over history+future in ONE index space.
m_tmp = Prophet()
freq = "MS"
future_index = pd.date_range(df["ds"].iloc[-1], periods=HORIZON + 1, freq=freq)[1:]
all_ds = list(df["ds"]) + list(future_index)
N = len(all_ds)

# Deterministic regressors over the whole index. The VALUES are dumped so the Rust side
# reads them rather than recomputing — one less class of mismatch.
promo    = [1.0 if i % 5 == 0 else 0.0 for i in range(N)]            # binary  -> auto = NOT standardised
price    = [100.0 + 0.05 * i + 7.0 * math.sin(2 * math.pi * i / 12) for i in range(N)]  # continuous -> standardised
discount = [0.10 + 0.05 * math.cos(2 * math.pi * i / 6) for i in range(N)]              # multiplicative
# An IDENTIFIABLE control: deterministic LCG noise, no trend and no seasonal structure, so
# it is not absorbed by the trend or the Fourier basis. `price` and `discount` turned out to
# be collinear with them (r = 0.76 and 0.999), which makes their betas unidentifiable.
def _lcg(i, s=0):
    x = (i * 1103515245 + 12345 + s) % 2147483648
    x = (x * 1103515245 + 12345) % 2147483648
    return x / 2147483648.0
weather = [20.0 * _lcg(i) - 10.0 for i in range(N)]

REGS = [
    ("promo",    "additive",       10.0, "auto"),
    ("price",    "additive",       10.0, "auto"),
    ("discount", "multiplicative",  5.0, "auto"),
    ("weather",  "additive",       10.0, "auto"),
]
VALS = {"promo": promo, "price": price, "discount": discount, "weather": weather}

hist = df.copy()
for name, vals in VALS.items():
    hist[name] = vals[:n]

m = Prophet(seasonality_mode="additive")
for name, mode, ps, std in REGS:
    m.add_regressor(name, mode=mode, prior_scale=ps, standardize=std)
m.fit(hist)

future = pd.DataFrame({"ds": all_ds})
for name, vals in VALS.items():
    future[name] = vals
fc = m.predict(future)

# ---- ladder rungs -------------------------------------------------------------
# Prophet 1.4 returns (seasonal_features, prior_scales, component_cols, modes).
# s_a / s_m are DERIVED in fit() from component_cols, not returned here.
feats, prior_scales, component_cols, modes = m.make_all_seasonality_features(m.history)
cols = list(feats.columns)
s_a = component_cols["additive_terms"].values
s_m = component_cols["multiplicative_terms"].values

extra = {}
for name, props in m.extra_regressors.items():
    extra[name] = {
        "mu": float(props["mu"]), "std": float(props["std"]),
        "mode": props["mode"], "prior_scale": float(props["prior_scale"]),
        "standardize": props["standardize"],
    }

comp_names = [c for c in fc.columns if not c.endswith("_lower") and not c.endswith("_upper")
              and c not in ("ds", "yhat", "trend", "cap", "floor")]

out = {
    "dataset": "retail_sales",
    "prophet_version": "1.4.0",
    "mode": "regressors",
    "n_history": int(n),
    "horizon": HORIZON,
    "freq": freq,
    "y_scale": float(m.y_scale),
    "start": str(m.start),
    "t_scale_days": float(m.t_scale / pd.Timedelta(days=1)),
    "changepoints_t": [float(v) for v in m.changepoints_t],
    "changepoint_prior_scale": float(m.changepoint_prior_scale),
    "seasonalities": {k: {"period": float(v["period"]), "fourier_order": int(v["fourier_order"]),
                          "prior_scale": float(v["prior_scale"]), "mode": v["mode"]}
                      for k, v in m.seasonalities.items()},
    "regressors_requested": [{"name": a, "mode": b, "prior_scale": c, "standardize": d} for a, b, c, d in REGS],
    "extra_regressors": extra,
    "regressor_values": {k: [float(x) for x in v] for k, v in VALS.items()},
    "columns": cols,
    "prior_scales": [float(v) for v in prior_scales],
    "s_a": [float(v) for v in s_a],
    "s_m": [float(v) for v in s_m],
    "component_cols": {c: [int(v) for v in component_cols[c].values] for c in component_cols.columns},
    "modes": {k: list(v) for k, v in modes.items()},
    "X_first3": feats.values[:3].tolist(),
    "X_last3": feats.values[-3:].tolist(),
    "params": {k: np.asarray(v).ravel().tolist() for k, v in m.params.items()},
    "history": {
        "ds": [str(d.date()) for d in m.history["ds"]],
        "t": [float(v) for v in m.history["t"]],
        "y": [float(v) for v in m.history["y"]],
        "y_scaled": [float(v) for v in m.history["y_scaled"]],
    },
    "forecast": {
        "ds": [str(pd.Timestamp(d).date()) for d in fc["ds"]],
        "yhat": [float(v) for v in fc["yhat"]],
        "trend": [float(v) for v in fc["trend"]],
        "yhat_lower": [float(v) for v in fc["yhat_lower"]],
        "yhat_upper": [float(v) for v in fc["yhat_upper"]],
        "components": {c: [float(v) for v in fc[c]] for c in comp_names},
    },
}
json.dump(out, open(OUT, "w"))
print("columns:", cols)
print("extra_regressors:", json.dumps(extra, indent=1))
print("component names:", comp_names)
print("wrote", OUT)

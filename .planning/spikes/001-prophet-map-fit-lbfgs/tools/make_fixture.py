import json, time, sys, numpy as np, pandas as pd
from prophet import Prophet
df = pd.read_csv(sys.argv[1])
m = Prophet()
t0 = time.perf_counter(); m.fit(df); fit_s = time.perf_counter() - t0
future = m.make_future_dataframe(periods=365)
t1 = time.perf_counter(); fc = m.predict(future); pred_s = time.perf_counter() - t1
hist = m.history
seasonal_features, prior_scales, component_cols, modes = m.make_all_seasonality_features(hist)
X = seasonal_features.values
p = m.params
k = float(np.ravel(p["k"])[0]); mm = float(np.ravel(p["m"])[0]); delta = np.ravel(p["delta"]).tolist(); beta = np.ravel(p["beta"]).tolist(); sigma = float(np.ravel(p["sigma_obs"])[0])
t = hist['t'].values; y = hist['y_scaled'].values; tc = m.changepoints_t
A = (t[:, None] >= tc[None, :]).astype(float)
trend = (k + A @ np.array(delta)) * t + (mm + A @ (-tc * np.array(delta)))
mu = trend + X @ np.array(beta)
tau = m.changepoint_prior_scale
lp = (-k**2/(2*25) - mm**2/(2*25) - np.sum(np.abs(delta))/tau - sigma**2/(2*0.25)
      - np.sum(np.array(beta)**2/(2*np.array(prior_scales)**2)) - len(t)*np.log(sigma) - np.sum((y-mu)**2)/(2*sigma**2))
comps = {n: fc[n].tolist() for n in m.seasonalities}
out = {
  "dataset": sys.argv[3], "prophet_version": __import__('prophet').__version__,
  "fit_seconds": fit_s, "predict_seconds": pred_s,
  "n_history": int(len(hist)), "y_scale": float(m.y_scale), "start": str(m.start.date()),
  "t_scale_days": float(m.t_scale / pd.Timedelta(days=1)),
  "changepoints_t": tc.tolist(), "changepoint_prior_scale": tau,
  "seasonality_columns": list(seasonal_features.columns), "prior_scales": [float(s) for s in prior_scales],
  "seasonalities": [{"name": n, "period": v["period"], "fourier_order": v["fourier_order"], "prior_scale": v["prior_scale"], "mode": v["mode"]} for n, v in m.seasonalities.items()],
  "params": {"k": k, "m": mm, "delta": delta, "beta": beta, "sigma_obs": sigma},
  "log_posterior_at_map_unnormalized": float(lp),
  "history": {"ds": hist['ds'].dt.strftime('%Y-%m-%d').tolist(), "t": t.tolist(), "y": hist['y'].tolist(), "y_scaled": y.tolist()},
  "X_first3": X[:3].tolist(), "X_last3": X[-3:].tolist(),
  "forecast": {"ds": fc['ds'].dt.strftime('%Y-%m-%d').tolist(), "trend": fc['trend'].tolist(), "yhat": fc['yhat'].tolist(),
               "yhat_lower": fc['yhat_lower'].tolist(), "yhat_upper": fc['yhat_upper'].tolist(), "components": comps},
}
json.dump(out, open(sys.argv[2], 'w'))
print(f"{sys.argv[3]}: fit={fit_s:.2f}s T={len(hist)} K={X.shape[1]} S={len(tc)} seas={[ (n, v['fourier_order']) for n,v in m.seasonalities.items()]} y_scale={m.y_scale:.3f} sigma={sigma:.5f} lp={lp:.3f} active_delta={int(np.sum(np.abs(delta)>1e-3))}")

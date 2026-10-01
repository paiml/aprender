import sys, json, time, numpy as np, pandas as pd, torch, warnings
warnings.filterwarnings("ignore")
from neuralprophet import NeuralProphet, set_log_level
set_log_level("ERROR")
df = pd.read_csv(sys.argv[1]); df['ds'] = pd.to_datetime(df['ds'])
H = 365; train = df.iloc[:-H].reset_index(drop=True); test = df.iloc[-H:].reset_index(drop=True)
out = {}
def run(**kw):
    torch.manual_seed(0); np.random.seed(0)
    m = NeuralProphet(**kw); t0 = time.perf_counter()
    try:
        metrics = m.fit(train, freq='D', progress=None)
    except Exception as e:
        print("AUTO-LR FAILED, falling back to learning_rate=0.01:", str(e)[:100])
        torch.manual_seed(0); np.random.seed(0); m = NeuralProphet(learning_rate=0.01, **kw); metrics = m.fit(train, freq='D', progress=None)
    secs = time.perf_counter() - t0
    cn = m.config_normalization; dp = getattr(cn, 'global_data_params', None) or getattr(cn, 'local_data_params', {}).get('__df__')
    cfg = {"lr": m.config_train.learning_rate, "epochs": m.config_train.epochs, "batch": m.config_train.batch_size, "secs": secs,
           "n_train_rows_after_impute": int(m.config_train.n_data) if hasattr(m.config_train, 'n_data') else None,
           "data_params": ({k: {"shift": str(v.shift), "scale": str(v.scale)} for k, v in dp.items()} if dp else None),
           "seasonality": ({n: p.resolution for n, p in m.config_seasonality.periods.items()} if m.config_seasonality else None),
           "changepoints_t": [float(x) for x in np.ravel(m.config_trend.changepoints)] if m.config_trend.changepoints is not None else None,
           "final_train_loss": float(metrics.iloc[-1]['Loss']) if metrics is not None and 'Loss' in metrics else None,
           "final_train_mae": float(metrics.iloc[-1]['MAE']) if metrics is not None and 'MAE' in metrics else None}
    return m, cfg
m, cfg = run(); fut = m.make_future_dataframe(train, periods=H, n_historic_predictions=False); fc = m.predict(fut)
yhat = fc['yhat1'].values[-H:]; fut_ds = fc['ds'].dt.strftime('%Y-%m-%d').values[-H:]
# align to the test rows that exist (Peyton has gaps)
m_ds = dict(zip(fut_ds, yhat)); aligned = np.array([m_ds.get(d.strftime('%Y-%m-%d'), np.nan) for d in test['ds']])
mae = np.nanmean(np.abs(aligned - test['y'].values))
out['trend_seasonality'] = {**cfg, "mae_365ahead_on_test_rows": float(mae), "n_test_rows": int(np.sum(~np.isnan(aligned))), "yhat_by_ds": {d: float(v) for d, v in zip(fut_ds, yhat)}}
for label, kw in [("ar30_linear", dict(n_lags=30)), ("ar30_hidden32", dict(n_lags=30, ar_layers=[32]))]:
    m, cfg = run(**kw); fc = m.predict(df)
    yh = fc['yhat1'].values; ds = fc['ds'].dt.strftime('%Y-%m-%d').values; mm = dict(zip(ds, yh))
    al = np.array([mm.get(d.strftime('%Y-%m-%d'), np.nan) for d in test['ds']]); ok = ~np.isnan(al)
    out[label] = {**cfg, "mae_1step_test": float(np.mean(np.abs(al[ok] - test['y'].values[ok]))), "n_test_rows": int(ok.sum())}
out['naive_1step_mae_test'] = float(np.mean(np.abs(df['y'].values[-H:] - df['y'].values[-H-1:-1])))
out['mean_forecast_mae_test'] = float(np.mean(np.abs(test['y'].values - train['y'].values[-365:].mean())))
json.dump(out, open(sys.argv[2], 'w'))
print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != 'yhat_by_ds'}) for k, v in out.items()}, indent=1))

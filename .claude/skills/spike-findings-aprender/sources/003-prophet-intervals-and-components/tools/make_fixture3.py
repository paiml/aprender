"""make_fixture3.py <mode> <csv> <out.json>   mode: default | holidays | logistic | multiplicative"""
import json, sys, time, logging, numpy as np, pandas as pd
logging.getLogger('cmdstanpy').setLevel(logging.ERROR); logging.getLogger('prophet').setLevel(logging.ERROR)
from prophet import Prophet
mode, csv, out_path = sys.argv[1:4]
df = pd.read_csv(csv); df['ds'] = pd.to_datetime(df['ds'])
CAP = 8.5
PLAYOFFS = ['2008-01-13','2009-01-03','2010-01-16','2010-01-24','2010-02-07','2011-01-08','2013-01-12','2014-01-12','2014-01-19','2014-02-02','2015-01-11','2016-01-17','2016-01-24','2016-02-07']
SUPERBOWLS = ['2010-02-07','2014-02-02','2016-02-07']
def build():
    kw = {}
    if mode == 'holidays':
        hol = pd.concat([pd.DataFrame({'holiday': 'playoff', 'ds': pd.to_datetime(PLAYOFFS), 'lower_window': 0, 'upper_window': 1}),
                         pd.DataFrame({'holiday': 'superbowl', 'ds': pd.to_datetime(SUPERBOWLS), 'lower_window': 0, 'upper_window': 1})])
        kw['holidays'] = hol
    if mode == 'logistic': kw['growth'] = 'logistic'
    if mode == 'multiplicative': kw['seasonality_mode'] = 'multiplicative'
    return Prophet(**kw)
def prep(d):
    d = d.copy()
    if mode == 'logistic': d['cap'] = CAP
    return d
np.random.seed(0)
m = build(); t0 = time.perf_counter(); m.fit(prep(df)); fit_s = time.perf_counter() - t0
future = prep(m.make_future_dataframe(periods=365)); fc = m.predict(future)
hist = m.history
X, prior_scales, component_cols, modes = m.make_all_seasonality_features(hist)
p = m.params; k = float(np.ravel(p['k'])[0]); mm = float(np.ravel(p['m'])[0]); delta = np.ravel(p['delta']).tolist(); beta = np.ravel(p['beta']).tolist(); sigma = float(np.ravel(p['sigma_obs'])[0])
hol_list = []
if mode == 'holidays':
    for _, r in m.holidays.iterrows(): hol_list.append({'holiday': r['holiday'], 'ds': r['ds'].strftime('%Y-%m-%d'), 'lower_window': int(r['lower_window']), 'upper_window': int(r['upper_window'])})
n = len(hist)
comps = {c: fc[c].tolist() for c in fc.columns if c not in ('ds','trend','yhat','yhat_lower','yhat_upper','trend_lower','trend_upper')}
future_mask = np.arange(len(fc)) >= n
out = {
  'dataset': csv.split('/')[-1].replace('.csv',''), 'mode': mode, 'growth': m.growth, 'seasonality_mode': m.seasonality_mode, 'prophet_version': __import__('prophet').__version__,
  'fit_seconds': fit_s, 'n_history': int(n), 'y_scale': float(m.y_scale), 'start': str(m.start.date()), 't_scale_days': float(m.t_scale / pd.Timedelta(days=1)),
  'changepoints_t': m.changepoints_t.tolist(), 'changepoint_prior_scale': m.changepoint_prior_scale, 'holidays_prior_scale': m.holidays_prior_scale,
  'cap': CAP if mode == 'logistic' else None, 'holidays': hol_list,
  'seasonalities': [{'name': nm, 'period': v['period'], 'fourier_order': v['fourier_order'], 'prior_scale': v['prior_scale'], 'mode': v['mode']} for nm, v in m.seasonalities.items()],
  'columns': list(X.columns), 'prior_scales': [float(s) for s in prior_scales],
  's_a': component_cols['additive_terms'].values.tolist(), 's_m': component_cols['multiplicative_terms'].values.tolist(),
  'component_cols': {c: [int(i) for i in np.where(component_cols[c].values == 1)[0]] for c in component_cols.columns},
  'component_modes': {k2: list(v) for k2, v in m.component_modes.items()},
  'params': {'k': k, 'm': mm, 'delta': delta, 'beta': beta, 'sigma_obs': sigma},
  'history': {'ds': hist['ds'].dt.strftime('%Y-%m-%d').tolist(), 't': hist['t'].tolist(), 'y': hist['y'].tolist(), 'y_scaled': hist['y_scaled'].tolist(), 'cap_scaled': hist['cap_scaled'].tolist() if 'cap_scaled' in hist else None},
  'X_first3': X.values[:3].tolist(), 'X_last3': X.values[-3:].tolist(),
  'forecast': {'ds': fc['ds'].dt.strftime('%Y-%m-%d').tolist(), 'trend': fc['trend'].tolist(), 'yhat': fc['yhat'].tolist(), 'yhat_lower': fc['yhat_lower'].tolist(), 'yhat_upper': fc['yhat_upper'].tolist(), 'trend_lower': fc['trend_lower'].tolist(), 'trend_upper': fc['trend_upper'].tolist(), 'components': comps},
  'uncertainty': {'interval_width': m.interval_width, 'uncertainty_samples': m.uncertainty_samples,
                  'future_band_mean_width': float(np.mean((fc['yhat_upper'] - fc['yhat_lower'])[future_mask])), 'hist_band_mean_width': float(np.mean((fc['yhat_upper'] - fc['yhat_lower'])[~future_mask])),
                  'future_trend_band_mean_width': float(np.mean((fc['trend_upper'] - fc['trend_lower'])[future_mask])),
                  'future_band_last30_mean_width': float(np.mean((fc['yhat_upper'] - fc['yhat_lower']).values[-30:]))},
}
if mode == 'default':
    H = 365; tr = df.iloc[:-H]; te = df.iloc[-H:]
    covs = []; widths = []
    for seed in range(3):
        np.random.seed(seed); m2 = Prophet().fit(tr); f2 = m2.predict(te[['ds']])
        inside = ((te['y'].values >= f2['yhat_lower'].values) & (te['y'].values <= f2['yhat_upper'].values)).mean()
        covs.append(float(inside)); widths.append(float(np.mean(f2['yhat_upper'] - f2['yhat_lower'])))
    out['holdout'] = {'n_train': int(len(tr)), 'coverage_by_seed': covs, 'mean_width_by_seed': widths, 'mae': float(np.mean(np.abs(f2['yhat'].values - te['y'].values)))}
json.dump(out, open(out_path, 'w'))
print(f"{out['dataset']}/{mode}: T={n} K={X.shape[1]} S={len(m.changepoints_t)} cols={list(X.columns)[:3]}...{list(X.columns)[-2:]} comps={list(component_cols.columns)} sigma={sigma:.5f} fit={fit_s:.2f}s band_future={out['uncertainty']['future_band_mean_width']:.4f}" + (f" holdout={out['holdout']}" if 'holdout' in out else ''))

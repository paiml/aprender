import json, sys, numpy as np, pandas as pd, logging
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
from prophet import Prophet
def lp_of(m):
    hist = m.history; X, ps, _, _ = m.make_all_seasonality_features(hist); X = X.values
    p = m.params; k=float(np.ravel(p['k'])[0]); mm=float(np.ravel(p['m'])[0]); d=np.ravel(p['delta']); b=np.ravel(p['beta']); s=float(np.ravel(p['sigma_obs'])[0])
    t=hist['t'].values; y=hist['y_scaled'].values; tc=m.changepoints_t; A=(t[:,None]>=tc[None,:]).astype(float)
    mu=(k+A@d)*t+(mm+A@(-tc*d))+X@b
    return -(-k**2/50-mm**2/50-np.abs(d).sum()/m.changepoint_prior_scale-s**2/0.5-np.sum(b**2/(2*np.array(ps)**2))-len(t)*np.log(s)-np.sum((y-mu)**2)/(2*s**2))
for name in sys.argv[1:]:
    df = pd.read_csv(f"data/{name}.csv")
    ref = Prophet().fit(df); fut = ref.make_future_dataframe(periods=365); fr = ref.predict(fut); n = len(df)
    print(f"\n{name} (T={n}): reference LBFGS f = {lp_of(ref):.4f}")
    variants = {"Newton": dict(algorithm='Newton'), "LBFGS init_alpha=1e-1": dict(algorithm='LBFGS', init_alpha=0.1), "LBFGS tol_rel_obj=1e-3": dict(algorithm='LBFGS', tol_rel_obj=1e-3)}
    for label, kw in variants.items():
        try:
            m = Prophet().fit(df, **kw); f = m.predict(fut)
            dh = np.max(np.abs(f['yhat'].values[:n]-fr['yhat'].values[:n])); dfut = np.max(np.abs(f['yhat'].values[n:]-fr['yhat'].values[n:]))
            ye = np.max(np.abs(f['yearly'].values-fr['yearly'].values)) if 'yearly' in f else float('nan')
            print(f"  {label:26s} f = {lp_of(m):.4f} (Δ {lp_of(m)-lp_of(ref):+.4f})  yhat hist max|Δ| {dh:.4f}  future(daily) max|Δ| {dfut:.4f}  yearly max|Δ| {ye:.4f}")
        except Exception as e:
            print(f"  {label:26s} FAILED: {str(e)[:120]}")
    # native-frequency future for monthly data: month starts
    if name != 'peyton_manning':
        futm = ref.make_future_dataframe(periods=24, freq='MS'); frm = ref.predict(futm)
        m2 = Prophet().fit(df, algorithm='Newton'); f2 = m2.predict(futm)
        print(f"  monthly-frequency future (24 MS): Newton vs LBFGS yhat max|Δ| hist {np.max(np.abs(f2['yhat'].values[:n]-frm['yhat'].values[:n])):.4f} future {np.max(np.abs(f2['yhat'].values[n:]-frm['yhat'].values[n:])):.4f}; y range {df['y'].max()-df['y'].min():.1f}")

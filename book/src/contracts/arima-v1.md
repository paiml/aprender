# arima-v1

**Version:** 1.0.0

ARIMA -- Autoregressive Integrated Moving Average time series forecasting

## References

- Box & Jenkins (1970) Time Series Analysis: Forecasting and Control
- Hamilton (1994) Time Series Analysis, Ch. 3-5

## Equations

### ar_forecast

```
y_hat_t = sum_{i=1}^{p} phi_i * y_{t-i}
```

**Domain:** $phi in R^p, y in R^T, t > p$

**Codomain:** `y_hat_t in R`

**Invariants:**

- $Forecast is a finite linear combination of past observations$
- $Deterministic given fixed parameters and history$

### differencing

$$
Delta^d y_t = sum_{k=0}^{d} C(d,k) * (-1)^k * y_{t-k}
$$

**Domain:** $y in R^T, d in {0, 1, 2}$

**Codomain:** $Delta^d y in R^{T-d}$

**Invariants:**

- $d-th order differencing reduces series length by d$
- $d=0 is identity (no differencing)$
- $Output length = T - d$

### forecast_finite

```
y_hat_{T+h} in R for h = 1, ..., n_periods
```

**Domain:** $fitted ARIMA(p,d,q) model, n_periods >= 1$

**Codomain:** $forecasts in R^{n_periods}, all finite$

**Invariants:**

- $Forecast length exactly equals n_periods$
- $All forecast values are finite (no NaN, no Inf)$

### ma_filter

$$
epsilon_weighted = sum_{j=1}^{q} theta_j * epsilon_{t-j}
$$

**Domain:** $theta in R^q, epsilon in R^T (residuals)$

**Codomain:** $epsilon_weighted in R$

**Invariants:**

- $MA component is a finite weighted sum of past residuals$
- $Deterministic given fixed parameters and residuals$

## Proof Obligations

| # | Type | Property | Formal |
|---|------|----------|--------|
| 1 | invariant | Forecast length equals n_periods | $\|forecast(model, n_periods)\| = n_periods$ |
| 2 | bound | All forecasts finite | `forall h in 1..n_periods: \|y_hat_{T+h}\| < infinity` |
| 3 | invariant | Differencing reduces order | $\|Delta^d y\| = \|y\| - d$ |
| 4 | invariant | Forecast deterministic | $forecast(model, n) = forecast(model, n) for same model and data$ |

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| FALSIFY-ARIMA-001 | Forecast length | forecast(model, n_periods) returns exactly n_periods values | Forecast horizon not respected |
| FALSIFY-ARIMA-002 | Forecast finiteness | All forecast values are finite (not NaN, not Inf) | Numerical instability in AR/MA recursion |
| FALSIFY-ARIMA-003 | Differencing length | d-th order differencing produces series of length T - d | Off-by-one in differencing implementation |
| FALSIFY-ARIMA-004 | Forecast deterministic | forecast(model, n) = forecast(model, n) | Non-deterministic state or random initialization leak |

## Kani Harnesses

| ID | Obligation | Bound | Strategy |
|----|------------|-------|----------|
| KANI-ARIMA-001 | ARIMA-INV-001 | 8 | stub_float |
| KANI-ARIMA-002 | ARIMA-BND-001 | 8 | stub_float |
| KANI-ARIMA_-003 | Forecast length equals n_periods | 8 | exhaustive |
| KANI-ARIMA_-004 | All forecasts finite | 8 | exhaustive |
| KANI-ARIMA_-005 | Differencing reduces order | 8 | exhaustive |
| KANI-ARIMA_-006 | Forecast deterministic | 8 | exhaustive |

## QA Gate

**ARIMA Contract** (F-ARIMA-001)

ARIMA time series forecasting correctness quality gate

**Checks:** forecast_length, forecast_finiteness, differencing_length, forecast_deterministic

**Pass criteria:** All 4 falsification tests pass + Kani harnesses verify


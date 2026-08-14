<!-- PCU: lib-time_series | contract: contracts/apr-page-lib-time_series-v1.yaml -->

# Module: `aprender::time_series`

Public module of the `aprender-core` crate.

## Source

[`crates/aprender-core/src/time_series/mod.rs`](https://github.com/paiml/aprender/blob/main/crates/aprender-core/src/time_series/mod.rs) or directory.

## Example

```rust
use aprender::time_series::ARIMA;
// See `cargo doc -p aprender-core --open` for full API reference.
```

## Module summary

`aprender::time_series` exposes classical statistical time-series forecasting.
Today it ships the `ARIMA(p, d, q)` model — autoregressive integrated moving
average — fitted by maximum likelihood, with both AR and MA coefficients
exposed for diagnostics. The API takes `Vector<f64>` rather than `f32` because
maximum-likelihood estimation of long-horizon AR processes is numerically
sensitive and benefits from double precision.

## Key types

| Type | Description |
|------|-------------|
| `ARIMA` | ARIMA(p, d, q) model. `fit` takes a `Vector<f64>` of observations; `forecast(n)` returns the next `n` predictions. |

Diagnostic accessors: `ar_coefficients`, `ma_coefficients`, `intercept`,
`order`.

## Usage patterns

### Pattern 1: Fit and forecast an AR(1) series

```rust
use aprender::time_series::ARIMA;
use aprender::primitives::Vector;

// Synthetic AR(1) data with phi=0.7
let mut series = vec![0.0_f64];
let mut x = 0.0;
for _ in 0..50 {
    x = 0.7 * x + 0.5;  // deterministic for the doc example
    series.push(x);
}
let data = Vector::from_vec(series);

// ARIMA(1, 0, 0) = pure AR(1)
let mut model = ARIMA::new(1, 0, 0);
model.fit(&data).expect("AR(1) fit");

let forecast = model.forecast(5).expect("5-step forecast");
println!("forecast: {:?}", forecast.as_slice());
```

### Pattern 2: Inspect ARIMA(1, 1, 1) coefficients

```rust
use aprender::time_series::ARIMA;
use aprender::primitives::Vector;

let observations = Vector::from_vec((0..40).map(|i| (i as f64).sin()).collect());

let mut model = ARIMA::new(1, 1, 1);
model.fit(&observations).expect("ARIMA(1,1,1) fit");

let (p, d, q) = model.order();
println!("order: ({}, {}, {})", p, d, q);
if let Some(ar) = model.ar_coefficients() {
    println!("AR: {:?}", ar.as_slice());
}
if let Some(ma) = model.ma_coefficients() {
    println!("MA: {:?}", ma.as_slice());
}
```

## See also

- [`bayesian`](bayesian.md) — for posterior forecasting with conjugate priors
- [`stats`](stats.md) — supporting statistical primitives (mean, variance, autocorrelation)
- [`metrics`](metrics.md) — RMSE / MAE for evaluating forecast accuracy

## Full API

Run `cargo doc -p aprender-core --open` for the rendered rustdoc, or browse
[docs.rs/aprender](https://docs.rs/aprender) for the published version.

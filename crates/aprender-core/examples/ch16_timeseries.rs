#![allow(clippy::disallowed_methods)]
//! Chapter 16: Time Series Analysis
//!
//! Demonstrates ARIMA(1,1,0) — fit on trending data, forecast future values.
//! Citation: Box & Jenkins, "Time Series Analysis," 1970
//! Contract: contracts/apr-book-ch16-v1.yaml (v2 — api_calls enforced)

use aprender::primitives::Vector;
use aprender::time_series::ARIMA;

fn main() {
    // Linear trend with noise: y = 2t + 1 + sin(t*0.1)
    let data_vec: Vec<f64> = (0..30)
        .map(|i| 2.0 * i as f64 + 1.0 + (i as f64 * 0.1).sin())
        .collect();
    let data_f64 = Vector::from_vec(data_vec.clone());

    println!("Time series: {} observations", data_f64.len());
    assert!(data_f64.len() >= 10, "Need enough data for ARIMA");

    // ARIMA(1,1,0): one AR term, first-order differencing, no MA
    let mut model = ARIMA::new(1, 1, 0);
    println!("ARIMA(p=1, d=1, q=0)");

    // Fit the model
    model.fit(&data_f64).expect("ARIMA fit should succeed");
    println!("  Model fitted successfully");

    // Check AR coefficients exist after fit
    let ar_coefs = model.ar_coefficients();
    assert!(ar_coefs.is_some(), "AR coefficients must exist after fit");
    let coefs = ar_coefs.unwrap();
    println!("  AR coefficients: {:?}", coefs.as_slice());
    assert_eq!(coefs.len(), 1, "ARIMA(1,_,_) has exactly 1 AR coefficient");

    // Forecast 5 periods ahead
    let n_forecast = 5;
    let forecast = model.forecast(n_forecast).expect("Forecast should succeed");
    println!(
        "  Forecast ({n_forecast} periods): {:?}",
        forecast.as_slice()
    );
    assert_eq!(
        forecast.len(),
        n_forecast,
        "Forecast length must match n_periods"
    );

    // Forecasted values should continue the upward trend
    let last_observed = data_vec[data_vec.len() - 1];
    let first_forecast = forecast.as_slice()[0];
    println!("  Last observed: {last_observed:.1}, First forecast: {first_forecast:.1}");
    assert!(
        first_forecast > last_observed * 0.5,
        "Forecast should roughly continue the trend"
    );

    println!("Chapter 16 contracts: PASSED");
}

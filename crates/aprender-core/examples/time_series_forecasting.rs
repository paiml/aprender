#![allow(clippy::disallowed_methods)]
//! ARIMA Time Series Forecasting
//!
//! Demonstrates ARIMA modeling for time series forecasting with different configurations:
//! - Trend forecasting with ARIMA(1,1,0)
//! - Stationary series with ARIMA(1,0,0)
//! - Complex patterns with ARIMA(2,1,1)
//!
//! # Run
//!
//! ```bash
//! cargo run --example time_series_forecasting
//! ```

use aprender::primitives::Vector;
use aprender::time_series::ARIMA;

fn main() {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║           ARIMA Time Series Forecasting Examples              ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    // Example 1: Forecasting monthly sales with trend
    example_1_sales_forecast();

    println!("\n{}", "═".repeat(64));

    // Example 2: Stationary series (temperature anomalies)
    example_2_stationary_series();

    println!("\n{}", "═".repeat(64));

    // Example 3: Complex pattern with AR and MA components
    example_3_complex_pattern();
}

/// Example 1: Monthly Sales Forecast with ARIMA(1,1,0)
///
/// Demonstrates forecasting sales data with a clear trend using differencing.
fn example_1_sales_forecast() {
    println!("EXAMPLE 1: Monthly Sales Forecast (Trend + Seasonality)");
    println!("{}", "─".repeat(64));

    // Synthetic monthly sales data (in thousands)
    let sales_data = Vector::from_slice(&[
        100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0, 135.0, 140.0, 145.0, 150.0, 155.0,
    ]);

    println!("\n📊 Historical Sales Data (12 months):");
    println!("   Month  1:  $100K");
    println!("   Month  2:  $105K");
    println!("   Month  3:  $110K");
    println!("   ...");
    println!("   Month 12:  $155K");
    println!("   \n   Trend: Clear upward trend (+$5K/month)");

    // Create ARIMA(1,1,0) model
    // p=1: Use previous value
    // d=1: Apply first-order differencing to remove trend
    // q=0: No moving average component
    let mut model = ARIMA::new(1, 1, 0);

    println!("\n🔧 Model Configuration: ARIMA(1, 1, 0)");
    println!("   p=1: Use 1 lagged observation (AR component)");
    println!("   d=1: First-order differencing (remove trend)");
    println!("   q=0: No moving average component");

    // Fit model to historical data
    model
        .fit(&sales_data)
        .expect("Model fitting should succeed");

    // Get model parameters
    let (p, d, q) = model.order();
    println!("\n📈 Fitted Model:");
    println!("   Order: ARIMA({p}, {d}, {q})");
    if let Some(ar_coef) = model.ar_coefficients() {
        println!("   AR coefficient: {:.4}", ar_coef[0]);
    }
    println!("   Intercept: {:.4}", model.intercept());

    // Forecast next 3 months
    let n_forecast = 3;
    let forecast = model
        .forecast(n_forecast)
        .expect("Forecasting should succeed");

    println!("\n🔮 Sales Forecast (next 3 months):");
    for i in 0..n_forecast {
        let month = 13 + i;
        println!("   Month {}: ${:.1}K", month, forecast[i]);
    }

    // Calculate expected growth
    let last_actual = sales_data[sales_data.len() - 1];
    let first_forecast = forecast[0];
    let growth = first_forecast - last_actual;
    println!("\n💡 Analysis:");
    println!("   Expected growth: ${growth:.1}K");
    println!("   Model captures upward trend effectively");
}

/// Example 2: Temperature Anomalies with ARIMA(1,0,0)
///
/// Demonstrates forecasting stationary series (no trend) using pure AR model.
fn example_2_stationary_series() {
    println!("EXAMPLE 2: Temperature Anomalies (Stationary Series)");
    println!("{}", "─".repeat(64));

    // Temperature anomalies (deviations from mean in °C)
    let temp_anomalies = Vector::from_slice(&[
        0.2, -0.1, 0.3, 0.1, -0.2, 0.0, 0.2, -0.3, 0.1, 0.0, -0.1, 0.2, 0.3, 0.1,
    ]);

    println!("\n🌡️  Temperature Anomaly Data (14 time points):");
    println!("   t= 1: +0.2°C");
    println!("   t= 2: -0.1°C");
    println!("   t= 3: +0.3°C");
    println!("   ...");
    println!("   t=14: +0.1°C");
    println!("\n   Series is already stationary (mean-reverting)");

    // Create ARIMA(1,0,0) model (pure AR model)
    // p=1: Use previous value
    // d=0: No differencing needed (already stationary)
    // q=0: No moving average
    let mut model = ARIMA::new(1, 0, 0);

    println!("\n🔧 Model Configuration: ARIMA(1, 0, 0) [AR(1) model]");
    println!("   p=1: First-order autoregression");
    println!("   d=0: No differencing (series is stationary)");
    println!("   q=0: No moving average");

    // Fit model
    model
        .fit(&temp_anomalies)
        .expect("Model fitting should succeed");

    // Display fitted parameters
    println!("\n📈 Fitted AR(1) Model:");
    if let Some(ar_coef) = model.ar_coefficients() {
        println!("   AR(1) coefficient: {:.4}", ar_coef[0]);
        let abs_coef = ar_coef[0].abs();
        if abs_coef < 1.0 {
            println!("   ✓ Series is stationary (|φ| < 1)");
        }
    }
    println!("   Mean anomaly: {:.4}°C", model.intercept());

    // Forecast next 5 time points
    let n_forecast = 5;
    let forecast = model
        .forecast(n_forecast)
        .expect("Forecasting should succeed");

    println!("\n🔮 Temperature Forecast (next 5 periods):");
    for i in 0..n_forecast {
        println!("   t={}: {:+.3}°C", 15 + i, forecast[i]);
    }

    println!("\n💡 Analysis:");
    println!("   Forecasts gradually revert to mean");
    println!("   Typical behavior for stationary AR(1) process");
}

/// Example 3: Complex Pattern with ARIMA(2,1,1)
///
/// Demonstrates full ARIMA model with AR, differencing, and MA components.
fn example_3_complex_pattern() {
    println!("EXAMPLE 3: Complex Time Series with ARIMA(2, 1, 1)");
    println!("{}", "─".repeat(64));

    // Simulated quarterly revenue data (with trend and autocorrelation)
    let revenue_data = Vector::from_slice(&[
        50.0, 52.0, 55.0, 59.0, 64.0, 68.0, 73.0, 79.0, 84.0, 90.0, 95.0, 101.0, 106.0, 112.0,
        118.0, 124.0,
    ]);

    println!("\n💰 Quarterly Revenue Data (16 quarters):");
    println!("   Q1:  $50M");
    println!("   Q2:  $52M");
    println!("   Q3:  $55M");
    println!("   ...");
    println!("   Q16: $124M");
    println!("\n   Pattern: Strong growth + complex autocorrelation");

    // Create ARIMA(2,1,1) model
    // p=2: Use 2 previous values (AR component)
    // d=1: First-order differencing (remove trend)
    // q=1: Use 1 previous forecast error (MA component)
    let mut model = ARIMA::new(2, 1, 1);

    println!("\n🔧 Model Configuration: ARIMA(2, 1, 1)");
    println!("   p=2: Second-order autoregression");
    println!("   d=1: First-order differencing");
    println!("   q=1: First-order moving average");

    // Fit model
    model
        .fit(&revenue_data)
        .expect("Model fitting should succeed");

    // Display model parameters
    println!("\n📈 Fitted ARIMA Model:");
    if let Some(ar_coef) = model.ar_coefficients() {
        println!("   AR coefficients: [{:.4}, {:.4}]", ar_coef[0], ar_coef[1]);
    }
    if let Some(ma_coef) = model.ma_coefficients() {
        println!("   MA coefficient: {:.4}", ma_coef[0]);
    }
    println!("   Intercept: {:.4}", model.intercept());

    // Forecast next 4 quarters
    let n_forecast = 4;
    let forecast = model
        .forecast(n_forecast)
        .expect("Forecasting should succeed");

    println!("\n🔮 Revenue Forecast (next 4 quarters):");
    let mut total_forecast = 0.0;
    for i in 0..n_forecast {
        println!("   Q{}: ${:.1}M", 17 + i, forecast[i]);
        total_forecast += forecast[i];
    }

    println!("\n💡 Forecast Analysis:");
    println!("   Total forecast (4 quarters): ${total_forecast:.1}M");
    let last_actual = revenue_data[revenue_data.len() - 1];
    let growth_rate = (forecast[n_forecast - 1] - last_actual) / last_actual * 100.0;
    println!("   Expected growth over period: {growth_rate:.1}%");
    println!("   Model captures both trend and short-term dynamics");

    println!("\n🎯 Key Takeaways:");
    println!("   • ARIMA(2,1,1) handles complex patterns well");
    println!("   • AR(2) captures momentum and reversals");
    println!("   • MA(1) accounts for forecast error correction");
    println!("   • Differencing removes non-stationarity");
}

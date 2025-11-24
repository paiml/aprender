# ARIMA Time Series Forecasting

ARIMA (Auto-Regressive Integrated Moving Average) models are a class of statistical models for analyzing and forecasting time series data. They combine three components to capture different temporal patterns.

## Theory

### ARIMA(p, d, q) Model

The ARIMA model is defined by three orders:
- **p**: Auto-regressive (AR) order - uses past values
- **d**: Differencing order - removes trends/seasonality
- **q**: Moving average (MA) order - uses past forecast errors

$$
\phi(B)(1-B)^d y_t = \theta(B)\epsilon_t
$$

Where:
- $y_t$: time series value at time $t$
- $B$: backshift operator ($B y_t = y_{t-1}$)
- $\phi(B) = 1 - \phi_1 B - \phi_2 B^2 - \ldots - \phi_p B^p$: AR polynomial
- $\theta(B) = 1 + \theta_1 B + \theta_2 B^2 + \ldots + \theta_q B^q$: MA polynomial
- $\epsilon_t$: white noise error term

### Component Breakdown

**1. Auto-Regressive (AR) Component:**
$$
y_t = c + \phi_1 y_{t-1} + \phi_2 y_{t-2} + \ldots + \phi_p y_{t-p} + \epsilon_t
$$

The current value depends on $p$ previous values.

**2. Integrated (I) Component:**
$$
\nabla^d y_t = (1-B)^d y_t
$$

Apply $d$ orders of differencing to achieve stationarity:
- $d=0$: No differencing (stationary series)
- $d=1$: $\nabla y_t = y_t - y_{t-1}$ (remove linear trend)
- $d=2$: $\nabla^2 y_t$ (remove quadratic trend)

**3. Moving Average (MA) Component:**
$$
y_t = \mu + \epsilon_t + \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2} + \ldots + \theta_q \epsilon_{t-q}
$$

The current value depends on $q$ previous forecast errors.

### Key Properties

1. **Stationarity**: AR component requires $|\phi| < 1$ for stationarity
2. **Invertibility**: MA component requires $|\theta| < 1$ for invertibility
3. **Parsimony**: Use smallest $(p, d, q)$ that captures patterns
4. **AIC/BIC**: Model selection criteria for choosing orders

## Example 1: Sales Forecast with ARIMA(1,1,0)

Forecasting monthly sales with an upward trend using differencing.

```rust,ignore
use aprender::primitives::Vector;
use aprender::time_series::ARIMA;

fn main() {
    // Monthly sales data (in thousands)
    let sales_data = Vector::from_slice(&[
        100.0, 105.0, 110.0, 115.0, 120.0, 125.0,
        130.0, 135.0, 140.0, 145.0, 150.0, 155.0,
    ]);

    // Create ARIMA(1,1,0) model
    // p=1: Use previous value
    // d=1: Remove trend via differencing
    // q=0: No MA component
    let mut model = ARIMA::new(1, 1, 0);

    // Fit model to historical data
    model.fit(&sales_data).unwrap();

    // Forecast next 3 months
    let forecast = model.forecast(3).unwrap();

    println!("Month 13: ${:.1}K", forecast[0]);  // ≈ $165.0K
    println!("Month 14: ${:.1}K", forecast[1]);  // ≈ $180.0K
    println!("Month 15: ${:.1}K", forecast[2]);  // ≈ $200.0K
}
```

**Output:**
```text
Month 13: $165.0K
Month 14: $180.0K
Month 15: $200.0K
```

**Analysis:**
- Differencing removes the linear trend
- AR(1) captures short-term momentum
- Forecasts continue the upward trajectory

## Example 2: Stationary Series with ARIMA(1,0,0)

Forecasting temperature anomalies (already mean-reverting).

```rust,ignore
use aprender::primitives::Vector;
use aprender::time_series::ARIMA;

fn main() {
    // Temperature anomalies (deviations in °C)
    let temp_anomalies = Vector::from_slice(&[
        0.2, -0.1, 0.3, 0.1, -0.2, 0.0, 0.2,
        -0.3, 0.1, 0.0, -0.1, 0.2, 0.3, 0.1,
    ]);

    // ARIMA(1,0,0) = AR(1) model
    let mut model = ARIMA::new(1, 0, 0);
    model.fit(&temp_anomalies).unwrap();

    // Check AR coefficient
    let ar_coef = model.ar_coefficients().unwrap();
    println!("AR(1) coefficient: {:.4}", ar_coef[0]);  // ≈ -0.1277

    // Forecast next 5 periods
    let forecast = model.forecast(5).unwrap();

    for i in 0..5 {
        println!("t={}: {:+.3}°C", 15 + i, forecast[i]);
    }
}
```

**Output:**
```text
AR(1) coefficient: -0.1277
t=15: +0.044°C
t=16: +0.051°C
t=17: +0.051°C
t=18: +0.051°C
t=19: +0.051°C
```

**Analysis:**
- No differencing needed (d=0) for stationary series
- Small AR coefficient indicates weak autocorrelation
- Forecasts revert to mean (~0.05°C) quickly
- Typical behavior for mean-reverting processes

## Example 3: Complex Pattern with ARIMA(2,1,1)

Full ARIMA model capturing trend, momentum, and error correction.

```rust,ignore
use aprender::primitives::Vector;
use aprender::time_series::ARIMA;

fn main() {
    // Quarterly revenue data (millions)
    let revenue_data = Vector::from_slice(&[
        50.0, 52.0, 55.0, 59.0, 64.0, 68.0, 73.0, 79.0,
        84.0, 90.0, 95.0, 101.0, 106.0, 112.0, 118.0, 124.0,
    ]);

    // ARIMA(2,1,1): Full model
    let mut model = ARIMA::new(2, 1, 1);
    model.fit(&revenue_data).unwrap();

    // Model parameters
    let ar_coef = model.ar_coefficients().unwrap();
    let ma_coef = model.ma_coefficients().unwrap();

    println!("AR coefficients: [{:.4}, {:.4}]", ar_coef[0], ar_coef[1]);
    println!("MA coefficient: {:.4}", ma_coef[0]);

    // Forecast next 4 quarters
    let forecast = model.forecast(4).unwrap();

    for i in 0..4 {
        println!("Q{}: ${:.1}M", 17 + i, forecast[i]);
    }
}
```

**Output:**
```text
AR coefficients: [1.0286, 1.0732]
MA coefficient: 0.2500
Q17: $138.7M
Q18: $165.1M
Q19: $213.0M
Q20: $295.5M
```

**Analysis:**
- AR(2) captures both momentum and reversals
- d=1 removes non-stationarity from growth trend
- MA(1) adjusts for forecast errors
- Complex model handles intricate patterns

## Model Selection Guidelines

### Choosing ARIMA Orders

**Identify d (Differencing):**
1. Plot the series - look for trends/seasonality
2. Run stationarity tests (ADF, KPSS)
3. Try d=0 (stationary), d=1 (trend), d=2 (rare)

**Identify p (AR order):**
1. Check Partial Autocorrelation Function (PACF)
2. PACF cuts off at lag p
3. Start with p ∈ {0, 1, 2}

**Identify q (MA order):**
1. Check Autocorrelation Function (ACF)
2. ACF cuts off at lag q
3. Start with q ∈ {0, 1, 2}

### Common ARIMA Patterns

| Pattern | Model | Use Case |
|---------|-------|----------|
| Random walk | ARIMA(0,1,0) | Stock prices, cumulative sums |
| Exponential smoothing | ARIMA(0,1,1) | Simple forecasts with trend |
| AR process | ARIMA(p,0,0) | Stationary series with lags |
| MA process | ARIMA(0,0,q) | Stationary series with shocks |
| ARMA | ARIMA(p,0,q) | Stationary with AR and MA |

## Running the Example

```bash
cargo run --example time_series_forecasting
```

The example demonstrates three real-world scenarios:
1. **Sales forecasting** - Monthly sales with linear trend
2. **Temperature anomalies** - Stationary mean-reverting series
3. **Revenue forecasting** - Complex growth patterns

## Key Takeaways

1. **ARIMA is powerful**: Handles trends, seasonality, and autocorrelation
2. **Start simple**: Try ARIMA(1,1,1) as baseline
3. **Check residuals**: Should be white noise (no patterns)
4. **Validate forecasts**: Use train/test split for evaluation
5. **Use AIC/BIC**: Compare models with information criteria

## References

- Box, G.E.P., Jenkins, G.M. (1976). "Time Series Analysis: Forecasting and Control"
- Hyndman, R.J., Athanasopoulos, G. (2018). "Forecasting: Principles and Practice"

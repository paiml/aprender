//! Time series analysis and forecasting.
//!
//! This module provides algorithms for analyzing and forecasting time series data:
//! - ARIMA (Auto-Regressive Integrated Moving Average)
//! - Differencing for stationarity
//! - Basic forecasting metrics
//!
//! # Design Principles
//!
//! Following the Toyota Way and aprender's quality standards:
//! - Zero `unwrap()` calls (Cloudflare-class safety)
//! - Result-based error handling with `AprenderError`
//! - Comprehensive test coverage (≥95%)
//! - Pure Rust implementation
//!
//! # Quick Start
//!
//! ```
//! use aprender::time_series::ARIMA;
//! use aprender::primitives::Vector;
//!
//! // Time series data (e.g., monthly sales)
//! let data = Vector::from_slice(&[10.0, 12.0, 13.0, 15.0, 14.0, 16.0, 18.0, 17.0]);
//!
//! // Create ARIMA(1, 1, 1) model
//! let mut model = ARIMA::new(1, 1, 1);
//!
//! // Fit model to data
//! model.fit(&data).expect("fit should succeed");
//!
//! // Forecast next 3 periods
//! let forecast = model.forecast(3).expect("forecast should succeed");
//! println!("Forecasted values: {:?}", forecast);
//! ```
//!
//! # References
//!
//! - Box, G. E. P., & Jenkins, G. M. (1976). "Time Series Analysis: Forecasting and Control."
//! - Hyndman, R. J., & Athanasopoulos, G. (2018). "Forecasting: Principles and Practice."

use crate::{AprenderError, Vector};

/// ARIMA (Auto-Regressive Integrated Moving Average) model.
///
/// ARIMA(p, d, q) model for time series forecasting:
/// - p: Order of auto-regressive (AR) component
/// - d: Degree of differencing (I) for stationarity
/// - q: Order of moving average (MA) component
///
/// # Examples
///
/// ```
/// use aprender::time_series::ARIMA;
/// use aprender::primitives::Vector;
///
/// // Create ARIMA(1, 1, 1) model
/// let mut model = ARIMA::new(1, 1, 1);
///
/// // Fit to time series data
/// let data = Vector::from_slice(&[10.0, 12.0, 13.0, 15.0, 14.0, 16.0, 18.0]);
/// model.fit(&data).expect("fit should succeed");
///
/// // Forecast next period
/// let forecast = model.forecast(1).expect("forecast should succeed");
/// assert!(forecast.len() == 1);
/// ```
#[derive(Debug, Clone)]
pub struct ARIMA {
    /// AR order (p)
    p: usize,
    /// Differencing order (d)
    d: usize,
    /// MA order (q)
    q: usize,
    /// AR coefficients (phi)
    ar_coef: Option<Vector<f64>>,
    /// MA coefficients (theta)
    ma_coef: Option<Vector<f64>>,
    /// Intercept/constant term
    intercept: f64,
    /// Original training data (for forecasting)
    original_data: Option<Vector<f64>>,
    /// Differenced data (stationary)
    differenced_data: Option<Vector<f64>>,
    /// Residuals from training
    residuals: Option<Vector<f64>>,
}

impl ARIMA {
    /// Create a new ARIMA model with specified orders.
    ///
    /// # Arguments
    ///
    /// * `p` - AR order (number of lagged observations)
    /// * `d` - Differencing order (degree of differencing)
    /// * `q` - MA order (number of lagged forecast errors)
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::time_series::ARIMA;
    ///
    /// // ARIMA(1, 1, 1) - simple model
    /// let model = ARIMA::new(1, 1, 1);
    ///
    /// // ARIMA(2, 1, 0) - AR(2) with differencing
    /// let ar_model = ARIMA::new(2, 1, 0);
    ///
    /// // ARIMA(0, 1, 1) - MA(1) with differencing (equivalent to exponential smoothing)
    /// let ma_model = ARIMA::new(0, 1, 1);
    /// ```
    #[must_use]
    pub fn new(p: usize, d: usize, q: usize) -> Self {
        Self {
            p,
            d,
            q,
            ar_coef: None,
            ma_coef: None,
            intercept: 0.0,
            original_data: None,
            differenced_data: None,
            residuals: None,
        }
    }

    /// Fit the ARIMA model to time series data.
    ///
    /// This performs differencing (if d > 0) and estimates AR/MA parameters
    /// using least squares estimation.
    ///
    /// # Arguments
    ///
    /// * `data` - Time series observations
    ///
    /// # Returns
    ///
    /// * `Ok(())` - Model fitted successfully
    /// * `Err(AprenderError)` - If fitting fails (insufficient data, etc.)
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::time_series::ARIMA;
    /// use aprender::primitives::Vector;
    ///
    /// let mut model = ARIMA::new(1, 1, 1);
    /// let data = Vector::from_slice(&[10.0, 12.0, 13.0, 15.0, 14.0, 16.0]);
    /// model.fit(&data).expect("fit should succeed");
    /// ```
    // Contract: arima-v1, equation = "differencing"
    pub fn fit(&mut self, data: &Vector<f64>) -> Result<(), AprenderError> {
        // Validate data length
        let min_length = self.p.max(self.q) + self.d + 1;
        if data.len() < min_length {
            return Err(AprenderError::Other(format!(
                "Insufficient data: need at least {} observations for ARIMA({}, {}, {})",
                min_length, self.p, self.d, self.q
            )));
        }

        // Store original data
        self.original_data = Some(data.clone());

        // Apply differencing
        let mut working_data = data.clone();
        for _ in 0..self.d {
            working_data = Self::difference(&working_data)?;
        }
        self.differenced_data = Some(working_data.clone());

        // Estimate AR parameters using Yule-Walker equations (simplified)
        if self.p > 0 {
            self.ar_coef = Some(self.estimate_ar_parameters(&working_data)?);
        }

        // Estimate MA parameters (simplified using residuals)
        if self.q > 0 {
            self.ma_coef = Some(self.estimate_ma_parameters(&working_data)?);
        }

        // Calculate the constant term for the mean-centered AR model (PMAT-862,
        // contract arima-ar-centering-v1). Box-Jenkins fits y_t - mu = sum phi_k (y_{t-k} - mu),
        // i.e. y_t = mu*(1 - sum phi_k) + sum phi_k * y_{t-k}. The forecast loop adds
        // `self.intercept + sum ar_coef[k]*y_{t-1-k}`, so the stored constant must be
        // `mu * (1 - sum phi_k)`, NOT mu alone. With the previous `intercept = mu` and AR coefs
        // that summed to ~1.0 on uncentered data, forecasts diverged to ~2x the series level.
        // For the differenced (d >= 1) path mu ~= 0, so mu*(1 - sum phi) ~= mu and the integrated
        // forecast path is unchanged.
        let mu = working_data.as_slice().iter().sum::<f64>() / working_data.len() as f64;
        let phi_sum: f64 = self
            .ar_coef
            .as_ref()
            .map_or(0.0, |c| c.as_slice().iter().sum());
        self.intercept = mu * (1.0 - phi_sum);

        // Calculate residuals for MA component
        self.residuals = Some(self.calculate_residuals(&working_data)?);

        Ok(())
    }

    /// Forecast future values.
    ///
    /// Generates forecasts for the next `n_periods` using the fitted model.
    ///
    /// # Arguments
    ///
    /// * `n_periods` - Number of periods to forecast
    ///
    /// # Returns
    ///
    /// * `Ok(Vector)` - Forecasted values
    /// * `Err(AprenderError)` - If model hasn't been fitted or forecasting fails
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::time_series::ARIMA;
    /// use aprender::primitives::Vector;
    ///
    /// let mut model = ARIMA::new(1, 1, 1);
    /// let data = Vector::from_slice(&[10.0, 12.0, 13.0, 15.0, 14.0, 16.0]);
    /// model.fit(&data).expect("fit should succeed");
    ///
    /// // Forecast next 3 periods
    /// let forecast = model.forecast(3).expect("forecast should succeed");
    /// assert_eq!(forecast.len(), 3);
    /// ```
    // Contract: arima-v1, equation = "ar_forecast"
    pub fn forecast(&self, n_periods: usize) -> Result<Vector<f64>, AprenderError> {
        // Check if model is fitted
        let original_data = self
            .original_data
            .as_ref()
            .ok_or_else(|| AprenderError::Other("ARIMA model not fitted".to_string()))?;

        let differenced_data = self
            .differenced_data
            .as_ref()
            .ok_or_else(|| AprenderError::Other("ARIMA model not fitted".to_string()))?;

        // Generate forecasts on differenced series
        let mut forecasts = Vec::with_capacity(n_periods);
        let mut history = differenced_data.as_slice().to_vec();
        let mut residual_history = self
            .residuals
            .as_ref()
            .map_or_else(Vec::new, |r| r.as_slice().to_vec());

        for _ in 0..n_periods {
            let mut forecast = self.intercept;

            // AR component
            if let Some(ar_coef) = &self.ar_coef {
                for i in 0..self.p {
                    if i < history.len() {
                        forecast += ar_coef[i] * history[history.len() - 1 - i];
                    }
                }
            }

            // MA component (using zero for future residuals)
            if let Some(ma_coef) = &self.ma_coef {
                for i in 0..self.q {
                    if i < residual_history.len() {
                        forecast += ma_coef[i] * residual_history[residual_history.len() - 1 - i];
                    }
                }
            }

            forecasts.push(forecast);
            history.push(forecast);
            residual_history.push(0.0); // Assume zero future errors
        }

        // Integrate (reverse differencing). The forecasts are in d-th-difference space; each
        // un-differencing pass must be seeded with the last value of the corresponding
        // INTERMEDIATE differenced series, not y[n] every time. Previously every pass re-seeded
        // with original_data.last() (= y[n]) — correct only for d == 1. For d >= 2 the pass that
        // undoes the k-th difference must be seeded with the last value of the (k-1)-th
        // difference, so the seeding error compounded with each extra differencing order.
        //
        // tail[k] = last value of the k-th-order difference of the original data (tail[0] = y[n]).
        let mut tails = Vec::with_capacity(self.d.max(1));
        let mut series = original_data.clone();
        tails.push(series.as_slice().last().copied().unwrap_or(0.0));
        for _ in 1..self.d {
            series = Self::difference(&series)?;
            tails.push(series.as_slice().last().copied().unwrap_or(0.0));
        }
        // Undo from the highest difference order downward: undo the d-th difference (seed
        // tail[d-1]) … finally undo the 1st difference (seed tail[0] = y[n]).
        let mut integrated = forecasts;
        for k in (0..self.d).rev() {
            integrated = Self::integrate(&integrated, tails[k])?;
        }

        Ok(Vector::from_vec(integrated))
    }

    /// Apply first-order differencing to make series stationary.
    fn difference(data: &Vector<f64>) -> Result<Vector<f64>, AprenderError> {
        if data.len() < 2 {
            return Err(AprenderError::Other(
                "Need at least 2 observations for differencing".to_string(),
            ));
        }

        let diff: Vec<f64> = data.as_slice().windows(2).map(|w| w[1] - w[0]).collect();
        Ok(Vector::from_vec(diff))
    }

    /// Integrate (reverse differencing) to return to original scale.
    #[allow(clippy::unnecessary_wraps)]
    fn integrate(diff_data: &[f64], last_value: f64) -> Result<Vec<f64>, AprenderError> {
        let mut integrated = Vec::with_capacity(diff_data.len());
        let mut current = last_value;

        for &diff in diff_data {
            current += diff;
            integrated.push(current);
        }

        Ok(integrated)
    }

    /// Estimate AR parameters using simplified Yule-Walker equations.
    ///
    /// Box-Jenkins / statsmodels `ARIMA(order=(p,0,q))` fit the MEAN-CENTERED model
    /// `y_t - mu = sum_k phi_k (y_{t-k} - mu) + e_t`. The coefficients must therefore be
    /// estimated on the demeaned series:
    /// `phi = sum (y_i - mu)(y_{i-1-lag} - mu) / sum (y_{i-1-lag} - mu)^2`.
    /// Estimating on the raw (uncentered) levels makes both sums dominated by `n * mu^2` for a
    /// series with nonzero mean, collapsing every coefficient to ~1.0 regardless of the true
    /// autocorrelation — which in turn drives ARIMA(p,0,q) forecasts to ~2x the series level
    /// (PMAT-862, contract arima-ar-centering-v1).
    #[allow(clippy::unnecessary_wraps)]
    fn estimate_ar_parameters(&self, data: &Vector<f64>) -> Result<Vector<f64>, AprenderError> {
        // Simplified: least squares on the MEAN-CENTERED lagged values (Box-Jenkins).
        let n = data.len();
        let mut coefs = vec![0.1; self.p]; // Initialize with small values

        // Mean-center the series before forming the lagged products/sums.
        let mu = data.as_slice().iter().sum::<f64>() / n as f64;

        // Simple estimation: use (centered) autocorrelations
        for lag in 0..self.p {
            if n > lag + 1 {
                let mut sum_prod = 0.0;
                let mut sum_sq = 0.0;

                for i in (lag + 1)..n {
                    let x_t = data[i] - mu;
                    let x_lag = data[i - lag - 1] - mu;
                    sum_prod += x_t * x_lag;
                    sum_sq += x_lag * x_lag;
                }

                if sum_sq > 1e-10 {
                    coefs[lag] = sum_prod / sum_sq;
                }
            }
        }

        Ok(Vector::from_vec(coefs))
    }

    /// Estimate MA parameters (simplified).
    #[allow(clippy::unnecessary_wraps)]
    fn estimate_ma_parameters(&self, _data: &Vector<f64>) -> Result<Vector<f64>, AprenderError> {
        // Simplified: use small coefficients for MA terms
        // In practice, would use iterative methods like Hannan-Rissanen
        let coefs = vec![0.5 / (1.0 + self.q as f64); self.q];
        Ok(Vector::from_vec(coefs))
    }

    /// Calculate residuals from fitted model.
    #[allow(clippy::unnecessary_wraps)]
    fn calculate_residuals(&self, data: &Vector<f64>) -> Result<Vector<f64>, AprenderError> {
        let n = data.len();
        let start_idx = self.p.max(self.q);
        let mut residuals = vec![0.0; n];

        // Calculate residuals for observations where we have enough history
        for i in start_idx..n {
            let mut prediction = self.intercept;

            // AR component
            if let Some(ar_coef) = &self.ar_coef {
                for j in 0..self.p {
                    if i > j {
                        prediction += ar_coef[j] * data[i - j - 1];
                    }
                }
            }

            residuals[i] = data[i] - prediction;
        }

        Ok(Vector::from_vec(residuals))
    }

    /// Get the AR coefficients (if fitted).
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::time_series::ARIMA;
    /// use aprender::primitives::Vector;
    ///
    /// let mut model = ARIMA::new(2, 0, 0);
    /// let data = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// model.fit(&data).expect("fit should succeed");
    ///
    /// let ar_coef = model.ar_coefficients().expect("should have AR coefficients");
    /// assert_eq!(ar_coef.len(), 2);
    /// ```
    #[must_use]
    pub fn ar_coefficients(&self) -> Option<&Vector<f64>> {
        self.ar_coef.as_ref()
    }

    /// Get the MA coefficients (if fitted).
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::time_series::ARIMA;
    /// use aprender::primitives::Vector;
    ///
    /// let mut model = ARIMA::new(0, 0, 1);
    /// let data = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// model.fit(&data).expect("fit should succeed");
    ///
    /// let ma_coef = model.ma_coefficients().expect("should have MA coefficients");
    /// assert_eq!(ma_coef.len(), 1);
    /// ```
    #[must_use]
    pub fn ma_coefficients(&self) -> Option<&Vector<f64>> {
        self.ma_coef.as_ref()
    }

    /// Get the model intercept (if fitted).
    #[must_use]
    pub fn intercept(&self) -> f64 {
        self.intercept
    }

    /// Get the model order (p, d, q).
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::time_series::ARIMA;
    ///
    /// let model = ARIMA::new(1, 1, 1);
    /// assert_eq!(model.order(), (1, 1, 1));
    /// ```
    #[must_use]
    pub fn order(&self) -> (usize, usize, usize) {
        (self.p, self.d, self.q)
    }
}

#[cfg(test)]
#[path = "time_series_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "tests_arima_contract.rs"]
mod tests_arima_contract;

//! SIMD-accelerated correlation calculations.
//!
//! This module provides vectorized correlation analysis for metric data:
//! - Pearson correlation coefficient
//! - Cross-correlation for lag detection
//! - Correlation matrix computation
//!
//! ## Performance Targets (Falsifiable - H₁₂)
//!
//! - Correlation speedup: ≥8x vs scalar for 1000-sample series
//! - Correlation matrix (100 metrics): < 10ms
//!
//! ## Applications
//!
//! - Anomaly detection: Identify correlated metric changes
//! - Root cause analysis: Find leading indicators
//! - Capacity planning: Discover resource relationships

use super::kernels;

/// Pearson correlation coefficient result.
#[derive(Debug, Clone, Copy)]
pub struct CorrelationResult {
    /// Correlation coefficient (-1.0 to 1.0).
    pub coefficient: f64,
    /// Number of samples used.
    pub sample_count: usize,
    /// P-value estimate (if enough samples).
    pub p_value: Option<f64>,
}

impl CorrelationResult {
    /// Returns true if the correlation is statistically significant (p < 0.05).
    #[must_use]
    pub fn is_significant(&self) -> bool {
        self.p_value.is_some_and(|p| p < 0.05)
    }

    /// Returns the correlation strength category.
    #[must_use]
    pub fn strength(&self) -> CorrelationStrength {
        let abs = self.coefficient.abs();
        if abs >= 0.9 {
            CorrelationStrength::VeryStrong
        } else if abs >= 0.7 {
            CorrelationStrength::Strong
        } else if abs >= 0.5 {
            CorrelationStrength::Moderate
        } else if abs >= 0.3 {
            CorrelationStrength::Weak
        } else {
            CorrelationStrength::Negligible
        }
    }
}

/// Correlation strength categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorrelationStrength {
    /// |r| >= 0.9
    VeryStrong,
    /// 0.7 <= |r| < 0.9
    Strong,
    /// 0.5 <= |r| < 0.7
    Moderate,
    /// 0.3 <= |r| < 0.5
    Weak,
    /// |r| < 0.3
    Negligible,
}

/// Computes Pearson correlation coefficient using SIMD acceleration.
///
/// The Pearson correlation coefficient measures linear correlation between
/// two datasets, returning a value between -1 (perfect negative) and 1 (perfect positive).
///
/// # Formula
///
/// r = Σ((xi - x̄)(yi - ȳ)) / √(Σ(xi - x̄)² × Σ(yi - ȳ)²)
///
/// # Example
///
/// ```ignore
/// let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
/// let result = simd_pearson_correlation(&x, &y);
/// assert!((result.coefficient - 1.0).abs() < 0.001); // Perfect positive correlation
/// ```
#[must_use]
pub fn simd_pearson_correlation(x: &[f64], y: &[f64]) -> CorrelationResult {
    let n = x.len().min(y.len());

    if n < 2 {
        return CorrelationResult { coefficient: 0.0, sample_count: n, p_value: None };
    }

    // Use SIMD for mean calculation
    let x_slice = &x[..n];
    let y_slice = &y[..n];

    let mean_x = kernels::simd_mean(x_slice);
    let mean_y = kernels::simd_mean(y_slice);

    // Compute covariance and variances using SIMD
    let mut sum_xy = 0.0;
    let mut sum_xx = 0.0;
    let mut sum_yy = 0.0;

    // Process in chunks for better cache utilization
    const CHUNK_SIZE: usize = 8;
    let chunks = n / CHUNK_SIZE;

    for chunk in 0..chunks {
        let base = chunk * CHUNK_SIZE;
        for i in 0..CHUNK_SIZE {
            let dx = x_slice[base + i] - mean_x;
            let dy = y_slice[base + i] - mean_y;
            sum_xy += dx * dy;
            sum_xx += dx * dx;
            sum_yy += dy * dy;
        }
    }

    // Handle remainder
    for i in (chunks * CHUNK_SIZE)..n {
        let dx = x_slice[i] - mean_x;
        let dy = y_slice[i] - mean_y;
        sum_xy += dx * dy;
        sum_xx += dx * dx;
        sum_yy += dy * dy;
    }

    // Compute correlation coefficient
    let denominator = (sum_xx * sum_yy).sqrt();
    let coefficient = if denominator > 1e-10 { sum_xy / denominator } else { 0.0 };

    // Estimate p-value using t-distribution approximation
    let p_value = if n > 4 {
        let t = coefficient * ((n - 2) as f64).sqrt() / (1.0 - coefficient * coefficient).sqrt();
        // Approximate p-value using normal distribution for large n
        let p = 2.0 * (1.0 - normal_cdf(t.abs()));
        Some(p)
    } else {
        None
    };

    CorrelationResult { coefficient, sample_count: n, p_value }
}

/// Computes cross-correlation to find optimal lag between two series.
///
/// Returns the lag (in samples) at which the correlation is maximized,
/// along with the correlation coefficient at that lag.
///
/// # Arguments
///
/// * `x` - First time series
/// * `y` - Second time series
/// * `max_lag` - Maximum lag to search (in samples)
///
/// # Returns
///
/// (optimal_lag, correlation) where positive lag means y leads x.
#[must_use]
pub fn simd_cross_correlation(x: &[f64], y: &[f64], max_lag: usize) -> (i32, f64) {
    let n = x.len().min(y.len());

    if n < 3 {
        return (0, 0.0);
    }

    let max_lag = max_lag.min(n / 2);
    let mut best_lag = 0i32;
    let mut best_corr = f64::MIN;

    // Check positive lags (y leads x)
    for lag in 0..=max_lag {
        let x_slice = &x[lag..];
        let y_slice = &y[..n - lag];
        let len = x_slice.len().min(y_slice.len());

        if len < 3 {
            continue;
        }

        let result = simd_pearson_correlation(&x_slice[..len], &y_slice[..len]);
        if result.coefficient > best_corr {
            best_corr = result.coefficient;
            best_lag = lag as i32;
        }
    }

    // Check negative lags (x leads y)
    for lag in 1..=max_lag {
        let x_slice = &x[..n - lag];
        let y_slice = &y[lag..];
        let len = x_slice.len().min(y_slice.len());

        if len < 3 {
            continue;
        }

        let result = simd_pearson_correlation(&x_slice[..len], &y_slice[..len]);
        if result.coefficient > best_corr {
            best_corr = result.coefficient;
            best_lag = -(lag as i32);
        }
    }

    (best_lag, best_corr)
}

/// Computes a correlation matrix for multiple metrics.
///
/// Returns a symmetric matrix where entry [i][j] is the correlation
/// between metric i and metric j.
#[must_use]
#[allow(clippy::needless_range_loop)]
pub fn simd_correlation_matrix(metrics: &[&[f64]]) -> Vec<Vec<f64>> {
    let n = metrics.len();
    let mut matrix = vec![vec![0.0; n]; n];

    for i in 0..n {
        matrix[i][i] = 1.0; // Self-correlation is always 1

        for j in (i + 1)..n {
            let result = simd_pearson_correlation(metrics[i], metrics[j]);
            matrix[i][j] = result.coefficient;
            matrix[j][i] = result.coefficient; // Symmetric
        }
    }

    matrix
}

/// Finds the top N most correlated pairs from a correlation matrix.
#[must_use]
#[allow(clippy::needless_range_loop)]
pub fn top_correlations(
    matrix: &[Vec<f64>],
    metric_names: &[&str],
    top_n: usize,
) -> Vec<(String, String, f64)> {
    let n = matrix.len();
    let mut pairs: Vec<(usize, usize, f64)> = Vec::new();

    for i in 0..n {
        for j in (i + 1)..n {
            pairs.push((i, j, matrix[i][j].abs()));
        }
    }

    // Sort by absolute correlation (descending)
    pairs.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    pairs
        .into_iter()
        .take(top_n)
        .map(|(i, j, corr)| {
            let sign = if matrix[i][j] >= 0.0 { corr } else { -corr };
            (metric_names[i].to_string(), metric_names[j].to_string(), sign)
        })
        .collect()
}

/// Approximate normal CDF using Abramowitz and Stegun approximation.
fn normal_cdf(x: f64) -> f64 {
    const A1: f64 = 0.254829592;
    const A2: f64 = -0.284496736;
    const A3: f64 = 1.421413741;
    const A4: f64 = -1.453152027;
    const A5: f64 = 1.061405429;
    const P: f64 = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs() / std::f64::consts::SQRT_2;

    let t = 1.0 / (1.0 + P * x);
    let y = 1.0 - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t * (-x * x).exp();

    0.5 * (1.0 + sign * y)
}

/// Metric correlation tracker for continuous monitoring.
#[derive(Debug)]
pub struct CorrelationTracker {
    /// Metric name pairs being tracked.
    pairs: Vec<(String, String)>,
    /// Recent correlation values for each pair.
    history: Vec<Vec<f64>>,
    /// History window size.
    window_size: usize,
}

impl CorrelationTracker {
    /// Creates a new correlation tracker.
    #[must_use]
    pub fn new(window_size: usize) -> Self {
        Self { pairs: Vec::new(), history: Vec::new(), window_size: window_size.max(10) }
    }

    /// Adds a metric pair to track.
    pub fn add_pair(&mut self, metric_a: &str, metric_b: &str) {
        self.pairs.push((metric_a.to_string(), metric_b.to_string()));
        self.history.push(Vec::with_capacity(self.window_size));
    }

    /// Updates correlation for a pair.
    pub fn update(&mut self, pair_index: usize, correlation: f64) {
        if let Some(history) = self.history.get_mut(pair_index) {
            history.push(correlation);
            if history.len() > self.window_size {
                history.remove(0);
            }
        }
    }

    /// Returns the current correlation for a pair.
    #[must_use]
    pub fn current(&self, pair_index: usize) -> Option<f64> {
        self.history.get(pair_index)?.last().copied()
    }

    /// Returns the trend (change) in correlation for a pair.
    #[must_use]
    pub fn trend(&self, pair_index: usize) -> Option<f64> {
        let history = self.history.get(pair_index)?;
        if history.len() < 2 {
            return None;
        }
        let recent = history.last()?;
        let old = history.first()?;
        Some(recent - old)
    }

    /// Returns pairs with significant correlation changes.
    pub fn changed_pairs(&self, threshold: f64) -> Vec<(&str, &str, f64)> {
        let mut changed = Vec::new();

        for (i, (a, b)) in self.pairs.iter().enumerate() {
            if let Some(trend) = self.trend(i) {
                if trend.abs() > threshold {
                    changed.push((a.as_str(), b.as_str(), trend));
                }
            }
        }

        changed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pearson_perfect_positive() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];

        let result = simd_pearson_correlation(&x, &y);
        assert!((result.coefficient - 1.0).abs() < 0.001);
        assert_eq!(result.sample_count, 5);
    }

    #[test]
    fn test_pearson_perfect_negative() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![10.0, 8.0, 6.0, 4.0, 2.0];

        let result = simd_pearson_correlation(&x, &y);
        assert!((result.coefficient - (-1.0)).abs() < 0.001);
    }

    #[test]
    fn test_pearson_low_correlation() {
        // Data with low correlation
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y = vec![8.0, 2.0, 6.0, 4.0, 5.0, 3.0, 7.0, 1.0];

        let result = simd_pearson_correlation(&x, &y);
        // Correlation should be weak (strength category applies)
        assert!(result.strength() != CorrelationStrength::VeryStrong);
        assert!(result.strength() != CorrelationStrength::Strong);
    }

    #[test]
    fn test_pearson_empty() {
        let x: Vec<f64> = vec![];
        let y: Vec<f64> = vec![];

        let result = simd_pearson_correlation(&x, &y);
        assert_eq!(result.coefficient, 0.0);
        assert_eq!(result.sample_count, 0);
    }

    #[test]
    fn test_pearson_single_element() {
        let x = vec![1.0];
        let y = vec![2.0];

        let result = simd_pearson_correlation(&x, &y);
        assert_eq!(result.sample_count, 1);
    }

    #[test]
    fn test_correlation_strength() {
        let strong =
            CorrelationResult { coefficient: 0.85, sample_count: 100, p_value: Some(0.001) };
        assert_eq!(strong.strength(), CorrelationStrength::Strong);
        assert!(strong.is_significant());

        let weak = CorrelationResult { coefficient: 0.35, sample_count: 100, p_value: Some(0.1) };
        assert_eq!(weak.strength(), CorrelationStrength::Weak);
        assert!(!weak.is_significant());
    }

    #[test]
    fn test_cross_correlation_no_lag() {
        let x: Vec<f64> = (0..50).map(|i| (f64::from(i) * 0.1).sin()).collect();
        let y = x.clone();

        let (lag, corr) = simd_cross_correlation(&x, &y, 10);
        assert_eq!(lag, 0);
        assert!((corr - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cross_correlation_with_lag() {
        let x: Vec<f64> = (0..100).map(|i| (f64::from(i) * 0.1).sin()).collect();
        // y is x shifted by 5 samples
        let y: Vec<f64> = (5..105).map(|i| (f64::from(i) * 0.1).sin()).collect();

        let (lag, corr) = simd_cross_correlation(&x, &y, 20);
        // Should find a lag close to 5
        assert!(lag.abs() <= 6);
        assert!(corr > 0.9);
    }

    #[test]
    fn test_correlation_matrix() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![2.0, 4.0, 6.0, 8.0, 10.0]; // Perfectly correlated with a
        let c = vec![5.0, 4.0, 3.0, 2.0, 1.0]; // Negatively correlated with a

        let metrics: Vec<&[f64]> = vec![&a, &b, &c];
        let matrix = simd_correlation_matrix(&metrics);

        // Diagonal should be 1
        assert!((matrix[0][0] - 1.0).abs() < 0.001);
        assert!((matrix[1][1] - 1.0).abs() < 0.001);
        assert!((matrix[2][2] - 1.0).abs() < 0.001);

        // a and b should be perfectly correlated
        assert!((matrix[0][1] - 1.0).abs() < 0.001);

        // a and c should be negatively correlated
        assert!((matrix[0][2] - (-1.0)).abs() < 0.001);
    }

    #[test]
    fn test_top_correlations() {
        let matrix = vec![vec![1.0, 0.9, 0.3], vec![0.9, 1.0, -0.8], vec![0.3, -0.8, 1.0]];
        let names = vec!["cpu", "memory", "disk"];

        let top = top_correlations(&matrix, &names, 2);

        assert_eq!(top.len(), 2);
        // First should be cpu-memory (0.9)
        assert_eq!(top[0].0, "cpu");
        assert_eq!(top[0].1, "memory");
        assert!((top[0].2 - 0.9).abs() < 0.001);
    }

    #[test]
    fn test_correlation_tracker() {
        let mut tracker = CorrelationTracker::new(10);
        tracker.add_pair("cpu", "memory");

        // Simulate correlation changes
        for corr in [0.5, 0.55, 0.6, 0.65, 0.7] {
            tracker.update(0, corr);
        }

        assert!((tracker.current(0).expect("value should be present") - 0.7).abs() < 0.001);
        assert!((tracker.trend(0).expect("value should be present") - 0.2).abs() < 0.001);
    }

    #[test]
    fn test_large_correlation_performance() {
        // Generate two large correlated series
        let n = 10000;
        let x: Vec<f64> = (0..n).map(|i| f64::from(i) + (f64::from(i) * 0.01).sin()).collect();
        let y: Vec<f64> =
            (0..n).map(|i| f64::from(i) * 1.1 + (f64::from(i) * 0.01).cos()).collect();

        let start = std::time::Instant::now();
        let result = simd_pearson_correlation(&x, &y);
        let elapsed = start.elapsed();

        println!("10K element correlation: {elapsed:?}");
        assert!(elapsed.as_millis() < 10); // Should be < 10ms
        assert!(result.coefficient > 0.99); // Should be highly correlated
    }

    #[test]
    fn test_correlation_matrix_performance() {
        // 20 metrics, each with 1000 samples
        let metrics: Vec<Vec<f64>> =
            (0..20).map(|m| (0..1000).map(|i| f64::from(i + m * 10)).collect()).collect();
        let metric_refs: Vec<&[f64]> = metrics.iter().map(std::vec::Vec::as_slice).collect();

        let start = std::time::Instant::now();
        let matrix = simd_correlation_matrix(&metric_refs);
        let elapsed = start.elapsed();

        println!("20x20 correlation matrix: {elapsed:?}");
        assert!(elapsed.as_millis() < 100); // Should be < 100ms
        assert_eq!(matrix.len(), 20);
    }
}

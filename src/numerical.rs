//! Numerical utilities for numerically stable computation
//!
//! Provides compensated summation (Kahan) and other numerical primitives
//! used in training loops and loss computation.

/// Kahan compensated summation — reduces floating-point accumulation error
/// from O(n * eps) to O(eps) for n addends.
///
/// Reference: Kahan (1965) "Pracniques: Further remarks on reducing truncation errors"
///
/// # Example
/// ```
/// use entrenar::numerical::kahan_sum;
///
/// // Many small values where naive summation drifts
/// let values: Vec<f32> = vec![0.1; 100_000];
/// let compensated = kahan_sum(&values);
/// assert!((compensated - 10_000.0).abs() < 0.01);
/// ```
pub fn kahan_sum(values: &[f32]) -> f32 {
    let mut sum = 0.0_f32;
    let mut compensation = 0.0_f32;

    for &val in values {
        let y = val - compensation;
        let t = sum + y;
        compensation = (t - sum) - y;
        sum = t;
    }

    sum
}

/// Kahan compensated summation for f64 values
pub fn kahan_sum_f64(values: &[f64]) -> f64 {
    let mut sum = 0.0_f64;
    let mut compensation = 0.0_f64;

    for &val in values {
        let y = val - compensation;
        let t = sum + y;
        compensation = (t - sum) - y;
        sum = t;
    }

    sum
}

/// Numerically stable mean computation using Kahan summation
pub fn kahan_mean(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    kahan_sum(values) / values.len() as f32
}

/// Numerically stable variance computation (two-pass with Kahan)
pub fn kahan_variance(values: &[f32]) -> f32 {
    if values.len() < 2 {
        return 0.0;
    }
    let mean = kahan_mean(values);
    let sq_diffs: Vec<f32> = values.iter().map(|&x| (x - mean) * (x - mean)).collect();
    kahan_sum(&sq_diffs) / values.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kahan_sum_basic() {
        let values = vec![1.0, 2.0, 3.0, 4.0];
        assert!((kahan_sum(&values) - 10.0).abs() < 1e-7);
    }

    #[test]
    fn test_kahan_sum_accumulated_error() {
        // Many small values: naive accumulation drifts, Kahan stays accurate
        // Sum of 100_000 values of 0.1 should be ~10000.0
        let values: Vec<f32> = vec![0.1; 100_000];
        let naive: f32 = values.iter().sum();
        let compensated = kahan_sum(&values);
        let expected = 10_000.0_f32;

        let kahan_err = (compensated - expected).abs();
        let naive_err = (naive - expected).abs();

        // Kahan should be more accurate than naive
        assert!(
            kahan_err <= naive_err + 1e-3,
            "Kahan error {kahan_err} should be <= naive error {naive_err}"
        );
        // Kahan should be close to expected
        assert!(
            kahan_err < 0.01,
            "Kahan sum = {compensated}, expected ~{expected}, error = {kahan_err}"
        );
    }

    #[test]
    fn test_kahan_sum_many_small_values() {
        // Sum of 1_000_000 values of 1e-7 should be ~0.1
        let values: Vec<f32> = vec![1e-7; 1_000_000];
        let compensated = kahan_sum(&values);
        let expected = 0.1_f32;

        assert!(
            (compensated - expected).abs() < 1e-6,
            "Kahan sum of 1M * 1e-7 = {compensated}, expected ~{expected}"
        );
    }

    #[test]
    fn test_kahan_sum_empty() {
        assert_eq!(kahan_sum(&[]), 0.0);
    }

    #[test]
    fn test_kahan_sum_single() {
        assert_eq!(kahan_sum(&[42.0]), 42.0);
    }

    #[test]
    fn test_kahan_mean() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((kahan_mean(&values) - 3.0).abs() < 1e-7);
    }

    #[test]
    fn test_kahan_variance() {
        let values = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let var = kahan_variance(&values);
        // Known variance for this dataset: 4.0
        assert!((var - 4.0).abs() < 1e-5, "Kahan variance = {var}, expected 4.0");
    }

    #[test]
    fn test_kahan_sum_f64_precision() {
        // Many small f64 values where accumulation error matters
        let values: Vec<f64> = vec![0.1; 1_000_000];
        let compensated = kahan_sum_f64(&values);
        let expected = 100_000.0_f64;
        assert!(
            (compensated - expected).abs() < 1e-8,
            "Kahan f64 sum = {compensated}, expected {expected}"
        );
    }

    /// Numerical validation: verify kahan sum against analytical_solution (EDD-03)
    #[test]
    fn verification_test_kahan_analytical_solution() {
        // Closed-form exact_solution: sum(1..=N) = N*(N+1)/2
        let n = 1000;
        let values: Vec<f32> = (1..=n).map(|i| i as f32).collect();
        let analytical_solution = (n * (n + 1) / 2) as f32;
        let computed = kahan_sum(&values);
        let tolerance = 1e-3;
        assert!(
            (computed - analytical_solution).abs() < tolerance,
            "convergence_test: computed={computed}, exact={analytical_solution}"
        );
    }

    #[test]
    fn test_kahan_vs_naive_accuracy() {
        // Alternating large and small values stress-test accumulation
        let n = 10_000;
        let values: Vec<f32> = (0..n).map(|i| if i % 2 == 0 { 1e6 } else { 1e-6 }).collect();

        let kahan = kahan_sum(&values);
        let naive: f32 = values.iter().sum();
        let exact = (n / 2) as f32 * 1e6 + (n / 2) as f32 * 1e-6;

        // Kahan should be closer to exact than naive
        let kahan_err = (kahan - exact).abs();
        let naive_err = (naive - exact).abs();
        assert!(
            kahan_err <= naive_err + 1e-3,
            "Kahan error {kahan_err} should be <= naive error {naive_err}"
        );
    }
}

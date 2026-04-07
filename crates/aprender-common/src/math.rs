//! Shared mathematical functions for the Batuta stack.
//!
//! Provides common math operations (statistics, special functions) used across
//! pmat, trueno, aprender, and trueno-viz.

// =============================================================================
// ERROR FUNCTION (Abramowitz & Stegun approximation)
// =============================================================================

/// Compute the error function erf(x) using the Abramowitz & Stegun approximation.
///
/// Maximum error: |ε| < 1.5 × 10⁻⁷
///
/// # Examples
/// ```
/// use batuta_common::math::erf;
/// assert!((erf(0.0) - 0.0).abs() < 1e-6);
/// assert!((erf(1.0) - 0.842_700_8).abs() < 1e-5);
/// assert!((erf(-1.0) + 0.842_700_8).abs() < 1e-5);
/// ```
#[must_use]
pub fn erf(x: f64) -> f64 {
    // Abramowitz and Stegun formula 7.1.26
    const A1: f64 = 0.254_829_592;
    const A2: f64 = -0.284_496_736;
    const A3: f64 = 1.421_413_741;
    const A4: f64 = -1.453_152_027;
    const A5: f64 = 1.061_405_429;
    const P: f64 = 0.327_591_1;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + P * x);
    let y = 1.0 - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t * (-x * x).exp();

    sign * y
}

/// Compute erf(x) with f32 precision.
///
/// Convenience wrapper for f32 callers; internally delegates to the f64 version.
#[must_use]
pub fn erf_f32(x: f32) -> f32 {
    erf(f64::from(x)) as f32
}

// =============================================================================
// STANDARD DEVIATION
// =============================================================================

/// Compute sample standard deviation of a slice (Bessel's correction, n-1).
///
/// Returns 0.0 if fewer than 2 elements.
///
/// # Examples
/// ```
/// use batuta_common::math::std_dev;
/// let data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
/// assert!((std_dev(&data) - 2.138).abs() < 0.01);
/// assert_eq!(std_dev(&[1.0]), 0.0);
/// assert_eq!(std_dev(&[]), 0.0);
/// ```
#[must_use]
pub fn std_dev(samples: &[f64]) -> f64 {
    if samples.len() < 2 {
        return 0.0;
    }
    let n = samples.len() as f64;
    let mean = samples.iter().sum::<f64>() / n;
    let variance = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    variance.sqrt()
}

/// Compute sample standard deviation for f32 data.
///
/// Returns 0.0 if fewer than 2 elements.
#[must_use]
pub fn std_dev_f32(samples: &[f32]) -> f32 {
    if samples.len() < 2 {
        return 0.0;
    }
    let n = samples.len() as f32;
    let mean = samples.iter().sum::<f32>() / n;
    let variance = samples.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / (n - 1.0);
    variance.sqrt()
}

/// Compute sample standard deviation given a pre-computed mean.
///
/// Useful when the mean has already been calculated separately.
#[must_use]
pub fn std_dev_with_mean(samples: &[f64], mean: f64) -> f64 {
    if samples.len() < 2 {
        return 0.0;
    }
    let variance =
        samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (samples.len() - 1) as f64;
    variance.sqrt()
}

/// Compute sample standard deviation for f32 data given a pre-computed mean.
#[must_use]
pub fn std_dev_f32_with_mean(samples: &[f32], mean: f32) -> f32 {
    if samples.len() < 2 {
        return 0.0;
    }
    let variance =
        samples.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / (samples.len() - 1) as f32;
    variance.sqrt()
}

// =============================================================================
// COSINE SIMILARITY
// =============================================================================

/// Compute cosine similarity between two f32 vectors.
///
/// Returns 0.0 if either vector has zero norm.
///
/// # Examples
/// ```
/// use batuta_common::math::cosine_similarity_f32;
/// let a = [1.0f32, 0.0, 0.0];
/// let b = [0.0f32, 1.0, 0.0];
/// assert!((cosine_similarity_f32(&a, &b) - 0.0).abs() < 1e-6);
///
/// let c = [1.0f32, 2.0, 3.0];
/// assert!((cosine_similarity_f32(&c, &c) - 1.0).abs() < 1e-6);
/// ```
#[must_use]
pub fn cosine_similarity_f32(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// Compute cosine similarity between two f64 vectors.
///
/// Returns 0.0 if either vector has zero norm.
#[must_use]
pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

// =============================================================================
// USAGE PERCENT
// =============================================================================

/// Compute usage percentage from used/total byte counts.
///
/// Returns 0.0 if `total` is 0 (avoids divide-by-zero).
///
/// # Examples
/// ```
/// use batuta_common::math::usage_percent;
/// assert!((usage_percent(750, 1000) - 75.0).abs() < 1e-10);
/// assert_eq!(usage_percent(0, 0), 0.0);
/// assert!((usage_percent(1024, 4096) - 25.0).abs() < 1e-10);
/// ```
#[must_use]
pub fn usage_percent(used: u64, total: u64) -> f64 {
    if total == 0 {
        return 0.0;
    }
    (used as f64 / total as f64) * 100.0
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- erf ---

    #[test]
    fn test_erf_zero() {
        assert!((erf(0.0) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_erf_positive() {
        assert!((erf(1.0) - 0.842_700_793).abs() < 1e-6);
    }

    #[test]
    fn test_erf_negative_symmetry() {
        assert!((erf(-1.0) + erf(1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_erf_large() {
        assert!((erf(5.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_erf_f32_matches() {
        let f32_val = erf_f32(1.0_f32);
        let f64_val = erf(1.0) as f32;
        assert!((f32_val - f64_val).abs() < 1e-6);
    }

    // --- std_dev ---

    #[test]
    fn test_std_dev_known_value() {
        // Sample std_dev with Bessel's correction (n-1):
        // Mean = 5.0, sum_sq_diff = 32, variance = 32/7 ≈ 4.571, sd ≈ 2.138
        let data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        assert!((std_dev(&data) - 2.138).abs() < 0.01);
    }

    #[test]
    fn test_std_dev_single_element() {
        assert_eq!(std_dev(&[42.0]), 0.0);
    }

    #[test]
    fn test_std_dev_empty() {
        assert_eq!(std_dev(&[]), 0.0);
    }

    #[test]
    fn test_std_dev_identical_values() {
        assert_eq!(std_dev(&[5.0, 5.0, 5.0, 5.0]), 0.0);
    }

    #[test]
    fn test_std_dev_f32() {
        let data: Vec<f32> = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        assert!((std_dev_f32(&data) - 2.138).abs() < 0.02);
    }

    #[test]
    fn test_std_dev_with_mean_matches() {
        let data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let sd1 = std_dev(&data);
        let sd2 = std_dev_with_mean(&data, mean);
        assert!((sd1 - sd2).abs() < 1e-10);
    }

    // --- cosine_similarity ---

    #[test]
    fn test_cosine_identical() {
        let a = [1.0, 2.0, 3.0];
        assert!((cosine_similarity(&a, &a) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_orthogonal() {
        let a = [1.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_opposite() {
        let a = [1.0, 0.0];
        let b = [-1.0, 0.0];
        assert!((cosine_similarity(&a, &b) + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_zero_vector() {
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 2.0, 3.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_cosine_f32() {
        let a = [1.0f32, 0.0, 0.0];
        let b = [0.0f32, 1.0, 0.0];
        assert!(cosine_similarity_f32(&a, &b).abs() < 1e-6);
    }

    // --- usage_percent ---

    #[test]
    fn test_usage_percent_normal() {
        assert!((usage_percent(750, 1000) - 75.0).abs() < 1e-10);
    }

    #[test]
    fn test_usage_percent_zero_total() {
        assert_eq!(usage_percent(0, 0), 0.0);
    }

    #[test]
    fn test_usage_percent_full() {
        assert!((usage_percent(1000, 1000) - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_usage_percent_empty() {
        assert!((usage_percent(0, 1000) - 0.0).abs() < 1e-10);
    }
}

// Statistical functions for regression detection using aprender
//
// This module wraps aprender's hypothesis testing and trueno's vector primitives
// to provide specialized interfaces for syscall trace analysis.
//
// Scientific Foundation:
// - Uses aprender's t-tests (parametric) for comparing distributions
// - Welch's t-test variant handles unequal variances between baseline/current
// - P-values indicate statistical significance (p < 0.05 = significant)
// - Uses trueno::Vector for SIMD-optimized statistics (mean, variance)
// - Uses aprender::stats::DescriptiveStats for quantiles/median

use anyhow::{Context, Result};
use aprender::stats::DescriptiveStats;
use trueno::Vector;

/// Result of statistical comparison between baseline and current distributions
#[derive(Debug, Clone)]
pub struct StatisticalTest {
    /// t-statistic value
    pub statistic: f32,

    /// p-value (two-tailed) - probability that difference is due to chance
    /// - p < 0.05: statistically significant (likely regression)
    /// - p >= 0.05: not significant (noise/random variation)
    pub pvalue: f32,

    /// Degrees of freedom
    pub df: f32,

    /// Median of baseline distribution
    pub baseline_median: f32,

    /// Median of current distribution
    pub current_median: f32,

    /// Variance of baseline distribution
    pub baseline_variance: f32,

    /// Variance of current distribution
    pub current_variance: f32,
}

/// Compare two distributions using Welch's independent t-test
///
/// Uses aprender's `ttest_ind()` with unequal variance assumption (Welch's test).
/// Appropriate for comparing syscall counts/durations between baseline and current traces.
///
/// # Arguments
/// * `baseline` - Counts/durations from golden trace
/// * `current` - Counts/durations from current trace
///
/// # Returns
/// `StatisticalTest` with statistic, p-value, and descriptive statistics
///
/// # Example
/// ```ignore
/// use renacer::regression::statistics::compare_distributions;
///
/// let baseline = vec![10.0, 12.0, 11.0, 13.0, 10.0]; // Stable baseline
/// let current = vec![25.0, 27.0, 26.0, 28.0, 25.0];  // Regressed!
///
/// let result = compare_distributions(&baseline, &current).expect("test");
/// assert!(result.pvalue < 0.05); // Significant difference
/// ```
pub fn compare_distributions(baseline: &[f32], current: &[f32]) -> Result<StatisticalTest> {
    // Validate inputs
    if baseline.is_empty() || current.is_empty() {
        anyhow::bail!("Cannot compare empty distributions");
    }

    if baseline.len() < 2 || current.len() < 2 {
        anyhow::bail!("Need at least 2 samples per distribution for t-test");
    }

    // Use aprender's independent t-test (Welch's variant: unequal variances)
    let ttest_result = aprender::stats::hypothesis::ttest_ind(baseline, current, false)
        .context("Failed to compute t-test")?;

    // Compute descriptive statistics using trueno Vector for SIMD optimization
    let baseline_vec = Vector::from_slice(baseline);
    let current_vec = Vector::from_slice(current);

    let baseline_median = median(&baseline_vec)?;
    let current_median = median(&current_vec)?;

    // trueno 0.7.0 returns Result<f32> for variance
    let baseline_variance =
        baseline_vec.variance().context("Failed to compute baseline variance")?;
    let current_variance = current_vec.variance().context("Failed to compute current variance")?;

    Ok(StatisticalTest {
        statistic: ttest_result.statistic,
        pvalue: ttest_result.pvalue,
        df: ttest_result.df,
        baseline_median,
        current_median,
        baseline_variance,
        current_variance,
    })
}

/// Calculate median using aprender's `DescriptiveStats`
///
/// Median is more robust to outliers than mean, making it suitable for
/// syscall traces which may have spikes.
///
/// Uses aprender's quantile(0.5) which implements the R-7 method with
/// `QuickSelect` for O(n) performance (Floyd & Rivest 1975).
pub fn median(vector: &Vector<f32>) -> Result<f32> {
    let stats = DescriptiveStats::new(vector);
    stats.quantile(0.5).map_err(|e| anyhow::anyhow!("Failed to compute median: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_median_odd_length() {
        let vec = Vector::from_slice(&[1.0, 3.0, 5.0, 7.0, 9.0]);
        assert_eq!(median(&vec).expect("test"), 5.0);
    }

    #[test]
    fn test_median_even_length() {
        let vec = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(median(&vec).expect("test"), 2.5);
    }

    #[test]
    fn test_variance_basic() {
        let vec = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0]);
        let var = vec.variance().expect("test");

        // Note: trueno uses population variance (divide by n), not sample variance (n-1)
        // Expected: mean=5, variance = ((2-5)^2 + (4-5)^2 + (6-5)^2 + (8-5)^2) / 4 = 20/4 = 5.0
        assert!((var - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_variance_constant() {
        let vec = Vector::from_slice(&[5.0, 5.0, 5.0, 5.0]);
        assert_eq!(vec.variance().expect("test"), 0.0);
    }

    #[test]
    fn test_compare_distributions_significant_difference() {
        // Baseline: low counts (stable)
        let baseline = vec![10.0, 12.0, 11.0, 13.0, 10.0];

        // Current: much higher counts (regression!)
        let current = vec![25.0, 27.0, 26.0, 28.0, 25.0];

        let result = compare_distributions(&baseline, &current).expect("test");

        // Should detect significant difference
        assert!(result.pvalue < 0.05, "p-value {} should be < 0.05", result.pvalue);
        assert!(result.current_median > result.baseline_median);
    }

    #[test]
    fn test_compare_distributions_no_difference() {
        // Both distributions similar (no regression)
        let baseline = vec![10.0, 12.0, 11.0, 13.0, 10.0];
        let current = vec![11.0, 13.0, 10.0, 12.0, 11.0];

        let result = compare_distributions(&baseline, &current).expect("test");

        // Should NOT detect significant difference
        assert!(result.pvalue >= 0.05, "p-value {} should be >= 0.05", result.pvalue);
    }

    #[test]
    fn test_compare_distributions_empty_baseline() {
        let baseline: Vec<f32> = vec![];
        let current = vec![10.0, 12.0];

        assert!(compare_distributions(&baseline, &current).is_err());
    }

    #[test]
    fn test_compare_distributions_insufficient_samples() {
        let baseline = vec![10.0]; // Only 1 sample
        let current = vec![12.0, 13.0];

        assert!(compare_distributions(&baseline, &current).is_err());
    }
}

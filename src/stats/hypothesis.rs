//! Statistical Hypothesis Testing
//!
//! Implements classical hypothesis tests for comparing distributions and testing relationships.
//!
//! # Tests
//!
//! - **t-tests**: Compare means (one-sample, two-sample, paired)
//! - **chi-square**: Test categorical distributions (goodness-of-fit, independence)
//! - **ANOVA**: Compare multiple group means (F-test)
//!
//! # Example
//!
//! ```ignore
//! use aprender::stats::hypothesis::{ttest_ind, TTestResult};
//!
//! let group1 = vec![2.3, 2.5, 2.7, 2.9, 3.1];
//! let group2 = vec![3.2, 3.4, 3.6, 3.8, 4.0];
//!
//! let result = ttest_ind(&group1, &group2, true).expect("valid t-test inputs");
//! println!("t-statistic: {:.4}, p-value: {:.4}", result.statistic, result.pvalue);
//! ```

use crate::error::{AprenderError, Result};
use std::f32::consts::PI;

/// Result of a t-test.
#[derive(Debug, Clone)]
pub struct TTestResult {
    /// t-statistic
    pub statistic: f32,

    /// p-value (two-tailed)
    pub pvalue: f32,

    /// Degrees of freedom
    pub df: f32,
}

/// Result of a chi-square test.
#[derive(Debug, Clone)]
pub struct ChiSquareResult {
    /// Chi-square statistic
    pub statistic: f32,

    /// p-value
    pub pvalue: f32,

    /// Degrees of freedom
    pub df: usize,
}

/// Result of an ANOVA F-test.
#[derive(Debug, Clone)]
pub struct AnovaResult {
    /// F-statistic
    pub statistic: f32,

    /// p-value
    pub pvalue: f32,

    /// Between-groups degrees of freedom
    pub df_between: usize,

    /// Within-groups degrees of freedom
    pub df_within: usize,
}

/// One-sample t-test: Tests if sample mean differs from population mean.
///
/// H₀: μ = `population_mean`
/// H₁: μ ≠ `population_mean`
///
/// # Arguments
///
/// * `sample` - Sample data
/// * `population_mean` - Hypothesized population mean
///
/// # Returns
///
/// `TTestResult` with statistic, p-value, and degrees of freedom
pub fn ttest_1samp(sample: &[f32], population_mean: f32) -> Result<TTestResult> {
    let n = sample.len();
    if n < 2 {
        return Err(AprenderError::Other(
            "t-test requires at least 2 samples".into(),
        ));
    }

    // Compute sample mean
    let sample_mean = sample.iter().sum::<f32>() / n as f32;

    // Compute sample standard deviation
    let variance = sample
        .iter()
        .map(|&x| (x - sample_mean).powi(2))
        .sum::<f32>()
        / (n - 1) as f32;
    let std = variance.sqrt();

    // Compute t-statistic: t = (x̄ - μ₀) / (s / √n)
    let se = std / (n as f32).sqrt();
    let t_stat = (sample_mean - population_mean) / se;

    // Degrees of freedom
    let df = (n - 1) as f32;

    // Compute p-value (two-tailed)
    let pvalue = t_distribution_pvalue(t_stat.abs(), df);

    Ok(TTestResult {
        statistic: t_stat,
        pvalue,
        df,
    })
}

/// Independent two-sample t-test: Tests if two independent samples have different means.
///
/// H₀: μ₁ = μ₂
/// H₁: μ₁ ≠ μ₂
///
/// # Arguments
///
/// * `sample1` - First sample
/// * `sample2` - Second sample
/// * `equal_var` - Assume equal variances (pooled t-test) or not (Welch's t-test)
///
/// # Returns
///
/// `TTestResult` with statistic, p-value, and degrees of freedom
pub fn ttest_ind(sample1: &[f32], sample2: &[f32], equal_var: bool) -> Result<TTestResult> {
    let n1 = sample1.len();
    let n2 = sample2.len();

    if n1 < 2 || n2 < 2 {
        return Err(AprenderError::Other(
            "Each sample must have at least 2 observations".into(),
        ));
    }

    // Compute means
    let mean1 = sample1.iter().sum::<f32>() / n1 as f32;
    let mean2 = sample2.iter().sum::<f32>() / n2 as f32;

    // Compute variances
    let var1 = sample1.iter().map(|&x| (x - mean1).powi(2)).sum::<f32>() / (n1 - 1) as f32;
    let var2 = sample2.iter().map(|&x| (x - mean2).powi(2)).sum::<f32>() / (n2 - 1) as f32;

    let (t_stat, df) = if equal_var {
        // Pooled t-test (Student's t-test)
        let pooled_var = ((n1 - 1) as f32 * var1 + (n2 - 1) as f32 * var2) / (n1 + n2 - 2) as f32;
        let se = (pooled_var * (1.0 / n1 as f32 + 1.0 / n2 as f32)).sqrt();
        let t = (mean1 - mean2) / se;
        let df = (n1 + n2 - 2) as f32;
        (t, df)
    } else {
        // Welch's t-test (unequal variances)
        let se = (var1 / n1 as f32 + var2 / n2 as f32).sqrt();
        let t = (mean1 - mean2) / se;

        // Welch-Satterthwaite degrees of freedom
        let numerator = (var1 / n1 as f32 + var2 / n2 as f32).powi(2);
        let denominator = (var1 / n1 as f32).powi(2) / (n1 - 1) as f32
            + (var2 / n2 as f32).powi(2) / (n2 - 1) as f32;
        let df = numerator / denominator;
        (t, df)
    };

    let pvalue = t_distribution_pvalue(t_stat.abs(), df);

    Ok(TTestResult {
        statistic: t_stat,
        pvalue,
        df,
    })
}

/// Paired t-test: Tests if paired samples have different means.
///
/// H₀: `μ_diff` = 0
/// H₁: `μ_diff` ≠ 0
///
/// # Arguments
///
/// * `sample1` - First sample (before)
/// * `sample2` - Second sample (after)
///
/// # Returns
///
/// `TTestResult` with statistic, p-value, and degrees of freedom
pub fn ttest_rel(sample1: &[f32], sample2: &[f32]) -> Result<TTestResult> {
    if sample1.len() != sample2.len() {
        return Err(AprenderError::DimensionMismatch {
            expected: format!("{} samples in sample1", sample1.len()),
            actual: format!("{} samples in sample2", sample2.len()),
        });
    }

    // Compute differences
    let diffs: Vec<f32> = sample1
        .iter()
        .zip(sample2.iter())
        .map(|(&x1, &x2)| x1 - x2)
        .collect();

    // Perform one-sample t-test on differences
    ttest_1samp(&diffs, 0.0)
}

/// Chi-square goodness-of-fit test: Tests if observed frequencies match expected.
///
/// H₀: Observed frequencies follow expected distribution
/// H₁: Observed frequencies do not follow expected distribution
///
/// # Arguments
///
/// * `observed` - Observed frequencies
/// * `expected` - Expected frequencies
///
/// # Returns
///
/// `ChiSquareResult` with statistic, p-value, and degrees of freedom
pub fn chisquare(observed: &[f32], expected: &[f32]) -> Result<ChiSquareResult> {
    if observed.len() != expected.len() {
        return Err(AprenderError::DimensionMismatch {
            expected: format!("{} categories in expected", expected.len()),
            actual: format!("{} categories in observed", observed.len()),
        });
    }

    let k = observed.len();
    if k < 2 {
        return Err(AprenderError::Other(
            "Chi-square test requires at least 2 categories".into(),
        ));
    }

    // Check for negative or zero expected frequencies
    for &exp in expected {
        if exp <= 0.0 {
            return Err(AprenderError::Other(
                "Expected frequencies must be positive".into(),
            ));
        }
    }

    // Compute chi-square statistic: χ² = Σ (O - E)² / E
    let chi2_stat = observed
        .iter()
        .zip(expected.iter())
        .map(|(&obs, &exp)| (obs - exp).powi(2) / exp)
        .sum::<f32>();

    let df = k - 1;
    let pvalue = chi_square_pvalue(chi2_stat, df);

    Ok(ChiSquareResult {
        statistic: chi2_stat,
        pvalue,
        df,
    })
}

/// One-way ANOVA: Tests if multiple groups have the same mean.
///
/// H₀: μ₁ = μ₂ = ... = μₖ
/// H₁: At least one mean is different
///
/// # Arguments
///
/// * `groups` - Vector of samples (each group is a `Vec<f32>`)
///
/// # Returns
///
/// `AnovaResult` with F-statistic, p-value, and degrees of freedom
pub fn f_oneway(groups: &[Vec<f32>]) -> Result<AnovaResult> {
    let k = groups.len();
    if k < 2 {
        return Err(AprenderError::Other(
            "ANOVA requires at least 2 groups".into(),
        ));
    }

    // Check each group has at least 1 observation
    for (i, group) in groups.iter().enumerate() {
        if group.is_empty() {
            return Err(AprenderError::Other(format!(
                "Group {i} is empty. All groups must have at least 1 observation"
            )));
        }
    }

    // Compute group means and overall mean
    let group_means: Vec<f32> = groups
        .iter()
        .map(|g| g.iter().sum::<f32>() / g.len() as f32)
        .collect();

    let n_total: usize = groups.iter().map(Vec::len).sum();
    let grand_mean = groups.iter().flat_map(|g| g.iter()).sum::<f32>() / n_total as f32;

    // Between-group sum of squares: SSB = Σ n_i * (ȳ_i - ȳ)²
    let ss_between = groups
        .iter()
        .zip(group_means.iter())
        .map(|(group, &mean)| group.len() as f32 * (mean - grand_mean).powi(2))
        .sum::<f32>();

    // Within-group sum of squares: SSW = Σ Σ (y_ij - ȳ_i)²
    let ss_within = groups
        .iter()
        .zip(group_means.iter())
        .map(|(group, &mean)| group.iter().map(|&val| (val - mean).powi(2)).sum::<f32>())
        .sum::<f32>();

    // Degrees of freedom
    let df_between = k - 1;
    let df_within = n_total - k;

    if df_within == 0 {
        return Err(AprenderError::Other(
            "Not enough observations for within-group variance".into(),
        ));
    }

    // Mean squares
    let ms_between = ss_between / df_between as f32;
    let ms_within = ss_within / df_within as f32;

    // F-statistic: F = MS_between / MS_within
    let f_stat = ms_between / ms_within;

    // Compute p-value
    let pvalue = f_distribution_pvalue(f_stat, df_between, df_within);

    Ok(AnovaResult {
        statistic: f_stat,
        pvalue,
        df_between,
        df_within,
    })
}

// ============================================================================
// Distribution p-value approximations
// ============================================================================

/// Approximates the two-tailed p-value for a t-distribution.
///
/// Uses numerical approximation for t-distribution CDF.
fn t_distribution_pvalue(t: f32, df: f32) -> f32 {
    // For large df, t-distribution approaches standard normal
    if df > 30.0 {
        return 2.0 * normal_cdf(-t.abs());
    }

    // Simple approximation using beta function relationship
    // P(T > t) ≈ 1 - I_x(df/2, 1/2) where x = df/(df + t²)
    let x = df / (df + t * t);
    let p_one_tail = 0.5 * incomplete_beta(df / 2.0, 0.5, x);
    2.0 * p_one_tail.clamp(0.0, 1.0)
}

/// Approximates the p-value for a chi-square distribution.
fn chi_square_pvalue(chi2: f32, df: usize) -> f32 {
    // P(χ² > x) ≈ 1 - I_x(df/2, 1) using incomplete gamma
    let k = df as f32 / 2.0;
    1.0 - incomplete_gamma(k, chi2 / 2.0)
}

/// Approximates the p-value for an F-distribution.
fn f_distribution_pvalue(f: f32, df1: usize, df2: usize) -> f32 {
    // P(F > x) using beta distribution relationship
    // x_beta = df2 / (df2 + df1 * F)
    let x = df2 as f32 / (df2 as f32 + df1 as f32 * f);
    incomplete_beta(df2 as f32 / 2.0, df1 as f32 / 2.0, x).clamp(0.0, 1.0)
}

/// Standard normal CDF approximation (using error function).
fn normal_cdf(x: f32) -> f32 {
    0.5 * (1.0 + erf(x / 2.0_f32.sqrt()))
}

/// Error function approximation.
fn erf(x: f32) -> f32 {
    // Abramowitz and Stegun approximation
    let a1 = 0.254_829_6_f32;
    let a2 = -0.284_496_72_f32;
    let a3 = 1.421_413_8_f32;
    let a4 = -1.453_152_1_f32;
    let a5 = 1.061_405_4_f32;
    let p = 0.327_591_1_f32;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

/// Incomplete gamma function approximation (series expansion).
fn incomplete_gamma(a: f32, x: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    if a <= 0.0 {
        return 1.0;
    }

    // Series expansion: γ(a,x) = e^(-x) * x^a * Σ x^n / Γ(a+n+1)
    let mut sum = 1.0 / a;
    let mut term = 1.0 / a;
    for n in 1..100 {
        term *= x / (a + n as f32);
        sum += term;
        if term.abs() < 1e-7 {
            break;
        }
    }

    ((-x).exp() * x.powf(a) * sum / gamma(a)).clamp(0.0, 1.0)
}

/// Incomplete beta function approximation.
fn incomplete_beta(a: f32, b: f32, x: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }

    // Simplified approximation using continued fraction
    let bt = (x.powf(a) * (1.0 - x).powf(b)) / (a * beta_function(a, b));

    if x < (a + 1.0) / (a + b + 2.0) {
        bt * beta_continued_fraction(a, b, x) / a
    } else {
        1.0 - bt * beta_continued_fraction(b, a, 1.0 - x) / b
    }
}

/// Beta function B(a, b) = Γ(a)Γ(b)/Γ(a+b).
fn beta_function(a: f32, b: f32) -> f32 {
    gamma(a) * gamma(b) / gamma(a + b)
}

/// Continued fraction for incomplete beta (Lentz's algorithm).
fn beta_continued_fraction(a: f32, b: f32, x: f32) -> f32 {
    let max_iter = 100;
    let eps = 1e-7;

    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;

    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < 1e-30 {
        d = 1e-30;
    }
    d = 1.0 / d;
    let mut h = d;

    for m in 1..=max_iter {
        let m_f = m as f32;
        let m2 = 2.0 * m_f;

        // Even step
        let aa = m_f * (b - m_f) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = 1.0 + aa / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        h *= d * c;

        // Odd step
        let aa = -(a + m_f) * (qab + m_f) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = 1.0 + aa / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;

        if (del - 1.0).abs() < eps {
            break;
        }
    }

    h
}

/// Gamma function approximation (Stirling's approximation).
fn gamma(z: f32) -> f32 {
    if z < 0.5 {
        // Reflection formula: Γ(z) = π / (sin(πz) * Γ(1-z))
        PI / ((PI * z).sin() * gamma(1.0 - z))
    } else {
        // Stirling's approximation
        let z = z - 1.0;
        let tmp = z + 5.5;
        let tmp = (z + 0.5) * tmp.ln() - tmp;
        let ser = 1.0 + 76.180_09_f32 / (z + 1.0) - 86.505_32_f32 / (z + 2.0)
            + 24.014_1_f32 / (z + 3.0)
            - 1.231_739_5_f32 / (z + 4.0)
            + 0.001_208_58_f32 / (z + 5.0)
            - 0.000_005_363_82_f32 / (z + 6.0);
        (tmp + ser.ln()).exp() * (2.0 * PI).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test: One-sample t-test
    #[test]
    fn test_ttest_1samp() {
        // Sample: [2.3, 2.5, 2.7, 2.9, 3.1]
        // Mean ≈ 2.7, testing against μ₀ = 2.5
        let sample = vec![2.3, 2.5, 2.7, 2.9, 3.1];
        let result = ttest_1samp(&sample, 2.5).expect("Valid t-test");

        // t = (2.7 - 2.5) / (s/√5)
        assert!(result.statistic > 0.0, "t-statistic should be positive");
        assert!(result.pvalue > 0.0 && result.pvalue < 1.0);
        assert_eq!(result.df, 4.0);
    }

    /// Test: Independent two-sample t-test (equal variances)
    #[test]
    fn test_ttest_ind_equal_var() {
        let group1 = vec![2.3, 2.5, 2.7, 2.9, 3.1];
        let group2 = vec![3.2, 3.4, 3.6, 3.8, 4.0];

        let result = ttest_ind(&group1, &group2, true).expect("Valid t-test");

        // Group2 mean > Group1 mean, so t should be negative
        assert!(result.statistic < 0.0, "t-statistic should be negative");
        assert!(result.pvalue > 0.0 && result.pvalue < 1.0);
        assert_eq!(result.df, 8.0); // n1 + n2 - 2 = 5 + 5 - 2 = 8
    }

    /// Test: Independent two-sample t-test (unequal variances - Welch's)
    #[test]
    fn test_ttest_ind_unequal_var() {
        let group1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let group2 = vec![10.0, 11.0, 12.0];

        let result = ttest_ind(&group1, &group2, false).expect("Valid Welch's t-test");

        assert!(result.statistic < 0.0); // group1 < group2
        assert!(result.pvalue > 0.0 && result.pvalue < 1.0);
        assert!(result.df > 0.0); // Welch-Satterthwaite df
    }

    /// Test: Paired t-test
    #[test]
    fn test_ttest_rel() {
        // Before-after measurements
        let before = vec![120.0, 122.0, 125.0, 128.0, 130.0];
        let after = vec![115.0, 118.0, 120.0, 123.0, 125.0];

        let result = ttest_rel(&before, &after).expect("Valid paired t-test");

        // After < Before, differences should be positive
        assert!(result.statistic > 0.0);
        assert!(result.pvalue > 0.0 && result.pvalue < 1.0);
        assert_eq!(result.df, 4.0); // n - 1
    }

    /// Test: Paired t-test with dimension mismatch
    #[test]
    fn test_ttest_rel_dimension_mismatch() {
        let sample1 = vec![1.0, 2.0, 3.0];
        let sample2 = vec![4.0, 5.0]; // Different size!

        let result = ttest_rel(&sample1, &sample2);
        assert!(result.is_err());
        let err = result.expect_err("Should be a dimension mismatch error");
        assert!(matches!(err, AprenderError::DimensionMismatch { .. }));
    }

    /// Test: t-test with insufficient data
    #[test]
    fn test_ttest_insufficient_data() {
        let sample = vec![1.0]; // Only 1 sample!
        let result = ttest_1samp(&sample, 0.0);
        assert!(result.is_err());
    }

    /// Test: Chi-square goodness-of-fit
    #[test]
    fn test_chisquare_goodness_of_fit() {
        // Testing if a die is fair
        // Observed: [8, 12, 10, 15, 9, 6] (60 rolls)
        // Expected: [10, 10, 10, 10, 10, 10] (uniform)
        let observed = vec![8.0, 12.0, 10.0, 15.0, 9.0, 6.0];
        let expected = vec![10.0, 10.0, 10.0, 10.0, 10.0, 10.0];

        let result = chisquare(&observed, &expected).expect("Valid chi-square test");

        // χ² = Σ (O-E)²/E
        assert!(result.statistic > 0.0);
        assert!(result.pvalue > 0.0 && result.pvalue < 1.0);
        assert_eq!(result.df, 5); // k - 1 = 6 - 1 = 5
    }

    /// Test: Chi-square with dimension mismatch
    #[test]
    fn test_chisquare_dimension_mismatch() {
        let observed = vec![10.0, 20.0, 30.0];
        let expected = vec![15.0, 25.0]; // Different size!

        let result = chisquare(&observed, &expected);
        assert!(result.is_err());
        let err = result.expect_err("Should be a dimension mismatch error");
        assert!(matches!(err, AprenderError::DimensionMismatch { .. }));
    }

    /// Test: Chi-square with invalid expected frequencies
    #[test]
    fn test_chisquare_invalid_expected() {
        let observed = vec![10.0, 20.0, 30.0];
        let expected = vec![15.0, 0.0, 25.0]; // Zero is invalid!

        let result = chisquare(&observed, &expected);
        assert!(result.is_err());
    }

    /// Test: One-way ANOVA with multiple groups
    #[test]
    fn test_f_oneway() {
        // Three groups with different means
        let group1 = vec![2.0, 2.5, 3.0, 2.8, 2.7];
        let group2 = vec![3.5, 3.8, 4.0, 3.7, 3.9];
        let group3 = vec![5.0, 5.2, 4.8, 5.1, 4.9];

        let result = f_oneway(&[group1, group2, group3]).expect("Valid ANOVA");

        // Groups have different means, F should be large
        assert!(result.statistic > 0.0);
        assert!(result.pvalue > 0.0 && result.pvalue < 1.0);
        assert_eq!(result.df_between, 2); // k - 1 = 3 - 1 = 2
        assert_eq!(result.df_within, 12); // n_total - k = 15 - 3 = 12
    }

    /// Test: ANOVA with identical groups (no difference)
    #[test]
    fn test_f_oneway_no_difference() {
        // Three groups with identical values
        let group1 = vec![3.0, 3.0, 3.0];
        let group2 = vec![3.0, 3.0, 3.0];
        let group3 = vec![3.0, 3.0, 3.0];

        let result = f_oneway(&[group1, group2, group3]).expect("Valid ANOVA");

        // No variance between groups, F should be ~0 (or NaN if MSwithin=0)
        assert!(result.statistic >= 0.0 || result.statistic.is_nan());
    }

    /// Test: ANOVA with insufficient groups
    #[test]
    fn test_f_oneway_insufficient_groups() {
        let group1 = vec![1.0, 2.0, 3.0];
        let result = f_oneway(&[group1]);
        assert!(result.is_err());
    }

    /// Test: ANOVA with empty group
    #[test]
    fn test_f_oneway_empty_group() {
        let group1 = vec![1.0, 2.0, 3.0];
        let group2 = vec![]; // Empty!
        let group3 = vec![4.0, 5.0, 6.0];

        let result = f_oneway(&[group1, group2, group3]);
        assert!(result.is_err());
    }

    /// Test: Normal CDF approximation
    #[test]
    fn test_normal_cdf() {
        // Standard normal values
        assert!((normal_cdf(0.0) - 0.5).abs() < 0.01);
        assert!(normal_cdf(1.96) > 0.97); // ~0.975
        assert!(normal_cdf(-1.96) < 0.03); // ~0.025
    }

    /// Test: Error function
    #[test]
    fn test_erf() {
        assert!((erf(0.0) - 0.0).abs() < 0.01);
        assert!(erf(1.0) > 0.8); // erf(1) ≈ 0.8427
        assert!(erf(-1.0) < -0.8); // erf(-1) ≈ -0.8427
    }

    /// Test: Gamma function
    #[test]
    fn test_gamma() {
        // Γ(1) = 1
        assert!((gamma(1.0) - 1.0).abs() < 0.1);
        // Γ(2) = 1! = 1
        assert!((gamma(2.0) - 1.0).abs() < 0.1);
        // Γ(3) = 2! = 2
        assert!((gamma(3.0) - 2.0).abs() < 0.2);
        // Γ(4) = 3! = 6
        assert!((gamma(4.0) - 6.0).abs() < 0.5);
    }

    /// Test: Real-world example - comparing two treatments
    #[test]
    fn test_real_world_treatment_comparison() {
        // Control group vs treatment group (blood pressure reduction)
        let control = vec![5.0, 7.0, 6.0, 8.0, 5.5, 6.5];
        let treatment = vec![12.0, 14.0, 13.0, 15.0, 11.0, 13.5];

        let result = ttest_ind(&control, &treatment, true).expect("Valid comparison");

        // Treatment should show significantly higher reduction
        assert!(result.statistic < 0.0); // control < treatment
                                         // With this difference, p-value should be small (< 0.05 typically)
        assert!(result.pvalue < 0.1, "Should show significant difference");
    }

    // =========================================================================
    // Additional coverage: Debug/Clone on result structs
    // =========================================================================

    #[test]
    fn test_ttest_result_debug_clone() {
        let result = ttest_1samp(&[2.0, 3.0, 4.0, 5.0, 6.0], 4.0).expect("Valid t-test");
        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("TTestResult"));
        assert!(debug_str.contains("statistic"));
        let cloned = result.clone();
        assert!((cloned.statistic - result.statistic).abs() < 1e-10);
        assert!((cloned.pvalue - result.pvalue).abs() < 1e-10);
        assert!((cloned.df - result.df).abs() < 1e-10);
    }

    #[test]
    fn test_chi_square_result_debug_clone() {
        let observed = vec![10.0, 20.0, 30.0];
        let expected = vec![20.0, 20.0, 20.0];
        let result = chisquare(&observed, &expected).expect("Valid chi-square");
        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("ChiSquareResult"));
        let cloned = result.clone();
        assert!((cloned.statistic - result.statistic).abs() < 1e-10);
        assert_eq!(cloned.df, result.df);
    }

    #[test]
    fn test_anova_result_debug_clone() {
        let g1 = vec![1.0, 2.0, 3.0];
        let g2 = vec![4.0, 5.0, 6.0];
        let result = f_oneway(&[g1, g2]).expect("Valid ANOVA");
        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("AnovaResult"));
        let cloned = result.clone();
        assert!((cloned.statistic - result.statistic).abs() < 1e-10);
        assert_eq!(cloned.df_between, result.df_between);
        assert_eq!(cloned.df_within, result.df_within);
    }

    // =========================================================================
    // Additional coverage: error return paths
    // =========================================================================

    #[test]
    fn test_ttest_ind_first_sample_too_small() {
        let s1 = vec![1.0]; // Only 1 element
        let s2 = vec![2.0, 3.0, 4.0];
        let result = ttest_ind(&s1, &s2, true);
        assert!(result.is_err());
    }

    #[test]
    fn test_ttest_ind_second_sample_too_small() {
        let s1 = vec![1.0, 2.0, 3.0];
        let s2 = vec![4.0]; // Only 1 element
        let result = ttest_ind(&s1, &s2, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_chisquare_too_few_categories() {
        let observed = vec![10.0];
        let expected = vec![10.0];
        let result = chisquare(&observed, &expected);
        assert!(result.is_err());
    }

    #[test]
    fn test_chisquare_negative_expected() {
        let observed = vec![10.0, 20.0, 30.0];
        let expected = vec![15.0, -5.0, 20.0]; // Negative expected
        let result = chisquare(&observed, &expected);
        assert!(result.is_err());
    }

    #[test]
    fn test_f_oneway_zero_groups() {
        let groups: &[Vec<f32>] = &[];
        let result = f_oneway(groups);
        assert!(result.is_err());
    }

    #[test]
    fn test_f_oneway_single_observation_per_group() {
        // Each group has 1 element => df_within = n_total - k = 2 - 2 = 0 => error
        let g1 = vec![1.0];
        let g2 = vec![5.0];
        let result = f_oneway(&[g1, g2]);
        assert!(result.is_err());
    }

    // =========================================================================
    // Additional coverage: distribution helper edge cases
    // =========================================================================

    #[test]
    fn test_t_distribution_large_df_uses_normal_approx() {
        // df > 30 triggers the normal approximation branch
        let large_sample: Vec<f32> = (0..50).map(|i| i as f32 * 0.1).collect();
        let result = ttest_1samp(&large_sample, 0.0).expect("Valid t-test with large n");
        // df = 49, which is > 30, so normal approximation path is taken
        assert!(result.df > 30.0);
        assert!(result.pvalue >= 0.0 && result.pvalue <= 1.0);
    }

    #[test]
    fn test_incomplete_beta_boundary_zero() {
        // x <= 0.0 returns 0.0
        let val = incomplete_beta(1.0, 1.0, 0.0);
        assert!((val - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_incomplete_beta_boundary_one() {
        // x >= 1.0 returns 1.0
        let val = incomplete_beta(1.0, 1.0, 1.0);
        assert!((val - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_incomplete_beta_else_branch() {
        // When x >= (a+1)/(a+b+2) the else branch is taken
        // For a=1, b=1: threshold = 2/4 = 0.5; x=0.8 triggers else branch
        let val = incomplete_beta(1.0, 1.0, 0.8);
        assert!(val > 0.0 && val <= 1.0);
    }

    #[test]
    fn test_incomplete_beta_if_branch() {
        // When x < (a+1)/(a+b+2) the if branch is taken
        // For a=1, b=1: threshold = 2/4 = 0.5; x=0.2 triggers if branch
        let val = incomplete_beta(1.0, 1.0, 0.2);
        assert!(val > 0.0 && val < 1.0);
    }

    #[test]
    fn test_incomplete_gamma_zero_x() {
        let val = incomplete_gamma(1.0, 0.0);
        assert!((val - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_incomplete_gamma_negative_x() {
        let val = incomplete_gamma(1.0, -1.0);
        assert!((val - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_incomplete_gamma_zero_a() {
        let val = incomplete_gamma(0.0, 1.0);
        assert!((val - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_incomplete_gamma_negative_a() {
        let val = incomplete_gamma(-1.0, 1.0);
        assert!((val - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_gamma_reflection_formula() {
        // z < 0.5 triggers the reflection formula branch
        let val = gamma(0.3);
        // Gamma(0.3) is approximately 2.9915...
        assert!(
            val > 2.5 && val < 3.5,
            "Gamma(0.3) should be ~2.99, got {val}"
        );
    }

    #[test]
    fn test_gamma_half() {
        // Gamma(0.5) = sqrt(pi) ~ 1.7724...
        let val = gamma(0.5);
        let sqrt_pi = std::f32::consts::PI.sqrt();
        assert!(
            (val - sqrt_pi).abs() < 0.2,
            "Gamma(0.5) should be ~sqrt(pi)={sqrt_pi}, got {val}"
        );
    }

    #[test]
    fn test_erf_large_positive() {
        // erf(3.0) should be very close to 1.0
        let val = erf(3.0);
        assert!(val > 0.99, "erf(3.0) should be ~1.0, got {val}");
    }

    #[test]
    fn test_erf_large_negative() {
        // erf(-3.0) should be very close to -1.0
        let val = erf(-3.0);
        assert!(val < -0.99, "erf(-3.0) should be ~-1.0, got {val}");
    }

    #[test]
    fn test_normal_cdf_extreme_positive() {
        let val = normal_cdf(5.0);
        assert!(val > 0.999, "Normal CDF at z=5 should be ~1.0");
    }

    #[test]
    fn test_normal_cdf_extreme_negative() {
        let val = normal_cdf(-5.0);
        assert!(val < 0.001, "Normal CDF at z=-5 should be ~0.0");
    }

    #[test]
    fn test_chi_square_pvalue_range() {
        // Ensure chi-square p-value is in [0, 1]
        let pval = chi_square_pvalue(5.0, 3);
        assert!(pval >= 0.0 && pval <= 1.0);
    }

    #[test]
    fn test_f_distribution_pvalue_range() {
        // Ensure f-distribution p-value is in [0, 1]
        let pval = f_distribution_pvalue(3.0, 2, 10);
        assert!(pval >= 0.0 && pval <= 1.0);
    }

    #[test]
    fn test_beta_function_basic() {
        // B(1,1) = Gamma(1)*Gamma(1)/Gamma(2) = 1*1/1 = 1
        let val = beta_function(1.0, 1.0);
        assert!((val - 1.0).abs() < 0.2, "B(1,1) should be ~1.0, got {val}");
    }

    #[test]
    fn test_ttest_1samp_mean_equals_population_mean() {
        // When the sample mean matches population mean, t-stat should be ~0
        let sample = vec![10.0, 10.0, 10.0, 10.0, 10.0];
        let result = ttest_1samp(&sample, 10.0).expect("Valid t-test");
        assert!(
            result.statistic.abs() < 1e-6 || result.statistic.is_nan(),
            "t-stat should be ~0 when means match"
        );
    }

    #[test]
    fn test_ttest_ind_equal_var_identical_samples() {
        let s = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = ttest_ind(&s, &s, true).expect("Valid t-test");
        assert!(
            result.statistic.abs() < 1e-6,
            "t-stat should be 0 for identical samples"
        );
    }

    #[test]
    fn test_ttest_ind_welch_identical_samples() {
        let s = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = ttest_ind(&s, &s, false).expect("Valid Welch's t-test");
        assert!(
            result.statistic.abs() < 1e-6,
            "t-stat should be 0 for identical samples"
        );
    }
}

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

/// Error function approximation (delegates to batuta-common).
fn erf(x: f32) -> f32 {
    batuta_common::math::erf_f32(x)
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

include!("beta_continued_fraction.rs");
include!("hypothesis_part_03.rs");

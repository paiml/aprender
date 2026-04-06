//! Statistical analysis for A/B testing with log-normal latency support
//!
//! Per Box et al. (2005), latency distributions are often log-normal.
//! This module provides log-transform and non-parametric tests.
//!
//! ## Features
//!
//! - **Welch's t-test**: Standard parametric comparison
//! - **Log-transformed t-test**: For log-normal latency data
//! - **Mann-Whitney U test**: Non-parametric test per Box et al. (2005)
//! - **Automatic test selection**: Based on sample size and skewness
//!
//! ## Citations
//!
//! - Box, G. E. P., Hunter, J. S., & Hunter, W. G. (2005).
//!   *Statistics for Experimenters*. Wiley-Interscience.
//! - Welch, B. L. (1947). "The Generalization of 'Student's' Problem."
//!   *Biometrika*, 34(1-2), 28-35.

#![allow(clippy::cast_precision_loss)] // Statistical functions need usize->f64

/// Configuration for experiment analysis
#[derive(Debug, Clone)]
pub struct AnalysisConfig {
    /// Significance level (default 0.05)
    pub alpha: f64,
    /// Whether to auto-detect skewness
    pub auto_detect_skew: bool,
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            alpha: 0.05,
            auto_detect_skew: true,
        }
    }
}

/// Result of statistical analysis
#[derive(Debug, Clone)]
pub struct AnalysisResult {
    /// Control group mean (or geometric mean if log-transformed)
    pub control_mean: f64,
    /// Treatment group mean
    pub treatment_mean: f64,
    /// Effect size (relative change)
    pub effect_size: f64,
    /// P-value for the test
    pub p_value: f64,
    /// Whether result is statistically significant
    pub significant: bool,
    /// Test method used
    pub method: TestMethod,
}

/// Statistical test method used
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TestMethod {
    /// Standard t-test (normal data)
    TTest,
    /// Log-transformed t-test (log-normal data)
    LogTransformTTest,
    /// Mann-Whitney U test (non-parametric)
    MannWhitneyU,
}

/// Analyze experiment results with automatic distribution detection
#[must_use]
pub fn analyze(control: &[f64], treatment: &[f64], config: &AnalysisConfig) -> AnalysisResult {
    let skewness = compute_skewness(control);

    // Auto-detect: use log-transform for highly skewed data (latency)
    let use_log = config.auto_detect_skew && skewness.abs() > 1.0;

    if use_log {
        analyze_log_transform(control, treatment, config.alpha)
    } else {
        analyze_t_test(control, treatment, config.alpha)
    }
}

/// Log-transformed t-test for log-normal latency data
#[must_use]
pub fn analyze_log_transform(control: &[f64], treatment: &[f64], alpha: f64) -> AnalysisResult {
    // Transform to log space
    let log_control: Vec<f64> = control.iter().map(|x| x.ln()).collect();
    let log_treatment: Vec<f64> = treatment.iter().map(|x| x.ln()).collect();

    let result = analyze_t_test(&log_control, &log_treatment, alpha);

    // Convert means back (geometric mean)
    AnalysisResult {
        control_mean: result.control_mean.exp(),
        treatment_mean: result.treatment_mean.exp(),
        effect_size: result.treatment_mean.exp() / result.control_mean.exp() - 1.0,
        p_value: result.p_value,
        significant: result.significant,
        method: TestMethod::LogTransformTTest,
    }
}

/// Standard Welch's t-test
#[must_use]
pub fn analyze_t_test(control: &[f64], treatment: &[f64], alpha: f64) -> AnalysisResult {
    let n1 = control.len().max(1) as f64;
    let n2 = treatment.len().max(1) as f64;

    let mean1 = control.iter().sum::<f64>() / n1;
    let mean2 = treatment.iter().sum::<f64>() / n2;

    let var1 = if n1 > 1.0 {
        control.iter().map(|x| (x - mean1).powi(2)).sum::<f64>() / (n1 - 1.0)
    } else {
        0.0
    };
    let var2 = if n2 > 1.0 {
        treatment.iter().map(|x| (x - mean2).powi(2)).sum::<f64>() / (n2 - 1.0)
    } else {
        0.0
    };

    let se = (var1 / n1 + var2 / n2).sqrt();
    let t_stat = (mean2 - mean1) / se;

    // Approximate p-value using normal distribution (valid for large n)
    let p_value = 2.0 * (1.0 - normal_cdf(t_stat.abs()));

    AnalysisResult {
        control_mean: mean1,
        treatment_mean: mean2,
        effect_size: if mean1.abs() > f64::EPSILON {
            (mean2 - mean1) / mean1
        } else {
            0.0
        },
        p_value,
        significant: p_value < alpha,
        method: TestMethod::TTest,
    }
}

/// Compute skewness of a distribution
fn compute_skewness(data: &[f64]) -> f64 {
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let std_dev = (data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n).sqrt();

    if std_dev < 1e-10 {
        return 0.0;
    }

    let m3 = data
        .iter()
        .map(|x| ((x - mean) / std_dev).powi(3))
        .sum::<f64>()
        / n;
    m3
}

// ============================================================================
// Mann-Whitney U Test (Box et al. 2005)
// ============================================================================

/// Effect size interpretation per Cohen's conventions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffectSizeInterpretation {
    /// |r| < 0.1 - trivial effect
    Negligible,
    /// 0.1 <= |r| < 0.3 - small effect
    Small,
    /// 0.3 <= |r| < 0.5 - medium effect
    Medium,
    /// |r| >= 0.5 - large effect
    Large,
}

/// Result of Mann-Whitney U test
#[derive(Debug, Clone)]
pub struct MannWhitneyResult {
    /// The U statistic (minimum of U1 and U2)
    pub u_statistic: f64,
    /// Z-score for normal approximation
    pub z_score: f64,
    /// P-value (two-tailed)
    pub p_value: f64,
    /// Whether result is significant at alpha=0.05
    pub significant: bool,
    /// Effect size (rank-biserial correlation)
    pub effect_size: f64,
    /// Interpretation of effect size
    pub effect_interpretation: EffectSizeInterpretation,
    /// Test method identifier
    pub method: TestMethod,
}

/// Mann-Whitney U test for non-parametric comparison
///
/// Also known as Wilcoxon rank-sum test. Compares two independent samples
/// without assuming normality. Preferred when:
/// - Distribution is heavily skewed (skewness > 2)
/// - Sample sizes are small (n < 15)
/// - Outliers are present and meaningful
///
/// ## Algorithm
///
/// 1. Combine and rank all observations
/// 2. Handle ties by assigning average ranks
/// 3. Compute U statistic from rank sums
/// 4. Use normal approximation for p-value
///
/// ## Citation
///
/// Box, G. E. P., Hunter, J. S., & Hunter, W. G. (2005).
/// *Statistics for Experimenters*. Wiley-Interscience.
#[must_use]
pub fn mann_whitney_u(control: &[f64], treatment: &[f64]) -> MannWhitneyResult {
    let n1 = control.len();
    let n2 = treatment.len();

    // Combine samples with group labels
    let mut combined: Vec<(f64, usize)> = control
        .iter()
        .map(|&x| (x, 0)) // Group 0 = control
        .chain(treatment.iter().map(|&x| (x, 1))) // Group 1 = treatment
        .collect();

    // Sort by value
    combined.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Assign ranks with tie handling
    let ranks = assign_ranks_with_ties(&combined);

    // Sum ranks for control group
    let r1: f64 = ranks
        .iter()
        .filter(|(_, group)| *group == 0)
        .map(|(rank, _)| rank)
        .sum();

    // Calculate U statistics
    let u1 = r1 - (n1 * (n1 + 1)) as f64 / 2.0;
    let u2 = (n1 * n2) as f64 - u1;
    let u_statistic = u1.min(u2);

    // Normal approximation (valid for n1, n2 >= 5)
    let mu = (n1 * n2) as f64 / 2.0;
    let sigma = ((n1 * n2 * (n1 + n2 + 1)) as f64 / 12.0).sqrt();

    let z_score = if sigma > 0.0 {
        (u_statistic - mu) / sigma
    } else {
        0.0
    };

    // Two-tailed p-value
    let p_value = 2.0 * (1.0 - normal_cdf(z_score.abs()));

    // Effect size: rank-biserial correlation
    // r = 1 - (2U)/(n1*n2)
    let effect_size = 1.0 - (2.0 * u_statistic) / (n1 * n2) as f64;

    let effect_interpretation = interpret_effect_size(effect_size.abs());

    MannWhitneyResult {
        u_statistic,
        z_score,
        p_value,
        significant: p_value < 0.05,
        effect_size,
        effect_interpretation,
        method: TestMethod::MannWhitneyU,
    }
}

/// Assign ranks to sorted values, handling ties by averaging
fn assign_ranks_with_ties(sorted: &[(f64, usize)]) -> Vec<(f64, usize)> {
    let mut ranks = Vec::with_capacity(sorted.len());
    let mut i = 0;

    while i < sorted.len() {
        let value = sorted[i].0;
        let mut j = i;

        // Find extent of tie
        while j < sorted.len() && (sorted[j].0 - value).abs() < 1e-10 {
            j += 1;
        }

        // Average rank for tied values
        // Ranks are 1-indexed: positions i..j get ranks (i+1)..(j+1)
        let avg_rank: f64 = ((i + 1)..=(j)).map(|r| r as f64).sum::<f64>() / (j - i) as f64;

        // Iterate over the slice instead of using index
        for item in sorted.iter().take(j).skip(i) {
            ranks.push((avg_rank, item.1));
        }

        i = j;
    }

    ranks
}

/// Interpret effect size using Cohen's conventions
fn interpret_effect_size(r: f64) -> EffectSizeInterpretation {
    if r < 0.1 {
        EffectSizeInterpretation::Negligible
    } else if r < 0.3 {
        EffectSizeInterpretation::Small
    } else if r < 0.5 {
        EffectSizeInterpretation::Medium
    } else {
        EffectSizeInterpretation::Large
    }
}

// ============================================================================
// Automatic Test Selection (per Gemini review)
// ============================================================================

/// Minimum sample size for parametric tests
const MIN_PARAMETRIC_SAMPLE_SIZE: usize = 15;

/// Skewness threshold for log-transform
const SKEWNESS_THRESHOLD: f64 = 1.0;

/// Analyze with automatic test selection based on data characteristics
///
/// Selection criteria (per Box et al. 2005 recommendations):
/// - Small samples (n < 15): Mann-Whitney U
/// - Highly skewed (|skewness| > 1): Log-transform if possible, else Mann-Whitney
/// - Normal-ish data: Welch's t-test
#[must_use]
pub fn analyze_with_auto_select(
    control: &[f64],
    treatment: &[f64],
    config: &AnalysisConfig,
) -> AnalysisResult {
    let min_n = control.len().min(treatment.len());

    // Small samples: always use non-parametric
    if min_n < MIN_PARAMETRIC_SAMPLE_SIZE {
        let mw = mann_whitney_u(control, treatment);
        return AnalysisResult {
            control_mean: median(control),
            treatment_mean: median(treatment),
            effect_size: mw.effect_size,
            p_value: mw.p_value,
            significant: mw.significant,
            method: TestMethod::MannWhitneyU,
        };
    }

    // Check skewness
    let skewness = compute_skewness(control);

    if config.auto_detect_skew && skewness.abs() > SKEWNESS_THRESHOLD {
        // Check if all values are positive (required for log-transform)
        let all_positive = control.iter().all(|&x| x > 0.0) && treatment.iter().all(|&x| x > 0.0);

        if all_positive {
            analyze_log_transform(control, treatment, config.alpha)
        } else {
            // Can't log-transform, use Mann-Whitney
            let mw = mann_whitney_u(control, treatment);
            AnalysisResult {
                control_mean: median(control),
                treatment_mean: median(treatment),
                effect_size: mw.effect_size,
                p_value: mw.p_value,
                significant: mw.significant,
                method: TestMethod::MannWhitneyU,
            }
        }
    } else {
        analyze_t_test(control, treatment, config.alpha)
    }
}

/// Calculate median of a slice
fn median(data: &[f64]) -> f64 {
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }

    if n.is_multiple_of(2) {
        f64::midpoint(sorted[n / 2 - 1], sorted[n / 2])
    } else {
        sorted[n / 2]
    }
}

/// Normal CDF approximation (Abramowitz and Stegun)
#[allow(clippy::unreadable_literal)] // Standard statistical constants
fn normal_cdf(x: f64) -> f64 {
    let a1 = 0.254_829_592;
    let a2 = -0.284_496_736;
    let a3 = 1.421_413_741;
    let a4 = -1.453_152_027;
    let a5 = 1.061_405_429;
    let p = 0.327_591;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs() / std::f64::consts::SQRT_2;

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    0.5 * (1.0 + sign * y)
}

include!("stats_assign_ranks.rs");

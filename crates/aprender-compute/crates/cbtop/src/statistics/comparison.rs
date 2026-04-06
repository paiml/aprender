//! Statistical comparison tests and outlier filtering.

use std::cmp::Ordering;

use super::analysis::EffectSize;
use super::helpers::{normal_cdf, percentile};

// ---------------------------------------------------------------------------
// Shared helpers extracted to eliminate repeated data-transformation patterns
// ---------------------------------------------------------------------------

/// Compute sample count (as f64), mean, and Bessel-corrected variance.
///
/// Returns `(n, mean, variance)` where `n = samples.len() as f64`.
/// Callers are responsible for ensuring `samples.len() >= 2` when the
/// variance value matters (single-element slices yield `variance = 0.0`).
fn sample_stats(samples: &[f64]) -> (f64, f64, f64) {
    let n = samples.len() as f64;
    let mean = samples.iter().sum::<f64>() / n;
    let divisor = (n - 1.0).max(1.0);
    let var = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / divisor;
    (n, mean, var)
}

/// Two-tailed p-value from an absolute z (or t) statistic using the normal
/// approximation.  Used by both Welch's t-test and Mann-Whitney U.
fn two_tailed_p(abs_stat: f64) -> f64 {
    2.0 * (1.0 - normal_cdf(abs_stat))
}

/// Filter non-finite values and return a sorted `Vec<f64>`.
///
/// Returns `None` when the resulting vector is empty.
fn sort_finite(samples: &[f64]) -> Option<Vec<f64>> {
    let mut v: Vec<f64> = samples.iter().copied().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    Some(v)
}

/// Result of statistical comparison between two samples
#[derive(Debug, Clone)]
pub struct ComparisonResult {
    /// Welch's t-statistic
    pub t_statistic: f64,
    /// Two-tailed p-value
    pub p_value: f64,
    /// Effect size
    pub effect_size: EffectSize,
    /// Whether difference is statistically significant (p < 0.05)
    pub statistically_significant: bool,
    /// Whether difference is practically significant (|d| >= 0.2)
    pub practically_significant: bool,
    /// Degrees of freedom (Welch-Satterthwaite)
    pub degrees_of_freedom: f64,
}

impl ComparisonResult {
    /// Perform Welch's t-test between two samples
    pub fn welch_t_test(sample1: &[f64], sample2: &[f64]) -> Option<Self> {
        if sample1.len() < 2 || sample2.len() < 2 {
            return None;
        }

        let (n1, mean1, var1) = sample_stats(sample1);
        let (n2, mean2, var2) = sample_stats(sample2);

        let se1 = var1 / n1;
        let se2 = var2 / n2;
        let se_diff = (se1 + se2).sqrt();

        if se_diff == 0.0 {
            return None;
        }

        let t = (mean1 - mean2) / se_diff;

        // Welch-Satterthwaite degrees of freedom
        let df = (se1 + se2).powi(2) / (se1.powi(2) / (n1 - 1.0) + se2.powi(2) / (n2 - 1.0));

        let p_value = two_tailed_p(t.abs());
        let effect_size = EffectSize::cohens_d(sample1, sample2)?;

        Some(Self {
            t_statistic: t,
            p_value,
            effect_size: effect_size.clone(),
            statistically_significant: p_value < 0.05,
            practically_significant: effect_size.is_significant(),
            degrees_of_freedom: df,
        })
    }

    /// Both statistically and practically significant
    pub fn is_meaningful(&self) -> bool {
        self.statistically_significant && self.practically_significant
    }
}

/// Mann-Whitney U test result (nonparametric)
#[derive(Debug, Clone)]
pub struct MannWhitneyResult {
    /// U statistic
    pub u_statistic: f64,
    /// Approximate p-value (normal approximation)
    pub p_value: f64,
    /// Rank-biserial correlation (effect size)
    pub effect_size: f64,
    /// Whether difference is significant (p < 0.05)
    pub significant: bool,
}

impl MannWhitneyResult {
    /// Perform Mann-Whitney U test between two samples
    pub fn test(sample1: &[f64], sample2: &[f64]) -> Option<Self> {
        if sample1.is_empty() || sample2.is_empty() {
            return None;
        }

        let n1 = sample1.len();
        let n2 = sample2.len();

        // Combine and rank
        let mut combined: Vec<(f64, usize)> =
            sample1.iter().map(|&x| (x, 0)).chain(sample2.iter().map(|&x| (x, 1))).collect();

        combined.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

        // Assign ranks (handling ties with average rank)
        let mut ranks: Vec<f64> = vec![0.0; combined.len()];
        let mut i = 0;
        while i < combined.len() {
            let mut j = i;
            while j < combined.len() && combined[j].0 == combined[i].0 {
                j += 1;
            }
            let avg_rank = (i + j + 1) as f64 / 2.0;
            for k in i..j {
                ranks[k] = avg_rank;
            }
            i = j;
        }

        // Sum of ranks for sample 1
        let r1: f64 = combined
            .iter()
            .enumerate()
            .filter(|(_, (_, group))| *group == 0)
            .map(|(idx, _)| ranks[idx])
            .sum();

        // U statistics
        let u1 = r1 - (n1 * (n1 + 1)) as f64 / 2.0;
        let u2 = (n1 * n2) as f64 - u1;
        let u = u1.min(u2);

        // Normal approximation for p-value
        let mean_u = (n1 * n2) as f64 / 2.0;
        let std_u = ((n1 * n2 * (n1 + n2 + 1)) as f64 / 12.0).sqrt();

        let z = if std_u > 0.0 { (u - mean_u) / std_u } else { 0.0 };
        let p_value = two_tailed_p(z.abs());

        // Rank-biserial correlation as effect size
        let effect_size = 1.0 - (2.0 * u) / (n1 * n2) as f64;

        Some(Self { u_statistic: u, p_value, effect_size, significant: p_value < 0.05 })
    }
}

/// IQR-based outlier filter
#[derive(Debug, Clone)]
pub struct OutlierFilter {
    /// Lower fence (Q1 - 1.5*IQR)
    pub lower_fence: f64,
    /// Upper fence (Q3 + 1.5*IQR)
    pub upper_fence: f64,
    /// Q1 (25th percentile)
    pub q1: f64,
    /// Q3 (75th percentile)
    pub q3: f64,
    /// IQR (Q3 - Q1)
    pub iqr: f64,
    /// Multiplier (default 1.5)
    pub multiplier: f64,
}

impl OutlierFilter {
    /// Create outlier filter from samples
    pub fn new(samples: &[f64]) -> Option<Self> {
        Self::with_multiplier(samples, 1.5)
    }

    /// Create outlier filter with custom multiplier
    pub fn with_multiplier(samples: &[f64], multiplier: f64) -> Option<Self> {
        let sorted = sort_finite(samples)?;

        let q1 = percentile(&sorted, 0.25);
        let q3 = percentile(&sorted, 0.75);
        let iqr = q3 - q1;

        let lower_fence = q1 - multiplier * iqr;
        let upper_fence = q3 + multiplier * iqr;

        Some(Self { lower_fence, upper_fence, q1, q3, iqr, multiplier })
    }

    /// Check if value is an outlier
    pub fn is_outlier(&self, value: f64) -> bool {
        value < self.lower_fence || value > self.upper_fence
    }

    /// Filter outliers from samples
    pub fn filter(&self, samples: &[f64]) -> Vec<f64> {
        samples.iter().copied().filter(|&x| !self.is_outlier(x)).collect()
    }

    /// Count outliers in samples
    pub fn count_outliers(&self, samples: &[f64]) -> usize {
        samples.iter().filter(|&&x| self.is_outlier(x)).count()
    }
}

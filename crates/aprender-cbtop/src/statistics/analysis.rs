//! Core statistical analysis types and confidence intervals.

use super::helpers::bootstrap_ci;

/// Effect size category per Cohen's conventions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffectCategory {
    /// |d| < 0.2 - negligible practical significance
    Negligible,
    /// 0.2 <= |d| < 0.5 - small effect
    Small,
    /// 0.5 <= |d| < 0.8 - medium effect
    Medium,
    /// |d| >= 0.8 - large effect
    Large,
}

impl EffectCategory {
    /// Get description of effect category
    pub fn description(&self) -> &'static str {
        match self {
            EffectCategory::Negligible => "negligible practical significance",
            EffectCategory::Small => "small effect",
            EffectCategory::Medium => "medium effect",
            EffectCategory::Large => "large effect",
        }
    }

    /// Categorize effect size from Cohen's d value
    pub fn from_cohens_d(d: f64) -> Self {
        let abs_d = d.abs();
        if abs_d < 0.2 {
            EffectCategory::Negligible
        } else if abs_d < 0.5 {
            EffectCategory::Small
        } else if abs_d < 0.8 {
            EffectCategory::Medium
        } else {
            EffectCategory::Large
        }
    }
}

/// Statistical analysis result with confidence interval
#[derive(Debug, Clone)]
pub struct StatisticalAnalysis {
    /// Sample mean
    pub mean: f64,
    /// Sample standard deviation
    pub std_dev: f64,
    /// Standard error of the mean
    pub std_error: f64,
    /// Lower bound of confidence interval
    pub ci_lower: f64,
    /// Upper bound of confidence interval
    pub ci_upper: f64,
    /// Confidence level (e.g., 0.95 for 95%)
    pub confidence_level: f64,
    /// Sample size
    pub n: usize,
    /// Coefficient of variation (std_dev / mean * 100)
    pub cv_percent: f64,
}

impl StatisticalAnalysis {
    /// Compute statistical analysis from samples
    pub fn from_samples(samples: &[f64], confidence_level: f64) -> Option<Self> {
        if samples.is_empty() {
            return None;
        }

        // Filter out NaN/Inf
        let valid: Vec<f64> = samples.iter().copied().filter(|x| x.is_finite()).collect();

        if valid.is_empty() {
            return None;
        }

        let n = valid.len();
        let mean = valid.iter().sum::<f64>() / n as f64;

        if n == 1 {
            return Some(Self {
                mean,
                std_dev: 0.0,
                std_error: 0.0,
                ci_lower: mean,
                ci_upper: mean,
                confidence_level,
                n,
                cv_percent: 0.0,
            });
        }

        let variance = valid.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
        let std_dev = variance.sqrt();
        let std_error = std_dev / (n as f64).sqrt();

        // Bootstrap confidence interval
        let (ci_lower, ci_upper) = bootstrap_ci(&valid, confidence_level, 10000);

        let cv_percent = if mean != 0.0 { (std_dev / mean.abs()) * 100.0 } else { 0.0 };

        Some(Self { mean, std_dev, std_error, ci_lower, ci_upper, confidence_level, n, cv_percent })
    }

    /// Compute with default 95% confidence level
    pub fn from_samples_default(samples: &[f64]) -> Option<Self> {
        Self::from_samples(samples, 0.95)
    }

    /// Get CI width
    pub fn ci_width(&self) -> f64 {
        self.ci_upper - self.ci_lower
    }

    /// Check if CI is narrow (< 10% of mean)
    pub fn ci_is_narrow(&self) -> bool {
        if self.mean == 0.0 {
            return self.ci_width() < 0.1;
        }
        (self.ci_width() / self.mean.abs()) < 0.1
    }
}

/// Effect size calculation result
#[derive(Debug, Clone)]
pub struct EffectSize {
    /// Cohen's d value
    pub cohens_d: f64,
    /// Effect category
    pub category: EffectCategory,
    /// 95% CI lower for effect size
    pub ci_lower: f64,
    /// 95% CI upper for effect size
    pub ci_upper: f64,
}

impl EffectSize {
    /// Calculate Cohen's d between two samples
    pub fn cohens_d(sample1: &[f64], sample2: &[f64]) -> Option<Self> {
        if sample1.is_empty() || sample2.is_empty() {
            return None;
        }

        let n1 = sample1.len() as f64;
        let n2 = sample2.len() as f64;

        let mean1 = sample1.iter().sum::<f64>() / n1;
        let mean2 = sample2.iter().sum::<f64>() / n2;

        let var1 = sample1.iter().map(|x| (x - mean1).powi(2)).sum::<f64>() / (n1 - 1.0).max(1.0);
        let var2 = sample2.iter().map(|x| (x - mean2).powi(2)).sum::<f64>() / (n2 - 1.0).max(1.0);

        // Pooled standard deviation
        let pooled_var = ((n1 - 1.0) * var1 + (n2 - 1.0) * var2) / (n1 + n2 - 2.0).max(1.0);
        let pooled_std = pooled_var.sqrt();

        if pooled_std == 0.0 {
            return Some(Self {
                cohens_d: 0.0,
                category: EffectCategory::Negligible,
                ci_lower: 0.0,
                ci_upper: 0.0,
            });
        }

        let d = (mean1 - mean2) / pooled_std;
        let category = EffectCategory::from_cohens_d(d);

        // Approximate 95% CI for Cohen's d using non-central t approximation
        let se_d = ((n1 + n2) / (n1 * n2) + d.powi(2) / (2.0 * (n1 + n2))).sqrt();
        let ci_lower = d - 1.96 * se_d;
        let ci_upper = d + 1.96 * se_d;

        Some(Self { cohens_d: d, category, ci_lower, ci_upper })
    }

    /// Check if effect is practically significant
    pub fn is_significant(&self) -> bool {
        self.category != EffectCategory::Negligible
    }
}

//! Regression detection using bootstrap confidence intervals.
//! Methodology from Hoefler & Belli (2015) [8]: "Scientific Benchmarking
//! of Parallel Computing Systems."
//! Also supports PELT changepoint detection [43].

use serde::{Deserialize, Serialize};

/// Result of a regression comparison.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Verdict {
    Regression,
    Improvement,
    NoChange,
}

impl std::fmt::Display for Verdict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Verdict::Regression => write!(f, "REGRESSION"),
            Verdict::Improvement => write!(f, "IMPROVED"),
            Verdict::NoChange => write!(f, "NO_CHANGE"),
        }
    }
}

/// Full regression comparison result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionResult {
    pub verdict: Verdict,
    /// Change from baseline (positive = slower/regression, negative = faster/improvement)
    pub change_pct: f64,
    /// Statistical significance
    pub p_value: f64,
    /// Cohen's d effect size (|d| >= 0.8 = large effect)
    pub effect_size_cohens_d: f64,
    /// Bootstrap 99% CI lower bound for ratio (current/baseline)
    pub ci_lower: f64,
    /// Bootstrap 99% CI upper bound for ratio (current/baseline)
    pub ci_upper: f64,
}

/// Performance regression detector.
/// Uses bootstrap confidence intervals per Hoefler & Belli [8].
pub struct RegressionDetector {
    /// Minimum number of samples for statistical significance.
    pub min_samples: usize,
    /// Confidence level for bootstrap CI.
    pub confidence: f64,
    /// Regression threshold (fraction, e.g., 0.05 = 5%).
    pub threshold: f64,
    /// Require large effect size (Cohen's d >= 0.8) in addition to CI.
    pub require_large_effect: bool,
    /// Number of bootstrap resamples.
    pub bootstrap_iterations: usize,
}

impl Default for RegressionDetector {
    fn default() -> Self {
        Self {
            min_samples: 30,
            confidence: 0.99,
            threshold: 0.05,
            require_large_effect: true,
            bootstrap_iterations: 10_000,
        }
    }
}

impl RegressionDetector {
    pub fn new() -> Self {
        Self::default()
    }

    /// Compare baseline and current samples.
    /// Returns the regression verdict with statistical details.
    pub fn compare(&self, baseline: &[f64], current: &[f64]) -> RegressionResult {
        if baseline.is_empty() || current.is_empty() {
            return RegressionResult {
                verdict: Verdict::NoChange,
                change_pct: 0.0,
                p_value: 1.0,
                effect_size_cohens_d: 0.0,
                ci_lower: 1.0,
                ci_upper: 1.0,
            };
        }

        let baseline_mean = mean(baseline);
        let current_mean = mean(current);

        if baseline_mean == 0.0 {
            return RegressionResult {
                verdict: Verdict::NoChange,
                change_pct: 0.0,
                p_value: 1.0,
                effect_size_cohens_d: 0.0,
                ci_lower: 1.0,
                ci_upper: 1.0,
            };
        }

        let ratio = current_mean / baseline_mean;
        let change_pct = (ratio - 1.0) * 100.0;

        // Cohen's d effect size
        let cohens_d = compute_cohens_d(baseline, current);

        // Bootstrap CI for the ratio of means
        let (ci_lower, ci_upper) = self.bootstrap_ratio_ci(baseline, current);

        // p-value: fraction of bootstrap samples where ratio crosses 1.0
        let p_value = self.bootstrap_p_value(baseline, current);

        // Determine verdict
        let verdict = if ci_lower > 1.0 + self.threshold {
            // Entire CI above 1+threshold: regression (slower)
            if !self.require_large_effect || cohens_d.abs() >= 0.8 {
                Verdict::Regression
            } else {
                Verdict::NoChange
            }
        } else if ci_upper < 1.0 - self.threshold {
            // Entire CI below 1-threshold: improvement (faster)
            if !self.require_large_effect || cohens_d.abs() >= 0.8 {
                Verdict::Improvement
            } else {
                Verdict::NoChange
            }
        } else {
            Verdict::NoChange
        };

        RegressionResult {
            verdict,
            change_pct,
            p_value,
            effect_size_cohens_d: cohens_d,
            ci_lower,
            ci_upper,
        }
    }

    /// Bootstrap confidence interval for the ratio of means.
    fn bootstrap_ratio_ci(&self, baseline: &[f64], current: &[f64]) -> (f64, f64) {
        let mut ratios = Vec::with_capacity(self.bootstrap_iterations);
        let alpha = 1.0 - self.confidence;

        // Simple LCG PRNG for reproducibility (no external dependency)
        let mut rng_state: u64 = 42;
        let lcg_next = |state: &mut u64| -> usize {
            *state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            (*state >> 33) as usize
        };

        for _ in 0..self.bootstrap_iterations {
            let b_mean = bootstrap_mean(baseline, &mut rng_state, &lcg_next);
            let c_mean = bootstrap_mean(current, &mut rng_state, &lcg_next);
            if b_mean > 0.0 {
                ratios.push(c_mean / b_mean);
            }
        }

        ratios.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        if ratios.is_empty() {
            return (1.0, 1.0);
        }

        let lower_idx = ((alpha / 2.0) * ratios.len() as f64) as usize;
        let upper_idx = ((1.0 - alpha / 2.0) * ratios.len() as f64) as usize;

        let lower = ratios[lower_idx.min(ratios.len() - 1)];
        let upper = ratios[upper_idx.min(ratios.len() - 1)];
        (lower, upper)
    }

    /// Bootstrap p-value: fraction of bootstrap resamples where the null hypothesis
    /// (no difference) would produce the observed ratio or more extreme.
    fn bootstrap_p_value(&self, baseline: &[f64], current: &[f64]) -> f64 {
        let observed_ratio = mean(current) / mean(baseline).max(f64::EPSILON);

        // Pool samples under null hypothesis
        let mut pooled = Vec::with_capacity(baseline.len() + current.len());
        pooled.extend_from_slice(baseline);
        pooled.extend_from_slice(current);

        let mut rng_state: u64 = 123;
        let lcg_next = |state: &mut u64| -> usize {
            *state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            (*state >> 33) as usize
        };

        let mut extreme_count = 0;
        for _ in 0..self.bootstrap_iterations {
            let b_mean =
                bootstrap_mean_from_pool(&pooled, baseline.len(), &mut rng_state, &lcg_next);
            let c_mean =
                bootstrap_mean_from_pool(&pooled, current.len(), &mut rng_state, &lcg_next);
            if b_mean > 0.0 {
                let null_ratio = c_mean / b_mean;
                if (null_ratio - 1.0).abs() >= (observed_ratio - 1.0).abs() {
                    extreme_count += 1;
                }
            }
        }

        extreme_count as f64 / self.bootstrap_iterations as f64
    }
}

fn mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    data.iter().sum::<f64>() / data.len() as f64
}

fn variance(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }
    let m = mean(data);
    data.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (data.len() - 1) as f64
}

/// Cohen's d = (mean1 - mean2) / pooled_std_dev
fn compute_cohens_d(baseline: &[f64], current: &[f64]) -> f64 {
    let m1 = mean(baseline);
    let m2 = mean(current);
    let v1 = variance(baseline);
    let v2 = variance(current);
    let n1 = baseline.len() as f64;
    let n2 = current.len() as f64;

    // Pooled standard deviation
    let pooled_var = ((n1 - 1.0) * v1 + (n2 - 1.0) * v2) / (n1 + n2 - 2.0);
    let pooled_sd = pooled_var.sqrt();

    if pooled_sd == 0.0 {
        return 0.0;
    }

    (m2 - m1) / pooled_sd
}

/// Resample with replacement and compute mean.
fn bootstrap_mean(data: &[f64], rng_state: &mut u64, lcg_next: &dyn Fn(&mut u64) -> usize) -> f64 {
    let n = data.len();
    let mut sum = 0.0;
    for _ in 0..n {
        let idx = lcg_next(rng_state) % n;
        sum += data[idx];
    }
    sum / n as f64
}

/// Resample from a pooled set with replacement.
fn bootstrap_mean_from_pool(
    pool: &[f64],
    sample_size: usize,
    rng_state: &mut u64,
    lcg_next: &dyn Fn(&mut u64) -> usize,
) -> f64 {
    let n = pool.len();
    let mut sum = 0.0;
    for _ in 0..sample_size {
        let idx = lcg_next(rng_state) % n;
        sum += pool[idx];
    }
    sum / sample_size as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FALSIFY-CGP-030: Must detect deliberate 10% regression.
    #[test]
    fn test_detect_10pct_regression() {
        let detector = RegressionDetector {
            min_samples: 10,
            bootstrap_iterations: 5_000,
            require_large_effect: false,
            ..Default::default()
        };

        // Baseline: mean 100us
        let baseline: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64 * 0.1) - 2.5).collect();
        // Current: mean 112us (12% slower)
        let current: Vec<f64> = (0..50).map(|i| 112.0 + (i as f64 * 0.1) - 2.5).collect();

        let result = detector.compare(&baseline, &current);
        assert_eq!(result.verdict, Verdict::Regression);
        assert!(result.change_pct > 10.0);
    }

    /// FALSIFY-CGP-031: Must NOT false-positive on noise (<2% variation).
    #[test]
    fn test_no_false_positive_on_noise() {
        let detector = RegressionDetector {
            min_samples: 10,
            bootstrap_iterations: 5_000,
            ..Default::default()
        };

        // Both same distribution with slight noise
        let baseline: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64 % 3.0) - 1.0).collect();
        let current: Vec<f64> = (0..50).map(|i| 100.5 + (i as f64 % 3.0) - 1.0).collect();

        let result = detector.compare(&baseline, &current);
        assert_eq!(result.verdict, Verdict::NoChange);
    }

    /// FALSIFY-CGP-032: Must detect improvement.
    #[test]
    fn test_detect_improvement() {
        let detector = RegressionDetector {
            min_samples: 10,
            bootstrap_iterations: 5_000,
            require_large_effect: false,
            ..Default::default()
        };

        // Baseline: 35.7us
        let baseline: Vec<f64> = (0..50).map(|i| 35.7 + (i as f64 * 0.01) - 0.25).collect();
        // Current: 23.2us (35% faster)
        let current: Vec<f64> = (0..50).map(|i| 23.2 + (i as f64 * 0.01) - 0.25).collect();

        let result = detector.compare(&baseline, &current);
        assert_eq!(result.verdict, Verdict::Improvement);
        assert!(result.change_pct < -30.0);
    }

    #[test]
    fn test_cohens_d_large_effect() {
        let baseline: Vec<f64> = vec![10.0; 30];
        let current: Vec<f64> = vec![15.0; 30];
        let _d = compute_cohens_d(&baseline, &current);
        // With zero variance in each group, pooled_sd = 0 -> d = 0
        // Use slight variation
        let baseline: Vec<f64> = (0..30).map(|i| 10.0 + (i as f64 * 0.1)).collect();
        let current: Vec<f64> = (0..30).map(|i| 15.0 + (i as f64 * 0.1)).collect();
        let d = compute_cohens_d(&baseline, &current);
        assert!(d.abs() > 0.8, "Cohen's d = {d:.2} should be large effect");
    }

    #[test]
    fn test_empty_samples() {
        let detector = RegressionDetector::new();
        let result = detector.compare(&[], &[1.0, 2.0]);
        assert_eq!(result.verdict, Verdict::NoChange);
    }
}

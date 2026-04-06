//! Profile comparator with Welch's t-test and statistical analysis (PMAT-045)
//!
//! Contains `CompareConfig` and `ProfileComparator` with all statistical
//! methods: Welch's t-test, Cohen's d, confidence intervals, and
//! distribution approximations.

use crate::profile_compare::types::{
    BenchmarkProfile, ChangeDirection, CompareError, CompareResult, ComparisonVerdict,
    EffectMagnitude, EffectSizeResult, MetricComparison, MetricSamples, ProfileComparison,
    WelchTestResult,
};

/// Configuration for profile comparison
#[derive(Debug, Clone)]
pub struct CompareConfig {
    /// Confidence level (default: 0.95)
    pub confidence_level: f64,
    /// Minimum samples required for comparison
    pub min_samples: usize,
    /// Apply Bonferroni correction for multiple comparisons
    pub bonferroni_correction: bool,
    /// Threshold for regression (percent change)
    pub regression_threshold_percent: f64,
    /// Metrics where higher is better (throughput-like)
    pub higher_is_better: Vec<String>,
    /// Metrics where lower is better (latency-like, default)
    pub lower_is_better: Vec<String>,
}

impl Default for CompareConfig {
    fn default() -> Self {
        Self {
            confidence_level: 0.95,
            min_samples: 5,
            bonferroni_correction: true,
            regression_threshold_percent: 5.0,
            higher_is_better: vec![
                "throughput".to_string(),
                "ops_per_sec".to_string(),
                "requests_per_sec".to_string(),
            ],
            lower_is_better: vec![
                "latency".to_string(),
                "latency_p50".to_string(),
                "latency_p99".to_string(),
                "memory".to_string(),
                "cpu_usage".to_string(),
            ],
        }
    }
}

/// Profile comparator
#[derive(Debug)]
pub struct ProfileComparator {
    /// Comparison configuration
    config: CompareConfig,
}

impl ProfileComparator {
    /// Create a new profile comparator
    pub fn new(config: CompareConfig) -> Self {
        Self { config }
    }

    /// Compare two benchmark profiles
    pub fn compare(
        &self,
        baseline: &BenchmarkProfile,
        comparison: &BenchmarkProfile,
    ) -> CompareResult<ProfileComparison> {
        // Find common metrics
        let common_metrics: Vec<String> = baseline
            .metric_names()
            .filter(|name| comparison.metrics.contains_key(*name))
            .cloned()
            .collect();

        if common_metrics.is_empty() {
            return Err(CompareError::NoCommonMetrics);
        }

        // Calculate corrected alpha for multiple comparisons
        let alpha = 1.0 - self.config.confidence_level;
        let corrected_alpha = if self.config.bonferroni_correction {
            alpha / common_metrics.len() as f64
        } else {
            alpha
        };

        let mut metric_comparisons = Vec::new();
        let mut regressions = Vec::new();
        let mut improvements = Vec::new();

        for metric_name in &common_metrics {
            let baseline_samples =
                baseline.get_metric(metric_name).expect("metric should exist in baseline");
            let comparison_samples =
                comparison.get_metric(metric_name).expect("metric should exist in comparison");

            // Check minimum samples
            if baseline_samples.count() < self.config.min_samples {
                continue;
            }
            if comparison_samples.count() < self.config.min_samples {
                continue;
            }

            let comparison_result = self.compare_metric(
                metric_name,
                baseline_samples,
                comparison_samples,
                corrected_alpha,
            )?;

            if comparison_result.is_regression {
                regressions.push(metric_name.clone());
            } else if comparison_result.t_test.significant
                && comparison_result.direction == ChangeDirection::Improved
            {
                improvements.push(metric_name.clone());
            }

            metric_comparisons.push(comparison_result);
        }

        // Determine verdict
        let verdict = if regressions.is_empty() {
            ComparisonVerdict::Pass
        } else {
            // Check if any regression exceeds threshold
            let severe_regression =
                metric_comparisons.iter().filter(|m| m.is_regression).any(|m| {
                    m.effect_size.percent_change.abs() >= self.config.regression_threshold_percent
                });

            if severe_regression {
                ComparisonVerdict::Fail
            } else {
                ComparisonVerdict::Warning
            }
        };

        Ok(ProfileComparison {
            baseline_name: baseline.name.clone(),
            comparison_name: comparison.name.clone(),
            metrics: metric_comparisons,
            regressions,
            improvements,
            verdict,
            corrected_alpha,
        })
    }

    /// Compare a single metric between two profiles
    pub(crate) fn compare_metric(
        &self,
        name: &str,
        baseline: &MetricSamples,
        comparison: &MetricSamples,
        alpha: f64,
    ) -> CompareResult<MetricComparison> {
        // Check for zero variance
        if baseline.variance() == 0.0 && comparison.variance() == 0.0 {
            return Err(CompareError::ZeroVariance { metric: name.to_string() });
        }

        // Perform Welch's t-test
        let t_test = self.welch_t_test(baseline, comparison, alpha)?;

        // Calculate effect size (Cohen's d with pooled std)
        let pooled_std = self.pooled_std(baseline, comparison);
        let cohens_d =
            if pooled_std > 0.0 { (comparison.mean() - baseline.mean()) / pooled_std } else { 0.0 };

        let percent_change = if baseline.mean().abs() > 1e-10 {
            ((comparison.mean() - baseline.mean()) / baseline.mean()) * 100.0
        } else {
            0.0
        };

        let effect_size = EffectSizeResult {
            cohens_d,
            magnitude: EffectMagnitude::from_cohens_d(cohens_d),
            percent_change,
        };

        // Determine direction and regression
        let higher_is_better = self.config.higher_is_better.iter().any(|m| name.contains(m));

        let direction = if !t_test.significant {
            ChangeDirection::NoChange
        } else if higher_is_better {
            if comparison.mean() > baseline.mean() {
                ChangeDirection::Improved
            } else {
                ChangeDirection::Regressed
            }
        } else {
            // Lower is better (latency-like)
            if comparison.mean() < baseline.mean() {
                ChangeDirection::Improved
            } else {
                ChangeDirection::Regressed
            }
        };

        let is_regression = direction == ChangeDirection::Regressed;

        // Calculate confidence interval for the difference
        let (ci_lower, ci_upper) = self.confidence_interval(baseline, comparison, alpha);

        Ok(MetricComparison {
            name: name.to_string(),
            baseline_mean: baseline.mean(),
            baseline_std: baseline.std_dev(),
            comparison_mean: comparison.mean(),
            comparison_std: comparison.std_dev(),
            t_test,
            effect_size,
            direction,
            is_regression,
            ci_lower,
            ci_upper,
        })
    }

    /// Perform Welch's t-test
    fn welch_t_test(
        &self,
        a: &MetricSamples,
        b: &MetricSamples,
        alpha: f64,
    ) -> CompareResult<WelchTestResult> {
        let n1 = a.count() as f64;
        let n2 = b.count() as f64;
        let var1 = a.variance();
        let var2 = b.variance();

        // Calculate t-statistic
        let mean_diff = b.mean() - a.mean();
        let se = ((var1 / n1) + (var2 / n2)).sqrt();

        let t_statistic = if se > 1e-10 { mean_diff / se } else { 0.0 };

        // Welch-Satterthwaite degrees of freedom
        let v1 = var1 / n1;
        let v2 = var2 / n2;
        let df = if (v1 + v2) > 1e-10 {
            (v1 + v2).powi(2) / (v1.powi(2) / (n1 - 1.0) + v2.powi(2) / (n2 - 1.0))
        } else {
            n1 + n2 - 2.0
        };

        // Approximate p-value using t-distribution CDF
        let p_value = self.t_distribution_p_value(t_statistic.abs(), df);

        let significant = p_value < alpha;

        Ok(WelchTestResult {
            t_statistic,
            degrees_of_freedom: df,
            p_value,
            significant,
            confidence_level: 1.0 - alpha,
        })
    }

    /// Approximate p-value from t-distribution using approximation
    fn t_distribution_p_value(&self, t: f64, df: f64) -> f64 {
        // Use approximation for two-tailed p-value
        // Based on Hill's approximation for Student's t-distribution

        if df <= 0.0 {
            return 1.0;
        }

        let x = df / (df + t * t);

        // Incomplete beta function approximation
        // For large df, use normal approximation
        if df > 100.0 {
            // Normal approximation
            let z = t;
            2.0 * (1.0 - self.normal_cdf(z.abs()))
        } else {
            // Beta approximation
            let a = df / 2.0;
            let b = 0.5;
            self.incomplete_beta(x, a, b)
        }
    }

    /// Normal CDF approximation
    fn normal_cdf(&self, x: f64) -> f64 {
        // Abramowitz and Stegun approximation
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let p = 0.3275911;

        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();

        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x / 2.0).exp();

        0.5 * (1.0 + sign * y)
    }

    /// Incomplete beta function approximation
    fn incomplete_beta(&self, x: f64, a: f64, b: f64) -> f64 {
        // Simple continued fraction approximation
        if !(0.0..=1.0).contains(&x) {
            return 0.0;
        }
        if x == 0.0 {
            return 0.0;
        }
        if x == 1.0 {
            return 1.0;
        }

        // Use continued fraction
        let bt = if x == 0.0 || x == 1.0 {
            0.0
        } else {
            (self.ln_gamma(a + b) - self.ln_gamma(a) - self.ln_gamma(b)
                + a * x.ln()
                + b * (1.0 - x).ln())
            .exp()
        };

        if x < (a + 1.0) / (a + b + 2.0) {
            bt * self.beta_cf(x, a, b) / a
        } else {
            1.0 - bt * self.beta_cf(1.0 - x, b, a) / b
        }
    }

    /// Beta continued fraction
    fn beta_cf(&self, x: f64, a: f64, b: f64) -> f64 {
        let max_iter = 100;
        let eps: f64 = 1e-10;

        let mut c: f64 = 1.0;
        let mut d: f64 = 1.0 / (1.0 - (a + b) * x / (a + 1.0)).max(eps);
        let mut h: f64 = d;

        for m in 1..=max_iter {
            let m = m as f64;

            // Even step
            let aa = m * (b - m) * x / ((a + 2.0 * m - 1.0) * (a + 2.0 * m));
            d = 1.0 / (1.0 + aa * d).max(eps);
            c = 1.0 + aa / c.max(eps);
            h *= d * c;

            // Odd step
            let aa = -(a + m) * (a + b + m) * x / ((a + 2.0 * m) * (a + 2.0 * m + 1.0));
            d = 1.0 / (1.0 + aa * d).max(eps);
            c = 1.0 + aa / c.max(eps);
            let del = d * c;
            h *= del;

            if (del - 1.0).abs() < eps {
                break;
            }
        }

        h
    }

    /// Log gamma function approximation (Stirling)
    #[allow(clippy::excessive_precision)]
    fn ln_gamma(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return f64::INFINITY;
        }

        // Lanczos approximation
        let g = 7;
        let c = [
            0.99999999999980993,
            676.5203681218851,
            -1259.1392167224028,
            771.32342877765313,
            -176.61502916214059,
            12.507343278686905,
            -0.13857109526572012,
            9.9843695780195716e-6,
            1.5056327351493116e-7,
        ];

        if x < 0.5 {
            std::f64::consts::PI.ln()
                - (std::f64::consts::PI * x).sin().ln()
                - self.ln_gamma(1.0 - x)
        } else {
            let x = x - 1.0;
            let mut a = c[0];
            for i in 1..=g {
                a += c[i] / (x + i as f64);
            }
            let t = x + g as f64 + 0.5;
            0.5 * (2.0 * std::f64::consts::PI).ln() + (t - 0.5) * t.ln() - t + a.ln()
        }
    }

    /// Calculate pooled standard deviation
    fn pooled_std(&self, a: &MetricSamples, b: &MetricSamples) -> f64 {
        let n1 = a.count() as f64;
        let n2 = b.count() as f64;

        if n1 + n2 <= 2.0 {
            return 0.0;
        }

        let pooled_var = ((n1 - 1.0) * a.variance() + (n2 - 1.0) * b.variance()) / (n1 + n2 - 2.0);

        pooled_var.sqrt()
    }

    /// Calculate confidence interval for the difference in means
    fn confidence_interval(&self, a: &MetricSamples, b: &MetricSamples, alpha: f64) -> (f64, f64) {
        let mean_diff = b.mean() - a.mean();
        let se = ((a.variance() / a.count() as f64) + (b.variance() / b.count() as f64)).sqrt();

        // Use normal approximation for large samples
        let z = self.normal_quantile(1.0 - alpha / 2.0);
        let margin = z * se;

        (mean_diff - margin, mean_diff + margin)
    }

    /// Normal quantile function (inverse CDF) approximation
    pub(crate) fn normal_quantile(&self, p: f64) -> f64 {
        // Rational approximation from Abramowitz and Stegun
        if p <= 0.0 {
            return f64::NEG_INFINITY;
        }
        if p >= 1.0 {
            return f64::INFINITY;
        }
        if p == 0.5 {
            return 0.0;
        }

        let t = if p < 0.5 { (-2.0 * p.ln()).sqrt() } else { (-2.0 * (1.0 - p).ln()).sqrt() };

        let c0 = 2.515517;
        let c1 = 0.802853;
        let c2 = 0.010328;
        let d1 = 1.432788;
        let d2 = 0.189269;
        let d3 = 0.001308;

        let x = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);

        if p < 0.5 {
            -x
        } else {
            x
        }
    }

    /// Get configuration
    pub fn config(&self) -> &CompareConfig {
        &self.config
    }
}

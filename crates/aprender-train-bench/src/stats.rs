//! Statistical analysis utilities.

/// Statistical analyzer for benchmark results.
pub struct StatisticalAnalyzer;

impl StatisticalAnalyzer {
    /// Perform Welch's t-test for two independent samples.
    ///
    /// Returns the p-value for the null hypothesis that the means are equal.
    pub fn welch_t_test(sample1: &[f64], sample2: &[f64]) -> TestResult {
        if sample1.len() < 2 || sample2.len() < 2 {
            return TestResult {
                statistic: 0.0,
                p_value: 1.0,
                significant: false,
                effect_size: 0.0,
            };
        }

        let n1 = sample1.len() as f64;
        let n2 = sample2.len() as f64;

        let mean1 = sample1.iter().sum::<f64>() / n1;
        let mean2 = sample2.iter().sum::<f64>() / n2;

        let var1 = sample1.iter().map(|x| (x - mean1).powi(2)).sum::<f64>() / (n1 - 1.0);
        let var2 = sample2.iter().map(|x| (x - mean2).powi(2)).sum::<f64>() / (n2 - 1.0);

        let se = ((var1 / n1) + (var2 / n2)).sqrt();
        if se == 0.0 {
            return TestResult {
                statistic: 0.0,
                p_value: 1.0,
                significant: false,
                effect_size: 0.0,
            };
        }

        let t = (mean1 - mean2) / se;

        // Welch-Satterthwaite degrees of freedom
        let df_num = ((var1 / n1) + (var2 / n2)).powi(2);
        let df_denom = ((var1 / n1).powi(2) / (n1 - 1.0)) + ((var2 / n2).powi(2) / (n2 - 1.0));
        let df = df_num / df_denom;

        // Approximate p-value using normal distribution for large df
        let p_value = Self::t_to_p(t.abs(), df);

        // Cohen's d effect size
        let pooled_std = f64::midpoint(var1, var2).sqrt();
        let effect_size = if pooled_std > 0.0 {
            (mean1 - mean2).abs() / pooled_std
        } else {
            0.0
        };

        TestResult {
            statistic: t,
            p_value,
            significant: p_value < 0.05,
            effect_size,
        }
    }

    /// Perform Mann-Whitney U test (non-parametric).
    pub fn mann_whitney_u(sample1: &[f64], sample2: &[f64]) -> TestResult {
        if sample1.is_empty() || sample2.is_empty() {
            return TestResult {
                statistic: 0.0,
                p_value: 1.0,
                significant: false,
                effect_size: 0.0,
            };
        }

        let n1 = sample1.len();
        let n2 = sample2.len();

        // Count ranks
        let mut u = 0.0;
        for &x in sample1 {
            for &y in sample2 {
                if x > y {
                    u += 1.0;
                } else if (x - y).abs() < 1e-10 {
                    u += 0.5;
                }
            }
        }

        // Normal approximation for large samples
        let mu = (n1 * n2) as f64 / 2.0;
        let sigma = ((n1 * n2 * (n1 + n2 + 1)) as f64 / 12.0).sqrt();

        let z = if sigma > 0.0 { (u - mu) / sigma } else { 0.0 };
        let p_value = 2.0 * Self::normal_cdf(-z.abs());

        // Effect size: r = z / sqrt(n)
        let effect_size = z.abs() / ((n1 + n2) as f64).sqrt();

        TestResult {
            statistic: u,
            p_value,
            significant: p_value < 0.05,
            effect_size,
        }
    }

    /// Perform one-way ANOVA.
    pub fn anova(groups: &[Vec<f64>]) -> TestResult {
        if groups.len() < 2 {
            return TestResult {
                statistic: 0.0,
                p_value: 1.0,
                significant: false,
                effect_size: 0.0,
            };
        }

        let k = groups.len() as f64;
        let n_total: usize = groups.iter().map(Vec::len).sum();

        // Grand mean
        let grand_mean: f64 = groups.iter().flat_map(|g| g.iter()).sum::<f64>() / n_total as f64;

        // Between-group sum of squares
        let ss_between: f64 = groups
            .iter()
            .map(|g| {
                let group_mean = g.iter().sum::<f64>() / g.len() as f64;
                g.len() as f64 * (group_mean - grand_mean).powi(2)
            })
            .sum();

        // Within-group sum of squares
        let ss_within: f64 = groups
            .iter()
            .map(|g| {
                let group_mean = g.iter().sum::<f64>() / g.len() as f64;
                g.iter().map(|x| (x - group_mean).powi(2)).sum::<f64>()
            })
            .sum();

        let df_between = k - 1.0;
        let df_within = n_total as f64 - k;

        let ms_between = ss_between / df_between;
        let ms_within = ss_within / df_within;

        let f = if ms_within > 0.0 {
            ms_between / ms_within
        } else {
            0.0
        };

        // Approximate p-value
        let p_value = Self::f_to_p(f, df_between, df_within);

        // Eta-squared effect size
        let effect_size = ss_between / (ss_between + ss_within);

        TestResult {
            statistic: f,
            p_value,
            significant: p_value < 0.05,
            effect_size,
        }
    }

    // Approximate t-distribution CDF using normal approximation
    fn t_to_p(t: f64, df: f64) -> f64 {
        // For large df, t-distribution approaches normal
        if df > 30.0 {
            2.0 * Self::normal_cdf(-t)
        } else {
            // Simple approximation for smaller df
            let adjusted_t = t * (1.0 + 1.0 / (4.0 * df));
            2.0 * Self::normal_cdf(-adjusted_t)
        }
    }

    // Approximate F-distribution p-value
    fn f_to_p(f: f64, df1: f64, _df2: f64) -> f64 {
        // Very rough approximation using chi-square
        let chi_approx = f * df1;
        Self::chi_square_p(chi_approx, df1)
    }

    // Approximate chi-square p-value
    fn chi_square_p(x: f64, df: f64) -> f64 {
        // Use normal approximation for large df
        if df > 30.0 {
            let z = (2.0 * x).sqrt() - (2.0 * df - 1.0).sqrt();
            Self::normal_cdf(-z)
        } else {
            // Rough approximation
            let mean = df;
            let std = (2.0 * df).sqrt();
            Self::normal_cdf(-(x - mean) / std)
        }
    }

    // Standard normal CDF approximation
    fn normal_cdf(x: f64) -> f64 {
        // Approximation using error function
        0.5 * (1.0 + Self::erf(x / std::f64::consts::SQRT_2))
    }

    // Error function approximation
    fn erf(x: f64) -> f64 {
        // Approximation from Abramowitz and Stegun
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let p = 0.3275911;

        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();

        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

        sign * y
    }
}

/// Result of a statistical test.
#[derive(Debug, Clone)]
pub struct TestResult {
    /// Test statistic (t, U, F, etc.)
    pub statistic: f64,
    /// P-value
    pub p_value: f64,
    /// Whether result is significant at α=0.05
    pub significant: bool,
    /// Effect size (Cohen's d, eta-squared, etc.)
    pub effect_size: f64,
}

impl TestResult {
    /// Interpret effect size (Cohen's d).
    pub fn effect_interpretation(&self) -> &'static str {
        if self.effect_size < 0.2 {
            "negligible"
        } else if self.effect_size < 0.5 {
            "small"
        } else if self.effect_size < 0.8 {
            "medium"
        } else {
            "large"
        }
    }
}

impl std::fmt::Display for TestResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "statistic={:.3}, p={:.4}, significant={}, effect_size={:.3} ({})",
            self.statistic,
            self.p_value,
            self.significant,
            self.effect_size,
            self.effect_interpretation()
        )
    }
}

/// Calculate confidence interval for a mean.
pub fn confidence_interval(sample: &[f64], confidence: f64) -> (f64, f64) {
    if sample.len() < 2 {
        let mean = sample.first().copied().unwrap_or(0.0);
        return (mean, mean);
    }

    let n = sample.len() as f64;
    let mean = sample.iter().sum::<f64>() / n;
    let std = (sample.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
    let se = std / n.sqrt();

    // Z-score for confidence level (approximate)
    let z = match confidence {
        c if c >= 0.99 => 2.576,
        c if c >= 0.95 => 1.96,
        c if c >= 0.90 => 1.645,
        _ => 1.96,
    };

    (mean - z * se, mean + z * se)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_welch_t_test_significant() {
        let sample1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sample2 = vec![6.0, 7.0, 8.0, 9.0, 10.0];

        let result = StatisticalAnalyzer::welch_t_test(&sample1, &sample2);
        assert!(result.significant);
        assert!(result.p_value < 0.05);
    }

    #[test]
    fn test_welch_t_test_not_significant() {
        let sample1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sample2 = vec![1.5, 2.5, 3.5, 4.5, 5.5];

        let result = StatisticalAnalyzer::welch_t_test(&sample1, &sample2);
        // Means are close, should not be significant
        assert!(result.p_value > 0.01);
    }

    #[test]
    fn test_mann_whitney() {
        let sample1 = vec![1.0, 2.0, 3.0];
        let sample2 = vec![4.0, 5.0, 6.0];

        let result = StatisticalAnalyzer::mann_whitney_u(&sample1, &sample2);
        assert!(result.statistic >= 0.0);
    }

    #[test]
    fn test_anova() {
        let groups = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];

        let result = StatisticalAnalyzer::anova(&groups);
        assert!(result.significant);
    }

    #[test]
    fn test_confidence_interval() {
        let sample = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (lower, upper) = confidence_interval(&sample, 0.95);

        assert!(lower < 3.0); // Mean is 3.0
        assert!(upper > 3.0);
    }

    #[test]
    fn test_effect_interpretation() {
        let small = TestResult {
            statistic: 0.0,
            p_value: 0.0,
            significant: false,
            effect_size: 0.3,
        };
        assert_eq!(small.effect_interpretation(), "small");

        let large = TestResult {
            statistic: 0.0,
            p_value: 0.0,
            significant: false,
            effect_size: 1.0,
        };
        assert_eq!(large.effect_interpretation(), "large");
    }

    #[test]
    fn test_effect_interpretation_all_levels() {
        // negligible
        let negligible = TestResult {
            statistic: 0.0,
            p_value: 0.0,
            significant: false,
            effect_size: 0.1,
        };
        assert_eq!(negligible.effect_interpretation(), "negligible");

        // medium
        let medium = TestResult {
            statistic: 0.0,
            p_value: 0.0,
            significant: false,
            effect_size: 0.6,
        };
        assert_eq!(medium.effect_interpretation(), "medium");
    }

    #[test]
    fn test_welch_t_test_small_samples() {
        let sample1 = vec![1.0];
        let sample2 = vec![2.0];

        let result = StatisticalAnalyzer::welch_t_test(&sample1, &sample2);
        assert_eq!(result.p_value, 1.0);
        assert!(!result.significant);
    }

    #[test]
    fn test_mann_whitney_empty_samples() {
        let result = StatisticalAnalyzer::mann_whitney_u(&[], &[1.0, 2.0]);
        assert_eq!(result.p_value, 1.0);
        assert!(!result.significant);
    }

    #[test]
    fn test_mann_whitney_ties() {
        // Sample with ties
        let sample1 = vec![1.0, 2.0, 3.0];
        let sample2 = vec![2.0, 3.0, 4.0]; // 2.0 and 3.0 tie with sample1

        let result = StatisticalAnalyzer::mann_whitney_u(&sample1, &sample2);
        assert!(result.statistic >= 0.0);
    }

    #[test]
    fn test_anova_single_group() {
        let groups = vec![vec![1.0, 2.0, 3.0]];

        let result = StatisticalAnalyzer::anova(&groups);
        assert_eq!(result.p_value, 1.0);
    }

    #[test]
    fn test_anova_identical_groups() {
        let groups = vec![
            vec![5.0, 5.0, 5.0],
            vec![5.0, 5.0, 5.0],
            vec![5.0, 5.0, 5.0],
        ];

        let result = StatisticalAnalyzer::anova(&groups);
        // No variance within or between, F should be 0
        assert!(!result.significant || result.statistic.abs() < 0.001);
    }

    #[test]
    fn test_confidence_interval_single_sample() {
        let sample = vec![5.0];
        let (lower, upper) = confidence_interval(&sample, 0.95);
        assert_eq!(lower, 5.0);
        assert_eq!(upper, 5.0);
    }

    #[test]
    fn test_confidence_interval_99() {
        let sample = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (lower_95, upper_95) = confidence_interval(&sample, 0.95);
        let (lower_99, upper_99) = confidence_interval(&sample, 0.99);

        // 99% CI should be wider than 95% CI
        assert!(lower_99 < lower_95);
        assert!(upper_99 > upper_95);
    }

    #[test]
    fn test_confidence_interval_90() {
        let sample = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (lower_95, upper_95) = confidence_interval(&sample, 0.95);
        let (lower_90, upper_90) = confidence_interval(&sample, 0.90);

        // 90% CI should be narrower than 95% CI
        assert!(lower_90 > lower_95);
        assert!(upper_90 < upper_95);
    }

    #[test]
    fn test_test_result_to_string() {
        let result = TestResult {
            statistic: 2.5,
            p_value: 0.025,
            significant: true,
            effect_size: 0.8,
        };

        let s = result.to_string();
        assert!(s.contains("statistic=2.5"));
        assert!(s.contains("p=0.0250"));
        assert!(s.contains("significant=true"));
        assert!(s.contains("large"));
    }

    #[test]
    fn test_welch_t_test_zero_variance() {
        // Samples with zero variance
        let sample1 = vec![5.0, 5.0, 5.0, 5.0, 5.0];
        let sample2 = vec![10.0, 10.0, 10.0, 10.0, 10.0];

        let result = StatisticalAnalyzer::welch_t_test(&sample1, &sample2);
        // Should return p=1 due to zero SE
        assert_eq!(result.p_value, 1.0);
    }

    #[test]
    fn test_normal_cdf_properties() {
        // Normal CDF at 0 should be 0.5
        let cdf_0 = StatisticalAnalyzer::normal_cdf(0.0);
        assert!((cdf_0 - 0.5).abs() < 0.01);

        // CDF at large positive should approach 1
        let cdf_large = StatisticalAnalyzer::normal_cdf(3.0);
        assert!(cdf_large > 0.99);

        // CDF at large negative should approach 0
        let cdf_neg = StatisticalAnalyzer::normal_cdf(-3.0);
        assert!(cdf_neg < 0.01);
    }
}

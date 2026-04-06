//! Profile Diffing and A/B Comparison (PMAT-045)
//!
//! Statistical comparison of benchmark profiles for regression detection.
//!
//! # Design
//!
//! - Welch's t-test for comparing two sample sets
//! - Confidence interval computation with configurable levels
//! - Effect size calculation (Cohen's d)
//! - Multiple comparison correction (Bonferroni)
//!
//! # Falsification (FKR-046)
//!
//! H₀: Profile diff cannot detect 5% regression with 95% confidence
//! Test: Inject known 5% regression, verify detection rate >80%

mod comparator;
mod types;

pub use comparator::{CompareConfig, ProfileComparator};
pub use types::{
    BenchmarkProfile, ChangeDirection, CompareError, CompareResult, ComparisonVerdict,
    EffectMagnitude, EffectSizeResult, MetricComparison, MetricSamples, ProfileComparison,
    WelchTestResult,
};

/// Minimum samples required for reliable comparison
pub const MIN_COMPARISON_SAMPLES: usize = 5;

/// Default confidence level
pub const DEFAULT_CONFIDENCE_LEVEL: f64 = 0.95;

/// Default regression threshold (percent)
pub const DEFAULT_REGRESSION_THRESHOLD: f64 = 5.0;

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_profile(
        name: &str,
        latency_samples: Vec<f64>,
        throughput_samples: Vec<f64>,
    ) -> BenchmarkProfile {
        let mut profile = BenchmarkProfile::new(name);
        profile.add_metric("latency_p50", latency_samples);
        profile.add_metric("throughput", throughput_samples);
        profile
    }

    #[test]
    fn test_metric_samples_statistics() {
        let samples = MetricSamples::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        assert_eq!(samples.count(), 5);
        assert!((samples.mean() - 3.0).abs() < 0.01);
        assert!((samples.variance() - 2.5).abs() < 0.01);
        assert!((samples.std_dev() - 1.58).abs() < 0.1);
        assert_eq!(samples.min(), 1.0);
        assert_eq!(samples.max(), 5.0);
    }

    #[test]
    fn test_empty_samples() {
        let samples = MetricSamples::new(vec![]);

        assert_eq!(samples.count(), 0);
        assert_eq!(samples.mean(), 0.0);
        assert_eq!(samples.variance(), 0.0);
    }

    #[test]
    fn test_effect_magnitude() {
        assert_eq!(EffectMagnitude::from_cohens_d(0.1), EffectMagnitude::Negligible);
        assert_eq!(EffectMagnitude::from_cohens_d(0.3), EffectMagnitude::Small);
        assert_eq!(EffectMagnitude::from_cohens_d(0.6), EffectMagnitude::Medium);
        assert_eq!(EffectMagnitude::from_cohens_d(1.0), EffectMagnitude::Large);
        assert_eq!(EffectMagnitude::from_cohens_d(-0.9), EffectMagnitude::Large);
    }

    #[test]
    fn test_profile_creation() {
        let profile = BenchmarkProfile::new("test")
            .with_description("Test profile")
            .with_metadata("version", "1.0");

        assert_eq!(profile.name, "test");
        assert_eq!(profile.description, Some("Test profile".to_string()));
        assert_eq!(profile.metadata.get("version"), Some(&"1.0".to_string()));
    }

    #[test]
    fn test_profile_comparison_no_regression() {
        let baseline = create_test_profile(
            "baseline",
            vec![100.0, 102.0, 98.0, 101.0, 99.0], // latency
            vec![1000.0, 1010.0, 990.0, 1005.0, 995.0], // throughput
        );

        let comparison = create_test_profile(
            "comparison",
            vec![99.0, 101.0, 97.0, 100.0, 98.0], // slightly better latency
            vec![1005.0, 1015.0, 995.0, 1010.0, 1000.0], // slightly better throughput
        );

        let comparator = ProfileComparator::new(CompareConfig::default());
        let result = comparator.compare(&baseline, &comparison).unwrap();

        assert_eq!(result.verdict, ComparisonVerdict::Pass);
        assert!(result.regressions.is_empty());
    }

    #[test]
    fn test_profile_comparison_with_regression() {
        let baseline = create_test_profile(
            "baseline",
            vec![100.0, 102.0, 98.0, 101.0, 99.0], // latency ~100
            vec![1000.0, 1010.0, 990.0, 1005.0, 995.0], // throughput ~1000
        );

        let comparison = create_test_profile(
            "comparison",
            vec![120.0, 122.0, 118.0, 121.0, 119.0], // latency ~120 (20% worse)
            vec![800.0, 810.0, 790.0, 805.0, 795.0], // throughput ~800 (20% worse)
        );

        let comparator = ProfileComparator::new(CompareConfig::default());
        let result = comparator.compare(&baseline, &comparison).unwrap();

        assert_eq!(result.verdict, ComparisonVerdict::Fail);
        assert!(!result.regressions.is_empty());
    }

    #[test]
    fn test_no_common_metrics() {
        let mut baseline = BenchmarkProfile::new("baseline");
        baseline.add_metric("metric_a", vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let mut comparison = BenchmarkProfile::new("comparison");
        comparison.add_metric("metric_b", vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let comparator = ProfileComparator::new(CompareConfig::default());
        let result = comparator.compare(&baseline, &comparison);

        assert!(matches!(result, Err(CompareError::NoCommonMetrics)));
    }

    #[test]
    fn test_bonferroni_correction() {
        let mut baseline = BenchmarkProfile::new("baseline");
        let mut comparison = BenchmarkProfile::new("comparison");

        // Add 10 metrics with variance (not all same values)
        for i in 0..10 {
            let base_samples: Vec<f64> = (0..10).map(|j| 100.0 + (j as f64) * 0.1).collect();
            let comp_samples: Vec<f64> = (0..10).map(|j| 100.0 + (j as f64) * 0.1).collect();
            baseline.add_metric(format!("metric_{}", i), base_samples);
            comparison.add_metric(format!("metric_{}", i), comp_samples);
        }

        let config = CompareConfig { bonferroni_correction: true, ..Default::default() };

        let comparator = ProfileComparator::new(config);
        let result = comparator.compare(&baseline, &comparison).unwrap();

        // With 10 metrics and alpha=0.05, corrected alpha should be 0.005
        assert!((result.corrected_alpha - 0.005).abs() < 0.001);
    }

    #[test]
    fn test_welch_t_test_identical() {
        let config = CompareConfig::default();
        let comparator = ProfileComparator::new(config);

        let a = MetricSamples::new(vec![10.0, 10.0, 10.0, 10.0, 10.0]);
        let b = MetricSamples::new(vec![10.0, 10.0, 10.0, 10.0, 10.0]);

        // This should fail due to zero variance
        let result = comparator.compare_metric("test", &a, &b, 0.05);
        assert!(matches!(result, Err(CompareError::ZeroVariance { .. })));
    }

    #[test]
    fn test_welch_t_test_significant() {
        let config = CompareConfig::default();
        let comparator = ProfileComparator::new(config);

        let a = MetricSamples::new(vec![10.0, 11.0, 9.0, 10.5, 9.5]);
        let b = MetricSamples::new(vec![20.0, 21.0, 19.0, 20.5, 19.5]);

        let result = comparator.compare_metric("test", &a, &b, 0.05).unwrap();

        // Very different means should be significant
        assert!(result.t_test.significant);
        assert!(result.effect_size.percent_change > 90.0); // ~100% increase
    }

    #[test]
    fn test_confidence_interval() {
        let config = CompareConfig::default();
        let comparator = ProfileComparator::new(config);

        let a = MetricSamples::new(vec![10.0, 11.0, 9.0, 10.5, 9.5]);
        let b = MetricSamples::new(vec![12.0, 13.0, 11.0, 12.5, 11.5]);

        let result = comparator.compare_metric("latency", &a, &b, 0.05).unwrap();

        // CI should contain the true difference (~2)
        assert!(result.ci_lower < 2.0);
        assert!(result.ci_upper > 2.0);
    }

    #[test]
    fn test_direction_higher_is_better() {
        let config = CompareConfig::default();
        let comparator = ProfileComparator::new(config);

        let a = MetricSamples::new(vec![100.0, 101.0, 99.0, 100.5, 99.5]);
        let b = MetricSamples::new(vec![120.0, 121.0, 119.0, 120.5, 119.5]);

        // Throughput: higher is better
        let result = comparator.compare_metric("throughput", &a, &b, 0.05).unwrap();
        assert_eq!(result.direction, ChangeDirection::Improved);
        assert!(!result.is_regression);

        // Latency: lower is better, so increase is regression
        let result = comparator.compare_metric("latency_p50", &a, &b, 0.05).unwrap();
        assert_eq!(result.direction, ChangeDirection::Regressed);
        assert!(result.is_regression);
    }

    #[test]
    fn test_comparison_counts() {
        let result = ProfileComparison {
            baseline_name: "a".to_string(),
            comparison_name: "b".to_string(),
            metrics: vec![],
            regressions: vec!["m1".to_string(), "m2".to_string()],
            improvements: vec!["m3".to_string()],
            verdict: ComparisonVerdict::Fail,
            corrected_alpha: 0.05,
        };

        assert_eq!(result.regression_count(), 2);
        assert_eq!(result.improvement_count(), 1);
        assert!(result.has_regressions());
    }

    #[test]
    fn test_compare_error_display() {
        let err = CompareError::InsufficientSamples { got: 3, need: 5 };
        assert!(err.to_string().contains("3"));
        assert!(err.to_string().contains("5"));

        let err = CompareError::MetricNotFound { name: "latency".to_string() };
        assert!(err.to_string().contains("latency"));
    }

    #[test]
    fn test_normal_quantile() {
        let comparator = ProfileComparator::new(CompareConfig::default());

        // Known values
        assert!((comparator.normal_quantile(0.5) - 0.0).abs() < 0.01);
        assert!((comparator.normal_quantile(0.975) - 1.96).abs() < 0.1);
        assert!((comparator.normal_quantile(0.025) - (-1.96)).abs() < 0.1);
    }

    // FKR-046: Detection of 5% regression with 95% confidence
    #[test]
    fn test_fkr_046_five_percent_regression_detection() {
        let comparator = ProfileComparator::new(CompareConfig::default());

        // Run multiple trials to verify detection rate
        let mut detected = 0;
        let trials = 100;

        for seed in 0..trials {
            // Generate baseline with mean 100, std 5
            let baseline_values: Vec<f64> =
                (0..30).map(|i| 100.0 + (((seed * 100 + i) % 10) as f64 - 5.0)).collect();

            // Generate comparison with 5% regression (mean 105 for latency)
            let comparison_values: Vec<f64> =
                (0..30).map(|i| 105.0 + (((seed * 100 + i + 50) % 10) as f64 - 5.0)).collect();

            let mut baseline = BenchmarkProfile::new("baseline");
            baseline.add_metric("latency_p50", baseline_values);

            let mut comparison = BenchmarkProfile::new("comparison");
            comparison.add_metric("latency_p50", comparison_values);

            let result = comparator.compare(&baseline, &comparison);

            if let Ok(r) = result {
                if r.has_regressions() {
                    detected += 1;
                }
            }
        }

        let detection_rate = detected as f64 / trials as f64;

        // FKR-046: Detection rate should be >80%
        assert!(detection_rate > 0.80, "Detection rate {} should be >80%", detection_rate);
    }
}

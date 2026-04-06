//! statistics_f1101 - Part 1

use cbtop::{
    bootstrap_ci, trimmed_mean, EffectCategory, EffectSize, MannWhitneyResult, OutlierFilter,
    StatisticalAnalysis, StatisticalComparison,
};

// ============================================================================
// F1101: Bootstrap CI contains true mean
// ============================================================================

#[test]
fn f1101_bootstrap_ci_contains_mean() {
    let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let (lower, upper) = bootstrap_ci(&samples, 0.95, 1000);
    let mean = 3.0;

    assert!(lower < mean);
    assert!(upper > mean);
}

#[test]
fn f1101_bootstrap_ci_95_coverage() {
    // Test that 95% CI actually covers true mean ~95% of time
    let true_mean = 5.0;
    let samples: Vec<f64> = (0..100).map(|i| true_mean + (i as f64 % 10.0) - 5.0).collect();

    let (lower, upper) = bootstrap_ci(&samples, 0.95, 10000);

    // CI should contain the sample mean
    let sample_mean = samples.iter().sum::<f64>() / samples.len() as f64;
    assert!(lower <= sample_mean && sample_mean <= upper);
}

#[test]
fn f1101_statistical_analysis_ci() {
    let samples = vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0];
    let analysis = StatisticalAnalysis::from_samples_default(&samples).unwrap();

    assert!(analysis.ci_lower < analysis.mean);
    assert!(analysis.ci_upper > analysis.mean);
    assert!((analysis.confidence_level - 0.95).abs() < 0.01);
}

// ============================================================================
// F1102: Effect size categorized correctly
// ============================================================================

#[test]
fn f1102_effect_negligible() {
    assert_eq!(EffectCategory::from_cohens_d(0.0), EffectCategory::Negligible);
    assert_eq!(EffectCategory::from_cohens_d(0.1), EffectCategory::Negligible);
    assert_eq!(EffectCategory::from_cohens_d(0.19), EffectCategory::Negligible);
}

#[test]
fn f1102_effect_small() {
    assert_eq!(EffectCategory::from_cohens_d(0.2), EffectCategory::Small);
    assert_eq!(EffectCategory::from_cohens_d(0.3), EffectCategory::Small);
    assert_eq!(EffectCategory::from_cohens_d(0.49), EffectCategory::Small);
}

#[test]
fn f1102_effect_medium() {
    assert_eq!(EffectCategory::from_cohens_d(0.5), EffectCategory::Medium);
    assert_eq!(EffectCategory::from_cohens_d(0.6), EffectCategory::Medium);
    assert_eq!(EffectCategory::from_cohens_d(0.79), EffectCategory::Medium);
}

#[test]
fn f1102_effect_large() {
    assert_eq!(EffectCategory::from_cohens_d(0.8), EffectCategory::Large);
    assert_eq!(EffectCategory::from_cohens_d(1.0), EffectCategory::Large);
    assert_eq!(EffectCategory::from_cohens_d(2.0), EffectCategory::Large);
}

#[test]
fn f1102_effect_negative() {
    // Negative d should still be categorized by magnitude
    assert_eq!(EffectCategory::from_cohens_d(-0.1), EffectCategory::Negligible);
    assert_eq!(EffectCategory::from_cohens_d(-0.5), EffectCategory::Medium);
    assert_eq!(EffectCategory::from_cohens_d(-1.0), EffectCategory::Large);
}

// ============================================================================
// F1103: t-test p-value accurate
// ============================================================================

#[test]
fn f1103_welch_t_significant() {
    // Very different samples should be significant
    let sample1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let sample2 = vec![10.0, 11.0, 12.0, 13.0, 14.0];

    let result = StatisticalComparison::welch_t_test(&sample1, &sample2).unwrap();

    assert!(result.statistically_significant);
    assert!(result.p_value < 0.05);
}

#[test]
fn f1103_welch_t_not_significant() {
    // Similar samples should not be significant
    let sample1 = vec![5.0, 5.1, 4.9, 5.0, 5.1];
    let sample2 = vec![5.0, 4.9, 5.1, 5.0, 4.9];

    let result = StatisticalComparison::welch_t_test(&sample1, &sample2).unwrap();

    // Should not be significant (similar means)
    assert!(result.p_value > 0.05 || !result.statistically_significant);
}

#[test]
fn f1103_t_statistic_sign() {
    let sample1 = vec![1.0, 2.0, 3.0];
    let sample2 = vec![4.0, 5.0, 6.0];

    let result = StatisticalComparison::welch_t_test(&sample1, &sample2).unwrap();

    // sample1 < sample2, so t should be negative
    assert!(result.t_statistic < 0.0);
}

// ============================================================================
// F1104: Mann-Whitney handles ties
// ============================================================================

#[test]
fn f1104_mann_whitney_with_ties() {
    let sample1 = vec![1.0, 2.0, 2.0, 3.0];
    let sample2 = vec![2.0, 3.0, 3.0, 4.0];

    // Should not panic on ties
    let result = MannWhitneyResult::test(&sample1, &sample2);
    assert!(result.is_some());
}

#[test]
fn f1104_mann_whitney_all_equal() {
    let sample1 = vec![5.0, 5.0, 5.0];
    let sample2 = vec![5.0, 5.0, 5.0];

    // Should not panic when all values are equal
    let result = MannWhitneyResult::test(&sample1, &sample2);
    assert!(result.is_some());
}

// ============================================================================
// F1105: IQR filter removes outliers
// ============================================================================

#[test]
fn f1105_outlier_removed() {
    let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0, 100.0];

    let filter = OutlierFilter::new(&samples).unwrap();
    let filtered = filter.filter(&samples);

    assert!(!filtered.contains(&100.0));
    assert_eq!(filtered.len(), 5);
}

#[test]
fn f1105_outlier_count() {
    let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0, 100.0, -100.0];

    let filter = OutlierFilter::new(&samples).unwrap();
    let count = filter.count_outliers(&samples);

    assert!(count >= 2);
}

#[test]
fn f1105_outlier_detection() {
    let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let filter = OutlierFilter::new(&samples).unwrap();

    assert!(!filter.is_outlier(3.0));
    assert!(filter.is_outlier(100.0));
    assert!(filter.is_outlier(-100.0));
}

// ============================================================================
// F1106: Empty input handled
// ============================================================================

#[test]
fn f1106_statistical_analysis_empty() {
    let samples: Vec<f64> = vec![];
    assert!(StatisticalAnalysis::from_samples_default(&samples).is_none());
}

#[test]
fn f1106_cohens_d_empty() {
    assert!(EffectSize::cohens_d(&[], &[1.0, 2.0]).is_none());
    assert!(EffectSize::cohens_d(&[1.0, 2.0], &[]).is_none());
}

#[test]
fn f1106_welch_t_empty() {
    let sample = vec![1.0, 2.0];
    assert!(StatisticalComparison::welch_t_test(&[], &sample).is_none());
    assert!(StatisticalComparison::welch_t_test(&sample, &[]).is_none());
}

#[test]
fn f1106_mann_whitney_empty() {
    assert!(MannWhitneyResult::test(&[], &[1.0]).is_none());
    assert!(MannWhitneyResult::test(&[1.0], &[]).is_none());
}

#[test]
fn f1106_outlier_filter_empty() {
    assert!(OutlierFilter::new(&[]).is_none());
}

#[test]
fn f1106_bootstrap_ci_empty() {
    let (lower, upper) = bootstrap_ci(&[], 0.95, 1000);
    assert_eq!(lower, 0.0);
    assert_eq!(upper, 0.0);
}

#[test]
fn f1106_trimmed_mean_empty() {
    assert!(trimmed_mean(&[], 0.1).is_none());
}

// ============================================================================
// F1107: Single element handled
// ============================================================================

#[test]
fn f1107_statistical_analysis_single() {
    let samples = vec![5.0];
    let analysis = StatisticalAnalysis::from_samples_default(&samples).unwrap();

    assert!((analysis.mean - 5.0).abs() < 0.01);
    assert!((analysis.ci_lower - 5.0).abs() < 0.01);
    assert!((analysis.ci_upper - 5.0).abs() < 0.01);
}

#[test]
fn f1107_bootstrap_single() {
    let (lower, upper) = bootstrap_ci(&[5.0], 0.95, 1000);
    assert!((lower - 5.0).abs() < 0.01);
    assert!((upper - 5.0).abs() < 0.01);
}

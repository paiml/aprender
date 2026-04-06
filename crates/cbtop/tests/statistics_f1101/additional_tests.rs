//! statistics_f1101 - Part 2

use cbtop::{
    bootstrap_ci, percentile, trimmed_mean, EffectCategory, EffectSize, OutlierFilter,
    StatisticalAnalysis, StatisticalComparison,
};

// ============================================================================
// F1108: Negative values accepted
// ============================================================================

#[test]
fn f1108_negative_samples() {
    let samples = vec![-5.0, -3.0, -1.0, 1.0, 3.0];
    let analysis = StatisticalAnalysis::from_samples_default(&samples).unwrap();

    assert!((analysis.mean - (-1.0)).abs() < 0.01);
}

#[test]
fn f1108_cohens_d_negative() {
    let sample1 = vec![-5.0, -4.0, -3.0];
    let sample2 = vec![-2.0, -1.0, 0.0];

    let effect = EffectSize::cohens_d(&sample1, &sample2);
    assert!(effect.is_some());
}

// ============================================================================
// F1109: NaN/Inf rejected
// ============================================================================

#[test]
fn f1109_nan_filtered() {
    let samples = vec![1.0, f64::NAN, 3.0, 5.0];
    let analysis = StatisticalAnalysis::from_samples_default(&samples).unwrap();

    assert!(analysis.mean.is_finite());
    assert_eq!(analysis.n, 3); // NaN excluded
}

#[test]
fn f1109_inf_filtered() {
    let samples = vec![1.0, f64::INFINITY, 3.0, 5.0];
    let analysis = StatisticalAnalysis::from_samples_default(&samples).unwrap();

    assert!(analysis.mean.is_finite());
    assert_eq!(analysis.n, 3); // Inf excluded
}

#[test]
fn f1109_all_nan() {
    let samples = vec![f64::NAN, f64::NAN];
    assert!(StatisticalAnalysis::from_samples_default(&samples).is_none());
}

// ============================================================================
// F1110: Large samples efficient
// ============================================================================

#[test]
#[ignore = "Environment-dependent: timing varies on CI runners under load"]
fn f1110_large_sample() {
    let samples: Vec<f64> = (0..10000).map(|i| i as f64).collect();

    // Should complete quickly (not O(n^2))
    let start = std::time::Instant::now();
    let analysis = StatisticalAnalysis::from_samples_default(&samples).unwrap();
    let elapsed = start.elapsed();

    assert!(elapsed.as_millis() < 1000); // Should be fast
    assert!(analysis.mean.is_finite());
}

// ============================================================================
// F1111: Bootstrap iterations configurable
// ============================================================================

#[test]
fn f1111_bootstrap_iterations() {
    let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    // Different iteration counts should work
    let (l1, u1) = bootstrap_ci(&samples, 0.95, 100);
    let (l2, u2) = bootstrap_ci(&samples, 0.95, 10000);

    // Both should produce valid intervals
    assert!(l1 < u1);
    assert!(l2 < u2);
}

// ============================================================================
// F1112: CI width decreases with n
// ============================================================================

#[test]
fn f1112_ci_width_vs_n() {
    let small_samples: Vec<f64> = (0..10).map(|i| i as f64).collect();
    let large_samples: Vec<f64> = (0..100).map(|i| i as f64).collect();

    let small_analysis = StatisticalAnalysis::from_samples_default(&small_samples).unwrap();
    let large_analysis = StatisticalAnalysis::from_samples_default(&large_samples).unwrap();

    // Larger sample should have narrower CI (relative to mean)
    let small_width = small_analysis.ci_width() / small_analysis.mean.abs();
    let large_width = large_analysis.ci_width() / large_analysis.mean.abs();

    assert!(large_width < small_width);
}

// ============================================================================
// F1113: Effect size sign correct
// ============================================================================

#[test]
fn f1113_effect_positive() {
    let sample1 = vec![10.0, 11.0, 12.0];
    let sample2 = vec![1.0, 2.0, 3.0];

    let effect = EffectSize::cohens_d(&sample1, &sample2).unwrap();
    assert!(effect.cohens_d > 0.0); // sample1 > sample2
}

#[test]
fn f1113_effect_negative() {
    let sample1 = vec![1.0, 2.0, 3.0];
    let sample2 = vec![10.0, 11.0, 12.0];

    let effect = EffectSize::cohens_d(&sample1, &sample2).unwrap();
    assert!(effect.cohens_d < 0.0); // sample1 < sample2
}

// ============================================================================
// F1114: Confidence level configurable
// ============================================================================

#[test]
fn f1114_confidence_90() {
    let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let analysis = StatisticalAnalysis::from_samples(&samples, 0.90).unwrap();

    assert!((analysis.confidence_level - 0.90).abs() < 0.01);
}

#[test]
fn f1114_confidence_99() {
    let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let analysis = StatisticalAnalysis::from_samples(&samples, 0.99).unwrap();

    assert!((analysis.confidence_level - 0.99).abs() < 0.01);
}

#[test]
fn f1114_higher_confidence_wider_ci() {
    let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

    let ci_90 = StatisticalAnalysis::from_samples(&samples, 0.90).unwrap();
    let ci_99 = StatisticalAnalysis::from_samples(&samples, 0.99).unwrap();

    // 99% CI should be wider than 90% CI
    assert!(ci_99.ci_width() > ci_90.ci_width());
}

// ============================================================================
// F1115: Thread-safe RNG
// ============================================================================

#[test]
fn f1115_deterministic_bootstrap() {
    let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    // Multiple calls with same input should give consistent results
    let (l1, u1) = bootstrap_ci(&samples, 0.95, 1000);
    let (l2, u2) = bootstrap_ci(&samples, 0.95, 1000);

    // Results should be deterministic (same seed)
    assert!((l1 - l2).abs() < 0.01);
    assert!((u1 - u2).abs() < 0.01);
}

// ============================================================================
// Additional Coverage Tests
// ============================================================================

#[test]
fn test_effect_category_description() {
    assert!(!EffectCategory::Negligible.description().is_empty());
    assert!(!EffectCategory::Small.description().is_empty());
    assert!(!EffectCategory::Medium.description().is_empty());
    assert!(!EffectCategory::Large.description().is_empty());
}

#[test]
fn test_effect_is_significant() {
    let sample1 = vec![1.0, 2.0, 3.0];
    let sample2 = vec![10.0, 11.0, 12.0];

    let effect = EffectSize::cohens_d(&sample1, &sample2).unwrap();
    assert!(effect.is_significant());
}

#[test]
fn test_comparison_is_meaningful() {
    let sample1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let sample2 = vec![10.0, 11.0, 12.0, 13.0, 14.0];

    let result = StatisticalComparison::welch_t_test(&sample1, &sample2).unwrap();
    assert!(result.is_meaningful());
}

#[test]
fn test_outlier_filter_custom_multiplier() {
    let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    // Tighter multiplier (1.0 instead of 1.5)
    let tight = OutlierFilter::with_multiplier(&samples, 1.0).unwrap();
    let normal = OutlierFilter::new(&samples).unwrap();

    // Tighter multiplier should have smaller fences
    assert!(tight.upper_fence < normal.upper_fence);
}

#[test]
fn test_statistical_analysis_cv() {
    let samples = vec![100.0, 100.0, 100.0, 100.0, 100.0];
    let analysis = StatisticalAnalysis::from_samples_default(&samples).unwrap();

    // Constant values should have 0% CV
    assert!(analysis.cv_percent < 0.01);
}

#[test]
fn test_ci_is_narrow() {
    let tight_samples = vec![100.0, 100.1, 99.9, 100.0, 100.1];
    let analysis = StatisticalAnalysis::from_samples_default(&tight_samples).unwrap();

    assert!(analysis.ci_is_narrow());
}

#[test]
fn test_percentile_function() {
    let sorted = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    assert!((percentile(&sorted, 0.0) - 1.0).abs() < 0.01);
    assert!((percentile(&sorted, 0.25) - 2.0).abs() < 0.01);
    assert!((percentile(&sorted, 0.5) - 3.0).abs() < 0.01);
    assert!((percentile(&sorted, 0.75) - 4.0).abs() < 0.01);
    assert!((percentile(&sorted, 1.0) - 5.0).abs() < 0.01);
}

#[test]
fn test_trimmed_mean_removes_extremes() {
    let samples = vec![1.0, 100.0, 200.0, 300.0, 10000.0];

    let normal_mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let trimmed = trimmed_mean(&samples, 0.2).unwrap();

    // Trimmed mean should be less affected by extremes
    assert!(trimmed < normal_mean);
}

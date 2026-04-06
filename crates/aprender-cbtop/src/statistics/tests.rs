use super::*;

#[test]
fn test_effect_category() {
    assert_eq!(
        EffectCategory::from_cohens_d(0.1),
        EffectCategory::Negligible
    );
    assert_eq!(EffectCategory::from_cohens_d(0.3), EffectCategory::Small);
    assert_eq!(EffectCategory::from_cohens_d(0.6), EffectCategory::Medium);
    assert_eq!(EffectCategory::from_cohens_d(1.0), EffectCategory::Large);
    assert_eq!(EffectCategory::from_cohens_d(-0.9), EffectCategory::Large);
}

#[test]
fn test_statistical_analysis() {
    let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let analysis = StatisticalAnalysis::from_samples_default(&samples).unwrap();

    assert!((analysis.mean - 3.0).abs() < 0.01);
    assert!(analysis.ci_lower < analysis.mean);
    assert!(analysis.ci_upper > analysis.mean);
    assert_eq!(analysis.n, 5);
}

#[test]
fn test_statistical_analysis_empty() {
    let samples: Vec<f64> = vec![];
    assert!(StatisticalAnalysis::from_samples_default(&samples).is_none());
}

#[test]
fn test_statistical_analysis_single() {
    let samples = vec![5.0];
    let analysis = StatisticalAnalysis::from_samples_default(&samples).unwrap();

    assert!((analysis.mean - 5.0).abs() < 0.01);
    assert!((analysis.ci_lower - 5.0).abs() < 0.01);
    assert!((analysis.ci_upper - 5.0).abs() < 0.01);
}

#[test]
fn test_cohens_d() {
    let sample1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let sample2 = vec![2.0, 3.0, 4.0, 5.0, 6.0];

    let effect = EffectSize::cohens_d(&sample1, &sample2).unwrap();
    // Difference of 1 unit with std of ~1.58 gives d ≈ -0.63
    assert!(effect.cohens_d < 0.0); // sample1 < sample2
    assert_eq!(effect.category, EffectCategory::Medium);
}

#[test]
fn test_welch_t_test() {
    let sample1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let sample2 = vec![6.0, 7.0, 8.0, 9.0, 10.0];

    let result = ComparisonResult::welch_t_test(&sample1, &sample2).unwrap();
    assert!(result.statistically_significant);
    assert!(result.practically_significant);
}

#[test]
fn test_mann_whitney() {
    let sample1 = vec![1.0, 2.0, 3.0];
    let sample2 = vec![4.0, 5.0, 6.0];

    let result = MannWhitneyResult::test(&sample1, &sample2).unwrap();
    // Completely separated groups
    assert!(result.significant || result.p_value < 0.1);
}

#[test]
fn test_outlier_filter() {
    let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0, 100.0]; // 100 is outlier

    let filter = OutlierFilter::new(&samples).unwrap();
    assert!(filter.is_outlier(100.0));
    assert!(!filter.is_outlier(3.0));

    let filtered = filter.filter(&samples);
    assert_eq!(filtered.len(), 5);
    assert!(!filtered.contains(&100.0));
}

#[test]
fn test_bootstrap_ci() {
    let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let (lower, upper) = bootstrap_ci(&samples, 0.95, 1000);

    assert!(lower < 3.0);
    assert!(upper > 3.0);
    assert!(lower < upper);
}

#[test]
fn test_percentile() {
    let sorted = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    assert!((percentile(&sorted, 0.5) - 3.0).abs() < 0.01);
    assert!((percentile(&sorted, 0.0) - 1.0).abs() < 0.01);
    assert!((percentile(&sorted, 1.0) - 5.0).abs() < 0.01);
}

#[test]
fn test_trimmed_mean() {
    let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0, 100.0];
    let mean = trimmed_mean(&samples, 0.2).unwrap();
    // Should exclude 1.0 and 100.0
    assert!(mean < 10.0); // Much less than with outlier
}

#[test]
fn test_nan_handling() {
    let samples = vec![1.0, f64::NAN, 3.0, f64::INFINITY, 5.0];
    let analysis = StatisticalAnalysis::from_samples_default(&samples).unwrap();
    assert!(analysis.mean.is_finite());
}

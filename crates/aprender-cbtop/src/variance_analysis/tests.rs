use super::*;

#[test]
fn test_variance_analysis_basic() {
    let input = VarianceInput {
        latencies: vec![10.0, 10.1, 10.2, 10.0, 10.1],
        frequencies: None,
        temperatures: None,
        warmup_count: 1,
    };

    let analysis = VarianceAnalysis::analyze(&input).unwrap();
    assert!(analysis.total_cv_percent < 5.0);
    assert!(analysis.budget_met);
}

#[test]
fn test_empty_input() {
    let input = VarianceInput {
        latencies: vec![],
        frequencies: None,
        temperatures: None,
        warmup_count: 0,
    };

    assert!(VarianceAnalysis::analyze(&input).is_none());
}

#[test]
fn test_high_variance() {
    let input = VarianceInput {
        latencies: vec![10.0, 20.0, 10.0, 20.0, 10.0],
        frequencies: None,
        temperatures: None,
        warmup_count: 1,
    };

    let analysis = VarianceAnalysis::analyze(&input).unwrap();
    assert!(analysis.total_cv_percent > 5.0);
    assert!(!analysis.budget_met);
}

#[test]
fn test_frequency_correlation() {
    let input = VarianceInput {
        latencies: vec![10.0, 12.0, 14.0, 16.0, 18.0],
        frequencies: Some(vec![3000.0, 2800.0, 2600.0, 2400.0, 2200.0]),
        temperatures: None,
        warmup_count: 1,
    };

    let analysis = VarianceAnalysis::analyze(&input).unwrap();
    // Higher latency correlates with lower frequency
    assert!(analysis.frequency_contribution > 0.0);
}

#[test]
fn test_thermal_correlation() {
    let input = VarianceInput {
        latencies: vec![10.0, 11.0, 12.0, 13.0, 14.0],
        frequencies: None,
        temperatures: Some(vec![60.0, 65.0, 70.0, 75.0, 80.0]),
        warmup_count: 1,
    };

    let analysis = VarianceAnalysis::analyze(&input).unwrap();
    // Higher latency correlates with higher temperature
    assert!(analysis.thermal_contribution > 0.0);
}

#[test]
fn test_cache_warmup_effect() {
    // Cold samples (first 3) are slower than warm samples (last 7)
    let input = VarianceInput {
        latencies: vec![20.0, 18.0, 15.0, 10.0, 10.1, 10.0, 10.1, 10.0, 10.1, 10.0],
        frequencies: None,
        temperatures: None,
        warmup_count: 3,
    };

    let analysis = VarianceAnalysis::analyze(&input).unwrap();
    assert!(analysis.warmup_effect > 1.0); // Cold/warm > 1
}

#[test]
fn test_recommendations_generated() {
    let input = VarianceInput {
        latencies: vec![10.0, 20.0, 10.0, 20.0, 10.0, 20.0],
        frequencies: None,
        temperatures: None,
        warmup_count: 1,
    };

    let analysis = VarianceAnalysis::analyze(&input).unwrap();
    assert!(!analysis.recommendations.is_empty());
}

#[test]
fn test_trend_calculation() {
    // Increasing trend
    let input = VarianceInput {
        latencies: vec![10.0, 11.0, 12.0, 13.0, 14.0],
        frequencies: None,
        temperatures: None,
        warmup_count: 0,
    };

    let analysis = VarianceAnalysis::analyze(&input).unwrap();
    assert!(analysis.trend_coefficient > 0.0); // Positive trend
}

#[test]
fn test_variance_source_names() {
    assert_eq!(
        VarianceSource::FrequencyScaling.name(),
        "CPU frequency scaling"
    );
    assert_eq!(
        VarianceSource::ThermalThrottling.name(),
        "thermal throttling"
    );
    assert_eq!(VarianceSource::CacheState.name(), "cache state variance");
    assert_eq!(VarianceSource::SystemNoise.name(), "system noise");
}

#[test]
fn test_correlation_calculation() {
    // Perfect positive correlation
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    let corr = calculate_correlation(&x, &y);
    assert!((corr - 1.0).abs() < 0.001);

    // Perfect negative correlation
    let y_neg = vec![10.0, 8.0, 6.0, 4.0, 2.0];
    let corr_neg = calculate_correlation(&x, &y_neg);
    assert!((corr_neg + 1.0).abs() < 0.001);
}

#[test]
fn test_cv_calculation() {
    let samples = vec![10.0, 10.0, 10.0, 10.0, 10.0];
    let cv = calculate_cv(&samples);
    assert_eq!(cv, 0.0); // No variance

    let samples2 = vec![10.0, 20.0, 10.0, 20.0];
    let cv2 = calculate_cv(&samples2);
    assert!(cv2 > 0.0); // Has variance
}

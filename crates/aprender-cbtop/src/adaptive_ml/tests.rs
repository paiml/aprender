use super::*;

/// Generate sinusoidal samples around a mean with given amplitude.
fn sine_samples(mean: f64, amplitude: f64, count: usize, freq: f64) -> Vec<f64> {
    (0..count)
        .map(|i| mean + (i as f64 * freq).sin() * amplitude)
        .collect()
}

/// Generate low-variance "normal" samples (near-linear ramp).
fn steady_samples(count: usize) -> Vec<f64> {
    (0..count).map(|i| 100.0 + (i as f64 * 0.01)).collect()
}

/// Build a TimeSeriesFeatures with given CV and autocorrelation.
fn features(cv: f64, autocorr: f64) -> TimeSeriesFeatures {
    TimeSeriesFeatures {
        mean: 100.0,
        std_dev: cv,
        cv,
        skewness: 0.0,
        kurtosis: 0.0,
        autocorr_lag1: autocorr,
        trend_slope: 0.0,
        sample_count: 100,
    }
}

/// Build an AdaptiveThresholdMl with custom min_training_samples and min_confidence.
fn ml_with(min_samples: usize, min_confidence: f64) -> AdaptiveThresholdMl {
    AdaptiveThresholdMl::new(MlThresholdConfig {
        min_training_samples: min_samples,
        min_confidence,
        ..Default::default()
    })
}

#[test]
fn test_time_series_features() {
    let values = sine_samples(100.0, 10.0, 100, 0.1);
    let feat = TimeSeriesFeatures::extract(&values).unwrap();

    assert!(feat.mean > 95.0 && feat.mean < 105.0);
    assert!(feat.std_dev > 0.0);
    assert!(feat.cv > 0.0 && feat.cv < 20.0);
    assert_eq!(feat.sample_count, 100);
}

#[test]
fn test_feature_extraction_insufficient_data() {
    assert!(TimeSeriesFeatures::extract(&[1.0, 2.0, 3.0]).is_none());
}

#[test]
fn test_workload_class_defaults() {
    assert!(
        WorkloadClass::Matmul.default_cv_threshold() < WorkloadClass::Ffn.default_cv_threshold()
    );
    assert!(
        WorkloadClass::ComputeBound.default_cv_threshold()
            < WorkloadClass::MemoryBound.default_cv_threshold()
    );
}

#[test]
fn test_learned_threshold_update() {
    let mut threshold = LearnedWorkloadThreshold::new(WorkloadClass::Matmul);
    let feat = features(8.0, 0.5);

    for _ in 0..10 {
        threshold.update(&feat, false);
    }

    assert!(threshold.training_samples > 0);
    assert!(threshold.confidence > 0.0);
}

#[test]
fn test_adaptive_threshold_detection() {
    let ml = AdaptiveThresholdMl::new(MlThresholdConfig::default());

    let result = ml.detect_anomaly(&steady_samples(100)).unwrap();
    assert!(!result.is_anomaly);

    let anomalous = sine_samples(100.0, 50.0, 100, 0.5);
    let result = ml.detect_anomaly(&anomalous).unwrap();
    assert!(result.score > 0.5);
}

#[test]
fn test_adaptive_threshold_training() {
    let mut ml = ml_with(5, 0.1);

    let normal = steady_samples(50);
    for _ in 0..10 {
        ml.train(&normal, false).unwrap();
    }

    let anomalous: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64 * 2.0)).collect();
    for _ in 0..5 {
        ml.train(&anomalous, true).unwrap();
    }

    assert!(!ml.learned_workloads().is_empty());

    let metrics = ml.get_metrics();
    assert!(
        metrics.true_positives
            + metrics.false_positives
            + metrics.true_negatives
            + metrics.false_negatives
            > 0
    );
}

#[test]
fn test_workload_classification() {
    let ml = AdaptiveThresholdMl::new(MlThresholdConfig::default());

    // Low CV, high autocorrelation -> ComputeBound
    assert_eq!(
        ml.classify_workload(&features(5.0, 0.8)),
        WorkloadClass::ComputeBound
    );
    // High CV, low autocorrelation -> MemoryBound
    assert_eq!(
        ml.classify_workload(&features(25.0, 0.1)),
        WorkloadClass::MemoryBound
    );
}

#[test]
fn test_drift_detection() {
    let mut ml = AdaptiveThresholdMl::new(MlThresholdConfig::default());

    let normal = steady_samples(100);
    for _ in 0..20 {
        ml.train(&normal, false).unwrap();
    }

    let similar: Vec<f64> = (0..100).map(|i| 100.5 + (i as f64 * 0.01)).collect();
    let drift = ml.check_drift(&similar).unwrap();
    assert!(drift.is_none() || drift.unwrap() < 3.0);

    let drifted: Vec<f64> = (0..100).map(|i| 200.0 + (i as f64 * 5.0)).collect();
    let _drift = ml.check_drift(&drifted).unwrap();
}

#[test]
fn test_classification_metrics() {
    let mut metrics = ClassificationMetrics::default();
    metrics.true_positives = 80;
    metrics.false_positives = 10;
    metrics.true_negatives = 90;
    metrics.false_negatives = 20;

    assert!((metrics.precision() - 0.889).abs() < 0.01);
    assert!((metrics.recall() - 0.8).abs() < 0.01);
    assert!((metrics.false_positive_rate() - 0.1).abs() < 0.01);
}

#[test]
fn test_model_persistence() {
    let config = MlThresholdConfig::default();
    let mut ml1 = AdaptiveThresholdMl::new(config.clone());

    let samples = steady_samples(100);
    for _ in 0..50 {
        ml1.train(&samples, false).unwrap();
    }

    let state = ml1.export_state();
    assert!(!state.is_empty());

    let mut ml2 = AdaptiveThresholdMl::new(config);
    ml2.import_state(state);
    assert!(!ml2.learned_workloads().is_empty());
}

#[test]
fn test_cold_start_conservative() {
    let ml = AdaptiveThresholdMl::new(MlThresholdConfig {
        cold_start_multiplier: 1.5,
        ..Default::default()
    });

    let threshold = ml.get_threshold(WorkloadClass::Matmul);
    let default = WorkloadClass::Matmul.default_cv_threshold();
    assert!(
        threshold > default,
        "Cold start threshold {threshold} should be > default {default}"
    );
}

#[test]
fn test_error_display() {
    let err = MlThresholdError::InsufficientData { have: 5, need: 10 };
    assert!(err.to_string().contains("5"));
    assert!(err.to_string().contains("10"));

    let err = MlThresholdError::DriftDetected {
        metric: "latency".to_string(),
        drift_score: 4.5,
    };
    assert!(err.to_string().contains("latency"));
    assert!(err.to_string().contains("4.5"));
}

// FKR-050: ML thresholds reduce false positives
#[test]
fn test_fkr_050_precision_improvement() {
    let mut ml = ml_with(10, 0.3);

    // FFN workload: naturally high CV (~18%)
    for _ in 0..50 {
        ml.train(&sine_samples(100.0, 18.0, 100, 0.2), false)
            .unwrap();
    }

    // Matmul workload: naturally low CV (~8%)
    for _ in 0..50 {
        ml.train(&sine_samples(100.0, 8.0, 100, 0.1), false)
            .unwrap();
    }

    // Verify workload-specific thresholds differ
    let ffn_t = ml
        .thresholds
        .get(&WorkloadClass::Ffn)
        .map(|t| t.cv_threshold);
    let matmul_t = ml
        .thresholds
        .get(&WorkloadClass::Matmul)
        .map(|t| t.cv_threshold);

    if let (Some(ft), Some(mt)) = (ffn_t, matmul_t) {
        assert!(
            ft != mt,
            "Workload thresholds should differ: FFN={ft}, Matmul={mt}"
        );
    }

    let fpr = ml.get_metrics().false_positive_rate();
    assert!(fpr < 0.20, "False positive rate {fpr} should be < 20%");
}

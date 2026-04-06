//! ConceptDriftStatus, TrainingStats, TrainingSample, defaults, and edge case tests

use super::*;

// ============================================================================
// ConceptDriftStatus
// ============================================================================

#[test]
fn concept_drift_status_debug_and_clone() {
    let status = ConceptDriftStatus {
        drift_detected: false,
        staleness_score: 0.5,
        samples_since_training: 50,
        recommend_retrain: false,
        explanation: "test".to_string(),
    };
    let cloned = status.clone();
    assert!(!cloned.drift_detected);
    assert_eq!(cloned.staleness_score, 0.5);
    assert_eq!(cloned.samples_since_training, 50);
    let debug = format!("{:?}", cloned);
    assert!(debug.contains("ConceptDriftStatus"));
}

// ============================================================================
// TrainingStats
// ============================================================================

#[test]
fn training_stats_debug_and_clone() {
    let stats = TrainingStats {
        total_samples: 100,
        samples_since_training: 10,
        accepted_count: 5,
        rejected_count: 2,
        alternative_count: 3,
        staleness_score: 0.1,
        drift_detected: false,
        online_learning_enabled: true,
    };
    let cloned = stats.clone();
    assert_eq!(cloned.total_samples, 100);
    let debug = format!("{:?}", cloned);
    assert!(debug.contains("TrainingStats"));
}

// ============================================================================
// TrainingSample
// ============================================================================

#[test]
fn training_sample_debug_and_clone() {
    let sample = make_sample(42.0);
    let cloned = sample.clone();
    assert_eq!(cloned.throughput_tps, 42.0);
    assert_eq!(cloned.hardware_id, "test-hw");
    let debug = format!("{:?}", cloned);
    assert!(debug.contains("TrainingSample"));
}

#[test]
fn training_sample_serialization_round_trip() {
    let sample = make_sample(200.0);
    let json = serde_json::to_string(&sample).expect("serialize sample");
    let deserialized: TrainingSample = serde_json::from_str(&json).expect("deserialize sample");
    assert_eq!(deserialized.throughput_tps, 200.0);
    assert_eq!(deserialized.hardware_id, "test-hw");
    assert!(matches!(deserialized.best_kernel, KernelType::TiledQ4K));
    assert!(matches!(deserialized.bottleneck, BottleneckClass::MemoryBound));
}

// ============================================================================
// TunerDataCollector Default impl
// ============================================================================

#[test]
fn default_collector_is_empty_and_learning_disabled() {
    let default_c = TunerDataCollector::default();
    assert!(default_c.is_empty());
    assert!(!default_c.is_online_learning_enabled());
}

#[test]
fn new_sets_retrain_threshold_unlike_default() {
    // new() sets retrain_threshold = 100, derived Default sets it to 0
    let new_c = TunerDataCollector::new();
    assert_eq!(new_c.retrain_threshold, 100);

    let default_c = TunerDataCollector::default();
    assert_eq!(default_c.retrain_threshold, 0);
}

// ============================================================================
// Edge case: error window exactly at boundary
// ============================================================================

#[test]
fn error_window_at_exact_boundary_for_drift_detection() {
    let mut c = TunerDataCollector::with_online_learning();
    // Add exactly 10 errors (minimum for drift detection)
    for _ in 0..10 {
        c.record_prediction_error(105.0, 100.0);
    }
    let status = c.detect_concept_drift();
    // Should NOT say "Insufficient" since we have exactly 10
    assert!(!status.explanation.contains("Insufficient"));
    // Error is 0.05 which is below 0.15 threshold
    assert!(!status.drift_detected);
}

#[test]
fn error_window_at_nine_insufficient_for_drift() {
    let mut c = TunerDataCollector::with_online_learning();
    for _ in 0..9 {
        c.record_prediction_error(200.0, 100.0);
    }
    let status = c.detect_concept_drift();
    assert!(status.explanation.contains("Insufficient"));
}

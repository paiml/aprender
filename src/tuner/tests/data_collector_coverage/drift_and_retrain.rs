//! detect_concept_drift (all branches), should_retrain (all branches),
//! mark_trained, auto_retrain, training_stats tests

use super::*;

// ============================================================================
// detect_concept_drift() - all branches
// ============================================================================

#[test]
fn detect_drift_insufficient_data() {
    let c = TunerDataCollector::new();
    let status = c.detect_concept_drift();
    assert!(!status.drift_detected);
    assert_eq!(status.staleness_score, 0.0);
    assert!(!status.recommend_retrain);
    assert!(status.explanation.contains("Insufficient"));
}

#[test]
fn detect_drift_insufficient_data_with_some_errors() {
    let mut c = TunerDataCollector::with_online_learning();
    // Add only 5 errors (below the 10 threshold)
    for _ in 0..5 {
        c.record_prediction_error(110.0, 100.0);
    }
    let status = c.detect_concept_drift();
    assert!(!status.drift_detected);
    assert!(status.explanation.contains("Insufficient"));
}

#[test]
fn detect_drift_no_drift_fresh_model() {
    let mut c = TunerDataCollector::with_online_learning();
    // Add 15 low-error predictions (error ~= 0.05)
    for _ in 0..15 {
        c.record_prediction_error(105.0, 100.0);
    }
    let status = c.detect_concept_drift();
    assert!(!status.drift_detected);
    assert!(!status.recommend_retrain);
    assert!(status.explanation.contains("fresh"));
}

#[test]
fn detect_drift_drift_detected_high_error() {
    let mut c = TunerDataCollector::with_online_learning();
    // Add 15 high-error predictions (error = 0.5, exceeds DRIFT_ERROR_THRESHOLD=0.15)
    for _ in 0..15 {
        c.record_prediction_error(150.0, 100.0);
    }
    let status = c.detect_concept_drift();
    assert!(status.drift_detected);
    assert!(status.recommend_retrain);
    assert!(status.explanation.contains("drift"));
}

#[test]
fn detect_drift_stale_model_no_drift() {
    let mut c = TunerDataCollector::with_online_learning();
    // Add many samples to make it stale
    for i in 0..90 {
        c.samples.push(make_sample(100.0 + i as f32));
    }
    // Add low-error predictions
    for _ in 0..15 {
        c.record_prediction_error(101.0, 100.0);
    }
    // staleness = 90 / 100 = 0.9 > 0.8 threshold
    let status = c.detect_concept_drift();
    assert!(!status.drift_detected);
    assert!(status.staleness_score > 0.8);
    assert!(status.recommend_retrain);
    assert!(status.explanation.contains("stale"));
}

#[test]
fn detect_drift_staleness_clamped_to_one() {
    let mut c = TunerDataCollector::with_online_learning();
    // Add far more samples than threshold
    for i in 0..200 {
        c.samples.push(make_sample(100.0 + i as f32));
    }
    for _ in 0..15 {
        c.record_prediction_error(101.0, 100.0);
    }
    let status = c.detect_concept_drift();
    // staleness_score should be clamped to 1.0
    assert!(status.staleness_score <= 1.0);
    assert_eq!(status.samples_since_training, 200);
}

#[test]
fn detect_drift_samples_since_training_uses_saturating_sub() {
    let mut c = TunerDataCollector::new();
    // samples_at_last_train > samples.len() (should never happen but test robustness)
    c.samples_at_last_train = 100;
    let status = c.detect_concept_drift();
    assert_eq!(status.samples_since_training, 0);
}

// ============================================================================
// should_retrain() - all branches
// ============================================================================

#[test]
fn should_retrain_returns_false_when_online_learning_disabled() {
    let c = TunerDataCollector::new();
    assert!(!c.should_retrain());
}

#[test]
fn should_retrain_returns_true_when_enough_new_samples() {
    let mut c = TunerDataCollector::with_online_learning();
    // Add retrain_threshold number of samples
    for i in 0..100 {
        c.samples.push(make_sample(100.0 + i as f32));
    }
    assert!(c.should_retrain());
}

#[test]
fn should_retrain_returns_false_below_threshold() {
    let mut c = TunerDataCollector::with_online_learning();
    for i in 0..10 {
        c.samples.push(make_sample(100.0 + i as f32));
    }
    // Not enough samples and no drift (insufficient data for drift)
    assert!(!c.should_retrain());
}

#[test]
fn should_retrain_returns_true_on_drift_with_min_samples() {
    let mut c = TunerDataCollector::with_online_learning();
    // Add enough samples for drift check but below retrain_threshold
    for i in 0..15 {
        c.samples.push(make_sample(100.0 + i as f32));
    }
    // Add high-error predictions to trigger drift
    for _ in 0..15 {
        c.record_prediction_error(200.0, 100.0);
    }
    // samples_since >= 10 and drift detected -> true
    assert!(c.should_retrain());
}

#[test]
fn should_retrain_returns_false_on_drift_with_too_few_samples() {
    let mut c = TunerDataCollector::with_online_learning();
    // Only 5 samples (below 10 minimum for drift retrain)
    for i in 0..5 {
        c.samples.push(make_sample(100.0 + i as f32));
    }
    // Add high errors to trigger drift
    for _ in 0..15 {
        c.record_prediction_error(300.0, 100.0);
    }
    // drift detected but samples_since < 10 -> false
    assert!(!c.should_retrain());
}

// ============================================================================
// mark_trained()
// ============================================================================

#[test]
fn mark_trained_resets_counters() {
    let mut c = TunerDataCollector::with_online_learning();
    for i in 0..20 {
        c.samples.push(make_sample(100.0 + i as f32));
    }
    for _ in 0..15 {
        c.record_prediction_error(120.0, 100.0);
    }
    assert!(!c.error_window.is_empty());

    c.mark_trained();
    assert_eq!(c.samples_at_last_train, 20);
    assert!(c.error_window.is_empty());
}

// ============================================================================
// training_stats()
// ============================================================================

#[test]
fn training_stats_with_no_feedback() {
    let c = make_collector_with_samples(5);
    let stats = c.training_stats();
    assert_eq!(stats.total_samples, 5);
    assert_eq!(stats.samples_since_training, 5);
    assert_eq!(stats.accepted_count, 0);
    assert_eq!(stats.rejected_count, 0);
    assert_eq!(stats.alternative_count, 0);
    assert!(!stats.drift_detected);
    assert!(!stats.online_learning_enabled);
}

#[test]
fn training_stats_with_mixed_feedback() {
    let mut c = make_collector_with_samples(5);
    c.record_feedback(0, UserFeedback::Accepted);
    c.record_feedback(1, UserFeedback::Accepted);
    c.record_feedback(2, UserFeedback::Rejected);
    c.record_feedback(3, UserFeedback::Alternative);
    // index 4 has no feedback (None)

    let stats = c.training_stats();
    assert_eq!(stats.accepted_count, 2);
    assert_eq!(stats.rejected_count, 1);
    assert_eq!(stats.alternative_count, 1);
}

#[test]
fn training_stats_reflects_online_learning_enabled() {
    let c = TunerDataCollector::with_online_learning();
    let stats = c.training_stats();
    assert!(stats.online_learning_enabled);
}

// ============================================================================
// auto_retrain() - edge cases
// ============================================================================

#[test]
fn auto_retrain_returns_false_when_should_retrain_is_false() {
    let mut c = TunerDataCollector::new();
    let mut tuner = BrickTuner::new();
    assert!(!c.auto_retrain(&mut tuner));
}

#[test]
fn auto_retrain_returns_false_when_weighted_data_too_small() {
    let mut c = TunerDataCollector::with_online_learning();
    // Add samples but mark ALL as rejected so weighted data is empty
    for i in 0..150 {
        c.samples.push(make_sample(100.0 + i as f32));
        c.record_feedback(i, UserFeedback::Rejected);
    }
    c.retrain_threshold = 50;
    let mut tuner = BrickTuner::new();
    // should_retrain() returns true (enough samples) but weighted data is empty
    assert!(!c.auto_retrain(&mut tuner));
}

#[test]
fn auto_retrain_marks_trained_on_success() {
    let mut c = TunerDataCollector::with_online_learning();
    for i in 0..150 {
        let features = TunerFeatures::builder()
            .model_params_b(1.0 + (i as f32) * 0.1)
            .hidden_dim(2048)
            .batch_size((i as u32) % 16 + 1)
            .quant_type(if i % 2 == 0 { QuantType::Q4K } else { QuantType::Q8_0 })
            .build();
        c.samples.push(TrainingSample {
            features,
            throughput_tps: 30.0 + (i as f32) * 2.0,
            best_kernel: KernelType::TiledQ4K,
            bottleneck: BottleneckClass::MemoryBound,
            timestamp: format!("{}", i),
            hardware_id: "test".to_string(),
        });
    }
    c.retrain_threshold = 50;
    let mut tuner = BrickTuner::new();
    let result = c.auto_retrain(&mut tuner);
    assert!(result);
    // After successful retrain, samples_at_last_train should be updated
    assert_eq!(c.samples_at_last_train, 150);
    assert!(c.error_window.is_empty());
}

// ============================================================================
// Drift threshold boundary testing
// ============================================================================

#[test]
fn drift_exactly_at_threshold_not_detected() {
    let mut c = TunerDataCollector::with_online_learning();
    // DRIFT_ERROR_THRESHOLD = 0.15
    // To get exactly 0.15 mean error: predicted/actual ratio = 1.15 or 0.85
    // |115 - 100| / 100 = 0.15
    for _ in 0..15 {
        c.record_prediction_error(115.0, 100.0);
    }
    let status = c.detect_concept_drift();
    // Mean error = 0.15 which is NOT > 0.15 (strictly greater)
    assert!(!status.drift_detected);
}

#[test]
fn drift_just_above_threshold_detected() {
    let mut c = TunerDataCollector::with_online_learning();
    // |116 - 100| / 100 = 0.16 > 0.15
    for _ in 0..15 {
        c.record_prediction_error(116.0, 100.0);
    }
    let status = c.detect_concept_drift();
    assert!(status.drift_detected);
    assert!(status.recommend_retrain);
}

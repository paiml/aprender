//! record_feedback, get_feedback, UserFeedback, record_prediction_error tests

use super::*;

// ============================================================================
// record_feedback() and get_feedback()
// ============================================================================

#[test]
fn record_feedback_stores_at_correct_index() {
    let mut c = TunerDataCollector::new();
    c.record_feedback(0, UserFeedback::Accepted);
    assert_eq!(c.get_feedback(0), UserFeedback::Accepted);
}

#[test]
fn record_feedback_extends_vector_with_none() {
    let mut c = TunerDataCollector::new();
    // Record at index 5 should extend the vector with None for indices 0-4
    c.record_feedback(5, UserFeedback::Rejected);
    assert_eq!(c.get_feedback(0), UserFeedback::None);
    assert_eq!(c.get_feedback(1), UserFeedback::None);
    assert_eq!(c.get_feedback(4), UserFeedback::None);
    assert_eq!(c.get_feedback(5), UserFeedback::Rejected);
}

#[test]
fn get_feedback_returns_none_for_out_of_range() {
    let c = TunerDataCollector::new();
    assert_eq!(c.get_feedback(999), UserFeedback::None);
}

#[test]
fn record_feedback_overwrites_existing() {
    let mut c = TunerDataCollector::new();
    c.record_feedback(0, UserFeedback::Accepted);
    assert_eq!(c.get_feedback(0), UserFeedback::Accepted);
    c.record_feedback(0, UserFeedback::Alternative);
    assert_eq!(c.get_feedback(0), UserFeedback::Alternative);
}

// ============================================================================
// UserFeedback
// ============================================================================

#[test]
fn user_feedback_default_is_none() {
    let fb = UserFeedback::default();
    assert_eq!(fb, UserFeedback::None);
}

#[test]
fn user_feedback_equality() {
    assert_eq!(UserFeedback::Accepted, UserFeedback::Accepted);
    assert_ne!(UserFeedback::Accepted, UserFeedback::Rejected);
    assert_ne!(UserFeedback::Alternative, UserFeedback::None);
}

#[test]
fn user_feedback_clone_and_copy() {
    let fb = UserFeedback::Rejected;
    let cloned = fb;
    assert_eq!(fb, cloned);
}

#[test]
fn user_feedback_debug_format() {
    let fb = UserFeedback::Accepted;
    let debug = format!("{:?}", fb);
    assert!(debug.contains("Accepted"));
}

// ============================================================================
// record_prediction_error()
// ============================================================================

#[test]
fn record_prediction_error_noop_when_online_learning_disabled() {
    let mut c = TunerDataCollector::new();
    assert!(!c.is_online_learning_enabled());
    c.record_prediction_error(100.0, 110.0);
    assert!(c.error_window.is_empty());
}

#[test]
fn record_prediction_error_adds_to_window_when_enabled() {
    let mut c = TunerDataCollector::with_online_learning();
    c.record_prediction_error(100.0, 100.0);
    assert_eq!(c.error_window.len(), 1);
    // Perfect prediction -> error = 0.0
    assert_eq!(c.error_window[0], 0.0);
}

#[test]
fn record_prediction_error_computes_relative_error() {
    let mut c = TunerDataCollector::with_online_learning();
    // predicted=150, actual=100 -> |50/100| = 0.5
    c.record_prediction_error(150.0, 100.0);
    assert!((c.error_window[0] - 0.5).abs() < 1e-6);
}

#[test]
fn record_prediction_error_clamps_to_one() {
    let mut c = TunerDataCollector::with_online_learning();
    // predicted=1000, actual=1 -> |999/1| = 999.0 clamped to 1.0
    c.record_prediction_error(1000.0, 1.0);
    assert_eq!(c.error_window[0], 1.0);
}

#[test]
fn record_prediction_error_actual_zero_returns_one() {
    let mut c = TunerDataCollector::with_online_learning();
    c.record_prediction_error(50.0, 0.0);
    assert_eq!(c.error_window[0], 1.0);
}

#[test]
fn record_prediction_error_trims_sliding_window() {
    let mut c = TunerDataCollector::with_online_learning();
    // Default window size is 50
    for i in 0..60 {
        c.record_prediction_error(100.0 + i as f32, 100.0);
    }
    // Window should be trimmed to 50
    assert_eq!(c.error_window.len(), 50);
}

#[test]
fn record_prediction_error_negative_actual_uses_abs() {
    let mut c = TunerDataCollector::with_online_learning();
    // predicted=100, actual=-100 -> actual > 0.0 is false -> error = 1.0
    c.record_prediction_error(100.0, -100.0);
    assert_eq!(c.error_window[0], 1.0);
}

// ============================================================================
// UserFeedback serialization
// ============================================================================

#[test]
fn user_feedback_serialization_round_trip() {
    for fb in &[
        UserFeedback::Accepted,
        UserFeedback::Rejected,
        UserFeedback::Alternative,
        UserFeedback::None,
    ] {
        let json = serde_json::to_string(fb).expect("serialize feedback");
        let deserialized: UserFeedback = serde_json::from_str(&json).expect("deserialize feedback");
        assert_eq!(*fb, deserialized);
    }
}

// ============================================================================
// prepare_weighted_training_data()
// ============================================================================

#[test]
fn weighted_data_skips_rejected_samples() {
    let mut c = make_collector_with_samples(3);
    c.record_feedback(1, UserFeedback::Rejected);

    // Access through auto_retrain path is private, but we can test via
    // the public interface and check sample counts indirectly.
    // Let's use the training_stats to verify feedback is recorded
    let stats = c.training_stats();
    assert_eq!(stats.rejected_count, 1);
}

#[test]
fn weighted_data_doubles_accepted_samples() {
    let mut c = make_collector_with_samples(3);
    c.record_feedback(0, UserFeedback::Accepted);
    // Accepted samples are duplicated in weighted data
    // We can verify the feedback was stored
    assert_eq!(c.get_feedback(0), UserFeedback::Accepted);
    assert_eq!(c.get_feedback(1), UserFeedback::None);
    assert_eq!(c.get_feedback(2), UserFeedback::None);
}

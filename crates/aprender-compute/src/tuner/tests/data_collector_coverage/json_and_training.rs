//! to_json/from_json round-trip, prepare_training_data, ready_to_train, training_progress tests

use super::*;

// ============================================================================
// to_json() / from_json() round-trip
// ============================================================================

#[test]
fn to_json_produces_valid_json() {
    let c = make_collector_with_samples(3);
    let json = c.to_json().expect("serialization should succeed");
    assert!(json.starts_with('['));
    assert!(json.contains("throughput_tps"));
}

#[test]
fn from_json_round_trip_preserves_data() {
    let original = make_collector_with_samples(2);
    let json = original.to_json().expect("serialize");
    let loaded = TunerDataCollector::from_json(&json).expect("deserialize");
    assert_eq!(loaded.len(), 2);
    assert_eq!(loaded.samples()[0].throughput_tps, original.samples()[0].throughput_tps);
    assert_eq!(loaded.samples()[1].throughput_tps, original.samples()[1].throughput_tps);
}

#[test]
fn from_json_returns_error_on_bad_json() {
    let result = TunerDataCollector::from_json("{broken");
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(err_msg.contains("Serialization"));
}

#[test]
fn to_json_on_empty_collector_returns_empty_array() {
    let c = TunerDataCollector::new();
    let json = c.to_json().expect("serialize empty");
    assert_eq!(json.trim(), "[]");
}

// ============================================================================
// prepare_training_data()
// ============================================================================

#[test]
fn prepare_training_data_extracts_features_and_throughput() {
    let c = make_collector_with_samples(3);
    let data = c.prepare_training_data();
    assert_eq!(data.len(), 3);
    // First sample throughput is 100.0
    assert_eq!(data[0].1, 100.0);
    assert_eq!(data[1].1, 101.0);
    assert_eq!(data[2].1, 102.0);
}

#[test]
fn prepare_training_data_empty_collector() {
    let c = TunerDataCollector::new();
    let data = c.prepare_training_data();
    assert!(data.is_empty());
}

// ============================================================================
// ready_to_train() and training_progress()
// ============================================================================

#[test]
fn ready_to_train_returns_false_below_threshold() {
    let c = make_collector_with_samples(999);
    assert!(!c.ready_to_train());
}

#[test]
fn ready_to_train_returns_true_at_threshold() {
    let c = make_collector_with_samples(TunerDataCollector::MIN_SAMPLES_FOR_TRAINING);
    assert!(c.ready_to_train());
}

#[test]
fn training_progress_reflects_sample_count() {
    let c = make_collector_with_samples(42);
    let (current, required) = c.training_progress();
    assert_eq!(current, 42);
    assert_eq!(required, TunerDataCollector::MIN_SAMPLES_FOR_TRAINING);
}

// ============================================================================
// train_if_ready()
// ============================================================================

#[test]
fn train_if_ready_returns_none_when_not_enough_samples() {
    let c = make_collector_with_samples(50);
    assert!(c.train_if_ready().is_none());
}

// ============================================================================
// bootstrap_from_five_whys()
// ============================================================================

#[test]
fn bootstrap_from_five_whys_returns_valid_collector() {
    let c = TunerDataCollector::bootstrap_from_five_whys();
    // Currently returns empty collector (TODO in source)
    assert!(c.is_empty());
    assert!(!c.is_online_learning_enabled());
    assert_eq!(c.retrain_threshold, 100);
}

// ============================================================================
// Constants
// ============================================================================

#[test]
fn constants_have_expected_values() {
    assert_eq!(TunerDataCollector::MIN_SAMPLES_FOR_TRAINING, 1000);
}

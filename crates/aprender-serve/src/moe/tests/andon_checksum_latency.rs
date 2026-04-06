//! Tests for AndonTrigger

use crate::moe::{AndonResponse, AndonTrigger};

#[test]
fn test_andon_checksum_triggers_rollback() {
    let trigger = AndonTrigger::ModelChecksumMismatch {
        model_id: "model-1".to_string(),
    };
    assert_eq!(trigger.response(), AndonResponse::Rollback);
    assert!(trigger.is_critical());
}

#[test]
fn test_andon_latency_triggers_notify() {
    let trigger = AndonTrigger::LatencyExceeded {
        p99_ms: 150.0,
        threshold_ms: 100.0,
    };
    assert_eq!(trigger.response(), AndonResponse::Notify);
    assert!(!trigger.is_critical());
}

#[test]
fn test_andon_high_error_rate_quarantines() {
    let trigger = AndonTrigger::ErrorRateThreshold {
        rate: 0.25,
        threshold: 0.1,
    };
    assert_eq!(trigger.response(), AndonResponse::Quarantine);
    assert!(trigger.is_critical());
}

#[test]
fn test_andon_moderate_error_rate_notifies() {
    let trigger = AndonTrigger::ErrorRateThreshold {
        rate: 0.15,
        threshold: 0.1,
    };
    assert_eq!(trigger.response(), AndonResponse::Notify);
}

#[test]
fn test_andon_expert_imbalance_triggers_notify() {
    let trigger = AndonTrigger::ExpertImbalance {
        imbalance_ratio: 3.5,
    };
    assert_eq!(trigger.response(), AndonResponse::Notify);
    assert!(!trigger.is_critical());
}

#[test]
fn test_andon_error_rate_exactly_at_threshold() {
    let trigger = AndonTrigger::ErrorRateThreshold {
        rate: 0.1,
        threshold: 0.1,
    };
    assert_eq!(trigger.response(), AndonResponse::Notify);
    assert!(!trigger.is_critical());
}

#[test]
fn test_andon_error_rate_exactly_2x_threshold() {
    let trigger = AndonTrigger::ErrorRateThreshold {
        rate: 0.2,
        threshold: 0.1,
    };
    assert_eq!(trigger.response(), AndonResponse::Notify);
    assert!(!trigger.is_critical());

    let trigger = AndonTrigger::ErrorRateThreshold {
        rate: 0.21,
        threshold: 0.1,
    };
    assert_eq!(trigger.response(), AndonResponse::Quarantine);
    assert!(trigger.is_critical());
}

#[test]
fn test_andon_trigger_debug_and_clone() {
    let trigger = AndonTrigger::ModelChecksumMismatch {
        model_id: "test-model".to_string(),
    };
    let cloned = trigger.clone();
    assert_eq!(trigger, cloned);

    let debug_str = format!("{:?}", trigger);
    assert!(debug_str.contains("ModelChecksumMismatch"));
    assert!(debug_str.contains("test-model"));
}

#[test]
fn test_andon_response_debug_and_eq() {
    assert_eq!(AndonResponse::Rollback, AndonResponse::Rollback);
    assert_eq!(AndonResponse::Notify, AndonResponse::Notify);
    assert_eq!(AndonResponse::Quarantine, AndonResponse::Quarantine);
    assert_ne!(AndonResponse::Rollback, AndonResponse::Notify);

    let debug_str = format!("{:?}", AndonResponse::Quarantine);
    assert_eq!(debug_str, "Quarantine");
}

#[test]
fn test_andon_trigger_latency_clone_and_eq() {
    let trigger = AndonTrigger::LatencyExceeded {
        p99_ms: 150.0,
        threshold_ms: 100.0,
    };
    let cloned = trigger.clone();
    assert_eq!(trigger, cloned);
}

#[test]
fn test_andon_trigger_error_rate_clone_and_eq() {
    let trigger = AndonTrigger::ErrorRateThreshold {
        rate: 0.15,
        threshold: 0.1,
    };
    let cloned = trigger.clone();
    assert_eq!(trigger, cloned);
}

#[test]
fn test_andon_trigger_imbalance_clone_and_eq() {
    let trigger = AndonTrigger::ExpertImbalance {
        imbalance_ratio: 2.5,
    };
    let cloned = trigger.clone();
    assert_eq!(trigger, cloned);
}

#[test]
fn test_andon_response_clone_all_variants() {
    let rollback = AndonResponse::Rollback;
    let rollback_cloned = rollback.clone();
    assert_eq!(rollback, rollback_cloned);

    let notify = AndonResponse::Notify;
    let notify_cloned = notify.clone();
    assert_eq!(notify, notify_cloned);

    let quarantine = AndonResponse::Quarantine;
    let quarantine_cloned = quarantine.clone();
    assert_eq!(quarantine, quarantine_cloned);
}

#[test]
fn test_andon_very_high_error_rate() {
    let trigger = AndonTrigger::ErrorRateThreshold {
        rate: 1.0, // 100% error rate
        threshold: 0.05,
    };
    assert_eq!(trigger.response(), AndonResponse::Quarantine);
    assert!(trigger.is_critical());
}

#[test]
fn test_andon_zero_error_rate() {
    let trigger = AndonTrigger::ErrorRateThreshold {
        rate: 0.0,
        threshold: 0.1,
    };
    assert_eq!(trigger.response(), AndonResponse::Notify);
    assert!(!trigger.is_critical());
}

#[test]
fn test_andon_high_imbalance_ratio() {
    let trigger = AndonTrigger::ExpertImbalance {
        imbalance_ratio: 100.0,
    };
    // ExpertImbalance always notifies, never quarantines
    assert_eq!(trigger.response(), AndonResponse::Notify);
    assert!(!trigger.is_critical());
}

#[test]
fn test_andon_latency_equal_to_threshold() {
    let trigger = AndonTrigger::LatencyExceeded {
        p99_ms: 100.0,
        threshold_ms: 100.0,
    };
    assert_eq!(trigger.response(), AndonResponse::Notify);
    assert!(!trigger.is_critical());
}

#[test]
fn test_andon_checksum_empty_model_id() {
    let trigger = AndonTrigger::ModelChecksumMismatch {
        model_id: String::new(),
    };
    assert_eq!(trigger.response(), AndonResponse::Rollback);
    assert!(trigger.is_critical());
}

#[test]
fn test_andon_checksum_long_model_id() {
    let trigger = AndonTrigger::ModelChecksumMismatch {
        model_id: "a".repeat(1000),
    };
    assert_eq!(trigger.response(), AndonResponse::Rollback);
    assert!(trigger.is_critical());
}

#[test]
fn test_andon_trigger_partial_eq() {
    let t1 = AndonTrigger::LatencyExceeded {
        p99_ms: 100.0,
        threshold_ms: 50.0,
    };
    let t2 = AndonTrigger::LatencyExceeded {
        p99_ms: 100.0,
        threshold_ms: 50.0,
    };
    let t3 = AndonTrigger::LatencyExceeded {
        p99_ms: 200.0,
        threshold_ms: 50.0,
    };

    assert_eq!(t1, t2);
    assert_ne!(t1, t3);
}

#[test]
fn test_andon_response_ne() {
    assert_ne!(AndonResponse::Rollback, AndonResponse::Notify);
    assert_ne!(AndonResponse::Notify, AndonResponse::Quarantine);
    assert_ne!(AndonResponse::Quarantine, AndonResponse::Rollback);
}

#[test]
fn test_andon_trigger_all_variants_is_critical() {
    // ModelChecksumMismatch is always critical
    let t = AndonTrigger::ModelChecksumMismatch {
        model_id: "x".to_string(),
    };
    assert!(t.is_critical());

    // LatencyExceeded is never critical
    let t = AndonTrigger::LatencyExceeded {
        p99_ms: 1000.0,
        threshold_ms: 10.0,
    };
    assert!(!t.is_critical());

    // ExpertImbalance is never critical
    let t = AndonTrigger::ExpertImbalance {
        imbalance_ratio: 1000.0,
    };
    assert!(!t.is_critical());

    // ErrorRateThreshold depends on rate vs threshold
    let t = AndonTrigger::ErrorRateThreshold {
        rate: 0.05,
        threshold: 0.1,
    };
    assert!(!t.is_critical()); // rate < threshold

    let t = AndonTrigger::ErrorRateThreshold {
        rate: 0.5,
        threshold: 0.1,
    };
    assert!(t.is_critical()); // rate > 2 * threshold
}

#[test]
fn test_andon_error_rate_boundary_cases() {
    // Exactly 2x threshold - should be Notify (not > 2x)
    let t = AndonTrigger::ErrorRateThreshold {
        rate: 0.2,
        threshold: 0.1,
    };
    assert_eq!(t.response(), AndonResponse::Notify);

    // Slightly over 2x threshold - should be Quarantine
    let t = AndonTrigger::ErrorRateThreshold {
        rate: 0.200001,
        threshold: 0.1,
    };
    assert_eq!(t.response(), AndonResponse::Quarantine);
}

//! Tests for Safety Andon

use super::*;
use crate::monitor::andon::AlertLevel;
use crate::monitor::inference::path::LinearPath;
use crate::monitor::inference::trace::DecisionTrace;

fn make_trace(confidence: f32, latency_ns: u64, output: f32) -> DecisionTrace<LinearPath> {
    let path = LinearPath::new(vec![0.5], 0.0, 0.0, output).with_probability(confidence);
    DecisionTrace::new(0, 0, 0, path, output, latency_ns)
}

#[test]
fn test_sil_levels() {
    assert_eq!(SafetyIntegrityLevel::QM.min_confidence(), 0.0);
    assert_eq!(SafetyIntegrityLevel::SIL4.min_confidence(), 0.9);

    assert_eq!(SafetyIntegrityLevel::QM.max_latency_ns(), u64::MAX);
    assert_eq!(SafetyIntegrityLevel::SIL4.max_latency_ns(), 1_000_000);
}

#[test]
fn test_sil_as_str() {
    assert_eq!(SafetyIntegrityLevel::QM.as_str(), "QM");
    assert_eq!(SafetyIntegrityLevel::SIL3.as_str(), "SIL3");
}

#[test]
fn test_safety_andon_new() {
    let andon = SafetyAndon::new(SafetyIntegrityLevel::SIL2);
    assert_eq!(andon.sil(), SafetyIntegrityLevel::SIL2);
    assert!(!andon.should_stop());
}

#[test]
fn test_safety_andon_confidence_check() {
    let mut andon = SafetyAndon::new(SafetyIntegrityLevel::SIL2);

    // High confidence - no alert
    let trace = make_trace(0.9, 1_000, 0.9);
    andon.check_trace(&trace, 10_000_000);
    assert!(!andon.should_stop());
    assert!(andon.history().is_empty());

    // Low confidence - warning
    let trace = make_trace(0.5, 1_000, 0.5);
    andon.check_trace(&trace, 10_000_000);
    assert!(!andon.should_stop());
    assert_eq!(andon.history().len(), 1);
    assert_eq!(andon.history()[0].level, AlertLevel::Warning);
}

#[test]
fn test_safety_andon_consecutive_low_confidence() {
    let mut andon = SafetyAndon::new(SafetyIntegrityLevel::SIL2).with_low_confidence_threshold(3);

    // 3 consecutive low confidence decisions
    for _ in 0..3 {
        let trace = make_trace(0.5, 1_000, 0.5);
        andon.check_trace(&trace, 10_000_000);
    }

    // Should have triggered emergency
    assert!(andon.history().len() >= 3);
}

#[test]
fn test_safety_andon_latency_check() {
    let mut andon = SafetyAndon::new(SafetyIntegrityLevel::SIL3);

    // Fast response - no alert
    let trace = make_trace(0.9, 1_000_000, 0.9);
    andon.check_trace(&trace, 10_000_000);
    assert!(andon.history().is_empty());

    // Slow response - warning
    let trace = make_trace(0.9, 15_000_000, 0.9);
    andon.check_trace(&trace, 10_000_000);
    assert_eq!(andon.history().len(), 1);
}

#[test]
fn test_safety_andon_invalid_output() {
    let mut andon = SafetyAndon::new(SafetyIntegrityLevel::QM);

    // NaN output - critical
    let trace = make_trace(0.9, 1_000, f32::NAN);
    andon.check_trace(&trace, 10_000_000);
    assert!(andon.should_stop());
}

#[test]
fn test_safety_andon_reset() {
    let mut andon = SafetyAndon::new(SafetyIntegrityLevel::QM);

    let trace = make_trace(0.9, 1_000, f32::NAN);
    andon.check_trace(&trace, 10_000_000);
    assert!(andon.should_stop());

    andon.reset();
    assert!(!andon.should_stop());
}

#[test]
fn test_emergency_condition_message() {
    let cond = EmergencyCondition::CollisionImminent { time_to_collision_ms: 50.0 };
    assert!(cond.message().contains("50.0ms"));

    let cond = EmergencyCondition::SensorDegraded { sensor: "lidar".to_string(), quality: 0.5 };
    assert!(cond.message().contains("lidar"));
    assert!(cond.message().contains("50.0%"));
}

#[test]
fn test_emergency_condition_alert_level() {
    assert_eq!(
        EmergencyCondition::CollisionImminent { time_to_collision_ms: 10.0 }.alert_level(),
        AlertLevel::Critical
    );
    assert_eq!(EmergencyCondition::AuditChainBroken.alert_level(), AlertLevel::Critical);
    assert_eq!(
        EmergencyCondition::DecisionTimeout { max_ms: 10.0 }.alert_level(),
        AlertLevel::Error
    );
}

#[test]
fn test_safety_andon_custom_thresholds() {
    let andon = SafetyAndon::new(SafetyIntegrityLevel::QM)
        .with_min_confidence(0.95)
        .with_max_latency_ns(500_000)
        .with_low_confidence_threshold(10);

    assert_eq!(andon.min_confidence, 0.95);
    assert_eq!(andon.max_latency_ns, 500_000);
    assert_eq!(andon.low_confidence_threshold, 10);
}

// Additional coverage tests

#[test]
fn test_sil_all_levels_as_str() {
    assert_eq!(SafetyIntegrityLevel::SIL1.as_str(), "SIL1");
    assert_eq!(SafetyIntegrityLevel::SIL2.as_str(), "SIL2");
    assert_eq!(SafetyIntegrityLevel::SIL4.as_str(), "SIL4");
}

#[test]
fn test_sil_all_levels_min_confidence() {
    assert_eq!(SafetyIntegrityLevel::SIL1.min_confidence(), 0.5);
    assert_eq!(SafetyIntegrityLevel::SIL2.min_confidence(), 0.7);
    assert_eq!(SafetyIntegrityLevel::SIL3.min_confidence(), 0.8);
}

#[test]
fn test_sil_all_levels_max_latency() {
    assert_eq!(SafetyIntegrityLevel::SIL1.max_latency_ns(), 100_000_000);
    assert_eq!(SafetyIntegrityLevel::SIL2.max_latency_ns(), 50_000_000);
    assert_eq!(SafetyIntegrityLevel::SIL3.max_latency_ns(), 10_000_000);
}

#[test]
fn test_safety_andon_default() {
    let andon = SafetyAndon::default();
    assert_eq!(andon.sil(), SafetyIntegrityLevel::QM);
}

#[test]
fn test_safety_andon_without_unknown_alerts() {
    let andon = SafetyAndon::new(SafetyIntegrityLevel::QM).without_unknown_alerts();
    assert!(!andon.alert_on_unknown);
}

#[test]
fn test_safety_andon_andon_accessor() {
    let andon = SafetyAndon::new(SafetyIntegrityLevel::QM);
    let _ = andon.andon();
    // Just verify we can access it
}

#[test]
fn test_emergency_condition_decision_timeout_message() {
    let cond = EmergencyCondition::DecisionTimeout { max_ms: 25.5 };
    let msg = cond.message();
    assert!(msg.contains("25.5"));
    assert!(msg.contains("timeout"));
}

#[test]
fn test_emergency_condition_consecutive_low_confidence_message() {
    let cond = EmergencyCondition::ConsecutiveLowConfidence { count: 7, threshold: 0.65 };
    let msg = cond.message();
    assert!(msg.contains('7'));
    assert!(msg.contains("65.0%"));
}

#[test]
fn test_emergency_condition_invalid_output_message() {
    let cond = EmergencyCondition::InvalidOutput;
    let msg = cond.message();
    assert!(msg.contains("NaN") || msg.contains("Inf"));
}

#[test]
fn test_emergency_condition_audit_chain_broken_message() {
    let cond = EmergencyCondition::AuditChainBroken;
    let msg = cond.message();
    assert!(msg.contains("chain") || msg.contains("integrity"));
}

#[test]
fn test_emergency_condition_sensor_degraded_critical() {
    // Quality < 0.3 should be Critical
    let cond = EmergencyCondition::SensorDegraded { sensor: "camera".to_string(), quality: 0.1 };
    assert_eq!(cond.alert_level(), AlertLevel::Critical);
}

#[test]
fn test_emergency_condition_sensor_degraded_error() {
    // Quality >= 0.3 should be Error
    let cond = EmergencyCondition::SensorDegraded { sensor: "camera".to_string(), quality: 0.5 };
    assert_eq!(cond.alert_level(), AlertLevel::Error);
}

#[test]
fn test_emergency_condition_consecutive_low_confidence_alert_level() {
    let cond = EmergencyCondition::ConsecutiveLowConfidence { count: 5, threshold: 0.7 };
    assert_eq!(cond.alert_level(), AlertLevel::Warning);
}

#[test]
fn test_emergency_condition_invalid_output_alert_level() {
    let cond = EmergencyCondition::InvalidOutput;
    assert_eq!(cond.alert_level(), AlertLevel::Critical);
}

#[test]
fn test_safety_andon_critical_latency() {
    let mut andon = SafetyAndon::new(SafetyIntegrityLevel::SIL3).with_max_latency_ns(1_000_000); // 1ms

    // More than 2x over budget should trigger emergency (DecisionTimeout)
    let trace = make_trace(0.95, 3_000_000, 0.95); // 3ms > 2*1ms
    andon.check_trace(&trace, 10_000_000);

    // Should have triggered an error/critical
    assert!(!andon.history().is_empty());
}

#[test]
fn test_safety_andon_infinite_output() {
    let mut andon = SafetyAndon::new(SafetyIntegrityLevel::QM);

    // Infinite output - critical
    let trace = make_trace(0.9, 1_000, f32::INFINITY);
    andon.check_trace(&trace, 10_000_000);
    assert!(andon.should_stop());
}

#[test]
fn test_safety_andon_neg_infinite_output() {
    let mut andon = SafetyAndon::new(SafetyIntegrityLevel::QM);

    // Negative infinite output - critical
    let trace = make_trace(0.9, 1_000, f32::NEG_INFINITY);
    andon.check_trace(&trace, 10_000_000);
    assert!(andon.should_stop());
}

#[test]
fn test_safety_andon_trigger_emergency() {
    let mut andon = SafetyAndon::new(SafetyIntegrityLevel::QM);

    andon.trigger_emergency(EmergencyCondition::AuditChainBroken);
    assert!(andon.should_stop());
    assert!(!andon.history().is_empty());
}

#[test]
fn test_serde_safety_integrity_level() {
    let sil = SafetyIntegrityLevel::SIL3;
    let json = serde_json::to_string(&sil).expect("JSON serialization should succeed");
    let deserialized: SafetyIntegrityLevel =
        serde_json::from_str(&json).expect("JSON deserialization should succeed");
    assert_eq!(sil, deserialized);
}

#[test]
fn test_serde_emergency_condition() {
    let cond = EmergencyCondition::CollisionImminent { time_to_collision_ms: 100.0 };
    let json = serde_json::to_string(&cond).expect("JSON serialization should succeed");
    let _: EmergencyCondition =
        serde_json::from_str(&json).expect("JSON deserialization should succeed");
}

#[test]
fn test_safety_andon_confidence_reset_on_good() {
    let mut andon = SafetyAndon::new(SafetyIntegrityLevel::SIL2).with_low_confidence_threshold(3);

    // Two low confidence decisions
    let trace = make_trace(0.5, 1_000, 0.5);
    andon.check_trace(&trace, 10_000_000);
    andon.check_trace(&trace, 10_000_000);

    // One good decision resets counter
    let trace = make_trace(0.9, 1_000, 0.9);
    andon.check_trace(&trace, 10_000_000);

    // Two more low confidence - should not trigger emergency yet
    let trace = make_trace(0.5, 1_000, 0.5);
    andon.check_trace(&trace, 10_000_000);
    andon.check_trace(&trace, 10_000_000);

    // Total alerts should be 4 (2 + 2 low confidence warnings)
    assert_eq!(andon.history().len(), 4);
}

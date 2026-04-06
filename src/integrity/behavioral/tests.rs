//! Tests for behavioral integrity module

use super::*;

#[test]
fn test_metamorphic_violation_new() {
    let violation = MetamorphicViolation::new(
        "V001",
        MetamorphicRelationType::Identity,
        "Output differs on identical input",
        "Input: [1, 2, 3]",
        "[0.5, 0.3, 0.2]",
        "[0.4, 0.4, 0.2]",
        0.7,
    );

    assert_eq!(violation.id, "V001");
    assert_eq!(violation.relation_type, MetamorphicRelationType::Identity);
    assert!((violation.severity - 0.7).abs() < f64::EPSILON);
}

#[test]
fn test_metamorphic_violation_severity_clamped() {
    let high = MetamorphicViolation::new(
        "V001",
        MetamorphicRelationType::Identity,
        "desc",
        "input",
        "exp",
        "act",
        1.5,
    );
    assert!((high.severity - 1.0).abs() < f64::EPSILON);

    let low = MetamorphicViolation::new(
        "V002",
        MetamorphicRelationType::Identity,
        "desc",
        "input",
        "exp",
        "act",
        -0.5,
    );
    assert!((low.severity - 0.0).abs() < f64::EPSILON);
}

#[test]
fn test_metamorphic_violation_is_critical() {
    let critical = MetamorphicViolation::new(
        "V001",
        MetamorphicRelationType::Identity,
        "desc",
        "input",
        "exp",
        "act",
        0.8,
    );
    assert!(critical.is_critical());

    let not_critical = MetamorphicViolation::new(
        "V002",
        MetamorphicRelationType::Identity,
        "desc",
        "input",
        "exp",
        "act",
        0.79,
    );
    assert!(!not_critical.is_critical());
}

#[test]
fn test_metamorphic_violation_is_warning() {
    let warning = MetamorphicViolation::new(
        "V001",
        MetamorphicRelationType::Identity,
        "desc",
        "input",
        "exp",
        "act",
        0.5,
    );
    assert!(warning.is_warning());

    let not_warning = MetamorphicViolation::new(
        "V002",
        MetamorphicRelationType::Identity,
        "desc",
        "input",
        "exp",
        "act",
        0.49,
    );
    assert!(!not_warning.is_warning());
}

#[test]
fn test_metamorphic_relation_type_display() {
    assert_eq!(format!("{}", MetamorphicRelationType::Additive), "additive");
    assert_eq!(format!("{}", MetamorphicRelationType::Permutation), "permutation");
    assert_eq!(format!("{}", MetamorphicRelationType::Identity), "identity");
}

#[test]
fn test_behavioral_integrity_new() {
    let integrity = BehavioralIntegrity::new(0.95, 0.90, 0.05, 0.92, "model-v1");

    assert!((integrity.equivalence_score - 0.95).abs() < f64::EPSILON);
    assert!((integrity.syscall_match - 0.90).abs() < f64::EPSILON);
    assert!((integrity.timing_variance - 0.05).abs() < f64::EPSILON);
    assert!((integrity.semantic_equiv - 0.92).abs() < f64::EPSILON);
    assert_eq!(integrity.model_id, "model-v1");
    assert!(integrity.violations.is_empty());
}

#[test]
fn test_behavioral_integrity_scores_clamped() {
    let integrity = BehavioralIntegrity::new(1.5, -0.1, 2.0, -0.5, "model");

    assert!((integrity.equivalence_score - 1.0).abs() < f64::EPSILON);
    assert!((integrity.syscall_match - 0.0).abs() < f64::EPSILON);
    assert!((integrity.timing_variance - 1.0).abs() < f64::EPSILON);
    assert!((integrity.semantic_equiv - 0.0).abs() < f64::EPSILON);
}

#[test]
fn test_behavioral_integrity_perfect() {
    let integrity = BehavioralIntegrity::perfect("model-v1");

    assert!((integrity.equivalence_score - 1.0).abs() < f64::EPSILON);
    assert!((integrity.syscall_match - 1.0).abs() < f64::EPSILON);
    assert!((integrity.timing_variance - 0.0).abs() < f64::EPSILON);
    assert!((integrity.semantic_equiv - 1.0).abs() < f64::EPSILON);
}

#[test]
fn test_behavioral_integrity_composite_score_perfect() {
    let integrity = BehavioralIntegrity::perfect("model");
    let score = integrity.composite_score();

    // Perfect scores should yield composite of 1.0
    assert!((score - 1.0).abs() < f64::EPSILON);
}

#[test]
fn test_behavioral_integrity_composite_score_mixed() {
    let integrity = BehavioralIntegrity::new(0.8, 0.7, 0.3, 0.9, "model");
    let score = integrity.composite_score();

    // Manual calculation:
    // 0.3 * 0.8 + 0.2 * 0.7 + 0.2 * (1-0.3) + 0.3 * 0.9
    // = 0.24 + 0.14 + 0.14 + 0.27 = 0.79
    assert!((score - 0.79).abs() < 0.01);
}

#[test]
fn test_behavioral_integrity_passes_gate() {
    let good = BehavioralIntegrity::perfect("model");
    assert!(good.passes_gate(0.9));

    let poor = BehavioralIntegrity::new(0.5, 0.5, 0.5, 0.5, "model");
    assert!(!poor.passes_gate(0.9));
}

#[test]
fn test_behavioral_integrity_passes_gate_timing_variance() {
    let mut integrity = BehavioralIntegrity::perfect("model");
    integrity.timing_variance = 0.25; // Too high

    // Even with perfect other scores, high timing variance fails
    assert!(!integrity.passes_gate(0.9));
}

#[test]
fn test_behavioral_integrity_passes_gate_critical_violation() {
    let mut integrity = BehavioralIntegrity::perfect("model");
    integrity.add_violation(MetamorphicViolation::new(
        "V001",
        MetamorphicRelationType::Identity,
        "desc",
        "input",
        "exp",
        "act",
        0.85, // Critical
    ));

    assert!(!integrity.passes_gate(0.9));
}

#[test]
fn test_behavioral_integrity_violation_counts() {
    let mut integrity = BehavioralIntegrity::new(0.9, 0.9, 0.1, 0.9, "model");

    // Add violations of different severities
    integrity.add_violation(MetamorphicViolation::new(
        "V1",
        MetamorphicRelationType::Identity,
        "d",
        "i",
        "e",
        "a",
        0.9, // Critical
    ));
    integrity.add_violation(MetamorphicViolation::new(
        "V2",
        MetamorphicRelationType::Identity,
        "d",
        "i",
        "e",
        "a",
        0.6, // Warning
    ));
    integrity.add_violation(MetamorphicViolation::new(
        "V3",
        MetamorphicRelationType::Identity,
        "d",
        "i",
        "e",
        "a",
        0.3, // Minor
    ));
    integrity.add_violation(MetamorphicViolation::new(
        "V4",
        MetamorphicRelationType::Identity,
        "d",
        "i",
        "e",
        "a",
        0.2, // Minor
    ));

    let counts = integrity.violation_counts();
    assert_eq!(counts.critical, 1);
    assert_eq!(counts.warnings, 1);
    assert_eq!(counts.minor, 2);
    assert_eq!(counts.total, 4);
}

#[test]
fn test_behavioral_integrity_violations_by_type() {
    let mut integrity = BehavioralIntegrity::new(0.9, 0.9, 0.1, 0.9, "model");

    integrity.add_violation(MetamorphicViolation::new(
        "V1",
        MetamorphicRelationType::Identity,
        "d",
        "i",
        "e",
        "a",
        0.5,
    ));
    integrity.add_violation(MetamorphicViolation::new(
        "V2",
        MetamorphicRelationType::Additive,
        "d",
        "i",
        "e",
        "a",
        0.5,
    ));
    integrity.add_violation(MetamorphicViolation::new(
        "V3",
        MetamorphicRelationType::Identity,
        "d",
        "i",
        "e",
        "a",
        0.5,
    ));

    let by_type = integrity.violations_by_type();
    assert_eq!(by_type.get(&MetamorphicRelationType::Identity).expect("key should exist").len(), 2);
    assert_eq!(by_type.get(&MetamorphicRelationType::Additive).expect("key should exist").len(), 1);
}

#[test]
fn test_behavioral_integrity_most_severe_violation() {
    let mut integrity = BehavioralIntegrity::new(0.9, 0.9, 0.1, 0.9, "model");

    integrity.add_violation(MetamorphicViolation::new(
        "V1",
        MetamorphicRelationType::Identity,
        "d",
        "i",
        "e",
        "a",
        0.3,
    ));
    integrity.add_violation(MetamorphicViolation::new(
        "V2",
        MetamorphicRelationType::Identity,
        "d",
        "i",
        "e",
        "a",
        0.9,
    ));
    integrity.add_violation(MetamorphicViolation::new(
        "V3",
        MetamorphicRelationType::Identity,
        "d",
        "i",
        "e",
        "a",
        0.5,
    ));

    let most_severe = integrity.most_severe_violation().expect("operation should succeed");
    assert_eq!(most_severe.id, "V2");
}

#[test]
fn test_behavioral_integrity_assessment() {
    let excellent = BehavioralIntegrity::perfect("model");
    assert_eq!(excellent.assessment(), IntegrityAssessment::Excellent);

    let good = BehavioralIntegrity::new(0.8, 0.8, 0.1, 0.8, "model");
    assert_eq!(good.assessment(), IntegrityAssessment::Good);

    let fair = BehavioralIntegrity::new(0.6, 0.6, 0.3, 0.6, "model");
    assert_eq!(fair.assessment(), IntegrityAssessment::Fair);

    let poor = BehavioralIntegrity::new(0.3, 0.3, 0.5, 0.3, "model");
    assert_eq!(poor.assessment(), IntegrityAssessment::Poor);

    let mut critical = BehavioralIntegrity::perfect("model");
    critical.add_violation(MetamorphicViolation::new(
        "V1",
        MetamorphicRelationType::Identity,
        "d",
        "i",
        "e",
        "a",
        0.85,
    ));
    assert_eq!(critical.assessment(), IntegrityAssessment::Critical);
}

#[test]
fn test_integrity_assessment_display() {
    assert_eq!(format!("{}", IntegrityAssessment::Excellent), "Excellent");
    assert_eq!(format!("{}", IntegrityAssessment::Critical), "Critical");
}

#[test]
fn test_behavioral_integrity_summary() {
    let integrity = BehavioralIntegrity::perfect("model-v1").with_test_count(100);

    let summary = integrity.summary();
    assert!(summary.contains("model-v1"));
    assert!(summary.contains("100.0%"));
    assert!(summary.contains("Excellent"));
    assert!(summary.contains("PASS"));
}

#[test]
fn test_behavioral_integrity_builder() {
    let integrity = BehavioralIntegrityBuilder::new("model-v2")
        .equivalence_score(0.95)
        .syscall_match(0.90)
        .timing_variance(0.05)
        .semantic_equiv(0.92)
        .test_count(500)
        .build();

    assert_eq!(integrity.model_id, "model-v2");
    assert!((integrity.equivalence_score - 0.95).abs() < f64::EPSILON);
    assert_eq!(integrity.test_count, 500);
}

#[test]
fn test_behavioral_integrity_builder_with_violation() {
    let violation = MetamorphicViolation::new(
        "V001",
        MetamorphicRelationType::Identity,
        "desc",
        "input",
        "exp",
        "act",
        0.5,
    );

    let integrity = BehavioralIntegrityBuilder::new("model")
        .equivalence_score(0.9)
        .violation(violation)
        .build();

    assert_eq!(integrity.violations.len(), 1);
}

#[test]
fn test_behavioral_integrity_serialization() {
    let integrity = BehavioralIntegrity::new(0.9, 0.85, 0.1, 0.88, "model-v1");
    let json = serde_json::to_string(&integrity).expect("JSON serialization should succeed");
    let parsed: BehavioralIntegrity =
        serde_json::from_str(&json).expect("JSON deserialization should succeed");

    assert!((parsed.equivalence_score - integrity.equivalence_score).abs() < f64::EPSILON);
    assert_eq!(parsed.model_id, integrity.model_id);
}

#[test]
fn test_behavioral_integrity_has_critical_violations_empty() {
    let integrity = BehavioralIntegrity::perfect("model");
    assert!(!integrity.has_critical_violations());
}

#[test]
fn test_behavioral_integrity_has_critical_violations_minor_only() {
    let mut integrity = BehavioralIntegrity::perfect("model");
    integrity.add_violation(MetamorphicViolation::new(
        "V1",
        MetamorphicRelationType::Identity,
        "d",
        "i",
        "e",
        "a",
        0.3,
    ));
    assert!(!integrity.has_critical_violations());
}

#[test]
fn test_behavioral_integrity_most_severe_violation_empty() {
    let integrity = BehavioralIntegrity::perfect("model");
    assert!(integrity.most_severe_violation().is_none());
}

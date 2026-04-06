//! Tests for counterfactual explanations.

use super::*;

#[test]
fn test_feature_change_new() {
    let change = FeatureChange::new(0, 1.0, 2.0);
    assert_eq!(change.feature_idx, 0);
    assert_eq!(change.original_value, 1.0);
    assert_eq!(change.counterfactual_value, 2.0);
    assert_eq!(change.delta, 1.0);
}

#[test]
fn test_feature_change_with_name() {
    let change = FeatureChange::new(0, 1.0, 2.0).with_name("income");
    assert_eq!(change.feature_name, Some("income".to_string()));
}

#[test]
fn test_feature_change_abs_delta() {
    let change_pos = FeatureChange::new(0, 1.0, 2.0);
    let change_neg = FeatureChange::new(0, 2.0, 1.0);
    assert_eq!(change_pos.abs_delta(), 1.0);
    assert_eq!(change_neg.abs_delta(), 1.0);
}

#[test]
fn test_counterfactual_new() {
    let cf = Counterfactual::new(vec![1.0, 2.0, 3.0], 0, 0.9, vec![1.5, 2.0, 4.0], 1, 0.85);

    assert_eq!(cf.original_decision, 0);
    assert_eq!(cf.alternative_decision, 1);
    assert_eq!(cf.n_changes(), 2); // features 0 and 2 changed
    assert!(cf.is_valid());
}

#[test]
fn test_counterfactual_metrics() {
    let cf = Counterfactual::new(vec![0.0, 0.0], 0, 0.9, vec![3.0, 4.0], 1, 0.85);

    // L1 = |3| + |4| = 7
    assert!((cf.sparsity - 7.0).abs() < 1e-6);
    // L2 = sqrt(9 + 16) = 5
    assert!((cf.distance - 5.0).abs() < 1e-6);
}

#[test]
fn test_counterfactual_explain() {
    let cf = Counterfactual::new(vec![45000.0, 0.42], 0, 0.7, vec![52000.0, 0.35], 1, 0.8)
        .with_feature_names(&["income".to_string(), "debt_ratio".to_string()]);

    let explanation = cf.explain();
    assert!(explanation.contains("Original decision: 0"));
    assert!(explanation.contains("Alternative decision: 1"));
    assert!(explanation.contains("income"));
    assert!(explanation.contains("debt_ratio"));
}

#[test]
fn test_counterfactual_serialization_roundtrip() {
    let cf = Counterfactual::new(vec![1.0, 2.0, 3.0], 0, 0.9, vec![1.5, 2.0, 4.0], 1, 0.85)
        .with_feature_names(&[
            "feature_a".to_string(),
            "feature_b".to_string(),
            "feature_c".to_string(),
        ]);

    let bytes = cf.to_bytes();
    let restored = Counterfactual::from_bytes(&bytes).expect("Failed to deserialize");

    assert_eq!(cf.original_decision, restored.original_decision);
    assert_eq!(cf.alternative_decision, restored.alternative_decision);
    assert!((cf.original_confidence - restored.original_confidence).abs() < 1e-6);
    assert_eq!(cf.original_input.len(), restored.original_input.len());
    assert_eq!(cf.changes.len(), restored.changes.len());
    assert!((cf.sparsity - restored.sparsity).abs() < 1e-6);
    assert!((cf.distance - restored.distance).abs() < 1e-6);
}

#[test]
fn test_counterfactual_no_changes() {
    let cf = Counterfactual::new(
        vec![1.0, 2.0, 3.0],
        0,
        0.9,
        vec![1.0, 2.0, 3.0], // Same input
        0,                   // Same decision
        0.9,
    );

    assert_eq!(cf.n_changes(), 0);
    assert!(!cf.is_valid()); // Decision didn't flip
}

#[test]
fn test_counterfactual_error_display() {
    let err = CounterfactualError::InsufficientData { expected: 100, actual: 50 };
    assert!(err.to_string().contains("expected 100"));

    let err = CounterfactualError::VersionMismatch { expected: 1, actual: 2 };
    assert!(err.to_string().contains("Version mismatch"));
}

#[test]
fn test_counterfactual_insufficient_data() {
    let result = Counterfactual::from_bytes(&[0; 10]);
    assert!(matches!(result, Err(CounterfactualError::InsufficientData { .. })));
}

#[test]
fn test_counterfactual_version_mismatch() {
    let mut bytes = vec![2u8]; // Invalid version
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&0.0f32.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&0.0f32.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());

    let result = Counterfactual::from_bytes(&bytes);
    assert!(matches!(result, Err(CounterfactualError::VersionMismatch { .. })));
}

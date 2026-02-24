//! Falsification tests for classification-finetune-v1.yaml
//!
//! Per Popper (1959), each validation rule has explicit falsification criteria.
//! If ANY test here passes when it shouldn't, the contract is BROKEN.
//!
//! Contract: contracts/classification-finetune-v1.yaml
//! Tests: FALSIFY-CLASS-001..006

#![allow(clippy::unwrap_used)]

use super::validated_classification::{
    ValidatedClassLogits, ValidatedClassifierWeight, ValidatedSafetyLabel,
};

// =============================================================================
// FALSIFY-CLASS-001: Logit shape mismatch must fail
// =============================================================================

#[test]
fn falsify_class_001_logit_shape_mismatch() {
    // 3 elements but num_classes=5 -> must fail
    let bad = vec![0.1f32; 3];
    let result = ValidatedClassLogits::new(bad, 5);
    assert!(result.is_err(), "Must reject wrong logit count");
    assert!(
        result.unwrap_err().rule_id.contains("F-CLASS-001"),
        "Must cite F-CLASS-001"
    );
}

#[test]
fn falsify_class_001_logit_shape_too_many() {
    // 7 elements but num_classes=5 -> must fail
    let bad = vec![0.1f32; 7];
    let result = ValidatedClassLogits::new(bad, 5);
    assert!(result.is_err(), "Must reject too many logits");
}

#[test]
fn falsify_class_001_logit_shape_empty() {
    // 0 elements, num_classes=5 -> must fail
    let result = ValidatedClassLogits::new(vec![], 5);
    assert!(result.is_err(), "Must reject empty logits");
}

#[test]
fn falsify_class_001_logit_shape_correct() {
    // 5 elements, num_classes=5 -> must succeed
    let good = vec![0.1, 0.2, 0.3, 0.4, 0.5];
    let result = ValidatedClassLogits::new(good, 5);
    assert!(result.is_ok(), "Must accept correct logit shape");
}

// =============================================================================
// FALSIFY-CLASS-002: Label out of range must fail
// =============================================================================

#[test]
fn falsify_class_002_label_out_of_range() {
    // index=5 with num_classes=5 -> must fail (valid: 0..4)
    let result = ValidatedSafetyLabel::new(5, 5);
    assert!(result.is_err(), "Must reject index >= num_classes");
    assert!(
        result.unwrap_err().rule_id.contains("F-CLASS-002"),
        "Must cite F-CLASS-002"
    );
}

#[test]
fn falsify_class_002_label_way_out_of_range() {
    let result = ValidatedSafetyLabel::new(100, 5);
    assert!(result.is_err(), "Must reject index >> num_classes");
}

#[test]
fn falsify_class_002_label_boundary_valid() {
    // index=4 with num_classes=5 -> must succeed (last valid)
    let result = ValidatedSafetyLabel::new(4, 5);
    assert!(result.is_ok(), "Must accept last valid index");
    assert_eq!(result.unwrap().label(), "unsafe");
}

#[test]
fn falsify_class_002_label_zero_valid() {
    let result = ValidatedSafetyLabel::new(0, 5);
    assert!(result.is_ok(), "Must accept index 0");
    assert_eq!(result.unwrap().label(), "safe");
}

#[test]
fn falsify_class_002_all_labels_valid() {
    for i in 0..5 {
        let result = ValidatedSafetyLabel::new(i, 5);
        assert!(result.is_ok(), "Must accept index {i}");
    }
}

// =============================================================================
// FALSIFY-CLASS-003: Softmax sum invariant
// =============================================================================

#[test]
fn falsify_class_003_softmax_sum_invariant() {
    let logits = ValidatedClassLogits::new(vec![1.0, 2.0, -1.0, 0.5, 3.0], 5).unwrap();
    let probs = logits.softmax();
    let sum: f32 = probs.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "Softmax must sum to 1.0, got {sum}"
    );
}

#[test]
fn falsify_class_003_softmax_all_zeros() {
    let logits = ValidatedClassLogits::new(vec![0.0; 5], 5).unwrap();
    let probs = logits.softmax();
    let sum: f32 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "Softmax of zeros must sum to 1.0");
    // All equal -> uniform distribution
    for &p in &probs {
        assert!((p - 0.2).abs() < 1e-5, "Uniform softmax should be 0.2");
    }
}

#[test]
fn falsify_class_003_softmax_large_values() {
    // Large values shouldn't cause overflow thanks to max subtraction
    let logits = ValidatedClassLogits::new(vec![100.0, 200.0, 300.0, 400.0, 500.0], 5).unwrap();
    let probs = logits.softmax();
    let sum: f32 = probs.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "Softmax with large values must sum to 1.0, got {sum}"
    );
}

#[test]
fn falsify_class_003_softmax_negative_values() {
    let logits = ValidatedClassLogits::new(vec![-10.0, -20.0, -5.0, -1.0, -100.0], 5).unwrap();
    let probs = logits.softmax();
    let sum: f32 = probs.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "Softmax with negative values must sum to 1.0"
    );
}

// =============================================================================
// FALSIFY-CLASS-004: Classifier weight shape mismatch must fail
// =============================================================================

#[test]
fn falsify_class_004_weight_shape_mismatch() {
    // 100 elements but hidden_size=128, num_classes=5 needs 640
    let bad = vec![0.1f32; 100];
    let result = ValidatedClassifierWeight::new(bad, 128, 5);
    assert!(result.is_err(), "Must reject wrong weight shape");
    assert!(
        result.unwrap_err().rule_id.contains("F-CLASS-004"),
        "Must cite F-CLASS-004"
    );
}

#[test]
fn falsify_class_004_weight_shape_correct() {
    let good = vec![0.01f32; 896 * 5]; // hidden_size=896, num_classes=5
    let result = ValidatedClassifierWeight::new(good, 896, 5);
    assert!(result.is_ok(), "Must accept correct weight shape");
}

#[test]
fn falsify_class_004_weight_zero_hidden() {
    let result = ValidatedClassifierWeight::new(vec![], 0, 5);
    assert!(result.is_err(), "Must reject hidden_size=0");
}

// =============================================================================
// FALSIFY-CLASS-005: NaN logits must be rejected
// =============================================================================

#[test]
fn falsify_class_005_nan_logits_rejected() {
    let bad = vec![0.1, f32::NAN, 0.3, 0.4, 0.5];
    let result = ValidatedClassLogits::new(bad, 5);
    assert!(result.is_err(), "Must reject NaN in logits");
}

#[test]
fn falsify_class_005_inf_logits_rejected() {
    let bad = vec![0.1, 0.2, f32::INFINITY, 0.4, 0.5];
    let result = ValidatedClassLogits::new(bad, 5);
    assert!(result.is_err(), "Must reject Inf in logits");
}

#[test]
fn falsify_class_005_neg_inf_logits_rejected() {
    let bad = vec![0.1, 0.2, 0.3, f32::NEG_INFINITY, 0.5];
    let result = ValidatedClassLogits::new(bad, 5);
    assert!(result.is_err(), "Must reject -Inf in logits");
}

#[test]
fn falsify_class_005_nan_weight_rejected() {
    let mut bad = vec![0.01f32; 128 * 5];
    bad[42] = f32::NAN;
    let result = ValidatedClassifierWeight::new(bad, 128, 5);
    assert!(result.is_err(), "Must reject NaN in classifier weight");
}

#[test]
fn falsify_class_005_inf_weight_rejected() {
    let mut bad = vec![0.01f32; 128 * 5];
    bad[42] = f32::INFINITY;
    let result = ValidatedClassifierWeight::new(bad, 128, 5);
    assert!(result.is_err(), "Must reject Inf in classifier weight");
}

// =============================================================================
// FALSIFY-CLASS-006: Single-class classifier must be rejected
// =============================================================================

#[test]
fn falsify_class_006_single_class_logits_rejected() {
    let result = ValidatedClassLogits::new(vec![1.0], 1);
    assert!(result.is_err(), "Must reject num_classes < 2 for logits");
}

#[test]
fn falsify_class_006_single_class_weight_rejected() {
    let result = ValidatedClassifierWeight::new(vec![0.1; 128], 128, 1);
    assert!(result.is_err(), "Must reject num_classes < 2 for weight");
}

#[test]
fn falsify_class_006_binary_class_accepted() {
    // num_classes=2 is the minimum valid
    let result = ValidatedClassLogits::new(vec![0.1, 0.9], 2);
    assert!(result.is_ok(), "Must accept num_classes=2");
}

// =============================================================================
// INTEGRATION: predicted_class and display
// =============================================================================

#[test]
fn test_predicted_class_argmax() {
    let logits = ValidatedClassLogits::new(vec![0.1, 0.2, 5.0, 0.4, 0.5], 5).unwrap();
    assert_eq!(
        logits.predicted_class(),
        2,
        "Should pick index with max logit"
    );
}

#[test]
fn test_predicted_class_with_confidence() {
    let logits = ValidatedClassLogits::new(vec![-100.0, -100.0, 100.0, -100.0, -100.0], 5).unwrap();
    let (cls, conf) = logits.predicted_class_with_confidence();
    assert_eq!(cls, 2);
    assert!(
        (conf - 1.0).abs() < 1e-5,
        "Extreme logit should give ~100% confidence"
    );
}

#[test]
fn test_safety_label_display() {
    let label = ValidatedSafetyLabel::new(3, 5).unwrap();
    let s = format!("{label}");
    assert!(
        s.contains("non-idempotent"),
        "Display should show label name"
    );
    assert!(s.contains("3"), "Display should show index");
}

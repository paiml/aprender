//! Falsification tests for classification contract -- entrenar side
//!
//! Contract: aprender/contracts/classification-finetune-v1.yaml
//! Tests: FALSIFY-CLASS-ENT-001..004, proptest variants
//!
//! These tests validate that entrenar's ClassificationHead and loss functions
//! enforce the same contract rules as aprender's validated newtypes.

use super::classification::{cross_entropy_loss, ClassificationHead};
use crate::autograd::Tensor;

// =============================================================================
// FALSIFY-CLASS-ENT-001: Forward output shape == num_classes
// =============================================================================

#[test]
fn falsify_class_ent_001_forward_output_shape() {
    // ClassificationHead::new(64, 5) -> forward with 3 tokens -> output has exactly 5 elements
    let head = ClassificationHead::new(64, 5);
    let hidden = Tensor::from_vec(vec![0.1f32; 3 * 64], false);
    let logits = head.forward(&hidden, 3);
    assert_eq!(
        logits.len(),
        5,
        "F-CLASS-001: forward must produce exactly num_classes={} logits, got {}",
        head.num_classes(),
        logits.len()
    );
}

// =============================================================================
// FALSIFY-CLASS-ENT-001b: Correct dims produce correct output shape
// =============================================================================

#[test]
fn falsify_class_ent_001b_correct_dims_correct_shape() {
    // Various valid configurations all produce correct output shape
    for (hidden_size, num_classes, seq_len) in [(32, 3, 1), (128, 10, 5), (256, 2, 8)] {
        let head = ClassificationHead::new(hidden_size, num_classes);
        let hidden = Tensor::from_vec(vec![0.05f32; seq_len * hidden_size], false);
        let logits = head.forward(&hidden, seq_len);
        assert_eq!(
            logits.len(),
            num_classes,
            "F-CLASS-001: hidden_size={hidden_size}, num_classes={num_classes}, seq_len={seq_len} \
             must produce {num_classes} logits, got {}",
            logits.len()
        );
    }
}

// =============================================================================
// FALSIFY-CLASS-ENT-002: cross_entropy_loss with label == num_classes panics
// =============================================================================

#[test]
#[should_panic(expected = "F-CLASS-002")]
fn falsify_class_ent_002_label_out_of_range() {
    // label=5 with num_classes=5 -> out-of-range (valid: 0..4)
    let logits = Tensor::from_vec(vec![1.0, 2.0, -1.0, 0.5, 3.0], false);
    let _ = cross_entropy_loss(&logits, 5, 5);
}

// =============================================================================
// FALSIFY-CLASS-ENT-002b: cross_entropy_loss with label=4, num_classes=5 succeeds
// =============================================================================

#[test]
fn falsify_class_ent_002b_label_boundary_valid() {
    // label=4, num_classes=5 -> last valid index, must succeed
    let logits = Tensor::from_vec(vec![1.0, 2.0, -1.0, 0.5, 3.0], false);
    let loss = cross_entropy_loss(&logits, 4, 5);
    let loss_val = loss.data()[0];
    assert!(
        loss_val.is_finite(),
        "F-CLASS-005: boundary label=4 must produce finite loss, got {loss_val}"
    );
}

// =============================================================================
// FALSIFY-CLASS-ENT-003: ClassificationHead::new(0, 5) panics (F-CLASS-004)
// =============================================================================

#[test]
#[should_panic(expected = "F-CLASS-004")]
fn falsify_class_ent_003_hidden_size_zero() {
    let _ = ClassificationHead::new(0, 5);
}

// =============================================================================
// FALSIFY-CLASS-ENT-003b: ClassificationHead::new(64, 1) panics (num_classes < 2)
// =============================================================================

#[test]
#[should_panic(expected = "F-CLASS-004")]
fn falsify_class_ent_003b_num_classes_one() {
    let _ = ClassificationHead::new(64, 1);
}

// =============================================================================
// FALSIFY-CLASS-ENT-004: cross_entropy_loss with valid logits -> finite loss
// =============================================================================

#[test]
fn falsify_class_ent_004_cross_entropy_finite() {
    let logits = Tensor::from_vec(vec![1.0, 2.0, -1.0, 0.5, 3.0], false);
    let loss = cross_entropy_loss(&logits, 2, 5);
    let loss_val = loss.data()[0];
    assert!(loss_val.is_finite(), "F-CLASS-005: cross_entropy_loss must be finite, got {loss_val}");
    assert!(
        loss_val > 0.0,
        "Cross-entropy loss must be positive for non-dominant class, got {loss_val}"
    );
}

// =============================================================================
// PROPTEST: FALSIFY-CLASS-ENT-001-prop through FALSIFY-CLASS-ENT-004-prop
// =============================================================================

mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        // FALSIFY-CLASS-ENT-001-prop: random hidden_size, num_classes, seq_len
        // -> output always has num_classes elements.
        #[test]
        fn falsify_class_ent_001_prop(
            hidden_size in 8usize..=512,
            num_classes in 2usize..=20,
            seq_len in 1usize..=10,
        ) {
            let head = ClassificationHead::new(hidden_size, num_classes);
            let hidden = Tensor::from_vec(vec![0.01f32; seq_len * hidden_size], false);
            let logits = head.forward(&hidden, seq_len);
            prop_assert_eq!(
                logits.len(),
                num_classes,
                "F-CLASS-001-prop: hidden_size={}, num_classes={}, seq_len={} produced {} logits",
                hidden_size,
                num_classes,
                seq_len,
                logits.len()
            );
        }

        // FALSIFY-CLASS-ENT-003-prop: random valid hidden_size and num_classes
        // -> construction always succeeds (no panic).
        #[test]
        fn falsify_class_ent_003_prop(
            hidden_size in 1usize..=512,
            num_classes in 2usize..=20,
        ) {
            // This must not panic -- valid args always produce a valid head
            let head = ClassificationHead::new(hidden_size, num_classes);
            prop_assert_eq!(head.hidden_size(), hidden_size);
            prop_assert_eq!(head.num_classes(), num_classes);
            prop_assert_eq!(head.num_parameters(), hidden_size * num_classes + num_classes);
        }

        // FALSIFY-CLASS-ENT-004-prop: random logits + valid labels -> loss is finite.
        #[test]
        fn falsify_class_ent_004_prop(
            num_classes in 2usize..=20,
            label_offset in 0usize..20,
        ) {
            let label = label_offset % num_classes; // always valid
            // Use moderate logit values to avoid extreme numerical edge cases
            let logits_data: Vec<f32> = (0..num_classes)
                .map(|i| (i as f32 - num_classes as f32 / 2.0) * 0.5)
                .collect();
            let logits = Tensor::from_vec(logits_data, false);
            let loss = cross_entropy_loss(&logits, label, num_classes);
            let loss_val = loss.data()[0];
            prop_assert!(
                loss_val.is_finite(),
                "F-CLASS-005-prop: loss must be finite for num_classes={}, label={}, got {}",
                num_classes,
                label,
                loss_val
            );
        }
    }
}

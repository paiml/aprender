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
// FALSIFY-CLASS-007: Qwen3.5 must have use_bias=false (F-CLASS-007)
//
// Contract: classification-finetune-v1.yaml F-CLASS-007
// Prediction: TransformerConfig::qwen3_5_9b().use_bias == false
// If fails: LoRA adapters would wrongly create bias tensors for Qwen3.5
// =============================================================================

#[test]
fn falsify_class_007_qwen35_no_bias() {
    let config = entrenar::transformer::TransformerConfig::qwen3_5_9b();
    assert!(
        !config.use_bias,
        "FALSIFIED F-CLASS-007: Qwen3.5 must have use_bias=false, got true"
    );
}

#[test]
fn falsify_class_007_qwen2_has_bias() {
    // Counterexample: Qwen2 DOES have bias — confirms 007 is Qwen3.5-specific
    let config = entrenar::transformer::TransformerConfig::qwen2_0_5b();
    assert!(
        config.use_bias,
        "Qwen2 should have use_bias=true (verifies 007 is discriminating)"
    );
}

// =============================================================================
// FALSIFY-CLASS-008: LoRA must target Q/V projections (F-CLASS-008)
//
// Contract: classification-finetune-v1.yaml F-CLASS-008
// Prediction: LoRA adapters are placed on q_proj and v_proj (2 per layer)
// If fails: LoRA would target wrong projections, breaking fine-tuning
// =============================================================================

#[test]
fn falsify_class_008_lora_adapter_count_per_layer() {
    // Each transformer layer should have 2 LoRA adapters (Q, V)
    let model_config = entrenar::transformer::TransformerConfig::qwen2_0_5b();
    let classify_config = entrenar::finetune::ClassifyConfig::default();
    let pipeline = entrenar::finetune::ClassifyPipeline::new(&model_config, classify_config);
    let expected = model_config.num_hidden_layers * 2; // Q + V per layer
    assert_eq!(
        pipeline.lora_layers.len(),
        expected,
        "FALSIFIED F-CLASS-008: Expected {} LoRA adapters ({}*2), got {}",
        expected,
        model_config.num_hidden_layers,
        pipeline.lora_layers.len()
    );
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

// =============================================================================
// PROPTEST: FALSIFY-CLASS-001-prop through FALSIFY-CLASS-006-prop
// =============================================================================

mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        // FALSIFY-CLASS-001-prop: Random data.len() vs num_classes --
        // when len != num_classes, ValidatedClassLogits::new must fail.
        #[test]
        fn falsify_class_001_prop_shape_mismatch(
            data_len in 1usize..100,
            num_classes in 2usize..50,
        ) {
            // Only test mismatch cases
            prop_assume!(data_len != num_classes);
            let data = vec![0.1f32; data_len];
            let result = ValidatedClassLogits::new(data, num_classes);
            prop_assert!(
                result.is_err(),
                "Must reject logits with {} elements when num_classes={}",
                data_len,
                num_classes
            );
        }

        // FALSIFY-CLASS-002-prop: Random label vs num_classes --
        // when label >= num_classes, ValidatedSafetyLabel::new must fail.
        #[test]
        fn falsify_class_002_prop_label_out_of_range(
            label in 0usize..200,
            num_classes in 2usize..50,
        ) {
            let result = ValidatedSafetyLabel::new(label, num_classes);
            if label >= num_classes {
                prop_assert!(
                    result.is_err(),
                    "Must reject label={} when num_classes={}",
                    label,
                    num_classes
                );
            } else {
                // label < num_classes -- only valid if SafetyClass::from_index succeeds
                // For safety labels, only indices 0..5 map to valid SafetyClass variants
                if label < 5 {
                    prop_assert!(
                        result.is_ok(),
                        "Must accept label={} when num_classes={}",
                        label,
                        num_classes
                    );
                }
            }
        }

        // FALSIFY-CLASS-003-prop: Random finite logits -> softmax sums to ~1.0.
        #[test]
        fn falsify_class_003_prop_softmax_sum(
            num_classes in 2usize..50,
            seed in 0u64..10000,
        ) {
            // Generate deterministic logits from seed
            let data: Vec<f32> = (0..num_classes)
                .map(|i| {
                    let x = (seed.wrapping_mul(6364136223846793005).wrapping_add(i as u64)) as f32;
                    (x / f32::MAX).clamp(-50.0, 50.0)
                })
                .collect();
            let logits = ValidatedClassLogits::new(data, num_classes).unwrap();
            let probs = logits.softmax();
            let sum: f32 = probs.iter().sum();
            prop_assert!(
                (sum - 1.0).abs() < 1e-4,
                "Softmax must sum to ~1.0, got {} for {} classes",
                sum,
                num_classes
            );
        }

        // FALSIFY-CLASS-004-prop: Random weight sizes vs hidden*num_classes --
        // mismatch detected by ValidatedClassifierWeight.
        #[test]
        fn falsify_class_004_prop_weight_shape_mismatch(
            actual_len in 1usize..1000,
            hidden_size in 1usize..128,
            num_classes in 2usize..20,
        ) {
            let expected_len = hidden_size * num_classes;
            prop_assume!(actual_len != expected_len);
            let data = vec![0.01f32; actual_len];
            let result = ValidatedClassifierWeight::new(data, hidden_size, num_classes);
            prop_assert!(
                result.is_err(),
                "Must reject weight with {} elements when expected {} ({}x{})",
                actual_len,
                expected_len,
                hidden_size,
                num_classes
            );
        }

        // FALSIFY-CLASS-005-prop: Inject NaN or Inf at random positions --
        // detected by ValidatedClassLogits.
        #[test]
        fn falsify_class_005_prop_nan_inf_detection(
            num_classes in 2usize..20,
            poison_idx in 0usize..20,
            poison_type in 0u8..3,
        ) {
            let idx = poison_idx % num_classes;
            let mut data = vec![0.5f32; num_classes];
            data[idx] = match poison_type {
                0 => f32::NAN,
                1 => f32::INFINITY,
                _ => f32::NEG_INFINITY,
            };
            let result = ValidatedClassLogits::new(data, num_classes);
            prop_assert!(
                result.is_err(),
                "Must reject logits with NaN/Inf at index {} (poison_type={})",
                idx,
                poison_type
            );
        }

        // FALSIFY-CLASS-006-prop: num_classes < 2 -> rejection by ValidatedClassLogits.
        #[test]
        fn falsify_class_006_prop_degenerate_class_count(
            num_classes in 0usize..2,
        ) {
            let data = vec![1.0f32; num_classes];
            let result = ValidatedClassLogits::new(data, num_classes);
            prop_assert!(
                result.is_err(),
                "Must reject num_classes={} (< 2)",
                num_classes
            );
        }
    }
}

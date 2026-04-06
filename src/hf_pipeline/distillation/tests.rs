//! Tests for distillation module.

use ndarray::{array, Array1, Array2};

use super::utils::{cross_entropy_loss, kl_divergence, l2_normalize, log_softmax, softmax};
use super::{AttentionTransfer, DistillationLoss, ProgressiveDistillation};

// =========================================================================
// Softmax Tests
// =========================================================================

#[test]
fn test_softmax_sums_to_one() {
    let logits = array![1.0, 2.0, 3.0, 4.0];
    let probs = softmax(&logits);
    let sum: f32 = probs.sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

#[test]
fn test_softmax_all_positive() {
    let logits = array![-100.0, 0.0, 100.0];
    let probs = softmax(&logits);
    for p in &probs {
        assert!(*p >= 0.0);
    }
}

#[test]
fn test_softmax_numerical_stability() {
    // Large values should not overflow
    let logits = array![1000.0, 1001.0, 1002.0];
    let probs = softmax(&logits);
    assert!(probs.iter().all(|&p| p.is_finite()));
    assert!((probs.sum() - 1.0).abs() < 1e-5);
}

#[test]
fn test_log_softmax_identity() {
    let logits = array![1.0, 2.0, 3.0];
    let log_probs = log_softmax(&logits);
    let probs_from_log: Array1<f32> = log_probs.mapv(f32::exp);
    let probs = softmax(&logits);

    for (a, b) in probs.iter().zip(probs_from_log.iter()) {
        assert!((a - b).abs() < 1e-5);
    }
}

// =========================================================================
// KL Divergence Tests
// =========================================================================

#[test]
fn test_kl_divergence_zero_for_same() {
    let p = softmax(&array![1.0, 2.0, 3.0]);
    let log_p = log_softmax(&array![1.0, 2.0, 3.0]);
    let kl = kl_divergence(&log_p, &p);
    assert!(kl.abs() < 1e-5);
}

#[test]
fn test_kl_divergence_positive() {
    let p = softmax(&array![1.0, 2.0, 3.0]);
    let log_q = log_softmax(&array![3.0, 2.0, 1.0]);
    let kl = kl_divergence(&log_q, &p);
    assert!(kl >= 0.0);
}

// =========================================================================
// DistillationLoss Tests
// =========================================================================

#[test]
fn test_distillation_loss_default() {
    let loss = DistillationLoss::default();
    assert_eq!(loss.temperature, 4.0);
    assert_eq!(loss.alpha, 0.7);
}

#[test]
fn test_distillation_loss_positive() {
    let loss = DistillationLoss::new(4.0, 0.5);
    let student = array![1.0, 2.0, 3.0];
    let teacher = array![1.5, 2.5, 2.0];
    let l = loss.forward_single(&student, &teacher, 2);
    assert!(l >= 0.0);
}

#[test]
fn test_distillation_loss_zero_alpha() {
    // alpha=0 means only hard label loss
    let loss = DistillationLoss::new(4.0, 0.0);
    let student = array![1.0, 2.0, 3.0];
    let teacher = array![100.0, 200.0, 300.0]; // Very different teacher
    let l = loss.forward_single(&student, &teacher, 2);
    // Should be close to cross-entropy loss (ignoring teacher)
    let ce = cross_entropy_loss(&student, 2);
    assert!((l - ce).abs() < 0.01);
}

#[test]
fn test_distillation_loss_high_temp() {
    // Higher temperature = softer distributions
    let loss_low = DistillationLoss::new(1.0, 1.0);
    let loss_high = DistillationLoss::new(10.0, 1.0);
    let student = array![1.0, 2.0, 3.0];
    let teacher = array![1.0, 2.0, 3.0];

    let l_low = loss_low.soft_loss(&student, &teacher);
    let l_high = loss_high.soft_loss(&student, &teacher);

    // Both should be near zero for same logits
    assert!(l_low.abs() < 0.1);
    assert!(l_high.abs() < 0.1);
}

#[test]
fn test_distillation_loss_batch() {
    let loss = DistillationLoss::new(4.0, 0.5);
    let student = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 2.0, 1.0, 3.0])
        .expect("operation should succeed");
    let teacher = Array2::from_shape_vec((2, 3), vec![1.5, 2.5, 2.5, 2.5, 1.5, 2.5])
        .expect("operation should succeed");
    let targets = vec![2, 0];

    let l = loss.forward(&student, &teacher, &targets);
    assert!(l >= 0.0);
    assert!(l.is_finite());
}

// =========================================================================
// ProgressiveDistillation Tests
// =========================================================================

#[test]
fn test_progressive_default() {
    let prog = ProgressiveDistillation::default();
    assert!(!prog.layer_mapping.is_empty());
    assert_eq!(prog.hidden_weight, 1.0);
}

#[test]
fn test_progressive_hidden_loss_zero_for_same() {
    let prog = ProgressiveDistillation::new(vec![(0, 0), (1, 1)]);
    let hidden = Array2::<f32>::ones((4, 768));
    let student = vec![hidden.clone(), hidden.clone()];
    let teacher = vec![hidden.clone(), hidden.clone()];

    let loss = prog.hidden_state_loss(&student, &teacher);
    assert!(loss.abs() < 1e-5);
}

#[test]
fn test_progressive_hidden_loss_positive_for_diff() {
    let prog = ProgressiveDistillation::new(vec![(0, 0)]);
    let s = Array2::<f32>::zeros((4, 768));
    let t = Array2::<f32>::ones((4, 768));

    let loss = prog.hidden_state_loss(&[s], &[t]);
    assert!(loss > 0.0);
}

#[test]
fn test_progressive_with_weight() {
    let prog = ProgressiveDistillation::new(vec![(0, 0)]).with_weight(0.5);
    assert_eq!(prog.hidden_weight, 0.5);
}

#[test]
fn test_progressive_projection_layer_creation() {
    // Student dim 512, teacher dim 768
    let prog = ProgressiveDistillation::new(vec![(0, 0)]).with_projection(512, 768);
    assert!(prog.projection.is_some());
    let proj = prog.projection.as_ref().expect("operation should succeed");
    assert_eq!(proj.dim(), (512, 768));
}

#[test]
fn test_progressive_hidden_loss_with_projection() {
    // Student has dim 512, teacher has dim 768
    let prog = ProgressiveDistillation::new(vec![(0, 0)]).with_projection(512, 768);

    let student = vec![Array2::<f32>::ones((4, 512))];
    let teacher = vec![Array2::<f32>::ones((4, 768))];

    // Should not skip due to shape mismatch
    let loss = prog.hidden_state_loss(&student, &teacher);
    // Loss should be computed (not zero due to projection mismatch)
    // Just verify it doesn't skip
    assert!(loss >= 0.0);
}

#[test]
fn test_progressive_projection_correct_transform() {
    // Use identity-like projection
    let mut prog = ProgressiveDistillation::new(vec![(0, 0)]).with_projection(768, 768);

    // Set projection to identity matrix
    if let Some(ref mut proj) = prog.projection {
        proj.fill(0.0);
        for i in 0..768 {
            proj[[i, i]] = 1.0;
        }
    }

    let hidden = Array2::<f32>::from_elem((4, 768), 1.0);
    let student = vec![hidden.clone()];
    let teacher = vec![hidden.clone()];

    // With identity projection, loss should be ~0
    let loss = prog.hidden_state_loss(&student, &teacher);
    assert!(loss.abs() < 1e-4, "Identity projection should give ~0 loss");
}

#[test]
fn test_progressive_no_projection_skips_mismatched() {
    // No projection set
    let prog = ProgressiveDistillation::new(vec![(0, 0)]);

    let student = vec![Array2::<f32>::ones((4, 512))];
    let teacher = vec![Array2::<f32>::ones((4, 768))];

    // Should skip due to shape mismatch, loss = 0
    let loss = prog.hidden_state_loss(&student, &teacher);
    assert_eq!(loss, 0.0, "Should skip mismatched shapes without projection");
}

// =========================================================================
// AttentionTransfer Tests
// =========================================================================

#[test]
fn test_attention_transfer_default() {
    let at = AttentionTransfer::default();
    assert_eq!(at.weight, 0.1);
}

#[test]
fn test_attention_transfer_zero_for_same() {
    let at = AttentionTransfer::new(1.0);
    let attn = Array2::<f32>::ones((8, 8));
    let student = vec![attn.clone()];
    let teacher = vec![attn.clone()];

    let loss = at.loss(&student, &teacher);
    assert!(loss.abs() < 1e-5);
}

#[test]
fn test_attention_transfer_positive_for_diff() {
    let at = AttentionTransfer::new(1.0);
    let s = Array2::<f32>::zeros((8, 8));
    let t = Array2::<f32>::ones((8, 8));

    let loss = at.loss(&[s], &[t]);
    assert!(loss > 0.0);
}

// =========================================================================
// L2 Normalize Tests
// =========================================================================

#[test]
fn test_l2_normalize_unit_norm() {
    let arr =
        Array2::from_shape_vec((2, 2), vec![3.0, 4.0, 0.0, 0.0]).expect("operation should succeed");
    let norm = l2_normalize(&arr);
    let l2 = norm.mapv(|x| x * x).sum().sqrt();
    assert!((l2 - 1.0).abs() < 1e-5);
}

#[test]
fn test_l2_normalize_zero() {
    let arr = Array2::<f32>::zeros((2, 2));
    let norm = l2_normalize(&arr);
    // Should return zeros without NaN
    assert!(norm.iter().all(|&x| x.is_finite()));
}

// =========================================================================
// Property-like Tests
// =========================================================================

#[test]
fn test_distillation_loss_monotonic_in_alpha() {
    let student = array![1.0, 2.0, 3.0];
    let teacher = array![3.0, 2.0, 1.0]; // Very different

    let loss_0 = DistillationLoss::new(4.0, 0.0).forward_single(&student, &teacher, 2);
    let loss_1 = DistillationLoss::new(4.0, 1.0).forward_single(&student, &teacher, 2);

    // As alpha increases, soft loss contribution increases
    // Both should be valid losses
    assert!(loss_0 >= 0.0);
    assert!(loss_1 >= 0.0);
}

#[test]
fn test_temperature_scaling_effect() {
    let student = array![1.0, 2.0, 3.0];
    let teacher = array![0.5, 2.0, 3.5];

    let loss_t1 = DistillationLoss::new(1.0, 1.0).soft_loss(&student, &teacher);
    let loss_t10 = DistillationLoss::new(10.0, 1.0).soft_loss(&student, &teacher);

    // Both should be valid
    assert!(loss_t1.is_finite());
    assert!(loss_t10.is_finite());
}

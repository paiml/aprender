//! Loss function accuracy tests — verifies mathematical correctness
//! against reference implementations.
//!
//! Batuta: NR-12 (Loss Function Accuracy)

use crate::train::{CrossEntropyLoss, LossFn, MSELoss};
use crate::Tensor;

/// Reference softmax (f64 for higher precision)
fn reference_softmax_f64(logits: &[f32]) -> Vec<f64> {
    let logits_f64: Vec<f64> = logits.iter().map(|&x| f64::from(x)).collect();
    let max = logits_f64.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exp_vals: Vec<f64> = logits_f64.iter().map(|&x| (x - max).exp()).collect();
    let sum: f64 = exp_vals.iter().sum();
    exp_vals.iter().map(|&e| e / sum).collect()
}

/// Reference cross-entropy (f64 precision)
fn reference_cross_entropy_f64(logits: &[f32], target_idx: usize) -> f64 {
    let probs = reference_softmax_f64(logits);
    -probs[target_idx].max(1e-30).ln()
}

#[test]
fn loss_test_cross_entropy_3class() {
    let logits = vec![2.0_f32, 1.0, 0.5];
    let target_idx = 0;
    let reference = reference_cross_entropy_f64(&logits, target_idx) as f32;
    let ce = CrossEntropyLoss;
    let pred = Tensor::from_vec(logits, false);
    let mut one_hot = vec![0.0_f32; 3];
    one_hot[target_idx] = 1.0;
    let tgt = Tensor::from_vec(one_hot, false);
    let loss = ce.forward(&pred, &tgt);
    let actual = loss.data()[0];
    let diff = (actual - reference).abs();
    assert!(diff < 1e-5, "CE accuracy: actual={actual}, ref={reference}, diff={diff}");
}

#[test]
fn loss_test_mse_reference_loss() {
    let pred_vals = vec![1.0_f32, 2.0, 3.0, 4.0];
    let target_vals = vec![1.5_f32, 2.5, 2.5, 4.5];
    let reference: f64 = pred_vals
        .iter()
        .zip(&target_vals)
        .map(|(p, t)| (f64::from(*p) - f64::from(*t)).powi(2))
        .sum::<f64>()
        / pred_vals.len() as f64;
    let mse = MSELoss;
    let pred = Tensor::from_vec(pred_vals, false);
    let tgt = Tensor::from_vec(target_vals, false);
    let loss = mse.forward(&pred, &tgt);
    let diff = (f64::from(loss.data()[0]) - reference).abs();
    assert!(diff < 1e-6, "MSE accuracy: diff={diff}");
}

#[test]
fn loss_test_cross_entropy_expected_loss_10class() {
    let logits: Vec<f32> = (0..10).map(|i| (i as f32 - 5.0) * 0.5).collect();
    for target_idx in 0..10 {
        let reference = reference_cross_entropy_f64(&logits, target_idx) as f32;
        let ce = CrossEntropyLoss;
        let pred = Tensor::from_vec(logits.clone(), false);
        let mut one_hot = vec![0.0_f32; 10];
        one_hot[target_idx] = 1.0;
        let tgt = Tensor::from_vec(one_hot, false);
        let loss = ce.forward(&pred, &tgt);
        let actual = loss.data()[0];
        let diff = (actual - reference).abs();
        assert!(diff < 1e-4, "CE accuracy 10-class[{target_idx}]: diff={diff}");
    }
}

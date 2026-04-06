//! Loss function accuracy tests — verifies mathematical correctness
//! against reference implementations.
//!
//! Batuta: NR-12 (Loss Function Accuracy)
//! Contract: Loss functions match reference implementations within tolerance.

use entrenar::train::{BCEWithLogitsLoss, CausalLMLoss, CrossEntropyLoss, LossFn, MSELoss};
use entrenar::Tensor;

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
fn test_cross_entropy_matches_reference_3class() {
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
    assert!(diff < 1e-5, "CE mismatch: actual={actual}, reference={reference}, diff={diff}");
}

#[test]
fn test_cross_entropy_matches_reference_10class() {
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
        assert!(
            diff < 1e-4,
            "CE 10-class target={target_idx}: actual={actual}, reference={reference}, diff={diff}"
        );
    }
}

#[test]
fn test_cross_entropy_uniform_equals_log_c() {
    for num_classes in [2, 3, 5, 10, 20] {
        let logits = vec![0.0_f32; num_classes];
        let expected = (num_classes as f32).ln();

        let ce = CrossEntropyLoss;
        let pred = Tensor::from_vec(logits, false);
        let mut one_hot = vec![0.0_f32; num_classes];
        one_hot[0] = 1.0;
        let tgt = Tensor::from_vec(one_hot, false);
        let loss = ce.forward(&pred, &tgt);
        let actual = loss.data()[0];

        let diff = (actual - expected).abs();
        assert!(
            diff < 1e-5,
            "CE(uniform, C={num_classes}): actual={actual}, expected=log({num_classes})={expected}"
        );
    }
}

#[test]
fn test_cross_entropy_perfect_prediction_approaches_zero() {
    let ce = CrossEntropyLoss;

    // With dominant logit, CE should approach 0
    let logits = vec![100.0_f32, -100.0, -100.0];
    let pred = Tensor::from_vec(logits, false);
    let tgt = Tensor::from_vec(vec![1.0, 0.0, 0.0], false);
    let loss = ce.forward(&pred, &tgt);

    assert!(
        loss.data()[0] < 1e-10,
        "Perfect prediction CE should be near 0, got {}",
        loss.data()[0]
    );
}

#[test]
fn test_mse_loss_matches_reference() {
    let pred_vals = vec![1.0_f32, 2.0, 3.0, 4.0];
    let target_vals = vec![1.5_f32, 2.5, 2.5, 4.5];

    // Reference: mean((pred - target)^2)
    let reference: f32 =
        pred_vals.iter().zip(&target_vals).map(|(p, t)| (p - t) * (p - t)).sum::<f32>()
            / pred_vals.len() as f32;

    let mse = MSELoss;
    let pred = Tensor::from_vec(pred_vals, false);
    let tgt = Tensor::from_vec(target_vals, false);
    let loss = mse.forward(&pred, &tgt);
    let actual = loss.data()[0];

    let diff = (actual - reference).abs();
    assert!(diff < 1e-6, "MSE mismatch: actual={actual}, reference={reference}, diff={diff}");
}

#[test]
fn test_mse_loss_zero_for_identical() {
    let vals = vec![1.0_f32, 2.0, 3.0];
    let mse = MSELoss;
    let pred = Tensor::from_vec(vals.clone(), false);
    let tgt = Tensor::from_vec(vals, false);
    let loss = mse.forward(&pred, &tgt);

    assert!(loss.data()[0].abs() < 1e-10, "MSE of identical should be 0, got {}", loss.data()[0]);
}

#[test]
fn test_bce_with_logits_matches_reference() {
    let logits = vec![0.5_f32, -0.5, 1.0, -1.0];
    let targets = vec![1.0_f32, 0.0, 1.0, 0.0];

    // Reference: -mean(target * log(sigmoid(logit)) + (1-target) * log(1-sigmoid(logit)))
    // Using stable form: max(logit, 0) - logit * target + log(1 + exp(-|logit|))
    let reference: f32 = logits
        .iter()
        .zip(&targets)
        .map(|(&z, &t)| z.max(0.0) - z * t + (1.0 + (-z.abs()).exp()).ln())
        .sum::<f32>()
        / logits.len() as f32;

    let bce = BCEWithLogitsLoss;
    let pred = Tensor::from_vec(logits, false);
    let tgt = Tensor::from_vec(targets, false);
    let loss = bce.forward(&pred, &tgt);
    let actual = loss.data()[0];

    let diff = (actual - reference).abs();
    assert!(diff < 1e-5, "BCE mismatch: actual={actual}, reference={reference}, diff={diff}");
}

#[test]
fn test_causal_lm_loss_finite_for_valid_input() {
    let vocab_size = 100;
    let seq_len = 10;

    // Create logits: seq_len * vocab_size values
    let logits: Vec<f32> = (0..seq_len * vocab_size).map(|i| ((i as f32) * 0.01).sin()).collect();

    // Create targets: token IDs as floats (CausalLMLoss expects seq_len target values)
    let targets: Vec<f32> = (0..seq_len).map(|s| (s % vocab_size) as f32).collect();

    let lm_loss = CausalLMLoss::new(vocab_size);
    let pred = Tensor::from_vec(logits, false);
    let tgt = Tensor::from_vec(targets, false);
    let loss = lm_loss.forward(&pred, &tgt);

    assert!(loss.data()[0].is_finite(), "CausalLM loss should be finite, got {}", loss.data()[0]);
    assert!(loss.data()[0] > 0.0, "CausalLM loss should be positive, got {}", loss.data()[0]);
}

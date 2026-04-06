//! Utility functions for distillation computations.
//!
//! Provides numerically stable softmax, log-softmax, KL divergence,
//! cross-entropy loss, and L2 normalization.

use ndarray::{Array1, Array2};

/// Softmax with numerical stability
pub(crate) fn softmax(logits: &Array1<f32>) -> Array1<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp: Array1<f32> = logits.mapv(|x| (x - max).exp());
    let sum = exp.sum();
    exp / sum
}

/// Log softmax with numerical stability
pub(crate) fn log_softmax(logits: &Array1<f32>) -> Array1<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let shifted = logits.mapv(|x| x - max);
    let log_sum_exp = shifted.mapv(f32::exp).sum().max(f32::MIN_POSITIVE).ln();
    shifted.mapv(|x| x - log_sum_exp)
}

/// KL divergence: KL(P || Q) = sum(P * log(P/Q))
pub(crate) fn kl_divergence(log_q: &Array1<f32>, p: &Array1<f32>) -> f32 {
    // KL(P || Q) = sum(P * (log(P) - log(Q)))
    // Since we have log(Q), we compute: sum(P * log(P)) - sum(P * log(Q))
    let p_log_p: f32 = p
        .iter()
        .map(|&pi| if pi > 1e-10 { pi * pi.max(f32::MIN_POSITIVE).ln() } else { 0.0 })
        .sum();
    let p_log_q: f32 = p.iter().zip(log_q.iter()).map(|(&pi, &lqi)| pi * lqi).sum();
    p_log_p - p_log_q
}

/// Cross-entropy loss
pub(crate) fn cross_entropy_loss(logits: &Array1<f32>, target: usize) -> f32 {
    let log_probs = log_softmax(logits);
    -log_probs[target]
}

/// L2 normalize a 2D array
pub(crate) fn l2_normalize(arr: &Array2<f32>) -> Array2<f32> {
    let norm = arr.mapv(|x| x * x).sum().sqrt();
    if norm > 1e-10 {
        arr / norm
    } else {
        arr.clone()
    }
}

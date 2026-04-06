//! Normalization correctness tests — verifies RMSNorm and LayerNorm
//! against reference implementations.
//!
//! Batuta: NR-14 (Normalization Layer Correctness)

use super::layer_norm;
use crate::autograd::Tensor;

/// Reference LayerNorm (f64 precision)
fn reference_layer_norm_f64(x: &[f32], gamma: &[f32], beta: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len() as f64;
    let x_f64: Vec<f64> = x.iter().map(|&v| f64::from(v)).collect();
    let mean: f64 = x_f64.iter().sum::<f64>() / n;
    let variance: f64 = x_f64.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / n;
    let std = (variance + f64::from(eps)).sqrt();
    x_f64
        .iter()
        .enumerate()
        .map(|(i, &v)| ((v - mean) / std * f64::from(gamma[i]) + f64::from(beta[i])) as f32)
        .collect()
}

#[test]
fn norm_test_matches_reference() {
    let x_data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
    let gamma_data = vec![1.0_f32; 5];
    let beta_data = vec![0.0_f32; 5];
    let eps = 1e-5;
    let reference = reference_layer_norm_f64(&x_data, &gamma_data, &beta_data, eps);
    let x = Tensor::from_vec(x_data, false);
    let gamma = Tensor::from_vec(gamma_data, false);
    let beta = Tensor::from_vec(beta_data, false);
    let result = layer_norm(&x, &gamma, &beta, eps);
    for (i, (&actual, &expected)) in result.data().iter().zip(reference.iter()).enumerate() {
        let diff = (actual - expected).abs();
        assert!(diff < 1e-5, "Correctness[{i}]: actual={actual}, ref={expected}, diff={diff}");
    }
}

#[test]
fn norm_test_mean_zero() {
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], false);
    let gamma = Tensor::from_vec(vec![1.0; 5], false);
    let beta = Tensor::from_vec(vec![0.0; 5], false);
    let result = layer_norm(&x, &gamma, &beta, 1e-5);
    let mean: f32 = result.data().iter().sum::<f32>() / result.len() as f32;
    assert!(mean.abs() < 1e-5, "Output mean should be ~0, got {mean}");
}

#[test]
fn norm_test_variance_one() {
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], false);
    let gamma = Tensor::from_vec(vec![1.0; 5], false);
    let beta = Tensor::from_vec(vec![0.0; 5], false);
    let result = layer_norm(&x, &gamma, &beta, 1e-5);
    let mean: f32 = result.data().iter().sum::<f32>() / result.len() as f32;
    let variance: f32 =
        result.data().iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / result.len() as f32;
    assert!((variance - 1.0).abs() < 1e-4, "Output variance should be ~1, got {variance}");
}

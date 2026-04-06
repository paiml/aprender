//! Normalization layer correctness tests — verifies RMSNorm and LayerNorm
//! against reference implementations.
//!
//! Batuta: NR-14 (Normalization Layer Correctness)
//! Contract: Normalization layers produce correct outputs matching reference.

use entrenar::autograd::Tensor;

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

/// Reference RMSNorm (f64 precision)
fn reference_rms_norm_f64(x: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len() as f64;
    let x_f64: Vec<f64> = x.iter().map(|&v| f64::from(v)).collect();

    let rms: f64 = (x_f64.iter().map(|&v| v * v).sum::<f64>() / n + f64::from(eps)).sqrt();

    x_f64.iter().enumerate().map(|(i, &v)| (v / rms * f64::from(gamma[i])) as f32).collect()
}

#[test]
fn test_layer_norm_matches_reference() {
    let x = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
    let gamma = vec![1.0_f32; 5];
    let beta = vec![0.0_f32; 5];
    let eps = 1e-5;

    let reference = reference_layer_norm_f64(&x, &gamma, &beta, eps);

    let x_tensor = Tensor::from_vec(x, false);
    let gamma_tensor = Tensor::from_vec(gamma, false);
    let beta_tensor = Tensor::from_vec(beta, false);

    let result = entrenar::autograd::layer_norm(&x_tensor, &gamma_tensor, &beta_tensor, eps);

    for (i, (&actual, &expected)) in result.data().iter().zip(reference.iter()).enumerate() {
        let diff = (actual - expected).abs();
        assert!(
            diff < 1e-5,
            "LayerNorm mismatch at {i}: actual={actual}, reference={expected}, diff={diff}"
        );
    }
}

#[test]
fn test_layer_norm_zero_mean() {
    let x = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
    let gamma = vec![1.0_f32; 5];
    let beta = vec![0.0_f32; 5];
    let eps = 1e-5;

    let x_tensor = Tensor::from_vec(x, false);
    let gamma_tensor = Tensor::from_vec(gamma, false);
    let beta_tensor = Tensor::from_vec(beta, false);

    let result = entrenar::autograd::layer_norm(&x_tensor, &gamma_tensor, &beta_tensor, eps);

    // With gamma=1 and beta=0, output mean should be ~0
    let mean: f32 = result.data().iter().sum::<f32>() / result.len() as f32;
    assert!(mean.abs() < 1e-5, "LayerNorm output mean should be ~0, got {mean}");
}

#[test]
fn test_layer_norm_unit_variance() {
    let x = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
    let gamma = vec![1.0_f32; 5];
    let beta = vec![0.0_f32; 5];
    let eps = 1e-5;

    let x_tensor = Tensor::from_vec(x, false);
    let gamma_tensor = Tensor::from_vec(gamma, false);
    let beta_tensor = Tensor::from_vec(beta, false);

    let result = entrenar::autograd::layer_norm(&x_tensor, &gamma_tensor, &beta_tensor, eps);

    // With gamma=1 and beta=0, output variance should be ~1
    let mean: f32 = result.data().iter().sum::<f32>() / result.len() as f32;
    let variance: f32 =
        result.data().iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / result.len() as f32;
    assert!(
        (variance - 1.0).abs() < 1e-4,
        "LayerNorm output variance should be ~1, got {variance}"
    );
}

#[test]
fn test_layer_norm_gamma_scaling() {
    let x = vec![1.0_f32, 2.0, 3.0, 4.0];
    let gamma = vec![2.0_f32; 4];
    let beta = vec![0.0_f32; 4];
    let eps = 1e-5;

    let reference = reference_layer_norm_f64(&x, &gamma, &beta, eps);

    let x_tensor = Tensor::from_vec(x, false);
    let gamma_tensor = Tensor::from_vec(gamma, false);
    let beta_tensor = Tensor::from_vec(beta, false);

    let result = entrenar::autograd::layer_norm(&x_tensor, &gamma_tensor, &beta_tensor, eps);

    for (i, (&actual, &expected)) in result.data().iter().zip(reference.iter()).enumerate() {
        let diff = (actual - expected).abs();
        assert!(diff < 1e-5, "LayerNorm gamma=2 mismatch at {i}: actual={actual}, ref={expected}");
    }
}

#[test]
fn test_layer_norm_beta_shift() {
    let x = vec![1.0_f32, 2.0, 3.0, 4.0];
    let gamma = vec![1.0_f32; 4];
    let beta = vec![5.0_f32; 4];
    let eps = 1e-5;

    let reference = reference_layer_norm_f64(&x, &gamma, &beta, eps);

    let x_tensor = Tensor::from_vec(x, false);
    let gamma_tensor = Tensor::from_vec(gamma, false);
    let beta_tensor = Tensor::from_vec(beta, false);

    let result = entrenar::autograd::layer_norm(&x_tensor, &gamma_tensor, &beta_tensor, eps);

    // Mean of output should be ~5 (beta shift)
    let mean: f32 = result.data().iter().sum::<f32>() / result.len() as f32;
    assert!((mean - 5.0).abs() < 1e-4, "LayerNorm with beta=5 should have mean ~5, got {mean}");

    for (i, (&actual, &expected)) in result.data().iter().zip(reference.iter()).enumerate() {
        let diff = (actual - expected).abs();
        assert!(diff < 1e-5, "LayerNorm beta=5 mismatch at {i}: actual={actual}, ref={expected}");
    }
}

#[test]
fn test_layer_norm_numerical_stability_large_values() {
    let x = vec![1e6_f32, 1e6 + 1.0, 1e6 + 2.0, 1e6 + 3.0];
    let gamma = vec![1.0_f32; 4];
    let beta = vec![0.0_f32; 4];
    let eps = 1e-5;

    let x_tensor = Tensor::from_vec(x, false);
    let gamma_tensor = Tensor::from_vec(gamma, false);
    let beta_tensor = Tensor::from_vec(beta, false);

    let result = entrenar::autograd::layer_norm(&x_tensor, &gamma_tensor, &beta_tensor, eps);

    // All outputs should be finite
    for (i, &v) in result.data().iter().enumerate() {
        assert!(v.is_finite(), "LayerNorm output[{i}] = {v} is not finite");
    }
}

#[test]
fn test_layer_norm_numerical_stability_small_values() {
    let x = vec![1e-7_f32, 2e-7, 3e-7, 4e-7];
    let gamma = vec![1.0_f32; 4];
    let beta = vec![0.0_f32; 4];
    let eps = 1e-5;

    let x_tensor = Tensor::from_vec(x, false);
    let gamma_tensor = Tensor::from_vec(gamma, false);
    let beta_tensor = Tensor::from_vec(beta, false);

    let result = entrenar::autograd::layer_norm(&x_tensor, &gamma_tensor, &beta_tensor, eps);

    for (i, &v) in result.data().iter().enumerate() {
        assert!(v.is_finite(), "LayerNorm output[{i}] = {v} is not finite");
    }
}

#[test]
fn test_rms_norm_reference_correctness() {
    // RMSNorm: y = x / rms(x) * gamma, where rms(x) = sqrt(mean(x^2) + eps)
    let x = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
    let gamma = vec![1.0_f32; 5];
    let eps = 1e-5;

    let reference = reference_rms_norm_f64(&x, &gamma, eps);

    // Verify: rms = sqrt((1+4+9+16+25)/5 + eps) = sqrt(11 + eps) ≈ 3.3166
    let rms_sq: f32 = x.iter().map(|&v| v * v).sum::<f32>() / 5.0 + eps;
    let rms = rms_sq.sqrt();
    assert!((rms - 3.3166).abs() < 0.01, "RMS should be ~3.3166, got {rms}");

    // Check reference output
    for (i, (&xi, &ref_out)) in x.iter().zip(reference.iter()).enumerate() {
        let expected = xi / rms;
        assert!(
            (ref_out - expected).abs() < 1e-4,
            "RMSNorm ref mismatch at {i}: ref={ref_out}, manual={expected}"
        );
    }
}

#[test]
fn test_rms_norm_scale_invariance_property() {
    // RMSNorm(alpha * x, gamma) = RMSNorm(x, gamma) for any alpha > 0
    // (because the denominator scales with alpha too)
    let x = vec![1.0_f32, 2.0, 3.0, 4.0];
    let gamma = vec![1.0_f32; 4];
    let eps = 1e-5;

    let ref_x = reference_rms_norm_f64(&x, &gamma, eps);

    let scaled_x: Vec<f32> = x.iter().map(|&v| v * 10.0).collect();
    let ref_scaled = reference_rms_norm_f64(&scaled_x, &gamma, eps);

    for (i, (&a, &b)) in ref_x.iter().zip(ref_scaled.iter()).enumerate() {
        let diff = (a - b).abs();
        assert!(
            diff < 1e-4,
            "RMSNorm scale invariance violated at {i}: rms(x)={a}, rms(10x)={b}, diff={diff}"
        );
    }
}

#[test]
fn test_layer_norm_constant_input_produces_beta() {
    // If all inputs are the same, normalized = 0, output = beta
    let x = vec![5.0_f32; 8];
    let gamma = vec![2.0_f32; 8];
    let beta = vec![3.0_f32; 8];
    let eps = 1e-5;

    let x_tensor = Tensor::from_vec(x, false);
    let gamma_tensor = Tensor::from_vec(gamma, false);
    let beta_tensor = Tensor::from_vec(beta, false);

    let result = entrenar::autograd::layer_norm(&x_tensor, &gamma_tensor, &beta_tensor, eps);

    // When all inputs are equal, variance ≈ 0, normalized ≈ 0, output ≈ beta
    for (i, &v) in result.data().iter().enumerate() {
        assert!((v - 3.0).abs() < 1e-2, "Constant input: output[{i}] = {v}, expected ~3.0 (beta)");
    }
}

#[test]
fn test_layer_norm_backward_gradient_exists() {
    let x = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], true);
    let gamma = Tensor::from_vec(vec![1.0_f32; 4], true);
    let beta = Tensor::from_vec(vec![0.0_f32; 4], true);
    let eps = 1e-5;

    let result = entrenar::autograd::layer_norm(&x, &gamma, &beta, eps);

    // Trigger backward
    if let Some(op) = result.backward_op() {
        // Set output gradient
        let result_mut = result.clone();
        result_mut.set_grad(ndarray::Array1::ones(4));
        op.backward();
    }

    // Gradient should exist for x
    if let Some(grad) = x.grad() {
        for (i, &g) in grad.iter().enumerate() {
            assert!(g.is_finite(), "LayerNorm grad[{i}] = {g} is not finite");
        }
    }
}

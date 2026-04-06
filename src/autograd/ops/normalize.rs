//! Normalization autograd operations: layer_norm

use crate::autograd::{BackwardOp, Tensor};
use ndarray::Array1;
use std::cell::RefCell;
use std::rc::Rc;

/// Layer Normalization
///
/// Normalizes input to have mean=0 and variance=1, then applies learned scale (gamma) and shift (beta)
/// LayerNorm(x) = gamma * (x - mean) / sqrt(var + epsilon) + beta
pub fn layer_norm(x: &Tensor, gamma: &Tensor, beta: &Tensor, epsilon: f32) -> Tensor {
    let n = x.len() as f32;

    // Compute mean
    let mean = x.data().sum() / n;

    // Compute variance
    let variance = x.data().mapv(|val| (val - mean).powi(2)).sum() / n;
    let std = (variance + epsilon).sqrt();

    // Normalize
    let normalized = x.data().mapv(|val| (val - mean) / std);

    // Scale and shift
    let data = &normalized * gamma.data() + beta.data();

    let requires_grad = x.requires_grad() || gamma.requires_grad() || beta.requires_grad();
    let mut result = Tensor::new(data, requires_grad);

    if requires_grad {
        let x_clone = x.clone();
        let gamma_clone = gamma.clone();
        let beta_clone = beta.clone();
        let backward_op = Rc::new(LayerNormBackward {
            x: x_clone,
            gamma: gamma_clone,
            beta: beta_clone,
            normalized: normalized.clone(),
            std,
            result_grad: result.grad_cell(),
        });
        result.set_backward_op(backward_op);
    }

    contract_post_layernorm!(result.data().as_slice().unwrap_or(&[]));
    result
}

struct LayerNormBackward {
    x: Tensor,
    gamma: Tensor,
    beta: Tensor,
    normalized: Array1<f32>,
    std: f32,
    result_grad: Rc<RefCell<Option<Array1<f32>>>>,
}

impl BackwardOp for LayerNormBackward {
    fn backward(&self) {
        if let Some(grad_output) = self.result_grad.borrow().as_ref() {
            let n = self.x.len() as f32;

            // ∂L/∂beta = ∂L/∂y (gradient flows directly through addition)
            if self.beta.requires_grad() {
                self.beta.accumulate_grad(grad_output.clone());
            }

            // ∂L/∂gamma = ∂L/∂y * x_normalized
            if self.gamma.requires_grad() {
                let grad_gamma = grad_output * &self.normalized;
                self.gamma.accumulate_grad(grad_gamma);
            }

            // ∂L/∂x is complex due to mean and variance dependencies
            if self.x.requires_grad() {
                // Gradient through scale: grad_normalized = grad_output * gamma
                let grad_normalized = grad_output * self.gamma.data();

                // Sum of gradients (for mean term)
                let sum_grad = grad_normalized.sum();

                // Sum of gradients weighted by normalized values (for variance term)
                let sum_grad_normalized = (&grad_normalized * &self.normalized).sum();

                // Full gradient formula:
                // ∂L/∂x_i = (1/std) * [grad_normalized_i - (1/n)*sum_grad - (1/n)*normalized_i*sum_grad_normalized]
                let grad_x: Vec<f32> = grad_normalized
                    .iter()
                    .zip(self.normalized.iter())
                    .map(|(&grad_norm, &norm)| {
                        (grad_norm - sum_grad / n - norm * sum_grad_normalized / n) / self.std
                    })
                    .collect();

                self.x.accumulate_grad(Array1::from(grad_x));
            }

            // Continue backward through the graph
            if let Some(op) = self.x.backward_op() {
                op.backward();
            }
            if let Some(op) = self.gamma.backward_op() {
                op.backward();
            }
            if let Some(op) = self.beta.backward_op() {
                op.backward();
            }
        }
    }
}

// =========================================================================
// FALSIFY-LN: layernorm-kernel-v1.yaml contract (entrenar layer_norm)
//
// Five-Whys (PMAT-354, Phase 10):
//   Why 1: entrenar had zero FALSIFY-LN-* tests despite a full LayerNorm impl
//   Why 2: autograd backward tests verify gradients, not output invariants
//   Why 3: no mapping from layernorm-kernel-v1.yaml to entrenar tests
//   Why 4: entrenar predates the provable-contracts YAML convention
//   Why 5: LayerNorm was "obviously correct" (y = (x-μ)/σ * γ + β)
//
// References:
//   - provable-contracts/contracts/layernorm-kernel-v1.yaml
//   - Ba et al. (2016) "Layer Normalization"
// =========================================================================

#[cfg(test)]
mod normalization_correctness_tests {
    use super::*;
    use crate::autograd::Tensor;

    /// Reference LayerNorm (f64 precision) for correctness verification
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
    fn test_normalization_correctness_matches_reference() {
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
            assert!(
                diff < 1e-5,
                "LayerNorm correctness[{i}]: actual={actual}, ref={expected}, diff={diff}"
            );
        }
    }

    #[test]
    fn test_normalization_correctness_with_scaling() {
        let x_data = vec![1.0_f32, 2.0, 3.0, 4.0];
        let gamma_data = vec![2.0_f32; 4];
        let beta_data = vec![1.0_f32; 4];
        let eps = 1e-5;
        let reference = reference_layer_norm_f64(&x_data, &gamma_data, &beta_data, eps);
        let x = Tensor::from_vec(x_data, false);
        let gamma = Tensor::from_vec(gamma_data, false);
        let beta = Tensor::from_vec(beta_data, false);
        let result = layer_norm(&x, &gamma, &beta, eps);
        for (i, (&actual, &expected)) in result.data().iter().zip(reference.iter()).enumerate() {
            let diff = (actual - expected).abs();
            assert!(diff < 1e-5, "LayerNorm correctness scaled[{i}]: diff={diff}");
        }
    }
}

#[cfg(test)]
mod ln_contract_tests {
    use super::*;
    use crate::autograd::Tensor;

    fn make_unit_params(dim: usize) -> (Tensor, Tensor) {
        let gamma = Tensor::from_vec(vec![1.0; dim], false);
        let beta = Tensor::from_vec(vec![0.0; dim], false);
        (gamma, beta)
    }

    /// FALSIFY-LN-001: Centering — mean of LN output ≈ 0 (with beta=0)
    #[test]
    fn falsify_ln_001_centering() {
        let (gamma, beta) = make_unit_params(8);
        let data = vec![1.0, -2.0, 3.0, 0.5, -1.5, 2.5, -0.5, 1.5];
        let x = Tensor::from_vec(data, false);
        let y = layer_norm(&x, &gamma, &beta, 1e-5);

        let mean: f32 = y.data().sum() / y.len() as f32;
        assert!(mean.abs() < 1e-5, "FALSIFIED LN-001: mean(LN(x)) = {mean}, expected ≈ 0");
    }

    /// FALSIFY-LN-002: Standardization — variance of LN output ≈ 1 (with gamma=1)
    #[test]
    fn falsify_ln_002_standardization() {
        let (gamma, beta) = make_unit_params(8);
        let data = vec![1.0, -2.0, 3.0, 0.5, -1.5, 2.5, -0.5, 1.5];
        let x = Tensor::from_vec(data, false);
        let y = layer_norm(&x, &gamma, &beta, 1e-5);
        let y_data = y.data();
        let n = y.len() as f32;

        let mean: f32 = y_data.sum() / n;
        let var: f32 = y_data.mapv(|v| (v - mean).powi(2)).sum() / n;
        assert!((var - 1.0).abs() < 0.05, "FALSIFIED LN-002: var(LN(x)) = {var}, expected ≈ 1.0");
    }

    /// FALSIFY-LN-003: Denominator safety — output finite for all finite input
    #[test]
    fn falsify_ln_003_denominator_safety() {
        let (gamma, beta) = make_unit_params(4);
        let test_cases: Vec<(&str, Vec<f32>)> = vec![
            ("normal", vec![1.0, 2.0, 3.0, 4.0]),
            ("small", vec![1e-7, 1e-7, 1e-7, 1e-7]),
            ("large", vec![1e6, 1e6, 1e6, 1e6]),
            ("mixed_sign", vec![-3.0, 2.0, -1.0, 4.0]),
            ("near_zero", vec![1e-20, 0.0, 1e-20, 0.0]),
            ("all_zero", vec![0.0, 0.0, 0.0, 0.0]),
        ];

        for (name, data) in &test_cases {
            let x = Tensor::from_vec(data.clone(), false);
            let y = layer_norm(&x, &gamma, &beta, 1e-5);
            for (i, &val) in y.data().iter().enumerate() {
                assert!(
                    val.is_finite(),
                    "FALSIFIED LN-003: output[{i}] = {val} not finite for case '{name}'"
                );
            }
        }
    }

    /// FALSIFY-LN-005: Idempotency — LN(LN(x)) ≈ LN(x)
    #[test]
    fn falsify_ln_005_idempotency() {
        let (gamma, beta) = make_unit_params(6);
        let x = Tensor::from_vec(vec![10.0, -5.0, 3.0, 7.0, -2.0, 0.5], false);
        let y1 = layer_norm(&x, &gamma, &beta, 1e-5);
        let y2 = layer_norm(&y1, &gamma, &beta, 1e-5);

        for (i, (&a, &b)) in y1.data().iter().zip(y2.data().iter()).enumerate() {
            let diff = (a - b).abs();
            assert!(
                diff < 1e-4,
                "FALSIFIED LN-005: LN(LN(x))[{i}] = {b}, LN(x)[{i}] = {a}, diff = {diff}"
            );
        }
    }

    /// FALSIFY-LN-006: Shift invariance — LN(x + c) = LN(x)
    #[test]
    fn falsify_ln_006_shift_invariance() {
        let (gamma, beta) = make_unit_params(5);
        let data = vec![1.0, -2.0, 3.0, 0.5, -1.5];
        let x = Tensor::from_vec(data.clone(), false);
        let y_base = layer_norm(&x, &gamma, &beta, 1e-5);

        for &c in &[10.0_f32, -100.0, 0.001, 1000.0] {
            let shifted: Vec<f32> = data.iter().map(|&v| v + c).collect();
            let x_shifted = Tensor::from_vec(shifted, false);
            let y_shifted = layer_norm(&x_shifted, &gamma, &beta, 1e-5);

            for (i, (&a, &b)) in y_base.data().iter().zip(y_shifted.data().iter()).enumerate() {
                let tol = 1e-3 * a.abs().max(1.0);
                assert!(
                    (a - b).abs() < tol,
                    "FALSIFIED LN-006: LN(x)[{i}]={a}, LN(x+{c})[{i}]={b}"
                );
            }
        }
    }

    /// FALSIFY-LN-007: Constant input → output ≈ beta (0)
    #[test]
    fn falsify_ln_007_constant_input() {
        let (gamma, beta) = make_unit_params(4);
        for &c in &[0.0_f32, 1.0, -5.0, 1e6, 1e-6] {
            let x = Tensor::from_vec(vec![c; 4], false);
            let y = layer_norm(&x, &gamma, &beta, 1e-5);

            for (i, &val) in y.data().iter().enumerate() {
                assert!(val.is_finite(), "FALSIFIED LN-003 (via LN-007): NaN/Inf for constant {c}");
                assert!(
                    val.abs() < 1e-3,
                    "FALSIFIED LN-007: LN([{c};4])[{i}] = {val}, expected ≈ 0"
                );
            }
        }
    }

    mod ln_proptest_falsify {
        use super::*;
        use proptest::prelude::*;

        // LN-001-prop: centering
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(200))]
            #[test]
            fn falsify_ln_001_prop_centering(
                dim in prop::sample::select(vec![4_usize, 8, 16, 32, 64]),
                scale in 0.01_f32..100.0,
            ) {
                let (gamma, beta) = make_unit_params(dim);
                let data: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.37 * scale).sin() * scale).collect();
                let x = Tensor::from_vec(data, false);
                let y = layer_norm(&x, &gamma, &beta, 1e-5);

                let mean: f32 = y.data().sum() / dim as f32;
                prop_assert!(
                    mean.abs() < 1e-4,
                    "FALSIFIED LN-001-prop: mean(LN(x)) = {} (d={}, scale={})",
                    mean, dim, scale
                );
            }
        }

        // LN-002-prop: standardization
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(200))]
            #[test]
            fn falsify_ln_002_prop_standardization(
                dim in prop::sample::select(vec![8_usize, 16, 32, 64]),
                scale in 0.1_f32..100.0,
            ) {
                let (gamma, beta) = make_unit_params(dim);
                let data: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.23).sin() * scale).collect();
                let x = Tensor::from_vec(data, false);
                let y = layer_norm(&x, &gamma, &beta, 1e-5);
                let y_data = y.data();
                let n = dim as f32;

                let mean: f32 = y_data.sum() / n;
                let var: f32 = y_data.mapv(|v| (v - mean).powi(2)).sum() / n;
                prop_assert!(
                    (var - 1.0).abs() < 0.1,
                    "FALSIFIED LN-002-prop: var(LN(x)) = {} (d={}, scale={})",
                    var, dim, scale
                );
            }
        }

        // LN-006-prop: shift invariance
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(100))]
            #[test]
            fn falsify_ln_006_prop_shift_invariance(
                dim in prop::sample::select(vec![4_usize, 8, 16, 32]),
                shift in prop::sample::select(vec![-100.0_f32, -1.0, 0.5, 10.0, 1000.0]),
            ) {
                let (gamma, beta) = make_unit_params(dim);
                let data: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.37).sin() * 5.0).collect();
                let x = Tensor::from_vec(data.clone(), false);
                let y_base = layer_norm(&x, &gamma, &beta, 1e-5);

                let shifted: Vec<f32> = data.iter().map(|&v| v + shift).collect();
                let x_shifted = Tensor::from_vec(shifted, false);
                let y_shifted = layer_norm(&x_shifted, &gamma, &beta, 1e-5);

                for (i, (&a, &b)) in y_base.data().iter().zip(y_shifted.data().iter()).enumerate() {
                    let tol = 1e-3 * a.abs().max(1.0);
                    prop_assert!(
                        (a - b).abs() < tol,
                        "FALSIFIED LN-006-prop: LN(x)[{i}]={a}, LN(x+{shift})[{i}]={b} (d={dim})"
                    );
                }
            }
        }

        // LN-007-prop: constant input
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(100))]
            #[test]
            fn falsify_ln_007_prop_constant_input(
                dim in prop::sample::select(vec![4_usize, 8, 16, 32]),
                c in prop::sample::select(vec![-1e6_f32, -1.0, 0.0, 1.0, 1e6]),
            ) {
                let (gamma, beta) = make_unit_params(dim);
                let x = Tensor::from_vec(vec![c; dim], false);
                let y = layer_norm(&x, &gamma, &beta, 1e-5);

                for (i, &val) in y.data().iter().enumerate() {
                    prop_assert!(
                        val.is_finite(),
                        "FALSIFIED LN-003-prop: NaN/Inf at [{i}] for constant {c} (d={dim})"
                    );
                    prop_assert!(
                        val.abs() < 1e-3,
                        "FALSIFIED LN-007-prop: LN([{c};{dim}])[{i}] = {val} (expected ≈ 0)"
                    );
                }
            }
        }
    }
}

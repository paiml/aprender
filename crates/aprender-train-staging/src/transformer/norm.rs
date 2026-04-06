//! RMS Normalization module
//!
//! This module provides RMS normalization layers for transformer models.

use crate::autograd::scale;
use crate::Tensor;
use std::collections::HashMap;

/// RMS Normalization layer
pub struct RMSNorm {
    /// Weight (scale) parameter
    pub weight: Tensor,
    /// Epsilon for numerical stability
    eps: f32,
}

impl RMSNorm {
    /// Create new RMS normalization layer
    pub fn new(hidden_size: usize, eps: f32) -> Self {
        Self { weight: Tensor::ones(hidden_size, true), eps }
    }

    /// Create from parameters
    ///
    /// # Contract (PMAT-332 norm)
    /// Validates weight.len() == hidden_size.
    /// Returns None if key is missing or length is wrong.
    pub fn from_params(
        params: &HashMap<String, Tensor>,
        prefix: &str,
        eps: f32,
        hidden_size: usize,
    ) -> Option<Self> {
        let weight = params.get(&format!("{prefix}.weight"))?.clone();
        if weight.len() != hidden_size {
            eprintln!(
                "[PMAT-332] {prefix}.weight: length mismatch — got {}, expected {hidden_size}",
                weight.len()
            );
            return None;
        }
        Some(Self { weight, eps })
    }

    /// Forward pass
    ///
    /// RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight
    pub fn forward(&self, x: &Tensor) -> Tensor {
        contract_pre_rmsnorm!(x.data());
        let n = x.len() as f32;

        // Compute RMS
        let sq_sum: f32 = x.data().iter().map(|v| v * v).sum();
        let rms = (sq_sum / n + self.eps).sqrt();

        // Normalize and scale
        let normalized = scale(x, 1.0 / rms);
        let result = crate::autograd::mul(&normalized, &self.weight);
        contract_post_rmsnorm!(result.data().as_slice().unwrap_or(&[]));
        result
    }

    /// Forward pass for batched input
    ///
    /// # Arguments
    /// * `x` - Input tensor (seq_len * hidden_size, flattened)
    /// * `seq_len` - Sequence length
    /// * `hidden_size` - Hidden dimension
    pub fn forward_batched(&self, x: &Tensor, seq_len: usize, hidden_size: usize) -> Tensor {
        let mut output = vec![0.0; seq_len * hidden_size];
        let mut rms_values = Vec::with_capacity(seq_len);

        // KAIZEN-020: Hoist data borrows outside the sequence loop.
        // Previously x.data().as_slice() was called per-position (seq_len times)
        // and self.weight.data()[i] was called per-element (seq_len * hidden_size times).
        let x_data = x.data();
        let x_slice = x_data.as_slice().expect("norm input must be contiguous");
        let w_data = self.weight.data();
        let w_slice = w_data.as_slice().expect("norm weight must be contiguous");

        for s in 0..seq_len {
            let start = s * hidden_size;
            let end = start + hidden_size;
            let slice = &x_slice[start..end];

            // Compute RMS for this position
            let sq_sum: f32 = slice.iter().map(|v| v * v).sum();
            let rms = (sq_sum / hidden_size as f32 + self.eps).sqrt();
            rms_values.push(rms);

            // Normalize and scale
            for (i, &val) in slice.iter().enumerate() {
                output[start + i] = (val / rms) * w_slice[i];
            }
        }

        let requires_grad = x.requires_grad() || self.weight.requires_grad();
        let mut result = Tensor::from_vec(output, requires_grad);

        if requires_grad {
            use crate::autograd::BackwardOp;
            use ndarray::Array1;
            use std::cell::RefCell;
            use std::rc::Rc;

            struct RMSNormBatchedBackward {
                x: Tensor,
                weight: Tensor,
                rms_values: Vec<f32>,
                seq_len: usize,
                hidden_size: usize,
                result_grad: Rc<RefCell<Option<Array1<f32>>>>,
            }

            impl BackwardOp for RMSNormBatchedBackward {
                fn backward(&self) {
                    if let Some(grad_output) = self.result_grad.borrow().as_ref() {
                        let h = self.hidden_size;
                        let x_data = self.x.data();
                        let x_sl = x_data.as_slice().expect("x contiguous");
                        let w_data = self.weight.data();
                        let w_sl = w_data.as_slice().expect("weight contiguous");
                        let go = grad_output.as_slice().expect("grad contiguous");

                        if self.x.requires_grad() {
                            // dx[s,j] = (go[s,j]*w[j] - x[s,j]*c_s) / rms_s
                            // c_s = sum_i(go[s,i]*w[i]*x[s,i]) / (n * rms_s^2)
                            let mut grad_x = vec![0.0_f32; self.seq_len * h];
                            let n = h as f32;

                            for s in 0..self.seq_len {
                                let off = s * h;
                                let rms = self.rms_values[s];

                                let mut dot = 0.0_f32;
                                for i in 0..h {
                                    dot += go[off + i] * w_sl[i] * x_sl[off + i];
                                }
                                let c = dot / (n * rms * rms);

                                for j in 0..h {
                                    grad_x[off + j] =
                                        (go[off + j] * w_sl[j] - x_sl[off + j] * c) / rms;
                                }
                            }

                            self.x.accumulate_grad(Array1::from(grad_x));
                        }

                        if self.weight.requires_grad() {
                            // dw[i] = sum_s(go[s,i] * x[s,i] / rms_s)
                            let mut grad_w = vec![0.0_f32; h];

                            for s in 0..self.seq_len {
                                let off = s * h;
                                let rms = self.rms_values[s];
                                for i in 0..h {
                                    grad_w[i] += go[off + i] * x_sl[off + i] / rms;
                                }
                            }

                            self.weight.accumulate_grad(Array1::from(grad_w));
                        }

                        // Continue backward propagation through inputs
                        if let Some(op) = self.x.backward_op() {
                            op.backward();
                        }
                        if let Some(op) = self.weight.backward_op() {
                            op.backward();
                        }
                    }
                }
            }

            let backward_op = Rc::new(RMSNormBatchedBackward {
                x: x.clone(),
                weight: self.weight.clone(),
                rms_values,
                seq_len,
                hidden_size,
                result_grad: result.grad_cell(),
            });
            result.set_backward_op(backward_op);
        }

        contract_post_rmsnorm!(result.data().as_slice().unwrap_or(&[]));
        result
    }
}

/// Layer Normalization with bias (used by BERT/RoBERTa/CodeBERT encoders).
///
/// Unlike RMSNorm (used by decoders), LayerNorm:
/// 1. Subtracts the mean (re-centering)
/// 2. Divides by standard deviation (not RMS)
/// 3. Has both weight (gamma) AND bias (beta) parameters
///
/// LayerNorm(x) = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias
///
/// # Contract (ENC-005)
/// - Output has zero mean and unit variance (before affine transform)
/// - weight.len() == bias.len() == hidden_size
pub struct LayerNorm {
    /// Scale parameter (gamma)
    pub weight: Tensor,
    /// Shift parameter (beta)
    pub bias: Tensor,
    /// Epsilon for numerical stability
    eps: f32,
    /// Hidden size
    hidden_size: usize,
}

impl LayerNorm {
    /// Create new LayerNorm (weight=1, bias=0)
    pub fn new(hidden_size: usize, eps: f32) -> Self {
        Self {
            weight: Tensor::ones(hidden_size, true),
            bias: Tensor::from_vec(vec![0.0; hidden_size], true),
            eps,
            hidden_size,
        }
    }

    /// Create from pre-trained parameters
    pub fn from_params(
        params: &HashMap<String, Tensor>,
        prefix: &str,
        eps: f32,
        hidden_size: usize,
    ) -> Option<Self> {
        let weight = params.get(&format!("{prefix}.weight"))?.clone();
        let bias = params.get(&format!("{prefix}.bias"))?.clone();
        if weight.len() != hidden_size || bias.len() != hidden_size {
            eprintln!(
                "[ENC-005] {prefix}: shape mismatch — weight={}, bias={}, expected {hidden_size}",
                weight.len(),
                bias.len()
            );
            return None;
        }
        Some(Self { weight, bias, eps, hidden_size })
    }

    /// Forward pass for batched input (seq_len positions, each of hidden_size)
    pub fn forward_batched(&self, x: &Tensor, seq_len: usize, hidden_size: usize) -> Tensor {
        let mut output = vec![0.0_f32; seq_len * hidden_size];
        let x_data = x.data();
        let x_slice = x_data.as_slice().expect("input contiguous");
        let w_data = self.weight.data();
        let w_slice = w_data.as_slice().expect("weight contiguous");
        let b_data = self.bias.data();
        let b_slice = b_data.as_slice().expect("bias contiguous");

        for s in 0..seq_len {
            let start = s * hidden_size;
            let end = start + hidden_size;
            let row = &x_slice[start..end];

            // Mean
            let mean: f32 = row.iter().sum::<f32>() / hidden_size as f32;

            // Variance
            let var: f32 =
                row.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / hidden_size as f32;
            let inv_std = 1.0 / (var + self.eps).sqrt();

            // Normalize, scale, shift
            for (i, &val) in row.iter().enumerate() {
                output[start + i] = (val - mean) * inv_std * w_slice[i] + b_slice[i];
            }
        }

        let result = Tensor::from_vec(output, x.requires_grad() || self.weight.requires_grad());
        contract_post_layernorm!(result.data().as_slice().unwrap_or(&[]));
        result
    }

    /// Get hidden size
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rms_norm_forward() {
        let norm = RMSNorm::new(4, 1e-6);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], true);
        let output = norm.forward(&x);
        assert_eq!(output.len(), 4);
        // Output should be normalized and scaled
        let data = output.data();
        assert!(data.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_rms_norm_batched() {
        let norm = RMSNorm::new(4, 1e-6);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], true);
        let output = norm.forward_batched(&x, 2, 4);
        assert_eq!(output.len(), 8);
    }

    #[test]
    fn test_rms_norm_normalization_property() {
        let norm = RMSNorm::new(4, 1e-6);
        let x = Tensor::from_vec(vec![2.0, 2.0, 2.0, 2.0], true);
        let output = norm.forward(&x);
        // After RMS normalization, if weights are 1, output should be x / rms(x)
        // rms(x) = sqrt(mean(x^2)) = sqrt(4) = 2
        // so output = [2/2, 2/2, 2/2, 2/2] = [1, 1, 1, 1]
        let data = output.data();
        for &val in data {
            assert!((val - 1.0).abs() < 1e-5, "Expected ~1.0, got {val}");
        }
    }

    #[test]
    fn test_rms_norm_with_zeros() {
        let norm = RMSNorm::new(4, 1e-6);
        let x = Tensor::from_vec(vec![0.0, 0.0, 0.0, 0.0], true);
        let output = norm.forward(&x);
        // With zeros input and eps, output should be finite (zeros)
        let data = output.data();
        assert!(data.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_rms_norm_weight_requires_grad() {
        let norm = RMSNorm::new(4, 1e-6);
        assert!(norm.weight.requires_grad());
    }

    #[test]
    fn test_rms_norm_from_params() {
        let mut params = HashMap::new();
        params.insert("test.weight".to_string(), Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], true));
        let norm = RMSNorm::from_params(&params, "test", 1e-6, 4);
        assert!(norm.is_some());
        let norm = norm.expect("operation should succeed");
        assert_eq!(norm.weight.len(), 4);
    }

    #[test]
    fn test_rms_norm_from_params_missing() {
        let params: HashMap<String, Tensor> = HashMap::new();
        let norm = RMSNorm::from_params(&params, "missing", 1e-6, 4);
        assert!(norm.is_none());
    }

    #[test]
    fn test_rms_norm_backward_gradient_exists() {
        let norm = RMSNorm::new(8, 1e-6);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], true);
        let mut output = norm.forward(&x);

        let grad_out = ndarray::Array1::ones(8);
        crate::autograd::backward(&mut output, Some(grad_out));

        assert!(norm.weight.grad().is_some());
        let grad = norm.weight.grad().expect("gradient should be available");
        assert!(grad.iter().all(|&v| v.is_finite()));
    }

    /// ALB-038 fix: forward_batched must propagate gradients (was creating tensors with no backward op)
    #[test]
    fn test_rms_norm_batched_backward_gradient_exists() {
        let norm = RMSNorm::new(4, 1e-6);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], true);
        let mut output = norm.forward_batched(&x, 2, 4);

        let grad_out = ndarray::Array1::ones(8);
        crate::autograd::backward(&mut output, Some(grad_out));

        // Weight must receive gradient (was broken before ALB-038 fix)
        assert!(norm.weight.grad().is_some(), "ALB-038: norm weight must have gradient");
        let wgrad = norm.weight.grad().expect("gradient available");
        assert!(wgrad.iter().all(|&v| v.is_finite()), "Weight gradients must be finite");
        assert!(wgrad.iter().any(|&v| v.abs() > 1e-10), "Weight gradients must be non-zero");

        // Input must receive gradient (enables gradient flow through model)
        assert!(x.grad().is_some(), "ALB-038: input x must have gradient");
        let xgrad = x.grad().expect("gradient available");
        assert!(xgrad.iter().all(|&v| v.is_finite()), "Input gradients must be finite");
        assert!(xgrad.iter().any(|&v| v.abs() > 1e-10), "Input gradients must be non-zero");
    }

    /// ALB-038 fix: batched backward produces correct weight gradients
    ///
    /// Note: forward() uses scale(x, 1/rms) which treats rms as constant w.r.t. x,
    /// giving an approximate input gradient. forward_batched() computes the exact
    /// RMSNorm gradient including d(rms)/d(x). Weight gradients match exactly since
    /// dL/dw_i = go_i * x_i / rms regardless.
    #[test]
    fn test_rms_norm_batched_backward_weight_grad_matches() {
        let hidden = 4;
        let data = vec![1.0_f32, -2.0, 3.0, -0.5];

        // Non-batched path (uses autograd ops)
        let norm1 = RMSNorm::new(hidden, 1e-6);
        let x1 = Tensor::from_vec(data.clone(), true);
        let mut out1 = norm1.forward(&x1);
        crate::autograd::backward(&mut out1, Some(ndarray::Array1::ones(hidden)));
        let wgrad1 = norm1.weight.grad().expect("gradient available");

        // Batched path (new backward op)
        let norm2 = RMSNorm::new(hidden, 1e-6);
        let x2 = Tensor::from_vec(data, true);
        let mut out2 = norm2.forward_batched(&x2, 1, hidden);
        crate::autograd::backward(&mut out2, Some(ndarray::Array1::ones(hidden)));
        let wgrad2 = norm2.weight.grad().expect("gradient available");

        // Weight gradients should match exactly (dw = go * x / rms)
        for i in 0..hidden {
            assert!(
                (wgrad1[i] - wgrad2[i]).abs() < 1e-5,
                "Weight grad mismatch at [{i}]: unbatched={}, batched={}",
                wgrad1[i],
                wgrad2[i]
            );
        }
    }

    // =========================================================================
    // ENC-005: LayerNorm tests
    // =========================================================================

    #[test]
    fn enc_005_layernorm_output_shape() {
        let ln = LayerNorm::new(8, 1e-5);
        let x = Tensor::from_vec(vec![1.0; 3 * 8], true);
        let output = ln.forward_batched(&x, 3, 8);
        assert_eq!(output.len(), 3 * 8);
    }

    #[test]
    fn enc_005_layernorm_zero_mean_unit_var() {
        let ln = LayerNorm::new(8, 1e-12);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], true);
        let output = ln.forward_batched(&x, 1, 8);
        let data = output.data();
        let slice = data.as_slice().expect("contiguous");

        // With weight=1, bias=0: output should have ~zero mean, ~unit variance
        let mean: f32 = slice.iter().sum::<f32>() / 8.0;
        assert!(mean.abs() < 1e-5, "LayerNorm output mean={mean}, expected ~0");

        let var: f32 = slice.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / 8.0;
        assert!((var - 1.0).abs() < 0.01, "LayerNorm output var={var}, expected ~1");
    }

    #[test]
    fn enc_005_layernorm_with_bias() {
        let mut ln = LayerNorm::new(4, 1e-12);
        // Set bias to shift output by 5.0
        ln.bias = Tensor::from_vec(vec![5.0; 4], true);
        let x = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], true);
        let output = ln.forward_batched(&x, 1, 4);
        // Constant input → normalized to 0, then shifted by bias → all 5.0
        let data = output.data();
        for &v in data {
            assert!((v - 5.0).abs() < 1e-3, "Expected ~5.0 with bias, got {v}");
        }
    }

    #[test]
    fn enc_005_layernorm_from_params() {
        let mut params = HashMap::new();
        params.insert("ln.weight".to_string(), Tensor::from_vec(vec![1.0; 32], true));
        params.insert("ln.bias".to_string(), Tensor::from_vec(vec![0.0; 32], true));
        let ln = LayerNorm::from_params(&params, "ln", 1e-5, 32);
        assert!(ln.is_some());
        assert_eq!(ln.expect("should succeed").hidden_size(), 32);
    }

    #[test]
    fn enc_005_layernorm_from_params_rejects_mismatch() {
        let mut params = HashMap::new();
        params.insert("ln.weight".to_string(), Tensor::from_vec(vec![1.0; 32], true));
        params.insert("ln.bias".to_string(), Tensor::from_vec(vec![0.0; 16], true)); // wrong size
        let ln = LayerNorm::from_params(&params, "ln", 1e-5, 32);
        assert!(ln.is_none());
    }

    #[test]
    fn enc_005_layernorm_finite_output() {
        let ln = LayerNorm::new(4, 1e-5);
        let x = Tensor::from_vec(vec![1e6, -1e6, 0.0, 1.0], true);
        let output = ln.forward_batched(&x, 1, 4);
        assert!(output.data().iter().all(|v| v.is_finite()));
    }

    // =========================================================================
    // FALSIFY-N: §2.1.5-6 Layer Norms — Five-Whys Gap Analysis (Refs PMAT-332)
    //
    // Contract: tensor-layout-v1.yaml §tensors.input_layernorm/post_attention_layernorm/final_norm
    //   apr_shape: "[hidden]"
    //   transpose: "false"
    //   kernel: "element-wise multiply"
    //
    // Five-Whys:
    //   Why 1: from_params accepts ANY tensor length without validation
    //   Why 2: RMSNorm stores raw Tensor with no length check
    //   Why 3: Wrong-length norm produces wrong-scale hidden states
    //   Why 4: Mismatched norm length panics at element-wise multiply
    //   Why 5: No constructor-time length check exists
    //
    // Popper (1959): "These tests attempt to falsify the claim that
    // entrenar's norm handling prevents shape-related runtime errors."
    // =========================================================================

    /// FALSIFY-N1e: from_params rejects wrong-length norm weight (PMAT-332 norm fix)
    ///
    /// RMSNorm.from_params now validates weight.len() == hidden_size.
    /// A wrong-length weight is rejected at construction time.
    #[test]
    fn falsify_n1e_from_params_rejects_wrong_length_norm() {
        let mut params = HashMap::new();
        // WRONG: weight has 7 elements, should be hidden_size=4
        params.insert("test.weight".to_string(), Tensor::from_vec(vec![1.0; 7], true));
        let norm = RMSNorm::from_params(&params, "test", 1e-6, 4);
        // FIXED: from_params now rejects wrong-length weight
        assert!(
            norm.is_none(),
            "FALSIFY-N1e: PMAT-332 fix — from_params MUST reject wrong-length norm weight"
        );
    }

    /// FALSIFY-N2e: RMSNorm forward produces finite output for valid input
    #[test]
    fn falsify_n2e_norm_output_finite() {
        let norm = RMSNorm::new(8, 1e-6);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], true);
        let output = norm.forward(&x);
        assert!(
            output.data().iter().all(|v| v.is_finite()),
            "FALSIFY-N2e: RMSNorm output must be finite for valid input"
        );
    }

    /// FALSIFY-N3e: RMSNorm weight length matches hidden_size when constructed via new()
    #[test]
    fn falsify_n3e_new_constructor_correct_length() {
        let hidden_sizes = [64, 128, 256, 896, 4096];
        for &hidden in &hidden_sizes {
            let norm = RMSNorm::new(hidden, 1e-6);
            assert_eq!(
                norm.weight.len(),
                hidden,
                "FALSIFY-N3e: RMSNorm::new({hidden}) weight must have {hidden} elements"
            );
        }
    }

    /// FALSIFY-N4e: Batched forward preserves sequence*hidden dimension
    #[test]
    fn falsify_n4e_batched_forward_preserves_dims() {
        let hidden = 8;
        let seq_len = 4;
        let norm = RMSNorm::new(hidden, 1e-6);
        let x = Tensor::from_vec(vec![0.5; seq_len * hidden], true);
        let output = norm.forward_batched(&x, seq_len, hidden);
        assert_eq!(
            output.len(),
            seq_len * hidden,
            "FALSIFY-N4e: Batched norm must preserve seq_len * hidden dimension"
        );
        assert!(
            output.data().iter().all(|v| v.is_finite()),
            "FALSIFY-N4e: Batched norm output must be finite"
        );
    }

    /// FALSIFY-N5e: RMSNorm handles extreme but finite input values
    ///
    /// Very large values should still produce finite output due to normalization.
    #[test]
    fn falsify_n5e_extreme_input_still_finite() {
        let norm = RMSNorm::new(4, 1e-6);
        let x = Tensor::from_vec(vec![1e30, -1e30, 1e30, -1e30], true);
        let output = norm.forward(&x);
        assert!(
            output.data().iter().all(|v| v.is_finite()),
            "FALSIFY-N5e: RMSNorm must handle extreme values without Inf/NaN"
        );
    }

    // =========================================================================
    // FALSIFY-RN: rmsnorm-kernel-v1.yaml contract (entrenar RMSNorm)
    //
    // Five-Whys (PMAT-354):
    //   Why 1: entrenar had 10+ RMSNorm tests but zero FALSIFY-RN-* tagged tests
    //   Why 2: existing tests verify API behavior, not mathematical invariants
    //   Why 3: no mapping from rmsnorm-kernel-v1.yaml to entrenar test names
    //   Why 4: entrenar predates the provable-contracts YAML convention
    //   Why 5: norm was "obviously correct" (divide by RMS, multiply by weight)
    //
    // References:
    //   - provable-contracts/contracts/rmsnorm-kernel-v1.yaml
    //   - Zhang & Sennrich (2019) "Root Mean Square Layer Normalization"
    // =========================================================================

    /// FALSIFY-RN-001: Finiteness — output must be finite for all finite input
    ///
    /// Contract: |RMSNorm(x)_i| < ∞ for all i when ε > 0
    #[test]
    fn falsify_rn_001_finiteness() {
        let norm = RMSNorm::new(4, 1e-6);

        let test_cases: Vec<(&str, Vec<f32>)> = vec![
            ("normal", vec![1.0, 2.0, 3.0, 4.0]),
            ("small", vec![1e-7, 1e-7, 1e-7, 1e-7]),
            ("large", vec![1e6, 1e6, 1e6, 1e6]),
            ("mixed_sign", vec![-3.0, 2.0, -1.0, 4.0]),
            ("near_zero", vec![1e-20, 0.0, 1e-20, 0.0]),
        ];

        for (name, data) in &test_cases {
            let x = Tensor::from_vec(data.clone(), true);
            let y = norm.forward(&x);

            for (i, &val) in y.data().iter().enumerate() {
                assert!(
                    val.is_finite(),
                    "FALSIFIED RN-001: output[{i}] = {val} not finite for case '{name}'"
                );
            }
        }
    }

    /// FALSIFY-RN-002: Scale invariance — RMSNorm(α·x) = sign(α)·RMSNorm(x)
    #[test]
    fn falsify_rn_002_scale_invariance() {
        let norm = RMSNorm::new(4, 1e-6);
        let x = Tensor::from_vec(vec![1.0, -2.0, 3.0, -0.5], true);
        let y_base = norm.forward(&x);

        for &alpha in &[2.0_f32, 0.5, -1.0, 10.0] {
            let x_scaled = Tensor::from_vec(x.data().iter().map(|&v| v * alpha).collect(), true);
            let y_scaled = norm.forward(&x_scaled);

            let sign = alpha.signum();
            for (i, (&ys, &yb)) in y_scaled.data().iter().zip(y_base.data().iter()).enumerate() {
                let expected = sign * yb;
                let diff = (ys - expected).abs();
                assert!(
                    diff < 1e-3,
                    "FALSIFIED RN-002: RMSNorm({alpha}·x)[{i}] = {ys}, expected {expected}"
                );
            }
        }
    }

    /// FALSIFY-RN-004: Zero vector — RMSNorm(0) = 0 (not NaN)
    #[test]
    fn falsify_rn_004_zero_vector() {
        let norm = RMSNorm::new(4, 1e-6);
        let x = Tensor::from_vec(vec![0.0, 0.0, 0.0, 0.0], true);
        let y = norm.forward(&x);

        for (i, &val) in y.data().iter().enumerate() {
            assert!(val.is_finite(), "FALSIFIED RN-004: RMSNorm(0)[{i}] = {val} (expected finite)");
        }
    }

    /// FALSIFY-RN-005: Unit γ normalized RMS ≈ 1
    #[test]
    fn falsify_rn_005_unit_gamma_normalized_rms() {
        let norm = RMSNorm::new(8, 1e-6);
        let x = Tensor::from_vec(vec![1.0, -2.0, 3.0, -0.5, 4.0, -1.0, 2.5, -3.0], true);
        let y = norm.forward(&x);
        let y_data = y.data();

        let rms_out: f32 =
            (y_data.iter().map(|&v| v * v).sum::<f32>() / y_data.len() as f32).sqrt();

        assert!(
            (rms_out - 1.0).abs() < 0.01,
            "FALSIFIED RN-005: RMS(RMSNorm(x)) = {rms_out}, expected ≈ 1.0"
        );
    }

    // =========================================================================
    // PROPTEST FALSIFY: RMSNorm property-based falsification
    //
    // Five-Whys (PMAT-354, Phase 10):
    //   Why 1: RN-001..005 used fixed dimensions (d=4 or d=8)
    //   Why 2: Scale invariance (RN-002) could break at edge float ranges
    //   Why 3: proptest explores dimension/value combos humans miss
    //   Why 4: RMSNorm eps-dominated regime untested at scale
    //   Why 5: YAML rmsnorm-kernel-v1 calls for proptest on all claims
    // =========================================================================

    mod rn_proptest_falsify {
        use super::*;
        use proptest::prelude::*;

        // RN-001-prop: finiteness for random vectors
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(200))]
            #[test]
            fn falsify_rn_001_prop_finiteness(
                dim in prop::sample::select(vec![4_usize, 8, 16, 32, 64]),
                scale in 0.001_f32..1000.0,
            ) {
                let norm = RMSNorm::new(dim, 1e-6);
                let data: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.13 * scale).sin()).collect();
                let x = Tensor::from_vec(data, true);
                let y = norm.forward(&x);
                for (i, &val) in y.data().iter().enumerate() {
                    prop_assert!(
                        val.is_finite(),
                        "FALSIFIED RN-001-prop: output[{}]={} not finite (d={}, scale={})",
                        i, val, dim, scale
                    );
                }
            }
        }

        // RN-002-prop: scale invariance for random vectors
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(100))]
            #[test]
            fn falsify_rn_002_prop_scale_invariance(
                dim in prop::sample::select(vec![4_usize, 8, 16, 32]),
                alpha in prop::sample::select(vec![-10.0_f32, -1.0, 0.5, 2.0, 100.0]),
            ) {
                let norm = RMSNorm::new(dim, 1e-6);
                let data: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.37).sin() * 5.0).collect();
                let x = Tensor::from_vec(data.clone(), true);
                let y_base = norm.forward(&x);

                let x_scaled = Tensor::from_vec(
                    data.iter().map(|&v| v * alpha).collect(),
                    true,
                );
                let y_scaled = norm.forward(&x_scaled);

                let sign = alpha.signum();
                for (i, (&ys, &yb)) in y_scaled.data().iter().zip(y_base.data().iter()).enumerate() {
                    let expected = sign * yb;
                    prop_assert!(
                        (ys - expected).abs() < 1e-3,
                        "FALSIFIED RN-002-prop: [{i}] got {ys}, expected {expected} (alpha={alpha}, d={dim})"
                    );
                }
            }
        }

        // RN-005-prop: unit gamma normalized RMS for random vectors
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(100))]
            #[test]
            fn falsify_rn_005_prop_unit_gamma_rms(
                dim in prop::sample::select(vec![8_usize, 16, 32, 64]),
            ) {
                let norm = RMSNorm::new(dim, 1e-6);
                // Use values large enough that eps doesn't dominate
                let data: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.23).sin() * 10.0).collect();
                let x = Tensor::from_vec(data, true);
                let y = norm.forward(&x);
                let y_data = y.data();

                let rms_out: f32 = (y_data.iter().map(|&v| v * v).sum::<f32>() / y_data.len() as f32).sqrt();
                prop_assert!(
                    (rms_out - 1.0).abs() < 0.05,
                    "FALSIFIED RN-005-prop: RMS(output)={} != 1.0 (d={})",
                    rms_out, dim
                );
            }
        }
    }
}

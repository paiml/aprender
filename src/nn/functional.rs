//! Functional interface for neural network operations.
//!
//! This module provides stateless functions that mirror the module-based
//! activations and operations. Use these when you don't need a module
//! wrapper (e.g., in custom forward passes).
//!
//! # Example
//!
//! ```ignore
//! use aprender::nn::F;
//! use aprender::autograd::Tensor;
//!
//! let x = Tensor::randn(&[32, 10]);
//! let y = F::relu(&x);
//! let probs = F::softmax(&y, -1);
//! ```

use crate::autograd::Tensor;

/// `ReLU` activation: max(0, x)
///
/// Contract: activation-kernel-v1, equation "relu"
#[provable_contracts_macros::contract("activation-kernel-v1", equation = "relu")]
#[must_use]
pub fn relu(x: &Tensor) -> Tensor {
    x.relu()
}

/// Scalar ReLU: max(0, x)
///
/// ONE PATH: Delegates to `trueno::activations::relu_scalar` (UCBD §4).
#[inline]
#[must_use]
pub fn relu_scalar(x: f32) -> f32 {
    trueno::relu_scalar(x)
}

/// Leaky `ReLU` activation: `max(negative_slope` * x, x)
#[must_use]
pub fn leaky_relu(x: &Tensor, negative_slope: f32) -> Tensor {
    let data: Vec<f32> = x
        .data()
        .iter()
        .map(|&v| if v > 0.0 { v } else { negative_slope * v })
        .collect();
    Tensor::new(&data, x.shape())
}

/// Sigmoid activation: 1 / (1 + exp(-x))
///
/// Contract: silu-kernel-v1, equation "sigmoid"
#[provable_contracts_macros::contract("silu-kernel-v1", equation = "sigmoid")]
#[must_use]
pub fn sigmoid(x: &Tensor) -> Tensor {
    x.sigmoid()
}

/// Scalar sigmoid: σ(x) = 1 / (1 + exp(-x))
///
/// ONE PATH: Delegates to `trueno::activations::sigmoid_scalar` (UCBD §4).
#[inline]
#[must_use]
pub fn sigmoid_scalar(x: f32) -> f32 {
    trueno::sigmoid_scalar(x)
}

/// Scalar sigmoid (f64): σ(x) = 1 / (1 + exp(-x))
///
/// ONE PATH: The canonical f64 scalar sigmoid for all aprender code (UCBD §4).
#[inline]
#[must_use]
pub fn sigmoid_scalar_f64(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// SiLU (Swish) activation: x * sigmoid(x)
///
/// Equation: SiLU(x) = x / (1 + exp(-x))
///
/// ONE PATH: Per-element delegates to `trueno::silu_scalar` (UCBD §4).
///
/// Contract: silu-kernel-v1, equation "silu"
// Contract: silu-kernel-v1, equation = "silu"
#[must_use]
pub fn silu(x: &Tensor) -> Tensor {
    let data: Vec<f32> = x.data().iter().map(|&v| trueno::silu_scalar(v)).collect();
    Tensor::new(&data, x.shape())
}

/// Scalar SiLU for non-Tensor contexts.
///
/// ONE PATH: Delegates to `trueno::activations::silu_scalar` (UCBD §4).
#[inline]
#[must_use]
pub fn silu_scalar(x: f32) -> f32 {
    trueno::silu_scalar(x)
}

/// SwiGLU activation: SiLU(gate) * x
///
/// Equation: SwiGLU(x, gate) = x * SiLU(gate) = x * gate / (1 + exp(-gate))
///
/// Used in FFN layers: output = down_proj(SwiGLU(up_proj(x), gate_proj(x)))
///
/// Contract: swiglu-kernel-v1, equation "swiglu"
// Contract: swiglu-kernel-v1, equation = "swiglu"
#[must_use]
pub fn swiglu(x: &Tensor, gate: &Tensor) -> Tensor {
    let data: Vec<f32> = x
        .data()
        .iter()
        .zip(gate.data().iter())
        .map(|(&xi, &gi)| xi * trueno::silu_scalar(gi))
        .collect();
    Tensor::new(&data, x.shape())
}

/// Scalar SwiGLU for non-Tensor contexts.
///
/// Contract: swiglu-kernel-v1, equation "swiglu"
#[inline]
#[must_use]
pub fn swiglu_scalar(x: f32, gate: f32) -> f32 {
    x * gate / (1.0 + (-gate).exp())
}

/// Softmax on a 1D slice of f32 values.
///
/// ONE PATH: All slice-based softmax in the codebase MUST delegate here (UCBD §4).
///
/// Equation: softmax(x)\_i = exp(x\_i - max) / sum\_j exp(x\_j - max)
#[must_use]
pub fn softmax_1d(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    exp.iter().map(|&x| x / sum).collect()
}

/// Softmax on a 1D slice of f64 values.
///
/// ONE PATH: All f64 slice-based softmax MUST delegate here (UCBD §4).
#[must_use]
pub fn softmax_1d_f64(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let exp: Vec<f64> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f64 = exp.iter().sum();
    exp.iter().map(|&x| x / sum).collect()
}

/// Log-softmax on a 1D slice of f32 values.
///
/// ONE PATH: All slice-based log-softmax MUST delegate here (UCBD §4).
///
/// More numerically stable than log(softmax(x)).
/// Equation: log\_softmax(x)\_i = x\_i - max - log(sum exp(x\_j - max))
#[must_use]
pub fn log_softmax_1d(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let log_sum_exp: f32 = logits.iter().map(|&x| (x - max).exp()).sum::<f32>().ln();
    logits.iter().map(|&x| x - max - log_sum_exp).collect()
}

/// Tanh activation
#[must_use]
pub fn tanh(x: &Tensor) -> Tensor {
    x.tanh_()
}

/// GELU activation (Gaussian Error Linear Unit)
///
/// Equation: GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
///
/// Contract: activation-kernel-v1, equation "gelu"
#[provable_contracts_macros::contract("activation-kernel-v1", equation = "gelu")]
#[must_use]
pub fn gelu(x: &Tensor) -> Tensor {
    // ONE PATH: Per-element delegates to trueno::gelu_scalar (UCBD §4).
    let data: Vec<f32> = x.data().iter().map(|&v| trueno::gelu_scalar(v)).collect();

    Tensor::new(&data, x.shape())
}

/// Softmax along the last dimension of an ND tensor.
///
/// Equation: softmax(x)\_i = exp(x\_i - max) / sum\_j exp(x\_j - max)
///
/// ONE PATH: This is the canonical Tensor softmax. All Tensor-based softmax
/// MUST delegate here (UCBD §4). Internally uses `softmax_1d` per row.
///
/// Contract: softmax-kernel-v1, equation "softmax"
#[provable_contracts_macros::contract("softmax-kernel-v1", equation = "softmax")]
#[must_use]
pub fn softmax(x: &Tensor, _dim: i32) -> Tensor {
    let shape = x.shape();
    let last_dim = shape[shape.len() - 1];
    let batch_size: usize = shape[..shape.len() - 1].iter().product();

    let mut output = Vec::with_capacity(x.data().len());
    for b in 0..batch_size {
        let start = b * last_dim;
        let row = &x.data()[start..start + last_dim];
        output.extend(softmax_1d(row));
    }

    Tensor::new(&output, shape)
}

/// Log softmax along the last dimension of an ND tensor.
///
/// More numerically stable than log(softmax(x)).
///
/// ONE PATH: This is the canonical Tensor log-softmax. All Tensor-based
/// log-softmax MUST delegate here (UCBD §4).
///
/// Contract: cross-entropy-kernel-v1, equation "log_softmax"
#[provable_contracts_macros::contract("cross-entropy-kernel-v1", equation = "log_softmax")]
#[must_use]
pub fn log_softmax(x: &Tensor, _dim: i32) -> Tensor {
    let shape = x.shape();
    let last_dim = shape[shape.len() - 1];
    let batch_size: usize = shape[..shape.len() - 1].iter().product();

    let mut output = vec![0.0f32; x.data().len()];

    for b in 0..batch_size {
        let start = b * last_dim;
        let row = &x.data()[start..start + last_dim];

        let max_val = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let log_sum_exp: f32 = row.iter().map(|&v| (v - max_val).exp()).sum::<f32>().ln();

        for j in 0..last_dim {
            output[start + j] = row[j] - max_val - log_sum_exp;
        }
    }

    Tensor::new(&output, shape)
}

/// Dropout (must be called with training flag)
#[must_use]
pub fn dropout(x: &Tensor, p: f32, training: bool) -> Tensor {
    if !training || p == 0.0 {
        return x.clone();
    }

    use rand::Rng;
    let mut rng = rand::rng();
    let scale = 1.0 / (1.0 - p);

    let data: Vec<f32> = x
        .data()
        .iter()
        .map(|&v| if rng.random::<f32>() < p { 0.0 } else { v * scale })
        .collect();

    Tensor::new(&data, x.shape())
}

/// Layer normalization over the last dimension of an ND tensor.
///
/// ONE PATH: This is the canonical functional layer norm. All standalone
/// layer_norm implementations MUST delegate here (UCBD §4).
///
/// Equation: y\_i = (x\_i - mean) / sqrt(var + eps) * weight\_i + bias\_i
///
/// Contract: layernorm-kernel-v1, equation "layernorm"
#[must_use]
pub fn layer_norm(x: &Tensor, weight: &Tensor, bias: &Tensor, eps: f32) -> Tensor {
    let shape = x.shape();
    let last_dim = shape[shape.len() - 1];
    let batch_size: usize = shape[..shape.len() - 1].iter().product();

    let data = x.data();
    let weight_data = weight.data();
    let bias_data = bias.data();
    let mut output = Vec::with_capacity(data.len());

    for b in 0..batch_size {
        let start = b * last_dim;
        let slice = &data[start..start + last_dim];

        let mean: f32 = slice.iter().sum::<f32>() / last_dim as f32;
        let var: f32 = slice.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / last_dim as f32;
        let inv_std = 1.0 / (var + eps).sqrt();

        for (i, &val) in slice.iter().enumerate() {
            let normalized = (val - mean) * inv_std;
            output.push(normalized * weight_data[i] + bias_data[i]);
        }
    }

    Tensor::new(&output, shape)
}

/// RMS normalization over the last dimension of an ND tensor.
///
/// ONE PATH: This is the canonical functional RMS norm. All standalone
/// rms_norm implementations MUST delegate here (UCBD §4).
///
/// Equation: y\_i = x\_i / rms(x) * weight\_i, where rms(x) = sqrt(mean(x^2) + eps)
///
/// Contract: rmsnorm-kernel-v1, equation "rmsnorm"
#[must_use]
pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f32) -> Tensor {
    let shape = x.shape();
    let last_dim = shape[shape.len() - 1];
    let batch_size: usize = shape[..shape.len() - 1].iter().product();

    let data = x.data();
    let weight_data = weight.data();
    let mut output = Vec::with_capacity(data.len());

    for b in 0..batch_size {
        let start = b * last_dim;
        let slice = &data[start..start + last_dim];

        let rms: f32 = (slice.iter().map(|&x| x * x).sum::<f32>() / last_dim as f32 + eps).sqrt();
        let inv_rms = 1.0 / rms;

        for (i, &val) in slice.iter().enumerate() {
            output.push(val * inv_rms * weight_data[i]);
        }
    }

    Tensor::new(&output, shape)
}

/// Linear transformation: y = x @ weight^T + bias
#[must_use]
pub fn linear(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Tensor {
    let weight_t = weight.transpose();
    let output = x.matmul(&weight_t);

    match bias {
        Some(b) => broadcast_add_1d(&output, b),
        None => output,
    }
}

/// Helper: broadcast-add 1D bias to 2D output
fn broadcast_add_1d(matrix: &Tensor, vector: &Tensor) -> Tensor {
    let (rows, cols) = (matrix.shape()[0], matrix.shape()[1]);
    let mut result = vec![0.0; rows * cols];

    for i in 0..rows {
        for j in 0..cols {
            result[i * cols + j] = matrix.data()[i * cols + j] + vector.data()[j];
        }
    }

    Tensor::new(&result, &[rows, cols])
}

// ============================================================================
// Distance Functions (ONE PATH canonical implementations)
// ============================================================================

/// Euclidean distance between two slices.
///
/// ONE PATH: Canonical euclidean distance (UCBD §4).
///
/// ```text
/// d(a, b) = sqrt(Σ(a_i - b_i)²)
/// ```
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// Cosine similarity between two slices.
///
/// ONE PATH: Canonical cosine similarity for `&[f32]` (UCBD §4).
///
/// ```text
/// cos(a, b) = (a · b) / (||a|| × ||b||)
/// ```
pub fn cosine_similarity_slice(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let mut dot = 0.0_f32;
    let mut norm_a = 0.0_f32;
    let mut norm_b = 0.0_f32;

    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < 1e-10 {
        0.0
    } else {
        (dot / denom).clamp(-1.0, 1.0)
    }
}

#[cfg(test)]
mod softmax_contract_tests {
    use super::*;

    // ========================================================================
    // FALSIFY-SM: softmax-kernel-v1.yaml contract falsification
    //
    // Five-Whys (PMAT-354):
    //   Why 1: aprender had 0 FALSIFY-SM-* tests for softmax_1d
    //   Why 2: softmax_1d had no inline tests at all (pure utility fn)
    //   Why 3: callers (calibration, regularization, gating) tested indirectly
    //   Why 4: no direct mapping from softmax-kernel-v1.yaml to test names
    //   Why 5: softmax was "obviously correct" — 3 lines of standard code
    //
    // References:
    //   - provable-contracts/contracts/softmax-kernel-v1.yaml
    //   - Bridle (1990) "Training Stochastic Model Recognition Algorithms"
    // ========================================================================

    /// FALSIFY-SM-001: Output sums to 1 (partition of unity)
    #[test]
    fn falsify_sm_001_sums_to_one() {
        let cases: Vec<Vec<f32>> = vec![
            vec![1.0, 2.0, 3.0],
            vec![-10.0, 0.0, 10.0],
            vec![100.0, 101.0, 102.0],
            (0..100).map(|i| (i as f32 * 0.37).sin() * 5.0).collect(),
        ];

        for (idx, logits) in cases.iter().enumerate() {
            let probs = softmax_1d(logits);
            let sum: f32 = probs.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "FALSIFIED SM-001: case {idx} sum={sum}"
            );
        }
    }

    /// FALSIFY-SM-001b: f64 variant also sums to 1
    #[test]
    fn falsify_sm_001b_f64_sums_to_one() {
        let logits: Vec<f64> = vec![1.0, 2.0, 3.0, -5.0, 10.0];
        let probs = softmax_1d_f64(&logits);
        let sum: f64 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "FALSIFIED SM-001b: f64 sum={sum}"
        );
    }

    /// FALSIFY-SM-002: All outputs strictly positive
    #[test]
    fn falsify_sm_002_strictly_positive() {
        let logits: Vec<f32> = (0..50).map(|i| (i as f32 - 25.0) * 2.0).collect();
        let probs = softmax_1d(&logits);

        for (i, &p) in probs.iter().enumerate() {
            assert!(
                p > 0.0,
                "FALSIFIED SM-002: probs[{i}] = {p} not strictly positive"
            );
        }
    }

    /// FALSIFY-SM-003: Order preservation (argmax invariant)
    #[test]
    fn falsify_sm_003_order_preservation() {
        let logits = vec![1.0f32, 5.0, 3.0, 2.0];
        let probs = softmax_1d(&logits);

        let input_argmax = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;
        let output_argmax = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;

        assert_eq!(
            input_argmax, output_argmax,
            "FALSIFIED SM-003: argmax changed from {input_argmax} to {output_argmax}"
        );
    }

    /// FALSIFY-SM-004: Each output bounded in (0, 1) when n > 1
    #[test]
    fn falsify_sm_004_bounded_zero_one() {
        let logits: Vec<f32> = (0..20).map(|i| (i as f32 * 1.7).sin() * 10.0).collect();
        let probs = softmax_1d(&logits);

        for (i, &p) in probs.iter().enumerate() {
            assert!(
                p > 0.0 && p < 1.0,
                "FALSIFIED SM-004: probs[{i}] = {p} not in (0, 1)"
            );
        }
    }

    /// FALSIFY-SM-005: Numerical stability — no NaN/Inf for extreme inputs
    #[test]
    fn falsify_sm_005_numerical_stability() {
        let extreme = vec![1000.0f32, 1001.0, 1002.0];
        let probs = softmax_1d(&extreme);
        assert!(
            probs.iter().all(|p| p.is_finite()),
            "FALSIFIED SM-005: extreme inputs produced non-finite"
        );

        let sum: f32 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "FALSIFIED SM-005: extreme sum={sum}"
        );
    }

    /// FALSIFY-SM-006: Tensor softmax preserves shape and per-row contract
    #[test]
    fn falsify_sm_006_tensor_softmax_shape() {
        let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let result = softmax(&x, -1);

        assert_eq!(result.shape(), &[2, 3], "FALSIFIED SM-006: shape changed");

        // Each row should sum to 1
        let data = result.data();
        let row1_sum: f32 = data[0..3].iter().sum();
        let row2_sum: f32 = data[3..6].iter().sum();

        assert!(
            (row1_sum - 1.0).abs() < 1e-5,
            "FALSIFIED SM-006: row 1 sum={row1_sum}"
        );
        assert!(
            (row2_sum - 1.0).abs() < 1e-5,
            "FALSIFIED SM-006: row 2 sum={row2_sum}"
        );
    }
}

include!("functional_include_01.rs");

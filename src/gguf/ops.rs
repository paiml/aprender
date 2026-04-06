//! Pure mathematical operations for GGUF inference
//!
//! This module contains standalone math functions used by both CPU and GPU
//! inference paths. By extracting these to a shared module, we enable:
//!
//! - Code reuse between `OwnedQuantizedModel` (CPU) and `OwnedQuantizedModelCuda` (GPU)
//! - Easier testing of mathematical correctness
//! - Clear separation of concerns
//!
//! ## Functions
//!
//! - `rms_norm`: RMSNorm normalization (LLaMA, Qwen, Mistral)
//! - `gelu`: GELU activation function
//! - `silu`: SiLU/Swish activation function
//! - `add_bias`: Add bias vector to output
//! - `argmax`: Find index of maximum value
//! - `softmax`: Numerically stable softmax

use provable_contracts_macros::contract;
use trueno::Vector as TruenoVector;

// =============================================================================
// Normalization Operations
// =============================================================================

/// RMSNorm (Root Mean Square Layer Normalization)
///
/// Used by LLaMA, TinyLlama, Qwen, Mistral instead of LayerNorm.
/// Formula: output = x / sqrt(mean(x^2) + eps) * weight
///
/// # Arguments
/// * `input` - Input tensor [seq_len * hidden_dim]
/// * `weight` - Normalization weights [hidden_dim]
/// * `eps` - Small constant for numerical stability (typically 1e-5 or 1e-6)
///
/// # Returns
/// Normalized output [seq_len * hidden_dim]
#[contract("forward-pass-v1", equation = "rms_norm")]
pub fn rms_norm(input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    contract_pre_rmsnorm!(input);
    let hidden_dim = weight.len();
    let seq_len = input.len() / hidden_dim;
    let mut output = Vec::with_capacity(input.len());

    let weight_vec = TruenoVector::from_slice(weight);

    for i in 0..seq_len {
        let start = i * hidden_dim;
        let end = start + hidden_dim;
        let x = &input[start..end];

        let x_vec = TruenoVector::from_slice(x);

        // SIMD: sum of squares
        let sum_sq = x_vec
            .sum_of_squares()
            .unwrap_or_else(|_| x.iter().map(|v| v * v).sum::<f32>());

        let mean_sq = sum_sq / hidden_dim as f32;
        let inv_rms = 1.0 / (mean_sq + eps).sqrt();

        // SIMD: scale by inv_rms, then multiply by weight
        match x_vec
            .scale(inv_rms)
            .and_then(|scaled| scaled.mul(&weight_vec))
        {
            Ok(result) => {
                output.extend_from_slice(result.as_slice());
            },
            Err(_) => {
                // Fallback to scalar
                for j in 0..hidden_dim {
                    output.push(x[j] * inv_rms * weight[j]);
                }
            },
        }
    }

    output
}

/// RMSNorm to pre-allocated buffer (zero-allocation path)
///
/// # Arguments
/// * `input` - Input tensor [hidden_dim] (single position)
/// * `weight` - Normalization weights [hidden_dim]
/// * `eps` - Small constant for numerical stability
/// * `output` - Pre-allocated output buffer [hidden_dim]
pub fn rms_norm_into(input: &[f32], weight: &[f32], eps: f32, output: &mut [f32]) {
    let hidden_dim = weight.len();
    let x = &input[..hidden_dim];

    let x_vec = TruenoVector::from_slice(x);
    let weight_vec = TruenoVector::from_slice(weight);

    let sum_sq = x_vec
        .sum_of_squares()
        .unwrap_or_else(|_| x.iter().map(|v| v * v).sum::<f32>());

    let mean_sq = sum_sq / hidden_dim as f32;
    let inv_rms = 1.0 / (mean_sq + eps).sqrt();

    match x_vec
        .scale(inv_rms)
        .and_then(|scaled| scaled.mul(&weight_vec))
    {
        Ok(result) => {
            output[..hidden_dim].copy_from_slice(result.as_slice());
        },
        Err(_) => {
            for j in 0..hidden_dim {
                output[j] = x[j] * inv_rms * weight[j];
            }
        },
    }
}

/// True Layer Normalization with optional bias
///
/// GH-278: Implements real LayerNorm with mean subtraction.
/// Formula: output = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias
/// Used by GPT-2 and phi-2 (models with attn_norm_bias).
///
/// # Arguments
/// * `input` - Input tensor [seq_len * hidden_dim]
/// * `weight` - Normalization weights [hidden_dim]
/// * `bias` - Optional bias [hidden_dim]
/// * `eps` - Small constant for numerical stability
pub fn layer_norm(input: &[f32], weight: &[f32], bias: Option<&[f32]>, eps: f32) -> Vec<f32> {
    contract_pre_layernorm!(input);
    let hidden_dim = weight.len();
    let seq_len = input.len() / hidden_dim;
    let mut output = Vec::with_capacity(input.len());
    let n = hidden_dim as f32;

    for i in 0..seq_len {
        let start = i * hidden_dim;
        let end = start + hidden_dim;
        let x = &input[start..end];

        let mean: f32 = x.iter().sum::<f32>() / n;
        let var: f32 = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
        let inv_std = 1.0 / (var + eps).sqrt();

        for j in 0..hidden_dim {
            let normalized = (x[j] - mean) * inv_std;
            let mut val = normalized * weight[j];
            if let Some(b) = bias {
                val += b[j];
            }
            output.push(val);
        }
    }

    output
}

/// Layer normalization to pre-allocated buffer
///
/// GH-278: Implements real LayerNorm with mean subtraction.
pub fn layer_norm_into(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    eps: f32,
    output: &mut [f32],
) {
    let hidden_dim = weight.len();
    let x = &input[..hidden_dim];
    let n = hidden_dim as f32;

    let mean: f32 = x.iter().sum::<f32>() / n;
    let var: f32 = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let inv_std = 1.0 / (var + eps).sqrt();

    for j in 0..hidden_dim {
        let normalized = (x[j] - mean) * inv_std;
        output[j] = normalized * weight[j];
        if let Some(b) = bias {
            output[j] += b[j];
        }
    }
}

// =============================================================================
// Activation Functions
// =============================================================================

/// GELU (Gaussian Error Linear Unit) activation
///
/// Approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
///
/// # Arguments
/// * `input` - Input tensor (modified in-place)
#[inline]
pub fn gelu(input: &mut [f32]) {
    if !input.is_empty() {
        contract_pre_gelu!(input);
    }
    // ONE PATH: Per-element delegates to trueno::gelu_scalar (UCBD §4).
    for x in input.iter_mut() {
        *x = trueno::gelu_scalar(*x);
    }
}

/// SiLU (Sigmoid Linear Unit) / Swish activation
///
/// SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
/// Used in SwiGLU FFN (LLaMA, Mistral, etc.)
///
/// # Arguments
/// * `input` - Input tensor (modified in-place)
#[inline]
pub fn silu(input: &mut [f32]) {
    if !input.is_empty() {
        contract_pre_silu!(input);
    }
    // ONE PATH: Per-element delegates to trueno::silu_scalar (UCBD §4).
    for x in input.iter_mut() {
        *x = trueno::silu_scalar(*x);
    }
}

// =============================================================================
// Utility Operations
// =============================================================================

/// Add bias vector to output tensor
///
/// # Arguments
/// * `output` - Output tensor [seq_len * out_dim] (modified in-place)
/// * `bias` - Bias vector [out_dim]
#[inline]
pub fn add_bias(output: &mut [f32], bias: &[f32]) {
    let out_dim = bias.len();
    let seq_len = output.len() / out_dim;
    for s in 0..seq_len {
        for o in 0..out_dim {
            output[s * out_dim + o] += bias[o];
        }
    }
}

/// Find index of maximum value (greedy decoding)
///
/// # Arguments
/// * `logits` - Logit values [vocab_size]
///
/// # Returns
/// Index of the maximum value
#[inline]
pub fn argmax(logits: &[f32]) -> u32 {
    let mut max_idx = 0u32;
    let mut max_val = f32::NEG_INFINITY;
    for (i, &val) in logits.iter().enumerate() {
        if val > max_val {
            max_val = val;
            max_idx = i as u32;
        }
    }
    max_idx
}

/// Numerically stable softmax
///
/// Computes softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
///
/// # Arguments
/// * `logits` - Input logits (modified in-place to probabilities)
#[contract("sampling-v1", equation = "softmax_inplace")]
pub fn softmax(logits: &mut [f32]) {
    contract_pre_softmax!(logits);
    // Find max for numerical stability
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // Compute exp(x - max) and sum
    let mut sum = 0.0f32;
    for x in logits.iter_mut() {
        *x = (*x - max_val).exp();
        sum += *x;
    }

    // Normalize
    let inv_sum = 1.0 / sum;
    for x in logits.iter_mut() {
        *x *= inv_sum;
    }
}

/// Per-head RMSNorm for QK normalization (GH-279: Qwen3)
///
/// Applies RMSNorm independently to each attention head's Q or K projection.
/// Weight shape is `[head_dim]` and is shared across all heads.
///
/// Formula per head: `head_out = RMSNorm(head_in, weight, eps)`
///
/// # Arguments
/// * `qk` - Q or K tensor `[num_heads * head_dim]` (modified in-place)
/// * `weight` - Norm weight `[head_dim]`
/// * `num_heads` - Number of heads
/// * `eps` - Epsilon for numerical stability
pub fn apply_per_head_rms_norm(qk: &mut [f32], weight: &[f32], num_heads: usize, eps: f32) {
    let head_dim = weight.len();
    debug_assert_eq!(
        qk.len(),
        num_heads * head_dim,
        "QK norm: expected {} elements, got {}",
        num_heads * head_dim,
        qk.len()
    );

    for h in 0..num_heads {
        let start = h * head_dim;
        let end = start + head_dim;
        let head = &mut qk[start..end];

        // RMSNorm: x / sqrt(mean(x^2) + eps) * weight
        let sum_sq: f32 = head.iter().map(|v| v * v).sum();
        let mean_sq = sum_sq / head_dim as f32;
        let inv_rms = 1.0 / (mean_sq + eps).sqrt();

        for (j, val) in head.iter_mut().enumerate() {
            *val = *val * inv_rms * weight[j];
        }
    }
}

include!("ops_gelu_zero_positive.rs");

#[cfg(test)]
mod rmsnorm_contract_tests {
    use super::*;

    // =========================================================================
    // FALSIFY-RN: rmsnorm-kernel-v1.yaml contract (realizar rms_norm)
    //
    // Five-Whys (PMAT-354):
    //   Why 1: realizar had zero FALSIFY-RN-* tests despite 15+ RMSNorm functions
    //   Why 2: ops.rs had no test module at all — tested only via integration
    //   Why 3: no mapping from rmsnorm-kernel-v1.yaml to realizar test names
    //   Why 4: realizar predates the provable-contracts YAML convention
    //   Why 5: rms_norm was tested via end-to-end model runs, not unit contracts
    //
    // References:
    //   - provable-contracts/contracts/rmsnorm-kernel-v1.yaml
    //   - Zhang & Sennrich (2019) "Root Mean Square Layer Normalization"
    // =========================================================================

    /// FALSIFY-RN-001: Finiteness — output must be finite for all finite input when eps > 0
    #[test]
    fn falsify_rn_001_finiteness() {
        let weight = vec![1.0f32; 8];
        let eps = 1e-5;

        let test_cases: Vec<(&str, Vec<f32>)> = vec![
            ("normal", vec![1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0]),
            ("small", vec![1e-7; 8]),
            ("large", vec![1e6; 8]),
            ("mixed", vec![-1e5, 1e5, -1e-5, 1e-5, 0.0, 0.0, 1.0, -1.0]),
        ];

        for (name, input) in &test_cases {
            let output = rms_norm(input, &weight, eps);

            for (i, &val) in output.iter().enumerate() {
                assert!(
                    val.is_finite(),
                    "FALSIFIED RN-001: output[{i}] = {val} not finite for case '{name}'"
                );
            }
        }
    }

    /// FALSIFY-RN-001b: rms_norm_into also produces finite output
    #[test]
    fn falsify_rn_001_into_finiteness() {
        let weight = vec![1.0f32; 4];
        let input = vec![1e-7, 1e7, -1e-7, -1e7];
        let mut output = vec![0.0f32; 4];

        rms_norm_into(&input, &weight, 1e-5, &mut output);

        for (i, &val) in output.iter().enumerate() {
            assert!(
                val.is_finite(),
                "FALSIFIED RN-001: rms_norm_into output[{i}] = {val} not finite"
            );
        }
    }

    /// FALSIFY-RN-002: Scale invariance — RMSNorm(α·x) = sign(α)·RMSNorm(x)
    #[test]
    fn falsify_rn_002_scale_invariance() {
        let weight = vec![1.0f32; 4];
        let eps = 1e-6;
        let x = vec![3.0f32, -1.0, 2.0, -4.0];
        let y_base = rms_norm(&x, &weight, eps);

        for &alpha in &[2.0_f32, 0.5, 100.0, -1.0, -3.0] {
            let x_scaled: Vec<f32> = x.iter().map(|&v| v * alpha).collect();
            let y_scaled = rms_norm(&x_scaled, &weight, eps);

            let sign = alpha.signum();
            for (i, (&ys, &yb)) in y_scaled.iter().zip(y_base.iter()).enumerate() {
                let expected = sign * yb;
                let diff = (ys - expected).abs();
                assert!(
                    diff < 1e-4,
                    "FALSIFIED RN-002: rms_norm({alpha}·x)[{i}] = {ys}, expected {expected}"
                );
            }
        }
    }

    /// FALSIFY-RN-004: Zero vector — RMSNorm(0) = 0 (not NaN)
    #[test]
    fn falsify_rn_004_zero_vector() {
        let weight = vec![1.0f32; 4];
        let x = vec![0.0f32; 4];
        let y = rms_norm(&x, &weight, 1e-5);

        for (i, &val) in y.iter().enumerate() {
            assert!(
                val.is_finite(),
                "FALSIFIED RN-004: rms_norm(0)[{i}] = {val} (expected finite)"
            );
            assert!(
                val.abs() < 1e-3,
                "FALSIFIED RN-004: rms_norm(0)[{i}] = {val} (expected ≈ 0)"
            );
        }
    }

    /// FALSIFY-RN-005: Unit γ normalized RMS ≈ 1
    ///
    /// After RMSNorm with unit weights, RMS of output should be ≈ 1
    #[test]
    fn falsify_rn_005_unit_gamma_normalized_rms() {
        let weight = vec![1.0f32; 8];
        let eps = 1e-6;

        let test_vectors: Vec<Vec<f32>> = vec![
            vec![1.0, -2.0, 3.0, -0.5, 4.0, -1.0, 2.5, -3.0],
            vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
        ];

        for (idx, x) in test_vectors.iter().enumerate() {
            let y = rms_norm(x, &weight, eps);

            let rms_out: f32 = (y.iter().map(|&v| v * v).sum::<f32>() / y.len() as f32).sqrt();

            assert!(
                (rms_out - 1.0).abs() < 0.01,
                "FALSIFIED RN-005: RMS(rms_norm(x)) = {rms_out}, expected ≈ 1.0 (case {idx})"
            );
        }
    }

    /// FALSIFY-RN-002b: rms_norm and rms_norm_into produce same result
    #[test]
    fn falsify_rn_consistency_norm_vs_norm_into() {
        let weight = vec![1.5f32, 0.5, 2.0, 0.8];
        let input = vec![3.0f32, -1.0, 2.0, -4.0];
        let eps = 1e-5;

        let y_alloc = rms_norm(&input, &weight, eps);
        let mut y_into = vec![0.0f32; 4];
        rms_norm_into(&input, &weight, eps, &mut y_into);

        for (i, (&a, &b)) in y_alloc.iter().zip(y_into.iter()).enumerate() {
            let diff = (a - b).abs();
            assert!(
                diff < 1e-6,
                "FALSIFIED: rms_norm vs rms_norm_into mismatch at [{i}]: {a} vs {b}"
            );
        }
    }

    // =========================================================================
    // PROPTEST FALSIFY: RMSNorm property-based falsification
    //
    // Five-Whys (PMAT-354, Phase 10):
    //   Why 1: RN-001..005 used fixed dimensions (d=4 or d=8)
    //   Why 2: Scale invariance (RN-002) could break at edge float ranges
    //   Why 3: proptest explores dimension/value combos humans miss
    //   Why 4: rms_norm and rms_norm_into consistency untested at scale
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
                let weight = vec![1.0f32; dim];
                let data: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.13 * scale).sin()).collect();
                let output = rms_norm(&data, &weight, 1e-5);
                for (i, &val) in output.iter().enumerate() {
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
                let weight = vec![1.0f32; dim];
                let eps = 1e-6;
                let data: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.37).sin() * 5.0).collect();
                let y_base = rms_norm(&data, &weight, eps);

                let x_scaled: Vec<f32> = data.iter().map(|&v| v * alpha).collect();
                let y_scaled = rms_norm(&x_scaled, &weight, eps);

                let sign = alpha.signum();
                for (i, (&ys, &yb)) in y_scaled.iter().zip(y_base.iter()).enumerate() {
                    let expected = sign * yb;
                    prop_assert!(
                        (ys - expected).abs() < 1e-3,
                        "FALSIFIED RN-002-prop: [{i}] got {ys}, expected {expected} (alpha={alpha}, d={dim})"
                    );
                }
            }
        }

        // RN-005-prop: unit gamma normalized RMS
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(100))]
            #[test]
            fn falsify_rn_005_prop_unit_gamma_rms(
                dim in prop::sample::select(vec![8_usize, 16, 32, 64]),
            ) {
                let weight = vec![1.0f32; dim];
                let data: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.23).sin() * 10.0).collect();
                let y = rms_norm(&data, &weight, 1e-6);

                let rms_out: f32 = (y.iter().map(|&v| v * v).sum::<f32>() / y.len() as f32).sqrt();
                prop_assert!(
                    (rms_out - 1.0).abs() < 0.05,
                    "FALSIFIED RN-005-prop: RMS(output)={} != 1.0 (d={})",
                    rms_out, dim
                );
            }
        }
    }
}

// =========================================================================
// FALSIFY-LN: layernorm-kernel-v1.yaml contract (realizar layer_norm)
//
// Five-Whys (PMAT-354, Phase 10):
//   Why 1: realizar had 10+ layer_norm tests but zero FALSIFY-LN-* tagged tests
//   Why 2: existing tests verify shapes/integration, not mathematical invariants
//   Why 3: no mapping from layernorm-kernel-v1.yaml to realizar test names
//   Why 4: realizar predates the provable-contracts YAML convention
//   Why 5: layer_norm was "obviously correct" (y = (x-μ)/σ * γ + β)
//
// References:
//   - provable-contracts/contracts/layernorm-kernel-v1.yaml
//   - Ba et al. (2016) "Layer Normalization"
// =========================================================================

#[cfg(test)]
mod ln_contract_tests {
    use super::*;

    /// FALSIFY-LN-001: Centering — mean of LN output ≈ 0 (with bias=0)
    #[test]
    fn falsify_ln_001_centering() {
        let dim = 8;
        let weight = vec![1.0f32; dim];
        let bias = vec![0.0f32; dim];
        let data = vec![1.0, -2.0, 3.0, 0.5, -1.5, 2.5, -0.5, 1.5];
        let y = layer_norm(&data, &weight, Some(&bias), 1e-5);

        let mean: f32 = y.iter().sum::<f32>() / dim as f32;
        assert!(
            mean.abs() < 1e-5,
            "FALSIFIED LN-001: mean(LN(x)) = {mean}, expected ≈ 0"
        );
    }

    /// FALSIFY-LN-002: Standardization — variance of LN output ≈ 1 (with weight=1)
    #[test]
    fn falsify_ln_002_standardization() {
        let dim = 8;
        let weight = vec![1.0f32; dim];
        let bias = vec![0.0f32; dim];
        let data = vec![1.0, -2.0, 3.0, 0.5, -1.5, 2.5, -0.5, 1.5];
        let y = layer_norm(&data, &weight, Some(&bias), 1e-5);

        let mean: f32 = y.iter().sum::<f32>() / dim as f32;
        let var: f32 = y.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / dim as f32;
        assert!(
            (var - 1.0).abs() < 0.05,
            "FALSIFIED LN-002: var(LN(x)) = {var}, expected ≈ 1.0"
        );
    }

    /// FALSIFY-LN-003: Denominator safety — output finite for all finite input
    #[test]
    fn falsify_ln_003_denominator_safety() {
        let weight = vec![1.0f32; 4];
        let bias = vec![0.0f32; 4];
        let test_cases: Vec<(&str, Vec<f32>)> = vec![
            ("normal", vec![1.0, 2.0, 3.0, 4.0]),
            ("small", vec![1e-7, 1e-7, 1e-7, 1e-7]),
            ("large", vec![1e6, 1e6, 1e6, 1e6]),
            ("mixed_sign", vec![-3.0, 2.0, -1.0, 4.0]),
            ("near_zero", vec![1e-20, 0.0, 1e-20, 0.0]),
            ("all_zero", vec![0.0, 0.0, 0.0, 0.0]),
        ];

        for (name, data) in &test_cases {
            let y = layer_norm(data, &weight, Some(&bias), 1e-5);
            for (i, &val) in y.iter().enumerate() {
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
        let dim = 6;
        let weight = vec![1.0f32; dim];
        let bias = vec![0.0f32; dim];
        let data = vec![10.0, -5.0, 3.0, 7.0, -2.0, 0.5];
        let y1 = layer_norm(&data, &weight, Some(&bias), 1e-5);
        let y2 = layer_norm(&y1, &weight, Some(&bias), 1e-5);

        for (i, (&a, &b)) in y1.iter().zip(y2.iter()).enumerate() {
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
        let dim = 5;
        let weight = vec![1.0f32; dim];
        let bias = vec![0.0f32; dim];
        let data = vec![1.0, -2.0, 3.0, 0.5, -1.5];
        let y_base = layer_norm(&data, &weight, Some(&bias), 1e-5);

        for &c in &[10.0_f32, -100.0, 0.001, 1000.0] {
            let shifted: Vec<f32> = data.iter().map(|&v| v + c).collect();
            let y_shifted = layer_norm(&shifted, &weight, Some(&bias), 1e-5);

            for (i, (&a, &b)) in y_base.iter().zip(y_shifted.iter()).enumerate() {
                let tol = 1e-3 * a.abs().max(1.0);
                assert!(
                    (a - b).abs() < tol,
                    "FALSIFIED LN-006: LN(x)[{i}]={a}, LN(x+{c})[{i}]={b}"
                );
            }
        }
    }

    /// FALSIFY-LN-007: Constant input → output ≈ 0 (bias=0)
    #[test]
    fn falsify_ln_007_constant_input() {
        let weight = vec![1.0f32; 4];
        let bias = vec![0.0f32; 4];
        for &c in &[0.0_f32, 1.0, -5.0, 1e6, 1e-6] {
            let data = vec![c; 4];
            let y = layer_norm(&data, &weight, Some(&bias), 1e-5);

            for (i, &val) in y.iter().enumerate() {
                assert!(
                    val.is_finite(),
                    "FALSIFIED LN-003 (via LN-007): NaN/Inf for constant {c}"
                );
                assert!(
                    val.abs() < 1e-3,
                    "FALSIFIED LN-007: LN([{c};4])[{i}] = {val}, expected ≈ 0"
                );
            }
        }
    }

    /// FALSIFY-LN-001b: layer_norm_into also centers
    #[test]
    fn falsify_ln_001_into_centering() {
        let dim = 8;
        let weight = vec![1.0f32; dim];
        let bias = vec![0.0f32; dim];
        let data = vec![1.0, -2.0, 3.0, 0.5, -1.5, 2.5, -0.5, 1.5];
        let mut output = vec![0.0f32; dim];
        layer_norm_into(&data, &weight, Some(&bias), 1e-5, &mut output);

        let mean: f32 = output.iter().sum::<f32>() / dim as f32;
        assert!(
            mean.abs() < 1e-5,
            "FALSIFIED LN-001b: mean(layer_norm_into(x)) = {mean}"
        );
    }

    /// FALSIFY-LN consistency: layer_norm and layer_norm_into produce same result
    #[test]
    fn falsify_ln_consistency_norm_vs_norm_into() {
        let dim = 8;
        let weight = vec![1.0f32; dim];
        let bias = vec![0.0f32; dim];
        let data = vec![3.0, -1.0, 2.0, -4.0, 5.0, -0.5, 1.5, -2.5];

        let y_alloc = layer_norm(&data, &weight, Some(&bias), 1e-5);
        let mut y_into = vec![0.0f32; dim];
        layer_norm_into(&data, &weight, Some(&bias), 1e-5, &mut y_into);

        for (i, (&a, &b)) in y_alloc.iter().zip(y_into.iter()).enumerate() {
            let diff = (a - b).abs();
            assert!(
                diff < 1e-6,
                "FALSIFIED LN consistency: layer_norm[{i}]={a}, layer_norm_into[{i}]={b}"
            );
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
                let weight = vec![1.0f32; dim];
                let bias = vec![0.0f32; dim];
                let data: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.37 * scale).sin() * scale).collect();
                let y = layer_norm(&data, &weight, Some(&bias), 1e-5);

                let mean: f32 = y.iter().sum::<f32>() / dim as f32;
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
                let weight = vec![1.0f32; dim];
                let bias = vec![0.0f32; dim];
                let data: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.23).sin() * scale).collect();
                let y = layer_norm(&data, &weight, Some(&bias), 1e-5);

                let mean: f32 = y.iter().sum::<f32>() / dim as f32;
                let var: f32 = y.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / dim as f32;
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
                let weight = vec![1.0f32; dim];
                let bias = vec![0.0f32; dim];
                let data: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.37).sin() * 5.0).collect();
                let y_base = layer_norm(&data, &weight, Some(&bias), 1e-5);

                let shifted: Vec<f32> = data.iter().map(|&v| v + shift).collect();
                let y_shifted = layer_norm(&shifted, &weight, Some(&bias), 1e-5);

                for (i, (&a, &b)) in y_base.iter().zip(y_shifted.iter()).enumerate() {
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
                let weight = vec![1.0f32; dim];
                let bias = vec![0.0f32; dim];
                let data = vec![c; dim];
                let y = layer_norm(&data, &weight, Some(&bias), 1e-5);

                for (i, &val) in y.iter().enumerate() {
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

// =========================================================================
// FALSIFY-SI: silu-kernel-v1.yaml contract (realizar silu in-place)
//
// Five-Whys (PMAT-354, Phase 11):
//   Why 1: realizar had zero FALSIFY-SI-* tests despite 6+ silu implementations
//   Why 2: unit tests verify SIMD parity, not mathematical invariants
//   Why 3: no mapping from silu-kernel-v1.yaml to realizar test names
//   Why 4: realizar predates the provable-contracts YAML convention
//   Why 5: SiLU was "obviously correct" (delegates to trueno::silu_scalar)
//
// References:
//   - provable-contracts/contracts/silu-kernel-v1.yaml
//   - Ramachandran et al. (2017) "Searching for Activation Functions"
// =========================================================================

#[cfg(test)]
mod silu_contract_tests {
    use super::*;

    /// FALSIFY-SI-001: Zero preservation — SiLU(0) = 0
    #[test]
    fn falsify_si_001_zero_preservation() {
        let mut input = vec![0.0];
        silu(&mut input);
        assert!(
            input[0].abs() < 1e-7,
            "FALSIFIED SI-001: SiLU(0) = {}",
            input[0]
        );
    }

    /// FALSIFY-SI-002: Global lower bound — SiLU(x) > -0.279 for all x
    #[test]
    fn falsify_si_002_global_lower_bound() {
        let mut input = vec![
            -100.0, -50.0, -10.0, -5.0, -2.0, -1.278, -1.0, -0.5, 0.0, 0.5, 1.0, 5.0, 100.0,
        ];
        silu(&mut input);
        for (i, &val) in input.iter().enumerate() {
            assert!(
                val > -0.28,
                "FALSIFIED SI-002: SiLU[{i}] = {val}, expected > -0.279"
            );
        }
    }

    /// FALSIFY-SI-003: Monotonic for positive inputs
    #[test]
    fn falsify_si_003_monotonic_positive() {
        let mut input = vec![0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0];
        silu(&mut input);
        for i in 1..input.len() {
            assert!(
                input[i] > input[i - 1],
                "FALSIFIED SI-003: SiLU not monotonic: [{i}]={} not > [{}]={}",
                input[i],
                i - 1,
                input[i - 1]
            );
        }
    }

    /// FALSIFY-SI-005: Asymptotic linearity — |SiLU(x) - x| < 0.01 for x > 10
    #[test]
    fn falsify_si_005_asymptotic_linearity() {
        let originals = [10.0f32, 20.0, 50.0, 100.0, 500.0];
        let mut input = originals.to_vec();
        silu(&mut input);
        for (i, (&val, &orig)) in input.iter().zip(originals.iter()).enumerate() {
            assert!(
                (val - orig).abs() < 0.01,
                "FALSIFIED SI-005: |SiLU({orig}) - {orig}| = {} >= 0.01",
                (val - orig).abs()
            );
        }
    }

    /// FALSIFY-SI-006: Large negative → 0
    #[test]
    fn falsify_si_006_large_negative_vanishes() {
        let mut input = vec![-10.0, -20.0, -50.0, -100.0, -500.0];
        silu(&mut input);
        for (i, &val) in input.iter().enumerate() {
            assert!(
                val.abs() < 0.01,
                "FALSIFIED SI-006: SiLU(neg)[{i}] = {val}, expected ≈ 0"
            );
        }
    }

    mod si_proptest_falsify {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(500))]
            #[test]
            fn falsify_si_002_prop_lower_bound(x in -1000.0_f32..1000.0) {
                let mut input = vec![x];
                silu(&mut input);
                prop_assert!(input[0] > -0.28, "FALSIFIED SI-002-prop: SiLU({x}) = {}", input[0]);
            }
        }

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(300))]
            #[test]
            fn falsify_si_003_prop_monotonic_positive(
                a in 0.001_f32..100.0,
                b in 0.001_f32..100.0,
            ) {
                if a != b {
                    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
                    let mut v_lo = vec![lo];
                    let mut v_hi = vec![hi];
                    silu(&mut v_lo);
                    silu(&mut v_hi);
                    prop_assert!(
                        v_hi[0] > v_lo[0],
                        "FALSIFIED SI-003-prop: SiLU({hi})={} not > SiLU({lo})={}",
                        v_hi[0], v_lo[0]
                    );
                }
            }
        }

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(200))]
            #[test]
            fn falsify_si_005_prop_asymptotic(x in 10.0_f32..500.0) {
                let mut input = vec![x];
                silu(&mut input);
                prop_assert!(
                    (input[0] - x).abs() < 0.01,
                    "FALSIFIED SI-005-prop: |SiLU({x}) - {x}| = {}",
                    (input[0] - x).abs()
                );
            }
        }
    }
}

// =========================================================================
// FALSIFY-GE: gelu-kernel-v1.yaml contract (realizar gelu in-place)
// =========================================================================
#[cfg(test)]
mod gelu_contract_tests {
    use super::*;

    /// FALSIFY-GE-001: Non-negativity — gelu(x) >= 0 for positive x
    #[test]
    fn falsify_ge_001_non_negativity() {
        let mut input = vec![0.001, 0.1, 1.0, 5.0, 10.0, 100.0];
        gelu(&mut input);
        for (i, &val) in input.iter().enumerate() {
            assert!(
                val >= 0.0,
                "FALSIFIED GE-001: gelu(positive)[{i}] = {val} < 0"
            );
        }
    }

    /// FALSIFY-GE-002: Monotonicity — ordering preserved for positive inputs
    #[test]
    fn falsify_ge_002_positive_monotonicity() {
        let mut input = vec![0.1, 0.5, 1.0, 2.0, 5.0, 10.0];
        gelu(&mut input);
        for i in 1..input.len() {
            assert!(
                input[i] > input[i - 1],
                "FALSIFIED GE-002: gelu not monotonic: [{i}]={} not > [{}]={}",
                input[i],
                i - 1,
                input[i - 1]
            );
        }
    }

    /// FALSIFY-GE-003: Zero preservation — gelu(0) = 0
    #[test]
    fn falsify_ge_003_zero_preservation() {
        let mut input = vec![0.0];
        gelu(&mut input);
        assert!(
            input[0].abs() < 1e-7,
            "FALSIFIED GE-003: gelu(0) = {}",
            input[0]
        );
    }

    /// FALSIFY-GE-006: Large input stability
    #[test]
    fn falsify_ge_006_large_input_stability() {
        let mut pos = vec![10.0, 50.0, 100.0];
        let mut neg = vec![-10.0, -50.0, -100.0];
        gelu(&mut pos);
        gelu(&mut neg);

        for (i, (&val, &orig)) in pos.iter().zip([10.0, 50.0, 100.0].iter()).enumerate() {
            assert!(
                (val - orig).abs() < 0.01,
                "FALSIFIED GE-006: gelu({orig}) = {val}, expected ≈ {orig}"
            );
        }
        for (i, &val) in neg.iter().enumerate() {
            assert!(
                val.abs() < 0.01,
                "FALSIFIED GE-006: gelu(neg)[{i}] = {val}, expected ≈ 0"
            );
        }
    }

    /// FALSIFY-GE-005: Tanh approximation accuracy
    #[test]
    fn falsify_ge_005_tanh_approx_accuracy() {
        use std::f32::consts::FRAC_2_PI;
        let c = FRAC_2_PI.sqrt();
        for x_int in -100..=100 {
            let x = x_int as f32 * 0.1;
            let mut input = vec![x];
            gelu(&mut input);
            let inner = c * (x + 0.044_715 * x * x * x);
            let expected = 0.5 * x * (1.0 + inner.tanh());
            assert!(
                (input[0] - expected).abs() < 0.005,
                "FALSIFIED GE-005: |gelu_approx({x}) - exact| = {}",
                (input[0] - expected).abs()
            );
        }
    }

    mod ge_proptest_falsify {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(500))]
            #[test]
            fn falsify_ge_001_prop_non_negativity(x in 0.0_f32..1000.0) {
                let mut input = vec![x];
                gelu(&mut input);
                prop_assert!(input[0] >= 0.0, "FALSIFIED GE-001-prop: gelu({x}) = {} < 0", input[0]);
            }
        }

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(300))]
            #[test]
            fn falsify_ge_002_prop_monotonic_positive(
                a in 0.001_f32..100.0,
                b in 0.001_f32..100.0,
            ) {
                if a != b {
                    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
                    let mut v_lo = vec![lo];
                    let mut v_hi = vec![hi];
                    gelu(&mut v_lo);
                    gelu(&mut v_hi);
                    prop_assert!(
                        v_hi[0] > v_lo[0],
                        "FALSIFIED GE-002-prop: gelu({hi})={} not > gelu({lo})={}",
                        v_hi[0], v_lo[0]
                    );
                }
            }
        }

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(200))]
            #[test]
            fn falsify_ge_006_prop_large_positive(x in 10.0_f32..500.0) {
                let mut input = vec![x];
                gelu(&mut input);
                prop_assert!(
                    (input[0] - x).abs() < 0.01,
                    "FALSIFIED GE-006-prop: |gelu({x}) - {x}| = {}",
                    (input[0] - x).abs()
                );
            }
        }
    }
}

// =========================================================================
// FALSIFY-SG: swiglu-kernel-v1.yaml contract (realizar fused_swiglu)
//
// Five-Whys (PMAT-354, Phase 11):
//   Why 1: realizar had 10+ swiglu unit tests but zero FALSIFY-SG-* tests
//   Why 2: unit tests verify SIMD parity, not mathematical invariants
//   Why 3: no mapping from swiglu-kernel-v1.yaml to realizar test names
//   Why 4: realizar predates the provable-contracts YAML convention
//   Why 5: SwiGLU was "obviously correct" (SiLU(gate) * up)
//
// Note: realizar's SwiGLU API is fused_swiglu_simd(gate, up) which modifies
// gate in-place. Tests use the silu() + element-wise multiply decomposition.
//
// References:
//   - provable-contracts/contracts/swiglu-kernel-v1.yaml
//   - Shazeer (2020) "GLU Variants Improve Transformer"
// =========================================================================

#[cfg(test)]
mod swiglu_contract_tests {
    use super::*;

    /// Scalar reference: SwiGLU(gate, up) = SiLU(gate) * up
    fn swiglu_ref(gate: f32, up: f32) -> f32 {
        trueno::silu_scalar(gate) * up
    }

    /// FALSIFY-SG-001: Zero preservation — SwiGLU(0, up) = 0 for any up
    #[test]
    fn falsify_sg_001_zero_gate_preservation() {
        for &up in &[-10.0f32, -1.0, 0.0, 1.0, 10.0] {
            let mut gate = vec![0.0];
            let up_vec = vec![up];
            silu(&mut gate);
            gate[0] *= up_vec[0];
            assert!(
                gate[0].abs() < 1e-7,
                "FALSIFIED SG-001: SwiGLU(0, {up}) = {}",
                gate[0]
            );
        }
    }

    /// FALSIFY-SG-002: Fused equivalence — fused matches decomposed
    #[test]
    fn falsify_sg_002_fused_equivalence() {
        let cases: Vec<(f32, f32)> = vec![
            (1.0, 1.0),
            (-2.0, 3.0),
            (5.0, -1.0),
            (0.5, 0.5),
            (100.0, 0.0),
        ];
        for &(g, u) in &cases {
            let expected = swiglu_ref(g, u);
            // Use the in-place silu + multiply approach
            let mut gate = vec![g];
            silu(&mut gate);
            let actual = gate[0] * u;
            assert!(
                (actual - expected).abs() < 1e-5,
                "FALSIFIED SG-002: silu({g})*{u} = {actual}, expected {expected}"
            );
        }
    }

    /// FALSIFY-SG-003: SiLU lower bound in gate — SiLU(z) > -0.279
    #[test]
    fn falsify_sg_003_silu_lower_bound() {
        let mut gates = vec![-1000.0f32, -1.278, -1.0, 0.0, 1.0, 1000.0];
        let orig = gates.clone();
        silu(&mut gates);
        for (i, &val) in gates.iter().enumerate() {
            assert!(val > -0.28, "FALSIFIED SG-003: SiLU({}) = {val}", orig[i]);
        }
    }

    /// FALSIFY-SG-004: Finite output for all finite inputs
    #[test]
    fn falsify_sg_004_finite_output() {
        let vals = vec![-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0];
        for &g in &vals {
            for &u in &vals {
                let y = swiglu_ref(g, u);
                assert!(y.is_finite(), "FALSIFIED SG-004: SwiGLU({g},{u}) = {y}");
            }
        }
    }

    /// FALSIFY-SG-005: Empty input produces empty output
    #[test]
    fn falsify_sg_005_empty_input() {
        let mut gate: Vec<f32> = vec![];
        silu(&mut gate);
        assert!(
            gate.is_empty(),
            "FALSIFIED SG-005: empty SiLU produced non-empty"
        );
    }

    /// FALSIFY-SG-006: Monotonicity of gate — for positive x and positive gates
    #[test]
    fn falsify_sg_006_gate_monotonicity() {
        let up = 5.0f32;
        let gate_values: Vec<f32> = vec![0.1, 0.5, 1.0, 2.0, 5.0, 10.0];
        let results: Vec<f32> = gate_values.iter().map(|&g| swiglu_ref(g, up)).collect();
        for i in 1..results.len() {
            assert!(
                results[i] > results[i - 1],
                "FALSIFIED SG-006: SwiGLU({},{up}) = {} not > SwiGLU({},{up}) = {}",
                gate_values[i],
                results[i],
                gate_values[i - 1],
                results[i - 1]
            );
        }
    }

    mod sg_proptest_falsify {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(300))]
            #[test]
            fn falsify_sg_001_prop_zero_gate(up in -100.0_f32..100.0) {
                let y = swiglu_ref(0.0, up);
                prop_assert!(y.abs() < 1e-6, "FALSIFIED SG-001-prop: SwiGLU(0,{up}) = {y}");
            }
        }

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(300))]
            #[test]
            fn falsify_sg_004_prop_finite(
                gate in -100.0_f32..100.0,
                up in -100.0_f32..100.0,
            ) {
                let y = swiglu_ref(gate, up);
                prop_assert!(y.is_finite(), "FALSIFIED SG-004-prop: SwiGLU({gate},{up}) = {y}");
            }
        }

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(200))]
            #[test]
            fn falsify_sg_006_prop_gate_monotonic(
                up in 1.0_f32..50.0,
                a in 0.1_f32..50.0,
                b in 0.1_f32..50.0,
            ) {
                if a != b {
                    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
                    let y_lo = swiglu_ref(lo, up);
                    let y_hi = swiglu_ref(hi, up);
                    prop_assert!(
                        y_hi > y_lo,
                        "FALSIFIED SG-006-prop: SwiGLU({hi},{up})={y_hi} not > SwiGLU({lo},{up})={y_lo}"
                    );
                }
            }
        }
    }
}

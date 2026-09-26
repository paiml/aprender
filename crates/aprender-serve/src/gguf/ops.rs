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

/// PMAT-809 (b): Gemma RMSNorm with `(1 + weight)` unit offset.
///
/// Gemma stores RMSNorm weights centered at 0, so the effective per-channel
/// scale is `(1 + w[j])`, NOT `w[j]`. Formula:
///   `output = x / sqrt(mean(x^2) + eps) * (1 + weight)`
///
/// This MUST only be used for Gemma-family models (`GGUFConfig::rmsnorm_unit_offset`).
/// Applying it to LLaMA/Qwen/Mistral (whose weights are centered at 1) would add a
/// spurious +1 and produce wrong output — hence it is a separate function gated at
/// the call site, never a silent default.
#[contract("forward-pass-v1", equation = "rms_norm")]
pub fn rms_norm_unit_offset(input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    contract_pre_rmsnorm!(input);
    let hidden_dim = weight.len();
    let seq_len = input.len() / hidden_dim;
    let mut output = Vec::with_capacity(input.len());

    for i in 0..seq_len {
        let start = i * hidden_dim;
        let end = start + hidden_dim;
        let x = &input[start..end];

        let sum_sq: f32 = x.iter().map(|v| v * v).sum();
        let mean_sq = sum_sq / hidden_dim as f32;
        let inv_rms = 1.0 / (mean_sq + eps).sqrt();

        for j in 0..hidden_dim {
            // (1 + w) unit offset — the load-bearing Gemma difference.
            output.push(x[j] * inv_rms * (1.0 + weight[j]));
        }
    }

    output
}

/// PMAT-809 (b): Gemma `(1 + weight)` RMSNorm into a pre-allocated buffer.
///
/// Zero-allocation single-position variant of [`rms_norm_unit_offset`] for the
/// decode hot path. See that function for the formula and gating rationale.
pub fn rms_norm_unit_offset_into(input: &[f32], weight: &[f32], eps: f32, output: &mut [f32]) {
    let hidden_dim = weight.len();
    let x = &input[..hidden_dim];

    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let mean_sq = sum_sq / hidden_dim as f32;
    let inv_rms = 1.0 / (mean_sq + eps).sqrt();

    for j in 0..hidden_dim {
        output[j] = x[j] * inv_rms * (1.0 + weight[j]);
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

/// PMAT-810: Gemma2 tanh logit softcapping, in place.
///
/// Gemma2 (and Gemma3) bound both the attention logits (cap = 50.0) and the
/// final lm_head logits (cap = 30.0) with `cap * tanh(x / cap)`. This squashes
/// extreme values into `(-cap, cap)` while staying near-linear for small `x`.
/// Without it, Gemma2 output diverges from the reference because the uncapped
/// attention scores and final logits put mass on the wrong tokens.
///
/// llama.cpp: `ggml_tanh(ggml_scale(kq, 1/cap)) * cap` (build_attn) and the same
/// on `cur` after the output layer when `hparams.f_logit_scale`/softcapping set.
///
/// # Arguments
/// * `values` - logits / scores (modified in place)
/// * `cap` - softcap constant (must be finite and > 0)
#[inline]
pub fn softcap(values: &mut [f32], cap: f32) {
    if cap <= 0.0 || !cap.is_finite() {
        return;
    }
    let inv_cap = 1.0 / cap;
    for v in values.iter_mut() {
        *v = cap * (*v * inv_cap).tanh();
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
    softmax_scalar_in_place(logits, SoftmaxNorm::MulInv);
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

/// RoPE pairing convention.
///
/// `Norm` rotates adjacent pairs `(2i, 2i + 1)` (LLaMA; GGUF `rope_type` 0).
/// `Neox` rotates split halves `(i, i + head_dim / 2)` (GPT-NeoX, Qwen; GGUF
/// `rope_type` 2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeStyle {
    /// Adjacent pairs `(2i, 2i + 1)`.
    Norm,
    /// Split halves `(i, i + head_dim / 2)`.
    Neox,
}

impl RopeStyle {
    /// GGUF `rope_type` 2 is NeoX; every other value is the adjacent-pair style.
    #[must_use]
    pub fn from_rope_type(rope_type: u32) -> Self {
        if rope_type == 2 {
            Self::Neox
        } else {
            Self::Norm
        }
    }
}

/// Rotary position embedding, in place, over `num_heads` contiguous heads of
/// `head_dim` elements in `x` (PP-ARCH-001 §9.2, the shared RoPE home).
///
/// Pair `i` rotates by `position * theta^(-2i / head_dim)`, computed in scalar
/// f32 as `1.0 / theta.powf(2.0 * i / head_dim)` with no fused multiply-add, so
/// the result is bit-identical to the per-site loops this replaced. The
/// sin/cos table is built once per call and shared by every head. A head that
/// does not fit in `x` is skipped, never partially rotated.
pub fn rope_into(
    x: &mut [f32],
    num_heads: usize,
    head_dim: usize,
    position: usize,
    theta: f32,
    style: RopeStyle,
) {
    let half_dim = head_dim / 2;
    if half_dim == 0 {
        return;
    }
    let pos_f32 = position as f32;
    let head_dim_f32 = head_dim as f32;

    let mut stack = [0.0f32; 256];
    let mut heap = Vec::new();
    let table: &mut [f32] = if half_dim <= 128 {
        &mut stack[..2 * half_dim]
    } else {
        heap.resize(2 * half_dim, 0.0);
        &mut heap
    };
    let (sin_t, cos_t) = table.split_at_mut(half_dim);
    for i in 0..half_dim {
        let freq = 1.0 / theta.powf(2.0 * i as f32 / head_dim_f32);
        let (sin_v, cos_v) = (pos_f32 * freq).sin_cos();
        sin_t[i] = sin_v;
        cos_t[i] = cos_v;
    }

    for h in 0..num_heads {
        let start = h * head_dim;
        let Some(head) = x.get_mut(start..start + head_dim) else {
            break;
        };
        for i in 0..half_dim {
            let (a, b) = match style {
                RopeStyle::Neox => (i, i + half_dim),
                RopeStyle::Norm => (2 * i, 2 * i + 1),
            };
            let x0 = head[a];
            let x1 = head[b];
            head[a] = x0 * cos_t[i] - x1 * sin_t[i];
            head[b] = x0 * sin_t[i] + x1 * cos_t[i];
        }
    }
}

#[cfg(test)]
mod rope_into_equivalence_tests {
    use super::{rope_into, RopeStyle};

    /// FROZEN copy of the per-site loop that `rope_into` replaced (#3422): the
    /// trig recomputed per head per pair, heads that do not fit skipped. Do not
    /// "improve" it; it is the bit-equality oracle for the migration.
    fn frozen_reference(
        x: &mut [f32],
        num_heads: usize,
        head_dim: usize,
        position: usize,
        theta: f32,
        rope_type: u32,
    ) {
        let half_dim = head_dim / 2;
        for h in 0..num_heads {
            let head_start = h * head_dim;
            if head_start + head_dim > x.len() {
                continue;
            }
            for i in 0..half_dim {
                let freq = 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32);
                let angle = position as f32 * freq;
                let cos_val = angle.cos();
                let sin_val = angle.sin();
                let (i1, i2) = if rope_type == 2 {
                    (head_start + i, head_start + half_dim + i)
                } else {
                    (head_start + 2 * i, head_start + 2 * i + 1)
                };
                let x1 = x[i1];
                let x2 = x[i2];
                x[i1] = x1 * cos_val - x2 * sin_val;
                x[i2] = x1 * sin_val + x2 * cos_val;
            }
        }
    }

    fn input(n: usize, seed: u32) -> Vec<f32> {
        let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
        (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (state as f32 / u32::MAX as f32) * 8.0 - 4.0
            })
            .collect()
    }

    #[test]
    fn rope_into_is_bit_identical_to_the_frozen_per_site_loop() {
        let mut cases = 0;
        for &head_dim in &[2usize, 64, 80, 128, 256, 512] {
            for &num_heads in &[1usize, 3, 8] {
                for &position in &[0usize, 1, 17, 4095, 131_071] {
                    for &theta in &[10_000.0f32, 1_000_000.0] {
                        for rope_type in [0u32, 2] {
                            // One extra partial head: it must be left untouched.
                            let n = num_heads * head_dim + head_dim / 2;
                            let mut want = input(n, (head_dim * 31 + position) as u32);
                            let mut got = want.clone();
                            frozen_reference(
                                &mut want,
                                num_heads + 1,
                                head_dim,
                                position,
                                theta,
                                rope_type,
                            );
                            rope_into(
                                &mut got,
                                num_heads + 1,
                                head_dim,
                                position,
                                theta,
                                RopeStyle::from_rope_type(rope_type),
                            );
                            let want_bits: Vec<u32> = want.iter().map(|v| v.to_bits()).collect();
                            let got_bits: Vec<u32> = got.iter().map(|v| v.to_bits()).collect();
                            assert_eq!(got_bits, want_bits, "head_dim={head_dim} heads={num_heads} pos={position} theta={theta} rope_type={rope_type}");
                            cases += 1;
                        }
                    }
                }
            }
        }
        assert_eq!(cases, 6 * 3 * 5 * 2 * 2);
    }

    #[test]
    fn rope_into_rotates_position_zero_to_identity_and_nonzero_away_from_it() {
        let orig = input(128, 7);
        let mut x = orig.clone();
        rope_into(&mut x, 2, 64, 0, 10_000.0, RopeStyle::Neox);
        assert_eq!(x, orig);
        rope_into(&mut x, 2, 64, 5, 10_000.0, RopeStyle::Neox);
        assert_ne!(x, orig);
    }
}

/// How a scalar RMSNorm site applies `rms` and the weight (PP-ARCH-001 §9.8).
///
/// The three forms round differently, so each migrated site keeps the one it had.
/// Merging them into one form changes numerics and needs a parity receipt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RmsScale {
    /// `(x / rms) * w`
    Divide,
    /// `(x * (1 / rms)) * w`
    ScaleThenWeight,
    /// `x * ((1 / rms) * w)`
    WeightedScale,
}

impl RmsScale {
    #[inline]
    fn apply(self, x: f32, rms: f32, inv_rms: f32, w: f32) -> f32 {
        match self {
            Self::Divide => x / rms * w,
            Self::ScaleThenWeight => x * inv_rms * w,
            Self::WeightedScale => x * (inv_rms * w),
        }
    }
}

/// Scalar RMS with a sequential sum: `sqrt(sum(x^2) / n + eps)`.
///
/// Unlike [`rms_norm`], which sums with trueno SIMD, this adds left to right,
/// so it is bit-identical to the plain `iter().map(|v| v * v).sum()` loops it replaces.
#[inline]
pub fn rms_scalar(x: &[f32], eps: f32) -> f32 {
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    (sum_sq / x.len() as f32 + eps).sqrt()
}

/// Scalar RMSNorm of one row into `out`.
///
/// The RMS is taken over all of `x`. The output covers the shortest of `out`, `x` and `weight`.
pub fn rms_norm_scalar_into(x: &[f32], weight: &[f32], eps: f32, form: RmsScale, out: &mut [f32]) {
    let rms = rms_scalar(x, eps);
    let inv_rms = 1.0 / rms;
    for ((o, &xi), &wi) in out.iter_mut().zip(x).zip(weight) {
        *o = form.apply(xi, rms, inv_rms, wi);
    }
}

/// Scalar RMSNorm of one row in place. It covers the shorter of `x` and `weight`.
pub fn rms_norm_scalar_in_place(x: &mut [f32], weight: &[f32], eps: f32, form: RmsScale) {
    let rms = rms_scalar(x, eps);
    let inv_rms = 1.0 / rms;
    for (xi, &wi) in x.iter_mut().zip(weight) {
        *xi = form.apply(*xi, rms, inv_rms, wi);
    }
}

#[cfg(test)]
mod rms_norm_scalar_equivalence_tests {
    use super::{rms_norm_scalar_in_place, rms_norm_scalar_into, RmsScale};

    /// The per-site loops, frozen as they were before §9.8 migrated them.
    fn frozen(x: &[f32], w: &[f32], eps: f32, form: RmsScale) -> Vec<f32> {
        let n = x.len();
        match form {
            RmsScale::Divide => {
                let sum_sq: f32 = x.iter().map(|v| v * v).sum();
                let rms = (sum_sq / n as f32 + eps).sqrt();
                x.iter().zip(w).map(|(xi, wi)| (xi / rms) * wi).collect()
            },
            RmsScale::ScaleThenWeight => {
                let mut sum_sq = 0.0f32;
                for &v in x {
                    sum_sq += v * v;
                }
                let inv_rms = 1.0 / (sum_sq / n as f32 + eps).sqrt();
                (0..n).map(|i| x[i] * inv_rms * w[i]).collect()
            },
            RmsScale::WeightedScale => {
                let mut sum_sq = 0.0f32;
                for &v in x {
                    sum_sq += v * v;
                }
                let inv_rms = 1.0 / (sum_sq / n as f32 + eps).sqrt();
                let mut d = x.to_vec();
                for i in 0..n {
                    d[i] *= inv_rms * w[i];
                }
                d
            },
        }
    }

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 40) as f32 / (1u64 << 24) as f32) * 8.0 - 4.0
    }

    #[test]
    fn rms_norm_scalar_is_bit_identical_to_the_frozen_per_site_loops() {
        let mut seed = 0x3422_u64;
        let mut cases = 0;
        for n in [1usize, 2, 7, 64, 128, 896, 1536, 4096] {
            for eps in [1e-5f32, 1e-6] {
                for scale in [1e-3f32, 1.0, 300.0] {
                    let x: Vec<f32> = (0..n).map(|_| lcg(&mut seed) * scale).collect();
                    let w: Vec<f32> = (0..n).map(|_| lcg(&mut seed)).collect();
                    for form in [
                        RmsScale::Divide,
                        RmsScale::ScaleThenWeight,
                        RmsScale::WeightedScale,
                    ] {
                        let want: Vec<u32> = frozen(&x, &w, eps, form)
                            .iter()
                            .map(|v| v.to_bits())
                            .collect();
                        let mut out = vec![0.0f32; n];
                        rms_norm_scalar_into(&x, &w, eps, form, &mut out);
                        let got: Vec<u32> = out.iter().map(|v| v.to_bits()).collect();
                        assert_eq!(got, want, "into n={n} eps={eps} scale={scale} {form:?}");
                        let mut inp = x.clone();
                        rms_norm_scalar_in_place(&mut inp, &w, eps, form);
                        let got: Vec<u32> = inp.iter().map(|v| v.to_bits()).collect();
                        assert_eq!(got, want, "in_place n={n} eps={eps} scale={scale} {form:?}");
                        cases += 1;
                    }
                }
            }
        }
        assert_eq!(cases, 144);
    }

    #[test]
    fn the_three_forms_are_distinct_so_the_enum_is_load_bearing() {
        let mut seed = 7_u64;
        let x: Vec<f32> = (0..4096).map(|_| lcg(&mut seed)).collect();
        let w: Vec<f32> = (0..4096).map(|_| lcg(&mut seed)).collect();
        let run = |form| {
            let mut o = vec![0.0f32; x.len()];
            rms_norm_scalar_into(&x, &w, 1e-6, form, &mut o);
            o.iter().map(|v| v.to_bits()).collect::<Vec<u32>>()
        };
        let (a, b, c) = (
            run(RmsScale::Divide),
            run(RmsScale::ScaleThenWeight),
            run(RmsScale::WeightedScale),
        );
        assert_ne!(a, b);
        assert_ne!(b, c);
        assert_ne!(a, c);
    }
}

/// How a scalar softmax divides its exponentials by their sum (PP-ARCH-001
/// Phase 2 step 4, #3422). The per-site loops this replaced used both, and
/// `e / sum` and `e * (1.0 / sum)` round differently, so each migrated site
/// names the form it had and stays bit-identical.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SoftmaxNorm {
    /// `e / sum`
    Divide,
    /// `e * (1.0 / sum)`
    MulInv,
}

/// Max-subtracted exponentials, in place, and their sequential sum.
///
/// The max is `f32::max` folded from `-inf` (NaN entries are skipped by it);
/// the sum accumulates left to right from `0.0`, the order of every scalar
/// site this replaced (including `iter().sum()`, which adds the same way).
pub fn softmax_exp_in_place(x: &mut [f32]) -> f32 {
    let max_val = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }
    sum
}

/// Divide exponentials by `sum` in the given [`SoftmaxNorm`] form.
pub fn softmax_normalize(x: &mut [f32], sum: f32, form: SoftmaxNorm) {
    match form {
        SoftmaxNorm::Divide => {
            for v in x.iter_mut() {
                *v /= sum;
            }
        },
        SoftmaxNorm::MulInv => {
            let inv_sum = 1.0 / sum;
            for v in x.iter_mut() {
                *v *= inv_sum;
            }
        },
    }
}

/// Scalar softmax in place: [`softmax_exp_in_place`] then [`softmax_normalize`].
/// Sites that skip normalisation when the sum is not positive call the two
/// halves themselves.
pub fn softmax_scalar_in_place(x: &mut [f32], form: SoftmaxNorm) {
    let sum = softmax_exp_in_place(x);
    softmax_normalize(x, sum, form);
}

/// How one [`attend_row_scalar`] site turns a dot product into a score:
/// `dot * s` or `dot / d`. The two differ in the last bit, so each site keeps
/// its own form.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ScoreScale {
    /// `dot * s` (sites that precompute `1 / sqrt(head_dim)`).
    Mul(f32),
    /// `dot / d` (sites that divide by `sqrt(head_dim)` per score).
    Div(f32),
}

/// Softmax flavour of one [`attend_row_scalar`] site: the final division form,
/// and whether normalisation is skipped when the sum is not positive. The max
/// term always contributes `exp(0) = 1`, so the sum is `>= 1` or NaN and the
/// guard is kept only to mirror each site's source faithfully.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowSoftmax {
    /// `e / sum` or `e * (1.0 / sum)`.
    pub norm: SoftmaxNorm,
    /// Normalise only when `sum > 0.0`.
    pub guard_positive_sum: bool,
}

/// Scalar attention for ONE query head row over `n_keys` keys (PP-ARCH-001
/// §9.11, the shared one-row attention home).
///
/// `score_j = dot(q, key(j))` scaled per [`ScoreScale`], each dot summed left to right from
/// `0.0` in scalar f32; a full [`softmax_exp_in_place`] +
/// [`softmax_normalize`] over the scores; then `out[d] += w_j * value(j)[d]`
/// in key order. `out` (`[q.len()]`) is accumulated into, so callers pass a
/// zeroed row. `scores` is scratch, reused across calls. The closures carry
/// each site's layout and GQA mapping; `key(j)`/`value(j)` must be at least
/// `q.len()` long.
#[allow(clippy::too_many_arguments)]
pub fn attend_row_scalar<'a>(
    q: &[f32],
    n_keys: usize,
    key: impl Fn(usize) -> &'a [f32],
    value: impl Fn(usize) -> &'a [f32],
    scale: ScoreScale,
    softmax: RowSoftmax,
    scores: &mut Vec<f32>,
    out: &mut [f32],
) {
    let head_dim = q.len();
    scores.clear();
    for j in 0..n_keys {
        let k = key(j);
        let mut dot = 0.0f32;
        for d in 0..head_dim {
            dot += q[d] * k[d];
        }
        scores.push(match scale {
            ScoreScale::Mul(s) => dot * s,
            ScoreScale::Div(d) => dot / d,
        });
    }
    let sum = softmax_exp_in_place(scores);
    if !softmax.guard_positive_sum || sum > 0.0 {
        softmax_normalize(scores, sum, softmax.norm);
    }
    for (j, &w) in scores.iter().enumerate() {
        let v = value(j);
        for d in 0..head_dim {
            out[d] += w * v[d];
        }
    }
}

#[cfg(test)]
mod attend_row_scalar_equivalence_tests {
    use super::{attend_row_scalar, RowSoftmax, ScoreScale, SoftmaxNorm};

    // Frozen per-site bodies (before step 5b), one query row each. Layout:
    // k/v rows of `head_dim`, key j at `j * head_dim`.

    // apr_transformer/pmat-260.rs::compute_causal_gqa_attention (guarded Divide).
    fn ref_pmat260(q: &[f32], k: &[f32], v: &[f32], n: usize, scale: f32, out: &mut [f32]) {
        let hd = q.len();
        let mut scores = Vec::with_capacity(n);
        for j in 0..n {
            let mut score = 0.0f32;
            for d in 0..hd {
                score += q[d] * k[j * hd + d];
            }
            scores.push(score * scale);
        }
        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut exp_sum = 0.0f32;
        for s in &mut scores {
            *s = (*s - max_score).exp();
            exp_sum += *s;
        }
        if exp_sum > 0.0 {
            for s in &mut scores {
                *s /= exp_sum;
            }
        }
        for (j, &weight) in scores.iter().enumerate() {
            for d in 0..hd {
                out[d] += weight * v[j * hd + d];
            }
        }
    }

    // gpu/adapters/apr_q4k.rs::gqa_attention (unguarded Divide).
    fn ref_apr_q4k(q: &[f32], k: &[f32], v: &[f32], n: usize, scale: f32, out: &mut [f32]) {
        let hd = q.len();
        let mut scores = vec![0.0f32; n];
        for pos in 0..n {
            let mut dot = 0.0f32;
            for d in 0..hd {
                dot += q[d] * k[pos * hd + d];
            }
            scores[pos] = dot * scale;
        }
        let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut exp_sum = 0.0f32;
        for s in &mut scores {
            *s = (*s - max_score).exp();
            exp_sum += *s;
        }
        for s in &mut scores {
            *s /= exp_sum;
        }
        for pos in 0..n {
            let w = scores[pos];
            for d in 0..hd {
                out[d] += w * v[pos * hd + d];
            }
        }
    }

    // gpu/scheduler/kv_forward_block.rs::gqa_attention_with_kv and
    // gqa_incremental_attention (iterator dot and sum, unguarded Divide).
    fn ref_kv_forward(q: &[f32], k: &[f32], v: &[f32], n: usize, scale: f32, out: &mut [f32]) {
        let hd = q.len();
        let mut scores = Vec::with_capacity(n);
        for kpos in 0..n {
            let k_slice = &k[kpos * hd..kpos * hd + hd];
            let score: f32 = q.iter().zip(k_slice.iter()).map(|(&a, &b)| a * b).sum();
            scores.push(score * scale);
        }
        let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
        let sum: f32 = exp_scores.iter().sum();
        let weights: Vec<f32> = exp_scores.iter().map(|&e| e / sum).collect();
        for (kpos, &weight) in weights.iter().enumerate() {
            for d in 0..hd {
                out[d] += weight * v[kpos * hd + d];
            }
        }
    }

    // gguf/inference/rope.rs::causal_attention (softmax_simd: AVX2 max and
    // normalise on x86_64 hosts that have it, MulInv otherwise).
    fn ref_rope(q: &[f32], k: &[f32], v: &[f32], n: usize, scale: f32, out: &mut [f32]) {
        let hd = q.len();
        let mut scores = Vec::with_capacity(n);
        for j in 0..n {
            let mut score = 0.0f32;
            for d in 0..hd {
                score += q[d] * k[j * hd + d];
            }
            scores.push(score * scale);
        }
        crate::quantize::softmax_simd(&mut scores);
        for (j, &weight) in scores.iter().enumerate() {
            for d in 0..hd {
                out[d] += weight * v[j * hd + d];
            }
        }
    }

    // gguf/inference/forward/batched.rs::compute_attention_output
    // (`&[&[f32]]` rows, iterator dot, softmax_simd).
    fn ref_batched(q: &[f32], k: &[f32], v: &[f32], n: usize, scale: f32, out: &mut [f32]) {
        let hd = q.len();
        let k_vecs: Vec<&[f32]> = (0..n).map(|j| &k[j * hd..(j + 1) * hd]).collect();
        let v_vecs: Vec<&[f32]> = (0..n).map(|j| &v[j * hd..(j + 1) * hd]).collect();
        let mut scores = Vec::with_capacity(n);
        for k_head in &k_vecs {
            let score: f32 = q.iter().zip(k_head.iter()).map(|(a, b)| a * b).sum();
            scores.push(score * scale);
        }
        crate::quantize::softmax_simd(&mut scores);
        for (attn, v_head) in scores.iter().zip(v_vecs.iter()) {
            for (i, &v_val) in v_head.iter().enumerate() {
                out[i] += attn * v_val;
            }
        }
    }

    // gguf/inference/forward/forward_qwen35.rs::forward_attention
    // (`dot / sqrt(hd)`, ops::softmax). `_scale` is unused: this site divides.
    fn ref_qwen35(q: &[f32], k: &[f32], v: &[f32], n: usize, _scale: f32, out: &mut [f32]) {
        let hd = q.len();
        let mut scores = vec![0.0; n];
        for p in 0..n {
            let mut dot = 0.0;
            let k_p = &k[p * hd..(p + 1) * hd];
            for i in 0..hd {
                dot += q[i] * k_p[i];
            }
            scores[p] = dot / (hd as f32).sqrt();
        }
        crate::gguf::ops::softmax(&mut scores);
        for p in 0..n {
            let w = scores[p];
            let v_p = &v[p * hd..(p + 1) * hd];
            for i in 0..hd {
                out[i] += w * v_p[i];
            }
        }
    }

    fn lcg(seed: &mut u64, mag: f32) -> f32 {
        *seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((*seed >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * mag
    }

    type Ref = fn(&[f32], &[f32], &[f32], usize, f32, &mut [f32]);

    #[test]
    fn shared_home_is_bit_identical_to_each_per_site_body() {
        let divide = |guard| RowSoftmax {
            norm: SoftmaxNorm::Divide,
            guard_positive_sum: guard,
        };
        let mul_inv = RowSoftmax {
            norm: SoftmaxNorm::MulInv,
            guard_positive_sum: false,
        };
        // (site, frozen body, softmax, divides by sqrt(hd) rather than
        // multiplying by its inverse)
        let sites: [(&str, Ref, RowSoftmax, bool); 6] = [
            ("pmat260", ref_pmat260, divide(true), false),
            ("apr_q4k", ref_apr_q4k, divide(false), false),
            ("kv_forward", ref_kv_forward, divide(false), false),
            ("rope", ref_rope, mul_inv, false),
            ("batched", ref_batched, mul_inv, false),
            ("qwen35", ref_qwen35, mul_inv, true),
        ];
        let mut cases = 0;
        let mut seed = 0x3422_005b_u64;
        let mut scratch = Vec::new();
        for (name, reference, sm, divides) in sites {
            for &hd in &[1usize, 8, 64, 128] {
                for &n in &[1usize, 2, 7, 64, 300] {
                    // mag 0.0 makes every dot an exact zero (the iterator-sum
                    // sign case); 60.0 drives most weights to underflow.
                    for &mag in &[0.0f32, 1e-3, 1.0, 8.0, 60.0] {
                        let q: Vec<f32> = (0..hd).map(|_| lcg(&mut seed, mag)).collect();
                        let k: Vec<f32> = (0..n * hd).map(|_| lcg(&mut seed, 1.0)).collect();
                        let v: Vec<f32> = (0..n * hd).map(|_| lcg(&mut seed, 4.0)).collect();
                        let scale = 1.0 / (hd as f32).sqrt();
                        let mut want = vec![0.0f32; hd];
                        let mut got = vec![0.0f32; hd];
                        reference(&q, &k, &v, n, scale, &mut want);
                        attend_row_scalar(
                            &q,
                            n,
                            |j| &k[j * hd..(j + 1) * hd],
                            |j| &v[j * hd..(j + 1) * hd],
                            if divides {
                                ScoreScale::Div((hd as f32).sqrt())
                            } else {
                                ScoreScale::Mul(scale)
                            },
                            sm,
                            &mut scratch,
                            &mut got,
                        );
                        assert_eq!(
                            want.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                            got.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                            "{name} hd={hd} n={n} mag={mag}"
                        );
                        cases += 1;
                    }
                }
            }
        }
        assert_eq!(cases, 600);
    }
}

/// Online-softmax attention for ONE query row over the first `n_keys` rows of
/// `k`/`v` (each `[_, head_dim]`), processed in tiles of `tile_size` keys
/// (PP-ARCH-001 §9.10, the shared tiled-attention home).
///
/// Scores are `dot(q_i, k_j) * scale`, dots summed left to right in scalar
/// f32. The running output is rescaled only when a tile raises the running
/// max. `out` (`[head_dim]`) is written with `acc / sum` only when the sum is
/// positive, so a caller's zeroed row stays zero otherwise. Bit-identical to
/// the causal, bidirectional and cross copies it replaced; they differ only in
/// `n_keys` (`i + 1`, `seq_len`, `encoder_len`).
#[allow(clippy::too_many_arguments)]
pub fn attend_row_online_tiled(
    q_i: &[f32],
    k: &[f32],
    v: &[f32],
    head_dim: usize,
    n_keys: usize,
    scale: f32,
    tile_size: usize,
    out: &mut [f32],
) {
    let tile_size = tile_size.max(1);
    let mut running_max = f32::NEG_INFINITY;
    let mut running_sum = 0.0f32;
    let mut running_output = vec![0.0f32; head_dim];

    for tile_start in (0..n_keys).step_by(tile_size) {
        let tile_end = (tile_start + tile_size).min(n_keys);

        let mut tile_scores = Vec::with_capacity(tile_end - tile_start);
        for j in tile_start..tile_end {
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q_i[d] * k[j * head_dim + d];
            }
            tile_scores.push(dot * scale);
        }

        let tile_max = tile_scores
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let new_max = running_max.max(tile_max);
        if new_max > running_max && running_sum > 0.0 {
            let rescale = (running_max - new_max).exp();
            running_sum *= rescale;
            for out_val in &mut running_output {
                *out_val *= rescale;
            }
        }
        running_max = new_max;

        for (idx, &score) in tile_scores.iter().enumerate() {
            let j = tile_start + idx;
            let weight = (score - running_max).exp();
            running_sum += weight;
            for d in 0..head_dim {
                running_output[d] += weight * v[j * head_dim + d];
            }
        }
    }

    if running_sum > 0.0 {
        for d in 0..head_dim {
            out[d] = running_output[d] / running_sum;
        }
    }
}

#[cfg(test)]
mod attend_row_online_tiled_equivalence_tests {
    use super::attend_row_online_tiled;

    // Frozen copy of the per-site body (tiled_causal/bidirectional/cross,
    // batch_tiled_causal_owned.rs before step 5), for one query row.
    #[allow(clippy::too_many_arguments)]
    fn reference(
        q_i: &[f32],
        k: &[f32],
        v: &[f32],
        head_dim: usize,
        n_keys: usize,
        scale: f32,
        tile_size: usize,
        out: &mut [f32],
    ) {
        let mut running_max = f32::NEG_INFINITY;
        let mut running_sum = 0.0f32;
        let mut running_output = vec![0.0f32; head_dim];
        for tile_start in (0..n_keys).step_by(tile_size) {
            let tile_end = (tile_start + tile_size).min(n_keys);
            let mut tile_scores = Vec::with_capacity(tile_end - tile_start);
            for j in tile_start..tile_end {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q_i[d] * k[j * head_dim + d];
                }
                tile_scores.push(dot * scale);
            }
            let tile_max = tile_scores
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            let new_max = running_max.max(tile_max);
            if new_max > running_max && running_sum > 0.0 {
                let rescale = (running_max - new_max).exp();
                running_sum *= rescale;
                for out_val in &mut running_output {
                    *out_val *= rescale;
                }
            }
            running_max = new_max;
            for (idx, &score) in tile_scores.iter().enumerate() {
                let j = tile_start + idx;
                let weight = (score - running_max).exp();
                running_sum += weight;
                for d in 0..head_dim {
                    running_output[d] += weight * v[j * head_dim + d];
                }
            }
        }
        if running_sum > 0.0 {
            for d in 0..head_dim {
                out[d] = running_output[d] / running_sum;
            }
        }
    }

    fn lcg(seed: &mut u64, mag: f32) -> f32 {
        *seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((*seed >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * mag
    }

    #[test]
    fn shared_home_is_bit_identical_to_the_per_site_body() {
        let mut cases = 0;
        let mut seed = 0x3422_0005_u64;
        for &head_dim in &[1usize, 8, 64, 128] {
            for &n_keys in &[0usize, 1, 3, 17, 64, 130] {
                for &tile in &[1usize, 4, 16, 64, 256] {
                    for &mag in &[1e-3f32, 1.0, 8.0, 60.0] {
                        let q: Vec<f32> = (0..head_dim).map(|_| lcg(&mut seed, mag)).collect();
                        let k: Vec<f32> = (0..n_keys * head_dim)
                            .map(|_| lcg(&mut seed, mag))
                            .collect();
                        let v: Vec<f32> = (0..n_keys * head_dim)
                            .map(|_| lcg(&mut seed, mag))
                            .collect();
                        let scale = 1.0 / (head_dim as f32).sqrt();
                        let mut want = vec![0.0f32; head_dim];
                        let mut got = vec![0.0f32; head_dim];
                        reference(&q, &k, &v, head_dim, n_keys, scale, tile, &mut want);
                        attend_row_online_tiled(
                            &q, &k, &v, head_dim, n_keys, scale, tile, &mut got,
                        );
                        let wb: Vec<u32> = want.iter().map(|x| x.to_bits()).collect();
                        let gb: Vec<u32> = got.iter().map(|x| x.to_bits()).collect();
                        assert_eq!(wb, gb, "hd={head_dim} n={n_keys} tile={tile} mag={mag}");
                        cases += 1;
                    }
                }
            }
        }
        assert_eq!(cases, 480);
    }

    #[test]
    fn tile_size_zero_is_clamped_to_one_like_the_callers_did() {
        let q = [0.5f32, -1.0];
        let k = [1.0f32, 2.0, -0.5, 0.25];
        let v = [3.0f32, 4.0, 5.0, 6.0];
        let mut a = [0.0f32; 2];
        let mut b = [0.0f32; 2];
        attend_row_online_tiled(&q, &k, &v, 2, 2, 0.7, 0, &mut a);
        attend_row_online_tiled(&q, &k, &v, 2, 2, 0.7, 1, &mut b);
        assert_eq!(a.map(f32::to_bits), b.map(f32::to_bits));
    }
}

#[cfg(test)]
mod softmax_scalar_equivalence_tests {
    use super::{softmax_exp_in_place, softmax_normalize, softmax_scalar_in_place, SoftmaxNorm};

    // Frozen copies of the per-site loops this step replaced.
    fn ref_divide(x: &mut [f32]) {
        let m = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut s = 0.0f32;
        for v in x.iter_mut() {
            *v = (*v - m).exp();
            s += *v;
        }
        for v in x.iter_mut() {
            *v /= s;
        }
    }
    fn ref_mul_inv(x: &mut [f32]) {
        let m = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut s = 0.0f32;
        for v in x.iter_mut() {
            *v = (*v - m).exp();
            s += *v;
        }
        let inv = 1.0 / s;
        for v in x.iter_mut() {
            *v *= inv;
        }
    }
    fn ref_collect_sum_divide(x: &[f32]) -> Vec<f32> {
        let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f32> = x.iter().map(|&v| (v - m).exp()).collect();
        let s: f32 = e.iter().sum();
        e.iter().map(|&v| v / s).collect()
    }
    fn ref_guarded_mul_inv(x: &mut [f32]) {
        let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut s = 0.0;
        for v in x.iter_mut() {
            *v = (*v - m).exp();
            s += *v;
        }
        if s > 0.0 {
            let inv = 1.0 / s;
            for v in x.iter_mut() {
                *v *= inv;
            }
        }
    }

    fn input(n: usize, scale: f32, seed: u32, masked: bool) -> Vec<f32> {
        let mut st = seed.wrapping_mul(2_654_435_761).wrapping_add(n as u32);
        (0..n)
            .map(|i| {
                st = st.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                if masked && i % 3 == 2 {
                    f32::NEG_INFINITY
                } else {
                    ((st >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * scale
                }
            })
            .collect()
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    #[test]
    fn softmax_scalar_is_bit_identical_to_the_frozen_per_site_loops() {
        let mut cases = 0;
        for &n in &[1usize, 2, 7, 64, 151, 1024, 32_000] {
            for &scale in &[1e-3f32, 1.0, 30.0, 300.0] {
                for &masked in &[false, true] {
                    let x = input(n, scale, 7, masked);

                    let (mut a, mut b) = (x.clone(), x.clone());
                    ref_divide(&mut a);
                    softmax_scalar_in_place(&mut b, SoftmaxNorm::Divide);
                    assert_eq!(
                        bits(&a),
                        bits(&b),
                        "Divide n={n} scale={scale} masked={masked}"
                    );

                    let (mut a, mut b) = (x.clone(), x.clone());
                    ref_mul_inv(&mut a);
                    softmax_scalar_in_place(&mut b, SoftmaxNorm::MulInv);
                    assert_eq!(
                        bits(&a),
                        bits(&b),
                        "MulInv n={n} scale={scale} masked={masked}"
                    );

                    let a = ref_collect_sum_divide(&x);
                    let mut b = x.clone();
                    softmax_scalar_in_place(&mut b, SoftmaxNorm::Divide);
                    assert_eq!(bits(&a), bits(&b), "iter().sum() n={n} scale={scale}");

                    let (mut a, mut b) = (x.clone(), x.clone());
                    ref_guarded_mul_inv(&mut a);
                    let s = softmax_exp_in_place(&mut b);
                    if s > 0.0 {
                        softmax_normalize(&mut b, s, SoftmaxNorm::MulInv);
                    }
                    assert_eq!(bits(&a), bits(&b), "guarded n={n} scale={scale}");
                    cases += 4;
                }
            }
        }
        assert_eq!(cases, 224);
    }

    #[test]
    fn the_two_forms_are_distinct_so_the_enum_is_load_bearing() {
        let x = input(1024, 30.0, 11, false);
        let (mut a, mut b) = (x.clone(), x);
        softmax_scalar_in_place(&mut a, SoftmaxNorm::Divide);
        softmax_scalar_in_place(&mut b, SoftmaxNorm::MulInv);
        assert_ne!(bits(&a), bits(&b));
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

#[cfg(test)]
mod gemma_rmsnorm_tests {
    use super::*;

    /// PMAT-809: the Gemma unit-offset RMSNorm equals the standard RMSNorm with
    /// `(1 + w)` substituted for `w` — i.e. `rms_norm_unit_offset(x, w) ==
    /// rms_norm(x, w+1)`. This is the load-bearing semantic difference.
    #[test]
    fn unit_offset_equals_standard_with_w_plus_one() {
        let x = vec![1.0f32, -2.0, 3.0, -4.0, 0.5, -0.25, 2.5, -1.5];
        let w = vec![0.1f32, -0.2, 0.3, 0.0, -0.5, 0.4, 0.05, -0.05];
        let w_plus_1: Vec<f32> = w.iter().map(|v| v + 1.0).collect();
        let eps = 1e-6;

        let unit_offset = rms_norm_unit_offset(&x, &w, eps);
        let standard_shifted = rms_norm(&x, &w_plus_1, eps);

        assert_eq!(unit_offset.len(), standard_shifted.len());
        for (a, b) in unit_offset.iter().zip(standard_shifted.iter()) {
            assert!((a - b).abs() < 1e-5, "unit-offset {a} != standard(w+1) {b}");
        }
    }

    /// The `_into` variant matches the allocating variant exactly (single position).
    #[test]
    fn unit_offset_into_matches_allocating() {
        let x = vec![0.7f32, -1.3, 2.1, -0.9, 1.1, -2.2, 0.4, 3.3];
        let w = vec![0.2f32, 0.0, -0.3, 0.5, -0.1, 0.6, -0.4, 0.15];
        let eps = 1e-6;
        let alloc = rms_norm_unit_offset(&x, &w, eps);
        let mut buf = vec![0.0f32; x.len()];
        rms_norm_unit_offset_into(&x, &w, eps, &mut buf);
        for (a, b) in alloc.iter().zip(buf.iter()) {
            assert!((a - b).abs() < 1e-6, "alloc {a} != into {b}");
        }
    }

    /// FALSIFIER: with zero-centered weights, unit-offset normalizes to RMS ≈ 1
    /// (since (1+0) == 1) while plain `rms_norm` would zero the output. This is
    /// exactly why Gemma (HF, weights centered at 0) needs the offset.
    #[test]
    fn zero_weights_give_unit_rms_under_offset_but_zero_under_standard() {
        let x = vec![1.0f32, 2.0, -3.0, 4.0, -1.0, 0.5, -2.5, 1.5];
        let w0 = vec![0.0f32; x.len()];
        let eps = 1e-6;

        let offset = rms_norm_unit_offset(&x, &w0, eps);
        let off_rms = (offset.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt();
        assert!((off_rms - 1.0).abs() < 1e-3, "offset RMS {off_rms} != 1");

        let standard = rms_norm(&x, &w0, eps);
        assert!(
            standard.iter().all(|v| v.abs() < 1e-6),
            "standard rms_norm with w=0 must be ~0, got {standard:?}"
        );
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

#[cfg(test)]
mod softcap_tests {
    use super::*;

    /// PMAT-810: softcap(x, cap) == cap * tanh(x / cap) elementwise.
    #[test]
    fn softcap_matches_reference() {
        let cap = 30.0f32;
        let mut v = vec![-100.0, -30.0, -1.0, 0.0, 1.0, 30.0, 100.0, 1000.0];
        let expect: Vec<f32> = v.iter().map(|&x| cap * (x / cap).tanh()).collect();
        softcap(&mut v, cap);
        for (got, want) in v.iter().zip(expect.iter()) {
            assert!((got - want).abs() < 1e-4, "softcap {got} != {want}");
        }
    }

    /// PMAT-810: softcap bounds every output into (-cap, cap).
    #[test]
    fn softcap_bounds_extremes() {
        let cap = 50.0f32;
        let mut v = vec![-1e9, -1e3, 1e3, 1e9, f32::MAX, f32::MIN];
        softcap(&mut v, cap);
        for &x in &v {
            assert!(x.abs() <= cap + 1e-3, "softcap output {x} escaped ±{cap}");
        }
    }

    /// PMAT-810: softcap is ~identity near 0 (tanh(t) ≈ t for small t).
    #[test]
    fn softcap_near_linear_at_origin() {
        let cap = 30.0f32;
        let mut v = vec![0.0, 0.01, -0.01, 0.5, -0.5];
        let original = v.clone();
        softcap(&mut v, cap);
        for (got, orig) in v.iter().zip(original.iter()) {
            // |error| grows like x^3/(3 cap^2); for |x|<=0.5, cap=30 it's ~1e-5.
            assert!(
                (got - orig).abs() < 1e-3,
                "softcap({orig}) = {got} not ≈ identity"
            );
        }
    }

    /// PMAT-810: a non-positive / non-finite cap is a no-op (defensive guard).
    #[test]
    fn softcap_zero_or_bad_cap_is_noop() {
        for bad in [0.0f32, -1.0, f32::NAN, f32::INFINITY] {
            let mut v = vec![1.0, 2.0, 3.0];
            let orig = v.clone();
            softcap(&mut v, bad);
            assert_eq!(v, orig, "softcap with cap={bad} must be a no-op");
        }
    }
}

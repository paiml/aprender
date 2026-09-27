//! GPU Scheduler Static Operations (PMAT-COMPLY)
//!
//! Extracted from model.rs for file health compliance.
//! Contains static helper functions for attention, normalization, and sampling.

use trueno::Vector;

/// Apply Rotary Position Embedding (RoPE) inline (Phase 21)
///
/// RoPE encodes position information by rotating pairs of elements
/// with position-dependent angles. This is CRITICAL for transformer attention.
///
/// # Arguments
/// * `x` - Mutable slice of Q or K vectors for a single position [num_heads * head_dim]
/// * `num_heads` - Number of attention heads
/// * `head_dim` - Dimension per head
/// * `rope_theta` - Base frequency (typically 10000.0)
/// * `position` - Token position for RoPE encoding
pub(super) fn apply_rope_inline(
    x: &mut [f32],
    num_heads: usize,
    head_dim: usize,
    rope_theta: f32,
    position: usize,
) {
    contract_pre_rope!(x);
    crate::gguf::ops::rope_into(
        x,
        num_heads,
        head_dim,
        position,
        rope_theta,
        crate::gguf::ops::RopeStyle::Neox,
    );
    contract_post_rope!(&x);
}

/// GQA multi-head attention (IMP-089, IMP-092, IMP-094)
///
/// IMP-094: trueno SIMD dot product (0.0 on a length mismatch, as before).
fn trueno_dot(a: &[f32], b: &[f32]) -> f32 {
    Vector::from_slice(a)
        .dot(&Vector::from_slice(b))
        .unwrap_or(0.0)
}

/// IMP-094: trueno SIMD softmax, in place; the shared scalar softmax if trueno
/// refuses the row.
fn trueno_softmax_in_place(row: &mut [f32]) {
    match Vector::from_slice(row).softmax() {
        Ok(w) => row.copy_from_slice(w.as_slice()),
        Err(_) => {
            crate::gguf::ops::softmax_scalar_in_place(row, crate::gguf::ops::SoftmaxNorm::Divide);
        },
    }
}

/// `out += w * v`, scalar (the SIMD benefit is marginal for small head_dim).
fn scalar_axpy(out: &mut [f32], w: f32, v: &[f32]) {
    for (o, &x) in out.iter_mut().zip(v) {
        *o += w * x;
    }
}

/// Grouped Query Attention where K/V have fewer heads than Q.
/// Each KV head serves (num_heads / num_kv_heads) Q heads.
///
/// IMP-094: Uses trueno SIMD-accelerated dot product and softmax
/// for ~10x speedup over scalar implementation.
///
/// Static method to avoid borrow conflicts with scheduler and weights.
#[allow(clippy::too_many_arguments)]
pub(super) fn gqa_multihead_attention(
    q: &[f32], // Q: [num_heads * head_dim]
    k: &[f32], // K: [kv_len * num_kv_heads * head_dim]
    v: &[f32], // V: [kv_len * num_kv_heads * head_dim]
    kv_len: usize,
    num_heads: usize,    // Number of Q heads
    num_kv_heads: usize, // Number of K/V heads (for GQA, < num_heads)
    head_dim: usize,
) -> Vec<f32> {
    contract_pre_attention!(q);
    let hidden_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let mut output = vec![0.0; hidden_dim];

    // PP-ARCH-001 §9.15: the shared cached-GQA home, with the last position as
    // its "current" K/V and trueno's SIMD dot and softmax as the kernels.
    // No keys means no weights: the output stays zero.
    if kv_len > 0 {
        let cached = (kv_len - 1) * kv_dim;
        crate::gguf::ops::attend_cached_gqa_into(
            q,
            &k[..cached],
            &v[..cached],
            &k[cached..cached + kv_dim],
            &v[cached..cached + kv_dim],
            &mut output,
            crate::gguf::ops::CachedGqa {
                num_heads,
                num_kv_heads,
                head_dim,
            },
            scale,
            None,
            trueno_dot,
            trueno_softmax_in_place,
            scalar_axpy,
        );
    }

    contract_post_attention!(&output);
    output
}

/// RMSNorm (Root Mean Square Layer Normalization)
///
/// PMAT-094 FIX: Qwen2, LLaMA, Mistral use RMSNorm, NOT LayerNorm.
/// Formula: output = x / sqrt(mean(x^2) + eps) * weight + bias
#[allow(clippy::cast_precision_loss)]
pub(crate) fn layer_norm_static(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    hidden_dim: usize,
    eps: f32,
) -> Vec<f32> {
    let num_rows = input.len() / hidden_dim;
    let mut output = Vec::with_capacity(input.len());

    for row in 0..num_rows {
        let start = row * hidden_dim;
        let row_data = &input[start..start + hidden_dim];

        // RMSNorm: compute root mean square (no mean subtraction!)
        let sum_sq: f32 = row_data.iter().map(|&x| x * x).sum();
        let rms = (sum_sq / hidden_dim as f32 + eps).sqrt();

        // Normalize and scale
        for (i, &x) in row_data.iter().enumerate() {
            let normalized = x / rms;
            output.push(normalized * weight[i] + bias[i]);
        }
    }

    output
}

/// Top-k sampling with temperature: one seeded draw through the shared sampler.
///
/// #3760: this sorted by probability and returned the FIRST entry, the argmax, so
/// GpuModel "top-k sampling" never drew and `temperature`/`top_k` changed nothing.
pub(super) fn sample_topk(
    logits: &[f32],
    temperature: f32,
    top_k: usize,
    rng: &mut rand::rngs::StdRng,
) -> usize {
    contract_pre_temperature!();
    let result = crate::sampling::draw_seeded(logits, temperature, top_k, 1.0, rng) as usize;
    contract_post_temperature!(&result);
    result
}

/// Transpose weight matrix from [rows, cols] to [cols, rows].
///
/// PMAT-285: Delegates to `contract_gate::transpose_f32` (single source of truth).
pub(super) fn transpose_weights(weights: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    crate::contract_gate::transpose_f32(weights, rows, cols)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_norm_static_single_row() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let bias = vec![0.0, 0.0, 0.0, 0.0];
        let eps = 1e-5;

        let output = layer_norm_static(&input, &weight, &bias, 4, eps);
        assert_eq!(output.len(), 4);

        // Verify RMSNorm: each element is x / rms
        let sum_sq: f32 = input.iter().map(|x| x * x).sum();
        let rms = (sum_sq / 4.0 + eps).sqrt();
        for (i, &x) in input.iter().enumerate() {
            let expected = x / rms;
            assert!((output[i] - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn test_layer_norm_static_with_weight_bias() {
        let input = vec![2.0, 2.0, 2.0, 2.0];
        let weight = vec![2.0, 2.0, 2.0, 2.0];
        let bias = vec![0.5, 0.5, 0.5, 0.5];
        let eps = 1e-5;

        let output = layer_norm_static(&input, &weight, &bias, 4, eps);

        // RMS of [2,2,2,2] = sqrt(16/4 + eps) = sqrt(4 + eps) ≈ 2.0
        // Normalized: 2.0 / 2.0 = 1.0
        // Scaled: 1.0 * 2.0 + 0.5 = 2.5
        for &val in &output {
            assert!((val - 2.5).abs() < 0.01);
        }
    }

    #[test]
    fn test_transpose_weights() {
        let weights = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
        let transposed = transpose_weights(&weights, 2, 3);
        // Expected: 3x2 = [1, 4, 2, 5, 3, 6]
        assert_eq!(transposed, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_sample_topk_deterministic() {
        let logits = vec![1.0, 5.0, 2.0, 0.5];
        let result = sample_topk(&logits, 1.0, 1, &mut rand::SeedableRng::seed_from_u64(1));
        assert_eq!(result, 1); // Highest logit is at index 1
    }

    #[test]
    fn test_apply_rope_inline() {
        let mut x = vec![1.0, 0.0, 0.0, 1.0]; // 1 head, head_dim=4
        apply_rope_inline(&mut x, 1, 4, 10000.0, 0);
        // At position 0, angle = 0, so cos=1, sin=0 -> no change
        assert!((x[0] - 1.0).abs() < 1e-5);
        assert!((x[1] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_gqa_multihead_attention_simple() {
        // 2 heads, 2 kv heads, head_dim=2, kv_len=1
        let q = vec![1.0, 0.0, 0.0, 1.0]; // 2 heads * 2 dim
        let k = vec![1.0, 0.0, 0.0, 1.0]; // 1 position * 2 kv_heads * 2 dim
        let v = vec![1.0, 2.0, 3.0, 4.0]; // Same shape as k

        let output = gqa_multihead_attention(&q, &k, &v, 1, 2, 2, 2);
        assert_eq!(output.len(), 4);
        // With softmax over single position, weights are all 1.0
        // So output should be same as v
        assert!((output[0] - 1.0).abs() < 1e-5);
        assert!((output[1] - 2.0).abs() < 1e-5);
    }
}

/// PP-ARCH-001 §9.15: `gqa_multihead_attention` moved onto the cached-GQA home.
/// The frozen pre-move body must agree bit for bit.
#[cfg(test)]
mod gqa_multihead_equivalence_tests {
    use super::gqa_multihead_attention;
    use trueno::Vector;

    fn frozen(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        kv_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let hidden_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let heads_per_kv = num_heads / num_kv_heads;
        let mut output = vec![0.0; hidden_dim];
        for h in 0..num_heads {
            let q_vec = Vector::from_slice(&q[h * head_dim..(h + 1) * head_dim]);
            let kv_head = h / heads_per_kv;
            let mut scores = Vec::with_capacity(kv_len);
            for pos in 0..kv_len {
                let k_offset = pos * kv_dim + kv_head * head_dim;
                let k_vec = Vector::from_slice(&k[k_offset..k_offset + head_dim]);
                scores.push(q_vec.dot(&k_vec).unwrap_or(0.0) * scale);
            }
            let attn_weights: Vec<f32> = Vector::from_slice(&scores).softmax().map_or_else(
                |_| {
                    let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let exp_scores: Vec<f32> =
                        scores.iter().map(|&s| (s - max_score).exp()).collect();
                    let sum_exp: f32 = exp_scores.iter().sum();
                    exp_scores.iter().map(|&e| e / sum_exp).collect()
                },
                |v| v.as_slice().to_vec(),
            );
            for (pos, &weight) in attn_weights.iter().enumerate() {
                let v_offset = pos * kv_dim + kv_head * head_dim;
                let v_head = &v[v_offset..v_offset + head_dim];
                for d in 0..head_dim {
                    output[h * head_dim + d] += weight * v_head[d];
                }
            }
        }
        output
    }

    fn fill(n: usize, seed: u32, mag: f32) -> Vec<f32> {
        let mut x = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                ((x as f32 / u32::MAX as f32) * 2.0 - 1.0) * mag
            })
            .collect()
    }

    #[test]
    fn matches_frozen_body_bit_for_bit() {
        let mut cases = 0;
        for &(nh, nkv) in &[(1, 1), (4, 4), (4, 2), (8, 1), (6, 3)] {
            for &hd in &[1usize, 7, 8, 16, 33] {
                for &kv_len in &[0usize, 1, 2, 5, 17, 40] {
                    for &mag in &[0.1f32, 3.0, 40.0] {
                        // `extra` > 0: K/V buffers longer than kv_len rows.
                        for &extra in &[0usize, 2] {
                            let seed = (nh * 1000 + hd * 31 + kv_len * 7 + extra) as u32;
                            let q = fill(nh * hd, seed, mag);
                            let n = (kv_len + extra) * nkv * hd;
                            let k = fill(n, seed + 1, mag);
                            let v = fill(n, seed + 2, mag);
                            let got = gqa_multihead_attention(&q, &k, &v, kv_len, nh, nkv, hd);
                            let want = frozen(&q, &k, &v, kv_len, nh, nkv, hd);
                            assert_eq!(got.len(), want.len());
                            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                                assert_eq!(
                                    g.to_bits(),
                                    w.to_bits(),
                                    "nh={nh} nkv={nkv} hd={hd} kv_len={kv_len} mag={mag} extra={extra} i={i}: {g} vs {w}"
                                );
                            }
                            cases += 1;
                        }
                    }
                }
            }
        }
        assert_eq!(cases, 5 * 5 * 6 * 3 * 2);
    }
}

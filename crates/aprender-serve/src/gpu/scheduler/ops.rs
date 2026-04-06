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
    let half_dim = head_dim / 2;
    let head_dim_f32 = head_dim as f32;
    let pos_f32 = position as f32;

    for h in 0..num_heads {
        let head_start = h * head_dim;
        let idx2_start = head_start + half_dim;

        for i in 0..half_dim {
            let freq = 1.0 / rope_theta.powf(2.0 * i as f32 / head_dim_f32);
            let angle = pos_f32 * freq;
            let (sin_val, cos_val) = angle.sin_cos();

            let x1 = x[head_start + i];
            let x2 = x[idx2_start + i];

            // Apply rotation: [cos -sin; sin cos] * [x1; x2]
            x[head_start + i] = x1 * cos_val - x2 * sin_val;
            x[idx2_start + i] = x1 * sin_val + x2 * cos_val;
        }
    }
    contract_post_rope!(&x);
}

/// GQA multi-head attention (IMP-089, IMP-092, IMP-094)
///
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

    // Number of Q heads per KV head
    let heads_per_kv = num_heads / num_kv_heads;

    let mut output = vec![0.0; hidden_dim];

    // Compute attention for all Q heads
    for h in 0..num_heads {
        let q_head = &q[h * head_dim..(h + 1) * head_dim];
        // IMP-094: Create trueno vector for SIMD dot product
        let q_vec = Vector::from_slice(q_head);

        // Map Q head to KV head (GQA: multiple Q heads share one KV head)
        let kv_head = h / heads_per_kv;

        // Compute attention scores for this head using SIMD dot product
        let mut scores = Vec::with_capacity(kv_len);
        for pos in 0..kv_len {
            // K offset: pos * kv_dim + kv_head * head_dim
            let k_offset = pos * kv_dim + kv_head * head_dim;
            let cached_key = &k[k_offset..k_offset + head_dim];

            // IMP-094: SIMD dot product via trueno
            let k_vec = Vector::from_slice(cached_key);
            let score = q_vec.dot(&k_vec).unwrap_or(0.0) * scale;
            scores.push(score);
        }

        // IMP-094: SIMD softmax via trueno
        let scores_vec = Vector::from_slice(&scores);
        let attn_weights: Vec<f32> = scores_vec.softmax().map_or_else(
            |_| {
                // Fallback to scalar softmax
                let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
                let sum_exp: f32 = exp_scores.iter().sum();
                exp_scores.iter().map(|&e| e / sum_exp).collect()
            },
            |v| v.as_slice().to_vec(),
        );

        // Weighted sum of values (still scalar - SIMD benefit is marginal for small head_dim)
        for (pos, &weight) in attn_weights.iter().enumerate() {
            // V offset: pos * kv_dim + kv_head * head_dim
            let v_offset = pos * kv_dim + kv_head * head_dim;
            let v_head = &v[v_offset..v_offset + head_dim];

            for d in 0..head_dim {
                output[h * head_dim + d] += weight * v_head[d];
            }
        }
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

/// Top-k sampling with temperature (returns highest prob token in top-k for determinism)
pub(super) fn sample_topk(logits: &[f32], temperature: f32, top_k: usize) -> usize {
    contract_pre_temperature!();
    // Apply temperature
    let scaled: Vec<f32> = logits.iter().map(|&x| x / temperature).collect();

    // Softmax with numerical stability
    let max_logit = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_logits: Vec<f32> = scaled.iter().map(|&x| (x - max_logit).exp()).collect();
    let sum: f32 = exp_logits.iter().sum();
    let probs: Vec<f32> = exp_logits.iter().map(|&x| x / sum).collect();

    // Get top-k indices by sorting
    let mut indexed: Vec<(usize, f32)> = probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Truncate to top_k and return highest probability token (deterministic)
    indexed.truncate(top_k);
    let result = indexed.first().map_or(0, |&(idx, _)| idx);
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
        let result = sample_topk(&logits, 1.0, 1);
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

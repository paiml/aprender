/// Compute scaled dot-product scores between a query vector and key vectors.
///
/// For each key position `j` in `0..num_keys`, computes `dot(q, k[j]) * scale`.
fn compute_attention_scores(
    q: &[f32],
    q_start: usize,
    keys: &[f32],
    kv_dim: usize,
    kv_head_offset: usize,
    head_dim: usize,
    num_keys: usize,
    scale: f32,
) -> Vec<f32> {
    let mut scores = Vec::with_capacity(num_keys);
    for j in 0..num_keys {
        let k_start = j * kv_dim + kv_head_offset;
        let mut score = 0.0f32;
        for d in 0..head_dim {
            score += q[q_start + d] * keys[k_start + d];
        }
        scores.push(score * scale);
    }
    scores
}

/// Apply softmax normalization in-place.
fn softmax_inplace(scores: &mut [f32]) {
    let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut exp_sum = 0.0f32;
    for s in scores.iter_mut() {
        *s = (*s - max_score).exp();
        exp_sum += *s;
    }
    for s in scores.iter_mut() {
        *s /= exp_sum;
    }
}

/// Accumulate weighted value vectors into the output buffer.
///
/// For each position `j`, adds `weight[j] * v[j]` element-wise into `output[out_start..]`.
fn accumulate_weighted_values(
    output: &mut [f32],
    out_start: usize,
    scores: &[f32],
    values: &[f32],
    kv_dim: usize,
    kv_head_offset: usize,
    head_dim: usize,
) {
    for (j, &weight) in scores.iter().enumerate() {
        let v_start = j * kv_dim + kv_head_offset;
        for d in 0..head_dim {
            output[out_start + d] += weight * values[v_start + d];
        }
    }
}

/// Merge per-head output buffers into a single interleaved output tensor.
///
/// Each `head_out` has shape `[seq_len, head_dim]`; the merged output has
/// shape `[seq_len, num_heads * head_dim]`.
fn merge_head_outputs(
    head_outputs: Vec<Vec<f32>>,
    seq_len: usize,
    head_dim: usize,
    q_dim: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; seq_len * q_dim];
    for (head, head_out) in head_outputs.into_iter().enumerate() {
        let head_offset = head * head_dim;
        for i in 0..seq_len {
            let src_start = i * head_dim;
            let dst_start = i * q_dim + head_offset;
            output[dst_start..dst_start + head_dim]
                .copy_from_slice(&head_out[src_start..src_start + head_dim]);
        }
    }
    output
}

/// Compute attention for one position of one head, writing into a per-head buffer.
///
/// Used by the parallel path where each head owns its own output buffer
/// with stride `head_dim` (not `q_dim`).
fn attend_position_per_head(
    head_out: &mut [f32],
    i: usize,
    q: &[f32],
    q_start: usize,
    keys: &[f32],
    values: &[f32],
    kv_dim: usize,
    kv_head_offset: usize,
    head_dim: usize,
    num_keys: usize,
    scale: f32,
) {
    let mut scores = compute_attention_scores(
        q, q_start, keys, kv_dim, kv_head_offset, head_dim, num_keys, scale,
    );
    softmax_inplace(&mut scores);
    let out_start = i * head_dim;
    accumulate_weighted_values(head_out, out_start, &scores, values, kv_dim, kv_head_offset, head_dim);
}

impl QuantizedAprTransformerQ4 {

    /// Attention with KV cache - new Q attends to all cached K/V
    ///
    /// Parallelizes across attention heads for efficiency.
    fn causal_attention_cached(
        &self,
        new_q: &[f32],
        full_k: &[f32],
        full_v: &[f32],
        new_seq_len: usize,
        _total_seq_len: usize,
        cache_len: usize,
    ) -> Vec<f32> {
        use rayon::prelude::*;

        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.hidden_dim / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let group_size = num_heads / num_kv_heads;

        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        const PARALLEL_HEAD_THRESHOLD: usize = 4;

        if num_heads < PARALLEL_HEAD_THRESHOLD {
            let mut output = vec![0.0f32; new_seq_len * q_dim];
            for head in 0..num_heads {
                let kv_head = head / group_size;
                let q_head_offset = head * head_dim;
                let kv_head_offset = kv_head * head_dim;

                for i in 0..new_seq_len {
                    let pos = cache_len + i;
                    let q_start = i * q_dim + q_head_offset;
                    let out_start = i * q_dim + q_head_offset;
                    let mut scores = compute_attention_scores(
                        new_q, q_start, full_k, kv_dim, kv_head_offset, head_dim, pos + 1, scale,
                    );
                    softmax_inplace(&mut scores);
                    accumulate_weighted_values(
                        &mut output, out_start, &scores, full_v, kv_dim, kv_head_offset, head_dim,
                    );
                }
            }
            output
        } else {
            let head_outputs: Vec<Vec<f32>> = (0..num_heads)
                .into_par_iter()
                .map(|head| {
                    let mut head_out = vec![0.0f32; new_seq_len * head_dim];
                    let kv_head = head / group_size;
                    let q_head_offset = head * head_dim;
                    let kv_head_offset = kv_head * head_dim;

                    for i in 0..new_seq_len {
                        let pos = cache_len + i;
                        let q_start = i * q_dim + q_head_offset;
                        attend_position_per_head(
                            &mut head_out, i, new_q, q_start,
                            full_k, full_v, kv_dim, kv_head_offset,
                            head_dim, pos + 1, scale,
                        );
                    }
                    head_out
                })
                .collect();

            merge_head_outputs(head_outputs, new_seq_len, head_dim, q_dim)
        }
    }

    /// Get memory footprint in bytes
    #[must_use]
    pub fn memory_size(&self) -> usize {
        let embed_size = self.token_embedding.len() * 4;
        let norm_size = self.output_norm_weight.len() * 4;
        let lm_head_size = self.lm_head_weight.data.len();

        let layer_size: usize = self
            .layers
            .iter()
            .map(|l| {
                l.attn_norm_weight.len() * 4
                    + l.qkv_weight.data.len()
                    + l.attn_output_weight.data.len()
                    + l.ffn_up_weight.data.len()
                    + l.ffn_down_weight.data.len()
                    + l.ffn_gate_weight.as_ref().map_or(0, |g| g.data.len())
                    + l.ffn_norm_weight.as_ref().map_or(0, |n| n.len() * 4)
            })
            .sum();

        embed_size + norm_size + lm_head_size + layer_size
    }

    /// Apply Rotary Position Embeddings (RoPE) to a tensor
    ///
    /// RoPE applies position-dependent rotation to pairs of dimensions,
    /// enabling the model to learn relative positional information.
    fn apply_rope(&self, x: &mut [f32], position: usize, num_heads_in_x: usize) {
        let head_dim = self.config.hidden_dim / self.config.num_heads;
        let half_dim = head_dim / 2;
        let theta = self.config.rope_theta;
        let pos_f32 = position as f32;
        let head_dim_f32 = head_dim as f32;

        for h in 0..num_heads_in_x {
            let head_start = h * head_dim;
            let idx2_start = head_start + half_dim;

            if idx2_start + half_dim > x.len() {
                continue;
            }

            apply_rope_to_head(x, head_start, idx2_start, half_dim, theta, pos_f32, head_dim_f32);
        }
    }

    /// Compute scaled dot-product attention with causal mask and GQA support
    ///
    /// Implements multi-head attention with Grouped Query Attention (GQA),
    /// where multiple Q heads share the same K/V heads.
    ///
    /// Optimized for single-token inference (seq_len=1).
    fn causal_attention(&self, q: &[f32], k: &[f32], v: &[f32], seq_len: usize) -> Vec<f32> {
        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.hidden_dim / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let group_size = num_heads / num_kv_heads;
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        // Fast path for single token (common case in autoregressive generation)
        // With seq_len=1 and causal mask, each head just copies its V vector
        // (softmax of single element is 1.0)
        if seq_len == 1 {
            let mut output = vec![0.0f32; q_dim];
            for head in 0..num_heads {
                let kv_head = head / group_size;
                let v_offset = kv_head * head_dim;
                let out_offset = head * head_dim;
                output[out_offset..out_offset + head_dim]
                    .copy_from_slice(&v[v_offset..v_offset + head_dim]);
            }
            return output;
        }

        use rayon::prelude::*;
        const PARALLEL_HEAD_THRESHOLD: usize = 4;

        if num_heads < PARALLEL_HEAD_THRESHOLD {
            let mut output = vec![0.0f32; seq_len * q_dim];
            for head in 0..num_heads {
                self.compute_head_attention(
                    head, group_size, head_dim, scale, q, k, v, seq_len, q_dim, kv_dim, &mut output,
                );
            }
            output
        } else {
            let head_outputs: Vec<Vec<f32>> = (0..num_heads)
                .into_par_iter()
                .map(|head| {
                    let mut head_out = vec![0.0f32; seq_len * head_dim];
                    let kv_head = head / group_size;
                    let q_head_offset = head * head_dim;
                    let kv_head_offset = kv_head * head_dim;

                    for i in 0..seq_len {
                        let q_start = i * q_dim + q_head_offset;
                        attend_position_per_head(
                            &mut head_out, i, q, q_start,
                            k, v, kv_dim, kv_head_offset,
                            head_dim, i + 1, scale,
                        );
                    }
                    head_out
                })
                .collect();

            merge_head_outputs(head_outputs, seq_len, head_dim, q_dim)
        }
    }

    /// Compute attention for a single head (helper for sequential path)
    #[allow(clippy::too_many_arguments)]
    fn compute_head_attention(
        &self,
        head: usize,
        group_size: usize,
        head_dim: usize,
        scale: f32,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        q_dim: usize,
        kv_dim: usize,
        output: &mut [f32],
    ) {
        let kv_head = head / group_size;
        let q_head_offset = head * head_dim;
        let kv_head_offset = kv_head * head_dim;

        for i in 0..seq_len {
            let q_start = i * q_dim + q_head_offset;
            let out_start = i * q_dim + q_head_offset;
            let mut scores = compute_attention_scores(
                q, q_start, k, kv_dim, kv_head_offset, head_dim, i + 1, scale,
            );
            softmax_inplace(&mut scores);
            accumulate_weighted_values(output, out_start, &scores, v, kv_dim, kv_head_offset, head_dim);
        }
    }
}

/// Apply RoPE rotation to a single head's dimensions, processing 4 elements at a time.
fn apply_rope_to_head(
    x: &mut [f32],
    head_start: usize,
    idx2_start: usize,
    half_dim: usize,
    theta: f32,
    pos_f32: f32,
    head_dim_f32: f32,
) {
    let mut i = 0;
    while i + 4 <= half_dim {
        apply_rope_quad(x, head_start, idx2_start, i, theta, pos_f32, head_dim_f32);
        i += 4;
    }
    // Handle remaining elements
    while i < half_dim {
        let freq = 1.0 / theta.powf(2.0 * i as f32 / head_dim_f32);
        let angle = pos_f32 * freq;
        let (sin_val, cos_val) = angle.sin_cos();

        let x1 = x[head_start + i];
        let x2 = x[idx2_start + i];

        x[head_start + i] = x1 * cos_val - x2 * sin_val;
        x[idx2_start + i] = x1 * sin_val + x2 * cos_val;

        i += 1;
    }
}

/// Apply RoPE rotation to 4 consecutive dimension pairs (ILP-friendly).
fn apply_rope_quad(
    x: &mut [f32],
    head_start: usize,
    idx2_start: usize,
    i: usize,
    theta: f32,
    pos_f32: f32,
    head_dim_f32: f32,
) {
    let freq0 = 1.0 / theta.powf(2.0 * i as f32 / head_dim_f32);
    let freq1 = 1.0 / theta.powf(2.0 * (i + 1) as f32 / head_dim_f32);
    let freq2 = 1.0 / theta.powf(2.0 * (i + 2) as f32 / head_dim_f32);
    let freq3 = 1.0 / theta.powf(2.0 * (i + 3) as f32 / head_dim_f32);

    let (sin0, cos0) = (pos_f32 * freq0).sin_cos();
    let (sin1, cos1) = (pos_f32 * freq1).sin_cos();
    let (sin2, cos2) = (pos_f32 * freq2).sin_cos();
    let (sin3, cos3) = (pos_f32 * freq3).sin_cos();

    let x1_0 = x[head_start + i];
    let x1_1 = x[head_start + i + 1];
    let x1_2 = x[head_start + i + 2];
    let x1_3 = x[head_start + i + 3];

    let x2_0 = x[idx2_start + i];
    let x2_1 = x[idx2_start + i + 1];
    let x2_2 = x[idx2_start + i + 2];
    let x2_3 = x[idx2_start + i + 3];

    x[head_start + i] = x1_0 * cos0 - x2_0 * sin0;
    x[head_start + i + 1] = x1_1 * cos1 - x2_1 * sin1;
    x[head_start + i + 2] = x1_2 * cos2 - x2_2 * sin2;
    x[head_start + i + 3] = x1_3 * cos3 - x2_3 * sin3;

    x[idx2_start + i] = x1_0 * sin0 + x2_0 * cos0;
    x[idx2_start + i + 1] = x1_1 * sin1 + x2_1 * cos1;
    x[idx2_start + i + 2] = x1_2 * sin2 + x2_2 * cos2;
    x[idx2_start + i + 3] = x1_3 * sin3 + x2_3 * cos3;
}

include!("q4_simd_from_gguf.rs");
include!("q4_simd_activations_cache.rs");

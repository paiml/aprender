/// Attend one query head row (`q[q_start..][..head_dim]`) over the first
/// `num_keys` key/value rows, accumulating into `output[out_start..][..head_dim]`
/// (PP-ARCH-001 §9.11: the shared `ops::attend_row_scalar` home, unguarded
/// Divide softmax).
#[allow(clippy::too_many_arguments)]
fn attend_row(
    output: &mut [f32],
    out_start: usize,
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
    let kv_row = |j: usize| j * kv_dim + kv_head_offset;
    crate::gguf::ops::attend_row_scalar(
        &q[q_start..q_start + head_dim],
        num_keys,
        |j| &keys[kv_row(j)..][..head_dim],
        |j| &values[kv_row(j)..][..head_dim],
        crate::gguf::ops::ScoreScale::Mul(scale),
        crate::gguf::ops::RowSoftmax {
            norm: crate::gguf::ops::SoftmaxNorm::Divide,
            guard_positive_sum: false,
        },
        &mut Vec::with_capacity(num_keys),
        &mut output[out_start..out_start + head_dim],
    );
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
    attend_row(
        head_out, i * head_dim, q, q_start, keys, values, kv_dim, kv_head_offset, head_dim,
        num_keys, scale,
    );
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
                    attend_row(
                        &mut output, out_start, new_q, q_start, full_k, full_v, kv_dim,
                        kv_head_offset, head_dim, pos + 1, scale,
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
    ///
    /// PMAT-797: honors the architecture's pairing convention. NORM
    /// (`rope_type == 0`, LLaMA-family) rotates adjacent pairs `(x[2i], x[2i+1])`;
    /// NEOX (`rope_type == 2`, Qwen/NeoX/Phi/Gemma) rotates split halves
    /// `(x[i], x[i+head_dim/2])`. Matches llama.cpp `ggml_rope`.
    fn apply_rope(&self, x: &mut [f32], position: usize, num_heads_in_x: usize) {
        let head_dim = self.config.hidden_dim / self.config.num_heads;
        let rope_type = crate::gguf::infer_rope_type(&self.config.architecture);
        crate::gguf::ops::rope_into(
            x,
            num_heads_in_x,
            head_dim,
            position,
            self.config.rope_theta,
            crate::gguf::ops::RopeStyle::from_rope_type(rope_type),
        );
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
            attend_row(
                output, out_start, q, q_start, k, v, kv_dim, kv_head_offset, head_dim, i + 1,
                scale,
            );
        }
    }
}



include!("q4_simd_from_gguf.rs");
include!("q4_simd_activations_cache.rs");

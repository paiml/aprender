
impl AprTransformer {
    /// Compute attention output using KV cache (extracted from forward_with_cache)
    ///
    /// Handles both first token (no cache, V pass-through with GQA expansion)
    /// and subsequent tokens (full softmax attention with cached K/V).
    #[allow(clippy::too_many_arguments)]
    fn compute_attention_with_cache(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        cache_len: usize,
        position: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        hidden_dim: usize,
    ) -> Vec<f32> {
        let kv_size = num_kv_heads * head_dim;

        // F-REGR-231 FIX: Handle first token specially
        // When cache is empty (cache_len = 0), we just appended K/V but len isn't incremented yet.
        // For the first token, attention(single token) = V directly (since softmax of one score = 1.0).
        if cache_len == 0 {
            // First token: just use V directly (expanded via GQA)
            let group_size = num_heads / num_kv_heads;
            (0..num_heads)
                .flat_map(|h| {
                    let kv_head = h / group_size;
                    let start = kv_head * head_dim;
                    v[start..start + head_dim].iter().copied()
                })
                .collect()
        } else {
            // Subsequent tokens: use full attention with cache + current K/V
            let mut attn_out = vec![0.0f32; hidden_dim];
            let scale = 1.0 / (head_dim as f32).sqrt();

            // seq_len includes cached positions plus current position
            let seq_len = cache_len + 1;

            for h in 0..num_heads {
                let kv_head = h * num_kv_heads / num_heads; // GQA mapping
                let q_start = h * head_dim;
                let q_slice = &q[q_start..q_start + head_dim];

                // Compute attention scores with SIMD dot product
                let mut scores = Vec::with_capacity(seq_len);

                // Scores for cached positions
                for pos in 0..cache_len {
                    let k_start = pos * kv_size + kv_head * head_dim;
                    let k_slice = &k_cache[k_start..k_start + head_dim];
                    let dot = simd_dot_f32(q_slice, k_slice);
                    scores.push(dot * scale);
                }

                // Score for current position (using current K)
                let k_start = kv_head * head_dim;
                let k_slice = &k[k_start..k_start + head_dim];
                let dot = simd_dot_f32(q_slice, k_slice);
                scores.push(dot * scale);

                // Causal mask: only attend to positions <= current
                for pos in (position + 1)..seq_len {
                    scores[pos] = f32::NEG_INFINITY;
                }

                // Softmax
                let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut exp_scores: Vec<f32> =
                    scores.iter().map(|s| (s - max_score).exp()).collect();
                let sum: f32 = exp_scores.iter().sum();
                if sum > 0.0 {
                    let inv_sum = 1.0 / sum;
                    for s in &mut exp_scores {
                        *s *= inv_sum;
                    }
                }

                // Weighted sum of V
                let attn_out_head = &mut attn_out[q_start..q_start + head_dim];

                // From cached positions
                for pos in 0..cache_len {
                    let v_start = pos * kv_size + kv_head * head_dim;
                    let v_slice = &v_cache[v_start..v_start + head_dim];
                    simd_add_weighted(attn_out_head, v_slice, exp_scores[pos]);
                }

                // From current position (using current V)
                let v_start = kv_head * head_dim;
                let v_slice = &v[v_start..v_start + head_dim];
                simd_add_weighted(attn_out_head, v_slice, exp_scores[cache_len]);
            }

            attn_out
        }
    }
}

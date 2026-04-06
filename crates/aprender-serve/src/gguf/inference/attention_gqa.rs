impl OwnedQuantizedModel {

    /// Compute attention with Grouped Query Attention (GQA) support (IMP-105)
    ///
    /// GQA uses fewer KV heads than Q heads, with multiple Q heads sharing each KV head.
    /// This reduces memory bandwidth and KV cache size for large models.
    ///
    /// # Arguments
    /// * `q` - Query vector for current position [hidden_dim] (num_heads Q heads)
    /// * `k_cache` - Cached keys [cache_len, kv_dim] (num_kv_heads KV heads)
    /// * `v_cache` - Cached values [cache_len, kv_dim] (num_kv_heads KV heads)
    /// * `current_k` - Key for current position [kv_dim]
    /// * `current_v` - Value for current position [kv_dim]
    ///
    /// # Returns
    /// Attention output [hidden_dim]
    ///
    /// # GQA Mapping
    /// Q head i uses KV head (i * num_kv_heads / num_heads)
    /// Example: 8 Q heads, 2 KV heads → Q heads 0-3 use KV head 0, Q heads 4-7 use KV head 1
    pub fn attention_with_cache_gqa(
        &self,
        q: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        current_k: &[f32],
        current_v: &[f32],
    ) -> Vec<f32> {
        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads;
        // GH-479: Use config methods (Qwen3 head_dim != hidden/heads)
        let head_dim = self.config.head_dim();
        let q_dim = self.config.q_dim();
        let kv_dim = self.config.kv_dim();
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Number of Q heads that share each KV head
        let q_per_kv = num_heads / num_kv_heads;

        // Total sequence length = cached + 1 (current)
        let cache_len = if kv_dim > 0 {
            k_cache.len() / kv_dim
        } else {
            0
        };
        let total_len = cache_len + 1;

        let mut output = vec![0.0f32; q_dim];

        // Score buffer for the current group.
        // Size: q_per_kv * total_len.
        // We reuse this buffer for each KV group to minimize allocation.
        let mut group_scores = vec![0.0f32; q_per_kv * total_len];

        // Process each KV head group (OPTIMIZATION: Scan KV cache once per group)
        for kv_head in 0..num_kv_heads {
            let kv_head_offset = kv_head * head_dim;

            // 1. Compute Scores (Scan K Cache Once)
            for pos in 0..cache_len {
                let k_start = pos * kv_dim + kv_head_offset;
                let cached_key = &k_cache[k_start..k_start + head_dim];

                // For each Q head in this group
                for i in 0..q_per_kv {
                    let q_head_idx = kv_head * q_per_kv + i;
                    let q_head_offset = q_head_idx * head_dim;
                    let q_head_data = &q[q_head_offset..q_head_offset + head_dim];

                    let score = Self::simd_dot_f32(q_head_data, cached_key) * scale;
                    group_scores[i * total_len + pos] = score;
                }
            }

            // Handle current position K
            let curr_key = &current_k[kv_head_offset..kv_head_offset + head_dim];
            for i in 0..q_per_kv {
                let q_head_idx = kv_head * q_per_kv + i;
                let q_head_offset = q_head_idx * head_dim;
                let q_head_data = &q[q_head_offset..q_head_offset + head_dim];

                let score = Self::simd_dot_f32(q_head_data, curr_key) * scale;
                group_scores[i * total_len + cache_len] = score;
            }

            // 2. Softmax (Per Q Head)
            for i in 0..q_per_kv {
                let start = i * total_len;
                let end = start + total_len;
                crate::quantize::softmax_simd(&mut group_scores[start..end]);
            }

            // 3. Accumulate Values (Scan V Cache Once)
            for pos in 0..cache_len {
                let v_start = pos * kv_dim + kv_head_offset;
                let cached_val = &v_cache[v_start..v_start + head_dim];

                for i in 0..q_per_kv {
                    let weight = group_scores[i * total_len + pos];
                    let q_head_idx = kv_head * q_per_kv + i;
                    let out_offset = q_head_idx * head_dim;
                    let out_head = &mut output[out_offset..out_offset + head_dim];

                    Self::simd_axpy_f32(out_head, weight, cached_val);
                }
            }

            // Handle current position V
            let curr_val = &current_v[kv_head_offset..kv_head_offset + head_dim];
            for i in 0..q_per_kv {
                let weight = group_scores[i * total_len + cache_len];
                let q_head_idx = kv_head * q_per_kv + i;
                let out_offset = q_head_idx * head_dim;
                let out_head = &mut output[out_offset..out_offset + head_dim];

                Self::simd_axpy_f32(out_head, weight, curr_val);
            }
        }

        output
    }

    /// Attention with cache - writes to pre-allocated buffer (IMP-131)
    pub fn attention_with_cache_gqa_into(
        &self,
        q: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        current_k: &[f32],
        current_v: &[f32],
        output: &mut [f32],
    ) {
        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads;
        // GH-479: Use config methods (Qwen3 head_dim != hidden/heads)
        let head_dim = self.config.head_dim();
        let q_dim = self.config.q_dim();
        let kv_dim = self.config.kv_dim();
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q_per_kv = num_heads / num_kv_heads;

        let cache_len = if kv_dim > 0 {
            k_cache.len() / kv_dim
        } else {
            0
        };
        let total_len = cache_len + 1;

        // Zero output buffer
        // GH-479: Use q_dim (may differ from hidden_dim for Qwen3)
        output[..q_dim].iter_mut().for_each(|x| *x = 0.0);

        // Score buffer for the current group.
        // Size: q_per_kv * total_len.
        // We reuse this buffer for each KV group to minimize allocation.
        let mut group_scores = vec![0.0f32; q_per_kv * total_len];

        // Process each KV head group (OPTIMIZATION: Scan KV cache once per group)
        for kv_head in 0..num_kv_heads {
            let kv_head_offset = kv_head * head_dim;

            // 1. Compute Scores (Scan K Cache Once)
            for pos in 0..cache_len {
                let k_start = pos * kv_dim + kv_head_offset;
                let cached_key = &k_cache[k_start..k_start + head_dim];

                // For each Q head in this group
                for i in 0..q_per_kv {
                    let q_head_idx = kv_head * q_per_kv + i;
                    let q_head_offset = q_head_idx * head_dim;
                    let q_head_data = &q[q_head_offset..q_head_offset + head_dim];

                    let score = Self::simd_dot_f32(q_head_data, cached_key) * scale;
                    group_scores[i * total_len + pos] = score;
                }
            }

            // Handle current position K
            let curr_key = &current_k[kv_head_offset..kv_head_offset + head_dim];
            for i in 0..q_per_kv {
                let q_head_idx = kv_head * q_per_kv + i;
                let q_head_offset = q_head_idx * head_dim;
                let q_head_data = &q[q_head_offset..q_head_offset + head_dim];

                let score = Self::simd_dot_f32(q_head_data, curr_key) * scale;
                group_scores[i * total_len + cache_len] = score;
            }

            // 2. Softmax (Per Q Head)
            for i in 0..q_per_kv {
                let start = i * total_len;
                let end = start + total_len;
                crate::quantize::softmax_simd(&mut group_scores[start..end]);
            }

            // 3. Accumulate Values (Scan V Cache Once)
            for pos in 0..cache_len {
                let v_start = pos * kv_dim + kv_head_offset;
                let cached_val = &v_cache[v_start..v_start + head_dim];

                for i in 0..q_per_kv {
                    let weight = group_scores[i * total_len + pos];
                    let q_head_idx = kv_head * q_per_kv + i;
                    let out_offset = q_head_idx * head_dim;
                    let out_head = &mut output[out_offset..out_offset + head_dim];

                    Self::simd_axpy_f32(out_head, weight, cached_val);
                }
            }

            // Handle current position V
            let curr_val = &current_v[kv_head_offset..kv_head_offset + head_dim];
            for i in 0..q_per_kv {
                let weight = group_scores[i * total_len + cache_len];
                let q_head_idx = kv_head * q_per_kv + i;
                let out_offset = q_head_idx * head_dim;
                let out_head = &mut output[out_offset..out_offset + head_dim];

                Self::simd_axpy_f32(out_head, weight, curr_val);
            }
        }
    }

    /// Adaptive attention with KV cache - auto-selects CPU or GPU backend (IMP-122)
    ///
    /// For short cache lengths (< 64), uses efficient CPU implementation.
    /// For long cache lengths (>= 64), uses GPU-accelerated computation.
    ///
    /// # Arguments
    /// * `q` - Query vector for current position [hidden_dim]
    /// * `k_cache` - Cached keys [cache_len, hidden_dim]
    /// * `v_cache` - Cached values [cache_len, hidden_dim]
    /// * `current_k` - Key for current position [hidden_dim]
    /// * `current_v` - Value for current position [hidden_dim]
    ///
    /// # Returns
    /// Result containing attention output [hidden_dim]
    ///
    /// # Errors
    /// Returns error if GPU operations fail (for GPU path)
    #[cfg(feature = "gpu")]
    pub fn adaptive_attention_with_cache(
        &self,
        q: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        current_k: &[f32],
        current_v: &[f32],
    ) -> Result<Vec<f32>> {
        let hidden_dim = self.config.hidden_dim;

        // Calculate cache length
        let cache_len = if hidden_dim > 0 {
            k_cache.len() / hidden_dim
        } else {
            0
        };

        // Threshold for GPU dispatch (matches IMP-119)
        const GPU_CACHE_LEN_THRESHOLD: usize = 64;

        if cache_len >= GPU_CACHE_LEN_THRESHOLD {
            // GPU path for long sequences
            self.gpu_attention_with_cache(q, k_cache, v_cache, current_k, current_v)
        } else {
            // CPU path for short sequences - use existing implementation
            Ok(self.attention_with_cache(q, k_cache, v_cache, current_k, current_v))
        }
    }

    /// CPU-only version of adaptive attention
    #[cfg(not(feature = "gpu"))]
    pub fn adaptive_attention_with_cache(
        &self,
        q: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        current_k: &[f32],
        current_v: &[f32],
    ) -> Result<Vec<f32>> {
        Ok(self.attention_with_cache(q, k_cache, v_cache, current_k, current_v))
    }

    /// GPU-accelerated attention with KV cache (IMP-122)
    ///
    /// Uses GPU for Q@K^T computation when cache is large enough.
    #[cfg(feature = "gpu")]
    fn gpu_attention_with_cache(
        &self,
        q: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        current_k: &[f32],
        current_v: &[f32],
    ) -> Result<Vec<f32>> {
        use crate::gpu::HybridScheduler;

        let num_heads = self.config.num_heads;
        // GH-479: Use config methods (Qwen3 head_dim != hidden/heads)
        let head_dim = self.config.head_dim();
        let q_dim = self.config.q_dim();
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Total sequence length = cached + 1 (current)
        let cache_len = k_cache.len() / q_dim;
        let total_len = cache_len + 1;

        let mut output = vec![0.0f32; q_dim];

        // Create scheduler for GPU operations
        let mut scheduler = HybridScheduler::with_threshold(1000).map_err(|e| {
            RealizarError::UnsupportedOperation {
                operation: "gpu_attention_with_cache".to_string(),
                reason: format!("Failed to create scheduler: {e}"),
            }
        })?;

        // Process each head
        for head in 0..num_heads {
            let head_offset = head * head_dim;
            let q_head = &q[head_offset..head_offset + head_dim];

            // Build full K matrix for this head: [total_len, head_dim]
            let mut k_full = Vec::with_capacity(total_len * head_dim);
            for pos in 0..cache_len {
                let k_start = pos * q_dim + head_offset;
                k_full.extend_from_slice(&k_cache[k_start..k_start + head_dim]);
            }
            k_full.extend_from_slice(&current_k[head_offset..head_offset + head_dim]);

            // Transpose K to [head_dim, total_len] for matmul
            let mut k_t = vec![0.0f32; head_dim * total_len];
            for pos in 0..total_len {
                for d in 0..head_dim {
                    k_t[d * total_len + pos] = k_full[pos * head_dim + d];
                }
            }

            // GPU matmul: Q[1, head_dim] @ K_T[head_dim, total_len] -> [1, total_len]
            let scores_raw = scheduler
                .matmul(q_head, &k_t, 1, head_dim, total_len)
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "gpu_attention_with_cache".to_string(),
                    reason: format!("GPU matmul failed: {e}"),
                })?;

            // Scale scores
            let mut scores: Vec<f32> = scores_raw.iter().map(|&s| s * scale).collect();

            // Softmax (SIMD-optimized)
            crate::quantize::softmax_simd(&mut scores);

            // Weighted sum of values
            let out_head = &mut output[head_offset..head_offset + head_dim];

            // Cached values
            for (pos, &weight) in scores.iter().enumerate().take(cache_len) {
                let v_start = pos * q_dim + head_offset;
                let cached_val = &v_cache[v_start..v_start + head_dim];
                for d in 0..head_dim {
                    out_head[d] += weight * cached_val[d];
                }
            }

            // Current value
            let curr_val = &current_v[head_offset..head_offset + head_dim];
            let current_weight = scores[cache_len];
            for d in 0..head_dim {
                out_head[d] += current_weight * curr_val[d];
            }
        }

        Ok(output)
    }
}

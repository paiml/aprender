impl OwnedQuantizedModel {
    /// Apply Rotary Position Embedding (RoPE) to Q or K vectors
    ///
    /// Supports two RoPE styles:
    /// - NORM (type 0): Adjacent pairs rotation (LLaMA default)
    /// - NEOX (type 2): Split halves rotation (GPT-NeoX)
    ///
    /// # Arguments
    /// * `x` - Vector to rotate in-place [num_heads_in_x * head_dim]
    /// * `position` - Position index for frequency calculation
    /// * `num_heads_in_x` - Number of heads in x (num_heads for Q, num_kv_heads for K)
    ///
    /// # GQA Support
    /// For GQA models, pass num_heads for Q vectors and num_kv_heads for K vectors.
    pub(crate) fn apply_rope(&self, x: &mut [f32], position: usize, num_heads_in_x: usize) {
        let head_dim = self.config.head_dim();
        let half_dim = head_dim / 2;
        let theta = self.config.rope_theta;
        let rope_type = self.config.rope_type;

        // Stack-based buffers (max 128 = 256 head_dim, covers all common models)
        // Avoids heap allocation on every call
        let mut cos_vals: [f32; 128] = [0.0; 128];
        let mut sin_vals: [f32; 128] = [0.0; 128];

        // Pre-compute cos/sin for this position (reused across all heads)
        let pos_f32 = position as f32;
        let head_dim_f32 = head_dim as f32;
        for i in 0..half_dim.min(128) {
            let freq = 1.0 / theta.powf(2.0 * i as f32 / head_dim_f32);
            let angle = pos_f32 * freq;
            let (sin_v, cos_v) = angle.sin_cos();
            cos_vals[i] = cos_v;
            sin_vals[i] = sin_v;
        }

        // Apply rotation to each head
        for h in 0..num_heads_in_x {
            let head_start = h * head_dim;

            if head_start + head_dim > x.len() {
                continue;
            }

            if rope_type == 2 {
                // NEOX style: split halves (x[0..half], x[half..])
                // Used by GPT-NeoX and some newer models
                let (first_half, second_half) =
                    x[head_start..head_start + head_dim].split_at_mut(half_dim);
                crate::quantize::apply_rope_rotation_simd(
                    first_half,
                    second_half,
                    &cos_vals[..half_dim],
                    &sin_vals[..half_dim],
                );
            } else {
                // NORM style (type 0): adjacent pairs (x[0], x[1]), (x[2], x[3]), ...
                // This is the default for LLaMA-family models
                let head_slice = &mut x[head_start..head_start + head_dim];
                for i in 0..half_dim {
                    let x0 = head_slice[2 * i];
                    let x1 = head_slice[2 * i + 1];
                    let cos_v = cos_vals[i];
                    let sin_v = sin_vals[i];
                    head_slice[2 * i] = x0 * cos_v - x1 * sin_v;
                    head_slice[2 * i + 1] = x0 * sin_v + x1 * cos_v;
                }
            }
        }
    }

    /// Compute scaled dot-product attention with causal mask (IMP-101b)
    ///
    /// Computes: softmax(QK^T / sqrt(d_k)) * V with causal masking
    ///
    /// # Arguments
    /// * `q` - Query vectors [seq_len, q_dim] where q_dim = num_heads * head_dim
    /// * `k` - Key vectors [seq_len, kv_dim] where kv_dim = num_kv_heads * head_dim
    /// * `v` - Value vectors [seq_len, kv_dim] where kv_dim = num_kv_heads * head_dim
    ///
    /// # Returns
    /// Attention output [seq_len, q_dim] where q_dim = num_heads * head_dim
    ///
    /// # GQA (Grouped Query Attention) Support
    /// For models where num_kv_heads < num_heads (e.g., TinyLlama: 4 vs 32),
    /// multiple Q heads share the same K/V head. The group size is num_heads / num_kv_heads.
    pub(crate) fn causal_attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
    ) -> Vec<f32> {
        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.head_dim();
        let scale = 1.0 / (head_dim as f32).sqrt();

        // GQA: multiple Q heads share each KV head
        // group_size = num_heads / num_kv_heads (e.g., 32/4 = 8 for TinyLlama)
        let group_size = num_heads / num_kv_heads;

        // Q has num_heads heads, K/V have num_kv_heads heads
        let q_dim = num_heads * head_dim; // e.g., 32 * 64 = 2048
        let kv_dim = num_kv_heads * head_dim; // e.g., 4 * 64 = 256

        let mut output = vec![0.0f32; seq_len * q_dim];

        // Process each Q head independently
        for head in 0..num_heads {
            // Map Q head to corresponding KV head (GQA grouping)
            let kv_head = head / group_size;

            let q_head_offset = head * head_dim;
            let kv_head_offset = kv_head * head_dim;

            // Process each query position
            for i in 0..seq_len {
                // Compute attention scores for this query against all keys up to position i (causal)
                let mut scores = Vec::with_capacity(i + 1);
                let q_start = i * q_dim + q_head_offset;

                for j in 0..=i {
                    // Only attend to positions 0..=i (causal mask)
                    let k_start = j * kv_dim + kv_head_offset;

                    // Dot product Q[i] · K[j]
                    let mut score = 0.0f32;
                    for d in 0..head_dim {
                        score += q[q_start + d] * k[k_start + d];
                    }
                    scores.push(score * scale);
                }

                // Softmax (SIMD-optimized)
                crate::quantize::softmax_simd(&mut scores);

                // Weighted sum of values
                let out_start = i * q_dim + q_head_offset;
                for (j, &weight) in scores.iter().enumerate() {
                    let v_start = j * kv_dim + kv_head_offset;
                    for d in 0..head_dim {
                        output[out_start + d] += weight * v[v_start + d];
                    }
                }
            }
        }

        output
    }

    /// Get model configuration
    pub fn config(&self) -> &GGUFConfig {
        &self.config
    }

    /// Check if CUDA is enabled
    #[cfg(feature = "cuda")]
    pub fn cuda_enabled(&self) -> bool {
        self.cuda_executor.is_some()
    }

    /// Check if CUDA acceleration is enabled (stub when cuda feature disabled)
    #[cfg(not(feature = "cuda"))]
    pub fn cuda_enabled(&self) -> bool {
        false
    }

    // ============================================================================
    // SIMD Helper Methods
    // ============================================================================

    /// SIMD-optimized dot product for f32 slices
    #[inline]
    fn simd_dot_f32(a: &[f32], b: &[f32]) -> f32 {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                // SAFETY: We've verified AVX2+FMA support
                unsafe { Self::simd_dot_f32_avx2(a, b) }
            } else {
                Self::simd_dot_f32_scalar(a, b)
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            Self::simd_dot_f32_scalar(a, b)
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2", enable = "fma")]
    #[inline]
    // SAFETY: Caller must satisfy the documented preconditions
    unsafe fn simd_dot_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            use std::arch::x86_64::{
                _mm256_castps256_ps128, _mm256_extractf128_ps, _mm256_fmadd_ps, _mm256_loadu_ps,
                _mm256_setzero_ps, _mm_add_ps, _mm_add_ss, _mm_cvtss_f32, _mm_movehdup_ps,
                _mm_movehl_ps,
            };

            let len = a.len().min(b.len());
            let mut acc = _mm256_setzero_ps();
            let mut i = 0;

            // Process 16 floats at a time (2x unrolled for better ILP)
            while i + 16 <= len {
                let va0 = _mm256_loadu_ps(a.as_ptr().add(i));
                let vb0 = _mm256_loadu_ps(b.as_ptr().add(i));
                let va1 = _mm256_loadu_ps(a.as_ptr().add(i + 8));
                let vb1 = _mm256_loadu_ps(b.as_ptr().add(i + 8));
                acc = _mm256_fmadd_ps(va0, vb0, acc);
                acc = _mm256_fmadd_ps(va1, vb1, acc);
                i += 16;
            }
            // Handle remaining 8-float chunk
            if i + 8 <= len {
                let va = _mm256_loadu_ps(a.as_ptr().add(i));
                let vb = _mm256_loadu_ps(b.as_ptr().add(i));
                acc = _mm256_fmadd_ps(va, vb, acc);
                i += 8;
            }

            // Horizontal sum
            let hi = _mm256_extractf128_ps(acc, 1);
            let lo = _mm256_castps256_ps128(acc);
            let sum128 = _mm_add_ps(lo, hi);
            let shuf = _mm_movehdup_ps(sum128);
            let sums = _mm_add_ps(sum128, shuf);
            let shuf2 = _mm_movehl_ps(sums, sums);
            let result = _mm_add_ss(sums, shuf2);
            let mut sum = _mm_cvtss_f32(result);

            // Handle remaining elements
            while i < len {
                sum += a[i] * b[i];
                i += 1;
            }

            sum
        }
    }

    #[inline]
    fn simd_dot_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    /// SIMD-optimized scaled accumulation: out[i] += weight * val[i]
    #[inline]
    fn simd_axpy_f32(out: &mut [f32], weight: f32, val: &[f32]) {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                // SAFETY: We've verified AVX2 support
                unsafe { Self::simd_axpy_f32_avx2(out, weight, val) }
            } else {
                Self::simd_axpy_f32_scalar(out, weight, val);
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            Self::simd_axpy_f32_scalar(out, weight, val);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2", enable = "fma")]
    #[inline]
    // SAFETY: Caller must satisfy the documented preconditions
    unsafe fn simd_axpy_f32_avx2(out: &mut [f32], weight: f32, val: &[f32]) {
        use std::arch::x86_64::{
            _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_storeu_ps,
        };

        let len = out.len().min(val.len());
        let w = _mm256_set1_ps(weight);
        let mut i = 0;

        // Process 8 floats at a time
        while i + 8 <= len {
            // SAFETY: bounds checked above, pointers valid
            let v_out = unsafe { _mm256_loadu_ps(out.as_ptr().add(i)) };
            // SAFETY: Memory safety ensured by bounds checking and alignment
            let v_val = unsafe { _mm256_loadu_ps(val.as_ptr().add(i)) };
            let result = _mm256_fmadd_ps(w, v_val, v_out);
            // SAFETY: Memory safety ensured by bounds checking and alignment
            unsafe { _mm256_storeu_ps(out.as_mut_ptr().add(i), result) };
            i += 8;
        }

        // Handle remaining elements
        while i < len {
            out[i] += weight * val[i];
            i += 1;
        }
    }

    #[inline]
    fn simd_axpy_f32_scalar(out: &mut [f32], weight: f32, val: &[f32]) {
        for (o, v) in out.iter_mut().zip(val.iter()) {
            *o += weight * *v;
        }
    }

    // ============================================================================
    // KV Cache Attention Methods
    // ============================================================================

    /// Attention with KV cache for autoregressive generation
    ///
    /// # Arguments
    /// * `q` - Query vector for current position [hidden_dim]
    /// * `k_cache` - Cached keys [cache_len, hidden_dim]
    /// * `v_cache` - Cached values [cache_len, hidden_dim]
    /// * `current_k` - Key for current position [hidden_dim]
    /// * `current_v` - Value for current position [hidden_dim]
    ///
    /// # Returns
    /// Attention output [hidden_dim]
    pub(crate) fn attention_with_cache(
        &self,
        q: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        current_k: &[f32],
        current_v: &[f32],
    ) -> Vec<f32> {
        let num_heads = self.config.num_heads;
        // GH-479: Use config methods (Qwen3 head_dim != hidden/heads)
        let head_dim = self.config.head_dim();
        let q_dim = self.config.q_dim();
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Total sequence length = cached + 1 (current)
        let cache_len = k_cache.len() / q_dim;
        let total_len = cache_len + 1;

        let mut output = vec![0.0f32; q_dim];

        // Process each head
        for head in 0..num_heads {
            let head_offset = head * head_dim;
            let q_head = &q[head_offset..head_offset + head_dim];

            // Compute attention scores against all positions (cached + current)
            let mut scores = Vec::with_capacity(total_len);

            // Scores against cached positions (SIMD-optimized)
            for pos in 0..cache_len {
                let k_start = pos * q_dim + head_offset;
                let cached_key = &k_cache[k_start..k_start + head_dim];
                let score = Self::simd_dot_f32(q_head, cached_key) * scale;
                scores.push(score);
            }

            // Score against current position (SIMD-optimized)
            let curr_key = &current_k[head_offset..head_offset + head_dim];
            let current_score = Self::simd_dot_f32(q_head, curr_key) * scale;
            scores.push(current_score);

            // Softmax (SIMD-optimized)
            crate::quantize::softmax_simd(&mut scores);

            // Weighted sum of values
            let out_head = &mut output[head_offset..head_offset + head_dim];

            // Sum over cached values (SIMD-optimized)
            for (pos, &weight) in scores.iter().enumerate().take(cache_len) {
                let v_start = pos * q_dim + head_offset;
                let cached_val = &v_cache[v_start..v_start + head_dim];
                Self::simd_axpy_f32(out_head, weight, cached_val);
            }

            // Add current value (SIMD-optimized)
            let curr_val = &current_v[head_offset..head_offset + head_dim];
            let current_weight = scores[cache_len];
            Self::simd_axpy_f32(out_head, current_weight, curr_val);
        }

        output
    }
}


impl Attention {

    /// SIMD-accelerated dot product
    ///
    /// Uses AVX2 on x86_64 for 8-way f32 parallelism
    #[inline]
    fn simd_dot_product(a: &[f32], b: &[f32]) -> f32 {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                // SAFETY: Feature detection above guarantees AVX2+FMA availability
                return unsafe { Self::simd_dot_avx2(a, b) };
            }
        }

        Self::scalar_dot_product(a, b)
    }

    /// AVX2 SIMD dot product (8-way f32 parallelism)
    ///
    /// # Safety
    /// Caller must ensure AVX2 and FMA are available on this CPU.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2", enable = "fma")]
    #[inline]
    #[allow(clippy::wildcard_imports, unsafe_op_in_unsafe_fn)]
    // SAFETY: Caller must satisfy the documented preconditions
    unsafe fn simd_dot_avx2(a: &[f32], b: &[f32]) -> f32 {
        use std::arch::x86_64::*;

        let len = a.len().min(b.len());
        let chunks = len / 8;
        let remainder = len % 8;

        let mut acc = _mm256_setzero_ps();

        for i in 0..chunks {
            let a_vec = _mm256_loadu_ps(a.as_ptr().add(i * 8));
            let b_vec = _mm256_loadu_ps(b.as_ptr().add(i * 8));
            acc = _mm256_fmadd_ps(a_vec, b_vec, acc);
        }

        // Horizontal sum of 8 floats
        let hi = _mm256_extractf128_ps(acc, 1);
        let lo = _mm256_castps256_ps128(acc);
        let sum128 = _mm_add_ps(lo, hi);
        let hi64 = _mm_movehl_ps(sum128, sum128);
        let sum64 = _mm_add_ps(sum128, hi64);
        let hi32 = _mm_shuffle_ps(sum64, sum64, 0x55);
        let sum32 = _mm_add_ss(sum64, hi32);
        let simd_sum = _mm_cvtss_f32(sum32);

        // Handle remainder
        let remainder_sum: f32 = (0..remainder)
            .map(|i| a[chunks * 8 + i] * b[chunks * 8 + i])
            .sum();

        simd_sum + remainder_sum
    }

    /// Scalar fallback dot product
    #[inline]
    fn scalar_dot_product(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    /// Parallel Flash Attention v2 using rayon
    ///
    /// Parallelizes over query positions for multi-core utilization.
    /// Each thread processes a subset of query rows independently.
    ///
    /// # Arguments
    ///
    /// * `query` - Query tensor `[seq_len, head_dim]`
    /// * `key` - Key tensor `[seq_len, head_dim]`
    /// * `value` - Value tensor `[seq_len, head_dim]`
    /// * `block_size` - Tile size for block-wise computation (e.g., 64, 128)
    ///
    /// # Returns
    ///
    /// Output tensor `[seq_len, head_dim]` (same as standard attention)
    ///
    /// # Errors
    ///
    /// Returns error if shapes don't match or `block_size` is zero
    #[allow(clippy::similar_names)]
    pub fn flash_forward_parallel(
        &self,
        query: &Tensor<f32>,
        key: &Tensor<f32>,
        value: &Tensor<f32>,
        block_size: usize,
    ) -> Result<Tensor<f32>> {
        use rayon::prelude::*;

        if block_size == 0 {
            return Err(RealizarError::InvalidShape {
                reason: "block_size must be > 0".to_string(),
            });
        }

        let q_shape = query.shape();
        let k_shape = key.shape();
        let v_shape = value.shape();

        // Validate shapes
        if q_shape.is_empty() || k_shape.is_empty() || v_shape.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Query, key, value tensors must have at least 1 dimension".to_string(),
            });
        }

        let q_last = q_shape[q_shape.len() - 1];
        let k_last = k_shape[k_shape.len() - 1];
        let v_last = v_shape[v_shape.len() - 1];

        if q_last != self.head_dim || k_last != self.head_dim || v_last != self.head_dim {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "Expected head_dim={}, got Q={}, K={}, V={}",
                    self.head_dim, q_last, k_last, v_last
                ),
            });
        }

        let q_seq_len = if q_shape.len() > 1 { q_shape[0] } else { 1 };
        let k_seq_len = if k_shape.len() > 1 { k_shape[0] } else { 1 };
        let v_seq_len = if v_shape.len() > 1 { v_shape[0] } else { 1 };

        if k_seq_len != v_seq_len {
            return Err(RealizarError::InvalidShape {
                reason: format!("Key seq_len {k_seq_len} != Value seq_len {v_seq_len}"),
            });
        }

        let q_data = query.data();
        let k_data = key.data();
        let v_data = value.data();
        let head_dim = self.head_dim;
        let scale = self.scale;

        // Parallel over query positions
        let output: Vec<f32> = (0..q_seq_len)
            .into_par_iter()
            .flat_map(|q_idx| {
                // Each query row is processed independently
                let mut row_output = vec![0.0; head_dim];
                let mut row_max = f32::NEG_INFINITY;
                let mut row_sum = 0.0;

                let num_kv_blocks = k_seq_len.div_ceil(block_size);

                for kv_block_idx in 0..num_kv_blocks {
                    let kv_start = kv_block_idx * block_size;
                    let kv_end = (kv_start + block_size).min(k_seq_len);

                    // Compute scores for this K/V block
                    let mut scores: Vec<f32> = (kv_start..kv_end)
                        .map(|kv_idx| {
                            let dot = Self::simd_dot_product(
                                &q_data[q_idx * head_dim..(q_idx + 1) * head_dim],
                                &k_data[kv_idx * head_dim..(kv_idx + 1) * head_dim],
                            );
                            dot * scale
                        })
                        .collect();

                    // Online softmax: find block max and update global max
                    let block_max = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                    let old_max = row_max;
                    let new_max = old_max.max(block_max);
                    row_max = new_max;

                    // Compute exp(scores - new_max)
                    let mut block_sum = 0.0;
                    for score in &mut scores {
                        let exp_val = (*score - new_max).exp();
                        *score = exp_val;
                        block_sum += exp_val;
                    }

                    // Rescale previous output
                    let scale_factor = (old_max - new_max).exp();
                    for out_val in &mut row_output {
                        *out_val *= scale_factor;
                    }
                    row_sum = row_sum * scale_factor + block_sum;

                    // Accumulate weighted values
                    for (j, kv_idx) in (kv_start..kv_end).enumerate() {
                        let weight = scores[j];
                        for k in 0..head_dim {
                            row_output[k] += weight * v_data[kv_idx * head_dim + k];
                        }
                    }
                }

                // Final normalization
                let inv_sum = 1.0 / row_sum;
                for out_val in &mut row_output {
                    *out_val *= inv_sum;
                }

                row_output
            })
            .collect();

        Tensor::from_vec(vec![q_seq_len, self.head_dim], output)
    }
}

include!("errors.rs");

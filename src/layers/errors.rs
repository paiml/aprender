impl Attention {
    /// Create a new attention layer
    ///
    /// # Arguments
    ///
    /// * `head_dim` - Dimension of each attention head
    ///
    /// # Errors
    ///
    /// Returns error if `head_dim` is zero
    pub fn new(head_dim: usize) -> Result<Self> {
        if head_dim == 0 {
            return Err(RealizarError::InvalidShape {
                reason: "head_dim must be > 0".to_string(),
            });
        }

        #[allow(clippy::cast_precision_loss)]
        let scale = 1.0 / (head_dim as f32).sqrt();

        Ok(Self { head_dim, scale })
    }

    /// Compute scaled dot-product attention
    ///
    /// # Arguments
    ///
    /// * `query` - Query tensor `[seq_len, head_dim]`
    /// * `key` - Key tensor `[seq_len, head_dim]`
    /// * `value` - Value tensor `[seq_len, head_dim]`
    ///
    /// # Returns
    ///
    /// Output tensor `[seq_len, head_dim]`
    ///
    /// # Errors
    ///
    /// Returns error if shapes don't match
    pub fn forward(
        &self,
        query: &Tensor<f32>,
        key: &Tensor<f32>,
        value: &Tensor<f32>,
    ) -> Result<Tensor<f32>> {
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

        // Get sequence lengths
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

        // Compute attention scores: Q @ K.T
        // scores[i][j] = sum(Q[i][k] * K[j][k]) for all k
        let mut scores = Vec::with_capacity(q_seq_len * k_seq_len);
        for i in 0..q_seq_len {
            for j in 0..k_seq_len {
                let mut dot = 0.0;
                for k in 0..self.head_dim {
                    dot += q_data[i * self.head_dim + k] * k_data[j * self.head_dim + k];
                }
                scores.push(dot * self.scale);
            }
        }

        // Apply softmax to each row of scores
        let scores_tensor = Tensor::from_vec(vec![q_seq_len, k_seq_len], scores)?;
        let attn_weights = softmax(&scores_tensor)?;
        let attn_data = attn_weights.data();

        // Compute output: attn_weights @ V
        // output[i][k] = sum(attn_weights[i][j] * V[j][k]) for all j
        let mut output = Vec::with_capacity(q_seq_len * self.head_dim);
        for i in 0..q_seq_len {
            for k in 0..self.head_dim {
                let mut sum = 0.0;
                for j in 0..k_seq_len {
                    sum += attn_data[i * k_seq_len + j] * v_data[j * self.head_dim + k];
                }
                output.push(sum);
            }
        }

        // Debug assertion for numerical stability
        debug_assert!(
            output.iter().all(|&x| x.is_finite()),
            "Attention layer produced NaN or Inf values - check input scaling"
        );

        Tensor::from_vec(vec![q_seq_len, self.head_dim], output)
    }

    /// Get head dimension
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Get scale factor
    #[must_use]
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Compute Flash Attention - memory-efficient block-wise attention
    ///
    /// Uses tiling and recomputation to reduce memory usage from O(N²) to O(N).
    /// Implements block-wise softmax with running max/sum statistics.
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
    ///
    /// # References
    ///
    /// - "`FlashAttention`: Fast and Memory-Efficient Exact Attention" - Dao et al., 2022
    /// - "FlashAttention-2: Faster Attention with Better Parallelism" - Dao, 2023
    pub fn flash_forward(
        &self,
        query: &Tensor<f32>,
        key: &Tensor<f32>,
        value: &Tensor<f32>,
        block_size: usize,
    ) -> Result<Tensor<f32>> {
        if block_size == 0 {
            return Err(RealizarError::InvalidShape {
                reason: "block_size must be > 0".to_string(),
            });
        }

        let q_shape = query.shape();
        let k_shape = key.shape();
        let v_shape = value.shape();

        // Validate shapes (same as standard attention)
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

        // Get sequence lengths
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

        // Initialize output and statistics
        let mut output = vec![0.0; q_seq_len * self.head_dim];
        let mut row_max = vec![f32::NEG_INFINITY; q_seq_len]; // Running max for each query row
        let mut row_sum = vec![0.0; q_seq_len]; // Running sum for each query row

        // Iterate over K/V blocks (outer loop)
        let num_kv_blocks = k_seq_len.div_ceil(block_size);
        for kv_block_idx in 0..num_kv_blocks {
            let kv_start = kv_block_idx * block_size;
            let kv_end = (kv_start + block_size).min(k_seq_len);
            let kv_block_len = kv_end - kv_start;

            // Iterate over Q blocks (inner loop)
            let num_q_blocks = q_seq_len.div_ceil(block_size);
            for q_block_idx in 0..num_q_blocks {
                let q_start = q_block_idx * block_size;
                let q_end = (q_start + block_size).min(q_seq_len);

                // Compute attention scores for this block: Q_block @ K_block.T
                let mut scores = vec![0.0; (q_end - q_start) * kv_block_len];
                for (i, q_idx) in (q_start..q_end).enumerate() {
                    for (j, kv_idx) in (kv_start..kv_end).enumerate() {
                        let mut dot = 0.0;
                        for k in 0..self.head_dim {
                            dot += q_data[q_idx * self.head_dim + k]
                                * k_data[kv_idx * self.head_dim + k];
                        }
                        scores[i * kv_block_len + j] = dot * self.scale;
                    }
                }

                // Update running max and apply softmax with new max
                for (i, q_idx) in (q_start..q_end).enumerate() {
                    // Find max in current block
                    let block_max = (0..kv_block_len)
                        .map(|j| scores[i * kv_block_len + j])
                        .fold(f32::NEG_INFINITY, f32::max);

                    // Update global max
                    let old_max = row_max[q_idx];
                    let new_max = old_max.max(block_max);
                    row_max[q_idx] = new_max;

                    // Compute exp(scores - new_max) and update running sum
                    let mut block_sum = 0.0;
                    for j in 0..kv_block_len {
                        let exp_val = (scores[i * kv_block_len + j] - new_max).exp();
                        scores[i * kv_block_len + j] = exp_val;
                        block_sum += exp_val;
                    }

                    // Rescale old output and sum based on new max
                    let scale_factor = (old_max - new_max).exp();
                    for k in 0..self.head_dim {
                        output[q_idx * self.head_dim + k] *= scale_factor;
                    }
                    row_sum[q_idx] = row_sum[q_idx] * scale_factor + block_sum;
                }

                // Accumulate weighted values: output += scores @ V_block
                for (i, q_idx) in (q_start..q_end).enumerate() {
                    for k in 0..self.head_dim {
                        let mut weighted_sum = 0.0;
                        for (j, kv_idx) in (kv_start..kv_end).enumerate() {
                            weighted_sum +=
                                scores[i * kv_block_len + j] * v_data[kv_idx * self.head_dim + k];
                        }
                        output[q_idx * self.head_dim + k] += weighted_sum;
                    }
                }
            }
        }

        // Final normalization by row_sum
        for i in 0..q_seq_len {
            for k in 0..self.head_dim {
                output[i * self.head_dim + k] /= row_sum[i];
            }
        }

        Tensor::from_vec(vec![q_seq_len, self.head_dim], output)
    }

    /// Flash Attention v2 with SIMD-accelerated dot products
    ///
    /// Optimized implementation using AVX2 SIMD for dot products.
    /// Uses parallel outer loop over query blocks for better multi-core utilization.
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
    ///
    /// # References
    ///
    /// - "FlashAttention-2: Faster Attention with Better Parallelism" - Dao, 2023
    #[allow(clippy::similar_names)]
    pub fn flash_forward_v2(
        &self,
        query: &Tensor<f32>,
        key: &Tensor<f32>,
        value: &Tensor<f32>,
        block_size: usize,
    ) -> Result<Tensor<f32>> {
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

        // Initialize output and statistics
        let mut output = vec![0.0; q_seq_len * head_dim];
        let mut row_max = vec![f32::NEG_INFINITY; q_seq_len];
        let mut row_sum = vec![0.0; q_seq_len];

        // Flash Attention v2: Iterate over K/V blocks in outer loop
        // This allows better memory access patterns
        let num_kv_blocks = k_seq_len.div_ceil(block_size);

        for kv_block_idx in 0..num_kv_blocks {
            let kv_start = kv_block_idx * block_size;
            let kv_end = (kv_start + block_size).min(k_seq_len);
            let kv_block_len = kv_end - kv_start;

            // Process all Q rows against this K/V block
            for q_idx in 0..q_seq_len {
                // SIMD-accelerated dot products for this row
                let mut scores = Vec::with_capacity(kv_block_len);
                for kv_idx in kv_start..kv_end {
                    let dot = Self::simd_dot_product(
                        &q_data[q_idx * head_dim..(q_idx + 1) * head_dim],
                        &k_data[kv_idx * head_dim..(kv_idx + 1) * head_dim],
                    );
                    scores.push(dot * scale);
                }

                // Find max in current block
                let block_max = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

                // Update global max
                let old_max = row_max[q_idx];
                let new_max = old_max.max(block_max);
                row_max[q_idx] = new_max;

                // Compute exp(scores - new_max) and update running sum
                let mut block_sum = 0.0;
                for score in &mut scores {
                    let exp_val = (*score - new_max).exp();
                    *score = exp_val;
                    block_sum += exp_val;
                }

                // Rescale old output and sum based on new max
                let scale_factor = (old_max - new_max).exp();
                for k in 0..head_dim {
                    output[q_idx * head_dim + k] *= scale_factor;
                }
                row_sum[q_idx] = row_sum[q_idx] * scale_factor + block_sum;

                // Accumulate weighted values: output += scores @ V_block
                for (j, kv_idx) in (kv_start..kv_end).enumerate() {
                    let weight = scores[j];
                    for k in 0..head_dim {
                        output[q_idx * head_dim + k] += weight * v_data[kv_idx * head_dim + k];
                    }
                }
            }
        }

        // Final normalization by row_sum
        for i in 0..q_seq_len {
            let inv_sum = 1.0 / row_sum[i];
            for k in 0..head_dim {
                output[i * head_dim + k] *= inv_sum;
            }
        }

        Tensor::from_vec(vec![q_seq_len, self.head_dim], output)
    }
}

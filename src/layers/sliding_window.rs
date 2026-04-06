
// ============================================================================
// Sliding Window Attention (Mistral/Mixtral style)
// ============================================================================
//
// Limits attention to a fixed window of recent tokens for efficient
// long-context inference. Used by Mistral-7B, Mixtral, and similar models.
//
// Benefits:
// - Reduces memory from O(n²) to O(n*w) where w = window_size
// - Enables very long context with bounded KV cache
// - Compatible with Flash Attention algorithms
//
// Reference: "Mistral 7B" - Jiang et al., 2023
// ============================================================================

/// Sliding Window Attention
///
/// Limits each token to attending only to the most recent `window_size` tokens.
/// This provides linear memory scaling for long sequences while maintaining
/// local context.
///
/// # Algorithm
///
/// For each query position i, attention is computed only over keys/values
/// in positions `[max(0, i - window_size + 1), i]`.
///
/// ```text
/// Standard Attention (full):  Sliding Window (w=3):
///   Q K K K K K                 Q K K K . .
///   Q K K K K K                 . Q K K K .
///   Q K K K K K                 . . Q K K K
///   Q K K K K K                 . . . Q K K
/// ```
///
/// # References
///
/// - "Mistral 7B" - Jiang et al., 2023
/// - "Longformer: The Long-Document Transformer" - Beltagy et al., 2020
#[derive(Debug, Clone)]
pub struct SlidingWindowAttention {
    /// Head dimension (`d_k` = `d_model` / `num_heads`)
    head_dim: usize,
    /// Scale factor: 1 / `sqrt(head_dim)`
    scale: f32,
    /// Window size (number of tokens each query can attend to)
    window_size: usize,
}

impl SlidingWindowAttention {
    /// Create a new sliding window attention layer
    ///
    /// # Arguments
    ///
    /// * `head_dim` - Dimension of each attention head
    /// * `window_size` - Number of tokens each query can attend to
    ///
    /// # Errors
    ///
    /// Returns error if `head_dim` is zero or `window_size` is zero
    pub fn new(head_dim: usize, window_size: usize) -> Result<Self> {
        if head_dim == 0 {
            return Err(RealizarError::InvalidShape {
                reason: "head_dim must be > 0".to_string(),
            });
        }
        if window_size == 0 {
            return Err(RealizarError::InvalidShape {
                reason: "window_size must be > 0".to_string(),
            });
        }

        #[allow(clippy::cast_precision_loss)]
        let scale = 1.0 / (head_dim as f32).sqrt();

        Ok(Self {
            head_dim,
            scale,
            window_size,
        })
    }

    /// Compute sliding window attention
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

        let mut output = Vec::with_capacity(q_seq_len * self.head_dim);

        // Process each query position with sliding window
        for i in 0..q_seq_len {
            // Compute window boundaries [window_start, window_end)
            // For causal attention: can only attend to positions <= i
            let window_end = (i + 1).min(k_seq_len);
            let window_start = window_end.saturating_sub(self.window_size);
            let window_len = window_end - window_start;

            if window_len == 0 {
                // No keys to attend to, output zeros
                output.extend(std::iter::repeat_n(0.0, self.head_dim));
                continue;
            }

            // Compute attention scores for this window
            let mut scores = Vec::with_capacity(window_len);
            for j in window_start..window_end {
                let mut dot = 0.0;
                for k in 0..self.head_dim {
                    dot += q_data[i * self.head_dim + k] * k_data[j * self.head_dim + k];
                }
                scores.push(dot * self.scale);
            }

            // Apply softmax over window scores
            let max_score = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mut exp_sum = 0.0;
            for score in &mut scores {
                let exp_val = (*score - max_score).exp();
                *score = exp_val;
                exp_sum += exp_val;
            }
            let inv_sum = 1.0 / exp_sum;
            for score in &mut scores {
                *score *= inv_sum;
            }

            // Compute weighted sum of values
            for k in 0..self.head_dim {
                let mut sum = 0.0;
                for (idx, j) in (window_start..window_end).enumerate() {
                    sum += scores[idx] * v_data[j * self.head_dim + k];
                }
                output.push(sum);
            }
        }

        Tensor::from_vec(vec![q_seq_len, self.head_dim], output)
    }

    /// Compute sliding window attention with mask
    ///
    /// Supports bidirectional attention (non-causal) with the sliding window.
    ///
    /// # Arguments
    ///
    /// * `query` - Query tensor `[seq_len, head_dim]`
    /// * `key` - Key tensor `[seq_len, head_dim]`
    /// * `value` - Value tensor `[seq_len, head_dim]`
    /// * `causal` - If true, only attend to past positions (causal/autoregressive)
    ///
    /// # Returns
    ///
    /// Output tensor `[seq_len, head_dim]`
    ///
    /// # Errors
    ///
    /// Returns error if shapes don't match
    pub fn forward_with_mask(
        &self,
        query: &Tensor<f32>,
        key: &Tensor<f32>,
        value: &Tensor<f32>,
        causal: bool,
    ) -> Result<Tensor<f32>> {
        if causal {
            // Causal is the default behavior
            return self.forward(query, key, value);
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

        let mut output = Vec::with_capacity(q_seq_len * self.head_dim);
        let half_window = self.window_size / 2;

        // Process each query position with bidirectional sliding window
        for i in 0..q_seq_len {
            // Bidirectional window centered on position i
            let window_start = i.saturating_sub(half_window);
            let window_end = (i + half_window + 1).min(k_seq_len);
            let window_len = window_end - window_start;

            if window_len == 0 {
                output.extend(std::iter::repeat_n(0.0, self.head_dim));
                continue;
            }

            // Compute attention scores for this window
            let mut scores = Vec::with_capacity(window_len);
            for j in window_start..window_end {
                let mut dot = 0.0;
                for k in 0..self.head_dim {
                    dot += q_data[i * self.head_dim + k] * k_data[j * self.head_dim + k];
                }
                scores.push(dot * self.scale);
            }

            // Apply softmax over window scores
            let max_score = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mut exp_sum = 0.0;
            for score in &mut scores {
                let exp_val = (*score - max_score).exp();
                *score = exp_val;
                exp_sum += exp_val;
            }
            let inv_sum = 1.0 / exp_sum;
            for score in &mut scores {
                *score *= inv_sum;
            }

            // Compute weighted sum of values
            for k in 0..self.head_dim {
                let mut sum = 0.0;
                for (idx, j) in (window_start..window_end).enumerate() {
                    sum += scores[idx] * v_data[j * self.head_dim + k];
                }
                output.push(sum);
            }
        }

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

    /// Get window size
    #[must_use]
    pub fn window_size(&self) -> usize {
        self.window_size
    }

    /// Compute the effective context at a given position
    ///
    /// Returns the number of tokens this position can attend to
    #[must_use]
    pub fn effective_context(&self, position: usize, seq_len: usize) -> usize {
        contract_pre_effective_context!();
        let window_end = (position + 1).min(seq_len);
        let window_start = window_end.saturating_sub(self.window_size);
        let result = window_end - window_start;
        contract_post_effective_context!(&result);
        result
    }

    /// Memory usage relative to full attention
    ///
    /// Returns the ratio of memory used compared to full attention.
    /// For window_size w and seq_len n: memory = O(n*w) vs O(n²)
    #[must_use]
    pub fn memory_ratio(&self, seq_len: usize) -> f32 {
        if seq_len == 0 {
            return 1.0;
        }
        #[allow(clippy::cast_precision_loss)]
        {
            (self.window_size.min(seq_len) as f32) / (seq_len as f32)
        }
    }
}

// ============================================================================
// Fused QKV + Attention (IMP-003)
// Per spec: performance-parity-ollama-llamacpp-gpu-inference-llms.md
// ============================================================================

/// Fused Query-Key-Value projection with scaled dot-product attention
///
/// Combines QKV projection and attention into a single fused operation for
/// improved memory efficiency and performance. Eliminates intermediate
/// materializations by computing attention in a single pass.
///
/// # Performance Benefits
///
/// - **Memory Bandwidth**: Single read of input, single write of output
/// - **Cache Efficiency**: QKV computed block-wise to maximize L1/L2 reuse
/// - **Numerical Stability**: Uses log-sum-exp trick for softmax
///
/// # Algorithm (Flash Attention style)
///
/// ```text
/// for each block of queries:
///     Q_block = input_block @ W_q
///     for each block of keys/values:
///         K_block = input_block @ W_k
///         V_block = input_block @ W_v
///         scores = Q_block @ K_block^T / sqrt(d)
///         update running softmax and output
/// ```
///
/// # References
///
/// - [1] Dao et al., "FlashAttention: Fast and Memory-Efficient Attention", 2022
/// - [2] Tri Dao, "FlashAttention-2: Faster Attention with Better Parallelism", 2023
#[derive(Debug, Clone)]
pub struct FusedQKVAttention {
    /// Dimension per attention head
    head_dim: usize,
    /// Total hidden dimension
    hidden_dim: usize,
    /// Number of attention heads
    num_heads: usize,
    /// Scale factor: 1 / sqrt(head_dim)
    scale: f32,
    /// Query projection weights: [hidden_dim, hidden_dim]
    w_q: Vec<f32>,
    /// Key projection weights: [hidden_dim, hidden_dim]
    w_k: Vec<f32>,
    /// Value projection weights: [hidden_dim, hidden_dim]
    w_v: Vec<f32>,
    /// Output projection weights: [hidden_dim, hidden_dim]
    w_o: Vec<f32>,
}

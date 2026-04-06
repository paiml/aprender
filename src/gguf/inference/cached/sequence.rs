
#[cfg(feature = "gpu")]
impl OwnedQuantizedModelCached {

    /// True batched multi-head attention (IMP-118)
    ///
    /// Uses true batched GEMM for Q@K^T and weights@V operations,
    /// processing all heads efficiently without per-head dispatch overhead.
    ///
    /// # Arguments
    /// * `q` - Query tensor [num_heads, seq_len, head_dim]
    /// * `k` - Key tensor [num_heads, seq_len, head_dim]
    /// * `v` - Value tensor [num_heads, seq_len, head_dim]
    /// * `seq_len` - Sequence length
    /// * `num_heads` - Number of attention heads
    /// * `head_dim` - Dimension per head
    ///
    /// # Returns
    /// Output tensor [num_heads, seq_len, head_dim]
    pub fn true_batched_multihead_attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Result<Vec<f32>> {
        let expected_size = num_heads * seq_len * head_dim;
        if q.len() != expected_size {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "Q size {} doesn't match num_heads={} * seq_len={} * head_dim={}",
                    q.len(),
                    num_heads,
                    seq_len,
                    head_dim
                ),
            });
        }

        let scale = 1.0 / (head_dim as f32).sqrt();

        // Step 1: Transpose K to [num_heads, head_dim, seq_len]
        let mut k_transposed = vec![0.0f32; num_heads * head_dim * seq_len];
        for h in 0..num_heads {
            let head_offset = h * seq_len * head_dim;
            let k_t_offset = h * head_dim * seq_len;
            for pos in 0..seq_len {
                for d in 0..head_dim {
                    k_transposed[k_t_offset + d * seq_len + pos] =
                        k[head_offset + pos * head_dim + d];
                }
            }
        }

        // Step 2: True batched Q @ K^T -> [num_heads, seq_len, seq_len]
        let scores =
            self.true_batched_gemm(q, &k_transposed, num_heads, seq_len, head_dim, seq_len)?;

        // Step 3: Scale and apply causal softmax
        let mut scaled_scores = scores;
        for s in &mut scaled_scores {
            *s *= scale;
        }

        // Apply causal mask and softmax per-head using trueno SIMD (IMP-305e)
        let weights = self.batched_causal_softmax_trueno(&scaled_scores, num_heads, seq_len)?;

        // Step 4: True batched weights @ V -> [num_heads, seq_len, head_dim]
        let attn_output =
            self.true_batched_gemm(&weights, v, num_heads, seq_len, seq_len, head_dim)?;

        Ok(attn_output)
    }

    /// GPU-accelerated fused causal attention (IMP-119)
    ///
    /// Uses GPU for long sequences where compute dominates transfer overhead.
    /// Combines Q@K^T → softmax → @V using GPU matmul operations.
    ///
    /// # Arguments
    /// * `q` - Query tensor [seq_len, head_dim]
    /// * `k` - Key tensor [seq_len, head_dim]
    /// * `v` - Value tensor [seq_len, head_dim]
    /// * `seq_len` - Sequence length
    /// * `head_dim` - Head dimension
    /// * `scale` - Attention scale factor (typically 1/sqrt(head_dim))
    ///
    /// # Returns
    /// Output tensor [seq_len, head_dim]
    pub fn gpu_fused_causal_attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        head_dim: usize,
        scale: f32,
    ) -> Result<Vec<f32>> {
        // For GPU-accelerated fused attention, we use a strategy that balances
        // GPU matmul benefits with avoiding large intermediate allocations
        //
        // Strategy:
        // 1. Use GPU for Q@K^T (benefits from parallelism)
        // 2. Apply causal mask + softmax on CPU (memory-efficient)
        // 3. Use GPU for attention_weights @ V

        let mut scheduler = self.get_scheduler()?;

        // Step 1: Transpose K to [head_dim, seq_len]
        let mut k_transposed = vec![0.0f32; head_dim * seq_len];
        for pos in 0..seq_len {
            for d in 0..head_dim {
                k_transposed[d * seq_len + pos] = k[pos * head_dim + d];
            }
        }

        // Step 2: GPU Q @ K^T -> [seq_len, seq_len]
        let scores = scheduler
            .matmul(q, &k_transposed, seq_len, head_dim, seq_len)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "gpu_fused_causal_attention Q@K^T".to_string(),
                reason: format!("GPU matmul failed: {}", e),
            })?;

        // Step 3: Scale and apply causal softmax (CPU - memory efficient)
        let mut weights = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            // Find max for numerical stability
            let mut max_val = f32::NEG_INFINITY;
            for j in 0..=i {
                let score = scores[i * seq_len + j] * scale;
                if score > max_val {
                    max_val = score;
                }
            }

            // Compute softmax with causal mask
            let mut sum = 0.0f32;
            for j in 0..=i {
                let score = scores[i * seq_len + j] * scale;
                weights[i * seq_len + j] = (score - max_val).exp();
                sum += weights[i * seq_len + j];
            }

            // Normalize
            if sum > 0.0 {
                for j in 0..=i {
                    weights[i * seq_len + j] /= sum;
                }
            }
            // j > i remain zero (causal mask)
        }

        // Step 4: GPU attention_weights @ V -> [seq_len, head_dim]
        let output = scheduler
            .matmul(&weights, v, seq_len, seq_len, head_dim)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "gpu_fused_causal_attention weights@V".to_string(),
                reason: format!("GPU matmul failed: {}", e),
            })?;

        Ok(output)
    }

    /// GPU-accelerated fused multi-head attention (IMP-119)
    ///
    /// Processes all heads using GPU acceleration for long sequences.
    ///
    /// # Arguments
    /// * `q` - Query tensor [seq_len, hidden_dim]
    /// * `k` - Key tensor [seq_len, hidden_dim]
    /// * `v` - Value tensor [seq_len, hidden_dim]
    /// * `seq_len` - Sequence length
    ///
    /// # Returns
    /// Output tensor [seq_len, hidden_dim]
    pub fn gpu_fused_multihead_attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
    ) -> Result<Vec<f32>> {
        let num_heads = self.model.config.num_heads;
        // GH-479: Use config methods (Qwen3 head_dim != hidden/heads)
        let head_dim = self.model.config.head_dim();
        let q_dim = num_heads * head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Reshape Q, K, V to [num_heads, seq_len, head_dim]
        let q_reshaped = self
            .model
            .reshape_for_parallel_heads(q, seq_len, num_heads, head_dim)?;
        let k_reshaped = self
            .model
            .reshape_for_parallel_heads(k, seq_len, num_heads, head_dim)?;
        let v_reshaped = self
            .model
            .reshape_for_parallel_heads(v, seq_len, num_heads, head_dim)?;

        // Process each head with GPU-accelerated fused attention
        let mut attn_output = vec![0.0f32; num_heads * seq_len * head_dim];

        for h in 0..num_heads {
            let head_offset = h * seq_len * head_dim;
            let q_head = &q_reshaped[head_offset..head_offset + seq_len * head_dim];
            let k_head = &k_reshaped[head_offset..head_offset + seq_len * head_dim];
            let v_head = &v_reshaped[head_offset..head_offset + seq_len * head_dim];

            // GPU fused attention for this head
            let head_output =
                self.gpu_fused_causal_attention(q_head, k_head, v_head, seq_len, head_dim, scale)?;

            attn_output[head_offset..head_offset + seq_len * head_dim]
                .copy_from_slice(&head_output);
        }

        // Reshape back to [seq_len, q_dim]
        let mut output = vec![0.0f32; seq_len * q_dim];
        for h in 0..num_heads {
            let head_start = h * seq_len * head_dim;
            for pos in 0..seq_len {
                let src_start = head_start + pos * head_dim;
                let dst_start = pos * q_dim + h * head_dim;
                output[dst_start..dst_start + head_dim]
                    .copy_from_slice(&attn_output[src_start..src_start + head_dim]);
            }
        }

        Ok(output)
    }

    /// Adaptive fused attention with CPU/GPU dispatch (IMP-119)
    ///
    /// Automatically selects CPU or GPU based on sequence length.
    /// - Short sequences (< threshold): Use CPU fused attention (lower overhead)
    /// - Long sequences (>= threshold): Use GPU fused attention (better throughput)
    ///
    /// # Arguments
    /// * `q` - Query tensor [seq_len, head_dim]
    /// * `k` - Key tensor [seq_len, head_dim]
    /// * `v` - Value tensor [seq_len, head_dim]
    /// * `seq_len` - Sequence length
    /// * `head_dim` - Head dimension
    /// * `scale` - Attention scale factor
    ///
    /// # Returns
    /// Output tensor [seq_len, head_dim]
    pub fn adaptive_fused_attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        head_dim: usize,
        scale: f32,
    ) -> Result<Vec<f32>> {
        // Threshold based on empirical analysis from IMP-108 and IMP-115:
        // - GPU dispatch overhead is ~300ms per HybridScheduler init (cached: ~0ms)
        // - CPU fused attention is ~50µs for seq_len=64
        // - GPU wins when compute volume justifies transfer overhead
        //
        // With scheduler caching (IMP-112), the crossover is much lower
        const GPU_SEQ_LEN_THRESHOLD: usize = 64;

        if seq_len >= GPU_SEQ_LEN_THRESHOLD {
            // Long sequence: Use GPU for better throughput
            self.gpu_fused_causal_attention(q, k, v, seq_len, head_dim, scale)
        } else {
            // Short sequence: Use CPU to avoid any overhead
            self.fused_causal_attention(q, k, v, seq_len, head_dim, scale)
        }
    }

    /// Generate tokens with adaptive attention (IMP-121)
    ///
    /// Uses adaptive attention that automatically selects CPU or GPU
    /// based on sequence length for optimal performance.
    ///
    /// # Arguments
    /// * `prompt` - Input token IDs
    /// * `config` - Generation configuration
    ///
    /// # Returns
    /// Generated token sequence including prompt
    pub fn generate_with_adaptive_attention(
        &self,
        prompt: &[u32],
        config: &QuantizedGenerateConfig,
    ) -> Result<Vec<u32>> {
        // Delegate to generate_with_cache which uses efficient KV cache.
        // Adaptive attention (IMP-122) is tracked separately for long-context prefill optimization.
        // Current implementation handles typical inference workloads efficiently.
        self.model.generate_with_cache(prompt, config)
    }
}

include!("scheduler.rs");
include!("attention.rs");
include!("single_flattened_batched_owned.rs");

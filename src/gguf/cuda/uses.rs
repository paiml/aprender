
impl OwnedQuantizedModelCuda {

    /// GPU-accelerated attention with KV cache using multi-head CUDA kernel (PARITY-044)
    ///
    /// Uses `CudaExecutor::flash_attention_multi_head` to process all heads in parallel.
    /// Memory layout: [n_heads, seq_len, head_dim]
    ///
    /// # Arguments
    ///
    /// * `num_heads` - Number of attention heads
    /// * `head_dim` - Dimension per head (hidden_dim / num_heads)
    #[allow(clippy::too_many_arguments)]
    fn cuda_attention_with_cache(
        &mut self,
        q: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        current_k: &[f32],
        current_v: &[f32],
        total_len: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Result<Vec<f32>> {
        let hidden_dim = num_heads * head_dim;
        let cache_len = total_len - 1;

        // Build full K and V tensors for all heads: [n_heads, total_len, head_dim]
        let tensor_size = num_heads * total_len * head_dim;

        // For GPU multi-head attention, we need Q repeated across all positions
        // Q is [hidden_dim] = [n_heads * head_dim], expand to [n_heads, total_len, head_dim]
        let mut q_full = vec![0.0f32; tensor_size];
        let mut k_full = vec![0.0f32; tensor_size];
        let mut v_full = vec![0.0f32; tensor_size];

        // Reorganize from [seq_len, n_heads * head_dim] to [n_heads, seq_len, head_dim]
        for head in 0..num_heads {
            let head_offset = head * head_dim;
            let gpu_head_offset = head * total_len * head_dim;

            // Q: single query expanded to all positions (for proper broadcast)
            for pos in 0..total_len {
                let gpu_pos_offset = gpu_head_offset + pos * head_dim;
                q_full[gpu_pos_offset..gpu_pos_offset + head_dim]
                    .copy_from_slice(&q[head_offset..head_offset + head_dim]);
            }

            // K: cached + current
            for pos in 0..cache_len {
                let cache_offset = pos * hidden_dim + head_offset;
                let gpu_pos_offset = gpu_head_offset + pos * head_dim;
                k_full[gpu_pos_offset..gpu_pos_offset + head_dim]
                    .copy_from_slice(&k_cache[cache_offset..cache_offset + head_dim]);
            }
            // Current K
            let gpu_current_offset = gpu_head_offset + cache_len * head_dim;
            k_full[gpu_current_offset..gpu_current_offset + head_dim]
                .copy_from_slice(&current_k[head_offset..head_offset + head_dim]);

            // V: cached + current
            for pos in 0..cache_len {
                let cache_offset = pos * hidden_dim + head_offset;
                let gpu_pos_offset = gpu_head_offset + pos * head_dim;
                v_full[gpu_pos_offset..gpu_pos_offset + head_dim]
                    .copy_from_slice(&v_cache[cache_offset..cache_offset + head_dim]);
            }
            // Current V
            v_full[gpu_current_offset..gpu_current_offset + head_dim]
                .copy_from_slice(&current_v[head_offset..head_offset + head_dim]);
        }

        // GPU multi-head attention using FlashAttention kernel
        let mut output_full = vec![0.0f32; tensor_size];
        self.executor
            .flash_attention_multi_head(
                &q_full,
                &k_full,
                &v_full,
                &mut output_full,
                total_len as u32,
                head_dim as u32,
                num_heads as u32,
                true, // causal masking for autoregressive decoding
            )
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "flash_attention_multi_head".to_string(),
                reason: format!("CUDA attention failed: {e}"),
            })?;

        // Extract output for the last position and reorganize to [hidden_dim]
        let mut output = vec![0.0f32; hidden_dim];
        let last_pos = total_len - 1;
        for head in 0..num_heads {
            let head_offset = head * head_dim;
            let gpu_head_offset = head * total_len * head_dim;
            let gpu_pos_offset = gpu_head_offset + last_pos * head_dim;
            output[head_offset..head_offset + head_dim]
                .copy_from_slice(&output_full[gpu_pos_offset..gpu_pos_offset + head_dim]);
        }

        Ok(output)
    }

    /// Generate tokens using CUDA acceleration with KV cache (PARITY-044)
    ///
    /// Uses `forward_single_cuda_with_cache` for GPU-accelerated incremental decoding.
    ///
    /// # Arguments
    ///
    /// * `prompt` - Initial token IDs
    /// * `config` - Generation configuration
    ///
    /// # Returns
    ///
    /// Generated token sequence including prompt
    pub fn forward_gpu_resident(
        &mut self,
        token_id: u32,
        cache: &mut OwnedQuantizedKVCache,
        position: usize,
    ) -> Result<Vec<f32>> {
        let hidden_dim = self.model.config.hidden_dim;
        let intermediate_dim = self.model.layers[0].ffn_up_weight.out_dim;
        let num_layers = self.model.layers.len();
        let vocab_size = self.model.lm_head_weight.out_dim;
        let eps = self.model.config.eps;

        // 1. Token embedding lookup (CPU - fast, single lookup, zero-alloc)
        // PAR-083: Use pre-allocated embed_buf to eliminate per-token heap allocation.
        self.model.embed_into(token_id, &mut self.embed_buf);

        // 2. Fully GPU-resident forward: layers + output norm + LM head
        // PAR-054: Use CUDA graph-captured path for decode (reduces 280 launches to 1)
        // Only 2 syncs total: embedding upload + logits download
        let mut logits = vec![0.0f32; vocab_size];
        self.executor
            .forward_all_layers_gpu_to_logits_graphed(
                &self.embed_buf,
                &mut logits,
                position as u32,
                num_layers,
                hidden_dim as u32,
                intermediate_dim as u32,
                vocab_size as u32,
                eps,
            )
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "forward_gpu_resident".to_string(),
                reason: format!("forward_all_layers_gpu_to_logits_graphed failed: {}", e),
            })?;

        // 3. Add LM head bias if present (CPU - fast)
        if let Some(ref bias) = self.model.lm_head_bias {
            ops::add_bias(&mut logits, bias);
        }

        // Advance cache position (for compatibility with cache-based generation)
        cache.advance();

        Ok(logits)
    }

    /// PAR-062: GPU-resident forward pass returning token ID directly
    ///
    /// Like `forward_gpu_resident` but uses GPU-side argmax for greedy sampling.
    /// Eliminates 600KB logits transfer per token, reducing to 4 bytes (token ID).
    ///
    /// # Performance Improvement
    ///
    /// - Before: Download 152064 x 4 = 600KB per token
    /// - After: Download 1 x 4 = 4 bytes per token
    /// - Expected speedup: ~1.2x overall throughput
    ///
    /// # Arguments
    ///
    /// * `token_id` - Input token
    /// * `cache` - KV cache (advanced but not used for logits)
    /// * `position` - Position in sequence
    ///
    /// # Returns
    ///
    /// Token ID with highest logit value (greedy sampling)
    ///
    /// # Errors
    ///
    /// Returns error if GPU operations fail or model has lm_head_bias (requires CPU path).
    pub fn forward_gpu_resident_to_token_id(
        &mut self,
        token_id: u32,
        cache: &mut OwnedQuantizedKVCache,
        position: usize,
    ) -> Result<u32> {
        // CORRECTNESS-013: Check if deterministic mode is requested
        // In this mode, download logits to CPU for argmax to ensure bit-exact
        // output matching between CPU and GPU inference paths.
        static CORRECTNESS_MODE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let use_cpu_argmax = *CORRECTNESS_MODE.get_or_init(|| {
            std::env::var("CORRECTNESS_MODE")
                .map(|v| v == "1")
                .unwrap_or(false)
        });

        // PAR-062: If model has LM head bias, fall back to CPU path
        // (bias addition requires CPU, so we'd download logits anyway)
        if self.model.lm_head_bias.is_some() || use_cpu_argmax {
            let logits = self.forward_gpu_resident(token_id, cache, position)?;
            return Ok(logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map_or(0, |(idx, _)| idx as u32));
        }

        // PAR-083: Per-phase decode timing for Five-Whys diagnosis
        static DECODE_TIMING: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let timing = *DECODE_TIMING.get_or_init(|| {
            std::env::var("DECODE_TIMING")
                .map(|v| v == "1")
                .unwrap_or(false)
        });

        let hidden_dim = self.model.config.hidden_dim;
        let intermediate_dim = self.model.layers[0].ffn_up_weight.out_dim;
        let num_layers = self.model.layers.len();
        let vocab_size = self.model.lm_head_weight.out_dim;
        let eps = self.model.config.eps;

        // 1. Token embedding lookup (CPU - fast, single lookup, zero-alloc)
        // PAR-083: Use pre-allocated embed_buf to eliminate per-token heap allocation.
        // Five-Whys: embed() allocated Vec<f32> per token → 14KB malloc/free overhead.
        let t0 = if timing {
            Some(std::time::Instant::now())
        } else {
            None
        };
        self.model.embed_into(token_id, &mut self.embed_buf);
        let t1 = t0.map(|_| std::time::Instant::now());

        // C-GDP-001: When profiling is enabled, always use eager path with per-brick
        // instrumentation. CUDA graph replay executes all kernels in one opaque launch,
        // hiding brick-level timing. Contract: gpu-decode-profiling-v1 FALSIFY-GDP-001.
        let profiling_enabled = self.executor.is_profiling_enabled();

        // 2. Check if CUDA graph is captured; if not, use regular path first
        // The graphed path needs to be initialized via forward_all_layers_gpu_to_logits_graphed
        if profiling_enabled || !self.executor.has_decode_graph() {
            // First call - need to capture graph, use regular path
            let mut logits = vec![0.0f32; vocab_size];
            self.executor
                .forward_all_layers_gpu_to_logits_graphed(
                    &self.embed_buf,
                    &mut logits,
                    position as u32,
                    num_layers,
                    hidden_dim as u32,
                    intermediate_dim as u32,
                    vocab_size as u32,
                    eps,
                )
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "forward_gpu_resident_to_token_id".to_string(),
                    reason: format!("forward_all_layers_gpu_to_logits_graphed failed: {}", e),
                })?;

            cache.advance();
            return Ok(logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map_or(0, |(idx, _)| idx as u32));
        }

        // 3. Use GPU argmax path - graph is captured, use optimized replay
        let next_token = self
            .executor
            .forward_graphed_replay_to_token_id(&self.embed_buf, position as u32, vocab_size as u32)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "forward_gpu_resident_to_token_id".to_string(),
                reason: format!("forward_graphed_replay_to_token_id failed: {}", e),
            })?;
        let t2 = t0.map(|_| std::time::Instant::now());

        // PAR-083: Per-phase decode timing output
        if let (Some(t0v), Some(t1v), Some(t2v)) = (t0, t1, t2) {
            let embed_us = t1v.duration_since(t0v).as_micros();
            let gpu_us = t2v.duration_since(t1v).as_micros();
            let total_us = t2v.duration_since(t0v).as_micros();
            eprintln!(
                "[DECODE-TIMING] pos={}: embed={}µs gpu={}µs total={}µs ({:.0} tok/s)",
                position,
                embed_us,
                gpu_us,
                total_us,
                if total_us > 0 {
                    1_000_000.0 / total_us as f64
                } else {
                    0.0
                }
            );
        }

        cache.advance();
        Ok(next_token)
    }
}

include!("cuda.rs");
include!("matmul.rs");
include!("clone_layer_weights.rs");

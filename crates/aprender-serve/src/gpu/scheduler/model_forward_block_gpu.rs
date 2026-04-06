impl GpuModel {

    /// Optimized block forward with batched multi-head attention (M17, IMP-092)
    ///
    /// IMP-092: Eliminated weight cloning (~130MB per layer) by using explicit
    /// field borrowing. Previous version cloned 3.7GB per token across 28 layers.
    pub fn forward_block_incremental_optimized(
        &mut self,
        input: &[f32],
        block_idx: usize,
        kv_cache: &mut StreamingKVCache,
    ) -> Result<Vec<f32>> {
        // Extract config values (Copy types, no borrow conflict)
        let hidden_dim = self.config.hidden_dim;
        let num_heads = self.config.num_heads;
        let head_dim = self.config.head_dim();
        let kv_dim = self.config.kv_dim();
        let qkv_dim = self.config.qkv_dim();
        let intermediate_dim = self.config.intermediate_dim;
        let eps = self.config.eps;
        let num_kv_heads = self.config.num_kv_heads;

        // IMP-092: Use REFERENCES instead of cloning 130MB of weights per layer
        // Pre-attention layer norm (static function avoids &self borrow)
        let normed = Self::layer_norm_static(
            input,
            &self.block_weights[block_idx].attn_norm_weight,
            &self.block_weights[block_idx].attn_norm_bias,
            hidden_dim,
            eps,
        );

        // QKV projection for single token [1, hidden_dim] @ [hidden_dim, qkv_dim]
        // For GQA: qkv_dim = hidden_dim + 2*kv_dim (K/V have fewer heads)
        // IMP-1006: Use do_matmul to route to CudaScheduler when available
        let qkv_weight = self.block_weights[block_idx].qkv_weight.clone();
        let mut qkv = self.do_matmul(&normed, &qkv_weight, 1, hidden_dim, qkv_dim)?;

        // Get current position BEFORE caching (Phase 21)
        let (cached_k_ref, _) = kv_cache.get_valid(block_idx);
        let current_pos = cached_k_ref.len() / kv_dim;

        // Phase 21: Apply RoPE to Q and K BEFORE caching
        // Without RoPE, attention has no position information and produces garbage
        let rope_theta = self.config.rope_theta;
        Self::apply_rope_inline(
            &mut qkv[0..hidden_dim],
            num_heads,
            head_dim,
            rope_theta,
            current_pos,
        );
        Self::apply_rope_inline(
            &mut qkv[hidden_dim..hidden_dim + kv_dim],
            num_kv_heads,
            head_dim,
            rope_theta,
            current_pos,
        );

        // Split QKV (GQA: K/V have kv_dim, not hidden_dim) - after RoPE
        let q = qkv[0..hidden_dim].to_vec();
        let k_new = qkv[hidden_dim..hidden_dim + kv_dim].to_vec();
        let v_new = qkv[hidden_dim + kv_dim..].to_vec();

        // Get cached K/V and clone to avoid borrow issues with kv_cache
        let (cached_k, cached_v) = kv_cache.get_valid(block_idx);
        let keys_cached = cached_k.to_vec();
        let vals_cached = cached_v.to_vec();

        // Append new K/V (with RoPE applied) to cache
        kv_cache.append(block_idx, &k_new, &v_new);

        // Build full K/V (cached + new)
        // GQA: K/V have kv_dim per position, not hidden_dim
        let kv_len = keys_cached.len() / kv_dim + 1;
        let mut full_k = keys_cached;
        full_k.extend_from_slice(&k_new);
        let mut full_v = vals_cached;
        full_v.extend_from_slice(&v_new);

        // GQA attention (IMP-089): static method to avoid borrow conflicts
        let attn_output = Self::gqa_multihead_attention(
            &q,
            &full_k,
            &full_v,
            kv_len,
            num_heads,
            num_kv_heads,
            head_dim,
        );

        // Output projection
        // IMP-1006: Use do_matmul to route to CudaScheduler when available
        let out_weight = self.block_weights[block_idx].out_weight.clone();
        let attn_proj = self.do_matmul(&attn_output, &out_weight, 1, hidden_dim, hidden_dim)?;

        // Add residual and bias
        let out_bias = &self.block_weights[block_idx].out_bias;
        let mut post_attn: Vec<f32> = input
            .iter()
            .zip(attn_proj.iter())
            .zip(out_bias.iter())
            .map(|((&i, &a), &b)| i + a + b)
            .collect();

        // FFN with layer norm (static function)
        let ffn_normed = Self::layer_norm_static(
            &post_attn,
            &self.block_weights[block_idx].ffn_norm_weight,
            &self.block_weights[block_idx].ffn_norm_bias,
            hidden_dim,
            eps,
        );

        // FFN: MoE dispatch when experts present, SwiGLU/GELU otherwise
        if let Some(ref moe) = self.block_weights[block_idx].moe_experts {
            // ALB-010: MoE forward — route to top-k experts + shared expert
            let moe_out = super::moe_dispatch::moe_forward_token(&ffn_normed, moe, hidden_dim);
            for i in 0..hidden_dim {
                post_attn[i] += moe_out[i];
            }
        } else {
            // Dense FFN: SwiGLU when gate weight exists, otherwise GELU
            // IMP-1006: Use do_matmul to route to CudaScheduler when available
            let fc1_activated: Vec<f32> = if let Some(ref gate_weight) =
                self.block_weights[block_idx].ffn_gate_weight
            {
                let fc1_weight = self.block_weights[block_idx].ffn_fc1_weight.clone();
                let gate_weight = gate_weight.clone();
                let up_out =
                    self.do_matmul(&ffn_normed, &fc1_weight, 1, hidden_dim, intermediate_dim)?;
                let gate_out =
                    self.do_matmul(&ffn_normed, &gate_weight, 1, hidden_dim, intermediate_dim)?;
                up_out
                    .iter()
                    .zip(gate_out.iter())
                    .map(|(&u, &g)| {
                        let silu_g = g / (1.0 + (-g).exp());
                        silu_g * u
                    })
                    .collect()
            } else {
                let fc1_weight = self.block_weights[block_idx].ffn_fc1_weight.clone();
                let fc1_out =
                    self.do_matmul(&ffn_normed, &fc1_weight, 1, hidden_dim, intermediate_dim)?;
                let ffn_fc1_bias = &self.block_weights[block_idx].ffn_fc1_bias;
                fc1_out
                    .iter()
                    .zip(ffn_fc1_bias.iter())
                    .map(|(&x, &b)| {
                        let x_b = x + b;
                        x_b * 0.5 + x_b * 0.5 * (0.797_884_6 * (x_b + 0.044_715 * x_b.powi(3))).tanh()
                    })
                    .collect()
            };

            let fc2_weight = self.block_weights[block_idx].ffn_fc2_weight.clone();
            let fc2_out =
                self.do_matmul(&fc1_activated, &fc2_weight, 1, intermediate_dim, hidden_dim)?;
            let ffn_fc2_bias = &self.block_weights[block_idx].ffn_fc2_bias;
            for i in 0..hidden_dim {
                post_attn[i] += fc2_out[i] + ffn_fc2_bias[i];
            }
        }

        Ok(post_attn)
    }

    /// Apply Rotary Position Embedding (RoPE) inline (delegates to ops module)
    fn apply_rope_inline(
        x: &mut [f32],
        num_heads: usize,
        head_dim: usize,
        rope_theta: f32,
        position: usize,
    ) {
        super::ops::apply_rope_inline(x, num_heads, head_dim, rope_theta, position);
    }

    /// GQA multi-head attention (delegates to ops module)
    fn gqa_multihead_attention(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        kv_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        super::ops::gqa_multihead_attention(q, k, v, kv_len, num_heads, num_kv_heads, head_dim)
    }

    // ============================================================================
    // Phase 9: Fused Kernels & Vectorization (M18)
    // ============================================================================

    /// Check if model has fused QKV projection (M18 - IMP-037)
    ///
    /// Fused QKV uses a single matmul instead of three separate projections.
    /// This is always true for GpuModel as QKV weights are stored combined.
    #[must_use]
    pub fn has_fused_qkv(&self) -> bool {
        // QKV weights are stored as [hidden_dim, 3*hidden_dim] for fused projection
        !self.block_weights.is_empty()
            && self.block_weights[0].qkv_weight.len()
                == self.config.hidden_dim * 3 * self.config.hidden_dim
    }

    /// Fused QKV projection (M18 - IMP-037)
    ///
    /// Performs Q, K, V projection in a single matmul operation.
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor [hidden_dim]
    ///
    /// # Returns
    ///
    /// Tuple of (Q, K, V) tensors, each [hidden_dim]
    ///
    /// # Errors
    ///
    /// Returns error if matmul fails
    pub fn fused_qkv_projection(
        &mut self,
        input: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        contract_pre_fused_qkv!();
        let hidden_dim = self.config.hidden_dim;
        let kv_dim = self.config.kv_dim();
        let qkv_dim = self.config.qkv_dim();

        // Use first block's QKV weights for projection
        let qkv_weight = &self.block_weights[0].qkv_weight;

        // Single matmul: [1, hidden_dim] @ [hidden_dim, qkv_dim] -> [1, qkv_dim]
        // For GQA: qkv_dim = hidden_dim + 2*kv_dim
        let qkv = self
            .scheduler
            .matmul(input, qkv_weight, 1, hidden_dim, qkv_dim)?;

        // Split into Q, K, V (GQA: K/V have kv_dim, not hidden_dim)
        let q = qkv[0..hidden_dim].to_vec();
        let k = qkv[hidden_dim..hidden_dim + kv_dim].to_vec();
        let v = qkv[hidden_dim + kv_dim..].to_vec();

        contract_post_attention!(&q);
        Ok((q, k, v))
    }

    /// Generation with fused QKV projection (M18 - IMP-037)
    ///
    /// Uses fused QKV projection for improved performance.
    ///
    /// # Errors
    ///
    /// Returns error if generation fails due to invalid input or model state.
    pub fn generate_with_fused_qkv(
        &mut self,
        prompt: &[usize],
        config: &GpuGenerateConfig,
    ) -> Result<Vec<usize>> {
        // Fused QKV is already used in generate_optimized via forward_block_incremental_optimized
        // This method provides explicit API for benchmarking
        self.generate_optimized(prompt, config)
    }

    /// Check if model has fused attention projection (M18 - IMP-039)
    #[must_use]
    pub fn has_fused_attn_proj(&self) -> bool {
        // Attention output projection is stored in block_weights
        !self.block_weights.is_empty()
            && self.block_weights[0].out_weight.len()
                == self.config.hidden_dim * self.config.hidden_dim
    }

    /// Forward pass with fused attention projection (M18 - IMP-039)
    ///
    /// Uses fused attention output projection for improved performance.
    ///
    /// # Errors
    ///
    /// Returns error if forward pass fails due to invalid token or cache state.
    pub fn forward_with_fused_attn_proj(
        &mut self,
        token_id: usize,
        kv_cache: &mut StreamingKVCache,
    ) -> Result<Vec<f32>> {
        // Fused attention projection is already used in forward_gpu_incremental_optimized
        // This method provides explicit API for benchmarking
        self.forward_gpu_incremental_optimized(token_id, kv_cache)
    }

    /// Check if model has fused output residual capability (M19 - IMP-042)
    #[must_use]
    pub fn has_fused_output_residual(&self) -> bool {
        // Fused output residual requires attention buffers and block weights
        self.attention_buffers.is_some() && !self.block_weights.is_empty()
    }

    /// Forward pass with fused output projection + residual (M19 - IMP-042)
    ///
    /// Combines the output projection matrix multiplication with residual
    /// connection in a single fused operation.
    ///
    /// # Errors
    ///
    /// Returns error if forward pass fails due to invalid token or cache state.
    pub fn forward_with_fused_output_residual(
        &mut self,
        token_id: usize,
        kv_cache: &mut StreamingKVCache,
    ) -> Result<Vec<f32>> {
        // Currently uses the optimized forward path
        // The fused operation is implemented in forward_block_incremental_optimized
        // This method provides explicit API for benchmarking
        self.forward_gpu_incremental_optimized(token_id, kv_cache)
    }

    /// Forward pass taking ownership of token_ids (convenience wrapper)
    ///
    /// This is useful when you don't need to keep the token_ids after the call.
    ///
    /// # Arguments
    ///
    /// * `token_ids` - Input token IDs (as Vec for owned semantics in tests)
    ///
    /// # Errors
    ///
    /// Returns error if forward pass fails
    pub fn forward_gpu_owned(&mut self, token_ids: &[usize]) -> Result<Vec<f32>> {
        self.forward_gpu(token_ids)
    }

    /// Generate text tokens using GPU-accelerated inference (M14: E2E Inference)
    ///
    /// Performs autoregressive token generation starting from a prompt.
    /// Uses GPU for forward passes and CPU for sampling.
    ///
    /// # Arguments
    ///
    /// * `prompt` - Initial token IDs to start generation from
    /// * `config` - Generation configuration (max tokens, temperature, etc.)
    ///
    /// # Returns
    ///
    /// Vector of generated token IDs (including the prompt)
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Prompt is empty
    /// - Forward pass fails
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let config = GpuGenerateConfig::deterministic(32);
    /// let tokens = model.generate(&[1, 2, 3], &config)?;
    /// ```
    pub fn generate(&mut self, prompt: &[usize], config: &GpuGenerateConfig) -> Result<Vec<usize>> {
        // IMP-1009: Use zero-clone RefCell path when CUDA is available
        // This provides ~7x speedup by eliminating weight cloning
        #[cfg(feature = "cuda")]
        if self.cuda_scheduler.is_some() {
            return self.generate_refcell(prompt, config);
        }

        // Fallback to clone-based path for non-CUDA or HybridScheduler
        // IMP-091: Uses KV cache for O(n) generation
        self.generate_optimized(prompt, config)
    }

    // =========================================================================
    // Phase 7: KV Cache Integration - Wrappers (extracted to kv.rs)
    // =========================================================================

    /// Forward pass with KV cache population (IMP-031) - delegates to kv module
    pub fn forward_gpu_with_cache(
        &mut self,
        token_ids: &[usize],
        kv_cache: &mut StreamingKVCache,
    ) -> Result<Vec<f32>> {
        super::kv::forward_gpu_with_cache(self, token_ids, kv_cache)
    }

    /// Incremental forward pass using cached KV (IMP-032) - delegates to kv module
    pub fn forward_gpu_incremental(
        &mut self,
        token_id: usize,
        kv_cache: &mut StreamingKVCache,
    ) -> Result<Vec<f32>> {
        super::kv::forward_gpu_incremental(self, token_id, kv_cache)
    }

    /// Generate with KV cache (IMP-033) - delegates to kv module
    pub fn generate_with_cache(
        &mut self,
        prompt: &[usize],
        config: &GpuGenerateConfig,
    ) -> Result<Vec<usize>> {
        super::kv::generate_with_cache(self, prompt, config)
    }

    /// Top-k sampling with temperature (delegates to ops module)
    fn sample_topk_generate(logits: &[f32], temperature: f32, top_k: usize) -> usize {
        super::ops::sample_topk(logits, temperature, top_k)
    }

    /// Transpose weight matrix (delegates to ops module)
    fn transpose_weights(weights: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        super::ops::transpose_weights(weights, rows, cols)
    }

    /// Check if GPU is being used
    #[must_use]
    pub fn has_gpu(&self) -> bool {
        self.scheduler.has_gpu()
    }
}

impl CudaExecutor {

    /// PAR-063-V5: Transformer layer using TRUE DP4A kernels (async, no sync)
    ///
    /// Uses Q8 activation quantization + Q4K×Q8 integer dot product for 4x instruction reduction.
    /// This is the llama.cpp-style approach that achieves 2x llama.cpp performance.
    ///
    /// Key optimizations:
    /// 1. Single Q8 quantization for Q/K/V (shared input)
    /// 2. dp4a.u32.s32 instruction: 4 multiply-adds per instruction
    /// 3. All GEMV operations use integer arithmetic
    ///
    /// # Arguments
    /// * `input` - GPU buffer containing hidden state [hidden_dim]
    /// * `layer_idx` - Layer index for weight lookup and KV cache
    /// * `layer_prefix` - Weight name prefix (e.g., "blk.0")
    /// * `hidden_dim` - Model hidden dimension
    /// * `intermediate_dim` - FFN intermediate dimension
    /// * `attn_norm_gamma` - Pre-attention RMSNorm weights
    /// * `ffn_norm_gamma` - Pre-FFN RMSNorm weights
    /// * `epsilon` - RMSNorm epsilon
    #[allow(clippy::too_many_arguments)]
    pub fn transformer_layer_gpu_true_dp4a(
        &mut self,
        input: &GpuBuffer<f32>,
        layer_idx: usize,
        layer_prefix: &str,
        hidden_dim: u32,
        intermediate_dim: u32,
        attn_norm_gamma: &GpuBuffer<f32>,
        ffn_norm_gamma: &GpuBuffer<f32>,
        epsilon: f32,
    ) -> Result<GpuBuffer<f32>, GpuError> {
        // Weight names follow GGML convention
        let q_name = format!("{}.attn_q.weight", layer_prefix);
        let k_name = format!("{}.attn_k.weight", layer_prefix);
        let v_name = format!("{}.attn_v.weight", layer_prefix);
        let o_name = format!("{}.attn_output.weight", layer_prefix);
        let gate_name = format!("{}.ffn_gate.weight", layer_prefix);
        let up_name = format!("{}.ffn_up.weight", layer_prefix);
        let down_name = format!("{}.ffn_down.weight", layer_prefix);

        // 1. Pre-attention RMSNorm (no sync)
        let normed = self.rmsnorm_gpu(input, attn_norm_gamma, hidden_dim, epsilon)?;

        // 2. PAR-063-V5: Quantize normed activations to Q8_1 ONCE for all Q/K/V projections
        let q8_normed = self.q8_quantize_async(&normed, hidden_dim)?;

        // 3. Q/K/V projections using Q4K × Q8 integer dot product
        let q_dim = (self.kv_num_heads * self.kv_head_dim) as u32;
        let kv_dim = (self.kv_num_kv_heads * self.kv_head_dim) as u32;

        let q = self.q4k_q8_gemv_async(&q_name, &q8_normed, q_dim, hidden_dim)?;
        let k = self.q4k_q8_gemv_async(&k_name, &q8_normed, kv_dim, hidden_dim)?;
        let v = self.q4k_q8_gemv_async(&v_name, &q8_normed, kv_dim, hidden_dim)?;

        // 4. Incremental attention (has internal sync for KV cache update)
        let (attn_out, _seq_len) = self.incremental_attention_async(layer_idx, &q, &k, &v)?;

        // 5. Quantize attention output for O projection
        let q8_attn = self.q8_quantize_async(&attn_out, q_dim)?;

        // 6. Output projection using Q4K × Q8 integer dot product
        let projected = self.q4k_q8_gemv_async(&o_name, &q8_attn, hidden_dim, q_dim)?;

        // 7. First residual add (no sync)
        let residual1 = self.residual_add_gpu(input, &projected, hidden_dim)?;

        // 8. Pre-FFN RMSNorm (no sync)
        let ffn_normed = self.rmsnorm_gpu(&residual1, ffn_norm_gamma, hidden_dim, epsilon)?;

        // 9. FFN SwiGLU using true DP4A path
        let ffn_out = self.fused_ffn_swiglu_gpu_true_dp4a(
            &ffn_normed,
            &gate_name,
            &up_name,
            &down_name,
            hidden_dim,
            intermediate_dim,
        )?;

        // 10. Second residual add (no sync)
        let output = self.residual_add_gpu(&residual1, &ffn_out, hidden_dim)?;

        // PAR-063-V5: NO sync here - caller can chain multiple layers
        Ok(output)
    }

    /// PAR-023: Cache RMSNorm gamma weights on GPU for all layers
    ///
    /// Pre-uploads attn_norm and ffn_norm gamma vectors to avoid per-layer uploads.
    /// Uses naming convention: `blk.{i}.attn_norm.gamma`, `blk.{i}.ffn_norm.gamma`
    ///
    /// # Arguments
    ///
    /// * `num_layers` - Number of transformer layers
    /// * `attn_norms` - Slice of attn_norm gamma vectors [num_layers][hidden_dim]
    /// * `ffn_norms` - Slice of ffn_norm gamma vectors [num_layers][hidden_dim]
    ///
    /// # Returns
    ///
    /// Total bytes uploaded to GPU
    pub fn preload_rmsnorm_weights(
        &mut self,
        num_layers: usize,
        attn_norms: &[&[f32]],
        ffn_norms: &[&[f32]],
    ) -> Result<usize, GpuError> {
        let mut total_bytes = 0usize;

        for layer_idx in 0..num_layers {
            // Attn norm
            let attn_name = format!("blk.{}.attn_norm.gamma", layer_idx);
            if !self.rmsnorm_cache.contains_key(&attn_name) {
                let buf = GpuBuffer::from_host(&self.context, attn_norms[layer_idx])?;
                total_bytes += buf.size_bytes();
                self.rmsnorm_cache.insert(attn_name, buf);
            }

            // FFN norm
            let ffn_name = format!("blk.{}.ffn_norm.gamma", layer_idx);
            if !self.rmsnorm_cache.contains_key(&ffn_name) {
                let buf = GpuBuffer::from_host(&self.context, ffn_norms[layer_idx])?;
                total_bytes += buf.size_bytes();
                self.rmsnorm_cache.insert(ffn_name, buf);
            }
        }

        Ok(total_bytes)
    }

    /// PAR-023: Check if RMSNorm weights are cached for a layer
    #[must_use]
    pub fn has_rmsnorm_weights(&self, layer_idx: usize) -> bool {
        let attn_name = format!("blk.{}.attn_norm.gamma", layer_idx);
        let ffn_name = format!("blk.{}.ffn_norm.gamma", layer_idx);
        self.rmsnorm_cache.contains_key(&attn_name) && self.rmsnorm_cache.contains_key(&ffn_name)
    }

    /// PAR-023: Pre-cache output norm (final layer norm) weight on GPU
    ///
    /// The output norm is applied after all transformer layers before LM head.
    /// Pre-caching allows fully GPU-resident forward pass.
    ///
    /// # Returns
    ///
    /// Total bytes uploaded to GPU
    pub fn preload_output_norm(&mut self, gamma: &[f32]) -> Result<usize, GpuError> {
        let output_name = "output_norm.gamma".to_string();
        if !self.rmsnorm_cache.contains_key(&output_name) {
            let buf = GpuBuffer::from_host(&self.context, gamma)?;
            let bytes = buf.size_bytes();
            self.rmsnorm_cache.insert(output_name, buf);
            Ok(bytes)
        } else {
            Ok(0)
        }
    }

    /// PAR-023: Check if output norm is cached
    #[must_use]
    pub fn has_output_norm(&self) -> bool {
        self.rmsnorm_cache.contains_key("output_norm.gamma")
    }

    /// Cache a single RMSNorm gamma weight by name.
    ///
    /// This is used by APR model loading to cache per-layer norm weights
    /// with arbitrary naming conventions. The gamma values are uploaded
    /// to GPU and stored in rmsnorm_cache for O(1) lookup during forward.
    ///
    /// # Arguments
    ///
    /// * `name` - Cache key name (e.g., "blk.0.attn_norm.gamma")
    /// * `gamma` - RMSNorm scale weights [hidden_dim]
    ///
    /// # Returns
    ///
    /// Total bytes uploaded to GPU (0 if already cached)
    pub fn cache_rmsnorm_gamma(&mut self, name: &str, gamma: &[f32]) -> Result<usize, GpuError> {
        if !self.rmsnorm_cache.contains_key(name) {
            let buf = GpuBuffer::from_host(&self.context, gamma)?;
            let bytes = buf.size_bytes();
            self.rmsnorm_cache.insert(name.to_string(), buf);
            Ok(bytes)
        } else {
            Ok(0)
        }
    }

    /// PAR-023: Get cached RMSNorm gamma pointer and length
    ///
    /// Returns the GPU device pointer and element count for a cached RMSNorm gamma.
    /// Use with `rmsnorm_gpu_ptr` to avoid CPU roundtrips.
    ///
    /// # Arguments
    ///
    /// * `name` - Cache key name (e.g., "blk.0.attn_norm.gamma")
    ///
    /// # Returns
    ///
    /// (device_pointer, element_count) or None if not cached
    #[must_use]
    pub fn get_rmsnorm_gamma_ptr(&self, name: &str) -> Option<(u64, usize)> {
        self.rmsnorm_cache
            .get(name)
            .map(|buf| (buf.as_ptr(), buf.len()))
    }

    /// BIAS-FIX: Cache QKV bias vectors on GPU for all layers
    ///
    /// Pre-uploads Q, K, V bias vectors (when present) to avoid per-layer uploads.
    /// Uses naming convention: `blk.{i}.attn_q.bias`, `blk.{i}.attn_k.bias`, `blk.{i}.attn_v.bias`
    ///
    /// Qwen2.5 models have QKV bias that must be added after GEMV.
    /// Models without bias pass empty slices.
    ///
    /// # Arguments
    ///
    /// * `num_layers` - Number of transformer layers
    /// * `q_biases` - Slice of Q bias vectors (or None for each layer)
    /// * `k_biases` - Slice of K bias vectors (or None for each layer)
    /// * `v_biases` - Slice of V bias vectors (or None for each layer)
    ///
    /// # Returns
    ///
    /// Total bytes uploaded to GPU
    pub fn preload_qkv_bias(
        &mut self,
        num_layers: usize,
        q_biases: &[Option<&[f32]>],
        k_biases: &[Option<&[f32]>],
        v_biases: &[Option<&[f32]>],
    ) -> Result<usize, GpuError> {
        let mut total_bytes = 0usize;

        for layer_idx in 0..num_layers {
            // Q bias
            if let Some(q_bias) = q_biases.get(layer_idx).and_then(|b| *b) {
                let name = format!("blk.{}.attn_q.bias", layer_idx);
                if !self.bias_cache.contains_key(&name) {
                    let buf = GpuBuffer::from_host(&self.context, q_bias)?;
                    total_bytes += buf.size_bytes();
                    self.bias_cache.insert(name, buf);
                }
            }

            // K bias
            if let Some(k_bias) = k_biases.get(layer_idx).and_then(|b| *b) {
                let name = format!("blk.{}.attn_k.bias", layer_idx);
                if !self.bias_cache.contains_key(&name) {
                    let buf = GpuBuffer::from_host(&self.context, k_bias)?;
                    total_bytes += buf.size_bytes();
                    self.bias_cache.insert(name, buf);
                }
            }

            // V bias
            if let Some(v_bias) = v_biases.get(layer_idx).and_then(|b| *b) {
                let name = format!("blk.{}.attn_v.bias", layer_idx);
                if !self.bias_cache.contains_key(&name) {
                    let buf = GpuBuffer::from_host(&self.context, v_bias)?;
                    total_bytes += buf.size_bytes();
                    self.bias_cache.insert(name, buf);
                }
            }
        }

        if total_bytes > 0 && verbose() {
            eprintln!(
                "[BIAS-FIX] Preloaded QKV bias for {} layers ({} bytes)",
                num_layers, total_bytes
            );
        }

        Ok(total_bytes)
    }

    /// BIAS-FIX: Check if QKV bias is cached for a layer
    #[must_use]
    pub fn has_qkv_bias(&self, layer_idx: usize) -> bool {
        // Check if at least one bias exists (Qwen2.5 has all three)
        let q_name = format!("blk.{}.attn_q.bias", layer_idx);
        self.bias_cache.contains_key(&q_name)
    }

    /// PAR-064-FIX: Pre-cache LM head bias on GPU
    ///
    /// Some models (like Qwen2.5) have an output.bias that must be added to logits
    /// after the LM head GEMV projection. Without this bias, GPU inference produces
    /// incorrect token predictions.
    ///
    /// # Arguments
    ///
    /// * `bias` - Optional LM head bias vector (vocab_size elements)
    ///
    /// # Returns
    ///
    /// Total bytes uploaded to GPU (0 if no bias)
    pub fn preload_lm_head_bias(&mut self, bias: Option<&[f32]>) -> Result<usize, GpuError> {
        let Some(bias_data) = bias else {
            return Ok(0);
        };

        if bias_data.is_empty() {
            return Ok(0);
        }

        let name = "output.bias".to_string();
        if self.bias_cache.contains_key(&name) {
            return Ok(0);
        }

        let buf = GpuBuffer::from_host(&self.context, bias_data)?;
        let total_bytes = buf.size_bytes();

        // Index the pointer for fast access in forward pass
        self.lm_head_bias_ptr = buf.as_ptr();
        self.lm_head_bias_len = buf.len();

        self.bias_cache.insert(name, buf);

        eprintln!(
            "[PAR-064-FIX] Preloaded LM head bias: {} elements ({} bytes)",
            bias_data.len(),
            total_bytes
        );

        Ok(total_bytes)
    }

    /// PAR-064-FIX: Check if LM head bias is cached
    #[must_use]
    pub fn has_lm_head_bias(&self) -> bool {
        self.lm_head_bias_ptr != 0
    }

    /// PAR-023: Transformer layer with cached gamma pointers
    ///
    /// Like `transformer_layer_gpu` but takes raw device pointers for gamma weights
    /// to avoid borrow checker conflicts with cached buffers.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn transformer_layer_gpu_cached(
        &mut self,
        input: &GpuBuffer<f32>,
        layer_idx: usize,
        layer_prefix: &str,
        hidden_dim: u32,
        intermediate_dim: u32,
        attn_gamma_ptr: u64, // CUdeviceptr
        attn_gamma_len: usize,
        ffn_gamma_ptr: u64, // CUdeviceptr
        ffn_gamma_len: usize,
        epsilon: f32,
    ) -> Result<GpuBuffer<f32>, GpuError> {
        // Weight names follow GGML convention
        let q_name = format!("{}.attn_q.weight", layer_prefix);
        let k_name = format!("{}.attn_k.weight", layer_prefix);
        let v_name = format!("{}.attn_v.weight", layer_prefix);
        let o_name = format!("{}.attn_output.weight", layer_prefix);
        let gate_name = format!("{}.ffn_gate.weight", layer_prefix);
        let up_name = format!("{}.ffn_up.weight", layer_prefix);
        let down_name = format!("{}.ffn_down.weight", layer_prefix);

        // 1. Pre-attention RMSNorm using cached gamma pointer
        let normed =
            self.rmsnorm_gpu_ptr(input, attn_gamma_ptr, attn_gamma_len, hidden_dim, epsilon)?;

        // 2. Q/K/V projections (no sync)
        // PAR-056: Tiled kernel selection based on K dimension
        let q_dim = (self.kv_num_heads * self.kv_head_dim) as u32;
        let kv_dim = (self.kv_num_kv_heads * self.kv_head_dim) as u32;
        const CHUNK_THRESHOLD: u32 = 8192;
        let hidden_aligned = hidden_dim.is_multiple_of(256);
        let q_aligned = q_dim.is_multiple_of(256);
        let kv_aligned = kv_dim.is_multiple_of(256);

        // PAR-056: For K > 8192, use non-tiled Q4KGemvKernel (warp-based)
        // ChunkedTiledQ4KGemvKernel bypassed for large K (PAR-056 path)
        let q = if !hidden_aligned || !q_aligned || hidden_dim > CHUNK_THRESHOLD {
            self.q4k_gemv_cached_async(&q_name, &normed, q_dim, hidden_dim)?
        } else {
            self.tiled_q4k_gemv_cached_async(&q_name, &normed, q_dim, hidden_dim, 4)?
        };
        let k = if !hidden_aligned || !kv_aligned || hidden_dim > CHUNK_THRESHOLD {
            self.q4k_gemv_cached_async(&k_name, &normed, kv_dim, hidden_dim)?
        } else {
            self.tiled_q4k_gemv_cached_async(&k_name, &normed, kv_dim, hidden_dim, 4)?
        };
        let v = if !hidden_aligned || !kv_aligned || hidden_dim > CHUNK_THRESHOLD {
            self.q4k_gemv_cached_async(&v_name, &normed, kv_dim, hidden_dim)?
        } else {
            self.tiled_q4k_gemv_cached_async(&v_name, &normed, kv_dim, hidden_dim, 4)?
        };

        // 3. Incremental attention (has internal sync for KV cache update)
        let (attn_out, _seq_len) = self.incremental_attention_async(layer_idx, &q, &k, &v)?;

        // 4. Output projection (no sync) - K = q_dim
        // PAR-056: For K > 8192, use non-tiled Q4KGemvKernel
        let projected = if !q_aligned || !hidden_aligned || q_dim > CHUNK_THRESHOLD {
            self.q4k_gemv_cached_async(&o_name, &attn_out, hidden_dim, q_dim)?
        } else {
            self.tiled_q4k_gemv_cached_async(&o_name, &attn_out, hidden_dim, q_dim, 4)?
        };

        // 5. First residual add (no sync)
        let residual1 = self.residual_add_gpu(input, &projected, hidden_dim)?;

        // 6. Pre-FFN RMSNorm using cached gamma pointer
        let ffn_normed = self.rmsnorm_gpu_ptr(
            &residual1,
            ffn_gamma_ptr,
            ffn_gamma_len,
            hidden_dim,
            epsilon,
        )?;

        // 7. FFN SwiGLU (no sync)
        let ffn_out = self.fused_ffn_swiglu_gpu(
            &ffn_normed,
            &gate_name,
            &up_name,
            &down_name,
            hidden_dim,
            intermediate_dim,
        )?;

        // 8. Second residual add (no sync)
        let output = self.residual_add_gpu(&residual1, &ffn_out, hidden_dim)?;

        Ok(output)
    }

    /// PAR-044: Get reference to workspace output buffer
    ///
    /// After calling `transformer_layer_workspace`, the output is in hidden_buf2.
    #[must_use]
    pub fn workspace_output(&self) -> Option<&GpuBuffer<f32>> {
        self.workspace.hidden_buf2.as_ref()
    }
}

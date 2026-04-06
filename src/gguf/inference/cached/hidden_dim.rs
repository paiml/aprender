impl OwnedQuantizedModelCachedSync {

    /// Adaptive multihead attention for production serving (IMP-121)
    ///
    /// Thread-safe multi-head attention that automatically selects backend.
    ///
    /// # Arguments
    /// * `q` - Query tensor [seq_len, hidden_dim]
    /// * `k` - Key tensor [seq_len, hidden_dim]
    /// * `v` - Value tensor [seq_len, hidden_dim]
    /// * `seq_len` - Sequence length
    ///
    /// # Returns
    /// Output tensor [seq_len, hidden_dim]
    pub fn adaptive_multihead_attention(
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

        let mut attn_output = vec![0.0f32; num_heads * seq_len * head_dim];

        for h in 0..num_heads {
            let head_offset = h * seq_len * head_dim;
            let q_head = &q_reshaped[head_offset..head_offset + seq_len * head_dim];
            let k_head = &k_reshaped[head_offset..head_offset + seq_len * head_dim];
            let v_head = &v_reshaped[head_offset..head_offset + seq_len * head_dim];

            let head_output =
                self.adaptive_fused_attention(q_head, k_head, v_head, seq_len, head_dim, scale)?;

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

    /// Warmup GPU weight cache for batch inference (PARITY-019)
    ///
    /// Pre-dequantizes all FFN weights to f32 for GPU GEMM operations.
    /// Call this once at server startup to avoid dequantization during inference.
    ///
    /// # Memory Usage
    /// - phi-2 (32 layers): ~6.4 GB
    /// - Per layer: 2 × hidden_dim × intermediate_dim × 4 bytes
    ///
    /// # Returns
    /// - Total memory allocated in bytes
    /// - Number of layers cached
    ///
    /// # Errors
    /// Returns error if dequantization fails
    pub fn warmup_gpu_cache(&self) -> Result<(usize, usize)> {
        let config = &self.model.config;
        let hidden_dim = config.hidden_dim;
        let intermediate_dim = config.intermediate_dim;
        let num_layers = self.model.layers.len();

        // Create cache with model dimensions
        let cache = DequantizedWeightCache::new(hidden_dim, intermediate_dim, num_layers);

        // Dequantize each layer's FFN weights
        // Note: warmup closure can't return Result, so we use unwrap_or_default
        // for robustness. In production, use warmup_gpu_cache_checked() for error handling.
        cache.warmup(|layer_idx| {
            let layer = &self.model.layers[layer_idx];

            // Dequantize using model's dequantize_weight method
            let up = self
                .model
                .dequantize_weight(&layer.ffn_up_weight)
                .unwrap_or_default();
            let down = self
                .model
                .dequantize_weight(&layer.ffn_down_weight)
                .unwrap_or_default();

            (up, down)
        });

        let memory_bytes = cache.memory_bytes();
        let cached_count = cache.cached_count();

        // Store in the cache field
        let mut cache_guard =
            self.dequant_cache
                .write()
                .map_err(|_| RealizarError::UnsupportedOperation {
                    operation: "warmup_gpu_cache".to_string(),
                    reason: "Cache lock poisoned".to_string(),
                })?;
        *cache_guard = Some(cache);

        Ok((memory_bytes, cached_count))
    }

    /// Check if GPU cache is warmed up
    pub fn is_gpu_cache_warm(&self) -> bool {
        self.dequant_cache
            .read()
            .map(|guard| guard.is_some())
            .unwrap_or(false)
    }

    /// Get GPU cache memory usage in bytes
    pub fn gpu_cache_memory(&self) -> usize {
        self.dequant_cache
            .read()
            .ok()
            .and_then(|guard| guard.as_ref().map(DequantizedWeightCache::memory_bytes))
            .unwrap_or(0)
    }

    /// Get dequantized weights for a layer (for GPU batch FFN)
    ///
    /// Returns None if cache not warmed up or layer not found.
    pub fn get_dequantized_ffn_weights(&self, layer_idx: usize) -> Option<DequantizedFFNWeights> {
        self.dequant_cache
            .read()
            .ok()
            .and_then(|guard| guard.as_ref().and_then(|c| c.get(layer_idx)))
    }

    /// Batch FFN forward pass using GPU (PARITY-019)
    ///
    /// Processes multiple tokens in parallel using GPU GEMM.
    /// Requires cache to be warmed up via `warmup_gpu_cache()`.
    ///
    /// # Arguments
    /// * `hidden_states` - Input tensor [batch_size × hidden_dim]
    /// * `layer_idx` - Layer index for weight lookup
    ///
    /// # Returns
    /// Output tensor [batch_size × hidden_dim]
    ///
    /// # Errors
    /// Returns error if cache not warmed or GPU operations fail
    /// PARITY-103: Batch FFN using CUDA when available
    ///
    /// Uses CudaScheduler first (no buffer limits), falls back to HybridScheduler (wgpu).
    /// This bypasses the wgpu 256MB buffer limit that was blocking GPU batch inference.
    pub fn batch_ffn_gpu(&self, hidden_states: &[f32], layer_idx: usize) -> Result<Vec<f32>> {
        let config = &self.model.config;
        let hidden_dim = config.hidden_dim;
        let intermediate_dim = config.intermediate_dim;
        let batch_size = hidden_states.len() / hidden_dim;

        if batch_size == 0 {
            return Err(RealizarError::UnsupportedOperation {
                operation: "batch_ffn_gpu".to_string(),
                reason: "Empty batch".to_string(),
            });
        }

        // Get cached weights
        let weights = self.get_dequantized_ffn_weights(layer_idx).ok_or_else(|| {
            RealizarError::UnsupportedOperation {
                operation: "batch_ffn_gpu".to_string(),
                reason: format!(
                    "Layer {} not cached. Call warmup_gpu_cache() first.",
                    layer_idx
                ),
            }
        })?;

        // PARITY-103: Up projection preferring CUDA
        let mut intermediate = self.batch_matmul_gpu_prefer_cuda(
            hidden_states,
            &weights.up,
            batch_size,
            hidden_dim,
            intermediate_dim,
        )?;

        // Add up bias if present
        if let Some(ref bias) = weights.up_bias {
            for b in 0..batch_size {
                for i in 0..intermediate_dim {
                    intermediate[b * intermediate_dim + i] += bias[i];
                }
            }
        }

        // GELU activation (CPU - fused in future)
        for x in &mut intermediate {
            let x64 = *x as f64;
            *x = (x64
                * 0.5
                * (1.0 + (x64 * 0.797_884_560_8 * (1.0 + 0.044_715 * x64 * x64)).tanh()))
                as f32;
        }

        // PARITY-103: Down projection preferring CUDA
        let mut output = self.batch_matmul_gpu_prefer_cuda(
            &intermediate,
            &weights.down,
            batch_size,
            intermediate_dim,
            hidden_dim,
        )?;

        // Add down bias if present
        if let Some(ref bias) = weights.down_bias {
            for b in 0..batch_size {
                for i in 0..hidden_dim {
                    output[b * hidden_dim + i] += bias[i];
                }
            }
        }

        Ok(output)
    }

    /// PARITY-103: Batch QKV projection using CUDA when available
    ///
    /// Projects hidden states to Q, K, V for all requests in batch.
    /// [batch, hidden] @ [hidden, 3*hidden] = [batch, 3*hidden]
    ///
    /// Uses CudaScheduler first (no buffer limits), falls back to HybridScheduler (wgpu).
    ///
    /// # Arguments
    /// * `hidden_states` - Flattened hidden states [batch * hidden_dim]
    /// * `layer_idx` - Layer index for weight lookup
    ///
    /// # Returns
    /// Flattened QKV projections [batch * 3 * hidden_dim]
    #[cfg(feature = "gpu")]
    pub fn batch_qkv_projection_gpu(
        &self,
        hidden_states: &[f32],
        layer_idx: usize,
    ) -> Result<Vec<f32>> {
        let hidden_dim = self.model.config.hidden_dim;
        let batch_size = hidden_states.len() / hidden_dim;
        let qkv_dim = 3 * hidden_dim;

        if batch_size == 0 {
            return Ok(Vec::new());
        }

        let layer = &self.model.layers[layer_idx];

        // Dequantize QKV weight for GPU GEMM
        let qkv_weight = self.model.dequantize_qkv(&layer.qkv_weight)?;

        // PARITY-103: QKV projection preferring CUDA
        let mut qkv = self.batch_matmul_gpu_prefer_cuda(
            hidden_states,
            &qkv_weight,
            batch_size,
            hidden_dim,
            qkv_dim,
        )?;

        // Add bias if present
        if let Some(ref bias) = layer.qkv_bias {
            for b in 0..batch_size {
                for i in 0..qkv_dim {
                    qkv[b * qkv_dim + i] += bias[i];
                }
            }
        }

        Ok(qkv)
    }

    /// Batch attention output projection using GPU GEMM (PARITY-024)
    ///
    /// Projects attention outputs for all requests in batch.
    /// [batch, hidden] @ [hidden, hidden] = [batch, hidden]
    ///
    /// # Arguments
    /// * `attention_outputs` - Flattened attention outputs [batch * hidden_dim]
    /// * `layer_idx` - Layer index for weight lookup
    ///
    /// # Returns
    /// Flattened projected outputs [batch * hidden_dim]
    #[cfg(feature = "gpu")]
    pub fn batch_attention_output_gpu(
        &self,
        attention_outputs: &[f32],
        layer_idx: usize,
    ) -> Result<Vec<f32>> {
        let hidden_dim = self.model.config.hidden_dim;
        let batch_size = attention_outputs.len() / hidden_dim;

        if batch_size == 0 {
            return Ok(Vec::new());
        }

        let layer = &self.model.layers[layer_idx];

        // Dequantize output weight for GPU GEMM
        let output_weight = self.model.dequantize_weight(&layer.attn_output_weight)?;

        // PARITY-103: Output projection preferring CUDA (bypasses wgpu 256MB limit)
        // [batch, hidden] @ [hidden, hidden] = [batch, hidden]
        let mut output = self.batch_matmul_gpu_prefer_cuda(
            attention_outputs,
            &output_weight,
            batch_size,
            hidden_dim,
            hidden_dim,
        )?;

        // Add bias if present
        if let Some(ref bias) = layer.attn_output_bias {
            for b in 0..batch_size {
                for i in 0..hidden_dim {
                    output[b * hidden_dim + i] += bias[i];
                }
            }
        }

        Ok(output)
    }

    /// Batch LM head projection using GPU GEMM (PARITY-025)
    ///
    /// Projects hidden states to vocabulary logits for all requests in batch.
    /// [batch, hidden] @ [hidden, vocab] = [batch, vocab]
    ///
    /// # Arguments
    /// * `hidden_states` - Flattened normalized hidden states [batch * hidden_dim]
    ///
    /// # Returns
    /// Flattened logits [batch * vocab_size]
    #[cfg(feature = "gpu")]
    pub fn batch_lm_head_gpu(&self, hidden_states: &[f32]) -> Result<Vec<f32>> {
        let hidden_dim = self.model.config.hidden_dim;
        let vocab_size = self.model.config.vocab_size;
        let batch_size = hidden_states.len() / hidden_dim;

        if batch_size == 0 {
            return Ok(Vec::new());
        }

        // Dequantize LM head weight for GPU GEMM
        let lm_head_weight = self.model.dequantize_weight(&self.model.lm_head_weight)?;

        // PARITY-103: LM head projection preferring CUDA (bypasses wgpu 256MB limit)
        // [batch, hidden] @ [hidden, vocab] = [batch, vocab]
        let mut logits = self.batch_matmul_gpu_prefer_cuda(
            hidden_states,
            &lm_head_weight,
            batch_size,
            hidden_dim,
            vocab_size,
        )?;

        // Add bias if present
        if let Some(ref bias) = self.model.lm_head_bias {
            for b in 0..batch_size {
                for i in 0..vocab_size {
                    logits[b * vocab_size + i] += bias[i];
                }
            }
        }

        Ok(logits)
    }
}

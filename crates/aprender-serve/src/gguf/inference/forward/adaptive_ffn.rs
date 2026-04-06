impl OwnedQuantizedModel {

    /// Compute FFN layer norm + activation for adaptive forward pass (PMAT-260)
    ///
    /// Extracts the shared FFN computation (norm + SwiGLU/GELU + down projection)
    /// from `forward_single_with_cache_adaptive` to reduce complexity.
    /// Used by both CUDA and CPU paths.
    ///
    /// # Arguments
    /// * `hidden` - Current hidden state
    /// * `layer_idx` - Layer index
    /// * `use_rmsnorm` - Whether to use RMSNorm (LLaMA) or LayerNorm (phi-2)
    ///
    /// # Returns
    /// FFN output vector to be added as residual
    ///
    /// # Errors
    /// Returns error if tensor operations fail
    fn adaptive_layer_ffn(
        &self,
        hidden: &[f32],
        layer_idx: usize,
        use_rmsnorm: bool,
    ) -> Result<Vec<f32>> {
        let layer = &self.layers[layer_idx];

        // Pre-FFN layer norm (LLaMA uses separate ffn_norm with RMSNorm)
        let ffn_input = if let Some(ref ffn_norm) = layer.ffn_norm_weight {
            if use_rmsnorm {
                ops::rms_norm(hidden, ffn_norm, self.config.eps)
            } else {
                ops::layer_norm(
                    hidden,
                    ffn_norm,
                    layer.ffn_norm_bias.as_deref(),
                    self.config.eps,
                )
            }
        } else {
            hidden.to_vec()
        };

        // FFN with SwiGLU or GELU activation (contract-driven, GH-278)
        let ffn_activated = if self.config.constraints.has_gate_ffn() {
            // SwiGLU path
            let (mut ffn_gate, mut ffn_up) = if let Some(ref gate_weight) = layer.ffn_gate_weight {
                let ffn_up = self.fused_matmul(&ffn_input, &layer.ffn_up_weight)?;
                let ffn_gate = self.fused_matmul(&ffn_input, gate_weight)?;
                (ffn_gate, ffn_up)
            } else {
                // GH-306: Fused gate_up weight (Phi-3.5)
                let fused = self.fused_matmul(&ffn_input, &layer.ffn_up_weight)?;
                let half = fused.len() / 2;
                (fused[..half].to_vec(), fused[half..].to_vec())
            };
            if let Some(ref bias) = layer.ffn_up_bias {
                ops::add_bias(&mut ffn_up, bias);
            }
            if let Some(ref bias) = layer.ffn_gate_bias {
                ops::add_bias(&mut ffn_gate, bias);
            }

            ops::silu(&mut ffn_gate);
            for i in 0..ffn_gate.len() {
                ffn_gate[i] *= ffn_up[i];
            }
            ffn_gate
        } else {
            // GELU path
            let mut ffn_hidden = self.fused_matmul(&ffn_input, &layer.ffn_up_weight)?;
            if let Some(ref bias) = layer.ffn_up_bias {
                ops::add_bias(&mut ffn_hidden, bias);
            }
            ops::gelu(&mut ffn_hidden);
            ffn_hidden
        };

        // FFN down projection
        let mut ffn_output = self.fused_matmul(&ffn_activated, &layer.ffn_down_weight)?;
        if let Some(ref bias) = layer.ffn_down_bias {
            ops::add_bias(&mut ffn_output, bias);
        }

        Ok(ffn_output)
    }

    /// Expand V for all Q heads when cache is empty (first token GQA) (PMAT-260)
    ///
    /// When processing the first token, there is no cached K/V history, so we
    /// simply expand the V vector across all query heads using the GQA mapping.
    ///
    /// # Arguments
    /// * `v` - Value vector from current token projection
    /// * `num_heads` - Number of query attention heads
    /// * `num_kv_heads` - Number of key/value heads (GQA grouping)
    /// * `head_dim` - Dimension per attention head
    /// * `hidden_dim` - Total hidden dimension (`num_heads * head_dim`)
    fn expand_first_token_v(
        v: &[f32],
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        q_dim: usize,
    ) -> Vec<f32> {
        // GH-479: Output size is q_dim (= num_heads * head_dim), not hidden_dim
        let mut expanded_v = vec![0.0f32; q_dim];
        let q_per_kv = num_heads / num_kv_heads;
        for q_head in 0..num_heads {
            let kv_head = q_head / q_per_kv;
            let v_start = kv_head * head_dim;
            let out_start = q_head * head_dim;
            expanded_v[out_start..out_start + head_dim]
                .copy_from_slice(&v[v_start..v_start + head_dim]);
        }
        expanded_v
    }

    /// CUDA-accelerated layer attention for adaptive forward pass (PMAT-260)
    ///
    /// Handles QKV projection, RoPE, attention computation, output projection,
    /// and residual connections using the CUDA path with GPU dispatch metrics.
    ///
    /// # Arguments
    /// * `hidden` - Current hidden state (modified in-place with residual connections)
    /// * `layer` - Transformer layer weights
    /// * `layer_idx` - Layer index for cache access
    /// * `normed` - Pre-normed hidden state for QKV projection
    /// * `cache` - KV cache for incremental decoding
    /// * `position` - Current sequence position for RoPE
    /// * `use_rmsnorm` - Whether to use RMSNorm (LLaMA) or LayerNorm (phi-2)
    /// * `metrics` - Dispatch metrics tracker for GPU timing
    /// * `num_kv_heads` - Number of key/value heads
    /// * `head_dim` - Dimension per attention head
    /// * `kv_dim` - Key/value dimension (`num_kv_heads * head_dim`)
    /// * `hidden_dim` - Total hidden dimension
    ///
    /// # Errors
    /// Returns error if tensor operations fail
    #[cfg(feature = "gpu")]
    #[allow(dead_code)]
    fn adaptive_layer_attention_cuda(
        &self,
        hidden: &mut [f32],
        layer: &OwnedQuantizedLayer,
        layer_idx: usize,
        normed: &[f32],
        cache: &mut OwnedQuantizedKVCache,
        position: usize,
        use_rmsnorm: bool,
        metrics: &std::sync::Arc<DispatchMetrics>,
        num_kv_heads: usize,
        head_dim: usize,
        kv_dim: usize,
        hidden_dim: usize,
    ) -> Result<()> {
        // GH-479: q_dim may differ from hidden_dim (Qwen3 head_dim != hidden/heads)
        let q_dim = self.config.q_dim();

        let start = std::time::Instant::now();
        let qkv_result = self.qkv_matmul(normed, &layer.qkv_weight)?;
        metrics.record_gpu_dispatch();
        metrics.record_gpu_latency(start.elapsed());
        let mut qkv = qkv_result;
        if let Some(ref bias) = layer.qkv_bias {
            ops::add_bias(&mut qkv, bias);
        }

        // GH-479: Use q_dim (may differ from hidden_dim for Qwen3)
        let mut q = qkv[0..q_dim].to_vec();
        let mut k = qkv[q_dim..q_dim + kv_dim].to_vec();
        let v = qkv[q_dim + kv_dim..q_dim + 2 * kv_dim].to_vec();

        // GH-279: Per-head QK RMSNorm (Qwen3) — after bias, before RoPE
        if let Some(ref q_norm) = layer.attn_q_norm_weight {
            ops::apply_per_head_rms_norm(&mut q, q_norm, self.config.num_heads, self.config.eps);
        }
        if let Some(ref k_norm) = layer.attn_k_norm_weight {
            ops::apply_per_head_rms_norm(&mut k, k_norm, num_kv_heads, self.config.eps);
        }

        // GH-278: Skip RoPE for models with learned position embeddings (GPT-2)
        if self.config.constraints.uses_rope() {
            self.apply_rope(&mut q, position, self.config.num_heads);
            self.apply_rope(&mut k, position, num_kv_heads);
        }

        let k_cache = cache.get_k(layer_idx);
        let v_cache = cache.get_v(layer_idx);

        let attn_out = if k_cache.is_empty() {
            Self::expand_first_token_v(&v, self.config.num_heads, num_kv_heads, head_dim, q_dim)
        } else {
            let start = std::time::Instant::now();
            let result = self.adaptive_attention_with_cache(&q, k_cache, v_cache, &k, &v)?;
            metrics.record_gpu_dispatch();
            metrics.record_gpu_latency(start.elapsed());
            result
        };

        cache.append(layer_idx, &k, &v);

        let start = std::time::Instant::now();
        let mut attn_output = self.fused_matmul(&attn_out, &layer.attn_output_weight)?;
        metrics.record_gpu_dispatch();
        metrics.record_gpu_latency(start.elapsed());
        if let Some(ref bias) = layer.attn_output_bias {
            ops::add_bias(&mut attn_output, bias);
        }

        for i in 0..hidden_dim {
            hidden[i] += attn_output[i];
        }

        let ffn_output = self.adaptive_layer_ffn_cuda(hidden, layer_idx, use_rmsnorm, metrics)?;
        for i in 0..hidden_dim {
            hidden[i] += ffn_output[i];
        }

        Ok(())
    }

    /// CPU layer attention for adaptive forward pass (PMAT-260)
    ///
    /// Handles QKV projection, RoPE, attention computation with adaptive
    /// CPU/GPU dispatch based on cache length, output projection, and
    /// residual connections.
    ///
    /// # Arguments
    /// * `hidden` - Current hidden state (modified in-place with residual connections)
    /// * `layer` - Transformer layer weights
    /// * `layer_idx` - Layer index for cache access
    /// * `normed` - Pre-normed hidden state for QKV projection
    /// * `cache` - KV cache for incremental decoding
    /// * `position` - Current sequence position for RoPE
    /// * `use_rmsnorm` - Whether to use RMSNorm (LLaMA) or LayerNorm (phi-2)
    /// * `metrics` - Dispatch metrics tracker for CPU/GPU timing
    /// * `num_kv_heads` - Number of key/value heads
    /// * `head_dim` - Dimension per attention head
    /// * `kv_dim` - Key/value dimension (`num_kv_heads * head_dim`)
    /// * `hidden_dim` - Total hidden dimension
    ///
    /// # Errors
    /// Returns error if tensor operations fail
    #[cfg(feature = "gpu")]
    fn adaptive_layer_attention_cpu(
        &self,
        hidden: &mut [f32],
        layer: &OwnedQuantizedLayer,
        layer_idx: usize,
        normed: &[f32],
        cache: &mut OwnedQuantizedKVCache,
        position: usize,
        use_rmsnorm: bool,
        metrics: &std::sync::Arc<DispatchMetrics>,
        num_kv_heads: usize,
        head_dim: usize,
        kv_dim: usize,
        hidden_dim: usize,
    ) -> Result<()> {
        // GH-479: q_dim may differ from hidden_dim (Qwen3 head_dim != hidden/heads)
        let q_dim = self.config.q_dim();

        let mut qkv = self.qkv_matmul(normed, &layer.qkv_weight)?;
        if let Some(ref bias) = layer.qkv_bias {
            ops::add_bias(&mut qkv, bias);
        }

        // GH-479: Use q_dim (may differ from hidden_dim for Qwen3)
        let mut q = qkv[0..q_dim].to_vec();
        let mut k = qkv[q_dim..q_dim + kv_dim].to_vec();
        let v = qkv[q_dim + kv_dim..q_dim + 2 * kv_dim].to_vec();

        // GH-279: Per-head QK RMSNorm (Qwen3) — after bias, before RoPE
        if let Some(ref q_norm) = layer.attn_q_norm_weight {
            ops::apply_per_head_rms_norm(&mut q, q_norm, self.config.num_heads, self.config.eps);
        }
        if let Some(ref k_norm) = layer.attn_k_norm_weight {
            ops::apply_per_head_rms_norm(&mut k, k_norm, num_kv_heads, self.config.eps);
        }

        // GH-278: Skip RoPE for models with learned position embeddings (GPT-2)
        if self.config.constraints.uses_rope() {
            self.apply_rope(&mut q, position, self.config.num_heads);
            self.apply_rope(&mut k, position, num_kv_heads);
        }

        let k_cache = cache.get_k(layer_idx);
        let v_cache = cache.get_v(layer_idx);

        let attn_out = if k_cache.is_empty() {
            Self::expand_first_token_v(&v, self.config.num_heads, num_kv_heads, head_dim, q_dim)
        } else {
            let cache_len = k_cache.len() / kv_dim;
            const GPU_CACHE_LEN_THRESHOLD: usize = 64;
            if cache_len >= GPU_CACHE_LEN_THRESHOLD {
                let start = std::time::Instant::now();
                let result = self.adaptive_attention_with_cache(&q, k_cache, v_cache, &k, &v)?;
                metrics.record_gpu_dispatch();
                metrics.record_gpu_latency(start.elapsed());
                result
            } else {
                let start = std::time::Instant::now();
                let result = self.attention_with_cache_gqa(&q, k_cache, v_cache, &k, &v);
                metrics.record_cpu_dispatch();
                metrics.record_cpu_latency(start.elapsed());
                result
            }
        };

        cache.append(layer_idx, &k, &v);

        let mut attn_output = self.fused_matmul(&attn_out, &layer.attn_output_weight)?;
        if let Some(ref bias) = layer.attn_output_bias {
            ops::add_bias(&mut attn_output, bias);
        }

        for i in 0..hidden_dim {
            hidden[i] += attn_output[i];
        }

        let ffn_output = self.adaptive_layer_ffn(hidden, layer_idx, use_rmsnorm)?;
        for i in 0..hidden_dim {
            hidden[i] += ffn_output[i];
        }

        Ok(())
    }

    /// Forward pass with adaptive CPU/GPU attention selection (IMP-124)
    ///
    /// This variant of `forward_single_with_cache` uses `adaptive_attention_with_cache`
    /// to automatically select between CPU and GPU backends based on cache length.
    /// It also records dispatch decisions to the provided metrics tracker.
    ///
    /// # Arguments
    /// * `token_id` - Token to process
    /// * `cache` - KV cache for incremental decoding
    /// * `position` - Position in sequence
    /// * `metrics` - Dispatch metrics tracker for CPU/GPU decision recording
    ///
    /// # Returns
    /// Logits for next token prediction [vocab_size]
    ///
    /// # Errors
    /// Returns error if tensor operations fail
    #[cfg(feature = "gpu")]
    pub fn forward_single_with_cache_adaptive(
        &self,
        token_id: u32,
        cache: &mut OwnedQuantizedKVCache,
        position: usize,
        metrics: &std::sync::Arc<DispatchMetrics>,
    ) -> Result<Vec<f32>> {
        let hidden_dim = self.config.hidden_dim;

        // 1. Token embedding lookup
        let mut hidden = self.embed(&[token_id]);

        // GH-278: Add learned position embedding for absolute encoding (GPT-2, BERT, whisper)
        if self.config.constraints.uses_absolute_positions() {
            if let Some(ref pos_emb) = self.position_embedding {
                let start = position * hidden_dim;
                let end = start + hidden_dim;
                if end <= pos_emb.len() {
                    for i in 0..hidden_dim {
                        hidden[i] += pos_emb[start + i];
                    }
                }
            }
        }

        // GH-278: Use contract-derived norm type.
        let use_rmsnorm = self.config.constraints.uses_rmsnorm();

        // GH-479: Use config methods (Qwen3 head_dim != hidden/heads)
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.head_dim();
        let kv_dim = self.config.kv_dim();

        // PARITY-113: Track CUDA kernel count for GPU dispatch metrics
        #[cfg(feature = "cuda")]
        let cuda_enabled = self.cuda_enabled();

        // 2. Process through transformer layers
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let normed = if use_rmsnorm {
                ops::rms_norm(&hidden, &layer.attn_norm_weight, self.config.eps)
            } else {
                ops::layer_norm(
                    &hidden,
                    &layer.attn_norm_weight,
                    layer.attn_norm_bias.as_deref(),
                    self.config.eps,
                )
            };

            #[cfg(feature = "cuda")]
            if cuda_enabled {
                self.adaptive_layer_attention_cuda(
                    &mut hidden, layer, layer_idx, &normed, cache, position,
                    use_rmsnorm, metrics, num_kv_heads, head_dim, kv_dim, hidden_dim,
                )?;
                continue;
            }

            self.adaptive_layer_attention_cpu(
                &mut hidden, layer, layer_idx, &normed, cache, position,
                use_rmsnorm, metrics, num_kv_heads, head_dim, kv_dim, hidden_dim,
            )?;
        }

        // Advance cache position after processing all layers
        cache.advance();

        // 3. Final layer norm (RMSNorm for LLaMA, LayerNorm for others)
        let normed = if use_rmsnorm {
            ops::rms_norm(&hidden, &self.output_norm_weight, self.config.eps)
        } else {
            ops::layer_norm(
                &hidden,
                &self.output_norm_weight,
                self.output_norm_bias.as_deref(),
                self.config.eps,
            )
        };

        // 4. LM head projection
        // PARITY-113: Record GPU dispatch for LM head when CUDA is enabled
        #[cfg(feature = "cuda")]
        if cuda_enabled {
            let start = std::time::Instant::now();
            let mut logits = self.fused_matmul(&normed, &self.lm_head_weight)?;
            metrics.record_gpu_dispatch();
            metrics.record_gpu_latency(start.elapsed());
            if let Some(ref bias) = self.lm_head_bias {
                ops::add_bias(&mut logits, bias);
            }
            return Ok(logits);
        }

        let mut logits = self.fused_matmul(&normed, &self.lm_head_weight)?;
        if let Some(ref bias) = self.lm_head_bias {
            ops::add_bias(&mut logits, bias);
        }

        Ok(logits)
    }

    /// FFN computation with GPU dispatch metrics for adaptive forward pass (PMAT-260)
    ///
    /// Same as `adaptive_layer_ffn` but records GPU dispatch metrics for each
    /// matmul operation. Used by the CUDA path of `forward_single_with_cache_adaptive`.
    ///
    /// # Arguments
    /// * `hidden` - Current hidden state
    /// * `layer_idx` - Layer index
    /// * `use_rmsnorm` - Whether to use RMSNorm (LLaMA) or LayerNorm (phi-2)
    /// * `metrics` - Dispatch metrics tracker
    ///
    /// # Returns
    /// FFN output vector to be added as residual
    ///
    /// # Errors
    /// Returns error if tensor operations fail
    #[cfg(feature = "gpu")]
    #[allow(dead_code)]
    fn adaptive_layer_ffn_cuda(
        &self,
        hidden: &[f32],
        layer_idx: usize,
        use_rmsnorm: bool,
        metrics: &std::sync::Arc<DispatchMetrics>,
    ) -> Result<Vec<f32>> {
        let layer = &self.layers[layer_idx];

        // Pre-FFN layer norm
        let ffn_input = if let Some(ref ffn_norm) = layer.ffn_norm_weight {
            if use_rmsnorm {
                ops::rms_norm(hidden, ffn_norm, self.config.eps)
            } else {
                ops::layer_norm(
                    hidden,
                    ffn_norm,
                    layer.ffn_norm_bias.as_deref(),
                    self.config.eps,
                )
            }
        } else {
            hidden.to_vec()
        };

        // FFN with SwiGLU or GELU activation (contract-driven, GH-278)
        let ffn_activated = if self.config.constraints.has_gate_ffn() {
            // SwiGLU path
            let (mut ffn_gate, mut ffn_up) = if let Some(ref gate_weight) = layer.ffn_gate_weight {
                let start = std::time::Instant::now();
                let ffn_up = self.fused_matmul(&ffn_input, &layer.ffn_up_weight)?;
                metrics.record_gpu_dispatch();
                metrics.record_gpu_latency(start.elapsed());
                let start = std::time::Instant::now();
                let ffn_gate = self.fused_matmul(&ffn_input, gate_weight)?;
                metrics.record_gpu_dispatch();
                metrics.record_gpu_latency(start.elapsed());
                (ffn_gate, ffn_up)
            } else {
                // GH-306: Fused gate_up weight (Phi-3.5)
                let start = std::time::Instant::now();
                let fused = self.fused_matmul(&ffn_input, &layer.ffn_up_weight)?;
                metrics.record_gpu_dispatch();
                metrics.record_gpu_latency(start.elapsed());
                let half = fused.len() / 2;
                (fused[..half].to_vec(), fused[half..].to_vec())
            };
            if let Some(ref bias) = layer.ffn_up_bias {
                ops::add_bias(&mut ffn_up, bias);
            }
            if let Some(ref bias) = layer.ffn_gate_bias {
                ops::add_bias(&mut ffn_gate, bias);
            }

            ops::silu(&mut ffn_gate);
            for i in 0..ffn_gate.len() {
                ffn_gate[i] *= ffn_up[i];
            }
            ffn_gate
        } else {
            // GELU path
            let start = std::time::Instant::now();
            let mut ffn_hidden = self.fused_matmul(&ffn_input, &layer.ffn_up_weight)?;
            metrics.record_gpu_dispatch();
            metrics.record_gpu_latency(start.elapsed());
            if let Some(ref bias) = layer.ffn_up_bias {
                ops::add_bias(&mut ffn_hidden, bias);
            }
            ops::gelu(&mut ffn_hidden);
            ffn_hidden
        };

        // FFN down projection
        let start = std::time::Instant::now();
        let mut ffn_output = self.fused_matmul(&ffn_activated, &layer.ffn_down_weight)?;
        metrics.record_gpu_dispatch();
        metrics.record_gpu_latency(start.elapsed());
        if let Some(ref bias) = layer.ffn_down_bias {
            ops::add_bias(&mut ffn_output, bias);
        }

        Ok(ffn_output)
    }
}

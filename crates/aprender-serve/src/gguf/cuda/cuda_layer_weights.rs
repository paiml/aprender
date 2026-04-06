impl OwnedQuantizedModelCuda {

    /// Clone layer weights and build cache keys for CUDA forward pass (PMAT-260)
    ///
    /// Extracts the weight cloning block from `forward_single_full_cuda_with_cache`.
    /// Clones weights to avoid borrow conflicts with `&mut self` (needed for
    /// `fused_matmul_cuda` which takes `&mut self`).
    ///
    /// Returns a tuple of:
    /// - (qkv_cache_key, qkv_weight, qkv_bias)
    /// - (attn_norm_weight, attn_norm_bias)
    /// - (attn_output_weight, attn_output_cache_key, attn_output_bias)
    /// - (ffn_up_weight, ffn_up_cache_key, ffn_up_bias)
    /// - (ffn_down_weight, ffn_down_cache_key, ffn_down_bias)
    /// - (ffn_gate_weight, ffn_gate_cache_key, ffn_gate_bias)
    #[allow(clippy::type_complexity)]
    fn cuda_clone_layer_weights(
        &self,
        layer_idx: usize,
    ) -> CudaLayerWeights {
        let layer = &self.model.layers[layer_idx];

        // PAR-014: Capture original weight pointers BEFORE cloning for stable cache keys
        let attn_output_cache_key = format!(
            "q4k_{:016x}",
            layer.attn_output_weight.data.as_ptr() as usize
        );
        let ffn_up_cache_key = format!(
            "q4k_{:016x}",
            layer.ffn_up_weight.data.as_ptr() as usize
        );
        let ffn_down_cache_key = format!(
            "q4k_{:016x}",
            layer.ffn_down_weight.data.as_ptr() as usize
        );
        let qkv_cache_key = match &layer.qkv_weight {
            OwnedQKVWeights::Fused(ref tensor) => {
                format!("q4k_{:016x}", tensor.data.as_ptr() as usize)
            },
            OwnedQKVWeights::Separate { ref q, .. } => {
                format!("q4k_{:016x}", q.data.as_ptr() as usize)
            },
        };

        let ffn_gate_weight = layer.ffn_gate_weight.clone();
        let ffn_gate_bias = layer.ffn_gate_bias.clone();
        let ffn_gate_cache_key = ffn_gate_weight
            .as_ref()
            .map(|w| format!("q4k_{:016x}", w.data.as_ptr() as usize));

        // Clone weights to avoid borrow conflicts with &mut self
        let qkv_weight = layer.qkv_weight.clone();
        let qkv_bias = layer.qkv_bias.clone();
        let attn_norm_weight = layer.attn_norm_weight.clone();
        let attn_norm_bias = layer.attn_norm_bias.clone();
        let attn_output_bias = layer.attn_output_bias.clone();
        let ffn_up_bias = layer.ffn_up_bias.clone();
        let ffn_down_bias = layer.ffn_down_bias.clone();

        // Reconstruct weight tensors
        let attn_output_weight = OwnedQuantizedTensor {
            data: layer.attn_output_weight.data.clone(),
            in_dim: layer.attn_output_weight.in_dim,
            out_dim: layer.attn_output_weight.out_dim,
            qtype: layer.attn_output_weight.qtype,
        };
        let ffn_up_weight = OwnedQuantizedTensor {
            data: layer.ffn_up_weight.data.clone(),
            in_dim: layer.ffn_up_weight.in_dim,
            out_dim: layer.ffn_up_weight.out_dim,
            qtype: layer.ffn_up_weight.qtype,
        };
        let ffn_down_weight = OwnedQuantizedTensor {
            data: layer.ffn_down_weight.data.clone(),
            in_dim: layer.ffn_down_weight.in_dim,
            out_dim: layer.ffn_down_weight.out_dim,
            qtype: layer.ffn_down_weight.qtype,
        };

        CudaLayerWeights {
            qkv_cache_key,
            qkv_weight,
            qkv_bias,
            attn_norm_weight,
            attn_norm_bias,
            attn_output_weight,
            attn_output_cache_key,
            attn_output_bias,
            ffn_up_weight,
            ffn_up_cache_key,
            ffn_up_bias,
            ffn_down_weight,
            ffn_down_cache_key,
            ffn_down_bias,
            ffn_gate_weight,
            ffn_gate_cache_key,
            ffn_gate_bias,
            // GH-479: QK norm weights
            attn_q_norm_weight: layer.attn_q_norm_weight.clone(),
            attn_k_norm_weight: layer.attn_k_norm_weight.clone(),
        }
    }

    /// Compute FFN input with optional layer norm (PMAT-260)
    ///
    /// Shared by SwiGLU and GELU paths. The `fallback` parameter differs:
    /// - SwiGLU passes `hidden` (no parallel residual)
    /// - GELU passes `normed` (parallel residual for phi-2)
    fn cuda_compute_ffn_input(
        &self,
        hidden: &[f32],
        fallback: &[f32],
        layer_idx: usize,
        use_rmsnorm: bool,
        eps: f32,
    ) -> Vec<f32> {
        if let Some(ref ffn_norm) = self.model.layers[layer_idx].ffn_norm_weight {
            if use_rmsnorm {
                ops::rms_norm(hidden, ffn_norm, eps)
            } else {
                ops::layer_norm(
                    hidden,
                    ffn_norm,
                    self.model.layers[layer_idx].ffn_norm_bias.as_deref(),
                    eps,
                )
            }
        } else {
            fallback.to_vec()
        }
    }

    /// CUDA fused Q4K FFN — single GPU round-trip (PMAT-260)
    fn cuda_fused_q4k_ffn(
        &mut self,
        hidden: &[f32],
        lw: &CudaLayerWeights,
        hidden_dim: usize,
    ) -> Result<Vec<f32>> {
        let intermediate_dim = lw.ffn_up_weight.out_dim;
        let mut output = vec![0.0f32; hidden_dim];

        if !self.executor.has_quantized_weights(&lw.ffn_up_cache_key) {
            self.executor
                .load_quantized_weights(&lw.ffn_up_cache_key, &lw.ffn_up_weight.data)
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "cuda_ffn_up_cache".to_string(),
                    reason: format!("Failed to cache FFN up weights: {e}"),
                })?;
        }
        if !self.executor.has_quantized_weights(&lw.ffn_down_cache_key) {
            self.executor
                .load_quantized_weights(&lw.ffn_down_cache_key, &lw.ffn_down_weight.data)
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "cuda_ffn_down_cache".to_string(),
                    reason: format!("Failed to cache FFN down weights: {e}"),
                })?;
        }

        self.executor
            .fused_ffn_q4k(
                hidden,
                &mut output,
                &lw.ffn_up_cache_key,
                &lw.ffn_down_cache_key,
                hidden_dim as u32,
                intermediate_dim as u32,
            )
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "cuda_fused_ffn".to_string(),
                reason: format!("CUDA fused FFN failed: {e}"),
            })?;

        Ok(output)
    }

    /// CUDA SwiGLU FFN path for LLaMA-style models (PMAT-260)
    fn cuda_swiglu_ffn(
        &mut self,
        hidden: &[f32],
        layer_idx: usize,
        lw: &CudaLayerWeights,
        gate_weight: &OwnedQuantizedTensor,
        gate_cache_key: &str,
        use_rmsnorm: bool,
        eps: f32,
    ) -> Result<Vec<f32>> {
        let ffn_input = self.cuda_compute_ffn_input(hidden, hidden, layer_idx, use_rmsnorm, eps);

        // UP projection on normalized input
        let mut ffn_up =
            self.fused_matmul_cuda_with_key(&ffn_input, &lw.ffn_up_weight, &lw.ffn_up_cache_key)?;
        if let Some(ref bias) = lw.ffn_up_bias {
            ops::add_bias(&mut ffn_up, bias);
        }

        // GATE projection on normalized input
        let mut ffn_gate =
            self.fused_matmul_cuda_with_key(&ffn_input, gate_weight, gate_cache_key)?;
        if let Some(ref bias) = lw.ffn_gate_bias {
            ops::add_bias(&mut ffn_gate, bias);
        }

        // SiLU on gate, then multiply with up
        ops::silu(&mut ffn_gate);
        for i in 0..ffn_gate.len() {
            ffn_gate[i] *= ffn_up[i];
        }

        // DOWN projection
        let mut ffn_output = self.fused_matmul_cuda_with_key(
            &ffn_gate,
            &lw.ffn_down_weight,
            &lw.ffn_down_cache_key,
        )?;
        if let Some(ref bias) = lw.ffn_down_bias {
            ops::add_bias(&mut ffn_output, bias);
        }
        Ok(ffn_output)
    }

    /// CUDA GELU FFN path for phi-2-style models (PMAT-260)
    fn cuda_gelu_ffn(
        &mut self,
        hidden: &[f32],
        normed: &[f32],
        layer_idx: usize,
        lw: &CudaLayerWeights,
        use_rmsnorm: bool,
        eps: f32,
    ) -> Result<Vec<f32>> {
        let ffn_input = self.cuda_compute_ffn_input(hidden, normed, layer_idx, use_rmsnorm, eps);

        let mut ffn_hidden =
            self.fused_matmul_cuda_with_key(&ffn_input, &lw.ffn_up_weight, &lw.ffn_up_cache_key)?;
        if let Some(ref bias) = lw.ffn_up_bias {
            ops::add_bias(&mut ffn_hidden, bias);
        }
        ops::gelu(&mut ffn_hidden);

        let mut ffn_output = self.fused_matmul_cuda_with_key(
            &ffn_hidden,
            &lw.ffn_down_weight,
            &lw.ffn_down_cache_key,
        )?;
        if let Some(ref bias) = lw.ffn_down_bias {
            ops::add_bias(&mut ffn_output, bias);
        }
        Ok(ffn_output)
    }

    /// CUDA FFN dispatcher — routes to fused Q4K, SwiGLU, or GELU path (PMAT-260)
    fn cuda_layer_ffn(
        &mut self,
        hidden: &[f32],
        normed: &[f32],
        layer_idx: usize,
        lw: &CudaLayerWeights,
        use_rmsnorm: bool,
        eps: f32,
        hidden_dim: usize,
    ) -> Result<Vec<f32>> {
        // PAR-057: Re-enable fused FFN path now that kernels are fixed
        #[allow(clippy::overly_complex_bool_expr)]
        if lw.ffn_up_bias.is_none()
            && lw.ffn_down_bias.is_none()
            && lw.ffn_up_weight.qtype == 12
            && lw.ffn_down_weight.qtype == 12
        {
            return self.cuda_fused_q4k_ffn(hidden, lw, hidden_dim);
        }

        if let (Some(ref gate_weight), Some(ref gate_cache_key)) =
            (&lw.ffn_gate_weight, &lw.ffn_gate_cache_key)
        {
            self.cuda_swiglu_ffn(hidden, layer_idx, lw, gate_weight, gate_cache_key, use_rmsnorm, eps)
        } else {
            self.cuda_gelu_ffn(hidden, normed, layer_idx, lw, use_rmsnorm, eps)
        }
    }

    /// IMP-1010: Full GPU forward pass for single token with KV cache
    ///
    /// This method uses GPU acceleration for ALL matmul operations:
    /// - QKV projection (3x hidden_dim x hidden_dim)
    /// - Attention output projection (hidden_dim x hidden_dim)
    /// - FFN up projection (hidden_dim x 4*hidden_dim)
    /// - FFN down projection (4*hidden_dim x hidden_dim)
    /// - LM head projection (hidden_dim x vocab_size)
    ///
    /// # Performance Target
    ///
    /// - CPU SIMD path: ~5 tok/s
    /// - Full GPU path: ~200 tok/s (matching Ollama)
    pub fn forward_single_full_cuda_with_cache(
        &mut self,
        token_id: u32,
        cache: &mut OwnedQuantizedKVCache,
        position: usize,
    ) -> Result<Vec<f32>> {
        let hidden_dim = self.model.config.hidden_dim;
        let num_heads = self.model.config.num_heads;
        let num_kv_heads = self.model.config.num_kv_heads;
        // GH-479: Use config methods for head_dim/q_dim/kv_dim (Qwen3 head_dim != hidden/heads)
        let head_dim = self.model.config.head_dim();
        let q_dim = self.model.config.q_dim();
        let num_layers = self.model.layers.len();
        let eps = self.model.config.eps;

        // PAR-021: GQA support
        let kv_dim = self.model.config.kv_dim();

        // 1. Token embedding lookup (CPU - fast enough, single lookup)
        let mut hidden = self.model.embed(&[token_id]);

        // PAR-016: Pre-capture LM head cache key for stable caching
        let lm_head_cache_key = format!(
            "q4k_{:016x}",
            self.model.lm_head_weight.data.as_ptr() as usize
        );

        // GH-278: Use contract-derived norm type.
        let use_rmsnorm = self.model.config.constraints.uses_rmsnorm();

        // 2. Process through transformer layers
        for layer_idx in 0..num_layers {
            // Clone layer weights to avoid borrow conflicts with &mut self
            let lw = self.cuda_clone_layer_weights(layer_idx);

            // 2a. Attention layer norm (CPU - fast for single vector)
            let normed = if use_rmsnorm {
                ops::rms_norm(&hidden, &lw.attn_norm_weight, eps)
            } else {
                ops::layer_norm(&hidden, &lw.attn_norm_weight, lw.attn_norm_bias.as_deref(), eps)
            };

            // 2b. QKV projection (GPU - PAR-014: use pre-captured cache key)
            let mut qkv = self.qkv_matmul_cuda_with_key(&normed, &lw.qkv_weight, &lw.qkv_cache_key)?;
            if let Some(ref bias) = lw.qkv_bias {
                ops::add_bias(&mut qkv, bias);
            }

            // 2c. Extract Q, K, V with GQA-aware sizes and apply RoPE
            // GH-479: Use q_dim (may differ from hidden_dim for Qwen3)
            let mut q = qkv[0..q_dim].to_vec();
            let mut k = qkv[q_dim..q_dim + kv_dim].to_vec();
            let v = qkv[q_dim + kv_dim..q_dim + 2 * kv_dim].to_vec();

            // GH-479: Per-head QK RMSNorm (Qwen3) — before RoPE
            if let Some(ref w) = lw.attn_q_norm_weight {
                ops::apply_per_head_rms_norm(&mut q, w, num_heads, eps);
            }
            if let Some(ref w) = lw.attn_k_norm_weight {
                ops::apply_per_head_rms_norm(&mut k, w, num_kv_heads, eps);
            }

            self.model
                .apply_rope(&mut q, position, self.model.config.num_heads);
            self.model.apply_rope(&mut k, position, num_kv_heads);

            // 2d. Get cached K/V and compute attention
            let k_cache = cache.get_k(layer_idx);
            let v_cache = cache.get_v(layer_idx);

            let attn_out = if k_cache.is_empty() {
                // First token - no cache yet
                if num_kv_heads < num_heads {
                    let q_per_kv = num_heads / num_kv_heads;
                    let mut expanded_v = vec![0.0f32; q_dim];
                    for q_head in 0..num_heads {
                        let kv_head = q_head / q_per_kv;
                        let v_start = kv_head * head_dim;
                        let out_start = q_head * head_dim;
                        expanded_v[out_start..out_start + head_dim]
                            .copy_from_slice(&v[v_start..v_start + head_dim]);
                    }
                    expanded_v
                } else {
                    v.clone()
                }
            } else {
                self.model
                    .attention_with_cache_gqa(&q, k_cache, v_cache, &k, &v)
            };

            // 2e. Store K and V in cache
            cache.append(layer_idx, &k, &v);

            // 2f. Attention output projection (GPU - PAR-014: use pre-captured cache key)
            let mut attn_output = self.fused_matmul_cuda_with_key(
                &attn_out,
                &lw.attn_output_weight,
                &lw.attn_output_cache_key,
            )?;
            if let Some(ref bias) = lw.attn_output_bias {
                ops::add_bias(&mut attn_output, bias);
            }

            // 2g. Residual connection
            for i in 0..hidden_dim {
                hidden[i] += attn_output[i];
            }

            // 2h/2i. FFN (extracted helper)
            let ffn_output = self.cuda_layer_ffn(
                &hidden, &normed, layer_idx, &lw, use_rmsnorm, eps, hidden_dim,
            )?;

            // Residual
            for i in 0..hidden_dim {
                hidden[i] += ffn_output[i];
            }

            // NaN safety check for early positions
            if position < 2 {
                let hidden_has_nan = hidden.iter().any(|x| x.is_nan());
                if hidden_has_nan {
                    let ffn_output_has_nan = ffn_output.iter().any(|x| x.is_nan());
                    let ffn_output_sum: f32 = ffn_output.iter().sum();
                    eprintln!(
                        "[IMP-1010] pos{} L{} ffn_out: sum={:.6e}, has_nan={}",
                        position, layer_idx, ffn_output_sum, ffn_output_has_nan
                    );
                    let hidden_sum: f32 = hidden.iter().sum();
                    eprintln!(
                        "[IMP-1010] pos{} L{} hidden after FFN: sum={:.6e}, has_nan={}",
                        position, layer_idx, hidden_sum, hidden_has_nan
                    );
                }
            }

        }

        // Advance cache position
        cache.advance();

        // 3. Final layer norm (CPU - fast for single vector)
        let normed = if use_rmsnorm {
            ops::rms_norm(
                &hidden,
                &self.model.output_norm_weight,
                self.model.config.eps,
            )
        } else {
            ops::layer_norm(
                &hidden,
                &self.model.output_norm_weight,
                self.model.output_norm_bias.as_deref(),
                self.model.config.eps,
            )
        };

        // 4. LM head projection (GPU - IMP-1010, PAR-016: use pre-captured cache key)
        let lm_head_weight_data = self.model.lm_head_weight.data.clone();
        let lm_head_weight_in_dim = self.model.lm_head_weight.in_dim;
        let lm_head_weight_out_dim = self.model.lm_head_weight.out_dim;
        let lm_head_weight_qtype = self.model.lm_head_weight.qtype;
        let lm_head_weight = OwnedQuantizedTensor {
            data: lm_head_weight_data,
            in_dim: lm_head_weight_in_dim,
            out_dim: lm_head_weight_out_dim,
            qtype: lm_head_weight_qtype,
        };

        let mut logits =
            self.fused_matmul_cuda_with_key(&normed, &lm_head_weight, &lm_head_cache_key)?;
        if let Some(ref bias) = self.model.lm_head_bias {
            ops::add_bias(&mut logits, bias);
        }

        Ok(logits)
    }
}

/// Cloned layer weights and cache keys for CUDA forward pass (PMAT-260)
///
/// This struct avoids the massive tuple return type and makes the weight
/// cloning interface self-documenting.
struct CudaLayerWeights {
    qkv_cache_key: String,
    qkv_weight: OwnedQKVWeights,
    qkv_bias: Option<Vec<f32>>,
    attn_norm_weight: Vec<f32>,
    attn_norm_bias: Option<Vec<f32>>,
    attn_output_weight: OwnedQuantizedTensor,
    attn_output_cache_key: String,
    attn_output_bias: Option<Vec<f32>>,
    ffn_up_weight: OwnedQuantizedTensor,
    ffn_up_cache_key: String,
    ffn_up_bias: Option<Vec<f32>>,
    ffn_down_weight: OwnedQuantizedTensor,
    ffn_down_cache_key: String,
    ffn_down_bias: Option<Vec<f32>>,
    ffn_gate_weight: Option<OwnedQuantizedTensor>,
    ffn_gate_cache_key: Option<String>,
    ffn_gate_bias: Option<Vec<f32>>,
    // GH-479: QK norm weights (Qwen3 per-head RMSNorm)
    attn_q_norm_weight: Option<Vec<f32>>,
    attn_k_norm_weight: Option<Vec<f32>>,
}

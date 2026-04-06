
impl OwnedQuantizedModelCuda {

    /// PAR-023: Pre-upload all layer weights to GPU with naming convention for
    /// GPU-resident transformer layer.
    ///
    /// This method uploads quantized weights using names expected by
    /// `CudaExecutor::transformer_layer_gpu`:
    /// - `blk.{i}.attn_q.weight`, `blk.{i}.attn_k.weight`, `blk.{i}.attn_v.weight`
    /// - `blk.{i}.attn_output.weight`
    /// - `blk.{i}.ffn_gate.weight`, `blk.{i}.ffn_up.weight`, `blk.{i}.ffn_down.weight`
    ///
    /// # Errors
    ///
    /// Returns error if weight upload fails or model uses fused QKV (phi-2 style).
    pub fn preload_weights_gpu(&mut self) -> Result<usize> {
        // THREAD-RESOLVED: Ensure CUDA context is current for this thread
        self.executor
            .make_current()
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "cuda_make_current".to_string(),
                reason: format!("Failed to set CUDA context current: {e}"),
            })?;

        let mut total_bytes = 0usize;

        // Upload per-layer projection weights (Q/K/V, O, FFN) + LM head
        total_bytes += self.preload_layer_projection_weights()?;

        // PAR-023: Pre-cache RMSNorm weights for all layers
        let num_layers = self.model.layers.len();
        let attn_norms: Vec<&[f32]> = self
            .model
            .layers
            .iter()
            .map(|l| l.attn_norm_weight.as_slice())
            .collect();
        let ffn_norms: Vec<&[f32]> = self
            .model
            .layers
            .iter()
            .map(|l| {
                l.ffn_norm_weight
                    .as_ref()
                    .map_or(l.attn_norm_weight.as_slice(), |w| w.as_slice())
            })
            .collect();

        total_bytes += self
            .executor
            .preload_rmsnorm_weights(num_layers, &attn_norms, &ffn_norms)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "preload_weights_gpu".to_string(),
                reason: format!("Failed to upload RMSNorm weights: {}", e),
            })?;

        // PAR-023: Pre-cache output norm (final layer norm) weight
        // This enables fully GPU-resident forward pass including output norm + LM head
        total_bytes += self
            .executor
            .preload_output_norm(&self.model.output_norm_weight)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "preload_weights_gpu".to_string(),
                reason: format!("Failed to upload output norm weights: {}", e),
            })?;

        // BIAS-FIX: Pre-cache QKV bias vectors for all layers (GQA-aware)
        total_bytes += self.preload_qkv_bias_weights(num_layers)?;

        // PAR-064-FIX: Pre-cache LM head bias (output.bias) for models that have it
        // Without this bias, GPU inference produces incorrect token predictions
        total_bytes += self
            .executor
            .preload_lm_head_bias(self.model.lm_head_bias.as_deref())
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "preload_weights_gpu".to_string(),
                reason: format!("Failed to upload LM head bias: {}", e),
            })?;

        // GH-279: Pre-cache QkNorm weights for Qwen3 per-head RMSNorm
        total_bytes += self.preload_qk_norm_weights()?;

        // PAR-043: Build indexed weight lookup table for O(1) access during decode
        // This eliminates ~10ms constant CPU overhead per token from string formatting + HashMap lookups
        // PAR-107: Skip if already indexed to preserve CUDA graph (graph captures buffer addresses)
        if !self.executor.has_indexed_weights() {
            // GH-279: Pass ArchConstraints for ValidatedLayerWeights enforcement
            let arch = &self.model.config.constraints;
            self.executor
                .build_indexed_weights(num_layers, |i| format!("blk.{}", i), arch)
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "preload_weights_gpu".to_string(),
                    reason: format!("PAR-043: Failed to build indexed weights: {}", e),
                })?;
        }

        // PAR-044: Initialize workspace buffers for zero-allocation forward pass
        // This eliminates ~288 buffer allocations per token
        // PAR-107: Skip if already initialized to preserve CUDA graph (graph captures buffer addresses)
        // ROOT CAUSE FIX: Reallocating workspace invalidates graph since addresses change
        if !self.executor.has_workspace() {
            self.executor
                .init_workspace(
                    self.model.config.hidden_dim,
                    self.model.config.intermediate_dim,
                )
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "preload_weights_gpu".to_string(),
                    reason: format!("PAR-044: Failed to initialize workspace: {}", e),
                })?;
        }

        Ok(total_bytes)
    }

    /// GH-279: Pre-cache QkNorm weights for Qwen3 per-head RMSNorm.
    /// Optional — only Qwen3+ models have attn_q_norm_weight/attn_k_norm_weight.
    fn preload_qk_norm_weights(&mut self) -> Result<usize> {
        let mut total_bytes = 0usize;
        for (layer_idx, layer) in self.model.layers.iter().enumerate() {
            if let Some(ref q_norm) = layer.attn_q_norm_weight {
                let name = format!("blk.{}.attn_q_norm.gamma", layer_idx);
                total_bytes += self
                    .executor
                    .cache_rmsnorm_gamma(&name, q_norm)
                    .map_err(|e| RealizarError::UnsupportedOperation {
                        operation: "preload_qk_norm_weights".to_string(),
                        reason: format!(
                            "Failed to upload Q norm weights for layer {}: {}",
                            layer_idx, e
                        ),
                    })?;
            }
            if let Some(ref k_norm) = layer.attn_k_norm_weight {
                let name = format!("blk.{}.attn_k_norm.gamma", layer_idx);
                total_bytes += self
                    .executor
                    .cache_rmsnorm_gamma(&name, k_norm)
                    .map_err(|e| RealizarError::UnsupportedOperation {
                        operation: "preload_qk_norm_weights".to_string(),
                        reason: format!(
                            "Failed to upload K norm weights for layer {}: {}",
                            layer_idx, e
                        ),
                    })?;
            }
        }
        Ok(total_bytes)
    }

    /// Upload per-layer projection weights (Q/K/V, O, FFN gate/up/down) and LM head to GPU.
    ///
    /// PAR-058: Passes qtype per weight for mixed-quant model support.
    /// Skips weights already present on GPU (idempotent).
    fn preload_layer_projection_weights(&mut self) -> Result<usize> {
        let mut total_bytes = 0usize;

        for (layer_idx, layer) in self.model.layers.iter().enumerate() {
            let prefix = format!("blk.{}", layer_idx);
            total_bytes += upload_layer_qkv(&mut self.executor, &prefix, layer_idx, layer)?;
            total_bytes += upload_layer_ffn(&mut self.executor, &prefix, layer)?;
        }

        // LM head weights
        total_bytes += upload_if_absent(
            &mut self.executor, "output.weight",
            &self.model.lm_head_weight.data, self.model.lm_head_weight.qtype,
        )?;

        Ok(total_bytes)
    }

    /// Collect and upload QKV bias vectors for all layers to GPU.
    ///
    /// BIAS-FIX: Qwen2.5 models have QKV bias that must be added after GEMV.
    /// GQA-FIX: Uses config-aware dimension calculation for GQA models
    /// where Q, K, V have different sizes (out_dim / 3 is wrong for GQA).
    ///
    /// # Errors
    ///
    /// Returns error if bias upload fails.
    fn preload_qkv_bias_weights(&mut self, num_layers: usize) -> Result<usize> {
        let num_heads = self.model.config.num_heads;
        let num_kv_heads = self.model.config.num_kv_heads;
        let hidden_dim = self.model.config.hidden_dim;
        let head_dim = self.model.config.head_dim();

        let q_biases: Vec<Option<&[f32]>> = self
            .model
            .layers
            .iter()
            .map(|l| {
                l.qkv_bias.as_ref().map(|b| {
                    // Q bias is first q_dim elements (GQA-aware)
                    let q_dim = l
                        .qkv_weight
                        .q_dim_for_config(num_heads, num_kv_heads, hidden_dim, head_dim);
                    &b[..q_dim]
                })
            })
            .collect();
        let k_biases: Vec<Option<&[f32]>> = self
            .model
            .layers
            .iter()
            .map(|l| {
                l.qkv_bias.as_ref().map(|b| {
                    let q_dim = l.qkv_weight.q_dim_for_config(num_heads, num_kv_heads, hidden_dim, head_dim);
                    let k_dim = l.qkv_weight.k_dim_for_config(num_heads, num_kv_heads, hidden_dim, head_dim);
                    &b[q_dim..q_dim + k_dim]
                })
            })
            .collect();
        let v_biases: Vec<Option<&[f32]>> = self
            .model
            .layers
            .iter()
            .map(|l| {
                l.qkv_bias.as_ref().map(|b| {
                    let q_dim = l
                        .qkv_weight
                        .q_dim_for_config(num_heads, num_kv_heads, hidden_dim, head_dim);
                    let k_dim = l
                        .qkv_weight
                        .k_dim_for_config(num_heads, num_kv_heads, hidden_dim, head_dim);
                    let v_dim = l
                        .qkv_weight
                        .v_dim_for_config(num_heads, num_kv_heads, hidden_dim, head_dim);
                    &b[q_dim + k_dim..q_dim + k_dim + v_dim]
                })
            })
            .collect();

        self.executor
            .preload_qkv_bias(num_layers, &q_biases, &k_biases, &v_biases)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "preload_qkv_bias_weights".to_string(),
                reason: format!("Failed to upload QKV bias: {}", e),
            })
    }

    /// Clear decode graph and related state
    ///
    /// Call this before starting a new generation session to ensure
    /// the graph is recaptured with fresh state.
    pub fn clear_decode_graph(&mut self) {
        self.executor.clear_decode_graph();
    }

    /// PAR-023: Check if model supports GPU-resident forward pass
    ///
    /// GPU-resident path requires:
    /// - Separate Q/K/V weights (not fused)
    /// - Gated FFN (SwiGLU or GatedMLP, per contract)
    /// - RMSNorm (per contract)
    #[must_use]
    pub fn supports_gpu_resident(&self) -> bool {
        // Contract-driven architecture checks (GH-278)
        let constraints = &self.model.config.constraints;
        let has_gated_ffn = constraints.has_gate_ffn();
        let has_rmsnorm = constraints.uses_rmsnorm();

        // Check first layer for QKV format (data-driven, not a heuristic)
        let has_separate_qkv = self.model.layers.first()
            .is_some_and(|l| matches!(l.qkv_weight, OwnedQKVWeights::Separate { .. }));

        has_separate_qkv && has_gated_ffn && has_rmsnorm
    }
}

/// Upload a single quantized weight to GPU if not already present (free function to avoid borrow conflicts).
fn upload_if_absent(
    executor: &mut crate::cuda::CudaExecutor,
    name: &str,
    data: &[u8],
    qtype: u32,
) -> Result<usize> {
    if executor.has_quantized_weights(name) {
        return Ok(0);
    }
    executor
        .load_quantized_weights_with_type(name, data, qtype)
        .map_err(|e| RealizarError::UnsupportedOperation {
            operation: "preload_layer_projection_weights".to_string(),
            reason: format!("Failed to upload '{}': {}", name, e),
        })
}

/// Upload Q/K/V and output projection weights for a single layer.
fn upload_layer_qkv(
    executor: &mut crate::cuda::CudaExecutor,
    prefix: &str,
    layer_idx: usize,
    layer: &crate::gguf::quantized::OwnedQuantizedLayer,
) -> Result<usize> {
    let mut total = 0usize;
    match &layer.qkv_weight {
        OwnedQKVWeights::Separate { q, k, v } => {
            total += upload_if_absent(executor, &format!("{prefix}.attn_q.weight"), &q.data, q.qtype)?;
            total += upload_if_absent(executor, &format!("{prefix}.attn_k.weight"), &k.data, k.qtype)?;
            total += upload_if_absent(executor, &format!("{prefix}.attn_v.weight"), &v.data, v.qtype)?;
        },
        OwnedQKVWeights::Fused(_) => {
            return Err(RealizarError::UnsupportedOperation {
                operation: "preload_layer_projection_weights".to_string(),
                reason: format!(
                    "Layer {} uses fused QKV (phi-2 style), GPU-resident path requires separate Q/K/V",
                    layer_idx
                ),
            });
        },
    }
    total += upload_if_absent(
        executor, &format!("{prefix}.attn_output.weight"),
        &layer.attn_output_weight.data, layer.attn_output_weight.qtype,
    )?;
    Ok(total)
}

/// Upload FFN weights (gate/up/down) for a single layer.
fn upload_layer_ffn(
    executor: &mut crate::cuda::CudaExecutor,
    prefix: &str,
    layer: &crate::gguf::quantized::OwnedQuantizedLayer,
) -> Result<usize> {
    let mut total = 0usize;
    if let Some(ref gate) = layer.ffn_gate_weight {
        total += upload_if_absent(executor, &format!("{prefix}.ffn_gate.weight"), &gate.data, gate.qtype)?;
    }
    total += upload_if_absent(
        executor, &format!("{prefix}.ffn_up.weight"),
        &layer.ffn_up_weight.data, layer.ffn_up_weight.qtype,
    )?;
    total += upload_if_absent(
        executor, &format!("{prefix}.ffn_down.weight"),
        &layer.ffn_down_weight.data, layer.ffn_down_weight.qtype,
    )?;
    Ok(total)
}

include!("cache.rs");

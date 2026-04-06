impl AprV2ModelCuda {

    // ========================================================================
    // Weight Pre-caching (2x performance optimization)
    // ========================================================================

    /// Extract model dimension config from metadata.
    ///
    /// Returns `(hidden_dim, num_layers, num_heads, num_kv_heads, intermediate_dim, head_dim, kv_dim)`.
    fn extract_model_dims(&self) -> (usize, usize, usize, usize, usize, usize, usize) {
        let hidden_dim = self.model.metadata.hidden_size.unwrap_or(0);
        let num_layers = self.model.metadata.num_layers.unwrap_or(0);
        let num_heads = self.model.metadata.num_heads.unwrap_or(1);
        let num_kv_heads = self.model.metadata.num_kv_heads.unwrap_or(num_heads);
        let intermediate_dim = self
            .model
            .metadata
            .intermediate_size
            .unwrap_or(hidden_dim * 4);
        let head_dim = if num_heads > 0 {
            hidden_dim / num_heads
        } else {
            0
        };
        let kv_dim = num_kv_heads * head_dim;
        (hidden_dim, num_layers, num_heads, num_kv_heads, intermediate_dim, head_dim, kv_dim)
    }

    /// Upload a single weight tensor (quantized or F32) to GPU.
    ///
    /// Uses GGUF-style cache names for compatibility with `build_indexed_weights()`.
    /// PMAT-113: Now caches F32 weights for GPU GEMM (was causing APR CUDA hang).
    ///
    /// # Returns
    ///
    /// `(bytes_uploaded, is_quantized)`
    fn upload_single_weight(&mut self, src_name: &str, cache_name: &str) -> (usize, bool) {
        // Clone dtype and shape upfront to release the immutable borrow on self.model
        // before calling &mut self methods below.
        let (dtype, shape) = match self.model.get_tensor(src_name) {
            Some(entry) => (entry.dtype.clone(), entry.shape.clone()),
            None => return (0, false),
        };
        if let Some(qtype) = dtype_to_ggml_qtype(&dtype) {
            // Quantized: upload raw bytes to quantized_weight_cache
            self.upload_quantized_weight(src_name, cache_name, qtype)
        } else {
            // PMAT-113: F32/F16 - cache on GPU for GEMM path
            // PMAT-222: Transpose 2D F32 weights from [n, k] to [k, n] for gemm_b_cached
            // HF convention stores weights as [out_dim, in_dim] but GEMM needs B[k, n]
            self.upload_f32_weight(src_name, cache_name, &shape)
        }
    }

    /// Upload a quantized weight tensor to GPU.
    fn upload_quantized_weight(
        &mut self,
        src_name: &str,
        cache_name: &str,
        qtype: u32,
    ) -> (usize, bool) {
        if let Ok(bytes) = self.model.get_tensor_bytes(src_name) {
            let size = self
                .executor
                .load_quantized_weights_with_type(cache_name, bytes, qtype)
                .unwrap_or(0);
            (size, true)
        } else {
            (0, false)
        }
    }

    /// Upload an F32 weight tensor to GPU, transposing 2D weights for GEMM.
    fn upload_f32_weight(
        &mut self,
        src_name: &str,
        cache_name: &str,
        shape: &[usize],
    ) -> (usize, bool) {
        if let Ok(weights) = self.model.get_tensor_f32(src_name) {
            let final_weights = if shape.len() == 2 {
                let rows = shape[0]; // out_dim (n)
                let cols = shape[1]; // in_dim (k)
                let mut transposed = vec![0.0f32; weights.len()];
                for i in 0..rows {
                    for j in 0..cols {
                        transposed[j * rows + i] = weights[i * cols + j];
                    }
                }
                transposed
            } else {
                weights
            };
            let size = self
                .executor
                .load_weights(cache_name, &final_weights)
                .unwrap_or(0);
            (size, false)
        } else {
            (0, false)
        }
    }

    /// Build per-layer weight name mappings for projection weights.
    ///
    /// Returns a vec of `(source_patterns, cache_suffix)` for Q/K/V/O and FFN gate/up/down.
    fn layer_projection_mappings(layer_idx: usize) -> Vec<(Vec<String>, &'static str)> {
        vec![
            (
                vec![
                    format!("model.layers.{layer_idx}.self_attn.q_proj.weight"),
                    format!("layers.{layer_idx}.self_attn.q_proj.weight"),
                    format!("blk.{layer_idx}.attn_q.weight"),
                ],
                "attn_q.weight",
            ),
            (
                vec![
                    format!("model.layers.{layer_idx}.self_attn.k_proj.weight"),
                    format!("layers.{layer_idx}.self_attn.k_proj.weight"),
                    format!("blk.{layer_idx}.attn_k.weight"),
                ],
                "attn_k.weight",
            ),
            (
                vec![
                    format!("model.layers.{layer_idx}.self_attn.v_proj.weight"),
                    format!("layers.{layer_idx}.self_attn.v_proj.weight"),
                    format!("blk.{layer_idx}.attn_v.weight"),
                ],
                "attn_v.weight",
            ),
            (
                vec![
                    format!("model.layers.{layer_idx}.self_attn.o_proj.weight"),
                    format!("layers.{layer_idx}.self_attn.o_proj.weight"),
                    format!("blk.{layer_idx}.attn_output.weight"),
                ],
                "attn_output.weight",
            ),
            (
                vec![
                    format!("model.layers.{layer_idx}.mlp.gate_proj.weight"),
                    format!("layers.{layer_idx}.mlp.gate_proj.weight"),
                    format!("blk.{layer_idx}.ffn_gate.weight"),
                ],
                "ffn_gate.weight",
            ),
            (
                vec![
                    format!("model.layers.{layer_idx}.mlp.up_proj.weight"),
                    format!("layers.{layer_idx}.mlp.up_proj.weight"),
                    format!("blk.{layer_idx}.ffn_up.weight"),
                ],
                "ffn_up.weight",
            ),
            (
                vec![
                    format!("model.layers.{layer_idx}.mlp.down_proj.weight"),
                    format!("layers.{layer_idx}.mlp.down_proj.weight"),
                    format!("blk.{layer_idx}.ffn_down.weight"),
                ],
                "ffn_down.weight",
            ),
        ]
    }

    /// Cache per-layer projection weights (Q/K/V/O + FFN gate/up/down).
    ///
    /// Uses GGUF naming convention to match `build_indexed_weights()` expectations.
    ///
    /// # Returns
    ///
    /// `(total_bytes, quantized_count, f32_count)`
    fn cache_layer_projections(&mut self, layer_idx: usize) -> (usize, usize, usize) {
        let prefix = format!("blk.{layer_idx}");
        let weight_mappings = Self::layer_projection_mappings(layer_idx);

        let mut total_bytes = 0;
        let mut quantized_count = 0;
        let mut f32_count = 0;

        for (patterns, suffix) in weight_mappings {
            let patterns_ref: Vec<&str> = patterns.iter().map(String::as_str).collect();
            if let Ok(src_name) = self.model.find_tensor_name(&patterns_ref) {
                let cache_name = format!("{prefix}.{suffix}");
                let (bytes, is_quantized) = self.upload_single_weight(&src_name, &cache_name);
                if bytes > 0 {
                    total_bytes += bytes;
                    if is_quantized {
                        quantized_count += 1;
                    } else {
                        f32_count += 1;
                    }
                }
            }
        }

        (total_bytes, quantized_count, f32_count)
    }

    /// Cache fused QKV weight by unfusing into separate Q/K/V tensors.
    ///
    /// PMAT-113: APR models from HuggingFace have Q/K/V fused into `qkv_proj.weight`.
    /// Unfuse and cache as separate Q/K/V with names the forward path expects.
    /// NOTE: P1 quality issue exists (SATD-WARNING in `generate_cuda_with_cache`).
    /// The APR import has corrupt tensor layouts - this caching doesn't fix that.
    ///
    /// # Returns
    ///
    /// `(total_bytes, f32_count)`
    fn cache_fused_qkv(
        &mut self,
        layer_idx: usize,
        hidden_dim: usize,
        kv_dim: usize,
    ) -> (usize, usize) {
        let fused_qkv_patterns = vec![format!(
            "model.layers.{layer_idx}.self_attn.qkv_proj.weight"
        )];
        let fused_patterns_ref: Vec<&str> =
            fused_qkv_patterns.iter().map(String::as_str).collect();

        let src_name = match self.model.find_tensor_name(&fused_patterns_ref) {
            Ok(name) => name,
            Err(_) => return (0, 0),
        };

        // Load and unfuse QKV for F32 models
        let qkv_weight = match self.model.get_tensor_f32(&src_name) {
            Ok(w) => w,
            Err(_) => return (0, 0),
        };

        // Unfuse: Q is first hidden_dim rows, K is next kv_dim, V is last kv_dim
        let q_size = hidden_dim * hidden_dim;
        let k_size = kv_dim * hidden_dim;
        let v_size = kv_dim * hidden_dim;

        if qkv_weight.len() < q_size + k_size + v_size {
            return (0, 0);
        }

        let q_weight: Vec<f32> = qkv_weight[0..q_size].to_vec();
        let k_weight: Vec<f32> = qkv_weight[q_size..q_size + k_size].to_vec();
        let v_weight: Vec<f32> =
            qkv_weight[q_size + k_size..q_size + k_size + v_size].to_vec();

        // PMAT-114: Trace K weight for layer 0 to debug 100x difference
        if layer_idx == 0 && std::env::var("APR_TRACE_LAYERS").is_ok() {
            Self::trace_k_weight_pre_transpose(&k_weight);
        }

        // Transpose for GPU GEMM (row-major to column-major)
        let q_weight_t = transpose_matrix(&q_weight, hidden_dim, hidden_dim);
        let k_weight_t = transpose_matrix(&k_weight, kv_dim, hidden_dim);
        let v_weight_t = transpose_matrix(&v_weight, kv_dim, hidden_dim);

        // PMAT-114: Trace K weight after transpose
        if layer_idx == 0 && std::env::var("APR_TRACE_LAYERS").is_ok() {
            Self::trace_k_weight_post_transpose(&k_weight_t);
        }

        // Cache with GGUF-style naming to match forward path (PMAT-805)
        let mut total_bytes = 0;
        let mut f32_count = 0;
        let weight_pairs = [
            (format!("blk.{layer_idx}.attn_q.weight"), q_weight_t),
            (format!("blk.{layer_idx}.attn_k.weight"), k_weight_t),
            (format!("blk.{layer_idx}.attn_v.weight"), v_weight_t),
        ];

        for (cache_name, weight_data) in &weight_pairs {
            if let Ok(bytes) = self.executor.load_weights(cache_name, weight_data) {
                total_bytes += bytes;
                f32_count += 1;
            }
        }

        (total_bytes, f32_count)
    }

    /// PMAT-114: Trace K weight statistics before transpose (layer 0 only).
    fn trace_k_weight_pre_transpose(k_weight: &[f32]) {
        let k_sum: f32 = k_weight.iter().sum();
        let k_mean = k_sum / k_weight.len() as f32;
        let k_min = k_weight.iter().cloned().fold(f32::INFINITY, f32::min);
        let k_max = k_weight.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        eprintln!("[PMAT-114] L0 K weight (pre-transpose): mean={:.6}, min={:.6}, max={:.6}, len={}",
            k_mean, k_min, k_max, k_weight.len());
        eprintln!(
            "[PMAT-114] L0 K weight first10={:?}",
            &k_weight[..10.min(k_weight.len())]
        );
    }

    /// PMAT-114: Trace K weight statistics after transpose (layer 0 only).
    fn trace_k_weight_post_transpose(k_weight_t: &[f32]) {
        let k_sum: f32 = k_weight_t.iter().sum();
        let k_mean = k_sum / k_weight_t.len() as f32;
        eprintln!(
            "[PMAT-114] L0 K weight (post-transpose): mean={:.6}, len={}",
            k_mean,
            k_weight_t.len()
        );
        eprintln!(
            "[PMAT-114] L0 K weight_t first10={:?}",
            &k_weight_t[..10.min(k_weight_t.len())]
        );
    }

    /// Build per-layer norm weight name mappings.
    ///
    /// Returns mappings for attn_norm, ffn_norm, and GH-279 QK norm weights.
    fn layer_norm_mappings(layer_idx: usize) -> Vec<(Vec<String>, &'static str)> {
        vec![
            (
                vec![
                    format!("model.layers.{layer_idx}.input_layernorm.weight"),
                    format!("layers.{layer_idx}.input_layernorm.weight"),
                    format!("blk.{layer_idx}.attn_norm.weight"),
                ],
                "attn_norm.gamma",
            ),
            (
                vec![
                    format!("model.layers.{layer_idx}.post_attention_layernorm.weight"),
                    format!("layers.{layer_idx}.post_attention_layernorm.weight"),
                    format!("blk.{layer_idx}.ffn_norm.weight"),
                ],
                "ffn_norm.gamma",
            ),
            // GH-279: QK norm weights (Qwen3 per-head RMSNorm on Q and K)
            (
                vec![
                    format!("model.layers.{layer_idx}.self_attn.q_norm.weight"),
                    format!("blk.{layer_idx}.attn_q_norm.weight"),
                ],
                "attn_q_norm.gamma",
            ),
            (
                vec![
                    format!("model.layers.{layer_idx}.self_attn.k_norm.weight"),
                    format!("blk.{layer_idx}.attn_k_norm.weight"),
                ],
                "attn_k_norm.gamma",
            ),
        ]
    }

    /// Cache RMSNorm gamma weights for a single layer (always F32).
    ///
    /// Uploads attn_norm, ffn_norm, and GH-279 QK norm weights.
    ///
    /// # Returns
    ///
    /// Total bytes uploaded.
    fn cache_layer_norms(&mut self, layer_idx: usize) -> usize {
        let prefix = format!("blk.{layer_idx}");
        let norm_mappings = Self::layer_norm_mappings(layer_idx);
        let mut total_bytes = 0;

        for (patterns, suffix) in norm_mappings {
            let patterns_ref: Vec<&str> = patterns.iter().map(String::as_str).collect();
            if let Ok(src_name) = self.model.find_tensor_name(&patterns_ref) {
                if let Ok(gamma) = self.model.get_tensor_f32(&src_name) {
                    let cache_name = format!("{prefix}.{suffix}");
                    if let Ok(bytes) = self.executor.cache_rmsnorm_gamma(&cache_name, &gamma) {
                        total_bytes += bytes;
                    }
                }
            }
        }

        total_bytes
    }

    /// Cache global weights: output norm + LM head.
    ///
    /// # Returns
    ///
    /// `(total_bytes, quantized_count)`
    fn cache_global_weights(&mut self) -> (usize, usize) {
        let mut total_bytes = 0;
        let mut quantized_count = 0;

        // Cache output norm
        total_bytes += self.cache_output_norm();

        // Cache LM head (may be quantized or F32)
        let (b, q) = self.cache_lm_head();
        total_bytes += b;
        quantized_count += q;

        (total_bytes, quantized_count)
    }

    /// Cache output norm weight.
    ///
    /// # Returns
    ///
    /// Bytes uploaded.
    fn cache_output_norm(&mut self) -> usize {
        let output_norm_patterns = [
            "model.norm.weight",
            "norm.weight",
            "transformer.ln_f.weight",
            "output_norm.weight",
        ];
        if let Ok(src_name) = self.model.find_tensor_name(&output_norm_patterns) {
            if let Ok(gamma) = self.model.get_tensor_f32(&src_name) {
                if let Ok(bytes) = self
                    .executor
                    .cache_rmsnorm_gamma("output_norm.gamma", &gamma)
                {
                    return bytes;
                }
            }
        }
        0
    }

    /// Cache LM head weight (may be quantized or F32).
    ///
    /// # Returns
    ///
    /// `(bytes_uploaded, quantized_count)`
    fn cache_lm_head(&mut self) -> (usize, usize) {
        let lm_head_patterns = [
            "lm_head.weight",
            "output.weight",
            "token_embd.weight", // GGUF (tied embeddings)
        ];
        let src_name = match self.model.find_tensor_name(&lm_head_patterns) {
            Ok(name) => name,
            Err(_) => return (0, 0),
        };
        let entry = match self.model.get_tensor(&src_name) {
            Some(e) => e.dtype.clone(),
            None => return (0, 0),
        };
        if let Some(qtype) = dtype_to_ggml_qtype(&entry) {
            // Quantized LM head
            self.cache_lm_head_quantized(&src_name, qtype)
        } else {
            // F32 LM head - store as quantized_weight_cache for compatibility
            // The forward path will handle F32 appropriately
            (self.cache_lm_head_f32(&src_name), 0)
        }
    }

    /// Cache quantized LM head weight.
    fn cache_lm_head_quantized(&mut self, src_name: &str, qtype: u32) -> (usize, usize) {
        if let Ok(bytes) = self.model.get_tensor_bytes(src_name) {
            if let Ok(size) = self.executor.load_quantized_weights_with_type(
                "output.weight",
                bytes,
                qtype,
            ) {
                return (size, 1);
            }
        }
        (0, 0)
    }

    /// Cache F32 LM head weight (uploaded as raw bytes with qtype 0).
    fn cache_lm_head_f32(&mut self, src_name: &str) -> usize {
        if let Ok(w) = self.model.get_tensor_f32(src_name) {
            // Upload F32 weights directly (no transpose needed for GEMV)
            // SAFETY: f32 slice to u8 view - valid because f32 has no padding,
            // alignment requirement of u8 is 1, and lifetime is preserved
            let w_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    w.as_ptr().cast::<u8>(),
                    w.len() * std::mem::size_of::<f32>(),
                )
            };
            // Use qtype 0 to indicate F32 (handled specially in forward)
            if let Ok(size) = self.executor.load_quantized_weights_with_type(
                "output.weight",
                w_bytes,
                0,
            ) {
                return size;
            }
        }
        0
    }

    /// Pre-cache all model weights on GPU using native quantized format.
    ///
    /// This uploads quantized weights (Q4K, Q6K, etc.) directly to GPU without
    /// CPU dequantization, enabling fused dequant+matmul kernels for maximum
    /// throughput (2x+ Ollama baseline per APR mandate).
    ///
    /// # Returns
    ///
    /// Total bytes uploaded to GPU.
    fn pre_cache_weights(&mut self) -> Result<()> {
        let (hidden_dim, num_layers, _num_heads, _num_kv_heads, intermediate_dim, _head_dim, kv_dim) =
            self.extract_model_dims();

        if hidden_dim == 0 || num_layers == 0 {
            return Ok(()); // Non-transformer model, nothing to cache
        }

        let mut total_bytes = 0usize;
        let mut quantized_count = 0usize;
        let mut f32_weight_count = 0usize;

        // Cache per-layer weights using GGUF naming convention
        // This matches build_indexed_weights() expectations
        for layer_idx in 0..num_layers {
            let (b, q, f) = self.cache_layer_projections(layer_idx);
            total_bytes += b;
            quantized_count += q;
            f32_weight_count += f;

            // PMAT-113: Cache fused QKV from APR import (PMAT-101)
            let (b, f) = self.cache_fused_qkv(layer_idx, hidden_dim, kv_dim);
            total_bytes += b;
            f32_weight_count += f;

            // Weights are loaded via GGUF-style names (blk.{}.attn_output.weight etc.)
            // in the first pass above. Biases are read directly from the model at
            // inference time. This avoids duplicate GPU memory for 1.5B F32 models.

            // Upload RMSNorm gamma weights (always F32)
            total_bytes += self.cache_layer_norms(layer_idx);
        }

        let (b, q) = self.cache_global_weights();
        total_bytes += b;
        quantized_count += q;

        // Build indexed weight lookup table for O(1) access during decode
        // This is the key optimization that enables fast token generation
        // GH-279: Always build + validate, even for F32-only models.
        // The validation catches missing architecture-required weights.
        {
            // R-01 (Meyer DbC): "unknown" — don't pretend unidentified model is LLaMA.
            let arch_name = self.model.metadata.model_type.as_deref().unwrap_or("unknown");
            let arch = crate::gguf::ArchConstraints::from_architecture(arch_name);
            self.executor
                .build_indexed_weights(num_layers, |i| format!("blk.{i}"), &arch)
                .map_err(|e| RealizarError::GpuError {
                    reason: format!(
                        "GH-279: Architecture weight validation failed: {e}"
                    ),
                })?;
            eprintln!(
                "[AprV2ModelCuda] Built indexed weights for {} layers (arch={})",
                num_layers, arch_name
            );
        }

        // Initialize workspace for zero-allocation forward pass
        if let Err(e) = self.executor.init_workspace(hidden_dim, intermediate_dim) {
            eprintln!("[AprV2ModelCuda] Warning: Could not init workspace: {e}");
        }

        // PMAT-113: Log both quantized and F32 weight counts
        eprintln!(
            "[AprV2ModelCuda] Pre-cached {} MB of weights on GPU ({} layers, {} quantized, {} F32 tensors)",
            total_bytes / (1024 * 1024),
            num_layers,
            quantized_count,
            f32_weight_count
        );

        Ok(())
    }
}

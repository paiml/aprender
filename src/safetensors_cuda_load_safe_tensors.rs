impl SafeTensorsCudaModel {
    /// Load SafeTensors model directly to GPU.
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to .safetensors file
    /// * `device_ordinal` - GPU device index (0 for first GPU)
    ///
    /// # Errors
    ///
    /// Returns error if file not found, config.json missing, or CUDA unavailable.
    pub fn load(model_path: &Path, device_ordinal: i32) -> Result<Self> {
        Self::load_with_max_seq_len(model_path, device_ordinal, 2048)
    }

    /// Load SafeTensors model with custom max sequence length.
    pub fn load_with_max_seq_len(
        model_path: &Path,
        device_ordinal: i32,
        max_seq_len: usize,
    ) -> Result<Self> {
        // 1. Load SafeTensors via mmap (F-PARSE-036)
        let st_model = MappedSafeTensorsModel::load(model_path)?;

        // 2. Load config.json (F-LOAD-063)
        let json_config = SafetensorsConfig::load_from_sibling(model_path).ok_or_else(|| {
            RealizarError::UnsupportedOperation {
                operation: "safetensors_cuda_load".to_string(),
                reason: "config.json not found (required for SafeTensors GPU inference)"
                    .to_string(),
            }
        })?;

        // 3. Extract config (F-LOAD-064, F-LOAD-065)
        let config = Self::extract_config(&json_config)?;

        // 3b. GH-279: Contract gate — validate architecture and dimensions
        let _proof = crate::contract_gate::validate_model_load_basic(
            &config.architecture,
            config.num_layers,
            config.hidden_dim,
            config.num_heads,
            config.num_kv_heads,
            config.intermediate_dim,
            config.vocab_size,
        )
        .map_err(crate::contract_gate::gate_error)?;

        // 3c. GH-279: Architecture completeness gate — validate all required
        // tensors exist in the SafeTensors file BEFORE GPU initialization.
        Self::validate_safetensors_completeness(&st_model, &config)?;

        // 4. Initialize CUDA executor (F-CUDA-011)
        let mut executor =
            CudaExecutor::new(device_ordinal).map_err(|e| RealizarError::UnsupportedOperation {
                operation: "CudaExecutor::new".to_string(),
                reason: format!("CUDA initialization failed: {e}"),
            })?;

        let device_name = executor
            .device_name()
            .unwrap_or_else(|_| "Unknown GPU".to_string());
        let memory_info = executor.memory_info().unwrap_or((0, 0));

        // GH-201 FIX: Check VRAM and select streaming mode
        let (free_vram, total_vram) = memory_info;
        let streaming_config = crate::cuda::StreamingConfig {
            hidden_dim: config.hidden_dim,
            num_layers: config.num_layers,
            num_heads: config.num_heads,
            num_kv_heads: config.num_kv_heads,
            vocab_size: config.vocab_size,
            intermediate_dim: config.intermediate_dim,
            max_seq_len,
        };

        let streaming_mode =
            match crate::cuda::check_vram_sufficient(free_vram, total_vram, &streaming_config) {
                Ok(crate::cuda::StreamingMode::FullCache) => false,
                Ok(crate::cuda::StreamingMode::LayerStreaming) => {
                    eprintln!(
                        "[GH-201] Using layer streaming mode (VRAM: {} MB free of {} MB)",
                        free_vram / (1024 * 1024),
                        total_vram / (1024 * 1024)
                    );
                    true
                },
                Err(msg) => {
                    return Err(RealizarError::UnsupportedOperation {
                        operation: "safetensors_cuda_load".to_string(),
                        reason: msg,
                    });
                },
            };

        // 5. Initialize GPU KV cache (F-PERF-085)
        let head_dim = config.hidden_dim / config.num_heads;
        executor
            .init_kv_cache_gpu(
                config.num_layers,
                config.num_heads,
                config.num_kv_heads,
                head_dim,
                max_seq_len,
            )
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "init_kv_cache_gpu".to_string(),
                reason: format!("GPU KV cache init failed: {e}"),
            })?;

        // 6. Set RoPE parameters
        // GH-329: Infer rope type from architecture, not hardcoded
        executor.set_rope_theta(config.rope_theta);
        executor.set_rope_type(crate::gguf::infer_rope_type(&config.architecture));

        // 7. Upload weights based on mode
        // GH-279: Full cache returns qk_norm_cache; streaming mode doesn't load QK norm
        let (embedding_cache, gamma_cache, qkv_bias_cache, o_bias_cache, qk_norm_loaded) =
            if streaming_mode {
                // GH-201: Streaming mode - only upload LM head and norms
                let (emb, gamma, qkv_bias, o_bias) =
                    Self::upload_weights_streaming(&mut executor, &st_model, &config)?;
                (emb, gamma, qkv_bias, o_bias, std::collections::HashMap::new())
            } else {
                // Full cache mode - upload all weights (including QK norm)
                Self::upload_weights(&mut executor, &st_model, &config)?
            };

        // Keep path for streaming mode (to reload weights on-demand)
        let model_path = if streaming_mode {
            Some(model_path.to_path_buf())
        } else {
            None
        };

        Ok(Self {
            executor,
            epsilon: config.eps,
            config,
            device_name,
            memory_info,
            kv_position: 0,
            embedding_cache,
            gamma_cache,
            qkv_bias_cache,
            o_bias_cache,
            qk_norm_cache: qk_norm_loaded,
            streaming_mode,
            model_path,
        })
    }

    /// GH-201 FIX: Estimate VRAM required for model weights and KV cache.
    ///
    /// SafeTensors/APR GPU path pre-caches ALL weights upfront (unlike GGUF streaming),
    /// which can cause OOM on GPUs with limited VRAM. This function estimates the
    /// total memory footprint so we can fail early with an actionable error message.
    ///
    /// Memory components:
    /// - LM head: hidden_dim × vocab_size × 4 bytes
    /// - Per layer (×num_layers):
    ///   - QKV weights: hidden_dim × (hidden_dim + 2×kv_dim) × 4
    ///   - O projection: hidden_dim × hidden_dim × 4
    ///   - FFN gate: intermediate_dim × hidden_dim × 4
    ///   - FFN up: intermediate_dim × hidden_dim × 4
    ///   - FFN down: hidden_dim × intermediate_dim × 4
    ///   - Norms: 2 × hidden_dim × 4 (attn + ffn)
    /// - KV cache: 2 × num_layers × max_seq_len × kv_dim × 4
    fn estimate_vram_bytes(config: &SafeTensorsCudaConfig, max_seq_len: usize) -> usize {
        let hidden_dim = config.hidden_dim;
        let num_layers = config.num_layers;
        let num_kv_heads = config.num_kv_heads;
        let num_heads = config.num_heads;
        let intermediate_dim = config.intermediate_dim;
        let vocab_size = config.vocab_size;
        let head_dim = hidden_dim / num_heads;
        let kv_dim = num_kv_heads * head_dim;

        // F32 = 4 bytes per element
        const F32_SIZE: usize = 4;

        // LM head (transposed: hidden_dim × vocab_size)
        let lm_head_bytes = hidden_dim * vocab_size * F32_SIZE;

        // Output norm gamma
        let output_norm_bytes = hidden_dim * F32_SIZE;

        // Per-layer weights
        let qkv_out_dim = hidden_dim + 2 * kv_dim;
        let per_layer_bytes = {
            // QKV (transposed: hidden_dim × qkv_out_dim)
            let qkv = hidden_dim * qkv_out_dim * F32_SIZE;
            // O projection (transposed: hidden_dim × hidden_dim)
            let o_proj = hidden_dim * hidden_dim * F32_SIZE;
            // FFN gate (transposed: hidden_dim × intermediate_dim)
            let ffn_gate = hidden_dim * intermediate_dim * F32_SIZE;
            // FFN up (transposed: hidden_dim × intermediate_dim)
            let ffn_up = hidden_dim * intermediate_dim * F32_SIZE;
            // FFN down (transposed: intermediate_dim × hidden_dim)
            let ffn_down = intermediate_dim * hidden_dim * F32_SIZE;
            // Attn + FFN norms (uploaded to rmsnorm_cache)
            let norms = 2 * hidden_dim * F32_SIZE;

            qkv + o_proj + ffn_gate + ffn_up + ffn_down + norms
        };

        let total_layer_bytes = num_layers * per_layer_bytes;

        // KV cache: 2 (K + V) × num_layers × max_seq_len × kv_dim
        let kv_cache_bytes = 2 * num_layers * max_seq_len * kv_dim * F32_SIZE;

        lm_head_bytes + output_norm_bytes + total_layer_bytes + kv_cache_bytes
    }

    /// Extract configuration from JSON config.
    fn extract_config(json: &SafetensorsConfig) -> Result<SafeTensorsCudaConfig> {
        let hidden_dim = json.hidden_size.ok_or_else(|| RealizarError::FormatError {
            reason: "config.json missing hidden_size".to_string(),
        })?;
        let num_layers = json
            .num_hidden_layers
            .ok_or_else(|| RealizarError::FormatError {
                reason: "config.json missing num_hidden_layers".to_string(),
            })?;
        let num_heads = json
            .num_attention_heads
            .ok_or_else(|| RealizarError::FormatError {
                reason: "config.json missing num_attention_heads".to_string(),
            })?;
        let vocab_size = json.vocab_size.ok_or_else(|| RealizarError::FormatError {
            reason: "config.json missing vocab_size".to_string(),
        })?;

        // GH-279: Derive architecture constraints for weight validation
        let arch_name = json.architecture();
        let arch_constraints = crate::gguf::ArchConstraints::from_architecture(&arch_name);
        // R-02 (Meyer DbC): rope_theta from config, or architecture-specific default.
        let rope_theta = json.rope_theta.unwrap_or_else(||
            crate::gguf::default_rope_theta_for_architecture(&arch_name));

        Ok(SafeTensorsCudaConfig {
            architecture: arch_name,
            hidden_dim,
            num_layers,
            num_heads,
            num_kv_heads: json.num_kv_heads(),
            vocab_size,
            intermediate_dim: json.intermediate_size.unwrap_or(hidden_dim * 4),
            context_length: json.max_position_embeddings.unwrap_or(0),
            rope_theta,
            eps: json.rms_norm_eps.unwrap_or(1e-6),
            tie_word_embeddings: json.tie_word_embeddings.unwrap_or(false),
            has_qk_norm: arch_constraints.has_qk_norm,
            has_bias: arch_constraints.has_bias,
            eos_token_id: json.eos_token_id.map(|v| v as u32),
        })
    }

    /// GH-279/316: Validate a single layer has all required tensors via contract.
    fn validate_layer_completeness(
        st_model: &MappedSafeTensorsModel,
        arch: &str,
        layer_idx: usize,
        config: &SafeTensorsCudaConfig,
        missing: &mut Vec<String>,
    ) {
        use crate::tensor_names::{
            has_layer, has_fused,
            LayerTensorRole, FusedTensorRole,
        };

        let mut check = |role: LayerTensorRole, label: &str| {
            if !has_layer(st_model, arch, layer_idx, role) {
                missing.push(format!("layer {layer_idx} {label}"));
            }
        };

        check(LayerTensorRole::AttnNormWeight, "AttnNormWeight");
        check(LayerTensorRole::OProjWeight, "OProjWeight");
        check(LayerTensorRole::FfnNormWeight, "FfnNormWeight");
        check(LayerTensorRole::FfnUpWeight, "FfnUpWeight");
        check(LayerTensorRole::FfnDownWeight, "FfnDownWeight");

        // QKV: either fused or separate
        if !has_fused(st_model, arch, layer_idx, FusedTensorRole::FusedQkv) {
            check(LayerTensorRole::QProjWeight, "QProjWeight");
            check(LayerTensorRole::KProjWeight, "KProjWeight");
            check(LayerTensorRole::VProjWeight, "VProjWeight");
        }

        // Architecture-conditional roles
        if config.has_qk_norm {
            check(LayerTensorRole::AttnQNormWeight, "AttnQNormWeight");
            check(LayerTensorRole::AttnKNormWeight, "AttnKNormWeight");
        }
        if config.has_bias {
            check(LayerTensorRole::QProjBias, "QProjBias");
            check(LayerTensorRole::KProjBias, "KProjBias");
            check(LayerTensorRole::VProjBias, "VProjBias");
        }
    }

    /// GH-279: Validate that all architecture-required tensors exist in the SafeTensors file.
    ///
    /// GH-316: Uses contract-driven tensor name resolution instead of hardcoded LLaMA names.
    /// This runs BEFORE GPU initialization so we fail fast with a clear error
    /// instead of discovering missing tensors halfway through weight upload.
    fn validate_safetensors_completeness(
        st_model: &MappedSafeTensorsModel,
        config: &SafeTensorsCudaConfig,
    ) -> Result<()> {
        use crate::tensor_names::{has_global, GlobalTensorRole};
        let arch = &config.architecture;
        let mut missing = Vec::new();

        // Global tensors
        if !has_global(st_model, arch, GlobalTensorRole::Embedding) {
            missing.push("Embedding".to_string());
        }
        if !has_global(st_model, arch, GlobalTensorRole::OutputNormWeight) {
            missing.push("OutputNormWeight".to_string());
        }

        for layer_idx in 0..config.num_layers {
            Self::validate_layer_completeness(st_model, arch, layer_idx, config, &mut missing);
        }

        if !missing.is_empty() {
            let first_few: Vec<&str> = missing.iter().take(5).map(String::as_str).collect();
            return Err(RealizarError::UnsupportedOperation {
                operation: "validate_safetensors_completeness".to_string(),
                reason: format!(
                    "GH-279: SafeTensors model missing {} required tensor(s) for architecture '{}'. \
                     First missing: [{}]",
                    missing.len(),
                    config.architecture,
                    first_few.join(", ")
                ),
            });
        }

        Ok(())
    }

    /// GH-316: Helper to map CUDA errors to `RealizarError`.
    fn cuda_err(op: &str, layer_idx: usize, what: &str, e: impl std::fmt::Display) -> RealizarError {
        RealizarError::UnsupportedOperation {
            operation: op.to_string(),
            reason: format!("Failed to upload layer {layer_idx} {what}: {e}"),
        }
    }

    /// GH-316: Split fused QKV tensor into separate Q, K, V components.
    fn split_fused_qkv(
        fused_qkv: &[f32],
        hidden_dim: usize,
        kv_dim: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let total_out = hidden_dim + kv_dim + kv_dim;
        if fused_qkv.len() >= total_out * hidden_dim {
            // Fused weight is [total_out, hidden_dim] — split by rows
            let q_end = hidden_dim * hidden_dim;
            let k_end = q_end + kv_dim * hidden_dim;
            let v_end = k_end + kv_dim * hidden_dim;
            (
                fused_qkv[..q_end].to_vec(),
                fused_qkv[q_end..k_end].to_vec(),
                fused_qkv[k_end..v_end].to_vec(),
            )
        } else {
            // Fallback: treat as concat of bias-sized vectors
            let k_end = hidden_dim + kv_dim;
            (
                fused_qkv[..hidden_dim].to_vec(),
                fused_qkv[hidden_dim..k_end].to_vec(),
                fused_qkv[k_end..].to_vec(),
            )
        }
    }

    /// GH-316: Upload a single layer's weights to GPU using contract-driven tensor names.
    #[allow(clippy::too_many_arguments)]
    fn upload_layer_weights(
        executor: &mut CudaExecutor,
        st_model: &MappedSafeTensorsModel,
        arch: &str,
        layer_idx: usize,
        hidden_dim: usize,
        kv_dim: usize,
        intermediate_dim: usize,
        gamma_cache: &mut std::collections::HashMap<String, Vec<f32>>,
        qkv_bias_cache: &mut std::collections::HashMap<String, Vec<f32>>,
        o_bias_cache: &mut std::collections::HashMap<String, Vec<f32>>,
        qk_norm_cache: &mut std::collections::HashMap<String, Vec<f32>>,
    ) -> Result<()> {
        use crate::tensor_names::{
            resolve_layer, resolve_layer_optional, resolve_fused,
            LayerTensorRole, FusedTensorRole,
        };

        // Attention norm
        let attn_norm = resolve_layer(st_model, arch, layer_idx, LayerTensorRole::AttnNormWeight)?;
        gamma_cache.insert(format!("attn.{layer_idx}"), attn_norm.clone());
        executor
            .cache_rmsnorm_gamma(&format!("blk.{layer_idx}.attn_norm.gamma"), &attn_norm)
            .map_err(|e| Self::cuda_err("cache_rmsnorm_gamma", layer_idx, "attn_norm", e))?;

        // QKV: try fused first, then separate
        let (q, k, v) = if let Some(fused) = resolve_fused(st_model, arch, layer_idx, FusedTensorRole::FusedQkv) {
            Self::split_fused_qkv(&fused, hidden_dim, kv_dim)
        } else {
            (
                resolve_layer(st_model, arch, layer_idx, LayerTensorRole::QProjWeight)?,
                resolve_layer(st_model, arch, layer_idx, LayerTensorRole::KProjWeight)?,
                resolve_layer(st_model, arch, layer_idx, LayerTensorRole::VProjWeight)?,
            )
        };
        let qkv = Self::concat_qkv_transposed(&q, &k, &v, hidden_dim, kv_dim);
        executor
            .load_weights(&format!("blk.{layer_idx}.attn_qkv"), &qkv)
            .map_err(|e| Self::cuda_err("load_weights", layer_idx, "qkv", e))?;

        // QKV bias
        let q_bias = resolve_layer_optional(st_model, arch, layer_idx, LayerTensorRole::QProjBias)
            .unwrap_or_else(|| vec![0.0f32; hidden_dim]);
        let k_bias = resolve_layer_optional(st_model, arch, layer_idx, LayerTensorRole::KProjBias)
            .unwrap_or_else(|| vec![0.0f32; kv_dim]);
        let v_bias = resolve_layer_optional(st_model, arch, layer_idx, LayerTensorRole::VProjBias)
            .unwrap_or_else(|| vec![0.0f32; kv_dim]);
        let mut qkv_bias = Vec::with_capacity(hidden_dim + 2 * kv_dim);
        qkv_bias.extend_from_slice(&q_bias);
        qkv_bias.extend_from_slice(&k_bias);
        qkv_bias.extend_from_slice(&v_bias);
        qkv_bias_cache.insert(format!("qkv_bias.{layer_idx}"), qkv_bias);

        // QK norm (Qwen3)
        if let Some(q_norm) = resolve_layer_optional(st_model, arch, layer_idx, LayerTensorRole::AttnQNormWeight) {
            qk_norm_cache.insert(format!("q_norm.{layer_idx}"), q_norm.clone());
            executor
                .cache_rmsnorm_gamma(&format!("blk.{layer_idx}.attn_q_norm.gamma"), &q_norm)
                .map_err(|e| Self::cuda_err("cache_rmsnorm_gamma", layer_idx, "q_norm", e))?;
        }
        if let Some(k_norm) = resolve_layer_optional(st_model, arch, layer_idx, LayerTensorRole::AttnKNormWeight) {
            qk_norm_cache.insert(format!("k_norm.{layer_idx}"), k_norm.clone());
            executor
                .cache_rmsnorm_gamma(&format!("blk.{layer_idx}.attn_k_norm.gamma"), &k_norm)
                .map_err(|e| Self::cuda_err("cache_rmsnorm_gamma", layer_idx, "k_norm", e))?;
        }

        // Output projection
        let o_raw = resolve_layer(st_model, arch, layer_idx, LayerTensorRole::OProjWeight)?;
        let o = Self::transpose_for_gemm(&o_raw, hidden_dim, hidden_dim);
        executor
            .load_weights(&format!("blk.{layer_idx}.attn_output"), &o)
            .map_err(|e| Self::cuda_err("load_weights", layer_idx, "attn_output", e))?;

        // O projection bias (rare)
        if let Ok(o_bias) = st_model.get_tensor_auto(&format!("model.layers.{layer_idx}.self_attn.o_proj.bias")) {
            o_bias_cache.insert(format!("o_bias.{layer_idx}"), o_bias);
        }

        // FFN norm
        let ffn_norm = resolve_layer(st_model, arch, layer_idx, LayerTensorRole::FfnNormWeight)?;
        gamma_cache.insert(format!("ffn.{layer_idx}"), ffn_norm.clone());
        executor
            .cache_rmsnorm_gamma(&format!("blk.{layer_idx}.ffn_norm.gamma"), &ffn_norm)
            .map_err(|e| Self::cuda_err("cache_rmsnorm_gamma", layer_idx, "ffn_norm", e))?;

        // FFN gate (optional — GPT-2/Phi-2 don't have it)
        if let Some(gate_raw) = resolve_layer_optional(st_model, arch, layer_idx, LayerTensorRole::FfnGateWeight) {
            let gate = Self::transpose_for_gemm(&gate_raw, intermediate_dim, hidden_dim);
            executor
                .load_weights(&format!("blk.{layer_idx}.ffn_gate"), &gate)
                .map_err(|e| Self::cuda_err("load_weights", layer_idx, "ffn_gate", e))?;
        }

        // FFN up
        let up_raw = resolve_layer(st_model, arch, layer_idx, LayerTensorRole::FfnUpWeight)?;
        let up = Self::transpose_for_gemm(&up_raw, intermediate_dim, hidden_dim);
        executor
            .load_weights(&format!("blk.{layer_idx}.ffn_up"), &up)
            .map_err(|e| Self::cuda_err("load_weights", layer_idx, "ffn_up", e))?;

        // FFN down
        let down_raw = resolve_layer(st_model, arch, layer_idx, LayerTensorRole::FfnDownWeight)?;
        let down = Self::transpose_for_gemm(&down_raw, hidden_dim, intermediate_dim);
        executor
            .load_weights(&format!("blk.{layer_idx}.ffn_down"), &down)
            .map_err(|e| Self::cuda_err("load_weights", layer_idx, "ffn_down", e))?;

        Ok(())
    }

    /// Upload all model weights to GPU.
    ///
    /// GH-316: Uses contract-driven tensor name resolution instead of hardcoded LLaMA names.
    /// Returns (embedding_table, gamma_cache, qkv_bias_cache, o_bias_cache) - embedding kept on CPU
    /// for token lookup, gamma_cache kept on CPU for RMS norm operations, bias caches for attention.
    #[allow(clippy::type_complexity)]
    #[allow(clippy::type_complexity)]
    fn upload_weights(
        executor: &mut CudaExecutor,
        st_model: &MappedSafeTensorsModel,
        config: &SafeTensorsCudaConfig,
    ) -> Result<(
        Vec<f32>,
        std::collections::HashMap<String, Vec<f32>>,
        std::collections::HashMap<String, Vec<f32>>,
        std::collections::HashMap<String, Vec<f32>>,
        std::collections::HashMap<String, Vec<f32>>,
    )> {
        use crate::tensor_names::{
            resolve_global, resolve_global_optional, has_global,
            resolve_layer, resolve_layer_optional, resolve_fused,
            GlobalTensorRole, LayerTensorRole, FusedTensorRole,
        };

        let hidden_dim = config.hidden_dim;
        let num_layers = config.num_layers;
        let num_heads = config.num_heads;
        let num_kv_heads = config.num_kv_heads;
        let intermediate_dim = config.intermediate_dim;
        let vocab_size = config.vocab_size;
        let head_dim = hidden_dim / num_heads;
        let kv_dim = num_kv_heads * head_dim;
        let arch = &config.architecture;

        // Gamma cache for CPU RMS norm
        let mut gamma_cache = std::collections::HashMap::new();
        // PMAT-120 FIX: Bias caches for attention projections
        let mut qkv_bias_cache = std::collections::HashMap::new();
        let mut o_bias_cache = std::collections::HashMap::new();
        // GH-279: QK norm weight cache (Qwen3 per-head RMSNorm)
        let mut qk_norm_cache = std::collections::HashMap::new();

        // GH-316: Embedding table via contract (keep on CPU for token lookup)
        let embedding = resolve_global(st_model, arch, GlobalTensorRole::Embedding)?;

        // GH-316: Output norm via contract - upload to rmsnorm_cache AND keep CPU copy
        let output_norm = resolve_global(st_model, arch, GlobalTensorRole::OutputNormWeight)?;
        gamma_cache.insert("output".to_string(), output_norm.clone());
        executor.preload_output_norm(&output_norm).map_err(|e| {
            RealizarError::UnsupportedOperation {
                operation: "preload_output_norm".to_string(),
                reason: format!("Failed to upload output_norm: {e}"),
            }
        })?;

        // GH-316: LM head via contract (may be tied to embeddings)
        // F-GT-002 FIX: Check tie_word_embeddings config FIRST, not just tensor existence
        let lm_head = if config.tie_word_embeddings {
            Self::transpose_for_gemm(&embedding, vocab_size, hidden_dim)
        } else if has_global(st_model, arch, GlobalTensorRole::LmHead) {
            let raw = resolve_global(st_model, arch, GlobalTensorRole::LmHead)?;
            Self::transpose_for_gemm(&raw, vocab_size, hidden_dim)
        } else {
            // Fallback: assume tied if no lm_head tensor exists
            Self::transpose_for_gemm(&embedding, vocab_size, hidden_dim)
        };
        executor.load_weights("lm_head", &lm_head).map_err(|e| {
            RealizarError::UnsupportedOperation {
                operation: "load_weights".to_string(),
                reason: format!("Failed to upload lm_head: {e}"),
            }
        })?;

        // Per-layer weights (F-LOAD-057, F-LOAD-061, F-LOAD-062)
        for layer_idx in 0..num_layers {
            Self::upload_layer_weights(
                executor, st_model, arch, layer_idx,
                hidden_dim, kv_dim, intermediate_dim,
                &mut gamma_cache, &mut qkv_bias_cache,
                &mut o_bias_cache, &mut qk_norm_cache,
            )?;
        }

        // GH-279: Validate QK norm completeness — if architecture requires it,
        // ALL layers must have Q and K norm weights loaded
        if config.has_qk_norm && qk_norm_cache.len() < 2 * num_layers {
            return Err(RealizarError::UnsupportedOperation {
                operation: "upload_weights".to_string(),
                reason: format!(
                    "GH-279: Architecture requires QK norm but only {}/{} norm weights found. \
                     Expected q_norm + k_norm for all {} layers.",
                    qk_norm_cache.len(),
                    2 * num_layers,
                    num_layers
                ),
            });
        }

        Ok((embedding, gamma_cache, qkv_bias_cache, o_bias_cache, qk_norm_cache))
    }
}

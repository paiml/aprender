
impl SafetensorsToAprConverter {
    /// Convert SafeTensors model to APR Transformer
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to model.safetensors file
    ///
    /// # Returns
    ///
    /// `AprTransformer` with F32 weights ready for inference
    ///
    /// # Errors
    ///
    /// Returns error if SafeTensors file, config.json, or required tensors are missing
    pub fn convert(model_path: &Path) -> Result<ValidatedAprTransformer> {
        // Load SafeTensors model using mmap for zero-copy access (T-QA-020)
        // This is critical for fast model loading - mmap is O(1) regardless of file size
        let st_model = MappedSafeTensorsModel::load(model_path)?;

        // Load config.json (required for architecture info)
        let config = SafetensorsConfig::load_from_sibling(model_path).ok_or_else(|| {
            RealizarError::UnsupportedOperation {
                operation: "safetensors_convert".to_string(),
                reason: "config.json not found (required for SafeTensors inference)".to_string(),
            }
        })?;

        Self::convert_from_source(&st_model, &config)
    }

    /// Convert a sharded SafeTensors model to APR Transformer (GH-213)
    ///
    /// # Arguments
    ///
    /// * `sharded` - Loaded sharded model (from index.json)
    /// * `config` - Model config.json
    ///
    /// # Errors
    ///
    /// Returns error if required tensors are missing from any shard
    #[cfg(not(target_arch = "wasm32"))]
    pub fn convert_sharded(
        sharded: &ShardedSafeTensorsModel,
        config: &SafetensorsConfig,
    ) -> Result<ValidatedAprTransformer> {
        Self::convert_from_source(sharded, config)
    }

    /// Core conversion logic shared between single-file and sharded paths
    fn convert_from_source<S: TensorSource>(
        source: &S,
        config: &SafetensorsConfig,
    ) -> Result<ValidatedAprTransformer> {
        // Extract architecture parameters
        let hidden_dim = config
            .hidden_size
            .ok_or_else(|| RealizarError::FormatError {
                reason: "config.json missing hidden_size".to_string(),
            })?;
        let num_layers = config
            .num_hidden_layers
            .ok_or_else(|| RealizarError::FormatError {
                reason: "config.json missing num_hidden_layers".to_string(),
            })?;
        let num_heads = config
            .num_attention_heads
            .ok_or_else(|| RealizarError::FormatError {
                reason: "config.json missing num_attention_heads".to_string(),
            })?;
        let num_kv_heads = config.num_kv_heads();
        let vocab_size = config
            .vocab_size
            .ok_or_else(|| RealizarError::FormatError {
                reason: "config.json missing vocab_size".to_string(),
            })?;
        let intermediate_dim = config.intermediate_size.unwrap_or(hidden_dim * 4);
        let context_length = config.max_position_embeddings.unwrap_or(0);
        let architecture = config.architecture();
        // R-02 (Meyer DbC): rope_theta from config, or architecture-specific default.
        let rope_theta = config.rope_theta.unwrap_or_else(||
            crate::gguf::default_rope_theta_for_architecture(&architecture));
        let eps = config.rms_norm_eps.unwrap_or(1e-6);

        // Phase 2: Validate model dimensions at construction boundary.
        // Non-fatal: logs a warning for invalid configs but allows conversion to
        // proceed. The final ValidatedAprTransformer::validate() at the end is
        // the hard gate. This catches obvious dimension errors early.
        if let Err(e) = crate::gguf::ValidatedModelConfig::from_safetensors_config(config) {
            eprintln!(
                "[Phase2-WARN] SafeTensors config validation: {e} — proceeding with conversion"
            );
        }

        // GH-278: Log Qwen3.5 detection with hybrid attention info
        if config.is_hybrid_attention() {
            let layer_count = config.layer_types.as_ref().map_or(0, Vec::len);
            let linear_count = config.layer_types.as_ref().map_or(0, |t| {
                t.iter()
                    .filter(|l| *l == "linear" || *l == "linear_attention")
                    .count()
            });
            eprintln!(
                "[GH-278] Hybrid attention model detected: {}/{} linear layers, head_dim={:?}",
                linear_count,
                layer_count,
                config.head_dim,
            );
        }

        // Build transformer config
        let apr_config = AprTransformerConfig {
            architecture,
            hidden_dim,
            num_layers,
            num_heads,
            num_kv_heads,
            vocab_size,
            intermediate_dim,
            context_length,
            rope_theta,
            eps,
            eos_token_id: config.eos_token_id,
            // GH-278: Hybrid attention fields
            explicit_head_dim: config.head_dim,
            layer_types: config.layer_types.clone(),
            linear_key_head_dim: config.linear_key_head_dim,
            linear_value_head_dim: config.linear_value_head_dim,
            linear_num_key_heads: config.linear_num_key_heads,
            linear_num_value_heads: config.linear_num_value_heads,
            linear_conv_kernel_dim: config.linear_conv_kernel_dim,
            // ALB-010: MoE fields
            num_experts: config.num_experts,
            num_experts_per_tok: config.num_experts_per_tok,
            expert_intermediate_size: config.moe_intermediate_size,
        };

        // ALB-010: Detect ConditionalGeneration wrapper prefix
        let model_prefix = Self::detect_model_prefix(source);
        let is_moe = config.num_experts.is_some();

        // Extract embeddings — use detected prefix for ConditionalGeneration wrapper
        let embed_name = format!("{model_prefix}.embed_tokens.weight");
        let token_embedding = Self::get_tensor_with_fallback_generic(
            source,
            &embed_name,
            "token_embd.weight",
        )?;

        // Extract output norm
        let norm_name = format!("{model_prefix}.norm.weight");
        let output_norm_weight = Self::get_tensor_with_fallback_generic(
            source,
            &norm_name,
            "output_norm.weight",
        )?;

        // F-GT-002 FIX: Check tie_word_embeddings config FIRST, not just tensor existence
        // When tie_word_embeddings=true, HuggingFace may store a placeholder lm_head.weight
        // that's all zeros - we MUST use the embedding matrix instead!
        let use_tied_embeddings = config.tie_word_embeddings.unwrap_or(false);

        let lm_head_weight = if use_tied_embeddings {
            Self::transpose_weight(&token_embedding, vocab_size, hidden_dim)
        } else if Self::has_tensor_with_fallback_generic(source, "lm_head.weight", "output.weight")
        {
            let raw =
                Self::get_tensor_with_fallback_generic(source, "lm_head.weight", "output.weight")?;
            Self::transpose_weight(&raw, vocab_size, hidden_dim)
        } else {
            // Fallback: assume tied if no lm_head tensor exists
            Self::transpose_weight(&token_embedding, vocab_size, hidden_dim)
        };

        // Extract layers
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            // GH-278: Dispatch to linear layer extractor for Gated Delta Net layers
            let is_linear = config
                .layer_types
                .as_ref()
                .and_then(|lt| lt.get(i))
                .is_some_and(|t| t == "linear" || t == "linear_attention");

            let mut layer = if is_linear {
                Self::extract_linear_layer_generic(
                    source,
                    i,
                    hidden_dim,
                    intermediate_dim,
                    config,
                    &model_prefix,
                )?
            } else {
                Self::extract_layer_generic_with_prefix(
                    source,
                    i,
                    hidden_dim,
                    num_heads,
                    num_kv_heads,
                    intermediate_dim,
                    &model_prefix,
                )?
            };

            // ALB-010: Load MoE weights if this is an MoE model
            if is_moe {
                Self::load_moe_weights(source, i, &model_prefix, config, hidden_dim, &mut layer)?;
            }

            layers.push(layer);
        }

        let transformer = AprTransformer {
            config: apr_config,
            token_embedding,
            layers,
            output_norm_weight,
            output_norm_bias: None,
            lm_head_weight,
            lm_head_bias: None,
            q4k_layers: None,
            lm_head_weight_q6k: None,
            lm_head_weight_q4k: None,
        };

        // PMAT-235: Validate ALL tensors at construction time
        ValidatedAprTransformer::validate(transformer).map_err(Into::into)
    }

    /// Extract a single transformer layer from SafeTensors (MappedSafeTensorsModel)
    #[allow(dead_code)]
    fn extract_layer(
        st_model: &MappedSafeTensorsModel,
        layer_idx: usize,
        hidden_dim: usize,
        num_heads: usize,
        num_kv_heads: usize,
        intermediate_dim: usize,
    ) -> Result<AprTransformerLayer> {
        Self::extract_layer_generic(
            st_model,
            layer_idx,
            hidden_dim,
            num_heads,
            num_kv_heads,
            intermediate_dim,
        )
    }

    /// Extract a single transformer layer from any TensorSource (GH-213)
    fn extract_layer_generic<S: TensorSource>(
        source: &S,
        layer_idx: usize,
        hidden_dim: usize,
        num_heads: usize,
        num_kv_heads: usize,
        intermediate_dim: usize,
    ) -> Result<AprTransformerLayer> {
        // Support both HuggingFace and GGUF naming conventions
        let hf_prefix = format!("model.layers.{layer_idx}");
        let gguf_prefix = format!("blk.{layer_idx}");

        let attn_norm_weight = Self::get_tensor_with_fallback_generic(
            source,
            &format!("{hf_prefix}.input_layernorm.weight"),
            &format!("{gguf_prefix}.attn_norm.weight"),
        )?;

        let q_weight = Self::get_tensor_with_fallback_generic(
            source,
            &format!("{hf_prefix}.self_attn.q_proj.weight"),
            &format!("{gguf_prefix}.attn_q.weight"),
        )?;
        let k_weight = Self::get_tensor_with_fallback_generic(
            source,
            &format!("{hf_prefix}.self_attn.k_proj.weight"),
            &format!("{gguf_prefix}.attn_k.weight"),
        )?;
        let v_weight = Self::get_tensor_with_fallback_generic(
            source,
            &format!("{hf_prefix}.self_attn.v_proj.weight"),
            &format!("{gguf_prefix}.attn_v.weight"),
        )?;

        let head_dim = hidden_dim / num_heads;
        let kv_dim = head_dim * num_kv_heads;
        let qkv_weight =
            Self::concat_qkv_transposed(&q_weight, &k_weight, &v_weight, hidden_dim, kv_dim);

        // QKV bias (optional)
        let qkv_bias = Self::try_concat_qkv_bias_dual_generic(
            source,
            &hf_prefix,
            &gguf_prefix,
            hidden_dim,
            kv_dim,
        );

        let attn_output_raw = Self::get_tensor_with_fallback_generic(
            source,
            &format!("{hf_prefix}.self_attn.o_proj.weight"),
            &format!("{gguf_prefix}.attn_output.weight"),
        )?;
        let attn_output_weight = Self::transpose_weight(&attn_output_raw, hidden_dim, hidden_dim);

        let ffn_norm_weight = Self::get_tensor_with_fallback_generic(
            source,
            &format!("{hf_prefix}.post_attention_layernorm.weight"),
            &format!("{gguf_prefix}.ffn_norm.weight"),
        )?;

        let ffn_gate_raw = Self::get_tensor_with_fallback_generic(
            source,
            &format!("{hf_prefix}.mlp.gate_proj.weight"),
            &format!("{gguf_prefix}.ffn_gate.weight"),
        )?;
        let ffn_gate_weight = Self::transpose_weight(&ffn_gate_raw, intermediate_dim, hidden_dim);

        let ffn_up_raw = Self::get_tensor_with_fallback_generic(
            source,
            &format!("{hf_prefix}.mlp.up_proj.weight"),
            &format!("{gguf_prefix}.ffn_up.weight"),
        )?;
        let ffn_up_weight = Self::transpose_weight(&ffn_up_raw, intermediate_dim, hidden_dim);

        let ffn_down_raw = Self::get_tensor_with_fallback_generic(
            source,
            &format!("{hf_prefix}.mlp.down_proj.weight"),
            &format!("{gguf_prefix}.ffn_down.weight"),
        )?;
        let ffn_down_weight = Self::transpose_weight(&ffn_down_raw, hidden_dim, intermediate_dim);

        Ok(AprTransformerLayer {
            attn_norm_weight,
            attn_norm_bias: None,
            qkv_weight,
            qkv_bias,
            attn_output_weight,
            attn_output_bias: None,
            ffn_gate_weight: Some(ffn_gate_weight),
            ffn_gate_bias: None,
            ffn_up_weight,
            ffn_up_bias: None,
            ffn_down_weight,
            ffn_down_bias: None,
            ffn_norm_weight: Some(ffn_norm_weight),
            ffn_norm_bias: None,
            // GH-279: QK norm weights (Qwen3 per-head RMSNorm)
            attn_q_norm_weight: source
                .get_tensor_auto(&format!("{hf_prefix}.self_attn.q_norm.weight"))
                .or_else(|_| source.get_tensor_auto(&format!("{gguf_prefix}.attn_q_norm.weight")))
                .ok(),
            attn_k_norm_weight: source
                .get_tensor_auto(&format!("{hf_prefix}.self_attn.k_norm.weight"))
                .or_else(|_| source.get_tensor_auto(&format!("{gguf_prefix}.attn_k_norm.weight")))
                .ok(),
            // GH-278: Linear attention weights extracted separately in extract_linear_layer_generic
            linear_attn_z_weight: None,
            linear_attn_b_weight: None,
            linear_attn_a_weight: None,
            linear_attn_conv1d_weight: None,
            linear_attn_a_log: None,
            linear_attn_dt_bias: None,
            linear_attn_norm_weight: None,
            // ALB-010: MoE weights loaded separately via load_moe_weights
            moe_gate_weight: None,
            moe_expert_gate_up: None,
            moe_expert_down: None,
            moe_shared_gate: None,
            moe_shared_up: None,
            moe_shared_down: None,
            moe_shared_expert_gate_weight: None,
        })
    }

    /// GH-278 + ALB-010: Extract a single Gated Delta Net (linear attention) transformer layer
    ///
    /// Supports two naming conventions:
    /// 1. Qwen3.5 non-MoE: `self_attn.in_proj_qkvz` (combined QKVZ) + `self_attn.in_proj_ba` (combined BA)
    /// 2. Qwen3.5-35B-A3B MoE: `linear_attn.in_proj_qkv` + separate `in_proj_z`, `in_proj_a`, `in_proj_b`
    fn extract_linear_layer_generic<S: TensorSource>(
        source: &S,
        layer_idx: usize,
        hidden_dim: usize,
        intermediate_dim: usize,
        config: &SafetensorsConfig,
        model_prefix: &str,
    ) -> Result<AprTransformerLayer> {
        let hf_prefix = format!("{model_prefix}.layers.{layer_idx}");

        // --- Attention layer norm (same as standard layers) ---
        let attn_norm_weight = Self::get_tensor_with_fallback_generic(
            source,
            &format!("{hf_prefix}.input_layernorm.weight"),
            &format!("blk.{layer_idx}.attn_norm.weight"),
        )?;

        // Compute split dimensions from config
        let key_head_dim = config.linear_key_head_dim.unwrap_or(128);
        let value_head_dim = config.linear_value_head_dim.unwrap_or(128);
        let num_key_heads = config.linear_num_key_heads.unwrap_or(16);
        let num_value_heads = config.linear_num_value_heads.unwrap_or(32);
        let key_dim = num_key_heads * key_head_dim;
        let value_dim = num_value_heads * value_head_dim;

        // --- GDN projections: try both naming conventions ---
        // Convention 1: combined `self_attn.in_proj_qkvz` + `self_attn.in_proj_ba`
        // Convention 2: separate `linear_attn.in_proj_qkv` + `linear_attn.in_proj_z/a/b`
        let combined_qkvz_name = format!("{hf_prefix}.self_attn.in_proj_qkvz.weight");
        let separate_qkv_name = format!("{hf_prefix}.linear_attn.in_proj_qkv.weight");

        let (qkv_weight, z_weight, b_weight, a_weight, attn_sub) =
            if source.has_tensor(&combined_qkvz_name) {
                // Convention 1: combined QKVZ + combined BA
                let in_proj_qkvz = source.get_tensor_auto(&combined_qkvz_name)?;
                let qkvz_out_dim = 2 * key_dim + 2 * value_dim;
                let expected_qkvz = qkvz_out_dim * hidden_dim;
                if in_proj_qkvz.len() != expected_qkvz {
                    return Err(RealizarError::FormatError {
                        reason: format!(
                            "GH-278: in_proj_qkvz size mismatch at layer {layer_idx}: \
                             expected {expected_qkvz}, got {}",
                            in_proj_qkvz.len()
                        ),
                    });
                }
                let q_end = key_dim * hidden_dim;
                let k_end = q_end + key_dim * hidden_dim;
                let v_end = k_end + value_dim * hidden_dim;
                let qkv = Self::concat_qkv(&in_proj_qkvz[..q_end], &in_proj_qkvz[q_end..k_end], &in_proj_qkvz[k_end..v_end]);
                let z = in_proj_qkvz[v_end..].to_vec();

                let in_proj_ba = source.get_tensor_auto(&format!("{hf_prefix}.self_attn.in_proj_ba.weight"))?;
                let ba_split = num_value_heads * hidden_dim;
                let b = in_proj_ba[..ba_split].to_vec();
                let a = in_proj_ba[ba_split..].to_vec();

                (qkv, z, b, a, "self_attn")
            } else {
                // Convention 2: separate projections (Qwen3.5-35B-A3B)
                let in_proj_qkv = source.get_tensor_auto(&separate_qkv_name)?;
                // in_proj_qkv: [Q(key_dim) + K(key_dim) + V(value_dim), hidden_dim]
                let qkv_out_dim = 2 * key_dim + value_dim;
                let expected_qkv = qkv_out_dim * hidden_dim;
                if in_proj_qkv.len() != expected_qkv {
                    return Err(RealizarError::FormatError {
                        reason: format!(
                            "ALB-010: in_proj_qkv size mismatch at layer {layer_idx}: \
                             expected {expected_qkv}, got {}",
                            in_proj_qkv.len()
                        ),
                    });
                }
                let qkv = in_proj_qkv;
                let z = source.get_tensor_auto(&format!("{hf_prefix}.linear_attn.in_proj_z.weight"))?;
                let b = source.get_tensor_auto(&format!("{hf_prefix}.linear_attn.in_proj_b.weight"))?;
                let a = source.get_tensor_auto(&format!("{hf_prefix}.linear_attn.in_proj_a.weight"))?;

                (qkv, z, b, a, "linear_attn")
            };

        // out_proj: [hidden_dim, value_dim] — GDN uses out_proj, not o_proj
        let out_proj_raw = source
            .get_tensor_auto(&format!("{hf_prefix}.{attn_sub}.out_proj.weight"))?;
        let attn_output_weight = Self::transpose_weight(&out_proj_raw, hidden_dim, value_dim);

        // Conv1D weight: HF stores as [conv_dim, 1, kernel_size], squeeze middle dim
        let conv1d_weight = source
            .get_tensor_auto(&format!("{hf_prefix}.{attn_sub}.conv1d.weight"))?;

        // A_log: [num_v_heads] — parameter, no .weight suffix
        let a_log = source
            .get_tensor_auto(&format!("{hf_prefix}.{attn_sub}.A_log"))?;

        // dt_bias: [num_v_heads] — parameter, no .weight suffix
        let dt_bias = source
            .get_tensor_auto(&format!("{hf_prefix}.{attn_sub}.dt_bias"))?;

        // Gated RMSNorm weight
        let norm_weight = source
            .get_tensor_auto(&format!("{hf_prefix}.{attn_sub}.norm.weight"))?;

        // --- FFN weights: MoE layers may not have dense FFN ---
        let ffn_norm_weight = Self::get_tensor_with_fallback_generic(
            source,
            &format!("{hf_prefix}.post_attention_layernorm.weight"),
            &format!("blk.{layer_idx}.ffn_norm.weight"),
        )?;

        let has_dense_ffn = source.has_tensor(&format!("{hf_prefix}.mlp.gate_proj.weight"))
            || source.has_tensor(&format!("blk.{layer_idx}.ffn_gate.weight"));

        let (ffn_gate_weight, ffn_up_weight, ffn_down_weight) = if has_dense_ffn {
            let gate_raw = Self::get_tensor_with_fallback_generic(
                source,
                &format!("{hf_prefix}.mlp.gate_proj.weight"),
                &format!("blk.{layer_idx}.ffn_gate.weight"),
            )?;
            let up_raw = Self::get_tensor_with_fallback_generic(
                source,
                &format!("{hf_prefix}.mlp.up_proj.weight"),
                &format!("blk.{layer_idx}.ffn_up.weight"),
            )?;
            let down_raw = Self::get_tensor_with_fallback_generic(
                source,
                &format!("{hf_prefix}.mlp.down_proj.weight"),
                &format!("blk.{layer_idx}.ffn_down.weight"),
            )?;
            (
                Some(Self::transpose_weight(&gate_raw, intermediate_dim, hidden_dim)),
                Self::transpose_weight(&up_raw, intermediate_dim, hidden_dim),
                Self::transpose_weight(&down_raw, hidden_dim, intermediate_dim),
            )
        } else {
            // MoE layer: no dense FFN
            (None, vec![0.0; intermediate_dim * hidden_dim], vec![0.0; hidden_dim * intermediate_dim])
        };

        Ok(AprTransformerLayer {
            attn_norm_weight,
            attn_norm_bias: None,
            qkv_weight,
            qkv_bias: None,
            attn_output_weight,
            attn_output_bias: None,
            ffn_gate_weight,
            ffn_gate_bias: None,
            ffn_up_weight,
            ffn_up_bias: None,
            ffn_down_weight,
            ffn_down_bias: None,
            ffn_norm_weight: Some(ffn_norm_weight),
            ffn_norm_bias: None,
            attn_q_norm_weight: None,
            attn_k_norm_weight: None,
            // GH-278: Gated Delta Net weights
            linear_attn_z_weight: Some(z_weight),
            linear_attn_b_weight: Some(b_weight),
            linear_attn_a_weight: Some(a_weight),
            linear_attn_conv1d_weight: Some(conv1d_weight),
            linear_attn_a_log: Some(a_log),
            linear_attn_dt_bias: Some(dt_bias),
            linear_attn_norm_weight: Some(norm_weight),
            // ALB-010: MoE weights loaded separately via load_moe_weights
            moe_gate_weight: None,
            moe_expert_gate_up: None,
            moe_expert_down: None,
            moe_shared_gate: None,
            moe_shared_up: None,
            moe_shared_down: None,
            moe_shared_expert_gate_weight: None,
        })
    }

    /// ALB-010: Detect model prefix for ConditionalGeneration wrappers
    ///
    /// Qwen3.5-35B-A3B stores tensors under `model.language_model.layers.*`
    /// instead of the standard `model.layers.*`. This detects the prefix
    /// by inspecting available tensor names.
    fn detect_model_prefix<S: TensorSource>(source: &S) -> String {
        let names = source.tensor_names();
        for name in &names {
            if name.starts_with("model.language_model.") {
                return "model.language_model".to_string();
            }
        }
        "model".to_string()
    }

    /// ALB-010: Extract a transformer layer with configurable model prefix
    ///
    /// Same as `extract_layer_generic` but uses the detected prefix instead of
    /// hardcoded `model.layers.{i}`. For MoE layers where `mlp.gate_proj`
    /// doesn't exist, FFN weights are zeroed (MoE replaces dense FFN).
    fn extract_layer_generic_with_prefix<S: TensorSource>(
        source: &S,
        layer_idx: usize,
        hidden_dim: usize,
        num_heads: usize,
        num_kv_heads: usize,
        intermediate_dim: usize,
        model_prefix: &str,
    ) -> Result<AprTransformerLayer> {
        let hf_prefix = format!("{model_prefix}.layers.{layer_idx}");
        let gguf_prefix = format!("blk.{layer_idx}");

        let attn_norm_weight = Self::get_tensor_with_fallback_generic(
            source,
            &format!("{hf_prefix}.input_layernorm.weight"),
            &format!("{gguf_prefix}.attn_norm.weight"),
        )?;

        let q_weight = Self::get_tensor_with_fallback_generic(
            source,
            &format!("{hf_prefix}.self_attn.q_proj.weight"),
            &format!("{gguf_prefix}.attn_q.weight"),
        )?;
        let k_weight = Self::get_tensor_with_fallback_generic(
            source,
            &format!("{hf_prefix}.self_attn.k_proj.weight"),
            &format!("{gguf_prefix}.attn_k.weight"),
        )?;
        let v_weight = Self::get_tensor_with_fallback_generic(
            source,
            &format!("{hf_prefix}.self_attn.v_proj.weight"),
            &format!("{gguf_prefix}.attn_v.weight"),
        )?;

        let head_dim = hidden_dim / num_heads;
        let kv_dim = head_dim * num_kv_heads;
        let qkv_weight =
            Self::concat_qkv_transposed(&q_weight, &k_weight, &v_weight, hidden_dim, kv_dim);

        let qkv_bias = Self::try_concat_qkv_bias_dual_generic(
            source,
            &hf_prefix,
            &gguf_prefix,
            hidden_dim,
            kv_dim,
        );

        let attn_output_raw = Self::get_tensor_with_fallback_generic(
            source,
            &format!("{hf_prefix}.self_attn.o_proj.weight"),
            &format!("{gguf_prefix}.attn_output.weight"),
        )?;
        let attn_output_weight = Self::transpose_weight(&attn_output_raw, hidden_dim, hidden_dim);

        let ffn_norm_weight = Self::get_tensor_with_fallback_generic(
            source,
            &format!("{hf_prefix}.post_attention_layernorm.weight"),
            &format!("{gguf_prefix}.ffn_norm.weight"),
        )?;

        // MoE layers may not have dense FFN — try to load, fallback to zeros
        let has_dense_ffn = source.has_tensor(&format!("{hf_prefix}.mlp.gate_proj.weight"))
            || source.has_tensor(&format!("{gguf_prefix}.ffn_gate.weight"));

        let (ffn_gate_weight, ffn_up_weight, ffn_down_weight) = if has_dense_ffn {
            let gate_raw = Self::get_tensor_with_fallback_generic(
                source,
                &format!("{hf_prefix}.mlp.gate_proj.weight"),
                &format!("{gguf_prefix}.ffn_gate.weight"),
            )?;
            let up_raw = Self::get_tensor_with_fallback_generic(
                source,
                &format!("{hf_prefix}.mlp.up_proj.weight"),
                &format!("{gguf_prefix}.ffn_up.weight"),
            )?;
            let down_raw = Self::get_tensor_with_fallback_generic(
                source,
                &format!("{hf_prefix}.mlp.down_proj.weight"),
                &format!("{gguf_prefix}.ffn_down.weight"),
            )?;
            (
                Some(Self::transpose_weight(&gate_raw, intermediate_dim, hidden_dim)),
                Self::transpose_weight(&up_raw, intermediate_dim, hidden_dim),
                Self::transpose_weight(&down_raw, hidden_dim, intermediate_dim),
            )
        } else {
            // MoE layer: no dense FFN, use empty placeholders
            (None, vec![0.0; intermediate_dim * hidden_dim], vec![0.0; hidden_dim * intermediate_dim])
        };

        Ok(AprTransformerLayer {
            attn_norm_weight,
            attn_norm_bias: None,
            qkv_weight,
            qkv_bias,
            attn_output_weight,
            attn_output_bias: None,
            ffn_gate_weight,
            ffn_gate_bias: None,
            ffn_up_weight,
            ffn_up_bias: None,
            ffn_down_weight,
            ffn_down_bias: None,
            ffn_norm_weight: Some(ffn_norm_weight),
            ffn_norm_bias: None,
            attn_q_norm_weight: source
                .get_tensor_auto(&format!("{hf_prefix}.self_attn.q_norm.weight"))
                .or_else(|_| source.get_tensor_auto(&format!("{gguf_prefix}.attn_q_norm.weight")))
                .ok(),
            attn_k_norm_weight: source
                .get_tensor_auto(&format!("{hf_prefix}.self_attn.k_norm.weight"))
                .or_else(|_| source.get_tensor_auto(&format!("{gguf_prefix}.attn_k_norm.weight")))
                .ok(),
            linear_attn_z_weight: None,
            linear_attn_b_weight: None,
            linear_attn_a_weight: None,
            linear_attn_conv1d_weight: None,
            linear_attn_a_log: None,
            linear_attn_dt_bias: None,
            linear_attn_norm_weight: None,
            moe_gate_weight: None,
            moe_expert_gate_up: None,
            moe_expert_down: None,
            moe_shared_gate: None,
            moe_shared_up: None,
            moe_shared_down: None,
            moe_shared_expert_gate_weight: None,
        })
    }

    /// ALB-010: Load MoE expert weights into an existing layer
    ///
    /// Supports two MoE weight layouts:
    ///
    /// **Layout 1 (packed, Qwen3.5-35B-A3B)**:
    /// - `mlp.experts.gate_up_proj` [num_experts, 2*intermediate, hidden]
    /// - `mlp.experts.down_proj` [num_experts, hidden, intermediate]
    /// - `mlp.shared_expert.{gate,up,down}_proj.weight`
    /// - `mlp.shared_expert_gate.weight` [1, hidden]
    ///
    /// **Layout 2 (per-expert, Qwen3-Coder-30B-A3B)**:
    /// - `mlp.experts.{e}.gate_proj.weight` [intermediate, hidden]
    /// - `mlp.experts.{e}.up_proj.weight` [intermediate, hidden]
    /// - `mlp.experts.{e}.down_proj.weight` [hidden, intermediate]
    /// - No shared expert
    fn load_moe_weights<S: TensorSource>(
        source: &S,
        layer_idx: usize,
        model_prefix: &str,
        config: &SafetensorsConfig,
        hidden_dim: usize,
        layer: &mut AprTransformerLayer,
    ) -> Result<()> {
        let prefix = format!("{model_prefix}.layers.{layer_idx}");

        // Router gate weight: [num_experts, hidden_dim]
        let gate_name = format!("{prefix}.mlp.gate.weight");
        if let Ok(gate) = source.get_tensor_auto(&gate_name) {
            layer.moe_gate_weight = Some(gate);
        }

        // Try Layout 1: packed expert tensors (no .weight suffix)
        let gate_up_name = format!("{prefix}.mlp.experts.gate_up_proj");
        if let Ok(gate_up) = source.get_tensor_auto(&gate_up_name) {
            layer.moe_expert_gate_up = Some(gate_up);

            let down_name = format!("{prefix}.mlp.experts.down_proj");
            if let Ok(down) = source.get_tensor_auto(&down_name) {
                layer.moe_expert_down = Some(down);
            }
        } else {
            // Layout 2: per-expert tensors (Qwen3-Coder style)
            // Pack individual expert weights into concatenated tensors
            let num_experts = config.num_experts.unwrap_or(0);
            let moe_intermediate = config.moe_intermediate_size.unwrap_or(0);
            if num_experts > 0 && moe_intermediate > 0 {
                let mut gate_up_packed = Vec::with_capacity(
                    num_experts * 2 * moe_intermediate * hidden_dim,
                );
                let mut down_packed = Vec::with_capacity(
                    num_experts * hidden_dim * moe_intermediate,
                );
                let mut found_any = false;

                for e in 0..num_experts {
                    let gate_name = format!("{prefix}.mlp.experts.{e}.gate_proj.weight");
                    let up_name = format!("{prefix}.mlp.experts.{e}.up_proj.weight");
                    let down_name = format!("{prefix}.mlp.experts.{e}.down_proj.weight");

                    match (
                        source.get_tensor_auto(&gate_name),
                        source.get_tensor_auto(&up_name),
                        source.get_tensor_auto(&down_name),
                    ) {
                        (Ok(gate), Ok(up), Ok(down)) => {
                            found_any = true;
                            // Pack gate+up into [2*intermediate, hidden] per expert
                            gate_up_packed.extend_from_slice(&gate);
                            gate_up_packed.extend_from_slice(&up);
                            down_packed.extend_from_slice(&down);
                        }
                        _ => {
                            if found_any {
                                // Partial experts — shouldn't happen, but handle gracefully
                                break;
                            }
                        }
                    }
                }

                if found_any {
                    layer.moe_expert_gate_up = Some(gate_up_packed);
                    layer.moe_expert_down = Some(down_packed);
                }
            }
        }

        // Shared expert FFN (only for models that have it, e.g. Qwen3.5)
        let shared_intermediate = config.shared_expert_intermediate_size
            .or(config.moe_intermediate_size)
            .unwrap_or(0);
        if shared_intermediate > 0 {
            let shared_gate_name = format!("{prefix}.mlp.shared_expert.gate_proj.weight");
            if let Ok(g) = source.get_tensor_auto(&shared_gate_name) {
                layer.moe_shared_gate = Some(g);
            }
            let shared_up_name = format!("{prefix}.mlp.shared_expert.up_proj.weight");
            if let Ok(u) = source.get_tensor_auto(&shared_up_name) {
                layer.moe_shared_up = Some(u);
            }
            let shared_down_name = format!("{prefix}.mlp.shared_expert.down_proj.weight");
            if let Ok(d) = source.get_tensor_auto(&shared_down_name) {
                layer.moe_shared_down = Some(d);
            }
        }

        // Shared expert gate: sigmoid scaling weight [1, hidden_dim]
        let shared_gate_name = format!("{prefix}.mlp.shared_expert_gate.weight");
        if let Ok(sg) = source.get_tensor_auto(&shared_gate_name) {
            layer.moe_shared_expert_gate_weight = Some(sg);
        }

        Ok(())
    }

    /// Pass through weight in matvec-optimal [out_dim, in_dim] format
    ///
    /// PMAT-095 FIX: HuggingFace stores Linear weights as [out_features, in_features]
    /// which is EXACTLY what trueno's matvec needs! Previous implementation transposed
    /// twice (here and in matmul), causing O(n²) overhead per forward pass.
    ///
    /// Now we keep HuggingFace format directly - no transposition needed.
    #[allow(clippy::unused_self)]
    pub fn transpose_weight(weight: &[f32], _out_dim: usize, _in_dim: usize) -> Vec<f32> {
        // PMAT-095: Keep [out_dim, in_dim] format - no transposition!
        // This eliminates the 75x performance gap vs GGUF.
        weight.to_vec()
    }

    /// Concatenate Q, K, V weights into combined QKV tensor (matvec-optimal)
    ///
    /// PMAT-095 FIX: Keep [out_dim, in_dim] format from HuggingFace.
    /// For QKV, we concatenate along the output dimension:
    /// - Q: [hidden_dim, hidden_dim]
    /// - K: [kv_dim, hidden_dim]
    /// - V: [kv_dim, hidden_dim]
    ///
    /// Result: [hidden_dim + kv_dim + kv_dim, hidden_dim] in row-major
    pub fn concat_qkv_transposed(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        _hidden_dim: usize,
        _kv_dim: usize,
    ) -> Vec<f32> {
        // PMAT-095: Simple concatenation - weights are already in optimal layout
        // Concatenate [Q; K; V] along output dimension
        let mut qkv = Vec::with_capacity(q.len() + k.len() + v.len());
        qkv.extend_from_slice(q);
        qkv.extend_from_slice(k);
        qkv.extend_from_slice(v);
        qkv
    }

    /// Concatenate Q, K, V weights into combined QKV tensor (legacy, no transpose)
    fn concat_qkv(q: &[f32], k: &[f32], v: &[f32]) -> Vec<f32> {
        let mut qkv = Vec::with_capacity(q.len() + k.len() + v.len());
        qkv.extend_from_slice(q);
        qkv.extend_from_slice(k);
        qkv.extend_from_slice(v);
        qkv
    }

    /// Try to concatenate Q, K, V biases if they exist
    #[allow(dead_code)]
    fn try_concat_qkv_bias(
        st_model: &MappedSafeTensorsModel,
        prefix: &str,
        hidden_dim: usize,
        kv_dim: usize,
    ) -> Option<Vec<f32>> {
        let q_bias = st_model
            .get_tensor_auto(&format!("{prefix}.self_attn.q_proj.bias"))
            .ok()?;
        let k_bias = st_model
            .get_tensor_auto(&format!("{prefix}.self_attn.k_proj.bias"))
            .ok()?;
        let v_bias = st_model
            .get_tensor_auto(&format!("{prefix}.self_attn.v_proj.bias"))
            .ok()?;

        let mut qkv_bias = Vec::with_capacity(hidden_dim + kv_dim + kv_dim);
        qkv_bias.extend_from_slice(&q_bias);
        qkv_bias.extend_from_slice(&k_bias);
        qkv_bias.extend_from_slice(&v_bias);

        Some(qkv_bias)
    }

    /// Try to concatenate Q, K, V biases with dual naming support
    #[allow(dead_code)]
    fn try_concat_qkv_bias_dual(
        st_model: &MappedSafeTensorsModel,
        hf_prefix: &str,
        gguf_prefix: &str,
        hidden_dim: usize,
        kv_dim: usize,
    ) -> Option<Vec<f32>> {
        Self::try_concat_qkv_bias_dual_generic(st_model, hf_prefix, gguf_prefix, hidden_dim, kv_dim)
    }

    /// Generic version of QKV bias concatenation (GH-213)
    fn try_concat_qkv_bias_dual_generic<S: TensorSource>(
        source: &S,
        hf_prefix: &str,
        gguf_prefix: &str,
        hidden_dim: usize,
        kv_dim: usize,
    ) -> Option<Vec<f32>> {
        let q_bias = source
            .get_tensor_auto(&format!("{hf_prefix}.self_attn.q_proj.bias"))
            .ok()
            .or_else(|| {
                source
                    .get_tensor_auto(&format!("{gguf_prefix}.attn_q.bias"))
                    .ok()
            })?;
        let k_bias = source
            .get_tensor_auto(&format!("{hf_prefix}.self_attn.k_proj.bias"))
            .ok()
            .or_else(|| {
                source
                    .get_tensor_auto(&format!("{gguf_prefix}.attn_k.bias"))
                    .ok()
            })?;
        let v_bias = source
            .get_tensor_auto(&format!("{hf_prefix}.self_attn.v_proj.bias"))
            .ok()
            .or_else(|| {
                source
                    .get_tensor_auto(&format!("{gguf_prefix}.attn_v.bias"))
                    .ok()
            })?;

        let mut qkv_bias = Vec::with_capacity(hidden_dim + kv_dim + kv_dim);
        qkv_bias.extend_from_slice(&q_bias);
        qkv_bias.extend_from_slice(&k_bias);
        qkv_bias.extend_from_slice(&v_bias);

        Some(qkv_bias)
    }

    /// Get tensor with fallback to alternative naming conventions
    ///
    /// Tries HuggingFace naming first, then GGUF-style naming, then bare name.
    /// This enables loading SafeTensors files regardless of their origin.
    ///
    /// GH-196: Also tries stripping `model.` prefix for APR canonical names,
    /// and adds diagnostic tensor name listing on failure.
    #[allow(dead_code)]
    fn get_tensor_with_fallback(
        st_model: &MappedSafeTensorsModel,
        hf_name: &str,
        gguf_name: &str,
    ) -> Result<Vec<f32>> {
        Self::get_tensor_with_fallback_generic(st_model, hf_name, gguf_name)
    }

    /// Generic version of tensor lookup with fallback naming (GH-213)
    fn get_tensor_with_fallback_generic<S: TensorSource>(
        source: &S,
        hf_name: &str,
        gguf_name: &str,
    ) -> Result<Vec<f32>> {
        // Try HuggingFace name first (e.g., "model.norm.weight")
        if let Ok(t) = source.get_tensor_auto(hf_name) {
            return Ok(t);
        }
        // Try GGUF name (e.g., "output_norm.weight")
        if let Ok(t) = source.get_tensor_auto(gguf_name) {
            return Ok(t);
        }
        // Try bare name without "model." prefix (APR canonical names)
        let bare_name = hf_name.strip_prefix("model.").unwrap_or(hf_name);
        if bare_name != hf_name {
            if let Ok(t) = source.get_tensor_auto(bare_name) {
                return Ok(t);
            }
        }

        // Diagnostic: list available tensor names for debugging
        let available = source.tensor_names();
        let sample: Vec<&str> = available.iter().take(5).copied().collect();
        Err(RealizarError::UnsupportedOperation {
            operation: "get_tensor_auto".to_string(),
            reason: format!(
                "Tensor not found with names: '{}', '{}', or '{}'. \
                 Available tensors ({} total): {:?}{}",
                hf_name,
                gguf_name,
                bare_name,
                available.len(),
                sample,
                if available.len() > 5 { ", ..." } else { "" }
            ),
        })
    }

    /// Check if tensor exists with either naming convention
    #[allow(dead_code)]
    fn has_tensor_with_fallback(
        st_model: &MappedSafeTensorsModel,
        hf_name: &str,
        gguf_name: &str,
    ) -> bool {
        Self::has_tensor_with_fallback_generic(st_model, hf_name, gguf_name)
    }

    /// Generic version of tensor existence check (GH-213)
    fn has_tensor_with_fallback_generic<S: TensorSource>(
        source: &S,
        hf_name: &str,
        gguf_name: &str,
    ) -> bool {
        if source.has_tensor(hf_name) || source.has_tensor(gguf_name) {
            return true;
        }
        // GH-196: Also check bare name without "model." prefix
        let bare_name = hf_name.strip_prefix("model.").unwrap_or(hf_name);
        bare_name != hf_name && source.has_tensor(bare_name)
    }

    /// Get optional tensor with fallback naming
    #[allow(dead_code)]
    fn get_optional_tensor_with_fallback(
        st_model: &MappedSafeTensorsModel,
        hf_name: &str,
        gguf_name: &str,
    ) -> Option<Vec<f32>> {
        st_model
            .get_tensor_auto(hf_name)
            .ok()
            .or_else(|| st_model.get_tensor_auto(gguf_name).ok())
            .or_else(|| {
                // GH-196: Try bare name without "model." prefix
                let bare_name = hf_name.strip_prefix("model.")?;
                st_model.get_tensor_auto(bare_name).ok()
            })
    }
}

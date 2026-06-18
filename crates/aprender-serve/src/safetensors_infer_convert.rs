
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
        // F-STRUCT-001 (PMAT-756): fail closed on cross-tensor dimension
        // inconsistencies (vocab/hidden mismatch) BEFORE assembly. The SafeTensors
        // format does not enforce these — safetensors-lib / Transformers / Ollama
        // load such artifacts and run garbage; apr rejects them at load.
        Self::validate_structural_consistency(source)?;

        let apr_config = Self::build_apr_config(config)?;
        Self::log_phase2_warning(config);
        Self::log_hybrid_attention_info(config);

        let model_prefix = Self::detect_model_prefix(source);
        let token_embedding = Self::get_tensor_with_fallback_generic(
            source,
            &format!("{model_prefix}.embed_tokens.weight"),
            "token_embd.weight",
        )?;
        let output_norm_weight = Self::get_tensor_with_fallback_generic(
            source,
            &format!("{model_prefix}.norm.weight"),
            "output_norm.weight",
        )?;
        let (lm_head_weight, lm_head_tied) = Self::resolve_lm_head_weight(
            source,
            config,
            &token_embedding,
            apr_config.vocab_size,
            apr_config.hidden_dim,
        )?;
        let layers = Self::build_layers(source, config, &apr_config, &model_prefix)?;

        let transformer = AprTransformer {
            config: apr_config,
            token_embedding,
            layers,
            output_norm_weight,
            output_norm_bias: None,
            lm_head_weight,
            lm_head_bias: None,
            lm_head_tied,
            q4k_layers: None,
            lm_head_weight_q6k: None,
            lm_head_weight_q4k: None,
        };

        ValidatedAprTransformer::validate(transformer).map_err(Into::into)
    }

    /// F-STRUCT-001 (PMAT-756): cross-tensor structural consistency gate.
    ///
    /// Collects the declared shapes of the model's tensors and runs
    /// [`validate_cross_tensor_structure`], which rejects a model whose embedding
    /// and output head disagree on vocab size, or whose embedding and attention
    /// disagree on hidden dim — structural garbage the SafeTensors container
    /// format does not catch (and which `safetensors`-lib / Transformers / Ollama
    /// load silently). No false positives: the gate only fires when it positively
    /// identifies disagreeing tensors.
    fn validate_structural_consistency<S: TensorSource>(source: &S) -> Result<()> {
        let names = source.tensor_names();
        let shapes: Vec<(String, Vec<usize>)> = names
            .iter()
            .filter_map(|n| source.tensor_shape(n).map(|s| ((*n).to_string(), s)))
            .collect();
        let view: Vec<(&str, &[usize])> = shapes
            .iter()
            .map(|(n, s)| (n.as_str(), s.as_slice()))
            .collect();
        crate::safetensors::validation::validate_cross_tensor_structure(view)
    }

    /// Build `AprTransformerConfig` from SafeTensors config.json.
    fn build_apr_config(config: &SafetensorsConfig) -> Result<AprTransformerConfig> {
        let hidden_dim = Self::required_config_field(config.hidden_size, "hidden_size")?;
        let num_layers = Self::required_config_field(config.num_hidden_layers, "num_hidden_layers")?;
        let num_heads =
            Self::required_config_field(config.num_attention_heads, "num_attention_heads")?;
        let vocab_size = Self::required_config_field(config.vocab_size, "vocab_size")?;
        let architecture = config.architecture();
        // R-02 (Meyer DbC): rope_theta from config, or architecture-specific default.
        let rope_theta = config
            .rope_theta
            .unwrap_or_else(|| crate::gguf::default_rope_theta_for_architecture(&architecture));

        Ok(AprTransformerConfig {
            architecture,
            hidden_dim,
            num_layers,
            num_heads,
            num_kv_heads: config.num_kv_heads(),
            vocab_size,
            intermediate_dim: config.intermediate_size.unwrap_or(hidden_dim * 4),
            context_length: config.max_position_embeddings.unwrap_or(0),
            rope_theta,
            eps: config.rms_norm_eps.unwrap_or(1e-6),
            eos_token_id: config.eos_token_id,
            explicit_head_dim: config.head_dim,
            layer_types: config.layer_types.clone(),
            linear_key_head_dim: config.linear_key_head_dim,
            linear_value_head_dim: config.linear_value_head_dim,
            linear_num_key_heads: config.linear_num_key_heads,
            linear_num_value_heads: config.linear_num_value_heads,
            linear_conv_kernel_dim: config.linear_conv_kernel_dim,
            num_experts: config.num_experts,
            num_experts_per_tok: config.num_experts_per_tok,
            expert_intermediate_size: config.moe_intermediate_size,
        })
    }

    fn required_config_field(value: Option<usize>, field: &str) -> Result<usize> {
        value.ok_or_else(|| RealizarError::FormatError {
            reason: format!("config.json missing {field}"),
        })
    }

    /// Phase 2: non-fatal dimension validation at construction boundary.
    /// The final `ValidatedAprTransformer::validate()` is the hard gate; this
    /// just surfaces obvious dimension errors early.
    fn log_phase2_warning(config: &SafetensorsConfig) {
        if let Err(e) = crate::gguf::ValidatedModelConfig::from_safetensors_config(config) {
            eprintln!(
                "[Phase2-WARN] SafeTensors config validation: {e} — proceeding with conversion"
            );
        }
    }

    /// GH-278: Log Qwen3.5 detection with hybrid attention info.
    fn log_hybrid_attention_info(config: &SafetensorsConfig) {
        if !config.is_hybrid_attention() {
            return;
        }
        let layer_count = config.layer_types.as_ref().map_or(0, Vec::len);
        let linear_count = config.layer_types.as_ref().map_or(0, |t| {
            t.iter()
                .filter(|l| *l == "linear" || *l == "linear_attention")
                .count()
        });
        eprintln!(
            "[GH-278] Hybrid attention model detected: {}/{} linear layers, head_dim={:?}",
            linear_count, layer_count, config.head_dim,
        );
    }

    /// Resolve the LM head weight, honoring `tie_word_embeddings` and falling
    /// back to the embedding matrix when no `lm_head` tensor exists.
    ///
    /// F-GT-002 FIX: Check `tie_word_embeddings` config FIRST, not just tensor
    /// existence — HuggingFace may store a placeholder all-zero `lm_head.weight`
    /// when tying is in effect.
    ///
    /// PMAT-788: returns `(lm_head_weight, tied)`. When the LM head is tied to
    /// the token embedding the weight buffer is byte-identical to
    /// `token_embedding` (`transpose_weight` is a documented no-op, PMAT-095), so
    /// instead of materializing a redundant full-size copy (≈519 MiB on a 0.5B
    /// model) we return an EMPTY `Vec` and `tied = true`. The logits matmul then
    /// sources `token_embedding` directly via `AprTransformer::lm_head_f32()`,
    /// producing bit-identical output. An UNtied model returns its own separate
    /// `lm_head` weights with `tied = false` and is completely unaffected.
    fn resolve_lm_head_weight<S: TensorSource>(
        source: &S,
        config: &SafetensorsConfig,
        token_embedding: &[f32],
        vocab_size: usize,
        hidden_dim: usize,
    ) -> Result<(Vec<f32>, bool)> {
        if config.tie_word_embeddings.unwrap_or(false) {
            // Tied: do not duplicate the embedding — `lm_head_f32()` reuses it.
            debug_assert_eq!(token_embedding.len(), vocab_size * hidden_dim);
            return Ok((Vec::new(), true));
        }
        if Self::has_tensor_with_fallback_generic(source, "lm_head.weight", "output.weight") {
            let raw =
                Self::get_tensor_with_fallback_generic(source, "lm_head.weight", "output.weight")?;
            // Untied: a genuine separate LM head — keep its own weights. The HF
            // [out, in] row-major layout is matvec-optimal as-is (PMAT-095:
            // `transpose_weight` is a no-op), so MOVE the already-owned buffer
            // (PMAT-787 `transpose_weight_owned`) rather than cloning it.
            return Ok((Self::transpose_weight_owned(raw), false));
        }
        // Fallback: no lm_head tensor exists → treat as tied to the embedding.
        Ok((Vec::new(), true))
    }

    /// Extract all transformer layers, dispatching to the linear (Gated Delta
    /// Net) extractor or the standard attention extractor per-layer, and
    /// layering MoE weights on top when configured.
    fn build_layers<S: TensorSource>(
        source: &S,
        config: &SafetensorsConfig,
        apr_config: &AprTransformerConfig,
        model_prefix: &str,
    ) -> Result<Vec<AprTransformerLayer>> {
        let is_moe = config.num_experts.is_some();
        let mut layers = Vec::with_capacity(apr_config.num_layers);
        for i in 0..apr_config.num_layers {
            let mut layer = Self::extract_single_layer(source, config, apr_config, i, model_prefix)?;
            if is_moe {
                Self::load_moe_weights(
                    source,
                    i,
                    model_prefix,
                    config,
                    apr_config.hidden_dim,
                    &mut layer,
                )?;
            }
            layers.push(layer);
        }
        Ok(layers)
    }

    fn extract_single_layer<S: TensorSource>(
        source: &S,
        config: &SafetensorsConfig,
        apr_config: &AprTransformerConfig,
        layer_idx: usize,
        model_prefix: &str,
    ) -> Result<AprTransformerLayer> {
        if Self::is_linear_attention_layer(config, layer_idx) {
            Self::extract_linear_layer_generic(
                source,
                layer_idx,
                apr_config.hidden_dim,
                apr_config.intermediate_dim,
                config,
                model_prefix,
            )
        } else {
            Self::extract_layer_generic_with_prefix(
                source,
                layer_idx,
                apr_config.hidden_dim,
                apr_config.num_heads,
                apr_config.num_kv_heads,
                apr_config.intermediate_dim,
                model_prefix,
            )
        }
    }

    fn is_linear_attention_layer(config: &SafetensorsConfig, layer_idx: usize) -> bool {
        config
            .layer_types
            .as_ref()
            .and_then(|lt| lt.get(layer_idx))
            .is_some_and(|t| t == "linear" || t == "linear_attention")
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
        // PMAT-787: move the owned buffer instead of cloning it (transpose is a no-op).
        let attn_output_weight = Self::transpose_weight_owned(attn_output_raw);

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
        let ffn_gate_weight = Self::transpose_weight_owned(ffn_gate_raw);

        let ffn_up_raw = Self::get_tensor_with_fallback_generic(
            source,
            &format!("{hf_prefix}.mlp.up_proj.weight"),
            &format!("{gguf_prefix}.ffn_up.weight"),
        )?;
        let ffn_up_weight = Self::transpose_weight_owned(ffn_up_raw);

        let ffn_down_raw = Self::get_tensor_with_fallback_generic(
            source,
            &format!("{hf_prefix}.mlp.down_proj.weight"),
            &format!("{gguf_prefix}.ffn_down.weight"),
        )?;
        let ffn_down_weight = Self::transpose_weight_owned(ffn_down_raw);

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
        // PMAT-787: move the owned buffer instead of cloning it (transpose is a no-op).
        let attn_output_weight = Self::transpose_weight_owned(out_proj_raw);

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
            // PMAT-787: move owned buffers instead of cloning (transpose is a no-op).
            (
                Some(Self::transpose_weight_owned(gate_raw)),
                Self::transpose_weight_owned(up_raw),
                Self::transpose_weight_owned(down_raw),
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
        // PMAT-787: q/k/v raws have been copied into `qkv_weight`; free them now
        // so they don't sit resident alongside the (larger) FFN buffers loaded below.
        drop(q_weight);
        drop(k_weight);
        drop(v_weight);

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
        // PMAT-787: move the owned buffer instead of cloning it (transpose is a no-op).
        let attn_output_weight = Self::transpose_weight_owned(attn_output_raw);

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
                Some(Self::transpose_weight_owned(gate_raw)),
                Self::transpose_weight_owned(up_raw),
                Self::transpose_weight_owned(down_raw),
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
        Self::load_moe_router_gate(source, &prefix, layer);
        Self::load_moe_expert_tensors(source, &prefix, config, hidden_dim, layer);
        Self::load_moe_shared_expert_ffn(source, &prefix, config, layer);
        Self::load_moe_shared_expert_gate(source, &prefix, layer);
        Ok(())
    }

    /// Router gate weight: `[num_experts, hidden_dim]`.
    fn load_moe_router_gate<S: TensorSource>(
        source: &S,
        prefix: &str,
        layer: &mut AprTransformerLayer,
    ) {
        if let Ok(gate) = source.get_tensor_auto(&format!("{prefix}.mlp.gate.weight")) {
            layer.moe_gate_weight = Some(gate);
        }
    }

    /// Expert `gate_up`/`down` tensors. Tries Layout 1 (packed `gate_up_proj` +
    /// `down_proj`) first, then falls back to Layout 2 (per-expert weights,
    /// Qwen3-Coder style), packing individual tensors into concatenated form.
    fn load_moe_expert_tensors<S: TensorSource>(
        source: &S,
        prefix: &str,
        config: &SafetensorsConfig,
        hidden_dim: usize,
        layer: &mut AprTransformerLayer,
    ) {
        if let Ok(gate_up) = source.get_tensor_auto(&format!("{prefix}.mlp.experts.gate_up_proj")) {
            layer.moe_expert_gate_up = Some(gate_up);
            if let Ok(down) = source.get_tensor_auto(&format!("{prefix}.mlp.experts.down_proj")) {
                layer.moe_expert_down = Some(down);
            }
            return;
        }
        if let Some((gate_up_packed, down_packed)) =
            Self::pack_per_expert_tensors(source, prefix, config, hidden_dim)
        {
            layer.moe_expert_gate_up = Some(gate_up_packed);
            layer.moe_expert_down = Some(down_packed);
        }
    }

    /// Layout 2: walk per-expert `gate_proj`/`up_proj`/`down_proj` tensors and
    /// concatenate them. Returns `None` when no complete expert was found.
    /// A partial expert encountered after complete ones short-circuits the
    /// loop (partial experts shouldn't happen, but handle gracefully).
    fn pack_per_expert_tensors<S: TensorSource>(
        source: &S,
        prefix: &str,
        config: &SafetensorsConfig,
        hidden_dim: usize,
    ) -> Option<(Vec<f32>, Vec<f32>)> {
        let num_experts = config.num_experts.unwrap_or(0);
        let moe_intermediate = config.moe_intermediate_size.unwrap_or(0);
        if num_experts == 0 || moe_intermediate == 0 {
            return None;
        }

        let mut gate_up_packed = Vec::with_capacity(num_experts * 2 * moe_intermediate * hidden_dim);
        let mut down_packed = Vec::with_capacity(num_experts * hidden_dim * moe_intermediate);
        let mut found_any = false;

        for e in 0..num_experts {
            let gate = source.get_tensor_auto(&format!("{prefix}.mlp.experts.{e}.gate_proj.weight"));
            let up = source.get_tensor_auto(&format!("{prefix}.mlp.experts.{e}.up_proj.weight"));
            let down = source.get_tensor_auto(&format!("{prefix}.mlp.experts.{e}.down_proj.weight"));

            if let (Ok(gate), Ok(up), Ok(down)) = (gate, up, down) {
                found_any = true;
                gate_up_packed.extend_from_slice(&gate);
                gate_up_packed.extend_from_slice(&up);
                down_packed.extend_from_slice(&down);
            } else if found_any {
                break;
            }
        }

        found_any.then_some((gate_up_packed, down_packed))
    }

    /// Shared expert FFN (`gate_proj`/`up_proj`/`down_proj`), present on
    /// models like Qwen3.5. Gated on a positive
    /// `shared_expert_intermediate_size` (falling back to
    /// `moe_intermediate_size`).
    fn load_moe_shared_expert_ffn<S: TensorSource>(
        source: &S,
        prefix: &str,
        config: &SafetensorsConfig,
        layer: &mut AprTransformerLayer,
    ) {
        let shared_intermediate = config
            .shared_expert_intermediate_size
            .or(config.moe_intermediate_size)
            .unwrap_or(0);
        if shared_intermediate == 0 {
            return;
        }
        if let Ok(g) = source.get_tensor_auto(&format!("{prefix}.mlp.shared_expert.gate_proj.weight"))
        {
            layer.moe_shared_gate = Some(g);
        }
        if let Ok(u) = source.get_tensor_auto(&format!("{prefix}.mlp.shared_expert.up_proj.weight"))
        {
            layer.moe_shared_up = Some(u);
        }
        if let Ok(d) = source.get_tensor_auto(&format!("{prefix}.mlp.shared_expert.down_proj.weight"))
        {
            layer.moe_shared_down = Some(d);
        }
    }

    /// Shared expert gate: sigmoid scaling weight `[1, hidden_dim]`.
    fn load_moe_shared_expert_gate<S: TensorSource>(
        source: &S,
        prefix: &str,
        layer: &mut AprTransformerLayer,
    ) {
        if let Ok(sg) = source.get_tensor_auto(&format!("{prefix}.mlp.shared_expert_gate.weight")) {
            layer.moe_shared_expert_gate_weight = Some(sg);
        }
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

    /// Owned, zero-copy pass-through of an already-materialized weight buffer.
    ///
    /// PMAT-787 (peak-RSS fix): `transpose_weight` is a documented no-op
    /// (PMAT-095 keeps HuggingFace `[out, in]` layout), but it takes `&[f32]`
    /// and returns `weight.to_vec()` — a full second allocation of the tensor.
    /// At load time every weight is fetched into an owned `Vec<f32>` and then
    /// cloned by `transpose_weight`, so each tensor was transiently resident
    /// **twice** (raw + clone) while the raw was still bound in the caller's
    /// scope. For the 0.5B qwen2.5-coder model that roughly doubled peak RSS
    /// at the attention/FFN extraction boundary.
    ///
    /// This variant takes the buffer **by value** and returns it unchanged,
    /// so the consumer simply moves the existing allocation (no extra copy).
    /// The byte content is identical to `transpose_weight(&buf, _, _)`, so
    /// inference output is bit-for-bit unchanged.
    #[inline]
    #[must_use]
    pub fn transpose_weight_owned(weight: Vec<f32>) -> Vec<f32> {
        // PMAT-095 layout is already matvec-optimal — move, don't copy.
        weight
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

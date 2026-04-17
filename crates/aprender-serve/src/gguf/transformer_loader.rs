
impl GGUFTransformer {
    /// Load transformer weights from GGUF model
    ///
    /// # Arguments
    ///
    /// * `model` - Parsed GGUF model
    /// * `file_data` - Original file bytes for tensor extraction
    ///
    /// # Errors
    ///
    /// Returns error if required tensors are missing or malformed
    pub fn from_gguf(model: &GGUFModel, file_data: &[u8]) -> Result<Self> {
        // Phase 2: Validate config at construction boundary.
        let config = ValidatedModelConfig::from_gguf(model)?.into_inner();

        // Load token embedding
        let token_embedding = model.get_tensor_f32("token_embd.weight", file_data)?;
        // GH-278: Position embedding — standard GGUF name + aprender export fallback
        let position_embedding = model
            .get_tensor_f32("token_pos_embd.weight", file_data)
            .or_else(|_| model.get_tensor_f32("model.position_embedding.weight", file_data))
            .ok();

        // Load layers
        let mut layers = Vec::with_capacity(config.num_layers);
        for layer_idx in 0..config.num_layers {
            let layer = Self::load_layer(model, file_data, layer_idx)?;
            layers.push(layer);
        }

        // Load output norm (raw gamma values - no delta transformation needed)
        let output_norm_weight = model.get_tensor_f32("output_norm.weight", file_data)?;
        // GH-278: Output norm bias — standard + aprender fallback
        let output_norm_bias = model
            .get_tensor_f32("output_norm.bias", file_data)
            .or_else(|_| model.get_tensor_f32("model.norm.bias", file_data))
            .ok();

        // Load LM head (output projection)
        // Fall back to token_embd.weight for tied embeddings (Qwen2, some LLaMA variants)
        let lm_head_weight = model
            .get_tensor_f32("output.weight", file_data)
            .or_else(|_| model.get_tensor_f32("token_embd.weight", file_data))?;
        let lm_head_bias = model.get_tensor_f32("output.bias", file_data).ok();

        Ok(Self {
            config,
            token_embedding,
            position_embedding,
            layers,
            output_norm_weight,
            output_norm_bias,
            lm_head_weight,
            lm_head_bias,
        })
    }

    /// Load a single transformer layer
    ///
    /// Supports both tensor naming conventions:
    /// - phi-2 style: combined `attn_qkv.weight`
    /// - llama style: separate `attn_q.weight`, `attn_k.weight`, `attn_v.weight`
    fn load_layer(
        model: &GGUFModel,
        file_data: &[u8],
        layer_idx: usize,
    ) -> Result<GGUFTransformerLayer> {
        let prefix = format!("blk.{}", layer_idx);

        // Attention norm weights
        let attn_norm_weight =
            model.get_tensor_f32(&format!("{}.attn_norm.weight", prefix), file_data)?;
        // GH-278: Attention norm bias — standard GGUF + aprender fallback
        let attn_norm_bias = model
            .get_tensor_f32(&format!("{}.attn_norm.bias", prefix), file_data)
            .or_else(|_| {
                model.get_tensor_f32(&format!("{}.input_layernorm.bias", prefix), file_data)
            })
            .ok();

        // QKV weights - try combined first (phi-2), fall back to separate (llama)
        let (qkv_weight, qkv_bias) = if let Ok(combined) =
            model.get_tensor_f32(&format!("{}.attn_qkv.weight", prefix), file_data)
        {
            // phi-2 style: combined QKV tensor
            let bias = model
                .get_tensor_f32(&format!("{}.attn_qkv.bias", prefix), file_data)
                .ok();
            (combined, bias)
        } else {
            // llama style: separate Q, K, V tensors - concatenate them
            let q_weight = model.get_tensor_f32(&format!("{}.attn_q.weight", prefix), file_data)?;
            let k_weight = model.get_tensor_f32(&format!("{}.attn_k.weight", prefix), file_data)?;
            let v_weight = model.get_tensor_f32(&format!("{}.attn_v.weight", prefix), file_data)?;

            // Concatenate Q, K, V weights
            let mut qkv = Vec::with_capacity(q_weight.len() + k_weight.len() + v_weight.len());
            qkv.extend_from_slice(&q_weight);
            qkv.extend_from_slice(&k_weight);
            qkv.extend_from_slice(&v_weight);

            // Try to get biases (llama usually doesn't have them)
            let q_bias = model
                .get_tensor_f32(&format!("{}.attn_q.bias", prefix), file_data)
                .ok();
            let k_bias = model
                .get_tensor_f32(&format!("{}.attn_k.bias", prefix), file_data)
                .ok();
            let v_bias = model
                .get_tensor_f32(&format!("{}.attn_v.bias", prefix), file_data)
                .ok();

            let bias = match (q_bias, k_bias, v_bias) {
                (Some(q), Some(k), Some(v)) => {
                    let mut combined_bias = Vec::with_capacity(q.len() + k.len() + v.len());
                    combined_bias.extend_from_slice(&q);
                    combined_bias.extend_from_slice(&k);
                    combined_bias.extend_from_slice(&v);
                    Some(combined_bias)
                },
                _ => None,
            };

            (qkv, bias)
        };

        // Attention output
        let attn_output_weight =
            model.get_tensor_f32(&format!("{}.attn_output.weight", prefix), file_data)?;
        let attn_output_bias = model
            .get_tensor_f32(&format!("{}.attn_output.bias", prefix), file_data)
            .ok();

        // FFN gate (SwiGLU models like llama have this)
        let ffn_gate_weight = model
            .get_tensor_f32(&format!("{}.ffn_gate.weight", prefix), file_data)
            .ok();
        let ffn_gate_bias = model
            .get_tensor_f32(&format!("{}.ffn_gate.bias", prefix), file_data)
            .ok();

        // FFN up/down projections
        let ffn_up_weight =
            model.get_tensor_f32(&format!("{}.ffn_up.weight", prefix), file_data)?;
        // GH-278: FFN biases — standard GGUF + aprender fallback
        let ffn_up_bias = model
            .get_tensor_f32(&format!("{}.ffn_up.bias", prefix), file_data)
            .or_else(|_| {
                model.get_tensor_f32(&format!("{}.mlp.up_proj.bias", prefix), file_data)
            })
            .ok();
        let ffn_down_weight =
            model.get_tensor_f32(&format!("{}.ffn_down.weight", prefix), file_data)?;
        let ffn_down_bias = model
            .get_tensor_f32(&format!("{}.ffn_down.bias", prefix), file_data)
            .or_else(|_| {
                model.get_tensor_f32(&format!("{}.mlp.down_proj.bias", prefix), file_data)
            })
            .ok();

        // FFN norm (models with separate FFN normalization)
        let ffn_norm_weight = model
            .get_tensor_f32(&format!("{}.ffn_norm.weight", prefix), file_data)
            .ok();
        // GH-278: FFN norm bias — standard GGUF + aprender fallback
        let ffn_norm_bias = model
            .get_tensor_f32(&format!("{}.ffn_norm.bias", prefix), file_data)
            .or_else(|_| {
                model.get_tensor_f32(
                    &format!("{}.post_attention_layernorm.bias", prefix),
                    file_data,
                )
            })
            .ok();

        // GH-279: QK norm weights (Qwen3 per-head RMSNorm on Q and K)
        let attn_q_norm_weight = model
            .get_tensor_f32(&format!("{prefix}.attn_q_norm.weight"), file_data)
            .ok();
        let attn_k_norm_weight = model
            .get_tensor_f32(&format!("{prefix}.attn_k_norm.weight"), file_data)
            .ok();

        Ok(GGUFTransformerLayer {
            attn_norm_weight,
            attn_norm_bias,
            qkv_weight,
            qkv_bias,
            attn_output_weight,
            attn_output_bias,
            ffn_gate_weight,
            ffn_gate_bias,
            ffn_up_weight,
            ffn_up_bias,
            ffn_down_weight,
            ffn_down_bias,
            ffn_norm_weight,
            ffn_norm_bias,
            attn_q_norm_weight,
            attn_k_norm_weight,
        })
    }
}

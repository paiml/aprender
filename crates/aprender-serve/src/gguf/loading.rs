/// Load a quantized tensor from APR format, trying multiple names.
///
/// Handles APR native q8/q4 formats by dequantizing to f32.
/// For Conv1D architectures, transposes weights to [out, in] layout.
fn apr_load_quantized_tensor(
    apr: &crate::apr::MappedAprModel,
    data: &[u8],
    data_offset: usize,
    names: &[&str],
    in_dim: usize,
    out_dim: usize,
    transpose: bool,
) -> Result<OwnedQuantizedTensor> {
    use crate::apr::MappedAprModel;

    let (tensor, found_name) = names
        .iter()
        .find_map(|name| apr.find_tensor(name).map(|t| (t, *name)))
        .ok_or_else(|| RealizarError::FormatError {
            reason: format!("APR: tensor not found (tried: {})", names.join(", ")),
        })?;
    let start = data_offset + tensor.offset as usize;
    let end = start + tensor.size as usize;
    if end > data.len() {
        return Err(RealizarError::FormatError {
            reason: format!("APR: tensor {found_name} extends past EOF"),
        });
    }
    let raw = &data[start..end];
    let dtype = tensor.dtype.as_str();
    let num_elements = in_dim * out_dim;

    match dtype {
        "q8" => {
            // GH-285: APR native q8 requires CPU dequant → F32 (slow).
            // Re-import with `apr import` for GPU-optimal Q4K loading.
            eprintln!(
                "[GH-285] APR native q8 tensor '{}': CPU dequant to F32 \
                 (slow — re-import with `apr import` for GPU-optimal Q4K)",
                found_name
            );
            let mut f32_data = crate::apr::dequant::dequantize_apr_q8(raw, num_elements);
            if transpose {
                f32_data = transpose_f32_matrix(&f32_data, in_dim, out_dim);
            }
            let f32_bytes: Vec<u8> = f32_data.iter().flat_map(|v| v.to_le_bytes()).collect();
            Ok(OwnedQuantizedTensor {
                data: f32_bytes,
                in_dim,
                out_dim,
                qtype: 0,
            })
        },
        "q4" => {
            // GH-285: APR native q4 requires CPU dequant → F32 (slow).
            // Re-import with `apr import` for GPU-optimal Q4K loading.
            eprintln!(
                "[GH-285] APR native q4 tensor '{}': CPU dequant to F32 \
                 (slow — re-import with `apr import` for GPU-optimal Q4K)",
                found_name
            );
            let mut f32_data = crate::apr::dequant::dequantize_apr_q4(raw, num_elements);
            if transpose {
                f32_data = transpose_f32_matrix(&f32_data, in_dim, out_dim);
            }
            let f32_bytes: Vec<u8> = f32_data.iter().flat_map(|v| v.to_le_bytes()).collect();
            Ok(OwnedQuantizedTensor {
                data: f32_bytes,
                in_dim,
                out_dim,
                qtype: 0,
            })
        },
        _ => {
            let qtype = MappedAprModel::dtype_to_qtype(dtype);
            Ok(OwnedQuantizedTensor {
                data: raw.to_vec(),
                in_dim,
                out_dim,
                qtype,
            })
        },
    }
}

/// Load an F32 tensor from APR format, trying multiple names.
fn apr_load_f32_tensor(
    apr: &crate::apr::MappedAprModel,
    data: &[u8],
    data_offset: usize,
    names: &[&str],
) -> Result<Vec<f32>> {
    let (tensor, found_name) = names
        .iter()
        .find_map(|name| apr.find_tensor(name).map(|t| (t, *name)))
        .ok_or_else(|| RealizarError::FormatError {
            reason: format!("APR: tensor not found (tried: {})", names.join(", ")),
        })?;
    let start = data_offset + tensor.offset as usize;
    let end = start + tensor.size as usize;
    if end > data.len() {
        return Err(RealizarError::FormatError {
            reason: format!("APR: tensor {found_name} extends past EOF"),
        });
    }
    Ok(data[start..end]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// Try loading an optional F32 bias tensor from APR format.
fn apr_try_load_f32(
    apr: &crate::apr::MappedAprModel,
    data: &[u8],
    data_offset: usize,
    name: &str,
) -> Option<Vec<f32>> {
    let tensor = apr.find_tensor(name)?;
    let start = data_offset + tensor.offset as usize;
    let end = start + tensor.size as usize;
    if end > data.len() {
        return None;
    }
    Some(
        data[start..end]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    )
}

/// Infer vocab_size from APR metadata or embedding tensor shape.
/// GH-324: Infer vocab size from embedding tensor using contract-driven names only.
///
/// No substring fallback — if the contract doesn't have the embedding name,
/// the model's naming convention needs to be added to tensor-names-v1.yaml.
fn apr_infer_vocab_size(apr: &crate::apr::MappedAprModel) -> usize {
    use crate::tensor_names::{normalize_architecture, global_names, global_fallback_names, GlobalTensorRole};

    if let Some(v) = apr.metadata.vocab_size {
        if v > 0 {
            return v;
        }
    }

    // R-01 (Meyer DbC): "unknown" — don't pretend unidentified APR model is LLaMA.
    let arch = apr.metadata.architecture.as_deref().unwrap_or("unknown");
    let arch_key = normalize_architecture(arch);

    // Contract-only: exact match against known embedding names
    let mut candidates: Vec<&str> = Vec::new();
    candidates.extend_from_slice(global_names(arch_key, GlobalTensorRole::Embedding));
    candidates.extend_from_slice(global_fallback_names(GlobalTensorRole::Embedding));

    for candidate in &candidates {
        if let Some(t) = apr.tensors.iter().find(|t| t.name == *candidate) {
            if let Some(&dim) = t.shape.first() {
                return dim;
            }
        }
    }

    // No fallback — contract is the source of truth. Return 0 to signal failure.
    0
}

impl OwnedQuantizedModel {
    /// Create model from memory-mapped APR file (SHOWCASE-APR-GPU)
    ///
    /// Converts APR Q4K format to GGUF-compatible model for GPU inference.
    /// The raw Q4K tensor data is byte-compatible between formats.
    ///
    /// # Arguments
    /// * `apr` - Memory-mapped APR model
    ///
    /// # Errors
    /// Returns error if APR format is invalid or missing required tensors.
    pub fn from_apr(apr: &crate::apr::MappedAprModel) -> Result<Self> {
        let data = apr.data();
        let data_offset = apr.data_offset() as usize;

        // Phase 2: Deduplicated APR config extraction + validated construction.
        let vocab_size = apr_infer_vocab_size(apr);
        let validated = ValidatedModelConfig::from_apr(apr, vocab_size)?;

        // GH-279: Contract gate — validate architecture and dimensions before loading weights
        let _proof = crate::contract_gate::validate_model_load_basic(
            validated.architecture(),
            validated.num_layers(),
            validated.hidden_dim(),
            validated.num_heads(),
            validated.num_kv_heads(),
            validated.intermediate_dim(),
            validated.vocab_size(),
        )
        .map_err(crate::contract_gate::gate_error)?;

        // Extract inner GGUFConfig for storage (struct field is typed GGUFConfig)
        let mut config = validated.into_inner();

        // GH-278: Detect Conv1D layout from contract (not string matching)
        let transpose = config.constraints.needs_transpose();

        // Extract dimensions from validated config for use below
        let hidden_dim = config.hidden_dim;
        let num_layers = config.num_layers;
        let intermediate_dim = config.intermediate_dim;

        // GH-479: Infer explicit head_dim from Q proj tensor shape (Qwen3 head_dim != hidden/heads)
        let q_tensor_name = "model.layers.0.self_attn.q_proj.weight";
        let gguf_q_name = "blk.0.attn_q.weight";
        if let Some(q_tensor) = apr.find_tensor(q_tensor_name).or_else(|| apr.find_tensor(gguf_q_name)) {
            if q_tensor.shape.len() == 2 {
                let q_out_dim = q_tensor.shape[0];
                let inferred_head_dim = if config.num_heads > 0 { q_out_dim / config.num_heads } else { 0 };
                let default_head_dim = if config.num_heads > 0 { hidden_dim / config.num_heads } else { 0 };
                if inferred_head_dim > 0 && inferred_head_dim != default_head_dim {
                    config.explicit_head_dim = Some(inferred_head_dim);
                }
            }
        }

        // Load token embeddings
        let token_embedding =
            Self::load_apr_token_embedding(apr, data, data_offset, vocab_size, hidden_dim)?;

        // Build layers
        // GH-479: q_dim may differ from hidden_dim (Qwen3 head_dim != hidden/heads)
        let q_dim = config.q_dim();
        let kv_dim = config.kv_dim();
        let mut layers = Vec::with_capacity(num_layers);

        for layer_idx in 0..num_layers {
            layers.push(Self::load_apr_layer(
                apr,
                data,
                data_offset,
                layer_idx,
                hidden_dim,
                q_dim,
                kv_dim,
                intermediate_dim,
                transpose,
            )?);
        }

        // Output norm
        let output_norm_weight =
            apr_load_f32_tensor(apr, data, data_offset, &["model.norm.weight", "output_norm.weight"])?;
        let output_norm_bias = apr_try_load_f32(apr, data, data_offset, "model.norm.bias");

        // LM head (try HF name first, then GGUF)
        let lm_head_weight = apr_load_quantized_tensor(
            apr, data, data_offset,
            &["lm_head.weight", "output.weight"],
            hidden_dim, vocab_size, transpose,
        )?;
        let lm_head_bias = apr_try_load_f32(apr, data, data_offset, "lm_head.bias");

        // GH-278: Load learned position embeddings (GPT-2 style)
        let position_embedding =
            apr_try_load_f32(apr, data, data_offset, "model.position_embedding.weight");

        Ok(Self {
            config,
            token_embedding,
            position_embedding,
            layers,
            encoder_layers: vec![],
            encoder_output_norm_weight: None,
            encoder_output_norm_bias: None,
            output_norm_weight,
            output_norm_bias,
            lm_head_weight,
            lm_head_bias,
            #[cfg(feature = "cuda")]
            cuda_executor: None,
            #[cfg(feature = "cuda")]
            cuda_kernel_count: std::sync::atomic::AtomicU64::new(0),
            #[cfg(feature = "cuda")]
            cached_weight_names: std::sync::Mutex::new(std::collections::HashSet::new()),
        })
    }

    /// Load token embeddings from APR format.
    fn load_apr_token_embedding(
        apr: &crate::apr::MappedAprModel,
        data: &[u8],
        data_offset: usize,
        vocab_size: usize,
        hidden_dim: usize,
    ) -> Result<Vec<f32>> {
        // GH-324: Use contract-driven names for embedding discovery
        // R-01 (Meyer DbC): "unknown" — don't pretend unidentified APR model is LLaMA.
    let arch = apr.metadata.architecture.as_deref().unwrap_or("unknown");
        let arch_key = crate::tensor_names::normalize_architecture(arch);
        let embed_candidates: Vec<&str> = {
            let mut c = Vec::new();
            c.extend_from_slice(crate::tensor_names::global_names(arch_key, crate::tensor_names::GlobalTensorRole::Embedding));
            c.extend_from_slice(crate::tensor_names::global_fallback_names(crate::tensor_names::GlobalTensorRole::Embedding));
            c
        };
        // GH-324: Contract-only matching, no substring fallback
        let embed_name = embed_candidates
            .iter()
            .find_map(|&name| apr.tensors.iter().find(|t| t.name == name))
            .map(|t| t.name.as_str())
            .ok_or_else(|| RealizarError::FormatError {
                reason: format!(
                    "APR: embedding tensor not found. Tried contract names: {:?}. \
                     Add this architecture's embedding name to tensor-names-v1.yaml.",
                    embed_candidates
                ),
            })?;

        let embed_tensor = apr.find_tensor(embed_name).ok_or_else(|| RealizarError::FormatError {
            reason: "APR: embedding tensor not found".to_string(),
        })?;
        let embed_start = data_offset + embed_tensor.offset as usize;
        let embed_end = embed_start + embed_tensor.size as usize;
        if embed_end > data.len() {
            return Err(RealizarError::FormatError {
                reason: "APR: embedding tensor extends past EOF".to_string(),
            });
        }
        let embed_data = &data[embed_start..embed_end];
        dequantize_embedding(embed_data, embed_tensor.dtype.as_str(), vocab_size * hidden_dim)
    }

    /// Load a single transformer layer from APR format.
    #[allow(clippy::too_many_arguments)]
    fn load_apr_layer(
        apr: &crate::apr::MappedAprModel,
        data: &[u8],
        data_offset: usize,
        layer_idx: usize,
        hidden_dim: usize,
        q_dim: usize,
        kv_dim: usize,
        intermediate_dim: usize,
        transpose: bool,
    ) -> Result<OwnedQuantizedLayer> {
        // HF names (primary, from SafeTensors->APR pipeline)
        let hf_q = format!("model.layers.{layer_idx}.self_attn.q_proj.weight");
        let hf_k = format!("model.layers.{layer_idx}.self_attn.k_proj.weight");
        let hf_v = format!("model.layers.{layer_idx}.self_attn.v_proj.weight");
        let hf_o = format!("model.layers.{layer_idx}.self_attn.o_proj.weight");
        let hf_gate = format!("model.layers.{layer_idx}.mlp.gate_proj.weight");
        let hf_up = format!("model.layers.{layer_idx}.mlp.up_proj.weight");
        let hf_down = format!("model.layers.{layer_idx}.mlp.down_proj.weight");
        let hf_attn_norm = format!("model.layers.{layer_idx}.input_layernorm.weight");
        let hf_ffn_norm = format!("model.layers.{layer_idx}.post_attention_layernorm.weight");

        // GGUF names (fallback, from GGUF->APR path)
        let gguf_q = format!("blk.{layer_idx}.attn_q.weight");
        let gguf_k = format!("blk.{layer_idx}.attn_k.weight");
        let gguf_v = format!("blk.{layer_idx}.attn_v.weight");
        let gguf_o = format!("blk.{layer_idx}.attn_output.weight");
        let gguf_gate = format!("blk.{layer_idx}.ffn_gate.weight");
        let gguf_up = format!("blk.{layer_idx}.ffn_up.weight");
        let gguf_down = format!("blk.{layer_idx}.ffn_down.weight");
        let gguf_attn_norm = format!("blk.{layer_idx}.attn_norm.weight");
        let gguf_ffn_norm = format!("blk.{layer_idx}.ffn_norm.weight");

        // GH-479: Q dim may differ from hidden_dim (Qwen3 head_dim != hidden/heads)
        let q_weight = apr_load_quantized_tensor(apr, data, data_offset, &[&hf_q, &gguf_q], hidden_dim, q_dim, transpose)?;
        let k_weight = apr_load_quantized_tensor(apr, data, data_offset, &[&hf_k, &gguf_k], hidden_dim, kv_dim, transpose)?;
        let v_weight = apr_load_quantized_tensor(apr, data, data_offset, &[&hf_v, &gguf_v], hidden_dim, kv_dim, transpose)?;

        let qkv_weight = OwnedQKVWeights::Separate {
            q: q_weight,
            k: k_weight,
            v: v_weight,
        };

        // QKV biases (Qwen2 has separate Q, K, V biases — concatenate for CUDA)
        // GH-87: Try both HF names (SafeTensors->APR) and GGUF names (GGUF->APR Q4K)
        let hf_q_bias = format!("model.layers.{layer_idx}.self_attn.q_proj.bias");
        let hf_k_bias = format!("model.layers.{layer_idx}.self_attn.k_proj.bias");
        let hf_v_bias = format!("model.layers.{layer_idx}.self_attn.v_proj.bias");
        let gguf_q_bias = format!("blk.{layer_idx}.attn_q.bias");
        let gguf_k_bias = format!("blk.{layer_idx}.attn_k.bias");
        let gguf_v_bias = format!("blk.{layer_idx}.attn_v.bias");
        let qkv_bias = apr_try_load_f32(apr, data, data_offset, &hf_q_bias)
            .or_else(|| apr_try_load_f32(apr, data, data_offset, &gguf_q_bias))
            .and_then(|q_b| {
                let k_b = apr_try_load_f32(apr, data, data_offset, &hf_k_bias)
                    .or_else(|| apr_try_load_f32(apr, data, data_offset, &gguf_k_bias))?;
                let v_b = apr_try_load_f32(apr, data, data_offset, &hf_v_bias)
                    .or_else(|| apr_try_load_f32(apr, data, data_offset, &gguf_v_bias))?;
                let mut combined = Vec::with_capacity(q_b.len() + k_b.len() + v_b.len());
                combined.extend_from_slice(&q_b);
                combined.extend_from_slice(&k_b);
                combined.extend_from_slice(&v_b);
                Some(combined)
            });

        // GH-479: O proj maps q_dim -> hidden_dim (Qwen3 q_dim != hidden_dim)
        let o_weight = apr_load_quantized_tensor(apr, data, data_offset, &[&hf_o, &gguf_o], q_dim, hidden_dim, transpose)?;

        // FFN weights (gate is optional — GPT-2 has no SwiGLU gate)
        let ffn_gate_weight = apr_load_quantized_tensor(apr, data, data_offset, &[&hf_gate, &gguf_gate], hidden_dim, intermediate_dim, transpose).ok();
        let ffn_up_weight = apr_load_quantized_tensor(apr, data, data_offset, &[&hf_up, &gguf_up], hidden_dim, intermediate_dim, transpose)?;
        let ffn_down_weight = apr_load_quantized_tensor(apr, data, data_offset, &[&hf_down, &gguf_down], intermediate_dim, hidden_dim, transpose)?;

        // Norm weights (F32)
        let attn_norm_weight = apr_load_f32_tensor(apr, data, data_offset, &[&hf_attn_norm, &gguf_attn_norm])?;
        let ffn_norm_weight = apr_load_f32_tensor(apr, data, data_offset, &[&hf_ffn_norm, &gguf_ffn_norm]).ok();

        // GH-278: Load biases (GPT-2/phi-2 style models have biases on all projections)
        let hf_attn_norm_bias = format!("model.layers.{layer_idx}.input_layernorm.bias");
        let hf_ffn_norm_bias = format!("model.layers.{layer_idx}.post_attention_layernorm.bias");
        let hf_o_bias = format!("model.layers.{layer_idx}.self_attn.o_proj.bias");
        let hf_up_bias = format!("model.layers.{layer_idx}.mlp.up_proj.bias");
        let hf_down_bias = format!("model.layers.{layer_idx}.mlp.down_proj.bias");

        Ok(OwnedQuantizedLayer {
            attn_norm_weight,
            attn_norm_bias: apr_try_load_f32(apr, data, data_offset, &hf_attn_norm_bias),
            qkv_weight,
            qkv_bias,
            attn_output_weight: o_weight,
            attn_output_bias: apr_try_load_f32(apr, data, data_offset, &hf_o_bias),
            ffn_norm_weight,
            ffn_norm_bias: apr_try_load_f32(apr, data, data_offset, &hf_ffn_norm_bias),
            ffn_gate_weight,
            ffn_gate_bias: None,
            ffn_up_weight,
            ffn_up_bias: apr_try_load_f32(apr, data, data_offset, &hf_up_bias),
            ffn_down_weight,
            ffn_down_bias: apr_try_load_f32(apr, data, data_offset, &hf_down_bias),
            // GH-479: QK norm weights (Qwen3 per-head RMSNorm)
            // Contract: qk-norm-apr-loader-v1 §QKN-LOAD-002
            attn_q_norm_weight: apr_try_load_f32(apr, data, data_offset,
                &format!("model.layers.{layer_idx}.self_attn.q_norm.weight"))
                .or_else(|| apr_try_load_f32(apr, data, data_offset,
                    &format!("blk.{layer_idx}.attn_q_norm.weight"))),
            attn_k_norm_weight: apr_try_load_f32(apr, data, data_offset,
                &format!("model.layers.{layer_idx}.self_attn.k_norm.weight"))
                .or_else(|| apr_try_load_f32(apr, data, data_offset,
                    &format!("blk.{layer_idx}.attn_k_norm.weight"))),
        })
    }
}

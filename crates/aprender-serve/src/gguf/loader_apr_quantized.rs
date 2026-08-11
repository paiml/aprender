/// Load a quantized tensor from APR format, trying multiple names.
///
/// GH-478: For native APR q4/q8, raw quantized bytes are stored in the
/// `OwnedQuantizedTensor` (no F32 expansion at load) and tagged with
/// `APR_TYPE_Q4` / `APR_TYPE_Q8` so `fused_matmul` can dequant per-tensor
/// during forward instead of holding the full F32 working set in RAM.
/// This bounds peak RAM at *one tensor's worth* of F32 scratch instead of
/// `4 × num_params` bytes (128 GB for a 32B model).
///
/// For Conv1D architectures (`transpose=true`), the legacy dequant→transpose
/// path is retained because re-laying-out quantized blocks would require a
/// dedicated routine. Conv1D models are small enough that F32 expansion is fine.
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
    use crate::gguf::types::{APR_TYPE_Q4, APR_TYPE_Q8};

    // A candidate whose data length is 0 is a PLACEHOLDER, not a weight:
    // tied-embedding exporters write `lm_head.weight` with the full shape and
    // zero bytes. Selecting it handed matmul an empty buffer, which surfaced
    // at request time as an HTTP 500 blaming "a MoE per-expert tensor" on a
    // dense model. Skip empty candidates so a later name (the tied embedding)
    // can win.
    let (tensor, found_name) = names
        .iter()
        .find_map(|name| apr.find_tensor(name).filter(|t| t.size > 0).map(|t| (t, *name)))
        .ok_or_else(|| RealizarError::FormatError {
            reason: format!(
                "APR: no tensor with data found (tried: {}); \
                 a listed name may exist with a zero-length data buffer",
                names.join(", ")
            ),
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
        "q8" if !transpose => Ok(OwnedQuantizedTensor {
            data: raw.to_vec(),
            in_dim,
            out_dim,
            qtype: APR_TYPE_Q8,
        }),
        "q4" if !transpose => Ok(OwnedQuantizedTensor {
            data: raw.to_vec(),
            in_dim,
            out_dim,
            qtype: APR_TYPE_Q4,
        }),
        "q8" => {
            // Conv1D fallback: dequant → transpose (rare; small models only).
            let mut f32_data = crate::apr::dequant::dequantize_apr_q8(raw, num_elements);
            f32_data = transpose_f32_matrix(&f32_data, in_dim, out_dim);
            let f32_bytes: Vec<u8> = f32_data.iter().flat_map(|v| v.to_le_bytes()).collect();
            Ok(OwnedQuantizedTensor {
                data: f32_bytes,
                in_dim,
                out_dim,
                qtype: 0,
            })
        },
        "q4" => {
            // Conv1D fallback: dequant → transpose (rare; small models only).
            let mut f32_data = crate::apr::dequant::dequantize_apr_q4(raw, num_elements);
            f32_data = transpose_f32_matrix(&f32_data, in_dim, out_dim);
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
    let raw = &data[start..end];
    // GH-180: Dispatch on dtype — FP16 APR models store biases as F16
    match tensor.dtype.as_str() {
        "F16" => Some(
            raw.chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect(),
        ),
        _ => Some(
            raw.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        ),
    }
}

/// Infer vocab_size from APR metadata or embedding tensor shape.
/// GH-337: Infer vocab size from metadata or embedding tensor shape.
///
/// **Design by Contract**: No hardcoded fallback. Returns 0 on failure
/// (callers validate via contract gate).
fn apr_infer_vocab_size(apr: &crate::apr::MappedAprModel) -> usize {
    if let Some(v) = apr.metadata.vocab_size {
        if v > 0 {
            return v;
        }
    }
    // Try embedding tensor shape (first dimension = vocab size)
    apr.tensors
        .iter()
        .find(|t| {
            t.name.contains("embed_tokens")
                || t.name.contains("tok_embeddings")
                || t.name.contains("token_embd")
        })
        .and_then(|t| t.shape.first().copied())
        .unwrap_or(0)
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
        let t0 = std::time::Instant::now();
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
        // PMAT-888: Gemma2 POST-attention / POST-FFN RMSNorms are loaded ONLY for
        // Gemma2. The HF tensor name `post_attention_layernorm.weight` is the
        // *FFN (pre-feedforward) norm* for qwen2/llama/mistral/phi/deepseek/qwen3
        // (see `tensor_names_fallback::FfnNormWeight`), NOT a post-attention norm.
        // Loading it into the Gemma2 `post_attn_norm_weight` slot — and the
        // `ffn_block` forward then applying it (it gates only on `is_some()`, not on
        // arch) — injected a spurious extra RMSNorm into EVERY non-Gemma2 `.apr`,
        // producing garbage output (PMAT-887 repro: `çļĦåıªæĺ¯…`). The GGUF loader
        // never hit this because it reads the disambiguated `post_attention_norm`
        // / `post_ffw_norm` names (no "layer"), which do not exist in non-Gemma2
        // GGUFs. Gate the APR post-norm load on the architecture, mirroring GGUF's
        // `None` for non-Gemma2.
        let is_gemma2 = config.is_gemma2();
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
                is_gemma2,
            )?);
        }

        // Output norm
        let output_norm_weight =
            apr_load_f32_tensor(apr, data, data_offset, &["model.norm.weight", "output_norm.weight"])?;
        let output_norm_bias = apr_try_load_f32(apr, data, data_offset, "model.norm.bias");

        // LM head (try HF name first, then GGUF, then the TIED EMBEDDING).
        // Weight-tied exports (Qwen2 0.5B, most <2B models) either omit
        // lm_head entirely or write it as a zero-length placeholder; the
        // embedding matrix IS the head and has the identical row-major
        // [vocab, hidden] layout, so it can be used verbatim.
        //
        // Say so out loud: a head silently sourced from somewhere other than
        // the named tensor is exactly the kind of thing the next person
        // debugging this model needs to know before anything else.
        if ["lm_head.weight", "output.weight"]
            .iter()
            .any(|n| apr.find_tensor(n).is_some_and(|t| t.size == 0))
        {
            eprintln!(
                "[tied-embeddings] LM head tensor is a 0-byte placeholder — \
                 using the token embedding matrix as the output head"
            );
        }
        let lm_head_weight = apr_load_quantized_tensor(
            apr, data, data_offset,
            &[
                "lm_head.weight",
                "output.weight",
                "model.embed_tokens.weight",
                "embed_tokens.weight",
                "token_embd.weight",
                "tok_embeddings.weight",
            ],
            hidden_dim, vocab_size, transpose,
        )?;
        let lm_head_bias = apr_try_load_f32(apr, data, data_offset, "lm_head.bias");

        // GH-278: Load learned position embeddings (GPT-2 style)
        let position_embedding =
            apr_try_load_f32(apr, data, data_offset, "model.position_embedding.weight");

        let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "[GH-175] OwnedQuantizedModel::from_apr: {} layers loaded in {:.1}ms",
            num_layers, load_ms
        );

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
        let embed_name = apr
            .tensors
            .iter()
            .find(|t| {
                t.name.contains("embed_tokens")
                    || t.name.contains("tok_embeddings")
                    || t.name.contains("token_embd")
            })
            .map(|t| t.name.as_str())
            .ok_or_else(|| RealizarError::FormatError {
                reason: "APR: embedding tensor not found".to_string(),
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
    ///
    /// PMAT-888: `is_gemma2` gates loading the Gemma2-only post-attention /
    /// post-FFN RMSNorms. For non-Gemma2 archs, `post_attention_layernorm.weight`
    /// is the FFN norm (already loaded as `ffn_norm_weight`), so it must NOT be
    /// loaded into the post-attn-norm slot.
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
        is_gemma2: bool,
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
        // GH-87: Try both HF names (SafeTensors→APR) and GGUF names (GGUF→APR Q4K)
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
        // GH-87: Try both HF names and GGUF names for all bias tensors
        let hf_attn_norm_bias = format!("model.layers.{layer_idx}.input_layernorm.bias");
        let hf_ffn_norm_bias = format!("model.layers.{layer_idx}.post_attention_layernorm.bias");
        let hf_o_bias = format!("model.layers.{layer_idx}.self_attn.o_proj.bias");
        let hf_up_bias = format!("model.layers.{layer_idx}.mlp.up_proj.bias");
        let hf_down_bias = format!("model.layers.{layer_idx}.mlp.down_proj.bias");
        let gguf_attn_norm_bias = format!("blk.{layer_idx}.attn_norm.bias");
        let gguf_ffn_norm_bias = format!("blk.{layer_idx}.ffn_norm.bias");
        let gguf_o_bias = format!("blk.{layer_idx}.attn_output.bias");
        let gguf_up_bias = format!("blk.{layer_idx}.ffn_up.bias");
        let gguf_down_bias = format!("blk.{layer_idx}.ffn_down.bias");

        Ok(OwnedQuantizedLayer {
            attn_norm_weight,
            attn_norm_bias: apr_try_load_f32(apr, data, data_offset, &hf_attn_norm_bias)
                .or_else(|| apr_try_load_f32(apr, data, data_offset, &gguf_attn_norm_bias)),
            qkv_weight,
            qkv_bias,
            attn_output_weight: o_weight,
            attn_output_bias: apr_try_load_f32(apr, data, data_offset, &hf_o_bias)
                .or_else(|| apr_try_load_f32(apr, data, data_offset, &gguf_o_bias)),
            ffn_norm_weight,
            ffn_norm_bias: apr_try_load_f32(apr, data, data_offset, &hf_ffn_norm_bias)
                .or_else(|| apr_try_load_f32(apr, data, data_offset, &gguf_ffn_norm_bias)),
            ffn_gate_weight,
            ffn_gate_bias: None,
            ffn_up_weight,
            ffn_up_bias: apr_try_load_f32(apr, data, data_offset, &hf_up_bias)
                .or_else(|| apr_try_load_f32(apr, data, data_offset, &gguf_up_bias)),
            ffn_down_weight,
            ffn_down_bias: apr_try_load_f32(apr, data, data_offset, &hf_down_bias)
                .or_else(|| apr_try_load_f32(apr, data, data_offset, &gguf_down_bias)),
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
            // PMAT-810 / PMAT-888: Gemma2 post-attention / post-FFN RMSNorm
            // (None for every other architecture). Gated on `is_gemma2` because the
            // HF name `post_attention_layernorm.weight` is the FFN norm for
            // qwen2/llama/mistral/phi/deepseek/qwen3 — loading it here would apply a
            // spurious extra RMSNorm in the forward (which gates only on `is_some()`)
            // and corrupt all output. Gemma2's true FFN norm is the distinct
            // `pre_feedforward_layernorm`, so this name collision only harms
            // non-Gemma2 models.
            post_attn_norm_weight: if is_gemma2 {
                apr_try_load_f32(apr, data, data_offset,
                    &format!("model.layers.{layer_idx}.post_attention_layernorm.weight"))
                    .or_else(|| apr_try_load_f32(apr, data, data_offset,
                        &format!("blk.{layer_idx}.post_attention_norm.weight")))
            } else {
                None
            },
            post_ffw_norm_weight: if is_gemma2 {
                apr_try_load_f32(apr, data, data_offset,
                    &format!("model.layers.{layer_idx}.post_feedforward_layernorm.weight"))
                    .or_else(|| apr_try_load_f32(apr, data, data_offset,
                        &format!("blk.{layer_idx}.post_ffw_norm.weight")))
            } else {
                None
            },
        })
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod gh478_per_layer_dequant_tests {
    //! GH-478: Falsifiable invariant — native APR q4/q8 tensors MUST stay
    //! quantized at load time. This test fails if the loader regresses to
    //! F32 expansion (would OOM 32B models on 128 GB hosts).

    use crate::apr::{HEADER_SIZE, MAGIC, MappedAprModel};
    use crate::gguf::types::{APR_TYPE_Q4, APR_TYPE_Q8};
    use std::io::Write;

    /// Build a minimal APR v2 file with a single quantized tensor.
    ///
    /// `dtype_byte` = 128 for APR q4, 129 for APR q8. `payload` is the raw
    /// quantized bytes the test wants to round-trip.
    fn build_single_tensor_apr(name: &str, dtype_byte: u8, shape: &[u64], payload: &[u8]) -> Vec<u8> {
        let metadata = b"{}";
        let metadata_padded = metadata.len().div_ceil(64) * 64;

        // Tensor index entry: name_len(u16) + name + dtype(u8) + rank(u8) +
        //                     shape(u64 × rank) + offset(u64) + size(u64)
        let mut entry = Vec::new();
        entry.extend_from_slice(&(name.len() as u16).to_le_bytes());
        entry.extend_from_slice(name.as_bytes());
        entry.push(dtype_byte);
        entry.push(shape.len() as u8);
        for &d in shape {
            entry.extend_from_slice(&d.to_le_bytes());
        }
        entry.extend_from_slice(&0u64.to_le_bytes()); // offset within data
        entry.extend_from_slice(&(payload.len() as u64).to_le_bytes());

        let tensor_index_offset = (HEADER_SIZE + metadata_padded) as u64;
        let data_offset = tensor_index_offset + entry.len() as u64;
        let total = data_offset as usize + payload.len();

        let mut out = vec![0u8; total];
        out[0..4].copy_from_slice(&MAGIC);
        out[4] = 2; // version major
        out[5] = 0; // version minor
        out[8..12].copy_from_slice(&1u32.to_le_bytes()); // tensor_count
        out[12..20].copy_from_slice(&(HEADER_SIZE as u64).to_le_bytes()); // metadata_offset
        out[20..24].copy_from_slice(&(metadata.len() as u32).to_le_bytes()); // metadata_size
        out[24..32].copy_from_slice(&tensor_index_offset.to_le_bytes());
        out[32..40].copy_from_slice(&data_offset.to_le_bytes());

        out[HEADER_SIZE..HEADER_SIZE + metadata.len()].copy_from_slice(metadata);
        let idx = tensor_index_offset as usize;
        out[idx..idx + entry.len()].copy_from_slice(&entry);
        let data_start = data_offset as usize;
        out[data_start..data_start + payload.len()].copy_from_slice(payload);
        out
    }

    fn write_tempfile(bytes: &[u8]) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().expect("tempfile");
        f.write_all(bytes).expect("write apr");
        f
    }

    #[test]
    fn apr_q4_load_keeps_raw_bytes_not_f32_expansion() {
        // 32×4 = 128 elements. q4 block = 18 bytes per 32 elems → 4 blocks = 72 bytes.
        let in_dim = 32usize;
        let out_dim = 4usize;
        let num_elements = in_dim * out_dim;
        let raw_q4 = vec![0u8; 4 * 18]; // 4 blocks of 18 bytes

        let file = write_tempfile(&build_single_tensor_apr(
            "ffn_up.weight",
            128, // APR-native q4
            &[out_dim as u64, in_dim as u64],
            &raw_q4,
        ));
        let apr = MappedAprModel::from_path(file.path()).expect("load apr");

        let tensor = super::apr_load_quantized_tensor(
            &apr,
            apr.data(),
            apr.data_offset() as usize,
            &["ffn_up.weight"],
            in_dim,
            out_dim,
            false, // transpose=false: per-layer dequant path
        )
        .expect("load tensor");

        // INVARIANT: raw quantized bytes, NOT F32 expansion.
        assert_eq!(tensor.data.len(), raw_q4.len(),
            "APR q4 loaded tensor must keep raw quantized bytes (got {}, expected {})",
            tensor.data.len(), raw_q4.len());
        assert_ne!(tensor.data.len(), num_elements * 4,
            "APR q4 loaded tensor must NOT be F32-expanded ({}B = 4×{})",
            num_elements * 4, num_elements);
        assert_eq!(tensor.qtype, APR_TYPE_Q4, "qtype must tag as APR_TYPE_Q4");
        assert_eq!(tensor.in_dim, in_dim);
        assert_eq!(tensor.out_dim, out_dim);
    }

    #[test]
    fn apr_q8_load_keeps_raw_bytes_not_f32_expansion() {
        // q8 layout = 4-byte scale + 1 byte/elem. 32×4 = 128 elems → 4 + 128 = 132 bytes.
        let in_dim = 32usize;
        let out_dim = 4usize;
        let num_elements = in_dim * out_dim;
        let raw_q8 = vec![0u8; 4 + num_elements];

        let file = write_tempfile(&build_single_tensor_apr(
            "ffn_up.weight",
            129, // APR-native q8
            &[out_dim as u64, in_dim as u64],
            &raw_q8,
        ));
        let apr = MappedAprModel::from_path(file.path()).expect("load apr");

        let tensor = super::apr_load_quantized_tensor(
            &apr,
            apr.data(),
            apr.data_offset() as usize,
            &["ffn_up.weight"],
            in_dim,
            out_dim,
            false,
        )
        .expect("load tensor");

        assert_eq!(tensor.data.len(), raw_q8.len(),
            "APR q8 loaded tensor must keep raw quantized bytes");
        assert_ne!(tensor.data.len(), num_elements * 4,
            "APR q8 loaded tensor must NOT be F32-expanded");
        assert_eq!(tensor.qtype, APR_TYPE_Q8, "qtype must tag as APR_TYPE_Q8");
    }

    #[test]
    fn apr_q4_conv1d_transpose_still_dequants_to_f32() {
        // Conv1D path (transpose=true) is intentionally kept on the legacy
        // dequant→transpose fallback. Assert that contract.
        let in_dim = 32usize;
        let out_dim = 4usize;
        let num_elements = in_dim * out_dim;
        let raw_q4 = vec![0u8; 4 * 18];

        let file = write_tempfile(&build_single_tensor_apr(
            "ffn_up.weight",
            128,
            &[out_dim as u64, in_dim as u64],
            &raw_q4,
        ));
        let apr = MappedAprModel::from_path(file.path()).expect("load apr");

        let tensor = super::apr_load_quantized_tensor(
            &apr,
            apr.data(),
            apr.data_offset() as usize,
            &["ffn_up.weight"],
            in_dim,
            out_dim,
            true, // Conv1D path
        )
        .expect("load tensor");

        assert_eq!(tensor.data.len(), num_elements * 4,
            "Conv1D (transpose=true) path keeps legacy F32 expansion");
        assert_eq!(tensor.qtype, 0, "Conv1D path flattens qtype to F32");
    }

    /// GH-478: End-to-end memory-bound check on a real APR-native q4/q8 model.
    ///
    /// Iterates all q4/q8 tensors via `apr_load_quantized_tensor` and asserts
    /// the total stored byte count stays at the on-disk raw-quantized size,
    /// never inflating to 4× (F32) — which would OOM large models.
    ///
    /// Gated on `GH478_APR_Q4_MODEL` so CI/regular `cargo test` skip it.
    /// Run:
    ///   GH478_APR_Q4_MODEL=/tmp/gh478-qwen-1.5b-aprq4.apr \
    ///   cargo test -p aprender-serve --lib \
    ///     gh478_real_model_load_stays_bounded -- --ignored --nocapture
    #[test]
    #[ignore]
    fn gh478_real_model_load_stays_bounded() {
        let path = match std::env::var("GH478_APR_Q4_MODEL") {
            Ok(p) => p,
            Err(_) => return, // gated
        };
        let apr = MappedAprModel::from_path(&path).expect("mmap apr");
        let data = apr.data();
        let data_offset = apr.data_offset() as usize;

        let mut total_raw_bytes: u64 = 0;
        let mut total_stored_bytes: u64 = 0;
        let mut total_elements: u64 = 0;
        let mut qtensor_count = 0usize;

        for tensor in &apr.tensors {
            let dtype = tensor.dtype.as_str();
            if dtype != "q4" && dtype != "q8" {
                continue;
            }
            if tensor.shape.len() != 2 {
                continue; // skip 1-D and Conv1D-transpose edge cases
            }
            let out_dim = tensor.shape[0] as usize;
            let in_dim = tensor.shape[1] as usize;
            let raw_size = tensor.size;
            let expected_f32_size = (in_dim * out_dim * 4) as u64;

            let loaded = super::apr_load_quantized_tensor(
                &apr, data, data_offset, &[tensor.name.as_str()],
                in_dim, out_dim, false,
            ).expect("load tensor");

            total_raw_bytes += raw_size;
            total_stored_bytes += loaded.data.len() as u64;
            total_elements += (in_dim * out_dim) as u64;
            qtensor_count += 1;

            // Per-tensor invariant: raw bytes, not F32 expansion.
            assert_eq!(loaded.data.len() as u64, raw_size,
                "tensor {}: data.len()={} raw_size={} expected_f32={} — regression!",
                tensor.name, loaded.data.len(), raw_size, expected_f32_size);
        }

        let stored_gb = total_stored_bytes as f64 / 1e9;
        let would_be_f32_gb = (total_elements * 4) as f64 / 1e9;
        eprintln!(
            "[GH-478] {} q-tensors  stored={:.3} GB  would-be-F32={:.3} GB  ratio={:.1}×",
            qtensor_count, stored_gb, would_be_f32_gb, would_be_f32_gb / stored_gb
        );
        assert!(qtensor_count > 0, "no q4/q8 tensors found — wrong model?");
        assert_eq!(total_stored_bytes, total_raw_bytes,
            "total stored bytes must equal on-disk raw quant bytes");
        assert!(would_be_f32_gb > stored_gb * 2.0,
            "falsification sanity: F32 expansion must be ≥2× the stored size");
    }

    fn read_rss_gb() -> f64 {
        let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
        for line in status.lines() {
            if let Some(rest) = line.strip_prefix("VmRSS:") {
                let kb: f64 = rest.trim().trim_end_matches(" kB")
                    .parse().unwrap_or(0.0);
                return kb / 1_048_576.0; // KiB → GiB (close enough)
            }
        }
        0.0
    }
}

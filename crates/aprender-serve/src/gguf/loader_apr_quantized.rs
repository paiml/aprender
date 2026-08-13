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

/// Names an APR file may use for the token-embedding matrix.
const APR_EMBED_NAME_FRAGMENTS: [&str; 3] = ["embed_tokens", "tok_embeddings", "token_embd"];

/// Names an APR file may use for the output projection.
const APR_LM_HEAD_NAMES: [&str; 2] = ["lm_head.weight", "output.weight"];

/// Find the token-embedding tensor's name in an APR file.
fn apr_find_embedding_name(apr: &crate::apr::MappedAprModel) -> Option<&str> {
    apr.tensors
        .iter()
        .find(|t| APR_EMBED_NAME_FRAGMENTS.iter().any(|frag| t.name.contains(frag)))
        .map(|t| t.name.as_str())
}

/// Is the output projection a tied-word-embedding placeholder? (#2309, #2441)
///
/// An `.apr` written from a `tie_word_embeddings=true` checkpoint records
/// `lm_head.weight` with its full `[vocab, hidden]` shape but ZERO bytes of data:
/// the matrix it names is the one already stored as `model.embed_tokens.weight`.
/// Absent an `lm_head` descriptor entirely, the tie is likewise implied.
fn apr_lm_head_is_tied(apr: &crate::apr::MappedAprModel) -> bool {
    APR_LM_HEAD_NAMES
        .iter()
        .find_map(|name| apr.find_tensor(name))
        .is_none_or(|t| t.size == 0)
}

/// Load the output projection, honoring tied word embeddings (#2309, #2441).
///
/// When `lm_head.weight` is a 0-byte placeholder, the embedding matrix IS the
/// output projection: both are row-major `[vocab, hidden]`, so the same bytes are
/// re-registered as `in_dim = hidden, out_dim = vocab` with no transpose. Loading
/// the placeholder verbatim instead produced an `OwnedQuantizedTensor` with an
/// empty `data` buffer, and every decode died in `fused_matmul` with
/// "matmul weight has EMPTY data buffer".
fn apr_load_lm_head(
    apr: &crate::apr::MappedAprModel,
    data: &[u8],
    data_offset: usize,
    hidden_dim: usize,
    vocab_size: usize,
    transpose: bool,
) -> Result<OwnedQuantizedTensor> {
    if apr_lm_head_is_tied(apr) {
        let embed_name = apr_find_embedding_name(apr).ok_or_else(|| RealizarError::FormatError {
            reason: "APR: lm_head.weight is a 0-byte tied-embedding placeholder but no \
                     embedding tensor exists to tie it to"
                .to_string(),
        })?;
        eprintln!(
            "[#2309] lm_head is a 0-byte tied-embedding placeholder — tying output projection to '{embed_name}'"
        );
        // The embedding is stored [vocab, hidden] row-major in every architecture,
        // including the Conv1D ones that set `transpose` for their projections, so
        // the tied head is never transposed.
        return apr_load_quantized_tensor(
            apr,
            data,
            data_offset,
            &[embed_name],
            hidden_dim,
            vocab_size,
            false,
        );
    }
    apr_load_quantized_tensor(
        apr,
        data,
        data_offset,
        &APR_LM_HEAD_NAMES,
        hidden_dim,
        vocab_size,
        transpose,
    )
}

/// Decode the raw bytes of a dense (unquantized) APR tensor into F32.
///
/// #2443: This is the ONLY place in the APR loader that turns tensor bytes into
/// `f32`, and it refuses to do so without consulting the dtype recorded in the
/// tensor index. The two helpers it replaced read every 1-D tensor as
/// `chunks_exact(4) -> f32::from_le_bytes` — the `F16` arm of one of them was
/// the only dtype ever checked, and everything else fell through to the 4-byte
/// read. A BF16 export (`qwen2.5-coder-0.5b-instruct.apr` stores 290 of its 291
/// tensors as BF16) therefore decoded all 49 RMSNorm weights and all 72 q/k/v
/// biases at HALF their element count with arbitrary values, and nothing
/// crashed: `gguf/ops.rs::rms_norm` takes `hidden_dim` from `weight.len()`, so a
/// 448-long norm silently re-sliced each 896-wide activation into two rows. The
/// model produced fluent-looking tokens that did not depend on the prompt.
///
/// Two properties make that class unrepresentable rather than merely fixed:
///
/// 1. **No silent fallthrough.** A dtype that is not a dense float is an error,
///    never a reinterpretation. Per the rule the CLI adopted in #2407, a path
///    that cannot work must fail instead of emitting plausible numbers.
/// 2. **The byte count must agree with the shape.** `raw.len()` has to equal
///    `shape.product() * width`, so decoding at the wrong width cannot succeed
///    even if a future dtype arm is added with the wrong element size — the very
///    mismatch this bug relied on (1792 bytes read as 448 F32 instead of 896
///    BF16) is now the thing that trips the guard.
fn apr_decode_dense_float(
    name: &str,
    dtype: &str,
    shape: &[usize],
    raw: &[u8],
) -> Result<Vec<f32>> {
    let width = match dtype {
        "F32" => 4usize,
        "F16" | "BF16" => 2usize,
        other => {
            return Err(RealizarError::FormatError {
                reason: format!(
                    "APR: tensor {name} has dtype {other}, which is not a dense float type; \
                     refusing to reinterpret its bytes as F32 (that yields a plausible but \
                     wrong tensor rather than an error)"
                ),
            });
        }
    };

    let expected_elems: usize = shape.iter().product();
    if shape.is_empty() {
        if raw.len() % width != 0 {
            return Err(RealizarError::FormatError {
                reason: format!(
                    "APR: tensor {name} ({dtype}) has {} bytes, not a multiple of {width}",
                    raw.len()
                ),
            });
        }
    } else if raw.len() != expected_elems * width {
        return Err(RealizarError::FormatError {
            reason: format!(
                "APR: tensor {name} ({dtype}) has {} bytes but shape {shape:?} \
                 needs {} ({expected_elems} elements x {width} bytes)",
                raw.len(),
                expected_elems * width
            ),
        });
    }

    let values: Vec<f32> = match dtype {
        "F32" => raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        "F16" => raw
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        // BF16 is the upper 16 bits of the F32 bit pattern.
        _ => raw
            .chunks_exact(2)
            .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
    };
    Ok(values)
}

/// Try loading an optional dense float tensor from APR format, trying multiple
/// names in order.
///
/// `Ok(None)` means NO listed name exists — the only benign reason to skip a
/// tensor. A name that DOES exist but cannot be decoded is `Err`, never `None`:
/// #2443's other half was that a bias silently dropped is as wrong an answer as
/// a bias decoded at the wrong width, and both used to be unobservable.
fn apr_try_load_dense_float(
    apr: &crate::apr::MappedAprModel,
    data: &[u8],
    data_offset: usize,
    names: &[&str],
) -> Result<Option<Vec<f32>>> {
    let Some((tensor, found_name)) = names
        .iter()
        .find_map(|name| apr.find_tensor(name).map(|t| (t, *name)))
    else {
        return Ok(None);
    };
    let start = data_offset + tensor.offset as usize;
    let end = start + tensor.size as usize;
    if end > data.len() {
        return Err(RealizarError::FormatError {
            reason: format!("APR: tensor {found_name} extends past EOF"),
        });
    }
    apr_decode_dense_float(found_name, &tensor.dtype, &tensor.shape, &data[start..end]).map(Some)
}

/// Reject an RMSNorm/LayerNorm weight whose element count is not `hidden_dim`.
///
/// #2443's damage was done downstream of loading: `gguf/ops.rs::rms_norm` infers
/// `hidden_dim` from `weight.len()` and `seq_len` from `input.len() /
/// hidden_dim`, so it cannot distinguish "a 448-wide norm and one 896-wide
/// token" from "a 448-wide norm and two 448-wide tokens" — a half-width norm is
/// arithmetically valid there and produces numbers instead of an error. The
/// loader is the last place that still knows `hidden_dim` from the config, so it
/// is where the check belongs: a norm of the wrong width cannot reach the
/// forward pass at all.
fn apr_check_norm_width(name: &str, weight: &[f32], hidden_dim: usize) -> Result<()> {
    if weight.len() == hidden_dim {
        return Ok(());
    }
    Err(RealizarError::FormatError {
        reason: format!(
            "APR: norm weight {name} decoded to {} elements but hidden_dim is {hidden_dim}; \
             refusing to load (a wrong-width norm silently re-slices every activation \
             instead of failing)",
            weight.len()
        ),
    })
}

/// Load a required dense float tensor from APR format, trying multiple names.
fn apr_load_f32_tensor(
    apr: &crate::apr::MappedAprModel,
    data: &[u8],
    data_offset: usize,
    names: &[&str],
) -> Result<Vec<f32>> {
    apr_try_load_dense_float(apr, data, data_offset, names)?.ok_or_else(|| {
        RealizarError::FormatError {
            reason: format!("APR: tensor not found (tried: {})", names.join(", ")),
        }
    })
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
                let inferred_head_dim = q_out_dim.checked_div(config.num_heads).unwrap_or(0);
                let default_head_dim = hidden_dim.checked_div(config.num_heads).unwrap_or(0);
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
        apr_check_norm_width("model.norm.weight", &output_norm_weight, hidden_dim)?;
        let output_norm_bias =
            apr_try_load_dense_float(apr, data, data_offset, &["model.norm.bias"])?;

        // LM head. Tied-embedding resolution lives in `apr_load_lm_head`:
        // weight-tied exports (Qwen2 0.5B, most <2B models) write lm_head as a
        // 0-byte placeholder and the embedding matrix IS the head.
        //
        // Preferred over the inline name-fallback list this replaced, for two
        // reasons that are not stylistic: it HARD-FAILS when the head is tied but
        // no embedding tensor exists, instead of falling through to a confusing
        // downstream shape error; and it loads the tied head with
        // `transpose: false` unconditionally, because the embedding is stored
        // [vocab, hidden] row-major in every architecture including the Conv1D
        // ones that set `transpose` for their projections.
        // LM head (try HF name first, then GGUF; tied embeddings resolved in-loader)
        let lm_head_weight =
            apr_load_lm_head(apr, data, data_offset, hidden_dim, vocab_size, transpose)?;
        let lm_head_bias = apr_try_load_dense_float(apr, data, data_offset, &["lm_head.bias"])?;

        // GH-278: Load learned position embeddings (GPT-2 style)
        let position_embedding = apr_try_load_dense_float(
            apr,
            data,
            data_offset,
            &["model.position_embedding.weight"],
        )?;

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
        let embed_name =
            apr_find_embedding_name(apr).ok_or_else(|| RealizarError::FormatError {
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
        let qkv_bias = match (
            apr_try_load_dense_float(apr, data, data_offset, &[&hf_q_bias, &gguf_q_bias])?,
            apr_try_load_dense_float(apr, data, data_offset, &[&hf_k_bias, &gguf_k_bias])?,
            apr_try_load_dense_float(apr, data, data_offset, &[&hf_v_bias, &gguf_v_bias])?,
        ) {
            (Some(q_b), Some(k_b), Some(v_b)) => {
                let mut combined = Vec::with_capacity(q_b.len() + k_b.len() + v_b.len());
                combined.extend_from_slice(&q_b);
                combined.extend_from_slice(&k_b);
                combined.extend_from_slice(&v_b);
                Some(combined)
            }
            // A model with only some of Q/K/V biased has no concatenated layout
            // to offer; that is the pre-existing contract, kept verbatim.
            _ => None,
        };

        // GH-479: O proj maps q_dim -> hidden_dim (Qwen3 q_dim != hidden_dim)
        let o_weight = apr_load_quantized_tensor(apr, data, data_offset, &[&hf_o, &gguf_o], q_dim, hidden_dim, transpose)?;

        // FFN weights (gate is optional — GPT-2 has no SwiGLU gate)
        let ffn_gate_weight = apr_load_quantized_tensor(apr, data, data_offset, &[&hf_gate, &gguf_gate], hidden_dim, intermediate_dim, transpose).ok();
        let ffn_up_weight = apr_load_quantized_tensor(apr, data, data_offset, &[&hf_up, &gguf_up], hidden_dim, intermediate_dim, transpose)?;
        let ffn_down_weight = apr_load_quantized_tensor(apr, data, data_offset, &[&hf_down, &gguf_down], intermediate_dim, hidden_dim, transpose)?;

        // Norm weights (dense float — F32, F16 or BF16 depending on the export)
        let attn_norm_weight =
            apr_load_f32_tensor(apr, data, data_offset, &[&hf_attn_norm, &gguf_attn_norm])?;
        // #2443: `.ok()` here used to swallow a decode failure as "this model has
        // no FFN norm" (true for GPT-2). Only ABSENCE may yield `None` now.
        let ffn_norm_weight =
            apr_try_load_dense_float(apr, data, data_offset, &[&hf_ffn_norm, &gguf_ffn_norm])?;
        apr_check_norm_width(&hf_attn_norm, &attn_norm_weight, hidden_dim)?;
        if let Some(w) = ffn_norm_weight.as_deref() {
            apr_check_norm_width(&hf_ffn_norm, w, hidden_dim)?;
        }

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

        let hf_q_norm = format!("model.layers.{layer_idx}.self_attn.q_norm.weight");
        let gguf_q_norm = format!("blk.{layer_idx}.attn_q_norm.weight");
        let hf_k_norm = format!("model.layers.{layer_idx}.self_attn.k_norm.weight");
        let gguf_k_norm = format!("blk.{layer_idx}.attn_k_norm.weight");
        let hf_post_attn_norm = format!("model.layers.{layer_idx}.post_attention_layernorm.weight");
        let gguf_post_attn_norm = format!("blk.{layer_idx}.post_attention_norm.weight");
        let hf_post_ffw_norm = format!("model.layers.{layer_idx}.post_feedforward_layernorm.weight");
        let gguf_post_ffw_norm = format!("blk.{layer_idx}.post_ffw_norm.weight");

        Ok(OwnedQuantizedLayer {
            attn_norm_weight,
            attn_norm_bias: apr_try_load_dense_float(
                apr,
                data,
                data_offset,
                &[&hf_attn_norm_bias, &gguf_attn_norm_bias],
            )?,
            qkv_weight,
            qkv_bias,
            attn_output_weight: o_weight,
            attn_output_bias: apr_try_load_dense_float(
                apr,
                data,
                data_offset,
                &[&hf_o_bias, &gguf_o_bias],
            )?,
            ffn_norm_weight,
            ffn_norm_bias: apr_try_load_dense_float(
                apr,
                data,
                data_offset,
                &[&hf_ffn_norm_bias, &gguf_ffn_norm_bias],
            )?,
            ffn_gate_weight,
            ffn_gate_bias: None,
            ffn_up_weight,
            ffn_up_bias: apr_try_load_dense_float(
                apr,
                data,
                data_offset,
                &[&hf_up_bias, &gguf_up_bias],
            )?,
            ffn_down_weight,
            ffn_down_bias: apr_try_load_dense_float(
                apr,
                data,
                data_offset,
                &[&hf_down_bias, &gguf_down_bias],
            )?,
            // GH-479: QK norm weights (Qwen3 per-head RMSNorm)
            // Contract: qk-norm-apr-loader-v1 §QKN-LOAD-002
            attn_q_norm_weight: apr_try_load_dense_float(
                apr,
                data,
                data_offset,
                &[&hf_q_norm, &gguf_q_norm],
            )?,
            attn_k_norm_weight: apr_try_load_dense_float(
                apr,
                data,
                data_offset,
                &[&hf_k_norm, &gguf_k_norm],
            )?,
            // PMAT-810 / PMAT-888: Gemma2 post-attention / post-FFN RMSNorm
            // (None for every other architecture). Gated on `is_gemma2` because the
            // HF name `post_attention_layernorm.weight` is the FFN norm for
            // qwen2/llama/mistral/phi/deepseek/qwen3 — loading it here would apply a
            // spurious extra RMSNorm in the forward (which gates only on `is_some()`)
            // and corrupt all output. Gemma2's true FFN norm is the distinct
            // `pre_feedforward_layernorm`, so this name collision only harms
            // non-Gemma2 models.
            post_attn_norm_weight: if is_gemma2 {
                apr_try_load_dense_float(
                    apr,
                    data,
                    data_offset,
                    &[&hf_post_attn_norm, &gguf_post_attn_norm],
                )?
            } else {
                None
            },
            post_ffw_norm_weight: if is_gemma2 {
                apr_try_load_dense_float(
                    apr,
                    data,
                    data_offset,
                    &[&hf_post_ffw_norm, &gguf_post_ffw_norm],
                )?
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

#[cfg(all(test, not(target_arch = "wasm32")))]
mod issue_2443_bf16_dense_tensor_tests {
    //! #2443: the APR bf16 body path emitted the SAME tokens for every prompt.
    //!
    //! `apr run qwen2.5-coder-0.5b-instruct.apr` answered three semantically
    //! unrelated prompts with a byte-identical run of 12 `<|fim_suffix|>`, exit
    //! code 0. The container was not at fault (the same weights as SafeTensors
    //! answered "4"), nor was bf16 arithmetic (the 2-D bf16 weights dispatch
    //! correctly in `matmul_fused.rs`). The intersection was: every 1-D tensor
    //! in an .apr was read as `chunks_exact(4) -> f32` with no dtype dispatch,
    //! so 290 BF16 tensors — all 49 RMSNorm weights, all 72 q/k/v biases —
    //! decoded to HALF their elements with unrelated values.
    //!
    //! These are the falsifiers for that. Each asserts a VALUE or a REFUSAL,
    //! not a shape: a loader that returns the right count of wrong numbers
    //! fails them just as hard as the original did.

    use crate::apr::{HEADER_SIZE, MAGIC, MappedAprModel};
    use std::io::Write;

    /// APR dtype bytes, as written into the tensor index.
    const DTYPE_F32: u8 = 0;
    const DTYPE_F16: u8 = 1;
    const DTYPE_Q4_K: u8 = 12;
    const DTYPE_BF16: u8 = 30;

    fn build_single_tensor_apr(
        name: &str,
        dtype_byte: u8,
        shape: &[u64],
        payload: &[u8],
    ) -> Vec<u8> {
        let metadata = b"{}";
        let metadata_padded = metadata.len().div_ceil(64) * 64;

        let mut entry = Vec::new();
        entry.extend_from_slice(&(name.len() as u16).to_le_bytes());
        entry.extend_from_slice(name.as_bytes());
        entry.push(dtype_byte);
        entry.push(shape.len() as u8);
        for &d in shape {
            entry.extend_from_slice(&d.to_le_bytes());
        }
        entry.extend_from_slice(&0u64.to_le_bytes());
        entry.extend_from_slice(&(payload.len() as u64).to_le_bytes());

        let tensor_index_offset = (HEADER_SIZE + metadata_padded) as u64;
        let data_offset = tensor_index_offset + entry.len() as u64;
        let total = data_offset as usize + payload.len();

        let mut out = vec![0u8; total];
        out[0..4].copy_from_slice(&MAGIC);
        out[4] = 2;
        out[5] = 0;
        out[8..12].copy_from_slice(&1u32.to_le_bytes());
        out[12..20].copy_from_slice(&(HEADER_SIZE as u64).to_le_bytes());
        out[20..24].copy_from_slice(&(metadata.len() as u32).to_le_bytes());
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

    fn load_norm(dtype_byte: u8, shape: &[u64], payload: &[u8]) -> crate::Result<Vec<f32>> {
        let file = write_tempfile(&build_single_tensor_apr(
            "model.norm.weight",
            dtype_byte,
            shape,
            payload,
        ));
        let apr = MappedAprModel::from_path(file.path()).expect("load apr");
        super::apr_load_f32_tensor(
            &apr,
            apr.data(),
            apr.data_offset() as usize,
            &["model.norm.weight"],
        )
    }

    /// The eight values below are exactly representable in bf16 (their low 16
    /// mantissa bits are zero), so "decoded correctly" is an EQUALITY, not a
    /// tolerance — and the wrong-width read cannot coincidentally match.
    const BF16_VALUES: [f32; 8] = [7.0625, -0.0625, 1.5, 17.25, -3.0, 0.03125, 128.0, -0.75];

    fn bf16_payload() -> Vec<u8> {
        BF16_VALUES
            .iter()
            .flat_map(|v| half::bf16::from_f32(*v).to_le_bytes())
            .collect()
    }

    #[test]
    fn bf16_norm_decodes_to_its_own_values_not_a_reinterpretation() {
        let payload = bf16_payload();
        assert_eq!(payload.len(), 16, "8 bf16 values are 16 bytes");

        let got = load_norm(DTYPE_BF16, &[8], &payload).expect("BF16 norm must load");

        // Before the fix this returned 4 values (16 bytes read as F32), the
        // first of which was 7.132_92 — a plausible-looking RMSNorm weight.
        assert_eq!(
            got,
            BF16_VALUES.to_vec(),
            "a BF16 tensor must decode as BF16; got {got:?}"
        );
    }

    #[test]
    fn f16_norm_still_decodes_as_f16() {
        // GH-180 regression guard: F16 was the ONE dtype the old code checked.
        let values = [1.5f32, -0.25, 8.0, 0.125];
        let payload: Vec<u8> = values
            .iter()
            .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
            .collect();

        let got = load_norm(DTYPE_F16, &[4], &payload).expect("F16 norm must load");
        assert_eq!(got, values.to_vec());
    }

    #[test]
    fn f32_norm_still_decodes_as_f32() {
        let values = [1.5f32, -0.25, 8.0, 0.125];
        let payload: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let got = load_norm(DTYPE_F32, &[4], &payload).expect("F32 norm must load");
        assert_eq!(got, values.to_vec());
    }

    #[test]
    fn unknown_dtype_is_refused_not_reinterpreted_as_f32() {
        // POKA-YOKE: the old `_ =>` arm turned every unhandled dtype into a
        // plausible f32 vector. A quantized 1-D tensor must now be an error —
        // the #2407 rule: a path that cannot work fails instead of emitting
        // numbers.
        let payload = vec![0x42u8; 144]; // one Q4_K super-block
        let err = load_norm(DTYPE_Q4_K, &[256], &payload)
            .expect_err("a Q4_K norm must not decode as dense float");
        let msg = err.to_string();
        assert!(
            msg.contains("Q4_K") && msg.contains("not a dense float"),
            "error must name the offending dtype, got: {msg}"
        );
    }

    #[test]
    fn byte_count_must_agree_with_shape_and_dtype_width() {
        // POKA-YOKE: this is the mismatch #2443 rode in on — 16 bytes of BF16
        // consumed as 4 F32 while the index said 8 elements. Even a future
        // dtype arm wired to the wrong width cannot decode silently now.
        let payload = bf16_payload(); // 16 bytes = 8 BF16, but only 4 F32
        let err = load_norm(DTYPE_F32, &[8], &payload)
            .expect_err("8 F32 elements need 32 bytes, not 16");
        let msg = err.to_string();
        assert!(
            msg.contains("16 bytes") && msg.contains("needs 32"),
            "error must state both byte counts, got: {msg}"
        );
    }

    #[test]
    fn absent_tensor_is_none_but_undecodable_tensor_is_err() {
        // An optional tensor that is MISSING is benign (`Ok(None)`). One that
        // is PRESENT and undecodable used to be indistinguishable from missing,
        // so a dropped bias was as invisible as a mis-decoded one.
        let payload = vec![0x42u8; 144];
        let file = write_tempfile(&build_single_tensor_apr(
            "model.layers.0.self_attn.q_proj.bias",
            DTYPE_Q4_K,
            &[256],
            &payload,
        ));
        let apr = MappedAprModel::from_path(file.path()).expect("load apr");

        let absent = super::apr_try_load_dense_float(
            &apr,
            apr.data(),
            apr.data_offset() as usize,
            &["model.layers.0.mlp.up_proj.bias"],
        )
        .expect("a missing optional tensor is not an error");
        assert!(absent.is_none(), "missing tensor must be None");

        let present_but_bad = super::apr_try_load_dense_float(
            &apr,
            apr.data(),
            apr.data_offset() as usize,
            &["model.layers.0.self_attn.q_proj.bias"],
        );
        assert!(
            present_but_bad.is_err(),
            "a present-but-undecodable bias must be an error, not silently dropped"
        );
    }

    #[test]
    fn a_norm_whose_width_is_not_hidden_dim_is_refused() {
        // POKA-YOKE: `rms_norm` infers hidden_dim from weight.len(), so a
        // half-width norm computes happily forever. The loader is the last
        // place that still knows the true hidden_dim.
        super::apr_check_norm_width("model.norm.weight", &[1.0; 896], 896)
            .expect("a full-width norm loads");

        let err = super::apr_check_norm_width("model.norm.weight", &[1.0; 448], 896)
            .expect_err("a half-width norm must be refused");
        let msg = err.to_string();
        assert!(
            msg.contains("448") && msg.contains("896"),
            "error must state both widths, got: {msg}"
        );
    }
}

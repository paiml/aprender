/// GH-278: Transpose a row-major f32 matrix from [rows x cols] to [cols x rows].
///
/// PMAT-285: Delegates to `contract_gate::transpose_f32` (single source of truth).
fn transpose_f32_matrix(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    crate::contract_gate::transpose_f32(data, rows, cols)
}

/// PMAT-895 (OBLIG-GGUF-LOAD-NANINF): scan the f16 scale field(s) of a quantized
/// tensor's raw bytes for a non-finite value (NaN/Inf). Returns the first offending
/// scale value if any, else `None`.
///
/// The block / super-block scale layouts mirror the dequant code
/// (`quantize/dequant.rs`, `quantize/dequant_q4k.rs`): each block leads with (or, for
/// Q6_K/Q2_K/Q3_K, ends with) one or two f16 scales. We read ONLY those f16 fields —
/// O(num_blocks), not O(num_elements). A non-finite `d`/`dmin` poisons every element
/// of its block at dequant, so checking the scales is both cheap and sufficient. F32
/// tensors (embeddings/norms) are stored separately and are not handled here.
fn quant_scale_first_nonfinite(data: &[u8], qtype: u32) -> Option<f32> {
    use crate::gguf::types::{
        GGUF_TYPE_Q2_K, GGUF_TYPE_Q3_K, GGUF_TYPE_Q4_0, GGUF_TYPE_Q4_1, GGUF_TYPE_Q4_K,
        GGUF_TYPE_Q5_0, GGUF_TYPE_Q5_1, GGUF_TYPE_Q5_K, GGUF_TYPE_Q6_K, GGUF_TYPE_Q8_0,
    };
    use crate::quantize::read_f16;

    if data.is_empty() {
        return None;
    }

    // (block_bytes, &[scale_offsets_within_block]) for each supported quant type.
    // Each offset marks a 2-byte little-endian f16 scale field.
    let (block_bytes, scale_offsets): (usize, &[usize]) = match qtype {
        // 32-element blocks (one or two leading f16 scales).
        t if t == GGUF_TYPE_Q4_0 => (18, &[0]),
        t if t == GGUF_TYPE_Q8_0 => (34, &[0]),
        t if t == GGUF_TYPE_Q4_1 => (20, &[0, 2]),
        t if t == GGUF_TYPE_Q5_0 => (22, &[0]),
        t if t == GGUF_TYPE_Q5_1 => (24, &[0, 2]),
        // 256-element K-quant super-blocks.
        t if t == GGUF_TYPE_Q4_K => (144, &[0, 2]),    // d, dmin
        t if t == GGUF_TYPE_Q5_K => (176, &[0, 2]),    // d, dmin
        t if t == GGUF_TYPE_Q6_K => (210, &[208]),     // d (trailing)
        t if t == GGUF_TYPE_Q2_K => (84, &[80, 82]),   // d, dmin (trailing)
        t if t == GGUF_TYPE_Q3_K => (110, &[108]),     // d_all (trailing)
        // F16/F32/BF16/unknown: not a scaled-block layout — skip here.
        _ => return None,
    };

    if block_bytes == 0 || !data.len().is_multiple_of(block_bytes) {
        // Malformed block size for this qtype: leave to the existing shape gates.
        return None;
    }

    for block in data.chunks_exact(block_bytes) {
        for &off in scale_offsets {
            let scale = read_f16(&block[off..off + 2]);
            if !scale.is_finite() {
                return Some(scale);
            }
        }
    }
    None
}

/// Dequantize token embedding from APR format to f32 based on dtype.
///
/// Refs realizar#85: Added BF16/F16 support for aprender's GH-205/GH-353 passthrough.
/// Refs realizar#86: Added all GGML quant types (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q5_K, Q6_K).
fn dequantize_embedding(
    embed_data: &[u8],
    dtype: &str,
    num_elements: usize,
) -> Result<Vec<f32>> {
    match dtype {
        "F32" | "f32" => Ok(embed_data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()),
        "BF16" | "bf16" => Ok(crate::inference::simd_bf16_to_f32(embed_data)),
        "F16" | "f16" => Ok(crate::apr::dequant::dequantize_f16(
            embed_data,
            num_elements,
        )),
        // GGML quant types (from GGUF-sourced APR files)
        "Q4_0" => crate::quantize::dequantize_q4_0(embed_data),
        "Q4_1" => crate::quantize::dequantize_q4_1(embed_data),
        "Q5_0" => crate::quantize::dequantize_q5_0(embed_data),
        "Q5_1" => crate::quantize::dequantize_q5_1(embed_data),
        "Q8_0" => crate::quantize::dequantize_q8_0(embed_data),
        "Q2_K" => crate::quantize::dequantize_q2_k(embed_data),
        "Q4_K" => crate::quantize::dequantize_q4_k(embed_data),
        "Q5_K" => crate::quantize::dequantize_q5_k(embed_data),
        "Q6_K" => crate::quantize::dequantize_q6_k(embed_data),
        // APR native quant types
        "q8" => Ok(crate::apr::dequant::dequantize_apr_q8(
            embed_data,
            num_elements,
        )),
        "q4" => Ok(crate::apr::dequant::dequantize_apr_q4(
            embed_data,
            num_elements,
        )),
        other => Err(RealizarError::FormatError {
            reason: format!("APR: unsupported embedding dtype: {other}"),
        }),
    }
}

impl OwnedQuantizedModel {
    /// Create owned model from memory-mapped GGUF file
    ///
    /// # Errors
    ///
    /// Returns error if model loading fails
    pub fn from_mapped(mapped: &crate::gguf::MappedGGUFModel) -> Result<Self> {
        let data = mapped.data();
        let transformer = QuantizedGGUFTransformer::from_gguf(&mapped.model, data)?;

        // Get config for dimension calculations
        let config = &transformer.config;
        let hidden_dim = config.hidden_dim;
        let vocab_size = config.vocab_size;

        // GH-279: Contract gate — validate architecture and dimensions before proceeding
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

        // Convert layers to owned (passing config for dimensions)
        // GH-278: Conv1D weight transpose is NOT needed for GGUF files.
        // Both llama.cpp (convert_hf_to_gguf.py) and aprender (transpose_weights: true)
        // already transpose Conv1D [in,out] -> Linear [out,in] during GGUF export.
        // Transposing again here would double-transpose F32 tensors.
        // The APR loading path (from_apr) still handles transpose for native APR formats.
        let layers: Vec<OwnedQuantizedLayer> = transformer
            .layers
            .iter()
            .map(|l| OwnedQuantizedLayer::from_borrowed(l, data, config))
            .collect();

        let model = Self {
            config: transformer.config.clone(),
            token_embedding: transformer.token_embedding,
            position_embedding: transformer.position_embedding,
            layers,
            encoder_layers: vec![],
            encoder_output_norm_weight: None,
            encoder_output_norm_bias: None,
            output_norm_weight: transformer.output_norm_weight,
            output_norm_bias: transformer.output_norm_bias,
            // LM head: [hidden_dim] -> [vocab_size]
            lm_head_weight: OwnedQuantizedTensor::from_ref_with_dims(
                &transformer.lm_head_weight,
                data,
                hidden_dim,
                vocab_size,
            ),
            lm_head_bias: transformer.lm_head_bias,
            #[cfg(feature = "cuda")]
            cuda_executor: None,
            #[cfg(feature = "cuda")]
            cuda_kernel_count: std::sync::atomic::AtomicU64::new(0),
            #[cfg(feature = "cuda")]
            cached_weight_names: std::sync::Mutex::new(std::collections::HashSet::new()),
        };
        // PMAT-750: fail closed on a truncated/corrupt model (a quantized weight
        // declares real dims but has no data because the file was incomplete) instead
        // of silently running inference on a dead weight and emitting garbage.
        model.validate_quantized_tensors()?;
        Ok(model)
    }

    /// PMAT-750: reject a truncated/corrupt model at load. `from_ref_with_dims`
    /// substitutes an empty data buffer when a tensor's bytes run past the file, so a
    /// truncated GGUF would otherwise load and produce garbage at inference (apr qa's
    /// density gate catches it, but `apr run` does not run those gates). This fails the
    /// load with a clear error naming the first truncated tensor — the fail-closed
    /// guarantee from the Pillar-4 beat (PMAT-744) extended to the load path.
    ///
    /// PMAT-895 (OBLIG-GGUF-LOAD-NANINF): also reject a model whose quantized weights
    /// dequantize to NaN/Inf. A super-block whose f16 scale `d`/`dmin` is f16 +Inf
    /// (`0x7C00`) or NaN (`0x7E00`) makes EVERY element of that block non-finite at
    /// dequant; inference then emits garbage. llama.cpp / Ollama load such a model
    /// (their `check_tensors` defaults to false), so apr failing closed here is a
    /// genuine Pillar-4 BEAT. The same NaN/Inf guarantee already exists on the
    /// SafeTensors path (F-DATA-QUALITY-002, `safetensors/validation.rs`); this wires
    /// it into the quantized load path. We scan only the f16 scale field(s) per block
    /// (O(num_blocks), not O(num_elements)) — the scales are what corrupt the dequant.
    pub(crate) fn validate_quantized_tensors(&self) -> Result<()> {
        fn check(t: &OwnedQuantizedTensor, name: &str) -> Result<()> {
            if t.is_truncated() {
                return Err(crate::error::RealizarError::InvalidShape {
                    reason: format!(
                        "truncated/corrupt model: tensor '{name}' declares {}x{} but has no data (file is incomplete)",
                        t.out_dim, t.in_dim
                    ),
                });
            }
            // PMAT-895: fail closed on a non-finite quant scale (NaN/Inf dequant).
            if let Some(bad) = quant_scale_first_nonfinite(&t.data, t.qtype) {
                return Err(crate::error::RealizarError::InvalidShape {
                    reason: format!(
                        "OBLIG-GGUF-LOAD-NANINF: tensor '{name}' (qtype {}) has a non-finite \
                         f16 quant scale (value {bad}); it dequantizes to NaN/Inf and produces \
                         garbage at inference — apr fails closed at load (F-DATA-QUALITY-002)",
                        t.qtype
                    ),
                });
            }
            Ok(())
        }
        fn check_layer(layer: &OwnedQuantizedLayer, prefix: &str) -> Result<()> {
            match &layer.qkv_weight {
                OwnedQKVWeights::Fused(t) => check(t, &format!("{prefix}.qkv"))?,
                OwnedQKVWeights::Separate { q, k, v } => {
                    check(q, &format!("{prefix}.q"))?;
                    check(k, &format!("{prefix}.k"))?;
                    check(v, &format!("{prefix}.v"))?;
                },
            }
            check(&layer.attn_output_weight, &format!("{prefix}.attn_output"))?;
            check(&layer.ffn_up_weight, &format!("{prefix}.ffn_up"))?;
            check(&layer.ffn_down_weight, &format!("{prefix}.ffn_down"))?;
            if let Some(g) = &layer.ffn_gate_weight {
                check(g, &format!("{prefix}.ffn_gate"))?;
            }
            Ok(())
        }
        for (i, layer) in self.layers.iter().enumerate() {
            check_layer(layer, &format!("layer.{i}"))?;
        }
        for (i, layer) in self.encoder_layers.iter().enumerate() {
            check_layer(layer, &format!("encoder_layer.{i}"))?;
        }
        check(&self.lm_head_weight, "lm_head")?;
        Ok(())
    }

    /// Create a model for testing purposes
    ///
    /// This constructor handles the internal CUDA fields automatically,
    /// allowing external tests to construct models without accessing pub(crate) fields.
    ///
    /// # Arguments
    /// * `config` - Model configuration
    /// * `token_embedding` - Token embedding weights
    /// * `layers` - Quantized transformer layers
    /// * `output_norm_weight` - Output normalization weight
    /// * `output_norm_bias` - Optional output normalization bias
    /// * `lm_head_weight` - Language model head weight
    /// * `lm_head_bias` - Optional language model head bias
    #[must_use]
    pub fn new_for_test(
        config: GGUFConfig,
        token_embedding: Vec<f32>,
        layers: Vec<OwnedQuantizedLayer>,
        output_norm_weight: Vec<f32>,
        output_norm_bias: Option<Vec<f32>>,
        lm_head_weight: OwnedQuantizedTensor,
        lm_head_bias: Option<Vec<f32>>,
    ) -> Self {
        Self {
            config,
            token_embedding,
            position_embedding: None,
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
        }
    }
}

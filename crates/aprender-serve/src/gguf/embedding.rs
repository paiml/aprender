/// GH-278: Transpose a row-major f32 matrix from [rows x cols] to [cols x rows].
///
/// PMAT-285: Delegates to `contract_gate::transpose_f32` (single source of truth).
fn transpose_f32_matrix(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    crate::contract_gate::transpose_f32(data, rows, cols)
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

        Ok(Self {
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
        })
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

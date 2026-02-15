
/// Convert f16 to f32
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = (bits >> 10) & 0x1F;
    let mantissa = (bits & 0x3FF) as u32;

    if exp == 0 {
        if mantissa == 0 {
            f32::from_bits(sign)
        } else {
            // Denormal
            let mut m = mantissa;
            let mut e = 0i32;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            let new_exp = ((127 - 15 + 1 + e) as u32) << 23;
            let new_mantissa = (m & 0x3FF) << 13;
            f32::from_bits(sign | new_exp | new_mantissa)
        }
    } else if exp == 31 {
        // Inf or NaN
        f32::from_bits(sign | 0x7F800000 | (mantissa << 13))
    } else {
        // exp is 1-30, bias conversion: f16 bias=15, f32 bias=127
        let new_exp = (exp as u32 + 127 - 15) << 23;
        let new_mantissa = mantissa << 13;
        f32::from_bits(sign | new_exp | new_mantissa)
    }
}

/// Quantize to int8 (symmetric quantization)
fn quantize_int8(data: &[f32]) -> Vec<f32> {
    if data.is_empty() {
        return vec![];
    }

    // Find scale factor (max absolute value)
    let max_abs = data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);

    if max_abs == 0.0 {
        return vec![0.0; data.len()];
    }

    let scale = max_abs / 127.0;

    // Quantize and dequantize
    data.iter()
        .map(|&v| {
            let quantized = (v / scale).round().clamp(-127.0, 127.0) as i8;
            f32::from(quantized) * scale
        })
        .collect()
}

/// Quantize to int4 (symmetric quantization)
fn quantize_int4(data: &[f32]) -> Vec<f32> {
    if data.is_empty() {
        return vec![];
    }

    // Find scale factor
    let max_abs = data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);

    if max_abs == 0.0 {
        return vec![0.0; data.len()];
    }

    let scale = max_abs / 7.0; // 4-bit signed range: -8 to 7

    // Quantize and dequantize
    data.iter()
        .map(|&v| {
            let quantized = (v / scale).round().clamp(-8.0, 7.0) as i8;
            f32::from(quantized) * scale
        })
        .collect()
}

// NOTE: quantize_q4_k moved to trueno-quant crate (Toyota Way consolidation)

// NOTE: quantize_q6_k moved to trueno-quant crate (Toyota Way consolidation)

// NOTE: quantize_q6_k_matrix moved to trueno-quant crate (Toyota Way consolidation)

// NOTE: quantize_q5_k moved to trueno-quant crate (Toyota Way consolidation)

// NOTE: quantize_q5_k_matrix moved to trueno-quant crate (Toyota Way consolidation)

// NOTE: quantize_q4_k_matrix moved to trueno-quant crate (Toyota Way consolidation)

// Transpose Q4K data for matmul kernel compatibility (LAYOUT-002)
// GGUF stores weight matrices in column-major order (GGML convention).
// NOTE: transpose_q4k_for_matmul moved to trueno-quant crate (Toyota Way consolidation)
// NOTE: transpose_q5k_for_matmul moved to trueno-quant crate (Toyota Way consolidation)
// NOTE: transpose_q6k_for_matmul moved to trueno-quant crate (Toyota Way consolidation)
// NOTE: dequantize_q6_k_to_f32 moved to trueno-quant crate (Toyota Way consolidation)
// NOTE: dequantize_q5_k_to_f32 moved to trueno-quant crate (Toyota Way consolidation)

/// Check if a tensor name represents a 2D weight that needs transposition
///
/// Note: Scaffolding for PMAT-103 layout conversion optimization.
#[allow(dead_code)]
fn needs_transpose(name: &str, shape: &[usize]) -> bool {
    // Only transpose 2D weight tensors
    if shape.len() != 2 {
        return false;
    }

    // Transpose these weight tensors for matmul compatibility
    let weight_patterns = [
        "attn_output.weight",
        "attn_k.weight",
        "attn_q.weight",
        "attn_v.weight",
        "ffn_gate.weight",
        "ffn_up.weight",
        "ffn_down.weight",
        "output.weight",
        "lm_head.weight",
        "q_proj.weight",
        "k_proj.weight",
        "v_proj.weight",
        "o_proj.weight",
        "gate_proj.weight",
        "up_proj.weight",
        "down_proj.weight",
    ];

    weight_patterns.iter().any(|pattern| name.contains(pattern))
}

/// GH-237: Should this tensor skip quantization?
///
/// Returns true for tensors where quantization causes quality loss:
/// - Embeddings lose lookup precision (GH-231/232)
/// - Biases are too small and precision-sensitive
/// - LayerNorm/RMSNorm weights are critical for numerical stability
/// - lm_head has same small-value distribution as embeddings (GH-234)
/// - Small tensors (<1024 elements) don't benefit from quantization
///
/// Used by both the convert path (`add_tensor_with_quantization`) and the
/// import path (`add_f32_tensor_to_writer` in write.rs).
pub(super) fn should_skip_quantization(name: &str, element_count: usize) -> bool {
    let is_embedding = name.contains("embed_tokens")
        || name.contains("token_embd")
        || name.contains("wte")
        || name.contains("wpe")
        || name.contains("word_embeddings")
        || name.contains("position_embedding");

    let is_lm_head = name.contains("lm_head") || name == "output.weight";

    is_embedding
        || is_lm_head
        || name.contains("bias")
        || name.contains("layernorm")
        || name.contains("layer_norm")
        || name.contains("norm.weight")
        || element_count < 1024
}

/// GH-237: Write a tensor to the APR writer with correct dtype dispatch.
///
/// Applies skip logic for sensitive tensors and dispatches to the correct
/// packing method (`add_q8_tensor`/`add_q4_tensor`/`add_f16_tensor`)
/// instead of always writing F32.
fn add_tensor_with_quantization(
    writer: &mut AprV2Writer,
    name: &str,
    shape: &[usize],
    data: &[f32],
    quantize: Option<QuantizationType>,
) {
    let should_skip = should_skip_quantization(name, data.len());

    match quantize {
        Some(QuantizationType::Fp16) => {
            writer.add_f16_tensor(name, shape.to_vec(), data);
        }
        Some(QuantizationType::Int8) if !should_skip => {
            writer.add_q8_tensor(name, shape.to_vec(), data);
        }
        Some(QuantizationType::Int4) if !should_skip => {
            writer.add_q4_tensor(name, shape.to_vec(), data);
        }
        Some(QuantizationType::Q4K) if !should_skip => {
            let q4k_bytes = quantize_q4_k(data);
            writer.add_q4k_raw_tensor(name, shape.to_vec(), q4k_bytes);
        }
        _ => {
            writer.add_f32_tensor(name, shape.to_vec(), data);
        }
    }
}

/// Save model tensors with optional compression
///
/// Note: For .apr output, use save_model_tensors_with_config() instead to embed metadata.
fn save_model_tensors(
    tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    output: &Path,
    compression: Option<Compression>,
    quantize: Option<QuantizationType>,
) -> Result<()> {
    // GH-165 FIX: If output is .apr, use APR format with embedded config
    let extension = output.extension().and_then(|e| e.to_str()).unwrap_or("");
    if extension == "apr" {
        return save_model_tensors_with_config(tensors, output, compression, quantize);
    }

    // For non-APR output (e.g., .safetensors), use plain SafeTensors
    save_safetensors(output, tensors).map_err(|e| AprenderError::FormatError {
        message: format!("Failed to save converted model: {e}"),
    })
}

/// Save model tensors to APR format with embedded config metadata (GH-165 fix)
///
/// Infers model configuration from tensor shapes and embeds it in APR metadata.
/// This ensures AprTransformer can load with correct dimensions.
/// If config cannot be inferred (generic tensors), saves with minimal metadata.
fn save_model_tensors_with_config(
    tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    output: &Path,
    _compression: Option<Compression>,
    quantize: Option<QuantizationType>,
) -> Result<()> {
    // Try to infer model configuration from tensor shapes
    let config = infer_model_config_from_tensors(tensors);

    // Build AprV2Metadata with inferred config (or defaults)
    let mut metadata = AprV2Metadata::new("unknown");
    metadata.original_format = Some("safetensors".to_string());

    if let Some(cfg) = config {
        metadata.model_type = "qwen2".to_string(); // Detected transformer model
        metadata.hidden_size = cfg.hidden_size;
        metadata.num_layers = cfg.num_layers;
        metadata.vocab_size = cfg.vocab_size;
        metadata.num_heads = cfg.num_heads;
        metadata.num_kv_heads = cfg.num_kv_heads;
        metadata.intermediate_size = cfg.intermediate_size;
    }

    // GH-237: Create writer and add tensors with correct dtype dispatch
    let mut writer = AprV2Writer::new(metadata);
    for (name, (data, shape)) in tensors {
        add_tensor_with_quantization(&mut writer, name, shape, data, quantize);
    }

    // Write to file
    let apr_bytes = writer.write().map_err(|e| AprenderError::FormatError {
        message: format!("Failed to write APR format: {e}"),
    })?;

    fs::write(output, apr_bytes).map_err(|e| AprenderError::FormatError {
        message: format!("Failed to write output file: {e}"),
    })
}

// GH-237: save_model_tensors_with_gguf_config removed — superseded by
// save_model_tensors_with_gguf_config_and_tokenizer which extends it with
// tokenizer embedding for standalone APR inference (PMAT-113).

/// Save model tensors to APR format with GGUF config AND tokenizer (PMAT-113 fix)
///
/// This extends `save_model_tensors_with_gguf_config` to also embed the tokenizer
/// vocabulary for standalone APR inference without sibling tokenizer.json files.
fn save_model_tensors_with_gguf_config_and_tokenizer(
    tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    output: &Path,
    _compression: Option<Compression>,
    gguf_config: &GgufModelConfig,
    tokenizer: Option<&GgufTokenizer>,
    quantize: Option<QuantizationType>,
) -> Result<()> {
    // Build AprV2Metadata with GGUF config (not inferred from tensor shapes)
    let mut metadata = AprV2Metadata::new(gguf_config.architecture.as_deref().unwrap_or("qwen2"));
    metadata.original_format = Some("gguf".to_string());
    metadata.model_type = gguf_config
        .architecture
        .clone()
        .unwrap_or_else(|| "qwen2".to_string());
    // PMAT-113 FIX: Set architecture for chat template detection
    metadata.architecture = gguf_config.architecture.clone();

    // Copy all GGUF config fields to APR metadata
    metadata.hidden_size = gguf_config.hidden_size;
    metadata.num_layers = gguf_config.num_layers;
    metadata.num_heads = gguf_config.num_heads;
    metadata.num_kv_heads = gguf_config.num_kv_heads;
    metadata.vocab_size = gguf_config.vocab_size;
    metadata.intermediate_size = gguf_config.intermediate_size;
    metadata.max_position_embeddings = gguf_config.max_position_embeddings;

    // F-REGR-231 FIX: These fields are CRITICAL for correct inference
    metadata.rope_theta = gguf_config.rope_theta;
    metadata.rope_type = gguf_config.rope_type;
    metadata.rms_norm_eps = gguf_config.rms_norm_eps;

    // PMAT-113 FIX: Embed tokenizer vocabulary for standalone APR inference
    if let Some(tok) = tokenizer {
        if !tok.vocabulary.is_empty() {
            eprintln!(
                "[PMAT-113] Embedding {} vocabulary tokens into APR metadata",
                tok.vocabulary.len()
            );
            // Store vocabulary as JSON array in custom metadata
            let vocab_array: Vec<serde_json::Value> = tok
                .vocabulary
                .iter()
                .map(|s| serde_json::Value::String(s.clone()))
                .collect();
            metadata.custom.insert(
                "tokenizer.vocabulary".to_string(),
                serde_json::Value::Array(vocab_array),
            );
            metadata.custom.insert(
                "tokenizer.vocab_size".to_string(),
                serde_json::Value::Number(serde_json::Number::from(tok.vocabulary.len())),
            );
        }
        if let Some(ref model_type) = tok.model_type {
            metadata.custom.insert(
                "tokenizer.model".to_string(),
                serde_json::Value::String(model_type.clone()),
            );
        }
        if let Some(bos_id) = tok.bos_token_id {
            metadata.custom.insert(
                "tokenizer.bos_token_id".to_string(),
                serde_json::Value::Number(serde_json::Number::from(bos_id)),
            );
        }
        if let Some(eos_id) = tok.eos_token_id {
            metadata.custom.insert(
                "tokenizer.eos_token_id".to_string(),
                serde_json::Value::Number(serde_json::Number::from(eos_id)),
            );
        }
        // PMAT-171: Embed BPE merge rules for standalone APR encoding
        if !tok.merges.is_empty() {
            eprintln!(
                "[PMAT-171] Embedding {} BPE merge rules into APR metadata",
                tok.merges.len()
            );
            let merges_array: Vec<serde_json::Value> = tok
                .merges
                .iter()
                .map(|s| serde_json::Value::String(s.clone()))
                .collect();
            metadata.custom.insert(
                "tokenizer.merges".to_string(),
                serde_json::Value::Array(merges_array),
            );
        }
    }

    // GH-237: Create writer and add tensors with correct dtype dispatch
    let mut writer = AprV2Writer::new(metadata);
    for (name, (data, shape)) in tensors {
        add_tensor_with_quantization(&mut writer, name, shape, data, quantize);
    }

    // Write to file
    let apr_bytes = writer.write().map_err(|e| AprenderError::FormatError {
        message: format!("Failed to write APR format: {e}"),
    })?;

    fs::write(output, apr_bytes).map_err(|e| AprenderError::FormatError {
        message: format!("Failed to write output file: {e}"),
    })
}

/// Inferred model configuration from tensor shapes for Q4K quantization.
///
/// Used by `save_model_tensors_q4k` to populate APR metadata fields.
struct InferredQ4kConfig {
    hidden_size: Option<usize>,
    num_layers: Option<usize>,
    num_kv_heads: Option<usize>,
    vocab_size: Option<usize>,
    intermediate_size: Option<usize>,
    num_heads: Option<usize>,
}

/// Infer model configuration from tensor names and shapes.
///
/// Scans the tensor map for well-known naming patterns (norm weights, embeddings,
/// layer indices, projection matrices, gate projections) and extracts architecture
/// dimensions. Assumes `head_dim=64` for head-count inference.
fn infer_q4k_config(tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>) -> InferredQ4kConfig {
    let mut cfg = InferredQ4kConfig {
        hidden_size: None,
        num_layers: None,
        num_kv_heads: None,
        vocab_size: None,
        intermediate_size: None,
        num_heads: None,
    };

    for (name, (_, shape)) in tensors {
        infer_q4k_single_tensor(&mut cfg, name, shape);
    }

    cfg
}

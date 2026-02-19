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

/// Symmetric quantize-then-dequantize: maps floats to integer levels and back.
///
/// `max_level` is the positive clamp bound (e.g. 127 for int8, 7 for int4).
/// `min_level` is the negative clamp bound (e.g. -127 for int8, -8 for int4).
fn symmetric_quantize_dequantize(data: &[f32], max_level: f32, min_level: f32) -> Vec<f32> {
    if data.is_empty() {
        return vec![];
    }

    let max_abs = data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    if max_abs == 0.0 {
        return vec![0.0; data.len()];
    }

    let scale = max_abs / max_level;
    data.iter()
        .map(|&v| {
            let quantized = (v / scale).round().clamp(min_level, max_level) as i8;
            f32::from(quantized) * scale
        })
        .collect()
}

/// Quantize to int8 (symmetric quantization)
fn quantize_int8(data: &[f32]) -> Vec<f32> {
    symmetric_quantize_dequantize(data, 127.0, -127.0)
}

/// Quantize to int4 (symmetric quantization)
fn quantize_int4(data: &[f32]) -> Vec<f32> {
    symmetric_quantize_dequantize(data, 7.0, -8.0) // 4-bit signed range: -8 to 7
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

    // PMAT-274 FIX: Apply quantization for SafeTensors output too
    if let Some(quant) = quantize {
        return save_safetensors_quantized(tensors, output, quant);
    }

    // For non-APR output without quantization, use plain SafeTensors (F32)
    save_safetensors(output, tensors).map_err(|e| AprenderError::FormatError {
        message: format!("Failed to save converted model: {e}"),
    })
}

/// PMAT-274: Save SafeTensors with quantized dtype.
/// Converts F32 tensors to the appropriate reduced-precision format.
/// Respects `should_skip_quantization` for sensitive tensors (embeddings, biases, norms).
fn save_safetensors_quantized(
    tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    output: &Path,
    quant: QuantizationType,
) -> Result<()> {
    let mut metadata = SafeTensorsMetadata::new();
    let mut raw_data = Vec::new();
    let mut current_offset = 0;

    for (name, (data, shape)) in tensors {
        let (dtype_str, tensor_bytes) = if should_skip_quantization(name, data.len()) {
            // Sensitive tensors stay F32
            let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
            ("F32", bytes)
        } else {
            quantize_for_safetensors(data, quant)
        };

        let start_offset = current_offset;
        let end_offset = current_offset + tensor_bytes.len();

        metadata.insert(
            name.clone(),
            TensorMetadata {
                dtype: dtype_str.to_string(),
                shape: shape.clone(),
                data_offsets: [start_offset, end_offset],
            },
        );

        raw_data.extend_from_slice(&tensor_bytes);
        current_offset = end_offset;
    }

    let metadata_json =
        serde_json::to_string(&metadata).map_err(|e| AprenderError::FormatError {
            message: format!("JSON serialization failed: {e}"),
        })?;

    let header_bytes = metadata_json.as_bytes();
    let header_len = header_bytes.len() as u64;

    let mut file_data = Vec::new();
    file_data.extend_from_slice(&header_len.to_le_bytes());
    file_data.extend_from_slice(header_bytes);
    file_data.extend_from_slice(&raw_data);

    fs::write(output, file_data).map_err(|e| AprenderError::FormatError {
        message: format!("Failed to write output file: {e}"),
    })
}

/// Convert F32 tensor data to quantized bytes for SafeTensors output.
/// Returns (dtype_string, raw_bytes).
fn quantize_for_safetensors(data: &[f32], quant: QuantizationType) -> (&'static str, Vec<u8>) {
    match quant {
        QuantizationType::Fp16 => ("F16", f32_slice_to_f16_le_bytes(data)),
        QuantizationType::Int8 | QuantizationType::Q4K => {
            ("I8", symmetric_quantize_i8(data, 127.0))
        }
        QuantizationType::Int4 => ("U8", quantize_int4_packed(data)),
    }
}

/// Symmetric quantization to i8: scale = max(|data|) / max_val.
fn symmetric_quantize_i8(data: &[f32], max_val: f32) -> Vec<u8> {
    let max_abs = data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let scale = if max_abs > 0.0 { max_val / max_abs } else { 1.0 };
    data.iter()
        .map(|&v| (v * scale).round().clamp(-128.0, 127.0) as i8 as u8)
        .collect()
}

/// Int4 quantization: pack 2 nibbles per byte.
fn quantize_int4_packed(data: &[f32]) -> Vec<u8> {
    let max_abs = data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let scale = if max_abs > 0.0 { 7.0 / max_abs } else { 1.0 };
    let quantized: Vec<u8> = data
        .iter()
        .map(|&v| ((v * scale).round().clamp(-8.0, 7.0) as i8 + 8) as u8)
        .collect();
    quantized
        .chunks(2)
        .map(|chunk| {
            let low = chunk[0] & 0x0F;
            let high = chunk.get(1).map_or(0, |v| v & 0x0F);
            low | (high << 4)
        })
        .collect()
}

/// Convert a single F32 value to F16 (IEEE 754 half-precision).
fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exponent = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x007F_FFFF;

    let f16 = if exponent == 0xFF {
        sign | 0x7C00 | if mantissa != 0 { 0x0200 } else { 0 }
    } else if exponent > 142 {
        sign | 0x7C00
    } else if exponent < 113 {
        sign
    } else {
        sign | (((exponent - 112) as u32) << 10) | ((mantissa >> 13) & 0x3FF)
    };
    f16 as u16
}

/// Convert F32 slice to IEEE 754 half-precision (F16) little-endian bytes.
fn f32_slice_to_f16_le_bytes(data: &[f32]) -> Vec<u8> {
    data.iter()
        .flat_map(|&v| f32_to_f16_bits(v).to_le_bytes())
        .collect()
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


include!("mod_part_03_include_01.rs");

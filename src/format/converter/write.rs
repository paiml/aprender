//! APR Write Functions
//! PMAT-197: Extracted from mod.rs

use crate::error::{AprenderError, Result};
use crate::format::converter_types::{ImportOptions, QuantizationType};
use crate::format::gguf::{
    dequantize_q4_0, dequantize_q4_1, dequantize_q5_0, dequantize_q8_0, GgufModelConfig,
    GgufRawTensor, GgufTokenizer,
};
use crate::serialization::safetensors::UserMetadata;

// Import quantization and transpose functions from parent module
// LAYOUT-002: transpose functions convert GGUF column-major to APR row-major
// NOTE: Local implementations until trueno-quant crate resolves cyclic dependency
use super::{
    quantize_q4_k, quantize_q4_k_matrix, quantize_q6_k_matrix, transpose_q4k_for_matmul,
    transpose_q6k_for_matmul,
};
use crate::format::v2::{AprV2Metadata, AprV2Writer, TensorDType};
use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::Path;

// ============================================================================
// LAYOUT-002 Helper Functions
// ============================================================================

/// Transpose an F32 matrix from column-major to row-major layout.
///
/// BUG-CONV-001 FIX: Legacy quant formats (Q4_0/Q4_1/Q5_0/Q8_0) dequantize to F32 in
/// column-major order. Before re-quantizing to row-major K-quants, we need to transpose.
///
/// # Arguments
/// * `data` - F32 values in column-major order (GGML: cols contiguous)
/// * `shape` - Original GGUF shape [ne0=cols, ne1=rows]
///
/// # Returns
/// (transposed F32 data in row-major order, new shape [rows, cols])
fn transpose_f32_colmajor_to_rowmajor(data: &[f32], shape: &[usize]) -> (Vec<f32>, Vec<usize>) {
    if shape.len() != 2 {
        // Only transpose 2D tensors
        return (data.to_vec(), shape.to_vec());
    }

    let cols = shape[0]; // GGML ne0 = cols (contiguous)
    let rows = shape[1]; // GGML ne1 = rows

    // Transpose: column-major [cols, rows] -> row-major [rows, cols]
    let mut transposed = vec![0.0f32; data.len()];
    for r in 0..rows {
        for c in 0..cols {
            // Column-major source: data[c * rows + r]
            // Row-major dest: transposed[r * cols + c]
            transposed[r * cols + c] = data[c * rows + r];
        }
    }

    (transposed, vec![rows, cols])
}

/// Transpose F32 bytes from column-major to row-major layout.
///
/// BUG-CONV-001 FIX: Raw F32 bytes in GGUF are column-major. Need to transpose for APR.
///
/// # Arguments
/// * `data` - F32 bytes in column-major order (little-endian f32s)
/// * `shape` - Original GGUF shape [ne0=cols, ne1=rows]
///
/// # Returns
/// (transposed F32 bytes in row-major order, new shape [rows, cols])
fn transpose_f32_bytes_colmajor_to_rowmajor(data: &[u8], shape: &[usize]) -> (Vec<u8>, Vec<usize>) {
    if shape.len() != 2 {
        // Only transpose 2D tensors
        return (data.to_vec(), shape.to_vec());
    }

    let cols = shape[0]; // GGML ne0 = cols (contiguous)
    let rows = shape[1]; // GGML ne1 = rows

    // Interpret bytes as f32 (little-endian)
    let f32_data: Vec<f32> = data
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    // Transpose: column-major [cols, rows] -> row-major [rows, cols]
    let mut transposed = vec![0.0f32; f32_data.len()];
    for r in 0..rows {
        for c in 0..cols {
            // Column-major source: data[c * rows + r]
            // Row-major dest: transposed[r * cols + c]
            transposed[r * cols + c] = f32_data[c * rows + r];
        }
    }

    // Convert back to bytes
    let transposed_bytes: Vec<u8> = transposed.iter().flat_map(|f| f.to_le_bytes()).collect();

    (transposed_bytes, vec![rows, cols])
}

/// Transpose F16 bytes from column-major to row-major layout.
///
/// BUG-CONV-001 FIX: Raw F16 bytes in GGUF are column-major. Need to transpose for APR.
/// Note: This preserves F16 format, unlike the F32 variant.
///
/// # Arguments
/// * `data` - F16 bytes in column-major order (little-endian f16s)
/// * `shape` - Original GGUF shape [ne0=cols, ne1=rows]
///
/// # Returns
/// (transposed F16 bytes in row-major order, new shape [rows, cols])
fn transpose_f16_bytes_colmajor_to_rowmajor(data: &[u8], shape: &[usize]) -> (Vec<u8>, Vec<usize>) {
    if shape.len() != 2 {
        // Only transpose 2D tensors
        return (data.to_vec(), shape.to_vec());
    }

    let cols = shape[0]; // GGML ne0 = cols (contiguous)
    let rows = shape[1]; // GGML ne1 = rows

    // F16 is 2 bytes per element
    let mut transposed = vec![0u8; data.len()];

    for r in 0..rows {
        for c in 0..cols {
            // Column-major source index: c * rows + r
            // Row-major dest index: r * cols + c
            let src_idx = (c * rows + r) * 2;
            let dst_idx = (r * cols + c) * 2;

            if src_idx + 1 < data.len() && dst_idx + 1 < transposed.len() {
                transposed[dst_idx] = data[src_idx];
                transposed[dst_idx + 1] = data[src_idx + 1];
            }
        }
    }

    (transposed, vec![rows, cols])
}

// ============================================================================
// High-level API
// ============================================================================

/// Write tensors to native APR format
///
/// PMAT-223: `user_metadata` preserves arbitrary user metadata from SafeTensors `__metadata__`
/// section through the conversion pipeline. Stored under `"source_metadata"` in APR custom field.
pub(crate) fn write_apr_file(
    tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    output: &Path,
    options: &ImportOptions,
    tokenizer: Option<&GgufTokenizer>,
    model_config: Option<&GgufModelConfig>,
    user_metadata: &UserMetadata,
) -> Result<()> {
    // PMAT-100: Handle tied embeddings (common in Qwen, LLaMA, etc.)
    // Many models share embed_tokens.weight with lm_head.weight to reduce parameters.
    // HuggingFace SafeTensors omits lm_head.weight when tied, but realizar's
    // AprTransformer::from_apr_bytes expects lm_head.weight to exist.
    // Solution: If lm_head.weight is missing but embed_tokens exists, copy it.
    // Do this FIRST so param_count and tensor_shapes include it.
    let tensors_with_lm_head: BTreeMap<String, (Vec<f32>, Vec<usize>)> = {
        let mut result = tensors.clone();
        let has_lm_head = tensors.keys().any(|k| k == "lm_head.weight");
        if !has_lm_head {
            // Try to find embed_tokens.weight (may have different prefixes)
            let embed_key = tensors
                .keys()
                .find(|k| k.contains("embed_tokens.weight") || *k == "token_embd.weight")
                .cloned();
            if let Some(embed_name) = embed_key {
                if let Some((embed_data, embed_shape)) = tensors.get(&embed_name) {
                    // For tied embeddings, lm_head shares weight with embed_tokens
                    // embed_tokens: [vocab_size, hidden_dim]
                    // lm_head: [vocab_size, hidden_dim] (same shape for realizar)
                    result.insert(
                        "lm_head.weight".to_string(),
                        (embed_data.clone(), embed_shape.clone()),
                    );
                }
            }
        }
        result
    };

    // Calculate total parameter count (includes lm_head if added)
    let param_count: u64 = tensors_with_lm_head
        .values()
        .map(|(data, _)| data.len() as u64)
        .sum();

    // ROSETTA-003: Track tied embeddings for round-trip export fidelity.
    // If we synthesized lm_head from embed_tokens, flag it so export paths
    // can remove the duplicate and restore the original tied structure.
    let has_tied_embeddings = !tensors.keys().any(|k| k == "lm_head.weight")
        && tensors_with_lm_head.contains_key("lm_head.weight");

    // Build tensor_shapes map for metadata (used by `apr tensors` command)
    // ROSETTA-003: Store all tensors individually (no QKV fusion)
    let tensor_shapes: serde_json::Map<String, serde_json::Value> = tensors_with_lm_head
        .iter()
        .map(|(name, (_, shape))| {
            let shape_array: Vec<serde_json::Value> = shape
                .iter()
                .map(|&dim| serde_json::Value::Number(serde_json::Number::from(dim as u64)))
                .collect();
            (name.clone(), serde_json::Value::Array(shape_array))
        })
        .collect();

    // Create metadata with architecture info and tensor shapes
    let mut custom = std::collections::HashMap::new();
    custom.insert(
        "tensor_shapes".to_string(),
        serde_json::Value::Object(tensor_shapes),
    );

    // PMAT-223: Preserve user metadata from SafeTensors __metadata__ section
    if !user_metadata.is_empty() {
        let meta_obj: serde_json::Map<String, serde_json::Value> = user_metadata
            .iter()
            .map(|(k, v)| (k.clone(), serde_json::Value::String(v.clone())))
            .collect();
        custom.insert(
            "source_metadata".to_string(),
            serde_json::Value::Object(meta_obj),
        );
    }

    // ROSETTA-003: Flag tied embeddings for round-trip export fidelity
    if has_tied_embeddings {
        custom.insert("tied_embeddings".to_string(), serde_json::Value::Bool(true));
    }

    // Add tokenizer data if available (CRITICAL for GGUF import)
    if let Some(tok) = tokenizer {
        if !tok.vocabulary.is_empty() {
            // Store vocabulary as JSON array
            let vocab_array: Vec<serde_json::Value> = tok
                .vocabulary
                .iter()
                .map(|s| serde_json::Value::String(s.clone()))
                .collect();
            custom.insert(
                "tokenizer.vocabulary".to_string(),
                serde_json::Value::Array(vocab_array),
            );
            custom.insert(
                "tokenizer.vocab_size".to_string(),
                serde_json::Value::Number(serde_json::Number::from(tok.vocabulary.len())),
            );
        }
        if let Some(ref model_type) = tok.model_type {
            custom.insert(
                "tokenizer.model_type".to_string(),
                serde_json::Value::String(model_type.clone()),
            );
        }
        if let Some(bos) = tok.bos_token_id {
            custom.insert(
                "tokenizer.bos_token_id".to_string(),
                serde_json::Value::Number(serde_json::Number::from(bos)),
            );
        }
        if let Some(eos) = tok.eos_token_id {
            custom.insert(
                "tokenizer.eos_token_id".to_string(),
                serde_json::Value::Number(serde_json::Number::from(eos)),
            );
        }
        if let Some(ref arch) = tok.architecture {
            custom.insert(
                "tokenizer.architecture".to_string(),
                serde_json::Value::String(arch.clone()),
            );
        }
        if let Some(ref name) = tok.model_name {
            custom.insert(
                "tokenizer.model_name".to_string(),
                serde_json::Value::String(name.clone()),
            );
        }
        // PMAT-221 FIX: Embed BPE merge rules for SafeTensors path
        // This was missing, causing SafeTensors→APR to produce garbage output
        // because the tokenizer couldn't properly encode input text without merges
        if !tok.merges.is_empty() {
            eprintln!(
                "[PMAT-221] Embedding {} BPE merge rules into APR metadata (SafeTensors path)",
                tok.merges.len()
            );
            let merges_array: Vec<serde_json::Value> = tok
                .merges
                .iter()
                .map(|s| serde_json::Value::String(s.clone()))
                .collect();
            custom.insert(
                "tokenizer.merges".to_string(),
                serde_json::Value::Array(merges_array),
            );
        }
    }

    // Extract transformer config from model_config (CRITICAL for inference)
    let (
        architecture,
        hidden_size,
        num_layers,
        num_heads,
        num_kv_heads,
        vocab_size,
        intermediate_size,
        max_position_embeddings,
        rope_theta,
        rope_type,
        rms_norm_eps,
    ) = if let Some(cfg) = model_config {
        (
            cfg.architecture.clone(),
            cfg.hidden_size,
            cfg.num_layers,
            cfg.num_heads,
            cfg.num_kv_heads,
            cfg.vocab_size,
            cfg.intermediate_size,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            cfg.rope_type,
            cfg.rms_norm_eps,
        )
    } else {
        (
            None, None, None, None, None, None, None, None, None, None, None,
        )
    };

    let metadata = AprV2Metadata {
        model_type: format!("{:?}", options.architecture),
        name: Some(
            output
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("model")
                .to_string(),
        ),
        param_count,
        custom,
        // Transformer config (CRITICAL for realizar inference)
        architecture,
        hidden_size,
        num_layers,
        num_heads,
        num_kv_heads,
        vocab_size,
        intermediate_size,
        max_position_embeddings,
        rope_theta,
        rope_type,
        rms_norm_eps,
        ..Default::default()
    };

    // Create APR writer
    let mut writer = AprV2Writer::new(metadata);

    // ROSETTA-003: Write all tensors individually (no QKV fusion).
    // Q, K, V are stored as separate tensors. Fusion happens at runtime in realizar.
    for (name, (data, shape)) in &tensors_with_lm_head {
        // Determine if tensor should skip quantization
        // - Biases are too small and precision-sensitive
        // - LayerNorm/RMSNorm weights are critical for numerical stability
        // - Small tensors (<1024 elements) don't benefit from quantization
        let should_skip_quant = name.contains("bias")
            || name.contains("layernorm")
            || name.contains("layer_norm")
            || name.contains("norm.weight")
            || data.len() < 1024;

        match options.quantize {
            Some(QuantizationType::Fp16) => {
                writer.add_f16_tensor(name, shape.clone(), data);
            }
            Some(QuantizationType::Int8) if !should_skip_quant => {
                writer.add_q8_tensor(name, shape.clone(), data);
            }
            Some(QuantizationType::Int4) if !should_skip_quant => {
                writer.add_q4_tensor(name, shape.clone(), data);
            }
            Some(QuantizationType::Q4K) if !should_skip_quant => {
                let q4k_bytes = quantize_q4_k(data);
                writer.add_q4k_raw_tensor(name, shape.clone(), q4k_bytes);
            }
            _ => {
                writer.add_f32_tensor(name, shape.clone(), data);
            }
        }
    }

    // Write to file
    let bytes = writer.write().map_err(|e| AprenderError::FormatError {
        message: format!("Failed to serialize APR format: {e}"),
    })?;

    let mut file = fs::File::create(output).map_err(|e| AprenderError::FormatError {
        message: format!("Failed to create output file: {e}"),
    })?;

    file.write_all(&bytes)
        .map_err(|e| AprenderError::FormatError {
            message: format!("Failed to write APR file: {e}"),
        })?;

    Ok(())
}

/// Write APR file from raw quantized GGUF tensors (preserves Q4_K/Q6_K exactly)
///
/// PMAT-103: This function preserves the original GGUF quantization format,
/// ensuring format parity with Ollama/llama.cpp. No dequantization occurs.
pub(crate) fn write_apr_file_raw(
    tensors: &BTreeMap<String, GgufRawTensor>,
    output: &Path,
    _options: &ImportOptions,
    tokenizer: Option<&GgufTokenizer>,
    model_config: Option<&GgufModelConfig>,
) -> Result<()> {
    // Calculate total parameter count (approximate - based on shapes)
    let param_count: u64 = tensors
        .values()
        .map(|t| t.shape.iter().product::<usize>() as u64)
        .sum();

    // Build tensor_shapes map for metadata
    // LAYOUT-002 FIX: Compute shapes AFTER transpose for K-quant tensors
    // Q4K/Q5K/Q6K dtypes are transposed during import, so their shapes must reflect
    // the transposed dimensions [cols, rows] rather than original GGML [rows, cols].
    // Other dtypes use effective_shape (GGML reversed to standard [ne1, ne0]).
    let tensor_shapes: serde_json::Map<String, serde_json::Value> = tensors
        .iter()
        .map(|(name, tensor)| {
            // LAYOUT-002: Determine the actual output shape after processing
            let output_shape = match (tensor.dtype, tensor.shape.len()) {
                // Q4K, Q5K, Q6K are transposed: [rows, cols] → [cols, rows]
                (12..=14, 2) => {
                    vec![tensor.shape[1], tensor.shape[0]]
                }
                // Other 2D tensors use effective_shape (GGML reversed)
                (_, 2) => {
                    vec![tensor.shape[1], tensor.shape[0]]
                }
                // 1D tensors keep original shape
                _ => tensor.shape.clone(),
            };
            let shape_array: Vec<serde_json::Value> = output_shape
                .iter()
                .map(|&dim| serde_json::Value::Number(serde_json::Number::from(dim as u64)))
                .collect();
            (name.clone(), serde_json::Value::Array(shape_array))
        })
        .collect();

    // Create metadata
    let mut custom = std::collections::HashMap::new();
    custom.insert(
        "tensor_shapes".to_string(),
        serde_json::Value::Object(tensor_shapes),
    );

    // Add tokenizer data if available (PMAT-171: embed vocabulary for standalone APR files)
    if let Some(tok) = tokenizer {
        if !tok.vocabulary.is_empty() {
            let vocab_array: Vec<serde_json::Value> = tok
                .vocabulary
                .iter()
                .map(|s| serde_json::Value::String(s.clone()))
                .collect();
            custom.insert(
                "tokenizer.vocabulary".to_string(),
                serde_json::Value::Array(vocab_array),
            );
            custom.insert(
                "tokenizer.vocab_size".to_string(),
                serde_json::Value::Number(serde_json::Number::from(tok.vocabulary.len())),
            );
        }
        if let Some(model_type) = &tok.model_type {
            custom.insert(
                "tokenizer.model".to_string(),
                serde_json::Value::String(model_type.clone()),
            );
        }
        if let Some(bos) = tok.bos_token_id {
            custom.insert(
                "tokenizer.bos_token_id".to_string(),
                serde_json::Value::Number(serde_json::Number::from(bos)),
            );
        }
        if let Some(eos) = tok.eos_token_id {
            custom.insert(
                "tokenizer.eos_token_id".to_string(),
                serde_json::Value::Number(serde_json::Number::from(eos)),
            );
        }
        // GH-185 FIX: Embed BPE merge rules for standalone APR encoding
        // Without merges, the tokenizer cannot properly encode input text
        if !tok.merges.is_empty() {
            eprintln!(
                "[GH-185] Embedding {} BPE merge rules into APR metadata",
                tok.merges.len()
            );
            let merges_array: Vec<serde_json::Value> = tok
                .merges
                .iter()
                .map(|s| serde_json::Value::String(s.clone()))
                .collect();
            custom.insert(
                "tokenizer.merges".to_string(),
                serde_json::Value::Array(merges_array),
            );
        }
    }

    // Add model config if available
    if let Some(cfg) = model_config {
        if let Some(arch) = &cfg.architecture {
            custom.insert(
                "model.architecture".to_string(),
                serde_json::Value::String(arch.clone()),
            );
        }
        if let Some(hidden_size) = cfg.hidden_size {
            custom.insert(
                "model.hidden_size".to_string(),
                serde_json::Value::Number(serde_json::Number::from(hidden_size)),
            );
        }
        if let Some(num_layers) = cfg.num_layers {
            custom.insert(
                "model.num_layers".to_string(),
                serde_json::Value::Number(serde_json::Number::from(num_layers)),
            );
        }
        if let Some(num_heads) = cfg.num_heads {
            custom.insert(
                "model.num_heads".to_string(),
                serde_json::Value::Number(serde_json::Number::from(num_heads)),
            );
        }
        if let Some(num_kv_heads) = cfg.num_kv_heads {
            custom.insert(
                "model.num_kv_heads".to_string(),
                serde_json::Value::Number(serde_json::Number::from(num_kv_heads)),
            );
        }
        if let Some(vocab_size) = cfg.vocab_size {
            custom.insert(
                "model.vocab_size".to_string(),
                serde_json::Value::Number(serde_json::Number::from(vocab_size)),
            );
        }
        if let Some(intermediate_size) = cfg.intermediate_size {
            custom.insert(
                "model.intermediate_size".to_string(),
                serde_json::Value::Number(serde_json::Number::from(intermediate_size)),
            );
        }
        if let Some(max_pos) = cfg.max_position_embeddings {
            custom.insert(
                "model.max_position_embeddings".to_string(),
                serde_json::Value::Number(serde_json::Number::from(max_pos)),
            );
        }
        if let Some(rope_theta) = cfg.rope_theta {
            custom.insert(
                "model.rope_theta".to_string(),
                serde_json::Value::Number(
                    serde_json::Number::from_f64(f64::from(rope_theta))
                        .unwrap_or_else(|| serde_json::Number::from(10000u64)),
                ),
            );
        }
        if let Some(rms_eps) = cfg.rms_norm_eps {
            custom.insert(
                "model.rms_norm_eps".to_string(),
                serde_json::Value::Number(
                    serde_json::Number::from_f64(f64::from(rms_eps))
                        .unwrap_or_else(|| serde_json::Number::from(0u64)),
                ),
            );
        }
        // PMAT-114: Write rope_type for correct RoPE style
        if let Some(rope_type) = cfg.rope_type {
            custom.insert(
                "model.rope_type".to_string(),
                serde_json::Value::Number(serde_json::Number::from(rope_type)),
            );
        }
    }

    // Build metadata using correct AprV2Metadata structure
    let metadata = AprV2Metadata {
        model_type: model_config
            .and_then(|c| c.architecture.clone())
            .unwrap_or_else(|| "qwen2".to_string()),
        name: model_config.and_then(|c| c.architecture.clone()),
        description: Some("GGUF Q4_K model imported with native quantization".to_string()),
        author: None,
        license: None,
        version: Some("1.0.0".to_string()),
        source: None,
        original_format: Some("gguf".to_string()),
        created_at: None,
        total_size: 0, // Will be calculated from tensor data
        param_count,
        quantization: None, // Q4_K stored as raw dtype, not quantization metadata
        sharding: None,
        chat_template: None,
        chat_format: None,
        special_tokens: None,
        architecture: model_config.and_then(|c| c.architecture.clone()),
        hidden_size: model_config.and_then(|c| c.hidden_size),
        num_layers: model_config.and_then(|c| c.num_layers),
        num_heads: model_config.and_then(|c| c.num_heads),
        num_kv_heads: model_config.and_then(|c| c.num_kv_heads),
        vocab_size: model_config.and_then(|c| c.vocab_size),
        intermediate_size: model_config.and_then(|c| c.intermediate_size),
        max_position_embeddings: model_config.and_then(|c| c.max_position_embeddings),
        rope_theta: model_config.and_then(|c| c.rope_theta),
        rope_type: model_config.and_then(|c| c.rope_type),
        rms_norm_eps: model_config.and_then(|c| c.rms_norm_eps),
        custom,
    };

    // Create APR writer
    let mut writer = AprV2Writer::new(metadata);

    // Add all tensors with their native quantization format
    // PMAT-222 FIX: Reverse 2D tensor shapes from GGML [ne0, ne1] to standard [ne1, ne0]
    // GGML uses column-major convention: ne[0] is contiguous in memory.
    // This means GGML [ne0, ne1] in column-major = [ne1, ne0] in row-major.
    // The DATA layout is identical - only shape metadata needs swapping.
    // Realizaer's GGUF loader does `dims.reverse()` for the same reason.
    let mut dtype_counts: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
    for (name, tensor) in tensors {
        *dtype_counts.entry(tensor.dtype).or_insert(0) += 1;

        // Calculate element count for dequantization
        let num_elements: usize = tensor.shape.iter().product();

        // PMAT-222: Reverse GGML shape [ne0, ne1] → standard [ne1, ne0] for 2D tensors
        // The data bytes are NOT modified - GGML column-major = row-major with reversed dims
        // Note: This was used by error paths that now return errors (BUG-LAYOUT-003 fix)
        let _effective_shape = if tensor.shape.len() == 2 {
            vec![tensor.shape[1], tensor.shape[0]]
        } else {
            tensor.shape.clone()
        };

        // Process tensor based on dtype
        // Q5_0/Q8_0 need dequantization since realizar doesn't support them natively
        // BUG-CONV-001 FIX: All GGUF tensors are column-major, need transpose for row-major APR
        match tensor.dtype {
            0 => {
                // F32 - BUG-CONV-001 FIX: Transpose column-major to row-major
                let (transposed_data, transposed_shape) =
                    transpose_f32_bytes_colmajor_to_rowmajor(&tensor.data, &tensor.shape);
                writer.add_tensor(name, TensorDType::F32, transposed_shape, transposed_data);
            }
            1 => {
                // F16 - BUG-CONV-001 FIX: Transpose column-major to row-major
                let (transposed_data, transposed_shape) =
                    transpose_f16_bytes_colmajor_to_rowmajor(&tensor.data, &tensor.shape);
                writer.add_tensor(name, TensorDType::F16, transposed_shape, transposed_data);
            }
            12 => {
                // Q4_K - LAYOUT-002: Dequantize and re-quantize with row-padded layout
                // GGUF has column-major super-blocks, APR needs row-padded super-blocks
                // dequant→requant converts layout while preserving values
                let (row_major_q4k, transposed_shape) =
                    transpose_q4k_for_matmul(&tensor.data, &tensor.shape);
                writer.add_tensor(name, TensorDType::Q4K, transposed_shape, row_major_q4k);
            }
            13 => {
                // Q5_K - LAYOUT-002: Convert column-major to row-major
                // APR doesn't have native Q5K dtype, so convert to Q6K (closer bit-width)
                // BUG-CONV-001 FIX: F32 data from dequant is column-major, need to transpose
                use crate::format::gguf::dequantize_q5_k;
                match dequantize_q5_k(&tensor.data, 0, num_elements) {
                    Ok(f32_data) => {
                        // Transpose from column-major to row-major before requantizing
                        let (transposed_f32, transposed_shape) =
                            transpose_f32_colmajor_to_rowmajor(&f32_data, &tensor.shape);
                        let q6k_bytes = quantize_q6_k_matrix(&transposed_f32, &transposed_shape);
                        writer.add_tensor(name, TensorDType::Q6K, transposed_shape, q6k_bytes);
                    }
                    Err(e) => {
                        // BUG-LAYOUT-003 FIX: Fail instead of silently corrupting data
                        // Writing raw column-major quantized bytes as F32 violates LAYOUT-002
                        // and corrupts the tensor interpretation
                        return Err(AprenderError::FormatError {
                            message: format!(
                                "Failed to dequantize Q5_K tensor '{}': {}. \
                                 Cannot proceed - would violate LAYOUT-002 mandate.",
                                name, e
                            ),
                        });
                    }
                }
            }
            14 => {
                // Q6_K - LAYOUT-002: Dequantize and re-quantize with row-padded layout
                // Same as Q4_K - convert column-major super-blocks to row-padded
                let (row_major_q6k, transposed_shape) =
                    transpose_q6k_for_matmul(&tensor.data, &tensor.shape);
                writer.add_tensor(name, TensorDType::Q6K, transposed_shape, row_major_q6k);
            }
            2 => {
                // Q4_0 - dequantize then requantize to Q4_K (realizar needs K-quants)
                // BUG-CONV-001 FIX: F32 data from dequant is column-major, need to transpose
                match dequantize_q4_0(&tensor.data, 0, num_elements) {
                    Ok(f32_data) => {
                        // Transpose from column-major to row-major before requantizing
                        let (transposed_f32, transposed_shape) =
                            transpose_f32_colmajor_to_rowmajor(&f32_data, &tensor.shape);
                        let q4k_bytes = quantize_q4_k_matrix(&transposed_f32, &transposed_shape);
                        writer.add_tensor(name, TensorDType::Q4K, transposed_shape, q4k_bytes);
                    }
                    Err(e) => {
                        // BUG-LAYOUT-003 FIX: Fail instead of silently corrupting data
                        return Err(AprenderError::FormatError {
                            message: format!(
                                "Failed to dequantize Q4_0 tensor '{}': {}. \
                                 Cannot proceed - would violate LAYOUT-002 mandate.",
                                name, e
                            ),
                        });
                    }
                }
            }
            3 => {
                // Q4_1 - dequantize then requantize to Q4_K (realizar needs K-quants)
                // BUG-CONV-001 FIX: F32 data from dequant is column-major, need to transpose
                match dequantize_q4_1(&tensor.data, 0, num_elements) {
                    Ok(f32_data) => {
                        // Transpose from column-major to row-major before requantizing
                        let (transposed_f32, transposed_shape) =
                            transpose_f32_colmajor_to_rowmajor(&f32_data, &tensor.shape);
                        let q4k_bytes = quantize_q4_k_matrix(&transposed_f32, &transposed_shape);
                        writer.add_tensor(name, TensorDType::Q4K, transposed_shape, q4k_bytes);
                    }
                    Err(e) => {
                        // BUG-LAYOUT-003 FIX: Fail instead of silently corrupting data
                        return Err(AprenderError::FormatError {
                            message: format!(
                                "Failed to dequantize Q4_1 tensor '{}': {}. \
                                 Cannot proceed - would violate LAYOUT-002 mandate.",
                                name, e
                            ),
                        });
                    }
                }
            }
            6 => {
                // Q5_0 - dequantize then requantize to Q6_K (closer bit-width match)
                // BUG-CONV-001 FIX: F32 data from dequant is column-major, need to transpose
                match dequantize_q5_0(&tensor.data, 0, num_elements) {
                    Ok(f32_data) => {
                        // Transpose from column-major to row-major before requantizing
                        let (transposed_f32, transposed_shape) =
                            transpose_f32_colmajor_to_rowmajor(&f32_data, &tensor.shape);
                        let q6k_bytes = quantize_q6_k_matrix(&transposed_f32, &transposed_shape);
                        writer.add_tensor(name, TensorDType::Q6K, transposed_shape, q6k_bytes);
                    }
                    Err(e) => {
                        // BUG-LAYOUT-003 FIX: Fail instead of silently corrupting data
                        return Err(AprenderError::FormatError {
                            message: format!(
                                "Failed to dequantize Q5_0 tensor '{}': {}. \
                                 Cannot proceed - would violate LAYOUT-002 mandate.",
                                name, e
                            ),
                        });
                    }
                }
            }
            8 => {
                // Q8_0 - dequantize then requantize to Q6_K (closer bit-width match)
                // BUG-CONV-001 FIX: F32 data from dequant is column-major, need to transpose
                match dequantize_q8_0(&tensor.data, 0, num_elements) {
                    Ok(f32_data) => {
                        // Transpose from column-major to row-major before requantizing
                        let (transposed_f32, transposed_shape) =
                            transpose_f32_colmajor_to_rowmajor(&f32_data, &tensor.shape);
                        let q6k_bytes = quantize_q6_k_matrix(&transposed_f32, &transposed_shape);
                        writer.add_tensor(name, TensorDType::Q6K, transposed_shape, q6k_bytes);
                    }
                    Err(e) => {
                        // BUG-LAYOUT-003 FIX: Fail instead of silently corrupting data
                        return Err(AprenderError::FormatError {
                            message: format!(
                                "Failed to dequantize Q8_0 tensor '{}': {}. \
                                 Cannot proceed - would violate LAYOUT-002 mandate.",
                                name, e
                            ),
                        });
                    }
                }
            }
            7 | 9 => {
                // Q5_1/Q8_1 - BUG-LAYOUT-003 FIX: Fail instead of silently corrupting
                // Writing column-major quantized bytes as F32 violates LAYOUT-002
                return Err(AprenderError::FormatError {
                    message: format!(
                        "GGUF dtype {} (Q5_1/Q8_1) for tensor '{}' not yet supported. \
                         Cannot store raw bytes - would violate LAYOUT-002 mandate.",
                        tensor.dtype, name
                    ),
                });
            }
            _ => {
                // BUG-LAYOUT-003 FIX: Fail instead of silently corrupting
                // Unknown dtypes cannot be safely converted
                return Err(AprenderError::FormatError {
                    message: format!(
                        "Unsupported GGUF dtype {} for tensor '{}'. \
                         Cannot store raw bytes - would violate LAYOUT-002 mandate.",
                        tensor.dtype, name
                    ),
                });
            }
        }
    }

    // Write to file
    let bytes = writer.write().map_err(|e| AprenderError::FormatError {
        message: format!("Failed to serialize APR format: {e}"),
    })?;

    let mut file = fs::File::create(output).map_err(|e| AprenderError::FormatError {
        message: format!("Failed to create output file: {e}"),
    })?;

    file.write_all(&bytes)
        .map_err(|e| AprenderError::FormatError {
            message: format!("Failed to write APR file: {e}"),
        })?;

    Ok(())
}

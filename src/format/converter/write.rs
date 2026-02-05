//! APR Write Functions
//! PMAT-197: Extracted from mod.rs

use crate::error::{AprenderError, Result};
use crate::format::converter_types::{ImportOptions, QuantizationType};
use crate::format::gguf::{GgufModelConfig, GgufRawTensor, GgufTokenizer};
use crate::serialization::safetensors::UserMetadata;

// Import quantization functions from parent module
// GH-202 FIX: transpose functions removed - GGML data is already row-major
// NOTE: Local implementations until trueno-quant crate resolves cyclic dependency
use super::quantize_q4_k;
use crate::format::v2::{AprV2Metadata, AprV2Writer, TensorDType};
use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::Path;

// GH-202 FIX: Removed transpose_f32_colmajor_to_rowmajor,
// transpose_f32_bytes_colmajor_to_rowmajor, transpose_f16_bytes_colmajor_to_rowmajor.
//
// These functions were based on a WRONG assumption that GGML stores data in
// Fortran-style column-major order. In reality, GGML data[i0 + i1*ne0] is
// C row-major with shape [ne1, ne0]. Only shape reversal is needed, not
// data transposition. The wrong transpose corrupted non-square tensors,
// causing 58-90% diff in GH-202 conversion fidelity tests.

// ============================================================================
// High-level API
// ============================================================================

/// Write tensors to native APR format
///
/// PMAT-223: `user_metadata` preserves arbitrary user metadata from SafeTensors `__metadata__`
/// section through the conversion pipeline. Stored under `"source_metadata"` in APR custom field.
///
/// GH-205: `f16_raw_tensors` contains raw F16 bytes for passthrough. When a tensor appears
/// in both `tensors` (as F32) and `f16_raw_tensors` (raw bytes), the raw bytes are preferred
/// to avoid precision loss from F16→F32→F16 conversion.
pub(crate) fn write_apr_file(
    tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    f16_raw_tensors: &BTreeMap<String, (Vec<u8>, Vec<usize>)>,
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
    //
    // GH-205: F16 passthrough - if a tensor has raw F16 bytes, use those directly
    // to avoid precision loss from F16→F32→F16 conversion.
    let mut f16_passthrough_count = 0usize;
    for (name, (data, shape)) in &tensors_with_lm_head {
        // GH-205: Check if we have raw F16 bytes for this tensor
        if let Some((f16_bytes, f16_shape)) = f16_raw_tensors.get(name) {
            // Use raw F16 bytes directly (passthrough)
            writer.add_tensor(name, TensorDType::F16, f16_shape.clone(), f16_bytes.clone());
            f16_passthrough_count += 1;
            continue;
        }

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

    if f16_passthrough_count > 0 {
        eprintln!(
            "[GH-205] F16 passthrough: {} tensors written as raw F16 (no precision loss)",
            f16_passthrough_count
        );
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
    // GH-202: Handle tied embeddings (common in Qwen2.5, LLaMA, etc.)
    // Many models share embed_tokens.weight with lm_head.weight to reduce parameters.
    // GGUF omits output.weight when tied, but realizar's AprTransformer expects
    // lm_head.weight to exist. Clone the embedding tensor as lm_head.weight.
    // NOTE: Check for BOTH lm_head.weight AND output.weight - the APR loader
    // checks both names, so if output.weight exists we must NOT synthesize a
    // wrong lm_head.weight from token_embd.weight.
    let original_has_lm_head = tensors
        .keys()
        .any(|k| k == "lm_head.weight" || k == "output.weight");
    let tensors = {
        let mut result = tensors.clone();
        if !original_has_lm_head {
            let embed_key = result
                .keys()
                .find(|k| k.contains("embed_tokens.weight") || *k == "token_embd.weight")
                .cloned();
            if let Some(embed_name) = embed_key {
                if let Some(embed_tensor) = result.get(&embed_name).cloned() {
                    eprintln!(
                        "[GH-202] Synthesizing lm_head.weight from {} (tied embeddings)",
                        embed_name
                    );
                    result.insert("lm_head.weight".to_string(), embed_tensor);
                }
            }
        }
        result
    };

    // ROSETTA-003: Track tied embeddings for round-trip export fidelity
    let has_tied_embeddings = !original_has_lm_head && tensors.contains_key("lm_head.weight");

    // Calculate total parameter count (approximate - based on shapes)
    let param_count: u64 = tensors
        .values()
        .map(|t| t.shape.iter().product::<usize>() as u64)
        .sum();

    // Build tensor_shapes map for metadata
    // GH-202 FIX: All 2D tensors get shape reversal [ne0, ne1] → [ne1, ne0]
    // This is the standard convention where shape[0]=rows, shape[1]=cols.
    // 1D tensors keep original shape.
    let tensor_shapes: serde_json::Map<String, serde_json::Value> = tensors
        .iter()
        .map(|(name, tensor)| {
            let output_shape = if tensor.shape.len() == 2 {
                vec![tensor.shape[1], tensor.shape[0]]
            } else {
                tensor.shape.clone()
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

    // ROSETTA-003: Record tied embeddings for round-trip export fidelity
    if has_tied_embeddings {
        custom.insert("tied_embeddings".to_string(), serde_json::Value::Bool(true));
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

        // Calculate element count (kept for future use/debugging)
        let _num_elements: usize = tensor.shape.iter().product();

        // GH-202 FIX: Process tensor based on dtype
        // CRITICAL LAYOUT INSIGHT: GGML data layout data[i0 + i1*ne0] is IDENTICAL
        // to C row-major data[row*cols + col] when shape is reversed [ne1, ne0].
        // NO data transpose is needed - only shape reversal.
        //
        // For quantized formats (Q4K/Q6K), the super-block layout in GGML already
        // has ne0 elements contiguous per "row", which IS row-major block layout.
        // Raw bytes can be used directly with reversed shape.
        //
        // For legacy quant formats (Q4_0/Q4_1/Q5_0/Q8_0): FAIL with clear error.
        // Import = passthrough only. Use `apr convert` for format transformation.
        //
        // GH-208: Shape is ALREADY in APR format after enforce_import_contract().
        // Do NOT reverse again - that was causing double-reversal bug.
        let apr_shape = tensor.shape.clone();

        match tensor.dtype {
            0 => {
                // F32 - GH-202 FIX: Data is already row-major, just reverse shape
                writer.add_tensor(name, TensorDType::F32, apr_shape, tensor.data.clone());
            }
            1 => {
                // F16 - GH-202 FIX: Data is already row-major, just reverse shape
                writer.add_tensor(name, TensorDType::F16, apr_shape, tensor.data.clone());
            }
            12 => {
                // Q4_K - GH-202 FIX: Raw Q4K bytes are already row-major block layout
                // Each "row" (ne1 index) has ceil(ne0/256) contiguous super-blocks.
                // Just reverse the shape metadata.
                writer.add_tensor(name, TensorDType::Q4K, apr_shape, tensor.data.clone());
            }
            13 => {
                // Q5_K - NOT SUPPORTED for passthrough import
                return Err(AprenderError::FormatError {
                    message: format!(
                        "GGUF tensor '{}' uses Q5_K quantization which APR cannot represent exactly. \
                         Import requires exact format preservation. \
                         Use `apr convert --quantize q6_k` to convert to a supported format.",
                        name
                    ),
                });
            }
            14 => {
                // Q6_K - GH-202 FIX: Raw Q6K bytes are already row-major block layout
                // Same as Q4K - just reverse shape metadata.
                writer.add_tensor(name, TensorDType::Q6K, apr_shape, tensor.data.clone());
            }
            2 => {
                // Q4_0 - NOT SUPPORTED for passthrough import
                return Err(AprenderError::FormatError {
                    message: format!(
                        "GGUF tensor '{}' uses Q4_0 quantization which APR cannot represent exactly. \
                         Import requires exact format preservation. \
                         Use `apr convert --quantize q4_k` to convert to a supported format.",
                        name
                    ),
                });
            }
            3 => {
                // Q4_1 - NOT SUPPORTED for passthrough import
                return Err(AprenderError::FormatError {
                    message: format!(
                        "GGUF tensor '{}' uses Q4_1 quantization which APR cannot represent exactly. \
                         Import requires exact format preservation. \
                         Use `apr convert --quantize q4_k` to convert to a supported format.",
                        name
                    ),
                });
            }
            6 => {
                // Q5_0 - NOT SUPPORTED for passthrough import
                // Use `apr convert` to transform to a supported format
                return Err(AprenderError::FormatError {
                    message: format!(
                        "GGUF tensor '{}' uses Q5_0 quantization which APR cannot represent exactly. \
                         Import requires exact format preservation. \
                         Use `apr convert --quantize q6_k` to convert to a supported format.",
                        name
                    ),
                });
            }
            8 => {
                // Q8_0 - NOT SUPPORTED for passthrough import
                // Use `apr convert` to transform to a supported format
                return Err(AprenderError::FormatError {
                    message: format!(
                        "GGUF tensor '{}' uses Q8_0 quantization which APR cannot represent exactly. \
                         Import requires exact format preservation. \
                         Use `apr convert --quantize q6_k` to convert to a supported format.",
                        name
                    ),
                });
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

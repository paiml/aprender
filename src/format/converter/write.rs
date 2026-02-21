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

/// Resolve tied embeddings for F32 tensor maps (PMAT-100).
///
/// If lm_head.weight is missing but embed_tokens exists, synthesizes lm_head from embed_tokens.
/// Returns (possibly modified tensor map, whether tied embeddings were detected).
fn resolve_f32_tied_embeddings(
    tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
) -> (BTreeMap<String, (Vec<f32>, Vec<usize>)>, bool) {
    let mut result = tensors.clone();
    let has_lm_head = tensors.keys().any(|k| k == "lm_head.weight");
    if !has_lm_head {
        let embed_key = tensors
            .keys()
            .find(|k| {
                k.contains("embed_tokens.weight") || *k == "token_embd.weight" || *k == "wte.weight"
            })
            .cloned();
        if let Some(embed_name) = embed_key {
            if let Some((embed_data, embed_shape)) = tensors.get(&embed_name) {
                result.insert(
                    "lm_head.weight".to_string(),
                    (embed_data.clone(), embed_shape.clone()),
                );
            }
        }
    }
    let has_tied = !has_lm_head && result.contains_key("lm_head.weight");
    (result, has_tied)
}

/// Insert tokenizer metadata into APR custom fields for F32 tensor import path.
fn insert_f32_tokenizer_metadata(
    tok: &GgufTokenizer,
    custom: &mut std::collections::HashMap<String, serde_json::Value>,
) {
    if !tok.vocabulary.is_empty() {
        insert_string_array(custom, "tokenizer.vocabulary", &tok.vocabulary);
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
    insert_common_tokenizer_fields(tok, custom);
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
    if !tok.merges.is_empty() {
        eprintln!(
            "[PMAT-221] Embedding {} BPE merge rules into APR metadata (SafeTensors path)",
            tok.merges.len()
        );
        insert_string_array(custom, "tokenizer.merges", &tok.merges);
    }
}

/// Build the custom metadata map for the F32 tensor import path.
///
/// Combines tensor shapes, user metadata, tied-embedding flags, and tokenizer data
/// into the `custom` field used by `AprV2Metadata`.
fn build_f32_custom_metadata(
    tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    user_metadata: &UserMetadata,
    has_tied_embeddings: bool,
    tokenizer: Option<&GgufTokenizer>,
) -> std::collections::HashMap<String, serde_json::Value> {
    // Build tensor_shapes map for metadata (used by `apr tensors` command)
    // ROSETTA-003: Store all tensors individually (no QKV fusion)
    let tensor_shapes: serde_json::Map<String, serde_json::Value> = tensors
        .iter()
        .map(|(name, (_, shape))| {
            let shape_array: Vec<serde_json::Value> = shape
                .iter()
                .map(|&dim| serde_json::Value::Number(serde_json::Number::from(dim as u64)))
                .collect();
            (name.clone(), serde_json::Value::Array(shape_array))
        })
        .collect();

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
        insert_f32_tokenizer_metadata(tok, &mut custom);
    }

    custom
}

/// Add a single F32 tensor (or its F16 passthrough) to the APR writer.
///
/// GH-205: If the tensor exists in `f16_raw_tensors`, raw F16 bytes are used
/// directly to avoid precision loss. Returns `true` if F16 passthrough was used.
fn add_f32_tensor_to_writer(
    writer: &mut AprV2Writer,
    name: &str,
    data: &[f32],
    shape: &[usize],
    f16_raw_tensors: &BTreeMap<String, (Vec<u8>, Vec<usize>)>,
    quantize: Option<QuantizationType>,
) -> bool {
    // GH-205: Check if we have raw F16 bytes for this tensor
    if let Some((f16_bytes, f16_shape)) = f16_raw_tensors.get(name) {
        writer.add_tensor(name, TensorDType::F16, f16_shape.clone(), f16_bytes.clone());
        return true;
    }

    let should_skip_quant = super::should_skip_quantization(name, data.len());

    match quantize {
        Some(QuantizationType::Fp16) => {
            writer.add_f16_tensor(name, shape.to_vec(), data);
        }
        Some(QuantizationType::Int8) if !should_skip_quant => {
            writer.add_q8_tensor(name, shape.to_vec(), data);
        }
        Some(QuantizationType::Int4) if !should_skip_quant => {
            writer.add_q4_tensor(name, shape.to_vec(), data);
        }
        Some(QuantizationType::Q4K) if !should_skip_quant => {
            let q4k_bytes = quantize_q4_k(data);
            writer.add_q4k_raw_tensor(name, shape.to_vec(), q4k_bytes);
        }
        _ => {
            writer.add_f32_tensor(name, shape.to_vec(), data);
        }
    }

    false
}

/// Serialize an `AprV2Writer` and write the resulting bytes to a file.
fn flush_writer_to_file(mut writer: AprV2Writer, output: &Path) -> Result<()> {
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
    let (tensors_with_lm_head, has_tied_embeddings) = resolve_f32_tied_embeddings(tensors);

    let param_count: u64 = tensors_with_lm_head
        .values()
        .map(|(data, _)| data.len() as u64)
        .sum();

    let custom = build_f32_custom_metadata(
        &tensors_with_lm_head,
        user_metadata,
        has_tied_embeddings,
        tokenizer,
    );

    // Extract transformer config from model_config (CRITICAL for inference)
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
        ..Default::default()
    };

    // Create APR writer and add tensors
    // ROSETTA-003: Write all tensors individually (no QKV fusion).
    // Q, K, V are stored as separate tensors. Fusion happens at runtime in realizar.
    let mut writer = AprV2Writer::new(metadata);

    let mut f16_passthrough_count = 0usize;
    for (name, (data, shape)) in &tensors_with_lm_head {
        if add_f32_tensor_to_writer(
            &mut writer,
            name,
            data,
            shape,
            f16_raw_tensors,
            options.quantize,
        ) {
            f16_passthrough_count += 1;
        }
    }

    if f16_passthrough_count > 0 {
        eprintln!(
            "[GH-205] F16 passthrough: {} tensors written as raw F16 (no precision loss)",
            f16_passthrough_count
        );
    }

    flush_writer_to_file(writer, output)
}

/// Resolve tied embeddings by synthesizing lm_head.weight from embed_tokens.weight
/// when the model doesn't have an output/lm_head weight (GH-202).
///
/// Returns the (possibly modified) tensor map and whether tied embeddings were detected.
fn resolve_tied_embeddings(
    tensors: &BTreeMap<String, GgufRawTensor>,
) -> (BTreeMap<String, GgufRawTensor>, bool) {
    let original_has_lm_head = tensors
        .keys()
        .any(|k| k == "lm_head.weight" || k == "output.weight");
    let mut result = tensors.clone();
    if !original_has_lm_head {
        let embed_key = result
            .keys()
            .find(|k| {
                k.contains("embed_tokens.weight") || *k == "token_embd.weight" || *k == "wte.weight"
            })
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
    let has_tied = !original_has_lm_head && result.contains_key("lm_head.weight");
    (result, has_tied)
}

/// Insert common tokenizer fields shared by both import paths.
fn insert_common_tokenizer_fields(
    tok: &GgufTokenizer,
    custom: &mut std::collections::HashMap<String, serde_json::Value>,
) {
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
    // GH-277: Store pre-tokenizer type for GGUF export round-trip
    if let Some(ref pre) = tok.pre_type {
        custom.insert(
            "tokenizer.pre_type".to_string(),
            serde_json::Value::String(pre.clone()),
        );
    }
}

/// Store string array as JSON in custom metadata.
fn insert_string_array(
    custom: &mut std::collections::HashMap<String, serde_json::Value>,
    key: &str,
    values: &[String],
) {
    let arr: Vec<serde_json::Value> = values
        .iter()
        .map(|s| serde_json::Value::String(s.clone()))
        .collect();
    custom.insert(key.to_string(), serde_json::Value::Array(arr));
}

/// Insert tokenizer metadata into the custom metadata map (PMAT-171).
fn insert_tokenizer_metadata(
    tok: &GgufTokenizer,
    custom: &mut std::collections::HashMap<String, serde_json::Value>,
) {
    if !tok.vocabulary.is_empty() {
        insert_string_array(custom, "tokenizer.vocabulary", &tok.vocabulary);
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
    insert_common_tokenizer_fields(tok, custom);
    if !tok.merges.is_empty() {
        eprintln!(
            "[GH-185] Embedding {} BPE merge rules into APR metadata",
            tok.merges.len()
        );
        insert_string_array(custom, "tokenizer.merges", &tok.merges);
    }
    // GH-253: Store additional tokenizer metadata for GGUF export round-trip
    insert_gh253_tokenizer_fields(tok, custom);
}

/// GH-253: Store extended tokenizer metadata for GGUF round-trip.
fn insert_gh253_tokenizer_fields(
    tok: &GgufTokenizer,
    custom: &mut std::collections::HashMap<String, serde_json::Value>,
) {
    if !tok.token_type.is_empty() {
        let type_array: Vec<serde_json::Value> = tok
            .token_type
            .iter()
            .map(|&t| serde_json::Value::Number(serde_json::Number::from(t)))
            .collect();
        custom.insert(
            "tokenizer.token_type".to_string(),
            serde_json::Value::Array(type_array),
        );
    }
    if let Some(pad_id) = tok.padding_token_id {
        custom.insert(
            "tokenizer.padding_token_id".to_string(),
            serde_json::Value::Number(serde_json::Number::from(pad_id)),
        );
    }
    if let Some(add_bos) = tok.add_bos_token {
        custom.insert(
            "tokenizer.add_bos_token".to_string(),
            serde_json::Value::Bool(add_bos),
        );
    }
    if let Some(ref tmpl) = tok.chat_template {
        custom.insert(
            "tokenizer.chat_template".to_string(),
            serde_json::Value::String(tmpl.clone()),
        );
    }
}

include!("write_model_config.rs");
include!("write_tests_tied_embeddings.rs");

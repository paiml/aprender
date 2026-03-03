//! Shared model configuration resolution (GH-376, GH-377)
//!
//! CONTRACT: The `.apr` file is the single source of truth for model architecture.
//! Architecture fields (hidden_size, num_heads, num_layers, vocab_size, etc.)
//! were validated at import time by `tensor-layout-v1`. This module propagates
//! that contract to all training/eval pipelines.
//!
//! `TransformerConfig::tiny()` MUST NOT appear outside `#[cfg(test)]` code.

use crate::error::{CliError, Result};
use std::path::Path;

/// Extract TransformerConfig from an `.apr` file's metadata header.
///
/// Reads only the 64-byte header + metadata JSON section (~4 KB), not the full
/// model file. Returns None if the file isn't a valid APR v2 file or if
/// required architecture fields are missing.
pub(crate) fn read_apr_architecture(
    path: &Path,
) -> Option<entrenar::transformer::TransformerConfig> {
    use aprender::format::v2::{AprV2Header, AprV2Metadata, HEADER_SIZE_V2, MAGIC_V2};
    use std::io::{Read, Seek, SeekFrom};

    let mut file = std::fs::File::open(path).ok()?;
    let mut header_buf = [0u8; HEADER_SIZE_V2];
    file.read_exact(&mut header_buf).ok()?;
    if header_buf[..4] != MAGIC_V2 {
        return None;
    }

    let header = AprV2Header::from_bytes(&header_buf).ok()?;
    file.seek(SeekFrom::Start(header.metadata_offset)).ok()?;
    let mut meta_buf = vec![0u8; header.metadata_size as usize];
    file.read_exact(&mut meta_buf).ok()?;

    let metadata = AprV2Metadata::from_json(&meta_buf).ok()?;
    entrenar::transformer::TransformerConfig::from_apr_metadata(
        metadata.hidden_size,
        metadata.num_heads,
        metadata.num_kv_heads,
        metadata.intermediate_size,
        metadata.num_layers,
        metadata.vocab_size,
        metadata.max_position_embeddings,
        metadata.rms_norm_eps,
        metadata.rope_theta,
        metadata.architecture.as_deref(),
    )
}

/// Resolve TransformerConfig from .apr metadata, HF config.json, or --model-size fallback.
///
/// Precedence:
///   1. `.apr` file metadata (provable, validated at import)
///   2. HuggingFace `config.json` in model directory
///   3. `--model-size` string match (legacy fallback, no .apr file)
///   4. Error (refuse to silently degrade to tiny)
pub(crate) fn resolve_transformer_config(
    model_path: Option<&Path>,
    model_size: Option<&str>,
) -> Result<entrenar::transformer::TransformerConfig> {
    // Attempt 1: Read architecture from .apr file metadata
    if let Some(path) = model_path.filter(|p| p.is_file()) {
        if let Some(config) = read_apr_architecture(path) {
            return Ok(config);
        }
        eprintln!(
            "[GH-376] WARNING: could not read architecture from .apr metadata, \
             falling back to --model-size"
        );
    }

    // Attempt 2: Read architecture from HuggingFace config.json in model directory
    if let Some(path) = model_path.filter(|p| p.is_dir()) {
        if let Some(config) = read_hf_config_json(path) {
            return Ok(config);
        }
    }

    // Attempt 3: Legacy --model-size string matching
    resolve_transformer_config_by_size(model_size)
}

/// Read TransformerConfig from a HuggingFace `config.json` in a model directory.
///
/// Parses the standard HF model config format used by Qwen, LLaMA, Mistral, etc.
/// Returns None if config.json doesn't exist or required fields are missing.
fn read_hf_config_json(dir: &Path) -> Option<entrenar::transformer::TransformerConfig> {
    let config_path = dir.join("config.json");
    let data = std::fs::read_to_string(&config_path).ok()?;
    let json: serde_json::Value = serde_json::from_str(&data).ok()?;

    let hidden_size = json.get("hidden_size")?.as_u64()? as usize;
    let num_heads = json.get("num_attention_heads")?.as_u64()? as usize;
    let num_kv_heads = json
        .get("num_key_value_heads")
        .and_then(|v| v.as_u64())
        .map_or(num_heads, |v| v as usize);
    let intermediate_size = json.get("intermediate_size")?.as_u64()? as usize;
    let num_layers = json.get("num_hidden_layers")?.as_u64()? as usize;
    let vocab_size = json.get("vocab_size")?.as_u64()? as usize;
    let max_pos = json
        .get("max_position_embeddings")
        .and_then(|v| v.as_u64())
        .map_or(4096, |v| v as usize);
    let rms_norm_eps = json
        .get("rms_norm_eps")
        .and_then(|v| v.as_f64())
        .unwrap_or(1e-6) as f32;
    let rope_theta = json
        .get("rope_theta")
        .and_then(|v| v.as_f64())
        .unwrap_or(10000.0) as f32;
    let head_dim = json
        .get("head_dim")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize);
    let use_bias = json
        .get("attention_bias")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    Some(entrenar::transformer::TransformerConfig {
        hidden_size,
        num_attention_heads: num_heads,
        num_kv_heads,
        intermediate_size,
        num_hidden_layers: num_layers,
        vocab_size,
        max_position_embeddings: max_pos,
        rms_norm_eps,
        rope_theta,
        head_dim_override: head_dim,
        use_bias,
    })
}

/// Resolve TransformerConfig from `--model-size` string only.
///
/// Delegates to `TransformerConfig::from_size_str()` — the single canonical
/// mapping from size strings to configs. No duplicated match tables.
pub(crate) fn resolve_transformer_config_by_size(
    model_size: Option<&str>,
) -> Result<entrenar::transformer::TransformerConfig> {
    match model_size {
        Some(size) => entrenar::transformer::TransformerConfig::from_size_str(size)
            .map_err(CliError::ValidationFailed),
        None => Err(CliError::ValidationFailed(
            "No model path or --model-size provided. Cannot determine architecture.".to_string(),
        )),
    }
}

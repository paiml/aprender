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
    transformer_config_from_apr_metadata(
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
            "[GH-376] WARNING: could not read architecture metadata from '{}' (format: {}), \
             falling back to --model-size",
            path.display(),
            describe_format(path)
        );
    }

    // Attempt 2: Read architecture from HuggingFace config.json in model directory
    if let Some(path) = model_path.filter(|p| p.is_dir()) {
        if let Some(config) = read_hf_config_json(path) {
            return Ok(config);
        }
    }

    // Attempt 3: Legacy --model-size string matching.
    //
    // When a model path WAS given, the failure is "this file carries no
    // readable architecture", not "you gave me no model". The old wording —
    // "No model path or --model-size provided. Cannot determine architecture."
    // — asserted the opposite of what the user did, so the obvious next move
    // (re-pass the path) could not possibly help.
    if model_size.is_none() {
        if let Some(path) = model_path {
            return Err(CliError::ValidationFailed(format!(
                "Could not read the model architecture from '{}' (format: {}). \
                 Pass --model-size to state it explicitly \
                 (known sizes: 0.5B, 1.5B, 7B, 9B, 13B).",
                path.display(),
                describe_format(path)
            )));
        }
    }

    resolve_transformer_config_by_size(model_size)
}

/// Name a model file's format for an error message, from its extension.
fn describe_format(path: &Path) -> String {
    match path.extension().and_then(|e| e.to_str()) {
        Some(ext) if !ext.is_empty() => ext.to_ascii_lowercase(),
        _ if path.is_dir() => "directory".to_string(),
        _ => "unknown".to_string(),
    }
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
    let _head_dim = json
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
        use_bias,
        head_dim_override: None,
        architecture: entrenar::transformer::ModelArchitecture::Decoder,
        hf_architecture: None,
        hf_model_type: None,
        tie_word_embeddings: false,
    })
}

/// Resolve TransformerConfig from `--model-size` string only.
///
/// Local implementation of the size-string-to-config mapping. The upstream
/// `TransformerConfig::from_size_str()` exists in local entrenar source but
/// is not yet published in entrenar 0.7.5.
pub(crate) fn resolve_transformer_config_by_size(
    model_size: Option<&str>,
) -> Result<entrenar::transformer::TransformerConfig> {
    use entrenar::transformer::TransformerConfig;
    match model_size {
        Some(size) => match size {
            "0.5B" | "500M" | "qwen2-0.5b" => Ok(TransformerConfig::qwen2_0_5b()),
            "1.5B" | "qwen2-1.5b" | "qwen2.5-1.5b" => Ok(TransformerConfig::qwen2_1_5b()),
            "7B" | "llama2-7b" => Ok(TransformerConfig::llama2_7b()),
            "13B" | "llama2-13b" => Ok(TransformerConfig::llama2_13b()),
            "mistral-7b" => Ok(TransformerConfig::mistral_7b()),
            "9B" | "qwen3.5-9b" | "qwen3_5" | "qwen3.5" => Ok(TransformerConfig::qwen3_5_9b()),
            unknown => Err(CliError::ValidationFailed(format!(
                "Unknown model size '{unknown}'. Known sizes: 0.5B, 1.5B, 7B, 9B, 13B"
            ))),
        },
        None => Err(CliError::ValidationFailed(
            "No model path or --model-size provided. Cannot determine architecture.".to_string(),
        )),
    }
}

/// Construct TransformerConfig from APR v2 metadata fields.
///
/// Local stub for `TransformerConfig::from_apr_metadata()` which exists in
/// local entrenar source but is not yet published in entrenar 0.7.5.
///
/// Returns None if any required field (hidden_size, num_heads, num_layers,
/// vocab_size, intermediate_size) is missing.
fn transformer_config_from_apr_metadata(
    hidden_size: Option<usize>,
    num_heads: Option<usize>,
    num_kv_heads: Option<usize>,
    intermediate_size: Option<usize>,
    num_layers: Option<usize>,
    vocab_size: Option<usize>,
    max_position_embeddings: Option<usize>,
    rms_norm_eps: Option<f32>,
    rope_theta: Option<f32>,
    architecture: Option<&str>,
) -> Option<entrenar::transformer::TransformerConfig> {
    use entrenar::transformer::TransformerConfig;

    let hidden = hidden_size?;
    let vocab = vocab_size?;

    // If critical fields are missing, try to match a known architecture preset
    // by (architecture, hidden_size). This handles APR files created from GGUF
    // that didn't store num_heads/num_layers in metadata (pre-GH-376 imports).
    let (heads, layers, intermediate, kv_heads) =
        match (num_heads, num_layers, intermediate_size, num_kv_heads) {
            (Some(h), Some(l), Some(i), kv) => (h, l, i, kv),
            _ => {
                // Fall back to known presets by architecture + hidden_size
                let preset = match (architecture, hidden) {
                    (Some(a), 896) if a.starts_with("qwen2") => {
                        Some(TransformerConfig::qwen2_0_5b())
                    }
                    (Some(a), 1536) if a.starts_with("qwen2") => {
                        Some(TransformerConfig::qwen2_1_5b())
                    }
                    (Some(a), 3584) if a.starts_with("qwen2") => {
                        Some(TransformerConfig::qwen2_7b())
                    }
                    _ => None,
                };
                if let Some(p) = preset {
                    eprintln!(
                        "[GH-376] Metadata incomplete (num_heads/num_layers missing), \
                         using {arch} preset for hidden_size={hidden}",
                        arch = architecture.unwrap_or("unknown"),
                    );
                    (
                        p.num_attention_heads,
                        p.num_hidden_layers,
                        p.intermediate_size,
                        Some(p.num_kv_heads),
                    )
                } else {
                    return None;
                }
            }
        };

    // Determine use_bias from architecture family
    let use_bias = matches!(architecture, Some(a) if a.starts_with("qwen2"));

    Some(TransformerConfig {
        hidden_size: hidden,
        num_attention_heads: heads,
        num_kv_heads: kv_heads.unwrap_or(heads),
        intermediate_size: intermediate,
        num_hidden_layers: layers,
        vocab_size: vocab,
        max_position_embeddings: max_position_embeddings.unwrap_or(32768),
        rms_norm_eps: rms_norm_eps.unwrap_or(1e-6),
        rope_theta: rope_theta.unwrap_or(10000.0),
        use_bias,
        head_dim_override: None,
        architecture: entrenar::transformer::ModelArchitecture::Decoder,
        hf_architecture: None,
        hf_model_type: None,
        tie_word_embeddings: false,
    })
}

// ─── Architecture-resolution errors (dogfood 0.63.0, issue #2374 finding 14) ──
//
// `apr finetune <model>.safetensors --task classify` replied "No model path or
// --model-size provided. Cannot determine architecture." — asserting the exact
// opposite of what the user had just done. The path was the first positional
// argument and was never echoed.
#[cfg(test)]
mod arch_resolution_tests {
    use super::*;

    /// A real file on disk whose architecture cannot be read (empty, wrong format).
    fn unreadable_model(ext: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!("apr-2374-arch-{}.{ext}", std::process::id()));
        std::fs::write(&path, b"not a model").expect("scratch write should succeed");
        path
    }

    #[test]
    fn unreadable_model_error_does_not_claim_no_path_was_given() {
        let path = unreadable_model("safetensors");
        let err = resolve_transformer_config(Some(&path), None)
            .expect_err("an unreadable architecture must be an error");
        let msg = err.to_string();
        let _ = std::fs::remove_file(&path);
        assert!(
            !msg.contains("No model path"),
            "a path WAS provided; the error must not claim otherwise: {msg}"
        );
        assert!(
            msg.contains(&path.display().to_string()),
            "the error must echo the path the user passed: {msg}"
        );
        assert!(
            msg.contains("safetensors"),
            "the error must name the format that could not be read: {msg}"
        );
        assert!(
            msg.contains("--model-size"),
            "the error must name the actual workaround: {msg}"
        );
    }

    #[test]
    fn gguf_gets_the_same_actionable_message() {
        // Two of the three formats CLAUDE.md promises reach classify only via
        // --model-size; both must say so plainly.
        let path = unreadable_model("gguf");
        let err = resolve_transformer_config(Some(&path), None)
            .expect_err("an unreadable architecture must be an error");
        let msg = err.to_string();
        let _ = std::fs::remove_file(&path);
        assert!(
            msg.contains("gguf"),
            "the error must name the format: {msg}"
        );
        assert!(!msg.contains("No model path"), "a path WAS provided: {msg}");
    }

    #[test]
    fn no_path_and_no_size_still_says_no_path_was_given() {
        // Guards against over-correcting: when the user really did pass
        // nothing, the original wording is the accurate one.
        let err = resolve_transformer_config(None, None)
            .expect_err("no path and no size must be an error");
        assert!(
            err.to_string().contains("No model path"),
            "with genuinely no input the message must say so: {err}"
        );
    }

    #[test]
    fn explicit_model_size_still_wins_when_metadata_is_unreadable() {
        let path = unreadable_model("safetensors");
        let config = resolve_transformer_config(Some(&path), Some("0.5B"))
            .expect("--model-size must still rescue an unreadable file");
        let _ = std::fs::remove_file(&path);
        assert_eq!(
            config.hidden_size, 896,
            "0.5B is qwen2-0.5b: hidden size 896"
        );
    }
}

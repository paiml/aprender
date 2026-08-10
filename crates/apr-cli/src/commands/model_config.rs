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

/// Whether `path` starts with an APR magic (`APR\0` v2 or `APRN` v1).
///
/// #2417: the LoRA training pipeline is `InstructPipeline::from_apr` — APR
/// only. A `.safetensors` base used to travel the whole way to that call and
/// surface as `Failed to open APR file '<model>.safetensors': Invalid magic`,
/// naming a format the caller never mentioned. Callers use this to reject an
/// unsupported base up front, with an actionable message.
pub(crate) fn is_apr_file(path: &Path) -> bool {
    use aprender::format::v2::MAGIC_V2;
    use std::io::Read;

    const MAGIC_V1: [u8; 4] = *b"APRN";

    let Ok(mut file) = std::fs::File::open(path) else {
        return false;
    };
    let mut magic = [0u8; 4];
    if file.read_exact(&mut magic).is_err() {
        return false;
    }
    magic == MAGIC_V2 || magic == MAGIC_V1
}

/// Resolve TransformerConfig from .apr metadata, HF config.json, or --model-size fallback.
///
/// Precedence:
///   1. `.apr` file metadata (provable, validated at import)
///   2. HuggingFace `config.json` beside the model file (#2417)
///   3. HuggingFace `config.json` in model directory
///   4. `--model-size` string match (legacy fallback, no .apr file)
///   5. Error naming the path that was actually supplied (refuse to silently
///      degrade to tiny, and refuse to claim no path was given when one was)
pub(crate) fn resolve_transformer_config(
    model_path: Option<&Path>,
    model_size: Option<&str>,
) -> Result<entrenar::transformer::TransformerConfig> {
    // Attempt 1: Read architecture from .apr file metadata
    if let Some(path) = model_path.filter(|p| p.is_file()) {
        if let Some(config) = read_apr_architecture(path) {
            return Ok(config);
        }
        // Attempt 2: a .safetensors / .gguf checkout ships its architecture in
        // a sibling config.json. Before #2417 this was only consulted when the
        // model path was a DIRECTORY, so `apr finetune model.safetensors`
        // could never resolve an architecture even with the config.json
        // sitting right next to the weights.
        if let Some(config) = read_sibling_hf_config(path) {
            return Ok(config);
        }
        eprintln!(
            "[GH-376] WARNING: could not read architecture from '{}' \
             (not APR v2 metadata, and no sibling config.json), \
             falling back to --model-size",
            path.display()
        );
    }

    // Attempt 3: Read architecture from HuggingFace config.json in model directory
    if let Some(path) = model_path.filter(|p| p.is_dir()) {
        if let Some(config) = read_hf_config_json(path) {
            return Ok(config);
        }
    }

    // Attempt 4: Legacy --model-size string matching.
    //
    // #2417: when no --model-size was given the legacy message was
    // "No model path or --model-size provided" even though a path WAS the
    // first positional argument and the CLI had just read tensors out of it.
    // Report what actually happened instead.
    if model_size.is_none() {
        if let Some(path) = model_path {
            return Err(CliError::ValidationFailed(
                unresolvable_architecture_message(path),
            ));
        }
    }
    resolve_transformer_config_by_size(model_size)
}

/// The accurate diagnostic for "a model path was supplied but no architecture
/// could be derived from it".
fn unresolvable_architecture_message(path: &Path) -> String {
    let kind = path
        .extension()
        .and_then(|e| e.to_str())
        .map_or_else(|| "file".to_string(), |e| format!(".{e} file"));
    format!(
        "Cannot determine architecture from '{}': the {kind} carries no APR v2 \
         architecture metadata and no HuggingFace config.json was found beside it \
         (looked for {}). Convert it with `apr convert`, place the model's \
         config.json alongside the weights, or pass --model-size.",
        path.display(),
        sibling_config_candidates(path)
            .iter()
            .map(|p| format!("'{}'", p.display()))
            .collect::<Vec<_>>()
            .join(" and "),
    )
}

/// Where a `config.json` for `file` may live: the pacha cache writes
/// `<hash>.config.json` next to `<hash>.safetensors`, while a HuggingFace
/// checkout puts a plain `config.json` in the same directory.
fn sibling_config_candidates(file: &Path) -> Vec<std::path::PathBuf> {
    let Some(dir) = file.parent() else {
        return Vec::new();
    };
    let mut candidates = Vec::new();
    if let Some(stem) = file.file_stem() {
        let mut name = stem.to_os_string();
        name.push(".config.json");
        candidates.push(dir.join(name));
    }
    candidates.push(dir.join("config.json"));
    candidates
}

/// Read TransformerConfig from a `config.json` sitting beside a model file.
fn read_sibling_hf_config(file: &Path) -> Option<entrenar::transformer::TransformerConfig> {
    sibling_config_candidates(file)
        .iter()
        .find_map(|c| read_hf_config_file(c))
}

/// Read TransformerConfig from a HuggingFace `config.json` in a model directory.
///
/// Parses the standard HF model config format used by Qwen, LLaMA, Mistral, etc.
/// Returns None if config.json doesn't exist or required fields are missing.
fn read_hf_config_json(dir: &Path) -> Option<entrenar::transformer::TransformerConfig> {
    read_hf_config_file(&dir.join("config.json"))
}

/// Parse one HuggingFace `config.json` file into a `TransformerConfig`.
fn read_hf_config_file(config_path: &Path) -> Option<entrenar::transformer::TransformerConfig> {
    let data = std::fs::read_to_string(config_path).ok()?;
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

#[cfg(test)]
mod tests_2417 {
    use super::*;

    const QWEN_CONFIG: &str = r#"{
        "hidden_size": 896,
        "num_attention_heads": 14,
        "num_key_value_heads": 2,
        "intermediate_size": 4864,
        "num_hidden_layers": 24,
        "vocab_size": 151936
    }"#;

    fn tmpdir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("apr-2417-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    /// #2417 — `apr finetune model.safetensors` reported "No model path or
    /// --model-size provided" while a model path WAS the first positional
    /// argument. The message named a condition that was false.
    #[test]
    fn safetensors_without_config_reports_the_path_it_was_given() {
        let dir = tmpdir("nocfg");
        let model = dir.join("weights.safetensors");
        std::fs::write(&model, b"not-an-apr-file").expect("write model");

        let err = resolve_transformer_config(Some(&model), None)
            .expect_err("architecture is genuinely underivable here");
        let msg = err.to_string();

        assert!(
            !msg.contains("No model path"),
            "message still claims no path was provided: {msg}"
        );
        assert!(
            msg.contains("weights.safetensors"),
            "message must name the path it was given: {msg}"
        );
        assert!(
            !msg.contains(".apr metadata"),
            "message must not cite .apr metadata for a .safetensors input: {msg}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// With no path AND no --model-size, the original message is still the
    /// accurate one.
    #[test]
    fn no_path_and_no_size_keeps_the_original_message() {
        let err = resolve_transformer_config(None, None).expect_err("nothing to go on");
        assert!(err.to_string().contains("No model path or --model-size"));
    }

    /// #2417 — a HuggingFace checkout keeps `config.json` beside the weights.
    /// Resolution used to consult it only when the model path was a DIRECTORY,
    /// so pointing at the .safetensors file itself could never work.
    #[test]
    fn safetensors_resolves_from_sibling_config_json() {
        let dir = tmpdir("hf");
        let model = dir.join("model.safetensors");
        std::fs::write(&model, b"not-an-apr-file").expect("write model");
        std::fs::write(dir.join("config.json"), QWEN_CONFIG).expect("write config");

        let config = resolve_transformer_config(Some(&model), None)
            .expect("architecture comes from the sibling config.json");
        assert_eq!(config.hidden_size, 896);
        assert_eq!(config.num_hidden_layers, 24);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The pacha cache stores `<hash>.safetensors` and `<hash>.config.json`
    /// side by side — the exact fixture layout the audit used.
    #[test]
    fn safetensors_resolves_from_hash_prefixed_sibling_config() {
        let dir = tmpdir("pacha");
        let model = dir.join("064a3693fa1ea02c.safetensors");
        std::fs::write(&model, b"not-an-apr-file").expect("write model");
        std::fs::write(dir.join("064a3693fa1ea02c.config.json"), QWEN_CONFIG)
            .expect("write config");

        let config = resolve_transformer_config(Some(&model), None)
            .expect("architecture comes from <stem>.config.json");
        assert_eq!(config.num_attention_heads, 14);
        assert_eq!(config.num_kv_heads, 2);
        std::fs::remove_dir_all(&dir).ok();
    }
}

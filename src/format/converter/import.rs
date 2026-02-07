//! APR Import Pipeline
//! PMAT-197: Extracted from mod.rs

use crate::error::{AprenderError, Result};
use crate::format::converter_types::{
    Architecture, ImportError, ImportOptions, Source, TensorExpectation, ValidationConfig,
};
use crate::format::gguf::{
    load_gguf_raw, load_gguf_with_tokenizer, GgufModelConfig, GgufRawTensor, GgufTokenizer,
};
use crate::format::layout_contract::contract;
use crate::format::validation::{AprValidator, TensorStats, ValidationReport};
use crate::serialization::safetensors::{MappedSafeTensors, UserMetadata};
use std::collections::BTreeMap;

// Import write functions and helpers from parent module
use super::{validate_tensor_values, write_apr_file, write_apr_file_raw};
use std::fs;
use std::path::{Path, PathBuf};

#[cfg(feature = "hf-hub-integration")]
use crate::format::converter_types::parse_import_error;

pub fn apr_import<P: AsRef<Path>>(
    source: &str,
    output: P,
    options: ImportOptions,
) -> Result<ValidationReport> {
    let parsed_source = Source::parse(source)?;
    let output_path = output.as_ref();

    // Step 1: Resolve source to local path
    let local_path = resolve_source(&parsed_source, options.cache)?;

    // Step 2: Check if GGUF - use raw import path to preserve quantization
    let extension = local_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");
    if extension == "gguf" {
        // PMAT-103: Use raw GGUF loading to preserve Q4_K/Q6_K quantization
        // This is critical for format parity - we don't want to dequantize and re-quantize
        return apr_import_gguf_raw(&local_path, output_path, &options);
    }

    // Non-GGUF path: Load tensors as f32, apply quantization during write
    let mut load_result = load_source_tensors(&local_path, &options)?;

    // PMAT-SAFETENSORS-TOK-001: For HuggingFace SafeTensors imports, try to find
    // tokenizer.json from the same repo if not found as sibling file
    if load_result.tokenizer.is_none() {
        if let Source::HuggingFace { org, repo, .. } = &parsed_source {
            // Try to find tokenizer.json in HuggingFace cache for this repo
            if let Some(tokenizer_path) = find_in_cache(org, repo, "tokenizer.json") {
                load_result.tokenizer = load_tokenizer_from_json(&tokenizer_path);
            }
        }
    }

    // PMAT-224: Warn about unverified architectures before proceeding
    let effective_arch = if options.architecture == Architecture::Auto {
        // Try to infer from model config
        load_result
            .model_config
            .as_ref()
            .and_then(|cfg| cfg.architecture.as_ref())
            .and_then(|arch_str| match arch_str.to_lowercase().as_str() {
                "qwen2" | "qwen" | "qwen2.5" => Some(Architecture::Qwen2),
                "llama" | "llama2" | "llama3" => Some(Architecture::Llama),
                "whisper" => Some(Architecture::Whisper),
                "bert" => Some(Architecture::Bert),
                _ => None,
            })
            .unwrap_or(Architecture::Auto)
    } else {
        options.architecture
    };

    if !effective_arch.is_inference_verified() {
        eprintln!(
            "[PMAT-224] WARNING: Architecture '{}' has not been verified for inference.",
            effective_arch.display_name()
        );
        eprintln!(
            "[PMAT-224] The imported APR file may not produce correct output with `apr run`."
        );
        eprintln!(
            "[PMAT-224] Verified architectures: Qwen2, LLaMA. Use --strict to reject unverified."
        );
        if options.strict {
            return Err(AprenderError::FormatError {
                message: format!(
                    "Architecture '{}' is not verified for inference (--strict mode). \
                     Remove --strict to import anyway, or specify --arch qwen2/llama.",
                    effective_arch.display_name()
                ),
            });
        }
    }

    // Step 3: Map tensor names to canonical APR names
    let mapped_tensors = map_tensor_names(&load_result.tensors, effective_arch);

    // GH-205: Also map F16 raw tensor names for passthrough
    let mapped_f16_raw: BTreeMap<String, (Vec<u8>, Vec<usize>)> = load_result
        .f16_raw_tensors
        .iter()
        .map(|(name, (bytes, shape))| {
            let mapped_name = effective_arch.map_name(name);
            (mapped_name, (bytes.clone(), shape.clone()))
        })
        .collect();

    // Step 4: ENFORCE CONTRACT (P0 - contracts/tensor-layout-v1.yaml)
    // The contract is the SOURCE OF TRUTH for tensor shapes.
    let layout_contract = contract();
    let vocab_size = load_result
        .model_config
        .as_ref()
        .and_then(|c| c.vocab_size)
        .unwrap_or(0);
    let hidden_dim = load_result
        .model_config
        .as_ref()
        .and_then(|c| c.hidden_size)
        .unwrap_or(0);

    if vocab_size > 0 && hidden_dim > 0 {
        for (name, (_data, shape)) in &mapped_tensors {
            if let Err(e) = layout_contract.validate_apr_shape(name, shape, vocab_size, hidden_dim)
            {
                eprintln!(
                    "[CONTRACT-VIOLATION] {}: {} (see contracts/tensor-layout-v1.yaml)",
                    name, e
                );
                if options.strict {
                    return Err(AprenderError::FormatError {
                        message: format!("Contract violation: {e}"),
                    });
                }
            }
        }
        eprintln!(
            "[CONTRACT] Validated {} tensors against tensor-layout-v1.yaml (vocab={}, hidden={})",
            mapped_tensors.len(),
            vocab_size,
            hidden_dim
        );
    } else {
        eprintln!(
            "[CONTRACT] WARNING: Cannot validate contract - missing vocab_size or hidden_dim"
        );
    }

    // Step 5: Validate tensors (inline validation)
    let validation_result = validate_tensors(&mapped_tensors, &options)?;

    // Step 5: Write APR format (with tokenizer AND model config - CRITICAL for inference)
    // Note: Quantization (fp16/int8/int4) is applied during write for true packed storage
    // PMAT-223: Pass user metadata for preservation in APR custom field
    // GH-205: Pass F16 raw tensors for passthrough
    write_apr_file(
        &mapped_tensors,
        &mapped_f16_raw,
        output_path,
        &options,
        load_result.tokenizer.as_ref(),
        load_result.model_config.as_ref(),
        &load_result.user_metadata,
    )?;

    Ok(validation_result)
}

/// Import GGUF file preserving original quantization (Q4_K, Q6_K, etc.)
///
/// This is the preferred path for GGUF import as it preserves the exact
/// quantization from the source file, ensuring format parity with Ollama/llama.cpp.
pub(crate) fn apr_import_gguf_raw(
    gguf_path: &Path,
    output_path: &Path,
    options: &ImportOptions,
) -> Result<ValidationReport> {
    // Load GGUF with raw quantized tensors (preserves Q4_K bytes)
    let raw_result = load_gguf_raw(gguf_path)?;

    // PMAT-232: Validate tokenizer data before import
    // GGUF files without embedded vocabulary cannot produce working APR files.
    // The APR format requires vocabulary+merges for text encoding/decoding.
    //
    // If GGUF has no tokenizer but --tokenizer was provided, load from external file.
    let effective_tokenizer = if raw_result.tokenizer.vocabulary.is_empty() {
        // Try external tokenizer if provided
        if let Some(ref tokenizer_path) = options.tokenizer_path {
            eprintln!(
                "[PMAT-232] GGUF has no embedded tokenizer, trying external: {}",
                tokenizer_path.display()
            );
            match load_tokenizer_from_explicit_path(tokenizer_path) {
                Some(tok) => {
                    eprintln!(
                        "[PMAT-232] External tokenizer loaded: {} vocab tokens, {} merge rules",
                        tok.vocabulary.len(),
                        tok.merges.len()
                    );
                    tok
                }
                None => {
                    let msg = format!(
                        "Failed to load external tokenizer from '{}'. \
                         Ensure the file is a valid HuggingFace tokenizer.json.",
                        tokenizer_path.display()
                    );
                    eprintln!("[PMAT-232] ERROR: {}", msg);
                    return Err(AprenderError::FormatError { message: msg });
                }
            }
        } else {
            let msg = format!(
                "GGUF file '{}' has no embedded tokenizer vocabulary. \
                 This is a 'weights-only' GGUF that cannot produce a working APR file. \
                 Solutions: (1) Use a GGUF with embedded tokenizer, or \
                 (2) Provide --tokenizer /path/to/tokenizer.json, or \
                 (3) Use SafeTensors format with sibling tokenizer.json, or \
                 (4) Import from HuggingFace source: apr import hf://ORG/REPO -o model.apr",
                gguf_path.display()
            );
            eprintln!("[PMAT-232] ERROR: {}", msg);
            return Err(AprenderError::FormatError { message: msg });
        }
    } else {
        // GGUF has embedded tokenizer
        if raw_result.tokenizer.merges.is_empty() {
            eprintln!(
                "[PMAT-232] WARNING: GGUF file has vocabulary but no BPE merges. \
                 Text encoding may fail for multi-character tokens. \
                 Consider using a GGUF with embedded tokenizer.ggml.merges."
            );
        } else {
            eprintln!(
                "[PMAT-232] Tokenizer validated: {} vocab tokens, {} merge rules",
                raw_result.tokenizer.vocabulary.len(),
                raw_result.tokenizer.merges.len()
            );
        }
        raw_result.tokenizer.clone()
    };

    // PMAT-222: Auto-detect architecture from GGUF model config
    // This ensures proper tensor name mapping (GGUF→HF convention)
    // Critical for format parity: GGUF uses blk.N.attn_q, APR uses layers.N.self_attn.q_proj
    let effective_arch = if options.architecture == Architecture::Auto {
        // Detect from model config
        raw_result
            .model_config
            .architecture
            .as_ref()
            .and_then(|arch_str| match arch_str.to_lowercase().as_str() {
                "qwen2" | "qwen" => Some(Architecture::Qwen2),
                "llama" | "llama2" | "llama3" => Some(Architecture::Llama),
                "whisper" => Some(Architecture::Whisper),
                "bert" => Some(Architecture::Bert),
                _ => None,
            })
            .unwrap_or(Architecture::Auto)
    } else {
        options.architecture.clone()
    };

    if effective_arch != Architecture::Auto {
        eprintln!(
            "[PMAT-222] Auto-detected architecture: {:?} (tensor names will be mapped)",
            effective_arch
        );
    }

    // PMAT-224: Warn about unverified architectures
    if !effective_arch.is_inference_verified() {
        eprintln!(
            "[PMAT-224] WARNING: Architecture '{}' has not been verified for inference.",
            effective_arch.display_name()
        );
        eprintln!("[PMAT-224] Verified architectures: Qwen2, LLaMA. Use --strict to reject.");
        if options.strict {
            return Err(AprenderError::FormatError {
                message: format!(
                    "Architecture '{}' is not verified for inference (--strict mode). \
                     Remove --strict to import anyway, or specify --arch qwen2/llama.",
                    effective_arch.display_name()
                ),
            });
        }
    }

    // Map tensor names to APR canonical format using detected architecture
    let mapped_tensors: BTreeMap<String, GgufRawTensor> = raw_result
        .tensors
        .into_iter()
        .map(|(name, tensor)| {
            let mapped_name = effective_arch.map_name(&name);
            (mapped_name, tensor)
        })
        .collect();

    // MANDATORY CONTRACT ENFORCEMENT (GH-208)
    // The contract is NOT A SUGGESTION - it is the SOURCE OF TRUTH.
    // ALL shape transformations go through enforce_import_contract().
    // See: contracts/tensor-layout-v1.yaml, Five Whys analysis in layout_contract.rs
    use crate::format::layout_contract::enforce_import_contract;

    let vocab_size = raw_result.model_config.vocab_size.unwrap_or(0);
    let hidden_dim = raw_result.model_config.hidden_size.unwrap_or(0);

    // Validate contract enforcement is possible
    if vocab_size == 0 || hidden_dim == 0 {
        return Err(AprenderError::FormatError {
            message: format!(
                "CONTRACT ENFORCEMENT FAILED: Missing vocab_size ({}) or hidden_dim ({}). \
                 Cannot validate tensor layouts without model config. \
                 This GGUF file may be malformed.",
                vocab_size, hidden_dim
            ),
        });
    }

    // Apply CONTRACT-ENFORCED shape transformation to all tensors
    let mapped_tensors: BTreeMap<String, GgufRawTensor> = mapped_tensors
        .into_iter()
        .map(|(name, mut tensor)| {
            // Use CONTRACT to determine shape transformation
            let (apr_shape, needs_data_transpose) =
                enforce_import_contract(&name, &tensor.shape, vocab_size, hidden_dim);

            // GH-208: Data transpose should NEVER be needed for GGUF import
            // If this fires, the contract is misconfigured
            assert!(
                !needs_data_transpose,
                "CONTRACT BUG: enforce_import_contract returned needs_data_transpose=true for '{}'. \
                 GGUF→APR NEVER needs data transpose. See GH-208.",
                name
            );

            tensor.shape = apr_shape;
            (name, tensor)
        })
        .collect();

    eprintln!(
        "[CONTRACT-ENFORCED] {} tensors transformed via tensor-layout-v1.yaml (vocab={}, hidden={})",
        mapped_tensors.len(),
        vocab_size,
        hidden_dim
    );

    // Basic validation (skip strict validation for quantized tensors - can't compute meaningful stats)
    let mut validation_result = ValidationReport::new();
    validation_result.total_score = 85; // Default score for raw import (tensors preserved)

    // Write APR file with raw quantized tensors
    // Use effective_tokenizer which may be from external file (PMAT-232)
    write_apr_file_raw(
        &mapped_tensors,
        output_path,
        options,
        Some(&effective_tokenizer),
        Some(&raw_result.model_config),
    )?;

    Ok(validation_result)
}

/// Resolve a source to a local file path
pub(crate) fn resolve_source(source: &Source, cache: bool) -> Result<PathBuf> {
    match source {
        Source::Local(path) => {
            if !path.exists() {
                // GH-129: Use ImportError for actionable message
                let err = ImportError::NotFound {
                    resource: path.display().to_string(),
                    status: 0, // Local file, not HTTP
                };
                return Err(AprenderError::from(err));
            }
            Ok(path.clone())
        }
        Source::HuggingFace { org, repo, file } => {
            // PMAT-168: Smart default filename based on repo type
            let filename = file.as_deref().unwrap_or_else(|| {
                // Detect GGUF repos by name convention
                if repo.to_lowercase().contains("gguf") {
                    // Try common GGUF naming patterns
                    // e.g., Qwen2.5-Coder-1.5B-Instruct-GGUF -> qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
                    "model.gguf" // We'll try multiple patterns in find_in_cache
                } else {
                    "model.safetensors"
                }
            });

            // Check standard cache locations first
            if cache {
                // PMAT-168: Try multiple common filenames for GGUF repos
                if repo.to_lowercase().contains("gguf") && file.is_none() {
                    // Try common GGUF naming patterns
                    let base_name = repo
                        .to_lowercase()
                        .replace("-gguf", "")
                        .replace("_gguf", "");
                    let gguf_patterns = [
                        format!("{}-q4_k_m.gguf", base_name),
                        format!("{}-q4_k.gguf", base_name),
                        format!("{}-q8_0.gguf", base_name),
                        "model.gguf".to_string(),
                    ];
                    for pattern in &gguf_patterns {
                        if let Some(path) = find_in_cache(org, repo, pattern) {
                            return Ok(path);
                        }
                    }
                }
                if let Some(path) = find_in_cache(org, repo, filename) {
                    return Ok(path);
                }
            }

            // Try to download using hf-hub if feature is enabled (GH-129: proper error handling)
            #[cfg(feature = "hf-hub-integration")]
            {
                let repo_id = format!("{org}/{repo}");
                // Return the result directly without explicit return statements
                download_from_hf(&repo_id, filename)
            }

            // Only reach here if hf-hub-integration feature is disabled
            #[cfg(not(feature = "hf-hub-integration"))]
            Err(AprenderError::FormatError {
                message: format!(
                    "HuggingFace model not found in cache. Download manually:\n\
                     huggingface-cli download {org}/{repo} {filename}\n\
                     Or provide a local path to the SafeTensors/GGUF file.",
                ),
            })
        }
        Source::Url(url) => Err(AprenderError::FormatError {
            message: format!("URL download not yet implemented: {url}"),
        }),
    }
}

/// Get XDG cache directory or fallback.
fn get_xdg_cache_dir() -> PathBuf {
    std::env::var("XDG_CACHE_HOME")
        .ok()
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::var("HOME")
                .map(|h| PathBuf::from(h).join(".cache"))
                .unwrap_or_else(|_| PathBuf::from(".cache"))
        })
}

/// Get `HuggingFace` cache directory.
fn get_hf_cache_dir() -> PathBuf {
    std::env::var("HF_HOME")
        .ok()
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::var("HOME")
                .map(|h| PathBuf::from(h).join(".cache").join("huggingface"))
                .unwrap_or_else(|_| PathBuf::from(".cache").join("huggingface"))
        })
}

/// Check aprender cache for a file.
fn find_in_aprender_cache(
    cache_base: &Path,
    org: &str,
    repo: &str,
    filename: &str,
) -> Option<PathBuf> {
    let apr_cache = cache_base
        .join("aprender")
        .join("hf")
        .join(org)
        .join(repo)
        .join(filename);
    apr_cache.exists().then_some(apr_cache)
}

/// Check `HuggingFace` hub cache for a file.
fn find_in_hf_hub_cache(
    cache_base: &Path,
    org: &str,
    repo: &str,
    filename: &str,
) -> Option<PathBuf> {
    let hf_cache = cache_base
        .join("hub")
        .join(format!("models--{org}--{repo}"));

    if !hf_cache.exists() {
        return None;
    }

    let snapshot_dir = hf_cache.join("snapshots");
    let entries = fs::read_dir(&snapshot_dir).ok()?;

    for entry in entries.flatten() {
        let file_path = entry.path().join(filename);
        if file_path.exists() {
            return Some(file_path);
        }
    }
    None
}

/// Find a model file in standard cache locations
fn find_in_cache(org: &str, repo: &str, filename: &str) -> Option<PathBuf> {
    let cache_paths = [get_xdg_cache_dir(), get_hf_cache_dir()];

    for cache_base in &cache_paths {
        if let Some(path) = find_in_aprender_cache(cache_base, org, repo, filename) {
            return Some(path);
        }
        if let Some(path) = find_in_hf_hub_cache(cache_base, org, repo, filename) {
            return Some(path);
        }
    }

    None
}

/// Download a file from HuggingFace Hub
#[cfg(feature = "hf-hub-integration")]
fn download_from_hf(repo_id: &str, filename: &str) -> Result<PathBuf> {
    use hf_hub::api::sync::ApiBuilder;

    // Build API client (uses HF_TOKEN if available)
    let token = std::env::var("HF_TOKEN").ok();
    let mut builder = ApiBuilder::new();
    if let Some(t) = token {
        builder = builder.with_token(Some(t));
    }

    let api = builder.build().map_err(|e| {
        let resource = format!("{repo_id}/{filename}");
        let err = parse_import_error(&e.to_string(), &resource);
        AprenderError::from(err)
    })?;

    // Get repo handle
    let repo = api.model(repo_id.to_string());

    // Download the file (GH-129: parse error for actionable messages)
    let path = repo.get(filename).map_err(|e| {
        let resource = format!("{repo_id}/{filename}");
        let err = parse_import_error(&e.to_string(), &resource);
        AprenderError::from(err)
    })?;

    Ok(path)
}

/// Result of loading source tensors (may include tokenizer data)
pub(crate) struct SourceLoadResult {
    /// Tensor data (name -> (data, shape))
    tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    /// GH-205: Raw F16 tensor bytes for passthrough (name -> bytes)
    /// Tensors in this map should NOT be converted - write raw to APR
    f16_raw_tensors: BTreeMap<String, (Vec<u8>, Vec<usize>)>,
    /// Tokenizer data (only present for GGUF files)
    tokenizer: Option<GgufTokenizer>,
    /// Model config (CRITICAL for inference - from GGUF)
    model_config: Option<GgufModelConfig>,
    /// PMAT-223: User metadata from SafeTensors `__metadata__` section
    user_metadata: UserMetadata,
}

/// Load model config from config.json alongside the model file (PMAT-098)
///
/// This is the preferred way to get model config for SafeTensors models.
/// Falls back to shape inference if config.json is not found.
pub(crate) fn load_model_config_from_json(model_path: &Path) -> Option<GgufModelConfig> {
    // Look for config.json alongside the model file
    let config_path = model_path.with_file_name("config.json");
    if !config_path.exists() {
        return None;
    }

    let content = fs::read_to_string(&config_path).ok()?;
    let json: serde_json::Value = serde_json::from_str(&content).ok()?;

    // Parse HuggingFace config.json format
    let hidden_size = json
        .get("hidden_size")
        .and_then(serde_json::Value::as_u64)
        .map(|v| v as usize);

    let num_layers = json
        .get("num_hidden_layers")
        .and_then(serde_json::Value::as_u64)
        .map(|v| v as usize);

    let num_heads = json
        .get("num_attention_heads")
        .and_then(serde_json::Value::as_u64)
        .map(|v| v as usize);

    let num_kv_heads = json
        .get("num_key_value_heads")
        .and_then(serde_json::Value::as_u64)
        .map(|v| v as usize)
        .or(num_heads); // Default to num_heads if not specified (no GQA)

    let vocab_size = json
        .get("vocab_size")
        .and_then(serde_json::Value::as_u64)
        .map(|v| v as usize);

    let intermediate_size = json
        .get("intermediate_size")
        .and_then(serde_json::Value::as_u64)
        .map(|v| v as usize);

    let max_position_embeddings = json
        .get("max_position_embeddings")
        .and_then(serde_json::Value::as_u64)
        .map(|v| v as usize);

    let rope_theta = json
        .get("rope_theta")
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(10000.0);

    let rms_norm_eps = json
        .get("rms_norm_eps")
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(1e-6);

    let architecture = json
        .get("model_type")
        .and_then(|v| v.as_str())
        .map(ToString::to_string);

    // PMAT-114: Infer rope_type from architecture
    // Qwen2/Qwen2.5 models use NEOX-style RoPE (type 2)
    let rope_type = match architecture.as_deref() {
        Some("qwen2" | "qwen2.5" | "qwen") => Some(2), // NEOX style
        _ => Some(0),                                  // Default to NORM style
    };

    Some(GgufModelConfig {
        architecture,
        hidden_size,
        num_layers,
        num_heads,
        num_kv_heads,
        vocab_size,
        intermediate_size,
        max_position_embeddings,
        rope_theta: Some(rope_theta as f32),
        rms_norm_eps: Some(rms_norm_eps as f32),
        rope_type,
    })
}

/// Parse tokenizer from already-loaded JSON content.
///
/// Extracted from `load_tokenizer_from_json` for testability. This is the pure
/// JSON-parsing core that receives parsed JSON values directly, with no filesystem I/O.
///
/// # Arguments
/// * `json` - Parsed tokenizer.json content
/// * `config_json` - Optional parsed config.json content (for `vocab_size`, BOS/EOS)
pub(crate) fn parse_tokenizer_json(
    json: &serde_json::Value,
    config_json: Option<&serde_json::Value>,
) -> Option<GgufTokenizer> {
    // Step 1: Extract base vocabulary from model.vocab
    let vocab_obj = json.get("model")?.get("vocab")?;
    let vocab_map = vocab_obj.as_object()?;

    // Build vocab map (token -> id) from base vocab
    let mut token_to_id: std::collections::BTreeMap<u32, String> = vocab_map
        .iter()
        .filter_map(|(token, id)| Some((id.as_u64()? as u32, token.clone())))
        .collect();

    // Step 2: Add special tokens from added_tokens
    if let Some(added) = json.get("added_tokens").and_then(|v| v.as_array()) {
        for token in added {
            if let (Some(content), Some(id)) = (
                token.get("content").and_then(|v| v.as_str()),
                token.get("id").and_then(|v| v.as_u64()),
            ) {
                token_to_id.insert(id as u32, content.to_string());
            }
        }
    }

    // Step 3: Get expected vocab_size from config.json for padding
    let expected_vocab_size = config_json
        .and_then(|cfg| cfg.get("vocab_size").and_then(|v| v.as_u64()))
        .map(|v| v as u32)
        .unwrap_or(0);

    // Step 4: Build vocabulary vector, padding with <unk> for missing IDs
    let max_id = token_to_id.keys().max().copied().unwrap_or(0);
    let final_size = (expected_vocab_size.max(max_id + 1)) as usize;

    let mut vocabulary: Vec<String> = vec!["<unk>".to_string(); final_size];
    for (id, token) in token_to_id {
        if (id as usize) < vocabulary.len() {
            vocabulary[id as usize] = token;
        }
    }

    eprintln!(
        "[BUG-EXPORT-004] Vocab: base={}, added={}, expected={}, final={}",
        vocab_map.len(),
        json.get("added_tokens")
            .and_then(|v| v.as_array())
            .map(|a| a.len())
            .unwrap_or(0),
        expected_vocab_size,
        vocabulary.len()
    );

    if vocabulary.is_empty() {
        return None;
    }

    // BUG-EXPORT-004: Extract special tokens (BOS/EOS)
    // PRIORITY 1: Read from config.json (authoritative source)
    // PRIORITY 2: Infer from added_tokens in tokenizer.json (fallback)
    let mut bos_token_id = None;
    let mut eos_token_id = None;

    if let Some(cfg) = config_json {
        bos_token_id = cfg
            .get("bos_token_id")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32);
        eos_token_id = cfg
            .get("eos_token_id")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32);
        eprintln!(
            "[BUG-EXPORT-004] Read BOS/EOS from config.json: bos={:?}, eos={:?}",
            bos_token_id, eos_token_id
        );
    }

    // Fallback: infer from added_tokens (less reliable)
    if bos_token_id.is_none() || eos_token_id.is_none() {
        if let Some(added_tokens) = json.get("added_tokens").and_then(|v| v.as_array()) {
            for token in added_tokens {
                let content = token.get("content").and_then(|v| v.as_str());
                let id = token.get("id").and_then(|v| v.as_u64()).map(|v| v as u32);

                if let (Some(content), Some(id)) = (content, id) {
                    // Common BOS/EOS token patterns
                    if bos_token_id.is_none()
                        && (content.contains("bos")
                            || content == "<s>"
                            || content == "<|startoftext|>")
                    {
                        bos_token_id = Some(id);
                    }
                    if eos_token_id.is_none()
                        && (content.contains("eos") || content == "</s>" || content == "<|eot_id|>")
                    {
                        eos_token_id = Some(id);
                    }
                }
            }
        }
    }

    // Try to get model type from tokenizer.json
    let model_type = json
        .get("model")
        .and_then(|m| m.get("type"))
        .and_then(|t| t.as_str())
        .map(String::from);

    // PMAT-171: Extract BPE merge rules for encoding
    let merges = json
        .get("model")
        .and_then(|m| m.get("merges"))
        .and_then(|m| m.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();

    Some(GgufTokenizer {
        vocabulary,
        merges,
        model_type,
        bos_token_id,
        eos_token_id,
        architecture: None,
        model_name: None,
    })
}

/// Load tokenizer from sibling tokenizer.json file (PMAT-APR-TOK-001)
///
/// For SafeTensors models, the tokenizer is stored in a separate tokenizer.json file.
/// This function reads it and converts to GgufTokenizer format for APR embedding.
///
/// BUG-TOK-002 FIX: Support both HuggingFace layout (tokenizer.json) and
/// Pacha cache layout ({hash}.tokenizer.json).
pub(crate) fn load_tokenizer_from_json(model_path: &Path) -> Option<GgufTokenizer> {
    // Try standard HuggingFace layout: tokenizer.json in same directory
    let standard_path = model_path.with_file_name("tokenizer.json");

    // Try Pacha cache layout: {hash}.tokenizer.json (same stem as model)
    let stem = model_path.file_stem()?.to_str()?;
    // Strip any existing extensions like .converted from the stem
    let base_stem = stem.split('.').next().unwrap_or(stem);
    let pacha_path = model_path.with_file_name(format!("{}.tokenizer.json", base_stem));

    // Try both paths
    eprintln!(
        "[DEBUG-TOK-PATH] standard_path={}, exists={}",
        standard_path.display(),
        standard_path.exists()
    );
    eprintln!(
        "[DEBUG-TOK-PATH] pacha_path={}, exists={}",
        pacha_path.display(),
        pacha_path.exists()
    );
    let tokenizer_path = if standard_path.exists() {
        standard_path
    } else if pacha_path.exists() {
        eprintln!(
            "[BUG-TOK-002] Found tokenizer at Pacha cache path: {}",
            pacha_path.display()
        );
        pacha_path
    } else {
        eprintln!("[DEBUG-TOK-PATH] No tokenizer found!");
        return None;
    };

    let content = fs::read_to_string(&tokenizer_path).ok()?;
    let json: serde_json::Value = serde_json::from_str(&content).ok()?;

    // Load config.json (try standard path, then Pacha cache path)
    let config_path = tokenizer_path.with_file_name("config.json");
    let pacha_config_path = tokenizer_path
        .file_stem()
        .and_then(|s| s.to_str())
        .map(|s| s.split('.').next().unwrap_or(s))
        .map(|stem| tokenizer_path.with_file_name(format!("{stem}.config.json")));

    let config_json = config_path
        .exists()
        .then(|| fs::read_to_string(&config_path).ok())
        .flatten()
        .or_else(|| {
            pacha_config_path
                .as_ref()
                .filter(|p| p.exists())
                .and_then(|p| fs::read_to_string(p).ok())
        })
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok());

    parse_tokenizer_json(&json, config_json.as_ref())
}

/// PMAT-232: Load tokenizer from explicit path (for --tokenizer CLI option)
///
/// Unlike `load_tokenizer_from_json` which searches for tokenizer.json in standard locations,
/// this function loads directly from the provided path. Used for weights-only GGUF imports
/// where the tokenizer is provided externally.
pub(crate) fn load_tokenizer_from_explicit_path(tokenizer_path: &Path) -> Option<GgufTokenizer> {
    if !tokenizer_path.exists() {
        eprintln!(
            "[PMAT-232] External tokenizer not found: {}",
            tokenizer_path.display()
        );
        return None;
    }

    let content = fs::read_to_string(tokenizer_path).ok()?;
    let json: serde_json::Value = serde_json::from_str(&content).ok()?;

    // Extract vocabulary from model.vocab
    let vocab_obj = json.get("model")?.get("vocab")?;
    let vocab_map = vocab_obj.as_object()?;

    // Build vocab map (token -> id) from base vocab
    let mut token_to_id: std::collections::BTreeMap<u32, String> = vocab_map
        .iter()
        .filter_map(|(token, id)| Some((id.as_u64()? as u32, token.clone())))
        .collect();

    // Add special tokens from added_tokens
    if let Some(added) = json.get("added_tokens").and_then(|v| v.as_array()) {
        for token in added {
            if let (Some(content), Some(id)) = (
                token.get("content").and_then(|v| v.as_str()),
                token.get("id").and_then(|v| v.as_u64()),
            ) {
                token_to_id.insert(id as u32, content.to_string());
            }
        }
    }

    // Read expected vocab_size from sibling config.json
    let config_path = tokenizer_path.with_file_name("config.json");
    let expected_vocab_size = config_path
        .exists()
        .then(|| fs::read_to_string(&config_path).ok())
        .flatten()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
        .and_then(|cfg| cfg.get("vocab_size").and_then(|v| v.as_u64()))
        .map(|v| v as u32)
        .unwrap_or(0);

    // Build vocabulary vector
    let max_id = token_to_id.keys().max().copied().unwrap_or(0);
    let final_size = (expected_vocab_size.max(max_id + 1)) as usize;
    let mut vocabulary: Vec<String> = vec!["<unk>".to_string(); final_size];
    for (id, token) in &token_to_id {
        if (*id as usize) < vocabulary.len() {
            vocabulary[*id as usize] = token.clone();
        }
    }

    eprintln!(
        "[PMAT-232] External tokenizer loaded: {} vocab tokens from {}",
        vocabulary.len(),
        tokenizer_path.display()
    );

    if vocabulary.is_empty() {
        return None;
    }

    // Extract BOS/EOS from config.json
    let mut bos_token_id = None;
    let mut eos_token_id = None;

    if let Some(cfg) = config_path
        .exists()
        .then(|| fs::read_to_string(&config_path).ok())
        .flatten()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
    {
        bos_token_id = cfg
            .get("bos_token_id")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32);
        eos_token_id = cfg
            .get("eos_token_id")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32);
    }

    // Fallback: infer from added_tokens
    if bos_token_id.is_none() || eos_token_id.is_none() {
        if let Some(added_tokens) = json.get("added_tokens").and_then(|v| v.as_array()) {
            for token in added_tokens {
                let content = token.get("content").and_then(|v| v.as_str());
                let id = token.get("id").and_then(|v| v.as_u64()).map(|v| v as u32);

                if let (Some(content), Some(id)) = (content, id) {
                    if bos_token_id.is_none()
                        && (content.contains("bos")
                            || content == "<s>"
                            || content == "<|startoftext|>")
                    {
                        bos_token_id = Some(id);
                    }
                    if eos_token_id.is_none()
                        && (content.contains("eos") || content == "</s>" || content == "<|eot_id|>")
                    {
                        eos_token_id = Some(id);
                    }
                }
            }
        }
    }

    // Extract model type and merge rules
    let model_type = json
        .get("model")
        .and_then(|m| m.get("type"))
        .and_then(|t| t.as_str())
        .map(String::from);

    let merges = json
        .get("model")
        .and_then(|m| m.get("merges"))
        .and_then(|m| m.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();

    Some(GgufTokenizer {
        vocabulary,
        merges,
        model_type,
        bos_token_id,
        eos_token_id,
        architecture: None,
        model_name: None,
    })
}

/// Infer model config from tensor shapes (for SafeTensors which has no metadata)
///
/// GH-165 FIX: Now handles both HuggingFace and GGUF tensor naming conventions:
/// - HuggingFace: model.layers.N.self_attn.q_proj.weight, embed_tokens.weight
/// - GGUF: blk.N.attn_q.weight, token_embd.weight
pub(crate) fn infer_model_config_from_tensors(
    tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
) -> Option<GgufModelConfig> {
    // Try to find embedding tensor to get vocab_size and hidden_size
    // Supports both HuggingFace (embed_tokens, wte) and GGUF (token_embd) naming
    // GH-165 FIX: Handle both shape orders:
    //   - HuggingFace: [vocab_size, hidden_size] (vocab first, larger)
    //   - GGUF: [hidden_size, vocab_size] (hidden first, smaller)
    // We detect by checking which dimension is larger (vocab_size >> hidden_size typically)
    let (vocab_size, hidden_size) = tensors
        .iter()
        .find(|(name, _)| {
            name.contains("embed_tokens")
                || name.contains("wte")
                || name.contains("word_embeddings")
                || name.contains("token_embd") // GGUF naming
        })
        .and_then(|(_, (_, shape))| {
            if shape.len() == 2 {
                // Vocab size is typically much larger than hidden_size (e.g., 151936 vs 896)
                // Detect order by comparing dimensions
                let (dim0, dim1) = (shape[0], shape[1]);
                if dim0 > dim1 {
                    // [vocab_size, hidden_size] - HuggingFace order
                    Some((dim0, dim1))
                } else {
                    // [hidden_size, vocab_size] - GGUF order
                    Some((dim1, dim0))
                }
            } else {
                None
            }
        })?;

    // Count transformer layers
    // Supports HuggingFace (layers.N., h.N., blocks.N.) and GGUF (blk.N.)
    let num_layers = tensors
        .keys()
        .filter_map(|name| {
            // Match patterns like "layers.N." or "h.N." or "blocks.N." or "blk.N."
            // GGUF uses "blk.N." pattern
            if let Some(start) = name.find("blk.") {
                let rest = &name[start + 4..]; // Skip "blk."
                if let Some(end) = rest.find('.') {
                    if let Ok(n) = rest[..end].parse::<usize>() {
                        return Some(n);
                    }
                }
            }
            // HuggingFace patterns
            let patterns = [
                (name.find("layers."), 7), // "layers." is 7 chars
                (name.find("h."), 2),      // "h." is 2 chars
                (name.find("blocks."), 7), // "blocks." is 7 chars
            ];
            for (pos, skip_len) in patterns {
                if let Some(start) = pos {
                    let rest = &name[start + skip_len..];
                    if let Some(end) = rest.find('.') {
                        if let Ok(n) = rest[..end].parse::<usize>() {
                            return Some(n);
                        }
                    }
                }
            }
            None
        })
        .max()
        .map(|n| n + 1)
        .unwrap_or(0);

    // BUG-EXPORT-004 FIX: Infer num_heads correctly using KV projection for GQA models
    //
    // For GQA models like Qwen2:
    // - q_proj: [q_dim, hidden_size] where q_dim = num_heads * head_dim
    // - k_proj: [kv_dim, hidden_size] where kv_dim = num_kv_heads * head_dim
    //
    // We can compute head_dim = kv_dim / num_kv_heads, then num_heads = q_dim / head_dim
    // Common num_kv_heads values: 2, 4, 8 (for efficient GQA)

    // First, get KV projection dimension
    let kv_dim = tensors
        .iter()
        .find(|(name, _)| {
            name.contains("k_proj.weight")
                || name.contains("key.weight")
                || name.contains("attn_k.weight") // GGUF naming
        })
        .and_then(|(_, (_, shape))| {
            if shape.len() == 2 {
                // k_proj shape is [kv_dim, hidden_size] or [hidden_size, kv_dim]
                // kv_dim is typically smaller (GQA) or equal (MHA) to hidden_size
                Some(shape[0].min(shape[1]))
            } else {
                None
            }
        });

    // Get Q projection dimension
    let q_dim = tensors
        .iter()
        .find(|(name, _)| {
            name.contains("q_proj.weight")
                || name.contains("query.weight")
                || name.contains("attn_q.weight") // GGUF naming
        })
        .and_then(|(_, (_, shape))| {
            if shape.len() == 2 {
                // q_proj shape is [q_dim, hidden_size] - q_dim typically equals hidden_size
                Some(shape[0].min(shape[1]))
            } else {
                None
            }
        });

    // Infer num_heads and num_kv_heads from Q and KV projection shapes
    let (num_heads, inferred_num_kv_heads) = match (q_dim, kv_dim) {
        (Some(q), Some(kv)) if kv < q => {
            // GQA model: kv_dim < q_dim
            // Derive from common head_dims rather than guessing n_kv
            let mut result = (None, None);
            for head_dim in [64, 128, 96, 80] {
                if kv % head_dim == 0 && q % head_dim == 0 {
                    let n_kv = kv / head_dim;
                    let n_heads = q / head_dim;
                    if n_heads >= n_kv && n_kv > 0 {
                        result = (Some(n_heads), Some(n_kv));
                        break;
                    }
                }
            }
            result
        }
        (Some(q), _) if q == hidden_size => {
            // MHA model: q_dim == hidden_size, kv_dim likely equal
            let mut result = (None, None);
            for head_dim in [64, 128, 96, 80] {
                if hidden_size % head_dim == 0 {
                    let n_heads = hidden_size / head_dim;
                    result = (Some(n_heads), Some(n_heads)); // MHA: same as num_heads
                    break;
                }
            }
            result
        }
        _ => (None, None),
    };

    // Try to get intermediate_size from gate/up projection
    // Supports HuggingFace (gate_proj, up_proj) and GGUF (ffn_gate, ffn_up)
    // GH-165 FIX: Handle both shape orders (intermediate_size > hidden_size typically)
    let intermediate_size = tensors
        .iter()
        .find(|(name, _)| {
            name.contains("gate_proj")
                || name.contains("up_proj")
                || name.contains("fc1")
                || name.contains("ffn_gate") // GGUF naming
                || name.contains("ffn_up") // GGUF naming
        })
        .and_then(|(_, (_, shape))| {
            if shape.len() == 2 {
                // intermediate_size is typically 4x hidden_size, so take the larger dimension
                Some(shape[0].max(shape[1]))
            } else {
                None
            }
        });

    // Infer architecture from tensor naming patterns
    // Supports both HuggingFace and GGUF naming conventions
    let architecture = if tensors.keys().any(|k| k.contains("model.layers")) {
        Some("qwen2".to_string()) // or llama
    } else if tensors.keys().any(|k| k.contains("transformer.h")) {
        Some("gpt2".to_string())
    } else if tensors.keys().any(|k| k.contains("blk.")) {
        Some("qwen2".to_string()) // GGUF models (likely qwen2 or llama variant)
    } else {
        Some("unknown".to_string())
    };

    // BUG-EXPORT-004: Use the already-inferred num_kv_heads from earlier KV dimension analysis
    // This is more accurate because it was computed together with num_heads using consistent head_dim
    let num_kv_heads = inferred_num_kv_heads.or(num_heads); // Fall back to MHA if inference fails

    // PMAT-114: Infer rope_type from architecture
    let rope_type = match architecture.as_deref() {
        Some("qwen2" | "qwen2.5" | "qwen") => Some(2), // NEOX style
        _ => Some(0),                                  // Default to NORM style
    };

    Some(GgufModelConfig {
        architecture,
        hidden_size: Some(hidden_size),
        num_layers: Some(num_layers),
        num_heads,
        num_kv_heads, // PMAT-107: Now correctly inferred for GQA models
        vocab_size: Some(vocab_size),
        intermediate_size,
        max_position_embeddings: Some(4096), // Default
        rope_theta: Some(10000.0),           // Default
        rms_norm_eps: Some(1e-6),            // Default
        rope_type,
    })
}

/// Load tensors from source file (`SafeTensors` format)
pub(crate) fn load_source_tensors(
    path: &Path,
    _options: &ImportOptions,
) -> Result<SourceLoadResult> {
    let extension = path.extension().and_then(|e| e.to_str()).unwrap_or("");

    match extension {
        "safetensors" => {
            // GH-205: Load tensors with F16 passthrough support
            let st_result = load_safetensors_with_f16_passthrough(path)?;
            // PMAT-098: Read config.json if available (CRITICAL for correct inference)
            // Fall back to shape inference only if config.json is missing
            let model_config = load_model_config_from_json(path)
                .or_else(|| infer_model_config_from_tensors(&st_result.tensors));
            // PMAT-APR-TOK-001: Load tokenizer from sibling tokenizer.json for APR embedding
            let tokenizer = load_tokenizer_from_json(path);
            Ok(SourceLoadResult {
                tensors: st_result.tensors,
                f16_raw_tensors: st_result.f16_raw_tensors, // GH-205: F16 passthrough
                tokenizer,
                model_config,
                user_metadata: st_result.user_metadata,
            })
        }
        "apr" => {
            // Already APR format - extract tensors
            Err(AprenderError::FormatError {
                message: "Cannot import from APR format - use direct loading instead".to_string(),
            })
        }
        "gguf" => {
            // Load GGUF with tokenizer AND model config (CRITICAL for inference)
            let result = load_gguf_with_tokenizer(path)?;
            Ok(SourceLoadResult {
                tensors: result.tensors,
                f16_raw_tensors: BTreeMap::new(), // GGUF uses different quant formats
                tokenizer: Some(result.tokenizer),
                model_config: Some(result.model_config),
                user_metadata: UserMetadata::new(),
            })
        }
        "bin" | "pt" | "pth" => Err(AprenderError::FormatError {
            message: format!(
                "PyTorch format ({extension}) not supported. Convert to SafeTensors first."
            ),
        }),
        other => Err(AprenderError::FormatError {
            message: format!("Unknown file format: .{other}. Supported: .safetensors"),
        }),
    }
}

/// Load tensors from `SafeTensors` file using memory-mapped I/O for efficiency
///
/// PMAT-187: Validates all tensors after loading to catch corruption early.
pub(crate) fn load_safetensors_tensors(
    path: &Path,
) -> Result<BTreeMap<String, (Vec<f32>, Vec<usize>)>> {
    // Use MappedSafeTensors for zero-copy mmap access (much faster for large models)
    let mapped = MappedSafeTensors::open(path).map_err(|e| AprenderError::FormatError {
        message: format!("Failed to mmap SafeTensors: {e}"),
    })?;

    let mut tensors = BTreeMap::new();
    let names: Vec<String> = mapped
        .tensor_names()
        .iter()
        .map(|&s| (*s).to_string())
        .collect();

    for name in &names {
        // Skip __metadata__ key if present
        if name.starts_with("__") {
            continue;
        }

        let meta = mapped
            .get_metadata(name)
            .ok_or_else(|| AprenderError::FormatError {
                message: format!("Tensor metadata not found for '{name}'"),
            })?;

        let data = mapped
            .get_tensor(name)
            .map_err(|e| AprenderError::FormatError {
                message: format!("Failed to extract tensor '{name}': {e}"),
            })?;

        // PMAT-187: Validate tensor values after loading (Jidoka - stop the line)
        validate_tensor_values(name, &data)?;

        tensors.insert(name.clone(), (data, meta.shape.clone()));
    }

    Ok(tensors)
}

/// GH-205: Result of loading SafeTensors with F16 passthrough support.
pub(crate) struct SafeTensorsLoadResult {
    /// F32 tensors (native F32 or converted from other dtypes for non-passthrough)
    pub tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    /// Raw F16 tensor bytes for passthrough (avoids F16→F32→F16 precision loss)
    pub f16_raw_tensors: BTreeMap<String, (Vec<u8>, Vec<usize>)>,
    /// User metadata from `__metadata__` section
    pub user_metadata: UserMetadata,
}

/// GH-205: Load SafeTensors with F16 passthrough support.
///
/// This function preserves raw F16 bytes for direct passthrough to APR format,
/// avoiding the precision loss from F16→F32→F16 round-trip conversion.
///
/// Returns:
/// - `tensors`: All tensors as F32 (for backward compatibility and validation)
/// - `f16_raw_tensors`: Raw F16 bytes for passthrough (only F16 tensors)
/// - `user_metadata`: User metadata from SafeTensors header
pub(crate) fn load_safetensors_with_f16_passthrough(path: &Path) -> Result<SafeTensorsLoadResult> {
    let mapped = MappedSafeTensors::open(path).map_err(|e| AprenderError::FormatError {
        message: format!("Failed to mmap SafeTensors: {e}"),
    })?;

    let user_metadata = mapped.user_metadata().clone();
    if !user_metadata.is_empty() {
        eprintln!(
            "[PMAT-223] Extracted {} user metadata key(s) from SafeTensors __metadata__",
            user_metadata.len()
        );
    }

    let mut tensors = BTreeMap::new();
    let mut f16_raw_tensors = BTreeMap::new();
    let mut f16_count = 0usize;

    let names: Vec<String> = mapped
        .tensor_names()
        .iter()
        .map(|&s| (*s).to_string())
        .collect();

    for name in &names {
        if name.starts_with("__") {
            continue;
        }

        let meta = mapped
            .get_metadata(name)
            .ok_or_else(|| AprenderError::FormatError {
                message: format!("Tensor metadata not found for '{name}'"),
            })?;

        // GH-205: Check if this is an F16 tensor for passthrough
        if meta.dtype == "F16" {
            // Get raw bytes for passthrough (no conversion)
            if let Some(raw_bytes) = mapped.get_tensor_bytes(name) {
                f16_raw_tensors.insert(name.clone(), (raw_bytes.to_vec(), meta.shape.clone()));
                f16_count += 1;
            }
        }

        // Always also get F32 representation (for validation and backward compat)
        let data = mapped
            .get_tensor(name)
            .map_err(|e| AprenderError::FormatError {
                message: format!("Failed to extract tensor '{name}': {e}"),
            })?;

        validate_tensor_values(name, &data)?;
        tensors.insert(name.clone(), (data, meta.shape.clone()));
    }

    if f16_count > 0 {
        eprintln!(
            "[GH-205] F16 passthrough: {} of {} tensors will be written as raw F16",
            f16_count,
            tensors.len()
        );
    }

    Ok(SafeTensorsLoadResult {
        tensors,
        f16_raw_tensors,
        user_metadata,
    })
}

/// Map tensor names to APR canonical format
pub(crate) fn map_tensor_names(
    tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    architecture: Architecture,
) -> BTreeMap<String, (Vec<f32>, Vec<usize>)> {
    tensors
        .iter()
        .map(|(name, data)| {
            let mapped_name = architecture.map_name(name);
            (mapped_name, data.clone())
        })
        .collect()
}

/// Check tensor expectations and return error message if failed.
fn check_tensor_expectation(
    name: &str,
    stats: &TensorStats,
    options: &ImportOptions,
) -> Option<String> {
    if options.validation == ValidationConfig::None {
        return None;
    }
    let expectation = TensorExpectation::for_tensor(name)?;
    let err = expectation.check(stats).err()?;
    if options.validation == ValidationConfig::Strict && options.strict {
        Some(format!("{name}: {err}"))
    } else {
        None
    }
}

/// Check for special values (NaN/Inf) and return error messages.
fn check_special_values(name: &str, stats: &TensorStats, options: &ImportOptions) -> Vec<String> {
    if options.validation == ValidationConfig::None {
        return Vec::new();
    }
    let mut errors = Vec::new();
    if stats.nan_count > 0 {
        errors.push(format!("{name}: contains {} NaN values", stats.nan_count));
    }
    if stats.inf_count > 0 {
        errors.push(format!("{name}: contains {} Inf values", stats.inf_count));
    }
    errors
}

/// Validate a single tensor and collect errors.
pub(crate) fn validate_single_tensor(
    name: &str,
    data: &[f32],
    options: &ImportOptions,
    validator: &mut AprValidator,
    errors: &mut Vec<String>,
) {
    let stats = compute_tensor_stats(name, data);

    if let Some(err) = check_tensor_expectation(name, &stats, options) {
        errors.push(err);
    }
    errors.extend(check_special_values(name, &stats, options));

    validator.add_tensor_stats(stats);
}

/// Required tensor alternatives for strict mode validation (DEFECT-001 fix)
/// At least one tensor from each group must be present for a model to be considered complete.
/// Each group represents equivalent tensors that may have different names across formats.
const STRICT_REQUIRED_TENSOR_GROUPS: &[&[&str]] = &[
    // Final layer norm - different formats use different names:
    // - SafeTensors (HuggingFace): model.norm.weight
    // - GGUF: output_norm.weight (mapped to model.norm.weight)
    // - Some models: norm.weight
    &["model.norm.weight", "norm.weight", "output_norm.weight"],
];

/// Check if any tensor from a group of alternatives is present
fn has_required_tensor(
    tensor_names: &std::collections::HashSet<&str>,
    alternatives: &[&str],
) -> bool {
    alternatives.iter().any(|&name| tensor_names.contains(name))
}

/// Validate tensors according to architecture expectations
pub(crate) fn validate_tensors(
    tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    options: &ImportOptions,
) -> Result<ValidationReport> {
    let mut validator = AprValidator::new();
    let mut validation_errors = Vec::new();

    // DEFECT-001 FIX: Check for required tensors in strict mode
    if options.strict {
        let tensor_names: std::collections::HashSet<&str> =
            tensors.keys().map(|s| s.as_str()).collect();

        for alternatives in STRICT_REQUIRED_TENSOR_GROUPS {
            if !has_required_tensor(&tensor_names, alternatives) {
                validation_errors.push(format!(
                    "Missing required tensor: {} (or equivalents: {})",
                    alternatives[0],
                    alternatives[1..].join(", ")
                ));
            }
        }
    }

    for (name, (data, _shape)) in tensors {
        validate_single_tensor(name, data, options, &mut validator, &mut validation_errors);
    }

    let report = validator.validate();

    if !validation_errors.is_empty() && options.strict {
        return Err(AprenderError::FormatError {
            message: format!(
                "Validation failed ({} errors):\n  - {}",
                validation_errors.len(),
                validation_errors.join("\n  - ")
            ),
        });
    }

    Ok(report)
}

/// Accumulator for tensor statistics during first pass.
pub(crate) struct TensorAccumulator {
    pub(crate) sum: f64,
    pub(crate) min: f32,
    pub(crate) max: f32,
    pub(crate) nan_count: usize,
    pub(crate) inf_count: usize,
    pub(crate) zero_count: usize,
    pub(crate) valid_count: usize,
}

impl TensorAccumulator {
    pub(crate) fn new() -> Self {
        Self {
            sum: 0.0,
            min: f32::INFINITY,
            max: f32::NEG_INFINITY,
            nan_count: 0,
            inf_count: 0,
            zero_count: 0,
            valid_count: 0,
        }
    }

    pub(crate) fn accumulate(&mut self, v: f32) {
        if v.is_nan() {
            self.nan_count += 1;
        } else if v.is_infinite() {
            self.inf_count += 1;
        } else {
            self.sum += v as f64;
            self.min = self.min.min(v);
            self.max = self.max.max(v);
            self.valid_count += 1;
            if v == 0.0 {
                self.zero_count += 1;
            }
        }
    }

    pub(crate) fn mean(&self) -> f32 {
        if self.valid_count > 0 {
            (self.sum / self.valid_count as f64) as f32
        } else {
            0.0
        }
    }

    pub(crate) fn safe_min(&self) -> f32 {
        if self.min == f32::INFINITY {
            0.0
        } else {
            self.min
        }
    }

    pub(crate) fn safe_max(&self) -> f32 {
        if self.max == f32::NEG_INFINITY {
            0.0
        } else {
            self.max
        }
    }
}

/// Compute standard deviation from data.
pub(crate) fn compute_std(data: &[f32], mean: f32, valid_count: usize) -> f32 {
    if valid_count <= 1 {
        return 0.0;
    }
    let variance_sum: f64 = data
        .iter()
        .filter(|v| !v.is_nan() && !v.is_infinite())
        .map(|&v| {
            let diff = v as f64 - mean as f64;
            diff * diff
        })
        .sum();
    ((variance_sum / (valid_count - 1) as f64).sqrt()) as f32
}

/// Compute statistics for a tensor
pub(crate) fn compute_tensor_stats(name: &str, data: &[f32]) -> TensorStats {
    if data.is_empty() {
        return TensorStats {
            name: name.to_string(),
            count: 0,
            min: 0.0,
            max: 0.0,
            mean: 0.0,
            std: 0.0,
            nan_count: 0,
            inf_count: 0,
            zero_count: 0,
        };
    }

    let mut acc = TensorAccumulator::new();
    for &v in data {
        acc.accumulate(v);
    }

    let mean = acc.mean();
    let std = compute_std(data, mean, acc.valid_count);

    TensorStats {
        name: name.to_string(),
        count: data.len(),
        min: acc.safe_min(),
        max: acc.safe_max(),
        mean,
        std,
        nan_count: acc.nan_count,
        inf_count: acc.inf_count,
        zero_count: acc.zero_count,
    }
}

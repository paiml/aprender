//! APR Import Pipeline
//! PMAT-197: Extracted from mod.rs

use crate::error::{AprenderError, Result};
use crate::format::converter_types::{
    Architecture, ImportError, ImportOptions, Source, TensorExpectation, ValidationConfig,
};
use crate::format::gguf::{
    load_gguf_raw, load_gguf_with_tokenizer, GgufModelConfig, GgufRawTensor, GgufTokenizer,
};
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
            "[PMAT-224] Verified architectures: Qwen2, LLaMA. Use --force to suppress this warning."
        );
        if !options.force {
            return Err(AprenderError::FormatError {
                message: format!(
                    "Architecture '{}' is not verified for inference. \
                     Use --force to import anyway, or specify --arch qwen2/llama.",
                    effective_arch.display_name()
                ),
            });
        }
    }

    // Step 3: Map tensor names to canonical APR names
    let mapped_tensors = map_tensor_names(&load_result.tensors, effective_arch);

    // Step 4: Validate tensors (inline validation)
    let validation_result = validate_tensors(&mapped_tensors, &options)?;

    // Step 5: Write APR format (with tokenizer AND model config - CRITICAL for inference)
    // Note: Quantization (fp16/int8/int4) is applied during write for true packed storage
    // PMAT-223: Pass user metadata for preservation in APR custom field
    write_apr_file(
        &mapped_tensors,
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
        eprintln!(
            "[PMAT-224] Verified architectures: Qwen2, LLaMA. Use --force to suppress."
        );
        if !options.force {
            return Err(AprenderError::FormatError {
                message: format!(
                    "Architecture '{}' is not verified for inference. \
                     Use --force to import anyway, or specify --arch qwen2/llama.",
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

    // Basic validation (skip strict validation for quantized tensors - can't compute meaningful stats)
    let mut validation_result = ValidationReport::new();
    validation_result.total_score = 85; // Default score for raw import (tensors preserved)

    // Write APR file with raw quantized tensors
    write_apr_file_raw(
        &mapped_tensors,
        output_path,
        options,
        Some(&raw_result.tokenizer),
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

/// Load tokenizer from sibling tokenizer.json file (PMAT-APR-TOK-001)
///
/// For SafeTensors models, the tokenizer is stored in a separate tokenizer.json file.
/// This function reads it and converts to GgufTokenizer format for APR embedding.
pub(crate) fn load_tokenizer_from_json(model_path: &Path) -> Option<GgufTokenizer> {
    let tokenizer_path = model_path.with_file_name("tokenizer.json");
    if !tokenizer_path.exists() {
        return None;
    }

    let content = fs::read_to_string(&tokenizer_path).ok()?;
    let json: serde_json::Value = serde_json::from_str(&content).ok()?;

    // Extract vocabulary from model.vocab (HuggingFace tokenizer.json format)
    let vocab_obj = json.get("model")?.get("vocab")?;
    let vocab_map = vocab_obj.as_object()?;

    // Build vocab vector sorted by ID
    let mut vocab_vec: Vec<(String, u32)> = vocab_map
        .iter()
        .filter_map(|(token, id)| Some((token.clone(), id.as_u64()? as u32)))
        .collect();
    vocab_vec.sort_by_key(|(_, id)| *id);

    let vocabulary: Vec<String> = vocab_vec.into_iter().map(|(token, _)| token).collect();

    if vocabulary.is_empty() {
        return None;
    }

    // Extract special tokens (BOS/EOS)
    let mut bos_token_id = None;
    let mut eos_token_id = None;

    if let Some(added_tokens) = json.get("added_tokens").and_then(|v| v.as_array()) {
        for token in added_tokens {
            let content = token.get("content").and_then(|v| v.as_str());
            let id = token.get("id").and_then(|v| v.as_u64()).map(|v| v as u32);

            if let (Some(content), Some(id)) = (content, id) {
                // Common BOS/EOS token patterns
                if content.contains("bos")
                    || content == "<s>"
                    || content == "<|startoftext|>"
                    || content == "<|im_start|>"
                {
                    bos_token_id = Some(id);
                }
                if content.contains("eos")
                    || content == "</s>"
                    || content == "<|endoftext|>"
                    || content == "<|im_end|>"
                    || content == "<|eot_id|>"
                {
                    eos_token_id = Some(id);
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

    // Try to infer num_heads from Q projection shape
    // Supports HuggingFace (q_proj.weight) and GGUF (attn_q.weight)
    let num_heads = tensors
        .iter()
        .find(|(name, _)| {
            name.contains("q_proj.weight")
                || name.contains("query.weight")
                || name.contains("attn_q.weight") // GGUF naming
        })
        .and_then(|(_, (_, shape))| {
            if shape.len() == 2 {
                // q_proj is typically [num_heads * head_dim, hidden_size]
                // hidden_size / head_dim = num_heads, and output_dim = num_heads * head_dim
                // For many models: output_dim == hidden_size, so num_heads = hidden_size / head_dim
                // Common head_dims: 64, 128. Try to find a reasonable factor.
                let q_dim = shape[0];
                if q_dim == hidden_size {
                    // Likely num_heads = hidden_size / head_dim
                    // Try common head_dims
                    for head_dim in [64, 128, 96, 80] {
                        if hidden_size % head_dim == 0 {
                            return Some(hidden_size / head_dim);
                        }
                    }
                }
                None
            } else {
                None
            }
        });

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

    // PMAT-107: Infer num_kv_heads from K projection tensor shape for GQA support
    // K tensor shape: [kv_dim, hidden_dim] where kv_dim = num_kv_heads * head_dim
    // Supports HuggingFace (k_proj.weight) and GGUF (attn_k.weight)
    let num_kv_heads = tensors
        .iter()
        .find(|(name, _)| {
            name.contains("k_proj.weight")
                || name.contains("key.weight")
                || name.contains("attn_k.weight") // GGUF naming
        })
        .and_then(|(_, (_, shape))| {
            if let (2, Some(num_h)) = (shape.len(), num_heads) {
                let kv_dim = shape[0];
                // head_dim = hidden_size / num_heads
                if hidden_size % num_h == 0 {
                    let head_dim = hidden_size / num_h;
                    // num_kv_heads = kv_dim / head_dim
                    if kv_dim % head_dim == 0 {
                        return Some(kv_dim / head_dim);
                    }
                }
            }
            None
        })
        .or(num_heads); // Fall back to MHA if inference fails

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
            // PMAT-223: Load tensors AND user metadata from SafeTensors
            let (tensors, user_metadata) = load_safetensors_with_user_metadata(path)?;
            // PMAT-098: Read config.json if available (CRITICAL for correct inference)
            // Fall back to shape inference only if config.json is missing
            let model_config = load_model_config_from_json(path)
                .or_else(|| infer_model_config_from_tensors(&tensors));
            // PMAT-APR-TOK-001: Load tokenizer from sibling tokenizer.json for APR embedding
            let tokenizer = load_tokenizer_from_json(path);
            Ok(SourceLoadResult {
                tensors,
                tokenizer,
                model_config,
                user_metadata,
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

/// Load tensors AND user metadata from SafeTensors file (PMAT-223).
///
/// Unlike `load_safetensors_tensors`, this also extracts the `__metadata__`
/// section from the SafeTensors header for preservation during format conversion.
pub(crate) fn load_safetensors_with_user_metadata(
    path: &Path,
) -> Result<(BTreeMap<String, (Vec<f32>, Vec<usize>)>, UserMetadata)> {
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

        let data = mapped
            .get_tensor(name)
            .map_err(|e| AprenderError::FormatError {
                message: format!("Failed to extract tensor '{name}': {e}"),
            })?;

        validate_tensor_values(name, &data)?;
        tensors.insert(name.clone(), (data, meta.shape.clone()));
    }

    Ok((tensors, user_metadata))
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
    if options.validation == ValidationConfig::Strict && !options.force {
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

/// Validate tensors according to architecture expectations
pub(crate) fn validate_tensors(
    tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    options: &ImportOptions,
) -> Result<ValidationReport> {
    let mut validator = AprValidator::new();
    let mut validation_errors = Vec::new();

    for (name, (data, _shape)) in tensors {
        validate_single_tensor(name, data, options, &mut validator, &mut validation_errors);
    }

    let report = validator.validate();

    if !validation_errors.is_empty() && !options.force {
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

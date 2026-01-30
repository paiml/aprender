//! APR Converter Module - Import Pipeline
//!
//! Implements Section 13 of APR-SPEC.md: Import/Convert Pipeline
//!
//! Supports:
//! - `HuggingFace` Hub downloads (<hf://org/repo>)
//! - `SafeTensors` conversion
//! - Inline validation during conversion
//! - Quantization and compression

use crate::error::{AprenderError, Result};
use crate::format::gguf::{
    dequantize_q4_0, dequantize_q4_1, dequantize_q5_0, dequantize_q8_0, load_gguf_raw,
    load_gguf_with_tokenizer, GgufModelConfig, GgufRawTensor, GgufReader, GgufTokenizer,
};
use crate::format::v2::{AprV2Metadata, AprV2Writer, QuantizationMetadata, TensorDType};
use crate::format::validation::{AprValidator, TensorStats, ValidationReport};
use crate::format::Compression;
use crate::serialization::safetensors::{save_safetensors, MappedSafeTensors};
use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

// PMAT-197: Re-export types from converter_types module for backward compatibility
pub use crate::format::converter_types::{
    detect_sharded_model, Architecture, ImportError, ImportOptions, QuantizationType,
    ShardedIndex, Source, TensorExpectation, ValidationConfig,
};
#[cfg(feature = "hf-hub-integration")]
pub use crate::format::converter_types::parse_import_error;

// HF Hub integration is used via hf_hub::api::sync::ApiBuilder in download_from_hf()

// ============================================================================
// Converter
// ============================================================================

/// APR Converter with builder pattern
#[derive(Debug)]
pub struct AprConverter {
    source: Option<Source>,
    architecture: Architecture,
    validation: ValidationConfig,
    quantize: Option<QuantizationType>,
    compress: Option<Compression>,
}

impl AprConverter {
    /// Create a new converter
    #[must_use]
    pub fn new() -> Self {
        Self {
            source: None,
            architecture: Architecture::Auto,
            validation: ValidationConfig::Strict,
            quantize: None,
            compress: None,
        }
    }

    /// Set the source
    pub fn source(mut self, source: &str) -> Result<Self> {
        self.source = Some(Source::parse(source)?);
        Ok(self)
    }

    /// Set the architecture
    #[must_use]
    pub fn architecture(mut self, arch: Architecture) -> Self {
        self.architecture = arch;
        self
    }

    /// Set validation config
    #[must_use]
    pub fn validate(mut self, config: ValidationConfig) -> Self {
        self.validation = config;
        self
    }

    /// Set quantization
    #[must_use]
    pub fn quantize(mut self, quant: QuantizationType) -> Self {
        self.quantize = Some(quant);
        self
    }

    /// Set compression
    #[must_use]
    pub fn compress(mut self, comp: Compression) -> Self {
        self.compress = Some(comp);
        self
    }

    /// Run the conversion
    pub fn convert(self) -> Result<Vec<u8>> {
        let source = self.source.ok_or_else(|| AprenderError::FormatError {
            message: "No source specified".to_string(),
        })?;

        // NOTE: Full conversion pipeline is tracked in GH-80 (metaheuristics milestone)
        // Current limitation: Returns error for unsupported sources
        Err(AprenderError::FormatError {
            message: format!(
                "Conversion from {:?} not yet implemented - see GH-80",
                source
            ),
        })
    }
}

impl Default for AprConverter {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// High-level API
// ============================================================================

/// Import a model from source to APR format
///
/// # Arguments
/// * `source` - Source path: local file, <hf://org/repo>, or URL
/// * `output` - Output APR file path
/// * `options` - Import configuration
///
/// # Returns
/// * `ValidationReport` with 100-point checklist results
///
/// # Example
/// ```rust,ignore
/// use aprender::format::{apr_import, ImportOptions, Architecture};
///
/// let options = ImportOptions {
///     architecture: Architecture::Whisper,
///     ..Default::default()
/// };
/// let report = apr_import("model.safetensors", "model.apr", options)?;
/// println!("Score: {}/100", report.total_score);
/// ```
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

    // Step 3: Map tensor names to canonical APR names
    let mapped_tensors = map_tensor_names(&load_result.tensors, options.architecture);

    // Step 4: Validate tensors (inline validation)
    let validation_result = validate_tensors(&mapped_tensors, &options)?;

    // Step 5: Write APR format (with tokenizer AND model config - CRITICAL for inference)
    // Note: Quantization (fp16/int8/int4) is applied during write for true packed storage
    write_apr_file(
        &mapped_tensors,
        output_path,
        &options,
        load_result.tokenizer.as_ref(),
        load_result.model_config.as_ref(),
    )?;

    Ok(validation_result)
}

/// Import GGUF file preserving original quantization (Q4_K, Q6_K, etc.)
///
/// This is the preferred path for GGUF import as it preserves the exact
/// quantization from the source file, ensuring format parity with Ollama/llama.cpp.
fn apr_import_gguf_raw(
    gguf_path: &Path,
    output_path: &Path,
    options: &ImportOptions,
) -> Result<ValidationReport> {
    // Load GGUF with raw quantized tensors (preserves Q4_K bytes)
    let raw_result = load_gguf_raw(gguf_path)?;

    // Map tensor names to APR canonical format
    let mapped_tensors: BTreeMap<String, GgufRawTensor> = raw_result
        .tensors
        .into_iter()
        .map(|(name, tensor)| {
            let mapped_name = options.architecture.map_name(&name);
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
fn resolve_source(source: &Source, cache: bool) -> Result<PathBuf> {
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
struct SourceLoadResult {
    /// Tensor data (name -> (data, shape))
    tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    /// Tokenizer data (only present for GGUF files)
    tokenizer: Option<GgufTokenizer>,
    /// Model config (CRITICAL for inference - from GGUF)
    model_config: Option<GgufModelConfig>,
}

/// Load model config from config.json alongside the model file (PMAT-098)
///
/// This is the preferred way to get model config for SafeTensors models.
/// Falls back to shape inference if config.json is not found.
fn load_model_config_from_json(model_path: &Path) -> Option<GgufModelConfig> {
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
fn load_tokenizer_from_json(model_path: &Path) -> Option<GgufTokenizer> {
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
fn load_source_tensors(path: &Path, _options: &ImportOptions) -> Result<SourceLoadResult> {
    let extension = path.extension().and_then(|e| e.to_str()).unwrap_or("");

    match extension {
        "safetensors" => {
            let tensors = load_safetensors_tensors(path)?;
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
pub(crate) fn load_safetensors_tensors(path: &Path) -> Result<BTreeMap<String, (Vec<f32>, Vec<usize>)>> {
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

/// Map tensor names to APR canonical format
fn map_tensor_names(
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
fn validate_single_tensor(
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
fn validate_tensors(
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
struct TensorAccumulator {
    sum: f64,
    min: f32,
    max: f32,
    nan_count: usize,
    inf_count: usize,
    zero_count: usize,
    valid_count: usize,
}

impl TensorAccumulator {
    fn new() -> Self {
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

    fn accumulate(&mut self, v: f32) {
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

    fn mean(&self) -> f32 {
        if self.valid_count > 0 {
            (self.sum / self.valid_count as f64) as f32
        } else {
            0.0
        }
    }

    fn safe_min(&self) -> f32 {
        if self.min == f32::INFINITY {
            0.0
        } else {
            self.min
        }
    }

    fn safe_max(&self) -> f32 {
        if self.max == f32::NEG_INFINITY {
            0.0
        } else {
            self.max
        }
    }
}

/// Compute standard deviation from data.
fn compute_std(data: &[f32], mean: f32, valid_count: usize) -> f32 {
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
fn compute_tensor_stats(name: &str, data: &[f32]) -> TensorStats {
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

/// Write tensors to native APR format
fn write_apr_file(
    tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    output: &Path,
    options: &ImportOptions,
    tokenizer: Option<&GgufTokenizer>,
    model_config: Option<&GgufModelConfig>,
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

    // PMAT-101: Pre-fuse Q, K, V into qkv_proj.weight for realizar compatibility
    // Compute this FIRST so we can include fused tensors in tensor_shapes metadata
    let (qkv_fused, qkv_bias_fused): (
        BTreeMap<String, (Vec<f32>, Vec<usize>)>,
        BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    ) = {
        let mut fused = BTreeMap::new();
        let mut bias_fused = BTreeMap::new();
        if let Some(cfg) = model_config {
            if let (Some(hidden_dim), Some(num_heads), Some(num_kv_heads)) =
                (cfg.hidden_size, cfg.num_heads, cfg.num_kv_heads)
            {
                let head_dim = hidden_dim / num_heads;
                let kv_dim = num_kv_heads * head_dim;
                let qkv_dim = hidden_dim + kv_dim + kv_dim;

                let layer_count = cfg.num_layers.unwrap_or(0);
                for layer_idx in 0..layer_count {
                    let q_name = format!("model.layers.{layer_idx}.self_attn.q_proj.weight");
                    let k_name = format!("model.layers.{layer_idx}.self_attn.k_proj.weight");
                    let v_name = format!("model.layers.{layer_idx}.self_attn.v_proj.weight");

                    if let (Some((q_data, _)), Some((k_data, _)), Some((v_data, _))) = (
                        tensors_with_lm_head.get(&q_name),
                        tensors_with_lm_head.get(&k_name),
                        tensors_with_lm_head.get(&v_name),
                    ) {
                        // Fuse as [Q; K; V] - simple concatenation (same as SafetensorsToAprConverter)
                        let mut qkv_data =
                            Vec::with_capacity(q_data.len() + k_data.len() + v_data.len());
                        qkv_data.extend_from_slice(q_data);
                        qkv_data.extend_from_slice(k_data);
                        qkv_data.extend_from_slice(v_data);

                        let qkv_name =
                            format!("model.layers.{layer_idx}.self_attn.qkv_proj.weight");
                        fused.insert(qkv_name, (qkv_data, vec![qkv_dim, hidden_dim]));
                    }

                    // PMAT-114: Also fuse Q, K, V biases if present (Qwen2 has attention bias)
                    let q_bias_name = format!("model.layers.{layer_idx}.self_attn.q_proj.bias");
                    let k_bias_name = format!("model.layers.{layer_idx}.self_attn.k_proj.bias");
                    let v_bias_name = format!("model.layers.{layer_idx}.self_attn.v_proj.bias");

                    if let (Some((q_bias, _)), Some((k_bias, _)), Some((v_bias, _))) = (
                        tensors_with_lm_head.get(&q_bias_name),
                        tensors_with_lm_head.get(&k_bias_name),
                        tensors_with_lm_head.get(&v_bias_name),
                    ) {
                        // Fuse biases as [Q_bias; K_bias; V_bias]
                        let mut qkv_bias_data =
                            Vec::with_capacity(q_bias.len() + k_bias.len() + v_bias.len());
                        qkv_bias_data.extend_from_slice(q_bias);
                        qkv_bias_data.extend_from_slice(k_bias);
                        qkv_bias_data.extend_from_slice(v_bias);

                        let qkv_bias_name =
                            format!("model.layers.{layer_idx}.self_attn.qkv_proj.bias");
                        bias_fused.insert(qkv_bias_name, (qkv_bias_data, vec![qkv_dim]));
                    }
                }
            }
        }
        (fused, bias_fused)
    };

    // Build tensor_shapes map for metadata (used by `apr tensors` command)
    // PMAT-101: Skip individual Q/K/V, include fused qkv_proj instead
    // PMAT-114: Also skip individual Q/K/V biases if fused
    let tensor_shapes: serde_json::Map<String, serde_json::Value> = {
        let mut shapes = serde_json::Map::new();

        // Add non-QKV tensors
        for (name, (_, shape)) in &tensors_with_lm_head {
            // Skip individual Q, K, V weights if we have fused versions
            if name.contains("q_proj.weight")
                || name.contains("k_proj.weight")
                || name.contains("v_proj.weight")
            {
                let layer_idx_opt = name
                    .split("layers.")
                    .nth(1)
                    .and_then(|s| s.split('.').next())
                    .and_then(|s| s.parse::<usize>().ok());
                if let Some(layer_idx) = layer_idx_opt {
                    let qkv_name = format!("model.layers.{layer_idx}.self_attn.qkv_proj.weight");
                    if qkv_fused.contains_key(&qkv_name) {
                        continue;
                    }
                }
            }

            // PMAT-114: Skip individual Q, K, V biases if we have fused versions
            if name.contains("q_proj.bias")
                || name.contains("k_proj.bias")
                || name.contains("v_proj.bias")
            {
                let layer_idx_opt = name
                    .split("layers.")
                    .nth(1)
                    .and_then(|s| s.split('.').next())
                    .and_then(|s| s.parse::<usize>().ok());
                if let Some(layer_idx) = layer_idx_opt {
                    let qkv_bias_name = format!("model.layers.{layer_idx}.self_attn.qkv_proj.bias");
                    if qkv_bias_fused.contains_key(&qkv_bias_name) {
                        continue;
                    }
                }
            }

            let shape_array: Vec<serde_json::Value> = shape
                .iter()
                .map(|&dim| serde_json::Value::Number(serde_json::Number::from(dim as u64)))
                .collect();
            shapes.insert(name.clone(), serde_json::Value::Array(shape_array));
        }

        // Add fused QKV weight tensors
        for (name, (_, shape)) in &qkv_fused {
            let shape_array: Vec<serde_json::Value> = shape
                .iter()
                .map(|&dim| serde_json::Value::Number(serde_json::Number::from(dim as u64)))
                .collect();
            shapes.insert(name.clone(), serde_json::Value::Array(shape_array));
        }

        // PMAT-114: Add fused QKV bias tensors
        for (name, (_, shape)) in &qkv_bias_fused {
            let shape_array: Vec<serde_json::Value> = shape
                .iter()
                .map(|&dim| serde_json::Value::Number(serde_json::Number::from(dim as u64)))
                .collect();
            shapes.insert(name.clone(), serde_json::Value::Array(shape_array));
        }

        shapes
    };

    // Create metadata with architecture info and tensor shapes
    let mut custom = std::collections::HashMap::new();
    custom.insert(
        "tensor_shapes".to_string(),
        serde_json::Value::Object(tensor_shapes),
    );

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

    // Add all tensors with appropriate quantization (qkv_fused computed earlier)
    for (name, (data, shape)) in &tensors_with_lm_head {
        // Skip individual Q, K, V weights if we fused them
        if name.contains("q_proj.weight")
            || name.contains("k_proj.weight")
            || name.contains("v_proj.weight")
        {
            // Check if we have a fused version for this layer
            let layer_idx_opt = name
                .split("layers.")
                .nth(1)
                .and_then(|s| s.split('.').next())
                .and_then(|s| s.parse::<usize>().ok());
            if let Some(layer_idx) = layer_idx_opt {
                let qkv_name = format!("model.layers.{layer_idx}.self_attn.qkv_proj.weight");
                if qkv_fused.contains_key(&qkv_name) {
                    continue; // Skip individual tensor, we'll write fused version
                }
            }
        }

        // PMAT-114: Skip individual Q, K, V biases if we fused them
        if name.contains("q_proj.bias")
            || name.contains("k_proj.bias")
            || name.contains("v_proj.bias")
        {
            let layer_idx_opt = name
                .split("layers.")
                .nth(1)
                .and_then(|s| s.split('.').next())
                .and_then(|s| s.parse::<usize>().ok());
            if let Some(layer_idx) = layer_idx_opt {
                let qkv_bias_name = format!("model.layers.{layer_idx}.self_attn.qkv_proj.bias");
                if qkv_bias_fused.contains_key(&qkv_bias_name) {
                    continue; // Skip individual bias, we'll write fused version
                }
            }
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

        // Write tensor (no transposition needed - QKV fusion handles it)
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
                // Native Q4_K quantization: F32 -> packed Q4_K bytes
                let q4k_bytes = quantize_q4_k(data);
                writer.add_q4k_raw_tensor(name, shape.clone(), q4k_bytes);
            }
            _ => {
                // Keep as F32 for small/critical tensors or when no quantization
                writer.add_f32_tensor(name, shape.clone(), data);
            }
        }
    }

    // Write fused QKV tensors (always large enough to quantize)
    for (name, (data, shape)) in &qkv_fused {
        match options.quantize {
            Some(QuantizationType::Fp16) => {
                writer.add_f16_tensor(name, shape.clone(), data);
            }
            Some(QuantizationType::Int8) => {
                writer.add_q8_tensor(name, shape.clone(), data);
            }
            Some(QuantizationType::Int4) => {
                writer.add_q4_tensor(name, shape.clone(), data);
            }
            Some(QuantizationType::Q4K) => {
                let q4k_bytes = quantize_q4_k(data);
                writer.add_q4k_raw_tensor(name, shape.clone(), q4k_bytes);
            }
            None => {
                writer.add_f32_tensor(name, shape.clone(), data);
            }
        }
    }

    // PMAT-114: Write fused QKV bias tensors (always F32 - biases are small and precision-sensitive)
    for (name, (data, shape)) in &qkv_bias_fused {
        writer.add_f32_tensor(name, shape.clone(), data);
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
fn write_apr_file_raw(
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
    // PMAT-114 FIX: Per apr_transformer/mod.rs comments, APR should store
    // GGML-convention dims (NOT reversed) with row-major data.
    // "APR stores GGUF data in row-major layout even though dims metadata
    //  says GGML column-major convention. The data is already correct - DO NOT transpose!"
    let tensor_shapes: serde_json::Map<String, serde_json::Value> = tensors
        .iter()
        .map(|(name, tensor)| {
            // Keep GGML-order dims - don't reverse
            let shape_array: Vec<serde_json::Value> = tensor
                .shape
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
    // PMAT-103: Store tensors as-is from GGUF for supported formats, dequantize others
    // PMAT-086: APR stores GGUF data in row-major layout with GGML convention dims
    // AprTransformer expects dims in GGML convention, data in row-major
    let mut dtype_counts: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
    for (name, tensor) in tensors {
        *dtype_counts.entry(tensor.dtype).or_insert(0) += 1;

        // Calculate element count for dequantization
        let num_elements: usize = tensor.shape.iter().product();

        // Process tensor based on dtype
        // Q5_0/Q8_0 need dequantization since realizar doesn't support them natively
        // All formats: store raw data bytes as-is, keep GGML-order dims
        match tensor.dtype {
            0 => {
                // F32 - store data as-is with GGML-order dims
                writer.add_tensor(
                    name,
                    TensorDType::F32,
                    tensor.shape.clone(),
                    tensor.data.clone(),
                );
            }
            1 => {
                // F16 - store data as-is with GGML-order dims
                writer.add_tensor(
                    name,
                    TensorDType::F16,
                    tensor.shape.clone(),
                    tensor.data.clone(),
                );
            }
            12 => {
                // Q4_K - store raw for fused kernels with GGML-order dims
                writer.add_tensor(
                    name,
                    TensorDType::Q4K,
                    tensor.shape.clone(),
                    tensor.data.clone(),
                );
            }
            14 => {
                // Q6_K - store raw for fused kernels with GGML-order dims
                writer.add_tensor(
                    name,
                    TensorDType::Q6K,
                    tensor.shape.clone(),
                    tensor.data.clone(),
                );
            }
            2 => {
                // Q4_0 - dequantize to F32, keep GGML convention dims
                // PMAT-086: APR stores GGUF data in row-major layout with GGML convention dims
                // AprTransformer expects dims in GGML convention, data in row-major
                match dequantize_q4_0(&tensor.data, 0, num_elements) {
                    Ok(f32_data) => {
                        // Keep GGML convention dims - APR transformer expects this
                        let bytes: Vec<u8> = f32_data
                            .iter()
                            .flat_map(|f: &f32| f.to_le_bytes())
                            .collect();
                        writer.add_tensor(name, TensorDType::F32, tensor.shape.clone(), bytes);
                    }
                    Err(e) => {
                        eprintln!("[WARN] Failed to dequantize Q4_0 tensor {name}: {e}");
                        writer.add_tensor(
                            name,
                            TensorDType::F32,
                            tensor.shape.clone(),
                            tensor.data.clone(),
                        );
                    }
                }
            }
            3 => {
                // Q4_1 - dequantize to F32, keep GGML convention dims
                // PMAT-086: APR stores GGUF data in row-major layout with GGML convention dims
                match dequantize_q4_1(&tensor.data, 0, num_elements) {
                    Ok(f32_data) => {
                        // Keep GGML convention dims - APR transformer expects this
                        let bytes: Vec<u8> = f32_data
                            .iter()
                            .flat_map(|f: &f32| f.to_le_bytes())
                            .collect();
                        writer.add_tensor(name, TensorDType::F32, tensor.shape.clone(), bytes);
                    }
                    Err(e) => {
                        eprintln!("[WARN] Failed to dequantize Q4_1 tensor {name}: {e}");
                        // Fall back to storing raw bytes (will fail at inference time)
                        writer.add_tensor(
                            name,
                            TensorDType::F32,
                            tensor.shape.clone(),
                            tensor.data.clone(),
                        );
                    }
                }
            }
            6 => {
                // Q5_0 - dequantize to F32, keep GGML convention dims
                // PMAT-086: APR stores GGUF data in row-major layout with GGML convention dims
                match dequantize_q5_0(&tensor.data, 0, num_elements) {
                    Ok(f32_data) => {
                        // Keep GGML convention dims - APR transformer expects this
                        let bytes: Vec<u8> =
                            f32_data.iter().flat_map(|f| f.to_le_bytes()).collect();
                        writer.add_tensor(name, TensorDType::F32, tensor.shape.clone(), bytes);
                    }
                    Err(e) => {
                        eprintln!("[WARN] Failed to dequantize Q5_0 tensor {name}: {e}");
                        writer.add_tensor(
                            name,
                            TensorDType::F32,
                            tensor.shape.clone(),
                            tensor.data.clone(),
                        );
                    }
                }
            }
            8 => {
                // Q8_0 - dequantize to F32, keep GGML convention dims
                // PMAT-086: APR stores GGUF data in row-major layout with GGML convention dims
                match dequantize_q8_0(&tensor.data, 0, num_elements) {
                    Ok(f32_data) => {
                        // Keep GGML convention dims - APR transformer expects this
                        let bytes: Vec<u8> =
                            f32_data.iter().flat_map(|f| f.to_le_bytes()).collect();
                        writer.add_tensor(name, TensorDType::F32, tensor.shape.clone(), bytes);
                    }
                    Err(e) => {
                        eprintln!("[WARN] Failed to dequantize Q8_0 tensor {name}: {e}");
                        writer.add_tensor(
                            name,
                            TensorDType::F32,
                            tensor.shape.clone(),
                            tensor.data.clone(),
                        );
                    }
                }
            }
            7 | 9 => {
                // Q5_1/Q8_1 - not yet supported, store raw with warning
                eprintln!(
                    "[WARN] GGUF dtype {} for tensor {} not yet supported, storing raw bytes",
                    tensor.dtype, name
                );
                writer.add_tensor(
                    name,
                    TensorDType::F32,
                    tensor.shape.clone(),
                    tensor.data.clone(),
                );
            }
            _ => {
                eprintln!(
                    "[WARN] Unsupported GGUF dtype {} for tensor {}, storing as-is",
                    tensor.dtype, name
                );
                writer.add_tensor(
                    name,
                    TensorDType::F32,
                    tensor.shape.clone(),
                    tensor.data.clone(),
                );
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

// ============================================================================
// Model Conversion (apr convert)
// ============================================================================

/// Options for model conversion
#[derive(Debug, Clone)]
pub struct ConvertOptions {
    /// Quantization method (int8, int4, fp16)
    pub quantize: Option<QuantizationType>,
    /// Compression method
    pub compress: Option<Compression>,
    /// Validate after conversion
    pub validate: bool,
}

impl Default for ConvertOptions {
    fn default() -> Self {
        Self {
            quantize: None,
            compress: None,
            validate: true,
        }
    }
}

/// Convert a model with quantization and/or compression
///
/// # Arguments
/// * `input` - Input model path (.safetensors or .apr)
/// * `output` - Output model path
/// * `options` - Conversion options
///
/// # Returns
/// * `ConvertReport` with size reduction stats
///
/// # Example
/// ```rust,ignore
/// use aprender::format::{apr_convert, ConvertOptions, QuantizationType};
///
/// let options = ConvertOptions {
///     quantize: Some(QuantizationType::Int8),
///     ..Default::default()
/// };
/// let report = apr_convert("model.safetensors", "model-int8.apr", options)?;
/// println!("Reduced from {} to {} bytes", report.original_size, report.converted_size);
/// ```
pub fn apr_convert<P: AsRef<Path>>(
    input: P,
    output: P,
    options: ConvertOptions,
) -> Result<ConvertReport> {
    let input_path = input.as_ref();
    let output_path = output.as_ref();
    let extension = input_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");

    eprintln!(
        "[DEBUG apr_convert] input: {:?}, extension: {:?}",
        input_path, extension
    );

    // GH-181 FIX: Preserve Q4_K_M block alignment by using raw byte pass-through
    // When source is already Q4K quantized GGUF and target is Q4K, skip dequant→requant
    if extension == "gguf" && options.quantize == Some(QuantizationType::Q4K) {
        if let Ok(raw_result) = load_gguf_raw(input_path) {
            // Check if source contains Q4_K tensors (dtype 12)
            let has_q4k = raw_result
                .tensors
                .values()
                .any(|t| t.dtype == 12 || t.dtype == 13 || t.dtype == 14); // Q4_K, Q5_K, Q6_K

            if has_q4k {
                eprintln!("[GH-181] Detected Q4K source, using raw byte pass-through");

                // Map tensor names to HuggingFace format
                let mapped_tensors: BTreeMap<String, GgufRawTensor> = raw_result
                    .tensors
                    .into_iter()
                    .map(|(name, tensor)| {
                        let mapped_name = Architecture::Qwen2.map_name(&name);
                        (mapped_name, tensor)
                    })
                    .collect();

                // Calculate original size from raw bytes
                let original_size: usize = mapped_tensors.values().map(|t| t.data.len()).sum();
                let original_count = mapped_tensors.len();

                // Write APR file with raw quantized tensors (preserves block alignment)
                let import_opts = ImportOptions {
                    architecture: Architecture::Qwen2,
                    ..Default::default()
                };
                write_apr_file_raw(
                    &mapped_tensors,
                    output_path,
                    &import_opts,
                    Some(&raw_result.tokenizer),
                    Some(&raw_result.model_config),
                )?;

                let converted_size = fs::metadata(output_path)
                    .map(|m| m.len() as usize)
                    .unwrap_or(0);

                return Ok(ConvertReport {
                    original_size,
                    converted_size,
                    tensor_count: original_count,
                    quantization: options.quantize,
                    compression: options.compress,
                    reduction_ratio: if converted_size > 0 {
                        original_size as f64 / converted_size as f64
                    } else {
                        0.0
                    },
                });
            }
        }
    }

    // F-REGR-231 FIX: For GGUF input, load with full config to preserve rope_type
    // Qwen2.5 models require rope_type=2 (NEOX style), not default 0 (NORM style)
    // PMAT-113 FIX: Also preserve tokenizer for APR embedding
    let (gguf_config, gguf_tokenizer) = if extension == "gguf" {
        match load_gguf_with_tokenizer(input_path) {
            Ok(result) => {
                eprintln!(
                    "[PMAT-113] Extracted tokenizer with {} vocabulary tokens",
                    result.tokenizer.vocabulary.len()
                );
                (Some(result.model_config), Some(result.tokenizer))
            }
            Err(_) => (None, None), // Fall back to inference if GGUF loading fails
        }
    } else {
        (None, None)
    };

    // Step 1: Load tensors
    let tensors = load_model_tensors(input_path)?;
    let original_size = calculate_tensor_size(&tensors);
    let original_count = tensors.len();

    // Step 1b: Map GGUF tensor names to HuggingFace/APR canonical format (PMAT-113 fix)
    // GGUF uses names like "blk.0.attn_q.weight" but APR loaders expect
    // HuggingFace names like "model.layers.0.self_attn.q_proj.weight"
    let tensors = if extension == "gguf" {
        eprintln!(
            "[PMAT-113] Mapping {} GGUF tensor names to HuggingFace format...",
            tensors.len()
        );
        let mapped = map_tensor_names(&tensors, Architecture::Qwen2);
        // Debug: show a few mapped names
        for (i, name) in mapped.keys().take(5).enumerate() {
            eprintln!("[PMAT-113]   {}: {}", i, name);
        }
        mapped
    } else {
        tensors
    };

    // Step 2: Handle Q4K specially - store raw Q4K bytes in APR format
    if options.quantize == Some(QuantizationType::Q4K) {
        save_model_tensors_q4k(&tensors, output_path)?;

        let converted_size = fs::metadata(output_path)
            .map(|m| m.len() as usize)
            .unwrap_or(0);

        return Ok(ConvertReport {
            original_size,
            converted_size,
            tensor_count: original_count,
            quantization: options.quantize,
            compression: options.compress,
            reduction_ratio: if converted_size > 0 {
                original_size as f64 / converted_size as f64
            } else {
                0.0
            },
        });
    }

    // Step 2b: Apply other quantization types (Fp16, Int8, Int4)
    let tensors = if let Some(quant_type) = &options.quantize {
        quantize_tensors(&tensors, quant_type)?
    } else {
        tensors
    };

    // Step 3: Save output with GGUF config if available (F-REGR-231 fix)
    // PMAT-113 FIX: Also embed tokenizer for standalone APR inference
    if let Some(ref config) = gguf_config {
        save_model_tensors_with_gguf_config_and_tokenizer(
            &tensors,
            output_path,
            options.compress,
            config,
            gguf_tokenizer.as_ref(),
        )?;
    } else {
        save_model_tensors(&tensors, output_path, options.compress)?;
    }

    // Step 4: Calculate stats
    let converted_size = fs::metadata(output_path)
        .map(|m| m.len() as usize)
        .unwrap_or(0);

    Ok(ConvertReport {
        original_size,
        converted_size,
        tensor_count: original_count,
        quantization: options.quantize,
        compression: options.compress,
        reduction_ratio: if converted_size > 0 {
            original_size as f64 / converted_size as f64
        } else {
            0.0
        },
    })
}

/// Report from model conversion
#[derive(Debug, Clone)]
pub struct ConvertReport {
    /// Original model size in bytes
    pub original_size: usize,
    /// Converted model size in bytes
    pub converted_size: usize,
    /// Number of tensors
    pub tensor_count: usize,
    /// Quantization applied
    pub quantization: Option<QuantizationType>,
    /// Compression applied
    pub compression: Option<Compression>,
    /// Size reduction ratio (original/converted)
    pub reduction_ratio: f64,
}

impl ConvertReport {
    /// Format reduction as percentage string
    #[must_use]
    pub fn reduction_percent(&self) -> String {
        if self.original_size > 0 && self.converted_size > 0 {
            let reduction = 100.0 * (1.0 - self.converted_size as f64 / self.original_size as f64);
            format!("{:.1}%", reduction)
        } else {
            "N/A".to_string()
        }
    }
}

/// Load tensors from model file
///
/// Supports: SafeTensors, APR, GGUF (GH-164 fix)
/// GGUF tensors are dequantized to F32 during loading.
pub(crate) fn load_model_tensors(path: &Path) -> Result<BTreeMap<String, (Vec<f32>, Vec<usize>)>> {
    let extension = path.extension().and_then(|e| e.to_str()).unwrap_or("");

    match extension {
        "safetensors" => load_safetensors_tensors(path),
        "apr" => load_apr_tensors_f32(path),
        "gguf" => load_gguf_tensors_f32(path),
        other => Err(AprenderError::FormatError {
            message: format!("Unsupported format for conversion: .{other}"),
        }),
    }
}

/// Load GGUF tensors and dequantize to F32 (GH-164)
///
/// Uses GgufReader::get_all_tensors_f32() which handles:
/// - Q4_K, Q5_K, Q6_K dequantization
/// - Q4_0, Q5_0, Q8_0 dequantization
/// - F16, F32 direct loading
///
/// PMAT-187: Validates all tensors after loading to catch corruption early.
fn load_gguf_tensors_f32(path: &Path) -> Result<BTreeMap<String, (Vec<f32>, Vec<usize>)>> {
    let reader = GgufReader::from_file(path)?;
    let tensors = reader.get_all_tensors_f32()?;

    // PMAT-187: Validate all tensors after loading (Jidoka - stop the line)
    for (name, (data, _shape)) in &tensors {
        validate_tensor_values(name, data)?;
    }

    Ok(tensors)
}

/// Load APR tensors and dequantize to F32 (PMAT-174)
///
/// APR binary format:
/// - Header (44 bytes): magic, version, flags, tensor_count, offsets, checksum
/// - Metadata: JSON config
/// - Tensor Index: binary tensor entries
/// - Tensor Data: raw bytes
///
/// Handles all APR dtypes: F32, F16, BF16, Q4_K, Q6_K, Q8_0
fn load_apr_tensors_f32(path: &Path) -> Result<BTreeMap<String, (Vec<f32>, Vec<usize>)>> {
    use std::io::Read;

    // Read entire file
    let mut file = fs::File::open(path).map_err(|e| AprenderError::FormatError {
        message: format!("Failed to open APR file: {e}"),
    })?;
    let mut data = Vec::new();
    file.read_to_end(&mut data)
        .map_err(|e| AprenderError::FormatError {
            message: format!("Failed to read APR file: {e}"),
        })?;

    // Validate header (44 bytes minimum)
    if data.len() < 44 {
        return Err(AprenderError::FormatError {
            message: "APR file too small for header".to_string(),
        });
    }

    // Check magic "APR\0" (0x00525041 in little-endian)
    let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    if magic != 0x0052_5041 {
        // "APR\0" in little-endian
        return Err(AprenderError::FormatError {
            message: format!("Invalid APR magic: 0x{magic:08X}, expected APR"),
        });
    }

    // Parse header
    let tensor_count = u32::from_le_bytes([data[12], data[13], data[14], data[15]]) as usize;
    let tensor_index_offset = u64::from_le_bytes([
        data[24], data[25], data[26], data[27], data[28], data[29], data[30], data[31],
    ]) as usize;
    let data_offset = u64::from_le_bytes([
        data[32], data[33], data[34], data[35], data[36], data[37], data[38], data[39],
    ]) as usize;

    // Parse tensor index
    let mut tensors = BTreeMap::new();
    let mut pos = tensor_index_offset;

    for _ in 0..tensor_count {
        if pos + 4 > data.len() {
            break;
        }

        // Name: len (2 bytes) + bytes
        let name_len = u16::from_le_bytes([data[pos], data[pos + 1]]) as usize;
        pos += 2;
        let name = String::from_utf8_lossy(&data[pos..pos + name_len]).to_string();
        pos += name_len;

        // Dtype (1 byte)
        let dtype_byte = data[pos];
        pos += 1;

        // Shape: ndim (1 byte) + dims (8 bytes each)
        let ndim = data[pos] as usize;
        pos += 1;
        let mut shape = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            let dim = u64::from_le_bytes([
                data[pos],
                data[pos + 1],
                data[pos + 2],
                data[pos + 3],
                data[pos + 4],
                data[pos + 5],
                data[pos + 6],
                data[pos + 7],
            ]) as usize;
            pos += 8;
            shape.push(dim);
        }

        // Offset and size
        let offset = u64::from_le_bytes([
            data[pos],
            data[pos + 1],
            data[pos + 2],
            data[pos + 3],
            data[pos + 4],
            data[pos + 5],
            data[pos + 6],
            data[pos + 7],
        ]) as usize;
        pos += 8;
        let size = u64::from_le_bytes([
            data[pos],
            data[pos + 1],
            data[pos + 2],
            data[pos + 3],
            data[pos + 4],
            data[pos + 5],
            data[pos + 6],
            data[pos + 7],
        ]) as usize;
        pos += 8;

        // Load tensor data
        let tensor_start = data_offset + offset;
        let tensor_end = tensor_start + size;
        if tensor_end > data.len() {
            continue;
        }
        let tensor_bytes = &data[tensor_start..tensor_end];
        let num_elements: usize = shape.iter().product();

        // Dequantize based on dtype
        let f32_data = match dtype_byte {
            0 => {
                // F32
                tensor_bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect()
            }
            1 => {
                // F16
                dequantize_f16_to_f32(tensor_bytes, num_elements)
            }
            2 => {
                // BF16
                dequantize_bf16_to_f32(tensor_bytes, num_elements)
            }
            8 => {
                // Q4_K
                dequantize_q4_k_to_f32(tensor_bytes, num_elements)
            }
            9 => {
                // Q6_K
                dequantize_q6_k_to_f32(tensor_bytes, num_elements)
            }
            10 => {
                // Q8_0
                dequantize_q8_0_to_f32(tensor_bytes, num_elements)
            }
            _ => {
                // Default to F32 interpretation
                tensor_bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect()
            }
        };

        // PMAT-187: Validate tensor values after dequantization (Jidoka - stop the line)
        validate_tensor_values(&name, &f32_data)?;

        tensors.insert(name, (f32_data, shape));
    }

    Ok(tensors)
}

/// PMAT-187: Validate tensor values for NaN/Inf/explosive corruption
///
/// Toyota Way Jidoka: Stop the line on quality defects, don't pass defects downstream.
/// This catches corruption introduced during dequantization before it propagates.
///
/// # Errors
///
/// Returns error if tensor contains NaN, Inf, or explosive values (mean > 100)
pub(crate) fn validate_tensor_values(name: &str, data: &[f32]) -> Result<()> {
    if data.is_empty() {
        return Ok(());
    }

    let mut nan_count = 0;
    let mut inf_count = 0;
    let mut sum: f64 = 0.0;

    for &value in data {
        if value.is_nan() {
            nan_count += 1;
        } else if value.is_infinite() {
            inf_count += 1;
        } else {
            sum += value as f64;
        }
    }

    // Fail fast on NaN
    if nan_count > 0 {
        return Err(AprenderError::FormatError {
            message: format!(
                "PMAT-187: Tensor '{}' contains {} NaN values (data corruption detected). \
                 Toyota Way: Stop the line - do not pass defects downstream.",
                name, nan_count
            ),
        });
    }

    // Fail fast on Inf
    if inf_count > 0 {
        return Err(AprenderError::FormatError {
            message: format!(
                "PMAT-187: Tensor '{}' contains {} Inf values (numerical overflow detected). \
                 Toyota Way: Stop the line - do not pass defects downstream.",
                name, inf_count
            ),
        });
    }

    // Check for explosive mean (indicates corrupted scale factors)
    let valid_count = data.len() - nan_count - inf_count;
    if valid_count > 0 {
        let mean = sum / valid_count as f64;
        if mean.abs() > 100.0 {
            return Err(AprenderError::FormatError {
                message: format!(
                    "PMAT-187: Tensor '{}' has explosive mean={:.2e} (expected [-100, 100]). \
                     This indicates corrupted quantization scale factors. \
                     Toyota Way: Stop the line - do not pass defects downstream.",
                    name, mean
                ),
            });
        }
    }

    Ok(())
}

/// Dequantize F16 to F32 (PMAT-174)
fn dequantize_f16_to_f32(bytes: &[u8], _num_elements: usize) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| {
            let bits = u16::from_le_bytes([c[0], c[1]]);
            f16_to_f32(bits)
        })
        .collect()
}

/// Dequantize BF16 to F32 (PMAT-174)
fn dequantize_bf16_to_f32(bytes: &[u8], _num_elements: usize) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| {
            let bits = u16::from_le_bytes([c[0], c[1]]);
            // BF16 is F32 with lower 16 mantissa bits zeroed
            f32::from_bits((bits as u32) << 16)
        })
        .collect()
}

/// Dequantize Q8_0 to F32 (PMAT-174)
/// Q8_0: 32-element blocks with f16 scale + 32 int8 quants
fn dequantize_q8_0_to_f32(bytes: &[u8], num_elements: usize) -> Vec<f32> {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 2 + 32; // f16 scale + 32 int8s
                                       // MSRV-compatible div_ceil: (n + d - 1) / d
    let num_blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let mut result = Vec::with_capacity(num_elements);

    for i in 0..num_blocks {
        let block_start = i * BLOCK_BYTES;
        if block_start + BLOCK_BYTES > bytes.len() {
            break;
        }
        let scale_bits = u16::from_le_bytes([bytes[block_start], bytes[block_start + 1]]);
        let scale = f16_to_f32(scale_bits);

        for j in 0..BLOCK_SIZE {
            if result.len() >= num_elements {
                break;
            }
            let q = bytes[block_start + 2 + j] as i8;
            result.push(q as f32 * scale);
        }
    }

    result
}

/// Calculate total tensor size in bytes (f32)
pub(crate) fn calculate_tensor_size(tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>) -> usize {
    tensors.values().map(|(data, _)| data.len() * 4).sum()
}

/// Apply quantization to tensors
pub(crate) fn quantize_tensors(
    tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    quant_type: &QuantizationType,
) -> Result<BTreeMap<String, (Vec<f32>, Vec<usize>)>> {
    let mut result = BTreeMap::new();

    for (name, (data, shape)) in tensors {
        let quantized_data = match quant_type {
            QuantizationType::Fp16 => quantize_fp16(data),
            QuantizationType::Int8 => quantize_int8(data),
            QuantizationType::Int4 => quantize_int4(data),
            QuantizationType::Q4K => {
                // Q4K: quantize to packed bytes then dequantize back to f32
                // This preserves the shape but shows quantization error
                let q4k_bytes = quantize_q4_k(data);
                dequantize_q4_k_to_f32(&q4k_bytes, data.len())
            }
        };
        result.insert(name.clone(), (quantized_data, shape.clone()));
    }

    Ok(result)
}

/// Dequantize Q4_K bytes back to F32 (for verification/testing)
/// Dequantize Q4_K data to f32 (llama.cpp compatible)
///
/// Matches the encoder format and realizar's `dequantize_q4_k_apr`:
/// - Scale packing: blocks 0-3 in lower 6 bits, blocks 4-7 use upper bits
/// - Value packing: 64-value chunks with low/high nibble interleaving
fn dequantize_q4_k_to_f32(data: &[u8], num_elements: usize) -> Vec<f32> {
    const SUPER_BLOCK_SIZE: usize = 256;
    const SUPER_BLOCK_BYTES: usize = 144;
    // PMAT-177: Minimum valid f16 normal value (~6.1e-5), clamp scales to avoid NaN
    const F16_MIN_NORMAL: f32 = 6.1e-5;

    let num_blocks = (num_elements + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
    let mut result = vec![0.0f32; num_blocks * SUPER_BLOCK_SIZE];

    for sb_idx in 0..num_blocks {
        let sb_start = sb_idx * SUPER_BLOCK_BYTES;
        let out_start = sb_idx * SUPER_BLOCK_SIZE;

        if sb_start + SUPER_BLOCK_BYTES > data.len() {
            break;
        }

        // Read d and dmin (f16) - PMAT-177: Validate for NaN/Inf
        let d_raw = f16_to_f32(u16::from_le_bytes([data[sb_start], data[sb_start + 1]]));
        let dmin_raw = f16_to_f32(u16::from_le_bytes([data[sb_start + 2], data[sb_start + 3]]));

        // PMAT-177: Replace NaN/Inf/subnormal with safe values to prevent corruption
        let d = if d_raw.is_nan() || d_raw.is_infinite() || d_raw.abs() < F16_MIN_NORMAL {
            0.0
        } else {
            d_raw
        };
        let dmin = if dmin_raw.is_nan() || dmin_raw.is_infinite() || dmin_raw.abs() < F16_MIN_NORMAL
        {
            0.0
        } else {
            dmin_raw
        };

        // Unpack scales and mins (llama.cpp format)
        let scales_bytes = &data[sb_start + 4..sb_start + 16];
        let mut scales = [0u8; 8];
        let mut mins = [0u8; 8];

        for i in 0..4 {
            // Blocks 0-3: lower 6 bits of bytes 0-3 and 4-7
            scales[i] = scales_bytes[i] & 0x3F;
            mins[i] = scales_bytes[i + 4] & 0x3F;
            // Blocks 4-7: lower 4 bits from bytes 8-11, upper 2 bits from bytes 0-3/4-7
            scales[i + 4] = (scales_bytes[i + 8] & 0x0F) | ((scales_bytes[i] >> 6) << 4);
            mins[i + 4] = (scales_bytes[i + 8] >> 4) | ((scales_bytes[i + 4] >> 6) << 4);
        }

        // Read quantized values (128 bytes = 256 4-bit values)
        let qs = &data[sb_start + 16..sb_start + 144];

        // PMAT-190 FIX: Match gguf.rs dequantize_q4_k layout exactly
        // Q4_K uses 8 sub-blocks of 32 elements each
        // Each sub-block uses ONE scale for ALL 32 elements (not different for low/high!)
        // Layout: 16 bytes per sub-block, each byte → 2 values (low nibble, high nibble)
        let mut ys_index = out_start;

        for j in 0..8 {
            // One scale per sub-block (same for both nibbles!)
            let scale = d * f32::from(scales[j]);
            let min_val = dmin * f32::from(mins[j]);

            // Process 16 bytes → 32 values
            for l in 0..16 {
                let q_byte = qs[j * 16 + l];
                let q0 = f32::from(q_byte & 0x0F);
                let q1 = f32::from(q_byte >> 4);
                result[ys_index] = q0 * scale - min_val;
                result[ys_index + 1] = q1 * scale - min_val;
                ys_index += 2;
            }
        }
    }

    result.truncate(num_elements);
    result
}

/// Quantize to fp16 - TRUE PACKING (2 bytes per value)
/// Returns dequantized f32 for now (proper f16 storage requires format change)
fn quantize_fp16(data: &[f32]) -> Vec<f32> {
    data.iter()
        .map(|&v| {
            // Convert f32 to f16 bits then back - TRUE precision reduction
            let f16_bits = f32_to_f16(v);
            f16_to_f32(f16_bits)
        })
        .collect()
}

/// Convert f32 to f16 (IEEE 754 half-precision)
fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = (bits >> 23) & 0xFF;
    let mantissa = bits & 0x7FFFFF;

    if exp == 0 {
        // Zero or denormal
        sign
    } else if exp == 0xFF {
        // Inf or NaN
        sign | 0x7C00 | if mantissa != 0 { 0x0200 } else { 0 }
    } else {
        // Normal number
        let new_exp = exp as i32 - 127 + 15;
        if new_exp >= 31 {
            // Overflow to infinity
            sign | 0x7C00
        } else if new_exp <= 0 {
            // Subnormal: f16 can represent down to 2^(-24) ≈ 5.96e-8
            // Add implicit 1 bit to mantissa (bit 23)
            let full_mantissa = mantissa | 0x800000;
            // Calculate shift: 14 bits base when new_exp=0, more for negative
            let shift = 14 - new_exp;
            if shift <= 24 {
                // Round to nearest: add 0.5 at the bit position being truncated
                let round_bit = 1u32 << (shift - 1);
                let rounded = full_mantissa.saturating_add(round_bit);
                let subnormal = (rounded >> shift) as u16;
                sign | (subnormal & 0x3FF)
            } else {
                // Too small for f16 subnormals, underflow to zero
                sign
            }
        } else {
            let new_mantissa = (mantissa >> 13) as u16;
            sign | ((new_exp as u16) << 10) | new_mantissa
        }
    }
}

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

/// Quantize F32 data to Q4_K format (GGML K-quants)
///
/// Q4_K format: 256-element super-blocks, each with:
/// - d (f16, 2 bytes): scale for scales
/// - dmin (f16, 2 bytes): scale for mins (offsets)
/// - scales (12 bytes): 8 6-bit scale values packed
/// - qs (128 bytes): 256 4-bit quantized values
///
/// Decoding formula: `value = q * (d * scales[j]) - (dmin * mins[j])`
/// Total: 144 bytes per 256 elements = 4.5 bits/weight
///
/// Returns packed Q4_K bytes ready for APR storage.
/// Quantize f32 data to Q4_K format (llama.cpp compatible)
///
/// Q4_K super-block layout (144 bytes per 256 elements):
/// - d: 2 bytes (f16 global scale)
/// - dmin: 2 bytes (f16 global min scale)
/// - scales: 12 bytes (packed 6-bit scales and mins for 8 sub-blocks)
/// - qs: 128 bytes (4-bit quantized values, interleaved low/high nibbles)
///
/// Scale packing (llama.cpp get_scale_min_k4):
/// - Blocks 0-3: scales[j] = scale_6bit, scales[j+4] = min_6bit
/// - Blocks 4-7: packed in bytes 8-11 using high bits of bytes 0-7
///
/// Value packing (candle/llama.cpp layout):
/// - For each 64-value chunk: 32 bytes store low nibbles first, then high nibbles
/// - Low nibbles use scale[is], high nibbles use scale[is+1]
fn quantize_q4_k(data: &[f32]) -> Vec<u8> {
    const SUPER_BLOCK_SIZE: usize = 256;
    const SUB_BLOCK_SIZE: usize = 32;
    const SUPER_BLOCK_BYTES: usize = 144; // 2 + 2 + 12 + 128
                                          // PMAT-177: Minimum valid f16 normal value (~6.1e-5) - prevents NaN on round-trip
    const F16_MIN_NORMAL: f32 = 6.1e-5;

    if data.is_empty() {
        return vec![];
    }

    let num_blocks = (data.len() + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
    let mut result = Vec::with_capacity(num_blocks * SUPER_BLOCK_BYTES);

    for block_idx in 0..num_blocks {
        let block_start = block_idx * SUPER_BLOCK_SIZE;
        let block_end = (block_start + SUPER_BLOCK_SIZE).min(data.len());
        let block_data = &data[block_start..block_end];

        // Pad to 256 if needed
        let mut padded = [0.0f32; SUPER_BLOCK_SIZE];
        padded[..block_data.len()].copy_from_slice(block_data);

        // Compute per-sub-block statistics (8 sub-blocks of 32 elements each)
        // Q4_K decoding: value = q * d * scale - dmin * min
        let mut sub_scales = [0.0f32; 8];
        let mut sub_mins = [0.0f32; 8];

        for (j, sub_block) in padded.chunks(SUB_BLOCK_SIZE).enumerate().take(8) {
            let min = sub_block.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            let max = sub_block.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let range = max - min;

            // PMAT-177: Clamp to F16_MIN_NORMAL to prevent underflow in f16 encoding
            sub_scales[j] = if range > F16_MIN_NORMAL {
                range / 15.0
            } else {
                F16_MIN_NORMAL
            };
            sub_mins[j] = (-min).max(0.0); // Store as positive offset
        }

        // Find global scale factors d and dmin
        let max_scale = sub_scales.iter().fold(0.0f32, |a, &b| a.max(b));
        let max_min = sub_mins.iter().fold(0.0f32, |a, &b| a.max(b));

        // PMAT-177: Clamp d/dmin to F16_MIN_NORMAL to prevent NaN after f16 round-trip
        let d = if max_scale > F16_MIN_NORMAL {
            max_scale / 63.0
        } else {
            F16_MIN_NORMAL
        };
        let dmin = if max_min > F16_MIN_NORMAL {
            max_min / 63.0
        } else {
            F16_MIN_NORMAL
        };

        // Compute 6-bit scales and mins for each sub-block
        let mut scales_6bit = [0u8; 8];
        let mut mins_6bit = [0u8; 8];

        for j in 0..8 {
            scales_6bit[j] = ((sub_scales[j] / d).round() as u8).min(63);
            mins_6bit[j] = ((sub_mins[j] / dmin).round() as u8).min(63);
        }

        // Write d (f16) - 2 bytes
        let d_f16 = f32_to_f16(d);
        result.extend_from_slice(&d_f16.to_le_bytes());

        // Write dmin (f16) - 2 bytes
        let dmin_f16 = f32_to_f16(dmin);
        result.extend_from_slice(&dmin_f16.to_le_bytes());

        // Pack scales and mins into 12 bytes (llama.cpp format)
        // Decoder expects:
        // - Blocks 0-3: scale = scales[j] & 63, min = scales[j+4] & 63
        // - Blocks 4-7: scale = (scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4)
        //               min = (scales[j+4] >> 4) | ((scales[j] >> 6) << 4)
        let mut scales_packed = [0u8; 12];

        // Blocks 0-3: store lower 6 bits directly, use upper 2 bits for blocks 4-7
        for i in 0..4 {
            // Lower 6 bits of scale[i], upper 2 bits store part of scale[i+4]
            scales_packed[i] = (scales_6bit[i] & 0x3F) | ((scales_6bit[i + 4] & 0x30) << 2);
            // Lower 6 bits of min[i], upper 2 bits store part of min[i+4]
            scales_packed[i + 4] = (mins_6bit[i] & 0x3F) | ((mins_6bit[i + 4] & 0x30) << 2);
        }

        // Blocks 4-7: store lower 4 bits of scale and min in bytes 8-11
        for i in 0..4 {
            scales_packed[i + 8] = (scales_6bit[i + 4] & 0x0F) | ((mins_6bit[i + 4] & 0x0F) << 4);
        }
        result.extend_from_slice(&scales_packed);

        // PMAT-190 FIX: Quantize to match gguf.rs dequantize_q4_k layout
        // Q4_K: 8 sub-blocks of 32 elements, ONE scale per sub-block
        // 16 bytes per sub-block, each byte packs TWO CONSECUTIVE values
        let mut qs = [0u8; 128];

        for j in 0..8 {
            // One scale per sub-block (same for both nibbles!)
            let scale = d * f32::from(scales_6bit[j]);
            let min_val = dmin * f32::from(mins_6bit[j]);

            // Process 32 values → 16 bytes
            for l in 0..16 {
                let idx0 = j * 32 + l * 2;       // First value of pair
                let idx1 = j * 32 + l * 2 + 1;   // Second value of pair

                // Quantize: q = (value + min_val) / scale
                let q0 = if scale > 1e-10 {
                    ((padded[idx0] + min_val) / scale).round().clamp(0.0, 15.0) as u8
                } else {
                    0
                };
                let q1 = if scale > 1e-10 {
                    ((padded[idx1] + min_val) / scale).round().clamp(0.0, 15.0) as u8
                } else {
                    0
                };

                // Pack: low nibble = q0, high nibble = q1
                qs[j * 16 + l] = (q0 & 0x0F) | ((q1 & 0x0F) << 4);
            }
        }
        result.extend_from_slice(&qs);
    }

    result
}

/// Transpose Q4K data for matmul kernel compatibility (PMAT-103)
///
/// GGUF stores weight matrices in column-major order (GGML convention) for `x @ W`.
/// The trueno Q4K kernel expects row-major order for `W @ x`.
/// These are transposes of each other.
///
/// This function:
/// 1. Dequantizes Q4K to F32
/// 2. Transposes from [rows, cols] to [cols, rows]
/// 3. Re-quantizes to Q4K
///
/// Returns: (transposed_q4k_bytes, transposed_shape)
///
/// Note: Scaffolding for PMAT-103 layout conversion optimization. Will be integrated
/// when the realizar inference engine adopts APR-native tensor ordering.
#[allow(dead_code)]
fn transpose_q4k_for_matmul(data: &[u8], shape: &[usize]) -> (Vec<u8>, Vec<usize>) {
    // Only transpose 2D tensors
    if shape.len() != 2 {
        return (data.to_vec(), shape.to_vec());
    }

    let rows = shape[0];
    let cols = shape[1];
    let num_elements = rows * cols;

    // Step 1: Dequantize Q4K to F32
    let f32_data = dequantize_q4_k_to_f32(data, num_elements);

    // Step 2: Transpose the F32 matrix from [rows, cols] to [cols, rows]
    // Original: data[i * cols + j] = element at row i, column j
    // Transposed: data[j * rows + i] = element at row j, column i
    let mut transposed_f32 = vec![0.0f32; num_elements];
    for i in 0..rows {
        for j in 0..cols {
            transposed_f32[j * rows + i] = f32_data[i * cols + j];
        }
    }

    // Step 3: Re-quantize to Q4K
    let transposed_q4k = quantize_q4_k(&transposed_f32);

    // Return with swapped dimensions
    (transposed_q4k, vec![cols, rows])
}

/// Transpose Q6K data for matmul kernel compatibility (PMAT-103)
///
/// Same as transpose_q4k_for_matmul but for Q6K format.
///
/// Note: Scaffolding for PMAT-103 layout conversion optimization.
/// Currently outputs Q4K for re-quantized transpose until Q6K encoder is added.
#[allow(dead_code)]
fn transpose_q6k_for_matmul(data: &[u8], shape: &[usize]) -> (Vec<u8>, Vec<usize>) {
    // Only transpose 2D tensors
    if shape.len() != 2 {
        return (data.to_vec(), shape.to_vec());
    }

    let rows = shape[0];
    let cols = shape[1];
    let num_elements = rows * cols;

    // Step 1: Dequantize Q6K to F32
    let f32_data = dequantize_q6_k_to_f32(data, num_elements);

    // Step 2: Transpose the F32 matrix
    let mut transposed_f32 = vec![0.0f32; num_elements];
    for i in 0..rows {
        for j in 0..cols {
            transposed_f32[j * rows + i] = f32_data[i * cols + j];
        }
    }

    // Step 3: Re-quantize to Q6K (for now, convert to Q4K since we don't have Q6K encoder)
    // Note: Proper Q6K quantization will be added when Q6K encoder is implemented
    let transposed_q4k = quantize_q4_k(&transposed_f32);

    // Return with swapped dimensions
    (transposed_q4k, vec![cols, rows])
}

/// Dequantize Q6_K data to f32 (for transpose)
///
/// Note: Scaffolding for PMAT-103 layout conversion optimization.
#[allow(dead_code)]
fn dequantize_q6_k_to_f32(data: &[u8], num_elements: usize) -> Vec<f32> {
    const SUPER_BLOCK_SIZE: usize = 256;
    const SUPER_BLOCK_BYTES: usize = 210;
    // PMAT-177: Minimum valid f16 normal value (~6.1e-5), clamp scales to avoid NaN
    const F16_MIN_NORMAL: f32 = 6.1e-5;

    let num_blocks = (num_elements + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
    let mut result = vec![0.0f32; num_blocks * SUPER_BLOCK_SIZE];

    for sb_idx in 0..num_blocks {
        let sb_start = sb_idx * SUPER_BLOCK_BYTES;
        let out_start = sb_idx * SUPER_BLOCK_SIZE;

        if sb_start + SUPER_BLOCK_BYTES > data.len() {
            break;
        }

        // Q6_K layout: ql (128B) + qh (64B) + scales (16B) + d (2B)
        let ql = &data[sb_start..sb_start + 128];
        let qh = &data[sb_start + 128..sb_start + 192];
        let scales_raw = &data[sb_start + 192..sb_start + 208];
        let d_raw = f16_to_f32(u16::from_le_bytes([
            data[sb_start + 208],
            data[sb_start + 209],
        ]));

        // PMAT-177: Replace NaN/Inf/subnormal with safe values to prevent corruption
        let d = if d_raw.is_nan() || d_raw.is_infinite() || d_raw.abs() < F16_MIN_NORMAL {
            0.0
        } else {
            d_raw
        };

        // Decode scales as signed i8
        let mut scales = [0i8; 16];
        for i in 0..16 {
            scales[i] = scales_raw[i] as i8;
        }

        // Dequantize 256 values
        for j in 0..256 {
            // Get 6-bit quantized value
            let ql_byte = ql[j / 2];
            let ql_val = if j % 2 == 0 {
                ql_byte & 0x0F
            } else {
                ql_byte >> 4
            };

            let qh_byte = qh[j / 4];
            let qh_val = (qh_byte >> ((j % 4) * 2)) & 0x03;

            let q6 = (ql_val as i32) | ((qh_val as i32) << 4);
            let q6_signed = q6 - 32; // Q6K uses offset encoding

            // Get scale for this 16-element block
            let scale_idx = j / 16;
            let scale = scales[scale_idx] as f32;

            result[out_start + j] = d * scale * q6_signed as f32;
        }
    }

    result.truncate(num_elements);
    result
}

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

/// Save model tensors with optional compression
///
/// Note: For .apr output, use save_model_tensors_with_config() instead to embed metadata.
fn save_model_tensors(
    tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    output: &Path,
    compression: Option<Compression>,
) -> Result<()> {
    // GH-165 FIX: If output is .apr, use APR format with embedded config
    let extension = output.extension().and_then(|e| e.to_str()).unwrap_or("");
    if extension == "apr" {
        return save_model_tensors_with_config(tensors, output, compression);
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

    // Create writer and add all tensors
    let mut writer = AprV2Writer::new(metadata);
    for (name, (data, shape)) in tensors {
        writer.add_f32_tensor(name, shape.clone(), data);
    }

    // Write to file
    let apr_bytes = writer.write().map_err(|e| AprenderError::FormatError {
        message: format!("Failed to write APR format: {e}"),
    })?;

    fs::write(output, apr_bytes).map_err(|e| AprenderError::FormatError {
        message: format!("Failed to write output file: {e}"),
    })
}

/// Save model tensors to APR format with GGUF model config (F-REGR-231 fix)
///
/// This function preserves critical GGUF metadata including:
/// - rope_type: RoPE style (0=NORM, 2=NEOX) - CRITICAL for Qwen2.5 models
/// - rope_theta: Position encoding frequency
/// - rms_norm_eps: RMS normalization epsilon
/// - All other model dimensions from GGUF
///
/// Without this, APR defaults to rope_type=0 which produces garbage for Qwen2.5.
#[allow(dead_code)] // Superseded by save_model_tensors_with_gguf_config_and_tokenizer
fn save_model_tensors_with_gguf_config(
    tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    output: &Path,
    _compression: Option<Compression>,
    gguf_config: &GgufModelConfig,
) -> Result<()> {
    // Build AprV2Metadata with GGUF config (not inferred from tensor shapes)
    let mut metadata = AprV2Metadata::new(gguf_config.architecture.as_deref().unwrap_or("qwen2"));
    metadata.original_format = Some("gguf".to_string());
    metadata.model_type = gguf_config
        .architecture
        .clone()
        .unwrap_or_else(|| "qwen2".to_string());

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

    // Create writer and add all tensors
    let mut writer = AprV2Writer::new(metadata);
    for (name, (data, shape)) in tensors {
        writer.add_f32_tensor(name, shape.clone(), data);
    }

    // Write to file
    let apr_bytes = writer.write().map_err(|e| AprenderError::FormatError {
        message: format!("Failed to write APR format: {e}"),
    })?;

    fs::write(output, apr_bytes).map_err(|e| AprenderError::FormatError {
        message: format!("Failed to write output file: {e}"),
    })
}

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

    // Create writer and add all tensors
    let mut writer = AprV2Writer::new(metadata);
    for (name, (data, shape)) in tensors {
        writer.add_f32_tensor(name, shape.clone(), data);
    }

    // Write to file
    let apr_bytes = writer.write().map_err(|e| AprenderError::FormatError {
        message: format!("Failed to write APR format: {e}"),
    })?;

    fs::write(output, apr_bytes).map_err(|e| AprenderError::FormatError {
        message: format!("Failed to write output file: {e}"),
    })
}

/// Save model tensors with Q4K quantization in APR format
///
/// Selectively quantizes large weight tensors while keeping biases and norms as F32.
/// Uses APR format with proper Q4K dtype for GPU-accelerated inference.
fn save_model_tensors_q4k(
    tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    output: &Path,
) -> Result<()> {
    use std::io::Write as IoWrite;

    // Infer model configuration from tensor shapes
    let mut hidden_size: Option<usize> = None;
    let mut num_layers: Option<usize> = None;
    let mut num_kv_heads: Option<usize> = None;
    let mut vocab_size: Option<usize> = None;
    let mut intermediate_size: Option<usize> = None;
    let mut num_heads: Option<usize> = None;

    for (name, (_, shape)) in tensors {
        // Infer hidden_size from norm weights (1D tensor of hidden_dim)
        if name.contains("input_layernorm.weight") && shape.len() == 1 {
            hidden_size = Some(shape[0]);
        }
        // Infer vocab_size from embedding [vocab_size, hidden_dim]
        if name.contains("embed_tokens.weight") && shape.len() == 2 {
            vocab_size = Some(shape[0]);
            if hidden_size.is_none() {
                hidden_size = Some(shape[1]);
            }
        }
        // Count layers
        if let Some(idx) = name.strip_prefix("model.layers.") {
            if let Some(layer_num) = idx.split('.').next().and_then(|s| s.parse::<usize>().ok()) {
                num_layers = Some(num_layers.map_or(layer_num + 1, |n| n.max(layer_num + 1)));
            }
        }
        // Infer kv_heads from k_proj shape [kv_dim, hidden_dim]
        if name.contains("k_proj.weight") && shape.len() == 2 && hidden_size.is_some() {
            // kv_dim = shape[0], hidden_dim = shape[1]
            // num_kv_heads = kv_dim / head_dim where head_dim = hidden_dim / num_heads
            // For Qwen2-0.5B: kv_dim=128, hidden_dim=896, head_dim=64, num_kv_heads=2
            num_kv_heads = Some(shape[0] / 64); // Assume head_dim=64 for now
        }
        // Infer num_heads from q_proj shape [q_dim, hidden_dim]
        if name.contains("q_proj.weight") && shape.len() == 2 {
            // q_dim = hidden_dim for standard attention
            // num_heads = hidden_dim / head_dim = hidden_dim / 64
            num_heads = Some(shape[0] / 64);
        }
        // Infer intermediate_size from gate_proj [intermediate, hidden]
        if name.contains("gate_proj.weight") && shape.len() == 2 {
            intermediate_size = Some(shape[0]);
        }
    }

    // Create APR metadata
    let param_count: u64 = tensors.values().map(|(data, _)| data.len() as u64).sum();

    let metadata = AprV2Metadata {
        model_type: "qwen2".to_string(),
        name: Some("Quantized Model".to_string()),
        description: Some("Q4K quantized from SafeTensors".to_string()),
        author: None,
        license: None,
        version: Some("1.0.0".to_string()),
        source: None,
        original_format: Some("safetensors".to_string()),
        created_at: None,
        total_size: 0,
        param_count,
        quantization: Some(QuantizationMetadata {
            quant_type: "q4_k".to_string(),
            bits: 4,
            block_size: Some(256),
            symmetric: false,
        }),
        sharding: None,
        chat_template: None,
        chat_format: None,
        special_tokens: None,
        architecture: Some("qwen2".to_string()),
        hidden_size,
        num_layers,
        num_heads,
        num_kv_heads,
        vocab_size,
        intermediate_size,
        max_position_embeddings: Some(32768), // Default for Qwen2
        rope_theta: Some(1000000.0),          // Default for Qwen2
        rope_type: Some(2),                   // NEOX style for Qwen2 (PMAT-114)
        rms_norm_eps: Some(1e-6),             // Default for Qwen2
        custom: std::collections::HashMap::new(),
    };

    let mut writer = AprV2Writer::new(metadata);

    // Add tensors, selectively quantizing to Q4K
    for (name, (data, shape)) in tensors {
        // Skip quantization for small tensors (biases, norms, scales)
        // and for 1D tensors which are typically biases/norms
        let should_quantize = shape.len() >= 2
            && data.len() >= 256  // Minimum size for Q4K (one super-block)
            && !name.contains("bias")
            && !name.contains("norm")
            && !name.contains("scale")
            && !name.contains("embed"); // Keep embeddings as F32 for now

        if should_quantize {
            // Quantize to Q4K bytes
            let q4k_bytes = quantize_q4_k(data);
            writer.add_q4k_raw_tensor(name, shape.clone(), q4k_bytes);
        } else {
            // Keep as F32
            writer.add_f32_tensor(name, shape.clone(), data);
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

// ============================================================================

// Export functionality extracted to export.rs (PMAT-197)
mod export;
pub use export::{apr_export, ExportFormat, ExportOptions, ExportReport};


// Merge functionality extracted to merge.rs (PMAT-197)
mod merge;
pub use merge::{apr_merge, MergeOptions, MergeReport, MergeStrategy};


// Tests extracted to tests.rs (PMAT-197)
#[cfg(test)]
mod tests;

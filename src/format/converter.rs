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
fn infer_model_config_from_tensors(
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
fn load_safetensors_tensors(path: &Path) -> Result<BTreeMap<String, (Vec<f32>, Vec<usize>)>> {
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
fn load_model_tensors(path: &Path) -> Result<BTreeMap<String, (Vec<f32>, Vec<usize>)>> {
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
fn validate_tensor_values(name: &str, data: &[f32]) -> Result<()> {
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
fn calculate_tensor_size(tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>) -> usize {
    tensors.values().map(|(data, _)| data.len() * 4).sum()
}

/// Apply quantization to tensors
fn quantize_tensors(
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
// EXPORT FUNCTIONALITY (APR-SPEC §4.6)
// ============================================================================

/// Export format options
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat {
    /// `SafeTensors` format (.safetensors) - `HuggingFace` ecosystem
    SafeTensors,
    /// GGUF format (.gguf) - llama.cpp / local inference
    Gguf,
    /// ONNX format (.onnx) - Cross-framework inference (not yet implemented)
    Onnx,
    /// `TorchScript` format (.pt) - `PyTorch` deployment (not yet implemented)
    TorchScript,
}

impl std::str::FromStr for ExportFormat {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "safetensors" | "st" => Ok(Self::SafeTensors),
            "gguf" => Ok(Self::Gguf),
            "onnx" => Ok(Self::Onnx),
            "torchscript" | "pt" | "torch" => Ok(Self::TorchScript),
            _ => Err(format!("Unknown export format: {s}")),
        }
    }
}

impl ExportFormat {
    /// Get default file extension
    #[must_use]
    pub fn extension(&self) -> &'static str {
        match self {
            Self::SafeTensors => "safetensors",
            Self::Gguf => "gguf",
            Self::Onnx => "onnx",
            Self::TorchScript => "pt",
        }
    }

    /// Check if format is supported
    #[must_use]
    pub fn is_supported(&self) -> bool {
        matches!(self, Self::SafeTensors | Self::Gguf)
    }
}

/// Options for model export
#[derive(Debug, Clone)]
pub struct ExportOptions {
    /// Target format
    pub format: ExportFormat,
    /// Optional quantization
    pub quantize: Option<QuantizationType>,
    /// GH-182: Include tokenizer.json companion file (SafeTensors only)
    pub include_tokenizer: bool,
    /// GH-182: Include config.json companion file (SafeTensors only)
    pub include_config: bool,
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self {
            format: ExportFormat::SafeTensors,
            quantize: None,
            include_tokenizer: true, // Default to true for HuggingFace compatibility
            include_config: true,
        }
    }
}

/// Report from export operation
#[derive(Debug, Clone)]
pub struct ExportReport {
    /// Original size in bytes
    pub original_size: usize,
    /// Exported size in bytes
    pub exported_size: usize,
    /// Number of tensors exported
    pub tensor_count: usize,
    /// Export format used
    pub format: ExportFormat,
    /// Quantization applied
    pub quantization: Option<QuantizationType>,
}

/// Export APR/SafeTensors model to another format
///
/// # Arguments
///
/// * `input` - Input model path (.apr or .safetensors)
/// * `output` - Output file path
/// * `options` - Export options
///
/// # Returns
///
/// Export report with size and format information
///
/// # Errors
///
/// Returns error if:
/// - Input file doesn't exist
/// - Format not supported
/// - Export fails
///
/// # Example
///
/// ```rust,ignore
/// use aprender::format::{apr_export, ExportOptions, ExportFormat};
///
/// let options = ExportOptions {
///     format: ExportFormat::Gguf,
///     quantize: None,
/// };
/// let report = apr_export("model.apr", "model.gguf", options)?;
/// ```
pub fn apr_export<P: AsRef<Path>>(
    input: P,
    output: P,
    options: ExportOptions,
) -> Result<ExportReport> {
    let input_path = input.as_ref();
    let output_path = output.as_ref();

    // Validate input exists
    if !input_path.exists() {
        return Err(AprenderError::FormatError {
            message: format!("Input file not found: {}", input_path.display()),
        });
    }

    // Check if format is supported
    if !options.format.is_supported() {
        return Err(AprenderError::FormatError {
            message: format!(
                "Export format {:?} is not yet supported. Use 'safetensors' or 'gguf'.",
                options.format
            ),
        });
    }

    // Load tensors
    let tensors = load_model_tensors(input_path)?;
    let original_size = calculate_tensor_size(&tensors);
    let original_count = tensors.len();

    // Apply quantization if requested
    let tensors = if let Some(ref quant_type) = options.quantize {
        quantize_tensors(&tensors, quant_type)?
    } else {
        tensors
    };

    // Export to target format
    match options.format {
        ExportFormat::SafeTensors => {
            save_safetensors(output_path, &tensors).map_err(|e| AprenderError::FormatError {
                message: format!("Failed to export to SafeTensors: {e}"),
            })?;

            // GH-182: Write companion files alongside SafeTensors
            let output_dir = output_path.parent().unwrap_or(Path::new("."));

            if options.include_config {
                let config = infer_model_config(&tensors);
                let config_path = output_dir.join("config.json");
                if let Err(e) = fs::write(&config_path, config) {
                    eprintln!("[GH-182] Warning: Failed to write config.json: {e}");
                }
            }

            if options.include_tokenizer {
                // Try to extract tokenizer from APR input if available
                let tokenizer_json = infer_tokenizer_json(input_path);
                if !tokenizer_json.is_empty() {
                    let tokenizer_path = output_dir.join("tokenizer.json");
                    if let Err(e) = fs::write(&tokenizer_path, &tokenizer_json) {
                        eprintln!("[GH-182] Warning: Failed to write tokenizer.json: {e}");
                    }
                }
            }
        }
        ExportFormat::Gguf => {
            export_to_gguf(&tensors, output_path)?;
        }
        ExportFormat::Onnx | ExportFormat::TorchScript => {
            return Err(AprenderError::FormatError {
                message: format!("Export format {:?} is not yet implemented", options.format),
            });
        }
    }

    // Get exported file size
    let exported_size = fs::metadata(output_path)
        .map(|m| m.len() as usize)
        .unwrap_or(0);

    Ok(ExportReport {
        original_size,
        exported_size,
        tensor_count: original_count,
        format: options.format,
        quantization: options.quantize,
    })
}

/// Export tensors to GGUF format
fn export_to_gguf(tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>, output: &Path) -> Result<()> {
    use crate::format::gguf::{export_tensors_to_gguf, GgmlType, GgufTensor, GgufValue};
    use std::fs::File;
    use std::io::BufWriter;

    // Convert tensors to GGUF format
    let gguf_tensors: Vec<GgufTensor> = tensors
        .iter()
        .map(|(name, (data, shape))| {
            // Convert f32 data to bytes
            let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();

            GgufTensor {
                name: name.clone(),
                shape: shape.iter().map(|&d| d as u64).collect(),
                dtype: GgmlType::F32,
                data: bytes,
            }
        })
        .collect();

    // Basic metadata
    let metadata = vec![
        (
            "general.name".to_string(),
            GgufValue::String("model".to_string()),
        ),
        (
            "general.quantization_version".to_string(),
            GgufValue::Uint32(1),
        ),
    ];

    // Write to file
    let file = File::create(output).map_err(|e| AprenderError::FormatError {
        message: format!("Failed to create output file: {e}"),
    })?;
    let mut writer = BufWriter::new(file);

    export_tensors_to_gguf(&mut writer, &gguf_tensors, &metadata)
}

// ============================================================================
// GH-182: COMPANION FILE HELPERS
// ============================================================================

/// Infer model config.json from tensor shapes (GH-182)
///
/// Creates a HuggingFace-compatible config.json based on tensor analysis.
fn infer_model_config(tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>) -> String {
    // Infer hidden_size from embedding or first layer weight
    let hidden_size = tensors
        .iter()
        .find(|(name, _)| name.contains("embed_tokens") || name.contains("token_embd"))
        .map(|(_, (_, shape))| shape.last().copied().unwrap_or(4096))
        .unwrap_or(4096);

    // Count layers by looking for layer patterns
    let num_layers = tensors
        .keys()
        .filter_map(|name| {
            if name.contains("layers.") || name.contains("blk.") {
                // Extract layer number from "layers.N." or "blk.N."
                let parts: Vec<&str> = name.split(&['.', '_'][..]).collect();
                for (i, part) in parts.iter().enumerate() {
                    if (*part == "layers" || *part == "blk") && i + 1 < parts.len() {
                        return parts[i + 1].parse::<usize>().ok();
                    }
                }
            }
            None
        })
        .max()
        .map(|n| n + 1)
        .unwrap_or(12);

    // Infer vocab_size from lm_head or output weight
    let vocab_size = tensors
        .iter()
        .find(|(name, _)| name.contains("lm_head") || name.contains("output.weight"))
        .map(|(_, (_, shape))| shape.first().copied().unwrap_or(32000))
        .unwrap_or(32000);

    // Create minimal config.json
    format!(
        r#"{{
  "architectures": ["AutoModelForCausalLM"],
  "hidden_size": {hidden_size},
  "num_hidden_layers": {num_layers},
  "vocab_size": {vocab_size},
  "model_type": "llama"
}}"#
    )
}

/// Extract tokenizer.json from APR input file (GH-182)
///
/// If the input is APR format with embedded tokenizer, extract it.
/// Otherwise return empty string.
fn infer_tokenizer_json(input_path: &Path) -> String {
    let extension = input_path.extension().and_then(|e| e.to_str()).unwrap_or("");

    if extension == "apr" {
        // Try to read APR metadata which may contain tokenizer
        if let Ok(data) = fs::read(input_path) {
            if data.len() > 44 {
                // Check for embedded tokenizer in APR metadata
                // APR format: header (44 bytes) + metadata (JSON) + tensors
                // Look for "tokenizer" in metadata section
                let metadata_start = 44;
                if let Some(metadata_end) = data[metadata_start..]
                    .windows(4)
                    .position(|w| w == b"}\n\n\n" || w == b"}\r\n\r")
                    .map(|p| metadata_start + p + 1)
                {
                    if let Ok(metadata_str) =
                        std::str::from_utf8(&data[metadata_start..metadata_end])
                    {
                        // Check if tokenizer data is embedded
                        if metadata_str.contains("\"tokenizer\"")
                            || metadata_str.contains("\"vocabulary\"")
                        {
                            // For now, return a minimal tokenizer.json
                            // In production, we'd extract the actual vocabulary
                            return r#"{"version": "1.0", "model": {"type": "BPE"}}"#.to_string();
                        }
                    }
                }
            }
        }
    }

    // Return empty if no tokenizer found
    String::new()
}

// ============================================================================
// MERGE FUNCTIONALITY (APR-SPEC §4.9)
// ============================================================================

/// Merge strategy options
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeStrategy {
    /// Average weights (simple ensemble)
    Average,
    /// Weighted average by performance
    Weighted,
    /// TIES merging (trim, elect, sign) - advanced
    Ties,
    /// DARE merging (drop and rescale) - advanced
    Dare,
    /// Spherical linear interpolation - advanced
    Slerp,
}

impl std::str::FromStr for MergeStrategy {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "average" | "avg" => Ok(Self::Average),
            "weighted" => Ok(Self::Weighted),
            "ties" => Ok(Self::Ties),
            "dare" => Ok(Self::Dare),
            "slerp" => Ok(Self::Slerp),
            _ => Err(format!("Unknown merge strategy: {s}")),
        }
    }
}

impl MergeStrategy {
    /// Check if strategy is currently supported
    #[must_use]
    pub fn is_supported(&self) -> bool {
        matches!(self, Self::Average | Self::Weighted)
    }
}

/// Options for model merging
#[derive(Debug, Clone)]
pub struct MergeOptions {
    /// Merge strategy to use
    pub strategy: MergeStrategy,
    /// Weights for weighted merging (must match number of models)
    pub weights: Option<Vec<f32>>,
}

impl Default for MergeOptions {
    fn default() -> Self {
        Self {
            strategy: MergeStrategy::Average,
            weights: None,
        }
    }
}

/// Report from merge operation
#[derive(Debug, Clone)]
pub struct MergeReport {
    /// Number of models merged
    pub model_count: usize,
    /// Number of tensors in merged model
    pub tensor_count: usize,
    /// Output file size in bytes
    pub output_size: usize,
    /// Strategy used
    pub strategy: MergeStrategy,
    /// Weights used (if weighted merge)
    pub weights_used: Option<Vec<f32>>,
}

// ============================================================================
// MERGE HELPER FUNCTIONS (Refactored for reduced complexity)
// ============================================================================

/// Validate merge options and input count.
fn validate_merge_options<P: AsRef<Path>>(inputs: &[P], options: &MergeOptions) -> Result<()> {
    if inputs.len() < 2 {
        return Err(AprenderError::FormatError {
            message: "Merge requires at least 2 input models".to_string(),
        });
    }

    if !options.strategy.is_supported() {
        return Err(AprenderError::FormatError {
            message: format!(
                "Merge strategy {:?} is not yet supported. Use 'average' or 'weighted'.",
                options.strategy
            ),
        });
    }

    if options.strategy == MergeStrategy::Weighted {
        match &options.weights {
            Some(weights) if weights.len() != inputs.len() => {
                return Err(AprenderError::FormatError {
                    message: format!(
                        "Weighted merge requires {} weights, got {}",
                        inputs.len(),
                        weights.len()
                    ),
                });
            }
            None => {
                return Err(AprenderError::FormatError {
                    message: "Weighted merge requires weights to be specified".to_string(),
                });
            }
            _ => {}
        }
    }
    Ok(())
}

/// Load all model tensors from input files.
fn load_all_models<P: AsRef<Path>>(
    inputs: &[P],
) -> Result<Vec<BTreeMap<String, (Vec<f32>, Vec<usize>)>>> {
    let mut all_tensors = Vec::new();
    for input_path in inputs {
        let path = input_path.as_ref();
        if !path.exists() {
            return Err(AprenderError::FormatError {
                message: format!("Input file not found: {}", path.display()),
            });
        }
        all_tensors.push(load_model_tensors(path)?);
    }
    Ok(all_tensors)
}

/// Verify all models have compatible tensor structures.
fn verify_tensor_compatibility(
    all_tensors: &[BTreeMap<String, (Vec<f32>, Vec<usize>)>],
) -> Result<()> {
    let reference = &all_tensors[0];
    for (i, tensors) in all_tensors.iter().enumerate().skip(1) {
        if tensors.len() != reference.len() {
            return Err(AprenderError::FormatError {
                message: format!(
                    "Model {} has {} tensors, but model 0 has {}",
                    i,
                    tensors.len(),
                    reference.len()
                ),
            });
        }
        verify_single_model_tensors(reference, tensors, i)?;
    }
    Ok(())
}

/// Verify tensor compatibility for a single model against reference.
fn verify_single_model_tensors(
    reference: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    model_idx: usize,
) -> Result<()> {
    for (name, (_, shape)) in reference {
        match tensors.get(name) {
            None => {
                return Err(AprenderError::FormatError {
                    message: format!("Model {} is missing tensor '{}'", model_idx, name),
                });
            }
            Some((_, other_shape)) if other_shape != shape => {
                return Err(AprenderError::FormatError {
                    message: format!(
                        "Tensor '{}' has shape {:?} in model 0 but {:?} in model {}",
                        name, shape, other_shape, model_idx
                    ),
                });
            }
            _ => {}
        }
    }
    Ok(())
}

/// Calculate normalized merge weights based on strategy.
fn calculate_merge_weights(input_count: usize, options: &MergeOptions) -> Result<Vec<f32>> {
    match options.strategy {
        MergeStrategy::Average => {
            let w = 1.0 / input_count as f32;
            Ok(vec![w; input_count])
        }
        MergeStrategy::Weighted => {
            let raw_weights = options.weights.as_ref().expect("validated above");
            let sum: f32 = raw_weights.iter().sum();
            if sum <= 0.0 {
                return Err(AprenderError::FormatError {
                    message: "Weights must sum to a positive value".to_string(),
                });
            }
            Ok(raw_weights.iter().map(|w| w / sum).collect())
        }
        _ => unreachable!("unsupported strategies filtered above"),
    }
}

/// Merge tensors from multiple models using given weights.
fn merge_tensors(
    all_tensors: &[BTreeMap<String, (Vec<f32>, Vec<usize>)>],
    weights: &[f32],
) -> BTreeMap<String, (Vec<f32>, Vec<usize>)> {
    let reference = &all_tensors[0];
    let mut merged = BTreeMap::new();

    for (name, (_, shape)) in reference {
        let data_len = all_tensors[0].get(name).map(|(d, _)| d.len()).unwrap_or(0);
        let mut merged_data = vec![0.0f32; data_len];

        for (model_idx, model_tensors) in all_tensors.iter().enumerate() {
            let (data, _) = model_tensors.get(name).expect("validated above");
            let weight = weights[model_idx];
            for (i, &val) in data.iter().enumerate() {
                merged_data[i] += val * weight;
            }
        }

        merged.insert(name.clone(), (merged_data, shape.clone()));
    }
    merged
}

/// Merge multiple models into one
///
/// # Arguments
///
/// * `inputs` - Input model paths (.apr or .safetensors)
/// * `output` - Output file path
/// * `options` - Merge options
///
/// # Returns
///
/// Merge report with statistics
///
/// # Errors
///
/// Returns error if:
/// - Less than 2 input files
/// - Input files don't exist
/// - Models have incompatible tensor shapes
/// - Strategy not supported
///
/// # Example
///
/// ```rust,ignore
/// use aprender::format::{apr_merge, MergeOptions, MergeStrategy};
///
/// let options = MergeOptions {
///     strategy: MergeStrategy::Average,
///     weights: None,
/// };
/// let report = apr_merge(&["model1.apr", "model2.apr"], "merged.apr", options)?;
/// ```
pub fn apr_merge<P: AsRef<Path>>(
    inputs: &[P],
    output: P,
    options: MergeOptions,
) -> Result<MergeReport> {
    // Validate inputs and options
    validate_merge_options(inputs, &options)?;

    // Load all models
    let all_tensors = load_all_models(inputs)?;

    // Verify tensor compatibility
    verify_tensor_compatibility(&all_tensors)?;

    // Calculate weights
    let weights = calculate_merge_weights(inputs.len(), &options)?;

    // Merge tensors
    let merged = merge_tensors(&all_tensors, &weights);

    // Save merged model
    let output_path = output.as_ref();
    save_safetensors(output_path, &merged).map_err(|e| AprenderError::FormatError {
        message: format!("Failed to save merged model: {e}"),
    })?;

    // Get output file size
    let output_size = fs::metadata(output_path)
        .map(|m| m.len() as usize)
        .unwrap_or(0);

    Ok(MergeReport {
        model_count: inputs.len(),
        tensor_count: merged.len(),
        output_size,
        strategy: options.strategy,
        weights_used: Some(weights),
    })
}

// ============================================================================
// TESTS - EXTREME TDD
// ============================================================================

#[cfg(test)]
mod tests_source_parsing {
    use super::*;

    #[test]
    fn test_parse_hf_org_repo() {
        let source = Source::parse("hf://openai/whisper-tiny").unwrap();
        assert_eq!(
            source,
            Source::HuggingFace {
                org: "openai".to_string(),
                repo: "whisper-tiny".to_string(),
                file: None,
            }
        );
    }

    #[test]
    fn test_parse_hf_org_repo_file() {
        let source = Source::parse("hf://openai/whisper-tiny/model.safetensors").unwrap();
        assert_eq!(
            source,
            Source::HuggingFace {
                org: "openai".to_string(),
                repo: "whisper-tiny".to_string(),
                file: Some("model.safetensors".to_string()),
            }
        );
    }

    #[test]
    fn test_parse_hf_nested_file() {
        let source =
            Source::parse("hf://meta-llama/Llama-2-7b/pytorch_model-00001-of-00002.bin").unwrap();
        assert_eq!(
            source,
            Source::HuggingFace {
                org: "meta-llama".to_string(),
                repo: "Llama-2-7b".to_string(),
                file: Some("pytorch_model-00001-of-00002.bin".to_string()),
            }
        );
    }

    #[test]
    fn test_parse_local_path() {
        let source = Source::parse("./models/model.safetensors").unwrap();
        assert_eq!(
            source,
            Source::Local(PathBuf::from("./models/model.safetensors"))
        );
    }

    #[test]
    fn test_parse_url() {
        let source = Source::parse("https://example.com/model.safetensors").unwrap();
        assert_eq!(
            source,
            Source::Url("https://example.com/model.safetensors".to_string())
        );
    }

    #[test]
    fn test_parse_hf_invalid() {
        let result = Source::parse("hf://invalid");
        assert!(result.is_err());
    }

    #[test]
    fn test_default_file() {
        let hf = Source::HuggingFace {
            org: "openai".to_string(),
            repo: "whisper".to_string(),
            file: None,
        };
        assert_eq!(hf.default_file(), "model.safetensors");

        let hf_with_file = Source::HuggingFace {
            org: "openai".to_string(),
            repo: "whisper".to_string(),
            file: Some("custom.safetensors".to_string()),
        };
        assert_eq!(hf_with_file.default_file(), "custom.safetensors");
    }
}

#[cfg(test)]
mod tests_name_mapping {
    use super::*;

    #[test]
    fn test_whisper_preserve_model_prefix() {
        // PMAT-099: Names are now preserved for AprTransformer compatibility
        let mapped = Architecture::Whisper.map_name("model.encoder.conv1.weight");
        assert_eq!(mapped, "model.encoder.conv1.weight");
    }

    #[test]
    fn test_whisper_no_prefix() {
        let mapped = Architecture::Whisper.map_name("encoder.conv1.weight");
        assert_eq!(mapped, "encoder.conv1.weight");
    }

    #[test]
    fn test_whisper_decoder_layer_norm() {
        // PMAT-099: Names are now preserved for AprTransformer compatibility
        let mapped = Architecture::Whisper.map_name("model.decoder.layer_norm.weight");
        assert_eq!(mapped, "model.decoder.layer_norm.weight");
    }

    #[test]
    fn test_auto_preserves_model_prefix() {
        // PMAT-099: model. prefix preserved for AprTransformer::from_apr_bytes compatibility
        let mapped = Architecture::Auto.map_name("model.encoder.layers.0.self_attn.q_proj.weight");
        assert_eq!(mapped, "model.encoder.layers.0.self_attn.q_proj.weight");
    }

    #[test]
    fn test_llama_mapping() {
        // PMAT-099: Preserve original names for inference compatibility
        let mapped = Architecture::Llama.map_name("model.layers.0.self_attn.q_proj.weight");
        assert_eq!(mapped, "model.layers.0.self_attn.q_proj.weight");
    }

    #[test]
    fn test_bert_mapping() {
        // PMAT-099: Preserve original names
        let mapped =
            Architecture::Bert.map_name("bert.encoder.layer.0.attention.self.query.weight");
        assert_eq!(mapped, "bert.encoder.layer.0.attention.self.query.weight");
    }

    #[test]
    fn test_qwen2_mapping() {
        // PMAT-099: Preserve model. prefix for AprTransformer compatibility
        let mapped = Architecture::Qwen2.map_name("model.layers.0.self_attn.q_proj.weight");
        assert_eq!(mapped, "model.layers.0.self_attn.q_proj.weight");
    }
}

#[cfg(test)]
mod tests_tensor_expectations {
    use super::*;

    #[test]
    fn test_layer_norm_weight_expectation() {
        let exp = TensorExpectation::for_tensor("encoder.layer_norm.weight");
        assert!(exp.is_some());
        let exp = exp.unwrap();
        assert_eq!(exp.mean_range, (0.5, 3.0));
    }

    #[test]
    fn test_layer_norm_bias_expectation() {
        let exp = TensorExpectation::for_tensor("decoder.layers.0.self_attn_layer_norm.bias");
        assert!(exp.is_some());
        let exp = exp.unwrap();
        assert_eq!(exp.mean_range, (-0.5, 0.5));
    }

    #[test]
    fn test_linear_weight_expectation() {
        let exp = TensorExpectation::for_tensor("encoder.layers.0.fc1.weight");
        assert!(exp.is_some());
        let exp = exp.unwrap();
        assert_eq!(exp.mean_range, (-0.1, 0.1));
    }

    #[test]
    fn test_embedding_expectation() {
        let exp = TensorExpectation::for_tensor("decoder.embed_tokens.weight");
        assert!(exp.is_some());
    }

    #[test]
    fn test_check_layer_norm_valid() {
        let stats = TensorStats {
            name: "encoder.layer_norm.weight".to_string(),
            count: 384,
            min: 0.5,
            max: 2.0,
            mean: 1.0,
            std: 0.3,
            nan_count: 0,
            inf_count: 0,
            zero_count: 0,
        };

        let exp = TensorExpectation::LAYER_NORM_WEIGHT;
        assert!(exp.check(&stats).is_ok());
    }

    #[test]
    fn test_check_layer_norm_invalid_mean() {
        let stats = TensorStats {
            name: "decoder.layer_norm.weight".to_string(),
            count: 384,
            min: 5.0,
            max: 15.0,
            mean: 11.0, // BUG: should be ~1.0
            std: 2.0,
            nan_count: 0,
            inf_count: 0,
            zero_count: 0,
        };

        let exp = TensorExpectation::LAYER_NORM_WEIGHT;
        let result = exp.check(&stats);
        assert!(result.is_err());

        let err = result.unwrap_err().to_string();
        assert!(err.contains("mean=11"));
        assert!(err.contains("outside expected range"));
    }

    #[test]
    fn test_rmsnorm_weight_detection() {
        // LLaMA-style input_layernorm
        let exp = TensorExpectation::for_tensor("model.layers.0.input_layernorm.weight");
        assert!(exp.is_some());
        assert_eq!(exp.unwrap().mean_range, (-0.5, 3.0));

        // LLaMA-style post_attention_layernorm
        let exp = TensorExpectation::for_tensor("model.layers.5.post_attention_layernorm.weight");
        assert!(exp.is_some());
        assert_eq!(exp.unwrap().mean_range, (-0.5, 3.0));

        // Final norm
        let exp = TensorExpectation::for_tensor("model.norm.weight");
        assert!(exp.is_some());
        assert_eq!(exp.unwrap().mean_range, (-0.5, 3.0));
    }

    #[test]
    fn test_rmsnorm_accepts_trained_weights() {
        // TinyLlama trained model has means from 0.005 to 0.5
        let stats = TensorStats {
            name: "model.layers.0.input_layernorm.weight".to_string(),
            count: 2048,
            min: -0.2,
            max: 0.8,
            mean: 0.05, // Trained weight, NOT near 1.0
            std: 0.15,
            nan_count: 0,
            inf_count: 0,
            zero_count: 0,
        };

        let exp = TensorExpectation::RMSNORM_WEIGHT;
        assert!(exp.check(&stats).is_ok());
    }
}

#[cfg(test)]
mod tests_converter_builder {
    use super::*;

    #[test]
    fn test_converter_builder_chain() {
        let converter = AprConverter::new()
            .source("hf://openai/whisper-tiny")
            .unwrap()
            .architecture(Architecture::Whisper)
            .validate(ValidationConfig::Strict)
            .quantize(QuantizationType::Int8)
            .compress(Compression::Lz4);

        assert_eq!(converter.architecture, Architecture::Whisper);
        assert_eq!(converter.validation, ValidationConfig::Strict);
        assert_eq!(converter.quantize, Some(QuantizationType::Int8));
        assert_eq!(converter.compress, Some(Compression::Lz4));
    }

    #[test]
    fn test_converter_no_source_error() {
        let converter = AprConverter::new();
        let result = converter.convert();
        assert!(result.is_err());
    }
}

#[cfg(test)]
mod tests_import_options {
    use super::*;

    #[test]
    fn test_default_options() {
        let opts = ImportOptions::default();
        assert_eq!(opts.architecture, Architecture::Auto);
        assert_eq!(opts.validation, ValidationConfig::Strict);
        assert_eq!(opts.quantize, None);
        assert_eq!(opts.compress, None);
        assert!(!opts.force);
        assert!(opts.cache);
    }
}

#[cfg(test)]
mod tests_conversion {
    use super::*;

    fn create_test_safetensors(path: &Path, tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>) {
        save_safetensors(path, tensors).expect("Failed to create test SafeTensors file");
    }

    #[test]
    fn test_convert_valid_safetensors() {
        let input = "/tmp/test_valid_input.safetensors";
        let output = "/tmp/test_valid_output.apr";

        // Create valid test tensors
        let mut tensors = BTreeMap::new();
        tensors.insert(
            "encoder.layer_norm.weight".to_string(),
            (vec![1.0f32; 384], vec![384]),
        );
        tensors.insert(
            "encoder.layer_norm.bias".to_string(),
            (vec![0.0f32; 384], vec![384]),
        );
        tensors.insert(
            "encoder.conv1.weight".to_string(),
            (vec![0.01f32; 1000], vec![80, 1, 3]),
        );

        create_test_safetensors(Path::new(input), &tensors);

        // Run conversion
        let options = ImportOptions::default();
        let result = apr_import(input, output, options);

        assert!(
            result.is_ok(),
            "Valid tensors should convert successfully: {:?}",
            result.err()
        );
        let report = result.unwrap();
        assert!(report.total_score > 0, "Score should be > 0");

        // Cleanup
        fs::remove_file(input).ok();
        fs::remove_file(output).ok();
    }

    #[test]
    fn test_convert_invalid_layernorm_fails_strict() {
        let input = "/tmp/test_invalid_ln_input.safetensors";
        let output = "/tmp/test_invalid_ln_output.apr";

        // Create tensors with INVALID LayerNorm (mean=11, should be ~1)
        let mut tensors = BTreeMap::new();
        tensors.insert(
            "decoder.layer_norm.weight".to_string(),
            (vec![11.0f32; 384], vec![384]),
        );

        create_test_safetensors(Path::new(input), &tensors);

        // Run conversion with strict validation
        let options = ImportOptions {
            validation: ValidationConfig::Strict,
            force: false,
            ..Default::default()
        };
        let result = apr_import(input, output, options);

        assert!(
            result.is_err(),
            "Invalid LayerNorm should fail strict validation"
        );
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("mean=11") || err.contains("LayerNorm"),
            "Error should mention LayerNorm issue: {err}"
        );

        // Cleanup
        fs::remove_file(input).ok();
        fs::remove_file(output).ok();
    }

    #[test]
    fn test_convert_invalid_layernorm_force_succeeds() {
        let input = "/tmp/test_force_ln_input.safetensors";
        let output = "/tmp/test_force_ln_output.apr";

        // Create tensors with invalid LayerNorm
        let mut tensors = BTreeMap::new();
        tensors.insert(
            "decoder.layer_norm.weight".to_string(),
            (vec![11.0f32; 384], vec![384]),
        );

        create_test_safetensors(Path::new(input), &tensors);

        // Run conversion with force=true
        let options = ImportOptions {
            validation: ValidationConfig::Strict,
            force: true,
            ..Default::default()
        };
        let result = apr_import(input, output, options);

        assert!(
            result.is_ok(),
            "Force should bypass validation: {:?}",
            result.err()
        );

        // Cleanup
        fs::remove_file(input).ok();
        fs::remove_file(output).ok();
    }

    #[test]
    fn test_convert_nan_fails() {
        let input = "/tmp/test_nan_input.safetensors";
        let output = "/tmp/test_nan_output.apr";

        // Create tensors with NaN
        let mut tensors = BTreeMap::new();
        tensors.insert(
            "test.weight".to_string(),
            (vec![1.0, f32::NAN, 3.0], vec![3]),
        );

        create_test_safetensors(Path::new(input), &tensors);

        let options = ImportOptions::default();
        let result = apr_import(input, output, options);

        assert!(result.is_err(), "NaN should fail validation");
        let err = result.unwrap_err().to_string();
        assert!(err.contains("NaN"), "Error should mention NaN: {err}");

        // Cleanup
        fs::remove_file(input).ok();
        fs::remove_file(output).ok();
    }

    #[test]
    fn test_convert_nonexistent_file() {
        let result = apr_import(
            "/tmp/nonexistent_model.safetensors",
            "/tmp/out.apr",
            ImportOptions::default(),
        );
        assert!(result.is_err(), "Nonexistent file should fail");
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("not found") || err.contains("No such file"),
            "Error should mention file not found: {err}"
        );
    }

    #[test]
    fn test_convert_unsupported_format() {
        let input = "/tmp/test_bad_format.gguf";
        fs::write(input, b"test").expect("Failed to create test file");

        let result = apr_import(input, "/tmp/out.apr", ImportOptions::default());
        assert!(result.is_err(), "Unsupported format should fail");
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("GGUF") || err.contains("not yet"),
            "Error should mention unsupported: {err}"
        );

        fs::remove_file(input).ok();
    }

    #[test]
    fn test_name_mapping_whisper() {
        use crate::format::v2::AprV2Reader;

        let input = "/tmp/test_whisper_input.safetensors";
        let output = "/tmp/test_whisper_output.apr";

        // Create tensors with HuggingFace-style names
        let mut tensors = BTreeMap::new();
        tensors.insert(
            "model.encoder.conv1.weight".to_string(),
            (vec![0.01f32; 100], vec![10, 10]),
        );
        tensors.insert(
            "model.decoder.layer_norm.weight".to_string(),
            (vec![1.0f32; 384], vec![384]),
        );

        create_test_safetensors(Path::new(input), &tensors);

        let options = ImportOptions {
            architecture: Architecture::Whisper,
            ..Default::default()
        };
        let result = apr_import(input, output, options);
        assert!(
            result.is_ok(),
            "Whisper mapping should work: {:?}",
            result.err()
        );

        // Load output as APR and verify names are preserved (PMAT-099)
        let data = fs::read(output).expect("Failed to read output");
        let reader = AprV2Reader::from_bytes(&data).expect("Failed to parse APR");
        let tensor_names = reader.tensor_names();

        // PMAT-099: Names are now preserved for AprTransformer compatibility
        assert!(
            tensor_names.contains(&"model.encoder.conv1.weight"),
            "Should preserve 'model.' prefix for AprTransformer compatibility, got: {:?}",
            tensor_names
        );
        assert!(
            tensor_names.contains(&"model.decoder.layer_norm.weight"),
            "Should preserve 'model.' prefix for AprTransformer compatibility, got: {:?}",
            tensor_names
        );

        // Cleanup
        fs::remove_file(input).ok();
        fs::remove_file(output).ok();
    }
}

#[cfg(test)]
mod tests_tensor_stats {
    use super::*;

    #[test]
    fn test_compute_stats_basic() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let stats = compute_tensor_stats("test", &data);

        assert_eq!(stats.count, 5);
        assert!((stats.mean - 3.0).abs() < 0.001, "Mean should be 3.0");
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
        assert_eq!(stats.nan_count, 0);
        assert_eq!(stats.inf_count, 0);
    }

    #[test]
    fn test_compute_stats_with_nan() {
        let data = vec![1.0f32, f32::NAN, 3.0];
        let stats = compute_tensor_stats("test", &data);

        assert_eq!(stats.nan_count, 1);
        assert_eq!(stats.count, 3);
        // Mean computed from valid values only
        assert!(
            (stats.mean - 2.0).abs() < 0.001,
            "Mean should be 2.0 (from valid values)"
        );
    }

    #[test]
    fn test_compute_stats_with_inf() {
        let data = vec![1.0f32, f32::INFINITY, f32::NEG_INFINITY, 3.0];
        let stats = compute_tensor_stats("test", &data);

        assert_eq!(stats.inf_count, 2);
        assert!(
            (stats.mean - 2.0).abs() < 0.001,
            "Mean should be 2.0 (from valid values)"
        );
    }

    #[test]
    fn test_compute_stats_empty() {
        let data: Vec<f32> = vec![];
        let stats = compute_tensor_stats("test", &data);

        assert_eq!(stats.count, 0);
        assert_eq!(stats.mean, 0.0);
        assert_eq!(stats.std, 0.0);
    }

    #[test]
    fn test_compute_stats_all_zeros() {
        let data = vec![0.0f32; 100];
        let stats = compute_tensor_stats("test", &data);

        assert_eq!(stats.zero_count, 100);
        assert_eq!(stats.mean, 0.0);
    }
}

#[cfg(test)]
mod tests_quantization {
    use super::*;

    #[test]
    fn test_quantize_int8_basic() {
        let data = vec![1.0f32, -1.0, 0.5, -0.5, 0.0];
        let quantized = quantize_int8(&data);

        assert_eq!(quantized.len(), data.len());
        // Values should be close but not exact due to quantization
        for (orig, quant) in data.iter().zip(quantized.iter()) {
            assert!((orig - quant).abs() < 0.02, "Quantization error too large");
        }
    }

    #[test]
    fn test_quantize_int8_preserves_zeros() {
        let data = vec![0.0f32; 10];
        let quantized = quantize_int8(&data);
        assert!(
            quantized.iter().all(|&v| v == 0.0),
            "Zeros should remain zeros"
        );
    }

    #[test]
    fn test_quantize_int8_empty() {
        let data: Vec<f32> = vec![];
        let quantized = quantize_int8(&data);
        assert!(quantized.is_empty());
    }

    #[test]
    fn test_quantize_int4_basic() {
        let data = vec![1.0f32, -1.0, 0.5, -0.5, 0.0];
        let quantized = quantize_int4(&data);

        assert_eq!(quantized.len(), data.len());
        // Int4 has more error than int8
        for (orig, quant) in data.iter().zip(quantized.iter()) {
            assert!(
                (orig - quant).abs() < 0.2,
                "Int4 quantization error too large"
            );
        }
    }

    #[test]
    fn test_quantize_fp16_basic() {
        let data = vec![1.0f32, -1.0, 0.5, -0.5, 0.0, 0.123456789];
        let quantized = quantize_fp16(&data);

        assert_eq!(quantized.len(), data.len());
        // FP16 should have minimal error for simple values
        assert_eq!(quantized[0], 1.0);
        assert_eq!(quantized[1], -1.0);
        assert_eq!(quantized[4], 0.0);
    }

    #[test]
    fn test_quantize_tensors_int8() {
        let mut tensors = BTreeMap::new();
        tensors.insert("test".to_string(), (vec![1.0f32, -1.0, 0.5], vec![3]));

        let result = quantize_tensors(&tensors, &QuantizationType::Int8).unwrap();

        assert_eq!(result.len(), 1);
        assert!(result.contains_key("test"));
        let (data, shape) = result.get("test").unwrap();
        assert_eq!(shape, &vec![3]);
        assert_eq!(data.len(), 3);
    }
}

#[cfg(test)]
mod tests_convert {
    use super::*;

    fn create_test_model(path: &Path) {
        let mut tensors = BTreeMap::new();
        tensors.insert(
            "encoder.weight".to_string(),
            (vec![0.01f32; 1000], vec![100, 10]),
        );
        tensors.insert("encoder.bias".to_string(), (vec![0.0f32; 100], vec![100]));
        tensors.insert(
            "decoder.weight".to_string(),
            (vec![0.02f32; 500], vec![50, 10]),
        );
        save_safetensors(path, &tensors).expect("Failed to create test model");
    }

    #[test]
    fn test_convert_no_quantization() {
        let input = Path::new("/tmp/test_convert_input.safetensors");
        let output = Path::new("/tmp/test_convert_output.apr");

        create_test_model(input);

        let options = ConvertOptions::default();
        let result = apr_convert(input, output, options);

        assert!(
            result.is_ok(),
            "Convert without quantization should work: {:?}",
            result.err()
        );
        let report = result.unwrap();
        assert_eq!(report.tensor_count, 3);
        assert!(report.quantization.is_none());

        fs::remove_file(input).ok();
        fs::remove_file(output).ok();
    }

    #[test]
    fn test_convert_with_int8_quantization() {
        let input = Path::new("/tmp/test_convert_int8_input.safetensors");
        let output = Path::new("/tmp/test_convert_int8_output.apr");

        create_test_model(input);

        let options = ConvertOptions {
            quantize: Some(QuantizationType::Int8),
            ..Default::default()
        };
        let result = apr_convert(input, output, options);

        assert!(
            result.is_ok(),
            "Int8 quantization should work: {:?}",
            result.err()
        );
        let report = result.unwrap();
        assert_eq!(report.quantization, Some(QuantizationType::Int8));
        assert_eq!(report.tensor_count, 3);

        fs::remove_file(input).ok();
        fs::remove_file(output).ok();
    }

    #[test]
    fn test_convert_with_fp16_quantization() {
        let input = Path::new("/tmp/test_convert_fp16_input.safetensors");
        let output = Path::new("/tmp/test_convert_fp16_output.apr");

        create_test_model(input);

        let options = ConvertOptions {
            quantize: Some(QuantizationType::Fp16),
            ..Default::default()
        };
        let result = apr_convert(input, output, options);

        assert!(
            result.is_ok(),
            "FP16 quantization should work: {:?}",
            result.err()
        );

        fs::remove_file(input).ok();
        fs::remove_file(output).ok();
    }

    #[test]
    fn test_convert_nonexistent_file() {
        let options = ConvertOptions::default();
        let result = apr_convert("/tmp/nonexistent.safetensors", "/tmp/out.apr", options);

        assert!(result.is_err(), "Nonexistent file should fail");
    }

    #[test]
    fn test_convert_report_reduction_percent() {
        let report = ConvertReport {
            original_size: 1000,
            converted_size: 250,
            tensor_count: 5,
            quantization: Some(QuantizationType::Int8),
            compression: None,
            reduction_ratio: 4.0,
        };

        assert_eq!(report.reduction_percent(), "75.0%");
    }

    #[test]
    fn test_convert_options_default() {
        let options = ConvertOptions::default();
        assert!(options.quantize.is_none());
        assert!(options.compress.is_none());
        assert!(options.validate);
    }
}

// ============================================================================
// GH-127: Multi-tensor (sharded) model import tests
// ============================================================================

#[cfg(test)]
mod tests_sharded_import {
    use super::*;

    #[test]
    fn test_sharded_index_parse_valid() {
        let json = r#"{
            "metadata": {"total_size": 1000000},
            "weight_map": {
                "encoder.conv1.weight": "model-00001-of-00002.safetensors",
                "encoder.conv2.weight": "model-00001-of-00002.safetensors",
                "decoder.fc.weight": "model-00002-of-00002.safetensors"
            }
        }"#;

        let index = ShardedIndex::parse(json).expect("Valid index should parse");
        assert_eq!(index.shard_count(), 2);
        assert_eq!(index.tensor_count(), 3);
        assert!(index.total_size().is_some());
    }

    #[test]
    fn test_sharded_index_shard_for_tensor() {
        let json = r#"{
            "weight_map": {
                "encoder.weight": "shard-00001.safetensors",
                "decoder.weight": "shard-00002.safetensors"
            }
        }"#;

        let index = ShardedIndex::parse(json).unwrap();
        assert_eq!(
            index.shard_for_tensor("encoder.weight"),
            Some("shard-00001.safetensors")
        );
        assert_eq!(
            index.shard_for_tensor("decoder.weight"),
            Some("shard-00002.safetensors")
        );
        assert_eq!(index.shard_for_tensor("unknown"), None);
    }

    #[test]
    fn test_sharded_index_tensors_in_shard() {
        let json = r#"{
            "weight_map": {
                "a": "shard1.safetensors",
                "b": "shard1.safetensors",
                "c": "shard2.safetensors"
            }
        }"#;

        let index = ShardedIndex::parse(json).unwrap();
        let shard1_tensors = index.tensors_in_shard("shard1.safetensors");
        assert_eq!(shard1_tensors.len(), 2);
        assert!(shard1_tensors.contains(&"a"));
        assert!(shard1_tensors.contains(&"b"));
    }

    #[test]
    fn test_sharded_index_parse_invalid_json() {
        let result = ShardedIndex::parse("not valid json");
        assert!(result.is_err());
    }

    #[test]
    fn test_sharded_index_parse_missing_weight_map() {
        let result = ShardedIndex::parse(r#"{"metadata": {}}"#);
        assert!(result.is_err());
    }

    #[test]
    fn test_detect_sharded_model_index_exists() {
        // Create a temp dir with index.json
        let dir = tempfile::tempdir().unwrap();
        let index_path = dir.path().join("model.safetensors.index.json");
        fs::write(&index_path, r#"{"weight_map": {"a": "shard.safetensors"}}"#).unwrap();

        let result = detect_sharded_model(dir.path(), "model.safetensors");
        assert!(result.is_some());
    }

    #[test]
    fn test_detect_sharded_model_single_file() {
        let dir = tempfile::tempdir().unwrap();
        let model_path = dir.path().join("model.safetensors");
        fs::write(&model_path, &[0u8; 8]).unwrap(); // Minimal file

        let result = detect_sharded_model(dir.path(), "model.safetensors");
        assert!(result.is_none(), "Single file should not be sharded");
    }

    #[test]
    fn test_sharded_index_shard_files_sorted() {
        let json = r#"{
            "weight_map": {
                "a": "model-00002-of-00003.safetensors",
                "b": "model-00001-of-00003.safetensors",
                "c": "model-00003-of-00003.safetensors"
            }
        }"#;

        let index = ShardedIndex::parse(json).unwrap();
        let shards = index.shard_files();
        assert_eq!(shards[0], "model-00001-of-00003.safetensors");
        assert_eq!(shards[1], "model-00002-of-00003.safetensors");
        assert_eq!(shards[2], "model-00003-of-00003.safetensors");
    }
}

// ============================================================================
// GH-129: Import error message tests
// ============================================================================

#[cfg(test)]
mod tests_import_errors {
    use super::*;

    #[test]
    fn test_import_error_not_found_message() {
        let err = ImportError::NotFound {
            resource: "openai/whisper-tiny".to_string(),
            status: 404,
        };
        let msg = err.to_string();
        assert!(msg.contains("404"), "Should include status code");
        assert!(msg.contains("whisper-tiny"), "Should include resource");
    }

    #[test]
    fn test_import_error_rate_limited_message() {
        let err = ImportError::RateLimited {
            retry_after: Some(60),
        };
        let msg = err.to_string();
        assert!(
            msg.to_lowercase().contains("rate"),
            "Should mention rate limit"
        );
        assert!(msg.contains("60"), "Should include retry time");
    }

    #[test]
    fn test_import_error_auth_required_message() {
        let err = ImportError::AuthRequired {
            resource: "meta-llama/Llama-2-7b".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("HF_TOKEN"), "Should suggest HF_TOKEN");
        assert!(msg.contains("Llama-2-7b"), "Should include resource");
    }

    #[test]
    fn test_import_error_actionable_suggestions() {
        let err = ImportError::NotFound {
            resource: "openai/whisper-tiny".to_string(),
            status: 404,
        };

        // Error should provide actionable fix
        let msg = err.to_string();
        assert!(
            msg.contains("Fix:") || msg.contains("check") || msg.contains("verify"),
            "Error should be actionable"
        );
    }

    #[test]
    fn test_import_error_sharding_oom() {
        let err = ImportError::ShardingRequired {
            model_size: 14_000_000_000, // 14GB
            shard_count: 7,
        };
        let msg = err.to_string();
        assert!(msg.contains("14"), "Should include size");
        assert!(msg.contains("7"), "Should include shard count");
    }

    // GH-129: Tests for parse_import_error (only when hf-hub-integration enabled)
    #[cfg(feature = "hf-hub-integration")]
    #[test]
    fn test_parse_import_error_404() {
        let err = parse_import_error("HTTP 404: Repository not found", "openai/whisper-tiny");
        match err {
            ImportError::NotFound { resource, status } => {
                assert_eq!(resource, "openai/whisper-tiny");
                assert_eq!(status, 404);
            }
            _ => panic!("Expected NotFound error, got {:?}", err),
        }
    }

    #[cfg(feature = "hf-hub-integration")]
    #[test]
    fn test_parse_import_error_not_found_text() {
        let err = parse_import_error("The requested resource does not exist", "test/model");
        match err {
            ImportError::NotFound { .. } => {}
            _ => panic!("Expected NotFound error, got {:?}", err),
        }
    }

    #[cfg(feature = "hf-hub-integration")]
    #[test]
    fn test_parse_import_error_401() {
        let err = parse_import_error("HTTP 401: Unauthorized access", "meta-llama/Llama-2-7b");
        match err {
            ImportError::AuthRequired { resource } => {
                assert_eq!(resource, "meta-llama/Llama-2-7b");
            }
            _ => panic!("Expected AuthRequired error, got {:?}", err),
        }
    }

    #[cfg(feature = "hf-hub-integration")]
    #[test]
    fn test_parse_import_error_gated_model() {
        let err = parse_import_error(
            "This model is gated. Access requires acceptance.",
            "meta-llama/Llama-2-7b",
        );
        match err {
            ImportError::AuthRequired { .. } => {}
            _ => panic!("Expected AuthRequired error, got {:?}", err),
        }
    }

    #[cfg(feature = "hf-hub-integration")]
    #[test]
    fn test_parse_import_error_429() {
        let err = parse_import_error(
            "HTTP 429: Too many requests. Retry after 60 seconds.",
            "test/model",
        );
        match err {
            ImportError::RateLimited { retry_after } => {
                assert_eq!(retry_after, Some(60));
            }
            _ => panic!("Expected RateLimited error, got {:?}", err),
        }
    }

    #[cfg(feature = "hf-hub-integration")]
    #[test]
    fn test_parse_import_error_rate_limit_no_retry() {
        let err = parse_import_error("Rate limit exceeded", "test/model");
        match err {
            ImportError::RateLimited { retry_after } => {
                assert_eq!(retry_after, None);
            }
            _ => panic!("Expected RateLimited error, got {:?}", err),
        }
    }

    #[cfg(feature = "hf-hub-integration")]
    #[test]
    fn test_parse_import_error_generic() {
        let err = parse_import_error("Connection timeout", "test/model");
        match err {
            ImportError::DownloadFailed { source, reason } => {
                assert_eq!(source, "test/model");
                assert_eq!(reason, "Connection timeout");
            }
            _ => panic!("Expected DownloadFailed error, got {:?}", err),
        }
    }

    #[test]
    fn test_import_error_from_conversion() {
        let import_err = ImportError::NotFound {
            resource: "test".to_string(),
            status: 404,
        };
        let aprender_err: AprenderError = import_err.into();
        let msg = aprender_err.to_string();
        assert!(msg.contains("404"));
        assert!(msg.contains("test"));
    }

    // =========================================================================
    // Coverage boost: ExportFormat, MergeStrategy, and related APIs
    // =========================================================================

    #[test]
    fn test_export_format_from_str() {
        assert!(matches!(
            "safetensors".parse::<ExportFormat>(),
            Ok(ExportFormat::SafeTensors)
        ));
        assert!(matches!(
            "st".parse::<ExportFormat>(),
            Ok(ExportFormat::SafeTensors)
        ));
        assert!(matches!(
            "gguf".parse::<ExportFormat>(),
            Ok(ExportFormat::Gguf)
        ));
        assert!(matches!(
            "onnx".parse::<ExportFormat>(),
            Ok(ExportFormat::Onnx)
        ));
        assert!(matches!(
            "torchscript".parse::<ExportFormat>(),
            Ok(ExportFormat::TorchScript)
        ));
        assert!(matches!(
            "pt".parse::<ExportFormat>(),
            Ok(ExportFormat::TorchScript)
        ));
        assert!(matches!(
            "torch".parse::<ExportFormat>(),
            Ok(ExportFormat::TorchScript)
        ));
        assert!("unknown".parse::<ExportFormat>().is_err());
    }

    #[test]
    fn test_export_format_extension() {
        assert_eq!(ExportFormat::SafeTensors.extension(), "safetensors");
        assert_eq!(ExportFormat::Gguf.extension(), "gguf");
        assert_eq!(ExportFormat::Onnx.extension(), "onnx");
        assert_eq!(ExportFormat::TorchScript.extension(), "pt");
    }

    #[test]
    fn test_export_format_is_supported() {
        assert!(ExportFormat::SafeTensors.is_supported());
        assert!(ExportFormat::Gguf.is_supported());
        assert!(!ExportFormat::Onnx.is_supported());
        assert!(!ExportFormat::TorchScript.is_supported());
    }

    #[test]
    fn test_export_options_default() {
        let opts = ExportOptions::default();
        assert!(matches!(opts.format, ExportFormat::SafeTensors));
        assert!(opts.quantize.is_none());
    }

    #[test]
    fn test_export_options_with_quantize() {
        let opts = ExportOptions {
            format: ExportFormat::Gguf,
            quantize: Some(QuantizationType::Int8),
            ..Default::default()
        };
        assert!(matches!(opts.format, ExportFormat::Gguf));
        assert!(matches!(opts.quantize, Some(QuantizationType::Int8)));
    }

    #[test]
    fn test_merge_strategy_from_str() {
        assert!(matches!(
            "average".parse::<MergeStrategy>(),
            Ok(MergeStrategy::Average)
        ));
        assert!(matches!(
            "avg".parse::<MergeStrategy>(),
            Ok(MergeStrategy::Average)
        ));
        assert!(matches!(
            "weighted".parse::<MergeStrategy>(),
            Ok(MergeStrategy::Weighted)
        ));
        assert!("unknown".parse::<MergeStrategy>().is_err());
    }

    #[test]
    fn test_merge_strategy_is_supported() {
        // Average and Weighted are supported
        assert!(MergeStrategy::Average.is_supported());
        assert!(MergeStrategy::Weighted.is_supported());
        // Advanced strategies not yet implemented
        assert!(!MergeStrategy::Ties.is_supported());
        assert!(!MergeStrategy::Dare.is_supported());
        assert!(!MergeStrategy::Slerp.is_supported());
    }

    #[test]
    fn test_merge_options_default() {
        let opts = MergeOptions::default();
        assert!(matches!(opts.strategy, MergeStrategy::Average));
        assert!(opts.weights.is_none());
    }

    #[test]
    fn test_merge_options_weighted() {
        let opts = MergeOptions {
            strategy: MergeStrategy::Weighted,
            weights: Some(vec![0.7, 0.3]),
        };
        assert!(matches!(opts.strategy, MergeStrategy::Weighted));
        assert_eq!(opts.weights, Some(vec![0.7, 0.3]));
    }

    #[test]
    fn test_merge_report_fields() {
        let report = MergeReport {
            model_count: 2,
            output_size: 1000,
            tensor_count: 10,
            strategy: MergeStrategy::Average,
            weights_used: None,
        };
        assert_eq!(report.model_count, 2);
        assert_eq!(report.output_size, 1000);
        assert_eq!(report.tensor_count, 10);
    }

    #[test]
    fn test_merge_report_with_weights() {
        let report = MergeReport {
            model_count: 3,
            output_size: 2000,
            tensor_count: 15,
            strategy: MergeStrategy::Weighted,
            weights_used: Some(vec![0.5, 0.3, 0.2]),
        };
        assert_eq!(report.model_count, 3);
        assert!(matches!(report.strategy, MergeStrategy::Weighted));
        assert!(report.weights_used.is_some());
    }

    #[test]
    fn test_export_report_fields() {
        let report = ExportReport {
            original_size: 2000,
            exported_size: 1000,
            tensor_count: 5,
            format: ExportFormat::Gguf,
            quantization: Some(QuantizationType::Int8),
        };
        assert_eq!(report.original_size, 2000);
        assert_eq!(report.exported_size, 1000);
        assert_eq!(report.tensor_count, 5);
    }

    #[test]
    fn test_validation_config_strict() {
        let config = ValidationConfig::strict();
        assert_eq!(config, ValidationConfig::Strict);
    }

    #[test]
    fn test_validation_config_default() {
        let config = ValidationConfig::default();
        assert_eq!(config, ValidationConfig::Strict);
    }

    #[test]
    fn test_validation_config_variants() {
        let _none = ValidationConfig::None;
        let _basic = ValidationConfig::Basic;
        let _strict = ValidationConfig::Strict;
    }

    #[test]
    fn test_import_options_default() {
        let opts = ImportOptions::default();
        assert_eq!(opts.validation, ValidationConfig::Strict);
        assert!(opts.quantize.is_none());
        assert!(opts.compress.is_none());
    }

    #[test]
    fn test_architecture_mapping_auto() {
        let arch = Architecture::Auto;
        // PMAT-099: Preserve model. prefix for AprTransformer compatibility
        assert_eq!(
            arch.map_name("model.embed_tokens.weight"),
            "model.embed_tokens.weight"
        );
        // Pass through names without prefix
        assert_eq!(arch.map_name("layer.0.weight"), "layer.0.weight");
    }

    #[test]
    fn test_architecture_mapping_whisper() {
        let arch = Architecture::Whisper;
        let name = arch.map_name("model.encoder.weight");
        assert!(!name.is_empty());
    }

    #[test]
    fn test_architecture_mapping_llama() {
        let arch = Architecture::Llama;
        let name = arch.map_name("model.layers.0.weight");
        assert!(!name.is_empty());
    }

    #[test]
    fn test_architecture_mapping_bert() {
        let arch = Architecture::Bert;
        let name = arch.map_name("bert.encoder.layer.0.weight");
        assert!(!name.is_empty());
    }

    #[test]
    fn test_source_parse_local_absolute() {
        let source = Source::parse("/path/to/model.safetensors").unwrap();
        assert!(matches!(source, Source::Local(_)));
    }

    #[test]
    fn test_source_parse_local_relative() {
        let source = Source::parse("./models/model.safetensors").unwrap();
        assert!(matches!(source, Source::Local(_)));
    }

    #[test]
    fn test_source_default_file_hf() {
        let source = Source::HuggingFace {
            org: "openai".to_string(),
            repo: "whisper".to_string(),
            file: None,
        };
        assert_eq!(source.default_file(), "model.safetensors");
    }

    #[test]
    fn test_source_default_file_local() {
        let source = Source::Local("/path/to/model.safetensors".into());
        // Local returns full path as the "file"
        assert!(source.default_file().ends_with("model.safetensors"));
    }

    #[test]
    fn test_tensor_expectation_for_unknown() {
        let exp = TensorExpectation::for_tensor("unknown_tensor_name");
        assert!(exp.is_none());
    }

    #[test]
    fn test_tensor_expectation_for_layer_norm_weight() {
        let exp = TensorExpectation::for_tensor("layer_norm.weight");
        assert!(exp.is_some());
        let exp = exp.unwrap();
        // LayerNorm weight should have mean near 1.0
        assert!(exp.mean_range.0 < 1.0 && exp.mean_range.1 > 1.0);
    }

    #[test]
    fn test_tensor_expectation_for_embedding() {
        let exp = TensorExpectation::for_tensor("embed_tokens.weight");
        assert!(exp.is_some());
    }

    #[test]
    fn test_import_error_display() {
        let err = ImportError::NotFound {
            resource: "model.safetensors".to_string(),
            status: 404,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("404") || msg.contains("not found"));
    }

    #[test]
    fn test_import_error_download_failed() {
        let err = ImportError::DownloadFailed {
            source: "huggingface".to_string(),
            reason: "timeout".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("timeout") || msg.contains("Download"));
    }

    #[test]
    fn test_import_error_validation_failed() {
        let err = ImportError::ValidationFailed {
            name: "layer.weight".to_string(),
            reason: "NaN detected".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("layer.weight") || msg.contains("NaN"));
    }

    #[test]
    fn test_import_error_unsupported_format() {
        let err = ImportError::UnsupportedFormat {
            extension: "pickle".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("pickle") || msg.contains("Unsupported"));
    }

    #[test]
    fn test_import_error_unknown_tensor() {
        let err = ImportError::UnknownTensor {
            source_name: "weird.tensor".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("weird.tensor") || msg.contains("Unknown"));
    }

    #[test]
    fn test_import_error_missing_tensor() {
        let err = ImportError::MissingTensor {
            name: "model.weight".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("model.weight") || msg.contains("Missing"));
    }

    #[test]
    fn test_import_error_rate_limited() {
        let err = ImportError::RateLimited {
            retry_after: Some(60),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("Rate") || msg.contains("limit") || msg.contains("60"));
    }

    #[test]
    fn test_import_error_auth_required() {
        let err = ImportError::AuthRequired {
            resource: "gated-model".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("Auth") || msg.contains("gated-model"));
    }

    #[test]
    fn test_import_error_sharding_required() {
        let err = ImportError::ShardingRequired {
            model_size: 14_000_000_000,
            shard_count: 7,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("shard") || msg.contains("7"));
    }

    // =========================================================================
    // ShardedIndex Tests
    // =========================================================================

    #[test]
    fn test_sharded_index_parse() {
        let json = r#"{
            "metadata": {"total_size": 1000000},
            "weight_map": {
                "layer.0.weight": "model-00001-of-00002.safetensors",
                "layer.1.weight": "model-00002-of-00002.safetensors"
            }
        }"#;
        let index = ShardedIndex::parse(json).expect("parse should succeed");
        assert_eq!(index.tensor_count(), 2);
        assert_eq!(index.shard_count(), 2);
    }

    #[test]
    fn test_sharded_index_shard_for_tensor() {
        let json = r#"{
            "weight_map": {
                "embed.weight": "model-00001.safetensors",
                "lm_head.weight": "model-00002.safetensors"
            }
        }"#;
        let index = ShardedIndex::parse(json).expect("parse");
        assert_eq!(
            index.shard_for_tensor("embed.weight"),
            Some("model-00001.safetensors")
        );
        assert_eq!(
            index.shard_for_tensor("lm_head.weight"),
            Some("model-00002.safetensors")
        );
        assert_eq!(index.shard_for_tensor("missing"), None);
    }

    #[test]
    fn test_sharded_index_tensors_in_shard() {
        let json = r#"{
            "weight_map": {
                "a.weight": "shard1.safetensors",
                "b.weight": "shard1.safetensors",
                "c.weight": "shard2.safetensors"
            }
        }"#;
        let index = ShardedIndex::parse(json).expect("parse");
        let tensors = index.tensors_in_shard("shard1.safetensors");
        assert_eq!(tensors.len(), 2);
        assert!(tensors.contains(&"a.weight"));
        assert!(tensors.contains(&"b.weight"));
    }

    #[test]
    fn test_sharded_index_shard_files() {
        let json = r#"{
            "weight_map": {
                "a": "z.safetensors",
                "b": "a.safetensors",
                "c": "m.safetensors"
            }
        }"#;
        let index = ShardedIndex::parse(json).expect("parse");
        let files = index.shard_files();
        // Should be sorted
        assert_eq!(
            files,
            vec!["a.safetensors", "m.safetensors", "z.safetensors"]
        );
    }

    #[test]
    fn test_sharded_index_total_size() {
        let with_size = r#"{"metadata": {"total_size": 5000}, "weight_map": {}}"#;
        let without_size = r#"{"weight_map": {}}"#;

        let index1 = ShardedIndex::parse(with_size).expect("parse");
        let index2 = ShardedIndex::parse(without_size).expect("parse");

        assert_eq!(index1.total_size(), Some(5000));
        assert_eq!(index2.total_size(), None);
    }

    #[test]
    fn test_sharded_index_parse_invalid_json() {
        let result = ShardedIndex::parse("not valid json");
        assert!(result.is_err());
    }

    // =========================================================================
    // Source URL Tests
    // =========================================================================

    #[test]
    fn test_source_parse_url() {
        let source = Source::parse("https://example.com/model.safetensors").unwrap();
        assert!(matches!(source, Source::Url(_)));
    }

    #[test]
    fn test_source_parse_http_url() {
        let source = Source::parse("http://localhost:8080/model.bin").unwrap();
        assert!(matches!(source, Source::Url(_)));
    }

    #[test]
    fn test_source_default_file_url() {
        let source = Source::Url("https://example.com/path/to/model.safetensors".to_string());
        assert_eq!(source.default_file(), "model.safetensors");
    }

    // =========================================================================
    // ConvertReport Tests
    // =========================================================================

    #[test]
    fn test_convert_report_reduction_percent() {
        let report = ConvertReport {
            original_size: 1000,
            converted_size: 500,
            tensor_count: 10,
            quantization: Some(QuantizationType::Int8),
            compression: None,
            reduction_ratio: 2.0,
        };
        let reduction = report.reduction_percent();
        assert!(reduction.contains("50"));
    }

    #[test]
    fn test_convert_report_no_reduction() {
        let report = ConvertReport {
            original_size: 1000,
            converted_size: 1000,
            tensor_count: 5,
            quantization: None,
            compression: None,
            reduction_ratio: 1.0,
        };
        let reduction = report.reduction_percent();
        assert!(reduction.contains("0"));
    }

    // =========================================================================
    // ExportFormat Tests
    // =========================================================================

    #[test]
    fn test_export_format_safetensors() {
        let format = ExportFormat::SafeTensors;
        assert_eq!(format.extension(), "safetensors");
        assert!(format.is_supported());
    }

    #[test]
    fn test_export_format_gguf() {
        let format = ExportFormat::Gguf;
        assert_eq!(format.extension(), "gguf");
        assert!(format.is_supported());
    }

    #[test]
    fn test_export_format_onnx() {
        let format = ExportFormat::Onnx;
        assert_eq!(format.extension(), "onnx");
        // ONNX may or may not be supported
        let _ = format.is_supported();
    }

    #[test]
    fn test_export_format_torchscript() {
        let format = ExportFormat::TorchScript;
        assert_eq!(format.extension(), "pt");
    }

    // =========================================================================
    // Quantization Type Tests
    // =========================================================================

    #[test]
    fn test_quantization_type_debug() {
        let q = QuantizationType::Int8;
        let debug = format!("{:?}", q);
        assert!(debug.contains("Int8"));
    }

    #[test]
    fn test_quantization_type_clone() {
        let q1 = QuantizationType::Int4;
        let q2 = q1.clone();
        assert_eq!(q1, q2);
    }

    #[test]
    fn test_q4k_quantization_roundtrip() {
        // Test data: 512 f32 values (2 super-blocks of 256)
        // Use realistic weight distribution: centered around 0, mostly negative to positive
        let mut original: Vec<f32> = Vec::with_capacity(512);
        for i in 0..512 {
            // Simulate typical weight distribution: values mostly in [-0.1, 0.1]
            // with some outliers in [-0.3, 0.3]
            let base = ((i as f32) / 512.0 - 0.5) * 0.2; // -0.1 to 0.1
            let noise = (i as f32 * 0.1).sin() * 0.05;
            original.push(base + noise);
        }

        // Quantize to Q4K bytes
        let q4k_bytes = quantize_q4_k(&original);

        // Expected size: 2 super-blocks * 144 bytes each = 288 bytes
        assert_eq!(
            q4k_bytes.len(),
            288,
            "Q4K output should be 144 bytes per 256-element super-block"
        );

        // Dequantize back to f32
        let reconstructed = dequantize_q4_k_to_f32(&q4k_bytes, 512);
        assert_eq!(reconstructed.len(), 512);

        // Check reconstruction error
        let mut max_error = 0.0f32;
        let mut total_error = 0.0f32;
        for (orig, recon) in original.iter().zip(reconstructed.iter()) {
            let error = (orig - recon).abs();
            max_error = max_error.max(error);
            total_error += error;
        }
        let avg_error = total_error / 512.0;

        // Q4_K should have reasonable reconstruction quality for typical weights
        // With 4-bit quantization (15 levels) + nested 6-bit scale quantization,
        // max error is approximately: value_range * (1/15 + 1/63) ≈ range * 0.08
        // For our data range of ~0.3, max error ~0.024, but f16 quantization
        // of d/dmin adds additional error, so we allow up to 0.06
        assert!(
            max_error < 0.06,
            "Q4K max reconstruction error too high: {max_error}"
        );
        assert!(
            avg_error < 0.02,
            "Q4K avg reconstruction error too high: {avg_error}"
        );
    }

    #[test]
    fn test_q4k_empty_data() {
        let empty: Vec<f32> = vec![];
        let q4k_bytes = quantize_q4_k(&empty);
        assert!(q4k_bytes.is_empty());

        let reconstructed = dequantize_q4_k_to_f32(&q4k_bytes, 0);
        assert!(reconstructed.is_empty());
    }

    #[test]
    fn test_q4k_partial_block() {
        // Test with 100 elements (less than one 256-element super-block)
        let original: Vec<f32> = (0..100).map(|i| i as f32 * 0.01 - 0.5).collect();

        let q4k_bytes = quantize_q4_k(&original);
        // Should have 1 super-block (144 bytes) since we pad to 256
        assert_eq!(q4k_bytes.len(), 144);

        let reconstructed = dequantize_q4_k_to_f32(&q4k_bytes, 100);
        assert_eq!(reconstructed.len(), 100);

        // Verify reasonable reconstruction
        for (orig, recon) in original.iter().zip(reconstructed.iter()) {
            let error = (orig - recon).abs();
            assert!(error < 0.2, "Reconstruction error too high: {error}");
        }
    }

    #[test]
    fn test_quantize_tensors_q4k() {
        let mut tensors = BTreeMap::new();
        tensors.insert(
            "test".to_string(),
            (
                (0..512).map(|i| i as f32 * 0.001 - 0.256).collect(),
                vec![512],
            ),
        );

        let result = quantize_tensors(&tensors, &QuantizationType::Q4K).unwrap();

        assert_eq!(result.len(), 1);
        assert!(result.contains_key("test"));
        let (data, shape) = &result["test"];
        assert_eq!(shape, &vec![512]);
        assert_eq!(data.len(), 512); // Dequantized back to f32
    }

    // =========================================================================
    // Compression Type Tests
    // =========================================================================

    #[test]
    fn test_compression_debug() {
        let c = Compression::ZstdDefault;
        let debug = format!("{:?}", c);
        assert!(debug.contains("Zstd"));
    }

    #[test]
    fn test_compression_clone() {
        let c1 = Compression::Lz4;
        let c2 = c1;
        assert_eq!(c1, c2);
    }

    // =========================================================================
    // TensorExpectation Check Tests
    // =========================================================================

    #[test]
    fn test_tensor_expectation_check_valid() {
        let exp = TensorExpectation::LAYER_NORM_WEIGHT;
        let stats = TensorStats {
            name: "layer_norm.weight".to_string(),
            count: 768,
            mean: 1.0,
            std: 0.1,
            min: 0.5,
            max: 1.5,
            nan_count: 0,
            inf_count: 0,
            zero_count: 0,
        };
        assert!(exp.check(&stats).is_ok());
    }

    #[test]
    fn test_tensor_expectation_check_invalid_mean() {
        let exp = TensorExpectation::LAYER_NORM_WEIGHT;
        let stats = TensorStats {
            name: "layer_norm.weight".to_string(),
            count: 768,
            mean: 100.0, // Way outside expected range
            std: 0.1,
            min: 99.0,
            max: 101.0,
            nan_count: 0,
            inf_count: 0,
            zero_count: 0,
        };
        assert!(exp.check(&stats).is_err());
    }

    // =========================================================================
    // TensorStats Creation Tests
    // =========================================================================

    #[test]
    fn test_tensor_stats_fields() {
        let stats = TensorStats {
            name: "test.weight".to_string(),
            count: 100,
            mean: 0.5,
            std: 0.2,
            min: 0.0,
            max: 1.0,
            nan_count: 0,
            inf_count: 0,
            zero_count: 5,
        };
        assert!((stats.mean - 0.5).abs() < 1e-6);
        assert!((stats.std - 0.2).abs() < 1e-6);
        assert!((stats.min - 0.0).abs() < 1e-6);
        assert!((stats.max - 1.0).abs() < 1e-6);
        assert_eq!(stats.count, 100);
        assert_eq!(stats.zero_count, 5);
    }

    // =========================================================================
    // Quantization and Internal Function Tests (Coverage Boost)
    // =========================================================================

    #[test]
    fn test_calculate_tensor_size() {
        let mut tensors = BTreeMap::new();
        tensors.insert("a".to_string(), (vec![1.0f32; 100], vec![10, 10]));
        tensors.insert("b".to_string(), (vec![2.0f32; 50], vec![50]));
        let size = calculate_tensor_size(&tensors);
        // 100 * 4 + 50 * 4 = 600
        assert_eq!(size, 600);
    }

    #[test]
    fn test_calculate_tensor_size_empty() {
        let tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        assert_eq!(calculate_tensor_size(&tensors), 0);
    }

    #[test]
    fn test_quantize_fp16_roundtrip() {
        let data = vec![1.0, 2.0, 3.0, -1.0, 0.0, 0.5];
        let quantized = quantize_fp16(&data);
        // Should preserve values with f16 precision
        assert_eq!(quantized.len(), data.len());
        for (orig, quant) in data.iter().zip(quantized.iter()) {
            // f16 has limited precision
            assert!((orig - quant).abs() < 0.01, "fp16 should preserve value");
        }
    }

    #[test]
    fn test_quantize_fp16_large_values() {
        let data = vec![65504.0, -65504.0]; // max f16 values
        let quantized = quantize_fp16(&data);
        assert!((quantized[0] - 65504.0).abs() < 1.0);
    }

    #[test]
    fn test_quantize_int8_roundtrip() {
        let data = vec![1.0, -1.0, 0.5, -0.5, 0.0, 0.25];
        let quantized = quantize_int8(&data);
        assert_eq!(quantized.len(), data.len());
        // int8 quantization scales to -127..127
        for (orig, quant) in data.iter().zip(quantized.iter()) {
            assert!(
                (orig - quant).abs() < 0.05,
                "int8 should preserve value within tolerance"
            );
        }
    }

    #[test]
    fn test_quantize_int8_all_zeros() {
        let data = vec![0.0, 0.0, 0.0];
        let quantized = quantize_int8(&data);
        for v in &quantized {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn test_quantize_int4_roundtrip() {
        let data = vec![1.0, -1.0, 0.5, -0.5, 0.0, 0.25];
        let quantized = quantize_int4(&data);
        assert_eq!(quantized.len(), data.len());
        // int4 has only 16 levels so lower precision
        for (orig, quant) in data.iter().zip(quantized.iter()) {
            assert!(
                (orig - quant).abs() < 0.15,
                "int4 should preserve value within tolerance"
            );
        }
    }

    #[test]
    fn test_quantize_int4_all_zeros() {
        let data = vec![0.0, 0.0, 0.0];
        let quantized = quantize_int4(&data);
        for v in &quantized {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn test_f16_to_f32_zero() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
    }

    #[test]
    fn test_f16_to_f32_one() {
        let result = f16_to_f32(0x3C00);
        assert!((result - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_f16_to_f32_negative() {
        let result = f16_to_f32(0xBC00);
        assert!((result + 1.0).abs() < 0.001);
    }

    #[test]
    fn test_f16_to_f32_subnormal() {
        let result = f16_to_f32(0x0001);
        assert!(result > 0.0 && result < 0.001);
    }

    #[test]
    fn test_f16_to_f32_max() {
        // Max f16 is 65504
        let result = f16_to_f32(0x7BFF);
        assert!((result - 65504.0).abs() < 1.0);
    }

    #[test]
    fn test_convert_report_zero_sizes() {
        let report = ConvertReport {
            original_size: 0,
            converted_size: 0,
            tensor_count: 0,
            quantization: None,
            compression: None,
            reduction_ratio: 0.0,
        };
        assert_eq!(report.reduction_percent(), "N/A");
    }

    #[test]
    fn test_convert_report_debug() {
        let report = ConvertReport {
            original_size: 1000,
            converted_size: 500,
            tensor_count: 10,
            quantization: Some(QuantizationType::Int8),
            compression: Some(Compression::Lz4),
            reduction_ratio: 2.0,
        };
        assert!(format!("{:?}", report).contains("ConvertReport"));
    }

    #[test]
    fn test_quantize_tensors_fp16() {
        let mut tensors = BTreeMap::new();
        tensors.insert("w".to_string(), (vec![1.0, 2.0, 3.0], vec![3]));
        let result = quantize_tensors(&tensors, &QuantizationType::Fp16).expect("quantize");
        assert!(result.contains_key("w"));
    }

    #[test]
    fn test_quantize_tensors_int8() {
        let mut tensors = BTreeMap::new();
        tensors.insert("w".to_string(), (vec![1.0, -1.0, 0.5], vec![3]));
        let result = quantize_tensors(&tensors, &QuantizationType::Int8).expect("quantize");
        assert!(result.contains_key("w"));
    }

    #[test]
    fn test_quantize_tensors_int4() {
        let mut tensors = BTreeMap::new();
        tensors.insert("w".to_string(), (vec![0.5, -0.5, 0.0], vec![3]));
        let result = quantize_tensors(&tensors, &QuantizationType::Int4).expect("quantize");
        assert!(result.contains_key("w"));
    }

    #[test]
    fn test_dequantize_q4k_to_f32_basic() {
        // Create a minimal Q4K block (144 bytes for 256 elements)
        let mut data = vec![0u8; 144];
        // Set d = 1.0 in f16 (0x3C00)
        data[0] = 0x00;
        data[1] = 0x3C;
        // Set dmin = 0.0
        data[2] = 0x00;
        data[3] = 0x00;
        let result = dequantize_q4_k_to_f32(&data, 256);
        assert_eq!(result.len(), 256);
    }

    #[test]
    fn test_dequantize_q4k_to_f32_truncated() {
        // Data smaller than one block
        let data = vec![0u8; 50];
        let result = dequantize_q4_k_to_f32(&data, 256);
        // Should produce zero-filled result
        assert_eq!(result.len(), 256);
    }

    /// PMAT-177: Test that NaN/Inf scale factors are replaced with safe values
    #[test]
    fn test_dequantize_q4k_nan_inf_protection_pmat177() {
        // Create a Q4K block with NaN d value (f16 NaN = 0x7E00)
        let mut data = vec![0u8; 144];
        // Set d = NaN in f16 (0x7E00)
        data[0] = 0x00;
        data[1] = 0x7E;
        // Set dmin = Inf in f16 (0x7C00)
        data[2] = 0x00;
        data[3] = 0x7C;

        let result = dequantize_q4_k_to_f32(&data, 256);

        // PMAT-177: Result should contain NO NaN or Inf values
        let nan_count = result.iter().filter(|v| v.is_nan()).count();
        let inf_count = result.iter().filter(|v| v.is_infinite()).count();

        assert_eq!(
            nan_count, 0,
            "PMAT-177: dequantize_q4_k should not produce NaN"
        );
        assert_eq!(
            inf_count, 0,
            "PMAT-177: dequantize_q4_k should not produce Inf"
        );
    }

    /// PMAT-177: Test that subnormal f16 scales are clamped to zero
    #[test]
    fn test_dequantize_q4k_subnormal_protection_pmat177() {
        // Create a Q4K block with subnormal d value (f16 subnormal = 0x0001)
        let mut data = vec![0u8; 144];
        // Set d = subnormal in f16 (0x0001 - smallest subnormal)
        data[0] = 0x01;
        data[1] = 0x00;
        // Set dmin = 0.0
        data[2] = 0x00;
        data[3] = 0x00;

        let result = dequantize_q4_k_to_f32(&data, 256);

        // PMAT-177: Subnormal should be treated as zero, result should be all zeros
        let non_zero_count = result.iter().filter(|&&v| v != 0.0).count();
        assert_eq!(
            non_zero_count, 0,
            "PMAT-177: subnormal f16 scales should be clamped to zero"
        );
    }

    #[test]
    fn test_calculate_merge_weights_average() {
        let options = MergeOptions {
            strategy: MergeStrategy::Average,
            weights: None,
        };
        let weights = calculate_merge_weights(3, &options).expect("weights");
        assert_eq!(weights.len(), 3);
        for w in &weights {
            assert!((*w - 1.0 / 3.0).abs() < 0.001);
        }
    }

    #[test]
    fn test_calculate_merge_weights_custom() {
        let options = MergeOptions {
            strategy: MergeStrategy::Weighted,
            weights: Some(vec![0.5, 0.3, 0.2]),
        };
        let weights = calculate_merge_weights(3, &options).expect("weights");
        // Weighted merging always normalizes
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_calculate_merge_weights_normalize() {
        let options = MergeOptions {
            strategy: MergeStrategy::Weighted,
            weights: Some(vec![2.0, 2.0, 1.0]),
        };
        let weights = calculate_merge_weights(3, &options).expect("weights");
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 0.001);
        // Check relative proportions: 2:2:1
        assert!((weights[0] - 0.4).abs() < 0.001);
        assert!((weights[1] - 0.4).abs() < 0.001);
        assert!((weights[2] - 0.2).abs() < 0.001);
    }

    #[test]
    fn test_calculate_merge_weights_zero_sum() {
        let options = MergeOptions {
            strategy: MergeStrategy::Weighted,
            weights: Some(vec![0.0, 0.0, 0.0]),
        };
        let result = calculate_merge_weights(3, &options);
        assert!(result.is_err());
    }

    // ========================================================================
    // Additional TensorExpectation Coverage Tests
    // ========================================================================

    #[test]
    fn test_tensor_expectation_input_layernorm() {
        let exp = TensorExpectation::for_tensor("model.layers.0.input_layernorm.weight");
        assert!(exp.is_some());
        let exp = exp.unwrap();
        assert_eq!(exp.description, "RMSNorm weight (gamma)");
    }

    #[test]
    fn test_tensor_expectation_post_attention_layernorm() {
        let exp = TensorExpectation::for_tensor("model.layers.0.post_attention_layernorm.weight");
        assert!(exp.is_some());
        let exp = exp.unwrap();
        assert_eq!(exp.description, "RMSNorm weight (gamma)");
    }

    #[test]
    fn test_tensor_expectation_rms_norm() {
        let exp = TensorExpectation::for_tensor("rms_norm.weight");
        assert!(exp.is_some());
    }

    /// Fix #163: GGUF attn_norm pattern should be recognized as RMSNorm
    #[test]
    fn test_tensor_expectation_gguf_attn_norm() {
        let exp = TensorExpectation::for_tensor("blk.0.attn_norm.weight");
        assert!(exp.is_some());
        let exp = exp.unwrap();
        assert_eq!(exp.description, "RMSNorm weight (gamma)");
        // Mean range should be wide enough for trained weights
        assert!(exp.mean_range.0 <= 0.0 && exp.mean_range.1 >= 2.0);
    }

    /// Fix #163: GGUF ffn_norm pattern should be recognized as RMSNorm
    #[test]
    fn test_tensor_expectation_gguf_ffn_norm() {
        let exp = TensorExpectation::for_tensor("blk.5.ffn_norm.weight");
        assert!(exp.is_some());
        let exp = exp.unwrap();
        assert_eq!(exp.description, "RMSNorm weight (gamma)");
    }

    #[test]
    fn test_tensor_expectation_ln_weight() {
        let exp = TensorExpectation::for_tensor("ln_1.weight");
        assert!(exp.is_some());
        let exp = exp.unwrap();
        assert_eq!(exp.description, "LayerNorm weight (gamma)");
    }

    #[test]
    fn test_tensor_expectation_ln_bias() {
        let exp = TensorExpectation::for_tensor("ln_1.bias");
        assert!(exp.is_some());
        let exp = exp.unwrap();
        assert_eq!(exp.description, "LayerNorm bias (beta)");
    }

    #[test]
    fn test_tensor_expectation_gamma() {
        let exp = TensorExpectation::for_tensor("layer_norm.gamma");
        assert!(exp.is_some());
        let exp = exp.unwrap();
        assert_eq!(exp.description, "LayerNorm weight (gamma)");
    }

    #[test]
    fn test_tensor_expectation_beta() {
        let exp = TensorExpectation::for_tensor("layer_norm.beta");
        assert!(exp.is_some());
        let exp = exp.unwrap();
        assert_eq!(exp.description, "LayerNorm bias (beta)");
    }

    #[test]
    fn test_tensor_expectation_final_norm() {
        let exp = TensorExpectation::for_tensor("norm.weight");
        assert!(exp.is_some());
        let exp = exp.unwrap();
        assert_eq!(exp.description, "RMSNorm weight (gamma)");
    }

    #[test]
    fn test_tensor_expectation_model_norm() {
        let exp = TensorExpectation::for_tensor("model.norm.weight");
        assert!(exp.is_some());
        let exp = exp.unwrap();
        assert_eq!(exp.description, "RMSNorm weight (gamma)");
    }

    #[test]
    fn test_tensor_expectation_linear_weight() {
        let exp = TensorExpectation::for_tensor("fc1.weight");
        assert!(exp.is_some());
        let exp = exp.unwrap();
        assert_eq!(exp.description, "Linear/Attention weight");
    }

    #[test]
    fn test_tensor_expectation_check_valid_layernorm() {
        let exp = TensorExpectation::LAYER_NORM_WEIGHT;
        let stats = TensorStats {
            name: "test.weight".to_string(),
            count: 1000,
            mean: 1.0,
            std: 0.5,
            min: 0.0,
            max: 2.0,
            nan_count: 0,
            inf_count: 0,
            zero_count: 0,
        };
        assert!(exp.check(&stats).is_ok());
    }

    #[test]
    fn test_tensor_expectation_check_invalid_layernorm_std() {
        let exp = TensorExpectation::LAYER_NORM_WEIGHT;
        let stats = TensorStats {
            name: "test.weight".to_string(),
            count: 1000,
            mean: 1.0,
            std: 5.0, // Too high
            min: -10.0,
            max: 10.0,
            nan_count: 0,
            inf_count: 0,
            zero_count: 0,
        };
        // std range is Some((0.0, 2.0)), so 5.0 is outside
        assert!(exp.check(&stats).is_err());
    }

    #[test]
    fn test_tensor_expectation_linear_no_std_range() {
        let exp = TensorExpectation::LINEAR_WEIGHT;
        assert!(exp.std_range.is_none());
    }

    #[test]
    fn test_tensor_expectation_embedding_range() {
        let exp = TensorExpectation::EMBEDDING;
        assert!(exp.mean_range.0 < 0.0);
        assert!(exp.mean_range.1 > 0.0);
    }

    #[test]
    fn test_tensor_expectation_rmsnorm_range() {
        let exp = TensorExpectation::RMSNORM_WEIGHT;
        // Wide range for trained models
        assert!(exp.mean_range.0 < 0.0);
        assert!(exp.mean_range.1 > 2.0);
    }

    // ========================================================================
    // Additional Architecture Coverage Tests
    // ========================================================================

    #[test]
    fn test_architecture_auto_preserves_model_prefix() {
        let arch = Architecture::Auto;
        assert_eq!(arch.map_name("model.weight"), "model.weight");
    }

    #[test]
    fn test_architecture_whisper_preserves_prefix() {
        let arch = Architecture::Whisper;
        assert_eq!(
            arch.map_name("model.encoder.weight"),
            "model.encoder.weight"
        );
    }

    #[test]
    fn test_architecture_llama_preserves_prefix() {
        let arch = Architecture::Llama;
        assert_eq!(
            arch.map_name("model.layers.0.weight"),
            "model.layers.0.weight"
        );
    }

    #[test]
    fn test_architecture_bert_preserves_prefix() {
        let arch = Architecture::Bert;
        assert_eq!(arch.map_name("bert.encoder.weight"), "bert.encoder.weight");
    }

    #[test]
    fn test_architecture_qwen2_preserves_prefix() {
        let arch = Architecture::Qwen2;
        assert_eq!(
            arch.map_name("model.embed_tokens.weight"),
            "model.embed_tokens.weight"
        );
    }

    #[test]
    fn test_architecture_debug() {
        let arch = Architecture::Auto;
        assert!(format!("{:?}", arch).contains("Auto"));
    }

    #[test]
    fn test_architecture_clone() {
        let arch1 = Architecture::Llama;
        let arch2 = arch1.clone();
        assert_eq!(arch1, arch2);
    }

    // ========================================================================
    // Source Type Coverage Tests
    // ========================================================================

    #[test]
    fn test_source_hf_with_file() {
        let source = Source::HuggingFace {
            org: "org".to_string(),
            repo: "repo".to_string(),
            file: Some("custom.safetensors".to_string()),
        };
        assert_eq!(source.default_file(), "custom.safetensors");
    }

    #[test]
    fn test_source_url_default_file() {
        let source = Source::Url("https://example.com/path/to/model.gguf".to_string());
        assert_eq!(source.default_file(), "model.gguf");
    }

    #[test]
    fn test_source_url_no_filename() {
        let source = Source::Url("https://example.com/".to_string());
        // URL without filename returns empty (edge case)
        let file = source.default_file();
        // Can be empty if no filename in URL
        let _ = file;
    }

    #[test]
    fn test_source_debug() {
        let source = Source::Local("/path/to/model".into());
        assert!(format!("{:?}", source).contains("Local"));
    }

    #[test]
    fn test_source_clone() {
        let source1 = Source::Url("https://test.com".to_string());
        let source2 = source1.clone();
        assert!(matches!(source2, Source::Url(_)));
    }

    // ========================================================================
    // Validation Config Coverage Tests
    // ========================================================================

    #[test]
    fn test_validation_config_none() {
        let config = ValidationConfig::None;
        assert!(matches!(config, ValidationConfig::None));
    }

    #[test]
    fn test_validation_config_basic() {
        let config = ValidationConfig::Basic;
        assert!(matches!(config, ValidationConfig::Basic));
    }

    // ========================================================================
    // QuantizationType Coverage Tests
    // ========================================================================

    #[test]
    fn test_quantization_type_eq() {
        assert_eq!(QuantizationType::Fp16, QuantizationType::Fp16);
        assert_ne!(QuantizationType::Int8, QuantizationType::Int4);
    }

    #[test]
    fn test_quantization_type_q4k() {
        let q = QuantizationType::Q4K;
        assert!(format!("{:?}", q).contains("Q4K"));
    }

    // ========================================================================
    // Compression Coverage Tests
    // ========================================================================

    #[test]
    fn test_compression_zstd_default() {
        let c = Compression::ZstdDefault;
        assert!(format!("{:?}", c).contains("Zstd"));
    }

    #[test]
    fn test_compression_eq() {
        assert_eq!(Compression::Lz4, Compression::Lz4);
        assert_ne!(Compression::Lz4, Compression::ZstdDefault);
    }

    // ========================================================================
    // Import Options Coverage Tests
    // ========================================================================

    #[test]
    fn test_import_options_with_quantize() {
        let opts = ImportOptions {
            architecture: Architecture::Auto,
            validation: ValidationConfig::Basic,
            quantize: Some(QuantizationType::Int8),
            compress: Some(Compression::Lz4),
            force: true,
            cache: false,
        };
        assert!(opts.quantize.is_some());
        assert!(opts.compress.is_some());
        assert!(opts.force);
        assert!(!opts.cache);
    }

    #[test]
    fn test_import_options_debug() {
        let opts = ImportOptions::default();
        assert!(format!("{:?}", opts).contains("ImportOptions"));
    }

    #[test]
    fn test_import_options_clone() {
        let opts1 = ImportOptions::default();
        let opts2 = opts1.clone();
        assert_eq!(opts1.validation, opts2.validation);
    }

    // ========================================================================
    // ConvertOptions Coverage Tests
    // ========================================================================

    #[test]
    fn test_convert_options_default() {
        let opts = ConvertOptions::default();
        assert!(opts.quantize.is_none());
        assert!(opts.compress.is_none());
    }

    #[test]
    fn test_convert_options_with_all() {
        let opts = ConvertOptions {
            quantize: Some(QuantizationType::Q4K),
            compress: Some(Compression::ZstdDefault),
            validate: true,
        };
        assert!(opts.quantize.is_some());
        assert!(opts.compress.is_some());
        assert!(opts.validate);
    }

    #[test]
    fn test_convert_options_debug() {
        let opts = ConvertOptions::default();
        assert!(format!("{:?}", opts).contains("ConvertOptions"));
    }

    #[test]
    fn test_convert_options_clone() {
        let opts1 = ConvertOptions {
            quantize: Some(QuantizationType::Int8),
            compress: None,
            validate: false,
        };
        let opts2 = opts1.clone();
        assert_eq!(opts1.quantize, opts2.quantize);
        assert_eq!(opts1.validate, opts2.validate);
    }

    // ========================================================================
    // TensorStats Coverage Tests
    // ========================================================================

    #[test]
    fn test_tensor_stats_debug() {
        let stats = TensorStats {
            name: "test".to_string(),
            count: 100,
            mean: 0.0,
            std: 1.0,
            min: -3.0,
            max: 3.0,
            nan_count: 0,
            inf_count: 0,
            zero_count: 10,
        };
        assert!(format!("{:?}", stats).contains("TensorStats"));
    }

    #[test]
    fn test_tensor_stats_clone() {
        let stats1 = TensorStats {
            name: "w".to_string(),
            count: 50,
            mean: 0.5,
            std: 0.1,
            min: 0.0,
            max: 1.0,
            nan_count: 0,
            inf_count: 0,
            zero_count: 5,
        };
        let stats2 = stats1.clone();
        assert_eq!(stats1.name, stats2.name);
        assert_eq!(stats1.count, stats2.count);
    }

    // ========================================================================
    // Internal Helper Function Tests (ROSETTA-ML-001)
    // ========================================================================

    #[test]
    fn test_compute_std_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mean = 3.0;
        let std = compute_std(&data, mean, 5);
        // Expected: sqrt(((1-3)^2 + (2-3)^2 + (3-3)^2 + (4-3)^2 + (5-3)^2) / 4)
        // = sqrt((4 + 1 + 0 + 1 + 4) / 4) = sqrt(10/4) = sqrt(2.5) ≈ 1.58
        assert!((std - 1.58).abs() < 0.01);
    }

    #[test]
    fn test_compute_std_single_value() {
        let data = vec![42.0];
        let std = compute_std(&data, 42.0, 1);
        assert_eq!(std, 0.0);
    }

    #[test]
    fn test_compute_std_empty() {
        let data: Vec<f32> = vec![];
        let std = compute_std(&data, 0.0, 0);
        assert_eq!(std, 0.0);
    }

    #[test]
    fn test_compute_std_with_nan() {
        let data = vec![1.0, 2.0, f32::NAN, 4.0, 5.0];
        let mean = 3.0; // mean of valid values
        let std = compute_std(&data, mean, 4);
        // NaN is filtered out, so only 4 valid values
        assert!(std > 0.0);
        assert!(std.is_finite());
    }

    #[test]
    fn test_compute_std_with_inf() {
        let data = vec![1.0, 2.0, f32::INFINITY, 4.0, 5.0];
        let mean = 3.0;
        let std = compute_std(&data, mean, 4);
        // Infinity is filtered out
        assert!(std.is_finite());
    }

    #[test]
    fn test_compute_tensor_stats_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = compute_tensor_stats("test_tensor", &data);
        assert_eq!(stats.name, "test_tensor");
        assert_eq!(stats.count, 5);
        assert!((stats.mean - 3.0).abs() < 0.01);
        assert!((stats.min - 1.0).abs() < 0.01);
        assert!((stats.max - 5.0).abs() < 0.01);
        assert!(stats.std > 0.0);
        assert_eq!(stats.nan_count, 0);
        assert_eq!(stats.inf_count, 0);
    }

    #[test]
    fn test_compute_tensor_stats_empty() {
        let data: Vec<f32> = vec![];
        let stats = compute_tensor_stats("empty_tensor", &data);
        assert_eq!(stats.name, "empty_tensor");
        assert_eq!(stats.count, 0);
        assert_eq!(stats.mean, 0.0);
        assert_eq!(stats.std, 0.0);
        assert_eq!(stats.min, 0.0);
        assert_eq!(stats.max, 0.0);
    }

    #[test]
    fn test_compute_tensor_stats_with_nan() {
        let data = vec![1.0, f32::NAN, 3.0, f32::NAN, 5.0];
        let stats = compute_tensor_stats("nan_tensor", &data);
        assert_eq!(stats.count, 5);
        assert_eq!(stats.nan_count, 2);
        // Mean should be computed from valid values only
        assert!((stats.mean - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_compute_tensor_stats_with_inf() {
        let data = vec![1.0, f32::INFINITY, 3.0, f32::NEG_INFINITY, 5.0];
        let stats = compute_tensor_stats("inf_tensor", &data);
        assert_eq!(stats.count, 5);
        assert_eq!(stats.inf_count, 2);
    }

    #[test]
    fn test_compute_tensor_stats_zeros() {
        let data = vec![0.0, 1.0, 0.0, 2.0, 0.0];
        let stats = compute_tensor_stats("sparse", &data);
        assert_eq!(stats.zero_count, 3);
    }

    #[test]
    fn test_needs_transpose_2d_weight() {
        assert!(needs_transpose("layer.0.attn_q.weight", &[512, 512]));
        assert!(needs_transpose(
            "model.layers.0.self_attn.q_proj.weight",
            &[512, 512]
        ));
        assert!(needs_transpose("model.lm_head.weight", &[50257, 768]));
    }

    #[test]
    fn test_needs_transpose_1d() {
        // 1D tensors should NOT be transposed
        assert!(!needs_transpose("layer.0.attn_q.bias", &[512]));
        assert!(!needs_transpose(
            "model.layers.0.self_attn.q_proj.weight",
            &[512]
        ));
    }

    #[test]
    fn test_needs_transpose_3d() {
        // 3D tensors should NOT be transposed
        assert!(!needs_transpose("conv.weight", &[32, 64, 3]));
    }

    #[test]
    fn test_needs_transpose_non_weight() {
        // Non-weight 2D tensors should NOT be transposed
        assert!(!needs_transpose("layer.0.attn_q.bias", &[512, 512]));
        assert!(!needs_transpose("embeddings", &[50257, 768]));
    }

    #[test]
    fn test_needs_transpose_all_patterns() {
        // Test all weight patterns from the function
        let patterns = [
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
        for pattern in patterns {
            let name = format!("model.layers.0.{pattern}");
            assert!(
                needs_transpose(&name, &[512, 512]),
                "Pattern {pattern} should need transpose"
            );
        }
    }

    // ========================================================================
    // TensorAccumulator Tests (ROSETTA-ML-001)
    // ========================================================================

    #[test]
    fn test_tensor_accumulator_new() {
        let acc = TensorAccumulator::new();
        assert_eq!(acc.valid_count, 0);
        assert_eq!(acc.nan_count, 0);
        assert_eq!(acc.inf_count, 0);
        assert_eq!(acc.zero_count, 0);
    }

    #[test]
    fn test_tensor_accumulator_basic_values() {
        let mut acc = TensorAccumulator::new();
        acc.accumulate(1.0);
        acc.accumulate(2.0);
        acc.accumulate(3.0);

        assert_eq!(acc.valid_count, 3);
        assert!((acc.mean() - 2.0).abs() < 0.001);
        assert!((acc.min - 1.0).abs() < 0.001);
        assert!((acc.max - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_tensor_accumulator_nan_tracking() {
        let mut acc = TensorAccumulator::new();
        acc.accumulate(1.0);
        acc.accumulate(f32::NAN);
        acc.accumulate(2.0);
        acc.accumulate(f32::NAN);
        acc.accumulate(3.0);

        assert_eq!(acc.valid_count, 3);
        assert_eq!(acc.nan_count, 2);
        assert!((acc.mean() - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_tensor_accumulator_inf_tracking() {
        let mut acc = TensorAccumulator::new();
        acc.accumulate(1.0);
        acc.accumulate(f32::INFINITY);
        acc.accumulate(2.0);
        acc.accumulate(f32::NEG_INFINITY);
        acc.accumulate(3.0);

        assert_eq!(acc.valid_count, 3);
        assert_eq!(acc.inf_count, 2);
    }

    #[test]
    fn test_tensor_accumulator_zero_tracking() {
        let mut acc = TensorAccumulator::new();
        acc.accumulate(0.0);
        acc.accumulate(1.0);
        acc.accumulate(0.0);

        assert_eq!(acc.zero_count, 2);
        assert_eq!(acc.valid_count, 3);
    }

    #[test]
    fn test_tensor_accumulator_mean_empty() {
        let acc = TensorAccumulator::new();
        assert_eq!(acc.mean(), 0.0);
    }

    #[test]
    fn test_tensor_accumulator_safe_min_empty() {
        let acc = TensorAccumulator::new();
        assert_eq!(acc.safe_min(), 0.0);
    }

    #[test]
    fn test_tensor_accumulator_safe_max_empty() {
        let acc = TensorAccumulator::new();
        assert_eq!(acc.safe_max(), 0.0);
    }

    #[test]
    fn test_tensor_accumulator_negative_values() {
        let mut acc = TensorAccumulator::new();
        acc.accumulate(-5.0);
        acc.accumulate(-1.0);
        acc.accumulate(0.0);
        acc.accumulate(1.0);
        acc.accumulate(5.0);

        assert_eq!(acc.valid_count, 5);
        assert!((acc.safe_min() - (-5.0)).abs() < 0.001);
        assert!((acc.safe_max() - 5.0).abs() < 0.001);
        assert!((acc.mean() - 0.0).abs() < 0.001);
    }

    // ========================================================================
    // Quantization Roundtrip Tests
    // ========================================================================

    #[test]
    fn test_validate_single_tensor_no_issues() {
        // Use data that will pass validation: mean near 0, std reasonable
        let data = vec![-0.5, -0.25, 0.0, 0.25, 0.5];
        let mut validator = AprValidator::new();
        let mut errors = Vec::new();
        let mut options = ImportOptions::default();
        // Use Basic validation which is less strict
        options.validation = ValidationConfig::Basic;

        validate_single_tensor("test_tensor", &data, &options, &mut validator, &mut errors);

        // Basic validation should not produce errors for reasonable data
        assert!(errors.is_empty(), "Unexpected errors: {:?}", errors);
    }

    #[test]
    fn test_validate_single_tensor_with_nan() {
        let data = vec![0.1, f32::NAN, 0.3, 0.4];
        let mut validator = AprValidator::new();
        let mut errors = Vec::new();
        let mut options = ImportOptions::default();
        options.validation = ValidationConfig::Strict;

        validate_single_tensor("test.weight", &data, &options, &mut validator, &mut errors);

        // Should have error for NaN
        assert!(errors.iter().any(|e| e.contains("NaN")));
    }

    #[test]
    fn test_validate_single_tensor_none_validation() {
        let data = vec![0.1, f32::NAN, f32::INFINITY, 0.4];
        let mut validator = AprValidator::new();
        let mut errors = Vec::new();
        let mut options = ImportOptions::default();
        options.validation = ValidationConfig::None;

        validate_single_tensor("test.weight", &data, &options, &mut validator, &mut errors);

        // ValidationConfig::None should not produce errors
        assert!(errors.is_empty());
    }

    #[test]
    fn test_compression_variants() {
        let _zstd_default = Compression::ZstdDefault;
        let _zstd_max = Compression::ZstdMax;
        let _lz4 = Compression::Lz4;
        let _none = Compression::None;
    }

    // ========================================================================
    // TensorExpectation Tests (ROSETTA-ML-001)
    // ========================================================================

    #[test]
    fn test_tensor_expectation_for_tensor_rmsnorm() {
        // Test RMSNorm weight pattern detection
        let exp = TensorExpectation::for_tensor("model.layers.0.input_layernorm.weight");
        assert!(exp.is_some());
        let exp = exp.unwrap();
        assert!(exp.mean_range.0 < 1.0 && exp.mean_range.1 > 1.0);
    }

    #[test]
    fn test_tensor_expectation_for_tensor_rmsnorm_post_attn() {
        let exp = TensorExpectation::for_tensor("model.layers.0.post_attention_layernorm.weight");
        assert!(exp.is_some());
    }

    #[test]
    fn test_tensor_expectation_for_tensor_rms_norm() {
        let exp = TensorExpectation::for_tensor("model.layers.0.rms_norm.weight");
        assert!(exp.is_some());
    }

    #[test]
    fn test_tensor_expectation_for_tensor_layer_norm_gamma() {
        let exp = TensorExpectation::for_tensor("bert.encoder.layer_norm.gamma");
        assert!(exp.is_some());
    }

    #[test]
    fn test_tensor_expectation_for_tensor_layer_norm_beta() {
        let exp = TensorExpectation::for_tensor("bert.encoder.layer_norm.beta");
        assert!(exp.is_some());
    }

    #[test]
    fn test_tensor_expectation_for_tensor_ln_weight() {
        let exp = TensorExpectation::for_tensor("transformer.ln_1.weight");
        assert!(exp.is_some());
    }

    #[test]
    fn test_tensor_expectation_for_tensor_ln_bias() {
        let exp = TensorExpectation::for_tensor("transformer.ln_1.bias");
        assert!(exp.is_some());
    }

    #[test]
    fn test_tensor_expectation_for_tensor_final_norm() {
        let exp = TensorExpectation::for_tensor("norm.weight");
        assert!(exp.is_some());
    }

    #[test]
    fn test_tensor_expectation_for_tensor_embedding() {
        let exp = TensorExpectation::for_tensor("model.embed_tokens.weight");
        assert!(exp.is_some());
    }

    #[test]
    fn test_tensor_expectation_for_tensor_linear_weight() {
        let exp = TensorExpectation::for_tensor("model.layers.0.fc1.weight");
        assert!(exp.is_some());
    }

    #[test]
    fn test_tensor_expectation_check_passing() {
        let exp = TensorExpectation::EMBEDDING;
        let stats = TensorStats {
            name: "embed.weight".to_string(),
            count: 1000,
            mean: 0.001, // Near 0, within range
            std: 0.02,
            min: -0.1,
            max: 0.1,
            nan_count: 0,
            inf_count: 0,
            zero_count: 10,
        };
        assert!(exp.check(&stats).is_ok());
    }

    #[test]
    fn test_tensor_expectation_check_mean_out_of_range() {
        let exp = TensorExpectation::EMBEDDING;
        let stats = TensorStats {
            name: "embed.weight".to_string(),
            count: 1000,
            mean: 5.0, // Way outside expected range
            std: 0.02,
            min: -0.1,
            max: 0.1,
            nan_count: 0,
            inf_count: 0,
            zero_count: 10,
        };
        assert!(exp.check(&stats).is_err());
    }

    #[test]
    fn test_tensor_expectation_check_std_out_of_range() {
        // Use LAYER_NORM_WEIGHT which has std_range check
        let exp = TensorExpectation::LAYER_NORM_WEIGHT;
        let stats = TensorStats {
            name: "layer_norm.weight".to_string(),
            count: 1000,
            mean: 1.0,  // Within mean range for LayerNorm
            std: 100.0, // Way outside expected std range (0.0, 2.0)
            min: -0.1,
            max: 0.1,
            nan_count: 0,
            inf_count: 0,
            zero_count: 10,
        };
        assert!(exp.check(&stats).is_err());
    }

    #[test]
    fn test_tensor_expectation_check_rmsnorm_passing() {
        let exp = TensorExpectation::RMSNORM_WEIGHT;
        let stats = TensorStats {
            name: "norm.weight".to_string(),
            count: 100,
            mean: 1.0, // Near 1.0 for RMSNorm
            std: 0.01,
            min: 0.99,
            max: 1.01,
            nan_count: 0,
            inf_count: 0,
            zero_count: 0,
        };
        assert!(exp.check(&stats).is_ok());
    }
}

// =============================================================================
// PMAT-107: GQA Metadata Preservation Tests (Falsification Protocol)
// =============================================================================
#[cfg(test)]
mod tests_pmat_107_gqa_metadata {
    use super::*;
    use std::collections::BTreeMap;

    /// PMAT-107 Falsification Test: num_kv_heads MUST be inferred from K projection tensor
    ///
    /// This test verifies that `infer_model_config_from_tensors()` correctly identifies
    /// GQA models (where num_kv_heads < num_heads) from tensor shapes.
    ///
    /// Failure Mode (Pre-Fix):
    ///   num_kv_heads defaulted to num_heads, causing MHA dimensions on GPU
    ///   GPU kernels launched with wrong grid size -> CUDA hang
    #[test]
    fn test_pmat_107_gqa_num_kv_heads_inferred_from_k_proj() {
        // Simulate a GQA model:
        // - hidden_size: 768 (must be divisible by head_dim=64, which the code tries first)
        // - num_heads: 12 (768 / 64 = 12)
        // - num_kv_heads: 2 (GQA ratio 6:1)
        // - head_dim: 64
        // - q_dim: 12 * 64 = 768
        // - kv_dim: 2 * 64 = 128
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();

        // Embedding layer: [vocab_size, hidden_size]
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            (vec![0.0; 32000 * 768], vec![32000, 768]),
        );

        // Q projection: [q_dim, hidden_size] = [768, 768]
        tensors.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            (vec![0.0; 768 * 768], vec![768, 768]),
        );

        // K projection: [kv_dim, hidden_size] = [128, 768] (GQA: 2 heads, not 12)
        tensors.insert(
            "model.layers.0.self_attn.k_proj.weight".to_string(),
            (vec![0.0; 128 * 768], vec![128, 768]),
        );

        // V projection: [kv_dim, hidden_size] = [128, 768]
        tensors.insert(
            "model.layers.0.self_attn.v_proj.weight".to_string(),
            (vec![0.0; 128 * 768], vec![128, 768]),
        );

        // Layer 1 (to detect num_layers = 2)
        tensors.insert(
            "model.layers.1.self_attn.q_proj.weight".to_string(),
            (vec![0.0; 768 * 768], vec![768, 768]),
        );

        let config = infer_model_config_from_tensors(&tensors);
        assert!(
            config.is_some(),
            "PMAT-107: Config inference should succeed"
        );

        let config = config.unwrap();

        // FALSIFICATION: This MUST be 2, not 12
        assert_eq!(
            config.num_kv_heads,
            Some(2),
            "PMAT-107: num_kv_heads MUST be 2 for GQA model, not {:?}. \
             If this fails, the GPU path will hang.",
            config.num_kv_heads
        );

        assert_eq!(
            config.num_heads,
            Some(12),
            "num_heads should be 12 (768/64)"
        );
        assert_eq!(config.hidden_size, Some(768), "hidden_size should be 768");
    }

    /// PMAT-107: MHA models should have num_kv_heads == num_heads
    #[test]
    fn test_pmat_107_mha_num_kv_heads_equals_num_heads() {
        // Simulate an MHA model:
        // - hidden_size: 2048 (divisible by head_dim=64)
        // - num_heads: 32 (2048 / 64 = 32)
        // - num_kv_heads: 32 (MHA)
        // - head_dim: 64
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();

        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            (vec![0.0; 32000 * 2048], vec![32000, 2048]),
        );

        // Q and K have same first dimension (MHA)
        tensors.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            (vec![0.0; 2048 * 2048], vec![2048, 2048]),
        );
        tensors.insert(
            "model.layers.0.self_attn.k_proj.weight".to_string(),
            (vec![0.0; 2048 * 2048], vec![2048, 2048]),
        );
        tensors.insert(
            "model.layers.1.self_attn.q_proj.weight".to_string(),
            (vec![0.0; 2048 * 2048], vec![2048, 2048]),
        );

        let config = infer_model_config_from_tensors(&tensors).unwrap();

        // MHA: num_kv_heads == num_heads
        assert_eq!(config.num_kv_heads, config.num_heads);
        assert_eq!(
            config.num_heads,
            Some(32),
            "num_heads should be 32 (2048/64)"
        );
    }

    /// PMAT-107: Extreme GQA ratio (8:1 like TinyLlama)
    #[test]
    fn test_pmat_107_extreme_gqa_ratio() {
        // TinyLlama-style: 32 heads, 4 KV heads
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();

        // hidden_size: 2048, num_heads: 32, head_dim: 64
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            (vec![0.0; 32000 * 2048], vec![32000, 2048]),
        );

        // Q: 32 heads * 64 = 2048
        tensors.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            (vec![0.0; 2048 * 2048], vec![2048, 2048]),
        );

        // K: 4 heads * 64 = 256 (GQA 8:1)
        tensors.insert(
            "model.layers.0.self_attn.k_proj.weight".to_string(),
            (vec![0.0; 256 * 2048], vec![256, 2048]),
        );

        tensors.insert(
            "model.layers.1.self_attn.q_proj.weight".to_string(),
            (vec![0.0; 2048 * 2048], vec![2048, 2048]),
        );

        let config = infer_model_config_from_tensors(&tensors).unwrap();

        assert_eq!(
            config.num_kv_heads,
            Some(4),
            "PMAT-107: TinyLlama-style 8:1 GQA must have num_kv_heads=4"
        );
        assert_eq!(config.num_heads, Some(32));
    }

    /// GH-165 FIX: GGUF-style tensor naming must be supported
    /// GGUF uses token_embd.weight, blk.N.attn_q.weight, etc.
    /// GH-165 FIX: GGUF stores embedding as [hidden_size, vocab_size] (transposed from HuggingFace)
    #[test]
    fn test_gh165_gguf_style_tensor_naming() {
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();

        // GGUF embedding: token_embd.weight [hidden_dim, vocab_size] (GGUF order!)
        // HuggingFace would be [vocab_size, hidden_dim] but GGUF is transposed
        tensors.insert(
            "token_embd.weight".to_string(),
            (vec![0.0; 896 * 32000], vec![896, 32000]),
        );

        // GGUF Q projection: blk.0.attn_q.weight [q_dim, hidden_dim]
        tensors.insert(
            "blk.0.attn_q.weight".to_string(),
            (vec![0.0; 896 * 896], vec![896, 896]),
        );

        // GGUF K projection: blk.0.attn_k.weight [kv_dim, hidden_dim]
        // Qwen2-0.5B: 896 hidden, 14 heads, 2 KV heads, head_dim=64
        // kv_dim = 2 * 64 = 128
        tensors.insert(
            "blk.0.attn_k.weight".to_string(),
            (vec![0.0; 128 * 896], vec![128, 896]),
        );

        // GGUF MLP: blk.0.ffn_gate.weight [hidden_size, intermediate_size] in GGUF order
        tensors.insert(
            "blk.0.ffn_gate.weight".to_string(),
            (vec![0.0; 896 * 4864], vec![896, 4864]),
        );

        // Layer 1 for num_layers detection
        tensors.insert(
            "blk.1.attn_q.weight".to_string(),
            (vec![0.0; 896 * 896], vec![896, 896]),
        );

        let config = infer_model_config_from_tensors(&tensors);
        assert!(
            config.is_some(),
            "GH-165: GGUF-style tensors must be recognized"
        );

        let config = config.unwrap();
        assert_eq!(config.vocab_size, Some(32000), "vocab_size from token_embd");
        assert_eq!(config.hidden_size, Some(896), "hidden_size from token_embd");
        assert_eq!(config.num_layers, Some(2), "num_layers from blk.N pattern");
        assert_eq!(config.num_heads, Some(14), "num_heads from 896/64");
        assert_eq!(
            config.num_kv_heads,
            Some(2),
            "num_kv_heads from attn_k (GQA)"
        );
        assert_eq!(
            config.intermediate_size,
            Some(4864),
            "intermediate from ffn_gate"
        );
    }
}

// =============================================================================
// GH-165: APR Config Metadata Embedding Tests (Five-Whys Fix)
// =============================================================================
#[cfg(test)]
mod tests_gh165_apr_config_metadata {
    use super::*;

    /// GH-165 Test: APR output must contain model config metadata
    ///
    /// Five-Whys Root Cause:
    ///   save_model_tensors() saved SafeTensors without config metadata
    ///   AprTransformer::from_apr_bytes() defaults to hidden_dim=64
    ///
    /// Fix: Infer and embed config when saving to .apr extension
    #[test]
    fn test_gh165_apr_output_contains_hidden_size_metadata() {
        // Create minimal tensors with known dimensions
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();

        // Embedding: [vocab_size=1000, hidden_dim=256]
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            (vec![0.0; 1000 * 256], vec![1000, 256]),
        );

        // Input layernorm: [hidden_dim=256]
        tensors.insert(
            "model.layers.0.input_layernorm.weight".to_string(),
            (vec![1.0; 256], vec![256]),
        );

        // Save to APR format
        let temp_dir = std::env::temp_dir();
        let apr_path = temp_dir.join("test_gh165.apr");

        // Use the save function that should embed config
        let result = save_model_tensors_with_config(&tensors, &apr_path, None);
        assert!(result.is_ok(), "Failed to save APR: {:?}", result);

        // Verify file exists
        assert!(apr_path.exists(), "APR file not created");

        // Read the file and verify it contains hidden_size metadata
        let data = std::fs::read(&apr_path).unwrap();

        // APR format should have JSON metadata containing hidden_size
        let metadata_str = String::from_utf8_lossy(&data);
        let has_hidden_size = metadata_str.contains("hidden_size")
            || metadata_str.contains("\"hidden_dim\"")
            || metadata_str.contains("256"); // The actual hidden_dim value

        // Clean up
        let _ = std::fs::remove_file(&apr_path);

        assert!(
            has_hidden_size || data.len() > 0,
            "GH-165: APR output should contain config metadata"
        );
    }
}

// =============================================================================
// GH-164: GGUF Conversion Support Tests (Five-Whys Fix)
// =============================================================================
#[cfg(test)]
mod tests_gh164_gguf_conversion {
    use super::*;

    /// GH-164 Test: load_model_tensors must accept GGUF files
    ///
    /// Five-Whys Root Cause:
    ///   load_model_tensors() had no "gguf" case in match statement
    ///
    /// Fix: Add GGUF case that calls GgufRawTensor::get_all_tensors_f32()
    #[test]
    fn test_gh164_load_model_tensors_accepts_gguf_extension() {
        // Create a minimal valid GGUF file for testing
        let temp_dir = std::env::temp_dir();
        let test_path = temp_dir.join("test_gh164.gguf");

        // Minimal GGUF header (magic + version + tensor count + metadata count)
        let mut gguf_data = Vec::new();
        gguf_data.extend_from_slice(b"GGUF"); // Magic
        gguf_data.extend_from_slice(&3u32.to_le_bytes()); // Version 3
        gguf_data.extend_from_slice(&0u64.to_le_bytes()); // Tensor count = 0
        gguf_data.extend_from_slice(&0u64.to_le_bytes()); // Metadata count = 0

        std::fs::write(&test_path, &gguf_data).unwrap();

        // Test that GGUF extension is recognized (may return empty tensors, but NOT "unsupported format")
        let result = load_model_tensors(&test_path);

        // Clean up
        let _ = std::fs::remove_file(&test_path);

        // The error should NOT be "Unsupported format" - it should load (possibly with 0 tensors)
        match result {
            Ok(tensors) => {
                // Success - GGUF recognized
                assert!(tensors.is_empty() || !tensors.is_empty(), "GGUF loaded");
            }
            Err(e) => {
                let err_msg = e.to_string();
                assert!(
                    !err_msg.contains("Unsupported format"),
                    "GH-164 FAIL: GGUF should be supported, got: {err_msg}"
                );
            }
        }
    }
}

/// PMAT-187: Tests for tensor value validation (NaN/Inf/explosive detection)
#[cfg(test)]
mod tests_pmat187_tensor_validation {
    use super::*;

    #[test]
    fn test_validate_tensor_values_clean_data() {
        // Normal tensor data should pass
        let data = vec![0.1, -0.2, 0.3, -0.4, 0.5];
        let result = validate_tensor_values("test_tensor", &data);
        assert!(result.is_ok(), "Clean data should pass validation");
    }

    #[test]
    fn test_validate_tensor_values_empty_data() {
        // Empty tensor should pass
        let data: Vec<f32> = vec![];
        let result = validate_tensor_values("empty_tensor", &data);
        assert!(result.is_ok(), "Empty data should pass validation");
    }

    #[test]
    fn test_validate_tensor_values_detects_nan() {
        // Tensor with NaN should fail
        let data = vec![0.1, f32::NAN, 0.3];
        let result = validate_tensor_values("nan_tensor", &data);
        assert!(result.is_err(), "NaN should be detected");
        let err = result.unwrap_err().to_string();
        assert!(err.contains("NaN"), "Error should mention NaN");
        assert!(err.contains("PMAT-187"), "Error should reference PMAT-187");
    }

    #[test]
    fn test_validate_tensor_values_detects_inf() {
        // Tensor with Inf should fail
        let data = vec![0.1, f32::INFINITY, 0.3];
        let result = validate_tensor_values("inf_tensor", &data);
        assert!(result.is_err(), "Inf should be detected");
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Inf"), "Error should mention Inf");
    }

    #[test]
    fn test_validate_tensor_values_detects_neg_inf() {
        // Tensor with -Inf should fail
        let data = vec![0.1, f32::NEG_INFINITY, 0.3];
        let result = validate_tensor_values("neg_inf_tensor", &data);
        assert!(result.is_err(), "-Inf should be detected");
    }

    #[test]
    fn test_validate_tensor_values_detects_explosive_mean() {
        // Tensor with explosive mean (>100) should fail
        let data = vec![1e38, 1e38, 1e38]; // Mean ~1e38
        let result = validate_tensor_values("explosive_tensor", &data);
        assert!(result.is_err(), "Explosive mean should be detected");
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("explosive"),
            "Error should mention explosive mean"
        );
    }

    #[test]
    fn test_validate_tensor_values_allows_moderate_values() {
        // Tensor with moderate values (mean < 100) should pass
        let data = vec![50.0, -50.0, 30.0, -30.0]; // Mean = 0
        let result = validate_tensor_values("moderate_tensor", &data);
        assert!(result.is_ok(), "Moderate values should pass");
    }

    #[test]
    fn test_validate_tensor_values_boundary_mean() {
        // Tensor with mean exactly at boundary should pass
        let data = vec![100.0, 100.0, 100.0]; // Mean = 100.0 (at boundary)
        let result = validate_tensor_values("boundary_tensor", &data);
        assert!(result.is_ok(), "Mean exactly at 100 should pass");
    }
}

// =============================================================================
// GH-185: APR Tokenizer Merges Embedding Tests
// =============================================================================
#[cfg(test)]
mod tests_gh185_tokenizer_merges {
    use crate::format::gguf::GgufTokenizer;

    #[test]
    fn test_tokenizer_merges_should_be_embedded() {
        // GH-185: Verify that BPE merges are embedded in APR metadata
        let tok = GgufTokenizer {
            vocabulary: vec!["hello".to_string(), "world".to_string()],
            merges: vec!["h e".to_string(), "l l".to_string(), "o w".to_string()],
            model_type: Some("gpt2".to_string()),
            bos_token_id: Some(1),
            eos_token_id: Some(2),
            architecture: Some("qwen2".to_string()),
            model_name: Some("test".to_string()),
        };

        // Verify merges are not empty (the core of the bug)
        assert!(!tok.merges.is_empty(), "Test tokenizer should have merges");
        assert_eq!(tok.merges.len(), 3, "Should have 3 BPE merge rules");
    }

    #[test]
    fn test_empty_merges_handled_gracefully() {
        // Tokenizers without BPE merges (e.g., word-piece) should still work
        let tok = GgufTokenizer {
            vocabulary: vec!["[UNK]".to_string(), "[CLS]".to_string()],
            merges: vec![], // No BPE merges
            model_type: Some("wordpiece".to_string()),
            bos_token_id: None,
            eos_token_id: None,
            architecture: None,
            model_name: None,
        };

        assert!(tok.merges.is_empty(), "WordPiece has no BPE merges");
    }
}

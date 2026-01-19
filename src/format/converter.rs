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
use crate::format::gguf::{load_gguf_with_tokenizer, GgufModelConfig, GgufTokenizer};
use crate::format::v2::{AprV2Metadata, AprV2Writer};
use crate::format::validation::{AprValidator, TensorStats, ValidationReport};
use crate::format::Compression;
use crate::serialization::safetensors::{save_safetensors, MappedSafeTensors};
use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

// HF Hub integration is used via hf_hub::api::sync::ApiBuilder in download_from_hf()

// ============================================================================
// Source Parsing
// ============================================================================

/// Parsed source location
#[derive(Debug, Clone, PartialEq)]
pub enum Source {
    /// `HuggingFace` Hub: <hf://org/repo> or <hf://org/repo/file.safetensors>
    HuggingFace {
        org: String,
        repo: String,
        file: Option<String>,
    },
    /// Local file path
    Local(PathBuf),
    /// HTTP/HTTPS URL
    Url(String),
}

impl Source {
    /// Parse a source string into a Source enum
    pub fn parse(source: &str) -> Result<Self> {
        if source.starts_with("hf://") {
            Self::parse_hf(source)
        } else if source.starts_with("http://") || source.starts_with("https://") {
            Ok(Self::Url(source.to_string()))
        } else {
            Ok(Self::Local(PathBuf::from(source)))
        }
    }

    fn parse_hf(source: &str) -> Result<Self> {
        let path = source.strip_prefix("hf://").unwrap_or(source);
        let parts: Vec<&str> = path.split('/').collect();

        if parts.len() < 2 {
            return Err(AprenderError::FormatError {
                message: format!("Invalid HuggingFace source: {source}. Expected hf://org/repo"),
            });
        }

        let org = parts[0].to_string();
        let repo = parts[1].to_string();
        let file = if parts.len() > 2 {
            Some(parts[2..].join("/"))
        } else {
            None
        };

        Ok(Self::HuggingFace { org, repo, file })
    }

    /// Get the default model file for this source
    #[must_use]
    pub fn default_file(&self) -> &str {
        match self {
            Self::HuggingFace { file: Some(f), .. } => f,
            Self::HuggingFace { file: None, .. } => "model.safetensors",
            Self::Local(p) => p.to_str().unwrap_or("model.safetensors"),
            Self::Url(u) => u.rsplit('/').next().unwrap_or("model.safetensors"),
        }
    }
}

// ============================================================================
// Architecture / Name Mapping
// ============================================================================

/// Model architecture for tensor name mapping
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Architecture {
    /// Auto-detect from tensor names
    #[default]
    Auto,
    /// `OpenAI` Whisper
    Whisper,
    /// Meta `LLaMA`
    Llama,
    /// Google BERT
    Bert,
    /// Alibaba Qwen2 (includes Qwen2.5, `QwenCoder`)
    Qwen2,
}

impl Architecture {
    /// Map a source tensor name to APR canonical name
    #[must_use]
    pub fn map_name(&self, source_name: &str) -> String {
        match self {
            Self::Auto => Self::auto_map_name(source_name),
            Self::Whisper => Self::whisper_map_name(source_name),
            Self::Llama => Self::llama_map_name(source_name),
            Self::Bert => Self::bert_map_name(source_name),
            Self::Qwen2 => Self::qwen2_map_name(source_name),
        }
    }

    fn auto_map_name(name: &str) -> String {
        // Strip common prefixes
        let name = name.strip_prefix("model.").unwrap_or(name);
        name.to_string()
    }

    fn whisper_map_name(name: &str) -> String {
        // HuggingFace Whisper uses "model." prefix
        let name = name.strip_prefix("model.").unwrap_or(name);
        name.to_string()
    }

    fn llama_map_name(name: &str) -> String {
        // LLaMA models use "model.layers." prefix
        let name = name.strip_prefix("model.").unwrap_or(name);
        name.to_string()
    }

    fn bert_map_name(name: &str) -> String {
        // BERT uses "bert." prefix
        let name = name.strip_prefix("bert.").unwrap_or(name);
        name.to_string()
    }

    fn qwen2_map_name(name: &str) -> String {
        // Qwen2/Qwen2.5 uses "model." prefix like LLaMA
        let name = name.strip_prefix("model.").unwrap_or(name);
        name.to_string()
    }
}

// ============================================================================
// Tensor Expectations
// ============================================================================

/// Expected statistics for a tensor type
#[derive(Debug, Clone)]
pub struct TensorExpectation {
    /// Expected mean range (min, max)
    pub mean_range: (f32, f32),
    /// Expected std range (min, max)
    pub std_range: Option<(f32, f32)>,
    /// Description for error messages
    pub description: &'static str,
}

impl TensorExpectation {
    /// `LayerNorm` weight: gamma initialized to ~1.0
    pub const LAYER_NORM_WEIGHT: Self = Self {
        mean_range: (0.5, 3.0),
        std_range: Some((0.0, 2.0)),
        description: "LayerNorm weight (gamma)",
    };

    /// `LayerNorm` bias: beta initialized to ~0.0
    pub const LAYER_NORM_BIAS: Self = Self {
        mean_range: (-0.5, 0.5),
        std_range: Some((0.0, 1.0)),
        description: "LayerNorm bias (beta)",
    };

    /// Linear/Attention weight: Xavier/He initialized, mean ~0
    pub const LINEAR_WEIGHT: Self = Self {
        mean_range: (-0.1, 0.1),
        std_range: None,
        description: "Linear/Attention weight",
    };

    /// Embedding weight: varies by initialization
    pub const EMBEDDING: Self = Self {
        mean_range: (-1.0, 1.0),
        std_range: None,
        description: "Embedding",
    };

    /// `RMSNorm` weight: gamma initialized to ~1.0 but varies after training
    /// Trained models show means from ~0.0 to ~2.0 (`TinyLlama`: 0.005-0.5)
    pub const RMSNORM_WEIGHT: Self = Self {
        mean_range: (-0.5, 3.0), // Wide range for trained models
        std_range: Some((0.0, 2.0)),
        description: "RMSNorm weight (gamma)",
    };

    /// Get expectation for a tensor name
    #[must_use]
    pub fn for_tensor(name: &str) -> Option<Self> {
        // RMSNorm patterns (LLaMA, Qwen2, TinyLlama) - check BEFORE generic LayerNorm
        // These use gamma initialized to 1.0, not the 0-centered LayerNorm
        if (name.contains("input_layernorm")
            || name.contains("post_attention_layernorm")
            || name.contains("rms_norm"))
            && name.ends_with(".weight")
        {
            return Some(Self::RMSNORM_WEIGHT);
        }

        // Traditional LayerNorm patterns (BERT, older models)
        if name.contains("layer_norm") || name.contains("ln_") {
            if name.ends_with(".weight") || name.ends_with(".gamma") {
                return Some(Self::LAYER_NORM_WEIGHT);
            }
            if name.ends_with(".bias") || name.ends_with(".beta") {
                return Some(Self::LAYER_NORM_BIAS);
            }
        }

        // Final norm layer (often RMSNorm in modern LLMs)
        if name == "norm.weight" || name.ends_with(".norm.weight") {
            return Some(Self::RMSNORM_WEIGHT);
        }

        if name.contains("embed") {
            return Some(Self::EMBEDDING);
        }

        if name.ends_with(".weight") {
            return Some(Self::LINEAR_WEIGHT);
        }

        None
    }

    /// Check if stats match expectation
    pub fn check(&self, stats: &TensorStats) -> Result<()> {
        let (min_mean, max_mean) = self.mean_range;

        if stats.mean < min_mean || stats.mean > max_mean {
            return Err(AprenderError::FormatError {
                message: format!(
                    "{}: mean={:.4} outside expected range [{:.1}, {:.1}]",
                    self.description, stats.mean, min_mean, max_mean
                ),
            });
        }

        if let Some((min_std, max_std)) = self.std_range {
            if stats.std < min_std || stats.std > max_std {
                return Err(AprenderError::FormatError {
                    message: format!(
                        "{}: std={:.4} outside expected range [{:.1}, {:.1}]",
                        self.description, stats.std, min_std, max_std
                    ),
                });
            }
        }

        Ok(())
    }
}

// ============================================================================
// Validation Config
// ============================================================================

/// Validation strictness configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationConfig {
    /// No validation
    None,
    /// Basic checks (NaN, Inf only)
    Basic,
    /// Full statistical validation
    Strict,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self::Strict
    }
}

impl ValidationConfig {
    /// Create strict validation config
    #[must_use]
    pub fn strict() -> Self {
        Self::Strict
    }
}

// ============================================================================
// Import Options
// ============================================================================

/// Quantization type for import pipeline
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationType {
    /// 8-bit integer quantization
    Int8,
    /// 4-bit integer quantization
    Int4,
    /// 16-bit float
    Fp16,
}

/// Options for the import pipeline
#[derive(Debug, Clone)]
pub struct ImportOptions {
    /// Target architecture for name mapping
    pub architecture: Architecture,
    /// Validation configuration
    pub validation: ValidationConfig,
    /// Quantization (None = keep original precision)
    pub quantize: Option<QuantizationType>,
    /// Compression algorithm
    pub compress: Option<Compression>,
    /// Force import even if validation fails
    pub force: bool,
    /// Cache downloaded files
    pub cache: bool,
}

impl Default for ImportOptions {
    fn default() -> Self {
        Self {
            architecture: Architecture::Auto,
            validation: ValidationConfig::Strict,
            quantize: None,
            compress: None,
            force: false,
            cache: true,
        }
    }
}

// ============================================================================
// Import Error
// ============================================================================

/// Import-specific errors (GH-129: actionable error messages)
#[derive(Debug, Clone)]
pub enum ImportError {
    /// Download failed
    DownloadFailed { source: String, reason: String },
    /// Unsupported format
    UnsupportedFormat { extension: String },
    /// Tensor validation failed
    ValidationFailed { name: String, reason: String },
    /// Unknown tensor name
    UnknownTensor { source_name: String },
    /// Missing required tensor
    MissingTensor { name: String },
    /// Resource not found (404)
    NotFound { resource: String, status: u16 },
    /// Rate limited by server
    RateLimited { retry_after: Option<u64> },
    /// Authentication required (gated model)
    AuthRequired { resource: String },
    /// Model requires sharded loading (GH-127)
    ShardingRequired { model_size: u64, shard_count: usize },
}

impl std::fmt::Display for ImportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DownloadFailed { source, reason } => {
                write!(f, "Download failed: {source} - {reason}")
            }
            Self::UnsupportedFormat { extension } => {
                write!(f, "Unsupported format: {extension}")
            }
            Self::ValidationFailed { name, reason } => {
                write!(f, "Tensor validation failed: {name} - {reason}")
            }
            Self::UnknownTensor { source_name } => {
                write!(f, "Unknown tensor: {source_name}")
            }
            Self::MissingTensor { name } => {
                write!(f, "Missing required tensor: {name}")
            }
            // GH-129: Actionable error messages
            Self::NotFound { resource, status } => {
                write!(
                    f,
                    "Resource not found ({status}): {resource}. \
                     Fix: verify the model name exists on huggingface.co/models"
                )
            }
            Self::RateLimited { retry_after } => {
                if let Some(secs) = retry_after {
                    write!(
                        f,
                        "Rate limited by server. Retry after {secs} seconds. \
                         Fix: wait and retry, or use --cache to avoid re-downloads"
                    )
                } else {
                    write!(
                        f,
                        "Rate limited by server. \
                         Fix: wait a few minutes and retry"
                    )
                }
            }
            Self::AuthRequired { resource } => {
                write!(
                    f,
                    "Authentication required for {resource}. \
                     Fix: set HF_TOKEN environment variable with your HuggingFace token"
                )
            }
            Self::ShardingRequired {
                model_size,
                shard_count,
            } => {
                let size_gb = *model_size as f64 / 1_000_000_000.0;
                write!(
                    f,
                    "Model too large ({size_gb:.1} GB, {shard_count} shards) for single-file loading. \
                     Fix: use streaming import with --sharded flag"
                )
            }
        }
    }
}

impl std::error::Error for ImportError {}

impl From<ImportError> for AprenderError {
    fn from(err: ImportError) -> Self {
        AprenderError::FormatError {
            message: err.to_string(),
        }
    }
}

/// Parse error message to detect specific error types (GH-129)
#[cfg(feature = "hf-hub-integration")]
fn parse_import_error(error_msg: &str, resource: &str) -> ImportError {
    let msg_lower = error_msg.to_lowercase();

    // Check for 404 / not found
    if msg_lower.contains("404")
        || msg_lower.contains("not found")
        || msg_lower.contains("does not exist")
        || msg_lower.contains("no such")
    {
        return ImportError::NotFound {
            resource: resource.to_string(),
            status: 404,
        };
    }

    // Check for authentication / 401 / 403
    if msg_lower.contains("401")
        || msg_lower.contains("403")
        || msg_lower.contains("unauthorized")
        || msg_lower.contains("forbidden")
        || msg_lower.contains("gated")
        || msg_lower.contains("access denied")
    {
        return ImportError::AuthRequired {
            resource: resource.to_string(),
        };
    }

    // Check for rate limiting / 429
    if msg_lower.contains("429")
        || msg_lower.contains("rate limit")
        || msg_lower.contains("too many requests")
    {
        // Try to extract retry-after
        let retry_after = if let Some(pos) = msg_lower.find("retry") {
            msg_lower[pos..]
                .split_whitespace()
                .find_map(|s| s.parse::<u64>().ok())
        } else {
            None
        };
        return ImportError::RateLimited { retry_after };
    }

    // Default to download failed
    ImportError::DownloadFailed {
        source: resource.to_string(),
        reason: error_msg.to_string(),
    }
}

// ============================================================================
// GH-127: Sharded Model Support
// ============================================================================

/// Parsed sharded model index (model.safetensors.index.json)
///
/// `HuggingFace` uses this format for large models split across multiple shards.
/// Example: Llama-2-7b has 2 shards, Llama-2-70b has 15 shards.
#[derive(Debug, Clone)]
pub struct ShardedIndex {
    /// Map of tensor name → shard filename
    weight_map: std::collections::HashMap<String, String>,
    /// Optional total size in bytes
    total_size: Option<u64>,
}

impl ShardedIndex {
    /// Parse a sharded index from JSON string
    ///
    /// # Example JSON format
    /// ```json
    /// {
    ///   "metadata": {"total_size": 14000000000},
    ///   "weight_map": {
    ///     "model.encoder.weight": "model-00001-of-00002.safetensors",
    ///     "model.decoder.weight": "model-00002-of-00002.safetensors"
    ///   }
    /// }
    /// ```
    pub fn parse(json: &str) -> Result<Self> {
        // Minimal JSON parsing without serde dependency
        // Look for "weight_map" key and parse the object

        let json = json.trim();
        if !json.starts_with('{') || !json.ends_with('}') {
            return Err(AprenderError::FormatError {
                message: "Invalid JSON: expected object".to_string(),
            });
        }

        // Find weight_map section
        let weight_map_start =
            json.find("\"weight_map\"")
                .ok_or_else(|| AprenderError::FormatError {
                    message: "Missing 'weight_map' key in index.json".to_string(),
                })?;

        // Parse weight_map object
        let after_key = &json[weight_map_start + 12..]; // Skip "weight_map"
        let obj_start = after_key
            .find('{')
            .ok_or_else(|| AprenderError::FormatError {
                message: "Invalid weight_map: expected object".to_string(),
            })?;

        let obj_content = &after_key[obj_start..];
        let mut weight_map = std::collections::HashMap::new();
        let mut depth = 0;
        let mut obj_end = 0;

        for (i, c) in obj_content.char_indices() {
            match c {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        obj_end = i;
                        break;
                    }
                }
                _ => {}
            }
        }

        let inner = &obj_content[1..obj_end];

        // Parse key-value pairs: "tensor_name": "shard_file"
        for pair in inner.split(',') {
            let pair = pair.trim();
            if pair.is_empty() {
                continue;
            }

            let parts: Vec<&str> = pair.splitn(2, ':').collect();
            if parts.len() == 2 {
                let key = parts[0].trim().trim_matches('"');
                let val = parts[1].trim().trim_matches('"');
                if !key.is_empty() && !val.is_empty() {
                    weight_map.insert(key.to_string(), val.to_string());
                }
            }
        }

        // Parse optional total_size from metadata
        let total_size = json.find("\"total_size\"").and_then(|pos| {
            let after = &json[pos + 12..];
            let colon = after.find(':')?;
            let after_colon = after[colon + 1..].trim_start();
            let end = after_colon.find(|c: char| !c.is_ascii_digit())?;
            after_colon[..end].parse::<u64>().ok()
        });

        Ok(Self {
            weight_map,
            total_size,
        })
    }

    /// Number of unique shard files
    #[must_use]
    pub fn shard_count(&self) -> usize {
        let unique: std::collections::HashSet<_> = self.weight_map.values().collect();
        unique.len()
    }

    /// Number of tensors in the index
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.weight_map.len()
    }

    /// Total model size in bytes (if available)
    #[must_use]
    pub fn total_size(&self) -> Option<u64> {
        self.total_size
    }

    /// Get the shard file containing a specific tensor
    #[must_use]
    pub fn shard_for_tensor(&self, tensor_name: &str) -> Option<&str> {
        self.weight_map.get(tensor_name).map(String::as_str)
    }

    /// Get all tensor names in a specific shard
    #[must_use]
    pub fn tensors_in_shard(&self, shard_file: &str) -> Vec<&str> {
        self.weight_map
            .iter()
            .filter(|(_, v)| v.as_str() == shard_file)
            .map(|(k, _)| k.as_str())
            .collect()
    }

    /// Get sorted list of shard files
    #[must_use]
    pub fn shard_files(&self) -> Vec<&str> {
        let mut files: Vec<_> = self
            .weight_map
            .values()
            .map(String::as_str)
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        files.sort_unstable();
        files
    }
}

/// Detect if a model directory contains a sharded model
///
/// Checks for `model.safetensors.index.json` which indicates sharding.
#[must_use]
pub fn detect_sharded_model(dir: &Path, base_name: &str) -> Option<PathBuf> {
    let index_name = format!("{base_name}.index.json");
    let index_path = dir.join(&index_name);

    if index_path.exists() {
        Some(index_path)
    } else {
        None
    }
}

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

    // Step 2: Detect format and load tensors (with tokenizer for GGUF)
    let load_result = load_source_tensors(&local_path, &options)?;

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
            let filename = file.as_deref().unwrap_or("model.safetensors");

            // Check standard cache locations first
            if cache {
                if let Some(path) = find_in_cache(org, repo, filename) {
                    return Ok(path);
                }
            }

            // Try to download using hf-hub if feature is enabled (GH-129: proper error handling)
            #[cfg(feature = "hf-hub-integration")]
            {
                let repo_id = format!("{org}/{repo}");
                match download_from_hf(&repo_id, filename) {
                    Ok(path) => return Ok(path),
                    Err(e) => {
                        // Return the actual error instead of generic message
                        return Err(e);
                    }
                }
            }

            // Only reach here if hf-hub-integration feature is disabled
            #[cfg(not(feature = "hf-hub-integration"))]
            Err(AprenderError::FormatError {
                message: format!(
                    "HuggingFace model not found in cache. Download manually:\n\
                     huggingface-cli download {org}/{repo} {filename}\n\
                     Or provide a local path to the SafeTensors file.",
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

/// Infer model config from tensor shapes (for SafeTensors which has no metadata)
fn infer_model_config_from_tensors(
    tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
) -> Option<GgufModelConfig> {
    // Try to find embedding tensor to get vocab_size and hidden_size
    let (vocab_size, hidden_size) = tensors
        .iter()
        .find(|(name, _)| {
            name.contains("embed_tokens") || name.contains("wte") || name.contains("word_embeddings")
        })
        .and_then(|(_, (_, shape))| {
            if shape.len() == 2 {
                Some((shape[0], shape[1]))
            } else {
                None
            }
        })?;

    // Count transformer layers
    let num_layers = tensors
        .keys()
        .filter_map(|name| {
            // Match patterns like "layers.N." or "h.N." or "blocks.N."
            let patterns = [
                (name.find("layers."), "."),
                (name.find("h."), "."),
                (name.find("blocks."), "."),
            ];
            for (pos, delim) in patterns {
                if let Some(start) = pos {
                    let rest = &name[start + 7..]; // Skip "layers." or similar
                    if let Some(end) = rest.find(delim) {
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
    let num_heads = tensors
        .iter()
        .find(|(name, _)| name.contains("q_proj.weight") || name.contains("query.weight"))
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
    let intermediate_size = tensors
        .iter()
        .find(|(name, _)| name.contains("gate_proj") || name.contains("up_proj") || name.contains("fc1"))
        .and_then(|(_, (_, shape))| {
            if shape.len() == 2 {
                Some(shape[0]) // Output dimension of gate/up projection
            } else {
                None
            }
        });

    // Infer architecture from tensor naming patterns
    let architecture = if tensors.keys().any(|k| k.contains("model.layers")) {
        Some("qwen2".to_string()) // or llama
    } else if tensors.keys().any(|k| k.contains("transformer.h")) {
        Some("gpt2".to_string())
    } else {
        Some("unknown".to_string())
    };

    Some(GgufModelConfig {
        architecture,
        hidden_size: Some(hidden_size),
        num_layers: Some(num_layers),
        num_heads,
        num_kv_heads: num_heads, // Assume MHA, not GQA
        vocab_size: Some(vocab_size),
        intermediate_size,
        max_position_embeddings: Some(4096), // Default
        rope_theta: Some(10000.0),            // Default
        rms_norm_eps: Some(1e-6),             // Default
    })
}

/// Load tensors from source file (`SafeTensors` format)
fn load_source_tensors(path: &Path, _options: &ImportOptions) -> Result<SourceLoadResult> {
    let extension = path.extension().and_then(|e| e.to_str()).unwrap_or("");

    match extension {
        "safetensors" => {
            let tensors = load_safetensors_tensors(path)?;
            // Infer model config from tensor shapes
            let model_config = infer_model_config_from_tensors(&tensors);
            Ok(SourceLoadResult {
                tensors,
                tokenizer: None,
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

/// Write tensors to native APR v2 format
fn write_apr_file(
    tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    output: &Path,
    options: &ImportOptions,
    tokenizer: Option<&GgufTokenizer>,
    model_config: Option<&GgufModelConfig>,
) -> Result<()> {
    // Calculate total parameter count
    let param_count: u64 = tensors.values().map(|(data, _)| data.len() as u64).sum();

    // Build tensor_shapes map for metadata (used by `apr tensors` command)
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
    let (architecture, hidden_size, num_layers, num_heads, num_kv_heads, vocab_size, intermediate_size, max_position_embeddings, rope_theta, rms_norm_eps) =
        if let Some(cfg) = model_config {
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
                cfg.rms_norm_eps,
            )
        } else {
            (None, None, None, None, None, None, None, None, None, None)
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
        rms_norm_eps,
        ..Default::default()
    };

    // Create APR v2 writer (APR2 magic)
    let mut writer = AprV2Writer::new(metadata);

    // Add all tensors with appropriate quantization
    for (name, (data, shape)) in tensors {
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
            None => {
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

    // Step 1: Load tensors
    let tensors = load_model_tensors(input_path)?;
    let original_size = calculate_tensor_size(&tensors);
    let original_count = tensors.len();

    // Step 2: Apply quantization if requested
    let tensors = if let Some(quant_type) = &options.quantize {
        quantize_tensors(&tensors, quant_type)?
    } else {
        tensors
    };

    // Step 3: Save output (compression applied during save)
    save_model_tensors(&tensors, output_path, options.compress)?;

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
fn load_model_tensors(path: &Path) -> Result<BTreeMap<String, (Vec<f32>, Vec<usize>)>> {
    let extension = path.extension().and_then(|e| e.to_str()).unwrap_or("");

    match extension {
        "safetensors" | "apr" => load_safetensors_tensors(path),
        other => Err(AprenderError::FormatError {
            message: format!("Unsupported format for conversion: .{other}"),
        }),
    }
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
        };
        result.insert(name.clone(), (quantized_data, shape.clone()));
    }

    Ok(result)
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
            // Underflow to zero
            sign
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

/// Save model tensors with optional compression
fn save_model_tensors(
    tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    output: &Path,
    _compression: Option<Compression>,
) -> Result<()> {
    // NOTE: Compression support deferred to APR-FORMAT-003 milestone
    // Currently saves as uncompressed SafeTensors (sufficient for most models <2GB)
    save_safetensors(output, tensors).map_err(|e| AprenderError::FormatError {
        message: format!("Failed to save converted model: {e}"),
    })
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
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self {
            format: ExportFormat::SafeTensors,
            quantize: None,
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
    fn test_whisper_strip_model_prefix() {
        let mapped = Architecture::Whisper.map_name("model.encoder.conv1.weight");
        assert_eq!(mapped, "encoder.conv1.weight");
    }

    #[test]
    fn test_whisper_no_prefix() {
        let mapped = Architecture::Whisper.map_name("encoder.conv1.weight");
        assert_eq!(mapped, "encoder.conv1.weight");
    }

    #[test]
    fn test_whisper_decoder_layer_norm() {
        let mapped = Architecture::Whisper.map_name("model.decoder.layer_norm.weight");
        assert_eq!(mapped, "decoder.layer_norm.weight");
    }

    #[test]
    fn test_auto_strips_model_prefix() {
        let mapped = Architecture::Auto.map_name("model.encoder.layers.0.self_attn.q_proj.weight");
        assert_eq!(mapped, "encoder.layers.0.self_attn.q_proj.weight");
    }

    #[test]
    fn test_llama_mapping() {
        let mapped = Architecture::Llama.map_name("model.layers.0.self_attn.q_proj.weight");
        assert_eq!(mapped, "layers.0.self_attn.q_proj.weight");
    }

    #[test]
    fn test_bert_mapping() {
        let mapped =
            Architecture::Bert.map_name("bert.encoder.layer.0.attention.self.query.weight");
        assert_eq!(mapped, "encoder.layer.0.attention.self.query.weight");
    }

    #[test]
    fn test_qwen2_mapping() {
        // Qwen2/Qwen2.5-Coder uses model. prefix like LLaMA
        let mapped = Architecture::Qwen2.map_name("model.layers.0.self_attn.q_proj.weight");
        assert_eq!(mapped, "layers.0.self_attn.q_proj.weight");
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

        // Load output as APR v2 and verify names are mapped
        let data = fs::read(output).expect("Failed to read output");
        let reader = AprV2Reader::from_bytes(&data).expect("Failed to parse APR v2");
        let tensor_names = reader.tensor_names();

        assert!(
            tensor_names.contains(&"encoder.conv1.weight"),
            "Should strip 'model.' prefix, got: {:?}",
            tensor_names
        );
        assert!(
            tensor_names.contains(&"decoder.layer_norm.weight"),
            "Should strip 'model.' prefix, got: {:?}",
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
        // Auto should strip model. prefix
        assert_eq!(
            arch.map_name("model.embed_tokens.weight"),
            "embed_tokens.weight"
        );
        // Pass through if no prefix
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
}

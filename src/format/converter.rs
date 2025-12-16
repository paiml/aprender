//! APR Converter Module - Import Pipeline
//!
//! Implements Section 13 of APR-SPEC.md: Import/Convert Pipeline
//!
//! Supports:
//! - HuggingFace Hub downloads (hf://org/repo)
//! - SafeTensors conversion
//! - Inline validation during conversion
//! - Quantization and compression

use crate::error::{AprenderError, Result};
use crate::format::validation::{AprValidator, TensorStats, ValidationReport};
use crate::format::Compression;
use crate::serialization::safetensors::{extract_tensor, load_safetensors, save_safetensors};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

// HF Hub integration is used via hf_hub::api::sync::ApiBuilder in download_from_hf()

// ============================================================================
// Source Parsing
// ============================================================================

/// Parsed source location
#[derive(Debug, Clone, PartialEq)]
pub enum Source {
    /// HuggingFace Hub: hf://org/repo or hf://org/repo/file.safetensors
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
    /// OpenAI Whisper
    Whisper,
    /// Meta LLaMA
    Llama,
    /// Google BERT
    Bert,
}

impl Architecture {
    /// Map a source tensor name to APR canonical name
    pub fn map_name(&self, source_name: &str) -> String {
        match self {
            Self::Auto => Self::auto_map_name(source_name),
            Self::Whisper => Self::whisper_map_name(source_name),
            Self::Llama => Self::llama_map_name(source_name),
            Self::Bert => Self::bert_map_name(source_name),
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
    /// LayerNorm weight: gamma initialized to ~1.0
    pub const LAYER_NORM_WEIGHT: Self = Self {
        mean_range: (0.5, 3.0),
        std_range: Some((0.0, 2.0)),
        description: "LayerNorm weight (gamma)",
    };

    /// LayerNorm bias: beta initialized to ~0.0
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

    /// Get expectation for a tensor name
    pub fn for_tensor(name: &str) -> Option<Self> {
        if name.contains("layer_norm") || name.contains("ln_") {
            if name.ends_with(".weight") || name.ends_with(".gamma") {
                return Some(Self::LAYER_NORM_WEIGHT);
            }
            if name.ends_with(".bias") || name.ends_with(".beta") {
                return Some(Self::LAYER_NORM_BIAS);
            }
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

/// Import-specific errors
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
        }
    }
}

impl std::error::Error for ImportError {}

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
    pub fn architecture(mut self, arch: Architecture) -> Self {
        self.architecture = arch;
        self
    }

    /// Set validation config
    pub fn validate(mut self, config: ValidationConfig) -> Self {
        self.validation = config;
        self
    }

    /// Set quantization
    pub fn quantize(mut self, quant: QuantizationType) -> Self {
        self.quantize = Some(quant);
        self
    }

    /// Set compression
    pub fn compress(mut self, comp: Compression) -> Self {
        self.compress = Some(comp);
        self
    }

    /// Run the conversion
    pub fn convert(self) -> Result<Vec<u8>> {
        let source = self.source.ok_or_else(|| AprenderError::FormatError {
            message: "No source specified".to_string(),
        })?;

        // TODO: Implement full conversion
        // 1. Download if HF source
        // 2. Load SafeTensors
        // 3. Validate each tensor
        // 4. Convert to APR format
        // 5. Apply quantization
        // 6. Apply compression

        Err(AprenderError::FormatError {
            message: format!("Conversion from {:?} not yet implemented", source),
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
/// * `source` - Source path: local file, hf://org/repo, or URL
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

    // Step 2: Detect format and load tensors
    let tensors = load_source_tensors(&local_path, &options)?;

    // Step 3: Map tensor names to canonical APR names
    let mapped_tensors = map_tensor_names(&tensors, options.architecture);

    // Step 4: Validate tensors (inline validation)
    let validation_result = validate_tensors(&mapped_tensors, &options)?;

    // Step 5: Write APR format
    write_apr_file(&mapped_tensors, output_path, &options)?;

    Ok(validation_result)
}

/// Resolve a source to a local file path
fn resolve_source(source: &Source, cache: bool) -> Result<PathBuf> {
    match source {
        Source::Local(path) => {
            if !path.exists() {
                return Err(AprenderError::FormatError {
                    message: format!("Source file not found: {}", path.display()),
                });
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

            // Try to download using hf-hub if feature is enabled
            #[cfg(feature = "hf-hub-integration")]
            {
                let repo_id = format!("{org}/{repo}");
                match download_from_hf(&repo_id, filename) {
                    Ok(path) => return Ok(path),
                    Err(e) => {
                        // Fall through to manual download instructions
                        eprintln!("HF download failed: {e}");
                    }
                }
            }

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

/// Find a model file in standard cache locations
fn find_in_cache(org: &str, repo: &str, filename: &str) -> Option<PathBuf> {
    // Try XDG cache dir first (Linux), then home dir fallback
    let cache_paths = [
        std::env::var("XDG_CACHE_HOME")
            .ok()
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                std::env::var("HOME")
                    .map(|h| PathBuf::from(h).join(".cache"))
                    .unwrap_or_else(|_| PathBuf::from(".cache"))
            }),
        // HuggingFace default cache location
        std::env::var("HF_HOME")
            .ok()
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                std::env::var("HOME")
                    .map(|h| PathBuf::from(h).join(".cache").join("huggingface"))
                    .unwrap_or_else(|_| PathBuf::from(".cache").join("huggingface"))
            }),
    ];

    for cache_base in &cache_paths {
        // Check aprender cache
        let apr_cache = cache_base
            .join("aprender")
            .join("hf")
            .join(org)
            .join(repo)
            .join(filename);
        if apr_cache.exists() {
            return Some(apr_cache);
        }

        // Check HuggingFace hub cache (different structure)
        let hf_cache = cache_base
            .join("hub")
            .join(format!("models--{org}--{repo}"));
        if hf_cache.exists() {
            // HF cache has snapshots/refs structure, look for the file
            let snapshot_dir = hf_cache.join("snapshots");
            if let Ok(entries) = fs::read_dir(&snapshot_dir) {
                for entry in entries.flatten() {
                    let file_path = entry.path().join(filename);
                    if file_path.exists() {
                        return Some(file_path);
                    }
                }
            }
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

    let api = builder.build().map_err(|e| AprenderError::FormatError {
        message: format!("Failed to initialize HF Hub API: {e}"),
    })?;

    // Get repo handle
    let repo = api.model(repo_id.to_string());

    // Download the file
    let path = repo.get(filename).map_err(|e| AprenderError::FormatError {
        message: format!("Failed to download {filename} from {repo_id}: {e}"),
    })?;

    Ok(path)
}

/// Load tensors from source file (SafeTensors format)
fn load_source_tensors(
    path: &Path,
    _options: &ImportOptions,
) -> Result<BTreeMap<String, (Vec<f32>, Vec<usize>)>> {
    let extension = path.extension().and_then(|e| e.to_str()).unwrap_or("");

    match extension {
        "safetensors" => load_safetensors_tensors(path),
        "apr" => {
            // Already APR format - extract tensors
            Err(AprenderError::FormatError {
                message: "Cannot import from APR format - use direct loading instead".to_string(),
            })
        }
        "gguf" => Err(AprenderError::FormatError {
            message: "GGUF import not yet implemented".to_string(),
        }),
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

/// Load tensors from SafeTensors file
fn load_safetensors_tensors(path: &Path) -> Result<BTreeMap<String, (Vec<f32>, Vec<usize>)>> {
    let (metadata, raw_data) = load_safetensors(path).map_err(|e| AprenderError::FormatError {
        message: format!("Failed to load SafeTensors: {e}"),
    })?;

    let mut tensors = BTreeMap::new();

    for (name, tensor_meta) in metadata.iter() {
        // Skip __metadata__ key if present
        if name.starts_with("__") {
            continue;
        }

        let data =
            extract_tensor(&raw_data, tensor_meta).map_err(|e| AprenderError::FormatError {
                message: format!("Failed to extract tensor '{name}': {e}"),
            })?;

        tensors.insert(name.clone(), (data, tensor_meta.shape.clone()));
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

/// Validate tensors according to architecture expectations
fn validate_tensors(
    tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    options: &ImportOptions,
) -> Result<ValidationReport> {
    // Create validator and build report
    let mut validator = AprValidator::new();

    // Compute stats and check expectations for each tensor
    let mut validation_errors = Vec::new();

    for (name, (data, _shape)) in tensors {
        let stats = compute_tensor_stats(name, data);

        // Check tensor expectations if validation is enabled
        if options.validation != ValidationConfig::None {
            if let Some(expectation) = TensorExpectation::for_tensor(name) {
                if let Err(e) = expectation.check(&stats) {
                    if options.validation == ValidationConfig::Strict && !options.force {
                        validation_errors.push(format!("{name}: {e}"));
                    }
                }
            }

            // Check for NaN/Inf
            if stats.nan_count > 0 {
                validation_errors.push(format!("{name}: contains {} NaN values", stats.nan_count));
            }
            if stats.inf_count > 0 {
                validation_errors.push(format!("{name}: contains {} Inf values", stats.inf_count));
            }
        }

        validator.add_tensor_stats(stats);
    }

    // Build report
    let report = validator.validate();

    // Fail if validation errors and not forced
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

    let mut sum = 0.0_f64;
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut nan_count = 0;
    let mut inf_count = 0;
    let mut zero_count = 0;
    let mut valid_count = 0;

    for &v in data {
        if v.is_nan() {
            nan_count += 1;
        } else if v.is_infinite() {
            inf_count += 1;
        } else {
            sum += v as f64;
            min = min.min(v);
            max = max.max(v);
            valid_count += 1;
            if v == 0.0 {
                zero_count += 1;
            }
        }
    }

    let mean = if valid_count > 0 {
        (sum / valid_count as f64) as f32
    } else {
        0.0
    };

    // Compute standard deviation
    let mut variance_sum = 0.0_f64;
    for &v in data {
        if !v.is_nan() && !v.is_infinite() {
            let diff = v as f64 - mean as f64;
            variance_sum += diff * diff;
        }
    }
    let std = if valid_count > 1 {
        ((variance_sum / (valid_count - 1) as f64).sqrt()) as f32
    } else {
        0.0
    };

    TensorStats {
        name: name.to_string(),
        count: data.len(),
        min: if min == f32::INFINITY { 0.0 } else { min },
        max: if max == f32::NEG_INFINITY { 0.0 } else { max },
        mean,
        std,
        nan_count,
        inf_count,
        zero_count,
    }
}

/// Write tensors to APR format
fn write_apr_file(
    tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    output: &Path,
    _options: &ImportOptions,
) -> Result<()> {
    // For now, write as SafeTensors format (simpler, still valid)
    // Full APR format with compression/quantization comes next
    save_safetensors(output, tensors).map_err(|e| AprenderError::FormatError {
        message: format!("Failed to write APR file: {e}"),
    })
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

/// Quantize to fp16 (simulate by reducing precision)
fn quantize_fp16(data: &[f32]) -> Vec<f32> {
    data.iter()
        .map(|&v| {
            // Convert to f16 precision by truncating mantissa
            let bits = v.to_bits();
            let sign = bits >> 31;
            let exp = (bits >> 23) & 0xFF;
            let mantissa = bits & 0x7FFFFF;

            // Truncate mantissa to 10 bits (f16 precision)
            let mantissa_16 = mantissa >> 13;

            // Reconstruct as f32 with reduced precision
            let new_bits = (sign << 31) | (exp << 23) | (mantissa_16 << 13);
            f32::from_bits(new_bits)
        })
        .collect()
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
    // TODO: Apply compression during save
    // For now, save as SafeTensors
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
    /// SafeTensors format (.safetensors) - HuggingFace ecosystem
    SafeTensors,
    /// GGUF format (.gguf) - llama.cpp / local inference
    Gguf,
    /// ONNX format (.onnx) - Cross-framework inference (not yet implemented)
    Onnx,
    /// TorchScript format (.pt) - PyTorch deployment (not yet implemented)
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
#[allow(clippy::too_many_lines)]
pub fn apr_merge<P: AsRef<Path>>(
    inputs: &[P],
    output: P,
    options: MergeOptions,
) -> Result<MergeReport> {
    // Validate inputs
    if inputs.len() < 2 {
        return Err(AprenderError::FormatError {
            message: "Merge requires at least 2 input models".to_string(),
        });
    }

    // Check strategy is supported
    if !options.strategy.is_supported() {
        return Err(AprenderError::FormatError {
            message: format!(
                "Merge strategy {:?} is not yet supported. Use 'average' or 'weighted'.",
                options.strategy
            ),
        });
    }

    // Validate weights for weighted merge
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

    // Load all models
    let mut all_tensors: Vec<BTreeMap<String, (Vec<f32>, Vec<usize>)>> = Vec::new();
    for input_path in inputs {
        let path = input_path.as_ref();
        if !path.exists() {
            return Err(AprenderError::FormatError {
                message: format!("Input file not found: {}", path.display()),
            });
        }
        let tensors = load_model_tensors(path)?;
        all_tensors.push(tensors);
    }

    // Verify all models have same tensors
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
        for (name, (_, shape)) in reference {
            match tensors.get(name) {
                None => {
                    return Err(AprenderError::FormatError {
                        message: format!("Model {} is missing tensor '{}'", i, name),
                    });
                }
                Some((_, other_shape)) if other_shape != shape => {
                    return Err(AprenderError::FormatError {
                        message: format!(
                            "Tensor '{}' has shape {:?} in model 0 but {:?} in model {}",
                            name, shape, other_shape, i
                        ),
                    });
                }
                _ => {}
            }
        }
    }

    // Calculate weights (normalize to sum to 1.0)
    let weights = match options.strategy {
        MergeStrategy::Average => {
            let w = 1.0 / inputs.len() as f32;
            vec![w; inputs.len()]
        }
        MergeStrategy::Weighted => {
            let raw_weights = options.weights.as_ref().expect("validated above");
            let sum: f32 = raw_weights.iter().sum();
            if sum <= 0.0 {
                return Err(AprenderError::FormatError {
                    message: "Weights must sum to a positive value".to_string(),
                });
            }
            raw_weights.iter().map(|w| w / sum).collect()
        }
        _ => unreachable!("unsupported strategies filtered above"),
    };

    // Merge tensors
    let mut merged: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
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
        let input = "/tmp/test_whisper_input.safetensors";
        let output = "/tmp/test_whisper_output.safetensors";

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

        // Load output and verify names are mapped
        let (metadata, _) = load_safetensors(output).expect("Failed to load output");
        assert!(
            metadata.contains_key("encoder.conv1.weight"),
            "Should strip 'model.' prefix"
        );
        assert!(
            metadata.contains_key("decoder.layer_norm.weight"),
            "Should strip 'model.' prefix"
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

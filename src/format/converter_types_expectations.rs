
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
    /// Issue #46: Qwen2.5-Coder-0.5B has mean=7.23 and std=2.11, widen ranges
    pub const RMSNORM_WEIGHT: Self = Self {
        mean_range: (-1.0, 10.0),    // Wider range for Qwen and other architectures
        std_range: Some((0.0, 5.0)), // Qwen has std up to 2.11
        description: "RMSNorm weight (gamma)",
    };

    /// Get expectation for a tensor name
    #[must_use]
    #[allow(clippy::case_sensitive_file_extension_comparisons)]
    pub fn for_tensor(name: &str) -> Option<Self> {
        // RMSNorm patterns (LLaMA, Qwen2, TinyLlama, GGUF) - check BEFORE generic LayerNorm
        // These use gamma initialized to 1.0, not the 0-centered LayerNorm
        // Fix #163: Also match GGUF attn_norm/ffn_norm patterns
        if (name.contains("input_layernorm")
            || name.contains("post_attention_layernorm")
            || name.contains("rms_norm")
            || name.contains("attn_norm") // GGUF pattern (blk.N.attn_norm.weight)
            || name.contains("ffn_norm") // GGUF pattern (blk.N.ffn_norm.weight)
            // GH-279: Qwen3 QK normalization (RMSNorm, not linear)
            || name.contains("q_norm")
            || name.contains("k_norm"))
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
    /// GGUF Q4_K format (K-quants, ~4.5 bits/weight)
    /// 256-element super-blocks with nested 32-element sub-blocks
    /// Achieves ~7x memory bandwidth improvement over F32
    Q4K,
}

// ============================================================================
// DOUBLE-QUANT-001: Compile-Time Double Quantization Prevention
// ============================================================================

/// F32 tensors that were NATIVELY F32 (e.g., SafeTensors F32/BF16 sources).
///
/// This is the ONLY type that `quantize_tensors()` accepts.
/// Constructed from sources known to be unquantized (SafeTensors, F32 APR files).
///
/// There is intentionally NO `From<DequantizedTensors> for NativeF32Tensors` —
/// attempting to pass dequantized tensors to `quantize_tensors()` is a compile error.
#[derive(Debug)]
pub struct NativeF32Tensors(BTreeMap<String, (Vec<f32>, Vec<usize>)>);

impl NativeF32Tensors {
    /// Wrap a tensor map as natively F32.
    #[must_use]
    pub fn new(map: BTreeMap<String, (Vec<f32>, Vec<usize>)>) -> Self {
        Self(map)
    }

    /// Consume the wrapper and return the inner map.
    #[must_use]
    pub fn into_inner(self) -> BTreeMap<String, (Vec<f32>, Vec<usize>)> {
        self.0
    }
}

impl AsRef<BTreeMap<String, (Vec<f32>, Vec<usize>)>> for NativeF32Tensors {
    fn as_ref(&self) -> &BTreeMap<String, (Vec<f32>, Vec<usize>)> {
        &self.0
    }
}

/// F32 tensors that were DEQUANTIZED from a quantized format (Q4K, Q6K, etc.).
///
/// CANNOT be passed to `quantize_tensors()` — compile error.
/// Re-quantizing dequantized data is a lossy double quantization that destroys
/// weight fidelity (PMAT-252).
#[derive(Debug)]
pub struct DequantizedTensors {
    tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    /// The original quantization format these tensors came from.
    pub original_quant: QuantizationType,
}

impl DequantizedTensors {
    /// Wrap a tensor map as dequantized from the given quantization type.
    #[must_use]
    pub fn new(map: BTreeMap<String, (Vec<f32>, Vec<usize>)>, quant: QuantizationType) -> Self {
        Self {
            tensors: map,
            original_quant: quant,
        }
    }

    /// Consume the wrapper and return the inner map.
    #[must_use]
    pub fn into_inner(self) -> BTreeMap<String, (Vec<f32>, Vec<usize>)> {
        self.tensors
    }
}

impl AsRef<BTreeMap<String, (Vec<f32>, Vec<usize>)>> for DequantizedTensors {
    fn as_ref(&self) -> &BTreeMap<String, (Vec<f32>, Vec<usize>)> {
        &self.tensors
    }
}

/// Tensor provenance: either natively F32 or dequantized from quantized format.
///
/// Used by export/convert pipelines to enforce compile-time safety against
/// double quantization (DOUBLE-QUANT-001).
#[derive(Debug)]
pub enum TensorProvenance {
    /// Tensors that are natively F32 — safe to quantize.
    Native(NativeF32Tensors),
    /// Tensors dequantized from a quantized format — must NOT be re-quantized.
    Dequantized(DequantizedTensors),
}

impl TensorProvenance {
    /// Get a read-only reference to the underlying tensor map.
    #[must_use]
    pub fn as_map(&self) -> &BTreeMap<String, (Vec<f32>, Vec<usize>)> {
        match self {
            Self::Native(n) => n.as_ref(),
            Self::Dequantized(d) => d.as_ref(),
        }
    }

    /// Consume the provenance wrapper and return the inner map.
    #[must_use]
    pub fn into_map(self) -> BTreeMap<String, (Vec<f32>, Vec<usize>)> {
        match self {
            Self::Native(n) => n.into_inner(),
            Self::Dequantized(d) => d.into_inner(),
        }
    }
}

// ============================================================================

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
    /// GH-196: Strict mode rejects unverified architectures.
    /// Default is permissive (imports proceed with a warning).
    /// Use `--strict` on CLI to enable this.
    pub strict: bool,
    /// Cache downloaded files
    pub cache: bool,
    /// PMAT-232: External tokenizer.json path for weights-only GGUF files.
    /// If the GGUF has no embedded tokenizer, this file will be used instead.
    pub tokenizer_path: Option<std::path::PathBuf>,
    /// GH-223: Allow import without config.json. By default, SafeTensors import
    /// errors when config.json is missing (inferred hyperparameters like rope_theta
    /// may be wrong). Pass `--allow-no-config` to proceed with warnings only.
    pub allow_no_config: bool,
}

impl Default for ImportOptions {
    fn default() -> Self {
        Self {
            architecture: Architecture::Auto,
            validation: ValidationConfig::Strict,
            quantize: None,
            compress: None,
            strict: false,
            cache: true,
            tokenizer_path: None,
            allow_no_config: false,
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

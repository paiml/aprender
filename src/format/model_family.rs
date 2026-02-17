//! Model Family Contract Types (PMAT-241)
//!
//! Defines the `ModelFamily` trait and associated configuration types for
//! compiler-enforced model family contracts.
//!
//! # Theoretical Foundation
//!
//! - Shingo (1986): Poka-Yoke / Zero Quality Control
//! - Strom & Yemini (1986): Typestate programming
//! - Parsons (2019): Parse, Don't Validate
//!
//! # Contract
//!
//! See `contracts/model-families/*.yaml` and
//! `docs/specifications/compiler-enforced-model-types-model-oracle.md`

use std::collections::HashMap;
use std::fmt;

use crate::error::{AprenderError, Result};

// ============================================================================
// Enums
// ============================================================================

/// Attention mechanism type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionType {
    /// Multi-Head Attention (standard transformer)
    Mha,
    /// Grouped Query Attention (GQA)
    Gqa,
    /// Multi-Query Attention (MQA)
    Mqa,
}

impl fmt::Display for AttentionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Mha => write!(f, "MHA"),
            Self::Gqa => write!(f, "GQA"),
            Self::Mqa => write!(f, "MQA"),
        }
    }
}

impl AttentionType {
    /// Parse from YAML string
    pub fn from_str_contract(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "mha" => Ok(Self::Mha),
            "gqa" => Ok(Self::Gqa),
            "mqa" => Ok(Self::Mqa),
            _ => Err(AprenderError::FormatError {
                message: format!("Unknown attention type: {s}. Expected: mha, gqa, mqa"),
            }),
        }
    }
}

/// Activation function type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// SiLU (Swish) activation
    Silu,
    /// GELU activation
    Gelu,
    /// ReLU activation
    Relu,
}

impl fmt::Display for Activation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Silu => write!(f, "SiLU"),
            Self::Gelu => write!(f, "GELU"),
            Self::Relu => write!(f, "ReLU"),
        }
    }
}

impl Activation {
    pub fn from_str_contract(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "silu" | "swish" => Ok(Self::Silu),
            "gelu" => Ok(Self::Gelu),
            "relu" => Ok(Self::Relu),
            _ => Err(AprenderError::FormatError {
                message: format!("Unknown activation: {s}. Expected: silu, gelu, relu"),
            }),
        }
    }
}

/// Normalization type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormType {
    /// RMS Normalization (LLaMA, Qwen2)
    RmsNorm,
    /// Layer Normalization (BERT, Whisper)
    LayerNorm,
}

impl fmt::Display for NormType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RmsNorm => write!(f, "RMSNorm"),
            Self::LayerNorm => write!(f, "LayerNorm"),
        }
    }
}

impl NormType {
    pub fn from_str_contract(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "rmsnorm" | "rms_norm" => Ok(Self::RmsNorm),
            "layernorm" | "layer_norm" => Ok(Self::LayerNorm),
            _ => Err(AprenderError::FormatError {
                message: format!("Unknown norm type: {s}. Expected: rmsnorm, layernorm"),
            }),
        }
    }
}

/// Positional encoding type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PositionalEncoding {
    /// Rotary Position Embeddings (LLaMA, Qwen2)
    Rope,
    /// ALiBi (Bloom)
    Alibi,
    /// Absolute position embeddings (BERT, Whisper)
    Absolute,
    /// Relative position embeddings
    Relative,
    /// No positional encoding (RWKV, Mamba — state carries temporal info)
    None,
}

impl fmt::Display for PositionalEncoding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Rope => write!(f, "RoPE"),
            Self::Alibi => write!(f, "ALiBi"),
            Self::Absolute => write!(f, "Absolute"),
            Self::Relative => write!(f, "Relative"),
            Self::None => write!(f, "None"),
        }
    }
}

impl PositionalEncoding {
    pub fn from_str_contract(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "rope" => Ok(Self::Rope),
            "alibi" => Ok(Self::Alibi),
            "absolute" | "sinusoidal" => Ok(Self::Absolute),
            "relative" => Ok(Self::Relative),
            "none" => Ok(Self::None),
            _ => Err(AprenderError::FormatError {
                message: format!(
                    "Unknown positional encoding: {s}. Expected: rope, alibi, absolute, relative, none"
                ),
            }),
        }
    }
}

/// MLP type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlpType {
    /// SwiGLU (LLaMA, Qwen2) - gated with SiLU
    SwiGlu,
    /// Standard GELU MLP (BERT, Whisper)
    GeluMlp,
    /// Gated MLP (generic)
    GatedMlp,
}

impl fmt::Display for MlpType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SwiGlu => write!(f, "SwiGLU"),
            Self::GeluMlp => write!(f, "GELU MLP"),
            Self::GatedMlp => write!(f, "Gated MLP"),
        }
    }
}

impl MlpType {
    pub fn from_str_contract(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "swiglu" => Ok(Self::SwiGlu),
            "gelu_mlp" | "gelu" => Ok(Self::GeluMlp),
            "gated_mlp" | "gated" => Ok(Self::GatedMlp),
            _ => Err(AprenderError::FormatError {
                message: format!("Unknown MLP type: {s}. Expected: swiglu, gelu_mlp, gated_mlp"),
            }),
        }
    }
}

// ============================================================================
// Configuration Structs
// ============================================================================

/// Configuration for a specific model size within a family.
#[derive(Debug, Clone)]
pub struct ModelSizeConfig {
    /// Human-readable parameter count (e.g., "0.5B", "7B")
    pub parameters: String,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of key-value heads (for GQA)
    pub num_kv_heads: usize,
    /// Intermediate (FFN) dimension
    pub intermediate_dim: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Maximum position embeddings
    pub max_position_embeddings: usize,
    /// Per-head dimension (`hidden_dim / num_heads`)
    pub head_dim: usize,
    /// RoPE theta frequency (0.0 if not using RoPE)
    pub rope_theta: f64,
    /// Normalization epsilon
    pub norm_eps: f64,
}

/// Architectural constraints for a model family.
#[derive(Debug, Clone)]
pub struct ModelConstraints {
    pub attention_type: AttentionType,
    pub activation: Activation,
    pub norm_type: NormType,
    pub has_bias: bool,
    pub tied_embeddings: bool,
    pub positional_encoding: PositionalEncoding,
    pub mlp_type: MlpType,
}

/// Tensor name template for a model family.
#[derive(Debug, Clone)]
pub struct TensorTemplate {
    /// Embedding tensor name (e.g., "model.embed\_tokens.weight")
    pub embedding: String,
    /// LM head tensor name (e.g., "lm\_head.weight")
    pub lm_head: Option<String>,
    /// Final normalization tensor name
    pub final_norm: Option<String>,
    /// Per-layer tensor name patterns (keys: q\_proj, k\_proj, etc., values contain {n} placeholder)
    pub per_layer: HashMap<String, Option<String>>,
}

/// Shape template for a model family (parameterized expressions).
#[derive(Debug, Clone)]
pub struct ShapeTemplate {
    /// Map of tensor role to parameterized shape expression
    /// e.g., "q\_proj" maps to "\[num\_heads * head\_dim, hidden\_dim\]"
    pub shapes: HashMap<String, String>,
}

/// Chat template configuration.
#[derive(Debug, Clone)]
pub struct ChatTemplateConfig {
    pub format: String,
    pub template: String,
    pub bos_token: String,
    pub eos_token: String,
    pub special_tokens: HashMap<String, String>,
}

/// Certification cross-reference configuration.
#[derive(Debug, Clone)]
pub struct CertificationConfig {
    pub playbook_path: String,
    pub csv_family_key: String,
    pub size_categories: HashMap<String, String>,
}

/// Complete configuration for a model family.
#[derive(Debug, Clone)]
pub struct ModelFamilyConfig {
    /// Canonical family name (e.g., "qwen2")
    pub family: String,
    /// Human-readable display name
    pub display_name: String,
    /// Vendor/organization
    pub vendor: String,
    /// HuggingFace architecture identifiers
    pub architectures: Vec<String>,
    /// HuggingFace repo name pattern
    pub hf_pattern: String,
    /// Size variants keyed by name (e.g., "0.5b", "7b")
    pub size_variants: HashMap<String, ModelSizeConfig>,
    /// Architectural constraints
    pub constraints: ModelConstraints,
    /// Tensor name template
    pub tensor_template: TensorTemplate,
    /// Shape template
    pub shape_template: ShapeTemplate,
    /// Supported quantization formats
    pub quantizations: Vec<String>,
    /// Chat template (None for non-chat models like Whisper, BERT)
    pub chat_template: Option<ChatTemplateConfig>,
    /// Certification cross-reference
    pub certification: Option<CertificationConfig>,
}

// ============================================================================
// Contract Error
// ============================================================================

/// Model family contract error
#[derive(Debug, Clone)]
pub struct ContractError {
    pub family: String,
    pub message: String,
}

impl fmt::Display for ContractError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Model family contract error [{}]: {}",
            self.family, self.message
        )
    }
}

impl std::error::Error for ContractError {}

impl From<ContractError> for AprenderError {
    fn from(err: ContractError) -> Self {
        AprenderError::FormatError {
            message: err.to_string(),
        }
    }
}

// ============================================================================
// ModelFamily Trait
// ============================================================================

/// Trait implemented by each model family.
///
/// This trait is the compile-time bridge between YAML contracts and Rust code.
/// Implementations can be generated by build.rs from model family YAMLs (PMAT-250)
/// or loaded at runtime from YAML files (PMAT-242).
pub trait ModelFamily: fmt::Debug + Send + Sync {
    /// Canonical family name (e.g., "qwen2")
    fn family_name(&self) -> &str;

    /// Human-readable display name
    fn display_name(&self) -> &str;

    /// Get the full configuration
    fn config(&self) -> &ModelFamilyConfig;

    /// Get configuration for a specific size variant
    fn size_config(&self, size: &str) -> Option<&ModelSizeConfig>;

    /// Detect size variant from model config (`hidden_dim`, `num_layers`)
    fn detect_size(&self, hidden_dim: usize, num_layers: usize) -> Option<String>;

    /// Get architectural constraints
    fn constraints(&self) -> &ModelConstraints;

    /// Expected tensor count for a given size variant
    fn expected_tensor_count(&self, size: &str) -> Option<usize>;

    /// Validate that a set of tensor names matches the contract
    fn validate_tensor_names(
        &self,
        names: &[&str],
        size: &str,
    ) -> std::result::Result<(), ContractError>;
}

// ============================================================================
// DynModelFamily - Runtime implementation backed by ModelFamilyConfig
// ============================================================================

/// Dynamic model family implementation backed by a `ModelFamilyConfig`.
/// Used when family is loaded from YAML at runtime.
#[derive(Debug, Clone)]
pub struct DynModelFamily {
    config: ModelFamilyConfig,
}

impl DynModelFamily {
    /// Create from a loaded config
    #[must_use]
    pub fn new(config: ModelFamilyConfig) -> Self {
        Self { config }
    }
}

include!("model_family_part_02.rs");
include!("model_family_part_03.rs");

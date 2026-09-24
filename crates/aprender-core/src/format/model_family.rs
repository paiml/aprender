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
    /// State Space Model (no attention — selective scan)
    Ssm,
    /// Linear Attention (WKV recurrence, no softmax)
    Linear,
    /// Hybrid: Gated DeltaNet linear-attention layers interleaved with full softmax-attention
    /// layers (the Qwen3.5 family). apr declares it unsupported on CPU and GPU (#3091); the
    /// family YAML must still LOAD, so the refusal can name the family instead of failing to parse.
    HybridGatedDeltaNet,
}

impl fmt::Display for AttentionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Mha => write!(f, "MHA"),
            Self::Gqa => write!(f, "GQA"),
            Self::Mqa => write!(f, "MQA"),
            Self::Ssm => write!(f, "SSM"),
            Self::Linear => write!(f, "Linear"),
            Self::HybridGatedDeltaNet => write!(f, "Hybrid Gated DeltaNet"),
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
            "ssm" => Ok(Self::Ssm),
            "linear" => Ok(Self::Linear),
            "hybrid_gated_deltanet" => Ok(Self::HybridGatedDeltaNet),
            _ => Err(AprenderError::FormatError {
                message: format!(
                    "Unknown attention type: {s}. Expected: mha, gqa, mqa, ssm, linear, hybrid_gated_deltanet"
                ),
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
            "gelu_mlp" | "gelu" | "standard" => Ok(Self::GeluMlp),
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

/// #3346: the Gated DeltaNet shape of a hybrid family.
///
/// `contracts/model-families/qwen3_5.yaml` declares `inner_size`,
/// `state_size`, `conv_kernel`, `group_count` and `full_attention_interval`
/// under `constraints:`, and until #3346 [`ModelConstraints`] carried none of
/// them. A DeltaNet layer's parameters live entirely in these dimensions — the
/// conv, the in/out projections and the gates — so a type that drops them
/// cannot count a hybrid model: dense accounting under-counted the real
/// `Qwen3.5-0.8B-Q4_K_M.gguf` by 14.4%.
///
/// Tensor names below are the GGUF names of that measured file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DeltaNetShape {
    /// Width of the DeltaNet mixer — `attn_gate`/`ssm_out` are `inner_size`
    /// wide and `attn_qkv`/`ssm_conv1d` are `3 * inner_size` wide.
    pub inner_size: usize,
    /// Per-head state width; `ssm_norm` is a vector of this length.
    pub state_size: usize,
    /// Depthwise conv width: `ssm_conv1d` is `[conv_kernel, 3 * inner_size]`.
    pub conv_kernel: usize,
    /// Number of DeltaNet heads — the length of `ssm_a` and `ssm_dt.bias`, and
    /// the output width of `ssm_alpha`/`ssm_beta`.
    pub group_count: usize,
    /// Period of the hybrid schedule: every `full_attention_interval`-th layer
    /// runs softmax attention instead, the LAST of each group (measured: layers
    /// 3, 7, 11, 15, 19, 23 of 24 at interval 4).
    pub full_attention_interval: usize,
}

impl DeltaNetShape {
    /// `group_count * state_size`, which must equal `inner_size` for the
    /// declared shape to describe one mixer.
    ///
    /// This is a CHECK, not a repair: the measured 0.8B satisfies it
    /// (16 * 128 = 2048) and the 9b descriptor does not (8 * 128 != 2048).
    #[must_use]
    pub const fn heads_span_the_mixer(&self) -> bool {
        match self.group_count.checked_mul(self.state_size) {
            Some(span) => span == self.inner_size,
            None => false,
        }
    }
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
    /// GH-280: Whether Q and K projections have per-head RMSNorm (e.g., Qwen3)
    pub qk_norm: bool,
    /// #3346: Gated DeltaNet shape, for hybrid families that declare one.
    /// `None` for every family whose descriptor has no `inner_size`.
    pub deltanet: Option<DeltaNetShape>,
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

/// GH-277: GGUF tensor name template for contract-driven export.
///
/// Maps APR canonical tensor roles to GGUF canonical tensor names
/// (from llama.cpp llama-arch.cpp). Used by the GGUF exporter to
/// produce llama.cpp-compatible files without hardcoded name tables.
#[derive(Debug, Clone, Default)]
pub struct GgufTensorTemplate {
    /// Embedding tensor GGUF name (e.g., "token_embd.weight")
    pub embedding: Option<String>,
    /// Position embedding GGUF name (e.g., "position_embd.weight"), None if not used
    pub position_embedding: Option<String>,
    /// LM head GGUF name (e.g., "output.weight")
    pub lm_head: Option<String>,
    /// Final norm weight GGUF name (e.g., "output_norm.weight")
    pub final_norm_weight: Option<String>,
    /// Final norm bias GGUF name (e.g., "output_norm.bias"), None for RMSNorm models
    pub final_norm_bias: Option<String>,
    /// Per-layer tensor role → GGUF suffix (after "blk.{n}.")
    /// None values mean the tensor should be SKIPPED during export
    pub per_layer: HashMap<String, Option<String>>,
    /// GH-277: If true, transpose 2D weight tensors from Conv1D to Linear layout during export.
    /// GPT-2 uses Conv1D `[in_features, out_features]`, llama.cpp expects `[out_features, in_features]`.
    pub transpose_weights: bool,
    /// GH-277: Fusion rules for concatenating multiple APR tensors into one GGUF tensor.
    /// Used for architectures like GPT-2 where llama.cpp expects fused QKV.
    pub fuse: Vec<GgufFusionRule>,
}

/// GH-277: Rule for fusing multiple APR per-layer tensors into a single GGUF tensor.
///
/// Example: GPT-2's separate Q, K, V projections are concatenated into a single
/// `attn_qkv.weight` tensor for llama.cpp compatibility.
#[derive(Debug, Clone)]
pub struct GgufFusionRule {
    /// GGUF suffix for the fused tensor (e.g., "attn_qkv.weight")
    pub gguf_suffix: String,
    /// APR tensor role names to concatenate, in order (e.g., `q_proj_weight`, `k_proj_weight`, `v_proj_weight`)
    pub source_roles: Vec<String>,
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
    /// GH-277: GGUF tensor name template for export
    pub gguf_tensor_template: GgufTensorTemplate,
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

include!("family_registry.rs");
include!("model_family_tests.rs");

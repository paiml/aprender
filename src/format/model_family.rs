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
}

impl fmt::Display for PositionalEncoding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Rope => write!(f, "RoPE"),
            Self::Alibi => write!(f, "ALiBi"),
            Self::Absolute => write!(f, "Absolute"),
            Self::Relative => write!(f, "Relative"),
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
            _ => Err(AprenderError::FormatError {
                message: format!(
                    "Unknown positional encoding: {s}. Expected: rope, alibi, absolute, relative"
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

impl ModelFamily for DynModelFamily {
    fn family_name(&self) -> &str {
        &self.config.family
    }

    fn display_name(&self) -> &str {
        &self.config.display_name
    }

    fn config(&self) -> &ModelFamilyConfig {
        &self.config
    }

    fn size_config(&self, size: &str) -> Option<&ModelSizeConfig> {
        self.config.size_variants.get(size)
    }

    fn detect_size(&self, hidden_dim: usize, num_layers: usize) -> Option<String> {
        for (name, variant) in &self.config.size_variants {
            if variant.hidden_dim == hidden_dim && variant.num_layers == num_layers {
                return Some(name.clone());
            }
        }
        None
    }

    fn constraints(&self) -> &ModelConstraints {
        &self.config.constraints
    }

    fn expected_tensor_count(&self, size: &str) -> Option<usize> {
        let variant = self.config.size_variants.get(size)?;
        let num_layers = variant.num_layers;

        // Count global tensors
        let mut count = 0usize;
        if !self.config.tensor_template.embedding.is_empty() {
            count += 1;
        }
        if self.config.tensor_template.lm_head.is_some() {
            count += 1;
        }
        if self.config.tensor_template.final_norm.is_some() {
            count += 1;
        }

        // Count per-layer tensors
        let tensors_per_layer = self
            .config
            .tensor_template
            .per_layer
            .values()
            .filter(|v| v.is_some())
            .count();
        count += tensors_per_layer * num_layers;

        Some(count)
    }

    fn validate_tensor_names(
        &self,
        names: &[&str],
        size: &str,
    ) -> std::result::Result<(), ContractError> {
        let variant = self
            .config
            .size_variants
            .get(size)
            .ok_or_else(|| ContractError {
                family: self.config.family.clone(),
                message: format!("Unknown size variant: {size}"),
            })?;

        // Build expected tensor names
        let mut expected: Vec<String> = Vec::new();
        expected.push(self.config.tensor_template.embedding.clone());
        if let Some(lm_head) = &self.config.tensor_template.lm_head {
            expected.push(lm_head.clone());
        }
        if let Some(final_norm) = &self.config.tensor_template.final_norm {
            expected.push(final_norm.clone());
        }

        for layer_idx in 0..variant.num_layers {
            for pat in self.config.tensor_template.per_layer.values().flatten() {
                expected.push(pat.replace("{n}", &layer_idx.to_string()));
            }
        }

        // Check for unexpected tensors (tensor names not in expected list)
        let expected_set: std::collections::HashSet<&str> =
            expected.iter().map(String::as_str).collect();
        let actual_set: std::collections::HashSet<&str> = names.iter().copied().collect();

        let missing: Vec<&str> = expected_set.difference(&actual_set).copied().collect();
        let unexpected: Vec<&str> = actual_set.difference(&expected_set).copied().collect();

        if !missing.is_empty() || !unexpected.is_empty() {
            let mut msg = String::new();
            if !missing.is_empty() {
                msg.push_str(&format!("Missing tensors: {}", missing.join(", ")));
            }
            if !unexpected.is_empty() {
                if !msg.is_empty() {
                    msg.push_str("; ");
                }
                msg.push_str(&format!("Unexpected tensors: {}", unexpected.join(", ")));
            }
            return Err(ContractError {
                family: self.config.family.clone(),
                message: msg,
            });
        }

        Ok(())
    }
}

// ============================================================================
// Family Registry
// ============================================================================

/// Registry of known model families for detection.
#[derive(Debug)]
pub struct FamilyRegistry {
    families: Vec<Box<dyn ModelFamily>>,
}

impl FamilyRegistry {
    /// Create an empty registry
    #[must_use]
    pub fn new() -> Self {
        Self {
            families: Vec::new(),
        }
    }

    /// Register a model family
    pub fn register(&mut self, family: Box<dyn ModelFamily>) {
        self.families.push(family);
    }

    /// Get all registered family names
    #[must_use]
    pub fn family_names(&self) -> Vec<&str> {
        self.families.iter().map(|f| f.family_name()).collect()
    }

    /// Look up a family by name
    #[must_use]
    pub fn get(&self, family_name: &str) -> Option<&dyn ModelFamily> {
        self.families
            .iter()
            .find(|f| f.family_name() == family_name)
            .map(|f| f.as_ref())
    }

    /// Detect model family from tensor names using best-match scoring.
    ///
    /// Scores each family by counting how many of its expected tensor patterns
    /// (embedding + per-layer for layer 0) match the given tensor names.
    /// Returns the family with the highest score, which disambiguates families
    /// with overlapping naming conventions (e.g., Qwen2's bias tensors
    /// distinguish it from LLaMA/DeepSeek/Mistral which share the same base
    /// naming but lack bias patterns).
    #[must_use]
    pub fn detect_family(&self, tensor_names: &[&str]) -> Option<&dyn ModelFamily> {
        let mut best: Option<(usize, &dyn ModelFamily)> = None;

        for family in &self.families {
            let config = family.config();

            // Must have the embedding tensor
            if !tensor_names.contains(&config.tensor_template.embedding.as_str()) {
                continue;
            }

            // Score: 1 point for embedding match + 1 for each per-layer pattern match
            let mut score = 1usize;
            for pattern in config.tensor_template.per_layer.values().flatten() {
                let layer0 = pattern.replace("{n}", "0");
                if tensor_names.contains(&layer0.as_str()) {
                    score += 1;
                }
            }

            // Need at least one per-layer match (score > 1)
            if score <= 1 {
                continue;
            }

            match best {
                None => best = Some((score, family.as_ref())),
                Some((best_score, _)) if score > best_score => {
                    best = Some((score, family.as_ref()));
                }
                _ => {}
            }
        }

        best.map(|(_, family)| family)
    }

    /// Detect model family from HuggingFace `model_type` string.
    #[must_use]
    pub fn detect_from_model_type(&self, model_type: &str) -> Option<&dyn ModelFamily> {
        let model_type_lower = model_type.to_lowercase();
        for family in &self.families {
            let config = family.config();
            for arch in &config.architectures {
                if arch.to_lowercase().contains(&model_type_lower)
                    || model_type_lower.contains(&config.family)
                {
                    return Some(family.as_ref());
                }
            }
        }
        None
    }

    /// Number of registered families
    #[must_use]
    pub fn len(&self) -> usize {
        self.families.len()
    }

    /// Check if registry is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.families.is_empty()
    }
}

impl Default for FamilyRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Build-Time Generated Code (PMAT-250)
// ============================================================================
//
// This include! pulls in code generated by build.rs from
// contracts/model-families/*.yaml. It provides:
// - KNOWN_FAMILIES: &[&str] — list of family names
// - Per-family const definitions (e.g., QWEN2_0_5B_HIDDEN_DIM)
// - build_default_registry() → FamilyRegistry with all families

include!(concat!(env!("OUT_DIR"), "/model_families_generated.rs"));

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_qwen2_config() -> ModelFamilyConfig {
        let mut size_variants = HashMap::new();
        size_variants.insert(
            "0.5b".to_string(),
            ModelSizeConfig {
                parameters: "0.5B".to_string(),
                hidden_dim: 896,
                num_layers: 24,
                num_heads: 14,
                num_kv_heads: 2,
                intermediate_dim: 4864,
                vocab_size: 151_936,
                max_position_embeddings: 32_768,
                head_dim: 64,
                rope_theta: 1_000_000.0,
                norm_eps: 0.000_001,
            },
        );
        size_variants.insert(
            "1.5b".to_string(),
            ModelSizeConfig {
                parameters: "1.5B".to_string(),
                hidden_dim: 1536,
                num_layers: 28,
                num_heads: 12,
                num_kv_heads: 2,
                intermediate_dim: 8960,
                vocab_size: 151_936,
                max_position_embeddings: 32_768,
                head_dim: 128,
                rope_theta: 1_000_000.0,
                norm_eps: 0.000_001,
            },
        );

        let mut per_layer = HashMap::new();
        per_layer.insert(
            "q_proj".to_string(),
            Some("model.layers.{n}.self_attn.q_proj.weight".to_string()),
        );
        per_layer.insert(
            "k_proj".to_string(),
            Some("model.layers.{n}.self_attn.k_proj.weight".to_string()),
        );
        per_layer.insert(
            "v_proj".to_string(),
            Some("model.layers.{n}.self_attn.v_proj.weight".to_string()),
        );
        per_layer.insert(
            "o_proj".to_string(),
            Some("model.layers.{n}.self_attn.o_proj.weight".to_string()),
        );
        per_layer.insert(
            "gate_proj".to_string(),
            Some("model.layers.{n}.mlp.gate_proj.weight".to_string()),
        );
        per_layer.insert(
            "up_proj".to_string(),
            Some("model.layers.{n}.mlp.up_proj.weight".to_string()),
        );
        per_layer.insert(
            "down_proj".to_string(),
            Some("model.layers.{n}.mlp.down_proj.weight".to_string()),
        );
        per_layer.insert(
            "input_layernorm".to_string(),
            Some("model.layers.{n}.input_layernorm.weight".to_string()),
        );
        per_layer.insert(
            "post_attention_layernorm".to_string(),
            Some("model.layers.{n}.post_attention_layernorm.weight".to_string()),
        );
        per_layer.insert(
            "q_proj_bias".to_string(),
            Some("model.layers.{n}.self_attn.q_proj.bias".to_string()),
        );
        per_layer.insert(
            "k_proj_bias".to_string(),
            Some("model.layers.{n}.self_attn.k_proj.bias".to_string()),
        );
        per_layer.insert(
            "v_proj_bias".to_string(),
            Some("model.layers.{n}.self_attn.v_proj.bias".to_string()),
        );

        let mut shapes = HashMap::new();
        shapes.insert(
            "embedding".to_string(),
            "[vocab_size, hidden_dim]".to_string(),
        );
        shapes.insert(
            "lm_head".to_string(),
            "[vocab_size, hidden_dim]".to_string(),
        );
        shapes.insert(
            "q_proj".to_string(),
            "[num_heads * head_dim, hidden_dim]".to_string(),
        );

        ModelFamilyConfig {
            family: "qwen2".to_string(),
            display_name: "Qwen2 / Qwen2.5-Coder".to_string(),
            vendor: "Alibaba".to_string(),
            architectures: vec!["Qwen2ForCausalLM".to_string()],
            hf_pattern: "Qwen/Qwen2*".to_string(),
            size_variants,
            constraints: ModelConstraints {
                attention_type: AttentionType::Gqa,
                activation: Activation::Silu,
                norm_type: NormType::RmsNorm,
                has_bias: true,
                tied_embeddings: false,
                positional_encoding: PositionalEncoding::Rope,
                mlp_type: MlpType::SwiGlu,
            },
            tensor_template: TensorTemplate {
                embedding: "model.embed_tokens.weight".to_string(),
                lm_head: Some("lm_head.weight".to_string()),
                final_norm: Some("model.norm.weight".to_string()),
                per_layer,
            },
            shape_template: ShapeTemplate { shapes },
            quantizations: vec!["q4_k_m".to_string(), "q6_k".to_string(), "f16".to_string()],
            chat_template: None,
            certification: None,
        }
    }

    #[test]
    fn test_dyn_family_detect_size() {
        let config = make_qwen2_config();
        let family = DynModelFamily::new(config);

        assert_eq!(family.detect_size(896, 24), Some("0.5b".to_string()));
        assert_eq!(family.detect_size(1536, 28), Some("1.5b".to_string()));
        assert_eq!(family.detect_size(999, 99), None);
    }

    #[test]
    fn test_dyn_family_expected_tensor_count() {
        let config = make_qwen2_config();
        let family = DynModelFamily::new(config);

        // 3 global (embedding, lm_head, final_norm) + 12 per-layer * 24 layers = 291
        assert_eq!(family.expected_tensor_count("0.5b"), Some(291));
        // 3 global + 12 per-layer * 28 layers = 339
        assert_eq!(family.expected_tensor_count("1.5b"), Some(339));
        assert_eq!(family.expected_tensor_count("unknown"), None);
    }

    #[test]
    fn test_registry_detect_family() {
        let config = make_qwen2_config();
        let family = DynModelFamily::new(config);

        let mut registry = FamilyRegistry::new();
        registry.register(Box::new(family));

        let names = vec![
            "model.embed_tokens.weight",
            "lm_head.weight",
            "model.norm.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
        ];

        let detected = registry.detect_family(&names);
        assert!(detected.is_some());
        assert_eq!(detected.expect("family detected").family_name(), "qwen2");
    }

    #[test]
    fn test_registry_detect_from_model_type() {
        let config = make_qwen2_config();
        let family = DynModelFamily::new(config);

        let mut registry = FamilyRegistry::new();
        registry.register(Box::new(family));

        let detected = registry.detect_from_model_type("qwen2");
        assert!(detected.is_some());
        assert_eq!(detected.expect("family detected").family_name(), "qwen2");
    }

    #[test]
    fn test_attention_type_parsing() {
        assert_eq!(
            AttentionType::from_str_contract("gqa").expect("parse"),
            AttentionType::Gqa
        );
        assert_eq!(
            AttentionType::from_str_contract("mha").expect("parse"),
            AttentionType::Mha
        );
        assert_eq!(
            AttentionType::from_str_contract("mqa").expect("parse"),
            AttentionType::Mqa
        );
        assert!(AttentionType::from_str_contract("unknown").is_err());
    }

    #[test]
    fn test_activation_parsing() {
        assert_eq!(
            Activation::from_str_contract("silu").expect("parse"),
            Activation::Silu
        );
        assert_eq!(
            Activation::from_str_contract("gelu").expect("parse"),
            Activation::Gelu
        );
        assert_eq!(
            Activation::from_str_contract("relu").expect("parse"),
            Activation::Relu
        );
        assert!(Activation::from_str_contract("unknown").is_err());
    }

    #[test]
    fn test_norm_type_parsing() {
        assert_eq!(
            NormType::from_str_contract("rmsnorm").expect("parse"),
            NormType::RmsNorm
        );
        assert_eq!(
            NormType::from_str_contract("layernorm").expect("parse"),
            NormType::LayerNorm
        );
        assert!(NormType::from_str_contract("unknown").is_err());
    }

    #[test]
    fn test_positional_encoding_parsing() {
        assert_eq!(
            PositionalEncoding::from_str_contract("rope").expect("parse"),
            PositionalEncoding::Rope
        );
        assert_eq!(
            PositionalEncoding::from_str_contract("absolute").expect("parse"),
            PositionalEncoding::Absolute
        );
        assert!(PositionalEncoding::from_str_contract("unknown").is_err());
    }

    #[test]
    fn test_mlp_type_parsing() {
        assert_eq!(
            MlpType::from_str_contract("swiglu").expect("parse"),
            MlpType::SwiGlu
        );
        assert_eq!(
            MlpType::from_str_contract("gelu_mlp").expect("parse"),
            MlpType::GeluMlp
        );
        assert!(MlpType::from_str_contract("unknown").is_err());
    }

    #[test]
    fn test_validate_tensor_names_rejects_wrong_family() {
        let config = make_qwen2_config();
        let family = DynModelFamily::new(config);

        let whisper_names = ["encoder.conv1.weight"];
        assert!(family
            .validate_tensor_names(&whisper_names, "0.5b")
            .is_err());
    }

    #[test]
    fn test_contract_error_display() {
        let err = ContractError {
            family: "qwen2".to_string(),
            message: "Missing tensor: lm_head.weight".to_string(),
        };
        assert_eq!(
            err.to_string(),
            "Model family contract error [qwen2]: Missing tensor: lm_head.weight"
        );
    }

    #[test]
    fn test_family_registry_empty() {
        let registry = FamilyRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
        assert!(registry.detect_family(&["foo"]).is_none());
    }

    // PMAT-250: Build-time generated code tests

    #[test]
    fn pmat_250_known_families_populated() {
        assert!(
            !KNOWN_FAMILIES.is_empty(),
            "KNOWN_FAMILIES should be populated by build.rs"
        );
        assert!(
            KNOWN_FAMILIES.contains(&"qwen2"),
            "KNOWN_FAMILIES should contain qwen2"
        );
        assert!(
            KNOWN_FAMILIES.contains(&"llama"),
            "KNOWN_FAMILIES should contain llama"
        );
    }

    #[test]
    fn pmat_250_build_default_registry() {
        let registry = build_default_registry();
        assert!(
            !registry.is_empty(),
            "Default registry should contain families from YAML contracts"
        );
        assert!(
            registry.len() >= 8,
            "Should have at least 8 families (bert, deepseek, gemma, llama, mistral, phi, qwen2, whisper), got {}",
            registry.len()
        );
    }

    #[test]
    fn pmat_250_default_registry_detects_qwen2() {
        let registry = build_default_registry();
        let detected = registry.detect_from_model_type("qwen2");
        assert!(detected.is_some(), "Should detect qwen2 family");
        let family = detected.expect("qwen2 detected");
        assert_eq!(family.family_name(), "qwen2");
    }

    #[test]
    fn pmat_250_default_registry_detects_llama() {
        let registry = build_default_registry();
        let detected = registry.detect_from_model_type("llama");
        assert!(detected.is_some(), "Should detect llama family");
    }

    #[test]
    fn pmat_250_generated_constants_correct() {
        assert_eq!(QWEN2_VENDOR, "Alibaba");
        assert_eq!(LLAMA_VENDOR, "Meta");
        assert_eq!(BERT_VENDOR, "Google");
        assert_eq!(WHISPER_VENDOR, "OpenAI");
        assert_eq!(MISTRAL_VENDOR, "Mistral AI");
        assert_eq!(PHI_VENDOR, "Microsoft");
        assert_eq!(GEMMA_VENDOR, "Google");
        assert_eq!(DEEPSEEK_VENDOR, "DeepSeek");

        // Verify some well-known size constants
        assert_eq!(QWEN2_0_5B_HIDDEN_DIM, 896);
        assert_eq!(QWEN2_0_5B_NUM_LAYERS, 24);
        assert_eq!(LLAMA_8B_HIDDEN_DIM, 4096);
        assert_eq!(LLAMA_8B_NUM_LAYERS, 32);
        assert_eq!(BERT_BASE_HIDDEN_DIM, 768);
        assert_eq!(WHISPER_TINY_HIDDEN_DIM, 384);
        assert_eq!(MISTRAL_7B_HIDDEN_DIM, 4096);
        assert_eq!(PHI_3_8B_HIDDEN_DIM, 3072);
        assert_eq!(GEMMA_2B_HIDDEN_DIM, 2048);
        assert_eq!(DEEPSEEK_7B_HIDDEN_DIM, 4096);
    }

    #[test]
    fn pmat_250_default_registry_size_detection() {
        let registry = build_default_registry();
        let qwen2 = registry
            .detect_from_model_type("qwen2")
            .expect("qwen2 detected");

        // Detect 0.5B by hidden_dim + num_layers
        let size = qwen2.detect_size(896, 24);
        assert_eq!(size.as_deref(), Some("0.5b"));

        // Detect 7B
        let size = qwen2.detect_size(3584, 28);
        assert_eq!(size.as_deref(), Some("7b"));
    }
}

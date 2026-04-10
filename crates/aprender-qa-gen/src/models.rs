//! Model registry and metadata
//!
//! Defines the `HuggingFace` model registry for qualification testing.

#![allow(clippy::struct_excessive_bools)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Model size category
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SizeCategory {
    /// 0.5B parameters
    Tiny,
    /// 1-2B parameters
    Small,
    /// 3-4B parameters
    Medium,
    /// 7-8B parameters
    Large,
    /// 13-14B parameters
    XLarge,
    /// 70B+ parameters
    Huge,
}

impl SizeCategory {
    /// Get the approximate parameter count
    #[must_use]
    pub const fn approx_params(&self) -> u64 {
        match self {
            Self::Tiny => 500_000_000,
            Self::Small => 1_500_000_000,
            Self::Medium => 3_500_000_000,
            Self::Large => 7_500_000_000,
            Self::XLarge => 13_500_000_000,
            Self::Huge => 70_000_000_000,
        }
    }

    /// Get memory estimate for F32 (4 bytes per param)
    #[must_use]
    pub const fn memory_f32_gb(&self) -> u64 {
        self.approx_params() * 4 / 1_000_000_000
    }

    /// Get memory estimate for `Q4_K` (0.5 bytes per param)
    #[must_use]
    pub const fn memory_q4k_gb(&self) -> u64 {
        self.approx_params() / 2 / 1_000_000_000
    }
}

/// Unique model identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelId {
    /// `HuggingFace` organization
    pub org: String,
    /// Model name
    pub name: String,
    /// Optional variant (e.g., "Instruct", "Chat")
    pub variant: Option<String>,
}

impl ModelId {
    /// Create a new model ID
    #[must_use]
    pub fn new(org: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            org: org.into(),
            name: name.into(),
            variant: None,
        }
    }

    /// Create a model ID with variant
    #[must_use]
    pub fn with_variant(
        org: impl Into<String>,
        name: impl Into<String>,
        variant: impl Into<String>,
    ) -> Self {
        Self {
            org: org.into(),
            name: name.into(),
            variant: Some(variant.into()),
        }
    }

    /// Get the full `HuggingFace` repo ID
    #[must_use]
    pub fn hf_repo(&self) -> String {
        self.variant.as_ref().map_or_else(
            || format!("{}/{}", self.org, self.name),
            |v| format!("{}/{}-{}", self.org, self.name, v),
        )
    }
}

impl std::fmt::Display for ModelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.hf_repo())
    }
}

/// Model metadata for qualification testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Model identifier
    pub id: ModelId,
    /// Size category
    pub size: SizeCategory,
    /// Architecture family (e.g., "qwen2", "llama", "mistral")
    pub architecture: String,
    /// Available quantizations
    pub quantizations: Vec<String>,
    /// Has chat template
    pub has_chat_template: bool,
    /// Supports system prompt
    pub supports_system_prompt: bool,
    /// Expected capabilities
    pub capabilities: ModelCapabilities,
}

/// Expected model capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCapabilities {
    /// Can do arithmetic (2+2=4)
    pub arithmetic: bool,
    /// Can do code completion
    pub code_completion: bool,
    /// Can follow instructions
    pub instruction_following: bool,
    /// Supports multi-turn conversation
    pub multi_turn: bool,
}

impl Default for ModelCapabilities {
    fn default() -> Self {
        Self {
            arithmetic: true,
            instruction_following: true,
            code_completion: false,
            multi_turn: true,
        }
    }
}

/// Registry of models for qualification testing
#[derive(Debug, Clone, Default)]
pub struct ModelRegistry {
    models: HashMap<String, ModelMetadata>,
}

impl ModelRegistry {
    /// Create a new empty registry
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create registry with default `HuggingFace` top models
    #[must_use]
    pub fn with_defaults() -> Self {
        let mut registry = Self::new();
        registry.add_default_models();
        registry
    }

    /// Add a model to the registry
    pub fn add(&mut self, metadata: ModelMetadata) {
        self.models.insert(metadata.id.hf_repo(), metadata);
    }

    /// Get model metadata by ID
    #[must_use]
    pub fn get(&self, id: &str) -> Option<&ModelMetadata> {
        self.models.get(id)
    }

    /// Get all models
    #[must_use]
    pub fn all(&self) -> Vec<&ModelMetadata> {
        self.models.values().collect()
    }

    /// Get models by size category
    #[must_use]
    pub fn by_size(&self, size: SizeCategory) -> Vec<&ModelMetadata> {
        self.models.values().filter(|m| m.size == size).collect()
    }

    /// Number of registered models
    #[must_use]
    pub fn len(&self) -> usize {
        self.models.len()
    }

    /// Check if registry is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }

    fn add_default_models(&mut self) {
        self.add_qwen25_general_models();
        self.add_qwen25_coder_models();
        self.add_qwen3_models();
        self.add_llama_models();
        self.add_mistral_models();
        self.add_gemma_models();
        self.add_phi_models();
        self.add_deepseek_coder_models();
        self.add_deepseek_r1_models();
        self.add_starcoder_models();
        self.add_yi_models();
        self.add_small_models();
        self.add_falcon_models();
        self.add_internlm_models();
        self.add_granite_models();
        self.add_olmo_models();
        self.add_nvidia_models();
        self.add_community_models();
    }

    fn add_qwen25_general_models(&mut self) {
        self.add(ModelMetadata {
            id: ModelId::with_variant("Qwen", "Qwen2.5-0.5B", "Instruct"),
            size: SizeCategory::Tiny,
            architecture: "qwen2".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        self.add(ModelMetadata {
            id: ModelId::with_variant("Qwen", "Qwen2.5-1.5B", "Instruct"),
            size: SizeCategory::Small,
            architecture: "qwen2".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string(), "f16".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        self.add(ModelMetadata {
            id: ModelId::with_variant("Qwen", "Qwen2.5-3B", "Instruct"),
            size: SizeCategory::Medium,
            architecture: "qwen2".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string(), "f16".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        self.add(ModelMetadata {
            id: ModelId::with_variant("Qwen", "Qwen2.5-7B", "Instruct"),
            size: SizeCategory::Large,
            architecture: "qwen2".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string(), "f16".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        self.add(ModelMetadata {
            id: ModelId::with_variant("Qwen", "Qwen2.5-14B", "Instruct"),
            size: SizeCategory::XLarge,
            architecture: "qwen2".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        self.add(ModelMetadata {
            id: ModelId::with_variant("Qwen", "Qwen2.5-32B", "Instruct"),
            size: SizeCategory::Huge,
            architecture: "qwen2".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        self.add(ModelMetadata {
            id: ModelId::with_variant("Qwen", "Qwen2.5-72B", "Instruct"),
            size: SizeCategory::Huge,
            architecture: "qwen2".to_string(),
            quantizations: vec!["q4_k_m".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        // QwQ reasoning model
        self.add(ModelMetadata {
            id: ModelId::new("Qwen", "QwQ-32B"),
            size: SizeCategory::Huge,
            architecture: "qwen2".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });
    }

    fn add_qwen25_coder_models(&mut self) {
        self.add(ModelMetadata {
            id: ModelId::with_variant("Qwen", "Qwen2.5-Coder-0.5B", "Instruct"),
            size: SizeCategory::Tiny,
            architecture: "qwen2".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities {
                code_completion: true,
                ..Default::default()
            },
        });

        self.add(ModelMetadata {
            id: ModelId::with_variant("Qwen", "Qwen2.5-Coder-1.5B", "Instruct"),
            size: SizeCategory::Small,
            architecture: "qwen2".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string(), "f16".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities {
                code_completion: true,
                ..Default::default()
            },
        });

        self.add(ModelMetadata {
            id: ModelId::with_variant("Qwen", "Qwen2.5-Coder-3B", "Instruct"),
            size: SizeCategory::Medium,
            architecture: "qwen2".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string(), "f16".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities {
                code_completion: true,
                ..Default::default()
            },
        });

        self.add(ModelMetadata {
            id: ModelId::with_variant("Qwen", "Qwen2.5-Coder-7B", "Instruct"),
            size: SizeCategory::Large,
            architecture: "qwen2".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string(), "f16".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities {
                code_completion: true,
                ..Default::default()
            },
        });

        self.add(ModelMetadata {
            id: ModelId::with_variant("Qwen", "Qwen2.5-Coder-14B", "Instruct"),
            size: SizeCategory::XLarge,
            architecture: "qwen2".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities {
                code_completion: true,
                ..Default::default()
            },
        });

        self.add(ModelMetadata {
            id: ModelId::with_variant("Qwen", "Qwen2.5-Coder-32B", "Instruct"),
            size: SizeCategory::Huge,
            architecture: "qwen2".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities {
                code_completion: true,
                ..Default::default()
            },
        });
    }

    fn add_qwen3_models(&mut self) {
        self.add(ModelMetadata {
            id: ModelId::new("Qwen", "Qwen3-0.6B"),
            size: SizeCategory::Tiny,
            architecture: "qwen3".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        self.add(ModelMetadata {
            id: ModelId::new("Qwen", "Qwen3-1.7B"),
            size: SizeCategory::Small,
            architecture: "qwen3".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string(), "f16".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        self.add(ModelMetadata {
            id: ModelId::new("Qwen", "Qwen3-4B"),
            size: SizeCategory::Medium,
            architecture: "qwen3".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string(), "f16".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        self.add(ModelMetadata {
            id: ModelId::new("Qwen", "Qwen3-8B"),
            size: SizeCategory::Large,
            architecture: "qwen3".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string(), "f16".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        self.add(ModelMetadata {
            id: ModelId::new("Qwen", "Qwen3-14B"),
            size: SizeCategory::XLarge,
            architecture: "qwen3".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        self.add(ModelMetadata {
            id: ModelId::new("Qwen", "Qwen3-32B"),
            size: SizeCategory::Huge,
            architecture: "qwen3".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        self.add(ModelMetadata {
            id: ModelId::new("Qwen", "Qwen3-30B-A3B"),
            size: SizeCategory::Huge,
            architecture: "qwen3_moe".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        self.add(ModelMetadata {
            id: ModelId::with_variant("Qwen", "Qwen3-Coder-30B-A3B", "Instruct"),
            size: SizeCategory::Huge,
            architecture: "qwen3_moe".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities {
                code_completion: true,
                ..Default::default()
            },
        });
    }
}

// Additional model families (llama, mistral, gemma, phi, deepseek-coder)
include!("models_registry_b.rs");

// Additional model families (deepseek-r1, starcoder, yi, community, etc.)
include!("models_registry_c.rs");

#[cfg(test)]
#[path = "models_tests.rs"]
mod tests;

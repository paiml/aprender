//! APR Converter Types
//!
//! Type definitions extracted from converter.rs for better modularity.
//! Part of PMAT-197: File size reduction.

use crate::error::{AprenderError, Result};
use crate::format::validation::TensorStats;
use crate::format::Compression;
use std::collections::BTreeMap;
use std::path::PathBuf;

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
            let joined = parts[2..].join("/");
            // GH-221: Strip HuggingFace web URL path components.
            // Users copy URLs like hf://org/repo/resolve/main/file.safetensors
            // or hf://org/repo/blob/main/file.safetensors from the browser.
            let cleaned = joined
                .strip_prefix("resolve/main/")
                .or_else(|| joined.strip_prefix("blob/main/"))
                .unwrap_or(&joined);
            // Also handle bare "resolve/main" or "blob/main" (no trailing slash, no file)
            let cleaned = if cleaned == "resolve/main" || cleaned == "blob/main" {
                ""
            } else {
                cleaned
            };
            if cleaned.is_empty() {
                None
            } else {
                Some(cleaned.to_string())
            }
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
    /// Meta `LLaMA` (also SmolLM2, Granite, Nemotron derivatives)
    Llama,
    /// Google BERT
    Bert,
    /// Alibaba Qwen2 (includes Qwen2.5, `QwenCoder`)
    Qwen2,
    /// Alibaba Qwen3
    Qwen3,
    /// Alibaba Qwen3.5 (hybrid linear/quadratic attention)
    Qwen3_5,
    /// `OpenAI` GPT-2 (also `StarCoder`, `bigcode` models)
    Gpt2,
    /// Microsoft Phi (Phi-3, Phi-4)
    Phi,
    /// `EleutherAI` GPT-NeoX (Pythia, GPT-NeoX-20B)
    GptNeoX,
    /// Meta OPT (Open Pre-trained Transformer)
    Opt,
}

/// Model category for tokenizer and preprocessor requirements (SPEC-MODEL-TYPE-001).
///
/// Orthogonal to kernel class (compute dispatch). See contract
/// `contracts/aprender/model-type-taxonomy-v1.yaml`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelCategory {
    /// Text-generating language model — requires text tokenizer for inference.
    Llm,
    /// Classical/statistical ML model — numeric input, no tokenizer needed.
    Ml,
    /// Speech/audio model — uses mel spectrogram features, not text tokenizer.
    Audio,
    /// Classification/embedding model — no text generation.
    Embedding,
}

impl Architecture {
    /// Returns true for text-generating LLM architectures that require a tokenizer.
    ///
    /// Audio (Whisper) and embedding (BERT) models do not require text tokenizers.
    /// `Auto` defaults to true (fail-safe: require tokenizer unless proven otherwise).
    ///
    /// Contract: `contracts/aprender/model-type-taxonomy-v1.yaml` equation `is_llm`.
    #[must_use]
    pub const fn is_llm(&self) -> bool {
        matches!(
            self,
            Self::Llama
                | Self::Qwen2
                | Self::Qwen3
                | Self::Qwen3_5
                | Self::Gpt2
                | Self::Phi
                | Self::GptNeoX
                | Self::Opt
                | Self::Auto
        )
    }

    /// Returns the model category for this architecture.
    #[must_use]
    pub const fn category(&self) -> ModelCategory {
        match self {
            Self::Whisper => ModelCategory::Audio,
            Self::Bert => ModelCategory::Embedding,
            _ => ModelCategory::Llm,
        }
    }
}

include!("tensor_expectation.rs");
include!("converter_types_expectations.rs");
include!("sharded_index.rs");
include!("converter_types_tests.rs");

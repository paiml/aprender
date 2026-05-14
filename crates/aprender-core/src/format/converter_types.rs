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
    ///
    /// GH-729: GGUF-only repos (name contains "GGUF") should not default to
    /// `model.safetensors`. We return a sentinel that the download layer
    /// can use to try GGUF files instead.
    #[must_use]
    pub fn default_file(&self) -> &str {
        match self {
            Self::HuggingFace { file: Some(f), .. } => f,
            Self::HuggingFace {
                file: None, repo, ..
            } => {
                if repo.to_lowercase().contains("gguf") {
                    // GGUF-only repo — don't try model.safetensors (404 guaranteed)
                    "model.gguf"
                } else {
                    "model.safetensors"
                }
            }
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
    /// DeepSeek (DeepSeek-V2, DeepSeek-Coder)
    DeepSeek,
    /// Google Gemma (Gemma-2, CodeGemma)
    Gemma,
    /// Mistral AI (Mistral, Mixtral)
    Mistral,
    /// TII Falcon-H1 (Hybrid Transformer+SSM)
    FalconH1,
    /// state-spaces Mamba (State Space Model)
    Mamba,
    /// `UsefulSensors` Moonshine (audio, encoder-decoder)
    Moonshine,
    /// Apple `OpenELM` (variable-width attention)
    OpenElm,
    /// RWKV-7 (linear attention / recurrence)
    Rwkv7,
    /// TII Falcon classic (`FalconForCausalLM`) — MQA/MGQA, RoPE,
    /// parallel attn+mlp residuals, `transformer.h.N.*` HF naming.
    /// GH-1587: tensor name mapping only — parallel-residual runtime
    /// inference is a separate concern.
    FalconClassic,
    /// `BigScience` BLOOM (ALiBi position bias, fused QKV).
    /// GH-1586: tensor name mapping only — full ALiBi runtime inference is
    /// a separate concern (`is_inference_verified()` returns false).
    Bloom,
}

include!("tensor_expectation.rs");
include!("converter_types_expectations.rs");
include!("sharded_index.rs");
include!("converter_types_tests.rs");

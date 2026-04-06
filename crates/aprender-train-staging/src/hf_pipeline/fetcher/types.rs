//! Type definitions for HuggingFace model fetching.
//!
//! Contains core enums and structs for model weights, architectures, and artifacts.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Model weight format
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WeightFormat {
    /// SafeTensors format (recommended, secure)
    SafeTensors,
    /// GGUF quantized format
    GGUF { quant_type: String },
    /// PyTorch pickle format (SECURITY RISK)
    PyTorchBin,
    /// ONNX format
    ONNX,
}

impl WeightFormat {
    /// Detect format from filename
    #[must_use]
    pub fn from_filename(filename: &str) -> Option<Self> {
        if filename.ends_with(".safetensors") {
            Some(Self::SafeTensors)
        } else if filename.ends_with(".gguf") {
            Some(Self::GGUF { quant_type: "unknown".into() })
        } else if filename.ends_with(".bin") {
            Some(Self::PyTorchBin)
        } else if filename.ends_with(".onnx") {
            Some(Self::ONNX)
        } else {
            None
        }
    }

    /// Check if format is safe (no arbitrary code execution)
    #[must_use]
    pub fn is_safe(&self) -> bool {
        matches!(self, Self::SafeTensors | Self::GGUF { .. } | Self::ONNX)
    }
}

/// Model architecture information
// CB-519: Serialize + Deserialize derive is intentional for config round-trip.
// PartialEq enables exact structural validation (not just param_count) after deserialization.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Architecture {
    /// BERT-style encoder
    BERT { num_layers: usize, hidden_size: usize, num_attention_heads: usize },
    /// GPT-style decoder
    GPT2 { num_layers: usize, hidden_size: usize, num_attention_heads: usize },
    /// Llama architecture
    Llama {
        num_layers: usize,
        hidden_size: usize,
        num_attention_heads: usize,
        intermediate_size: usize,
    },
    /// T5 encoder-decoder
    T5 { encoder_layers: usize, decoder_layers: usize, hidden_size: usize },
    /// Custom/unknown architecture
    Custom { config: serde_json::Value },
}

impl Architecture {
    /// Estimate parameter count
    #[must_use]
    pub fn param_count(&self) -> u64 {
        match self {
            Self::BERT { num_layers, hidden_size, num_attention_heads: _ } => {
                // Rough estimate: 4 * hidden^2 per layer (Q, K, V, O projections + FFN)
                let per_layer = 4 * (*hidden_size as u64).pow(2) + 4 * (*hidden_size as u64).pow(2);
                per_layer * (*num_layers as u64)
            }
            Self::GPT2 { num_layers, hidden_size, .. } => {
                let per_layer = 4 * (*hidden_size as u64).pow(2) + 4 * (*hidden_size as u64).pow(2);
                per_layer * (*num_layers as u64)
            }
            Self::Llama { num_layers, hidden_size, intermediate_size, .. } => {
                let attn = 4 * (*hidden_size as u64).pow(2);
                let ffn = 2 * (*hidden_size as u64) * (*intermediate_size as u64);
                (attn + ffn) * (*num_layers as u64)
            }
            Self::T5 { encoder_layers, decoder_layers, hidden_size } => {
                let per_layer = 8 * (*hidden_size as u64).pow(2);
                per_layer * ((*encoder_layers + *decoder_layers) as u64)
            }
            Self::Custom { .. } => 0, // Unknown
        }
    }
}

/// Downloaded model artifact
#[derive(Debug)]
pub struct ModelArtifact {
    /// Local path to downloaded files
    pub path: PathBuf,
    /// Detected weight format
    pub format: WeightFormat,
    /// Model architecture (parsed from config.json)
    pub architecture: Option<Architecture>,
    /// SHA256 hash of model file
    pub sha256: Option<String>,
}

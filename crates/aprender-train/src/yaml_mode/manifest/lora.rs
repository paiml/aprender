//! LoRA Configuration
//!
//! Contains Low-Rank Adaptation configuration types for training manifests.

use serde::{Deserialize, Serialize};

/// LoRA (Low-Rank Adaptation) configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraConfig {
    /// Enable LoRA
    pub enabled: bool,

    /// Rank of low-rank matrices (r)
    pub rank: usize,

    /// Scaling factor (alpha)
    pub alpha: f64,

    /// LoRA dropout
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dropout: Option<f64>,

    /// Target modules for LoRA
    pub target_modules: Vec<String>,

    /// Target modules pattern (regex)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_modules_pattern: Option<String>,

    /// Bias training (none, all, lora_only)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bias: Option<String>,

    /// Weight initialization (gaussian, xavier, kaiming)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub init_weights: Option<String>,

    /// QLoRA: Quantize base model
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quantize_base: Option<bool>,

    /// QLoRA: Quantization bits
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quantize_bits: Option<u8>,

    /// QLoRA: Double quantization
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub double_quantize: Option<bool>,

    /// QLoRA: Quantization type (nf4, fp4)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quant_type: Option<String>,
}

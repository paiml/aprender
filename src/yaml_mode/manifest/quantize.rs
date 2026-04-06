//! Quantization Configuration
//!
//! Contains quantization-related configuration types for training manifests.

use serde::{Deserialize, Serialize};

/// Quantization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizeConfig {
    /// Enable quantization
    pub enabled: bool,

    /// Quantization bits (2, 4, 8)
    pub bits: u8,

    /// Quantization scheme (symmetric, asymmetric, dynamic)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scheme: Option<String>,

    /// Granularity (per_tensor, per_channel, per_group)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub granularity: Option<String>,

    /// Group size for per_group quantization
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub group_size: Option<usize>,

    /// QAT configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub qat: Option<QatConfig>,

    /// PTQ calibration configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calibration: Option<CalibrationConfig>,

    /// Layers to exclude from quantization
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exclude: Option<Vec<String>>,
}

/// Quantization-aware training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QatConfig {
    pub enabled: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub observer: Option<String>,
}

/// Post-training quantization calibration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub samples: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub method: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub percentile: Option<f64>,
}

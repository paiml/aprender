//! Output Configuration
//!
//! Contains output and artifact configuration types for training manifests.

use serde::{Deserialize, Serialize};

/// Output and artifact configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    /// Output directory (supports templates)
    pub dir: String,

    /// Model output configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<ModelOutputConfig>,

    /// Metrics export configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metrics: Option<MetricsOutputConfig>,

    /// Training report configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub report: Option<ReportConfig>,

    /// Artifact registry configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub registry: Option<RegistryConfig>,
}

/// Model output configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelOutputConfig {
    /// Output format (safetensors, pt, gguf, apr)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,

    /// Save optimizer state
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub save_optimizer: Option<bool>,

    /// Save scheduler state
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub save_scheduler: Option<bool>,
}

/// Metrics output configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsOutputConfig {
    /// Output format (parquet, csv, json)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,

    /// Metrics to include
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub include: Option<Vec<String>>,
}

/// Training report configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportConfig {
    pub enabled: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub include_plots: Option<bool>,
}

/// Artifact registry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryConfig {
    pub enabled: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub include_config: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub include_metrics: Option<bool>,
}

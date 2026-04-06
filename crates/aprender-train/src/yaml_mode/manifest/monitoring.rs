//! Monitoring Configuration
//!
//! Contains monitoring and tracking configuration types for training manifests.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Monitoring configuration (genchi genbutsu - go and see)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// Terminal visualization
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub terminal: Option<TerminalMonitor>,

    /// Experiment tracking
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tracking: Option<TrackingConfig>,

    /// System metrics
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<SystemMonitorConfig>,

    /// Alerts (Andon system)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alerts: Option<Vec<AlertConfig>>,

    /// Drift detection configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub drift_detection: Option<DriftDetectionConfig>,
}

/// Drift detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftDetectionConfig {
    /// Whether drift detection is enabled
    pub enabled: bool,

    /// Path to baseline statistics
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub baseline: Option<String>,

    /// Threshold for triggering drift alert (e.g., PSI threshold)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threshold: Option<f64>,

    /// Window size for drift detection
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub window: Option<usize>,

    /// Features to monitor for drift
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub features: Option<Vec<String>>,

    /// Drift detection method (e.g., "psi", "ks", "wasserstein")
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub method: Option<String>,
}

/// Terminal monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerminalMonitor {
    /// Enable terminal monitoring
    pub enabled: bool,

    /// Refresh rate in ms
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refresh_rate: Option<usize>,

    /// Metrics to display
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metrics: Option<Vec<String>>,

    /// Charts configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub charts: Option<Vec<ChartConfig>>,
}

/// Chart configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartConfig {
    #[serde(rename = "type")]
    pub chart_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metric: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub window: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub show_eta: Option<bool>,
}

/// Experiment tracking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackingConfig {
    /// Enable tracking
    pub enabled: bool,

    /// Backend (trueno-db, mlflow, wandb, tensorboard)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend: Option<String>,

    /// Project name
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project: Option<String>,

    /// Experiment name (supports templates)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub experiment: Option<String>,

    /// Tags
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tags: Option<HashMap<String, String>>,
}

/// System monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMonitorConfig {
    pub enabled: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interval: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metrics: Option<Vec<String>>,
}

/// Alert configuration (Andon system)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertConfig {
    pub condition: String,
    pub action: String,
    pub message: String,
}

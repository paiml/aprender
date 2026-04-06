//! Scheduler Configuration
//!
//! Contains learning rate scheduler configuration types for training manifests.

use serde::{Deserialize, Serialize};

/// Learning rate scheduler configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    /// Scheduler name (step, cosine, linear, exponential, plateau, one_cycle)
    pub name: String,

    /// Warmup configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub warmup: Option<WarmupConfig>,

    /// Cosine annealing T_max
    #[serde(rename = "T_max", default, skip_serializing_if = "Option::is_none")]
    pub t_max: Option<usize>,

    /// Cosine annealing eta_min
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eta_min: Option<f64>,

    /// Step scheduler step_size
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub step_size: Option<usize>,

    /// Step/exponential gamma
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gamma: Option<f64>,

    /// Plateau scheduler mode (min, max)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<String>,

    /// Plateau scheduler factor
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub factor: Option<f64>,

    /// Plateau scheduler patience
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub patience: Option<usize>,

    /// Plateau scheduler threshold
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threshold: Option<f64>,

    /// One-cycle max_lr
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_lr: Option<f64>,

    /// One-cycle pct_start
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pct_start: Option<f64>,

    /// One-cycle anneal_strategy
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub anneal_strategy: Option<String>,

    /// One-cycle div_factor
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub div_factor: Option<f64>,

    /// One-cycle final_div_factor
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub final_div_factor: Option<f64>,
}

/// Warmup configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarmupConfig {
    /// Warmup steps
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub steps: Option<usize>,

    /// Warmup ratio (alternative to steps)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ratio: Option<f64>,

    /// Starting learning rate
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start_lr: Option<f64>,
}

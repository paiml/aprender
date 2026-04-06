//! Optimizer Configuration
//!
//! Contains optimizer-related configuration types for training manifests.

use serde::{Deserialize, Serialize};

/// Optimizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerConfig {
    /// Optimizer name (sgd, adam, adamw, rmsprop, adagrad, lamb)
    pub name: String,

    /// Learning rate
    pub lr: f64,

    /// Weight decay (L2 regularization)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weight_decay: Option<f64>,

    /// Adam/AdamW betas
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub betas: Option<Vec<f64>>,

    /// Adam epsilon
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eps: Option<f64>,

    /// AMSGrad variant
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub amsgrad: Option<bool>,

    /// SGD momentum
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub momentum: Option<f64>,

    /// Nesterov momentum
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nesterov: Option<bool>,

    /// SGD dampening
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dampening: Option<f64>,

    /// RMSprop alpha
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alpha: Option<f64>,

    /// RMSprop centered
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub centered: Option<bool>,

    /// Per-parameter groups
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub param_groups: Option<Vec<ParamGroup>>,
}

/// Per-parameter group configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParamGroup {
    pub params: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lr: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weight_decay: Option<f64>,
}

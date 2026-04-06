//! Configuration for DP-SGD.

use serde::{Deserialize, Serialize};

use super::budget::PrivacyBudget;
use super::error::{DpError, Result};

/// Configuration for DP-SGD
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DpSgdConfig {
    /// Maximum gradient norm for clipping
    pub max_grad_norm: f64,
    /// Noise multiplier (sigma = noise_multiplier * max_grad_norm)
    pub noise_multiplier: f64,
    /// Privacy budget
    pub budget: PrivacyBudget,
    /// Sampling rate (batch_size / dataset_size)
    pub sample_rate: f64,
    /// Whether to stop training when budget is exhausted
    pub strict_budget: bool,
}

impl DpSgdConfig {
    /// Create a new DP-SGD configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum gradient norm
    pub fn with_max_grad_norm(mut self, norm: f64) -> Self {
        self.max_grad_norm = norm;
        self
    }

    /// Set noise multiplier
    pub fn with_noise_multiplier(mut self, multiplier: f64) -> Self {
        self.noise_multiplier = multiplier.max(0.0);
        self
    }

    /// Set privacy budget
    pub fn with_budget(mut self, budget: PrivacyBudget) -> Self {
        self.budget = budget;
        self
    }

    /// Set sampling rate
    pub fn with_sample_rate(mut self, rate: f64) -> Self {
        self.sample_rate = rate.clamp(0.0, 1.0);
        self
    }

    /// Set strict budget enforcement
    pub fn with_strict_budget(mut self, strict: bool) -> Self {
        self.strict_budget = strict;
        self
    }

    /// Compute noise standard deviation
    pub fn noise_std(&self) -> f64 {
        self.noise_multiplier * self.max_grad_norm
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        if self.max_grad_norm <= 0.0 {
            return Err(DpError::InvalidConfig("max_grad_norm must be positive".to_string()));
        }
        if self.noise_multiplier < 0.0 {
            return Err(DpError::InvalidConfig(
                "noise_multiplier must be non-negative".to_string(),
            ));
        }
        if self.budget.epsilon <= 0.0 {
            return Err(DpError::InvalidConfig("epsilon must be positive".to_string()));
        }
        if self.budget.delta <= 0.0 || self.budget.delta >= 1.0 {
            return Err(DpError::InvalidConfig("delta must be in (0, 1)".to_string()));
        }
        Ok(())
    }
}

impl Default for DpSgdConfig {
    fn default() -> Self {
        Self {
            max_grad_norm: 1.0,
            noise_multiplier: 1.1,
            budget: PrivacyBudget::default(),
            sample_rate: 0.01, // 1% batch size
            strict_budget: true,
        }
    }
}

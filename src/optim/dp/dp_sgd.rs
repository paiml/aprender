//! Differentially private SGD wrapper.
//!
//! Wraps any optimizer with DP guarantees by:
//! 1. Per-sample gradient clipping
//! 2. Adding calibrated Gaussian noise
//! 3. Privacy accounting

use super::accountant::RdpAccountant;
use super::config::DpSgdConfig;
use super::error::{DpError, Result};
use super::gradient::{add_gaussian_noise, clip_gradient};

/// Differentially private SGD wrapper
///
/// Wraps any optimizer with DP guarantees by:
/// 1. Per-sample gradient clipping
/// 2. Adding calibrated Gaussian noise
/// 3. Privacy accounting
#[derive(Debug, Clone)]
pub struct DpSgd {
    /// DP configuration
    config: DpSgdConfig,
    /// Privacy accountant
    accountant: RdpAccountant,
    /// Learning rate
    learning_rate: f64,
}

impl DpSgd {
    /// Create a new DP-SGD optimizer
    pub fn new(learning_rate: f64, config: DpSgdConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self { config, accountant: RdpAccountant::new(), learning_rate })
    }

    /// Get current privacy spent
    pub fn privacy_spent(&self) -> (f64, f64) {
        self.accountant.get_privacy_spent(self.config.budget.delta)
    }

    /// Get current epsilon
    pub fn current_epsilon(&self) -> f64 {
        self.privacy_spent().0
    }

    /// Get remaining budget
    pub fn remaining_budget(&self) -> f64 {
        self.config.budget.remaining(self.current_epsilon())
    }

    /// Check if budget is exhausted
    pub fn is_budget_exhausted(&self) -> bool {
        !self.config.budget.allows(self.current_epsilon())
    }

    /// Get number of training steps
    pub fn n_steps(&self) -> usize {
        self.accountant.n_steps()
    }

    /// Get configuration
    pub fn config(&self) -> &DpSgdConfig {
        &self.config
    }

    /// Process per-sample gradients with DP mechanism
    ///
    /// Returns the privatized aggregated gradient
    pub fn privatize_gradients(&mut self, per_sample_grads: &[Vec<f64>]) -> Result<Vec<f64>> {
        if per_sample_grads.is_empty() {
            return Err(DpError::GradientError("No gradients provided".to_string()));
        }

        // Check budget
        if self.config.strict_budget && self.is_budget_exhausted() {
            return Err(DpError::BudgetExhausted {
                spent: self.current_epsilon(),
                budget: self.config.budget.epsilon,
            });
        }

        let n_samples = per_sample_grads.len();
        let grad_dim = per_sample_grads[0].len();

        // Step 1: Clip each per-sample gradient
        let clipped: Vec<Vec<f64>> =
            per_sample_grads.iter().map(|g| clip_gradient(g, self.config.max_grad_norm)).collect();

        // Step 2: Average clipped gradients
        let mut averaged = vec![0.0; grad_dim];
        for g in &clipped {
            for (i, &val) in g.iter().enumerate() {
                averaged[i] += val / n_samples as f64;
            }
        }

        // Step 3: Add Gaussian noise
        let mut rng = rand::rng();
        let noise_std = self.config.noise_std() / n_samples as f64;
        let noised = add_gaussian_noise(&averaged, noise_std, &mut rng);

        // Step 4: Update privacy accounting
        self.accountant.step(self.config.noise_multiplier, self.config.sample_rate);

        Ok(noised)
    }

    /// Apply gradient update to parameters
    pub fn apply_update(&self, params: &mut [f64], grad: &[f64]) {
        for (p, g) in params.iter_mut().zip(grad.iter()) {
            *p -= self.learning_rate * g;
        }
    }

    /// Full DP-SGD step: privatize gradients and update parameters
    pub fn step(
        &mut self,
        params: &mut [f64],
        per_sample_grads: &[Vec<f64>],
    ) -> Result<(f64, f64)> {
        let grad = self.privatize_gradients(per_sample_grads)?;
        self.apply_update(params, &grad);
        Ok(self.privacy_spent())
    }

    /// Reset privacy accountant
    pub fn reset(&mut self) {
        self.accountant.reset();
    }
}

//! Direct Preference Optimization (DPO) (GH-449)
//!
//! Implements the DPO algorithm from Rafailov et al. 2023:
//! "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
//!
//! DPO directly optimizes the policy using preference pairs (chosen/rejected)
//! without training a separate reward model (unlike RLHF/PPO).
//!
//! Loss: L_DPO = -E[log σ(β · (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))]
//!
//! where y_w = chosen response, y_l = rejected response, β = temperature

use crate::error::{AprenderError, Result};

/// Configuration for DPO training.
#[derive(Debug, Clone)]
pub struct DpoConfig {
    /// Temperature parameter (β). Higher β = more conservative updates.
    /// Typical range: 0.1 to 0.5
    pub beta: f64,
    /// Learning rate
    pub learning_rate: f64,
    /// Label smoothing (0.0 = none). Reduces overfitting to preference pairs.
    pub label_smoothing: f64,
    /// Whether to use reference model log-probs (standard DPO)
    /// If false, uses SimPO variant (reference-free)
    pub use_reference: bool,
    /// Length normalization for SimPO variant
    pub length_normalize: bool,
}

impl Default for DpoConfig {
    fn default() -> Self {
        Self {
            beta: 0.1,
            learning_rate: 5e-7,
            label_smoothing: 0.0,
            use_reference: true,
            length_normalize: false,
        }
    }
}

impl DpoConfig {
    /// Validate configuration constraints.
    pub fn validate(&self) -> Result<()> {
        if self.beta <= 0.0 || !self.beta.is_finite() {
            return Err(AprenderError::FormatError {
                message: format!("beta must be positive finite, got {}", self.beta),
            });
        }
        if self.learning_rate <= 0.0 || !self.learning_rate.is_finite() {
            return Err(AprenderError::FormatError {
                message: format!(
                    "learning_rate must be positive finite, got {}",
                    self.learning_rate
                ),
            });
        }
        if self.label_smoothing < 0.0 || self.label_smoothing >= 1.0 {
            return Err(AprenderError::FormatError {
                message: format!(
                    "label_smoothing must be in [0.0, 1.0), got {}",
                    self.label_smoothing
                ),
            });
        }
        Ok(())
    }
}

/// A preference pair for DPO training.
#[derive(Debug, Clone)]
pub struct PreferencePair {
    /// Log-probability of chosen response under current policy
    pub chosen_logprob: f64,
    /// Log-probability of rejected response under current policy
    pub rejected_logprob: f64,
    /// Log-probability of chosen response under reference policy
    pub ref_chosen_logprob: f64,
    /// Log-probability of rejected response under reference policy
    pub ref_rejected_logprob: f64,
}

/// DPO loss calculator.
#[derive(Debug, Clone)]
pub struct DpoLoss {
    config: DpoConfig,
}

impl DpoLoss {
    /// Create a new DPO loss calculator.
    #[must_use]
    pub fn new(config: DpoConfig) -> Self {
        Self { config }
    }

    /// Compute DPO loss for a single preference pair.
    ///
    /// L = -log σ(β · (log_ratio_chosen - log_ratio_rejected))
    ///
    /// where log_ratio = log π(y|x) - log π_ref(y|x)
    #[must_use]
    pub fn compute(&self, pair: &PreferencePair) -> f64 {
        let log_ratio_chosen = if self.config.use_reference {
            pair.chosen_logprob - pair.ref_chosen_logprob
        } else {
            pair.chosen_logprob
        };

        let log_ratio_rejected = if self.config.use_reference {
            pair.rejected_logprob - pair.ref_rejected_logprob
        } else {
            pair.rejected_logprob
        };

        let logit = self.config.beta * (log_ratio_chosen - log_ratio_rejected);

        // -log σ(logit) with label smoothing
        if self.config.label_smoothing > 0.0 {
            let eps = self.config.label_smoothing;
            // Smoothed: -(1-eps) * log σ(logit) - eps * log σ(-logit)
            -(1.0 - eps) * log_sigmoid(logit) - eps * log_sigmoid(-logit)
        } else {
            -log_sigmoid(logit)
        }
    }

    /// Compute DPO loss for a batch of preference pairs.
    #[must_use]
    pub fn compute_batch(&self, pairs: &[PreferencePair]) -> f64 {
        if pairs.is_empty() {
            return 0.0;
        }
        pairs.iter().map(|p| self.compute(p)).sum::<f64>() / pairs.len() as f64
    }

    /// Compute gradient of DPO loss w.r.t. policy log-probs.
    ///
    /// Returns (grad_chosen, grad_rejected) — gradients for the policy's
    /// log-probabilities of chosen and rejected responses.
    #[must_use]
    pub fn gradient(&self, pair: &PreferencePair) -> (f64, f64) {
        let log_ratio_chosen = if self.config.use_reference {
            pair.chosen_logprob - pair.ref_chosen_logprob
        } else {
            pair.chosen_logprob
        };

        let log_ratio_rejected = if self.config.use_reference {
            pair.rejected_logprob - pair.ref_rejected_logprob
        } else {
            pair.rejected_logprob
        };

        let logit = self.config.beta * (log_ratio_chosen - log_ratio_rejected);

        // σ(-logit) = 1 - σ(logit)
        let s = sigmoid(-logit);

        // dL/d(log_ratio_chosen) = -β * σ(-logit) = -β * (1 - σ(logit))
        // dL/d(log_ratio_rejected) = β * σ(-logit)
        let grad_chosen = -self.config.beta * s;
        let grad_rejected = self.config.beta * s;

        (grad_chosen, grad_rejected)
    }

    /// Compute implicit reward for a response.
    ///
    /// r(x, y) = β * (log π(y|x) - log π_ref(y|x))
    #[must_use]
    pub fn implicit_reward(&self, policy_logprob: f64, ref_logprob: f64) -> f64 {
        self.config.beta * (policy_logprob - ref_logprob)
    }

    /// Compute the accuracy of the policy's preference ranking.
    ///
    /// Returns the fraction of pairs where the policy assigns higher
    /// probability to the chosen response.
    #[must_use]
    pub fn accuracy(&self, pairs: &[PreferencePair]) -> f64 {
        if pairs.is_empty() {
            return 0.0;
        }
        let correct = pairs
            .iter()
            .filter(|p| {
                let chosen_ratio = p.chosen_logprob - p.ref_chosen_logprob;
                let rejected_ratio = p.rejected_logprob - p.ref_rejected_logprob;
                chosen_ratio > rejected_ratio
            })
            .count();
        correct as f64 / pairs.len() as f64
    }

    /// Get configuration.
    #[must_use]
    pub fn config(&self) -> &DpoConfig {
        &self.config
    }
}

/// Numerically stable log-sigmoid: log(σ(x)) = -log(1 + exp(-x))
fn log_sigmoid(x: f64) -> f64 {
    if x > 20.0 {
        // For large x: log(σ(x)) ≈ 0
        -(-x).exp()
    } else if x < -20.0 {
        // For very negative x: log(σ(x)) ≈ x
        x
    } else {
        -(1.0 + (-x).exp()).ln()
    }
}

/// Standard sigmoid function.
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// DPO training metrics.
#[derive(Debug, Clone, Default)]
pub struct DpoMetrics {
    /// Average loss over epoch
    pub avg_loss: f64,
    /// Preference accuracy (chosen > rejected)
    pub accuracy: f64,
    /// Average chosen reward
    pub avg_chosen_reward: f64,
    /// Average rejected reward
    pub avg_rejected_reward: f64,
    /// Reward margin (chosen - rejected)
    pub reward_margin: f64,
    /// Number of pairs processed
    pub num_pairs: usize,
}

impl DpoMetrics {
    /// Compute metrics from a batch of preference pairs.
    #[must_use]
    pub fn from_batch(loss: &DpoLoss, pairs: &[PreferencePair]) -> Self {
        if pairs.is_empty() {
            return Self::default();
        }

        let avg_loss = loss.compute_batch(pairs);
        let accuracy = loss.accuracy(pairs);

        let (total_chosen, total_rejected) = pairs.iter().fold((0.0, 0.0), |(tc, tr), p| {
            let rc = loss.implicit_reward(p.chosen_logprob, p.ref_chosen_logprob);
            let rr = loss.implicit_reward(p.rejected_logprob, p.ref_rejected_logprob);
            (tc + rc, tr + rr)
        });

        let n = pairs.len() as f64;
        let avg_chosen = total_chosen / n;
        let avg_rejected = total_rejected / n;

        Self {
            avg_loss,
            accuracy,
            avg_chosen_reward: avg_chosen,
            avg_rejected_reward: avg_rejected,
            reward_margin: avg_chosen - avg_rejected,
            num_pairs: pairs.len(),
        }
    }
}

#[cfg(test)]
#[path = "dpo_tests.rs"]
mod tests;

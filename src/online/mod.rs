//! Online Learning Infrastructure for Dynamic Model Retraining
//!
//! This module provides incremental model updates without full retraining,
//! supporting continuous improvement in production ML systems.
//!
//! # References
//!
//! - [Bottou 2010] "Large-Scale Machine Learning with Stochastic Gradient Descent"
//! - [Crammer et al. 2006] "Online Passive-Aggressive Algorithms"
//!
//! # Toyota Way Principles
//!
//! - **Kaizen**: Continuous model improvement via online learning
//! - **Jidoka**: Drift detection stops bad predictions automatically
//! - **Just-in-Time**: Retrain only when drift detected, not on schedule

pub mod corpus;
pub mod curriculum;
pub mod distillation;
pub mod drift;
pub mod orchestrator;

use crate::error::{AprenderError, Result};

/// Online learning capability for incremental model updates
///
/// Reference: [Bottou 2010] "Large-Scale Machine Learning with Stochastic
/// Gradient Descent" - Online learning converges to optimal solution with
/// O(1/t) regret bound under convex loss.
///
/// # Example
///
/// ```rust,ignore
/// use aprender::online::OnlineLearner;
///
/// let mut model = LogisticRegression::new();
/// for (x, y) in stream {
///     let loss = model.partial_fit(&x, &y, None)?;
///     println!("Sample loss: {}", loss);
/// }
/// ```
pub trait OnlineLearner {
    /// Update model with single sample (or mini-batch)
    ///
    /// # Arguments
    /// * `x` - Feature matrix (`n_samples` × `n_features`)
    /// * `y` - Target vector (`n_samples`)
    /// * `learning_rate` - Optional step size (uses adaptive if None)
    ///
    /// # Returns
    /// Loss on this sample before update (for monitoring)
    fn partial_fit(&mut self, x: &[f64], y: &[f64], learning_rate: Option<f64>) -> Result<f64>;

    /// Check if model supports warm-starting from checkpoint
    fn supports_warm_start(&self) -> bool {
        true
    }

    /// Get current effective learning rate
    fn current_learning_rate(&self) -> f64;

    /// Number of samples seen so far
    fn n_samples_seen(&self) -> u64;

    /// Reset internal state (for retraining from scratch)
    fn reset(&mut self);
}

/// Passive-Aggressive online learning for classification
///
/// Reference: [Crammer et al. 2006] "Online Passive-Aggressive Algorithms"
/// - Margin-based updates with bounded aggressiveness
/// - Suitable for non-stationary distributions
pub trait PassiveAggressive: OnlineLearner {
    /// Aggressiveness parameter C (higher = more aggressive updates)
    fn aggressiveness(&self) -> f64;

    /// Set aggressiveness for PA-I or PA-II variants
    fn set_aggressiveness(&mut self, c: f64);
}

/// Configuration for online learning
#[derive(Debug, Clone)]
pub struct OnlineLearnerConfig {
    /// Base learning rate
    pub learning_rate: f64,
    /// Learning rate decay schedule
    pub decay: LearningRateDecay,
    /// Regularization strength
    pub l2_reg: f64,
    /// Momentum coefficient (0.0 for pure SGD)
    pub momentum: f64,
    /// Clip gradients to this magnitude
    pub gradient_clip: Option<f64>,
}

impl Default for OnlineLearnerConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            decay: LearningRateDecay::InverseSqrt,
            l2_reg: 0.0,
            momentum: 0.0,
            gradient_clip: None,
        }
    }
}

/// Learning rate decay schedules
///
/// Reference: [Duchi et al. 2011] "Adaptive Subgradient Methods"
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum LearningRateDecay {
    /// No decay (constant learning rate)
    Constant,
    /// lr = `lr_0` / sqrt(t)
    #[default]
    InverseSqrt,
    /// lr = `lr_0` / t
    Inverse,
    /// lr = `lr_0` / (1 + `decay_rate` * t)
    Step { decay_rate: f64 },
    /// AdaGrad-style per-parameter adaptive decay
    AdaGrad { epsilon: f64 },
}

/// Simple online linear regression using SGD
///
/// Implements `OnlineLearner` for incremental least squares fitting.
#[derive(Debug, Clone)]
pub struct OnlineLinearRegression {
    /// Model weights
    weights: Vec<f64>,
    /// Bias term
    bias: f64,
    /// Accumulated squared gradients (for `AdaGrad`)
    accum_grad: Vec<f64>,
    /// Number of samples processed
    n_samples: u64,
    /// Configuration
    config: OnlineLearnerConfig,
}

impl OnlineLinearRegression {
    /// Create a new online linear regression model
    ///
    /// # Arguments
    /// * `n_features` - Number of input features
    #[must_use]
    pub fn new(n_features: usize) -> Self {
        Self {
            weights: vec![0.0; n_features],
            bias: 0.0,
            accum_grad: vec![1e-8; n_features], // Small value to avoid div by zero
            n_samples: 0,
            config: OnlineLearnerConfig::default(),
        }
    }

    /// Create with custom configuration
    #[must_use]
    pub fn with_config(n_features: usize, config: OnlineLearnerConfig) -> Self {
        Self {
            weights: vec![0.0; n_features],
            bias: 0.0,
            accum_grad: vec![1e-8; n_features],
            n_samples: 0,
            config,
        }
    }

    /// Get model weights
    #[must_use]
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Get bias term
    #[must_use]
    pub fn bias(&self) -> f64 {
        self.bias
    }

    /// Predict for a single sample
    pub fn predict_one(&self, x: &[f64]) -> Result<f64> {
        if x.len() != self.weights.len() {
            return Err(AprenderError::dimension_mismatch(
                "input features",
                self.weights.len(),
                x.len(),
            ));
        }

        let pred: f64 = x.iter().zip(&self.weights).map(|(xi, wi)| xi * wi).sum();
        Ok(pred + self.bias)
    }

    fn compute_lr(&self) -> f64 {
        let base = self.config.learning_rate;
        let t = self.n_samples.max(1) as f64;

        match self.config.decay {
            LearningRateDecay::InverseSqrt => base / t.sqrt(),
            LearningRateDecay::Inverse => base / t,
            LearningRateDecay::Step { decay_rate } => base / (1.0 + decay_rate * t),
            LearningRateDecay::Constant | LearningRateDecay::AdaGrad { .. } => base,
        }
    }
}

impl OnlineLearner for OnlineLinearRegression {
    fn partial_fit(&mut self, x: &[f64], y: &[f64], learning_rate: Option<f64>) -> Result<f64> {
        if x.is_empty() || y.is_empty() {
            return Err(AprenderError::empty_input("partial_fit input"));
        }

        let n_features = self.weights.len();
        if x.len() % n_features != 0 {
            return Err(AprenderError::dimension_mismatch(
                "input features",
                n_features,
                x.len() % n_features,
            ));
        }

        let n_samples = x.len() / n_features;
        if n_samples != y.len() {
            return Err(AprenderError::dimension_mismatch(
                "samples in y",
                n_samples,
                y.len(),
            ));
        }

        let lr = learning_rate.unwrap_or_else(|| self.compute_lr());
        let mut total_loss = 0.0;

        // Process each sample
        for i in 0..n_samples {
            let xi = &x[i * n_features..(i + 1) * n_features];
            let yi = y[i];

            // Predict before update
            let pred = self.predict_one(xi)?;
            let error = pred - yi;
            total_loss += error * error; // MSE

            // Gradient: 2 * error * x (we drop the 2 and absorb into lr)
            for (j, &xij) in xi.iter().enumerate() {
                let grad = error * xij + self.config.l2_reg * self.weights[j];

                // Clip gradient if configured
                let grad = if let Some(clip) = self.config.gradient_clip {
                    grad.clamp(-clip, clip)
                } else {
                    grad
                };

                // Update with optional AdaGrad
                let effective_lr = match self.config.decay {
                    LearningRateDecay::AdaGrad { epsilon } => {
                        self.accum_grad[j] += grad * grad;
                        lr / (self.accum_grad[j].sqrt() + epsilon)
                    }
                    _ => lr,
                };

                self.weights[j] -= effective_lr * grad;
            }

            // Update bias
            self.bias -= lr * error;
            self.n_samples += 1;
        }

        Ok(total_loss / n_samples as f64)
    }

    fn current_learning_rate(&self) -> f64 {
        self.compute_lr()
    }

    fn n_samples_seen(&self) -> u64 {
        self.n_samples
    }

    fn reset(&mut self) {
        self.weights.fill(0.0);
        self.bias = 0.0;
        self.accum_grad.fill(1e-8);
        self.n_samples = 0;
    }
}

/// Simple online logistic regression using SGD
///
/// Implements `OnlineLearner` for binary classification.
///
/// Reference: [Duchi et al. 2011] `AdaGrad` for adaptive learning rates
#[derive(Debug, Clone)]
pub struct OnlineLogisticRegression {
    /// Model weights
    weights: Vec<f64>,
    /// Bias term
    bias: f64,
    /// Accumulated squared gradients (for `AdaGrad`)
    accum_grad: Vec<f64>,
    /// Number of samples processed
    n_samples: u64,
    /// Configuration
    config: OnlineLearnerConfig,
}

impl OnlineLogisticRegression {
    /// Create a new online logistic regression model
    ///
    /// # Arguments
    /// * `n_features` - Number of input features
    #[must_use]
    pub fn new(n_features: usize) -> Self {
        Self {
            weights: vec![0.0; n_features],
            bias: 0.0,
            accum_grad: vec![1e-8; n_features],
            n_samples: 0,
            config: OnlineLearnerConfig::default(),
        }
    }

    /// Create with custom configuration
    #[must_use]
    pub fn with_config(n_features: usize, config: OnlineLearnerConfig) -> Self {
        Self {
            weights: vec![0.0; n_features],
            bias: 0.0,
            accum_grad: vec![1e-8; n_features],
            n_samples: 0,
            config,
        }
    }

    /// Get model weights
    #[must_use]
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Get bias term
    #[must_use]
    pub fn bias(&self) -> f64 {
        self.bias
    }

    /// Sigmoid function
    fn sigmoid(z: f64) -> f64 {
        1.0 / (1.0 + (-z).exp())
    }

    /// Predict probability for a single sample
    pub fn predict_proba_one(&self, x: &[f64]) -> Result<f64> {
        if x.len() != self.weights.len() {
            return Err(AprenderError::dimension_mismatch(
                "input features",
                self.weights.len(),
                x.len(),
            ));
        }

        let logit: f64 = x.iter().zip(&self.weights).map(|(xi, wi)| xi * wi).sum();
        Ok(Self::sigmoid(logit + self.bias))
    }

    fn compute_lr(&self) -> f64 {
        let base = self.config.learning_rate;
        let t = self.n_samples.max(1) as f64;

        match self.config.decay {
            LearningRateDecay::InverseSqrt => base / t.sqrt(),
            LearningRateDecay::Inverse => base / t,
            LearningRateDecay::Step { decay_rate } => base / (1.0 + decay_rate * t),
            LearningRateDecay::Constant | LearningRateDecay::AdaGrad { .. } => base,
        }
    }
}

impl OnlineLearner for OnlineLogisticRegression {
    fn partial_fit(&mut self, x: &[f64], y: &[f64], learning_rate: Option<f64>) -> Result<f64> {
        if x.is_empty() || y.is_empty() {
            return Err(AprenderError::empty_input("partial_fit input"));
        }

        let n_features = self.weights.len();
        if x.len() % n_features != 0 {
            return Err(AprenderError::dimension_mismatch(
                "input features",
                n_features,
                x.len() % n_features,
            ));
        }

        let n_samples = x.len() / n_features;
        if n_samples != y.len() {
            return Err(AprenderError::dimension_mismatch(
                "samples in y",
                n_samples,
                y.len(),
            ));
        }

        let lr = learning_rate.unwrap_or_else(|| self.compute_lr());
        let mut total_loss = 0.0;

        // Process each sample
        for i in 0..n_samples {
            let xi = &x[i * n_features..(i + 1) * n_features];
            let yi = y[i];

            // Predict before update
            let pred = self.predict_proba_one(xi)?;

            // Cross-entropy loss: -y*log(p) - (1-y)*log(1-p)
            let eps = 1e-15;
            let pred_clipped = pred.clamp(eps, 1.0 - eps);
            total_loss += -yi * pred_clipped.ln() - (1.0 - yi) * (1.0 - pred_clipped).ln();

            // Gradient: (pred - y) * x
            let error = pred - yi;

            for (j, &xij) in xi.iter().enumerate() {
                let grad = error * xij + self.config.l2_reg * self.weights[j];

                // Clip gradient if configured
                let grad = if let Some(clip) = self.config.gradient_clip {
                    grad.clamp(-clip, clip)
                } else {
                    grad
                };

                // Update with optional AdaGrad
                let effective_lr = match self.config.decay {
                    LearningRateDecay::AdaGrad { epsilon } => {
                        self.accum_grad[j] += grad * grad;
                        lr / (self.accum_grad[j].sqrt() + epsilon)
                    }
                    _ => lr,
                };

                self.weights[j] -= effective_lr * grad;
            }

            // Update bias
            self.bias -= lr * error;
            self.n_samples += 1;
        }

        Ok(total_loss / n_samples as f64)
    }

    fn current_learning_rate(&self) -> f64 {
        self.compute_lr()
    }

    fn n_samples_seen(&self) -> u64 {
        self.n_samples
    }

    fn reset(&mut self) {
        self.weights.fill(0.0);
        self.bias = 0.0;
        self.accum_grad.fill(1e-8);
        self.n_samples = 0;
    }
}

#[cfg(test)]
#[path = "online_tests.rs"]
mod tests;

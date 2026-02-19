//! Knowledge Distillation for Model Compression
//!
//! Transfer knowledge from large teacher models to small student models
//! using soft targets (probabilities) rather than hard labels.
//!
//! # References
//!
//! - [Hinton et al. 2015] "Distilling the Knowledge in a Neural Network"
//!
//! # Toyota Way Principles
//!
//! - **Muda Elimination**: Compress models to eliminate resource waste
//! - **Standardization**: Consistent soft-target training process

use crate::error::{AprenderError, Result};

/// Default distillation temperature (recommended by review)
///
/// Per Toyota Way review: "Start fixed (T=2.0-4.0). Hinton's original paper
/// suggests T=2.0-5.0 works well for a wide range of tasks. T=3.0 is a safe
/// starting point."
pub const DEFAULT_TEMPERATURE: f64 = 3.0;

/// Default alpha (weight for distillation loss vs hard label loss)
pub const DEFAULT_ALPHA: f64 = 0.7;

/// Configuration for knowledge distillation
#[derive(Debug, Clone)]
pub struct DistillationConfig {
    /// Temperature for softening probabilities
    /// Higher T = softer distribution = more "dark knowledge"
    pub temperature: f64,
    /// Weight for distillation loss vs hard label loss
    /// alpha * KL(student || teacher) + (1-alpha) * CE(student, labels)
    pub alpha: f64,
    /// Learning rate for student updates
    pub learning_rate: f64,
    /// L2 regularization strength
    pub l2_reg: f64,
}

impl Default for DistillationConfig {
    fn default() -> Self {
        Self {
            temperature: DEFAULT_TEMPERATURE,
            alpha: DEFAULT_ALPHA,
            learning_rate: 0.01,
            l2_reg: 0.0,
        }
    }
}

impl DistillationConfig {
    /// Create with custom temperature
    #[must_use]
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = temperature;
        self
    }

    /// Create with custom alpha
    #[must_use]
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }
}

/// Softmax with temperature scaling
///
/// ONE PATH: Scales then delegates to `nn::functional::softmax_1d_f64` (UCBD §4).
///
/// `softmax_T(z_i)` = `exp(z_i/T)` / `sum(exp(z_j/T))`
pub fn softmax_temperature(logits: &[f64], temperature: f64) -> Vec<f64> {
    if logits.is_empty() {
        return vec![];
    }
    let t = temperature.max(1e-10);
    let scaled: Vec<f64> = logits.iter().map(|&z| z / t).collect();
    crate::nn::functional::softmax_1d_f64(&scaled)
}

/// Regular softmax (T=1)
///
/// ONE PATH: Delegates to `nn::functional::softmax_1d_f64` (UCBD §4).
#[must_use]
pub fn softmax(logits: &[f64]) -> Vec<f64> {
    crate::nn::functional::softmax_1d_f64(logits)
}

/// KL divergence: `D_KL(P` || Q) = sum(P * log(P/Q))
///
/// Returns sum of KL divergence over all classes
#[must_use]
pub fn kl_divergence(p: &[f64], q: &[f64]) -> f64 {
    if p.len() != q.len() {
        return f64::INFINITY;
    }

    let eps = 1e-15;
    p.iter()
        .zip(q.iter())
        .map(|(&pi, &qi)| {
            let pi = pi.clamp(eps, 1.0 - eps);
            let qi = qi.clamp(eps, 1.0 - eps);
            pi * (pi / qi).ln()
        })
        .sum()
}

/// Cross-entropy loss: CE(p, y) = -sum(y * log(p))
#[must_use]
pub fn cross_entropy(probs: &[f64], targets: &[f64]) -> f64 {
    if probs.len() != targets.len() {
        return f64::INFINITY;
    }

    let eps = 1e-15;
    probs
        .iter()
        .zip(targets.iter())
        .map(|(&p, &y)| -y * p.clamp(eps, 1.0 - eps).ln())
        .sum()
}

/// Binary cross-entropy for single-class prediction
#[must_use]
pub fn binary_cross_entropy(prob: f64, target: f64) -> f64 {
    let eps = 1e-15;
    let p = prob.clamp(eps, 1.0 - eps);
    -target * p.ln() - (1.0 - target) * (1.0 - p).ln()
}

/// Soft target generator from logits
#[derive(Debug, Clone)]
pub struct SoftTargetGenerator {
    /// Temperature for softening
    temperature: f64,
}

impl Default for SoftTargetGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl SoftTargetGenerator {
    /// Create with default temperature
    #[must_use]
    pub fn new() -> Self {
        Self {
            temperature: DEFAULT_TEMPERATURE,
        }
    }

    /// Create with custom temperature
    #[must_use]
    pub fn with_temperature(temperature: f64) -> Self {
        Self { temperature }
    }

    /// Generate soft targets from logits
    #[must_use]
    pub fn generate(&self, logits: &[f64]) -> Vec<f64> {
        softmax_temperature(logits, self.temperature)
    }

    /// Generate soft targets for batch
    #[must_use]
    pub fn generate_batch(&self, logits: &[f64], n_classes: usize) -> Vec<f64> {
        if logits.is_empty() || n_classes == 0 || logits.len() % n_classes != 0 {
            return vec![];
        }

        let n_samples = logits.len() / n_classes;
        let mut result = Vec::with_capacity(logits.len());

        for i in 0..n_samples {
            let sample_logits = &logits[i * n_classes..(i + 1) * n_classes];
            result.extend(self.generate(sample_logits));
        }

        result
    }
}

/// Knowledge distillation loss calculator
#[derive(Debug, Clone)]
pub struct DistillationLoss {
    /// Configuration
    config: DistillationConfig,
}

impl Default for DistillationLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl DistillationLoss {
    /// Create with default config
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: DistillationConfig::default(),
        }
    }

    /// Create with custom config
    #[must_use]
    pub fn with_config(config: DistillationConfig) -> Self {
        Self { config }
    }

    /// Compute distillation loss
    ///
    /// Loss = α * T² * `KL(student_soft` || `teacher_soft`) + (1-α) * `CE(student_hard`, labels)
    ///
    /// The T² factor compensates for the gradient magnitude change when using temperature.
    ///
    /// # Arguments
    /// * `student_logits` - Raw logits from student model
    /// * `teacher_logits` - Raw logits from teacher model
    /// * `hard_labels` - One-hot encoded true labels
    ///
    /// # Returns
    /// Total loss value
    pub fn compute(
        &self,
        student_logits: &[f64],
        teacher_logits: &[f64],
        hard_labels: &[f64],
    ) -> Result<f64> {
        if student_logits.len() != teacher_logits.len() || student_logits.len() != hard_labels.len()
        {
            return Err(AprenderError::dimension_mismatch(
                "logits/labels",
                student_logits.len(),
                teacher_logits.len(),
            ));
        }

        let t = self.config.temperature;

        // Soft targets from teacher and student
        let teacher_soft = softmax_temperature(teacher_logits, t);
        let student_soft = softmax_temperature(student_logits, t);

        // Hard predictions from student (T=1)
        let student_hard = softmax(student_logits);

        // Distillation loss (KL divergence with T² scaling)
        let kl_loss = kl_divergence(&student_soft, &teacher_soft);
        let distill_loss = t * t * kl_loss;

        // Hard label loss
        let hard_loss = cross_entropy(&student_hard, hard_labels);

        // Combined loss
        let total = self.config.alpha * distill_loss + (1.0 - self.config.alpha) * hard_loss;

        Ok(total)
    }

    /// Compute gradient of distillation loss w.r.t. student logits
    ///
    /// # Returns
    /// Gradient vector same size as `student_logits`
    pub fn gradient(
        &self,
        student_logits: &[f64],
        teacher_logits: &[f64],
        hard_labels: &[f64],
    ) -> Result<Vec<f64>> {
        if student_logits.len() != teacher_logits.len() || student_logits.len() != hard_labels.len()
        {
            return Err(AprenderError::dimension_mismatch(
                "logits/labels",
                student_logits.len(),
                teacher_logits.len(),
            ));
        }

        let t = self.config.temperature;

        // Soft distributions
        let teacher_soft = softmax_temperature(teacher_logits, t);
        let student_soft = softmax_temperature(student_logits, t);
        let student_hard = softmax(student_logits);

        // Gradient of distill loss: T * (student_soft - teacher_soft)
        // Gradient of hard loss: (student_hard - hard_labels)
        let grad: Vec<f64> = student_soft
            .iter()
            .zip(teacher_soft.iter())
            .zip(student_hard.iter())
            .zip(hard_labels.iter())
            .map(|(((&ss, &ts), &sh), &hl)| {
                let distill_grad = t * (ss - ts);
                let hard_grad = sh - hl;
                self.config.alpha * distill_grad + (1.0 - self.config.alpha) * hard_grad
            })
            .collect();

        Ok(grad)
    }

    /// Get configuration
    #[must_use]
    pub fn config(&self) -> &DistillationConfig {
        &self.config
    }
}

/// Simple linear distillation model (for testing/simple cases)
#[derive(Debug, Clone)]
pub struct LinearDistiller {
    /// Student weights (`n_classes` × `n_features`)
    weights: Vec<f64>,
    /// Student biases (`n_classes`)
    biases: Vec<f64>,
    /// Number of features
    n_features: usize,
    /// Number of classes (stored for validation)
    #[allow(dead_code)]
    n_classes: usize,
    /// Loss calculator
    loss: DistillationLoss,
}

impl LinearDistiller {
    /// Create a new linear distiller
    ///
    /// # Arguments
    /// * `n_features` - Number of input features
    /// * `n_classes` - Number of output classes
    #[must_use]
    pub fn new(n_features: usize, n_classes: usize) -> Self {
        Self {
            weights: vec![0.0; n_classes * n_features],
            biases: vec![0.0; n_classes],
            n_features,
            n_classes,
            loss: DistillationLoss::new(),
        }
    }

    /// Create with custom config
    #[must_use]
    pub fn with_config(n_features: usize, n_classes: usize, config: DistillationConfig) -> Self {
        Self {
            weights: vec![0.0; n_classes * n_features],
            biases: vec![0.0; n_classes],
            n_features,
            n_classes,
            loss: DistillationLoss::with_config(config),
        }
    }

    /// Compute student logits for a single sample
    pub fn forward(&self, features: &[f64]) -> Result<Vec<f64>> {
        if features.len() != self.n_features {
            return Err(AprenderError::dimension_mismatch(
                "features",
                self.n_features,
                features.len(),
            ));
        }

        let mut logits = self.biases.clone();
        for (c, logit) in logits.iter_mut().enumerate() {
            for (f, &feat) in features.iter().enumerate() {
                *logit += self.weights[c * self.n_features + f] * feat;
            }
        }

        Ok(logits)
    }

    /// Train on a single sample using teacher's soft targets
    ///
    /// # Arguments
    /// * `features` - Input features
    /// * `teacher_logits` - Teacher's output logits
    /// * `hard_labels` - True one-hot labels
    ///
    /// # Returns
    /// Loss value before update
    pub fn train_step(
        &mut self,
        features: &[f64],
        teacher_logits: &[f64],
        hard_labels: &[f64],
    ) -> Result<f64> {
        // Forward pass
        let student_logits = self.forward(features)?;

        // Compute loss
        let loss_val = self
            .loss
            .compute(&student_logits, teacher_logits, hard_labels)?;

        // Compute gradient
        let grad = self
            .loss
            .gradient(&student_logits, teacher_logits, hard_labels)?;

        // Update weights (SGD)
        let lr = self.loss.config().learning_rate;
        let l2 = self.loss.config().l2_reg;

        for (c, (&g, bias)) in grad.iter().zip(self.biases.iter_mut()).enumerate() {
            for (f, &feat) in features.iter().enumerate() {
                let idx = c * self.n_features + f;
                let weight_grad = g * feat + l2 * self.weights[idx];
                self.weights[idx] -= lr * weight_grad;
            }
            *bias -= lr * g;
        }

        Ok(loss_val)
    }

    /// Get weights
    #[must_use]
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Get biases
    #[must_use]
    pub fn biases(&self) -> &[f64] {
        &self.biases
    }

    /// Predict class probabilities
    pub fn predict_proba(&self, features: &[f64]) -> Result<Vec<f64>> {
        let logits = self.forward(features)?;
        Ok(softmax(&logits))
    }

    /// Predict class label
    pub fn predict(&self, features: &[f64]) -> Result<usize> {
        let probs = self.predict_proba(features)?;
        Ok(probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(i, _)| i))
    }
}

/// Distillation training result
#[derive(Debug, Clone)]
pub struct DistillationResult {
    /// Final loss
    pub final_loss: f64,
    /// Number of samples trained
    pub n_samples: usize,
    /// Loss history (per epoch/batch if tracked)
    pub loss_history: Vec<f64>,
    /// Student accuracy on training data
    pub train_accuracy: Option<f64>,
}

#[cfg(test)]
#[path = "distillation_tests.rs"]
mod tests;

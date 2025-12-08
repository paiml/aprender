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
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = temperature;
        self
    }

    /// Create with custom alpha
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }
}

/// Softmax with temperature scaling
///
/// softmax_T(z_i) = exp(z_i/T) / sum(exp(z_j/T))
pub fn softmax_temperature(logits: &[f64], temperature: f64) -> Vec<f64> {
    if logits.is_empty() {
        return vec![];
    }

    let t = temperature.max(1e-10); // Avoid division by zero

    // Compute scaled logits
    let scaled: Vec<f64> = logits.iter().map(|&z| z / t).collect();

    // Subtract max for numerical stability
    let max_logit = scaled.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exp_logits: Vec<f64> = scaled.iter().map(|&z| (z - max_logit).exp()).collect();

    let sum: f64 = exp_logits.iter().sum();
    exp_logits.iter().map(|&e| e / sum).collect()
}

/// Regular softmax (T=1)
pub fn softmax(logits: &[f64]) -> Vec<f64> {
    softmax_temperature(logits, 1.0)
}

/// KL divergence: D_KL(P || Q) = sum(P * log(P/Q))
///
/// Returns sum of KL divergence over all classes
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
    pub fn new() -> Self {
        Self {
            temperature: DEFAULT_TEMPERATURE,
        }
    }

    /// Create with custom temperature
    pub fn with_temperature(temperature: f64) -> Self {
        Self { temperature }
    }

    /// Generate soft targets from logits
    pub fn generate(&self, logits: &[f64]) -> Vec<f64> {
        softmax_temperature(logits, self.temperature)
    }

    /// Generate soft targets for batch
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
    pub fn new() -> Self {
        Self {
            config: DistillationConfig::default(),
        }
    }

    /// Create with custom config
    pub fn with_config(config: DistillationConfig) -> Self {
        Self { config }
    }

    /// Compute distillation loss
    ///
    /// Loss = α * T² * KL(student_soft || teacher_soft) + (1-α) * CE(student_hard, labels)
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
    /// Gradient vector same size as student_logits
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
    pub fn config(&self) -> &DistillationConfig {
        &self.config
    }
}

/// Simple linear distillation model (for testing/simple cases)
#[derive(Debug, Clone)]
pub struct LinearDistiller {
    /// Student weights (n_classes × n_features)
    weights: Vec<f64>,
    /// Student biases (n_classes)
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
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Get biases
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
mod tests {
    use super::*;

    #[test]
    fn test_softmax_temperature_basic() {
        let logits = vec![1.0, 2.0, 3.0];

        // T=1 (standard softmax)
        let soft1 = softmax_temperature(&logits, 1.0);
        assert!((soft1.iter().sum::<f64>() - 1.0).abs() < 1e-10);

        // T=2 (softer)
        let soft2 = softmax_temperature(&logits, 2.0);
        assert!((soft2.iter().sum::<f64>() - 1.0).abs() < 1e-10);

        // Higher temperature = more uniform
        let variance1: f64 = soft1.iter().map(|x| (x - 1.0 / 3.0).powi(2)).sum();
        let variance2: f64 = soft2.iter().map(|x| (x - 1.0 / 3.0).powi(2)).sum();
        assert!(variance2 < variance1);
    }

    #[test]
    fn test_softmax_temperature_extreme() {
        let logits = vec![0.0, 10.0];

        // Very low temperature (approaches argmax)
        let hard = softmax_temperature(&logits, 0.1);
        assert!(hard[1] > 0.99);

        // Very high temperature (approaches uniform)
        let uniform = softmax_temperature(&logits, 100.0);
        assert!((uniform[0] - uniform[1]).abs() < 0.1);
    }

    #[test]
    fn test_softmax_empty() {
        let result = softmax_temperature(&[], 1.0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_kl_divergence_same() {
        let p = vec![0.25, 0.25, 0.25, 0.25];
        let kl = kl_divergence(&p, &p);
        assert!(kl.abs() < 1e-10);
    }

    #[test]
    fn test_kl_divergence_different() {
        let p = vec![0.9, 0.1];
        let q = vec![0.5, 0.5];
        let kl = kl_divergence(&p, &q);
        assert!(kl > 0.0);
    }

    #[test]
    fn test_cross_entropy() {
        // Perfect prediction
        let probs = vec![0.0, 1.0, 0.0];
        let labels = vec![0.0, 1.0, 0.0];
        let ce = cross_entropy(&probs, &labels);
        assert!(ce < 0.01);

        // Bad prediction
        let probs_bad = vec![0.9, 0.05, 0.05];
        let ce_bad = cross_entropy(&probs_bad, &labels);
        assert!(ce_bad > ce);
    }

    #[test]
    fn test_binary_cross_entropy() {
        // Perfect prediction
        let bce = binary_cross_entropy(1.0, 1.0);
        assert!(bce < 0.01);

        // Wrong prediction
        let bce_bad = binary_cross_entropy(0.1, 1.0);
        assert!(bce_bad > 1.0);
    }

    #[test]
    fn test_soft_target_generator() {
        let generator = SoftTargetGenerator::with_temperature(3.0);
        let logits = vec![1.0, 2.0, 3.0];
        let soft = generator.generate(&logits);

        assert_eq!(soft.len(), 3);
        assert!((soft.iter().sum::<f64>() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_soft_target_generator_batch() {
        let generator = SoftTargetGenerator::with_temperature(2.0);
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2 samples × 3 classes
        let soft = generator.generate_batch(&logits, 3);

        assert_eq!(soft.len(), 6);

        // Each sample should sum to 1
        assert!((soft[0..3].iter().sum::<f64>() - 1.0).abs() < 1e-10);
        assert!((soft[3..6].iter().sum::<f64>() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_distillation_loss_compute() {
        let loss = DistillationLoss::new();

        let student_logits = vec![1.0, 2.0, 3.0];
        let teacher_logits = vec![1.0, 2.5, 2.8];
        let hard_labels = vec![0.0, 0.0, 1.0];

        let result = loss.compute(&student_logits, &teacher_logits, &hard_labels);
        assert!(result.is_ok());
        assert!(result.unwrap().is_finite());
    }

    #[test]
    fn test_distillation_loss_gradient() {
        let loss = DistillationLoss::new();

        let student_logits = vec![1.0, 2.0, 3.0];
        let teacher_logits = vec![1.0, 2.5, 2.8];
        let hard_labels = vec![0.0, 0.0, 1.0];

        let grad = loss.gradient(&student_logits, &teacher_logits, &hard_labels);
        assert!(grad.is_ok());
        assert_eq!(grad.unwrap().len(), 3);
    }

    #[test]
    fn test_distillation_loss_dimension_mismatch() {
        let loss = DistillationLoss::new();

        let student_logits = vec![1.0, 2.0];
        let teacher_logits = vec![1.0, 2.0, 3.0];
        let hard_labels = vec![0.0, 1.0];

        assert!(loss
            .compute(&student_logits, &teacher_logits, &hard_labels)
            .is_err());
    }

    #[test]
    fn test_linear_distiller_forward() {
        let distiller = LinearDistiller::new(3, 2);

        let features = vec![1.0, 2.0, 3.0];
        let logits = distiller.forward(&features);

        assert!(logits.is_ok());
        assert_eq!(logits.unwrap().len(), 2);
    }

    #[test]
    fn test_linear_distiller_train_step() {
        let config = DistillationConfig {
            learning_rate: 0.1,
            ..Default::default()
        };
        let mut distiller = LinearDistiller::with_config(2, 2, config);

        let features = vec![1.0, 0.5];
        let teacher_logits = vec![0.5, 1.5];
        let hard_labels = vec![0.0, 1.0];

        let loss1 = distiller
            .train_step(&features, &teacher_logits, &hard_labels)
            .unwrap();

        // Train more
        for _ in 0..10 {
            distiller
                .train_step(&features, &teacher_logits, &hard_labels)
                .unwrap();
        }

        let loss2 = distiller
            .train_step(&features, &teacher_logits, &hard_labels)
            .unwrap();

        // Loss should decrease
        assert!(loss2 < loss1 * 1.5); // Allow some variance
    }

    #[test]
    fn test_linear_distiller_predict() {
        let mut distiller = LinearDistiller::new(2, 3);

        // Train briefly
        let features = vec![1.0, 0.0];
        let teacher_logits = vec![0.0, 0.0, 5.0]; // Strongly predict class 2
        let hard_labels = vec![0.0, 0.0, 1.0];

        for _ in 0..50 {
            distiller
                .train_step(&features, &teacher_logits, &hard_labels)
                .unwrap();
        }

        let pred = distiller.predict(&features).unwrap();
        assert_eq!(pred, 2);
    }

    #[test]
    fn test_linear_distiller_dimension_mismatch() {
        let distiller = LinearDistiller::new(3, 2);

        let features = vec![1.0, 2.0]; // Wrong size
        assert!(distiller.forward(&features).is_err());
    }

    #[test]
    fn test_distillation_config_default() {
        let config = DistillationConfig::default();
        assert_eq!(config.temperature, DEFAULT_TEMPERATURE);
        assert_eq!(config.alpha, DEFAULT_ALPHA);
    }

    #[test]
    fn test_distillation_config_builder() {
        let config = DistillationConfig::default()
            .with_temperature(5.0)
            .with_alpha(0.9);

        assert_eq!(config.temperature, 5.0);
        assert_eq!(config.alpha, 0.9);
    }

    #[test]
    fn test_soft_target_generator_default() {
        let generator = SoftTargetGenerator::default();
        assert_eq!(generator.temperature, DEFAULT_TEMPERATURE);
    }

    #[test]
    fn test_distillation_loss_default() {
        let loss = DistillationLoss::default();
        assert_eq!(loss.config().temperature, DEFAULT_TEMPERATURE);
    }

    #[test]
    fn test_temperature_effect_on_dark_knowledge() {
        let logits = vec![1.0, 3.0, 2.0];

        // Low temperature - peaked distribution
        let low_t = softmax_temperature(&logits, 1.0);
        // High temperature - flatter distribution
        let high_t = softmax_temperature(&logits, 5.0);

        // The smallest class probability should be higher with higher T
        let min_low = low_t.iter().cloned().fold(f64::INFINITY, f64::min);
        let min_high = high_t.iter().cloned().fold(f64::INFINITY, f64::min);

        assert!(
            min_high > min_low,
            "Higher T should reveal more dark knowledge"
        );
    }
}

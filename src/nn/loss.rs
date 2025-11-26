//! Differentiable loss functions for neural network training.
//!
//! These loss functions work with autograd Tensors and support
//! backpropagation for gradient-based optimization.
//!
//! # Example
//!
//! ```ignore
//! use aprender::nn::loss::{MSELoss, CrossEntropyLoss};
//! use aprender::autograd::Tensor;
//!
//! // Regression loss
//! let criterion = MSELoss::new();
//! let pred = Tensor::from_slice(&[1.0, 2.0, 3.0]).requires_grad();
//! let target = Tensor::from_slice(&[1.1, 2.0, 2.9]);
//! let loss = criterion.forward(&pred, &target);
//! loss.backward();
//!
//! // Classification loss
//! let criterion = CrossEntropyLoss::new();
//! let logits = Tensor::new(&[1.0, 2.0, 0.5, 0.1, 3.0, 0.2], &[2, 3]).requires_grad();
//! let targets = Tensor::from_slice(&[1.0, 2.0]);  // class indices
//! let loss = criterion.forward(&logits, &targets);
//! ```
//!
//! # References
//!
//! - Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer.

use crate::autograd::grad_fn::CrossEntropyBackward;
use crate::autograd::{is_grad_enabled, with_graph, Tensor};
use std::sync::Arc;

/// Reduction mode for loss functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Reduction {
    /// Return loss per element (no reduction)
    None,
    /// Return mean of losses (default)
    #[default]
    Mean,
    /// Return sum of losses
    Sum,
}

/// Mean Squared Error loss for regression.
///
/// Computes: MSE = mean((pred - target)²)
///
/// Gradient: ∂MSE/∂pred = 2 * (pred - target) / n
///
/// # Example
///
/// ```ignore
/// let criterion = MSELoss::new();
/// let pred = Tensor::from_slice(&[1.0, 2.0, 3.0]).requires_grad();
/// let target = Tensor::from_slice(&[1.0, 2.0, 3.0]);
/// let loss = criterion.forward(&pred, &target);
/// assert!(loss.item() < 1e-6);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct MSELoss {
    reduction: Reduction,
}

impl MSELoss {
    /// Create a new MSELoss with default reduction (mean).
    pub fn new() -> Self {
        Self::default()
    }

    /// Create MSELoss with specified reduction.
    pub fn with_reduction(reduction: Reduction) -> Self {
        Self { reduction }
    }

    /// Compute MSE loss between predictions and targets.
    pub fn forward(&self, pred: &Tensor, target: &Tensor) -> Tensor {
        assert_eq!(
            pred.shape(),
            target.shape(),
            "Prediction and target shapes must match"
        );

        // Compute (pred - target)²
        let diff = pred.sub(target);
        let squared = diff.pow(2.0);

        match self.reduction {
            Reduction::None => squared,
            Reduction::Mean => squared.mean(),
            Reduction::Sum => squared.sum(),
        }
    }
}

/// Mean Absolute Error loss for regression.
///
/// Computes: MAE = mean(|pred - target|)
///
/// More robust to outliers than MSE.
#[derive(Debug, Clone, Copy, Default)]
pub struct L1Loss {
    reduction: Reduction,
}

impl L1Loss {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_reduction(reduction: Reduction) -> Self {
        Self { reduction }
    }

    /// Compute L1 loss between predictions and targets.
    pub fn forward(&self, pred: &Tensor, target: &Tensor) -> Tensor {
        assert_eq!(pred.shape(), target.shape());

        // Compute |pred - target|
        let diff = pred.sub(target);
        let abs_diff = abs(&diff);

        match self.reduction {
            Reduction::None => abs_diff,
            Reduction::Mean => abs_diff.mean(),
            Reduction::Sum => abs_diff.sum(),
        }
    }
}

/// Smooth L1 Loss (Huber Loss).
///
/// Combines MSE and L1: uses squared error for small values,
/// linear error for large values.
///
/// ```text
/// loss = 0.5 * x² / beta,  if |x| < beta
///      = |x| - 0.5 * beta, otherwise
/// ```
#[derive(Debug, Clone, Copy)]
pub struct SmoothL1Loss {
    beta: f32,
    reduction: Reduction,
}

impl SmoothL1Loss {
    pub fn new() -> Self {
        Self {
            beta: 1.0,
            reduction: Reduction::Mean,
        }
    }

    pub fn with_beta(beta: f32) -> Self {
        Self {
            beta,
            reduction: Reduction::Mean,
        }
    }

    pub fn forward(&self, pred: &Tensor, target: &Tensor) -> Tensor {
        assert_eq!(pred.shape(), target.shape());

        let diff = pred.sub(target);
        let loss_data: Vec<f32> = diff
            .data()
            .iter()
            .map(|&x| {
                let abs_x = x.abs();
                if abs_x < self.beta {
                    0.5 * x * x / self.beta
                } else {
                    abs_x - 0.5 * self.beta
                }
            })
            .collect();

        let loss = Tensor::new(&loss_data, pred.shape());

        match self.reduction {
            Reduction::None => loss,
            Reduction::Mean => loss.mean(),
            Reduction::Sum => loss.sum(),
        }
    }
}

impl Default for SmoothL1Loss {
    fn default() -> Self {
        Self::new()
    }
}

/// Cross-Entropy Loss for classification.
///
/// Combines log_softmax and negative log likelihood for numerical stability.
///
/// For multi-class classification:
/// ```text
/// loss = -log(softmax(logits)[target_class])
/// ```
///
/// # Arguments
///
/// * `logits` - Raw model outputs, shape [batch, num_classes]
/// * `targets` - Target class indices, shape [batch]
#[derive(Debug, Clone, Default)]
pub struct CrossEntropyLoss {
    reduction: Reduction,
    label_smoothing: f32,
}

impl CrossEntropyLoss {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_reduction(reduction: Reduction) -> Self {
        Self {
            reduction,
            label_smoothing: 0.0,
        }
    }

    pub fn with_label_smoothing(label_smoothing: f32) -> Self {
        assert!(
            (0.0..1.0).contains(&label_smoothing),
            "Label smoothing must be in [0, 1)"
        );
        Self {
            reduction: Reduction::Mean,
            label_smoothing,
        }
    }

    /// Compute cross-entropy loss.
    ///
    /// # Arguments
    ///
    /// * `logits` - Shape [batch, num_classes]
    /// * `targets` - Shape [batch], integer class indices (as f32)
    pub fn forward(&self, logits: &Tensor, targets: &Tensor) -> Tensor {
        assert_eq!(logits.ndim(), 2, "Logits must be 2D [batch, classes]");
        assert_eq!(targets.ndim(), 1, "Targets must be 1D [batch]");
        assert_eq!(
            logits.shape()[0],
            targets.shape()[0],
            "Batch sizes must match"
        );

        let batch_size = logits.shape()[0];
        let num_classes = logits.shape()[1];

        // Compute softmax for gradient computation
        let softmax_output = softmax_2d(logits);

        // Compute log_softmax for numerical stability in loss calculation
        let log_probs = log_softmax(logits);

        // Extract target indices for gradient computation
        let target_indices: Vec<usize> = targets
            .data()
            .iter()
            .map(|&t| {
                let idx = t as usize;
                assert!(
                    idx < num_classes,
                    "Target class {idx} out of bounds for {num_classes} classes"
                );
                idx
            })
            .collect();

        // Compute negative log likelihood
        let mut losses = Vec::with_capacity(batch_size);

        for (b, &target_class) in target_indices.iter().enumerate() {
            if self.label_smoothing > 0.0 {
                // Label smoothing: distribute some probability mass to other classes
                let smooth_target = (1.0 - self.label_smoothing) / num_classes as f32;
                let mut loss = 0.0;
                for c in 0..num_classes {
                    let target_prob = if c == target_class {
                        1.0 - self.label_smoothing + smooth_target
                    } else {
                        smooth_target
                    };
                    loss -= target_prob * log_probs.data()[b * num_classes + c];
                }
                losses.push(loss);
            } else {
                // Standard cross-entropy: -log(p[target])
                losses.push(-log_probs.data()[b * num_classes + target_class]);
            }
        }

        let per_sample_loss = Tensor::new(&losses, &[batch_size]);

        // Apply reduction
        let mut loss = match self.reduction {
            Reduction::None => per_sample_loss,
            Reduction::Mean => {
                let mean_val = losses.iter().sum::<f32>() / batch_size as f32;
                Tensor::from_slice(&[mean_val])
            }
            Reduction::Sum => {
                let sum_val = losses.iter().sum::<f32>();
                Tensor::from_slice(&[sum_val])
            }
        };

        // Record autograd if enabled
        if is_grad_enabled() && logits.requires_grad_enabled() && self.label_smoothing == 0.0 {
            loss.requires_grad_(true);
            let grad_fn = Arc::new(CrossEntropyBackward {
                softmax_output: softmax_output.clone(),
                targets: target_indices,
            });
            loss.set_grad_fn(grad_fn.clone());

            with_graph(|graph| {
                graph.register_tensor(logits.clone());
                graph.record(loss.id(), grad_fn, vec![logits.id()]);
            });
        }

        loss
    }
}

/// Binary Cross-Entropy with Logits loss.
///
/// Combines sigmoid and binary cross-entropy for numerical stability.
///
/// ```text
/// loss = -[y * log(σ(x)) + (1-y) * log(1-σ(x))]
///      = max(x, 0) - x*y + log(1 + exp(-|x|))
/// ```
#[derive(Debug, Clone, Default)]
pub struct BCEWithLogitsLoss {
    reduction: Reduction,
    pos_weight: Option<f32>,
}

impl BCEWithLogitsLoss {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_reduction(reduction: Reduction) -> Self {
        Self {
            reduction,
            pos_weight: None,
        }
    }

    /// Set positive class weight for imbalanced datasets.
    pub fn with_pos_weight(pos_weight: f32) -> Self {
        Self {
            reduction: Reduction::Mean,
            pos_weight: Some(pos_weight),
        }
    }

    /// Compute BCE loss from logits.
    ///
    /// # Arguments
    ///
    /// * `logits` - Raw model outputs (before sigmoid)
    /// * `targets` - Binary targets (0 or 1)
    pub fn forward(&self, logits: &Tensor, targets: &Tensor) -> Tensor {
        assert_eq!(logits.shape(), targets.shape());

        // Numerically stable computation:
        // loss = max(x, 0) - x*y + log(1 + exp(-|x|))
        let loss_data: Vec<f32> = logits
            .data()
            .iter()
            .zip(targets.data().iter())
            .map(|(&x, &y)| {
                let max_val = x.max(0.0);
                let base_loss = max_val - x * y + (1.0 + (-x.abs()).exp()).ln();

                // Apply positive weight if specified
                match self.pos_weight {
                    Some(w) => {
                        // weight positive samples more
                        let weight = y * (w - 1.0) + 1.0;
                        base_loss * weight
                    }
                    None => base_loss,
                }
            })
            .collect();

        let loss = Tensor::new(&loss_data, logits.shape());

        match self.reduction {
            Reduction::None => loss,
            Reduction::Mean => loss.mean(),
            Reduction::Sum => loss.sum(),
        }
    }
}

/// Negative Log Likelihood loss.
///
/// Expects log-probabilities as input (use after log_softmax).
#[derive(Debug, Clone, Copy, Default)]
pub struct NLLLoss {
    reduction: Reduction,
}

impl NLLLoss {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn forward(&self, log_probs: &Tensor, targets: &Tensor) -> Tensor {
        assert_eq!(log_probs.ndim(), 2);
        assert_eq!(targets.ndim(), 1);

        let batch_size = log_probs.shape()[0];
        let num_classes = log_probs.shape()[1];

        let mut losses = Vec::with_capacity(batch_size);
        for b in 0..batch_size {
            let target = targets.data()[b] as usize;
            losses.push(-log_probs.data()[b * num_classes + target]);
        }

        let loss = Tensor::new(&losses, &[batch_size]);

        match self.reduction {
            Reduction::None => loss,
            Reduction::Mean => loss.mean(),
            Reduction::Sum => loss.sum(),
        }
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Element-wise absolute value.
fn abs(x: &Tensor) -> Tensor {
    let data: Vec<f32> = x.data().iter().map(|&v| v.abs()).collect();
    Tensor::new(&data, x.shape())
}

/// Softmax computation for gradient tracking.
fn softmax_2d(x: &Tensor) -> Tensor {
    assert_eq!(x.ndim(), 2);

    let (batch, features) = (x.shape()[0], x.shape()[1]);
    let mut output = vec![0.0; batch * features];

    for b in 0..batch {
        let row_start = b * features;

        // Find max for numerical stability
        let max_val = x.data()[row_start..row_start + features]
            .iter()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        // Compute exp(x - max)
        let mut sum = 0.0;
        for j in 0..features {
            let exp_val = (x.data()[row_start + j] - max_val).exp();
            output[row_start + j] = exp_val;
            sum += exp_val;
        }

        // Normalize
        for j in 0..features {
            output[row_start + j] /= sum;
        }
    }

    Tensor::new(&output, x.shape())
}

/// Log-softmax for numerical stability.
fn log_softmax(x: &Tensor) -> Tensor {
    assert_eq!(x.ndim(), 2);

    let (batch, features) = (x.shape()[0], x.shape()[1]);
    let mut output = vec![0.0; batch * features];

    for b in 0..batch {
        let row_start = b * features;

        // Find max for numerical stability
        let max_val = x.data()[row_start..row_start + features]
            .iter()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        // Compute log(sum(exp(x - max)))
        let log_sum_exp: f32 = x.data()[row_start..row_start + features]
            .iter()
            .map(|&v| (v - max_val).exp())
            .sum::<f32>()
            .ln();

        // log_softmax = x - max - log_sum_exp
        for j in 0..features {
            output[row_start + j] = x.data()[row_start + j] - max_val - log_sum_exp;
        }
    }

    Tensor::new(&output, x.shape())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::clear_graph;

    #[test]
    fn test_mse_loss_zero() {
        let pred = Tensor::from_slice(&[1.0, 2.0, 3.0]);
        let target = Tensor::from_slice(&[1.0, 2.0, 3.0]);

        let criterion = MSELoss::new();
        let loss = criterion.forward(&pred, &target);

        assert!(loss.item() < 1e-6);
    }

    #[test]
    fn test_mse_loss_nonzero() {
        let pred = Tensor::from_slice(&[1.0, 2.0, 3.0]);
        let target = Tensor::from_slice(&[2.0, 2.0, 2.0]);

        let criterion = MSELoss::new();
        let loss = criterion.forward(&pred, &target);

        // MSE = ((1-2)² + (2-2)² + (3-2)²) / 3 = (1 + 0 + 1) / 3 = 2/3
        assert!((loss.item() - 2.0 / 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_mse_loss_gradient() {
        clear_graph();

        let pred = Tensor::from_slice(&[1.0, 2.0, 3.0]).requires_grad();
        let pred_id = pred.id();
        let target = Tensor::from_slice(&[2.0, 2.0, 2.0]);

        let criterion = MSELoss::new();
        let loss = criterion.forward(&pred, &target);
        loss.backward();

        let grad = crate::autograd::get_grad(pred_id).expect("Should have gradient");

        // Gradient of MSE: 2 * (pred - target) / n
        // = 2 * [-1, 0, 1] / 3 = [-2/3, 0, 2/3]
        let expected = [-2.0 / 3.0, 0.0, 2.0 / 3.0];
        for (g, e) in grad.data().iter().zip(expected.iter()) {
            assert!((g - e).abs() < 1e-5, "Expected {e}, got {g}");
        }
    }

    #[test]
    fn test_l1_loss() {
        let pred = Tensor::from_slice(&[1.0, 2.0, 3.0]);
        let target = Tensor::from_slice(&[2.0, 2.0, 2.0]);

        let criterion = L1Loss::new();
        let loss = criterion.forward(&pred, &target);

        // MAE = (|1-2| + |2-2| + |3-2|) / 3 = (1 + 0 + 1) / 3 = 2/3
        assert!((loss.item() - 2.0 / 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_smooth_l1_loss() {
        let pred = Tensor::from_slice(&[0.0]);
        let target = Tensor::from_slice(&[0.5]);

        let criterion = SmoothL1Loss::new();
        let loss = criterion.forward(&pred, &target);

        // |x| = 0.5 < 1.0 (beta), so loss = 0.5 * 0.5² / 1.0 = 0.125
        assert!((loss.item() - 0.125).abs() < 1e-5);
    }

    #[test]
    fn test_cross_entropy_loss() {
        // Two samples, 3 classes
        // Sample 0: logits [1, 2, 0.5], target 1 (class with logit 2)
        // Sample 1: logits [0.1, 3, 0.2], target 1 (class with logit 3)
        let logits = Tensor::new(&[1.0, 2.0, 0.5, 0.1, 3.0, 0.2], &[2, 3]);
        let targets = Tensor::from_slice(&[1.0, 1.0]);

        let criterion = CrossEntropyLoss::new();
        let loss = criterion.forward(&logits, &targets);

        // Loss should be relatively small since we're predicting the correct class
        // (the class with highest logit matches target)
        assert!(loss.item() < 1.0);
    }

    #[test]
    fn test_cross_entropy_wrong_prediction() {
        // Logits favor class 0, but target is class 2
        let logits = Tensor::new(&[10.0, 0.0, 0.0], &[1, 3]);
        let targets = Tensor::from_slice(&[2.0]);

        let criterion = CrossEntropyLoss::new();
        let loss = criterion.forward(&logits, &targets);

        // Loss should be high since prediction is wrong
        assert!(loss.item() > 5.0);
    }

    #[test]
    fn test_bce_with_logits() {
        // Perfect prediction: logit = 10 (high sigmoid) for target = 1
        let logits = Tensor::from_slice(&[10.0]);
        let targets = Tensor::from_slice(&[1.0]);

        let criterion = BCEWithLogitsLoss::new();
        let loss = criterion.forward(&logits, &targets);

        // Loss should be very small
        assert!(loss.item() < 0.001);
    }

    #[test]
    fn test_bce_with_logits_wrong() {
        // Wrong prediction: logit = 10 (high sigmoid) for target = 0
        let logits = Tensor::from_slice(&[10.0]);
        let targets = Tensor::from_slice(&[0.0]);

        let criterion = BCEWithLogitsLoss::new();
        let loss = criterion.forward(&logits, &targets);

        // Loss should be high
        assert!(loss.item() > 5.0);
    }

    #[test]
    fn test_nll_loss() {
        // Log-probs where class 1 has highest probability
        let log_probs = Tensor::new(&[-2.0, -0.1, -3.0], &[1, 3]);
        let targets = Tensor::from_slice(&[1.0]);

        let criterion = NLLLoss::new();
        let loss = criterion.forward(&log_probs, &targets);

        // NLL = -log_probs[target] = -(-0.1) = 0.1
        assert!((loss.item() - 0.1).abs() < 1e-5);
    }

    #[test]
    fn test_reduction_modes() {
        let pred = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let target = Tensor::from_slice(&[0.0, 0.0, 0.0, 0.0]);

        // None: returns per-element loss
        let criterion = MSELoss::with_reduction(Reduction::None);
        let loss = criterion.forward(&pred, &target);
        assert_eq!(loss.shape(), &[4]);
        assert_eq!(loss.data(), &[1.0, 4.0, 9.0, 16.0]);

        // Sum
        let criterion = MSELoss::with_reduction(Reduction::Sum);
        let loss = criterion.forward(&pred, &target);
        assert!((loss.item() - 30.0).abs() < 1e-5);

        // Mean
        let criterion = MSELoss::with_reduction(Reduction::Mean);
        let loss = criterion.forward(&pred, &target);
        assert!((loss.item() - 7.5).abs() < 1e-5);
    }

    #[test]
    fn test_cross_entropy_gradient() {
        clear_graph();

        // Simple classification: 2 samples, 3 classes
        // logits favor class 1 for both samples
        let logits = Tensor::new(&[0.0, 2.0, 0.0, 0.0, 2.0, 0.0], &[2, 3]).requires_grad();
        let logits_id = logits.id();
        let targets = Tensor::from_slice(&[1.0, 1.0]); // Both target class 1

        let criterion = CrossEntropyLoss::new();
        let loss = criterion.forward(&logits, &targets);

        // Verify loss is computed
        assert!(
            loss.item() < 1.0,
            "Loss should be small for correct predictions"
        );

        // Backward pass
        loss.backward();

        // Verify gradients exist
        let grad = crate::autograd::get_grad(logits_id).expect("Should have gradient");
        assert_eq!(grad.shape(), &[2, 3], "Gradient shape should match logits");

        // Gradient for cross-entropy: softmax(logits) - one_hot(targets)
        // For logits [0, 2, 0], softmax ≈ [0.106, 0.788, 0.106]
        // Target class 1, so gradient ≈ [0.106, -0.212, 0.106] (after mean reduction)
        // Check that gradient for target class is negative (should decrease)
        let grad_data = grad.data();
        // Sample 0, class 1 (target): gradient should be negative
        assert!(
            grad_data[1] < 0.0,
            "Gradient for target class should be negative"
        );
        // Sample 0, class 0 (non-target): gradient should be positive
        assert!(
            grad_data[0] > 0.0,
            "Gradient for non-target class should be positive"
        );
    }
}

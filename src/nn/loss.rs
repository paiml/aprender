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
    /// Create a new `MSELoss` with default reduction (mean).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create `MSELoss` with specified reduction.
    #[must_use]
    pub fn with_reduction(reduction: Reduction) -> Self {
        Self { reduction }
    }

    /// Compute MSE loss between predictions and targets.
    #[must_use]
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
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn with_reduction(reduction: Reduction) -> Self {
        Self { reduction }
    }

    /// Compute L1 loss between predictions and targets.
    #[must_use]
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
    #[must_use]
    pub fn new() -> Self {
        Self {
            beta: 1.0,
            reduction: Reduction::Mean,
        }
    }

    #[must_use]
    pub fn with_beta(beta: f32) -> Self {
        Self {
            beta,
            reduction: Reduction::Mean,
        }
    }

    #[must_use]
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
/// Combines `log_softmax` and negative log likelihood for numerical stability.
///
/// For multi-class classification:
/// ```text
/// loss = -log(softmax(logits)[target_class])
/// ```
///
/// # Arguments
///
/// * `logits` - Raw model outputs, shape `[batch, num_classes]`
/// * `targets` - Target class indices, shape `[batch]`
#[derive(Debug, Clone, Default)]
pub struct CrossEntropyLoss {
    reduction: Reduction,
    label_smoothing: f32,
}

impl CrossEntropyLoss {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn with_reduction(reduction: Reduction) -> Self {
        Self {
            reduction,
            label_smoothing: 0.0,
        }
    }

    #[must_use]
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
    /// * `logits` - Shape `[batch, num_classes]`
    /// * `targets` - Shape `[batch]`, integer class indices (as f32)
    #[must_use]
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
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn with_reduction(reduction: Reduction) -> Self {
        Self {
            reduction,
            pos_weight: None,
        }
    }

    /// Set positive class weight for imbalanced datasets.
    #[must_use]
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
    #[must_use]
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
/// Expects log-probabilities as input (use after `log_softmax`).
#[derive(Debug, Clone, Copy, Default)]
pub struct NLLLoss {
    reduction: Reduction,
}

#[path = "loss_part_02.rs"]
mod loss_part_02;
#[allow(clippy::wildcard_imports)]
use loss_part_02::*;
#[path = "loss_part_03.rs"]
mod loss_part_03;

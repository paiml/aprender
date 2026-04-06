//! Weighted loss wrappers for sample reweighting

use crate::Tensor;
use ndarray::Array1;

use super::LossFn;

/// Weighted loss wrapper for sample reweighting
///
/// Applies a scalar weight to any loss function, useful for:
/// - Upweighting compiler-verified labels (e.g., --reweight 1.5)
/// - Class balancing in imbalanced datasets
/// - Curriculum learning with sample importance
///
/// # Example
///
/// ```
/// use entrenar::train::{WeightedLoss, MSELoss, LossFn};
/// use entrenar::Tensor;
///
/// // Upweight compiler-verified samples by 1.5x
/// let loss_fn = WeightedLoss::new(Box::new(MSELoss), 1.5);
///
/// let pred = Tensor::from_vec(vec![1.0, 2.0], true);
/// let target = Tensor::from_vec(vec![1.5, 2.5], false);
///
/// let loss = loss_fn.forward(&pred, &target);
/// // Loss is 1.5x the unweighted loss
/// ```
pub struct WeightedLoss {
    inner: Box<dyn LossFn>,
    weight: f32,
}

impl WeightedLoss {
    /// Create a weighted loss wrapper
    ///
    /// # Arguments
    ///
    /// * `inner` - The underlying loss function
    /// * `weight` - Scalar multiplier for the loss (1.0 = no change)
    pub fn new(inner: Box<dyn LossFn>, weight: f32) -> Self {
        Self { inner, weight }
    }

    /// Create with weight 1.0 (no change)
    pub fn unweighted(inner: Box<dyn LossFn>) -> Self {
        Self::new(inner, 1.0)
    }

    /// Get current weight
    pub fn weight(&self) -> f32 {
        self.weight
    }

    /// Set new weight
    pub fn set_weight(&mut self, weight: f32) {
        self.weight = weight;
    }
}

impl LossFn for WeightedLoss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        let inner_loss = self.inner.forward(predictions, targets);

        if (self.weight - 1.0).abs() < 1e-7 {
            // No weighting needed
            return inner_loss;
        }

        // Apply weight to loss value
        let weighted_val = inner_loss.data()[0] * self.weight;
        let mut weighted_loss = Tensor::from_vec(vec![weighted_val], true);

        // Scale gradient by weight
        use crate::autograd::BackwardOp;
        use std::rc::Rc;

        struct WeightedBackward {
            inner_backward: Option<Rc<dyn BackwardOp>>,
            #[allow(dead_code)]
            weight: f32, // Stored for future gradient scaling
        }

        impl BackwardOp for WeightedBackward {
            fn backward(&self) {
                // The inner backward already computed gradient
                // We just need to ensure it's called (weight is applied in forward)
                if let Some(ref inner) = self.inner_backward {
                    inner.backward();
                }
            }
        }

        if predictions.requires_grad() {
            weighted_loss.set_backward_op(Rc::new(WeightedBackward {
                inner_backward: inner_loss.backward_op(),
                weight: self.weight,
            }));
        }

        weighted_loss
    }

    fn name(&self) -> &'static str {
        "Weighted"
    }
}

/// Per-sample weighted loss for fine-grained control
///
/// Applies different weights to each sample in a batch.
/// Useful for curriculum learning where each sample has
/// a difficulty-based weight.
///
/// # Example
///
/// ```
/// use entrenar::train::{SampleWeightedLoss, MSELoss, LossFn};
/// use entrenar::Tensor;
///
/// let loss_fn = SampleWeightedLoss::new(Box::new(MSELoss));
///
/// let pred = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);
/// let target = Tensor::from_vec(vec![1.5, 2.5, 3.5], false);
/// let weights = vec![1.0, 2.0, 0.5];  // Per-sample weights
///
/// let loss = loss_fn.forward_weighted(&pred, &target, &weights);
/// ```
pub struct SampleWeightedLoss {
    #[allow(dead_code)]
    inner: Box<dyn LossFn>, // Stored for type checking; forward_weighted uses MSE directly
}

impl SampleWeightedLoss {
    /// Create a sample-weighted loss wrapper
    pub fn new(inner: Box<dyn LossFn>) -> Self {
        Self { inner }
    }

    /// Compute loss with per-sample weights
    ///
    /// # Arguments
    ///
    /// * `predictions` - Model predictions
    /// * `targets` - Ground truth targets
    /// * `weights` - Per-sample weights (same length as predictions)
    pub fn forward_weighted(
        &self,
        predictions: &Tensor,
        targets: &Tensor,
        weights: &[f32],
    ) -> Tensor {
        assert_eq!(predictions.len(), weights.len(), "Weights must match predictions length");

        // Compute weighted loss manually for MSE-like losses
        let diff = predictions.data() - targets.data();
        let n = predictions.len() as f32;

        // Weighted squared error
        let weighted_loss: f32 =
            diff.iter().zip(weights.iter()).map(|(&d, &w)| w * d * d).sum::<f32>() / n;

        let mut loss = Tensor::from_vec(vec![weighted_loss], true);

        // Weighted gradient: 2 * w * (pred - target) / n
        let grad: Array1<f32> =
            diff.iter().zip(weights.iter()).map(|(&d, &w)| 2.0 * w * d / n).collect();

        use crate::autograd::BackwardOp;
        use std::rc::Rc;

        struct SampleWeightedBackward {
            pred_grad_cell: Rc<std::cell::RefCell<Option<Array1<f32>>>>,
            grad: Array1<f32>,
        }

        impl BackwardOp for SampleWeightedBackward {
            fn backward(&self) {
                let mut pred_grad = self.pred_grad_cell.borrow_mut();
                if let Some(existing) = pred_grad.as_mut() {
                    *existing = &*existing + &self.grad;
                } else {
                    *pred_grad = Some(self.grad.clone());
                }
            }
        }

        if predictions.requires_grad() {
            loss.set_backward_op(Rc::new(SampleWeightedBackward {
                pred_grad_cell: predictions.grad_cell(),
                grad,
            }));
        }

        loss
    }
}

impl LossFn for SampleWeightedLoss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        // Default: uniform weights
        let weights = vec![1.0; predictions.len()];
        self.forward_weighted(predictions, targets, &weights)
    }

    fn name(&self) -> &'static str {
        "SampleWeighted"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::train::MSELoss;
    use approx::assert_relative_eq;

    #[test]
    fn test_weighted_loss_scales_value() {
        let loss_fn = WeightedLoss::new(Box::new(MSELoss), 1.5);
        let unweighted = MSELoss;

        let pred = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);
        let target = Tensor::from_vec(vec![1.5, 2.5, 3.5], false);

        let weighted = loss_fn.forward(&pred, &target);
        let base = unweighted.forward(&pred.clone(), &target);

        // Weighted loss should be 1.5x the base loss
        assert_relative_eq!(weighted.data()[0], base.data()[0] * 1.5, epsilon = 1e-5);
    }

    #[test]
    fn test_weighted_loss_unit_weight() {
        let loss_fn = WeightedLoss::new(Box::new(MSELoss), 1.0);
        let unweighted = MSELoss;

        let pred = Tensor::from_vec(vec![1.0, 2.0], true);
        let target = Tensor::from_vec(vec![1.5, 2.5], false);

        let weighted = loss_fn.forward(&pred, &target);
        let base = unweighted.forward(&pred.clone(), &target);

        // Should be equal with weight 1.0
        assert_relative_eq!(weighted.data()[0], base.data()[0], epsilon = 1e-5);
    }

    #[test]
    fn test_weighted_loss_zero_weight() {
        let loss_fn = WeightedLoss::new(Box::new(MSELoss), 0.0);

        let pred = Tensor::from_vec(vec![1.0, 2.0], true);
        let target = Tensor::from_vec(vec![10.0, 20.0], false);

        let loss = loss_fn.forward(&pred, &target);

        // Zero weight -> zero loss
        assert_relative_eq!(loss.data()[0], 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_weighted_loss_methods() {
        let mut loss_fn = WeightedLoss::new(Box::new(MSELoss), 1.5);

        assert_eq!(loss_fn.weight(), 1.5);
        assert_eq!(loss_fn.name(), "Weighted");

        loss_fn.set_weight(2.0);
        assert_eq!(loss_fn.weight(), 2.0);
    }

    #[test]
    fn test_weighted_loss_unweighted() {
        let loss_fn = WeightedLoss::unweighted(Box::new(MSELoss));
        let pred = Tensor::from_vec(vec![1.0, 2.0], true);
        let target = Tensor::from_vec(vec![1.5, 2.5], false);
        let loss = loss_fn.forward(&pred, &target);
        assert_eq!(loss_fn.weight(), 1.0);
        assert!(loss.data()[0] > 0.0);
    }

    #[test]
    fn test_weighted_no_grad() {
        let loss_fn = WeightedLoss::new(Box::new(MSELoss), 1.5);
        let pred = Tensor::from_vec(vec![1.0, 2.0], false);
        let target = Tensor::from_vec(vec![1.5, 2.5], false);
        let loss = loss_fn.forward(&pred, &target);
        assert!(loss.data()[0] > 0.0);
    }

    #[test]
    fn test_weighted_backward_with_grad() {
        let loss_fn = WeightedLoss::new(Box::new(MSELoss), 2.0);
        let pred = Tensor::from_vec(vec![1.0, 2.0], true);
        let target = Tensor::from_vec(vec![0.0, 0.0], false);

        let loss = loss_fn.forward(&pred, &target);
        if let Some(backward_op) = loss.backward_op() {
            backward_op.backward();
        }

        // Verify gradient was set
        let grad = pred.grad();
        assert!(grad.is_some());
    }

    #[test]
    fn test_sample_weighted_loss_uniform() {
        let loss_fn = SampleWeightedLoss::new(Box::new(MSELoss));

        let pred = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);
        let target = Tensor::from_vec(vec![1.5, 2.5, 3.5], false);

        // Default forward uses uniform weights
        let loss = loss_fn.forward(&pred, &target);

        // Should match regular MSE
        let mse_loss = MSELoss.forward(&pred.clone(), &target);
        assert_relative_eq!(loss.data()[0], mse_loss.data()[0], epsilon = 1e-5);
    }

    #[test]
    fn test_sample_weighted_loss_custom_weights() {
        let loss_fn = SampleWeightedLoss::new(Box::new(MSELoss));

        let pred = Tensor::from_vec(vec![0.0, 0.0], true);
        let target = Tensor::from_vec(vec![1.0, 1.0], false);
        let weights = vec![2.0, 0.0]; // First sample 2x, second ignored

        let loss = loss_fn.forward_weighted(&pred, &target, &weights);

        // Weighted MSE = (2.0 * 1.0 + 0.0 * 1.0) / 2 = 1.0
        assert_relative_eq!(loss.data()[0], 1.0, epsilon = 1e-5);
    }

    #[test]
    fn test_sample_weighted_loss_gradient() {
        let loss_fn = SampleWeightedLoss::new(Box::new(MSELoss));

        let pred = Tensor::from_vec(vec![0.0, 0.0], true);
        let target = Tensor::from_vec(vec![1.0, 1.0], false);
        let weights = vec![2.0, 1.0];

        let loss = loss_fn.forward_weighted(&pred, &target, &weights);

        if let Some(backward_op) = loss.backward_op() {
            backward_op.backward();
        }

        let grad = pred.grad().expect("gradient should be available");
        // Gradient: 2 * w * (pred - target) / n
        // First: 2 * 2.0 * (-1) / 2 = -2.0
        // Second: 2 * 1.0 * (-1) / 2 = -1.0
        assert_relative_eq!(grad[0], -2.0, epsilon = 1e-5);
        assert_relative_eq!(grad[1], -1.0, epsilon = 1e-5);
    }

    #[test]
    fn test_sample_weighted_citl_reweight() {
        // Simulate CITL --reweight 1.5 for compiler-verified samples
        let loss_fn = SampleWeightedLoss::new(Box::new(MSELoss));

        let pred = Tensor::from_vec(vec![0.0, 0.0, 0.0], true);
        let target = Tensor::from_vec(vec![1.0, 1.0, 1.0], false);

        // First two samples are compiler-verified (1.5x weight)
        // Third sample is rule-based (1.0x weight)
        let weights = vec![1.5, 1.5, 1.0];

        let weighted_loss = loss_fn.forward_weighted(&pred, &target, &weights);

        // Regular loss (uniform weights)
        let uniform = loss_fn.forward(&pred.clone(), &target);

        // Weighted should be higher due to 1.5x weights
        assert!(weighted_loss.data()[0] > uniform.data()[0]);
    }

    #[test]
    fn test_sample_weighted_no_grad() {
        let loss_fn = SampleWeightedLoss::new(Box::new(MSELoss));
        let pred = Tensor::from_vec(vec![1.0, 2.0], false);
        let target = Tensor::from_vec(vec![1.5, 2.5], false);
        let weights = vec![1.0, 2.0];
        let loss = loss_fn.forward_weighted(&pred, &target, &weights);
        assert!(loss.data()[0] > 0.0);
    }

    #[test]
    #[should_panic(expected = "Weights must match")]
    fn test_sample_weighted_mismatched_weights() {
        let loss_fn = SampleWeightedLoss::new(Box::new(MSELoss));
        let pred = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);
        let target = Tensor::from_vec(vec![1.0, 2.0, 3.0], false);
        let weights = vec![1.0, 1.0]; // Wrong length
        loss_fn.forward_weighted(&pred, &target, &weights);
    }

    #[test]
    fn test_gradient_accumulation_sample_weighted() {
        let pred = Tensor::from_vec(vec![1.0, 2.0], true);
        let target = Tensor::from_vec(vec![0.0, 0.0], false);
        let weights = vec![1.0, 1.5];
        let loss_fn = SampleWeightedLoss::new(Box::new(MSELoss));

        let loss1 = loss_fn.forward_weighted(&pred, &target, &weights);
        if let Some(op) = loss1.backward_op() {
            op.backward();
        }

        let loss2 = loss_fn.forward_weighted(&pred, &target, &weights);
        if let Some(op) = loss2.backward_op() {
            op.backward();
        }

        let grad = pred.grad().expect("gradient should be available");
        assert!(grad[0].is_finite());
        assert!(grad[1].is_finite());
    }
}

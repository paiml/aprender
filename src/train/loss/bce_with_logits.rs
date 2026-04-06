//! Binary Cross-Entropy with Logits Loss for multi-label classification
//!
//! Combines a sigmoid activation with binary cross-entropy loss.
//! Each output is treated as an independent binary classification,
//! allowing multiple labels to be active simultaneously.
//!
//! # Formula
//!
//! Numerically stable computation:
//! ```text
//! L_i = max(x_i, 0) - x_i * t_i + log(1 + exp(-|x_i|))
//! L = mean(L_i) over all i
//! ```
//!
//! Gradient: `∂L/∂x_i = σ(x_i) - t_i`
//!
//! # Multi-label vs single-label
//!
//! - **CrossEntropyLoss**: softmax → mutual exclusion (single label)
//! - **BCEWithLogitsLoss**: sigmoid → independent per-class (multi-label)

use crate::Tensor;
use ndarray::Array1;

use super::LossFn;

/// Binary Cross-Entropy with Logits Loss.
///
/// For multi-label classification where each class is an independent binary decision.
/// Targets are multi-hot vectors (e.g., `[1.0, 0.0, 1.0, 0.0, 1.0]` for classes 0, 2, 4).
///
/// # Example
///
/// ```
/// use entrenar::train::{BCEWithLogitsLoss, LossFn};
/// use entrenar::Tensor;
///
/// let loss_fn = BCEWithLogitsLoss;
/// let logits = Tensor::from_vec(vec![2.0, -1.0, 0.5], true);
/// let targets = Tensor::from_vec(vec![1.0, 0.0, 1.0], false); // multi-hot
///
/// let loss = loss_fn.forward(&logits, &targets);
/// assert!(loss.data()[0] > 0.0);
/// ```
pub struct BCEWithLogitsLoss;

impl BCEWithLogitsLoss {
    /// Compute element-wise sigmoid: σ(x) = 1 / (1 + exp(-x))
    pub(crate) fn sigmoid(x: &Array1<f32>) -> Array1<f32> {
        contract_pre_sigmoid!();
        let result = x.mapv(|v| {
            // Numerically stable sigmoid
            if v >= 0.0 {
                let exp_neg = (-v).exp();
                1.0 / (1.0 + exp_neg)
            } else {
                let exp_v = v.exp();
                exp_v / (1.0 + exp_v)
            }
        });
        contract_post_silu!(result);
        result
    }

    /// Numerically stable BCE: max(x, 0) - x*t + log(1 + exp(-|x|))
    fn stable_bce(logit: f32, target: f32) -> f32 {
        let relu = logit.max(0.0);
        let abs_x = logit.abs();
        relu - logit * target + (1.0 + (-abs_x).exp()).ln()
    }
}

impl LossFn for BCEWithLogitsLoss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        assert_eq!(
            predictions.len(),
            targets.len(),
            "Predictions and targets must have same length"
        );

        // Compute per-element BCE loss
        let total_loss: f32 = predictions
            .data()
            .iter()
            .zip(targets.data().iter())
            .map(|(&logit, &target)| Self::stable_bce(logit, target))
            .sum::<f32>()
            / predictions.len() as f32;

        let mut loss = Tensor::from_vec(vec![total_loss], true);

        // Gradient: ∂L/∂x_i = (σ(x_i) - t_i) / N
        let sigmoid_vals = Self::sigmoid(predictions.data());
        let n = predictions.len() as f32;
        let grad = (&sigmoid_vals - targets.data()) / n;

        use crate::autograd::BackwardOp;
        use std::rc::Rc;

        struct BCEBackward {
            pred_grad_cell: Rc<std::cell::RefCell<Option<Array1<f32>>>>,
            grad: Array1<f32>,
        }

        impl BackwardOp for BCEBackward {
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
            loss.set_backward_op(Rc::new(BCEBackward {
                pred_grad_cell: predictions.grad_cell(),
                grad,
            }));
        }

        loss
    }

    fn name(&self) -> &'static str {
        "BCEWithLogits"
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_bce_with_logits_loss_basic() {
        let loss_fn = BCEWithLogitsLoss;
        let logits = Tensor::from_vec(vec![2.0, -1.0, 0.5], true);
        let targets = Tensor::from_vec(vec![1.0, 0.0, 1.0], false);

        let loss = loss_fn.forward(&logits, &targets);
        assert!(loss.data()[0] > 0.0);
        assert!(loss.data()[0].is_finite());
    }

    #[test]
    fn test_sigmoid_basic() {
        let x = Array1::from(vec![0.0, 100.0, -100.0]);
        let s = BCEWithLogitsLoss::sigmoid(&x);

        assert_relative_eq!(s[0], 0.5, epsilon = 1e-5);
        assert_relative_eq!(s[1], 1.0, epsilon = 1e-5);
        assert_relative_eq!(s[2], 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_sigmoid_symmetry() {
        // σ(x) + σ(-x) = 1
        let x = Array1::from(vec![1.0, 2.0, -3.0, 0.5]);
        let neg_x = x.mapv(|v| -v);
        let s_x = BCEWithLogitsLoss::sigmoid(&x);
        let s_neg_x = BCEWithLogitsLoss::sigmoid(&neg_x);

        for i in 0..x.len() {
            assert_relative_eq!(s_x[i] + s_neg_x[i], 1.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_bce_perfect_prediction() {
        let loss_fn = BCEWithLogitsLoss;
        // Logits that strongly match targets → low loss
        let logits = Tensor::from_vec(vec![100.0, -100.0, 100.0, -100.0, 100.0], true);
        let targets = Tensor::from_vec(vec![1.0, 0.0, 1.0, 0.0, 1.0], false);

        let loss = loss_fn.forward(&logits, &targets);
        assert!(loss.data()[0] < 0.01, "Perfect prediction should have near-zero loss");
    }

    #[test]
    fn test_bce_wrong_prediction() {
        let loss_fn = BCEWithLogitsLoss;
        // Logits that strongly disagree with targets → high loss
        let logits = Tensor::from_vec(vec![-100.0, 100.0, -100.0], true);
        let targets = Tensor::from_vec(vec![1.0, 0.0, 1.0], false);

        let loss = loss_fn.forward(&logits, &targets);
        assert!(loss.data()[0] > 10.0, "Wrong prediction should have high loss");
    }

    #[test]
    fn test_bce_gradient_direction() {
        let loss_fn = BCEWithLogitsLoss;
        let logits = Tensor::from_vec(vec![2.0, -1.0, 0.5], true);
        let targets = Tensor::from_vec(vec![1.0, 0.0, 1.0], false);

        let loss = loss_fn.forward(&logits, &targets);
        if let Some(backward_op) = loss.backward_op() {
            backward_op.backward();
        }

        let grad = logits.grad().expect("gradient should be available");
        // For target=1 with positive logit: grad should be negative (push logit higher)
        assert!(grad[0] < 0.0, "grad[0] should be negative (target=1, logit=2.0)");
        // For target=0 with negative logit: grad should be positive (push logit lower)
        assert!(grad[1] > 0.0, "grad[1] should be positive (target=0, logit=-1.0)");
        // All gradients finite
        for g in &grad {
            assert!(g.is_finite());
        }
    }

    #[test]
    fn test_bce_gradient_at_zero() {
        let loss_fn = BCEWithLogitsLoss;
        // At logit=0, sigmoid=0.5
        let logits = Tensor::from_vec(vec![0.0], true);
        let targets = Tensor::from_vec(vec![1.0], false);

        let loss = loss_fn.forward(&logits, &targets);
        if let Some(op) = loss.backward_op() {
            op.backward();
        }

        let grad = logits.grad().expect("gradient should be available");
        // ∂L/∂x = (σ(0) - 1) / 1 = (0.5 - 1) / 1 = -0.5
        assert_relative_eq!(grad[0], -0.5, epsilon = 1e-5);
    }

    #[test]
    fn test_bce_all_zeros_target() {
        let loss_fn = BCEWithLogitsLoss;
        let logits = Tensor::from_vec(vec![0.0; 5], true);
        let targets = Tensor::from_vec(vec![0.0; 5], false);

        let loss = loss_fn.forward(&logits, &targets);
        // log(1 + exp(0)) = log(2) ≈ 0.693 per element
        assert_relative_eq!(loss.data()[0], 2.0_f32.ln(), epsilon = 1e-5);
    }

    #[test]
    fn test_bce_all_ones_target() {
        let loss_fn = BCEWithLogitsLoss;
        let logits = Tensor::from_vec(vec![0.0; 5], true);
        let targets = Tensor::from_vec(vec![1.0; 5], false);

        let loss = loss_fn.forward(&logits, &targets);
        // Same: log(2) per element (symmetric when logit=0)
        assert_relative_eq!(loss.data()[0], 2.0_f32.ln(), epsilon = 1e-5);
    }

    #[test]
    fn test_bce_numerical_stability_large_positive() {
        let loss_fn = BCEWithLogitsLoss;
        let logits = Tensor::from_vec(vec![1000.0, 500.0, 100.0], true);
        let targets = Tensor::from_vec(vec![1.0, 1.0, 1.0], false);

        let loss = loss_fn.forward(&logits, &targets);
        assert!(loss.data()[0].is_finite(), "Must be stable for large positive logits");
        assert!(loss.data()[0] < 0.01, "Loss should be near-zero for correct large logits");
    }

    #[test]
    fn test_bce_numerical_stability_large_negative() {
        let loss_fn = BCEWithLogitsLoss;
        let logits = Tensor::from_vec(vec![-1000.0, -500.0, -100.0], true);
        let targets = Tensor::from_vec(vec![0.0, 0.0, 0.0], false);

        let loss = loss_fn.forward(&logits, &targets);
        assert!(loss.data()[0].is_finite(), "Must be stable for large negative logits");
        assert!(loss.data()[0] < 0.01, "Loss should be near-zero for correct large logits");
    }

    #[test]
    #[should_panic(expected = "must have same length")]
    fn test_bce_mismatched_lengths() {
        let loss_fn = BCEWithLogitsLoss;
        let pred = Tensor::from_vec(vec![1.0, 2.0], true);
        let target = Tensor::from_vec(vec![1.0, 2.0, 3.0], false);
        loss_fn.forward(&pred, &target);
    }

    #[test]
    fn test_bce_no_grad() {
        let loss_fn = BCEWithLogitsLoss;
        let pred = Tensor::from_vec(vec![2.0, -1.0], false);
        let target = Tensor::from_vec(vec![1.0, 0.0], false);
        let loss = loss_fn.forward(&pred, &target);
        assert!(loss.data()[0] > 0.0);
    }

    #[test]
    fn test_bce_gradient_accumulation() {
        let logits = Tensor::from_vec(vec![1.0, -1.0], true);
        let targets = Tensor::from_vec(vec![1.0, 0.0], false);

        let loss1 = BCEWithLogitsLoss.forward(&logits, &targets);
        if let Some(op) = loss1.backward_op() {
            op.backward();
        }

        let loss2 = BCEWithLogitsLoss.forward(&logits, &targets);
        if let Some(op) = loss2.backward_op() {
            op.backward();
        }

        let grad = logits.grad().expect("gradient should be available");
        assert!(grad[0].is_finite());
        assert!(grad[1].is_finite());
    }

    #[test]
    fn test_bce_name() {
        assert_eq!(BCEWithLogitsLoss.name(), "BCEWithLogits");
    }

    #[test]
    fn test_stable_bce_formula() {
        // Verify against naive (potentially unstable) formula
        // For moderate values, both should agree
        let logit = 1.5f32;
        let target = 0.7f32;

        let stable = BCEWithLogitsLoss::stable_bce(logit, target);

        // Naive: -[t * log(σ(x)) + (1-t) * log(1 - σ(x))]
        let sigma = 1.0 / (1.0 + (-logit).exp());
        let naive = -(target * sigma.ln() + (1.0 - target) * (1.0 - sigma).ln());

        assert_relative_eq!(stable, naive, epsilon = 1e-5);
    }

    #[test]
    fn test_multi_label_scenario() {
        // Real multi-label: script is both non-deterministic AND needs-quoting
        let loss_fn = BCEWithLogitsLoss;
        // 5 classes: safe, needs-quoting, non-det, non-idem, unsafe
        let logits = Tensor::from_vec(vec![-2.0, 3.0, 4.0, -1.0, -3.0], true);
        let targets = Tensor::from_vec(vec![0.0, 1.0, 1.0, 0.0, 0.0], false);

        let loss = loss_fn.forward(&logits, &targets);
        assert!(loss.data()[0].is_finite());
        assert!(loss.data()[0] > 0.0);
    }
}

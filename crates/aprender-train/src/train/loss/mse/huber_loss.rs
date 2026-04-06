//! Huber Loss (Smooth L1 Loss)
//!
//! Forward scalar computation delegates to [`aprender::loss::huber_loss`].
//! Gradient computation (backward) is entrenar's autograd concern.

use aprender::primitives::Vector;

use crate::autograd::BackwardOp;
use crate::Tensor;
use ndarray::Array1;
use std::rc::Rc;

use crate::train::loss::LossFn;

/// Huber Loss (Smooth L1 Loss)
///
/// Combines MSE for small errors and MAE for large errors,
/// making it robust to outliers.
///
/// For |error| <= delta:  L = 0.5 * error^2
/// For |error| > delta:   L = delta * (|error| - 0.5 * delta)
///
/// # Example
///
/// ```
/// use entrenar::train::{HuberLoss, LossFn};
/// use entrenar::Tensor;
///
/// let loss_fn = HuberLoss::new(1.0);
/// let pred = Tensor::from_vec(vec![1.0, 2.0, 10.0], true);  // 10.0 is outlier
/// let target = Tensor::from_vec(vec![1.5, 2.5, 0.0], false);
///
/// let loss = loss_fn.forward(&pred, &target);
/// assert!(loss.data()[0] > 0.0);
/// ```
pub struct HuberLoss {
    /// Threshold for switching between quadratic and linear
    delta: f32,
}

impl HuberLoss {
    /// Create Huber loss with given delta threshold
    pub fn new(delta: f32) -> Self {
        assert!(delta > 0.0, "delta must be positive");
        Self { delta }
    }

    /// Create Huber loss with default delta = 1.0
    pub fn default_delta() -> Self {
        Self::new(1.0)
    }
}

impl Default for HuberLoss {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl LossFn for HuberLoss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        assert_eq!(
            predictions.len(),
            targets.len(),
            "Predictions and targets must have same length"
        );

        // Delegate forward scalar to aprender
        let pred_vec =
            Vector::from_slice(predictions.data().as_slice().expect("contiguous tensor data"));
        let tgt_vec =
            Vector::from_slice(targets.data().as_slice().expect("contiguous tensor data"));
        let mean_loss = aprender::loss::huber_loss(&pred_vec, &tgt_vec, self.delta);

        let mut loss = Tensor::from_vec(vec![mean_loss], true);

        // Gradient computation is entrenar's autograd concern
        let diff = predictions.data() - targets.data();
        let n = predictions.len() as f32;
        let delta = self.delta;

        // Compute gradient per element
        // d(Huber)/d(pred) = error if |error| <= delta
        //                  = delta * sign(error) if |error| > delta
        let grad: Array1<f32> = diff
            .iter()
            .map(|&d| {
                let abs_d = d.abs();
                if abs_d <= delta {
                    d / n
                } else {
                    delta * d.signum() / n
                }
            })
            .collect();

        struct HuberBackward {
            pred_grad_cell: Rc<std::cell::RefCell<Option<Array1<f32>>>>,
            grad: Array1<f32>,
        }

        impl BackwardOp for HuberBackward {
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
            loss.set_backward_op(Rc::new(HuberBackward {
                pred_grad_cell: predictions.grad_cell(),
                grad,
            }));
        }

        loss
    }

    fn name(&self) -> &'static str {
        "Huber"
    }
}

/// Smooth L1 Loss (alias for HuberLoss with delta=1.0)
///
/// Equivalent to HuberLoss with delta=1.0, commonly used in
/// object detection (e.g., Faster R-CNN).
pub type SmoothL1Loss = HuberLoss;

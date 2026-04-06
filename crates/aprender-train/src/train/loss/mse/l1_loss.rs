//! L1 Loss (Mean Absolute Error)
//!
//! Forward scalar computation delegates to [`aprender::loss::mae_loss`].
//! Gradient computation (backward) is entrenar's autograd concern.

use aprender::primitives::Vector;

use crate::autograd::BackwardOp;
use crate::Tensor;
use ndarray::Array1;
use std::rc::Rc;

use crate::train::loss::LossFn;

/// L1 Loss (Mean Absolute Error)
///
/// L = mean(|predictions - targets|)
///
/// Forward scalar delegates to [`aprender::loss::mae_loss`].
/// Backward gradient is computed by entrenar's autograd.
///
/// # Example
///
/// ```
/// use entrenar::train::{L1Loss, LossFn};
/// use entrenar::Tensor;
///
/// let loss_fn = L1Loss;
/// let pred = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);
/// let target = Tensor::from_vec(vec![1.5, 2.5, 3.5], false);
///
/// let loss = loss_fn.forward(&pred, &target);
/// assert!(loss.data()[0] > 0.0);
/// ```
pub struct L1Loss;

impl LossFn for L1Loss {
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
        let mae = aprender::loss::mae_loss(&pred_vec, &tgt_vec);

        let mut loss = Tensor::from_vec(vec![mae], true);

        // Gradient computation is entrenar's autograd concern
        // d(L1)/d(pred) = sign(pred - target) / n
        let diff = predictions.data() - targets.data();
        let n = predictions.len() as f32;
        let grad: Array1<f32> = diff.mapv(|d| d.signum() / n);

        struct L1Backward {
            pred_grad_cell: Rc<std::cell::RefCell<Option<Array1<f32>>>>,
            grad: Array1<f32>,
        }

        impl BackwardOp for L1Backward {
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
            loss.set_backward_op(Rc::new(L1Backward {
                pred_grad_cell: predictions.grad_cell(),
                grad,
            }));
        }

        loss
    }

    fn name(&self) -> &'static str {
        "L1"
    }
}

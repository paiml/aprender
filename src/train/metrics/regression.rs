//! Regression metrics: R2Score, MAE, RMSE
//!
//! These metrics delegate computation to `aprender::metrics` for the core math,
//! while wrapping them in entrenar's `Metric` trait for integration with the
//! training loop and evaluation framework.

use crate::Tensor;
use aprender::primitives::Vector;

use super::Metric;

/// Convert a Tensor's data to an aprender Vector for delegation.
fn tensor_to_vector(t: &Tensor) -> Vector<f32> {
    Vector::from_slice(t.data().as_slice().expect("contiguous tensor data"))
}

/// R² (coefficient of determination) for regression
///
/// Delegates to [`aprender::metrics::r_squared`] for computation.
///
/// R² = 1 - SS_res / SS_tot
///
/// Where:
/// - SS_res = sum((y - y_pred)²)
/// - SS_tot = sum((y - y_mean)²)
///
/// R² = 1.0 is perfect prediction, 0.0 means predicting the mean
///
/// # Example
///
/// ```
/// use entrenar::train::{R2Score, Metric};
/// use entrenar::Tensor;
///
/// let metric = R2Score;
/// let pred = Tensor::from_vec(vec![1.0, 2.0, 3.0], false);
/// let target = Tensor::from_vec(vec![1.0, 2.0, 3.0], false);
///
/// let r2 = metric.compute(&pred, &target);
/// assert!((r2 - 1.0).abs() < 1e-5);  // Perfect prediction
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct R2Score;

impl Metric for R2Score {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> f32 {
        assert_eq!(predictions.len(), targets.len());

        if predictions.is_empty() {
            return 0.0;
        }

        let y_pred = tensor_to_vector(predictions);
        let y_true = tensor_to_vector(targets);
        let r2 = aprender::metrics::r_squared(&y_pred, &y_true);

        // aprender returns 0.0 for constant targets (ss_tot == 0);
        // entrenar returns 1.0 when prediction is also perfect (ss_res == 0)
        if r2 == 0.0 {
            let ss_res: f32 = predictions
                .data()
                .iter()
                .zip(targets.data().iter())
                .map(|(&p, &t)| (t - p).powi(2))
                .sum();
            if ss_res == 0.0 {
                return 1.0;
            }
        }

        r2
    }

    fn name(&self) -> &'static str {
        "R²"
    }
}

/// Mean Absolute Error (MAE) metric
///
/// Delegates to [`aprender::metrics::mae`] for computation.
///
/// MAE = mean(|y - y_pred|)
///
/// # Example
///
/// ```
/// use entrenar::train::{MAE, Metric};
/// use entrenar::Tensor;
///
/// let metric = MAE;
/// let pred = Tensor::from_vec(vec![1.0, 2.0, 3.0], false);
/// let target = Tensor::from_vec(vec![1.5, 2.5, 3.5], false);
///
/// let mae = metric.compute(&pred, &target);
/// assert!((mae - 0.5).abs() < 1e-5);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct MAE;

impl Metric for MAE {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> f32 {
        assert_eq!(predictions.len(), targets.len());

        if predictions.is_empty() {
            return 0.0;
        }

        let y_pred = tensor_to_vector(predictions);
        let y_true = tensor_to_vector(targets);
        aprender::metrics::mae(&y_pred, &y_true)
    }

    fn name(&self) -> &'static str {
        "MAE"
    }

    fn higher_is_better(&self) -> bool {
        false
    }
}

/// Root Mean Squared Error (RMSE) metric
///
/// Delegates to [`aprender::metrics::rmse`] for computation.
///
/// RMSE = sqrt(mean((y - y_pred)²))
///
/// # Example
///
/// ```
/// use entrenar::train::{RMSE, Metric};
/// use entrenar::Tensor;
///
/// let metric = RMSE;
/// let pred = Tensor::from_vec(vec![1.0, 2.0, 3.0], false);
/// let target = Tensor::from_vec(vec![1.0, 2.0, 3.0], false);
///
/// let rmse = metric.compute(&pred, &target);
/// assert!(rmse < 1e-5);  // Perfect prediction
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct RMSE;

impl Metric for RMSE {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> f32 {
        assert_eq!(predictions.len(), targets.len());

        if predictions.is_empty() {
            return 0.0;
        }

        let y_pred = tensor_to_vector(predictions);
        let y_true = tensor_to_vector(targets);
        aprender::metrics::rmse(&y_pred, &y_true)
    }

    fn name(&self) -> &'static str {
        "RMSE"
    }

    fn higher_is_better(&self) -> bool {
        false
    }
}

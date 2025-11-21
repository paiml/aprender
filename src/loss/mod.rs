//! Loss functions for training machine learning models.
//!
//! # Usage
//!
//! ```
//! use aprender::loss::{mse_loss, mae_loss, huber_loss};
//! use aprender::primitives::Vector;
//!
//! let y_true = Vector::from_slice(&[1.0, 2.0, 3.0]);
//! let y_pred = Vector::from_slice(&[1.1, 2.1, 2.9]);
//!
//! let mse = mse_loss(&y_pred, &y_true);
//! let mae = mae_loss(&y_pred, &y_true);
//! let huber = huber_loss(&y_pred, &y_true, 1.0);
//! ```

use crate::primitives::Vector;

/// Mean Squared Error (MSE) loss.
///
/// Computes the average squared difference between predictions and targets:
///
/// ```text
/// MSE = (1/n) * Σ(y_pred - y_true)²
/// ```
///
/// MSE is differentiable everywhere and heavily penalizes large errors.
///
/// # Arguments
///
/// * `y_pred` - Predicted values
/// * `y_true` - True target values
///
/// # Returns
///
/// The mean squared error
///
/// # Panics
///
/// Panics if `y_pred` and `y_true` have different lengths.
///
/// # Example
///
/// ```
/// use aprender::loss::mse_loss;
/// use aprender::primitives::Vector;
///
/// let y_true = Vector::from_slice(&[1.0, 2.0, 3.0]);
/// let y_pred = Vector::from_slice(&[1.0, 2.0, 3.0]);
///
/// let loss = mse_loss(&y_pred, &y_true);
/// assert!((loss - 0.0).abs() < 1e-6);
/// ```
pub fn mse_loss(y_pred: &Vector<f32>, y_true: &Vector<f32>) -> f32 {
    assert_eq!(
        y_pred.len(),
        y_true.len(),
        "Predicted and true values must have same length"
    );

    let n = y_pred.len() as f32;
    let mut sum = 0.0;

    for i in 0..y_pred.len() {
        let diff = y_pred[i] - y_true[i];
        sum += diff * diff;
    }

    sum / n
}

/// Mean Absolute Error (MAE) loss.
///
/// Computes the average absolute difference between predictions and targets:
///
/// ```text
/// MAE = (1/n) * Σ|y_pred - y_true|
/// ```
///
/// MAE is more robust to outliers than MSE but not differentiable at zero.
///
/// # Arguments
///
/// * `y_pred` - Predicted values
/// * `y_true` - True target values
///
/// # Returns
///
/// The mean absolute error
///
/// # Panics
///
/// Panics if `y_pred` and `y_true` have different lengths.
///
/// # Example
///
/// ```
/// use aprender::loss::mae_loss;
/// use aprender::primitives::Vector;
///
/// let y_true = Vector::from_slice(&[1.0, 2.0, 3.0]);
/// let y_pred = Vector::from_slice(&[1.5, 2.5, 2.5]);
///
/// let loss = mae_loss(&y_pred, &y_true);
/// assert!((loss - 0.5).abs() < 1e-6);
/// ```
pub fn mae_loss(y_pred: &Vector<f32>, y_true: &Vector<f32>) -> f32 {
    assert_eq!(
        y_pred.len(),
        y_true.len(),
        "Predicted and true values must have same length"
    );

    let n = y_pred.len() as f32;
    let mut sum = 0.0;

    for i in 0..y_pred.len() {
        sum += (y_pred[i] - y_true[i]).abs();
    }

    sum / n
}

/// Huber loss (smooth approximation of MAE).
///
/// Combines the benefits of MSE and MAE by being quadratic for small errors
/// and linear for large errors. This makes it robust to outliers while
/// remaining differentiable everywhere.
///
/// ```text
/// Huber(δ) = { 0.5 * (y_pred - y_true)²           if |y_pred - y_true| ≤ δ
///            { δ * (|y_pred - y_true| - 0.5 * δ)  otherwise
/// ```
///
/// where δ (delta) is a threshold parameter.
///
/// # Arguments
///
/// * `y_pred` - Predicted values
/// * `y_true` - True target values
/// * `delta` - Threshold for switching from quadratic to linear (typically 1.0)
///
/// # Returns
///
/// The Huber loss
///
/// # Panics
///
/// Panics if:
/// - `y_pred` and `y_true` have different lengths
/// - `delta` is not positive
///
/// # Example
///
/// ```
/// use aprender::loss::huber_loss;
/// use aprender::primitives::Vector;
///
/// let y_true = Vector::from_slice(&[1.0, 2.0, 3.0]);
/// let y_pred = Vector::from_slice(&[1.5, 2.0, 5.0]);
///
/// // With delta=1.0, small errors use MSE, large errors use MAE
/// let loss = huber_loss(&y_pred, &y_true, 1.0);
/// assert!(loss > 0.0);
/// ```
pub fn huber_loss(y_pred: &Vector<f32>, y_true: &Vector<f32>, delta: f32) -> f32 {
    assert_eq!(
        y_pred.len(),
        y_true.len(),
        "Predicted and true values must have same length"
    );
    assert!(delta > 0.0, "Delta must be positive");

    let n = y_pred.len() as f32;
    let mut sum = 0.0;

    for i in 0..y_pred.len() {
        let diff = (y_pred[i] - y_true[i]).abs();
        if diff <= delta {
            // Quadratic for small errors
            sum += 0.5 * diff * diff;
        } else {
            // Linear for large errors
            sum += delta * (diff - 0.5 * delta);
        }
    }

    sum / n
}

/// Trait for loss functions.
///
/// Implement this trait to create custom loss functions compatible with
/// training algorithms.
pub trait Loss {
    /// Computes the loss between predictions and targets.
    ///
    /// # Arguments
    ///
    /// * `y_pred` - Predicted values
    /// * `y_true` - True target values
    ///
    /// # Returns
    ///
    /// The computed loss value
    fn compute(&self, y_pred: &Vector<f32>, y_true: &Vector<f32>) -> f32;

    /// Returns the name of the loss function.
    fn name(&self) -> &str;
}

/// Mean Squared Error loss function (struct wrapper).
#[derive(Debug, Clone, Copy)]
pub struct MSELoss;

impl Loss for MSELoss {
    fn compute(&self, y_pred: &Vector<f32>, y_true: &Vector<f32>) -> f32 {
        mse_loss(y_pred, y_true)
    }

    fn name(&self) -> &'static str {
        "MSE"
    }
}

/// Mean Absolute Error loss function (struct wrapper).
#[derive(Debug, Clone, Copy)]
pub struct MAELoss;

impl Loss for MAELoss {
    fn compute(&self, y_pred: &Vector<f32>, y_true: &Vector<f32>) -> f32 {
        mae_loss(y_pred, y_true)
    }

    fn name(&self) -> &'static str {
        "MAE"
    }
}

/// Huber loss function (struct wrapper).
#[derive(Debug, Clone, Copy)]
pub struct HuberLoss {
    delta: f32,
}

impl HuberLoss {
    /// Creates a new Huber loss with the given delta parameter.
    ///
    /// # Arguments
    ///
    /// * `delta` - Threshold for switching from quadratic to linear
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::loss::{HuberLoss, Loss};
    /// use aprender::primitives::Vector;
    ///
    /// let loss_fn = HuberLoss::new(1.0);
    /// let y_true = Vector::from_slice(&[1.0, 2.0]);
    /// let y_pred = Vector::from_slice(&[1.5, 2.5]);
    ///
    /// let loss = loss_fn.compute(&y_pred, &y_true);
    /// assert!(loss > 0.0);
    /// ```
    #[must_use]
    pub fn new(delta: f32) -> Self {
        Self { delta }
    }

    /// Returns the delta parameter.
    #[must_use]
    pub fn delta(&self) -> f32 {
        self.delta
    }
}

impl Loss for HuberLoss {
    fn compute(&self, y_pred: &Vector<f32>, y_true: &Vector<f32>) -> f32 {
        huber_loss(y_pred, y_true, self.delta)
    }

    fn name(&self) -> &'static str {
        "Huber"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mse_loss_perfect() {
        let y_true = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let y_pred = Vector::from_slice(&[1.0, 2.0, 3.0]);

        let loss = mse_loss(&y_pred, &y_true);
        assert!((loss - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_mse_loss_basic() {
        let y_true = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let y_pred = Vector::from_slice(&[2.0, 3.0, 4.0]);

        // Errors: [1, 1, 1], squared: [1, 1, 1], mean: 1.0
        let loss = mse_loss(&y_pred, &y_true);
        assert!((loss - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_mse_loss_different_errors() {
        let y_true = Vector::from_slice(&[0.0, 0.0, 0.0]);
        let y_pred = Vector::from_slice(&[1.0, 2.0, 3.0]);

        // Errors: [1, 2, 3], squared: [1, 4, 9], mean: 14/3 ≈ 4.667
        let loss = mse_loss(&y_pred, &y_true);
        assert!((loss - 14.0 / 3.0).abs() < 1e-5);
    }

    #[test]
    #[should_panic(expected = "same length")]
    fn test_mse_loss_mismatched_lengths() {
        let y_true = Vector::from_slice(&[1.0, 2.0]);
        let y_pred = Vector::from_slice(&[1.0, 2.0, 3.0]);

        let _ = mse_loss(&y_pred, &y_true);
    }

    #[test]
    fn test_mae_loss_perfect() {
        let y_true = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let y_pred = Vector::from_slice(&[1.0, 2.0, 3.0]);

        let loss = mae_loss(&y_pred, &y_true);
        assert!((loss - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_mae_loss_basic() {
        let y_true = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let y_pred = Vector::from_slice(&[1.5, 2.5, 2.5]);

        // Errors: [0.5, 0.5, -0.5], abs: [0.5, 0.5, 0.5], mean: 0.5
        let loss = mae_loss(&y_pred, &y_true);
        assert!((loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_mae_loss_outlier_robustness() {
        let y_true = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let y_pred = Vector::from_slice(&[2.0, 3.0, 100.0]);

        // MAE: (1 + 1 + 97) / 3 = 33.0
        let mae = mae_loss(&y_pred, &y_true);

        // MSE: (1 + 1 + 9409) / 3 = 3137.0
        let mse = mse_loss(&y_pred, &y_true);

        // MAE should be much less affected by outlier
        assert!(mae < mse / 10.0);
    }

    #[test]
    #[should_panic(expected = "same length")]
    fn test_mae_loss_mismatched_lengths() {
        let y_true = Vector::from_slice(&[1.0, 2.0]);
        let y_pred = Vector::from_slice(&[1.0, 2.0, 3.0]);

        let _ = mae_loss(&y_pred, &y_true);
    }

    #[test]
    fn test_huber_loss_small_errors() {
        let y_true = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let y_pred = Vector::from_slice(&[1.5, 2.5, 3.5]);

        // All errors (0.5) <= delta (1.0), so use quadratic
        // 0.5 * (0.5)² * 3 / 3 = 0.125
        let loss = huber_loss(&y_pred, &y_true, 1.0);
        assert!((loss - 0.125).abs() < 1e-6);
    }

    #[test]
    fn test_huber_loss_large_errors() {
        let y_true = Vector::from_slice(&[0.0, 0.0, 0.0]);
        let y_pred = Vector::from_slice(&[5.0, 5.0, 5.0]);

        // All errors (5.0) > delta (1.0), so use linear
        // 1.0 * (5.0 - 0.5 * 1.0) * 3 / 3 = 4.5
        let loss = huber_loss(&y_pred, &y_true, 1.0);
        assert!((loss - 4.5).abs() < 1e-6);
    }

    #[test]
    fn test_huber_loss_mixed_errors() {
        let y_true = Vector::from_slice(&[0.0, 0.0]);
        let y_pred = Vector::from_slice(&[0.5, 5.0]);

        // First: 0.5 <= 1.0, use 0.5 * 0.5² = 0.125
        // Second: 5.0 > 1.0, use 1.0 * (5.0 - 0.5) = 4.5
        // Mean: (0.125 + 4.5) / 2 = 2.3125
        let loss = huber_loss(&y_pred, &y_true, 1.0);
        assert!((loss - 2.3125).abs() < 1e-5);
    }

    #[test]
    #[should_panic(expected = "Delta must be positive")]
    fn test_huber_loss_zero_delta() {
        let y_true = Vector::from_slice(&[1.0]);
        let y_pred = Vector::from_slice(&[2.0]);

        let _ = huber_loss(&y_pred, &y_true, 0.0);
    }

    #[test]
    #[should_panic(expected = "Delta must be positive")]
    fn test_huber_loss_negative_delta() {
        let y_true = Vector::from_slice(&[1.0]);
        let y_pred = Vector::from_slice(&[2.0]);

        let _ = huber_loss(&y_pred, &y_true, -1.0);
    }

    #[test]
    fn test_huber_vs_mse_small_errors() {
        let y_true = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let y_pred = Vector::from_slice(&[1.1, 2.1, 3.1]);

        let huber = huber_loss(&y_pred, &y_true, 1.0);
        let mse = mse_loss(&y_pred, &y_true);

        // For small errors, Huber should be close to MSE
        assert!((huber - mse).abs() < 0.01);
    }

    #[test]
    fn test_huber_vs_mae_large_errors() {
        let y_true = Vector::from_slice(&[0.0, 0.0, 0.0]);
        let y_pred = Vector::from_slice(&[10.0, 10.0, 10.0]);

        let huber = huber_loss(&y_pred, &y_true, 1.0);
        let mae = mae_loss(&y_pred, &y_true);

        // For large errors, Huber should approximate MAE (with offset)
        assert!(huber < mae);
        assert!((huber - (mae - 0.5)).abs() < 0.1);
    }

    #[test]
    fn test_mse_loss_struct() {
        let loss_fn = MSELoss;
        let y_true = Vector::from_slice(&[1.0, 2.0]);
        let y_pred = Vector::from_slice(&[1.0, 2.0]);

        let loss = loss_fn.compute(&y_pred, &y_true);
        assert!((loss - 0.0).abs() < 1e-6);
        assert_eq!(loss_fn.name(), "MSE");
    }

    #[test]
    fn test_mae_loss_struct() {
        let loss_fn = MAELoss;
        let y_true = Vector::from_slice(&[1.0, 2.0]);
        let y_pred = Vector::from_slice(&[1.5, 2.5]);

        let loss = loss_fn.compute(&y_pred, &y_true);
        assert!((loss - 0.5).abs() < 1e-6);
        assert_eq!(loss_fn.name(), "MAE");
    }

    #[test]
    fn test_huber_loss_struct() {
        let loss_fn = HuberLoss::new(1.0);
        let y_true = Vector::from_slice(&[1.0, 2.0]);
        let y_pred = Vector::from_slice(&[1.5, 2.5]);

        let loss = loss_fn.compute(&y_pred, &y_true);
        assert!(loss > 0.0);
        assert_eq!(loss_fn.name(), "Huber");
        assert!((loss_fn.delta() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_loss_trait_polymorphism() {
        let loss_fns: Vec<Box<dyn Loss>> = vec![
            Box::new(MSELoss),
            Box::new(MAELoss),
            Box::new(HuberLoss::new(1.0)),
        ];

        let y_true = Vector::from_slice(&[1.0, 2.0]);
        let y_pred = Vector::from_slice(&[1.5, 2.5]);

        for loss_fn in loss_fns {
            let loss = loss_fn.compute(&y_pred, &y_true);
            assert!(loss > 0.0);
            assert!(!loss_fn.name().is_empty());
        }
    }

    #[test]
    fn test_negative_values() {
        let y_true = Vector::from_slice(&[-1.0, -2.0, -3.0]);
        let y_pred = Vector::from_slice(&[-1.5, -2.5, -3.5]);

        let mse = mse_loss(&y_pred, &y_true);
        let mae = mae_loss(&y_pred, &y_true);
        let huber = huber_loss(&y_pred, &y_true, 1.0);

        assert!(mse > 0.0);
        assert!(mae > 0.0);
        assert!(huber > 0.0);
    }

    #[test]
    fn test_zero_values() {
        let y_true = Vector::from_slice(&[0.0, 0.0, 0.0]);
        let y_pred = Vector::from_slice(&[0.0, 0.0, 0.0]);

        let mse = mse_loss(&y_pred, &y_true);
        let mae = mae_loss(&y_pred, &y_true);
        let huber = huber_loss(&y_pred, &y_true, 1.0);

        assert!((mse - 0.0).abs() < 1e-6);
        assert!((mae - 0.0).abs() < 1e-6);
        assert!((huber - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_single_value() {
        let y_true = Vector::from_slice(&[5.0]);
        let y_pred = Vector::from_slice(&[3.0]);

        let mse = mse_loss(&y_pred, &y_true);
        let mae = mae_loss(&y_pred, &y_true);
        let huber = huber_loss(&y_pred, &y_true, 1.0);

        assert!((mse - 4.0).abs() < 1e-6);
        assert!((mae - 2.0).abs() < 1e-6);
        assert!(huber > 0.0);
    }
}

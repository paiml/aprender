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
#[must_use]
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
#[must_use]
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
#[must_use]
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

/// Cross-entropy loss with one-hot (soft) targets.
///
/// Computes: L = -sum(y_true * log(softmax(y_pred)))
///
/// Uses numerically stable log-softmax (max subtraction).
///
/// # Arguments
///
/// * `y_pred` - Raw logits (pre-softmax predictions)
/// * `y_true` - Target distribution (one-hot or soft labels, must sum to ~1.0)
///
/// # Returns
///
/// The cross-entropy loss (non-negative scalar)
///
/// # Panics
///
/// Panics if `y_pred` and `y_true` have different lengths or are empty.
///
/// # Example
///
/// ```
/// use aprender::loss::cross_entropy_loss;
/// use aprender::primitives::Vector;
///
/// let logits = Vector::from_slice(&[2.0, 1.0, 0.5]);
/// let targets = Vector::from_slice(&[1.0, 0.0, 0.0]); // one-hot
///
/// let loss = cross_entropy_loss(&logits, &targets);
/// assert!(loss > 0.0);
/// ```
#[must_use]
pub fn cross_entropy_loss(y_pred: &Vector<f32>, y_true: &Vector<f32>) -> f32 {
    assert_eq!(
        y_pred.len(),
        y_true.len(),
        "Predicted and true values must have same length"
    );
    assert!(!y_pred.is_empty(), "Vectors cannot be empty");

    // ONE PATH: Delegates log-softmax computation to nn::functional (UCBD §4).
    let log_probs = crate::nn::functional::log_softmax_1d(y_pred.as_slice());

    // CE = -sum(y_true_i * log_softmax(y_pred_i))
    y_true
        .as_slice()
        .iter()
        .zip(log_probs.iter())
        .filter(|(&y, _)| y > 0.0)
        .map(|(&y, &lp)| -y * lp)
        .sum()
}

/// Trait for loss functions.
///
/// Implement this trait to create custom loss functions compatible with
/// training algorithms.
///
/// The `Send + Sync` bounds enable safe sharing across threads,
/// required for multithreaded training loops (e.g., entrenar).
pub trait Loss: Send + Sync {
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

// =============================================================================
// Contrastive Loss Functions (spec: more-learning-specs.md §3)
// =============================================================================

/// Triplet loss for metric learning.
///
/// Computes loss to ensure anchor is closer to positive than negative:
///
/// ```text
/// L = max(0, d(anchor, positive) - d(anchor, negative) + margin)
/// ```
///
/// # Arguments
///
/// * `anchor` - Anchor embedding
/// * `positive` - Positive example (same class as anchor)
/// * `negative` - Negative example (different class)
/// * `margin` - Minimum distance margin between positive and negative
///
/// # Returns
///
/// The triplet loss value (0 if constraint satisfied)
///
/// # Example
///
/// ```
/// use aprender::loss::triplet_loss;
/// use aprender::primitives::Vector;
///
/// let anchor = Vector::from_slice(&[1.0, 0.0]);
/// let positive = Vector::from_slice(&[0.9, 0.1]);  // close to anchor
/// let negative = Vector::from_slice(&[0.0, 1.0]);  // far from anchor
///
/// let loss = triplet_loss(&anchor, &positive, &negative, 0.2);
/// assert!(loss >= 0.0);
/// ```
#[must_use]
pub fn triplet_loss(
    anchor: &Vector<f32>,
    positive: &Vector<f32>,
    negative: &Vector<f32>,
    margin: f32,
) -> f32 {
    assert_eq!(
        anchor.len(),
        positive.len(),
        "Anchor and positive must have same dimension"
    );
    assert_eq!(
        anchor.len(),
        negative.len(),
        "Anchor and negative must have same dimension"
    );

    let d_pos = euclidean_distance(anchor, positive);
    let d_neg = euclidean_distance(anchor, negative);

    (d_pos - d_neg + margin).max(0.0)
}

/// Euclidean distance between two vectors.
fn euclidean_distance(a: &Vector<f32>, b: &Vector<f32>) -> f32 {
    let mut sum = 0.0;
    for i in 0..a.len() {
        let diff = a[i] - b[i];
        sum += diff * diff;
    }
    sum.sqrt()
}

/// Cosine similarity between two vectors.
fn cosine_similarity(a: &Vector<f32>, b: &Vector<f32>) -> f32 {
    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    let denom = (norm_a.sqrt() * norm_b.sqrt()).max(1e-8);
    dot / denom
}

include!("mod_part_02.rs");
include!("mod_part_03.rs");

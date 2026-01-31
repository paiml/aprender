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

/// `InfoNCE` (Noise Contrastive Estimation) loss for contrastive learning.
///
/// Also known as NT-Xent (Normalized Temperature-scaled Cross Entropy).
///
/// ```text
/// L = -log(exp(sim(z_i, z_j)/τ) / Σ_k exp(sim(z_i, z_k)/τ))
/// ```
///
/// # Arguments
///
/// * `anchor` - Anchor embedding
/// * `positive` - Positive example embedding
/// * `negatives` - Slice of negative example embeddings
/// * `temperature` - Temperature scaling parameter (typically 0.07-0.5)
///
/// # Returns
///
/// The `InfoNCE` loss value
///
/// # Example
///
/// ```
/// use aprender::loss::info_nce_loss;
/// use aprender::primitives::Vector;
///
/// let anchor = Vector::from_slice(&[1.0, 0.0, 0.0]);
/// let positive = Vector::from_slice(&[0.9, 0.1, 0.0]);
/// let negatives = vec![
///     Vector::from_slice(&[0.0, 1.0, 0.0]),
///     Vector::from_slice(&[0.0, 0.0, 1.0]),
/// ];
///
/// let loss = info_nce_loss(&anchor, &positive, &negatives, 0.1);
/// assert!(loss >= 0.0);
/// ```
#[must_use]
pub fn info_nce_loss(
    anchor: &Vector<f32>,
    positive: &Vector<f32>,
    negatives: &[Vector<f32>],
    temperature: f32,
) -> f32 {
    assert_eq!(
        anchor.len(),
        positive.len(),
        "Anchor and positive must have same dimension"
    );
    for neg in negatives {
        assert_eq!(
            anchor.len(),
            neg.len(),
            "All embeddings must have same dimension"
        );
    }
    assert!(temperature > 0.0, "Temperature must be positive");

    // Compute similarity with positive
    let sim_pos = cosine_similarity(anchor, positive) / temperature;

    // Compute log-sum-exp over all (positive + negatives)
    let mut max_sim = sim_pos;
    for neg in negatives {
        let sim_neg = cosine_similarity(anchor, neg) / temperature;
        max_sim = max_sim.max(sim_neg);
    }

    // Numerically stable log-sum-exp
    let mut sum_exp = (sim_pos - max_sim).exp();
    for neg in negatives {
        let sim_neg = cosine_similarity(anchor, neg) / temperature;
        sum_exp += (sim_neg - max_sim).exp();
    }

    // -log(exp(sim_pos) / sum_exp) = -sim_pos + log(sum_exp)
    -sim_pos + max_sim + sum_exp.ln()
}

/// Focal loss for class imbalance (spec: more-learning-specs.md §18).
///
/// Down-weights easy examples, focuses on hard examples:
///
/// ```text
/// FL(p) = -α * (1 - p)^γ * log(p)
/// ```
///
/// # Arguments
///
/// * `predictions` - Predicted probabilities (after sigmoid/softmax)
/// * `targets` - Binary targets (0 or 1)
/// * `alpha` - Balancing factor (typically 0.25 for rare class)
/// * `gamma` - Focusing parameter (typically 2.0)
///
/// # Returns
///
/// The focal loss value
///
/// # Example
///
/// ```
/// use aprender::loss::focal_loss;
/// use aprender::primitives::Vector;
///
/// let predictions = Vector::from_slice(&[0.9, 0.1, 0.8]);
/// let targets = Vector::from_slice(&[1.0, 0.0, 1.0]);
///
/// let loss = focal_loss(&predictions, &targets, 0.25, 2.0);
/// assert!(loss >= 0.0);
/// ```
#[must_use]
pub fn focal_loss(predictions: &Vector<f32>, targets: &Vector<f32>, alpha: f32, gamma: f32) -> f32 {
    assert_eq!(
        predictions.len(),
        targets.len(),
        "Predictions and targets must have same length"
    );

    let n = predictions.len() as f32;
    let mut sum = 0.0;

    for i in 0..predictions.len() {
        let p = predictions[i].clamp(1e-7, 1.0 - 1e-7);
        let t = targets[i];

        // For positive class (t=1): -α * (1-p)^γ * log(p)
        // For negative class (t=0): -(1-α) * p^γ * log(1-p)
        let loss = if t > 0.5 {
            -alpha * (1.0 - p).powf(gamma) * p.ln()
        } else {
            -(1.0 - alpha) * p.powf(gamma) * (1.0 - p).ln()
        };

        sum += loss;
    }

    sum / n
}

/// KL Divergence loss between two probability distributions.
///
/// ```text
/// KL(P || Q) = Σ P(x) * log(P(x) / Q(x))
/// ```
///
/// # Arguments
///
/// * `p` - True distribution (targets)
/// * `q` - Predicted distribution
///
/// # Returns
///
/// The KL divergence (always >= 0)
///
/// # Example
///
/// ```
/// use aprender::loss::kl_divergence;
/// use aprender::primitives::Vector;
///
/// let p = Vector::from_slice(&[0.5, 0.3, 0.2]);
/// let q = Vector::from_slice(&[0.4, 0.4, 0.2]);
///
/// let kl = kl_divergence(&p, &q);
/// assert!(kl >= 0.0);
/// ```
#[must_use]
pub fn kl_divergence(p: &Vector<f32>, q: &Vector<f32>) -> f32 {
    assert_eq!(p.len(), q.len(), "Distributions must have same length");

    let mut sum = 0.0;
    for i in 0..p.len() {
        if p[i] > 1e-10 {
            let q_safe = q[i].max(1e-10);
            sum += p[i] * (p[i] / q_safe).ln();
        }
    }

    sum
}

/// Triplet loss function (struct wrapper).
#[derive(Debug, Clone, Copy)]
pub struct TripletLoss {
    margin: f32,
}

impl TripletLoss {
    /// Creates a new Triplet loss with the given margin.
    #[must_use]
    pub fn new(margin: f32) -> Self {
        Self { margin }
    }

    /// Returns the margin parameter.
    #[must_use]
    pub fn margin(&self) -> f32 {
        self.margin
    }

    /// Compute triplet loss for given embeddings.
    #[must_use]
    pub fn compute_triplet(
        &self,
        anchor: &Vector<f32>,
        positive: &Vector<f32>,
        negative: &Vector<f32>,
    ) -> f32 {
        triplet_loss(anchor, positive, negative, self.margin)
    }
}

/// Focal loss function (struct wrapper).
#[derive(Debug, Clone, Copy)]
pub struct FocalLoss {
    alpha: f32,
    gamma: f32,
}

impl FocalLoss {
    /// Creates a new Focal loss with given parameters.
    ///
    /// # Arguments
    ///
    /// * `alpha` - Balancing factor for rare class (typically 0.25)
    /// * `gamma` - Focusing parameter (typically 2.0)
    #[must_use]
    pub fn new(alpha: f32, gamma: f32) -> Self {
        Self { alpha, gamma }
    }

    /// Returns the alpha parameter.
    #[must_use]
    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    /// Returns the gamma parameter.
    #[must_use]
    pub fn gamma(&self) -> f32 {
        self.gamma
    }
}

impl Loss for FocalLoss {
    fn compute(&self, y_pred: &Vector<f32>, y_true: &Vector<f32>) -> f32 {
        focal_loss(y_pred, y_true, self.alpha, self.gamma)
    }

    fn name(&self) -> &'static str {
        "Focal"
    }
}

/// `InfoNCE` / NT-Xent loss function (struct wrapper).
#[derive(Debug, Clone, Copy)]
pub struct InfoNCELoss {
    temperature: f32,
}

impl InfoNCELoss {
    /// Creates a new `InfoNCE` loss with given temperature.
    ///
    /// # Arguments
    ///
    /// * `temperature` - Temperature scaling (typically 0.07-0.5)
    #[must_use]
    pub fn new(temperature: f32) -> Self {
        Self { temperature }
    }

    /// Returns the temperature parameter.
    #[must_use]
    pub fn temperature(&self) -> f32 {
        self.temperature
    }

    /// Compute `InfoNCE` loss for contrastive learning.
    #[must_use]
    pub fn compute_contrastive(
        &self,
        anchor: &Vector<f32>,
        positive: &Vector<f32>,
        negatives: &[Vector<f32>],
    ) -> f32 {
        info_nce_loss(anchor, positive, negatives, self.temperature)
    }
}

/// Dice loss for segmentation tasks.
///
/// Measures overlap between predicted and ground truth masks:
/// ```text
/// Dice = 2 * |X ∩ Y| / (|X| + |Y|)
/// Loss = 1 - Dice
/// ```
///
/// # Arguments
/// * `y_pred` - Predicted probabilities (0-1)
/// * `y_true` - Ground truth binary mask (0 or 1)
/// * `smooth` - Smoothing factor to avoid division by zero
#[must_use]
pub fn dice_loss(y_pred: &Vector<f32>, y_true: &Vector<f32>, smooth: f32) -> f32 {
    assert_eq!(y_pred.len(), y_true.len());

    let mut intersection = 0.0;
    let mut pred_sum = 0.0;
    let mut true_sum = 0.0;

    for i in 0..y_pred.len() {
        intersection += y_pred[i] * y_true[i];
        pred_sum += y_pred[i];
        true_sum += y_true[i];
    }

    let dice = (2.0 * intersection + smooth) / (pred_sum + true_sum + smooth);
    1.0 - dice
}

/// Hinge loss for SVM-style margin classification.
///
/// ```text
/// L = max(0, margin - y_true * y_pred)
/// ```
///
/// # Arguments
/// * `y_pred` - Predicted scores (raw, not probabilities)
/// * `y_true` - True labels (-1 or 1)
/// * `margin` - Margin threshold (typically 1.0)
#[must_use]
pub fn hinge_loss(y_pred: &Vector<f32>, y_true: &Vector<f32>, margin: f32) -> f32 {
    assert_eq!(y_pred.len(), y_true.len());

    let mut sum = 0.0;
    for i in 0..y_pred.len() {
        let loss = (margin - y_true[i] * y_pred[i]).max(0.0);
        sum += loss;
    }
    sum / y_pred.len() as f32
}

/// Squared hinge loss (smoother gradient).
#[must_use]
pub fn squared_hinge_loss(y_pred: &Vector<f32>, y_true: &Vector<f32>, margin: f32) -> f32 {
    assert_eq!(y_pred.len(), y_true.len());

    let mut sum = 0.0;
    for i in 0..y_pred.len() {
        let loss = (margin - y_true[i] * y_pred[i]).max(0.0);
        sum += loss * loss;
    }
    sum / y_pred.len() as f32
}

/// Dice loss struct wrapper.
#[derive(Debug, Clone, Copy)]
pub struct DiceLoss {
    smooth: f32,
}

impl DiceLoss {
    #[must_use]
    pub fn new(smooth: f32) -> Self {
        Self { smooth }
    }

    #[must_use]
    pub fn smooth(&self) -> f32 {
        self.smooth
    }
}

impl Loss for DiceLoss {
    fn compute(&self, y_pred: &Vector<f32>, y_true: &Vector<f32>) -> f32 {
        dice_loss(y_pred, y_true, self.smooth)
    }

    fn name(&self) -> &'static str {
        "Dice"
    }
}

/// Hinge loss struct wrapper.
#[derive(Debug, Clone, Copy)]
pub struct HingeLoss {
    margin: f32,
}

impl HingeLoss {
    #[must_use]
    pub fn new(margin: f32) -> Self {
        Self { margin }
    }

    #[must_use]
    pub fn margin(&self) -> f32 {
        self.margin
    }
}

impl Loss for HingeLoss {
    fn compute(&self, y_pred: &Vector<f32>, y_true: &Vector<f32>) -> f32 {
        hinge_loss(y_pred, y_true, self.margin)
    }

    fn name(&self) -> &'static str {
        "Hinge"
    }
}

/// Connectionist Temporal Classification (CTC) Loss.
///
/// Used for sequence-to-sequence tasks where alignment is unknown.
/// Common in speech recognition and OCR.
///
/// Reference: Graves et al., "Connectionist Temporal Classification" (2006)
#[derive(Debug, Clone)]
pub struct CTCLoss {
    blank_idx: usize,
}

impl CTCLoss {
    /// Create CTC loss with specified blank token index.
    #[must_use]
    pub fn new(blank_idx: usize) -> Self {
        Self { blank_idx }
    }

    #[must_use]
    pub fn blank_idx(&self) -> usize {
        self.blank_idx
    }

    /// Compute CTC loss using forward-backward algorithm.
    ///
    /// # Arguments
    /// * `log_probs` - Log probabilities [T, C] (time x classes)
    /// * `targets` - Target sequence (class indices, no blanks)
    /// * `input_length` - Length of input sequence
    /// * `target_length` - Length of target sequence
    #[must_use]
    pub fn forward(
        &self,
        log_probs: &[Vec<f32>],
        targets: &[usize],
        input_length: usize,
        target_length: usize,
    ) -> f32 {
        if target_length == 0 || input_length == 0 {
            return 0.0;
        }

        // Create extended labels with blanks: b, l1, b, l2, b, ...
        let extended_len = 2 * target_length + 1;
        let mut labels = vec![self.blank_idx; extended_len];
        for (i, &t) in targets.iter().take(target_length).enumerate() {
            labels[2 * i + 1] = t;
        }

        // Forward pass: alpha[t][s] = P(prefix up to s at time t)
        let neg_inf = f32::NEG_INFINITY;
        let mut alpha = vec![vec![neg_inf; extended_len]; input_length];

        // Initialize
        alpha[0][0] = log_probs[0][labels[0]];
        if extended_len > 1 {
            alpha[0][1] = log_probs[0][labels[1]];
        }

        // Forward recursion
        for t in 1..input_length {
            for s in 0..extended_len {
                let label = labels[s];
                let mut val = alpha[t - 1][s];

                if s > 0 {
                    val = log_sum_exp(val, alpha[t - 1][s - 1]);
                }

                // Skip blank (allow skipping to same label only if not blank and different)
                if s > 1 && label != self.blank_idx && labels[s - 2] != label {
                    val = log_sum_exp(val, alpha[t - 1][s - 2]);
                }

                alpha[t][s] = val + log_probs[t][label];
            }
        }

        // Final probability: sum of last two positions
        let last_t = input_length - 1;
        let last_s = extended_len - 1;
        let total = if extended_len > 1 {
            log_sum_exp(alpha[last_t][last_s], alpha[last_t][last_s - 1])
        } else {
            alpha[last_t][last_s]
        };

        -total // Negative log-likelihood
    }
}

/// Log-sum-exp for numerical stability: log(exp(a) + exp(b))
fn log_sum_exp(a: f32, b: f32) -> f32 {
    if a == f32::NEG_INFINITY {
        b
    } else if b == f32::NEG_INFINITY {
        a
    } else if a > b {
        a + (b - a).exp().ln_1p()
    } else {
        b + (a - b).exp().ln_1p()
    }
}

/// Wasserstein (Earth Mover's) Distance Loss.
///
/// Measures the minimum cost to transform one distribution to another.
/// More stable for GAN training than cross-entropy.
///
/// For 1D sorted distributions: W1 = Σ|CDF1 - CDF2|
///
/// Reference: Arjovsky et al., "Wasserstein GAN" (2017)
#[must_use]
pub fn wasserstein_loss(real_scores: &Vector<f32>, fake_scores: &Vector<f32>) -> f32 {
    let real_mean: f32 = real_scores.as_slice().iter().sum::<f32>() / real_scores.len() as f32;
    let fake_mean: f32 = fake_scores.as_slice().iter().sum::<f32>() / fake_scores.len() as f32;
    fake_mean - real_mean
}

/// Wasserstein loss for discriminator (critic).
/// Maximizes distance between real and fake.
#[must_use]
pub fn wasserstein_discriminator_loss(real_scores: &Vector<f32>, fake_scores: &Vector<f32>) -> f32 {
    -wasserstein_loss(real_scores, fake_scores)
}

/// Wasserstein loss for generator.
/// Minimizes negative fake score.
#[must_use]
pub fn wasserstein_generator_loss(fake_scores: &Vector<f32>) -> f32 {
    -fake_scores.as_slice().iter().sum::<f32>() / fake_scores.len() as f32
}

/// Gradient penalty for WGAN-GP.
/// Enforces Lipschitz constraint via gradient norm penalty.
#[must_use]
pub fn gradient_penalty(gradients: &[f32], lambda: f32) -> f32 {
    let grad_norm: f32 = gradients.iter().map(|&g| g * g).sum::<f32>().sqrt();
    lambda * (grad_norm - 1.0).powi(2)
}

/// Wasserstein Loss struct wrapper.
#[derive(Debug, Clone, Copy)]
pub struct WassersteinLoss {
    lambda_gp: f32,
}

impl WassersteinLoss {
    #[must_use]
    pub fn new(lambda_gp: f32) -> Self {
        Self { lambda_gp }
    }

    #[must_use]
    pub fn lambda_gp(&self) -> f32 {
        self.lambda_gp
    }

    #[must_use]
    pub fn discriminator_loss(&self, real: &Vector<f32>, fake: &Vector<f32>) -> f32 {
        wasserstein_discriminator_loss(real, fake)
    }

    #[must_use]
    pub fn generator_loss(&self, fake: &Vector<f32>) -> f32 {
        wasserstein_generator_loss(fake)
    }
}

impl Loss for WassersteinLoss {
    fn compute(&self, y_pred: &Vector<f32>, y_true: &Vector<f32>) -> f32 {
        wasserstein_loss(y_pred, y_true)
    }

    fn name(&self) -> &'static str {
        "Wasserstein"
    }
}


#[cfg(test)]
mod tests;

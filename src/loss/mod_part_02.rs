
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

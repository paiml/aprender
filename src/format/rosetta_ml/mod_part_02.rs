
/// Jidoka stop conditions detected in tensor features
#[derive(Debug, Clone)]
pub enum JidokaViolation {
    /// NaN values detected
    NaN { count: usize },
    /// Infinite values detected
    Inf { count: usize },
    /// Zero variance with non-zero mean (degenerate tensor)
    ZeroVariance { mean: f32 },
    /// Shape mismatch between source and target
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    /// Checksum verification failed
    ChecksumFailed { expected: u32, actual: u32 },
}

impl fmt::Display for JidokaViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NaN { count } => write!(f, "NaN values detected: {count}"),
            Self::Inf { count } => write!(f, "Infinite values detected: {count}"),
            Self::ZeroVariance { mean } => write!(f, "Zero variance with mean={mean}"),
            Self::ShapeMismatch { expected, actual } => {
                write!(f, "Shape mismatch: expected {expected:?}, got {actual:?}")
            }
            Self::ChecksumFailed { expected, actual } => {
                write!(
                    f,
                    "Checksum failed: expected {expected:#x}, got {actual:#x}"
                )
            }
        }
    }
}

// ============================================================================
// Anomaly Detection with Regularized Mahalanobis Distance
// ============================================================================

/// Anomaly detector using regularized Mahalanobis distance.
///
/// Per Dr. Popper's audit: "The covariance matrix Σ may be singular or
/// unstable. Use a regularized covariance estimator."
///
/// We use Ledoit-Wolf shrinkage: Σ_reg = (1-α)Σ + α·trace(Σ)/p·I
///
/// # Falsification Criterion (F-ANOM-001)
///
/// "If anomaly detector has >5% false positive rate on known-good
/// conversions, the threshold is falsified and must be recalibrated."
#[derive(Debug, Clone)]
pub struct AnomalyDetector {
    /// Mean feature vector from training corpus
    mean: Vec<f32>,
    /// Inverse covariance matrix (regularized)
    inv_cov: Vec<Vec<f32>>,
    /// Shrinkage parameter (Ledoit-Wolf)
    shrinkage: f32,
    /// Number of training samples
    n_samples: usize,
    /// Anomaly threshold (Mahalanobis distance²)
    threshold: f32,
}

impl AnomalyDetector {
    /// Create new detector from training data.
    ///
    /// Uses Ledoit-Wolf shrinkage for regularization.
    ///
    /// # Arguments
    ///
    /// * `training_data` - Feature vectors from known-good conversions
    /// * `shrinkage` - Shrinkage parameter (0.0 = no regularization, 1.0 = identity)
    /// * `threshold` - Mahalanobis distance² threshold for anomaly
    #[must_use]
    pub fn fit(training_data: &[TensorFeatures], shrinkage: f32, threshold: f32) -> Option<Self> {
        if training_data.len() < 13 {
            // Need n > p for covariance estimation
            return None;
        }

        let n = training_data.len();
        let p = 12; // Feature dimension

        let mean = compute_feature_mean(training_data, p, n);
        let mut cov = compute_covariance(training_data, &mean, p, n);
        apply_ledoit_wolf_shrinkage(&mut cov, p, shrinkage);

        // Invert covariance matrix (simple Gauss-Jordan for 12x12)
        let inv_cov = invert_matrix(&cov)?;

        Some(Self {
            mean,
            inv_cov,
            shrinkage,
            n_samples: n,
            threshold,
        })
    }

    /// Compute Mahalanobis distance² for a tensor
    #[must_use]
    pub fn mahalanobis_distance_sq(&self, features: &TensorFeatures) -> f32 {
        let x = features.to_vec();
        let p = x.len();

        // d² = (x - μ)ᵀ Σ⁻¹ (x - μ)
        let mut diff = vec![0.0f32; p];
        for i in 0..p {
            diff[i] = x[i] - self.mean[i];
        }

        let mut result = 0.0f32;
        for i in 0..p {
            for j in 0..p {
                result += diff[i] * self.inv_cov[i][j] * diff[j];
            }
        }

        result
    }

    /// Check if tensor is anomalous
    #[must_use]
    pub fn is_anomaly(&self, features: &TensorFeatures) -> bool {
        self.mahalanobis_distance_sq(features) > self.threshold
    }

    /// Get anomaly score (0 = normal, 1 = at threshold, >1 = anomaly)
    #[must_use]
    pub fn anomaly_score(&self, features: &TensorFeatures) -> f32 {
        self.mahalanobis_distance_sq(features) / self.threshold
    }

    /// Get shrinkage parameter used for regularization
    #[must_use]
    pub fn shrinkage(&self) -> f32 {
        self.shrinkage
    }

    /// Get number of training samples used to fit the detector
    #[must_use]
    pub fn n_samples(&self) -> usize {
        self.n_samples
    }

    /// Get the anomaly threshold (Mahalanobis distance²)
    #[must_use]
    pub fn threshold(&self) -> f32 {
        self.threshold
    }
}

/// Compute mean feature vector from training data.
fn compute_feature_mean(training_data: &[TensorFeatures], p: usize, n: usize) -> Vec<f32> {
    let mut mean = vec![0.0f32; p];
    for features in training_data {
        let vec = features.to_vec();
        for (i, &v) in vec.iter().enumerate() {
            mean[i] += v;
        }
    }
    for m in &mut mean {
        *m /= n as f32;
    }
    mean
}

/// Compute covariance matrix from training data and mean vector.
fn compute_covariance(
    training_data: &[TensorFeatures],
    mean: &[f32],
    p: usize,
    n: usize,
) -> Vec<Vec<f32>> {
    let mut cov = vec![vec![0.0f32; p]; p];
    for features in training_data {
        let vec = features.to_vec();
        for i in 0..p {
            for j in 0..p {
                cov[i][j] += (vec[i] - mean[i]) * (vec[j] - mean[j]);
            }
        }
    }
    for row in &mut cov {
        for c in row {
            *c /= (n - 1) as f32;
        }
    }
    cov
}

/// Apply Ledoit-Wolf shrinkage: Sigma_reg = (1-alpha)*Sigma + alpha*trace(Sigma)/p*I
fn apply_ledoit_wolf_shrinkage(cov: &mut [Vec<f32>], p: usize, shrinkage: f32) {
    let trace: f32 = (0..p).map(|i| cov[i][i]).sum();
    let shrink_target = trace / p as f32;

    for i in 0..p {
        for j in 0..p {
            cov[i][j] *= 1.0 - shrinkage;
            if i == j {
                cov[i][j] += shrinkage * shrink_target;
            }
        }
    }
}

/// Simple matrix inversion using Gauss-Jordan elimination
fn invert_matrix(matrix: &[Vec<f32>]) -> Option<Vec<Vec<f32>>> {
    let n = matrix.len();
    if n == 0 || matrix[0].len() != n {
        return None;
    }

    let mut aug = build_augmented_matrix(matrix, n);

    for i in 0..n {
        partial_pivot(&mut aug, i);

        let pivot = aug[i][i];
        if pivot.abs() < 1e-10 {
            return None; // Singular matrix
        }

        scale_row(&mut aug[i], pivot);
        eliminate_column(&mut aug, i, n);
    }

    Some(extract_inverse(&aug, n))
}

/// Build augmented matrix [A | I] for Gauss-Jordan elimination.
fn build_augmented_matrix(matrix: &[Vec<f32>], n: usize) -> Vec<Vec<f32>> {
    let mut aug = vec![vec![0.0f32; 2 * n]; n];
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = matrix[i][j];
        }
        aug[i][n + i] = 1.0;
    }
    aug
}

/// Find the row with the largest absolute value in the given column
/// at or below the diagonal, then swap it into position.
fn partial_pivot(aug: &mut [Vec<f32>], col: usize) {
    let mut max_row = col;
    for k in (col + 1)..aug.len() {
        if aug[k][col].abs() > aug[max_row][col].abs() {
            max_row = k;
        }
    }
    aug.swap(col, max_row);
}

/// Scale a row by dividing all elements by the pivot value.
fn scale_row(row: &mut [f32], pivot: f32) {
    for val in row.iter_mut() {
        *val /= pivot;
    }
}

/// Eliminate all entries in the given column except the pivot row.
fn eliminate_column(aug: &mut [Vec<f32>], col: usize, n: usize) {
    for k in 0..n {
        if k == col {
            continue;
        }
        let factor = aug[k][col];
        for j in 0..(2 * n) {
            aug[k][j] -= factor * aug[col][j];
        }
    }
}

/// Extract the inverse matrix from the right half of the augmented matrix.
fn extract_inverse(aug: &[Vec<f32>], n: usize) -> Vec<Vec<f32>> {
    let mut inverse = vec![vec![0.0f32; n]; n];
    for i in 0..n {
        for j in 0..n {
            inverse[i][j] = aug[i][n + j];
        }
    }
    inverse
}

// ============================================================================
// Wilson Score Confidence Intervals
// ============================================================================

/// Wilson score confidence interval for binomial proportions.
///
/// More accurate than normal approximation for small samples.
/// From Wilson (1927).
///
/// # Falsification Criterion (F-CONF-001)
///
/// "If 95% confidence interval excludes the true population rate in
/// >5% of samples, the confidence calculation is falsified."
#[derive(Debug, Clone, Copy)]
pub struct WilsonScore {
    /// Observed proportion
    pub proportion: f32,
    /// Lower bound of confidence interval
    pub lower: f32,
    /// Upper bound of confidence interval
    pub upper: f32,
    /// Sample size
    pub n: usize,
    /// Confidence level (e.g., 0.95 for 95%)
    pub confidence: f32,
}

impl WilsonScore {
    /// Calculate Wilson score interval.
    ///
    /// # Arguments
    ///
    /// * `successes` - Number of successful outcomes
    /// * `total` - Total number of trials
    /// * `confidence` - Confidence level (e.g., 0.95)
    #[must_use]
    pub fn calculate(successes: usize, total: usize, confidence: f32) -> Self {
        if total == 0 {
            return Self {
                proportion: 0.0,
                lower: 0.0,
                upper: 0.0,
                n: 0,
                confidence,
            };
        }

        let n = total as f32;
        let p = successes as f32 / n;

        // Z-score for confidence level (approximation)
        let z = match confidence {
            c if c >= 0.99 => 2.576,
            c if c >= 0.95 => 1.96,
            c if c >= 0.90 => 1.645,
            _ => 1.96,
        };

        let z2 = z * z;
        let denominator = 1.0 + z2 / n;
        let center = p + z2 / (2.0 * n);
        let margin = z * (p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt();

        let lower = ((center - margin) / denominator).max(0.0);
        let upper = ((center + margin) / denominator).min(1.0);

        Self {
            proportion: p,
            lower,
            upper,
            n: total,
            confidence,
        }
    }

    /// Get Andon alert level based on target.
    ///
    /// - Green: proportion >= target
    /// - Yellow: proportion in [0.5*target, target)
    /// - Red: proportion < 0.5*target
    #[must_use]
    pub fn andon_level(&self, target: f32) -> AndonLevel {
        if self.proportion >= target {
            AndonLevel::Green
        } else if self.proportion >= 0.5 * target {
            AndonLevel::Yellow
        } else {
            AndonLevel::Red
        }
    }
}

/// Andon alert levels (Toyota Production System)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AndonLevel {
    /// All good, target met
    Green,
    /// Warning, below target but above 50% of target
    Yellow,
    /// Critical, below 50% of target
    Red,
}

impl fmt::Display for AndonLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Green => write!(f, "GREEN"),
            Self::Yellow => write!(f, "YELLOW"),
            Self::Red => write!(f, "RED"),
        }
    }
}

// ============================================================================
// Error Pattern Library with Success Tracking
// ============================================================================

/// Fix action for conversion errors
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FixAction {
    /// Swap dimensions (row-major ↔ column-major)
    SwapDimensions,
    /// Requantize with different block size
    Requantize { block_size: usize },
    /// Recompute header checksums
    RecomputeChecksum,
    /// Pad tensor to alignment
    PadAlignment { alignment: usize },
    /// Skip problematic tensor
    SkipTensor,
    /// Fallback to F32 precision
    FallbackF32,
    /// Custom fix with description
    Custom { description: String },
}

/// Error pattern with learned success rate
#[derive(Debug, Clone)]
pub struct ErrorPattern {
    /// Pattern identifier
    pub id: String,
    /// Keywords that match this error
    pub keywords: Vec<String>,
    /// Suggested fix action
    pub fix: FixAction,
    /// Number of times pattern was applied
    pub applications: usize,
    /// Number of successful fixes
    pub successes: usize,
    /// Source of pattern (bootstrap, llm, corpus)
    pub source: PatternSource,
}

/// Source of error pattern
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PatternSource {
    /// Bootstrapped from golden traces
    Bootstrap,
    /// Learned from corpus of successful conversions
    Corpus,
    /// Suggested by LLM
    Llm,
    /// Manually entered by developer
    Manual,
}

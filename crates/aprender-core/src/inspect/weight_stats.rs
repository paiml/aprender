
impl WeightStats {
    /// Create weight stats from a slice of values
    #[must_use]
    pub fn from_slice(weights: &[f32]) -> Self {
        if weights.is_empty() {
            return Self::empty();
        }

        let count = weights.len() as u64;
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        let mut sum = 0.0_f64;
        let mut zero_count = 0_u64;
        let mut nan_count = 0_u64;
        let mut inf_count = 0_u64;
        let mut l1_sum = 0.0_f64;
        let mut l2_sum = 0.0_f64;

        for &w in weights {
            let w = f64::from(w);
            if w.is_nan() {
                nan_count += 1;
                continue;
            }
            if w.is_infinite() {
                inf_count += 1;
                continue;
            }
            if w == 0.0 {
                zero_count += 1;
            }
            min = min.min(w);
            max = max.max(w);
            sum += w;
            l1_sum += w.abs();
            l2_sum += w * w;
        }

        let valid_count = count - nan_count - inf_count;
        let mean = if valid_count > 0 {
            sum / valid_count as f64
        } else {
            0.0
        };

        // Calculate standard deviation
        let mut variance_sum = 0.0_f64;
        for &w in weights {
            let w = f64::from(w);
            if !w.is_nan() && !w.is_infinite() {
                variance_sum += (w - mean).powi(2);
            }
        }
        let std = if valid_count > 1 {
            (variance_sum / (valid_count - 1) as f64).sqrt()
        } else {
            0.0
        };

        let sparsity = zero_count as f64 / count as f64;

        Self {
            count,
            min: if min == f64::INFINITY { 0.0 } else { min },
            max: if max == f64::NEG_INFINITY { 0.0 } else { max },
            mean,
            std,
            zero_count,
            nan_count,
            inf_count,
            sparsity,
            l1_norm: l1_sum,
            l2_norm: l2_sum.sqrt(),
        }
    }

    /// Create empty weight stats
    #[must_use]
    pub fn empty() -> Self {
        Self {
            count: 0,
            min: 0.0,
            max: 0.0,
            mean: 0.0,
            std: 0.0,
            zero_count: 0,
            nan_count: 0,
            inf_count: 0,
            sparsity: 0.0,
            l1_norm: 0.0,
            l2_norm: 0.0,
        }
    }

    /// Check if weights have issues (NaN or Inf)
    #[must_use]
    pub fn has_issues(&self) -> bool {
        self.nan_count > 0 || self.inf_count > 0
    }

    /// Get health status
    #[must_use]
    pub fn health_status(&self) -> WeightHealth {
        if self.nan_count > 0 || self.inf_count > 0 {
            WeightHealth::Critical
        } else if self.sparsity > 0.99 || self.std < 1e-10 {
            WeightHealth::Warning
        } else {
            WeightHealth::Healthy
        }
    }
}

/// Weight health status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightHealth {
    /// Weights are healthy
    Healthy,
    /// Weights have potential issues
    Warning,
    /// Weights have critical issues
    Critical,
}

impl WeightHealth {
    /// Get description
    #[must_use]
    pub const fn description(&self) -> &'static str {
        match self {
            Self::Healthy => "Weights are within normal parameters",
            Self::Warning => "Weights may have potential issues (high sparsity or low variance)",
            Self::Critical => "Weights have critical issues (NaN or Inf values)",
        }
    }
}

/// Inspection warning
#[derive(Debug, Clone)]
pub struct InspectionWarning {
    /// Warning code
    pub code: String,
    /// Warning message
    pub message: String,
    /// Recommendation
    pub recommendation: Option<String>,
}

impl InspectionWarning {
    /// Create a new warning
    #[must_use]
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
            recommendation: None,
        }
    }

    /// Add recommendation
    #[must_use]
    pub fn with_recommendation(mut self, recommendation: impl Into<String>) -> Self {
        self.recommendation = Some(recommendation.into());
        self
    }
}

impl fmt::Display for InspectionWarning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.code, self.message)?;
        if let Some(rec) = &self.recommendation {
            write!(f, " (Recommendation: {rec})")?;
        }
        Ok(())
    }
}

/// Inspection error
#[derive(Debug, Clone)]
pub struct InspectionError {
    /// Error code
    pub code: String,
    /// Error message
    pub message: String,
    /// Whether error is fatal
    pub fatal: bool,
}

impl InspectionError {
    /// Create a new error
    #[must_use]
    pub fn new(code: impl Into<String>, message: impl Into<String>, fatal: bool) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
            fatal,
        }
    }
}

impl fmt::Display for InspectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let severity = if self.fatal { "FATAL" } else { "ERROR" };
        write!(f, "[{} {}] {}", severity, self.code, self.message)
    }
}

/// Model diff result
#[derive(Debug, Clone)]
pub struct DiffResult {
    /// First model info
    pub model_a: String,
    /// Second model info
    pub model_b: String,
    /// Header differences
    pub header_diff: Vec<DiffItem>,
    /// Metadata differences
    pub metadata_diff: Vec<DiffItem>,
    /// Weight differences
    pub weight_diff: Option<WeightDiff>,
    /// Overall similarity (0.0 - 1.0)
    pub similarity: f64,
}

impl DiffResult {
    /// Create a new diff result
    #[must_use]
    pub fn new(model_a: impl Into<String>, model_b: impl Into<String>) -> Self {
        Self {
            model_a: model_a.into(),
            model_b: model_b.into(),
            header_diff: Vec::new(),
            metadata_diff: Vec::new(),
            weight_diff: None,
            similarity: 1.0,
        }
    }

    /// Check if models are identical
    #[must_use]
    pub fn is_identical(&self) -> bool {
        self.header_diff.is_empty()
            && self.metadata_diff.is_empty()
            && self
                .weight_diff
                .as_ref()
                .map_or(true, WeightDiff::is_identical)
    }

    /// Get total difference count
    #[must_use]
    pub fn diff_count(&self) -> usize {
        let weight_count = self.weight_diff.as_ref().map_or(0, WeightDiff::diff_count);
        self.header_diff.len() + self.metadata_diff.len() + weight_count
    }
}

/// Diff item for scalar values
#[derive(Debug, Clone)]
pub struct DiffItem {
    /// Field name
    pub field: String,
    /// Value in model A
    pub value_a: String,
    /// Value in model B
    pub value_b: String,
}

impl DiffItem {
    /// Create a new diff item
    #[must_use]
    pub fn new(
        field: impl Into<String>,
        value_a: impl Into<String>,
        value_b: impl Into<String>,
    ) -> Self {
        Self {
            field: field.into(),
            value_a: value_a.into(),
            value_b: value_b.into(),
        }
    }
}

impl fmt::Display for DiffItem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {} -> {}", self.field, self.value_a, self.value_b)
    }
}

/// Weight difference statistics
#[derive(Debug, Clone)]
pub struct WeightDiff {
    /// Number of weights that differ
    pub changed_count: u64,
    /// Maximum absolute difference
    pub max_diff: f64,
    /// Mean absolute difference
    pub mean_diff: f64,
    /// L2 distance between weight vectors
    pub l2_distance: f64,
    /// Cosine similarity
    pub cosine_similarity: f64,
}

impl WeightDiff {
    /// Create empty weight diff
    #[must_use]
    pub fn empty() -> Self {
        Self {
            changed_count: 0,
            max_diff: 0.0,
            mean_diff: 0.0,
            l2_distance: 0.0,
            cosine_similarity: 1.0,
        }
    }

    /// Create from two weight slices
    #[must_use]
    pub fn from_slices(a: &[f32], b: &[f32]) -> Self {
        if a.len() != b.len() || a.is_empty() {
            return Self::empty();
        }

        let mut changed_count = 0_u64;
        let mut max_diff = 0.0_f64;
        let mut diff_sum = 0.0_f64;
        let mut l2_sum = 0.0_f64;
        let mut dot_product = 0.0_f64;
        let mut norm_a = 0.0_f64;
        let mut norm_b = 0.0_f64;

        for (&va, &vb) in a.iter().zip(b.iter()) {
            let va = f64::from(va);
            let vb = f64::from(vb);
            let diff = (va - vb).abs();

            if diff > 1e-10 {
                changed_count += 1;
            }
            max_diff = max_diff.max(diff);
            diff_sum += diff;
            l2_sum += diff * diff;
            dot_product += va * vb;
            norm_a += va * va;
            norm_b += vb * vb;
        }

        let count = a.len() as f64;
        let mean_diff = diff_sum / count;
        let l2_distance = l2_sum.sqrt();

        let cosine_similarity = if norm_a > 0.0 && norm_b > 0.0 {
            dot_product / (norm_a.sqrt() * norm_b.sqrt())
        } else {
            1.0
        };

        Self {
            changed_count,
            max_diff,
            mean_diff,
            l2_distance,
            cosine_similarity,
        }
    }

    /// Check if weights are identical
    #[must_use]
    pub fn is_identical(&self) -> bool {
        self.changed_count == 0
    }

    /// Get diff count (treat any changes as a single diff)
    #[must_use]
    pub fn diff_count(&self) -> usize {
        usize::from(self.changed_count > 0)
    }
}

/// Inspection options
#[derive(Debug, Clone)]
pub struct InspectOptions {
    /// Include weight statistics
    pub include_weights: bool,
    /// Include quality scoring
    pub include_quality: bool,
    /// Maximum weights to analyze (for large models)
    pub max_weights: usize,
    /// Verbose output
    pub verbose: bool,
}

impl Default for InspectOptions {
    fn default() -> Self {
        Self {
            include_weights: true,
            include_quality: true,
            max_weights: 10_000_000, // 10M weights
            verbose: false,
        }
    }
}

impl InspectOptions {
    /// Create new inspection options
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Quick inspection (header and metadata only)
    #[must_use]
    pub fn quick() -> Self {
        Self {
            include_weights: false,
            include_quality: false,
            max_weights: 0,
            verbose: false,
        }
    }

    /// Full inspection with all analysis
    #[must_use]
    pub fn full() -> Self {
        Self {
            include_weights: true,
            include_quality: true,
            max_weights: usize::MAX,
            verbose: true,
        }
    }
}

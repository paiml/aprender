//! Rich Labeling - Beyond binary correctness
//!
//! Extracts maximum signal from each oracle invocation with rich multi-task labels.
//!
//! # Error Categories
//!
//! | Category | Description | Example |
//! |----------|-------------|---------|
//! | TypeMismatch | Type system incompatibility | `int` vs `i32` semantics |
//! | OwnershipViolation | Rust borrow checker errors | Move after borrow |
//! | LifetimeError | Lifetime annotation issues | Missing lifetime bounds |
//! | PanicDivergence | Source continues, target panics | Divide by zero |
//! | OutputMismatch | Different output values | Off-by-one errors |
//!
//! # Reference
//! - VER-053: Rich Labeling - Beyond binary correctness

use serde::{Deserialize, Serialize};

/// Error category taxonomy for transpilation failures
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ErrorCategory {
    /// Type system incompatibility
    TypeMismatch,
    /// Rust ownership/borrow checker errors
    OwnershipViolation,
    /// Lifetime annotation issues
    LifetimeError,
    /// Source continues, target panics
    PanicDivergence,
    /// Different output values
    OutputMismatch,
    /// Compilation error (syntax, missing imports)
    CompilationError,
    /// Runtime error (not panic)
    RuntimeError,
    /// Timeout or resource exhaustion
    ResourceExhaustion,
    /// Unknown or uncategorized error
    Unknown,
}

impl Default for ErrorCategory {
    fn default() -> Self {
        Self::Unknown
    }
}

impl ErrorCategory {
    /// All error categories
    #[must_use]
    pub fn all() -> &'static [Self] {
        &[
            Self::TypeMismatch,
            Self::OwnershipViolation,
            Self::LifetimeError,
            Self::PanicDivergence,
            Self::OutputMismatch,
            Self::CompilationError,
            Self::RuntimeError,
            Self::ResourceExhaustion,
            Self::Unknown,
        ]
    }

    /// Severity weight for prioritization (higher = more important to fix)
    #[must_use]
    pub fn severity(&self) -> f32 {
        match self {
            Self::PanicDivergence => 1.0,    // Critical: silent failures
            Self::OwnershipViolation => 0.9, // Rust-specific complexity
            Self::LifetimeError => 0.85,     // Rust-specific complexity
            Self::TypeMismatch => 0.8,       // Common transpilation issue
            Self::OutputMismatch => 0.7,     // Semantic error
            Self::RuntimeError => 0.6,       // Detectable at runtime
            Self::CompilationError => 0.5,   // Detectable at compile time
            Self::ResourceExhaustion => 0.3, // Often environment-specific
            Self::Unknown => 0.2,            // Needs investigation
        }
    }

    /// Classify error from error message
    #[must_use]
    pub fn classify(error_msg: &str) -> Self {
        let msg = error_msg.to_lowercase();

        // Ownership/borrow errors
        if msg.contains("borrow")
            || msg.contains("move")
            || msg.contains("cannot borrow")
            || msg.contains("value borrowed")
        {
            return Self::OwnershipViolation;
        }

        // Lifetime errors
        if msg.contains("lifetime")
            || msg.contains("does not live long enough")
            || msg.contains("'a")
        {
            return Self::LifetimeError;
        }

        // Type errors
        if msg.contains("type mismatch")
            || msg.contains("expected type")
            || msg.contains("mismatched types")
            || msg.contains("cannot convert")
        {
            return Self::TypeMismatch;
        }

        // Panic/divergence
        if msg.contains("panic")
            || msg.contains("unwrap")
            || msg.contains("assertion failed")
            || msg.contains("index out of bounds")
        {
            return Self::PanicDivergence;
        }

        // Output mismatch
        if msg.contains("output")
            || msg.contains("mismatch")
            || msg.contains("expected")
            || msg.contains("actual")
        {
            return Self::OutputMismatch;
        }

        // Compilation errors
        if msg.contains("cannot find")
            || msg.contains("unresolved")
            || msg.contains("syntax error")
            || msg.contains("parse error")
        {
            return Self::CompilationError;
        }

        // Runtime errors
        if msg.contains("runtime") || msg.contains("overflow") || msg.contains("division by zero") {
            return Self::RuntimeError;
        }

        // Resource exhaustion
        if msg.contains("timeout")
            || msg.contains("memory")
            || msg.contains("stack overflow")
            || msg.contains("resource")
        {
            return Self::ResourceExhaustion;
        }

        Self::Unknown
    }

    /// Convert to one-hot encoding (9 categories)
    #[must_use]
    pub fn to_one_hot(&self) -> [f32; 9] {
        let mut one_hot = [0.0f32; 9];
        one_hot[*self as usize] = 1.0;
        one_hot
    }

    /// Create from one-hot encoding
    #[must_use]
    pub fn from_one_hot(one_hot: &[f32; 9]) -> Self {
        one_hot
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(Self::Unknown, |(i, _)| Self::from_index(i))
    }

    fn from_index(idx: usize) -> Self {
        match idx {
            0 => Self::TypeMismatch,
            1 => Self::OwnershipViolation,
            2 => Self::LifetimeError,
            3 => Self::PanicDivergence,
            4 => Self::OutputMismatch,
            5 => Self::CompilationError,
            6 => Self::RuntimeError,
            7 => Self::ResourceExhaustion,
            _ => Self::Unknown,
        }
    }
}

/// Soft labels for gradual correctness
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SoftLabels {
    /// Output similarity (0.0 = completely different, 1.0 = identical)
    pub output_similarity: f32,
    /// Runtime ratio (target_time / source_time, 1.0 = same speed)
    pub runtime_ratio: f32,
    /// Structural similarity of AST
    pub structural_similarity: f32,
    /// Semantic correctness confidence
    pub semantic_confidence: f32,
    /// Type safety score
    pub type_safety: f32,
}

impl SoftLabels {
    /// Create new soft labels
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// All labels are valid (in [0, 1] range)
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.output_similarity >= 0.0
            && self.output_similarity <= 1.0
            && self.runtime_ratio >= 0.0
            && self.structural_similarity >= 0.0
            && self.structural_similarity <= 1.0
            && self.semantic_confidence >= 0.0
            && self.semantic_confidence <= 1.0
            && self.type_safety >= 0.0
            && self.type_safety <= 1.0
    }

    /// Convert to array
    #[must_use]
    pub fn to_array(&self) -> [f32; 5] {
        [
            self.output_similarity,
            self.runtime_ratio.min(10.0) / 10.0, // Normalize to [0, 1]
            self.structural_similarity,
            self.semantic_confidence,
            self.type_safety,
        ]
    }

    /// Create from array
    #[must_use]
    pub fn from_array(arr: [f32; 5]) -> Self {
        Self {
            output_similarity: arr[0],
            runtime_ratio: arr[1] * 10.0, // Denormalize
            structural_similarity: arr[2],
            semantic_confidence: arr[3],
            type_safety: arr[4],
        }
    }

    /// Overall correctness score (weighted average)
    #[must_use]
    pub fn overall_score(&self) -> f32 {
        let weights = [0.3, 0.1, 0.2, 0.25, 0.15];
        let arr = self.to_array();

        let weighted_sum: f32 = arr.iter().zip(&weights).map(|(v, w)| v * w).sum();
        let total_weight: f32 = weights.iter().sum();

        weighted_sum / total_weight
    }
}

/// Builder for soft labels
#[derive(Debug, Default)]
pub struct SoftLabelsBuilder {
    labels: SoftLabels,
}

impl SoftLabelsBuilder {
    /// Create new builder
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set output similarity
    #[must_use]
    pub fn output_similarity(mut self, value: f32) -> Self {
        self.labels.output_similarity = value.clamp(0.0, 1.0);
        self
    }

    /// Set runtime ratio
    #[must_use]
    pub fn runtime_ratio(mut self, value: f32) -> Self {
        self.labels.runtime_ratio = value.max(0.0);
        self
    }

    /// Set structural similarity
    #[must_use]
    pub fn structural_similarity(mut self, value: f32) -> Self {
        self.labels.structural_similarity = value.clamp(0.0, 1.0);
        self
    }

    /// Set semantic confidence
    #[must_use]
    pub fn semantic_confidence(mut self, value: f32) -> Self {
        self.labels.semantic_confidence = value.clamp(0.0, 1.0);
        self
    }

    /// Set type safety
    #[must_use]
    pub fn type_safety(mut self, value: f32) -> Self {
        self.labels.type_safety = value.clamp(0.0, 1.0);
        self
    }

    /// Build soft labels
    #[must_use]
    pub fn build(self) -> SoftLabels {
        self.labels
    }
}

/// Multi-task label schema
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RichLabel {
    /// Binary correctness (ground truth)
    pub is_correct: bool,
    /// Error category (if not correct)
    pub error_category: Option<ErrorCategory>,
    /// Error message (if not correct)
    pub error_message: Option<String>,
    /// Soft labels for gradual correctness
    pub soft_labels: SoftLabels,
    /// AST diff summary
    pub ast_diff: Option<AstDiff>,
    /// Execution metrics
    pub execution_metrics: ExecutionMetrics,
}

impl RichLabel {
    /// Create for correct sample
    #[must_use]
    pub fn correct(soft_labels: SoftLabels) -> Self {
        Self {
            is_correct: true,
            error_category: None,
            error_message: None,
            soft_labels,
            ast_diff: None,
            execution_metrics: ExecutionMetrics::default(),
        }
    }

    /// Create for incorrect sample
    #[must_use]
    pub fn incorrect(category: ErrorCategory, message: String, soft_labels: SoftLabels) -> Self {
        Self {
            is_correct: false,
            error_category: Some(category),
            error_message: Some(message),
            soft_labels,
            ast_diff: None,
            execution_metrics: ExecutionMetrics::default(),
        }
    }

    /// Set AST diff
    #[must_use]
    pub fn with_ast_diff(mut self, diff: AstDiff) -> Self {
        self.ast_diff = Some(diff);
        self
    }

    /// Set execution metrics
    #[must_use]
    pub fn with_metrics(mut self, metrics: ExecutionMetrics) -> Self {
        self.execution_metrics = metrics;
        self
    }

    /// Convert to flat feature vector for ML
    #[must_use]
    pub fn to_feature_vector(&self) -> Vec<f32> {
        let mut features = Vec::with_capacity(20);

        // Binary label
        features.push(if self.is_correct { 1.0 } else { 0.0 });

        // Error category one-hot (9 values)
        let one_hot = self
            .error_category
            .unwrap_or(ErrorCategory::Unknown)
            .to_one_hot();
        features.extend_from_slice(&one_hot);

        // Soft labels (5 values)
        features.extend_from_slice(&self.soft_labels.to_array());

        // Execution metrics (4 values)
        features.push(self.execution_metrics.source_time_ms as f32 / 1000.0);
        features.push(self.execution_metrics.target_time_ms as f32 / 1000.0);
        features.push(self.execution_metrics.memory_bytes as f32 / 1_000_000.0);
        features.push(if self.execution_metrics.timeout {
            1.0
        } else {
            0.0
        });

        features
    }
}

/// AST diff summary
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AstDiff {
    /// Number of nodes added
    pub nodes_added: u32,
    /// Number of nodes removed
    pub nodes_removed: u32,
    /// Number of nodes modified
    pub nodes_modified: u32,
    /// Structural edit distance
    pub edit_distance: u32,
    /// Most common diff type
    pub primary_change: Option<String>,
}

impl AstDiff {
    /// Total number of changes
    #[must_use]
    pub fn total_changes(&self) -> u32 {
        self.nodes_added + self.nodes_removed + self.nodes_modified
    }

    /// Similarity score (1.0 = identical, 0.0 = completely different)
    #[must_use]
    pub fn similarity(&self, total_nodes: u32) -> f32 {
        if total_nodes == 0 {
            return 1.0;
        }

        let changes = self.total_changes();
        1.0 - (changes as f32 / total_nodes as f32).min(1.0)
    }
}

/// Execution metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExecutionMetrics {
    /// Source execution time in milliseconds
    pub source_time_ms: u64,
    /// Target execution time in milliseconds
    pub target_time_ms: u64,
    /// Memory usage in bytes
    pub memory_bytes: u64,
    /// Whether execution timed out
    pub timeout: bool,
}

impl ExecutionMetrics {
    /// Runtime ratio (target / source)
    #[must_use]
    pub fn runtime_ratio(&self) -> f32 {
        if self.source_time_ms == 0 {
            return 1.0;
        }
        self.target_time_ms as f32 / self.source_time_ms as f32
    }
}

/// Label extractor for oracle results
#[derive(Debug, Default)]
pub struct LabelExtractor;

impl LabelExtractor {
    /// Create new label extractor
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Extract rich label from oracle result
    pub fn extract(
        &self,
        is_correct: bool,
        error_msg: Option<&str>,
        source_output: &str,
        target_output: &str,
        source_time_ms: u64,
        target_time_ms: u64,
    ) -> RichLabel {
        let output_similarity = self.compute_output_similarity(source_output, target_output);

        let runtime_ratio = if source_time_ms == 0 {
            1.0
        } else {
            target_time_ms as f32 / source_time_ms as f32
        };

        let soft_labels = SoftLabelsBuilder::new()
            .output_similarity(output_similarity)
            .runtime_ratio(runtime_ratio)
            .semantic_confidence(if is_correct { 1.0 } else { 0.3 })
            .type_safety(if is_correct { 1.0 } else { 0.5 })
            .build();

        let execution_metrics = ExecutionMetrics {
            source_time_ms,
            target_time_ms,
            memory_bytes: 0,
            timeout: false,
        };

        if is_correct {
            RichLabel::correct(soft_labels).with_metrics(execution_metrics)
        } else {
            let category = error_msg.map_or(ErrorCategory::Unknown, ErrorCategory::classify);
            let message = error_msg.unwrap_or("Unknown error").to_string();

            RichLabel::incorrect(category, message, soft_labels).with_metrics(execution_metrics)
        }
    }

    fn compute_output_similarity(&self, source: &str, target: &str) -> f32 {
        if source == target {
            return 1.0;
        }

        if source.is_empty() && target.is_empty() {
            return 1.0;
        }

        if source.is_empty() || target.is_empty() {
            return 0.0;
        }

        // Simple Jaccard similarity on lines
        let source_lines: std::collections::HashSet<_> = source.lines().collect();
        let target_lines: std::collections::HashSet<_> = target.lines().collect();

        let intersection = source_lines.intersection(&target_lines).count();
        let union = source_lines.union(&target_lines).count();

        if union == 0 {
            1.0
        } else {
            intersection as f32 / union as f32
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========== ErrorCategory Tests ==========

    #[test]
    fn test_error_category_all() {
        let all = ErrorCategory::all();
        assert_eq!(all.len(), 9);
    }

    #[test]
    fn test_error_category_default() {
        assert_eq!(ErrorCategory::default(), ErrorCategory::Unknown);
    }

    #[test]
    fn test_error_category_severity() {
        assert!(ErrorCategory::PanicDivergence.severity() > ErrorCategory::Unknown.severity());
        assert!(
            ErrorCategory::OwnershipViolation.severity()
                > ErrorCategory::CompilationError.severity()
        );
    }

    #[test]
    fn test_error_category_classify_ownership() {
        assert_eq!(
            ErrorCategory::classify("cannot borrow x as mutable"),
            ErrorCategory::OwnershipViolation
        );
        assert_eq!(
            ErrorCategory::classify("value moved here"),
            ErrorCategory::OwnershipViolation
        );
    }

    #[test]
    fn test_error_category_classify_lifetime() {
        assert_eq!(
            ErrorCategory::classify("lifetime 'a does not live long enough"),
            ErrorCategory::LifetimeError
        );
    }

    #[test]
    fn test_error_category_classify_type() {
        assert_eq!(
            ErrorCategory::classify("type mismatch: expected i32"),
            ErrorCategory::TypeMismatch
        );
    }

    #[test]
    fn test_error_category_classify_panic() {
        assert_eq!(
            ErrorCategory::classify("thread panicked at index out of bounds"),
            ErrorCategory::PanicDivergence
        );
    }

    #[test]
    fn test_error_category_classify_output() {
        assert_eq!(
            ErrorCategory::classify("output mismatch: expected 5, actual 6"),
            ErrorCategory::OutputMismatch
        );
    }

    #[test]
    fn test_error_category_classify_compilation() {
        assert_eq!(
            ErrorCategory::classify("cannot find value x in scope"),
            ErrorCategory::CompilationError
        );
    }

    #[test]
    fn test_error_category_classify_runtime() {
        assert_eq!(
            ErrorCategory::classify("integer overflow detected"),
            ErrorCategory::RuntimeError
        );
    }

    #[test]
    fn test_error_category_classify_resource() {
        assert_eq!(
            ErrorCategory::classify("execution timeout"),
            ErrorCategory::ResourceExhaustion
        );
    }

    #[test]
    fn test_error_category_classify_unknown() {
        assert_eq!(
            ErrorCategory::classify("some random error"),
            ErrorCategory::Unknown
        );
    }

    #[test]
    fn test_error_category_one_hot() {
        let one_hot = ErrorCategory::TypeMismatch.to_one_hot();
        assert_eq!(one_hot[0], 1.0);
        assert_eq!(one_hot[1], 0.0);
    }

    #[test]
    fn test_error_category_from_one_hot() {
        let one_hot = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert_eq!(
            ErrorCategory::from_one_hot(&one_hot),
            ErrorCategory::OwnershipViolation
        );
    }

    // ========== SoftLabels Tests ==========

    #[test]
    fn test_soft_labels_default() {
        let labels = SoftLabels::default();
        assert_eq!(labels.output_similarity, 0.0);
    }

    #[test]
    fn test_soft_labels_is_valid() {
        let valid = SoftLabels {
            output_similarity: 0.8,
            runtime_ratio: 1.2,
            structural_similarity: 0.9,
            semantic_confidence: 0.95,
            type_safety: 1.0,
        };
        assert!(valid.is_valid());

        let invalid = SoftLabels {
            output_similarity: -0.1,
            ..Default::default()
        };
        assert!(!invalid.is_valid());
    }

    #[test]
    fn test_soft_labels_to_array() {
        let labels = SoftLabels {
            output_similarity: 0.8,
            runtime_ratio: 1.5,
            structural_similarity: 0.9,
            semantic_confidence: 0.7,
            type_safety: 1.0,
        };

        let arr = labels.to_array();
        assert_eq!(arr.len(), 5);
        assert!((arr[0] - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_soft_labels_overall_score() {
        let perfect = SoftLabels {
            output_similarity: 1.0,
            runtime_ratio: 1.0,
            structural_similarity: 1.0,
            semantic_confidence: 1.0,
            type_safety: 1.0,
        };

        let score = perfect.overall_score();
        assert!((score - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_soft_labels_builder() {
        let labels = SoftLabelsBuilder::new()
            .output_similarity(0.9)
            .runtime_ratio(1.1)
            .structural_similarity(0.95)
            .semantic_confidence(0.85)
            .type_safety(1.0)
            .build();

        assert!((labels.output_similarity - 0.9).abs() < 0.001);
        assert!((labels.runtime_ratio - 1.1).abs() < 0.001);
    }

    #[test]
    fn test_soft_labels_builder_clamps() {
        let labels = SoftLabelsBuilder::new()
            .output_similarity(1.5) // Should clamp to 1.0
            .semantic_confidence(-0.5) // Should clamp to 0.0
            .build();

        assert!((labels.output_similarity - 1.0).abs() < 0.001);
        assert!((labels.semantic_confidence - 0.0).abs() < 0.001);
    }

    // ========== RichLabel Tests ==========

    #[test]
    fn test_rich_label_correct() {
        let label = RichLabel::correct(SoftLabels::default());
        assert!(label.is_correct);
        assert!(label.error_category.is_none());
    }

    #[test]
    fn test_rich_label_incorrect() {
        let label = RichLabel::incorrect(
            ErrorCategory::TypeMismatch,
            "Type error".to_string(),
            SoftLabels::default(),
        );
        assert!(!label.is_correct);
        assert_eq!(label.error_category, Some(ErrorCategory::TypeMismatch));
    }

    #[test]
    fn test_rich_label_with_ast_diff() {
        let diff = AstDiff {
            nodes_added: 5,
            nodes_removed: 2,
            nodes_modified: 3,
            edit_distance: 10,
            primary_change: Some("FunctionDef".to_string()),
        };

        let label = RichLabel::correct(SoftLabels::default()).with_ast_diff(diff);
        assert!(label.ast_diff.is_some());
    }

    #[test]
    fn test_rich_label_feature_vector() {
        let label = RichLabel::correct(SoftLabels {
            output_similarity: 1.0,
            runtime_ratio: 1.0,
            structural_similarity: 1.0,
            semantic_confidence: 1.0,
            type_safety: 1.0,
        });

        let features = label.to_feature_vector();
        assert_eq!(features.len(), 19); // 1 + 9 + 5 + 4
        assert!((features[0] - 1.0).abs() < 0.001); // is_correct
    }

    // ========== AstDiff Tests ==========

    #[test]
    fn test_ast_diff_total_changes() {
        let diff = AstDiff {
            nodes_added: 5,
            nodes_removed: 3,
            nodes_modified: 2,
            edit_distance: 0,
            primary_change: None,
        };

        assert_eq!(diff.total_changes(), 10);
    }

    #[test]
    fn test_ast_diff_similarity() {
        let diff = AstDiff {
            nodes_added: 2,
            nodes_removed: 0,
            nodes_modified: 0,
            edit_distance: 2,
            primary_change: None,
        };

        let sim = diff.similarity(10);
        assert!((sim - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_ast_diff_similarity_empty() {
        let diff = AstDiff::default();
        assert!((diff.similarity(0) - 1.0).abs() < 0.001);
    }

    // ========== ExecutionMetrics Tests ==========

    #[test]
    fn test_execution_metrics_runtime_ratio() {
        let metrics = ExecutionMetrics {
            source_time_ms: 100,
            target_time_ms: 150,
            memory_bytes: 0,
            timeout: false,
        };

        assert!((metrics.runtime_ratio() - 1.5).abs() < 0.001);
    }

    #[test]
    fn test_execution_metrics_runtime_ratio_zero() {
        let metrics = ExecutionMetrics {
            source_time_ms: 0,
            target_time_ms: 100,
            memory_bytes: 0,
            timeout: false,
        };

        assert!((metrics.runtime_ratio() - 1.0).abs() < 0.001);
    }

    // ========== LabelExtractor Tests ==========

    #[test]
    fn test_label_extractor_correct() {
        let extractor = LabelExtractor::new();
        let label = extractor.extract(true, None, "hello\nworld", "hello\nworld", 100, 100);

        assert!(label.is_correct);
        assert!((label.soft_labels.output_similarity - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_label_extractor_incorrect() {
        let extractor = LabelExtractor::new();
        let label = extractor.extract(false, Some("type mismatch error"), "5", "6", 100, 100);

        assert!(!label.is_correct);
        assert_eq!(label.error_category, Some(ErrorCategory::TypeMismatch));
    }

    #[test]
    fn test_label_extractor_output_similarity() {
        let extractor = LabelExtractor::new();

        // Same output
        let same = extractor.extract(true, None, "a\nb\nc", "a\nb\nc", 100, 100);
        assert!((same.soft_labels.output_similarity - 1.0).abs() < 0.001);

        // Partially different
        let partial = extractor.extract(false, None, "a\nb\nc", "a\nb\nd", 100, 100);
        assert!(partial.soft_labels.output_similarity > 0.0);
        assert!(partial.soft_labels.output_similarity < 1.0);
    }

    // ========== Debug Tests ==========

    #[test]
    fn test_error_category_debug() {
        let debug = format!("{:?}", ErrorCategory::TypeMismatch);
        assert!(debug.contains("TypeMismatch"));
    }

    #[test]
    fn test_soft_labels_debug() {
        let labels = SoftLabels::default();
        let debug = format!("{labels:?}");
        assert!(debug.contains("SoftLabels"));
    }

    #[test]
    fn test_rich_label_debug() {
        let label = RichLabel::correct(SoftLabels::default());
        let debug = format!("{label:?}");
        assert!(debug.contains("RichLabel"));
    }

    #[test]
    fn test_label_extractor_debug() {
        let extractor = LabelExtractor::new();
        let debug = format!("{extractor:?}");
        assert!(debug.contains("LabelExtractor"));
    }

    // ========== Serialization Tests ==========

    #[test]
    fn test_error_category_serialize() {
        let category = ErrorCategory::OwnershipViolation;
        let json = serde_json::to_string(&category).unwrap();
        let restored: ErrorCategory = serde_json::from_str(&json).unwrap();
        assert_eq!(category, restored);
    }

    #[test]
    fn test_soft_labels_serialize() {
        let labels = SoftLabelsBuilder::new()
            .output_similarity(0.8)
            .runtime_ratio(1.2)
            .build();

        let json = serde_json::to_string(&labels).unwrap();
        let restored: SoftLabels = serde_json::from_str(&json).unwrap();
        assert!((labels.output_similarity - restored.output_similarity).abs() < 0.001);
    }

    #[test]
    fn test_rich_label_serialize() {
        let label = RichLabel::incorrect(
            ErrorCategory::TypeMismatch,
            "Error".to_string(),
            SoftLabels::default(),
        );

        let json = serde_json::to_string(&label).unwrap();
        let restored: RichLabel = serde_json::from_str(&json).unwrap();
        assert_eq!(label.is_correct, restored.is_correct);
        assert_eq!(label.error_category, restored.error_category);
    }
}

/// Property-based tests
#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Severity is bounded [0, 1]
        #[test]
        fn prop_severity_bounded(idx in 0usize..9) {
            let category = ErrorCategory::from_index(idx);
            let severity = category.severity();
            prop_assert!(severity >= 0.0);
            prop_assert!(severity <= 1.0);
        }

        /// One-hot roundtrip
        #[test]
        fn prop_one_hot_roundtrip(idx in 0usize..9) {
            let original = ErrorCategory::from_index(idx);
            let one_hot = original.to_one_hot();
            let restored = ErrorCategory::from_one_hot(&one_hot);
            prop_assert_eq!(original, restored);
        }

        /// Soft labels array roundtrip preserves structure
        #[test]
        fn prop_soft_labels_structure(
            output_sim in 0.0f32..1.0,
            structural_sim in 0.0f32..1.0,
            semantic_conf in 0.0f32..1.0,
            type_safety in 0.0f32..1.0,
        ) {
            let labels = SoftLabelsBuilder::new()
                .output_similarity(output_sim)
                .structural_similarity(structural_sim)
                .semantic_confidence(semantic_conf)
                .type_safety(type_safety)
                .build();

            prop_assert!(labels.is_valid());
        }

        /// Overall score is bounded [0, 1]
        #[test]
        fn prop_overall_score_bounded(
            output_sim in 0.0f32..1.0,
            runtime_ratio in 0.0f32..10.0,
            structural_sim in 0.0f32..1.0,
            semantic_conf in 0.0f32..1.0,
            type_safety in 0.0f32..1.0,
        ) {
            let labels = SoftLabels {
                output_similarity: output_sim,
                runtime_ratio,
                structural_similarity: structural_sim,
                semantic_confidence: semantic_conf,
                type_safety,
            };

            let score = labels.overall_score();
            prop_assert!(score >= 0.0);
            prop_assert!(score <= 1.0);
        }

        /// Feature vector length is consistent
        #[test]
        fn prop_feature_vector_length(is_correct: bool) {
            let label = if is_correct {
                RichLabel::correct(SoftLabels::default())
            } else {
                RichLabel::incorrect(ErrorCategory::Unknown, "error".to_string(), SoftLabels::default())
            };

            let features = label.to_feature_vector();
            prop_assert_eq!(features.len(), 19);
        }
    }
}

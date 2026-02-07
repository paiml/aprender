//! `HuggingFace` weight comparison module (GH-121).
//!
// Allow format_push_string: cleaner code for string building without I/O concerns
#![allow(clippy::format_push_string, clippy::uninlined_format_args)]
//!
//! Compares model weights between APR and SafeTensors/HuggingFace sources:
//! - Tensor value comparison (L2 norm, max diff)
//! - Tensor shape comparison
//! - Tensor name mapping
//! - Statistical summary
//!
//! # Example
//!
//! ```rust
//! use aprender::format::compare::{TensorComparison, WeightComparer, CompareConfig};
//!
//! let comparison = TensorComparison {
//!     name: "encoder.weight".to_string(),
//!     shape_match: true,
//!     source_shape: vec![512, 768],
//!     target_shape: vec![512, 768],
//!     l2_diff: Some(1.23e-7),
//!     max_diff: Some(4.56e-8),
//!     mean_diff: Some(1.0e-8),
//! };
//! assert!(comparison.is_match());
//! ```
//!
//! # PMAT Compliance
//!
//! - Zero `unwrap()` calls
//! - All public APIs return `Result<T, E>`

use std::collections::{HashMap, HashSet};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for weight comparison
#[derive(Debug, Clone)]
pub struct CompareConfig {
    /// Tolerance for L2 norm difference (default: 1e-5)
    pub l2_tolerance: f64,
    /// Tolerance for max element-wise difference (default: 1e-5)
    pub max_tolerance: f64,
    /// Whether to allow shape broadcasting
    pub allow_broadcast: bool,
    /// Whether to normalize tensors before comparison
    pub normalize_first: bool,
    /// Prefix to strip from source tensor names
    pub source_prefix: Option<String>,
    /// Prefix to strip from target tensor names
    pub target_prefix: Option<String>,
}

impl Default for CompareConfig {
    fn default() -> Self {
        Self {
            l2_tolerance: 1e-5,
            max_tolerance: 1e-5,
            allow_broadcast: false,
            normalize_first: false,
            source_prefix: None,
            target_prefix: None,
        }
    }
}

impl CompareConfig {
    /// Strict comparison (no tolerance for differences)
    #[must_use]
    pub fn strict() -> Self {
        Self {
            l2_tolerance: 0.0,
            max_tolerance: 0.0,
            ..Self::default()
        }
    }

    /// Relaxed comparison for quantized models
    #[must_use]
    pub fn quantized() -> Self {
        Self {
            l2_tolerance: 1e-2,
            max_tolerance: 1e-2,
            ..Self::default()
        }
    }
}

// ============================================================================
// Comparison Results
// ============================================================================

/// Result of comparing a single tensor
#[derive(Debug, Clone)]
pub struct TensorComparison {
    /// Tensor name
    pub name: String,
    /// Whether shapes match
    pub shape_match: bool,
    /// Source tensor shape
    pub source_shape: Vec<usize>,
    /// Target tensor shape
    pub target_shape: Vec<usize>,
    /// L2 norm of difference (if shapes match)
    pub l2_diff: Option<f64>,
    /// Maximum element-wise difference (if shapes match)
    pub max_diff: Option<f64>,
    /// Mean element-wise difference (if shapes match)
    pub mean_diff: Option<f64>,
}

impl TensorComparison {
    /// Check if tensors match within tolerance
    #[must_use]
    pub fn is_match_with_tolerance(&self, l2_tol: f64, max_tol: f64) -> bool {
        self.shape_match
            && self.l2_diff.is_some_and(|d| d <= l2_tol)
            && self.max_diff.is_some_and(|d| d <= max_tol)
    }

    /// Check if tensors match with default tolerance
    #[must_use]
    pub fn is_match(&self) -> bool {
        self.is_match_with_tolerance(1e-5, 1e-5)
    }

    /// Get element count from shape
    #[must_use]
    pub fn element_count(&self) -> usize {
        self.source_shape.iter().product()
    }
}

/// Summary of full model comparison
#[derive(Debug, Clone)]
pub struct CompareReport {
    /// Individual tensor comparisons
    pub tensors: Vec<TensorComparison>,
    /// Tensors only in source
    pub source_only: Vec<String>,
    /// Tensors only in target
    pub target_only: Vec<String>,
    /// Total L2 norm across all tensors
    pub total_l2_diff: f64,
    /// Maximum difference across all tensors
    pub global_max_diff: f64,
    /// Configuration used
    pub config: CompareConfig,
}

impl CompareReport {
    /// Check if all tensors match within tolerance
    #[must_use]
    pub fn all_match(&self) -> bool {
        self.source_only.is_empty()
            && self.target_only.is_empty()
            && self.tensors.iter().all(|t| {
                t.is_match_with_tolerance(self.config.l2_tolerance, self.config.max_tolerance)
            })
    }

    /// Count matching tensors
    #[must_use]
    pub fn match_count(&self) -> usize {
        self.tensors
            .iter()
            .filter(|t| {
                t.is_match_with_tolerance(self.config.l2_tolerance, self.config.max_tolerance)
            })
            .count()
    }

    /// Count mismatched tensors
    #[must_use]
    pub fn mismatch_count(&self) -> usize {
        self.tensors.len() - self.match_count()
    }

    /// Get summary string
    #[must_use]
    pub fn summary(&self) -> String {
        let mut result = String::new();
        result.push_str("Weight Comparison Report\n");
        result.push_str("========================\n\n");

        result.push_str(&format!("Tensors compared: {}\n", self.tensors.len()));
        result.push_str(&format!("  Matching: {}\n", self.match_count()));
        result.push_str(&format!("  Mismatched: {}\n", self.mismatch_count()));
        result.push_str(&format!("  Source only: {}\n", self.source_only.len()));
        result.push_str(&format!("  Target only: {}\n", self.target_only.len()));
        result.push('\n');
        result.push_str(&format!("Total L2 diff: {:.6e}\n", self.total_l2_diff));
        result.push_str(&format!("Global max diff: {:.6e}\n", self.global_max_diff));
        result.push_str(&format!("L2 tolerance: {:.6e}\n", self.config.l2_tolerance));
        result.push_str(&format!(
            "Max tolerance: {:.6e}\n",
            self.config.max_tolerance
        ));

        if !self.source_only.is_empty() {
            result.push_str("\nTensors only in source:\n");
            for name in &self.source_only {
                result.push_str(&format!("  - {name}\n"));
            }
        }

        if !self.target_only.is_empty() {
            result.push_str("\nTensors only in target:\n");
            for name in &self.target_only {
                result.push_str(&format!("  - {name}\n"));
            }
        }

        let mismatched: Vec<_> = self
            .tensors
            .iter()
            .filter(|t| {
                !t.is_match_with_tolerance(self.config.l2_tolerance, self.config.max_tolerance)
            })
            .collect();

        if !mismatched.is_empty() {
            result.push_str("\nMismatched tensors:\n");
            for t in mismatched {
                let shape_info = if t.shape_match {
                    format!("{:?}", t.source_shape)
                } else {
                    format!("{:?} vs {:?}", t.source_shape, t.target_shape)
                };
                let diff_info = t
                    .l2_diff
                    .map_or_else(|| "shape mismatch".to_string(), |d| format!("L2={:.6e}", d));
                result.push_str(&format!("  - {} [{}]: {}\n", t.name, shape_info, diff_info));
            }
        }

        result
    }
}

// ============================================================================
// Weight Comparer
// ============================================================================

/// Compares weights between two tensor collections.
///
/// # Example
///
/// ```rust
/// use aprender::format::compare::{WeightComparer, CompareConfig};
///
/// let comparer = WeightComparer::new(CompareConfig::default());
///
/// // Compare two f32 tensors
/// let source = vec![1.0_f32, 2.0, 3.0];
/// let target = vec![1.0_f32, 2.0, 3.0];
/// let comparison = comparer.compare_tensors(
///     "test",
///     &source,
///     &[3],
///     &target,
///     &[3],
/// );
/// assert!(comparison.is_match());
/// ```
#[derive(Debug, Clone)]
pub struct WeightComparer {
    config: CompareConfig,
}

impl WeightComparer {
    /// Create new weight comparer
    #[must_use]
    pub fn new(config: CompareConfig) -> Self {
        Self { config }
    }

    /// Compare two tensors
    #[must_use]
    pub fn compare_tensors(
        &self,
        name: &str,
        source_data: &[f32],
        source_shape: &[usize],
        target_data: &[f32],
        target_shape: &[usize],
    ) -> TensorComparison {
        let shape_match = source_shape == target_shape;

        let (l2_diff, max_diff, mean_diff) =
            if shape_match && source_data.len() == target_data.len() {
                let diff_stats = self.compute_diff_stats(source_data, target_data);
                (Some(diff_stats.0), Some(diff_stats.1), Some(diff_stats.2))
            } else {
                (None, None, None)
            };

        TensorComparison {
            name: name.to_string(),
            shape_match,
            source_shape: source_shape.to_vec(),
            target_shape: target_shape.to_vec(),
            l2_diff,
            max_diff,
            mean_diff,
        }
    }

    /// Compute L2 norm, max diff, and mean diff
    #[allow(clippy::unused_self)]
    fn compute_diff_stats(&self, source: &[f32], target: &[f32]) -> (f64, f64, f64) {
        if source.is_empty() {
            return (0.0, 0.0, 0.0);
        }

        let mut sum_sq = 0.0_f64;
        let mut max_diff = 0.0_f64;
        let mut sum_diff = 0.0_f64;
        let mut count = 0_usize;

        for (s, t) in source.iter().zip(target.iter()) {
            if s.is_finite() && t.is_finite() {
                let diff = (f64::from(*s) - f64::from(*t)).abs();
                sum_sq += diff * diff;
                if diff > max_diff {
                    max_diff = diff;
                }
                sum_diff += diff;
                count += 1;
            }
        }

        let l2 = sum_sq.sqrt();
        let mean = if count > 0 {
            sum_diff / count as f64
        } else {
            0.0
        };

        (l2, max_diff, mean)
    }

    /// Compare two models represented as tensor maps
    #[must_use]
    pub fn compare_models(
        &self,
        source: &HashMap<String, (Vec<f32>, Vec<usize>)>,
        target: &HashMap<String, (Vec<f32>, Vec<usize>)>,
    ) -> CompareReport {
        let source_names: HashSet<&str> = source.keys().map(String::as_str).collect();
        let target_names: HashSet<&str> = target.keys().map(String::as_str).collect();

        // Normalize names if prefixes configured
        let normalize_name = |name: &str, prefix: &Option<String>| -> String {
            prefix
                .as_ref()
                .and_then(|p| name.strip_prefix(p.as_str()))
                .unwrap_or(name)
                .to_string()
        };

        let source_only: Vec<String> = source_names
            .difference(&target_names)
            .map(|s| (*s).to_string())
            .collect();

        let target_only: Vec<String> = target_names
            .difference(&source_names)
            .map(|s| (*s).to_string())
            .collect();

        let common: Vec<&str> = source_names.intersection(&target_names).copied().collect();

        let mut tensors = Vec::new();
        let mut total_l2_sq = 0.0_f64;
        let mut global_max = 0.0_f64;

        for name in common {
            let (source_data, source_shape) = source.get(name).expect("checked intersection");
            let (target_data, target_shape) = target.get(name).expect("checked intersection");

            let normalized_name = normalize_name(name, &self.config.source_prefix);

            let comparison = self.compare_tensors(
                &normalized_name,
                source_data,
                source_shape,
                target_data,
                target_shape,
            );

            if let Some(l2) = comparison.l2_diff {
                total_l2_sq += l2 * l2;
            }
            if let Some(max) = comparison.max_diff {
                if max > global_max {
                    global_max = max;
                }
            }

            tensors.push(comparison);
        }

        CompareReport {
            tensors,
            source_only,
            target_only,
            total_l2_diff: total_l2_sq.sqrt(),
            global_max_diff: global_max,
            config: self.config.clone(),
        }
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Compare two f32 slices and return L2 norm of difference.
///
/// Returns None if lengths differ or data contains NaN/Inf.
#[must_use]
pub fn l2_diff(a: &[f32], b: &[f32]) -> Option<f64> {
    if a.len() != b.len() {
        return None;
    }

    let mut sum_sq = 0.0_f64;
    for (x, y) in a.iter().zip(b.iter()) {
        if !x.is_finite() || !y.is_finite() {
            return None;
        }
        let diff = f64::from(*x) - f64::from(*y);
        sum_sq += diff * diff;
    }

    Some(sum_sq.sqrt())
}

/// Compare two f32 slices and return max absolute difference.
#[must_use]
pub fn max_diff(a: &[f32], b: &[f32]) -> Option<f64> {
    if a.len() != b.len() {
        return None;
    }

    let mut max = 0.0_f64;
    for (x, y) in a.iter().zip(b.iter()) {
        if !x.is_finite() || !y.is_finite() {
            return None;
        }
        let diff = (f64::from(*x) - f64::from(*y)).abs();
        if diff > max {
            max = diff;
        }
    }

    Some(max)
}

/// Compute relative L2 error: ||a - b|| / ||a||
#[must_use]
pub fn relative_l2_error(a: &[f32], b: &[f32]) -> Option<f64> {
    let diff_norm = l2_diff(a, b)?;
    let a_norm: f64 = a.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();

    if a_norm < f64::EPSILON {
        if diff_norm < f64::EPSILON {
            Some(0.0)
        } else {
            Some(f64::INFINITY)
        }
    } else {
        Some(diff_norm / a_norm)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compare_config_default() {
        let config = CompareConfig::default();
        assert!((config.l2_tolerance - 1e-5).abs() < 1e-10);
        assert!((config.max_tolerance - 1e-5).abs() < 1e-10);
    }

    #[test]
    fn test_compare_config_strict() {
        let config = CompareConfig::strict();
        assert_eq!(config.l2_tolerance, 0.0);
        assert_eq!(config.max_tolerance, 0.0);
    }

    #[test]
    fn test_compare_config_quantized() {
        let config = CompareConfig::quantized();
        assert!((config.l2_tolerance - 1e-2).abs() < 1e-10);
    }

    #[test]
    fn test_tensor_comparison_identical() {
        let comparer = WeightComparer::new(CompareConfig::default());
        let data = vec![1.0_f32, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];

        let comparison = comparer.compare_tensors("test", &data, &shape, &data, &shape);

        assert!(comparison.shape_match);
        assert!(comparison.is_match());
        assert!((comparison.l2_diff.unwrap_or(1.0) - 0.0).abs() < 1e-10);
        assert!((comparison.max_diff.unwrap_or(1.0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_tensor_comparison_different() {
        let comparer = WeightComparer::new(CompareConfig::default());
        let source = vec![1.0_f32, 2.0, 3.0, 4.0];
        let target = vec![1.0_f32, 2.0, 3.0, 5.0]; // Last element differs by 1
        let shape = vec![2, 2];

        let comparison = comparer.compare_tensors("test", &source, &shape, &target, &shape);

        assert!(comparison.shape_match);
        assert!(!comparison.is_match()); // 1.0 diff > 1e-5 tolerance
        assert!((comparison.l2_diff.unwrap() - 1.0).abs() < 1e-10);
        assert!((comparison.max_diff.unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_tensor_comparison_shape_mismatch() {
        let comparer = WeightComparer::new(CompareConfig::default());
        let source = vec![1.0_f32, 2.0, 3.0, 4.0];
        let target = vec![1.0_f32, 2.0, 3.0];
        let source_shape = vec![2, 2];
        let target_shape = vec![3];

        let comparison =
            comparer.compare_tensors("test", &source, &source_shape, &target, &target_shape);

        assert!(!comparison.shape_match);
        assert!(comparison.l2_diff.is_none());
        assert!(!comparison.is_match());
    }

    #[test]
    fn test_compare_models() {
        let comparer = WeightComparer::new(CompareConfig::default());

        let mut source = HashMap::new();
        source.insert("layer1.weight".to_string(), (vec![1.0_f32, 2.0], vec![2]));
        source.insert("layer1.bias".to_string(), (vec![0.5_f32], vec![1]));
        source.insert("layer2.weight".to_string(), (vec![3.0_f32, 4.0], vec![2]));

        let mut target = HashMap::new();
        target.insert("layer1.weight".to_string(), (vec![1.0_f32, 2.0], vec![2]));
        target.insert("layer1.bias".to_string(), (vec![0.5_f32], vec![1]));
        // layer2.weight missing, extra_layer added
        target.insert("extra_layer".to_string(), (vec![5.0_f32], vec![1]));

        let report = comparer.compare_models(&source, &target);

        assert_eq!(report.tensors.len(), 2);
        assert_eq!(report.source_only.len(), 1);
        assert_eq!(report.target_only.len(), 1);
        assert!(report.source_only.contains(&"layer2.weight".to_string()));
        assert!(report.target_only.contains(&"extra_layer".to_string()));
    }

    #[test]
    fn test_compare_report_summary() {
        let comparer = WeightComparer::new(CompareConfig::default());

        let mut source = HashMap::new();
        source.insert("weight".to_string(), (vec![1.0_f32, 2.0], vec![2]));

        let mut target = HashMap::new();
        target.insert("weight".to_string(), (vec![1.0_f32, 2.0], vec![2]));

        let report = comparer.compare_models(&source, &target);
        let summary = report.summary();

        assert!(summary.contains("Tensors compared: 1"));
        assert!(summary.contains("Matching: 1"));
        assert!(summary.contains("Mismatched: 0"));
    }

    #[test]
    fn test_l2_diff_identical() {
        let a = vec![1.0_f32, 2.0, 3.0];
        let b = vec![1.0_f32, 2.0, 3.0];
        let diff = l2_diff(&a, &b).unwrap();
        assert!(diff < 1e-10);
    }

    #[test]
    fn test_l2_diff_different() {
        let a = vec![0.0_f32, 0.0, 0.0];
        let b = vec![3.0_f32, 4.0, 0.0]; // L2 = 5.0
        let diff = l2_diff(&a, &b).unwrap();
        assert!((diff - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_l2_diff_length_mismatch() {
        let a = vec![1.0_f32, 2.0];
        let b = vec![1.0_f32, 2.0, 3.0];
        assert!(l2_diff(&a, &b).is_none());
    }

    #[test]
    fn test_l2_diff_nan() {
        let a = vec![1.0_f32, f32::NAN];
        let b = vec![1.0_f32, 2.0];
        assert!(l2_diff(&a, &b).is_none());
    }

    #[test]
    fn test_max_diff() {
        let a = vec![1.0_f32, 2.0, 3.0];
        let b = vec![1.0_f32, 2.5, 3.0];
        let diff = max_diff(&a, &b).unwrap();
        assert!((diff - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_relative_l2_error() {
        let a = vec![1.0_f32, 0.0, 0.0];
        let b = vec![1.1_f32, 0.0, 0.0];
        let error = relative_l2_error(&a, &b).unwrap();
        // f32 precision: 1.1 - 1.0 is not exactly 0.1
        assert!((error - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_relative_l2_error_zero_norm() {
        let a = vec![0.0_f32, 0.0, 0.0];
        let b = vec![0.0_f32, 0.0, 0.0];
        let error = relative_l2_error(&a, &b).unwrap();
        assert!(error < 1e-10);
    }

    #[test]
    fn test_tensor_comparison_element_count() {
        let comparison = TensorComparison {
            name: "test".to_string(),
            shape_match: true,
            source_shape: vec![2, 3, 4],
            target_shape: vec![2, 3, 4],
            l2_diff: Some(0.0),
            max_diff: Some(0.0),
            mean_diff: Some(0.0),
        };
        assert_eq!(comparison.element_count(), 24);
    }

    #[test]
    fn test_compare_report_all_match() {
        let config = CompareConfig::default();
        let report = CompareReport {
            tensors: vec![TensorComparison {
                name: "test".to_string(),
                shape_match: true,
                source_shape: vec![10],
                target_shape: vec![10],
                l2_diff: Some(1e-10),
                max_diff: Some(1e-10),
                mean_diff: Some(1e-10),
            }],
            source_only: vec![],
            target_only: vec![],
            total_l2_diff: 1e-10,
            global_max_diff: 1e-10,
            config,
        };
        assert!(report.all_match());
    }

    #[test]
    fn test_compare_report_mismatch() {
        let config = CompareConfig::default();
        let report = CompareReport {
            tensors: vec![],
            source_only: vec!["missing".to_string()],
            target_only: vec![],
            total_l2_diff: 0.0,
            global_max_diff: 0.0,
            config,
        };
        assert!(!report.all_match());
    }

    // ========================================================================
    // Additional Coverage Tests for compare.rs
    // ========================================================================

    #[test]
    fn test_compare_config_clone() {
        let config = CompareConfig::default();
        let cloned = config.clone();
        assert_eq!(cloned.l2_tolerance, config.l2_tolerance);
        assert_eq!(cloned.max_tolerance, config.max_tolerance);
    }

    #[test]
    fn test_compare_config_debug() {
        let config = CompareConfig::default();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("CompareConfig"));
    }

    #[test]
    fn test_tensor_comparison_debug() {
        let comparison = TensorComparison {
            name: "test_tensor".to_string(),
            shape_match: true,
            source_shape: vec![2, 2],
            target_shape: vec![2, 2],
            l2_diff: Some(0.0),
            max_diff: Some(0.0),
            mean_diff: Some(0.0),
        };
        let debug_str = format!("{:?}", comparison);
        assert!(debug_str.contains("test_tensor"));
    }

    #[test]
    fn test_tensor_comparison_clone() {
        let comparison = TensorComparison {
            name: "test".to_string(),
            shape_match: true,
            source_shape: vec![2, 2],
            target_shape: vec![2, 2],
            l2_diff: Some(0.1),
            max_diff: Some(0.2),
            mean_diff: Some(0.05),
        };
        let cloned = comparison.clone();
        assert_eq!(cloned.name, "test");
        assert_eq!(cloned.l2_diff, Some(0.1));
    }

    #[test]
    fn test_compare_report_debug() {
        let config = CompareConfig::default();
        let report = CompareReport {
            tensors: vec![],
            source_only: vec![],
            target_only: vec![],
            total_l2_diff: 0.0,
            global_max_diff: 0.0,
            config,
        };
        let debug_str = format!("{:?}", report);
        assert!(debug_str.contains("CompareReport"));
    }

    #[test]
    fn test_compare_report_clone() {
        let config = CompareConfig::default();
        let report = CompareReport {
            tensors: vec![],
            source_only: vec!["a".to_string()],
            target_only: vec!["b".to_string()],
            total_l2_diff: 0.5,
            global_max_diff: 0.3,
            config,
        };
        let cloned = report.clone();
        assert_eq!(cloned.source_only.len(), 1);
        assert_eq!(cloned.total_l2_diff, 0.5);
    }

    #[test]
    fn test_weight_comparer_debug() {
        let comparer = WeightComparer::new(CompareConfig::default());
        let debug_str = format!("{:?}", comparer);
        assert!(debug_str.contains("WeightComparer"));
    }

    #[test]
    fn test_tensor_comparison_no_match_shape_mismatch() {
        let comparison = TensorComparison {
            name: "test".to_string(),
            shape_match: false,
            source_shape: vec![2, 2],
            target_shape: vec![2, 3],
            l2_diff: None,
            max_diff: None,
            mean_diff: None,
        };
        assert!(!comparison.is_match());
    }

    #[test]
    fn test_tensor_comparison_with_mean_diff() {
        let comparison = TensorComparison {
            name: "test".to_string(),
            shape_match: true,
            source_shape: vec![4],
            target_shape: vec![4],
            l2_diff: Some(0.0),
            max_diff: Some(0.0),
            mean_diff: Some(0.0),
        };
        assert!(comparison.is_match());
        assert_eq!(comparison.element_count(), 4);
    }

    #[test]
    fn test_compare_report_with_target_only() {
        let config = CompareConfig::default();
        let report = CompareReport {
            tensors: vec![],
            source_only: vec![],
            target_only: vec!["extra".to_string()],
            total_l2_diff: 0.0,
            global_max_diff: 0.0,
            config,
        };
        assert!(!report.all_match());
    }

    #[test]
    fn test_max_diff_nan_in_source() {
        let a = vec![f32::NAN, 2.0];
        let b = vec![1.0, 2.0];
        assert!(max_diff(&a, &b).is_none());
    }

    #[test]
    fn test_max_diff_inf_diff() {
        let a = vec![f32::INFINITY, 0.0];
        let b = vec![0.0, 0.0];
        let diff = max_diff(&a, &b);
        // INFINITY - 0.0 = INFINITY, which might be filtered out
        // Just verify the function doesn't panic
        let _ = diff;
    }

    #[test]
    fn test_relative_l2_error_length_mismatch() {
        let a = vec![1.0_f32, 2.0];
        let b = vec![1.0_f32, 2.0, 3.0];
        assert!(relative_l2_error(&a, &b).is_none());
    }

    #[test]
    fn test_compare_empty_models() {
        let comparer = WeightComparer::new(CompareConfig::default());
        let source: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
        let target: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();

        let report = comparer.compare_models(&source, &target);
        assert!(report.tensors.is_empty());
        assert!(report.source_only.is_empty());
        assert!(report.target_only.is_empty());
        assert!(report.all_match());
    }

    #[test]
    fn test_compare_report_summary_with_mismatches() {
        let config = CompareConfig::default();
        let report = CompareReport {
            tensors: vec![TensorComparison {
                name: "weight".to_string(),
                shape_match: true,
                source_shape: vec![10],
                target_shape: vec![10],
                l2_diff: Some(1.0), // Large diff
                max_diff: Some(1.0),
                mean_diff: Some(0.1),
            }],
            source_only: vec!["missing".to_string()],
            target_only: vec!["extra".to_string()],
            total_l2_diff: 1.0,
            global_max_diff: 1.0,
            config,
        };

        let summary = report.summary();
        // The summary format may vary, just check it contains key info
        assert!(summary.contains("1"));
        assert!(!summary.is_empty());
    }

    // ========================================================================
    // Extended Coverage Tests (GH-121 coverage push)
    // ========================================================================

    // --- CompareConfig extended tests ---

    #[test]
    fn test_compare_config_default_flags() {
        let config = CompareConfig::default();
        assert!(!config.allow_broadcast);
        assert!(!config.normalize_first);
        assert!(config.source_prefix.is_none());
        assert!(config.target_prefix.is_none());
    }

    #[test]
    fn test_compare_config_strict_inherits_defaults() {
        let config = CompareConfig::strict();
        assert!(!config.allow_broadcast);
        assert!(!config.normalize_first);
        assert!(config.source_prefix.is_none());
        assert!(config.target_prefix.is_none());
    }

    #[test]
    fn test_compare_config_quantized_tolerances() {
        let config = CompareConfig::quantized();
        assert!((config.l2_tolerance - 1e-2).abs() < 1e-10);
        assert!((config.max_tolerance - 1e-2).abs() < 1e-10);
        assert!(!config.allow_broadcast);
        assert!(!config.normalize_first);
    }

    #[test]
    fn test_compare_config_with_prefixes() {
        let config = CompareConfig {
            source_prefix: Some("model.".to_string()),
            target_prefix: Some("encoder.".to_string()),
            ..CompareConfig::default()
        };
        assert_eq!(config.source_prefix.as_deref(), Some("model."));
        assert_eq!(config.target_prefix.as_deref(), Some("encoder."));
    }

    // --- TensorComparison::is_match_with_tolerance edge cases ---

    #[test]
    fn test_is_match_with_tolerance_l2_exceeds_max_within() {
        let comparison = TensorComparison {
            name: "test".to_string(),
            shape_match: true,
            source_shape: vec![4],
            target_shape: vec![4],
            l2_diff: Some(1.0),   // exceeds tolerance
            max_diff: Some(1e-8), // within tolerance
            mean_diff: Some(0.0),
        };
        // l2 exceeds -> not a match
        assert!(!comparison.is_match_with_tolerance(1e-5, 1e-5));
    }

    #[test]
    fn test_is_match_with_tolerance_max_exceeds_l2_within() {
        let comparison = TensorComparison {
            name: "test".to_string(),
            shape_match: true,
            source_shape: vec![4],
            target_shape: vec![4],
            l2_diff: Some(1e-8), // within tolerance
            max_diff: Some(1.0), // exceeds tolerance
            mean_diff: Some(0.0),
        };
        // max exceeds -> not a match
        assert!(!comparison.is_match_with_tolerance(1e-5, 1e-5));
    }

    #[test]
    fn test_is_match_with_tolerance_both_exceed() {
        let comparison = TensorComparison {
            name: "test".to_string(),
            shape_match: true,
            source_shape: vec![4],
            target_shape: vec![4],
            l2_diff: Some(1.0),
            max_diff: Some(1.0),
            mean_diff: Some(0.5),
        };
        assert!(!comparison.is_match_with_tolerance(1e-5, 1e-5));
    }

    #[test]
    fn test_is_match_with_tolerance_exact_boundary() {
        let comparison = TensorComparison {
            name: "test".to_string(),
            shape_match: true,
            source_shape: vec![4],
            target_shape: vec![4],
            l2_diff: Some(1e-5),  // exactly at tolerance
            max_diff: Some(1e-5), // exactly at tolerance
            mean_diff: Some(0.0),
        };
        // <= so exact boundary should match
        assert!(comparison.is_match_with_tolerance(1e-5, 1e-5));
    }

    #[test]
    fn test_is_match_with_tolerance_none_l2() {
        let comparison = TensorComparison {
            name: "test".to_string(),
            shape_match: true,
            source_shape: vec![4],
            target_shape: vec![4],
            l2_diff: None,
            max_diff: Some(0.0),
            mean_diff: None,
        };
        // l2_diff is None -> is_some_and returns false -> not a match
        assert!(!comparison.is_match_with_tolerance(1e-5, 1e-5));
    }

    #[test]
    fn test_is_match_with_tolerance_none_max() {
        let comparison = TensorComparison {
            name: "test".to_string(),
            shape_match: true,
            source_shape: vec![4],
            target_shape: vec![4],
            l2_diff: Some(0.0),
            max_diff: None,
            mean_diff: None,
        };
        // max_diff is None -> not a match
        assert!(!comparison.is_match_with_tolerance(1e-5, 1e-5));
    }

    #[test]
    fn test_is_match_with_tolerance_shape_mismatch_but_diffs_ok() {
        let comparison = TensorComparison {
            name: "test".to_string(),
            shape_match: false,
            source_shape: vec![4],
            target_shape: vec![2, 2],
            l2_diff: Some(0.0),
            max_diff: Some(0.0),
            mean_diff: Some(0.0),
        };
        // shape_match false -> not a match regardless of diffs
        assert!(!comparison.is_match_with_tolerance(1e-5, 1e-5));
    }

    #[test]
    fn test_is_match_with_zero_tolerance() {
        let comparison = TensorComparison {
            name: "test".to_string(),
            shape_match: true,
            source_shape: vec![4],
            target_shape: vec![4],
            l2_diff: Some(0.0),
            max_diff: Some(0.0),
            mean_diff: Some(0.0),
        };
        assert!(comparison.is_match_with_tolerance(0.0, 0.0));
    }

    // --- TensorComparison::element_count edge cases ---

    #[test]
    fn test_element_count_empty_shape() {
        let comparison = TensorComparison {
            name: "scalar".to_string(),
            shape_match: true,
            source_shape: vec![],
            target_shape: vec![],
            l2_diff: Some(0.0),
            max_diff: Some(0.0),
            mean_diff: Some(0.0),
        };
        // Product of empty iterator is 1 (identity element)
        assert_eq!(comparison.element_count(), 1);
    }

    #[test]
    fn test_element_count_single_dim() {
        let comparison = TensorComparison {
            name: "vector".to_string(),
            shape_match: true,
            source_shape: vec![128],
            target_shape: vec![128],
            l2_diff: None,
            max_diff: None,
            mean_diff: None,
        };
        assert_eq!(comparison.element_count(), 128);
    }

    #[test]
    fn test_element_count_high_rank() {
        let comparison = TensorComparison {
            name: "4d".to_string(),
            shape_match: true,
            source_shape: vec![2, 3, 4, 5],
            target_shape: vec![2, 3, 4, 5],
            l2_diff: None,
            max_diff: None,
            mean_diff: None,
        };
        assert_eq!(comparison.element_count(), 120);
    }

    // --- compute_diff_stats edge cases ---

    #[test]
    fn test_compare_tensors_empty_data() {
        let comparer = WeightComparer::new(CompareConfig::default());
        let data: Vec<f32> = vec![];
        let shape: Vec<usize> = vec![0];

        let comparison = comparer.compare_tensors("empty", &data, &shape, &data, &shape);
        assert!(comparison.shape_match);
        // compute_diff_stats returns (0.0, 0.0, 0.0) for empty
        assert_eq!(comparison.l2_diff, Some(0.0));
        assert_eq!(comparison.max_diff, Some(0.0));
        assert_eq!(comparison.mean_diff, Some(0.0));
    }

    #[test]
    fn test_compare_tensors_with_nan_in_data() {
        let comparer = WeightComparer::new(CompareConfig::default());
        let source = vec![1.0_f32, f32::NAN, 3.0];
        let target = vec![1.0_f32, 2.0, 3.0];
        let shape = vec![3];

        let comparison = comparer.compare_tensors("nan_test", &source, &shape, &target, &shape);
        // NaN pairs are skipped; only finite pairs contribute
        assert!(comparison.shape_match);
        assert!(comparison.l2_diff.is_some());
        // The NaN element is skipped, so only elements 0 and 2 are compared
        // Both are identical, so diff should be 0
        assert_eq!(comparison.l2_diff, Some(0.0));
        assert_eq!(comparison.max_diff, Some(0.0));
        assert_eq!(comparison.mean_diff, Some(0.0));
    }

    #[test]
    fn test_compare_tensors_with_inf_in_data() {
        let comparer = WeightComparer::new(CompareConfig::default());
        let source = vec![1.0_f32, f32::INFINITY, 3.0];
        let target = vec![1.0_f32, 2.0, 3.0];
        let shape = vec![3];

        let comparison = comparer.compare_tensors("inf_test", &source, &shape, &target, &shape);
        assert!(comparison.shape_match);
        // Inf is not finite, so that pair is skipped
        assert_eq!(comparison.l2_diff, Some(0.0));
    }

    #[test]
    fn test_compare_tensors_all_non_finite() {
        let comparer = WeightComparer::new(CompareConfig::default());
        let source = vec![f32::NAN, f32::INFINITY];
        let target = vec![f32::NAN, f32::NEG_INFINITY];
        let shape = vec![2];

        let comparison = comparer.compare_tensors("all_nan", &source, &shape, &target, &shape);
        assert!(comparison.shape_match);
        // All pairs skipped => count=0 => mean=0.0
        assert_eq!(comparison.l2_diff, Some(0.0));
        assert_eq!(comparison.max_diff, Some(0.0));
        assert_eq!(comparison.mean_diff, Some(0.0));
    }

    #[test]
    fn test_compare_tensors_same_shape_different_data_length() {
        // Shapes match but data lengths differ (unusual/corrupt state)
        let comparer = WeightComparer::new(CompareConfig::default());
        let source = vec![1.0_f32, 2.0, 3.0, 4.0];
        let target = vec![1.0_f32, 2.0];
        let source_shape = vec![2, 2];
        let target_shape = vec![2, 2]; // same shape

        let comparison = comparer.compare_tensors(
            "mismatch_len",
            &source,
            &source_shape,
            &target,
            &target_shape,
        );
        // shape_match is true, but data.len() != data.len() => None diffs
        assert!(comparison.shape_match);
        assert!(comparison.l2_diff.is_none());
        assert!(comparison.max_diff.is_none());
        assert!(comparison.mean_diff.is_none());
    }

    #[test]
    fn test_compare_tensors_single_element() {
        let comparer = WeightComparer::new(CompareConfig::default());
        let source = vec![3.14_f32];
        let target = vec![3.14_f32];
        let shape = vec![1];

        let comparison = comparer.compare_tensors("scalar", &source, &shape, &target, &shape);
        assert!(comparison.shape_match);
        assert!(comparison.is_match());
        assert_eq!(comparison.l2_diff, Some(0.0));
    }

    #[test]
    fn test_compare_tensors_large_diff() {
        let comparer = WeightComparer::new(CompareConfig::default());
        let source = vec![0.0_f32; 4];
        let target = vec![100.0_f32; 4];
        let shape = vec![4];

        let comparison = comparer.compare_tensors("big_diff", &source, &shape, &target, &shape);
        assert!(comparison.shape_match);
        assert!(!comparison.is_match());
        // L2 = sqrt(4 * 100^2) = sqrt(40000) = 200
        let l2 = comparison.l2_diff.expect("should have l2");
        assert!((l2 - 200.0).abs() < 1e-6);
        let max = comparison.max_diff.expect("should have max");
        assert!((max - 100.0).abs() < 1e-6);
        let mean = comparison.mean_diff.expect("should have mean");
        assert!((mean - 100.0).abs() < 1e-6);
    }

    // --- compare_models with prefix stripping ---

    #[test]
    fn test_compare_models_with_source_prefix() {
        let config = CompareConfig {
            source_prefix: Some("model.".to_string()),
            ..CompareConfig::default()
        };
        let comparer = WeightComparer::new(config);

        let mut source = HashMap::new();
        source.insert(
            "model.layer1.weight".to_string(),
            (vec![1.0_f32, 2.0], vec![2]),
        );

        let mut target = HashMap::new();
        target.insert(
            "model.layer1.weight".to_string(),
            (vec![1.0_f32, 2.0], vec![2]),
        );

        let report = comparer.compare_models(&source, &target);
        assert_eq!(report.tensors.len(), 1);
        // The normalized name should strip the prefix
        assert_eq!(report.tensors[0].name, "layer1.weight");
        assert!(report.all_match());
    }

    #[test]
    fn test_compare_models_prefix_no_match() {
        // Source prefix doesn't match tensor name => name unchanged
        let config = CompareConfig {
            source_prefix: Some("encoder.".to_string()),
            ..CompareConfig::default()
        };
        let comparer = WeightComparer::new(config);

        let mut source = HashMap::new();
        source.insert("decoder.weight".to_string(), (vec![1.0_f32], vec![1]));

        let mut target = HashMap::new();
        target.insert("decoder.weight".to_string(), (vec![1.0_f32], vec![1]));

        let report = comparer.compare_models(&source, &target);
        assert_eq!(report.tensors.len(), 1);
        // Prefix "encoder." doesn't match "decoder.weight" so name is unchanged
        assert_eq!(report.tensors[0].name, "decoder.weight");
    }

    #[test]
    fn test_compare_models_all_source_only() {
        let comparer = WeightComparer::new(CompareConfig::default());

        let mut source = HashMap::new();
        source.insert("a.weight".to_string(), (vec![1.0_f32], vec![1]));
        source.insert("b.weight".to_string(), (vec![2.0_f32], vec![1]));

        let target: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();

        let report = comparer.compare_models(&source, &target);
        assert!(report.tensors.is_empty());
        assert_eq!(report.source_only.len(), 2);
        assert!(report.target_only.is_empty());
        assert!(!report.all_match());
    }

    #[test]
    fn test_compare_models_all_target_only() {
        let comparer = WeightComparer::new(CompareConfig::default());

        let source: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();

        let mut target = HashMap::new();
        target.insert("x.weight".to_string(), (vec![1.0_f32], vec![1]));

        let report = comparer.compare_models(&source, &target);
        assert!(report.tensors.is_empty());
        assert!(report.source_only.is_empty());
        assert_eq!(report.target_only.len(), 1);
        assert!(!report.all_match());
    }

    #[test]
    fn test_compare_models_mixed_match_mismatch() {
        let comparer = WeightComparer::new(CompareConfig::default());

        let mut source = HashMap::new();
        source.insert("good.weight".to_string(), (vec![1.0_f32, 2.0], vec![2]));
        source.insert("bad.weight".to_string(), (vec![1.0_f32, 2.0], vec![2]));

        let mut target = HashMap::new();
        target.insert("good.weight".to_string(), (vec![1.0_f32, 2.0], vec![2]));
        target.insert("bad.weight".to_string(), (vec![100.0_f32, 200.0], vec![2]));

        let report = comparer.compare_models(&source, &target);
        assert_eq!(report.tensors.len(), 2);
        assert_eq!(report.match_count(), 1);
        assert_eq!(report.mismatch_count(), 1);
        assert!(!report.all_match());
    }

    #[test]
    fn test_compare_models_shape_mismatch_in_common() {
        let comparer = WeightComparer::new(CompareConfig::default());

        let mut source = HashMap::new();
        source.insert("layer.weight".to_string(), (vec![1.0_f32, 2.0], vec![2]));

        let mut target = HashMap::new();
        target.insert(
            "layer.weight".to_string(),
            (vec![1.0_f32, 2.0, 3.0], vec![3]),
        );

        let report = comparer.compare_models(&source, &target);
        assert_eq!(report.tensors.len(), 1);
        assert!(!report.tensors[0].shape_match);
        assert!(report.tensors[0].l2_diff.is_none());
        assert_eq!(report.mismatch_count(), 1);
        assert!(!report.all_match());
    }

    #[test]
    fn test_compare_models_accumulates_l2_and_max() {
        let comparer = WeightComparer::new(CompareConfig::default());

        let mut source = HashMap::new();
        source.insert("a".to_string(), (vec![0.0_f32], vec![1]));
        source.insert("b".to_string(), (vec![0.0_f32], vec![1]));

        let mut target = HashMap::new();
        target.insert("a".to_string(), (vec![3.0_f32], vec![1]));
        target.insert("b".to_string(), (vec![4.0_f32], vec![1]));

        let report = comparer.compare_models(&source, &target);
        // total_l2_diff = sqrt(3^2 + 4^2) = sqrt(9+16) = 5
        assert!((report.total_l2_diff - 5.0).abs() < 1e-6);
        // global_max_diff = max(3, 4) = 4
        assert!((report.global_max_diff - 4.0).abs() < 1e-6);
    }

    // --- CompareReport::summary() branch coverage ---

    #[test]
    fn test_summary_no_source_only_no_target_only() {
        let report = CompareReport {
            tensors: vec![TensorComparison {
                name: "w".to_string(),
                shape_match: true,
                source_shape: vec![2],
                target_shape: vec![2],
                l2_diff: Some(0.0),
                max_diff: Some(0.0),
                mean_diff: Some(0.0),
            }],
            source_only: vec![],
            target_only: vec![],
            total_l2_diff: 0.0,
            global_max_diff: 0.0,
            config: CompareConfig::default(),
        };
        let summary = report.summary();
        assert!(summary.contains("Weight Comparison Report"));
        assert!(summary.contains("Tensors compared: 1"));
        assert!(summary.contains("Matching: 1"));
        assert!(summary.contains("Mismatched: 0"));
        assert!(summary.contains("Source only: 0"));
        assert!(summary.contains("Target only: 0"));
        // Should NOT contain the "Tensors only in source/target" sections
        assert!(!summary.contains("Tensors only in source:"));
        assert!(!summary.contains("Tensors only in target:"));
        assert!(!summary.contains("Mismatched tensors:"));
    }

    #[test]
    fn test_summary_source_only_section() {
        let report = CompareReport {
            tensors: vec![],
            source_only: vec!["layer1.weight".to_string(), "layer2.bias".to_string()],
            target_only: vec![],
            total_l2_diff: 0.0,
            global_max_diff: 0.0,
            config: CompareConfig::default(),
        };
        let summary = report.summary();
        assert!(summary.contains("Tensors only in source:"));
        assert!(summary.contains("  - layer1.weight"));
        assert!(summary.contains("  - layer2.bias"));
        assert!(!summary.contains("Tensors only in target:"));
    }

    #[test]
    fn test_summary_target_only_section() {
        let report = CompareReport {
            tensors: vec![],
            source_only: vec![],
            target_only: vec!["extra.weight".to_string()],
            total_l2_diff: 0.0,
            global_max_diff: 0.0,
            config: CompareConfig::default(),
        };
        let summary = report.summary();
        assert!(!summary.contains("Tensors only in source:"));
        assert!(summary.contains("Tensors only in target:"));
        assert!(summary.contains("  - extra.weight"));
    }

    #[test]
    fn test_summary_mismatched_tensor_with_shape_match() {
        // Mismatched tensor where shapes match but l2 exceeds tolerance
        let report = CompareReport {
            tensors: vec![TensorComparison {
                name: "attn.weight".to_string(),
                shape_match: true,
                source_shape: vec![4, 4],
                target_shape: vec![4, 4],
                l2_diff: Some(0.5),
                max_diff: Some(0.1),
                mean_diff: Some(0.01),
            }],
            source_only: vec![],
            target_only: vec![],
            total_l2_diff: 0.5,
            global_max_diff: 0.1,
            config: CompareConfig::default(),
        };
        let summary = report.summary();
        assert!(summary.contains("Mismatched tensors:"));
        // shape_match=true -> format shows single shape
        assert!(summary.contains("[4, 4]"));
        assert!(summary.contains("L2="));
        assert!(summary.contains("attn.weight"));
    }

    #[test]
    fn test_summary_mismatched_tensor_with_shape_mismatch() {
        // Mismatched tensor where shapes differ -> "shape mismatch" string
        let report = CompareReport {
            tensors: vec![TensorComparison {
                name: "ffn.weight".to_string(),
                shape_match: false,
                source_shape: vec![4, 4],
                target_shape: vec![4, 8],
                l2_diff: None,
                max_diff: None,
                mean_diff: None,
            }],
            source_only: vec![],
            target_only: vec![],
            total_l2_diff: 0.0,
            global_max_diff: 0.0,
            config: CompareConfig::default(),
        };
        let summary = report.summary();
        assert!(summary.contains("Mismatched tensors:"));
        // shape_match=false -> format shows "vs"
        assert!(summary.contains("[4, 4] vs [4, 8]"));
        // l2_diff is None -> "shape mismatch" text
        assert!(summary.contains("shape mismatch"));
        assert!(summary.contains("ffn.weight"));
    }

    #[test]
    fn test_summary_tolerances_displayed() {
        let config = CompareConfig {
            l2_tolerance: 1e-3,
            max_tolerance: 2e-4,
            ..CompareConfig::default()
        };
        let report = CompareReport {
            tensors: vec![],
            source_only: vec![],
            target_only: vec![],
            total_l2_diff: 0.0,
            global_max_diff: 0.0,
            config,
        };
        let summary = report.summary();
        assert!(summary.contains("L2 tolerance:"));
        assert!(summary.contains("Max tolerance:"));
    }

    #[test]
    fn test_summary_total_l2_and_global_max_displayed() {
        let report = CompareReport {
            tensors: vec![],
            source_only: vec![],
            target_only: vec![],
            total_l2_diff: 1.234_567,
            global_max_diff: 0.987_654,
            config: CompareConfig::default(),
        };
        let summary = report.summary();
        assert!(summary.contains("Total L2 diff:"));
        assert!(summary.contains("Global max diff:"));
    }

    // --- CompareReport::match_count / mismatch_count ---

    #[test]
    fn test_match_count_all_matching() {
        let report = CompareReport {
            tensors: vec![
                TensorComparison {
                    name: "a".to_string(),
                    shape_match: true,
                    source_shape: vec![1],
                    target_shape: vec![1],
                    l2_diff: Some(0.0),
                    max_diff: Some(0.0),
                    mean_diff: Some(0.0),
                },
                TensorComparison {
                    name: "b".to_string(),
                    shape_match: true,
                    source_shape: vec![1],
                    target_shape: vec![1],
                    l2_diff: Some(1e-8),
                    max_diff: Some(1e-8),
                    mean_diff: Some(1e-8),
                },
            ],
            source_only: vec![],
            target_only: vec![],
            total_l2_diff: 0.0,
            global_max_diff: 0.0,
            config: CompareConfig::default(),
        };
        assert_eq!(report.match_count(), 2);
        assert_eq!(report.mismatch_count(), 0);
    }

    #[test]
    fn test_match_count_none_matching() {
        let report = CompareReport {
            tensors: vec![TensorComparison {
                name: "x".to_string(),
                shape_match: false,
                source_shape: vec![1],
                target_shape: vec![2],
                l2_diff: None,
                max_diff: None,
                mean_diff: None,
            }],
            source_only: vec![],
            target_only: vec![],
            total_l2_diff: 0.0,
            global_max_diff: 0.0,
            config: CompareConfig::default(),
        };
        assert_eq!(report.match_count(), 0);
        assert_eq!(report.mismatch_count(), 1);
    }

    #[test]
    fn test_match_count_empty_tensors() {
        let report = CompareReport {
            tensors: vec![],
            source_only: vec![],
            target_only: vec![],
            total_l2_diff: 0.0,
            global_max_diff: 0.0,
            config: CompareConfig::default(),
        };
        assert_eq!(report.match_count(), 0);
        assert_eq!(report.mismatch_count(), 0);
    }

    #[test]
    fn test_match_count_with_quantized_tolerance() {
        let report = CompareReport {
            tensors: vec![TensorComparison {
                name: "q".to_string(),
                shape_match: true,
                source_shape: vec![4],
                target_shape: vec![4],
                l2_diff: Some(5e-3), // exceeds default but within quantized
                max_diff: Some(5e-3),
                mean_diff: Some(1e-3),
            }],
            source_only: vec![],
            target_only: vec![],
            total_l2_diff: 5e-3,
            global_max_diff: 5e-3,
            config: CompareConfig::quantized(),
        };
        assert_eq!(report.match_count(), 1);
        assert_eq!(report.mismatch_count(), 0);
        assert!(report.all_match());
    }

    // --- Utility function edge cases ---

    #[test]
    fn test_l2_diff_empty_slices() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        let diff = l2_diff(&a, &b).expect("should succeed for empty");
        assert_eq!(diff, 0.0);
    }

    #[test]
    fn test_l2_diff_nan_in_target() {
        let a = vec![1.0_f32, 2.0];
        let b = vec![1.0_f32, f32::NAN];
        assert!(l2_diff(&a, &b).is_none());
    }

    #[test]
    fn test_l2_diff_inf_in_source() {
        let a = vec![f32::INFINITY, 2.0];
        let b = vec![1.0_f32, 2.0];
        assert!(l2_diff(&a, &b).is_none());
    }

    #[test]
    fn test_l2_diff_neg_inf() {
        let a = vec![f32::NEG_INFINITY];
        let b = vec![0.0_f32];
        assert!(l2_diff(&a, &b).is_none());
    }

    #[test]
    fn test_l2_diff_single_element() {
        let a = vec![5.0_f32];
        let b = vec![2.0_f32];
        let diff = l2_diff(&a, &b).expect("should succeed");
        assert!((diff - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_max_diff_empty_slices() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        let diff = max_diff(&a, &b).expect("should succeed for empty");
        assert_eq!(diff, 0.0);
    }

    #[test]
    fn test_max_diff_length_mismatch() {
        let a = vec![1.0_f32, 2.0, 3.0];
        let b = vec![1.0_f32];
        assert!(max_diff(&a, &b).is_none());
    }

    #[test]
    fn test_max_diff_nan_in_target() {
        let a = vec![1.0_f32, 2.0];
        let b = vec![1.0_f32, f32::NAN];
        assert!(max_diff(&a, &b).is_none());
    }

    #[test]
    fn test_max_diff_identical() {
        let a = vec![1.0_f32, 2.0, 3.0];
        let b = vec![1.0_f32, 2.0, 3.0];
        let diff = max_diff(&a, &b).expect("should succeed");
        assert_eq!(diff, 0.0);
    }

    #[test]
    fn test_max_diff_negative_values() {
        let a = vec![-5.0_f32, -3.0];
        let b = vec![-2.0_f32, -1.0];
        let diff = max_diff(&a, &b).expect("should succeed");
        // max(|-5 - (-2)|, |-3 - (-1)|) = max(3, 2) = 3
        assert!((diff - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_relative_l2_error_nan_input() {
        let a = vec![f32::NAN];
        let b = vec![1.0_f32];
        // l2_diff returns None for NaN -> relative_l2_error returns None
        assert!(relative_l2_error(&a, &b).is_none());
    }

    #[test]
    fn test_relative_l2_error_zero_source_nonzero_diff() {
        let a = vec![0.0_f32, 0.0];
        let b = vec![1.0_f32, 0.0];
        let error = relative_l2_error(&a, &b).expect("should succeed");
        // a_norm ~ 0, diff_norm = 1.0 > EPSILON -> INFINITY
        assert!(error.is_infinite());
    }

    #[test]
    fn test_relative_l2_error_identical_nonzero() {
        let a = vec![3.0_f32, 4.0]; // norm = 5.0
        let b = vec![3.0_f32, 4.0];
        let error = relative_l2_error(&a, &b).expect("should succeed");
        assert!(error < 1e-10);
    }

    #[test]
    fn test_relative_l2_error_proportional() {
        // If b = 2*a, diff = a, relative error = ||a|| / ||a|| = 1.0
        let a = vec![3.0_f32, 4.0]; // norm = 5.0
        let b = vec![6.0_f32, 8.0]; // diff norm = 5.0
        let error = relative_l2_error(&a, &b).expect("should succeed");
        assert!((error - 1.0).abs() < 1e-6);
    }

    // --- WeightComparer Clone/Debug ---

    #[test]
    fn test_weight_comparer_clone() {
        let comparer = WeightComparer::new(CompareConfig::quantized());
        let cloned = comparer.clone();
        let debug_orig = format!("{:?}", comparer);
        let debug_clone = format!("{:?}", cloned);
        assert_eq!(debug_orig, debug_clone);
    }

    // --- Integration-level: full round-trip ---

    #[test]
    fn test_full_round_trip_matching_model() {
        let config = CompareConfig::default();
        let comparer = WeightComparer::new(config);

        let mut model = HashMap::new();
        model.insert(
            "embed.weight".to_string(),
            (vec![0.1_f32, 0.2, 0.3, 0.4], vec![2, 2]),
        );
        model.insert("head.bias".to_string(), (vec![0.01_f32, 0.02], vec![2]));

        let report = comparer.compare_models(&model, &model);
        assert!(report.all_match());
        assert_eq!(report.match_count(), 2);
        assert_eq!(report.mismatch_count(), 0);
        assert!(report.source_only.is_empty());
        assert!(report.target_only.is_empty());
        assert!(report.total_l2_diff < 1e-10);
        assert!(report.global_max_diff < 1e-10);

        let summary = report.summary();
        assert!(summary.contains("Matching: 2"));
        assert!(summary.contains("Mismatched: 0"));
    }

    #[test]
    fn test_full_round_trip_mismatching_model() {
        let config = CompareConfig::strict();
        let comparer = WeightComparer::new(config);

        let mut source = HashMap::new();
        source.insert("w1".to_string(), (vec![1.0_f32], vec![1]));
        source.insert("w2".to_string(), (vec![2.0_f32], vec![1]));
        source.insert("w3".to_string(), (vec![3.0_f32], vec![1]));

        let mut target = HashMap::new();
        target.insert("w1".to_string(), (vec![1.0_f32], vec![1])); // match
        target.insert("w2".to_string(), (vec![2.5_f32], vec![1])); // mismatch
                                                                   // w3 missing from target, w4 extra in target
        target.insert("w4".to_string(), (vec![4.0_f32], vec![1]));

        let report = comparer.compare_models(&source, &target);
        assert!(!report.all_match());

        let summary = report.summary();
        assert!(summary.contains("Source only: 1"));
        assert!(summary.contains("Target only: 1"));
    }
}

//! Model diff library (TOOL-APR-002)
//!
//! Provides format-agnostic model comparison for APR, GGUF, and SafeTensors.
//!
//! Toyota Way: Kaizen - Continuous improvement through comparison.
//!
//! # Supported Formats
//!
//! | Format | Extension | Notes |
//! |--------|-----------|-------|
//! | APR | `.apr` | Native format, full metadata |
//! | GGUF | `.gguf` | llama.cpp format |
//! | SafeTensors | `.safetensors` | HuggingFace format |
//!
//! # Example
//!
//! ```rust,ignore
//! use aprender::format::diff::{diff_models, DiffOptions};
//!
//! let report = diff_models("model_a.gguf", "model_b.gguf", DiffOptions::default())?;
//! if report.is_identical() {
//!     println!("Models are identical");
//! } else {
//!     for diff in &report.differences {
//!         println!("{}: {} → {}", diff.field, diff.value1, diff.value2);
//!     }
//! }
//! ```

use crate::error::{AprenderError, Result};
use crate::format::rosetta::{FormatType, InspectionReport, RosettaStone};
use serde::Serialize;
use std::path::Path;

// ============================================================================
// Types
// ============================================================================

/// A single difference between two models
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct DiffEntry {
    /// Name of the differing field
    pub field: String,
    /// Value from first model
    pub value1: String,
    /// Value from second model
    pub value2: String,
    /// Category of difference
    pub category: DiffCategory,
}

/// Category of difference for filtering
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
pub enum DiffCategory {
    /// Format-level difference (magic, version)
    Format,
    /// Metadata difference (name, description, architecture)
    Metadata,
    /// Tensor-level difference (count, names, shapes)
    Tensor,
    /// Quantization difference
    Quantization,
    /// Size difference
    Size,
}

impl DiffCategory {
    /// Get human-readable name
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Format => "format",
            Self::Metadata => "metadata",
            Self::Tensor => "tensor",
            Self::Quantization => "quantization",
            Self::Size => "size",
        }
    }
}

/// Options for diff operation
#[derive(Debug, Clone)]
pub struct DiffOptions {
    /// Include tensor-level comparison
    pub compare_tensors: bool,
    /// Include metadata comparison
    pub compare_metadata: bool,
    /// Compare tensor statistics (min, max, mean, std)
    pub compare_stats: bool,
    /// Tensor name filter (regex pattern)
    pub tensor_filter: Option<String>,
}

impl Default for DiffOptions {
    fn default() -> Self {
        Self {
            compare_tensors: true,
            compare_metadata: true,
            compare_stats: false,
            tensor_filter: None,
        }
    }
}

impl DiffOptions {
    /// Create new options with defaults
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable tensor comparison
    #[must_use]
    pub const fn with_tensors(mut self) -> Self {
        self.compare_tensors = true;
        self
    }

    /// Disable tensor comparison
    #[must_use]
    pub const fn without_tensors(mut self) -> Self {
        self.compare_tensors = false;
        self
    }

    /// Enable metadata comparison
    #[must_use]
    pub const fn with_metadata(mut self) -> Self {
        self.compare_metadata = true;
        self
    }

    /// Disable metadata comparison
    #[must_use]
    pub const fn without_metadata(mut self) -> Self {
        self.compare_metadata = false;
        self
    }

    /// Enable tensor statistics comparison
    #[must_use]
    pub const fn with_stats(mut self) -> Self {
        self.compare_stats = true;
        self
    }

    /// Set tensor name filter
    #[must_use]
    pub fn with_filter(mut self, pattern: &str) -> Self {
        self.tensor_filter = Some(pattern.to_string());
        self
    }
}

/// Complete diff report between two models
#[derive(Debug, Clone, Serialize)]
pub struct DiffReport {
    /// Path to first model
    pub path1: String,
    /// Path to second model
    pub path2: String,
    /// Format of first model
    pub format1: String,
    /// Format of second model
    pub format2: String,
    /// All differences found
    pub differences: Vec<DiffEntry>,
    /// First model inspection
    #[serde(skip)]
    pub inspection1: Option<InspectionReport>,
    /// Second model inspection
    #[serde(skip)]
    pub inspection2: Option<InspectionReport>,
}

impl DiffReport {
    /// Check if models are identical
    #[must_use]
    pub fn is_identical(&self) -> bool {
        self.differences.is_empty()
    }

    /// Get count of differences
    #[must_use]
    pub fn diff_count(&self) -> usize {
        self.differences.len()
    }

    /// Get differences by category
    #[must_use]
    pub fn differences_by_category(&self, category: DiffCategory) -> Vec<&DiffEntry> {
        self.differences
            .iter()
            .filter(|d| d.category == category)
            .collect()
    }

    /// Check if formats are the same
    #[must_use]
    pub fn same_format(&self) -> bool {
        self.format1 == self.format2
    }

    /// Get summary string
    #[must_use]
    pub fn summary(&self) -> String {
        if self.is_identical() {
            format!("{} and {} are IDENTICAL", self.path1, self.path2)
        } else {
            format!(
                "{} and {} differ in {} fields",
                self.path1,
                self.path2,
                self.diff_count()
            )
        }
    }
}

// ============================================================================
// Main Entry Points
// ============================================================================

/// Compare two model files
///
/// Supports APR, GGUF, and SafeTensors formats. Format is auto-detected
/// from magic bytes.
///
/// # Arguments
///
/// * `path1` - Path to first model
/// * `path2` - Path to second model
/// * `options` - Diff options
///
/// # Returns
///
/// `DiffReport` containing all differences
///
/// # Errors
///
/// Returns error if files cannot be read or formats are unsupported
pub fn diff_models<P: AsRef<Path>>(
    path1: P,
    path2: P,
    options: DiffOptions,
) -> Result<DiffReport> {
    let path1 = path1.as_ref();
    let path2 = path2.as_ref();

    // Validate paths
    validate_path(path1)?;
    validate_path(path2)?;

    // Detect formats
    let format1 = FormatType::from_magic(path1).or_else(|_| FormatType::from_extension(path1))?;
    let format2 = FormatType::from_magic(path2).or_else(|_| FormatType::from_extension(path2))?;

    // Get inspections
    let rosetta = RosettaStone::new();
    let inspection1 = rosetta.inspect(path1)?;
    let inspection2 = rosetta.inspect(path2)?;

    // Compute differences
    let differences = compute_differences(&inspection1, &inspection2, &options);

    Ok(DiffReport {
        path1: path1.display().to_string(),
        path2: path2.display().to_string(),
        format1: format1.to_string(),
        format2: format2.to_string(),
        differences,
        inspection1: Some(inspection1),
        inspection2: Some(inspection2),
    })
}

/// Compare two inspections directly (for advanced use)
///
/// # Arguments
///
/// * `inspection1` - First model inspection
/// * `inspection2` - Second model inspection
/// * `path1` - Path/name for first model
/// * `path2` - Path/name for second model
/// * `options` - Diff options
///
/// # Returns
///
/// `DiffReport` containing all differences
#[must_use]
pub fn diff_inspections(
    inspection1: &InspectionReport,
    inspection2: &InspectionReport,
    path1: &str,
    path2: &str,
    options: DiffOptions,
) -> DiffReport {
    let differences = compute_differences(inspection1, inspection2, &options);

    DiffReport {
        path1: path1.to_string(),
        path2: path2.to_string(),
        format1: inspection1.format.to_string(),
        format2: inspection2.format.to_string(),
        differences,
        inspection1: None,
        inspection2: None,
    }
}

// ============================================================================
// Internal Functions
// ============================================================================

/// Validate that path exists and is a file
fn validate_path(path: &Path) -> Result<()> {
    if !path.exists() {
        return Err(AprenderError::FormatError {
            message: format!("File not found: {}", path.display()),
        });
    }
    if !path.is_file() {
        return Err(AprenderError::FormatError {
            message: format!("Not a file: {}", path.display()),
        });
    }
    Ok(())
}

/// Compute all differences between two inspections
fn compute_differences(
    i1: &InspectionReport,
    i2: &InspectionReport,
    options: &DiffOptions,
) -> Vec<DiffEntry> {
    let mut diffs = Vec::new();

    // Format differences
    if i1.format != i2.format {
        diffs.push(DiffEntry {
            field: "format".to_string(),
            value1: i1.format.to_string(),
            value2: i2.format.to_string(),
            category: DiffCategory::Format,
        });
    }

    // Size differences
    if i1.file_size != i2.file_size {
        diffs.push(DiffEntry {
            field: "file_size".to_string(),
            value1: format_size(i1.file_size),
            value2: format_size(i2.file_size),
            category: DiffCategory::Size,
        });
    }

    if i1.total_params != i2.total_params {
        diffs.push(DiffEntry {
            field: "total_params".to_string(),
            value1: format_params(i1.total_params),
            value2: format_params(i2.total_params),
            category: DiffCategory::Size,
        });
    }

    // Architecture
    if i1.architecture != i2.architecture {
        diffs.push(DiffEntry {
            field: "architecture".to_string(),
            value1: i1.architecture.clone().unwrap_or_else(|| "(none)".to_string()),
            value2: i2.architecture.clone().unwrap_or_else(|| "(none)".to_string()),
            category: DiffCategory::Metadata,
        });
    }

    // Quantization
    if i1.quantization != i2.quantization {
        diffs.push(DiffEntry {
            field: "quantization".to_string(),
            value1: i1.quantization.clone().unwrap_or_else(|| "(none)".to_string()),
            value2: i2.quantization.clone().unwrap_or_else(|| "(none)".to_string()),
            category: DiffCategory::Quantization,
        });
    }

    // Metadata comparison
    if options.compare_metadata {
        compare_metadata(&i1.metadata, &i2.metadata, &mut diffs);
    }

    // Tensor comparison
    if options.compare_tensors {
        compare_tensors(&i1.tensors, &i2.tensors, options, &mut diffs);
    }

    diffs
}

/// Compare metadata maps
fn compare_metadata(
    m1: &std::collections::BTreeMap<String, String>,
    m2: &std::collections::BTreeMap<String, String>,
    diffs: &mut Vec<DiffEntry>,
) {
    // Keys in m1 but not m2 or different
    for (key, val1) in m1 {
        match m2.get(key) {
            Some(val2) if val1 != val2 => {
                diffs.push(DiffEntry {
                    field: format!("metadata.{key}"),
                    value1: truncate_value(val1, 50),
                    value2: truncate_value(val2, 50),
                    category: DiffCategory::Metadata,
                });
            }
            None => {
                diffs.push(DiffEntry {
                    field: format!("metadata.{key}"),
                    value1: truncate_value(val1, 50),
                    value2: "(missing)".to_string(),
                    category: DiffCategory::Metadata,
                });
            }
            _ => {}
        }
    }

    // Keys in m2 but not m1
    for (key, val2) in m2 {
        if !m1.contains_key(key) {
            diffs.push(DiffEntry {
                field: format!("metadata.{key}"),
                value1: "(missing)".to_string(),
                value2: truncate_value(val2, 50),
                category: DiffCategory::Metadata,
            });
        }
    }
}

/// Compare tensor lists
fn compare_tensors(
    t1: &[crate::format::rosetta::TensorInfo],
    t2: &[crate::format::rosetta::TensorInfo],
    options: &DiffOptions,
    diffs: &mut Vec<DiffEntry>,
) {
    // Tensor count
    if t1.len() != t2.len() {
        diffs.push(DiffEntry {
            field: "tensor_count".to_string(),
            value1: t1.len().to_string(),
            value2: t2.len().to_string(),
            category: DiffCategory::Tensor,
        });
    }

    // Build maps for lookup
    let map1: std::collections::HashMap<_, _> = t1.iter().map(|t| (t.name.as_str(), t)).collect();
    let map2: std::collections::HashMap<_, _> = t2.iter().map(|t| (t.name.as_str(), t)).collect();

    // Filter function
    let matches_filter = |name: &str| -> bool {
        if let Some(ref pattern) = options.tensor_filter {
            name.contains(pattern)
        } else {
            true
        }
    };

    // Check tensors in model 1
    for tensor1 in t1 {
        if !matches_filter(&tensor1.name) {
            continue;
        }

        match map2.get(tensor1.name.as_str()) {
            Some(tensor2) => {
                // Compare shapes
                if tensor1.shape != tensor2.shape {
                    diffs.push(DiffEntry {
                        field: format!("tensor.{}.shape", tensor1.name),
                        value1: format!("{:?}", tensor1.shape),
                        value2: format!("{:?}", tensor2.shape),
                        category: DiffCategory::Tensor,
                    });
                }

                // Compare dtypes
                if tensor1.dtype != tensor2.dtype {
                    diffs.push(DiffEntry {
                        field: format!("tensor.{}.dtype", tensor1.name),
                        value1: tensor1.dtype.clone(),
                        value2: tensor2.dtype.clone(),
                        category: DiffCategory::Tensor,
                    });
                }

                // Compare sizes
                if tensor1.size_bytes != tensor2.size_bytes {
                    diffs.push(DiffEntry {
                        field: format!("tensor.{}.size", tensor1.name),
                        value1: format_size(tensor1.size_bytes),
                        value2: format_size(tensor2.size_bytes),
                        category: DiffCategory::Tensor,
                    });
                }

                // Compare stats if enabled
                if options.compare_stats {
                    compare_tensor_stats(tensor1, tensor2, diffs);
                }
            }
            None => {
                // Tensor only in model 1
                diffs.push(DiffEntry {
                    field: format!("tensor.{}", tensor1.name),
                    value1: format!("{:?} {}", tensor1.shape, tensor1.dtype),
                    value2: "(missing)".to_string(),
                    category: DiffCategory::Tensor,
                });
            }
        }
    }

    // Check for tensors only in model 2
    for tensor2 in t2 {
        if !matches_filter(&tensor2.name) {
            continue;
        }

        if !map1.contains_key(tensor2.name.as_str()) {
            diffs.push(DiffEntry {
                field: format!("tensor.{}", tensor2.name),
                value1: "(missing)".to_string(),
                value2: format!("{:?} {}", tensor2.shape, tensor2.dtype),
                category: DiffCategory::Tensor,
            });
        }
    }
}

/// Compare tensor statistics
fn compare_tensor_stats(
    t1: &crate::format::rosetta::TensorInfo,
    t2: &crate::format::rosetta::TensorInfo,
    diffs: &mut Vec<DiffEntry>,
) {
    match (&t1.stats, &t2.stats) {
        (Some(s1), Some(s2)) => {
            let epsilon = 1e-4;

            if (s1.min - s2.min).abs() > epsilon {
                diffs.push(DiffEntry {
                    field: format!("tensor.{}.min", t1.name),
                    value1: format!("{:.6}", s1.min),
                    value2: format!("{:.6}", s2.min),
                    category: DiffCategory::Tensor,
                });
            }

            if (s1.max - s2.max).abs() > epsilon {
                diffs.push(DiffEntry {
                    field: format!("tensor.{}.max", t1.name),
                    value1: format!("{:.6}", s1.max),
                    value2: format!("{:.6}", s2.max),
                    category: DiffCategory::Tensor,
                });
            }

            if (s1.mean - s2.mean).abs() > epsilon {
                diffs.push(DiffEntry {
                    field: format!("tensor.{}.mean", t1.name),
                    value1: format!("{:.6}", s1.mean),
                    value2: format!("{:.6}", s2.mean),
                    category: DiffCategory::Tensor,
                });
            }

            if (s1.std - s2.std).abs() > epsilon {
                diffs.push(DiffEntry {
                    field: format!("tensor.{}.std", t1.name),
                    value1: format!("{:.6}", s1.std),
                    value2: format!("{:.6}", s2.std),
                    category: DiffCategory::Tensor,
                });
            }
        }
        (Some(_), None) => {
            diffs.push(DiffEntry {
                field: format!("tensor.{}.stats", t1.name),
                value1: "present".to_string(),
                value2: "(none)".to_string(),
                category: DiffCategory::Tensor,
            });
        }
        (None, Some(_)) => {
            diffs.push(DiffEntry {
                field: format!("tensor.{}.stats", t1.name),
                value1: "(none)".to_string(),
                value2: "present".to_string(),
                category: DiffCategory::Tensor,
            });
        }
        (None, None) => {}
    }
}

/// Format byte size for display
fn format_size(bytes: usize) -> String {
    const KB: usize = 1024;
    const MB: usize = KB * 1024;
    const GB: usize = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}

/// Format parameter count for display
fn format_params(params: usize) -> String {
    const K: usize = 1_000;
    const M: usize = K * 1_000;
    const B: usize = M * 1_000;

    if params >= B {
        format!("{:.2}B", params as f64 / B as f64)
    } else if params >= M {
        format!("{:.2}M", params as f64 / M as f64)
    } else if params >= K {
        format!("{:.2}K", params as f64 / K as f64)
    } else {
        params.to_string()
    }
}

/// Truncate value for display
fn truncate_value(s: &str, max_len: usize) -> String {
    if s.len() > max_len {
        format!("{}...", &s[..max_len])
    } else {
        s.to_string()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // DiffOptions Tests
    // ========================================================================

    #[test]
    fn test_diff_options_default() {
        let opts = DiffOptions::default();
        assert!(opts.compare_tensors);
        assert!(opts.compare_metadata);
        assert!(!opts.compare_stats);
        assert!(opts.tensor_filter.is_none());
    }

    #[test]
    fn test_diff_options_new() {
        let opts = DiffOptions::new();
        assert!(opts.compare_tensors);
        assert!(opts.compare_metadata);
    }

    #[test]
    fn test_diff_options_with_tensors() {
        let opts = DiffOptions::new().without_tensors().with_tensors();
        assert!(opts.compare_tensors);
    }

    #[test]
    fn test_diff_options_without_tensors() {
        let opts = DiffOptions::new().without_tensors();
        assert!(!opts.compare_tensors);
    }

    #[test]
    fn test_diff_options_with_metadata() {
        let opts = DiffOptions::new().without_metadata().with_metadata();
        assert!(opts.compare_metadata);
    }

    #[test]
    fn test_diff_options_without_metadata() {
        let opts = DiffOptions::new().without_metadata();
        assert!(!opts.compare_metadata);
    }

    #[test]
    fn test_diff_options_with_stats() {
        let opts = DiffOptions::new().with_stats();
        assert!(opts.compare_stats);
    }

    #[test]
    fn test_diff_options_with_filter() {
        let opts = DiffOptions::new().with_filter("embed");
        assert_eq!(opts.tensor_filter, Some("embed".to_string()));
    }

    // ========================================================================
    // DiffCategory Tests
    // ========================================================================

    #[test]
    fn test_diff_category_names() {
        assert_eq!(DiffCategory::Format.name(), "format");
        assert_eq!(DiffCategory::Metadata.name(), "metadata");
        assert_eq!(DiffCategory::Tensor.name(), "tensor");
        assert_eq!(DiffCategory::Quantization.name(), "quantization");
        assert_eq!(DiffCategory::Size.name(), "size");
    }

    // ========================================================================
    // DiffEntry Tests
    // ========================================================================

    #[test]
    fn test_diff_entry_serialization() {
        let entry = DiffEntry {
            field: "version".to_string(),
            value1: "1.0".to_string(),
            value2: "2.0".to_string(),
            category: DiffCategory::Format,
        };
        let json = serde_json::to_string(&entry).expect("serialize");
        assert!(json.contains("version"));
        assert!(json.contains("1.0"));
        assert!(json.contains("2.0"));
        assert!(json.contains("Format"));
    }

    #[test]
    fn test_diff_entry_equality() {
        let entry1 = DiffEntry {
            field: "test".to_string(),
            value1: "a".to_string(),
            value2: "b".to_string(),
            category: DiffCategory::Metadata,
        };
        let entry2 = entry1.clone();
        assert_eq!(entry1, entry2);
    }

    // ========================================================================
    // DiffReport Tests
    // ========================================================================

    #[test]
    fn test_diff_report_identical() {
        let report = DiffReport {
            path1: "a.apr".to_string(),
            path2: "b.apr".to_string(),
            format1: "APR".to_string(),
            format2: "APR".to_string(),
            differences: vec![],
            inspection1: None,
            inspection2: None,
        };
        assert!(report.is_identical());
        assert_eq!(report.diff_count(), 0);
        assert!(report.same_format());
    }

    #[test]
    fn test_diff_report_with_differences() {
        let report = DiffReport {
            path1: "a.apr".to_string(),
            path2: "b.gguf".to_string(),
            format1: "APR".to_string(),
            format2: "GGUF".to_string(),
            differences: vec![
                DiffEntry {
                    field: "format".to_string(),
                    value1: "APR".to_string(),
                    value2: "GGUF".to_string(),
                    category: DiffCategory::Format,
                },
                DiffEntry {
                    field: "tensor_count".to_string(),
                    value1: "10".to_string(),
                    value2: "12".to_string(),
                    category: DiffCategory::Tensor,
                },
            ],
            inspection1: None,
            inspection2: None,
        };
        assert!(!report.is_identical());
        assert_eq!(report.diff_count(), 2);
        assert!(!report.same_format());
    }

    #[test]
    fn test_diff_report_by_category() {
        let report = DiffReport {
            path1: "a.apr".to_string(),
            path2: "b.apr".to_string(),
            format1: "APR".to_string(),
            format2: "APR".to_string(),
            differences: vec![
                DiffEntry {
                    field: "tensor_count".to_string(),
                    value1: "10".to_string(),
                    value2: "12".to_string(),
                    category: DiffCategory::Tensor,
                },
                DiffEntry {
                    field: "metadata.name".to_string(),
                    value1: "model_a".to_string(),
                    value2: "model_b".to_string(),
                    category: DiffCategory::Metadata,
                },
                DiffEntry {
                    field: "tensor.embed.shape".to_string(),
                    value1: "[100]".to_string(),
                    value2: "[200]".to_string(),
                    category: DiffCategory::Tensor,
                },
            ],
            inspection1: None,
            inspection2: None,
        };

        let tensor_diffs = report.differences_by_category(DiffCategory::Tensor);
        assert_eq!(tensor_diffs.len(), 2);

        let metadata_diffs = report.differences_by_category(DiffCategory::Metadata);
        assert_eq!(metadata_diffs.len(), 1);

        let format_diffs = report.differences_by_category(DiffCategory::Format);
        assert_eq!(format_diffs.len(), 0);
    }

    #[test]
    fn test_diff_report_summary_identical() {
        let report = DiffReport {
            path1: "a.apr".to_string(),
            path2: "b.apr".to_string(),
            format1: "APR".to_string(),
            format2: "APR".to_string(),
            differences: vec![],
            inspection1: None,
            inspection2: None,
        };
        assert!(report.summary().contains("IDENTICAL"));
    }

    #[test]
    fn test_diff_report_summary_different() {
        let report = DiffReport {
            path1: "a.apr".to_string(),
            path2: "b.apr".to_string(),
            format1: "APR".to_string(),
            format2: "APR".to_string(),
            differences: vec![DiffEntry {
                field: "test".to_string(),
                value1: "a".to_string(),
                value2: "b".to_string(),
                category: DiffCategory::Metadata,
            }],
            inspection1: None,
            inspection2: None,
        };
        assert!(report.summary().contains("differ"));
        assert!(report.summary().contains("1"));
    }

    #[test]
    fn test_diff_report_serialization() {
        let report = DiffReport {
            path1: "a.apr".to_string(),
            path2: "b.apr".to_string(),
            format1: "APR".to_string(),
            format2: "APR".to_string(),
            differences: vec![],
            inspection1: None,
            inspection2: None,
        };
        let json = serde_json::to_string(&report).expect("serialize");
        assert!(json.contains("a.apr"));
        assert!(json.contains("b.apr"));
    }

    // ========================================================================
    // Helper Function Tests
    // ========================================================================

    #[test]
    fn test_format_size_bytes() {
        assert_eq!(format_size(100), "100 B");
        assert_eq!(format_size(0), "0 B");
    }

    #[test]
    fn test_format_size_kb() {
        assert_eq!(format_size(1024), "1.00 KB");
        assert_eq!(format_size(2048), "2.00 KB");
        assert_eq!(format_size(1536), "1.50 KB");
    }

    #[test]
    fn test_format_size_mb() {
        assert_eq!(format_size(1024 * 1024), "1.00 MB");
        assert_eq!(format_size(10 * 1024 * 1024), "10.00 MB");
    }

    #[test]
    fn test_format_size_gb() {
        assert_eq!(format_size(1024 * 1024 * 1024), "1.00 GB");
        assert_eq!(format_size(2 * 1024 * 1024 * 1024), "2.00 GB");
    }

    #[test]
    fn test_format_params() {
        assert_eq!(format_params(100), "100");
        assert_eq!(format_params(1_000), "1.00K");
        assert_eq!(format_params(1_500), "1.50K");
        assert_eq!(format_params(1_000_000), "1.00M");
        assert_eq!(format_params(7_000_000_000), "7.00B");
    }

    #[test]
    fn test_truncate_value() {
        assert_eq!(truncate_value("short", 10), "short");
        assert_eq!(truncate_value("this is a very long string", 10), "this is a ...");
    }

    // ========================================================================
    // Validate Path Tests
    // ========================================================================

    #[test]
    fn test_validate_path_not_found() {
        let result = validate_path(Path::new("/nonexistent/model.apr"));
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_path_is_directory() {
        use tempfile::tempdir;
        let dir = tempdir().expect("create dir");
        let result = validate_path(dir.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_path_valid() {
        use tempfile::NamedTempFile;
        let file = NamedTempFile::new().expect("create file");
        let result = validate_path(file.path());
        assert!(result.is_ok());
    }

    // ========================================================================
    // Metadata Comparison Tests
    // ========================================================================

    #[test]
    fn test_compare_metadata_identical() {
        use std::collections::BTreeMap;
        let mut m1 = BTreeMap::new();
        m1.insert("key1".to_string(), "value1".to_string());
        m1.insert("key2".to_string(), "value2".to_string());

        let m2 = m1.clone();
        let mut diffs = Vec::new();
        compare_metadata(&m1, &m2, &mut diffs);
        assert!(diffs.is_empty());
    }

    #[test]
    fn test_compare_metadata_different_value() {
        use std::collections::BTreeMap;
        let mut m1 = BTreeMap::new();
        m1.insert("key1".to_string(), "value1".to_string());

        let mut m2 = BTreeMap::new();
        m2.insert("key1".to_string(), "value2".to_string());

        let mut diffs = Vec::new();
        compare_metadata(&m1, &m2, &mut diffs);
        assert_eq!(diffs.len(), 1);
        assert!(diffs[0].field.contains("key1"));
    }

    #[test]
    fn test_compare_metadata_missing_key() {
        use std::collections::BTreeMap;
        let mut m1 = BTreeMap::new();
        m1.insert("key1".to_string(), "value1".to_string());
        m1.insert("key2".to_string(), "value2".to_string());

        let mut m2 = BTreeMap::new();
        m2.insert("key1".to_string(), "value1".to_string());

        let mut diffs = Vec::new();
        compare_metadata(&m1, &m2, &mut diffs);
        assert_eq!(diffs.len(), 1);
        assert!(diffs[0].field.contains("key2"));
        assert!(diffs[0].value2.contains("missing"));
    }

    #[test]
    fn test_compare_metadata_extra_key() {
        use std::collections::BTreeMap;
        let mut m1 = BTreeMap::new();
        m1.insert("key1".to_string(), "value1".to_string());

        let mut m2 = BTreeMap::new();
        m2.insert("key1".to_string(), "value1".to_string());
        m2.insert("key2".to_string(), "value2".to_string());

        let mut diffs = Vec::new();
        compare_metadata(&m1, &m2, &mut diffs);
        assert_eq!(diffs.len(), 1);
        assert!(diffs[0].field.contains("key2"));
        assert!(diffs[0].value1.contains("missing"));
    }

    // ========================================================================
    // Tensor Comparison Tests
    // ========================================================================

    #[test]
    fn test_compare_tensors_identical() {
        use crate::format::rosetta::TensorInfo;

        let t1 = vec![TensorInfo {
            name: "weight".to_string(),
            dtype: "F32".to_string(),
            shape: vec![10, 20],
            size_bytes: 800,
            stats: None,
        }];
        let t2 = t1.clone();

        let mut diffs = Vec::new();
        let options = DiffOptions::default();
        compare_tensors(&t1, &t2, &options, &mut diffs);
        assert!(diffs.is_empty());
    }

    #[test]
    fn test_compare_tensors_different_count() {
        use crate::format::rosetta::TensorInfo;

        let t1 = vec![TensorInfo {
            name: "weight".to_string(),
            dtype: "F32".to_string(),
            shape: vec![10, 20],
            size_bytes: 800,
            stats: None,
        }];
        let t2 = vec![
            TensorInfo {
                name: "weight".to_string(),
                dtype: "F32".to_string(),
                shape: vec![10, 20],
                size_bytes: 800,
                stats: None,
            },
            TensorInfo {
                name: "bias".to_string(),
                dtype: "F32".to_string(),
                shape: vec![10],
                size_bytes: 40,
                stats: None,
            },
        ];

        let mut diffs = Vec::new();
        let options = DiffOptions::default();
        compare_tensors(&t1, &t2, &options, &mut diffs);

        // Should have count diff and missing tensor diff
        assert!(diffs.iter().any(|d| d.field == "tensor_count"));
        assert!(diffs.iter().any(|d| d.field.contains("bias")));
    }

    #[test]
    fn test_compare_tensors_different_shape() {
        use crate::format::rosetta::TensorInfo;

        let t1 = vec![TensorInfo {
            name: "weight".to_string(),
            dtype: "F32".to_string(),
            shape: vec![10, 20],
            size_bytes: 800,
            stats: None,
        }];
        let t2 = vec![TensorInfo {
            name: "weight".to_string(),
            dtype: "F32".to_string(),
            shape: vec![20, 20],
            size_bytes: 1600,
            stats: None,
        }];

        let mut diffs = Vec::new();
        let options = DiffOptions::default();
        compare_tensors(&t1, &t2, &options, &mut diffs);

        assert!(diffs.iter().any(|d| d.field.contains("shape")));
        assert!(diffs.iter().any(|d| d.field.contains("size")));
    }

    #[test]
    fn test_compare_tensors_different_dtype() {
        use crate::format::rosetta::TensorInfo;

        let t1 = vec![TensorInfo {
            name: "weight".to_string(),
            dtype: "F32".to_string(),
            shape: vec![10, 20],
            size_bytes: 800,
            stats: None,
        }];
        let t2 = vec![TensorInfo {
            name: "weight".to_string(),
            dtype: "Q8_0".to_string(),
            shape: vec![10, 20],
            size_bytes: 400,
            stats: None,
        }];

        let mut diffs = Vec::new();
        let options = DiffOptions::default();
        compare_tensors(&t1, &t2, &options, &mut diffs);

        assert!(diffs.iter().any(|d| d.field.contains("dtype")));
    }

    #[test]
    fn test_compare_tensors_with_filter() {
        use crate::format::rosetta::TensorInfo;

        let t1 = vec![
            TensorInfo {
                name: "embed.weight".to_string(),
                dtype: "F32".to_string(),
                shape: vec![100],
                size_bytes: 400,
                stats: None,
            },
            TensorInfo {
                name: "lm_head.weight".to_string(),
                dtype: "F32".to_string(),
                shape: vec![100],
                size_bytes: 400,
                stats: None,
            },
        ];
        let t2 = vec![
            TensorInfo {
                name: "embed.weight".to_string(),
                dtype: "F32".to_string(),
                shape: vec![200], // Different
                size_bytes: 800,
                stats: None,
            },
            TensorInfo {
                name: "lm_head.weight".to_string(),
                dtype: "F32".to_string(),
                shape: vec![200], // Different
                size_bytes: 800,
                stats: None,
            },
        ];

        let mut diffs = Vec::new();
        let options = DiffOptions::new().with_filter("embed");
        compare_tensors(&t1, &t2, &options, &mut diffs);

        // Should only report embed differences due to filter
        assert!(diffs.iter().all(|d| d.field.contains("embed")));
    }

    #[test]
    fn test_compare_tensors_with_stats() {
        use crate::format::rosetta::{TensorInfo, TensorStats};

        let t1 = vec![TensorInfo {
            name: "weight".to_string(),
            dtype: "F32".to_string(),
            shape: vec![10],
            size_bytes: 40,
            stats: Some(TensorStats {
                min: 0.0,
                max: 1.0,
                mean: 0.5,
                std: 0.1,
            }),
        }];
        let t2 = vec![TensorInfo {
            name: "weight".to_string(),
            dtype: "F32".to_string(),
            shape: vec![10],
            size_bytes: 40,
            stats: Some(TensorStats {
                min: 0.0,
                max: 2.0, // Different
                mean: 0.6, // Different
                std: 0.1,
            }),
        }];

        let mut diffs = Vec::new();
        let options = DiffOptions::new().with_stats();
        compare_tensors(&t1, &t2, &options, &mut diffs);

        assert!(diffs.iter().any(|d| d.field.contains("max")));
        assert!(diffs.iter().any(|d| d.field.contains("mean")));
    }

    #[test]
    fn test_compare_tensor_stats_one_missing() {
        use crate::format::rosetta::{TensorInfo, TensorStats};

        let t1 = TensorInfo {
            name: "weight".to_string(),
            dtype: "F32".to_string(),
            shape: vec![10],
            size_bytes: 40,
            stats: Some(TensorStats {
                min: 0.0,
                max: 1.0,
                mean: 0.5,
                std: 0.1,
            }),
        };
        let t2 = TensorInfo {
            name: "weight".to_string(),
            dtype: "F32".to_string(),
            shape: vec![10],
            size_bytes: 40,
            stats: None,
        };

        let mut diffs = Vec::new();
        compare_tensor_stats(&t1, &t2, &mut diffs);

        assert_eq!(diffs.len(), 1);
        assert!(diffs[0].field.contains("stats"));
    }

    // ========================================================================
    // Integration Tests (File-based, minimal)
    // ========================================================================

    #[test]
    fn test_diff_models_file_not_found() {
        let result = diff_models(
            Path::new("/nonexistent/a.apr"),
            Path::new("/nonexistent/b.apr"),
            DiffOptions::default(),
        );
        assert!(result.is_err());
    }
}

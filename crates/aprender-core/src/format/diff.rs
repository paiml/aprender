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
pub fn diff_models<P: AsRef<Path>>(path1: P, path2: P, options: DiffOptions) -> Result<DiffReport> {
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
            value1: i1
                .architecture
                .clone()
                .unwrap_or_else(|| "(none)".to_string()),
            value2: i2
                .architecture
                .clone()
                .unwrap_or_else(|| "(none)".to_string()),
            category: DiffCategory::Metadata,
        });
    }

    // Quantization
    if i1.quantization != i2.quantization {
        diffs.push(DiffEntry {
            field: "quantization".to_string(),
            value1: i1
                .quantization
                .clone()
                .unwrap_or_else(|| "(none)".to_string()),
            value2: i2
                .quantization
                .clone()
                .unwrap_or_else(|| "(none)".to_string()),
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

include!("diff_gguf_name_map.rs");

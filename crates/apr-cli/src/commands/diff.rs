//! Diff command implementation (Thin Shim)
//!
//! Compares two model files across APR, GGUF, and SafeTensors formats.
//! This is a thin CLI wrapper around the library's diff functions.
//!
//! Toyota Way: Kaizen - Continuous improvement through comparison.
//!
//! # TOOL-APR-002 Library Extraction
//!
//! All diff logic lives in `aprender::format::diff`. This CLI only handles:
//! - Path validation
//! - Option parsing
//! - Output formatting (text/JSON)

use crate::error::CliError;
use crate::output;
use aprender::format::diff::{diff_models, DiffCategory, DiffOptions, DiffReport};
use aprender::format::rosetta::RosettaStone;
use colored::Colorize;
use serde::Serialize;
use std::collections::HashMap;
use std::path::Path;

// ============================================================================
// Serializable Types (for JSON output)
// ============================================================================

/// Diff entry for JSON output (mirrors library type)
#[derive(Serialize)]
struct DiffEntryJson {
    field: String,
    file1_value: String,
    file2_value: String,
    category: String,
    /// GH-256: Alias for category using parity-checker-compatible names
    /// Maps: quantization→dtype, size→size, metadata→metadata, format→format, tensor→data
    #[serde(rename = "type")]
    diff_type: String,
}

/// GH-256: Per-category diff counts for structured analysis
#[derive(Serialize)]
struct CategoryCounts {
    format: usize,
    metadata: usize,
    tensor: usize,
    quantization: usize,
    size: usize,
}

/// Diff result for JSON output
#[derive(Serialize)]
struct DiffResultJson {
    file1: String,
    file2: String,
    format1: String,
    format2: String,
    identical: bool,
    difference_count: usize,
    /// GH-256: Structural diffs exclude quantization and size (expected for int4 vs int8)
    structural_diffs: usize,
    /// GH-256: Per-category breakdown
    category_counts: CategoryCounts,
    differences: Vec<DiffEntryJson>,
}

impl From<&DiffReport> for DiffResultJson {
    fn from(report: &DiffReport) -> Self {
        let format_count = report.differences_by_category(DiffCategory::Format).len();
        let metadata_count = report.differences_by_category(DiffCategory::Metadata).len();
        let tensor_count = report.differences_by_category(DiffCategory::Tensor).len();
        let quant_count = report
            .differences_by_category(DiffCategory::Quantization)
            .len();
        let size_count = report.differences_by_category(DiffCategory::Size).len();

        // GH-256: Structural = format + metadata + tensor (excludes quantization + size)
        let structural_diffs = format_count + metadata_count + tensor_count;

        Self {
            file1: report.path1.clone(),
            file2: report.path2.clone(),
            format1: report.format1.clone(),
            format2: report.format2.clone(),
            identical: report.is_identical(),
            difference_count: report.diff_count(),
            structural_diffs,
            category_counts: CategoryCounts {
                format: format_count,
                metadata: metadata_count,
                tensor: tensor_count,
                quantization: quant_count,
                size: size_count,
            },
            differences: report
                .differences
                .iter()
                .map(|d| {
                    let cat_name = d.category.name().to_string();
                    // GH-256: Map category to parity-checker-compatible type names
                    // dtype/size/data are non-structural; metadata treated as data
                    // since metadata diffs (e.g. model_name) are expected between quant variants
                    let diff_type = match d.category {
                        DiffCategory::Quantization => "dtype".to_string(),
                        DiffCategory::Size => "size".to_string(),
                        DiffCategory::Tensor => "data".to_string(),
                        DiffCategory::Metadata => "data".to_string(),
                        DiffCategory::Format => "format".to_string(),
                    };
                    DiffEntryJson {
                        field: d.field.clone(),
                        file1_value: d.value1.clone(),
                        file2_value: d.value2.clone(),
                        category: cat_name,
                        diff_type,
                    }
                })
                .collect(),
        }
    }
}

/// Tensor value comparison statistics
#[derive(Debug, Clone, Serialize)]
struct TensorValueStats {
    name: String,
    shape_a: Vec<usize>,
    shape_b: Vec<usize>,
    element_count: usize,
    mean_diff: f32,
    max_diff: f32,
    rmse: f32,
    cosine_similarity: f32,
    identical_count: usize,
    small_diff_count: usize,  // |diff| < 0.001
    medium_diff_count: usize, // 0.001 <= |diff| < 0.01
    large_diff_count: usize,  // |diff| >= 0.01
    status: TensorDiffStatus,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq)]
enum TensorDiffStatus {
    Identical,
    NearlyIdentical, // max_diff < 0.001
    SmallDiff,       // max_diff < 0.01
    MediumDiff,      // max_diff < 0.1
    LargeDiff,       // max_diff < 1.0
    Transposed,      // shapes are transposed but values match (linear comparison)
    Critical,        // max_diff >= 1.0 or incompatible shape mismatch
}

impl TensorDiffStatus {
    fn from_diff_info(
        max_diff: f32,
        shape_a: &[usize],
        shape_b: &[usize],
        identical_count: usize,
        element_count: usize,
    ) -> Self {
        let shape_match = shape_a == shape_b;

        // Check if shapes are transposed (2D tensors with swapped dimensions)
        let is_transpose = !shape_match
            && shape_a.len() == 2
            && shape_b.len() == 2
            && shape_a[0] == shape_b[1]
            && shape_a[1] == shape_b[0];

        if is_transpose {
            // If transposed and values are mostly identical in linear order, it's just a layout diff
            let ident_ratio = identical_count as f64 / element_count as f64;
            if ident_ratio > 0.99 {
                return TensorDiffStatus::Transposed;
            }
            // Otherwise, transposed + value differences is concerning but not "critical"
            if max_diff < 0.1 {
                return TensorDiffStatus::MediumDiff;
            }
        }

        if !shape_match && !is_transpose {
            // Incompatible shapes
            return TensorDiffStatus::Critical;
        }

        // Shape matches or is transposed - classify by value differences
        if max_diff == 0.0 {
            TensorDiffStatus::Identical
        } else if max_diff < 0.001 {
            TensorDiffStatus::NearlyIdentical
        } else if max_diff < 0.01 {
            TensorDiffStatus::SmallDiff
        } else if max_diff < 0.1 {
            TensorDiffStatus::MediumDiff
        } else if max_diff < 1.0 {
            TensorDiffStatus::LargeDiff
        } else {
            TensorDiffStatus::Critical
        }
    }

    fn colored_string(self) -> colored::ColoredString {
        match self {
            TensorDiffStatus::Identical => "IDENTICAL".green().bold(),
            TensorDiffStatus::NearlyIdentical => "~IDENTICAL".green(),
            TensorDiffStatus::SmallDiff => "SMALL".blue(),
            TensorDiffStatus::MediumDiff => "MEDIUM".yellow(),
            TensorDiffStatus::LargeDiff => "LARGE".red(),
            TensorDiffStatus::Transposed => "TRANSPOSED".cyan(),
            TensorDiffStatus::Critical => "CRITICAL".red().bold(),
        }
    }
}

// ============================================================================
// Main Entry Point
// ============================================================================

/// Run the diff command
///
/// This is a thin shim that delegates to the library's diff functions.
/// All actual logic is in `aprender::format::diff`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run(
    path1: &Path,
    path2: &Path,
    show_weights: bool,
    compare_values: bool,
    filter: Option<&str>,
    limit: usize,
    transpose_aware: bool,
    json_output: bool,
) -> Result<(), CliError> {
    // Validate paths exist
    validate_paths(path1, path2)?;

    if compare_values {
        // Run tensor value comparison
        run_tensor_value_diff(path1, path2, filter, limit, transpose_aware, json_output)
    } else {
        // Run standard metadata/structure diff
        let options = DiffOptions::new().with_tensors().with_metadata();
        let report = diff_models(path1, path2, options)
            .map_err(|e| CliError::InvalidFormat(format!("Failed to diff models: {e}")))?;

        if json_output {
            output_json(&report);
        } else {
            output_text(&report, show_weights);
        }
        Ok(())
    }
}

// ============================================================================
// Tensor Value Comparison
// ============================================================================

// serde_json::json!() macro uses infallible unwrap internally
#[allow(clippy::disallowed_methods)]
fn run_tensor_value_diff(
    path1: &Path,
    path2: &Path,
    filter: Option<&str>,
    limit: usize,
    transpose_aware: bool,
    json_output: bool,
) -> Result<(), CliError> {
    let rosetta = RosettaStone::new();

    // Inspect both models
    let report1 = rosetta
        .inspect(path1)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to inspect model 1: {e}")))?;
    let report2 = rosetta
        .inspect(path2)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to inspect model 2: {e}")))?;

    // Build tensor maps
    let tensors1: HashMap<String, _> = report1
        .tensors
        .iter()
        .map(|t| (normalize_tensor_name(&t.name), t))
        .collect();
    let tensors2: HashMap<String, _> = report2
        .tensors
        .iter()
        .map(|t| (normalize_tensor_name(&t.name), t))
        .collect();

    // Find common tensors
    let mut common_names: Vec<_> = tensors1
        .keys()
        .filter(|k| tensors2.contains_key(*k))
        .cloned()
        .collect();
    common_names.sort();

    // Apply filter
    if let Some(pattern) = filter {
        common_names.retain(|n| n.contains(pattern));
    }

    // Limit number of tensors to compare
    common_names.truncate(limit);

    if !json_output {
        print_diff_header(path1, path2);
    }

    let mut results: Vec<TensorValueStats> = Vec::new();
    let mut critical_count = 0;
    let mut large_count = 0;
    let mut medium_count = 0;
    let mut transposed_count = 0;
    let mut identical_count = 0;

    for name in &common_names {
        let t1 = tensors1.get(name).expect("tensor exists");
        let t2 = tensors2.get(name).expect("tensor exists");

        // Load actual tensor data
        let data1 = rosetta.load_tensor_f32(path1, &t1.name).map_err(|e| {
            CliError::ValidationFailed(format!("Failed to load tensor {}: {e}", t1.name))
        })?;
        let data2 = rosetta.load_tensor_f32(path2, &t2.name).map_err(|e| {
            CliError::ValidationFailed(format!("Failed to load tensor {}: {e}", t2.name))
        })?;

        let stats =
            compute_tensor_diff_stats(name, &t1.shape, &t2.shape, &data1, &data2, transpose_aware);

        match stats.status {
            TensorDiffStatus::Critical => critical_count += 1,
            TensorDiffStatus::LargeDiff => large_count += 1,
            TensorDiffStatus::MediumDiff => medium_count += 1,
            TensorDiffStatus::Transposed => transposed_count += 1,
            TensorDiffStatus::Identical | TensorDiffStatus::NearlyIdentical => identical_count += 1,
            TensorDiffStatus::SmallDiff => {}
        }

        if !json_output {
            print_tensor_diff_row(&stats);
        }

        results.push(stats);
    }

    if json_output {
        print_diff_json(
            path1,
            path2,
            &results,
            identical_count,
            transposed_count,
            critical_count,
            large_count,
            medium_count,
        );
    } else {
        print_diff_summary(
            &results,
            identical_count,
            transposed_count,
            critical_count,
            large_count,
            medium_count,
        );
    }

    Ok(())
}

/// Print the diff box header.
fn print_diff_header(path1: &Path, path2: &Path) {
    let sep = "╠══════════════════════════════════════════════════════════════════════════════╣";
    println!(
        "{}",
        "╔══════════════════════════════════════════════════════════════════════════════╗".cyan()
    );
    println!(
        "{}",
        "║           TENSOR VALUE DIFF (Statistical Comparison)                         ║".cyan()
    );
    println!("{}", sep.cyan());
    println!(
        "║ Model A: {:<66} ║",
        truncate_path(&path1.display().to_string(), 66)
    );
    println!(
        "║ Model B: {:<66} ║",
        truncate_path(&path2.display().to_string(), 66)
    );
    println!("{}", sep.cyan());
    println!(
        "║ Legend: {} {} {} {} {} {} ║",
        "IDENTICAL".green().bold(),
        "~IDENT".green(),
        "SMALL".blue(),
        "MEDIUM".yellow(),
        "LARGE".red(),
        "CRITICAL".red().bold()
    );
    println!("{}", sep.cyan());
}

/// Print diff results as JSON.
// serde_json::json!() macro uses infallible unwrap internally
#[allow(clippy::disallowed_methods, clippy::too_many_arguments)]
fn print_diff_json(
    path1: &Path,
    path2: &Path,
    results: &[TensorValueStats],
    identical: usize,
    transposed: usize,
    critical: usize,
    large: usize,
    medium: usize,
) {
    let json = serde_json::json!({
        "model_a": path1.display().to_string(),
        "model_b": path2.display().to_string(),
        "tensors_compared": results.len(),
        "identical_count": identical,
        "transposed_count": transposed,
        "critical_count": critical,
        "large_count": large,
        "medium_count": medium,
        "results": results,
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&json).unwrap_or_default()
    );
}

include!("diff_accumulator.rs");
include!("diff_part_03.rs");
include!("diff_part_04.rs");

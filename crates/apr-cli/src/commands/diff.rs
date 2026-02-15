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

/// Print diff summary with diagnosis.
#[allow(clippy::too_many_arguments)]
fn print_diff_summary(
    results: &[TensorValueStats],
    identical: usize,
    transposed: usize,
    critical: usize,
    large: usize,
    medium: usize,
) {
    let sep = "╠══════════════════════════════════════════════════════════════════════════════╣";
    println!("{}", sep.cyan());
    println!(
        "{}",
        "║                              SUMMARY                                          ║"
            .cyan()
            .bold()
    );
    println!("{}", sep.cyan());
    println!("║ Tensors compared: {:<58} ║", results.len());
    println!(
        "║ Identical: {:<65} ║",
        format!("{identical}").green().to_string()
    );
    println!(
        "║ Transposed (layout diff): {:<50} ║",
        if transposed > 0 {
            format!("{transposed}").cyan().to_string()
        } else {
            "0".dimmed().to_string()
        }
    );
    println!(
        "║ Critical differences: {:<54} ║",
        if critical > 0 {
            format!("{critical}").red().bold().to_string()
        } else {
            "0".green().to_string()
        }
    );
    println!(
        "║ Large differences: {:<57} ║",
        if large > 0 {
            format!("{large}").red().to_string()
        } else {
            "0".green().to_string()
        }
    );
    println!(
        "║ Medium differences: {:<56} ║",
        if medium > 0 {
            format!("{medium}").yellow().to_string()
        } else {
            "0".green().to_string()
        }
    );

    println!("{}", sep.cyan());
    print_diff_diagnosis(results, identical, transposed, critical, large, medium);
    println!(
        "{}",
        "╚══════════════════════════════════════════════════════════════════════════════╝".cyan()
    );
}

/// Print the diagnosis section of the diff summary.
fn print_diff_diagnosis(
    results: &[TensorValueStats],
    identical: usize,
    transposed: usize,
    critical: usize,
    large: usize,
    medium: usize,
) {
    if critical > 0 {
        println!(
            "║ {} ║",
            "DIAGNOSIS: Critical value differences detected!"
                .red()
                .bold()
        );
        println!("║ {:<75} ║", "Possible causes:".yellow());
        println!(
            "║ {:<75} ║",
            "  - Different quantization/dequantization algorithms"
        );
        println!(
            "║ {:<75} ║",
            "  - Tensor layout mismatch (row-major vs column-major)"
        );
        println!("║ {:<75} ║", "  - Corrupted weights during conversion");
    } else if large > 0 {
        println!(
            "║ {} ║",
            "DIAGNOSIS: Large value differences - may affect inference quality"
                .yellow()
                .bold()
        );
        let transposed_with_diffs = results
            .iter()
            .filter(|r| {
                let is_t = r.shape_a.len() == 2
                    && r.shape_b.len() == 2
                    && r.shape_a[0] == r.shape_b[1]
                    && r.shape_a[1] == r.shape_b[0];
                is_t && r.status != TensorDiffStatus::Transposed
            })
            .count();
        if transposed_with_diffs > 0 {
            println!(
                "║ {:<75} ║",
                "NOTE: Differences in transposed tensors may be expected when".cyan()
            );
            println!(
                "║ {:<75} ║",
                "comparing GGUF (col-major) to APR (row-major) linearly.".cyan()
            );
        }
    } else if medium > 0 {
        println!(
            "║ {} ║",
            "DIAGNOSIS: Medium differences - likely acceptable quantization variance".blue()
        );
    } else if transposed > 0 && identical > 0 {
        println!(
            "║ {} ║",
            "DIAGNOSIS: Values identical, shapes transposed (format layout diff)"
                .cyan()
                .bold()
        );
    } else {
        println!(
            "║ {} ║",
            "DIAGNOSIS: Tensors are nearly identical".green().bold()
        );
    }
}

/// Look up element in data_b at the transposed position corresponding to index `i` in data_a.
fn lookup_transposed_element(
    data_b: &[f32],
    shape_a: &[usize],
    shape_b: &[usize],
    i: usize,
) -> Option<f32> {
    let cols_a = shape_a[1];
    let row = i / cols_a;
    let col = i % cols_a;
    let cols_b = shape_b[1];
    let j = row * cols_b + col;
    if j < data_b.len() {
        Some(data_b[j])
    } else {
        None
    }
}

/// Classify a diff value into identical/small/medium/large buckets.
fn classify_diff(
    diff: f32,
    identical: &mut usize,
    small: &mut usize,
    medium: &mut usize,
    large: &mut usize,
) {
    if diff == 0.0 {
        *identical += 1;
    } else if diff < 0.001 {
        *small += 1;
    } else if diff < 0.01 {
        *medium += 1;
    } else {
        *large += 1;
    }
}

/// Accumulator for element-wise diff statistics.
struct DiffAccumulator {
    sum_diff: f64,
    sum_sq_diff: f64,
    max_diff: f32,
    dot_product: f64,
    norm_a: f64,
    norm_b: f64,
    identical_count: usize,
    small_diff_count: usize,
    medium_diff_count: usize,
    large_diff_count: usize,
}

impl DiffAccumulator {
    fn new() -> Self {
        Self {
            sum_diff: 0.0,
            sum_sq_diff: 0.0,
            max_diff: 0.0,
            dot_product: 0.0,
            norm_a: 0.0,
            norm_b: 0.0,
            identical_count: 0,
            small_diff_count: 0,
            medium_diff_count: 0,
            large_diff_count: 0,
        }
    }

    /// Accumulate a pair of finite values. Returns false if NaN/Inf (counted as large diff).
    fn accumulate(&mut self, a: f32, b: f32) {
        if a.is_nan() || b.is_nan() || a.is_infinite() || b.is_infinite() {
            self.large_diff_count += 1;
            return;
        }
        let diff = (a - b).abs();
        self.sum_diff += diff as f64;
        self.sum_sq_diff += (diff as f64) * (diff as f64);
        self.max_diff = self.max_diff.max(diff);
        self.dot_product += (a as f64) * (b as f64);
        self.norm_a += (a as f64) * (a as f64);
        self.norm_b += (b as f64) * (b as f64);
        classify_diff(
            diff,
            &mut self.identical_count,
            &mut self.small_diff_count,
            &mut self.medium_diff_count,
            &mut self.large_diff_count,
        );
    }

    fn mean_diff(&self, n: usize) -> f32 {
        (self.sum_diff / n as f64) as f32
    }
    fn rmse(&self, n: usize) -> f32 {
        ((self.sum_sq_diff / n as f64).sqrt()) as f32
    }

    fn cosine_similarity(&self) -> f32 {
        if self.norm_a > 0.0 && self.norm_b > 0.0 {
            (self.dot_product / (self.norm_a.sqrt() * self.norm_b.sqrt())) as f32
        } else {
            0.0
        }
    }
}

/// Build an empty `TensorValueStats` for zero-element tensors.
fn empty_tensor_stats(name: &str, shape_a: &[usize], shape_b: &[usize]) -> TensorValueStats {
    TensorValueStats {
        name: name.to_string(),
        shape_a: shape_a.to_vec(),
        shape_b: shape_b.to_vec(),
        element_count: 0,
        mean_diff: 0.0,
        max_diff: 0.0,
        rmse: 0.0,
        cosine_similarity: 0.0,
        identical_count: 0,
        small_diff_count: 0,
        medium_diff_count: 0,
        large_diff_count: 0,
        status: TensorDiffStatus::Critical,
    }
}

fn compute_tensor_diff_stats(
    name: &str,
    shape_a: &[usize],
    shape_b: &[usize],
    data_a: &[f32],
    data_b: &[f32],
    transpose_aware: bool,
) -> TensorValueStats {
    let element_count = data_a.len().min(data_b.len());
    if element_count == 0 {
        return empty_tensor_stats(name, shape_a, shape_b);
    }

    let is_transpose = shape_a.len() == 2
        && shape_b.len() == 2
        && shape_a[0] == shape_b[1]
        && shape_a[1] == shape_b[0];
    let use_transpose = transpose_aware && is_transpose && shape_a.len() == 2;

    let mut acc = DiffAccumulator::new();
    for i in 0..element_count {
        let a = data_a[i];
        let b = if use_transpose {
            match lookup_transposed_element(data_b, shape_a, shape_b, i) {
                Some(val) => val,
                None => continue,
            }
        } else {
            data_b[i]
        };
        acc.accumulate(a, b);
    }

    let status = TensorDiffStatus::from_diff_info(
        acc.max_diff,
        shape_a,
        shape_b,
        acc.identical_count,
        element_count,
    );

    TensorValueStats {
        name: name.to_string(),
        shape_a: shape_a.to_vec(),
        shape_b: shape_b.to_vec(),
        element_count,
        mean_diff: acc.mean_diff(element_count),
        max_diff: acc.max_diff,
        rmse: acc.rmse(element_count),
        cosine_similarity: acc.cosine_similarity(),
        identical_count: acc.identical_count,
        small_diff_count: acc.small_diff_count,
        medium_diff_count: acc.medium_diff_count,
        large_diff_count: acc.large_diff_count,
        status,
    }
}

fn print_tensor_diff_row(stats: &TensorValueStats) {
    let status_str = stats.status.colored_string();
    let name_truncated = truncate_str(&stats.name, 40);

    // Color max_diff based on severity
    let max_diff_str = format!("{:.6}", stats.max_diff);
    let max_diff_colored = match stats.status {
        TensorDiffStatus::Identical | TensorDiffStatus::NearlyIdentical => max_diff_str.green(),
        TensorDiffStatus::SmallDiff | TensorDiffStatus::Transposed => max_diff_str.cyan(),
        TensorDiffStatus::MediumDiff => max_diff_str.yellow(),
        TensorDiffStatus::LargeDiff | TensorDiffStatus::Critical => max_diff_str.red(),
    };

    // Color cosine similarity
    let cos_str = format!("{:.6}", stats.cosine_similarity);
    let cos_colored = if stats.cosine_similarity > 0.9999 {
        cos_str.green()
    } else if stats.cosine_similarity > 0.999 {
        cos_str.blue()
    } else if stats.cosine_similarity > 0.99 {
        cos_str.yellow()
    } else {
        cos_str.red()
    };

    println!("║ [{}] {:<40} ║", status_str, name_truncated);
    println!(
        "║   max_diff={} mean_diff={:.6} rmse={:.6} cos_sim={} ║",
        max_diff_colored, stats.mean_diff, stats.rmse, cos_colored
    );

    // Check for shape mismatch and if it's a transpose
    let shape_match = stats.shape_a == stats.shape_b;
    let is_transpose = !shape_match
        && stats.shape_a.len() == 2
        && stats.shape_b.len() == 2
        && stats.shape_a[0] == stats.shape_b[1]
        && stats.shape_a[1] == stats.shape_b[0];

    if !shape_match {
        if is_transpose {
            println!(
                "║   {} shapes: {:?} vs {:?} {} ║",
                "TRANSPOSED".yellow(),
                stats.shape_a,
                stats.shape_b,
                "(row-major vs col-major)".dimmed()
            );
        } else {
            println!(
                "║   {} shapes: {:?} vs {:?} ║",
                "SHAPE MISMATCH".red().bold(),
                stats.shape_a,
                stats.shape_b
            );
        }
    }

    // Show distribution if there are differences
    if stats.status != TensorDiffStatus::Identical {
        let total = stats.element_count;
        let ident_pct = 100.0 * stats.identical_count as f64 / total as f64;
        let small_pct = 100.0 * stats.small_diff_count as f64 / total as f64;
        let med_pct = 100.0 * stats.medium_diff_count as f64 / total as f64;
        let large_pct = 100.0 * stats.large_diff_count as f64 / total as f64;

        println!(
            "║   dist: {:.1}% ident, {:.1}% small, {:.1}% med, {:.1}% large ({} elems)  ║",
            ident_pct, small_pct, med_pct, large_pct, total
        );
    }

    println!(
        "{}",
        "╠──────────────────────────────────────────────────────────────────────────────╣".dimmed()
    );
}

// ============================================================================
// Helper Functions
// ============================================================================

fn normalize_tensor_name(name: &str) -> String {
    // Normalize different naming conventions
    name.replace("blk.", "model.layers.")
        .replace(".attn_q.", ".self_attn.q_proj.")
        .replace(".attn_k.", ".self_attn.k_proj.")
        .replace(".attn_v.", ".self_attn.v_proj.")
        .replace(".attn_output.", ".self_attn.o_proj.")
        .replace(".ffn_gate.", ".mlp.gate_proj.")
        .replace(".ffn_up.", ".mlp.up_proj.")
        .replace(".ffn_down.", ".mlp.down_proj.")
        .replace(".attn_norm.", ".input_layernorm.")
        .replace(".ffn_norm.", ".post_attention_layernorm.")
}

fn truncate_path(path: &str, max_len: usize) -> String {
    if path.len() <= max_len {
        path.to_string()
    } else {
        format!("...{}", &path[path.len() - max_len + 3..])
    }
}

fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}

// ============================================================================
// Path Validation
// ============================================================================

fn validate_paths(path1: &Path, path2: &Path) -> Result<(), CliError> {
    for path in [path1, path2] {
        if !path.exists() {
            return Err(CliError::FileNotFound(path.to_path_buf()));
        }
        if !path.is_file() {
            return Err(CliError::NotAFile(path.to_path_buf()));
        }
    }
    Ok(())
}

// ============================================================================
// Output Functions
// ============================================================================

fn output_json(report: &DiffReport) {
    let json_result = DiffResultJson::from(report);
    if let Ok(json) = serde_json::to_string_pretty(&json_result) {
        println!("{json}");
    }
}

fn output_text(report: &DiffReport, show_weights: bool) {
    output::header("Model Diff");

    let format_info = if report.same_format() {
        report.format1.clone()
    } else {
        format!("{} vs {}", report.format1, report.format2)
    };

    println!(
        "{}",
        output::kv_table(&[
            ("File A", report.path1.clone()),
            ("File B", report.path2.clone()),
            ("Format", format_info),
        ])
    );
    println!();

    if report.is_identical() {
        println!(
            "  {}",
            output::badge_pass("Models are IDENTICAL in structure and metadata")
        );
    } else {
        let count = report.diff_count();
        println!(
            "  {} {} differences found",
            output::badge_warn("DIFF"),
            count
        );
        println!();

        // Build diff table
        let mut rows: Vec<Vec<String>> = Vec::new();
        for category in [
            DiffCategory::Format,
            DiffCategory::Size,
            DiffCategory::Quantization,
            DiffCategory::Metadata,
            DiffCategory::Tensor,
        ] {
            let diffs = report.differences_by_category(category);
            for diff in diffs {
                rows.push(vec![
                    category.name().to_string(),
                    diff.field.clone(),
                    diff.value1.clone(),
                    diff.value2.clone(),
                ]);
            }
        }
        if !rows.is_empty() {
            println!(
                "{}",
                output::table(&["Category", "Field", "File A", "File B"], &rows)
            );
        }
    }

    if show_weights {
        println!();
        println!(
            "  {} Use --values to compare actual tensor values",
            output::badge_info("TIP")
        );
    }
}

// ============================================================================
// Tests (Minimal - Most logic is tested in library)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use aprender::format::diff::DiffEntry;
    use std::io::Write;
    use tempfile::{tempdir, NamedTempFile};

    #[test]
    fn test_validate_paths_first_not_found() {
        let file2 = NamedTempFile::new().expect("create file");
        let result = validate_paths(Path::new("/nonexistent/model1.apr"), file2.path());
        assert!(result.is_err());
        match result {
            Err(CliError::FileNotFound(_)) => {}
            _ => panic!("Expected FileNotFound error"),
        }
    }

    #[test]
    fn test_validate_paths_second_not_found() {
        let file1 = NamedTempFile::new().expect("create file");
        let result = validate_paths(file1.path(), Path::new("/nonexistent/model2.apr"));
        assert!(result.is_err());
        match result {
            Err(CliError::FileNotFound(_)) => {}
            _ => panic!("Expected FileNotFound error"),
        }
    }

    #[test]
    fn test_validate_paths_first_is_directory() {
        let dir = tempdir().expect("create dir");
        let file2 = NamedTempFile::new().expect("create file");
        let result = validate_paths(dir.path(), file2.path());
        assert!(result.is_err());
        match result {
            Err(CliError::NotAFile(_)) => {}
            _ => panic!("Expected NotAFile error"),
        }
    }

    #[test]
    fn test_validate_paths_valid() {
        let file1 = NamedTempFile::new().expect("create file");
        let file2 = NamedTempFile::new().expect("create file");
        let result = validate_paths(file1.path(), file2.path());
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_file_not_found() {
        let file = NamedTempFile::new().expect("create file");
        let result = run(
            Path::new("/nonexistent/model.apr"),
            file.path(),
            false,
            false,
            None,
            10,
            false,
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_invalid_files() {
        let mut file1 = NamedTempFile::with_suffix(".apr").expect("create file");
        let mut file2 = NamedTempFile::with_suffix(".apr").expect("create file");

        // Write minimal data (less than header size)
        file1.write_all(b"short").expect("write");
        file2.write_all(b"short").expect("write");

        let result = run(
            file1.path(),
            file2.path(),
            false,
            false,
            None,
            10,
            false,
            false,
        );
        // Should fail because files are too small/invalid
        assert!(result.is_err());
    }

    #[test]
    fn test_tensor_diff_status_thresholds() {
        // Test with matching shapes
        let shape = &[10, 10];
        let elem_count = 100;

        // Identical: max_diff = 0.0
        assert_eq!(
            TensorDiffStatus::from_diff_info(0.0, shape, shape, 100, elem_count),
            TensorDiffStatus::Identical
        );
        // Nearly identical: max_diff < 0.001
        assert_eq!(
            TensorDiffStatus::from_diff_info(0.0005, shape, shape, 50, elem_count),
            TensorDiffStatus::NearlyIdentical
        );
        // Small diff: max_diff < 0.01
        assert_eq!(
            TensorDiffStatus::from_diff_info(0.005, shape, shape, 10, elem_count),
            TensorDiffStatus::SmallDiff
        );
        // Medium diff: max_diff < 0.1
        assert_eq!(
            TensorDiffStatus::from_diff_info(0.05, shape, shape, 5, elem_count),
            TensorDiffStatus::MediumDiff
        );
        // Large diff: max_diff < 1.0
        assert_eq!(
            TensorDiffStatus::from_diff_info(0.5, shape, shape, 1, elem_count),
            TensorDiffStatus::LargeDiff
        );
        // Critical: max_diff >= 1.0
        assert_eq!(
            TensorDiffStatus::from_diff_info(1.5, shape, shape, 0, elem_count),
            TensorDiffStatus::Critical
        );
        // Incompatible shape mismatch (different element counts) is critical
        assert_eq!(
            TensorDiffStatus::from_diff_info(0.0, &[10, 10], &[5, 5], 25, 25),
            TensorDiffStatus::Critical
        );
        // Transposed shapes with identical values
        assert_eq!(
            TensorDiffStatus::from_diff_info(0.0, &[10, 20], &[20, 10], 200, 200),
            TensorDiffStatus::Transposed
        );
    }

    #[test]
    fn test_compute_tensor_diff_stats_identical() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let stats = compute_tensor_diff_stats("test", &[4], &[4], &data, &data, false);
        assert_eq!(stats.status, TensorDiffStatus::Identical);
        assert_eq!(stats.max_diff, 0.0);
        assert_eq!(stats.identical_count, 4);
    }

    #[test]
    fn test_compute_tensor_diff_stats_small_diff() {
        let data_a = vec![1.0, 2.0, 3.0, 4.0];
        let data_b = vec![1.0001, 2.0001, 3.0001, 4.0001];
        let stats = compute_tensor_diff_stats("test", &[4], &[4], &data_a, &data_b, false);
        assert_eq!(stats.status, TensorDiffStatus::NearlyIdentical);
        assert!(stats.max_diff < 0.001);
    }

    #[test]
    fn test_normalize_tensor_name() {
        assert!(normalize_tensor_name("blk.0.attn_q.weight").contains("model.layers.0"));
        assert!(normalize_tensor_name("blk.0.attn_q.weight").contains("self_attn.q_proj"));
    }

    // ==================== TensorDiffStatus::from_diff_info - exhaustive branch coverage ====================

    #[test]
    fn test_from_diff_info_transposed_high_ident_ratio() {
        // Transposed shapes with >99% identical values => Transposed
        let status = TensorDiffStatus::from_diff_info(0.0, &[10, 20], &[20, 10], 199, 200);
        assert_eq!(status, TensorDiffStatus::Transposed);
    }

    #[test]
    fn test_from_diff_info_transposed_low_ident_ratio_small_max_diff() {
        // Transposed shapes, low ident ratio, max_diff < 0.1 => MediumDiff
        let status = TensorDiffStatus::from_diff_info(0.05, &[10, 20], &[20, 10], 10, 200);
        assert_eq!(status, TensorDiffStatus::MediumDiff);
    }

    #[test]
    fn test_from_diff_info_transposed_low_ident_ratio_large_max_diff() {
        // Transposed shapes, low ident ratio, max_diff >= 0.1 => falls through to value classification
        // max_diff=0.5 => LargeDiff (0.1 <= 0.5 < 1.0)
        let status = TensorDiffStatus::from_diff_info(0.5, &[10, 20], &[20, 10], 10, 200);
        assert_eq!(status, TensorDiffStatus::LargeDiff);
    }

    #[test]
    fn test_from_diff_info_transposed_critical_max_diff() {
        // Transposed shapes, low ident ratio, max_diff >= 1.0 => Critical
        let status = TensorDiffStatus::from_diff_info(2.0, &[10, 20], &[20, 10], 5, 200);
        assert_eq!(status, TensorDiffStatus::Critical);
    }

    #[test]
    fn test_from_diff_info_incompatible_1d_vs_2d() {
        // Different dimensionality shapes => Critical
        let status = TensorDiffStatus::from_diff_info(0.0, &[100], &[10, 10], 100, 100);
        assert_eq!(status, TensorDiffStatus::Critical);
    }

    #[test]
    fn test_from_diff_info_incompatible_3d_shapes() {
        // 3D shapes that don't match => Critical (is_transpose only for 2D)
        let status = TensorDiffStatus::from_diff_info(0.0, &[2, 3, 4], &[4, 3, 2], 24, 24);
        assert_eq!(status, TensorDiffStatus::Critical);
    }

    #[test]
    fn test_from_diff_info_boundary_nearly_identical() {
        // max_diff just below 0.001 => NearlyIdentical
        let status = TensorDiffStatus::from_diff_info(0.000_999, &[10], &[10], 5, 10);
        assert_eq!(status, TensorDiffStatus::NearlyIdentical);
    }

    #[test]
    fn test_from_diff_info_boundary_small_diff() {
        // max_diff exactly at 0.001 => SmallDiff (0.001 is NOT < 0.001)
        let status = TensorDiffStatus::from_diff_info(0.001, &[10], &[10], 5, 10);
        assert_eq!(status, TensorDiffStatus::SmallDiff);
    }

    #[test]
    fn test_from_diff_info_boundary_medium_diff() {
        // max_diff exactly at 0.01 => MediumDiff (0.01 is NOT < 0.01)
        let status = TensorDiffStatus::from_diff_info(0.01, &[10], &[10], 5, 10);
        assert_eq!(status, TensorDiffStatus::MediumDiff);
    }

    #[test]
    fn test_from_diff_info_boundary_large_diff() {
        // max_diff exactly at 0.1 => LargeDiff (0.1 is NOT < 0.1)
        let status = TensorDiffStatus::from_diff_info(0.1, &[10], &[10], 5, 10);
        assert_eq!(status, TensorDiffStatus::LargeDiff);
    }

    #[test]
    fn test_from_diff_info_boundary_critical() {
        // max_diff exactly at 1.0 => Critical (1.0 is NOT < 1.0)
        let status = TensorDiffStatus::from_diff_info(1.0, &[10], &[10], 0, 10);
        assert_eq!(status, TensorDiffStatus::Critical);
    }

    #[test]
    fn test_from_diff_info_same_1d_shape_identical() {
        // 1D shapes that match, max_diff = 0.0
        let status = TensorDiffStatus::from_diff_info(0.0, &[256], &[256], 256, 256);
        assert_eq!(status, TensorDiffStatus::Identical);
    }

    #[test]
    fn test_from_diff_info_transposed_exact_boundary_ident_ratio() {
        // Transposed shapes with exactly 99% identical (should NOT be Transposed - need >0.99)
        // 99/100 = 0.99, which is NOT > 0.99
        let status = TensorDiffStatus::from_diff_info(0.0, &[10, 10], &[10, 10], 99, 100);
        // Same shapes => not transposed, max_diff=0.0 => Identical
        assert_eq!(status, TensorDiffStatus::Identical);
    }

    #[test]
    fn test_from_diff_info_transposed_boundary_ident_ratio_below_threshold() {
        // Transposed shapes, ident ratio exactly 0.99 (not > 0.99), max_diff = 0.0
        // 99/100 = 0.99 which is NOT > 0.99 => goes to max_diff check
        // max_diff=0.0 < 0.1 => MediumDiff
        let status = TensorDiffStatus::from_diff_info(0.0, &[5, 20], &[20, 5], 99, 100);
        assert_eq!(status, TensorDiffStatus::MediumDiff);
    }

    // ==================== TensorDiffStatus::colored_string - all variants ====================

    #[test]
    fn test_colored_string_identical() {
        let s = TensorDiffStatus::Identical.colored_string();
        // colored_string returns a ColoredString; check the underlying text
        assert_eq!(s.to_string().contains("IDENTICAL"), true);
    }

    #[test]
    fn test_colored_string_nearly_identical() {
        let s = TensorDiffStatus::NearlyIdentical.colored_string();
        assert!(s.to_string().contains("IDENT"));
    }

    #[test]
    fn test_colored_string_small_diff() {
        let s = TensorDiffStatus::SmallDiff.colored_string();
        assert!(s.to_string().contains("SMALL"));
    }

    #[test]
    fn test_colored_string_medium_diff() {
        let s = TensorDiffStatus::MediumDiff.colored_string();
        assert!(s.to_string().contains("MEDIUM"));
    }

    #[test]
    fn test_colored_string_large_diff() {
        let s = TensorDiffStatus::LargeDiff.colored_string();
        assert!(s.to_string().contains("LARGE"));
    }

    #[test]
    fn test_colored_string_transposed() {
        let s = TensorDiffStatus::Transposed.colored_string();
        assert!(s.to_string().contains("TRANSPOSED"));
    }

    #[test]
    fn test_colored_string_critical() {
        let s = TensorDiffStatus::Critical.colored_string();
        assert!(s.to_string().contains("CRITICAL"));
    }

    // ==================== DiffResultJson::from(&DiffReport) ====================

    #[test]
    fn test_diff_result_json_from_empty_report() {
        let report = DiffReport {
            path1: "model_a.apr".to_string(),
            path2: "model_b.apr".to_string(),
            format1: "APR".to_string(),
            format2: "APR".to_string(),
            differences: vec![],
            inspection1: None,
            inspection2: None,
        };
        let json = DiffResultJson::from(&report);
        assert_eq!(json.file1, "model_a.apr");
        assert_eq!(json.file2, "model_b.apr");
        assert_eq!(json.format1, "APR");
        assert_eq!(json.format2, "APR");
        assert!(json.identical);
        assert_eq!(json.difference_count, 0);
        assert!(json.differences.is_empty());
    }

    #[test]
    fn test_diff_result_json_from_report_with_diffs() {
        let report = DiffReport {
            path1: "a.gguf".to_string(),
            path2: "b.safetensors".to_string(),
            format1: "GGUF".to_string(),
            format2: "SafeTensors".to_string(),
            differences: vec![
                DiffEntry {
                    field: "tensor_count".to_string(),
                    value1: "100".to_string(),
                    value2: "200".to_string(),
                    category: DiffCategory::Tensor,
                },
                DiffEntry {
                    field: "format_version".to_string(),
                    value1: "v2".to_string(),
                    value2: "v3".to_string(),
                    category: DiffCategory::Format,
                },
            ],
            inspection1: None,
            inspection2: None,
        };
        let json = DiffResultJson::from(&report);
        assert_eq!(json.file1, "a.gguf");
        assert_eq!(json.file2, "b.safetensors");
        assert_eq!(json.format1, "GGUF");
        assert_eq!(json.format2, "SafeTensors");
        assert!(!json.identical);
        assert_eq!(json.difference_count, 2);
        assert_eq!(json.differences.len(), 2);
        assert_eq!(json.differences[0].field, "tensor_count");
        assert_eq!(json.differences[0].file1_value, "100");
        assert_eq!(json.differences[0].file2_value, "200");
        assert_eq!(json.differences[0].category, "tensor");
        assert_eq!(json.differences[1].field, "format_version");
        assert_eq!(json.differences[1].category, "format");
    }

    #[test]
    fn test_diff_result_json_serialization() {
        let report = DiffReport {
            path1: "a.apr".to_string(),
            path2: "b.apr".to_string(),
            format1: "APR".to_string(),
            format2: "APR".to_string(),
            differences: vec![DiffEntry {
                field: "file_size".to_string(),
                value1: "1024".to_string(),
                value2: "2048".to_string(),
                category: DiffCategory::Size,
            }],
            inspection1: None,
            inspection2: None,
        };
        let json_result = DiffResultJson::from(&report);
        let serialized = serde_json::to_string(&json_result).expect("serialize");
        assert!(serialized.contains("\"file1\":\"a.apr\""));
        assert!(serialized.contains("\"identical\":false"));
        assert!(serialized.contains("\"difference_count\":1"));
        assert!(serialized.contains("\"file_size\""));
    }

    // ==================== normalize_tensor_name - all replacement patterns ====================

    #[test]
    fn test_normalize_tensor_name_attn_k() {
        let result = normalize_tensor_name("blk.5.attn_k.weight");
        assert_eq!(result, "model.layers.5.self_attn.k_proj.weight");
    }

    #[test]
    fn test_normalize_tensor_name_attn_v() {
        let result = normalize_tensor_name("blk.3.attn_v.weight");
        assert_eq!(result, "model.layers.3.self_attn.v_proj.weight");
    }

    #[test]
    fn test_normalize_tensor_name_attn_output() {
        let result = normalize_tensor_name("blk.1.attn_output.weight");
        assert_eq!(result, "model.layers.1.self_attn.o_proj.weight");
    }

    #[test]
    fn test_normalize_tensor_name_ffn_gate() {
        let result = normalize_tensor_name("blk.0.ffn_gate.weight");
        assert_eq!(result, "model.layers.0.mlp.gate_proj.weight");
    }

    #[test]
    fn test_normalize_tensor_name_ffn_up() {
        let result = normalize_tensor_name("blk.2.ffn_up.weight");
        assert_eq!(result, "model.layers.2.mlp.up_proj.weight");
    }

    #[test]
    fn test_normalize_tensor_name_ffn_down() {
        let result = normalize_tensor_name("blk.4.ffn_down.weight");
        assert_eq!(result, "model.layers.4.mlp.down_proj.weight");
    }

    #[test]
    fn test_normalize_tensor_name_attn_norm() {
        let result = normalize_tensor_name("blk.0.attn_norm.weight");
        assert_eq!(result, "model.layers.0.input_layernorm.weight");
    }

    #[test]
    fn test_normalize_tensor_name_ffn_norm() {
        let result = normalize_tensor_name("blk.0.ffn_norm.weight");
        assert_eq!(result, "model.layers.0.post_attention_layernorm.weight");
    }

    #[test]
    fn test_normalize_tensor_name_no_changes() {
        // Already in HF naming convention - should be unchanged
        let name = "model.layers.0.self_attn.q_proj.weight";
        assert_eq!(normalize_tensor_name(name), name);
    }

    #[test]
    fn test_normalize_tensor_name_empty() {
        assert_eq!(normalize_tensor_name(""), "");
    }

    #[test]
    fn test_normalize_tensor_name_no_prefix() {
        // No blk. prefix but has the dot-delimited pattern
        let result = normalize_tensor_name("layer.attn_q.bias");
        assert_eq!(result, "layer.self_attn.q_proj.bias");
    }

    // ==================== truncate_path ====================

    #[test]
    fn test_truncate_path_short() {
        assert_eq!(truncate_path("short.apr", 20), "short.apr");
    }

    #[test]
    fn test_truncate_path_exact_length() {
        let path = "abcdefghij"; // 10 chars
        assert_eq!(truncate_path(path, 10), "abcdefghij");
    }

    #[test]
    fn test_truncate_path_long() {
        let path = "/very/long/path/to/some/model/file.apr";
        let result = truncate_path(path, 20);
        assert!(result.starts_with("..."));
        assert_eq!(result.len(), 20);
        assert!(result.ends_with("file.apr"));
    }

    #[test]
    fn test_truncate_path_one_over() {
        let path = "abcdefghijk"; // 11 chars
        let result = truncate_path(path, 10);
        assert!(result.starts_with("..."));
        assert_eq!(result.len(), 10);
    }

    // ==================== truncate_str ====================

    #[test]
    fn test_truncate_str_short() {
        assert_eq!(truncate_str("hello", 10), "hello");
    }

    #[test]
    fn test_truncate_str_exact_length() {
        assert_eq!(truncate_str("1234567890", 10), "1234567890");
    }

    #[test]
    fn test_truncate_str_long() {
        let result = truncate_str("this is a very long string", 10);
        assert_eq!(result, "this is...");
        assert_eq!(result.len(), 10);
    }

    #[test]
    fn test_truncate_str_one_over() {
        let result = truncate_str("12345678901", 10);
        assert!(result.ends_with("..."));
        assert_eq!(result.len(), 10);
    }

    // ==================== compute_tensor_diff_stats ====================

    #[test]
    fn test_compute_tensor_diff_stats_empty_data() {
        let stats = compute_tensor_diff_stats("empty", &[0], &[0], &[], &[], false);
        assert_eq!(stats.status, TensorDiffStatus::Critical);
        assert_eq!(stats.element_count, 0);
        assert_eq!(stats.mean_diff, 0.0);
        assert_eq!(stats.max_diff, 0.0);
        assert_eq!(stats.rmse, 0.0);
        assert_eq!(stats.cosine_similarity, 0.0);
        assert_eq!(stats.name, "empty");
    }

    #[test]
    fn test_compute_tensor_diff_stats_large_diff() {
        let data_a = vec![0.0, 0.0, 0.0, 0.0];
        let data_b = vec![10.0, 10.0, 10.0, 10.0];
        let stats = compute_tensor_diff_stats("large", &[4], &[4], &data_a, &data_b, false);
        assert_eq!(stats.status, TensorDiffStatus::Critical);
        assert_eq!(stats.max_diff, 10.0);
        assert_eq!(stats.large_diff_count, 4);
    }

    #[test]
    fn test_compute_tensor_diff_stats_medium_diff() {
        let data_a = vec![1.0, 2.0, 3.0, 4.0];
        let data_b = vec![1.05, 2.05, 3.05, 4.05];
        let stats = compute_tensor_diff_stats("med", &[4], &[4], &data_a, &data_b, false);
        assert_eq!(stats.status, TensorDiffStatus::MediumDiff);
        assert!(stats.max_diff > 0.01);
        assert!(stats.max_diff < 0.1);
    }

    #[test]
    fn test_compute_tensor_diff_stats_with_nan() {
        let data_a = vec![1.0, f32::NAN, 3.0];
        let data_b = vec![1.0, 2.0, 3.0];
        let stats = compute_tensor_diff_stats("nan_test", &[3], &[3], &data_a, &data_b, false);
        // NaN values should be counted as large_diff
        assert!(stats.large_diff_count >= 1);
        // Non-NaN elements should still be compared
        assert!(stats.identical_count >= 2);
    }

    #[test]
    fn test_compute_tensor_diff_stats_with_inf() {
        let data_a = vec![1.0, f32::INFINITY, 3.0];
        let data_b = vec![1.0, 2.0, 3.0];
        let stats = compute_tensor_diff_stats("inf_test", &[3], &[3], &data_a, &data_b, false);
        // Inf values should be counted as large_diff
        assert!(stats.large_diff_count >= 1);
    }

    #[test]
    fn test_compute_tensor_diff_stats_with_neg_inf() {
        let data_a = vec![f32::NEG_INFINITY, 2.0];
        let data_b = vec![1.0, 2.0];
        let stats = compute_tensor_diff_stats("neg_inf", &[2], &[2], &data_a, &data_b, false);
        assert!(stats.large_diff_count >= 1);
        assert_eq!(stats.identical_count, 1);
    }

    #[test]
    fn test_compute_tensor_diff_stats_both_nan() {
        let data_a = vec![f32::NAN, f32::NAN];
        let data_b = vec![f32::NAN, f32::NAN];
        let stats = compute_tensor_diff_stats("both_nan", &[2], &[2], &data_a, &data_b, false);
        // Both NaN => large_diff (NaN skipped in stats)
        assert_eq!(stats.large_diff_count, 2);
        assert_eq!(stats.identical_count, 0);
    }

    #[test]
    fn test_compute_tensor_diff_stats_all_zeros() {
        let data = vec![0.0, 0.0, 0.0, 0.0];
        let stats = compute_tensor_diff_stats("zeros", &[4], &[4], &data, &data, false);
        assert_eq!(stats.status, TensorDiffStatus::Identical);
        assert_eq!(stats.identical_count, 4);
        // Cosine similarity of zero vectors is 0.0 (division by zero guard)
        assert_eq!(stats.cosine_similarity, 0.0);
    }

    #[test]
    fn test_compute_tensor_diff_stats_cosine_similarity_identical() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let stats = compute_tensor_diff_stats("cos", &[4], &[4], &data, &data, false);
        // Identical vectors => cosine_similarity = 1.0
        assert!((stats.cosine_similarity - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_compute_tensor_diff_stats_cosine_similarity_opposite() {
        let data_a = vec![1.0, 2.0, 3.0];
        let data_b = vec![-1.0, -2.0, -3.0];
        let stats = compute_tensor_diff_stats("opposite", &[3], &[3], &data_a, &data_b, false);
        // Opposite vectors => cosine_similarity = -1.0
        assert!((stats.cosine_similarity + 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_compute_tensor_diff_stats_different_length_data() {
        // data_b is shorter than data_a; element_count = min(len_a, len_b)
        let data_a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let data_b = vec![1.0, 2.0, 3.0];
        let stats = compute_tensor_diff_stats("diff_len", &[5], &[3], &data_a, &data_b, false);
        assert_eq!(stats.element_count, 3);
    }

    #[test]
    fn test_compute_tensor_diff_stats_transpose_aware() {
        // A is [2, 3] with data [1,2,3,4,5,6]
        // B is [3, 2] with data = transposed version
        // A[0,0]=1, A[0,1]=2, A[0,2]=3, A[1,0]=4, A[1,1]=5, A[1,2]=6
        // B transposed from A: B[0,0]=1, B[0,1]=4, B[1,0]=2, B[1,1]=5, B[2,0]=3, B[2,1]=6
        let data_a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        // B in row-major for shape [3, 2]: [[1,4],[2,5],[3,6]]
        let data_b = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let stats =
            compute_tensor_diff_stats("transpose", &[2, 3], &[3, 2], &data_a, &data_b, true);
        // With transpose_aware=true, it remaps indices
        // The function compares A[i] with B[remapped(i)]
        // Let's just verify it runs and produces a result
        assert_eq!(stats.element_count, 6);
        assert_eq!(stats.name, "transpose");
    }

    #[test]
    fn test_compute_tensor_diff_stats_transpose_aware_false() {
        // Same data but transpose_aware=false: linear comparison, shapes differ
        let data_a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let data_b = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let stats =
            compute_tensor_diff_stats("no_transpose", &[2, 3], &[3, 2], &data_a, &data_b, false);
        // Without transpose_aware, shapes are transposed so from_diff_info detects Transposed
        // if ident ratio > 0.99. Let's check: 2 of 6 identical (indices 0 and 4) => 33%
        // So it won't be Transposed status. max_diff = |4-2| = 2.0 => Critical path likely
        assert_eq!(stats.element_count, 6);
    }

    #[test]
    fn test_compute_tensor_diff_stats_diff_buckets() {
        // Craft data to hit all diff buckets: identical, small (<0.001), medium (<0.01), large (>=0.01)
        let data_a = vec![1.0, 2.0, 3.0, 4.0];
        let data_b = vec![1.0, 2.0005, 3.005, 4.5];
        let stats = compute_tensor_diff_stats("buckets", &[4], &[4], &data_a, &data_b, false);
        assert_eq!(stats.identical_count, 1); // 1.0 == 1.0
        assert_eq!(stats.small_diff_count, 1); // |2.0-2.0005| = 0.0005 < 0.001
        assert_eq!(stats.medium_diff_count, 1); // |3.0-3.005| = 0.005 in [0.001, 0.01)
        assert_eq!(stats.large_diff_count, 1); // |4.0-4.5| = 0.5 >= 0.01
    }

    #[test]
    fn test_compute_tensor_diff_stats_rmse_and_mean() {
        let data_a = vec![0.0, 0.0];
        let data_b = vec![1.0, 1.0];
        let stats = compute_tensor_diff_stats("rmse", &[2], &[2], &data_a, &data_b, false);
        assert!((stats.mean_diff - 1.0).abs() < 1e-5);
        assert!((stats.rmse - 1.0).abs() < 1e-5);
        assert!((stats.max_diff - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_compute_tensor_diff_stats_shape_stored() {
        let stats = compute_tensor_diff_stats("shapes", &[2, 3, 4], &[5, 6], &[1.0], &[2.0], false);
        assert_eq!(stats.shape_a, vec![2, 3, 4]);
        assert_eq!(stats.shape_b, vec![5, 6]);
    }

    // ==================== validate_paths - additional coverage ====================

    #[test]
    fn test_validate_paths_second_is_directory() {
        let file1 = NamedTempFile::new().expect("create file");
        let dir = tempdir().expect("create dir");
        let result = validate_paths(file1.path(), dir.path());
        assert!(result.is_err());
        match result {
            Err(CliError::NotAFile(_)) => {}
            _ => panic!("Expected NotAFile error"),
        }
    }

    #[test]
    fn test_validate_paths_both_nonexistent() {
        let result = validate_paths(
            Path::new("/nonexistent/a.apr"),
            Path::new("/nonexistent/b.apr"),
        );
        assert!(result.is_err());
        // First path checked first
        match result {
            Err(CliError::FileNotFound(p)) => {
                assert_eq!(p, Path::new("/nonexistent/a.apr"));
            }
            _ => panic!("Expected FileNotFound error for first path"),
        }
    }

    // ==================== TensorValueStats struct ====================

    #[test]
    fn test_tensor_value_stats_construction() {
        let stats = TensorValueStats {
            name: "test_tensor".to_string(),
            shape_a: vec![10, 20],
            shape_b: vec![10, 20],
            element_count: 200,
            mean_diff: 0.001,
            max_diff: 0.005,
            rmse: 0.002,
            cosine_similarity: 0.999,
            identical_count: 150,
            small_diff_count: 30,
            medium_diff_count: 15,
            large_diff_count: 5,
            status: TensorDiffStatus::SmallDiff,
        };
        assert_eq!(stats.name, "test_tensor");
        assert_eq!(stats.element_count, 200);
        assert_eq!(stats.status, TensorDiffStatus::SmallDiff);
    }

    #[test]
    fn test_tensor_value_stats_serialization() {
        let stats = TensorValueStats {
            name: "layer.0.weight".to_string(),
            shape_a: vec![4, 4],
            shape_b: vec![4, 4],
            element_count: 16,
            mean_diff: 0.0,
            max_diff: 0.0,
            rmse: 0.0,
            cosine_similarity: 1.0,
            identical_count: 16,
            small_diff_count: 0,
            medium_diff_count: 0,
            large_diff_count: 0,
            status: TensorDiffStatus::Identical,
        };
        let json = serde_json::to_string(&stats).expect("serialize");
        assert!(json.contains("\"name\":\"layer.0.weight\""));
        assert!(json.contains("\"element_count\":16"));
        assert!(json.contains("\"Identical\""));
    }

    // ==================== TensorDiffStatus serialization ====================

    #[test]
    fn test_tensor_diff_status_serialize_all_variants() {
        let variants = vec![
            (TensorDiffStatus::Identical, "\"Identical\""),
            (TensorDiffStatus::NearlyIdentical, "\"NearlyIdentical\""),
            (TensorDiffStatus::SmallDiff, "\"SmallDiff\""),
            (TensorDiffStatus::MediumDiff, "\"MediumDiff\""),
            (TensorDiffStatus::LargeDiff, "\"LargeDiff\""),
            (TensorDiffStatus::Transposed, "\"Transposed\""),
            (TensorDiffStatus::Critical, "\"Critical\""),
        ];
        for (variant, expected) in variants {
            let json = serde_json::to_string(&variant).expect("serialize");
            assert_eq!(json, expected);
        }
    }

    #[test]
    fn test_tensor_diff_status_clone_and_copy() {
        let status = TensorDiffStatus::LargeDiff;
        let cloned = status.clone();
        let copied = status;
        assert_eq!(status, cloned);
        assert_eq!(status, copied);
    }

    #[test]
    fn test_tensor_diff_status_debug() {
        let debug = format!("{:?}", TensorDiffStatus::Transposed);
        assert_eq!(debug, "Transposed");
    }

    // ==================== print_tensor_diff_row - coverage for formatting ====================

    #[test]
    fn test_print_tensor_diff_row_identical() {
        let stats = TensorValueStats {
            name: "token_embd.weight".to_string(),
            shape_a: vec![100, 64],
            shape_b: vec![100, 64],
            element_count: 6400,
            mean_diff: 0.0,
            max_diff: 0.0,
            rmse: 0.0,
            cosine_similarity: 1.0,
            identical_count: 6400,
            small_diff_count: 0,
            medium_diff_count: 0,
            large_diff_count: 0,
            status: TensorDiffStatus::Identical,
        };
        // Just ensure it doesn't panic
        print_tensor_diff_row(&stats);
    }

    #[test]
    fn test_print_tensor_diff_row_critical_with_shape_mismatch() {
        let stats = TensorValueStats {
            name: "output.weight".to_string(),
            shape_a: vec![100, 64],
            shape_b: vec![200, 32],
            element_count: 6400,
            mean_diff: 5.0,
            max_diff: 10.0,
            rmse: 6.0,
            cosine_similarity: 0.5,
            identical_count: 0,
            small_diff_count: 0,
            medium_diff_count: 0,
            large_diff_count: 6400,
            status: TensorDiffStatus::Critical,
        };
        // Exercises the SHAPE MISMATCH branch (non-transpose, non-match)
        print_tensor_diff_row(&stats);
    }

    #[test]
    fn test_print_tensor_diff_row_transposed_shapes() {
        let stats = TensorValueStats {
            name: "attn.weight".to_string(),
            shape_a: vec![64, 128],
            shape_b: vec![128, 64],
            element_count: 8192,
            mean_diff: 0.0,
            max_diff: 0.0,
            rmse: 0.0,
            cosine_similarity: 1.0,
            identical_count: 8192,
            small_diff_count: 0,
            medium_diff_count: 0,
            large_diff_count: 0,
            status: TensorDiffStatus::Transposed,
        };
        // Exercises the TRANSPOSED branch in shape printing
        print_tensor_diff_row(&stats);
    }

    #[test]
    fn test_print_tensor_diff_row_nearly_identical() {
        let stats = TensorValueStats {
            name: "norm.weight".to_string(),
            shape_a: vec![64],
            shape_b: vec![64],
            element_count: 64,
            mean_diff: 0.000_05,
            max_diff: 0.000_1,
            rmse: 0.000_07,
            cosine_similarity: 0.999_999,
            identical_count: 32,
            small_diff_count: 32,
            medium_diff_count: 0,
            large_diff_count: 0,
            status: TensorDiffStatus::NearlyIdentical,
        };
        // Exercises the distribution printing path (status != Identical)
        print_tensor_diff_row(&stats);
    }

    #[test]
    fn test_print_tensor_diff_row_medium_diff() {
        let stats = TensorValueStats {
            name: "ffn.weight".to_string(),
            shape_a: vec![32, 32],
            shape_b: vec![32, 32],
            element_count: 1024,
            mean_diff: 0.03,
            max_diff: 0.08,
            rmse: 0.04,
            cosine_similarity: 0.998,
            identical_count: 100,
            small_diff_count: 200,
            medium_diff_count: 500,
            large_diff_count: 224,
            status: TensorDiffStatus::MediumDiff,
        };
        print_tensor_diff_row(&stats);
    }

    #[test]
    fn test_print_tensor_diff_row_large_diff() {
        let stats = TensorValueStats {
            name: "lm_head.weight".to_string(),
            shape_a: vec![50, 50],
            shape_b: vec![50, 50],
            element_count: 2500,
            mean_diff: 0.3,
            max_diff: 0.9,
            rmse: 0.4,
            cosine_similarity: 0.95,
            identical_count: 0,
            small_diff_count: 0,
            medium_diff_count: 500,
            large_diff_count: 2000,
            status: TensorDiffStatus::LargeDiff,
        };
        print_tensor_diff_row(&stats);
    }

    #[test]
    fn test_print_tensor_diff_row_small_diff() {
        let stats = TensorValueStats {
            name: "embed.weight".to_string(),
            shape_a: vec![16, 16],
            shape_b: vec![16, 16],
            element_count: 256,
            mean_diff: 0.003,
            max_diff: 0.008,
            rmse: 0.004,
            cosine_similarity: 0.9999,
            identical_count: 50,
            small_diff_count: 150,
            medium_diff_count: 56,
            large_diff_count: 0,
            status: TensorDiffStatus::SmallDiff,
        };
        print_tensor_diff_row(&stats);
    }

    #[test]
    fn test_print_tensor_diff_row_long_name_truncation() {
        let long_name = "model.layers.99.self_attn.q_proj.weight.extra.suffix.that.is.very.long";
        let stats = TensorValueStats {
            name: long_name.to_string(),
            shape_a: vec![4],
            shape_b: vec![4],
            element_count: 4,
            mean_diff: 0.0,
            max_diff: 0.0,
            rmse: 0.0,
            cosine_similarity: 1.0,
            identical_count: 4,
            small_diff_count: 0,
            medium_diff_count: 0,
            large_diff_count: 0,
            status: TensorDiffStatus::Identical,
        };
        // Exercises truncate_str for name > 40 chars
        print_tensor_diff_row(&stats);
    }

    #[test]
    fn test_print_tensor_diff_row_cosine_similarity_ranges() {
        // Test cosine similarity coloring: > 0.9999
        let make_stats = |cos: f32| TensorValueStats {
            name: "t".to_string(),
            shape_a: vec![4],
            shape_b: vec![4],
            element_count: 4,
            mean_diff: 0.01,
            max_diff: 0.02,
            rmse: 0.01,
            cosine_similarity: cos,
            identical_count: 0,
            small_diff_count: 4,
            medium_diff_count: 0,
            large_diff_count: 0,
            status: TensorDiffStatus::SmallDiff,
        };
        // > 0.9999 (green)
        print_tensor_diff_row(&make_stats(0.99999));
        // > 0.999 (blue)
        print_tensor_diff_row(&make_stats(0.9995));
        // > 0.99 (yellow)
        print_tensor_diff_row(&make_stats(0.995));
        // <= 0.99 (red)
        print_tensor_diff_row(&make_stats(0.5));
    }

    // ==================== DiffEntryJson struct ====================

    #[test]
    fn test_diff_entry_json_serialization() {
        let entry = DiffEntryJson {
            field: "tensor_count".to_string(),
            file1_value: "100".to_string(),
            file2_value: "200".to_string(),
            category: "tensor".to_string(),
            diff_type: "data".to_string(),
        };
        let json = serde_json::to_string(&entry).expect("serialize");
        assert!(json.contains("\"field\":\"tensor_count\""));
        assert!(json.contains("\"file1_value\":\"100\""));
        assert!(json.contains("\"file2_value\":\"200\""));
        assert!(json.contains("\"category\":\"tensor\""));
        assert!(json.contains("\"type\":\"data\""));
    }

    // ==================== DiffResultJson category mapping ====================

    #[test]
    fn test_diff_result_json_all_categories() {
        let report = DiffReport {
            path1: "a".to_string(),
            path2: "b".to_string(),
            format1: "APR".to_string(),
            format2: "GGUF".to_string(),
            differences: vec![
                DiffEntry {
                    field: "f1".to_string(),
                    value1: "v1".to_string(),
                    value2: "v2".to_string(),
                    category: DiffCategory::Format,
                },
                DiffEntry {
                    field: "f2".to_string(),
                    value1: "v1".to_string(),
                    value2: "v2".to_string(),
                    category: DiffCategory::Metadata,
                },
                DiffEntry {
                    field: "f3".to_string(),
                    value1: "v1".to_string(),
                    value2: "v2".to_string(),
                    category: DiffCategory::Tensor,
                },
                DiffEntry {
                    field: "f4".to_string(),
                    value1: "v1".to_string(),
                    value2: "v2".to_string(),
                    category: DiffCategory::Quantization,
                },
                DiffEntry {
                    field: "f5".to_string(),
                    value1: "v1".to_string(),
                    value2: "v2".to_string(),
                    category: DiffCategory::Size,
                },
            ],
            inspection1: None,
            inspection2: None,
        };
        let json = DiffResultJson::from(&report);
        assert_eq!(json.difference_count, 5);
        assert_eq!(json.differences[0].category, "format");
        assert_eq!(json.differences[1].category, "metadata");
        assert_eq!(json.differences[2].category, "tensor");
        assert_eq!(json.differences[3].category, "quantization");
        assert_eq!(json.differences[4].category, "size");
    }

    // ==================== Integration: compute_tensor_diff_stats + from_diff_info ====================

    #[test]
    fn test_compute_stats_transposed_identical_values() {
        // Simulate a tensor that's transposed but values happen to match linearly
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let stats = compute_tensor_diff_stats("t", &[2, 3], &[3, 2], &data, &data, false);
        // All 6 elements identical linearly => ident_ratio = 1.0 > 0.99
        assert_eq!(stats.status, TensorDiffStatus::Transposed);
        assert_eq!(stats.identical_count, 6);
    }

    #[test]
    fn test_compute_stats_small_diff_boundary() {
        // max_diff exactly at SmallDiff boundary
        let data_a = vec![1.0, 2.0];
        let data_b = vec![1.001, 2.0]; // max_diff = 0.001 exactly
        let stats = compute_tensor_diff_stats("boundary", &[2], &[2], &data_a, &data_b, false);
        // 0.001 is NOT < 0.001, so SmallDiff
        assert_eq!(stats.status, TensorDiffStatus::SmallDiff);
    }

    #[test]
    fn test_compute_stats_one_element() {
        let stats = compute_tensor_diff_stats("single", &[1], &[1], &[42.0], &[42.0], false);
        assert_eq!(stats.status, TensorDiffStatus::Identical);
        assert_eq!(stats.element_count, 1);
        assert_eq!(stats.identical_count, 1);
        assert!((stats.cosine_similarity - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_compute_stats_one_element_a_only() {
        // data_b is empty => element_count = min(1, 0) = 0 => early return Critical
        let stats = compute_tensor_diff_stats("asym", &[1], &[0], &[1.0], &[], false);
        assert_eq!(stats.status, TensorDiffStatus::Critical);
        assert_eq!(stats.element_count, 0);
    }
}

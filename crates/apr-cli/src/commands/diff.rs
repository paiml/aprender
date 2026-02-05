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
    differences: Vec<DiffEntryJson>,
}

impl From<&DiffReport> for DiffResultJson {
    fn from(report: &DiffReport) -> Self {
        Self {
            file1: report.path1.clone(),
            file2: report.path2.clone(),
            format1: report.format1.clone(),
            format2: report.format2.clone(),
            identical: report.is_identical(),
            difference_count: report.diff_count(),
            differences: report
                .differences
                .iter()
                .map(|d| DiffEntryJson {
                    field: d.field.clone(),
                    file1_value: d.value1.clone(),
                    file2_value: d.value2.clone(),
                    category: d.category.name().to_string(),
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

    fn colored_string(&self) -> colored::ColoredString {
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
        println!(
            "{}",
            "╔══════════════════════════════════════════════════════════════════════════════╗"
                .cyan()
        );
        println!(
            "{}",
            "║           TENSOR VALUE DIFF (Statistical Comparison)                         ║"
                .cyan()
        );
        println!(
            "{}",
            "╠══════════════════════════════════════════════════════════════════════════════╣"
                .cyan()
        );
        println!(
            "║ Model A: {:<66} ║",
            truncate_path(&path1.display().to_string(), 66)
        );
        println!(
            "║ Model B: {:<66} ║",
            truncate_path(&path2.display().to_string(), 66)
        );
        println!(
            "{}",
            "╠══════════════════════════════════════════════════════════════════════════════╣"
                .cyan()
        );
        println!(
            "║ Legend: {} {} {} {} {} {} ║",
            "IDENTICAL".green().bold(),
            "~IDENT".green(),
            "SMALL".blue(),
            "MEDIUM".yellow(),
            "LARGE".red(),
            "CRITICAL".red().bold()
        );
        println!(
            "{}",
            "╠══════════════════════════════════════════════════════════════════════════════╣"
                .cyan()
        );
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
        let data1 = rosetta
            .load_tensor_f32(path1, &t1.name)
            .map_err(|e| CliError::ValidationFailed(format!("Failed to load tensor {}: {e}", t1.name)))?;
        let data2 = rosetta
            .load_tensor_f32(path2, &t2.name)
            .map_err(|e| CliError::ValidationFailed(format!("Failed to load tensor {}: {e}", t2.name)))?;

        let stats = compute_tensor_diff_stats(name, &t1.shape, &t2.shape, &data1, &data2, transpose_aware);

        match stats.status {
            TensorDiffStatus::Critical => critical_count += 1,
            TensorDiffStatus::LargeDiff => large_count += 1,
            TensorDiffStatus::MediumDiff => medium_count += 1,
            TensorDiffStatus::Transposed => transposed_count += 1,
            TensorDiffStatus::Identical | TensorDiffStatus::NearlyIdentical => identical_count += 1,
            _ => {}
        }

        if !json_output {
            print_tensor_diff_row(&stats);
        }

        results.push(stats);
    }

    if !json_output {
        // Summary
        println!(
            "{}",
            "╠══════════════════════════════════════════════════════════════════════════════╣"
                .cyan()
        );
        println!(
            "{}",
            "║                              SUMMARY                                          ║"
                .cyan()
                .bold()
        );
        println!(
            "{}",
            "╠══════════════════════════════════════════════════════════════════════════════╣"
                .cyan()
        );
        println!("║ Tensors compared: {:<58} ║", results.len());
        println!(
            "║ Identical: {:<65} ║",
            format!("{}", identical_count).green().to_string()
        );
        println!(
            "║ Transposed (layout diff): {:<50} ║",
            if transposed_count > 0 {
                format!("{}", transposed_count).cyan().to_string()
            } else {
                "0".dimmed().to_string()
            }
        );
        println!(
            "║ Critical differences: {:<54} ║",
            if critical_count > 0 {
                format!("{}", critical_count).red().bold().to_string()
            } else {
                "0".green().to_string()
            }
        );
        println!(
            "║ Large differences: {:<57} ║",
            if large_count > 0 {
                format!("{}", large_count).red().to_string()
            } else {
                "0".green().to_string()
            }
        );
        println!(
            "║ Medium differences: {:<56} ║",
            if medium_count > 0 {
                format!("{}", medium_count).yellow().to_string()
            } else {
                "0".green().to_string()
            }
        );

        // Diagnosis
        println!(
            "{}",
            "╠══════════════════════════════════════════════════════════════════════════════╣"
                .cyan()
        );
        if critical_count > 0 {
            println!(
                "║ {} ║",
                "DIAGNOSIS: Critical value differences detected!".red().bold()
            );
            println!(
                "║ {:<75} ║",
                "Possible causes:".yellow()
            );
            println!(
                "║ {:<75} ║",
                "  - Different quantization/dequantization algorithms"
            );
            println!(
                "║ {:<75} ║",
                "  - Tensor layout mismatch (row-major vs column-major)"
            );
            println!(
                "║ {:<75} ║",
                "  - Corrupted weights during conversion"
            );
        } else if large_count > 0 {
            println!(
                "║ {} ║",
                "DIAGNOSIS: Large value differences - may affect inference quality"
                    .yellow()
                    .bold()
            );
            // Check if these are mostly transposed tensors
            let transposed_with_diffs = results
                .iter()
                .filter(|r| {
                    let is_transpose = r.shape_a.len() == 2
                        && r.shape_b.len() == 2
                        && r.shape_a[0] == r.shape_b[1]
                        && r.shape_a[1] == r.shape_b[0];
                    is_transpose && r.status != TensorDiffStatus::Transposed
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
        } else if medium_count > 0 {
            println!(
                "║ {} ║",
                "DIAGNOSIS: Medium differences - likely acceptable quantization variance"
                    .blue()
            );
        } else if transposed_count > 0 && identical_count > 0 {
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
        println!(
            "{}",
            "╚══════════════════════════════════════════════════════════════════════════════╝"
                .cyan()
        );
    } else {
        // JSON output
        let json = serde_json::json!({
            "model_a": path1.display().to_string(),
            "model_b": path2.display().to_string(),
            "tensors_compared": results.len(),
            "identical_count": identical_count,
            "transposed_count": transposed_count,
            "critical_count": critical_count,
            "large_count": large_count,
            "medium_count": medium_count,
            "results": results,
        });
        println!("{}", serde_json::to_string_pretty(&json).unwrap_or_default());
    }

    Ok(())
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
        return TensorValueStats {
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
        };
    }

    // Check if shapes are transposed (2D tensors with swapped dimensions)
    let is_transpose = shape_a.len() == 2
        && shape_b.len() == 2
        && shape_a[0] == shape_b[1]
        && shape_a[1] == shape_b[0];

    let mut sum_diff = 0.0f64;
    let mut sum_sq_diff = 0.0f64;
    let mut max_diff = 0.0f32;
    let mut dot_product = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;
    let mut identical_count = 0usize;
    let mut small_diff_count = 0usize;
    let mut medium_diff_count = 0usize;
    let mut large_diff_count = 0usize;

    for i in 0..element_count {
        let a = data_a[i];
        // If transpose_aware and shapes are transposed, look up corresponding element
        let b = if transpose_aware && is_transpose && shape_a.len() == 2 {
            // For transposed comparison: element at (row, col) in A should compare to (row, col) in B
            // A: linear i = row * cols_a + col, where row = i / cols_a, col = i % cols_a
            // B: linear j = row * cols_b + col, where cols_b = shape_b[1]
            // Since shapes are transposed: cols_a = shape_a[1], cols_b = shape_b[1] = shape_a[0]
            let cols_a = shape_a[1];
            let row = i / cols_a;
            let col = i % cols_a;
            // B's linear index for the same logical (row, col)
            let cols_b = shape_b[1];
            let j = row * cols_b + col;
            if j < data_b.len() {
                data_b[j]
            } else {
                continue; // Skip if out of bounds
            }
        } else {
            data_b[i]
        };

        // Skip NaN/Inf for statistics
        if a.is_nan() || b.is_nan() || a.is_infinite() || b.is_infinite() {
            large_diff_count += 1;
            continue;
        }

        let diff = (a - b).abs();
        sum_diff += diff as f64;
        sum_sq_diff += (diff as f64) * (diff as f64);
        max_diff = max_diff.max(diff);

        dot_product += (a as f64) * (b as f64);
        norm_a += (a as f64) * (a as f64);
        norm_b += (b as f64) * (b as f64);

        if diff == 0.0 {
            identical_count += 1;
        } else if diff < 0.001 {
            small_diff_count += 1;
        } else if diff < 0.01 {
            medium_diff_count += 1;
        } else {
            large_diff_count += 1;
        }
    }

    let mean_diff = (sum_diff / element_count as f64) as f32;
    let rmse = ((sum_sq_diff / element_count as f64).sqrt()) as f32;

    // Cosine similarity
    let cosine_similarity = if norm_a > 0.0 && norm_b > 0.0 {
        (dot_product / (norm_a.sqrt() * norm_b.sqrt())) as f32
    } else {
        0.0
    };

    let status = TensorDiffStatus::from_diff_info(
        max_diff,
        shape_a,
        shape_b,
        identical_count,
        element_count,
    );

    TensorValueStats {
        name: name.to_string(),
        shape_a: shape_a.to_vec(),
        shape_b: shape_b.to_vec(),
        element_count,
        mean_diff,
        max_diff,
        rmse,
        cosine_similarity,
        identical_count,
        small_diff_count,
        medium_diff_count,
        large_diff_count,
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

    println!(
        "║ [{}] {:<40} ║",
        status_str,
        name_truncated
    );
    println!(
        "║   max_diff={} mean_diff={:.6} rmse={:.6} cos_sim={} ║",
        max_diff_colored,
        stats.mean_diff,
        stats.rmse,
        cos_colored
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
        "╠──────────────────────────────────────────────────────────────────────────────╣"
            .dimmed()
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
    println!(
        "Comparing {} vs {}",
        report.path1.cyan(),
        report.path2.cyan()
    );
    println!();

    // Format info
    if report.same_format() {
        println!("Format: {}", report.format1.white().bold());
        println!();
    } else {
        println!(
            "{} Comparing different formats: {} vs {}",
            "NOTE:".yellow(),
            report.format1.white().bold(),
            report.format2.white().bold()
        );
        println!();
    }

    if report.is_identical() {
        println!(
            "{}",
            "Models are IDENTICAL in structure and metadata"
                .green()
                .bold()
        );
    } else {
        let count = report.diff_count();
        println!("{} {} differences found:", "DIFF:".yellow().bold(), count);
        println!();

        // Group by category
        for category in [
            DiffCategory::Format,
            DiffCategory::Size,
            DiffCategory::Quantization,
            DiffCategory::Metadata,
            DiffCategory::Tensor,
        ] {
            let diffs = report.differences_by_category(category);
            if !diffs.is_empty() {
                println!("  {} ({}):", category.name().white().bold(), diffs.len());
                for diff in diffs {
                    println!(
                        "    {}: {} → {}",
                        diff.field.white(),
                        diff.value1.red(),
                        diff.value2.green()
                    );
                }
                println!();
            }
        }
    }

    if show_weights {
        println!();
        println!(
            "{} Use --values to compare actual tensor values",
            "TIP:".blue()
        );
    }
}

// ============================================================================
// Tests (Minimal - Most logic is tested in library)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
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

        let result = run(file1.path(), file2.path(), false, false, None, 10, false, false);
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
}

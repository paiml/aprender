//! Tensors command implementation (Thin Shim)
//!
//! Lists tensor names, shapes, and statistics from APR model files.
//! This is a thin CLI wrapper around the library's tensor listing functions.
//!
//! Toyota Way: Genchi Genbutsu - Go to the actual tensors to understand.
//!
//! # TOOL-APR-001 Fix
//!
//! Previous implementation read from `tensor_shapes` metadata field.
//! New implementation uses library code that reads from actual tensor index.

use crate::error::CliError;
use crate::output;
use aprender::format::tensors::{
    format_size, list_tensors, TensorInfo, TensorListOptions, TensorListResult,
};
use colored::Colorize;
use serde::Serialize;
use std::path::Path;

// ============================================================================
// Serializable Types (for JSON output)
// ============================================================================

/// Tensor information for JSON output
#[derive(Serialize)]
struct TensorInfoJson {
    name: String,
    shape: Vec<usize>,
    dtype: String,
    size_bytes: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    mean: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    std: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    min: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    nan_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    inf_count: Option<usize>,
}

impl From<&TensorInfo> for TensorInfoJson {
    fn from(info: &TensorInfo) -> Self {
        Self {
            name: info.name.clone(),
            shape: info.shape.clone(),
            dtype: info.dtype.clone(),
            size_bytes: info.size_bytes,
            mean: info.mean,
            std: info.std,
            min: info.min,
            max: info.max,
            nan_count: info.nan_count,
            inf_count: info.inf_count,
        }
    }
}

/// Tensors listing result for JSON output
#[derive(Serialize)]
struct TensorsResultJson {
    file: String,
    format_version: String,
    tensor_count: usize,
    total_size_bytes: usize,
    tensors: Vec<TensorInfoJson>,
}

impl From<&TensorListResult> for TensorsResultJson {
    fn from(result: &TensorListResult) -> Self {
        Self {
            file: result.file.clone(),
            format_version: result.format_version.clone(),
            tensor_count: result.tensor_count,
            total_size_bytes: result.total_size_bytes,
            tensors: result.tensors.iter().map(TensorInfoJson::from).collect(),
        }
    }
}

// ============================================================================
// Main Entry Point
// ============================================================================

/// Run the tensors command
///
/// This is a thin shim that delegates to the library's tensor listing functions.
/// All actual logic is in `aprender::format::tensors`.
pub(crate) fn run(
    path: &Path,
    show_stats: bool,
    filter: Option<&str>,
    json_output: bool,
    limit: usize,
) -> Result<(), CliError> {
    // Validate path exists
    validate_path(path)?;

    // Build options
    let mut options = TensorListOptions::new();
    if show_stats {
        options = options.with_stats();
    }
    if let Some(pattern) = filter {
        options = options.with_filter(pattern);
    }
    if limit > 0 && limit < usize::MAX {
        options = options.with_limit(limit);
    }

    // Call library function
    let result = list_tensors(path, options)
        .map_err(|e| CliError::InvalidFormat(format!("Failed to list tensors: {e}")))?;

    // Output results
    if json_output {
        output_json(&result);
    } else {
        output_text(&result, show_stats);
    }

    Ok(())
}

// ============================================================================
// Path Validation
// ============================================================================

fn validate_path(path: &Path) -> Result<(), CliError> {
    if !path.exists() {
        return Err(CliError::FileNotFound(path.to_path_buf()));
    }
    if !path.is_file() {
        return Err(CliError::NotAFile(path.to_path_buf()));
    }
    Ok(())
}

// ============================================================================
// Output Functions
// ============================================================================

fn output_json(result: &TensorListResult) {
    let json_result = TensorsResultJson::from(result);
    if let Ok(json) = serde_json::to_string_pretty(&json_result) {
        println!("{json}");
    }
}

/// Format an optional f64 stat value, handling NaN.
fn format_stat(value: Option<f32>) -> String {
    match value {
        Some(v) if v.is_nan() => "NaN".to_string(),
        Some(v) => format!("{v:.4}"),
        None => "—".to_string(),
    }
}

/// Build a tensor row with optional stat columns, collecting anomaly warnings.
fn build_tensor_row(
    tensor: &TensorInfo,
    show_stats: bool,
    anomaly_warnings: &mut Vec<String>,
) -> Vec<String> {
    let mut row = vec![
        tensor.name.clone(),
        format!("{:?}", tensor.shape),
        tensor.dtype.clone(),
        format_size(tensor.size_bytes as u64),
    ];

    if show_stats {
        let range_str = match (tensor.min, tensor.max) {
            (Some(min), Some(max)) => format!("[{min:.4}, {max:.4}]"),
            _ => "—".to_string(),
        };
        row.extend(vec![
            format_stat(tensor.mean),
            format_stat(tensor.std),
            range_str,
        ]);

        if tensor.nan_count.is_some_and(|c| c > 0) {
            anomaly_warnings.push(format!(
                "  {} {}: {} NaN values (spec H8 violation)",
                "✗".red().bold(),
                tensor.name,
                tensor.nan_count.unwrap_or(0)
            ));
        }
        if tensor.inf_count.is_some_and(|c| c > 0) {
            anomaly_warnings.push(format!(
                "  {} {}: {} Inf values",
                "⚠".yellow().bold(),
                tensor.name,
                tensor.inf_count.unwrap_or(0)
            ));
        }
    }

    row
}

fn output_text(result: &TensorListResult, show_stats: bool) {
    output::header(&format!("Tensors: {}", result.file));

    if result.tensors.is_empty() {
        println!("  No tensor information available");
        return;
    }

    let summary = vec![
        ("Format", result.format_version.clone()),
        ("Tensors", output::count_fmt(result.tensor_count)),
        ("Total Size", format_size(result.total_size_bytes as u64)),
    ];
    println!("{}", output::kv_table(&summary));

    let mut dtype_counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for tensor in &result.tensors {
        *dtype_counts.entry(&tensor.dtype).or_insert(0) += 1;
    }
    let dominant_dtype = dtype_counts
        .iter()
        .max_by_key(|(_, c)| **c)
        .map_or("unknown", |(dt, _)| *dt);

    let mut headers: Vec<&str> = vec!["Name", "Shape", "DType", "Size"];
    if show_stats {
        headers.extend(&["Mean", "Std", "Range"]);
    }

    let mut anomaly_warnings: Vec<String> = Vec::new();
    let rows: Vec<Vec<String>> = result
        .tensors
        .iter()
        .map(|t| build_tensor_row(t, show_stats, &mut anomaly_warnings))
        .collect();

    println!("{}", output::table(&headers, &rows));

    if !anomaly_warnings.is_empty() {
        output::subheader("Anomalies");
        for w in &anomaly_warnings {
            println!("{w}");
        }
    }

    println!(
        "\n  {} {} {} {}",
        output::count_fmt(result.tensor_count).white().bold(),
        "tensors".dimmed(),
        format_size(result.total_size_bytes as u64).white().bold(),
        output::dtype_color(dominant_dtype),
    );
}

// ============================================================================
// Tests (Minimal - Most logic is tested in library)
// ============================================================================

#[cfg(test)]
#[path = "tensors_tests.rs"]
mod tests;

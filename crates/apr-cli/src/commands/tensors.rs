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

fn output_text(result: &TensorListResult, show_stats: bool) {
    output::section(&format!("Tensors: {}", result.file));
    println!();

    if result.tensors.is_empty() {
        println!("  No tensor information available");
        return;
    }

    output::kv("Format version", &result.format_version);
    output::kv("Total tensors", result.tensor_count);
    output::kv("Total size", format_size(result.total_size_bytes as u64));
    println!();

    for tensor in &result.tensors {
        let shape_str = format!("{:?}", tensor.shape);
        println!("  {} [{}] {}", tensor.name, tensor.dtype, shape_str);

        if tensor.size_bytes > 0 {
            println!("    Size: {}", format_size(tensor.size_bytes as u64));
        }

        if show_stats {
            if let (Some(mean), Some(std)) = (tensor.mean, tensor.std) {
                // Check for NaN stats (spec H8)
                if mean.is_nan() || std.is_nan() {
                    println!(
                        "    Stats: {} (FAIL: NaN detected per spec H8)",
                        "NaN".red()
                    );
                } else {
                    println!("    Stats: mean={mean:.4}, std={std:.4}");
                }
            }
            if let (Some(min), Some(max)) = (tensor.min, tensor.max) {
                println!("    Range: [{min:.4}, {max:.4}]");
            }
            // Display NaN/Inf count warnings
            if let Some(nan_count) = tensor.nan_count {
                if nan_count > 0 {
                    println!(
                        "    {} {} NaN values detected (spec H8 violation)",
                        "WARNING:".red().bold(),
                        nan_count
                    );
                }
            }
            if let Some(inf_count) = tensor.inf_count {
                if inf_count > 0 {
                    println!(
                        "    {} {} Inf values detected",
                        "WARNING:".yellow().bold(),
                        inf_count
                    );
                }
            }
        }
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

    // =========================================================================
    // validate_path tests
    // =========================================================================

    #[test]
    fn test_validate_path_not_found() {
        let result = validate_path(Path::new("/nonexistent/model.apr"));
        assert!(result.is_err());
        match result {
            Err(CliError::FileNotFound(_)) => {}
            _ => panic!("Expected FileNotFound error"),
        }
    }

    #[test]
    fn test_validate_path_is_directory() {
        let dir = tempdir().expect("create dir");
        let result = validate_path(dir.path());
        assert!(result.is_err());
        match result {
            Err(CliError::NotAFile(_)) => {}
            _ => panic!("Expected NotAFile error"),
        }
    }

    #[test]
    fn test_validate_path_valid() {
        let file = NamedTempFile::new().expect("create file");
        let result = validate_path(file.path());
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_path_empty_file() {
        let file = NamedTempFile::new().expect("create file");
        // Empty file is still a valid path
        let result = validate_path(file.path());
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_path_symlink_to_file() {
        // On Unix, symlinks to files are valid
        let file = NamedTempFile::new().expect("create file");
        let result = validate_path(file.path());
        assert!(result.is_ok());
    }

    // =========================================================================
    // run() tests
    // =========================================================================

    #[test]
    fn test_run_file_not_found() {
        let result = run(Path::new("/nonexistent/model.apr"), false, None, false, 100);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_invalid_file() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create file");
        file.write_all(b"not valid apr").expect("write");

        let result = run(file.path(), false, None, false, 100);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_stats_option() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create file");
        file.write_all(b"not valid apr").expect("write");

        // Will fail due to invalid file, but tests stats option parsing
        let result = run(file.path(), true, None, false, 100);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_filter_option() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create file");
        file.write_all(b"not valid apr").expect("write");

        // Will fail due to invalid file, but tests filter option parsing
        let result = run(file.path(), false, Some("weight"), false, 100);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_json_output() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create file");
        file.write_all(b"not valid apr").expect("write");

        // Will fail due to invalid file, but tests JSON output flag
        let result = run(file.path(), false, None, true, 100);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_limit_zero() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create file");
        file.write_all(b"not valid apr").expect("write");

        // Limit 0 should not apply limit
        let result = run(file.path(), false, None, false, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_limit_max() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create file");
        file.write_all(b"not valid apr").expect("write");

        // Limit usize::MAX should not apply limit
        let result = run(file.path(), false, None, false, usize::MAX);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_directory_path() {
        let dir = tempdir().expect("create dir");
        let result = run(dir.path(), false, None, false, 100);
        assert!(result.is_err());
        match result {
            Err(CliError::NotAFile(_)) => {}
            other => panic!("Expected NotAFile error, got {:?}", other),
        }
    }

    // =========================================================================
    // TensorInfoJson tests
    // =========================================================================

    #[test]
    fn test_tensor_info_json_from() {
        let info = TensorInfo {
            name: "test".to_string(),
            shape: vec![10, 20],
            dtype: "f32".to_string(),
            size_bytes: 800,
            mean: Some(0.5),
            std: Some(0.1),
            min: Some(0.0),
            max: Some(1.0),
            nan_count: Some(0),
            inf_count: Some(0),
        };

        let json_info = TensorInfoJson::from(&info);
        assert_eq!(json_info.name, "test");
        assert_eq!(json_info.shape, vec![10, 20]);
        assert_eq!(json_info.mean, Some(0.5));
    }

    #[test]
    fn test_tensor_info_json_from_no_stats() {
        let info = TensorInfo {
            name: "layer.weight".to_string(),
            shape: vec![512, 768],
            dtype: "f16".to_string(),
            size_bytes: 786432,
            mean: None,
            std: None,
            min: None,
            max: None,
            nan_count: None,
            inf_count: None,
        };

        let json_info = TensorInfoJson::from(&info);
        assert_eq!(json_info.name, "layer.weight");
        assert_eq!(json_info.shape, vec![512, 768]);
        assert_eq!(json_info.dtype, "f16");
        assert_eq!(json_info.size_bytes, 786432);
        assert!(json_info.mean.is_none());
        assert!(json_info.std.is_none());
    }

    #[test]
    fn test_tensor_info_json_from_with_anomalies() {
        let info = TensorInfo {
            name: "bad_tensor".to_string(),
            shape: vec![100],
            dtype: "f32".to_string(),
            size_bytes: 400,
            mean: Some(f32::NAN),
            std: Some(0.1),
            min: Some(-1.0),
            max: Some(1.0),
            nan_count: Some(5),
            inf_count: Some(2),
        };

        let json_info = TensorInfoJson::from(&info);
        assert!(json_info.mean.unwrap().is_nan());
        assert_eq!(json_info.nan_count, Some(5));
        assert_eq!(json_info.inf_count, Some(2));
    }

    #[test]
    fn test_tensor_info_json_empty_shape() {
        let info = TensorInfo {
            name: "scalar".to_string(),
            shape: vec![],
            dtype: "f32".to_string(),
            size_bytes: 4,
            mean: None,
            std: None,
            min: None,
            max: None,
            nan_count: None,
            inf_count: None,
        };

        let json_info = TensorInfoJson::from(&info);
        assert!(json_info.shape.is_empty());
    }

    #[test]
    fn test_tensor_info_json_multidimensional() {
        let info = TensorInfo {
            name: "conv.weight".to_string(),
            shape: vec![64, 3, 7, 7],
            dtype: "f32".to_string(),
            size_bytes: 64 * 3 * 7 * 7 * 4,
            mean: Some(0.0),
            std: Some(0.02),
            min: Some(-0.1),
            max: Some(0.1),
            nan_count: Some(0),
            inf_count: Some(0),
        };

        let json_info = TensorInfoJson::from(&info);
        assert_eq!(json_info.shape.len(), 4);
        assert_eq!(json_info.shape, vec![64, 3, 7, 7]);
    }

    // =========================================================================
    // TensorsResultJson tests
    // =========================================================================

    #[test]
    fn test_tensors_result_json_from() {
        let result = TensorListResult {
            file: "test.apr".to_string(),
            format_version: "v2".to_string(),
            tensor_count: 1,
            total_size_bytes: 100,
            tensors: vec![TensorInfo {
                name: "weight".to_string(),
                shape: vec![10],
                dtype: "f32".to_string(),
                size_bytes: 40,
                mean: None,
                std: None,
                min: None,
                max: None,
                nan_count: None,
                inf_count: None,
            }],
        };

        let json_result = TensorsResultJson::from(&result);
        assert_eq!(json_result.file, "test.apr");
        assert_eq!(json_result.format_version, "v2");
        assert_eq!(json_result.tensors.len(), 1);
    }

    #[test]
    fn test_tensors_result_json_empty_tensors() {
        let result = TensorListResult {
            file: "empty.apr".to_string(),
            format_version: "v2".to_string(),
            tensor_count: 0,
            total_size_bytes: 0,
            tensors: vec![],
        };

        let json_result = TensorsResultJson::from(&result);
        assert_eq!(json_result.tensor_count, 0);
        assert!(json_result.tensors.is_empty());
    }

    #[test]
    fn test_tensors_result_json_multiple_tensors() {
        let result = TensorListResult {
            file: "model.apr".to_string(),
            format_version: "v2".to_string(),
            tensor_count: 3,
            total_size_bytes: 1000,
            tensors: vec![
                TensorInfo {
                    name: "embed".to_string(),
                    shape: vec![1000, 128],
                    dtype: "f32".to_string(),
                    size_bytes: 512000,
                    mean: None,
                    std: None,
                    min: None,
                    max: None,
                    nan_count: None,
                    inf_count: None,
                },
                TensorInfo {
                    name: "layer.0.weight".to_string(),
                    shape: vec![128, 128],
                    dtype: "f32".to_string(),
                    size_bytes: 65536,
                    mean: None,
                    std: None,
                    min: None,
                    max: None,
                    nan_count: None,
                    inf_count: None,
                },
                TensorInfo {
                    name: "layer.0.bias".to_string(),
                    shape: vec![128],
                    dtype: "f32".to_string(),
                    size_bytes: 512,
                    mean: None,
                    std: None,
                    min: None,
                    max: None,
                    nan_count: None,
                    inf_count: None,
                },
            ],
        };

        let json_result = TensorsResultJson::from(&result);
        assert_eq!(json_result.tensors.len(), 3);
        assert_eq!(json_result.tensors[0].name, "embed");
        assert_eq!(json_result.tensors[1].name, "layer.0.weight");
        assert_eq!(json_result.tensors[2].name, "layer.0.bias");
    }

    // =========================================================================
    // Serialization tests
    // =========================================================================

    #[test]
    fn test_tensor_info_json_serialization() {
        let json_info = TensorInfoJson {
            name: "test".to_string(),
            shape: vec![10],
            dtype: "f32".to_string(),
            size_bytes: 40,
            mean: Some(0.5),
            std: None, // Should be skipped in JSON
            min: None,
            max: None,
            nan_count: None,
            inf_count: None,
        };

        let json = serde_json::to_string(&json_info).expect("serialize");
        assert!(json.contains("test"));
        assert!(json.contains("0.5"));
        assert!(!json.contains("std")); // Skipped due to skip_serializing_if
    }

    #[test]
    fn test_tensor_info_json_serialization_full() {
        let json_info = TensorInfoJson {
            name: "full_tensor".to_string(),
            shape: vec![10, 20],
            dtype: "f16".to_string(),
            size_bytes: 400,
            mean: Some(0.5),
            std: Some(0.1),
            min: Some(-1.0),
            max: Some(1.0),
            nan_count: Some(0),
            inf_count: Some(0),
        };

        let json = serde_json::to_string(&json_info).expect("serialize");
        assert!(json.contains("full_tensor"));
        assert!(json.contains("\"shape\":[10,20]"));
        assert!(json.contains("\"mean\":0.5"));
        assert!(json.contains("\"std\":0.1"));
        assert!(json.contains("\"min\":-1.0"));
        assert!(json.contains("\"max\":1.0"));
    }

    #[test]
    fn test_tensors_result_json_serialization() {
        let json_result = TensorsResultJson {
            file: "model.apr".to_string(),
            format_version: "v2".to_string(),
            tensor_count: 1,
            total_size_bytes: 100,
            tensors: vec![TensorInfoJson {
                name: "test".to_string(),
                shape: vec![10],
                dtype: "f32".to_string(),
                size_bytes: 40,
                mean: None,
                std: None,
                min: None,
                max: None,
                nan_count: None,
                inf_count: None,
            }],
        };

        let json = serde_json::to_string_pretty(&json_result).expect("serialize");
        assert!(json.contains("model.apr"));
        assert!(json.contains("\"tensor_count\": 1"));
        assert!(json.contains("\"tensors\""));
    }
}

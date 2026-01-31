//! Tensors command implementation
//!
//! Lists tensor names, shapes, and statistics from APR model files.
//! Useful for debugging model structure and identifying issues.
//!
//! Toyota Way: Genchi Genbutsu - Go to the actual tensors to understand.

use crate::error::CliError;
use crate::output;
use aprender::format::HEADER_SIZE;
use colored::Colorize;
use serde::Serialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

/// Tensor information for display/JSON
#[derive(Serialize, Clone)]
struct TensorInfo {
    name: String,
    shape: Vec<usize>,
    dtype: String,
    size_bytes: usize,
    mean: Option<f32>,
    std: Option<f32>,
    min: Option<f32>,
    max: Option<f32>,
    /// NaN count (spec H8: should be 0)
    nan_count: Option<usize>,
    /// Inf count
    inf_count: Option<usize>,
}

/// Tensors listing result
#[derive(Serialize)]
struct TensorsResult {
    file: String,
    tensor_count: usize,
    total_size_bytes: usize,
    tensors: Vec<TensorInfo>,
}

/// Run the tensors command
pub(crate) fn run(
    path: &Path,
    show_stats: bool,
    filter: Option<&str>,
    json_output: bool,
    limit: usize,
) -> Result<(), CliError> {
    validate_path(path)?;

    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    // Validate header and detect format version
    let magic = validate_header(&mut reader)?;
    // APR v2 uses "APR\0" (null byte) or "APR2" magic
    let is_v2 = &magic == b"APR2" || &magic == b"APR\0";

    // Read metadata size and offset based on format version
    // APR v1 layout: offset 8 = metadata_size, header = 32 bytes
    // APR v2 layout: offset 20 = metadata_size, header = 64 bytes
    let (metadata_size_offset, header_size) = if is_v2 { (20, 64) } else { (8, HEADER_SIZE) };

    let mut size_buf = [0u8; 4];
    reader.seek(SeekFrom::Start(metadata_size_offset))?;
    reader.read_exact(&mut size_buf)?;
    let metadata_size = u32::from_le_bytes(size_buf) as usize;

    // Read metadata
    reader.seek(SeekFrom::Start(header_size as u64))?;
    let mut metadata_bytes = vec![0u8; metadata_size];
    reader.read_exact(&mut metadata_bytes)?;

    // Try to extract tensor info from metadata
    let tensors = extract_tensor_info(&metadata_bytes, show_stats, filter, limit);

    if json_output {
        output_json(path, &tensors);
    } else {
        output_text(path, &tensors, show_stats);
    }

    Ok(())
}

fn validate_path(path: &Path) -> Result<(), CliError> {
    if !path.exists() {
        return Err(CliError::FileNotFound(path.to_path_buf()));
    }
    if !path.is_file() {
        return Err(CliError::NotAFile(path.to_path_buf()));
    }
    Ok(())
}

fn validate_header(reader: &mut BufReader<File>) -> Result<[u8; 4], CliError> {
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic).map_err(|_| {
        CliError::InvalidFormat("File too small to contain valid header".to_string())
    })?;

    if !output::is_valid_magic(&magic) {
        return Err(CliError::InvalidFormat(format!(
            "Invalid magic bytes: expected APRN, APR1, APR2, or APR\\0, got {magic:?}"
        )));
    }

    Ok(magic)
}

fn extract_tensor_info(
    metadata_bytes: &[u8],
    _show_stats: bool,
    filter: Option<&str>,
    limit: usize,
) -> Vec<TensorInfo> {
    // APR v2 uses JSON, APR v1 uses msgpack - try both
    let metadata: HashMap<String, serde_json::Value> = serde_json::from_slice(metadata_bytes)
        .or_else(|_| rmp_serde::from_slice(metadata_bytes))
        .unwrap_or_else(|_| HashMap::new());

    // Try tensor_shapes first, then hyperparameters, then placeholder
    let mut tensors = extract_from_tensor_shapes(&metadata, filter, limit);

    if tensors.is_empty() {
        tensors = extract_from_hyperparameters(&metadata);
    }

    if tensors.is_empty() {
        tensors.push(create_unavailable_tensor_info());
    }

    tensors
}

/// Extract tensor info from tensor_shapes metadata.
fn extract_from_tensor_shapes(
    metadata: &HashMap<String, serde_json::Value>,
    filter: Option<&str>,
    limit: usize,
) -> Vec<TensorInfo> {
    let Some(shapes) = metadata.get("tensor_shapes").and_then(|s| s.as_object()) else {
        return Vec::new();
    };

    shapes
        .iter()
        .filter(|(name, _)| filter.map_or(true, |f| name.contains(f)))
        .take(limit)
        .map(|(name, shape_val)| {
            let shape = parse_shape_array(shape_val);
            let size_bytes = shape.iter().product::<usize>() * 4; // Assume f32
            TensorInfo {
                name: name.clone(),
                shape,
                dtype: "f32".to_string(),
                size_bytes,
                mean: None,
                std: None,
                min: None,
                max: None,
                nan_count: None,
                inf_count: None,
            }
        })
        .collect()
}

/// Parse shape array from JSON value.
fn parse_shape_array(shape_val: &serde_json::Value) -> Vec<usize> {
    shape_val.as_array().map_or(Vec::new(), |arr| {
        arr.iter()
            .filter_map(|v| v.as_u64().map(|n| n as usize))
            .collect()
    })
}

/// Extract tensor info from hyperparameters as fallback.
fn extract_from_hyperparameters(metadata: &HashMap<String, serde_json::Value>) -> Vec<TensorInfo> {
    let Some(hp_obj) = metadata
        .get("hyperparameters")
        .and_then(|hp| hp.as_object())
    else {
        return Vec::new();
    };

    let shape: Vec<usize> = hp_obj
        .iter()
        .filter(|(key, _)| key.contains("dim") || key.contains("size") || key.contains("layers"))
        .filter_map(|(_, val)| val.as_u64().map(|n| n as usize))
        .collect();

    if shape.is_empty() {
        return Vec::new();
    }

    vec![TensorInfo {
        name: "(model structure from hyperparameters)".to_string(),
        shape,
        dtype: "mixed".to_string(),
        size_bytes: 0,
        mean: None,
        std: None,
        min: None,
        max: None,
        nan_count: None,
        inf_count: None,
    }]
}

/// Create placeholder tensor info when no metadata is available.
fn create_unavailable_tensor_info() -> TensorInfo {
    TensorInfo {
        name: "(tensor metadata not available in this APR file)".to_string(),
        shape: vec![],
        dtype: "unknown".to_string(),
        size_bytes: 0,
        mean: None,
        std: None,
        min: None,
        max: None,
        nan_count: None,
        inf_count: None,
    }
}

fn output_json(path: &Path, tensors: &[TensorInfo]) {
    let total_size: usize = tensors.iter().map(|t| t.size_bytes).sum();
    let result = TensorsResult {
        file: path.display().to_string(),
        tensor_count: tensors.len(),
        total_size_bytes: total_size,
        tensors: tensors.to_vec(),
    };

    if let Ok(json) = serde_json::to_string_pretty(&result) {
        println!("{json}");
    }
}

fn output_text(path: &Path, tensors: &[TensorInfo], show_stats: bool) {
    output::section(&format!("Tensors: {}", path.display()));
    println!();

    if tensors.is_empty() {
        println!("  No tensor information available");
        return;
    }

    let total_size: usize = tensors.iter().map(|t| t.size_bytes).sum();
    output::kv("Total tensors", tensors.len());
    output::kv("Total size", output::format_size(total_size as u64));
    println!();

    for tensor in tensors {
        let shape_str = format!("{:?}", tensor.shape);
        println!("  {} [{}] {}", tensor.name, tensor.dtype, shape_str);

        if tensor.size_bytes > 0 {
            println!(
                "    Size: {}",
                output::format_size(tensor.size_bytes as u64)
            );
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::{tempdir, NamedTempFile};

    // ========================================================================
    // Path Validation Tests
    // ========================================================================

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

    // ========================================================================
    // Run Command Tests
    // ========================================================================

    #[test]
    fn test_run_file_not_found() {
        let result = run(
            Path::new("/nonexistent/model.apr"),
            false,
            None,
            false,
            100,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_file_too_small() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create file");
        file.write_all(b"short").expect("write");

        let result = run(file.path(), false, None, false, 100);
        assert!(result.is_err());
        match result {
            Err(CliError::InvalidFormat(msg)) => {
                assert!(msg.contains("too small") || msg.contains("Invalid"));
            }
            _ => panic!("Expected InvalidFormat error"),
        }
    }

    #[test]
    fn test_run_invalid_magic() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create file");
        // Write 32 bytes with invalid magic
        let mut data = [0u8; 32];
        data[0..4].copy_from_slice(b"XXXX");
        file.write_all(&data).expect("write");

        let result = run(file.path(), false, None, false, 100);
        assert!(result.is_err());
        match result {
            Err(CliError::InvalidFormat(msg)) => {
                assert!(msg.contains("Invalid magic"));
            }
            _ => panic!("Expected InvalidFormat error"),
        }
    }

    // ========================================================================
    // TensorInfo Tests
    // ========================================================================

    #[test]
    fn test_tensor_info_serialization() {
        let info = TensorInfo {
            name: "encoder.weight".to_string(),
            shape: vec![384, 80, 3],
            dtype: "f32".to_string(),
            size_bytes: 384 * 80 * 3 * 4,
            mean: Some(0.001),
            std: Some(0.04),
            min: Some(-0.1),
            max: Some(0.1),
            nan_count: Some(0),
            inf_count: Some(0),
        };

        let json = serde_json::to_string(&info).expect("serialize");
        assert!(json.contains("encoder.weight"));
        assert!(json.contains("384"));
        assert!(json.contains("f32"));
    }

    #[test]
    fn test_tensors_result_serialization() {
        let result = TensorsResult {
            file: "model.apr".to_string(),
            tensor_count: 2,
            total_size_bytes: 1000,
            tensors: vec![
                TensorInfo {
                    name: "weight1".to_string(),
                    shape: vec![100],
                    dtype: "f32".to_string(),
                    size_bytes: 400,
                    mean: None,
                    std: None,
                    min: None,
                    max: None,
                    nan_count: None,
                    inf_count: None,
                },
                TensorInfo {
                    name: "weight2".to_string(),
                    shape: vec![150],
                    dtype: "f32".to_string(),
                    size_bytes: 600,
                    mean: None,
                    std: None,
                    min: None,
                    max: None,
                    nan_count: None,
                    inf_count: None,
                },
            ],
        };

        let json = serde_json::to_string(&result).expect("serialize");
        assert!(json.contains("model.apr"));
        assert!(json.contains("tensor_count"));
        assert!(json.contains("weight1"));
        assert!(json.contains("weight2"));
    }

    // ========================================================================
    // Helper Function Tests
    // ========================================================================

    #[test]
    fn test_parse_shape_array_valid() {
        let val = serde_json::json!([10, 20, 30]);
        let shape = parse_shape_array(&val);
        assert_eq!(shape, vec![10, 20, 30]);
    }

    #[test]
    fn test_parse_shape_array_empty() {
        let val = serde_json::json!([]);
        let shape = parse_shape_array(&val);
        assert!(shape.is_empty());
    }

    #[test]
    fn test_parse_shape_array_not_array() {
        let val = serde_json::json!("not an array");
        let shape = parse_shape_array(&val);
        assert!(shape.is_empty());
    }

    #[test]
    fn test_parse_shape_array_mixed_types() {
        let val = serde_json::json!([10, "invalid", 30]);
        let shape = parse_shape_array(&val);
        assert_eq!(shape, vec![10, 30]); // Skips invalid elements
    }

    #[test]
    fn test_create_unavailable_tensor_info() {
        let info = create_unavailable_tensor_info();
        assert!(info.name.contains("not available"));
        assert!(info.shape.is_empty());
        assert_eq!(info.dtype, "unknown");
    }

    // ========================================================================
    // Extract Functions Tests
    // ========================================================================

    #[test]
    fn test_extract_from_tensor_shapes_empty() {
        let metadata: HashMap<String, serde_json::Value> = HashMap::new();
        let tensors = extract_from_tensor_shapes(&metadata, None, 100);
        assert!(tensors.is_empty());
    }

    #[test]
    fn test_extract_from_tensor_shapes_valid() {
        let mut metadata: HashMap<String, serde_json::Value> = HashMap::new();
        let mut shapes = serde_json::Map::new();
        shapes.insert("layer1.weight".to_string(), serde_json::json!([100, 200]));
        shapes.insert("layer2.weight".to_string(), serde_json::json!([200, 300]));
        metadata.insert("tensor_shapes".to_string(), serde_json::Value::Object(shapes));

        let tensors = extract_from_tensor_shapes(&metadata, None, 100);
        assert_eq!(tensors.len(), 2);
    }

    #[test]
    fn test_extract_from_tensor_shapes_with_filter() {
        let mut metadata: HashMap<String, serde_json::Value> = HashMap::new();
        let mut shapes = serde_json::Map::new();
        shapes.insert("encoder.weight".to_string(), serde_json::json!([100, 200]));
        shapes.insert("decoder.weight".to_string(), serde_json::json!([200, 300]));
        metadata.insert("tensor_shapes".to_string(), serde_json::Value::Object(shapes));

        let tensors = extract_from_tensor_shapes(&metadata, Some("encoder"), 100);
        assert_eq!(tensors.len(), 1);
        assert!(tensors[0].name.contains("encoder"));
    }

    #[test]
    fn test_extract_from_tensor_shapes_with_limit() {
        let mut metadata: HashMap<String, serde_json::Value> = HashMap::new();
        let mut shapes = serde_json::Map::new();
        for i in 0..10 {
            shapes.insert(format!("layer{i}.weight"), serde_json::json!([100]));
        }
        metadata.insert("tensor_shapes".to_string(), serde_json::Value::Object(shapes));

        let tensors = extract_from_tensor_shapes(&metadata, None, 3);
        assert_eq!(tensors.len(), 3);
    }

    #[test]
    fn test_extract_from_hyperparameters_empty() {
        let metadata: HashMap<String, serde_json::Value> = HashMap::new();
        let tensors = extract_from_hyperparameters(&metadata);
        assert!(tensors.is_empty());
    }

    #[test]
    fn test_extract_from_hyperparameters_valid() {
        let mut metadata: HashMap<String, serde_json::Value> = HashMap::new();
        let mut hp = serde_json::Map::new();
        hp.insert("hidden_dim".to_string(), serde_json::json!(768));
        hp.insert("num_layers".to_string(), serde_json::json!(12));
        metadata.insert("hyperparameters".to_string(), serde_json::Value::Object(hp));

        let tensors = extract_from_hyperparameters(&metadata);
        assert_eq!(tensors.len(), 1);
        assert!(!tensors[0].shape.is_empty());
    }

    #[test]
    fn test_extract_tensor_info_fallback() {
        // Empty metadata should return unavailable placeholder
        let metadata_bytes = b"{}";
        let tensors = extract_tensor_info(metadata_bytes, false, None, 100);
        assert_eq!(tensors.len(), 1);
        assert!(tensors[0].name.contains("not available"));
    }

    // ========================================================================
    // Output Functions Tests
    // ========================================================================

    #[test]
    fn test_output_json() {
        let tensors = vec![TensorInfo {
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
        }];

        // Should not panic
        output_json(Path::new("test.apr"), &tensors);
    }

    #[test]
    fn test_output_text_empty() {
        output_text(Path::new("test.apr"), &[], false);
    }

    #[test]
    fn test_output_text_with_tensors() {
        let tensors = vec![TensorInfo {
            name: "weight".to_string(),
            shape: vec![100, 200],
            dtype: "f32".to_string(),
            size_bytes: 80000,
            mean: None,
            std: None,
            min: None,
            max: None,
            nan_count: None,
            inf_count: None,
        }];

        output_text(Path::new("test.apr"), &tensors, false);
    }

    #[test]
    fn test_output_text_with_stats() {
        let tensors = vec![TensorInfo {
            name: "weight".to_string(),
            shape: vec![100],
            dtype: "f32".to_string(),
            size_bytes: 400,
            mean: Some(0.001),
            std: Some(0.04),
            min: Some(-0.1),
            max: Some(0.1),
            nan_count: Some(0),
            inf_count: Some(0),
        }];

        output_text(Path::new("test.apr"), &tensors, true);
    }

    #[test]
    fn test_output_text_with_nan_warning() {
        let tensors = vec![TensorInfo {
            name: "weight".to_string(),
            shape: vec![100],
            dtype: "f32".to_string(),
            size_bytes: 400,
            mean: Some(f32::NAN),
            std: Some(0.04),
            min: None,
            max: None,
            nan_count: Some(5),
            inf_count: Some(0),
        }];

        output_text(Path::new("test.apr"), &tensors, true);
    }

    #[test]
    fn test_output_text_with_inf_warning() {
        let tensors = vec![TensorInfo {
            name: "weight".to_string(),
            shape: vec![100],
            dtype: "f32".to_string(),
            size_bytes: 400,
            mean: Some(0.0),
            std: Some(0.04),
            min: None,
            max: None,
            nan_count: Some(0),
            inf_count: Some(3),
        }];

        output_text(Path::new("test.apr"), &tensors, true);
    }
}

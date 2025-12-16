//! Tensors command implementation
//!
//! Lists tensor names, shapes, and statistics from APR model files.
//! Useful for debugging model structure and identifying issues.
//!
//! Toyota Way: Genchi Genbutsu - Go to the actual tensors to understand.

use crate::error::CliError;
use crate::output;
use aprender::format::HEADER_SIZE;
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

    // Validate header
    validate_header(&mut reader)?;

    // Read metadata size
    let mut size_buf = [0u8; 4];
    reader.seek(SeekFrom::Start(8))?;
    reader.read_exact(&mut size_buf)?;
    let metadata_size = u32::from_le_bytes(size_buf) as usize;

    // Read metadata
    reader.seek(SeekFrom::Start(HEADER_SIZE as u64))?;
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

fn validate_header(reader: &mut BufReader<File>) -> Result<(), CliError> {
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic).map_err(|_| {
        CliError::InvalidFormat("File too small to contain valid header".to_string())
    })?;

    if !output::is_valid_magic(&magic) {
        return Err(CliError::InvalidFormat(format!(
            "Invalid magic bytes: expected APRN or APR1, got {magic:?}"
        )));
    }

    Ok(())
}

fn extract_tensor_info(
    metadata_bytes: &[u8],
    _show_stats: bool,
    filter: Option<&str>,
    limit: usize,
) -> Vec<TensorInfo> {
    // Try to parse metadata as MessagePack
    let metadata: HashMap<String, serde_json::Value> =
        rmp_serde::from_slice(metadata_bytes).unwrap_or_else(|_| HashMap::new());

    let mut tensors = Vec::new();

    // Look for tensor_shapes or similar metadata
    if let Some(shapes) = metadata.get("tensor_shapes") {
        if let Some(shapes_map) = shapes.as_object() {
            for (name, shape_val) in shapes_map {
                if let Some(filter_str) = filter {
                    if !name.contains(filter_str) {
                        continue;
                    }
                }

                let shape: Vec<usize> = shape_val.as_array().map_or(Vec::new(), |arr| {
                    arr.iter()
                        .filter_map(|v| v.as_u64().map(|n| n as usize))
                        .collect()
                });

                let size_bytes = shape.iter().product::<usize>() * 4; // Assume f32

                tensors.push(TensorInfo {
                    name: name.clone(),
                    shape,
                    dtype: "f32".to_string(),
                    size_bytes,
                    mean: None,
                    std: None,
                    min: None,
                    max: None,
                });

                if tensors.len() >= limit {
                    break;
                }
            }
        }
    }

    // Also check for hyperparameters that might indicate tensor structure
    if tensors.is_empty() {
        if let Some(hp) = metadata.get("hyperparameters") {
            if let Some(hp_obj) = hp.as_object() {
                // Extract structural info from hyperparameters
                let mut info = TensorInfo {
                    name: "(model structure from hyperparameters)".to_string(),
                    shape: vec![],
                    dtype: "mixed".to_string(),
                    size_bytes: 0,
                    mean: None,
                    std: None,
                    min: None,
                    max: None,
                };

                // Look for dimension info
                for (key, val) in hp_obj {
                    if key.contains("dim") || key.contains("size") || key.contains("layers") {
                        if let Some(n) = val.as_u64() {
                            info.shape.push(n as usize);
                        }
                    }
                }

                if !info.shape.is_empty() {
                    tensors.push(info);
                }
            }
        }
    }

    // If still empty, report that tensor info isn't available
    if tensors.is_empty() {
        tensors.push(TensorInfo {
            name: "(tensor metadata not available in this APR file)".to_string(),
            shape: vec![],
            dtype: "unknown".to_string(),
            size_bytes: 0,
            mean: None,
            std: None,
            min: None,
            max: None,
        });
    }

    tensors
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
                println!("    Stats: mean={mean:.4}, std={std:.4}");
            }
            if let (Some(min), Some(max)) = (tensor.min, tensor.max) {
                println!("    Range: [{min:.4}, {max:.4}]");
            }
        }
    }
}

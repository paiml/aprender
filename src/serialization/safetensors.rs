//! SafeTensors format implementation for model serialization.
//!
//! Implements the SafeTensors format:
//! ```text
//! [8-byte header: u64 metadata length (little-endian)]
//! [JSON metadata: tensor names, dtypes, shapes, data_offsets]
//! [Raw tensor data: F32 values in little-endian]
//! ```
//!
//! Compatible with:
//! - HuggingFace ecosystem
//! - Ollama (can convert to GGUF)
//! - PyTorch, TensorFlow
//! - realizar inference engine

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

/// Metadata for a single tensor in SafeTensors format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorMetadata {
    /// Data type of the tensor (e.g., "F32").
    pub dtype: String,
    /// Shape of the tensor (e.g., `[n_features]` or `[1]`).
    pub shape: Vec<usize>,
    /// Data offsets `[start, end]` in the raw data section.
    pub data_offsets: [usize; 2],
}

/// Complete SafeTensors metadata structure.
/// Uses BTreeMap for deterministic JSON serialization (sorted keys).
pub type SafeTensorsMetadata = BTreeMap<String, TensorMetadata>;

/// Saves tensors to SafeTensors format.
///
/// # Arguments
///
/// * `path` - File path to write to
/// * `tensors` - Map of tensor names to (data, shape) tuples
///
/// # Errors
///
/// Returns an error if:
/// - File writing fails
/// - JSON serialization fails
pub fn save_safetensors<P: AsRef<Path>>(
    path: P,
    tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
) -> Result<(), String> {
    let mut metadata = SafeTensorsMetadata::new();
    let mut raw_data = Vec::new();
    let mut current_offset = 0;

    // Process each tensor (BTreeMap already provides sorted iteration)
    for (name, (data, shape)) in tensors {
        // Calculate data offsets
        let start_offset = current_offset;
        let data_size = data.len() * 4; // F32 = 4 bytes
        let end_offset = current_offset + data_size;

        // Add metadata
        metadata.insert(
            name.clone(),
            TensorMetadata {
                dtype: "F32".to_string(),
                shape: shape.clone(),
                data_offsets: [start_offset, end_offset],
            },
        );

        // Append raw data (little-endian F32)
        for &value in data {
            raw_data.extend_from_slice(&value.to_le_bytes());
        }

        current_offset = end_offset;
    }

    // Serialize metadata to JSON
    let metadata_json =
        serde_json::to_string(&metadata).map_err(|e| format!("JSON serialization failed: {e}"))?;
    let metadata_bytes = metadata_json.as_bytes();
    let metadata_len = metadata_bytes.len() as u64;

    // Write SafeTensors format:
    // [8-byte header: metadata length]
    // [JSON metadata]
    // [Raw tensor data]
    let mut output = Vec::new();
    output.extend_from_slice(&metadata_len.to_le_bytes());
    output.extend_from_slice(metadata_bytes);
    output.extend_from_slice(&raw_data);

    fs::write(path, output).map_err(|e| format!("File write failed: {e}"))?;
    Ok(())
}

/// Loads tensors from SafeTensors format.
///
/// # Arguments
///
/// * `path` - File path to read from
///
/// # Returns
///
/// Returns `(metadata, raw_data)` where:
/// - `metadata` - Tensor metadata mapping
/// - `raw_data` - Raw tensor bytes
///
/// # Errors
///
/// Returns an error if:
/// - File reading fails
/// - Header is invalid (< 8 bytes)
/// - JSON parsing fails
/// - Data section is truncated
pub fn load_safetensors<P: AsRef<Path>>(path: P) -> Result<(SafeTensorsMetadata, Vec<u8>), String> {
    // Read file
    let bytes = fs::read(path).map_err(|e| format!("File read failed: {e}"))?;

    // Validate minimum size (8-byte header)
    if bytes.len() < 8 {
        return Err(format!(
            "Invalid SafeTensors file: file is {} bytes, need at least 8 bytes for header",
            bytes.len()
        ));
    }

    // Read header (8-byte u64 little-endian)
    let header_bytes: [u8; 8] = bytes[0..8]
        .try_into()
        .map_err(|_| "Failed to read header bytes".to_string())?;
    let metadata_len = u64::from_le_bytes(header_bytes) as usize;

    // Validate metadata length
    if metadata_len == 0 {
        return Err("Invalid SafeTensors file: metadata length is 0".to_string());
    }

    if 8 + metadata_len > bytes.len() {
        return Err(format!(
            "Invalid SafeTensors file: metadata length {metadata_len} exceeds file size"
        ));
    }

    // Extract metadata JSON
    let metadata_json = &bytes[8..8 + metadata_len];
    let metadata_str = std::str::from_utf8(metadata_json)
        .map_err(|e| format!("Metadata is not valid UTF-8: {e}"))?;

    // Parse metadata JSON as generic Value first (handles __metadata__ and tensor entries)
    let raw_metadata: serde_json::Value =
        serde_json::from_str(metadata_str).map_err(|e| format!("JSON parsing failed: {e}"))?;

    // Extract tensor metadata only (skip __metadata__ and other non-tensor entries)
    let mut metadata = SafeTensorsMetadata::new();
    if let serde_json::Value::Object(map) = raw_metadata {
        for (key, value) in map {
            // Skip special keys like __metadata__
            if key.starts_with("__") {
                continue;
            }

            // Try to parse as TensorMetadata
            if let Ok(tensor_meta) = serde_json::from_value::<TensorMetadata>(value) {
                metadata.insert(key, tensor_meta);
            }
        }
    }

    // Extract raw data section
    let raw_data = bytes[8 + metadata_len..].to_vec();

    Ok((metadata, raw_data))
}

/// Extracts a tensor from raw SafeTensors data.
///
/// # Arguments
///
/// * `raw_data` - Raw tensor bytes from SafeTensors file
/// * `tensor_meta` - Metadata for the tensor to extract
///
/// # Returns
///
/// Vector of F32 values
///
/// # Errors
///
/// Returns an error if:
/// - Data offsets are invalid
/// - Data is not a multiple of 4 bytes (F32 size)
pub fn extract_tensor(raw_data: &[u8], tensor_meta: &TensorMetadata) -> Result<Vec<f32>, String> {
    let [start, end] = tensor_meta.data_offsets;

    // Validate offsets
    if end > raw_data.len() {
        return Err(format!(
            "Invalid data offset: end={} exceeds data size={}",
            end,
            raw_data.len()
        ));
    }

    if start >= end {
        return Err(format!("Invalid data offset: start={start} >= end={end}"));
    }

    // Extract bytes
    let tensor_bytes = &raw_data[start..end];

    // Validate size (must be multiple of 4 for F32)
    if tensor_bytes.len() % 4 != 0 {
        return Err(format!(
            "Invalid tensor data: size {} is not a multiple of 4 bytes (F32)",
            tensor_bytes.len()
        ));
    }

    // Parse F32 values (little-endian)
    let mut values = Vec::new();
    for chunk in tensor_bytes.chunks_exact(4) {
        let f32_bytes: [u8; 4] = chunk
            .try_into()
            .map_err(|_| "Failed to read F32 bytes".to_string())?;
        values.push(f32::from_le_bytes(f32_bytes));
    }

    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_save_and_load_safetensors() {
        let path = "/tmp/test_safetensors_module.safetensors";

        // Create test tensors
        let mut tensors = BTreeMap::new();
        tensors.insert("weights".to_string(), (vec![1.0, 2.0, 3.0], vec![3]));
        tensors.insert("bias".to_string(), (vec![0.5], vec![1]));

        // Save
        save_safetensors(path, &tensors)
            .expect("Failed to save test tensors to SafeTensors format");

        // Load
        let (metadata, raw_data) =
            load_safetensors(path).expect("Failed to load test SafeTensors file");

        // Verify metadata
        assert!(metadata.contains_key("weights"));
        assert!(metadata.contains_key("bias"));

        // Extract and verify tensors
        let weights = extract_tensor(&raw_data, &metadata["weights"])
            .expect("Failed to extract weights tensor from raw data");
        assert_eq!(weights, vec![1.0, 2.0, 3.0]);

        let bias = extract_tensor(&raw_data, &metadata["bias"])
            .expect("Failed to extract bias tensor from raw data");
        assert_eq!(bias, vec![0.5]);

        // Cleanup
        fs::remove_file(path).ok();
    }

    #[test]
    fn test_save_safetensors_header_format() {
        let path = "/tmp/test_header_format.safetensors";

        let mut tensors = BTreeMap::new();
        tensors.insert("test".to_string(), (vec![1.0], vec![1]));

        save_safetensors(path, &tensors)
            .expect("Failed to save test tensor for header format verification");

        // Read and verify header
        let bytes =
            fs::read(path).expect("Failed to read test SafeTensors file for header verification");
        assert!(bytes.len() >= 8, "File must have at least 8-byte header");

        let header_bytes: [u8; 8] = bytes[0..8]
            .try_into()
            .expect("Failed to convert first 8 bytes to header array (file has at least 8 bytes)");
        let metadata_len = u64::from_le_bytes(header_bytes);
        assert!(metadata_len > 0, "Metadata length must be > 0");

        fs::remove_file(path).ok();
    }

    #[test]
    fn test_load_safetensors_corrupted_header() {
        let path = "/tmp/test_corrupted_header.safetensors";

        // Write invalid file (< 8 bytes)
        fs::write(path, [1, 2, 3]).expect("Failed to write test file with corrupted header");

        let result = load_safetensors(path);
        assert!(result.is_err());
        assert!(result
            .expect_err("Should fail with corrupted header size check")
            .contains("8 bytes"));

        fs::remove_file(path).ok();
    }

    #[test]
    fn test_load_safetensors_nonexistent_file() {
        let result = load_safetensors("/tmp/nonexistent_file_xyz.safetensors");
        assert!(result.is_err());
        let err = result.expect_err("Should fail when file not found");
        assert!(
            err.contains("No such file") || err.contains("not found"),
            "Error should mention file not found: {err}"
        );
    }

    #[test]
    fn test_extract_tensor_invalid_offsets() {
        let raw_data = vec![0u8; 16];
        let meta = TensorMetadata {
            dtype: "F32".to_string(),
            shape: vec![1],
            data_offsets: [0, 100], // Exceeds data size
        };

        let result = extract_tensor(&raw_data, &meta);
        assert!(result.is_err());
        assert!(result
            .expect_err("Should fail when tensor offset exceeds data size")
            .contains("exceeds data size"));
    }

    #[test]
    fn test_deterministic_serialization() {
        // Verify that serialization is deterministic (sorted keys)
        let path1 = "/tmp/test_det1.safetensors";
        let path2 = "/tmp/test_det2.safetensors";

        let mut tensors = BTreeMap::new();
        tensors.insert("z_last".to_string(), (vec![3.0], vec![1]));
        tensors.insert("a_first".to_string(), (vec![1.0], vec![1]));
        tensors.insert("m_middle".to_string(), (vec![2.0], vec![1]));

        // Save twice
        save_safetensors(path1, &tensors).expect("Failed to save first deterministic test file");
        save_safetensors(path2, &tensors).expect("Failed to save second deterministic test file");

        // Files should be identical (deterministic)
        let bytes1 = fs::read(path1).expect("Failed to read first deterministic test file");
        let bytes2 = fs::read(path2).expect("Failed to read second deterministic test file");
        assert_eq!(bytes1, bytes2, "Serialization must be deterministic");

        fs::remove_file(path1).ok();
        fs::remove_file(path2).ok();
    }
}

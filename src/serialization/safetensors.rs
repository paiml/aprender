//! `SafeTensors` format implementation for model serialization.
//!
//! Implements the `SafeTensors` format:
//! ```text
//! [8-byte header: u64 metadata length (little-endian)]
//! [JSON metadata: tensor names, dtypes, shapes, data_offsets]
//! [Raw tensor data: F32 values in little-endian]
//! ```
//!
//! Compatible with:
//! - `HuggingFace` ecosystem
//! - Ollama (can convert to GGUF)
//! - `PyTorch`, TensorFlow
//! - realizar inference engine

use crate::bundle::MappedFile;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

/// Metadata for a single tensor in `SafeTensors` format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorMetadata {
    /// Data type of the tensor (e.g., "F32").
    pub dtype: String,
    /// Shape of the tensor (e.g., `[n_features]` or `[1]`).
    pub shape: Vec<usize>,
    /// Data offsets `[start, end]` in the raw data section.
    pub data_offsets: [usize; 2],
}

/// Complete `SafeTensors` metadata structure.
/// Uses `BTreeMap` for deterministic JSON serialization (sorted keys).
pub type SafeTensorsMetadata = BTreeMap<String, TensorMetadata>;

/// Saves tensors to `SafeTensors` format.
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

/// Loads tensors from `SafeTensors` format.
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
    let bytes = fs::read(path).map_err(|e| format!("File read failed: {e}"))?;
    let metadata_len = validate_and_read_header(&bytes)?;
    let metadata = parse_metadata(&bytes, metadata_len)?;
    let raw_data = bytes[8 + metadata_len..].to_vec();
    Ok((metadata, raw_data))
}

/// Memory-mapped `SafeTensors` file for zero-copy tensor access.
///
/// Uses `bundle::MappedFile` for efficient large model loading.
/// Tensors are accessed directly from the mapped memory without copying.
///
/// # Example
///
/// ```rust,ignore
/// use aprender::serialization::safetensors::MappedSafeTensors;
///
/// let mapped = MappedSafeTensors::open("model.safetensors")?;
/// let weight = mapped.get_tensor("model.embed_tokens.weight")?;
/// ```
#[derive(Debug)]
pub struct MappedSafeTensors {
    /// Memory-mapped file
    mmap: MappedFile,
    /// Parsed metadata
    metadata: SafeTensorsMetadata,
    /// Offset where tensor data begins (after header + metadata JSON)
    data_offset: usize,
}

impl MappedSafeTensors {
    /// Open a `SafeTensors` file with memory mapping.
    ///
    /// # Errors
    ///
    /// Returns error if file cannot be opened or format is invalid.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let mmap = MappedFile::open(path).map_err(|e| format!("mmap failed: {e}"))?;
        let bytes = mmap.as_slice();

        let metadata_len = validate_and_read_header(bytes)?;
        let metadata = parse_metadata(bytes, metadata_len)?;
        let data_offset = 8 + metadata_len;

        Ok(Self {
            mmap,
            metadata,
            data_offset,
        })
    }

    /// Get tensor metadata by name.
    #[must_use]
    pub fn get_metadata(&self, name: &str) -> Option<&TensorMetadata> {
        self.metadata.get(name)
    }

    /// Get all tensor names.
    #[must_use]
    pub fn tensor_names(&self) -> Vec<&str> {
        self.metadata.keys().map(String::as_str).collect()
    }

    /// Extract tensor data as f32 values (BF16/F16 are converted to F32).
    ///
    /// # Errors
    ///
    /// Returns error if tensor not found or data is invalid.
    pub fn get_tensor(&self, name: &str) -> Result<Vec<f32>, String> {
        let meta = self
            .metadata
            .get(name)
            .ok_or_else(|| format!("Tensor '{name}' not found"))?;

        let bytes = self.mmap.as_slice();
        let [start, end] = meta.data_offsets;
        let abs_start = self.data_offset + start;
        let abs_end = self.data_offset + end;

        if abs_end > bytes.len() {
            return Err(format!(
                "Tensor '{name}' data out of bounds: {abs_end} > {}",
                bytes.len()
            ));
        }

        let tensor_bytes = &bytes[abs_start..abs_end];

        // Handle different dtypes
        match meta.dtype.as_str() {
            "F32" => extract_f32(tensor_bytes),
            "BF16" => extract_bf16_to_f32(tensor_bytes),
            "F16" => extract_f16_to_f32(tensor_bytes),
            other => Err(format!("Unsupported dtype for '{name}': {other}")),
        }
    }

    /// Get raw tensor bytes (zero-copy).
    #[must_use]
    pub fn get_tensor_bytes(&self, name: &str) -> Option<&[u8]> {
        let meta = self.metadata.get(name)?;
        let [start, end] = meta.data_offsets;
        let abs_start = self.data_offset + start;
        let abs_end = self.data_offset + end;

        self.mmap.slice(abs_start, abs_end)
    }

    /// Number of tensors in the file.
    #[must_use]
    pub fn len(&self) -> usize {
        self.metadata.len()
    }

    /// Check if file has no tensors.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.metadata.is_empty()
    }
}

fn validate_and_read_header(bytes: &[u8]) -> Result<usize, String> {
    if bytes.len() < 8 {
        return Err(format!(
            "Invalid SafeTensors file: file is {} bytes, need at least 8 bytes for header",
            bytes.len()
        ));
    }

    let header_bytes: [u8; 8] = bytes[0..8]
        .try_into()
        .map_err(|_| "Failed to read header bytes".to_string())?;
    let metadata_len = u64::from_le_bytes(header_bytes) as usize;

    if metadata_len == 0 {
        return Err("Invalid SafeTensors file: metadata length is 0".to_string());
    }

    if 8 + metadata_len > bytes.len() {
        return Err(format!(
            "Invalid SafeTensors file: metadata length {metadata_len} exceeds file size"
        ));
    }

    Ok(metadata_len)
}

fn parse_metadata(bytes: &[u8], metadata_len: usize) -> Result<SafeTensorsMetadata, String> {
    let metadata_json = &bytes[8..8 + metadata_len];
    let metadata_str = std::str::from_utf8(metadata_json)
        .map_err(|e| format!("Metadata is not valid UTF-8: {e}"))?;

    let raw_metadata: serde_json::Value =
        serde_json::from_str(metadata_str).map_err(|e| format!("JSON parsing failed: {e}"))?;

    let mut metadata = SafeTensorsMetadata::new();
    if let serde_json::Value::Object(map) = raw_metadata {
        for (key, value) in map {
            if key.starts_with("__") {
                continue;
            }
            if let Ok(tensor_meta) = serde_json::from_value::<TensorMetadata>(value) {
                metadata.insert(key, tensor_meta);
            }
        }
    }

    Ok(metadata)
}

/// Extracts a tensor from raw `SafeTensors` data.
///
/// # Arguments
///
/// * `raw_data` - Raw tensor bytes from `SafeTensors` file
/// * `tensor_meta` - Metadata for the tensor to extract
///
/// # Returns
///
/// Vector of F32 values (BF16/F16 are converted to F32)
///
/// # Errors
///
/// Returns an error if:
/// - Data offsets are invalid
/// - Data size doesn't match dtype requirements
/// - Unsupported dtype
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

    // Handle different dtypes
    match tensor_meta.dtype.as_str() {
        "F32" => extract_f32(tensor_bytes),
        "BF16" => extract_bf16_to_f32(tensor_bytes),
        "F16" => extract_f16_to_f32(tensor_bytes),
        other => Err(format!(
            "Unsupported dtype: {other}. Supported: F32, BF16, F16"
        )),
    }
}

/// Extract F32 tensor data
fn extract_f32(tensor_bytes: &[u8]) -> Result<Vec<f32>, String> {
    if tensor_bytes.len() % 4 != 0 {
        return Err(format!(
            "Invalid F32 tensor data: size {} is not a multiple of 4 bytes",
            tensor_bytes.len()
        ));
    }

    let values: Vec<f32> = tensor_bytes
        .chunks_exact(4)
        .map(|chunk| {
            let bytes: [u8; 4] = chunk.try_into().expect("chunk is 4 bytes");
            f32::from_le_bytes(bytes)
        })
        .collect();

    Ok(values)
}

/// Extract BF16 tensor data and convert to F32
fn extract_bf16_to_f32(tensor_bytes: &[u8]) -> Result<Vec<f32>, String> {
    if tensor_bytes.len() % 2 != 0 {
        return Err(format!(
            "Invalid BF16 tensor data: size {} is not a multiple of 2 bytes",
            tensor_bytes.len()
        ));
    }

    let values: Vec<f32> = tensor_bytes
        .chunks_exact(2)
        .map(|chunk| {
            let bytes: [u8; 2] = chunk.try_into().expect("chunk is 2 bytes");
            bf16_to_f32(bytes)
        })
        .collect();

    Ok(values)
}

/// Extract F16 tensor data and convert to F32
fn extract_f16_to_f32(tensor_bytes: &[u8]) -> Result<Vec<f32>, String> {
    if tensor_bytes.len() % 2 != 0 {
        return Err(format!(
            "Invalid F16 tensor data: size {} is not a multiple of 2 bytes",
            tensor_bytes.len()
        ));
    }

    let values: Vec<f32> = tensor_bytes
        .chunks_exact(2)
        .map(|chunk| {
            let bytes: [u8; 2] = chunk.try_into().expect("chunk is 2 bytes");
            f16_to_f32(bytes)
        })
        .collect();

    Ok(values)
}

/// Convert BF16 (Brain Float 16) to F32
///
/// BF16 has the same exponent range as F32 (8 bits) but only 7 mantissa bits.
/// Conversion is done by zero-padding the mantissa.
#[inline]
fn bf16_to_f32(bytes: [u8; 2]) -> f32 {
    // BF16 is the upper 16 bits of an F32
    let bits = u32::from_le_bytes([0, 0, bytes[0], bytes[1]]);
    f32::from_bits(bits)
}

/// Convert F16 (IEEE 754 half-precision) to F32
///
/// F16 has 5 exponent bits and 10 mantissa bits.
#[inline]
fn f16_to_f32(bytes: [u8; 2]) -> f32 {
    let h = u16::from_le_bytes(bytes);

    // Extract components
    let sign = (h >> 15) & 1;
    let exp = (h >> 10) & 0x1F;
    let mant = h & 0x3FF;

    let f32_bits = if exp == 0 {
        if mant == 0 {
            // Zero (positive or negative)
            u32::from(sign) << 31
        } else {
            // Subnormal: convert to normalized F32
            let mut m = mant;
            let mut e: i32 = -14; // F16 subnormal exponent bias adjustment
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3FF; // Remove implicit 1
            let exp32 = (e + 127) as u32; // F32 bias
            (u32::from(sign) << 31) | (exp32 << 23) | (u32::from(m) << 13)
        }
    } else if exp == 31 {
        // Inf or NaN
        let mant32 = u32::from(mant) << 13;
        (u32::from(sign) << 31) | (0xFF << 23) | mant32
    } else {
        // Normal number
        let exp32 = u32::from(exp) - 15 + 127; // Adjust bias
        let mant32 = u32::from(mant) << 13;
        (u32::from(sign) << 31) | (exp32 << 23) | mant32
    };

    f32::from_bits(f32_bits)
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

    // =========================================================================
    // Coverage boost: MappedSafeTensors API tests
    // =========================================================================

    #[test]
    fn test_mapped_safetensors_full_api() {
        let path = "/tmp/test_mapped_api.safetensors";

        // Create multi-tensor file
        let mut tensors = BTreeMap::new();
        tensors.insert("weight".to_string(), (vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]));
        tensors.insert("bias".to_string(), (vec![0.5, 0.5], vec![2]));
        tensors.insert("scale".to_string(), (vec![1.0], vec![1]));

        save_safetensors(path, &tensors).expect("save");

        // Test MappedSafeTensors API
        let mapped = MappedSafeTensors::open(path).expect("open");

        // len/is_empty
        assert_eq!(mapped.len(), 3);
        assert!(!mapped.is_empty());

        // tensor_names
        let names = mapped.tensor_names();
        assert!(names.contains(&"weight"));
        assert!(names.contains(&"bias"));
        assert!(names.contains(&"scale"));

        // get_metadata
        let meta = mapped.get_metadata("weight").expect("metadata");
        assert_eq!(meta.dtype, "F32");
        assert_eq!(meta.shape, vec![2, 2]);

        assert!(mapped.get_metadata("nonexistent").is_none());

        // get_tensor
        let weight = mapped.get_tensor("weight").expect("tensor");
        assert_eq!(weight, vec![1.0, 2.0, 3.0, 4.0]);

        // get_tensor_bytes
        let bytes = mapped.get_tensor_bytes("bias").expect("bytes");
        assert_eq!(bytes.len(), 8); // 2 f32 = 8 bytes

        // Error: tensor not found
        let err = mapped.get_tensor("missing").unwrap_err();
        assert!(err.contains("not found"));

        fs::remove_file(path).ok();
    }

    #[test]
    fn test_mapped_safetensors_empty_file() {
        let path = "/tmp/test_empty_tensors.safetensors";

        let tensors = BTreeMap::new();
        save_safetensors(path, &tensors).expect("save empty");

        let mapped = MappedSafeTensors::open(path).expect("open empty");
        assert!(mapped.is_empty());
        assert_eq!(mapped.len(), 0);
        assert!(mapped.tensor_names().is_empty());

        fs::remove_file(path).ok();
    }

    #[test]
    fn test_validate_header_metadata_zero() {
        let path = "/tmp/test_zero_meta.safetensors";

        // Create file with 0 metadata length
        let bytes: Vec<u8> = vec![0, 0, 0, 0, 0, 0, 0, 0];
        fs::write(path, &bytes).expect("write");

        let result = MappedSafeTensors::open(path);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("metadata length is 0"));

        fs::remove_file(path).ok();
    }

    #[test]
    fn test_validate_header_metadata_exceeds_file() {
        let path = "/tmp/test_exceed_meta.safetensors";

        // Create file with metadata length > file size
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&1000u64.to_le_bytes()); // claim 1000 bytes
        bytes.extend_from_slice(b"{}"); // only 2 bytes of metadata
        fs::write(path, &bytes).expect("write");

        let result = MappedSafeTensors::open(path);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("exceeds file size"));

        fs::remove_file(path).ok();
    }

    #[test]
    fn test_parse_metadata_with_dunder_keys() {
        let path = "/tmp/test_dunder.safetensors";

        // Manually create file with __metadata__ key (should be skipped)
        let metadata = r#"{"__metadata__":{"format":"pt"},"tensor":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let meta_bytes = metadata.as_bytes();

        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(meta_bytes.len() as u64).to_le_bytes());
        bytes.extend_from_slice(meta_bytes);
        bytes.extend_from_slice(&1.0f32.to_le_bytes());
        fs::write(path, &bytes).expect("write");

        let mapped = MappedSafeTensors::open(path).expect("open");
        assert_eq!(mapped.len(), 1); // only "tensor", not "__metadata__"
        assert!(mapped.get_metadata("__metadata__").is_none());
        assert!(mapped.get_metadata("tensor").is_some());

        fs::remove_file(path).ok();
    }

    #[test]
    fn test_extract_bf16() {
        // BF16: 0x3F80 = 1.0 in BF16
        let bf16_bytes = vec![0x80, 0x3F, 0x00, 0x40]; // 1.0, 2.0
        let result = extract_bf16_to_f32(&bf16_bytes).expect("bf16");
        assert_eq!(result.len(), 2);
        assert!((result[0] - 1.0).abs() < 0.01);
        assert!((result[1] - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_extract_f16() {
        // F16: 0x3C00 = 1.0 in F16
        let f16_bytes = vec![0x00, 0x3C, 0x00, 0x40]; // 1.0, 2.0
        let result = extract_f16_to_f32(&f16_bytes).expect("f16");
        assert_eq!(result.len(), 2);
        assert!((result[0] - 1.0).abs() < 0.01);
        assert!((result[1] - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_unsupported_dtype() {
        let path = "/tmp/test_unsupported.safetensors";

        // Create file with unsupported dtype
        let metadata = r#"{"tensor":{"dtype":"INT8","shape":[1],"data_offsets":[0,1]}}"#;
        let meta_bytes = metadata.as_bytes();

        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(meta_bytes.len() as u64).to_le_bytes());
        bytes.extend_from_slice(meta_bytes);
        bytes.push(42); // 1 byte of data
        fs::write(path, &bytes).expect("write");

        let mapped = MappedSafeTensors::open(path).expect("open");
        let result = mapped.get_tensor("tensor");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Unsupported dtype"));

        fs::remove_file(path).ok();
    }

    #[test]
    fn test_tensor_out_of_bounds() {
        let path = "/tmp/test_oob.safetensors";

        // Create file with tensor pointing past end
        let metadata = r#"{"tensor":{"dtype":"F32","shape":[100],"data_offsets":[0,400]}}"#;
        let meta_bytes = metadata.as_bytes();

        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(meta_bytes.len() as u64).to_le_bytes());
        bytes.extend_from_slice(meta_bytes);
        bytes.extend_from_slice(&[0u8; 16]); // only 16 bytes, not 400
        fs::write(path, &bytes).expect("write");

        let mapped = MappedSafeTensors::open(path).expect("open");
        let result = mapped.get_tensor("tensor");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("out of bounds"));

        fs::remove_file(path).ok();
    }

    #[test]
    fn test_invalid_utf8_metadata() {
        let path = "/tmp/test_invalid_utf8.safetensors";

        // Create file with invalid UTF-8 in metadata
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&4u64.to_le_bytes());
        bytes.extend_from_slice(&[0xFF, 0xFE, 0x00, 0x01]); // Invalid UTF-8
        fs::write(path, &bytes).expect("write");

        let result = MappedSafeTensors::open(path);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("UTF-8"));

        fs::remove_file(path).ok();
    }

    #[test]
    fn test_invalid_json_metadata() {
        let path = "/tmp/test_invalid_json.safetensors";

        let invalid_json = b"not valid json{";
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(invalid_json.len() as u64).to_le_bytes());
        bytes.extend_from_slice(invalid_json);
        fs::write(path, &bytes).expect("write");

        let result = MappedSafeTensors::open(path);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("JSON"));

        fs::remove_file(path).ok();
    }
}

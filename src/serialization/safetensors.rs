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

/// GH-205: Original dtype from SafeTensors file.
///
/// Used for F16 passthrough to avoid precision loss from F16→F32→F16 conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafeTensorsDType {
    /// 32-bit float
    F32,
    /// 16-bit float (IEEE 754 half-precision)
    F16,
    /// Brain float 16
    BF16,
}

impl SafeTensorsDType {
    /// Bytes per element
    #[must_use]
    pub fn bytes_per_element(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
        }
    }
}

/// GH-205: Raw tensor data with dtype information preserved.
///
/// This struct carries tensor data without dtype conversion, enabling
/// lossless F16 passthrough through the import pipeline.
#[derive(Debug, Clone)]
pub struct RawTensorData {
    /// Original dtype from SafeTensors
    pub dtype: SafeTensorsDType,
    /// Tensor shape
    pub shape: Vec<usize>,
    /// Raw bytes (no conversion applied)
    pub bytes: Vec<u8>,
}

impl RawTensorData {
    /// Convert to f32 values (performs conversion if needed)
    pub fn to_f32(&self) -> Result<Vec<f32>, String> {
        match self.dtype {
            SafeTensorsDType::F32 => extract_f32(&self.bytes),
            SafeTensorsDType::F16 => extract_f16_to_f32(&self.bytes),
            SafeTensorsDType::BF16 => extract_bf16_to_f32(&self.bytes),
        }
    }

    /// Check if this is F16 data (for passthrough optimization)
    #[must_use]
    pub fn is_f16(&self) -> bool {
        self.dtype == SafeTensorsDType::F16
    }

    /// Check if this is BF16 data
    #[must_use]
    pub fn is_bf16(&self) -> bool {
        self.dtype == SafeTensorsDType::BF16
    }
}

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

/// User metadata from `SafeTensors` `__metadata__` header section.
/// SafeTensors stores arbitrary string→string metadata under `__metadata__`.
pub type UserMetadata = BTreeMap<String, String>;

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

/// Saves tensors to `SafeTensors` format with user metadata (PMAT-223).
///
/// Like `save_safetensors`, but includes a `__metadata__` section in the
/// SafeTensors header for preserving arbitrary user metadata through
/// format conversion round-trips.
///
/// # Errors
///
/// Returns an error if file writing or JSON serialization fails.
pub fn save_safetensors_with_metadata<P: AsRef<Path>>(
    path: P,
    tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    user_metadata: &UserMetadata,
) -> Result<(), String> {
    // Build a serde_json::Value containing both __metadata__ and tensor entries
    let mut header = serde_json::Map::new();

    // Add __metadata__ if non-empty
    if !user_metadata.is_empty() {
        let meta_obj: serde_json::Map<String, serde_json::Value> = user_metadata
            .iter()
            .map(|(k, v)| (k.clone(), serde_json::Value::String(v.clone())))
            .collect();
        header.insert(
            "__metadata__".to_string(),
            serde_json::Value::Object(meta_obj),
        );
    }

    // Add tensor metadata (BTreeMap already provides sorted iteration)
    let mut raw_data = Vec::new();
    let mut current_offset = 0;

    for (name, (data, shape)) in tensors {
        let start_offset = current_offset;
        let data_size = data.len() * 4;
        let end_offset = current_offset + data_size;

        #[allow(clippy::disallowed_methods)] // serde_json::json! macro uses unwrap() internally
        let tensor_meta = serde_json::json!({
            "dtype": "F32",
            "shape": shape,
            "data_offsets": [start_offset, end_offset]
        });
        header.insert(name.clone(), tensor_meta);

        for &value in data {
            raw_data.extend_from_slice(&value.to_le_bytes());
        }
        current_offset = end_offset;
    }

    let metadata_json =
        serde_json::to_string(&header).map_err(|e| format!("JSON serialization failed: {e}"))?;
    let metadata_bytes = metadata_json.as_bytes();
    let metadata_len = metadata_bytes.len() as u64;

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
    let (metadata, _user_metadata) = parse_metadata(&bytes, metadata_len)?;
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
    /// User metadata from `__metadata__` header section (PMAT-223)
    user_metadata: UserMetadata,
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
        let (metadata, user_metadata) = parse_metadata(bytes, metadata_len)?;
        let data_offset = 8 + metadata_len;

        Ok(Self {
            mmap,
            metadata,
            user_metadata,
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

    /// GH-205 FIX: Get tensor with original dtype preserved (no F16→F32 conversion).
    ///
    /// Returns the raw tensor bytes along with dtype and shape information.
    /// This enables F16 passthrough: SafeTensors F16 → APR F16 without precision loss.
    ///
    /// For F32 tensors, returns f32 data directly.
    /// For F16/BF16 tensors, returns the raw bytes without conversion.
    ///
    /// # Errors
    ///
    /// Returns error if tensor not found or data is invalid.
    pub fn get_tensor_raw(&self, name: &str) -> Result<RawTensorData, String> {
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

        // Parse dtype string to enum
        let dtype = match meta.dtype.as_str() {
            "F32" => SafeTensorsDType::F32,
            "F16" => SafeTensorsDType::F16,
            "BF16" => SafeTensorsDType::BF16,
            other => return Err(format!("Unsupported dtype for '{name}': {other}")),
        };

        Ok(RawTensorData {
            dtype,
            shape: meta.shape.clone(),
            bytes: tensor_bytes.to_vec(),
        })
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

    /// Get user metadata from `__metadata__` header section (PMAT-223).
    ///
    /// SafeTensors files may contain arbitrary string→string metadata under
    /// the `__metadata__` key. This method exposes that data for preservation
    /// during format conversion.
    #[must_use]
    pub fn user_metadata(&self) -> &UserMetadata {
        &self.user_metadata
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

fn parse_metadata(
    bytes: &[u8],
    metadata_len: usize,
) -> Result<(SafeTensorsMetadata, UserMetadata), String> {
    let metadata_json = &bytes[8..8 + metadata_len];
    let metadata_str = std::str::from_utf8(metadata_json)
        .map_err(|e| format!("Metadata is not valid UTF-8: {e}"))?;

    let raw_metadata: serde_json::Value =
        serde_json::from_str(metadata_str).map_err(|e| format!("JSON parsing failed: {e}"))?;

    let serde_json::Value::Object(map) = raw_metadata else {
        return Ok((SafeTensorsMetadata::new(), UserMetadata::new()));
    };

    let mut metadata = SafeTensorsMetadata::new();
    let mut user_metadata = UserMetadata::new();

    for (key, value) in map {
        if key == "__metadata__" {
            // PMAT-223: Extract user metadata instead of discarding it
            extract_user_metadata(value, &mut user_metadata);
            continue;
        }
        if key.starts_with("__") {
            continue;
        }
        if let Ok(tensor_meta) = serde_json::from_value::<TensorMetadata>(value) {
            metadata.insert(key, tensor_meta);
        }
    }

    Ok((metadata, user_metadata))
}

/// Extracts string key-value pairs from a `__metadata__` JSON object into `UserMetadata`.
fn extract_user_metadata(value: serde_json::Value, user_metadata: &mut UserMetadata) {
    let serde_json::Value::Object(meta_map) = value else {
        return;
    };
    for (mk, mv) in meta_map {
        if let serde_json::Value::String(s) = mv {
            user_metadata.insert(mk, s);
        }
    }
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
        // GH-205 FIX: Rearrange to avoid underflow in debug mode
        // F16 bias is 15, F32 bias is 127, so we add 112 (127 - 15)
        let exp32 = u32::from(exp) + 112; // Was: exp - 15 + 127 (underflows if exp < 15)
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

        // PMAT-223: __metadata__ is now extracted as user metadata, not discarded
        let metadata = r#"{"__metadata__":{"format":"pt","training_run_id":"12345"},"tensor":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let meta_bytes = metadata.as_bytes();

        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(meta_bytes.len() as u64).to_le_bytes());
        bytes.extend_from_slice(meta_bytes);
        bytes.extend_from_slice(&1.0f32.to_le_bytes());
        fs::write(path, &bytes).expect("write");

        let mapped = MappedSafeTensors::open(path).expect("open");
        assert_eq!(mapped.len(), 1); // only "tensor", not "__metadata__"
        assert!(mapped.get_metadata("__metadata__").is_none()); // Not a tensor
        assert!(mapped.get_metadata("tensor").is_some());

        // PMAT-223: User metadata IS extracted
        let user_meta = mapped.user_metadata();
        assert_eq!(user_meta.len(), 2);
        assert_eq!(user_meta.get("format"), Some(&"pt".to_string()));
        assert_eq!(user_meta.get("training_run_id"), Some(&"12345".to_string()));

        fs::remove_file(path).ok();
    }

    #[test]
    fn test_save_safetensors_with_metadata_round_trip() {
        let path = "/tmp/test_metadata_roundtrip.safetensors";
        let mut tensors = BTreeMap::new();
        tensors.insert(
            "weight".to_string(),
            (vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]),
        );

        let mut user_metadata = UserMetadata::new();
        user_metadata.insert("my_run_id".to_string(), "test_123".to_string());
        user_metadata.insert("framework".to_string(), "pytorch".to_string());

        // Write with metadata
        save_safetensors_with_metadata(path, &tensors, &user_metadata).expect("save");

        // Read back and verify metadata round-trips
        let mapped = MappedSafeTensors::open(path).expect("open");
        assert_eq!(mapped.len(), 1);
        assert!(mapped.get_metadata("weight").is_some());

        let restored = mapped.user_metadata();
        assert_eq!(restored.get("my_run_id"), Some(&"test_123".to_string()));
        assert_eq!(restored.get("framework"), Some(&"pytorch".to_string()));

        fs::remove_file(path).ok();
    }

    #[test]
    fn test_empty_user_metadata_no_dunder_section() {
        let path = "/tmp/test_no_dunder.safetensors";

        // File without __metadata__ should have empty user_metadata
        let metadata = r#"{"tensor":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let meta_bytes = metadata.as_bytes();

        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(meta_bytes.len() as u64).to_le_bytes());
        bytes.extend_from_slice(meta_bytes);
        bytes.extend_from_slice(&1.0f32.to_le_bytes());
        fs::write(path, &bytes).expect("write");

        let mapped = MappedSafeTensors::open(path).expect("open");
        assert!(mapped.user_metadata().is_empty());

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

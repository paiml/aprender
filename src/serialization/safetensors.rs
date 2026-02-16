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

/// PMAT-260: Serialize F32 data as BF16 bytes.
///
/// BF16 is the upper 16 bits of F32. For data that originated as BF16,
/// the F32 representation has zeros in the lower 16 bits, so this
/// truncation is lossless.
fn f32_slice_to_bf16_bytes(data: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(data.len() * 2);
    for &value in data {
        let bits = value.to_bits();
        let bf16 = (bits >> 16) as u16;
        bytes.extend_from_slice(&bf16.to_le_bytes());
    }
    bytes
}

/// PMAT-260: Serialize F32 data as F16 bytes.
///
/// Uses IEEE 754 half-precision format. For data that originated as F16,
/// the F32 representation is exact, so this round-trip is lossless.
fn f32_slice_to_f16_bytes(data: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(data.len() * 2);
    for &value in data {
        let bits = value.to_bits();
        let sign = (bits >> 16) & 0x8000;
        let exponent = ((bits >> 23) & 0xFF) as i32;
        let mantissa = bits & 0x007F_FFFF;

        let f16_bits = if exponent == 0xFF {
            // Inf/NaN
            sign | 0x7C00 | if mantissa != 0 { 0x0200 } else { 0 }
        } else if exponent > 142 {
            // Overflow → Inf
            sign | 0x7C00
        } else if exponent < 113 {
            // Underflow → zero (or denorm, but we simplify)
            sign
        } else {
            let e = (exponent - 112) as u32;
            let m = mantissa >> 13;
            sign | (e << 10) | (m & 0x3FF)
        };
        bytes.extend_from_slice(&(f16_bits as u16).to_le_bytes());
    }
    bytes
}

/// PMAT-260: Encode tensor data according to dtype, returning (dtype_str, raw_bytes).
fn encode_tensor_for_dtype(data: &[f32], original_dtype: Option<&str>) -> (&'static str, Vec<u8>) {
    match original_dtype {
        Some("BF16") => ("BF16", f32_slice_to_bf16_bytes(data)),
        Some("F16") => ("F16", f32_slice_to_f16_bytes(data)),
        _ => ("F32", data.iter().flat_map(|f| f.to_le_bytes()).collect()),
    }
}

/// PMAT-260: Save SafeTensors with original dtype preservation.
///
/// When `original_dtypes` contains entries, tensors with BF16/F16 origin
/// are written back in their original dtype instead of being widened to F32.
///
/// # Errors
///
/// Returns error if file writing or JSON serialization fails.
pub fn save_safetensors_typed<P: AsRef<Path>>(
    path: P,
    tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    original_dtypes: &BTreeMap<String, String>,
) -> Result<(), String> {
    let mut metadata = SafeTensorsMetadata::new();
    let mut raw_data = Vec::new();
    let mut current_offset = 0;

    for (name, (data, shape)) in tensors {
        let orig = original_dtypes.get(name).map(String::as_str);
        let (dtype_str, tensor_bytes) = encode_tensor_for_dtype(data, orig);

        let start_offset = current_offset;
        let end_offset = current_offset + tensor_bytes.len();

        metadata.insert(
            name.clone(),
            TensorMetadata {
                dtype: dtype_str.to_string(),
                shape: shape.clone(),
                data_offsets: [start_offset, end_offset],
            },
        );

        raw_data.extend_from_slice(&tensor_bytes);
        current_offset = end_offset;
    }

    let metadata_json =
        serde_json::to_string(&metadata).map_err(|e| format!("JSON serialization failed: {e}"))?;
    let metadata_bytes = metadata_json.as_bytes();
    let metadata_len = metadata_bytes.len() as u64;

    let mut output = Vec::new();
    output.extend_from_slice(&metadata_len.to_le_bytes());
    output.extend_from_slice(metadata_bytes);
    output.extend_from_slice(&raw_data);

    fs::write(path, output).map_err(|e| format!("File write failed: {e}"))?;
    Ok(())
}

/// PMAT-260: Save SafeTensors with user metadata AND original dtype preservation.
///
/// # Errors
///
/// Returns error if file writing or JSON serialization fails.
pub fn save_safetensors_with_metadata_typed<P: AsRef<Path>>(
    path: P,
    tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    user_metadata: &UserMetadata,
    original_dtypes: &BTreeMap<String, String>,
) -> Result<(), String> {
    let mut header = serde_json::Map::new();

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

    let mut raw_data = Vec::new();
    let mut current_offset = 0;

    for (name, (data, shape)) in tensors {
        let orig = original_dtypes.get(name).map(String::as_str);
        let (dtype_str, tensor_bytes) = encode_tensor_for_dtype(data, orig);

        let start_offset = current_offset;
        let end_offset = current_offset + tensor_bytes.len();

        #[allow(clippy::disallowed_methods)]
        let tensor_meta = serde_json::json!({
            "dtype": dtype_str,
            "shape": shape,
            "data_offsets": [start_offset, end_offset]
        });
        header.insert(name.clone(), tensor_meta);

        raw_data.extend_from_slice(&tensor_bytes);
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

    /// Get the offset where tensor data begins (after header + metadata JSON).
    #[must_use]
    pub fn data_offset(&self) -> usize {
        self.data_offset
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

    /// PMAT-260: Extract original dtype for each tensor.
    ///
    /// Returns a map of tensor name → dtype string (e.g., "F32", "F16", "BF16").
    /// Used by the export pipeline to preserve original dtypes during round-trip.
    #[must_use]
    pub fn dtype_map(&self) -> BTreeMap<String, String> {
        self.metadata
            .iter()
            .map(|(name, meta)| (name.clone(), meta.dtype.clone()))
            .collect()
    }
}

#[path = "safetensors_part_02.rs"]
mod safetensors_part_02;
pub use safetensors_part_02::extract_tensor;
use safetensors_part_02::{
    extract_bf16_to_f32, extract_f16_to_f32, extract_f32, parse_metadata, validate_and_read_header,
};

#[cfg(test)]
#[path = "safetensors_part_03.rs"]
mod safetensors_part_03;

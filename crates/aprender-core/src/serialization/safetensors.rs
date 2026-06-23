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

/// PMAT-260 / PMAT-859: Serialize F32 data as BF16 bytes.
///
/// BF16 is the upper 16 bits of F32. For data that originated as BF16,
/// the F32 representation has zeros in the lower 16 bits, so this is
/// lossless.
///
/// PMAT-859: Earlier this function truncated (`(bits >> 16) as u16`), which
/// (a) biased every value toward zero — not the IEEE / PyTorch / HF-safetensors
/// behavior — and (b) silently turned f32 NaNs whose mantissa bits live only
/// in the low 16 bits into +/-Inf. It now performs round-to-nearest-even and
/// preserves NaN, matching `half::bf16::from_f32`.
///
/// Round-to-nearest-even adds a bias of `0x7FFF + (lsb of the kept half)` before
/// the shift; this rounds halfway cases to even (the IEEE-754 default). NaN is
/// preserved by forcing a mantissa bit in the retained half so the result stays
/// NaN rather than collapsing to Inf.
fn f32_slice_to_bf16_bytes(data: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(data.len() * 2);
    for &value in data {
        let bits = value.to_bits();
        let bf16 = if value.is_nan() {
            // Force a mantissa bit in the retained half so NaN stays NaN, not Inf.
            ((bits >> 16) as u16) | 0x0040
        } else {
            // Round-to-nearest-even: bias by 0x7FFF plus the lsb of the kept half.
            let rounding_bias = 0x7FFF + ((bits >> 16) & 1);
            ((bits + rounding_bias) >> 16) as u16
        };
        bytes.extend_from_slice(&bf16.to_le_bytes());
    }
    bytes
}

/// PMAT-905: Encode a single f32 as IEEE 754 half-precision (F16) bits using
/// round-to-nearest-even, with full subnormal handling and NaN preservation.
///
/// This is the IEEE / PyTorch / HuggingFace-safetensors / `half::f16::from_f32`
/// reference behavior. The prior implementation (PMAT-260) truncated the
/// mantissa (`mantissa >> 13`) and flushed the entire subnormal range to signed
/// zero — both biased the cast and lost the smallest representable magnitudes.
///
/// F16 layout: 1 sign / 5 exponent (bias 15) / 10 mantissa.
///
/// Algorithm (matches `half::f16::from_f32` / IEEE-754 default rounding):
/// re-bias the f32 exponent to f16, then round the 23-bit mantissa down to 10
/// bits using round-to-nearest-even. An exact halfway case (discarded low bits
/// == `0x1000`) rounds toward the even mantissa; anything above halfway rounds
/// up. Values that land in the f16 subnormal range build the full significand
/// (with the implicit leading 1) and shift it into the subnormal field with the
/// same RNE rule, rather than flushing to zero. Overflow saturates to Inf and a
/// mantissa carry-out bumps the exponent (potentially to Inf), exactly as in
/// `half::f16`. NaN is preserved by forcing a mantissa bit.
fn f32_to_f16_bits_rne(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x007F_FFFF;

    if exponent == 0xFF {
        // Inf or NaN. Preserve NaN (force a mantissa bit so it never becomes
        // Inf); Inf keeps a zero mantissa.
        if mantissa != 0 {
            return sign | 0x7C00 | ((mantissa >> 13) as u16) | 0x0200;
        }
        return sign | 0x7C00;
    }

    // Re-bias exponent from f32 (bias 127) to f16 (bias 15).
    let f16_exp = exponent - 127 + 15;

    if f16_exp >= 0x1F {
        // Overflow → Inf.
        return sign | 0x7C00;
    }

    if f16_exp <= 0 {
        // Subnormal f16 (or underflow to signed zero). Build the significand
        // with its implicit leading 1, then shift it right into the 10-bit
        // subnormal field, rounding to nearest even on the discarded bits.
        if f16_exp < -10 {
            // Below half the smallest subnormal → rounds to signed zero.
            return sign;
        }
        let significand = mantissa | 0x0080_0000;
        let shift = (14 - f16_exp) as u32; // 14..=24
        let half = 1u32 << (shift - 1);
        let mask = (1u32 << shift) - 1;
        let low = significand & mask;
        let mut frac = significand >> shift;
        if low > half || (low == half && (frac & 1) == 1) {
            frac += 1;
        }
        return sign | (frac as u16);
    }

    // Normal f16. Round the 23-bit mantissa to 10 bits, round-to-nearest-even.
    let low = mantissa & 0x1FFF; // 13 discarded bits
    let half = 0x1000u32; // 2^12 = exact halfway
    let mut frac = mantissa >> 13;
    if low > half || (low == half && (frac & 1) == 1) {
        frac += 1;
    }
    // A mantissa carry-out (frac == 0x400) propagates into the exponent via the
    // `+` below; if that pushes the exponent to 0x1F the result is Inf, exactly
    // as half::f16 produces near the overflow boundary.
    let assembled = ((f16_exp as u32) << 10) + frac;
    sign | (assembled as u16)
}

/// PMAT-260 / PMAT-905: Serialize F32 data as F16 bytes.
///
/// Uses IEEE 754 half-precision round-to-nearest-even, matching
/// `half::f16::from_f32`. For data that originated as F16, the F32
/// representation is exact, so this round-trip is lossless.
fn f32_slice_to_f16_bytes(data: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(data.len() * 2);
    for &value in data {
        bytes.extend_from_slice(&f32_to_f16_bits_rne(value).to_le_bytes());
    }
    bytes
}

/// PMAT-260: Encode tensor data according to dtype, returning (dtype_str, raw_bytes).
///
/// GH-439 (poka-yoke): Explicit handling for known dtypes. Unknown dtype strings
/// emit a warning instead of silently falling back to F32 (the GH-186 pattern).
fn encode_tensor_for_dtype(data: &[f32], original_dtype: Option<&str>) -> (&'static str, Vec<u8>) {
    match original_dtype {
        Some("BF16") => ("BF16", f32_slice_to_bf16_bytes(data)),
        Some("F16") => ("F16", f32_slice_to_f16_bytes(data)),
        Some("F32") | None => ("F32", data.iter().flat_map(|f| f.to_le_bytes()).collect()),
        Some(unknown) => {
            eprintln!(
                "[GH-439] encode_tensor_for_dtype: unknown dtype '{}' — \
                 falling back to F32. This may produce incorrect output.",
                unknown
            );
            ("F32", data.iter().flat_map(|f| f.to_le_bytes()).collect())
        }
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

include!("safetensors_include_01.rs");

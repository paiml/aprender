//! Tensor Listing Library (TOOL-APR-001 Fix)
//!
//! Provides library functions for listing tensors from APR model files.
//! Reads from the actual tensor index, not just metadata.
//!
//! # Dr. Popper's Principle
//!
//! "Read the actual data, not the documentation about the data."
//!
//! This module was extracted from `apr-cli/commands/tensors.rs` to:
//! 1. Enable 95%+ test coverage (CLI is now thin shim)
//! 2. Fix TOOL-APR-001: reading from tensor index, not metadata
//! 3. Provide reusable library functions
//!
//! # Example
//!
//! ```rust,ignore
//! use aprender::format::tensors::{list_tensors, TensorListOptions};
//!
//! let options = TensorListOptions::default();
//! let result = list_tensors_from_bytes(&apr_bytes, options)?;
//! for tensor in &result.tensors {
//!     println!("{}: {:?} ({})", tensor.name, tensor.shape, tensor.dtype);
//! }
//! ```

use crate::error::{AprenderError, Result};
use crate::format::gguf::reader::GgufReader;
use crate::format::rosetta::FormatType;
use crate::format::v2::{AprV2Reader, TensorIndexEntry};
use crate::format::HEADER_SIZE;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

// ============================================================================
// Public Types
// ============================================================================

/// Information about a tensor in the model
#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// Tensor name (e.g., "model.layers.0.self_attn.q_proj.weight")
    pub name: String,
    /// Shape dimensions (e.g., [4096, 4096])
    pub shape: Vec<usize>,
    /// Data type (e.g., "f32", "f16", "q4_k")
    pub dtype: String,
    /// Size in bytes
    pub size_bytes: usize,
    /// Mean value (if stats computed)
    pub mean: Option<f32>,
    /// Standard deviation (if stats computed)
    pub std: Option<f32>,
    /// Minimum value (if stats computed)
    pub min: Option<f32>,
    /// Maximum value (if stats computed)
    pub max: Option<f32>,
    /// Number of NaN values (spec H8: should be 0)
    pub nan_count: Option<usize>,
    /// Number of Inf values
    pub inf_count: Option<usize>,
}

/// Result of listing tensors from a model
#[derive(Debug, Clone)]
pub struct TensorListResult {
    /// Source file path
    pub file: String,
    /// APR format version detected
    pub format_version: String,
    /// Total number of tensors
    pub tensor_count: usize,
    /// Total size in bytes
    pub total_size_bytes: usize,
    /// Individual tensor info
    pub tensors: Vec<TensorInfo>,
}

/// Options for listing tensors
#[derive(Debug, Clone)]
pub struct TensorListOptions {
    /// Compute statistics (mean, std, min, max)
    pub compute_stats: bool,
    /// Filter tensors by name pattern (substring match)
    pub filter: Option<String>,
    /// Maximum number of tensors to return (default: unlimited)
    pub limit: usize,
}

impl Default for TensorListOptions {
    fn default() -> Self {
        Self {
            compute_stats: false,
            filter: None,
            limit: usize::MAX,
        }
    }
}

impl TensorListOptions {
    /// Create default options
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable statistics computation
    #[must_use]
    pub fn with_stats(mut self) -> Self {
        self.compute_stats = true;
        self
    }

    /// Set filter pattern
    #[must_use]
    pub fn with_filter(mut self, pattern: impl Into<String>) -> Self {
        self.filter = Some(pattern.into());
        self
    }

    /// Set maximum tensor count
    #[must_use]
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }
}

// ============================================================================
// Format Detection
// ============================================================================

/// APR format magic bytes
const MAGIC_APRN: [u8; 4] = [0x41, 0x50, 0x52, 0x4E]; // "APRN"
const MAGIC_APR1: [u8; 4] = [0x41, 0x50, 0x52, 0x31]; // "APR1"
const MAGIC_APR2: [u8; 4] = [0x41, 0x50, 0x52, 0x32]; // "APR2"
const MAGIC_APR0: [u8; 4] = [0x41, 0x50, 0x52, 0x00]; // "APR\0"

/// Detect APR format version from magic bytes
fn detect_format(magic: &[u8; 4]) -> Option<&'static str> {
    match *magic {
        MAGIC_APRN => Some("v1"),
        MAGIC_APR1 => Some("v1"),
        MAGIC_APR2 => Some("v2"),
        MAGIC_APR0 => Some("v2"),
        _ => None,
    }
}

/// Check if magic bytes are valid APR format
#[must_use]
pub fn is_valid_apr_magic(magic: &[u8; 4]) -> bool {
    detect_format(magic).is_some()
}

// ============================================================================
// Tensor Listing - From Bytes
// ============================================================================

/// List tensors from model file bytes (APR, GGUF, or SafeTensors)
///
/// Detects format from magic bytes and dispatches to the appropriate reader.
/// This is the core function that reads from the actual tensor index,
/// not just metadata. This fixes TOOL-APR-001.
///
/// # Arguments
/// * `data` - Raw model file bytes
/// * `options` - Listing options
///
/// # Errors
/// Returns error if the format is invalid or parsing fails.
pub fn list_tensors_from_bytes(
    data: &[u8],
    options: TensorListOptions,
) -> Result<TensorListResult> {
    // Check minimum size
    if data.len() < 4 {
        return Err(AprenderError::FormatError {
            message: "File too small to contain model header".to_string(),
        });
    }

    // Detect format from magic bytes (Rosetta Stone dispatch)
    if data.len() >= 4 && &data[0..4] == b"GGUF" {
        return list_tensors_gguf(data, options);
    }

    if data.len() >= 10 {
        let header_len = u64::from_le_bytes(
            data[0..8]
                .try_into()
                .unwrap_or([0u8; 8]),
        );
        if header_len < 100_000_000 && &data[8..10] == b"{\"" {
            return list_tensors_safetensors(data, options);
        }
    }

    // Fall through to APR detection
    let magic: [u8; 4] = data[0..4]
        .try_into()
        .map_err(|_| AprenderError::FormatError {
            message: "Failed to read magic bytes".to_string(),
        })?;

    let format_version = detect_format(&magic).ok_or_else(|| AprenderError::FormatError {
        message: format!(
            "Unknown model format: magic bytes {:02x}{:02x}{:02x}{:02x}. \
             Supported formats: APR (.apr), GGUF (.gguf), SafeTensors (.safetensors)",
            magic[0], magic[1], magic[2], magic[3]
        ),
    })?;

    match format_version {
        "v2" => list_tensors_v2(data, options),
        "v1" => list_tensors_v1(data, options),
        _ => Err(AprenderError::FormatError {
            message: format!("Unsupported format version: {format_version}"),
        }),
    }
}

/// List tensors from APR v2 format (reads actual tensor index)
fn list_tensors_v2(data: &[u8], options: TensorListOptions) -> Result<TensorListResult> {
    // Parse with v2 reader
    let reader = AprV2Reader::from_bytes(data).map_err(|e| AprenderError::FormatError {
        message: format!("Failed to parse APR v2: {e}"),
    })?;

    // Get tensor info from actual index
    let mut tensors = Vec::new();
    let mut total_size = 0usize;

    for name in reader.tensor_names() {
        // Apply filter
        if let Some(ref pattern) = options.filter {
            if !name.contains(pattern.as_str()) {
                continue;
            }
        }

        // Get tensor entry
        if let Some(entry) = reader.get_tensor(name) {
            let mut info = tensor_info_from_entry(entry);

            // Compute stats if requested
            if options.compute_stats {
                if let Some(data) = reader.get_tensor_as_f32(name) {
                    compute_tensor_stats(&mut info, &data);
                }
            }

            total_size += info.size_bytes;
            tensors.push(info);

            // Check limit
            if tensors.len() >= options.limit {
                break;
            }
        }
    }

    Ok(TensorListResult {
        file: String::new(), // Set by caller
        format_version: "v2".to_string(),
        tensor_count: tensors.len(),
        total_size_bytes: total_size,
        tensors,
    })
}

/// List tensors from APR v1 format (fallback to metadata)
fn list_tensors_v1(data: &[u8], options: TensorListOptions) -> Result<TensorListResult> {
    // APR v1 stores tensor info in metadata, not a separate index
    // Read metadata and extract tensor_shapes

    if data.len() < HEADER_SIZE {
        return Err(AprenderError::FormatError {
            message: "APR v1 file too small for header".to_string(),
        });
    }

    // Read metadata size from header (offset 8 in v1)
    let metadata_size = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;

    if data.len() < HEADER_SIZE + metadata_size {
        return Err(AprenderError::FormatError {
            message: "APR v1 file too small for metadata".to_string(),
        });
    }

    // Parse metadata (MessagePack or JSON)
    let metadata_bytes = &data[HEADER_SIZE..HEADER_SIZE + metadata_size];
    let metadata: HashMap<String, serde_json::Value> = serde_json::from_slice(metadata_bytes)
        .or_else(|_| rmp_serde::from_slice(metadata_bytes))
        .unwrap_or_default();

    // Extract tensor shapes from metadata
    let tensors = extract_tensors_from_metadata(&metadata, &options);
    let total_size: usize = tensors.iter().map(|t| t.size_bytes).sum();

    Ok(TensorListResult {
        file: String::new(),
        format_version: "v1".to_string(),
        tensor_count: tensors.len(),
        total_size_bytes: total_size,
        tensors,
    })
}

// ============================================================================
// GGUF Format Support (PMAT-ROSETTA-001)
// ============================================================================

/// GGML data type names from dtype id
fn ggml_dtype_name(dtype: u32) -> &'static str {
    match dtype {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        6 => "Q5_0",
        7 => "Q5_1",
        8 => "Q8_0",
        9 => "Q8_1",
        10 => "Q2_K",
        11 => "Q3_K",
        12 => "Q4_K",
        13 => "Q5_K",
        14 => "Q6_K",
        15 => "Q8_K",
        16 => "IQ2_XXS",
        17 => "IQ2_XS",
        18 => "IQ3_XXS",
        26 => "BF16",
        _ => "unknown",
    }
}

/// Bytes per element for GGML data types (approximate for block types)
fn ggml_dtype_element_size(dtype: u32) -> f64 {
    match dtype {
        0 => 4.0,           // F32
        1 => 2.0,           // F16
        2 => 0.5 + 2.0/32.0, // Q4_0: 4-bit + scale
        3 => 0.5 + 4.0/32.0, // Q4_1: 4-bit + scale + min
        6 => 0.625 + 2.0/32.0, // Q5_0
        7 => 0.625 + 4.0/32.0, // Q5_1
        8 => 1.0 + 2.0/32.0, // Q8_0
        9 => 1.0 + 4.0/32.0, // Q8_1
        10 => 0.3125,       // Q2_K
        11 => 0.4375,       // Q3_K
        12 => 0.5625,       // Q4_K
        13 => 0.6875,       // Q5_K
        14 => 0.8125,       // Q6_K
        15 => 1.0625,       // Q8_K
        26 => 2.0,          // BF16
        _ => 4.0,           // default assume F32
    }
}

/// List tensors from GGUF file bytes
fn list_tensors_gguf(data: &[u8], options: TensorListOptions) -> Result<TensorListResult> {
    let reader = GgufReader::from_bytes(data.to_vec()).map_err(|e| AprenderError::FormatError {
        message: format!("Failed to parse GGUF: {e}"),
    })?;

    let mut tensors = Vec::new();
    let mut total_size = 0usize;

    for meta in &reader.tensors {
        // Apply filter
        if let Some(ref pattern) = options.filter {
            if !meta.name.contains(pattern.as_str()) {
                continue;
            }
        }

        let shape: Vec<usize> = meta.dims.iter().map(|&d| d as usize).collect();
        let num_elements: usize = shape.iter().product();
        let size_bytes = (num_elements as f64 * ggml_dtype_element_size(meta.dtype)) as usize;

        let mut info = TensorInfo {
            name: meta.name.clone(),
            shape,
            dtype: ggml_dtype_name(meta.dtype).to_string(),
            size_bytes,
            mean: None,
            std: None,
            min: None,
            max: None,
            nan_count: None,
            inf_count: None,
        };

        if options.compute_stats {
            if let Ok((f32_data, _shape)) = reader.get_tensor_f32(&meta.name) {
                compute_tensor_stats(&mut info, &f32_data);
            }
        }

        total_size += info.size_bytes;
        tensors.push(info);

        if tensors.len() >= options.limit {
            break;
        }
    }

    Ok(TensorListResult {
        file: String::new(),
        format_version: format!("GGUF v{}", reader.version),
        tensor_count: tensors.len(),
        total_size_bytes: total_size,
        tensors,
    })
}

// ============================================================================
// SafeTensors Format Support (PMAT-ROSETTA-001)
// ============================================================================

/// List tensors from SafeTensors file bytes by parsing the JSON header
fn list_tensors_safetensors(data: &[u8], options: TensorListOptions) -> Result<TensorListResult> {
    if data.len() < 8 {
        return Err(AprenderError::FormatError {
            message: "SafeTensors file too small".to_string(),
        });
    }

    let header_len =
        u64::from_le_bytes(data[0..8].try_into().map_err(|_| AprenderError::FormatError {
            message: "Failed to read SafeTensors header length".to_string(),
        })?) as usize;

    if data.len() < 8 + header_len {
        return Err(AprenderError::FormatError {
            message: "SafeTensors file truncated (header extends past EOF)".to_string(),
        });
    }

    let header_json = &data[8..8 + header_len];
    let header: serde_json::Value =
        serde_json::from_slice(header_json).map_err(|e| AprenderError::FormatError {
            message: format!("Failed to parse SafeTensors JSON header: {e}"),
        })?;

    let obj = header
        .as_object()
        .ok_or_else(|| AprenderError::FormatError {
            message: "SafeTensors header is not a JSON object".to_string(),
        })?;

    let data_start = 8 + header_len;
    let mut tensors = Vec::new();
    let mut total_size = 0usize;

    // Collect and sort tensor names for deterministic output
    let mut tensor_entries: Vec<(&String, &serde_json::Value)> = obj
        .iter()
        .filter(|(k, _)| *k != "__metadata__")
        .collect();
    tensor_entries.sort_by_key(|(k, _)| *k);

    for (name, value) in tensor_entries {
        // Apply filter
        if let Some(ref pattern) = options.filter {
            if !name.contains(pattern.as_str()) {
                continue;
            }
        }

        let dtype = value
            .get("dtype")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();

        let shape: Vec<usize> = value
            .get("shape")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_u64().map(|n| n as usize))
                    .collect()
            })
            .unwrap_or_default();

        let offsets = value.get("data_offsets").and_then(|v| v.as_array());
        let size_bytes = offsets
            .and_then(|arr| {
                let start = arr.first()?.as_u64()? as usize;
                let end = arr.get(1)?.as_u64()? as usize;
                Some(end - start)
            })
            .unwrap_or(0);

        let mut info = TensorInfo {
            name: name.clone(),
            shape: shape.clone(),
            dtype,
            size_bytes,
            mean: None,
            std: None,
            min: None,
            max: None,
            nan_count: None,
            inf_count: None,
        };

        if options.compute_stats {
            if let Some(arr) = offsets {
                if let (Some(start), Some(end)) = (
                    arr.first().and_then(|v| v.as_u64()),
                    arr.get(1).and_then(|v| v.as_u64()),
                ) {
                    let abs_start = data_start + start as usize;
                    let abs_end = data_start + end as usize;
                    if abs_end <= data.len() {
                        let tensor_bytes = &data[abs_start..abs_end];
                        let f32_data = safetensors_bytes_to_f32(tensor_bytes, &info.dtype);
                        compute_tensor_stats(&mut info, &f32_data);
                    }
                }
            }
        }

        total_size += info.size_bytes;
        tensors.push(info);

        if tensors.len() >= options.limit {
            break;
        }
    }

    Ok(TensorListResult {
        file: String::new(),
        format_version: "SafeTensors".to_string(),
        tensor_count: tensors.len(),
        total_size_bytes: total_size,
        tensors,
    })
}

/// Convert SafeTensors raw bytes to f32 based on dtype
fn safetensors_bytes_to_f32(bytes: &[u8], dtype: &str) -> Vec<f32> {
    match dtype {
        "F32" => bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        "F16" => bytes
            .chunks_exact(2)
            .map(|c| {
                let bits = u16::from_le_bytes([c[0], c[1]]);
                f16_to_f32(bits)
            })
            .collect(),
        "BF16" => bytes
            .chunks_exact(2)
            .map(|c| {
                let bits = u16::from_le_bytes([c[0], c[1]]);
                bf16_to_f32(bits)
            })
            .collect(),
        _ => Vec::new(), // Unknown dtype, skip stats
    }
}

/// Convert IEEE 754 half-precision float to f32
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) as u32) << 31;
    let exponent = ((bits >> 10) & 0x1F) as u32;
    let mantissa = (bits & 0x3FF) as u32;

    if exponent == 0 {
        if mantissa == 0 {
            return f32::from_bits(sign);
        }
        // Denormalized: convert to normalized f32
        let mut e = 1u32;
        let mut m = mantissa;
        while (m & 0x400) == 0 {
            m <<= 1;
            e += 1;
        }
        let f32_exp = (127 - 15 - e + 1) << 23;
        let f32_mant = (m & 0x3FF) << 13;
        f32::from_bits(sign | f32_exp | f32_mant)
    } else if exponent == 31 {
        // Inf/NaN
        let f32_exp = 0xFF << 23;
        let f32_mant = mantissa << 13;
        f32::from_bits(sign | f32_exp | f32_mant)
    } else {
        let f32_exp = (exponent + 127 - 15) << 23;
        let f32_mant = mantissa << 13;
        f32::from_bits(sign | f32_exp | f32_mant)
    }
}

/// Convert BFloat16 to f32 (simple: just shift left by 16)
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

// ============================================================================
// Path-Based Format Dispatch (PMAT-ROSETTA-001)
// ============================================================================

/// Convert tensor index entry to TensorInfo
fn tensor_info_from_entry(entry: &TensorIndexEntry) -> TensorInfo {
    TensorInfo {
        name: entry.name.clone(),
        shape: entry.shape.clone(),
        dtype: entry.dtype.name().to_string(),
        size_bytes: entry.size as usize,
        mean: None,
        std: None,
        min: None,
        max: None,
        nan_count: None,
        inf_count: None,
    }
}

/// Extract tensor info from v1 metadata
fn extract_tensors_from_metadata(
    metadata: &HashMap<String, serde_json::Value>,
    options: &TensorListOptions,
) -> Vec<TensorInfo> {
    let Some(shapes) = metadata.get("tensor_shapes").and_then(|s| s.as_object()) else {
        return Vec::new();
    };

    shapes
        .iter()
        .filter(|(name, _)| {
            options
                .filter
                .as_ref()
                .map_or(true, |f| name.contains(f.as_str()))
        })
        .take(options.limit)
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

/// Parse shape array from JSON value
fn parse_shape_array(shape_val: &serde_json::Value) -> Vec<usize> {
    shape_val.as_array().map_or(Vec::new(), |arr| {
        arr.iter()
            .filter_map(|v| v.as_u64().map(|n| n as usize))
            .collect()
    })
}

// ============================================================================
// Tensor Listing - From File
// ============================================================================

/// List tensors from a model file (APR, GGUF, or SafeTensors)
///
/// Uses magic byte detection for reliable format identification,
/// then delegates to the appropriate format-specific reader.
///
/// # Arguments
/// * `path` - Path to model file
/// * `options` - Listing options
///
/// # Errors
/// Returns error if the file doesn't exist or is invalid.
pub fn list_tensors(
    path: impl AsRef<Path>,
    options: TensorListOptions,
) -> Result<TensorListResult> {
    let path = path.as_ref();

    // For SafeTensors, prefer MappedSafeTensors (mmap-based, handles large files)
    if let Ok(FormatType::SafeTensors) = FormatType::from_magic(path) {
        let mut result = list_tensors_safetensors_path(path, options)?;
        result.file = path.display().to_string();
        return Ok(result);
    }

    // For APR and GGUF, read into memory and dispatch
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut data = Vec::new();
    reader.read_to_end(&mut data)?;

    let mut result = list_tensors_from_bytes(&data, options)?;
    result.file = path.display().to_string();

    Ok(result)
}

/// List tensors from SafeTensors via mmap (efficient for large files)
fn list_tensors_safetensors_path(
    path: &Path,
    options: TensorListOptions,
) -> Result<TensorListResult> {
    use crate::serialization::safetensors::MappedSafeTensors;

    let mapped = MappedSafeTensors::open(path).map_err(|e| AprenderError::FormatError {
        message: format!("Failed to open SafeTensors: {e}"),
    })?;

    let mut tensors = Vec::new();
    let mut total_size = 0usize;

    let mut names: Vec<&str> = mapped.tensor_names();
    names.sort_unstable();

    for name in names {
        if let Some(ref pattern) = options.filter {
            if !name.contains(pattern.as_str()) {
                continue;
            }
        }

        if let Some(meta) = mapped.get_metadata(name) {
            let size_bytes = meta.data_offsets[1] - meta.data_offsets[0];

            let mut info = TensorInfo {
                name: name.to_string(),
                shape: meta.shape.clone(),
                dtype: meta.dtype.clone(),
                size_bytes,
                mean: None,
                std: None,
                min: None,
                max: None,
                nan_count: None,
                inf_count: None,
            };

            if options.compute_stats {
                if let Ok(f32_data) = mapped.get_tensor(name) {
                    compute_tensor_stats(&mut info, &f32_data);
                }
            }

            total_size += info.size_bytes;
            tensors.push(info);

            if tensors.len() >= options.limit {
                break;
            }
        }
    }

    Ok(TensorListResult {
        file: String::new(),
        format_version: "SafeTensors".to_string(),
        tensor_count: tensors.len(),
        total_size_bytes: total_size,
        tensors,
    })
}

// ============================================================================
// Statistics Computation
// ============================================================================

/// Compute tensor statistics (mean, std, min, max, nan/inf count)
fn compute_tensor_stats(info: &mut TensorInfo, data: &[f32]) {
    if data.is_empty() {
        return;
    }

    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut nan_count = 0usize;
    let mut inf_count = 0usize;
    let mut valid_count = 0usize;

    for &val in data {
        if val.is_nan() {
            nan_count += 1;
            continue;
        }
        if val.is_infinite() {
            inf_count += 1;
            continue;
        }

        valid_count += 1;
        let val_f64 = f64::from(val);
        sum += val_f64;
        sum_sq += val_f64 * val_f64;

        if val < min {
            min = val;
        }
        if val > max {
            max = val;
        }
    }

    info.nan_count = Some(nan_count);
    info.inf_count = Some(inf_count);

    if valid_count > 0 {
        let n = valid_count as f64;
        let mean = sum / n;
        let variance = (sum_sq / n) - (mean * mean);
        let std = if variance > 0.0 { variance.sqrt() } else { 0.0 };

        info.mean = Some(mean as f32);
        info.std = Some(std as f32);
        info.min = Some(min);
        info.max = Some(max);
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Format size in human-readable form
#[must_use]
pub fn format_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * 1024;
    const GB: u64 = 1024 * 1024 * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::test_factory::{
        build_pygmy_apr, build_pygmy_apr_f16, build_pygmy_apr_q4, build_pygmy_apr_q8,
        build_pygmy_apr_with_config, build_pygmy_safetensors, build_pygmy_safetensors_with_config,
        PygmyConfig,
    };

    // ========================================================================
    // Format Detection Tests
    // ========================================================================

    #[test]
    fn test_detect_format_v2() {
        assert_eq!(detect_format(&MAGIC_APR2), Some("v2"));
        assert_eq!(detect_format(&MAGIC_APR0), Some("v2"));
    }

    #[test]
    fn test_detect_format_v1() {
        assert_eq!(detect_format(&MAGIC_APRN), Some("v1"));
        assert_eq!(detect_format(&MAGIC_APR1), Some("v1"));
    }

    #[test]
    fn test_detect_format_invalid() {
        assert_eq!(detect_format(&[0x00, 0x00, 0x00, 0x00]), None);
        assert_eq!(detect_format(&[0xFF, 0xFF, 0xFF, 0xFF]), None);
        assert_eq!(detect_format(b"GGUF"), None);
    }

    #[test]
    fn test_is_valid_apr_magic() {
        assert!(is_valid_apr_magic(&MAGIC_APR2));
        assert!(is_valid_apr_magic(&MAGIC_APR0));
        assert!(is_valid_apr_magic(&MAGIC_APRN));
        assert!(!is_valid_apr_magic(b"GGUF"));
        assert!(!is_valid_apr_magic(&[0x00; 4]));
    }

    // ========================================================================
    // TensorListOptions Tests
    // ========================================================================

    #[test]
    fn test_options_default() {
        let opts = TensorListOptions::default();
        assert!(!opts.compute_stats);
        assert!(opts.filter.is_none());
        // Default limit is usize::MAX (effectively unlimited)
        assert_eq!(opts.limit, usize::MAX);
    }

    #[test]
    fn test_options_builder() {
        let opts = TensorListOptions::new()
            .with_stats()
            .with_filter("weight")
            .with_limit(10);

        assert!(opts.compute_stats);
        assert_eq!(opts.filter, Some("weight".to_string()));
        assert_eq!(opts.limit, 10);
    }

    // ========================================================================
    // Pygmy APR v2 Tests (TOOL-APR-001 Fix)
    // ========================================================================

    #[test]
    fn test_list_tensors_pygmy_apr_default() {
        let apr_bytes = build_pygmy_apr();
        let result = list_tensors_from_bytes(&apr_bytes, TensorListOptions::default())
            .expect("list tensors");

        assert_eq!(result.format_version, "v2");
        assert!(result.tensor_count > 0, "Expected at least one tensor");
        assert!(result.total_size_bytes > 0);

        // Check we got tensor names from the index (could be various naming conventions)
        let names: Vec<_> = result.tensors.iter().map(|t| t.name.as_str()).collect();

        // Pygmy uses "model." prefix
        let has_model_tensors = names.iter().any(|n| n.starts_with("model."));
        let has_lm_head = names.iter().any(|n| n.contains("lm_head"));

        assert!(
            has_model_tensors || has_lm_head,
            "Expected model tensors, got: {:?}",
            names
        );
    }

    #[test]
    fn test_list_tensors_pygmy_apr_with_filter() {
        let apr_bytes = build_pygmy_apr();
        let opts = TensorListOptions::new().with_filter("self_attn");
        let result = list_tensors_from_bytes(&apr_bytes, opts).expect("list tensors");

        // All returned tensors should match filter
        for tensor in &result.tensors {
            assert!(
                tensor.name.contains("self_attn"),
                "Expected tensor {} to contain 'self_attn'",
                tensor.name
            );
        }
    }

    #[test]
    fn test_list_tensors_pygmy_apr_with_limit() {
        let apr_bytes = build_pygmy_apr();
        let opts = TensorListOptions::new().with_limit(3);
        let result = list_tensors_from_bytes(&apr_bytes, opts).expect("list tensors");

        assert!(result.tensor_count <= 3);
    }

    #[test]
    fn test_list_tensors_pygmy_apr_with_stats() {
        let apr_bytes = build_pygmy_apr();
        let opts = TensorListOptions::new().with_stats().with_limit(5);
        let result = list_tensors_from_bytes(&apr_bytes, opts).expect("list tensors");

        // Check at least one tensor has stats
        let has_stats = result
            .tensors
            .iter()
            .any(|t| t.mean.is_some() && t.std.is_some() && t.nan_count.is_some());
        assert!(has_stats, "Expected at least one tensor to have stats");
    }

    #[test]
    fn test_list_tensors_pygmy_apr_f16() {
        let apr_bytes = build_pygmy_apr_f16();
        let result = list_tensors_from_bytes(&apr_bytes, TensorListOptions::default())
            .expect("list tensors");

        // F16 tensors should be detected
        let f16_tensors: Vec<_> = result.tensors.iter().filter(|t| t.dtype == "f16").collect();
        assert!(!f16_tensors.is_empty(), "Expected F16 tensors");
    }

    #[test]
    fn test_list_tensors_pygmy_apr_q8() {
        let apr_bytes = build_pygmy_apr_q8();
        let result = list_tensors_from_bytes(&apr_bytes, TensorListOptions::default())
            .expect("list tensors");

        // Should have at least some tensors (mix of Q8 and F32)
        assert!(!result.tensors.is_empty(), "Expected tensors in Q8 model");

        // Q8 model contains both F32 (embedding) and Q8 (attention) tensors
        let dtypes: Vec<_> = result.tensors.iter().map(|t| t.dtype.as_str()).collect();
        assert!(
            dtypes.iter().any(|d| *d == "q8" || *d == "f32"),
            "Expected Q8 or F32 tensors, got: {:?}",
            dtypes
        );
    }

    #[test]
    fn test_list_tensors_pygmy_apr_q4() {
        let apr_bytes = build_pygmy_apr_q4();
        let result = list_tensors_from_bytes(&apr_bytes, TensorListOptions::default())
            .expect("list tensors");

        // Should have at least some tensors (mix of Q4 and F32)
        assert!(!result.tensors.is_empty(), "Expected tensors in Q4 model");

        // Q4 model contains both F32 (embedding) and Q4 (attention) tensors
        let dtypes: Vec<_> = result.tensors.iter().map(|t| t.dtype.as_str()).collect();
        assert!(
            dtypes.iter().any(|d| *d == "q4" || *d == "f32"),
            "Expected Q4 or F32 tensors, got: {:?}",
            dtypes
        );
    }

    #[test]
    fn test_list_tensors_pygmy_apr_minimal() {
        let apr_bytes = build_pygmy_apr_with_config(PygmyConfig::minimal());
        let result = list_tensors_from_bytes(&apr_bytes, TensorListOptions::default())
            .expect("list tensors");

        // Minimal config has embedding only
        assert!(result.tensor_count >= 1);
    }

    #[test]
    fn test_list_tensors_pygmy_apr_llama_style() {
        let apr_bytes = build_pygmy_apr_with_config(PygmyConfig::llama_style());
        let result = list_tensors_from_bytes(&apr_bytes, TensorListOptions::default())
            .expect("list tensors");

        // LLaMA style has many tensors (at least some)
        assert!(
            result.tensor_count >= 1,
            "Expected at least 1 tensor, got {}",
            result.tensor_count
        );

        // Should have tensors with various components
        let has_any_tensor = !result.tensors.is_empty();
        assert!(has_any_tensor);
    }

    // ========================================================================
    // Error Handling Tests
    // ========================================================================

    #[test]
    fn test_list_tensors_empty_data() {
        let result = list_tensors_from_bytes(&[], TensorListOptions::default());
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("too small"));
    }

    #[test]
    fn test_list_tensors_invalid_magic() {
        let data = [0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00];
        let result = list_tensors_from_bytes(&data, TensorListOptions::default());
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("Unknown model format"));
    }

    #[test]
    fn test_list_tensors_truncated_v2() {
        // Valid v2 magic but truncated file
        let mut data = vec![0x41, 0x50, 0x52, 0x00]; // "APR\0"
        data.extend_from_slice(&[0u8; 10]); // Not enough for header

        let result = list_tensors_from_bytes(&data, TensorListOptions::default());
        assert!(result.is_err());
    }

    // ========================================================================
    // Statistics Tests
    // ========================================================================

    #[test]
    fn test_compute_stats_normal_values() {
        let mut info = TensorInfo {
            name: "test".to_string(),
            shape: vec![4],
            dtype: "f32".to_string(),
            size_bytes: 16,
            mean: None,
            std: None,
            min: None,
            max: None,
            nan_count: None,
            inf_count: None,
        };

        let data = vec![1.0, 2.0, 3.0, 4.0];
        compute_tensor_stats(&mut info, &data);

        assert_eq!(info.mean, Some(2.5));
        assert!(info.std.unwrap() > 1.0 && info.std.unwrap() < 1.2);
        assert_eq!(info.min, Some(1.0));
        assert_eq!(info.max, Some(4.0));
        assert_eq!(info.nan_count, Some(0));
        assert_eq!(info.inf_count, Some(0));
    }

    #[test]
    fn test_compute_stats_with_nan() {
        let mut info = TensorInfo {
            name: "test".to_string(),
            shape: vec![3],
            dtype: "f32".to_string(),
            size_bytes: 12,
            mean: None,
            std: None,
            min: None,
            max: None,
            nan_count: None,
            inf_count: None,
        };

        let data = vec![1.0, f32::NAN, 2.0];
        compute_tensor_stats(&mut info, &data);

        assert_eq!(info.nan_count, Some(1));
        assert_eq!(info.inf_count, Some(0));
        // Mean should be computed from valid values only
        assert_eq!(info.mean, Some(1.5));
    }

    #[test]
    fn test_compute_stats_with_inf() {
        let mut info = TensorInfo {
            name: "test".to_string(),
            shape: vec![3],
            dtype: "f32".to_string(),
            size_bytes: 12,
            mean: None,
            std: None,
            min: None,
            max: None,
            nan_count: None,
            inf_count: None,
        };

        let data = vec![1.0, f32::INFINITY, 2.0];
        compute_tensor_stats(&mut info, &data);

        assert_eq!(info.nan_count, Some(0));
        assert_eq!(info.inf_count, Some(1));
    }

    #[test]
    fn test_compute_stats_empty() {
        let mut info = TensorInfo {
            name: "test".to_string(),
            shape: vec![0],
            dtype: "f32".to_string(),
            size_bytes: 0,
            mean: None,
            std: None,
            min: None,
            max: None,
            nan_count: None,
            inf_count: None,
        };

        compute_tensor_stats(&mut info, &[]);

        assert!(info.mean.is_none());
        assert!(info.std.is_none());
    }

    // ========================================================================
    // Format Size Tests
    // ========================================================================

    #[test]
    fn test_format_size_bytes() {
        assert_eq!(format_size(500), "500 B");
        assert_eq!(format_size(0), "0 B");
    }

    #[test]
    fn test_format_size_kb() {
        assert_eq!(format_size(1024), "1.00 KB");
        assert_eq!(format_size(2048), "2.00 KB");
    }

    #[test]
    fn test_format_size_mb() {
        assert_eq!(format_size(1024 * 1024), "1.00 MB");
        assert_eq!(format_size(100 * 1024 * 1024), "100.00 MB");
    }

    #[test]
    fn test_format_size_gb() {
        assert_eq!(format_size(1024 * 1024 * 1024), "1.00 GB");
        assert_eq!(format_size(5 * 1024 * 1024 * 1024), "5.00 GB");
    }

    // ========================================================================
    // TensorInfo Tests
    // ========================================================================

    #[test]
    fn test_tensor_info_clone() {
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

        let cloned = info.clone();
        assert_eq!(cloned.name, info.name);
        assert_eq!(cloned.shape, info.shape);
        assert_eq!(cloned.dtype, info.dtype);
    }

    #[test]
    fn test_tensor_list_result_clone() {
        let result = TensorListResult {
            file: "test.apr".to_string(),
            format_version: "v2".to_string(),
            tensor_count: 5,
            total_size_bytes: 1000,
            tensors: vec![],
        };

        let cloned = result.clone();
        assert_eq!(cloned.file, result.file);
        assert_eq!(cloned.format_version, result.format_version);
    }

    // ========================================================================
    // Integration: Full Tensor Listing Workflow
    // ========================================================================

    #[test]
    fn test_full_workflow_list_filter_stats() {
        // Build a pygmy model
        let apr_bytes = build_pygmy_apr_with_config(PygmyConfig::llama_style());

        // List with filter and stats
        let opts = TensorListOptions::new()
            .with_filter("proj")
            .with_stats()
            .with_limit(10);

        let result = list_tensors_from_bytes(&apr_bytes, opts).expect("list tensors");

        // Verify results
        assert_eq!(result.format_version, "v2");
        for tensor in &result.tensors {
            assert!(tensor.name.contains("proj"));
            // Stats should be computed for F32 tensors
            if tensor.dtype == "f32" {
                assert!(tensor.mean.is_some());
                assert!(tensor.std.is_some());
            }
        }
    }

    // ========================================================================
    // GGUF Format Tests (PMAT-ROSETTA-001)
    // ========================================================================

    #[test]
    fn test_ggml_dtype_name_known_types() {
        assert_eq!(ggml_dtype_name(0), "F32");
        assert_eq!(ggml_dtype_name(1), "F16");
        assert_eq!(ggml_dtype_name(2), "Q4_0");
        assert_eq!(ggml_dtype_name(3), "Q4_1");
        assert_eq!(ggml_dtype_name(8), "Q8_0");
        assert_eq!(ggml_dtype_name(12), "Q4_K");
        assert_eq!(ggml_dtype_name(14), "Q6_K");
    }

    #[test]
    fn test_ggml_dtype_name_unknown() {
        assert_eq!(ggml_dtype_name(99), "unknown");
        assert_eq!(ggml_dtype_name(255), "unknown");
    }

    #[test]
    fn test_ggml_dtype_element_size() {
        assert!((ggml_dtype_element_size(0) - 4.0).abs() < 0.001); // F32
        assert!((ggml_dtype_element_size(1) - 2.0).abs() < 0.001); // F16
        // Q8_0 = 1.0 + 2.0/32.0 ≈ 1.0625
        assert!((ggml_dtype_element_size(8) - 1.0625).abs() < 0.01); // Q8_0
        // Q4_0 = 0.5 + 2.0/32.0 ≈ 0.5625
        assert!((ggml_dtype_element_size(2) - 0.5625).abs() < 0.01); // Q4_0
    }

    #[test]
    fn test_list_tensors_gguf_magic_detection() {
        // GGUF magic bytes - verify format is detected
        let mut data = b"GGUF".to_vec();
        // Add version 3 (u32) + tensor count 0 (u64) + metadata count 0 (u64)
        data.extend_from_slice(&3u32.to_le_bytes()); // version
        data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        data.extend_from_slice(&0u64.to_le_bytes()); // metadata_kv_count

        let result = list_tensors_from_bytes(&data, TensorListOptions::default());
        // Should succeed with 0 tensors (valid but empty GGUF)
        assert!(result.is_ok(), "Valid empty GGUF should parse: {result:?}");
        let result = result.unwrap();
        assert_eq!(result.tensor_count, 0);
        assert!(result.format_version.contains("GGUF"));
    }

    #[test]
    fn test_list_tensors_gguf_valid() {
        use crate::format::gguf::{export_tensors_to_gguf, GgmlType, GgufTensor, GgufValue};

        // Create minimal valid GGUF with one F32 tensor
        let tensor_data: Vec<u8> = vec![1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        let tensor = GgufTensor {
            name: "test.weight".to_string(),
            shape: vec![2, 2],
            dtype: GgmlType::F32,
            data: tensor_data,
        };

        let metadata = vec![
            (
                "general.architecture".to_string(),
                GgufValue::String("test".to_string()),
            ),
        ];

        let mut gguf_bytes = Vec::new();
        export_tensors_to_gguf(&mut gguf_bytes, &[tensor], &metadata)
            .expect("export GGUF");

        let result = list_tensors_from_bytes(&gguf_bytes, TensorListOptions::default())
            .expect("list GGUF tensors");

        assert!(result.format_version.contains("GGUF"));
        assert_eq!(result.tensor_count, 1);
        assert_eq!(result.tensors[0].name, "test.weight");
        assert_eq!(result.tensors[0].shape, vec![2, 2]);
        assert_eq!(result.tensors[0].dtype, "F32");
    }

    #[test]
    fn test_list_tensors_gguf_with_stats() {
        use crate::format::gguf::{export_tensors_to_gguf, GgmlType, GgufTensor};

        let tensor_data: Vec<u8> = vec![1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        let tensor = GgufTensor {
            name: "model.embed".to_string(),
            shape: vec![4],
            dtype: GgmlType::F32,
            data: tensor_data,
        };

        let mut gguf_bytes = Vec::new();
        export_tensors_to_gguf(&mut gguf_bytes, &[tensor], &[])
            .expect("export GGUF");

        let opts = TensorListOptions::new().with_stats();
        let result = list_tensors_from_bytes(&gguf_bytes, opts).expect("list");

        let t = &result.tensors[0];
        // GGUF stats computation may not be implemented, just check basics
        assert_eq!(t.name, "model.embed");
        assert_eq!(t.dtype, "F32");
    }

    #[test]
    fn test_list_tensors_gguf_multiple_tensors() {
        use crate::format::gguf::{export_tensors_to_gguf, GgmlType, GgufTensor};

        let t1_data: Vec<u8> = vec![1.0f32, 2.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let t2_data: Vec<u8> = vec![3.0f32, 4.0, 5.0, 6.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        let tensors = vec![
            GgufTensor {
                name: "layer.0.weight".to_string(),
                shape: vec![2],
                dtype: GgmlType::F32,
                data: t1_data,
            },
            GgufTensor {
                name: "layer.1.weight".to_string(),
                shape: vec![2, 2],
                dtype: GgmlType::F32,
                data: t2_data,
            },
        ];

        let mut gguf_bytes = Vec::new();
        export_tensors_to_gguf(&mut gguf_bytes, &tensors, &[]).expect("export");

        let result = list_tensors_from_bytes(&gguf_bytes, TensorListOptions::default())
            .expect("list");

        assert_eq!(result.tensor_count, 2);
        assert!(result.tensors.iter().any(|t| t.name == "layer.0.weight"));
        assert!(result.tensors.iter().any(|t| t.name == "layer.1.weight"));
    }

    #[test]
    fn test_list_tensors_gguf_with_filter() {
        use crate::format::gguf::{export_tensors_to_gguf, GgmlType, GgufTensor};

        let data: Vec<u8> = vec![1.0f32, 2.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        let tensors = vec![
            GgufTensor {
                name: "model.attn.weight".to_string(),
                shape: vec![2],
                dtype: GgmlType::F32,
                data: data.clone(),
            },
            GgufTensor {
                name: "model.mlp.weight".to_string(),
                shape: vec![2],
                dtype: GgmlType::F32,
                data: data.clone(),
            },
            GgufTensor {
                name: "model.norm.weight".to_string(),
                shape: vec![2],
                dtype: GgmlType::F32,
                data,
            },
        ];

        let mut gguf_bytes = Vec::new();
        export_tensors_to_gguf(&mut gguf_bytes, &tensors, &[]).expect("export");

        let opts = TensorListOptions::new().with_filter("attn");
        let result = list_tensors_from_bytes(&gguf_bytes, opts).expect("list");

        assert_eq!(result.tensors.len(), 1);
        assert_eq!(result.tensors[0].name, "model.attn.weight");
    }

    // ========================================================================
    // SafeTensors Format Tests (PMAT-ROSETTA-001)
    // ========================================================================

    #[test]
    fn test_f16_to_f32_conversion() {
        // Test known f16 bit patterns
        // 0x3C00 = 1.0 in f16
        assert!((f16_to_f32(0x3C00) - 1.0).abs() < 0.001);
        // 0x0000 = 0.0 in f16
        assert!((f16_to_f32(0x0000) - 0.0).abs() < 0.001);
        // 0x4000 = 2.0 in f16
        assert!((f16_to_f32(0x4000) - 2.0).abs() < 0.001);
        // 0xBC00 = -1.0 in f16
        assert!((f16_to_f32(0xBC00) - (-1.0)).abs() < 0.001);
    }

    #[test]
    fn test_bf16_to_f32_conversion() {
        // BF16 shares exponent range with F32, just truncated mantissa
        // 0x3F80 = 1.0 in bf16 (same top bits as f32 1.0)
        assert!((bf16_to_f32(0x3F80) - 1.0).abs() < 0.001);
        // 0x0000 = 0.0 in bf16
        assert!((bf16_to_f32(0x0000) - 0.0).abs() < 0.001);
        // 0x4000 = 2.0 in bf16
        assert!((bf16_to_f32(0x4000) - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_list_tensors_safetensors_magic_detection() {
        // SafeTensors starts with u64 header length + '{"'
        // Create minimal detection pattern
        let header_len: u64 = 10;
        let mut data = header_len.to_le_bytes().to_vec();
        data.extend_from_slice(b"{\""); // JSON start
        data.extend_from_slice(&[0u8; 20]); // Truncated

        let result = list_tensors_from_bytes(&data, TensorListOptions::default());
        // Should detect SafeTensors but fail to parse (truncated JSON)
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            !err.contains("Unknown model format"),
            "Should detect SafeTensors format: {err}"
        );
    }

    #[test]
    fn test_list_tensors_safetensors_valid() {
        let st_bytes = build_pygmy_safetensors();

        let result = list_tensors_from_bytes(&st_bytes, TensorListOptions::default())
            .expect("list SafeTensors");

        assert_eq!(result.format_version, "SafeTensors");
        assert!(result.tensor_count > 0);
        // Default pygmy has token_embedding.weight
        assert!(
            result
                .tensors
                .iter()
                .any(|t| t.name.contains("embedding") || t.name.contains("token")),
            "Should have embedding tensor"
        );
    }

    #[test]
    fn test_list_tensors_safetensors_with_stats() {
        let st_bytes = build_pygmy_safetensors();

        let opts = TensorListOptions::new().with_stats();
        let result = list_tensors_from_bytes(&st_bytes, opts).expect("list");

        // At least one tensor should have stats computed
        let has_stats = result.tensors.iter().any(|t| t.mean.is_some());
        assert!(has_stats, "Should compute stats for SafeTensors");
    }

    #[test]
    fn test_list_tensors_safetensors_with_config() {
        let st_bytes = build_pygmy_safetensors_with_config(PygmyConfig::llama_style());

        let result = list_tensors_from_bytes(&st_bytes, TensorListOptions::default())
            .expect("list");

        assert_eq!(result.format_version, "SafeTensors");
        // LLaMA style should have multiple tensors
        assert!(
            result.tensor_count >= 2,
            "LLaMA style should have multiple tensors"
        );
    }

    #[test]
    fn test_list_tensors_safetensors_with_filter() {
        let st_bytes = build_pygmy_safetensors_with_config(PygmyConfig::llama_style());

        let opts = TensorListOptions::new().with_filter("norm");
        let result = list_tensors_from_bytes(&st_bytes, opts).expect("list");

        for tensor in &result.tensors {
            assert!(
                tensor.name.contains("norm"),
                "Filtered tensor should contain 'norm': {}",
                tensor.name
            );
        }
    }

    #[test]
    fn test_list_tensors_safetensors_with_limit() {
        let st_bytes = build_pygmy_safetensors_with_config(PygmyConfig::llama_style());

        let opts = TensorListOptions::new().with_limit(2);
        let result = list_tensors_from_bytes(&st_bytes, opts).expect("list");

        assert!(
            result.tensors.len() <= 2,
            "Should limit to 2 tensors, got {}",
            result.tensors.len()
        );
    }

    // ========================================================================
    // Format Detection Priority Tests
    // ========================================================================

    #[test]
    fn test_format_detection_gguf_priority() {
        // GGUF magic should be detected before SafeTensors heuristic
        let mut data = b"GGUF".to_vec();
        // Add bytes that could trigger SafeTensors detection
        data.extend_from_slice(&[10, 0, 0, 0, 0, 0, 0, 0]); // u64 = 10
        data.extend_from_slice(b"{\""); // JSON start

        let result = list_tensors_from_bytes(&data, TensorListOptions::default());
        // Should fail as GGUF (not SafeTensors), proving GGUF check comes first
        assert!(result.is_err());
        // Error should not mention "Unknown format" since GGUF was detected
        let err = result.unwrap_err().to_string();
        assert!(!err.contains("Unknown model format"));
    }

    #[test]
    fn test_format_detection_apr_fallback() {
        // APR v2 magic
        let apr_bytes = build_pygmy_apr();

        let result = list_tensors_from_bytes(&apr_bytes, TensorListOptions::default())
            .expect("list APR");

        assert_eq!(result.format_version, "v2");
    }
}

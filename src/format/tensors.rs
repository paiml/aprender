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
        let header_len = u64::from_le_bytes(data[0..8].try_into().unwrap_or([0u8; 8]));
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
    let mut total_matching = 0usize;
    let limit_reached = options.limit < usize::MAX;

    for name in reader.tensor_names() {
        // Apply filter
        if let Some(ref pattern) = options.filter {
            if !name.contains(pattern.as_str()) {
                continue;
            }
        }

        // Get tensor entry
        if let Some(entry) = reader.get_tensor(name) {
            let size = entry.size as usize;
            total_size += size;
            total_matching += 1;

            // Only collect details up to the limit
            if tensors.len() < options.limit {
                let mut info = tensor_info_from_entry(entry);

                // Compute stats if requested
                if options.compute_stats {
                    if let Some(data) = reader.get_tensor_as_f32(name) {
                        compute_tensor_stats(&mut info, &data);
                    }
                }

                tensors.push(info);
            }
        }
    }

    // When no limit is applied, total_matching == tensors.len().
    // When a limit truncates, total_matching reflects the true count.
    let _ = limit_reached;

    Ok(TensorListResult {
        file: String::new(), // Set by caller
        format_version: "v2".to_string(),
        tensor_count: total_matching,
        total_size_bytes: total_size,
        tensors,
    })
}

/// Parse shape array from JSON value
fn parse_shape_array(shape_val: &serde_json::Value) -> Vec<usize> {
    shape_val.as_array().map_or(Vec::new(), |arr| {
        arr.iter()
            .filter_map(|v| v.as_u64().map(|n| n as usize))
            .collect()
    })
}

/// GH-195 FIX: Extract tensors with accurate total count and size
/// Returns (tensors_up_to_limit, total_matching_count, total_size_bytes)
fn extract_tensors_from_metadata_with_counts(
    metadata: &HashMap<String, serde_json::Value>,
    options: &TensorListOptions,
) -> (Vec<TensorInfo>, usize, usize) {
    let Some(shapes) = metadata.get("tensor_shapes").and_then(|s| s.as_object()) else {
        return (Vec::new(), 0, 0);
    };

    let mut tensors = Vec::new();
    let mut total_matching = 0usize;
    let mut total_size = 0usize;

    for (name, shape_val) in shapes {
        // Apply filter
        if let Some(ref pattern) = options.filter {
            if !name.contains(pattern.as_str()) {
                continue;
            }
        }

        let shape = parse_shape_array(shape_val);
        let size_bytes = shape.iter().product::<usize>() * 4; // Assume f32

        total_size += size_bytes;
        total_matching += 1;

        // Only collect details up to the limit
        if tensors.len() < options.limit {
            tensors.push(TensorInfo {
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
            });
        }
    }

    (tensors, total_matching, total_size)
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

    // GH-195 FIX: Extract ALL matching tensors first to get true count and total size
    let (tensors, total_matching, total_size) =
        extract_tensors_from_metadata_with_counts(&metadata, &options);

    Ok(TensorListResult {
        file: String::new(),
        format_version: "v1".to_string(),
        tensor_count: total_matching,
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
        19 => "IQ1_S",
        20 => "IQ4_NL",
        21 => "IQ3_S",
        22 => "IQ2_S",
        23 => "IQ4_XS",
        24 => "I8",
        25 => "I16",
        26 => "BF16",
        27 => "I32",
        28 => "I64",
        29 => "F64",
        30 => "IQ1_M",
        _ => "unknown",
    }
}

/// Bytes per element for GGML data types (approximate for block types)
fn ggml_dtype_element_size(dtype: u32) -> f64 {
    match dtype {
        0 => 4.0,                // F32
        1 => 2.0,                // F16
        2 => 0.5 + 2.0 / 32.0,   // Q4_0: 4-bit + scale
        3 => 0.5 + 4.0 / 32.0,   // Q4_1: 4-bit + scale + min
        6 => 0.625 + 2.0 / 32.0, // Q5_0
        7 => 0.625 + 4.0 / 32.0, // Q5_1
        8 => 1.0 + 2.0 / 32.0,   // Q8_0
        9 => 1.0 + 4.0 / 32.0,   // Q8_1
        10 => 0.3125,            // Q2_K
        11 => 0.4375,            // Q3_K
        12 => 0.5625,            // Q4_K
        13 => 0.6875,            // Q5_K
        14 => 0.8125,            // Q6_K
        15 => 1.0625,            // Q8_K
        26 => 2.0,               // BF16
        // GGML I-quants (importance matrix quantization)
        16 => 0.5625, // IQ2_XXS
        17 => 0.625,  // IQ2_XS
        18 => 0.6875, // IQ3_XXS
        19 => 0.4375, // IQ1_S
        20 => 0.5625, // IQ4_NL
        21 => 0.4375, // IQ3_S
        22 => 0.625,  // IQ2_S
        23 => 0.5,    // IQ4_XS
        24 => 1.0,    // I8
        25 => 2.0,    // I16
        27 => 4.0,    // I32
        28 => 8.0,    // I64
        29 => 8.0,    // F64
        30 => 0.375,  // IQ1_M
        // Unknown dtype: use F32 size (4 bytes) as conservative estimate.
        // This is intentional — for size estimation purposes, overestimating
        // is safer than underestimating. The dtype name function above will
        // report "unknown" for diagnostics.
        _ => 4.0,
    }
}

/// List tensors from GGUF file bytes
fn list_tensors_gguf(data: &[u8], options: TensorListOptions) -> Result<TensorListResult> {
    let reader = GgufReader::from_bytes(data.to_vec()).map_err(|e| AprenderError::FormatError {
        message: format!("Failed to parse GGUF: {e}"),
    })?;

    let mut tensors = Vec::new();
    let mut total_size = 0usize;
    let mut total_matching = 0usize;

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

        total_size += size_bytes;
        total_matching += 1;

        // Only collect details up to the limit
        if tensors.len() < options.limit {
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

            tensors.push(info);
        }
    }

    Ok(TensorListResult {
        file: String::new(),
        format_version: format!("GGUF v{}", reader.version),
        tensor_count: total_matching,
        total_size_bytes: total_size,
        tensors,
    })
}

// ============================================================================
// SafeTensors Format Support (PMAT-ROSETTA-001)
// ============================================================================

/// Parse and validate the SafeTensors JSON header, returning the parsed header
/// as a `serde_json::Value` (guaranteed to be an object) and the byte offset
/// where tensor data begins.
fn parse_safetensors_header(data: &[u8]) -> Result<(serde_json::Value, usize)> {
    if data.len() < 8 {
        return Err(AprenderError::FormatError {
            message: "SafeTensors file too small".to_string(),
        });
    }

    let header_len =
        u64::from_le_bytes(
            data[0..8]
                .try_into()
                .map_err(|_| AprenderError::FormatError {
                    message: "Failed to read SafeTensors header length".to_string(),
                })?,
        ) as usize;

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

    if !header.is_object() {
        return Err(AprenderError::FormatError {
            message: "SafeTensors header is not a JSON object".to_string(),
        });
    }

    let data_start = 8 + header_len;
    Ok((header, data_start))
}

/// Extract a `TensorInfo` from a SafeTensors JSON tensor entry.
/// Returns the info and the relative byte offsets `(start, end)` within the
/// data section (if present in the entry).
fn extract_safetensors_tensor_info(
    name: &str,
    value: &serde_json::Value,
) -> (TensorInfo, Option<(usize, usize)>) {
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

    let relative_offsets = value
        .get("data_offsets")
        .and_then(|v| v.as_array())
        .and_then(|arr| {
            let start = arr.first()?.as_u64()? as usize;
            let end = arr.get(1)?.as_u64()? as usize;
            Some((start, end))
        });

    let size_bytes = relative_offsets
        .map(|(start, end)| end - start)
        .unwrap_or(0);

    let info = TensorInfo {
        name: name.to_string(),
        shape,
        dtype,
        size_bytes,
        mean: None,
        std: None,
        min: None,
        max: None,
        nan_count: None,
        inf_count: None,
    };

    (info, relative_offsets)
}

/// Compute and populate stats on a `TensorInfo` from its SafeTensors byte
/// range. `data` is the full file buffer; `data_start` is the byte offset
/// where the tensor data section begins; `relative_offsets` are
/// `(start, end)` relative to that section.
fn populate_safetensors_stats(
    info: &mut TensorInfo,
    data: &[u8],
    data_start: usize,
    relative_offsets: (usize, usize),
) {
    let (start, end) = relative_offsets;
    let abs_start = data_start + start;
    let abs_end = data_start + end;
    if abs_end > data.len() {
        return;
    }
    let tensor_bytes = &data[abs_start..abs_end];
    let f32_data = safetensors_bytes_to_f32(tensor_bytes, &info.dtype);
    compute_tensor_stats(info, &f32_data);
}

/// Check whether a tensor name passes the optional filter pattern.
fn matches_filter(name: &str, filter: Option<&String>) -> bool {
    match filter {
        Some(pattern) => name.contains(pattern.as_str()),
        None => true,
    }
}

/// List tensors from SafeTensors file bytes by parsing the JSON header
fn list_tensors_safetensors(data: &[u8], options: TensorListOptions) -> Result<TensorListResult> {
    let (header, data_start) = parse_safetensors_header(data)?;

    // Safety: parse_safetensors_header validated this is an object
    let obj = header
        .as_object()
        .expect("parse_safetensors_header guarantees object");

    let mut tensors = Vec::new();
    let mut total_size = 0usize;
    let mut total_matching = 0usize;

    // Collect and sort tensor names for deterministic output
    let mut tensor_entries: Vec<(&String, &serde_json::Value)> =
        obj.iter().filter(|(k, _)| *k != "__metadata__").collect();
    tensor_entries.sort_by_key(|(k, _)| *k);

    for (name, value) in tensor_entries {
        if !matches_filter(name, options.filter.as_ref()) {
            continue;
        }

        let (mut info, relative_offsets) = extract_safetensors_tensor_info(name, value);

        total_size += info.size_bytes;
        total_matching += 1;

        if tensors.len() >= options.limit {
            continue;
        }

        if options.compute_stats {
            if let Some(offsets) = relative_offsets {
                populate_safetensors_stats(&mut info, data, data_start, offsets);
            }
        }

        tensors.push(info);
    }

    Ok(TensorListResult {
        file: String::new(),
        format_version: "SafeTensors".to_string(),
        tensor_count: total_matching,
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
    let mut total_matching = 0usize;

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

            total_size += size_bytes;
            total_matching += 1;

            // Only collect details up to the limit
            if tensors.len() < options.limit {
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

                tensors.push(info);
            }
        }
    }

    Ok(TensorListResult {
        file: String::new(),
        format_version: "SafeTensors".to_string(),
        tensor_count: total_matching,
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

#[cfg(test)]
#[path = "tensors_tests.rs"]
mod tests;

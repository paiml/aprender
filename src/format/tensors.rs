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
    if data.get(0..4) == Some(b"GGUF") {
        return list_tensors_gguf(data, options);
    }

    if data.len() >= 10 {
        let header_len = u64::from_le_bytes(
            data.get(0..8)
                .and_then(|s| s.try_into().ok())
                .unwrap_or([0u8; 8]),
        );
        if header_len < 100_000_000 && data.get(8..10) == Some(b"{\"") {
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

include!("tensors_part_02.rs");
include!("tensors_part_03.rs");

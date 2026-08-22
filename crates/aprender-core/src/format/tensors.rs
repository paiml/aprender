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
use crate::format::v2::{AprV2Reader, AprV2ReaderRef, TensorIndexEntry};
// issue #2231: `get_tensor_as_f32` is the re-attached `AprV2DequantExt` method.
use crate::format::AprV2DequantExt;
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

    /// Set filter pattern (supports substring match and simple glob: `*` and `?`)
    #[must_use]
    pub fn with_filter(mut self, pattern: impl Into<String>) -> Self {
        self.filter = Some(pattern.into());
        self
    }

    /// Check if a tensor name matches the filter pattern.
    /// GH-669: Supports glob-style `*` (any chars) and `?` (one char).
    /// Falls back to substring match when no glob chars present.
    pub fn matches_filter(&self, name: &str) -> bool {
        match &self.filter {
            None => true,
            Some(pattern) => {
                if pattern.contains('*') || pattern.contains('?') {
                    glob_match(pattern, name)
                } else {
                    name.contains(pattern.as_str())
                }
            }
        }
    }

    /// Set maximum tensor count
    #[must_use]
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }
}

/// Simple glob matching: `*` matches any sequence, `?` matches one char.
/// GH-669: Enables `--filter 'blk.0.*'` to match `blk.0.attn_k.weight` etc.
fn glob_match(pattern: &str, text: &str) -> bool {
    let p = pattern.as_bytes();
    let t = text.as_bytes();
    let (mut pi, mut ti) = (0, 0);
    let (mut star_pi, mut star_ti) = (usize::MAX, 0);
    while ti < t.len() {
        if pi < p.len() && (p[pi] == b'?' || p[pi] == t[ti]) {
            pi += 1;
            ti += 1;
        } else if pi < p.len() && p[pi] == b'*' {
            star_pi = pi;
            star_ti = ti;
            pi += 1;
        } else if star_pi != usize::MAX {
            pi = star_pi + 1;
            star_ti += 1;
            ti = star_ti;
        } else {
            return false;
        }
    }
    while pi < p.len() && p[pi] == b'*' {
        pi += 1;
    }
    pi == p.len()
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
// Tensor Table Bounds Check (#2569)
// ============================================================================

/// Refuse a tensor table whose declared extents do not fit inside the file.
///
/// #2569: `apr tensors` computed each row's `size_bytes` from the DECLARED
/// shape and never asked whether those bytes exist. On a 128-byte GGUF
/// declaring 8192 bytes of tensor data it printed the whole table and exited 0
/// while `apr validate` on the same file exited 5 with
/// "(file is N bytes too short)" — two commands, one binary, one file, opposite
/// answers, and the one that said OK is the JSON the MCP tool and CI consume.
///
/// The check and its wording are lifted from `RosettaStone::validate_gguf`
/// (`format/rosetta/arch_inference.rs`, S1-FIX), which has had it all along.
/// This one is stricter in the way that matters: `validate` only asks whether
/// the LAST tensor's start byte exists, so it cannot see a file that is short
/// by less than the final tensor's length. Here every tensor's full extent —
/// `data_start + offset + size` — must be inside the file, because that extent
/// is exactly what the printed row asserts.
///
/// `tensors` yields `(name, offset_within_data_section, size_bytes)` and must
/// cover EVERY tensor in the file, not just the ones a `--filter` selected: a
/// filtered listing must not be able to hide a truncation.
///
/// # Errors
/// Returns `FormatError` naming the first tensor that overruns the file, or
/// whose declared extent overflows `u64`.
fn check_tensor_table_fits<'a, I>(
    format: &str,
    file_len: u64,
    data_start: u64,
    tensors: I,
) -> Result<()>
where
    I: IntoIterator<Item = (&'a str, u64, u64)>,
{
    for (name, offset, size) in tensors {
        let start = data_start
            .checked_add(offset)
            .ok_or_else(|| AprenderError::FormatError {
                message: format!(
                    "Corrupt {format}: tensor '{name}' offset {offset} overflows the data \
                     section start {data_start} (file is {file_len} bytes)"
                ),
            })?;
        let end = start
            .checked_add(size)
            .ok_or_else(|| AprenderError::FormatError {
                message: format!(
                    "Corrupt {format}: tensor '{name}' declares {size} bytes at {start}, which \
                     overflows u64 (file is {file_len} bytes)"
                ),
            })?;
        if end > file_len {
            return Err(AprenderError::FormatError {
                message: format!(
                    "Truncated {format}: file is {file_len} bytes but tensor '{name}' declares \
                     bytes {start}..{end} (file is {short} bytes too short)",
                    short = end - file_len,
                ),
            });
        }
    }
    Ok(())
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
/// Build a `TensorInfo` from a v2 reader entry, optionally computing stats.
fn build_v2_tensor_info(
    reader: &AprV2Reader,
    name: &str,
    entry: &TensorIndexEntry,
    compute_stats: bool,
) -> TensorInfo {
    let mut info = tensor_info_from_entry(entry);
    if compute_stats {
        if let Some(data) = reader.get_tensor_as_f32(name) {
            compute_tensor_stats(&mut info, &data);
        }
    }
    info
}

fn list_tensors_v2(data: &[u8], options: TensorListOptions) -> Result<TensorListResult> {
    // Parse with v2 reader
    let reader = AprV2Reader::from_bytes(data).map_err(|e| AprenderError::FormatError {
        message: format!("Failed to parse APR v2: {e}"),
    })?;

    // #2569 / follow-up filed in #2564: the APR v2 index carries each tensor's real
    // byte size, and this listing believed it without checking the file was that
    // long. A 50 MB head of a 991 MB model printed "291 tensors 942.3 MB", 19x the
    // file. Unlike GGUF there is no dtype estimate here — `entry.size` is exact.
    check_tensor_table_fits(
        "APR v2",
        data.len() as u64,
        reader.header().data_offset,
        reader
            .tensor_names()
            .into_iter()
            .filter_map(|name| reader.get_tensor(name).map(|e| (name, e.offset, e.size)))
            .collect::<Vec<_>>(),
    )?;

    // Get tensor info from actual index
    let mut tensors = Vec::new();
    let mut total_size = 0usize;
    let mut total_matching = 0usize;

    for name in reader.tensor_names() {
        if !options.matches_filter(name) {
            continue;
        }

        if let Some(entry) = reader.get_tensor(name) {
            total_size += entry.size as usize;
            total_matching += 1;

            if tensors.len() < options.limit {
                tensors.push(build_v2_tensor_info(
                    &reader,
                    name,
                    entry,
                    options.compute_stats,
                ));
            }
        }
    }

    Ok(TensorListResult {
        file: String::new(), // Set by caller
        format_version: "v2".to_string(),
        tensor_count: total_matching,
        total_size_bytes: total_size,
        tensors,
    })
}

/// List tensors from APR v2 via mmap (realizar#136 — constant memory)
///
/// Uses `AprV2ReaderRef` which borrows the mmap'd slice instead of copying
/// the entire file into a `Vec<u8>`. Peak RSS = header + index (~180KB).
fn list_tensors_v2_mmap(data: &[u8], options: TensorListOptions) -> Result<TensorListResult> {
    let reader = AprV2ReaderRef::from_bytes(data).map_err(|e| AprenderError::FormatError {
        message: format!("Failed to parse APR v2: {e}"),
    })?;

    // #2569: same check as `list_tensors_v2`. This mmap path is the one `apr tensors`
    // actually takes for an APR v2 file on disk, so leaving it out would have made
    // the sibling fix theater.
    check_tensor_table_fits(
        "APR v2",
        data.len() as u64,
        reader.header().data_offset,
        reader
            .tensor_names()
            .into_iter()
            .filter_map(|name| reader.get_tensor(name).map(|e| (name, e.offset, e.size)))
            .collect::<Vec<_>>(),
    )?;

    let mut tensors = Vec::new();
    let mut total_size = 0usize;
    let mut total_matching = 0usize;

    for name in reader.tensor_names() {
        if !options.matches_filter(name) {
            continue;
        }

        if let Some(entry) = reader.get_tensor(name) {
            total_size += entry.size as usize;
            total_matching += 1;

            if tensors.len() < options.limit {
                let mut info = tensor_info_from_entry(entry);
                if options.compute_stats {
                    if let Some(data) = reader.get_tensor_as_f32(name) {
                        compute_tensor_stats(&mut info, &data);
                    }
                }
                tensors.push(info);
            }
        }
    }

    Ok(TensorListResult {
        file: String::new(),
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
        if !options.matches_filter(name) {
            continue;
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

/// GGML dtype id to human-readable name (table lookup, O(1))
///
/// Contract: apr-inspect-dtype-naming-v1 F-INSPECT-DTYPE-001 (paiml/aprender#619).
/// Made pub(crate) so `format::rosetta::validate_inspect` can render dtype names
/// consistently with `apr tensors` output.
pub(crate) fn ggml_dtype_name(dtype: u32) -> &'static str {
    // Indexed by GGML type code (ggml.h GGML_TYPE_*). The tail (24-30) was
    // misordered — BF16 was at 26, mislabeling BF16 tensors (code 30) as
    // "IQ1_M", I32 (26) as "BF16", etc. Correct order per ggml.h:
    //   I8=24, I16=25, I32=26, I64=27, F64=28, IQ1_M=29, BF16=30.
    const NAMES: [&str; 31] = [
        "F32", "F16", "Q4_0", "Q4_1", "unknown", "unknown", "Q5_0", "Q5_1", "Q8_0", "Q8_1", "Q2_K",
        "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_K", "IQ2_XXS", "IQ2_XS", "IQ3_XXS", "IQ1_S", "IQ4_NL",
        "IQ3_S", "IQ2_S", "IQ4_XS", "I8", "I16", "I32", "I64", "F64", "IQ1_M", "BF16",
    ];
    NAMES.get(dtype as usize).copied().unwrap_or("unknown")
}

include!("safetensors.rs");
include!("tensors_safetensors.rs");

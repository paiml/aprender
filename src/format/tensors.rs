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
    match magic {
        &MAGIC_APRN => Some("v1"),
        &MAGIC_APR1 => Some("v1"),
        &MAGIC_APR2 => Some("v2"),
        &MAGIC_APR0 => Some("v2"),
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

/// List tensors from APR file bytes
///
/// This is the core function that reads from the actual tensor index,
/// not just metadata. This fixes TOOL-APR-001.
///
/// # Arguments
/// * `data` - Raw APR file bytes
/// * `options` - Listing options
///
/// # Errors
/// Returns error if the format is invalid or parsing fails.
pub fn list_tensors_from_bytes(data: &[u8], options: TensorListOptions) -> Result<TensorListResult> {
    // Check minimum size
    if data.len() < 4 {
        return Err(AprenderError::FormatError {
            message: "File too small to contain APR header".to_string(),
        });
    }

    // Read magic bytes
    let magic: [u8; 4] = data[0..4].try_into().map_err(|_| AprenderError::FormatError {
        message: "Failed to read magic bytes".to_string(),
    })?;

    // Detect format version
    let format_version = detect_format(&magic).ok_or_else(|| AprenderError::FormatError {
        message: format!(
            "Invalid APR magic: expected APRN/APR1/APR2/APR\\0, got {:?}",
            magic
        ),
    })?;

    // Parse based on format version
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

/// List tensors from an APR file
///
/// # Arguments
/// * `path` - Path to APR file
/// * `options` - Listing options
///
/// # Errors
/// Returns error if the file doesn't exist or is invalid.
pub fn list_tensors(path: impl AsRef<Path>, options: TensorListOptions) -> Result<TensorListResult> {
    let path = path.as_ref();

    // Read file
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut data = Vec::new();
    reader.read_to_end(&mut data)?;

    // Parse
    let mut result = list_tensors_from_bytes(&data, options)?;
    result.file = path.display().to_string();

    Ok(result)
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
        let std = if variance > 0.0 {
            variance.sqrt()
        } else {
            0.0
        };

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
        build_pygmy_apr_with_config, PygmyConfig,
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
        let has_stats = result.tensors.iter().any(|t| {
            t.mean.is_some() && t.std.is_some() && t.nan_count.is_some()
        });
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
        assert!(err.to_string().contains("Invalid APR magic"));
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
}

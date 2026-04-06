//! Unified Model Format Detection and Loading
//!
//! Per spec §3: Format Support Matrix - auto-detect APR, GGUF, SafeTensors from magic bytes.
//!
//! ## Jidoka (Built-in Quality)
//!
//! - CRC32 verification for APR format
//! - Header size validation for SafeTensors (DOS protection)
//! - Magic byte validation for GGUF
//!
//! ## Supported Formats
//!
//! | Format | Magic | Extension |
//! |--------|-------|-----------|
//! | APR    | `APR\0` | `.apr` |
//! | GGUF   | `GGUF` | `.gguf` |
//! | SafeTensors | (u64 header size) | `.safetensors` |

use std::path::Path;

/// Detected model format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFormat {
    /// Aprender native format (first-class support)
    Apr,
    /// GGUF format (llama.cpp compatible)
    Gguf,
    /// SafeTensors format (HuggingFace compatible)
    SafeTensors,
}

impl std::fmt::Display for ModelFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Apr => write!(f, "APR"),
            Self::Gguf => write!(f, "GGUF"),
            Self::SafeTensors => write!(f, "SafeTensors"),
        }
    }
}

/// Errors during format detection
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FormatError {
    /// Data too short for format detection (need at least 8 bytes)
    TooShort {
        /// Actual length
        len: usize,
    },
    /// Unknown format (no magic bytes matched)
    UnknownFormat,
    /// SafeTensors header too large (DOS protection per spec §7.1)
    HeaderTooLarge {
        /// Header size in bytes
        size: u64,
    },
    /// File extension doesn't match detected format
    ExtensionMismatch {
        /// Detected format
        detected: ModelFormat,
        /// Extension from filename
        extension: String,
    },
}

impl std::fmt::Display for FormatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooShort { len } => {
                write!(
                    f,
                    "Data too short for format detection: {len} bytes (need 8)"
                )
            },
            Self::UnknownFormat => write!(f, "Unknown model format (no magic bytes matched)"),
            Self::HeaderTooLarge { size } => write!(
                f,
                "SafeTensors header too large: {size} bytes (max 100MB for DOS protection)"
            ),
            Self::ExtensionMismatch {
                detected,
                extension,
            } => {
                write!(
                    f,
                    "Extension mismatch: detected {detected} but file has extension .{extension}"
                )
            },
        }
    }
}

impl std::error::Error for FormatError {}

/// APR format magic bytes (first 3 bytes, 4th is version)
///
/// APR v1: `APR1` (0x41505231)
/// APR v2: `APR2` (0x41505232)
/// Legacy: `APR\0` (0x41505200)
pub const APR_MAGIC: &[u8; 3] = b"APR";

/// GGUF format magic bytes
pub const GGUF_MAGIC: &[u8; 4] = b"GGUF";

/// Maximum SafeTensors header size (100MB for DOS protection per spec §7.1)
pub const MAX_SAFETENSORS_HEADER: u64 = 100_000_000;

/// Valid APR version bytes
const APR_VERSIONS: [u8; 4] = [b'N', b'1', b'2', 0];

/// Try to detect APR format from magic bytes
#[inline]
fn try_detect_apr(data: &[u8]) -> Option<ModelFormat> {
    if data.len() >= 4
        && data.get(0..3).expect("length checked >= 4") == APR_MAGIC
        && APR_VERSIONS.contains(&data[3])
    {
        return Some(ModelFormat::Apr);
    }
    None
}

/// Try to detect GGUF format from magic bytes
#[inline]
fn try_detect_gguf(data: &[u8]) -> Option<ModelFormat> {
    if data.len() >= 4 && data.get(0..4).expect("length checked >= 4") == GGUF_MAGIC {
        return Some(ModelFormat::Gguf);
    }
    None
}

/// Try to detect SafeTensors format from header size
#[inline]
fn try_detect_safetensors(data: &[u8]) -> Result<Option<ModelFormat>, FormatError> {
    let header_size = u64::from_le_bytes(data[0..8].try_into().expect("slice is exactly 8 bytes"));
    if header_size > 0 && header_size < MAX_SAFETENSORS_HEADER {
        return Ok(Some(ModelFormat::SafeTensors));
    }
    if header_size >= MAX_SAFETENSORS_HEADER {
        return Err(FormatError::HeaderTooLarge { size: header_size });
    }
    Ok(None)
}

/// Detect model format from magic bytes (Jidoka: fail-fast)
///
/// Per spec §3.2: Format Detection
///
/// # Arguments
///
/// * `data` - First 8+ bytes of the model file
///
/// # Returns
///
/// Detected format or error
///
/// # Errors
///
/// Returns error if:
/// - Data is too short (<8 bytes)
/// - No known magic bytes detected
/// - SafeTensors header size exceeds limit (DOS protection)
///
/// # Example
///
/// ```
/// use realizar::format::{detect_format, ModelFormat};
///
/// // APR format
/// let apr_data = b"APR\0xxxxxxxxxxxx";
/// assert_eq!(detect_format(apr_data).expect("test"), ModelFormat::Apr);
///
/// // GGUF format
/// let gguf_data = b"GGUFxxxxxxxxxxxx";
/// assert_eq!(detect_format(gguf_data).expect("test"), ModelFormat::Gguf);
/// ```
pub fn detect_format(data: &[u8]) -> Result<ModelFormat, FormatError> {
    if data.len() < 8 {
        return Err(FormatError::TooShort { len: data.len() });
    }

    // Try each format in order of specificity
    if let Some(format) = try_detect_apr(data) {
        return Ok(format);
    }
    if let Some(format) = try_detect_gguf(data) {
        return Ok(format);
    }
    if let Some(format) = try_detect_safetensors(data)? {
        return Ok(format);
    }

    Err(FormatError::UnknownFormat)
}

/// Detect format from file path (using extension as hint, then verify magic)
///
/// # Arguments
///
/// * `path` - Path to model file
///
/// # Returns
///
/// Detected format (verified against magic bytes if data provided)
///
/// # Errors
///
/// Returns `FormatError::UnknownFormat` if extension is not recognized.
///
/// # Example
///
/// ```
/// use realizar::format::{detect_format_from_path, ModelFormat};
/// use std::path::Path;
///
/// assert_eq!(
///     detect_format_from_path(Path::new("model.apr")).expect("test"),
///     ModelFormat::Apr
/// );
/// ```
pub fn detect_format_from_path(path: &Path) -> Result<ModelFormat, FormatError> {
    let extension = path.extension().and_then(|e| e.to_str()).unwrap_or("");

    match extension.to_lowercase().as_str() {
        "apr" => Ok(ModelFormat::Apr),
        "gguf" => Ok(ModelFormat::Gguf),
        "safetensors" => Ok(ModelFormat::SafeTensors),
        _ => Err(FormatError::UnknownFormat),
    }
}

/// Detect format from path and verify against data magic bytes
///
/// Per Jidoka: stop immediately if extension doesn't match magic
///
/// # Arguments
///
/// * `path` - Path to model file
/// * `data` - First 8+ bytes of model data
///
/// # Returns
///
/// Verified format or error if mismatch
///
/// # Errors
///
/// Returns error if:
/// - Format cannot be detected from magic bytes
/// - File extension doesn't match detected format
pub fn detect_and_verify_format(path: &Path, data: &[u8]) -> Result<ModelFormat, FormatError> {
    let from_data = detect_format(data)?;
    let from_path = detect_format_from_path(path);

    // If path detection succeeded, verify it matches data
    if let Ok(path_format) = from_path {
        if path_format != from_data {
            return Err(FormatError::ExtensionMismatch {
                detected: from_data,
                extension: path
                    .extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("unknown")
                    .to_string(),
            });
        }
    }

    // Data-based detection is authoritative
    Ok(from_data)
}

include!("format_detect.rs");

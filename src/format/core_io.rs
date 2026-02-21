//! APR format core I/O operations (save, load, inspect)

use super::{Compression, Header, Metadata, ModelInfo, ModelType, SaveOptions, HEADER_SIZE};
use crate::error::{AprenderError, Result};
use serde::{de::DeserializeOwned, Serialize};
use std::fs::File;
#[cfg(feature = "format-compression")]
use std::io::Cursor;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

#[cfg(feature = "format-encryption")]
use super::{KEY_SIZE, NONCE_SIZE, SALT_SIZE};

/// Compress payload based on algorithm (spec 3.3)
#[allow(clippy::unnecessary_wraps)] // Returns Result to handle compression errors when feature enabled
pub(crate) fn compress_payload(
    data: &[u8],
    compression: Compression,
) -> Result<(Vec<u8>, Compression)> {
    match compression {
        Compression::None => Ok((data.to_vec(), Compression::None)),
        #[cfg(feature = "format-compression")]
        Compression::ZstdDefault => {
            // Zstd level 3 (good balance of speed and ratio)
            let compressed = zstd::encode_all(Cursor::new(data), 3).map_err(|e| {
                AprenderError::Serialization(format!("Zstd compression failed: {e}"))
            })?;
            Ok((compressed, Compression::ZstdDefault))
        }
        #[cfg(feature = "format-compression")]
        Compression::ZstdMax => {
            // Zstd level 19 (maximum compression for archival)
            let compressed = zstd::encode_all(Cursor::new(data), 19).map_err(|e| {
                AprenderError::Serialization(format!("Zstd compression failed: {e}"))
            })?;
            Ok((compressed, Compression::ZstdMax))
        }
        #[cfg(not(feature = "format-compression"))]
        Compression::ZstdDefault | Compression::ZstdMax => {
            // Feature not enabled, fall back to no compression
            Ok((data.to_vec(), Compression::None))
        }
        #[cfg(feature = "format-compression")]
        Compression::Lz4 => {
            // LZ4 compression using lz4_flex with prepended size (GH-146)
            let compressed = lz4_flex::compress_prepend_size(data);
            Ok((compressed, Compression::Lz4))
        }
        #[cfg(not(feature = "format-compression"))]
        Compression::Lz4 => {
            // Feature not enabled, fall back to no compression
            Ok((data.to_vec(), Compression::None))
        }
    }
}

/// Decompress payload based on algorithm (spec 3.3)
pub(crate) fn decompress_payload(data: &[u8], compression: Compression) -> Result<Vec<u8>> {
    match compression {
        Compression::None => Ok(data.to_vec()),
        #[cfg(feature = "format-compression")]
        Compression::ZstdDefault | Compression::ZstdMax => zstd::decode_all(Cursor::new(data))
            .map_err(|e| AprenderError::Serialization(format!("Zstd decompression failed: {e}"))),
        #[cfg(not(feature = "format-compression"))]
        Compression::ZstdDefault | Compression::ZstdMax => Err(AprenderError::FormatError {
            message: "Zstd compression not supported (enable format-compression feature)"
                .to_string(),
        }),
        #[cfg(feature = "format-compression")]
        Compression::Lz4 => lz4_flex::decompress_size_prepended(data)
            .map_err(|e| AprenderError::Serialization(format!("LZ4 decompression failed: {e}"))),
        #[cfg(not(feature = "format-compression"))]
        Compression::Lz4 => Err(AprenderError::FormatError {
            message: "LZ4 compression not supported (enable format-compression feature)"
                .to_string(),
        }),
    }
}

/// CRC32 checksum (IEEE polynomial)
pub(crate) fn crc32(data: &[u8]) -> u32 {
    // CRC32 lookup table (IEEE polynomial 0xEDB88320)
    const TABLE: [u32; 256] = {
        let mut table = [0u32; 256];
        let mut i = 0;
        while i < 256 {
            let mut crc = i as u32;
            let mut j = 0;
            while j < 8 {
                if crc & 1 != 0 {
                    crc = (crc >> 1) ^ 0xEDB8_8320;
                } else {
                    crc >>= 1;
                }
                j += 1;
            }
            table[i] = crc;
            i += 1;
        }
        table
    };

    let mut crc = 0xFFFF_FFFF_u32;
    for &byte in data {
        let idx = ((crc ^ u32::from(byte)) & 0xFF) as usize;
        crc = (crc >> 8) ^ TABLE[idx];
    }
    !crc
}

// ============================================================================
// FILE LOADING HELPER FUNCTIONS (Refactored for reduced complexity)
// ============================================================================

/// Read entire file content into a buffer.
#[cfg(any(feature = "format-signing", feature = "format-encryption"))]
pub(crate) fn read_file_content(path: &Path) -> Result<Vec<u8>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut content = Vec::new();
    reader.read_to_end(&mut content)?;
    Ok(content)
}

/// Verify CRC32 checksum at end of file content.
#[cfg(any(feature = "format-signing", feature = "format-encryption"))]
pub(crate) fn verify_file_checksum(content: &[u8]) -> Result<()> {
    if content.len() < 4 {
        return Err(AprenderError::FormatError {
            message: "File too small for checksum".to_string(),
        });
    }
    let stored_checksum = u32::from_le_bytes([
        content[content.len() - 4],
        content[content.len() - 3],
        content[content.len() - 2],
        content[content.len() - 1],
    ]);
    let computed_checksum = crc32(&content[..content.len() - 4]);
    if stored_checksum != computed_checksum {
        return Err(AprenderError::ChecksumMismatch {
            expected: stored_checksum,
            actual: computed_checksum,
        });
    }
    Ok(())
}

/// Parse header and validate model type.
#[cfg(any(feature = "format-signing", feature = "format-encryption"))]
pub(crate) fn parse_and_validate_header(
    content: &[u8],
    expected_type: ModelType,
) -> Result<Header> {
    let header = Header::from_bytes(&content[..HEADER_SIZE])?;
    if header.model_type != expected_type {
        return Err(AprenderError::FormatError {
            message: format!(
                "Model type mismatch: file contains {:?}, expected {:?}",
                header.model_type, expected_type
            ),
        });
    }
    Ok(header)
}

/// Verify header flag is set for signed files.
#[cfg(feature = "format-signing")]
pub(crate) fn verify_signed_flag(header: &Header) -> Result<()> {
    if !header.flags.is_signed() {
        return Err(AprenderError::FormatError {
            message: "File is not signed (SIGNED flag not set)".to_string(),
        });
    }
    Ok(())
}

/// Verify header flag is set for encrypted files.
#[cfg(feature = "format-encryption")]
pub(crate) fn verify_encrypted_flag(header: &Header) -> Result<()> {
    if !header.flags.is_encrypted() {
        return Err(AprenderError::FormatError {
            message: "File is not encrypted (ENCRYPTED flag not set)".to_string(),
        });
    }
    Ok(())
}

/// Verify payload boundary is within file content.
#[cfg(any(feature = "format-signing", feature = "format-encryption"))]
pub(crate) fn verify_payload_boundary(payload_end: usize, content_len: usize) -> Result<()> {
    if payload_end > content_len - 4 {
        return Err(AprenderError::FormatError {
            message: "Payload extends beyond file boundary".to_string(),
        });
    }
    Ok(())
}

/// Decompress and deserialize payload.
#[cfg(feature = "format-signing")]
pub(crate) fn decompress_and_deserialize<M: DeserializeOwned>(
    payload_compressed: &[u8],
    compression: Compression,
) -> Result<M> {
    let payload_uncompressed = decompress_payload(payload_compressed, compression)?;
    bincode::deserialize(&payload_uncompressed)
        .map_err(|e| AprenderError::Serialization(format!("Failed to deserialize model: {e}")))
}

/// Save a model to .apr format
///
/// # Arguments
/// * `model` - The model to save (must implement Serialize)
/// * `model_type` - Model type identifier
/// * `path` - Output file path
/// * `options` - Save options (compression, metadata)
///
/// # Errors
/// Returns error on I/O failure or serialization error
#[allow(clippy::needless_pass_by_value)] // SaveOptions is small and passed by value for ergonomics
pub fn save<M: Serialize>(
    model: &M,
    model_type: ModelType,
    path: impl AsRef<Path>,
    options: SaveOptions,
) -> Result<()> {
    let path = path.as_ref();

    // APR-POKA-001: Jidoka gate - refuse to write if validation explicitly failed
    // Score 0 means "validation rules exist but model failed all of them"
    if options.quality_score == Some(0) {
        return Err(AprenderError::ValidationError {
            message: "Jidoka: Refusing to save model with quality_score=0. \
                      Fix validation errors or use score=None to skip validation."
                .to_string(),
        });
    }

    // Serialize payload with bincode
    let payload_uncompressed = bincode::serialize(model)
        .map_err(|e| AprenderError::Serialization(format!("Failed to serialize model: {e}")))?;

    // Compress payload
    let (payload_compressed, compression) =
        compress_payload(&payload_uncompressed, options.compression)?;

    // Serialize metadata as MessagePack with named fields (spec 2)
    // Must use to_vec_named() for map mode to preserve field names with skip_serializing_if
    let metadata_bytes = rmp_serde::to_vec_named(&options.metadata)
        .map_err(|e| AprenderError::Serialization(format!("Failed to serialize metadata: {e}")))?;

    // Build header
    let mut header = Header::new(model_type);
    header.compression = compression;
    header.metadata_size = metadata_bytes.len() as u32;
    header.payload_size = payload_compressed.len() as u32;
    header.uncompressed_size = payload_uncompressed.len() as u32;

    // Set LICENSED flag if license info present (spec 9.1)
    if options.metadata.license.is_some() {
        header.flags = header.flags.with_licensed();
    }

    // APR-POKA-001: Set quality score in header (0 = no validation performed)
    header.quality_score = options.quality_score.unwrap_or(0);

    // Assemble file content (without checksum)
    let mut content = Vec::new();
    content.extend_from_slice(&header.to_bytes());
    content.extend_from_slice(&metadata_bytes);
    content.extend_from_slice(&payload_compressed);

    // Calculate and append checksum
    let checksum = crc32(&content);
    content.extend_from_slice(&checksum.to_le_bytes());

    // Write to file
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    writer.write_all(&content)?;
    writer.flush()?;

    Ok(())
}

/// Load a model from .apr format
///
/// # Arguments
/// * `path` - Input file path
/// * `expected_type` - Expected model type (for type safety)
///
/// # Errors
/// Returns error on I/O failure, format error, or type mismatch
pub fn load<M: DeserializeOwned>(path: impl AsRef<Path>, expected_type: ModelType) -> Result<M> {
    let path = path.as_ref();

    // Read entire file
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut content = Vec::new();
    reader.read_to_end(&mut content)?;

    // Verify minimum size
    if content.len() < HEADER_SIZE + 4 {
        return Err(AprenderError::FormatError {
            message: format!("File too small: {} bytes", content.len()),
        });
    }

    // Verify checksum (Jidoka: stop the line on corruption)
    let stored_checksum = u32::from_le_bytes([
        content[content.len() - 4],
        content[content.len() - 3],
        content[content.len() - 2],
        content[content.len() - 1],
    ]);
    let computed_checksum = crc32(&content[..content.len() - 4]);
    if stored_checksum != computed_checksum {
        return Err(AprenderError::ChecksumMismatch {
            expected: stored_checksum,
            actual: computed_checksum,
        });
    }

    // Parse header
    let header = Header::from_bytes(&content[..HEADER_SIZE])?;

    // Verify model type
    if header.model_type != expected_type {
        return Err(AprenderError::FormatError {
            message: format!(
                "Model type mismatch: file contains {:?}, expected {:?}",
                header.model_type, expected_type
            ),
        });
    }

    // Extract payload
    let metadata_end = HEADER_SIZE + header.metadata_size as usize;
    let payload_end = metadata_end + header.payload_size as usize;

    if payload_end > content.len() - 4 {
        return Err(AprenderError::FormatError {
            message: "Payload extends beyond file boundary".to_string(),
        });
    }

    let payload_compressed = &content[metadata_end..payload_end];

    // Decompress payload
    let payload_uncompressed = decompress_payload(payload_compressed, header.compression)?;

    // Deserialize model
    bincode::deserialize(&payload_uncompressed)
        .map_err(|e| AprenderError::Serialization(format!("Failed to deserialize model: {e}")))
}

/// Load a model from a byte slice (spec 1.1 - Single Binary Deployment)
///
/// Enables the `include_bytes!()` pattern for embedding models directly
/// in executables. This is the key function for zero-dependency ML deployment.
///
/// # Arguments
/// * `data` - Raw .apr file bytes (e.g., from `include_bytes!()`)
/// * `expected_type` - Expected model type (for type safety)
///
/// # Example
///
/// ```rust,ignore
/// use aprender::format::{load_from_bytes, ModelType};
///
/// // Embed model at compile time
/// const MODEL: &[u8] = include_bytes!("sentiment.apr");
///
/// fn main() -> Result<()> {
///     let model: LogisticRegression = load_from_bytes(MODEL, ModelType::LogisticRegression)?;
///     let prediction = model.predict(&input)?;
///     Ok(())
/// }
/// ```
///
/// # Errors
/// Returns error on format error, type mismatch, or checksum failure
pub fn load_from_bytes<M: DeserializeOwned>(data: &[u8], expected_type: ModelType) -> Result<M> {
    // Verify minimum size
    if data.len() < HEADER_SIZE + 4 {
        return Err(AprenderError::FormatError {
            message: format!("Data too small: {} bytes", data.len()),
        });
    }

    // Verify checksum (Jidoka: stop the line on corruption)
    let stored_checksum = u32::from_le_bytes([
        data[data.len() - 4],
        data[data.len() - 3],
        data[data.len() - 2],
        data[data.len() - 1],
    ]);
    let computed_checksum = crc32(&data[..data.len() - 4]);
    if stored_checksum != computed_checksum {
        return Err(AprenderError::ChecksumMismatch {
            expected: stored_checksum,
            actual: computed_checksum,
        });
    }

    // Parse header
    let header = Header::from_bytes(&data[..HEADER_SIZE])?;

    // Verify model type
    if header.model_type != expected_type {
        return Err(AprenderError::FormatError {
            message: format!(
                "Model type mismatch: data contains {:?}, expected {:?}",
                header.model_type, expected_type
            ),
        });
    }

    // Extract payload
    let metadata_end = HEADER_SIZE + header.metadata_size as usize;
    let payload_end = metadata_end + header.payload_size as usize;

    if payload_end > data.len() - 4 {
        return Err(AprenderError::FormatError {
            message: "Payload extends beyond data boundary".to_string(),
        });
    }

    let payload_compressed = &data[metadata_end..payload_end];

    // Decompress payload
    let payload_uncompressed = decompress_payload(payload_compressed, header.compression)?;

    // Deserialize model
    bincode::deserialize(&payload_uncompressed)
        .map_err(|e| AprenderError::Serialization(format!("Failed to deserialize model: {e}")))
}

/// Threshold for switching to mmap loading (1MB)
///
/// Files larger than this will use memory-mapped I/O for better performance.
/// Smaller files use standard read-to-heap which has lower overhead for small data.
pub const MMAP_THRESHOLD: u64 = 1024 * 1024;

include!("core_io_mmap.rs");
include!("test_model.rs");

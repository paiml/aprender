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

/// Load a model using memory-mapped I/O (zero-copy where possible)
///
/// Toyota Way Principle: *Muda* (Waste Elimination) - Eliminates redundant
/// data copies by mapping the file directly into the process address space.
///
/// # Performance
///
/// - Cold load: ~4x faster than standard `load()` for large models
/// - Memory: Uses ~1x file size vs ~2x for standard load
/// - Syscalls: Reduces `brk` calls from ~970 to ~50
///
/// # Safety
///
/// Uses OS-level memory mapping. The file must not be modified while loaded.
/// See `bundle-mmap-spec.md` Section 4 for safety considerations.
///
/// # Example
///
/// ```rust,ignore
/// use aprender::format::{load_mmap, ModelType};
///
/// // Load large model efficiently
/// let model: RandomForest = load_mmap("large_model.apr", ModelType::RandomForest)?;
/// ```
///
/// # Feature Flag
///
/// When `format-mmap` is enabled, uses real OS mmap via `memmap2`.
/// Otherwise, falls back to standard file I/O (same API, heap-allocated).
///
/// # Errors
///
/// Returns error on file not found, format error, type mismatch, or checksum failure
pub fn load_mmap<M: DeserializeOwned>(
    path: impl AsRef<Path>,
    expected_type: ModelType,
) -> Result<M> {
    use crate::bundle::MappedFile;

    let mapped = MappedFile::open(path.as_ref())?;

    load_from_bytes(mapped.as_slice(), expected_type)
}

/// Load a model with automatic strategy selection based on file size
///
/// Toyota Way Principle: *Heijunka* (Level Loading) - Chooses the optimal
/// loading strategy based on file size to balance memory and performance.
///
/// # Strategy
///
/// - Files <= 1MB: Standard `load()` (lower overhead for small files)
/// - Files > 1MB: Memory-mapped `load_mmap()` (better for large files)
///
/// # Example
///
/// ```rust,ignore
/// use aprender::format::{load_auto, ModelType};
///
/// // Automatically chooses best loading strategy
/// let model: KMeans = load_auto("model.apr", ModelType::KMeans)?;
/// ```
///
/// # Errors
///
/// Returns error on file not found, format error, type mismatch, or checksum failure
pub fn load_auto<M: DeserializeOwned>(
    path: impl AsRef<Path>,
    expected_type: ModelType,
) -> Result<M> {
    let metadata = std::fs::metadata(path.as_ref())?;

    if metadata.len() > MMAP_THRESHOLD {
        load_mmap(path, expected_type)
    } else {
        load(path, expected_type)
    }
}

// ============================================================================
// ENCRYPTION HELPER FUNCTIONS
// ============================================================================

/// Verify encrypted data has minimum required size.
#[cfg(feature = "format-encryption")]
pub(crate) fn verify_encrypted_data_size(data: &[u8]) -> Result<()> {
    if data.len() < HEADER_SIZE + SALT_SIZE + NONCE_SIZE + 4 {
        return Err(AprenderError::FormatError {
            message: format!("Data too small for encrypted model: {} bytes", data.len()),
        });
    }
    Ok(())
}

/// Verify encrypted data checksum.
#[cfg(feature = "format-encryption")]
pub(crate) fn verify_encrypted_checksum(data: &[u8]) -> Result<()> {
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
    Ok(())
}

/// Verify header has ENCRYPTED flag and correct model type.
#[cfg(feature = "format-encryption")]
pub(crate) fn verify_encrypted_header(header: &Header, expected_type: ModelType) -> Result<()> {
    if !header.flags.is_encrypted() {
        return Err(AprenderError::FormatError {
            message: "Data is not encrypted (ENCRYPTED flag not set)".to_string(),
        });
    }
    if header.model_type != expected_type {
        return Err(AprenderError::FormatError {
            message: format!(
                "Model type mismatch: data contains {:?}, expected {:?}",
                header.model_type, expected_type
            ),
        });
    }
    Ok(())
}

/// Extract salt, nonce, and ciphertext from encrypted data.
#[cfg(feature = "format-encryption")]
pub(crate) fn extract_encrypted_components<'a>(
    data: &'a [u8],
    header: &Header,
) -> Result<([u8; SALT_SIZE], [u8; NONCE_SIZE], &'a [u8])> {
    let metadata_end = HEADER_SIZE + header.metadata_size as usize;
    let salt_end = metadata_end + SALT_SIZE;
    let nonce_end = salt_end + NONCE_SIZE;
    let payload_end = metadata_end + header.payload_size as usize;

    if payload_end > data.len() - 4 {
        return Err(AprenderError::FormatError {
            message: "Encrypted payload extends beyond data boundary".to_string(),
        });
    }

    let salt: [u8; SALT_SIZE] =
        data[metadata_end..salt_end]
            .try_into()
            .map_err(|_| AprenderError::FormatError {
                message: "Invalid salt size".to_string(),
            })?;
    let nonce: [u8; NONCE_SIZE] =
        data[salt_end..nonce_end]
            .try_into()
            .map_err(|_| AprenderError::FormatError {
                message: "Invalid nonce size".to_string(),
            })?;
    let ciphertext = &data[nonce_end..payload_end];

    Ok((salt, nonce, ciphertext))
}

/// Decrypt payload using password and extracted components.
#[cfg(feature = "format-encryption")]
pub(crate) fn decrypt_encrypted_payload(
    password: &str,
    salt: &[u8; SALT_SIZE],
    nonce_bytes: &[u8; NONCE_SIZE],
    ciphertext: &[u8],
) -> Result<Vec<u8>> {
    use aes_gcm::{
        aead::{Aead, KeyInit},
        Aes256Gcm, Nonce,
    };
    use argon2::Argon2;

    let mut key = [0u8; KEY_SIZE];
    Argon2::default()
        .hash_password_into(password.as_bytes(), salt, &mut key)
        .map_err(|e| AprenderError::Other(format!("Key derivation failed: {e}")))?;

    let cipher = Aes256Gcm::new_from_slice(&key)
        .map_err(|e| AprenderError::Other(format!("Failed to create cipher: {e}")))?;
    let nonce = Nonce::from_slice(nonce_bytes);

    cipher
        .decrypt(nonce, ciphertext)
        .map_err(|_| AprenderError::DecryptionFailed {
            message: "Decryption failed (wrong password or corrupted data)".to_string(),
        })
}

/// Load an encrypted model from a byte slice (spec 1.1 + 4.1.2)
///
/// Enables the `include_bytes!()` pattern for embedding encrypted models.
/// Combines single binary deployment with password-based encryption.
///
/// # Arguments
/// * `data` - Raw encrypted .apr file bytes
/// * `expected_type` - Expected model type
/// * `password` - Password for decryption
///
/// # Example
///
/// ```rust,ignore
/// use aprender::format::{load_from_bytes_encrypted, ModelType};
///
/// // Embed encrypted model at compile time
/// const MODEL: &[u8] = include_bytes!("model.apr.enc");
///
/// fn main() -> Result<()> {
///     let model: NaiveBayes = load_from_bytes_encrypted(
///         MODEL,
///         ModelType::NaiveBayes,
///         &get_password_from_env(),
///     )?;
///     Ok(())
/// }
/// ```
///
/// # Errors
/// Returns error on format error, type mismatch, or decryption failure
#[cfg(feature = "format-encryption")]
pub fn load_from_bytes_encrypted<M: DeserializeOwned>(
    data: &[u8],
    expected_type: ModelType,
    password: &str,
) -> Result<M> {
    // Validate data integrity (Jidoka: stop the line on corruption)
    verify_encrypted_data_size(data)?;
    verify_encrypted_checksum(data)?;

    // Parse and verify header
    let header = Header::from_bytes(&data[..HEADER_SIZE])?;
    verify_encrypted_header(&header, expected_type)?;

    // Extract encryption components and decrypt
    let (salt, nonce, ciphertext) = extract_encrypted_components(data, &header)?;
    let payload_compressed = decrypt_encrypted_payload(password, &salt, &nonce, ciphertext)?;

    // Decompress and deserialize
    let payload_uncompressed = decompress_payload(&payload_compressed, header.compression)?;
    bincode::deserialize(&payload_uncompressed)
        .map_err(|e| AprenderError::Serialization(format!("Failed to deserialize model: {e}")))
}

/// Inspect model data without loading the payload (spec 1.1)
///
/// Useful for validating embedded models or checking metadata
/// without deserializing the full model.
///
/// # Arguments
/// * `data` - Raw .apr file bytes
///
/// # Errors
/// Returns error on format error
pub fn inspect_bytes(data: &[u8]) -> Result<ModelInfo> {
    // Verify minimum size
    if data.len() < HEADER_SIZE {
        return Err(AprenderError::FormatError {
            message: format!("Data too small: {} bytes", data.len()),
        });
    }

    // Parse header
    let header = Header::from_bytes(&data[..HEADER_SIZE])?;

    // Extract metadata
    let metadata_end = HEADER_SIZE + header.metadata_size as usize;
    if metadata_end > data.len() {
        return Err(AprenderError::FormatError {
            message: "Metadata extends beyond data boundary".to_string(),
        });
    }

    let metadata_bytes = &data[HEADER_SIZE..metadata_end];
    let metadata: Metadata = rmp_serde::from_slice(metadata_bytes)
        .map_err(|e| AprenderError::Serialization(format!("Failed to parse metadata: {e}")))?;

    Ok(ModelInfo {
        model_type: header.model_type,
        format_version: header.version,
        metadata,
        payload_size: header.payload_size as usize,
        uncompressed_size: header.uncompressed_size as usize,
        encrypted: header.flags.is_encrypted(),
        signed: header.flags.is_signed(),
        streaming: header.flags.is_streaming(),
        licensed: header.flags.is_licensed(),
        trueno_native: header.flags.is_trueno_native(),
        quantized: header.flags.is_quantized(),
        has_model_card: header.flags.has_model_card(),
    })
}

/// Inspect a model file without loading the payload
///
/// # Arguments
/// * `path` - Input file path
///
/// # Errors
/// Returns error on I/O failure or format error
pub fn inspect(path: impl AsRef<Path>) -> Result<ModelInfo> {
    let path = path.as_ref();

    // Read header + metadata only
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    // Read header
    let mut header_bytes = [0u8; HEADER_SIZE];
    reader.read_exact(&mut header_bytes)?;
    let header = Header::from_bytes(&header_bytes)?;

    // Read metadata (MessagePack per spec 2)
    let mut metadata_bytes = vec![0u8; header.metadata_size as usize];
    reader.read_exact(&mut metadata_bytes)?;
    let metadata: Metadata = rmp_serde::from_slice(&metadata_bytes)
        .map_err(|e| AprenderError::Serialization(format!("Failed to parse metadata: {e}")))?;

    Ok(ModelInfo {
        model_type: header.model_type,
        format_version: header.version,
        metadata,
        payload_size: header.payload_size as usize,
        uncompressed_size: header.uncompressed_size as usize,
        encrypted: header.flags.is_encrypted(),
        signed: header.flags.is_signed(),
        streaming: header.flags.is_streaming(),
        licensed: header.flags.is_licensed(),
        trueno_native: header.flags.is_trueno_native(),
        quantized: header.flags.is_quantized(),
        has_model_card: header.flags.has_model_card(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    // ============================================================================
    // CRC32 Tests
    // ============================================================================

    #[test]
    fn test_crc32_empty() {
        // CRC32 of empty data (IEEE polynomial)
        assert_eq!(crc32(&[]), 0x0000_0000);
    }

    #[test]
    fn test_crc32_known_values() {
        // "123456789" should give CRC32 = 0xCBF43926
        let data = b"123456789";
        assert_eq!(crc32(data), 0xCBF4_3926);
    }

    #[test]
    fn test_crc32_single_byte() {
        // Single byte values
        assert_eq!(crc32(&[0x00]), 0xD202_EF8D);
        assert_eq!(crc32(&[0xFF]), 0xFF00_0000);
    }

    #[test]
    fn test_crc32_multiple_bytes() {
        let data = b"Hello, World!";
        let crc = crc32(data);
        // Verify it's deterministic
        assert_eq!(crc, crc32(data));
        // Verify different data gives different CRC
        assert_ne!(crc, crc32(b"Hello, World"));
    }

    // ============================================================================
    // Compression Tests
    // ============================================================================

    #[test]
    fn test_compress_payload_none() {
        let data = b"test data for compression";
        let (compressed, compression) =
            compress_payload(data, Compression::None).expect("compress");
        assert_eq!(compression, Compression::None);
        assert_eq!(compressed, data);
    }

    #[test]
    fn test_decompress_payload_none() {
        let data = b"test data for decompression";
        let decompressed = decompress_payload(data, Compression::None).expect("decompress");
        assert_eq!(decompressed, data);
    }

    #[cfg(feature = "format-compression")]
    #[test]
    fn test_compress_decompress_zstd_default() {
        let data = b"test data that should compress well with zstd compression";
        let (compressed, compression) =
            compress_payload(data, Compression::ZstdDefault).expect("compress");
        assert_eq!(compression, Compression::ZstdDefault);
        let decompressed = decompress_payload(&compressed, compression).expect("decompress");
        assert_eq!(decompressed, data);
    }

    #[cfg(feature = "format-compression")]
    #[test]
    fn test_compress_decompress_lz4() {
        let data = b"test data for lz4 compression";
        let (compressed, compression) = compress_payload(data, Compression::Lz4).expect("compress");
        assert_eq!(compression, Compression::Lz4);
        let decompressed = decompress_payload(&compressed, compression).expect("decompress");
        assert_eq!(decompressed, data);
    }

    // ============================================================================
    // Save/Load Round-Trip Tests
    // ============================================================================

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    struct TestModel {
        name: String,
        values: Vec<f32>,
    }

    #[test]
    fn test_save_load_roundtrip() {
        let model = TestModel {
            name: "test_model".to_string(),
            values: vec![1.0, 2.0, 3.0, 4.0],
        };

        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("test.apr");

        let options = SaveOptions::default();
        save(&model, ModelType::LinearRegression, &path, options).expect("save");

        let loaded: TestModel = load(&path, ModelType::LinearRegression).expect("load");
        assert_eq!(model, loaded);
    }

    #[test]
    fn test_save_with_metadata() {
        let model = TestModel {
            name: "metadata_test".to_string(),
            values: vec![1.0],
        };

        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("test_metadata.apr");

        let mut metadata = Metadata::default();
        metadata.description = Some("A test model".to_string());

        let options = SaveOptions {
            metadata,
            compression: Compression::None,
            quality_score: Some(85),
        };
        save(&model, ModelType::LinearRegression, &path, options).expect("save");

        let info = inspect(&path).expect("inspect");
        assert_eq!(info.metadata.description, Some("A test model".to_string()));
    }

    #[test]
    fn test_save_rejects_quality_score_zero() {
        let model = TestModel {
            name: "bad_model".to_string(),
            values: vec![],
        };

        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("should_not_exist.apr");

        let options = SaveOptions {
            quality_score: Some(0),
            ..Default::default()
        };

        let result = save(&model, ModelType::LinearRegression, &path, options);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_wrong_model_type() {
        let model = TestModel {
            name: "type_test".to_string(),
            values: vec![1.0],
        };

        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("type_test.apr");

        save(
            &model,
            ModelType::LinearRegression,
            &path,
            SaveOptions::default(),
        )
        .expect("save");

        let result: Result<TestModel> = load(&path, ModelType::KMeans);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_nonexistent_file() {
        let result: Result<TestModel> =
            load("/nonexistent/path/model.apr", ModelType::LinearRegression);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_file_too_small() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("tiny.apr");

        std::fs::write(&path, &[0u8; 10]).expect("write tiny file");

        let result: Result<TestModel> = load(&path, ModelType::LinearRegression);
        assert!(result.is_err());
    }

    // ============================================================================
    // Inspect Tests
    // ============================================================================

    #[test]
    fn test_inspect_model() {
        let model = TestModel {
            name: "inspect_test".to_string(),
            values: vec![1.0, 2.0, 3.0],
        };

        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("inspect_test.apr");

        let mut metadata = Metadata::default();
        metadata.model_name = Some("Test Model".to_string());

        let options = SaveOptions {
            metadata,
            compression: Compression::None,
            quality_score: Some(90),
        };
        save(&model, ModelType::LinearRegression, &path, options).expect("save");

        let info = inspect(&path).expect("inspect");
        assert_eq!(info.model_type, ModelType::LinearRegression);
        assert_eq!(info.metadata.model_name, Some("Test Model".to_string()));
    }

    #[test]
    fn test_inspect_with_license_flag() {
        use super::super::{LicenseInfo, LicenseTier};

        let model = TestModel {
            name: "licensed".to_string(),
            values: vec![1.0],
        };

        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("licensed.apr");

        let mut metadata = Metadata::default();
        metadata.license = Some(LicenseInfo {
            uuid: "test-uuid".to_string(),
            hash: "test-hash".to_string(),
            expiry: None,
            seats: None,
            licensee: Some("Test User".to_string()),
            tier: LicenseTier::Enterprise,
        });

        let options = SaveOptions {
            metadata,
            compression: Compression::None,
            quality_score: None,
        };
        save(&model, ModelType::LinearRegression, &path, options).expect("save");

        let info = inspect(&path).expect("inspect");
        assert!(info.licensed);
    }

    #[test]
    fn test_inspect_bytes_valid() {
        let model = TestModel {
            name: "bytes_test".to_string(),
            values: vec![1.0],
        };

        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("bytes_test.apr");

        save(
            &model,
            ModelType::LinearRegression,
            &path,
            SaveOptions::default(),
        )
        .expect("save");

        let data = std::fs::read(&path).expect("read file");
        let info = inspect_bytes(&data).expect("inspect bytes");
        assert_eq!(info.model_type, ModelType::LinearRegression);
    }

    #[test]
    fn test_inspect_bytes_too_small() {
        let data = vec![0u8; 10];
        let result = inspect_bytes(&data);
        assert!(result.is_err());
    }
}

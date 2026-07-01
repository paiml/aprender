//! APR format core I/O — re-export seam over the sovereign `apr-format` leaf.
//!
//! Issue #2231: the v1 container save/load/inspect path moved to the `apr-format`
//! leaf crate. This module re-exports the leaf's public API so existing
//! `aprender::format::*` paths keep resolving, and keeps the framework-only bits
//! that genuinely depend on `aprender-core` (the `bundle::MappedFile`-backed
//! `load_mmap`, and the feature-gated encryption/signing helpers, which pull
//! `aes_gcm` / `argon2` / `ed25519_dalek`).

use super::{Compression, ModelInfo, ModelType, SaveOptions};
use crate::error::{AprenderError, Result};
use serde::de::DeserializeOwned;
use serde::Serialize;
use std::path::Path;

// Imports used only by the feature-gated signing/encryption helpers below.
#[cfg(any(feature = "format-signing", feature = "format-encryption"))]
use super::{Header, HEADER_SIZE};
#[cfg(any(feature = "format-signing", feature = "format-encryption"))]
use std::fs::File;
#[cfg(any(feature = "format-signing", feature = "format-encryption"))]
use std::io::{BufReader, Read};

// The leaf's `MMAP_THRESHOLD` re-exports byte-identically.
pub use apr_format::core_io::MMAP_THRESHOLD;

// Thin wrappers over the leaf's container I/O so the public signatures keep
// returning `aprender::Result<_>` (i.e. `AprenderError`), NOT the leaf's
// `apr_format::Result`. This preserves the EXACT pre-extraction API for every
// downstream caller (a bare `crate::format::save(...)` tail expression must
// still type-check as `Result<(), AprenderError>`); the `?` lifts the leaf
// error via `impl From<AprFormatError> for AprenderError` (issue #2231).

/// Save a model to `.apr` (v1 `APRN`) format. Delegates to the sovereign leaf.
///
/// # Errors
/// Returns an error on I/O failure, serialization error, or a refused quality gate.
#[allow(clippy::needless_pass_by_value)]
pub fn save<M: Serialize>(
    model: &M,
    model_type: ModelType,
    path: impl AsRef<Path>,
    options: SaveOptions,
) -> Result<()> {
    Ok(apr_format::core_io::save(model, model_type, path, options)?)
}

/// Load a model from a `.apr` (v1 `APRN`) file. Delegates to the sovereign leaf.
///
/// # Errors
/// Returns an error on I/O failure, format error, checksum failure, or type mismatch.
pub fn load<M: DeserializeOwned>(path: impl AsRef<Path>, expected_type: ModelType) -> Result<M> {
    Ok(apr_format::core_io::load(path, expected_type)?)
}

/// Load a model from a byte slice. Delegates to the sovereign leaf.
///
/// # Errors
/// Returns an error on format error, type mismatch, or checksum failure.
pub fn load_from_bytes<M: DeserializeOwned>(data: &[u8], expected_type: ModelType) -> Result<M> {
    Ok(apr_format::core_io::load_from_bytes(data, expected_type)?)
}

/// Load with automatic strategy selection based on file size. Delegates to the leaf.
///
/// # Errors
/// Returns an error on file-not-found, format error, type mismatch, or checksum failure.
pub fn load_auto<M: DeserializeOwned>(
    path: impl AsRef<Path>,
    expected_type: ModelType,
) -> Result<M> {
    Ok(apr_format::core_io::load_auto(path, expected_type)?)
}

/// Inspect model bytes without loading the payload. Delegates to the leaf.
///
/// # Errors
/// Returns an error on a too-small buffer, a bad header, or out-of-bounds metadata.
pub fn inspect_bytes(data: &[u8]) -> Result<ModelInfo> {
    Ok(apr_format::core_io::inspect_bytes(data)?)
}

/// Inspect a model file without loading the payload. Delegates to the leaf.
///
/// # Errors
/// Returns an error on I/O failure or a format error.
pub fn inspect(path: impl AsRef<Path>) -> Result<ModelInfo> {
    Ok(apr_format::core_io::inspect(path)?)
}

#[cfg(feature = "format-encryption")]
use super::{KEY_SIZE, NONCE_SIZE, SALT_SIZE};

// ============================================================================
// Feature-gated helpers retained in core for the signing / encryption modules.
// These pull `aes_gcm` / `argon2` / `ed25519_dalek` and so cannot live in the
// sovereign leaf. They are byte-identical to the pre-extraction core helpers.
// ============================================================================

/// Compress payload based on algorithm (spec §3.3).
///
/// Retained in core (not the leaf) because it switches on the optional
/// `format-compression` zstd/lz4 deps; used by signing/encryption and the
/// compression unit tests. Byte-identical to the pre-extraction core helper.
#[allow(clippy::unnecessary_wraps, dead_code)]
pub(crate) fn compress_payload(
    data: &[u8],
    compression: Compression,
) -> Result<(Vec<u8>, Compression)> {
    match compression {
        Compression::None => Ok((data.to_vec(), Compression::None)),
        #[cfg(feature = "format-compression")]
        Compression::ZstdDefault => {
            let compressed = zstd::encode_all(std::io::Cursor::new(data), 3).map_err(|e| {
                AprenderError::Serialization(format!("Zstd compression failed: {e}"))
            })?;
            Ok((compressed, Compression::ZstdDefault))
        }
        #[cfg(feature = "format-compression")]
        Compression::ZstdMax => {
            let compressed = zstd::encode_all(std::io::Cursor::new(data), 19).map_err(|e| {
                AprenderError::Serialization(format!("Zstd compression failed: {e}"))
            })?;
            Ok((compressed, Compression::ZstdMax))
        }
        #[cfg(not(feature = "format-compression"))]
        Compression::ZstdDefault | Compression::ZstdMax => Ok((data.to_vec(), Compression::None)),
        #[cfg(feature = "format-compression")]
        Compression::Lz4 => {
            let compressed = lz4_flex::compress_prepend_size(data);
            Ok((compressed, Compression::Lz4))
        }
        #[cfg(not(feature = "format-compression"))]
        Compression::Lz4 => Ok((data.to_vec(), Compression::None)),
    }
}

/// Decompress payload based on algorithm (spec §3.3).
///
/// Retained in core for the same reason as [`compress_payload`].
#[allow(dead_code)]
pub(crate) fn decompress_payload(data: &[u8], compression: Compression) -> Result<Vec<u8>> {
    match compression {
        Compression::None => Ok(data.to_vec()),
        #[cfg(feature = "format-compression")]
        Compression::ZstdDefault | Compression::ZstdMax => {
            zstd::decode_all(std::io::Cursor::new(data)).map_err(|e| {
                AprenderError::Serialization(format!("Zstd decompression failed: {e}"))
            })
        }
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
    let computed_checksum = apr_format::crc32(&content[..content.len() - 4]);
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
    if content.len() < HEADER_SIZE {
        return Err(AprenderError::FormatError {
            message: format!("File too small for header: {} bytes", content.len()),
        });
    }
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

/// Load a model using memory-mapped I/O via `aprender`'s `bundle::MappedFile`.
///
/// This is the framework-side `load_mmap` (distinct from the leaf's
/// `memmap2`-direct one): it uses `aprender-core`'s bundle abstraction. Existing
/// `aprender::format::load_mmap` callers resolve here.
///
/// # Errors
/// Returns an error on file-not-found, a format error, a type mismatch, or a
/// checksum failure.
pub fn load_mmap<M: DeserializeOwned>(
    path: impl AsRef<Path>,
    expected_type: ModelType,
) -> Result<M> {
    use crate::bundle::MappedFile;
    let mapped = MappedFile::open(path.as_ref())?;
    load_from_bytes(mapped.as_slice(), expected_type)
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
    let computed_checksum = apr_format::crc32(&data[..data.len() - 4]);
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

/// Load an encrypted model from a byte slice (spec §1.1 + §4.1.2).
///
/// # Errors
/// Returns an error on a format error, a type mismatch, or a decryption failure.
#[cfg(feature = "format-encryption")]
pub fn load_from_bytes_encrypted<M: DeserializeOwned>(
    data: &[u8],
    expected_type: ModelType,
    password: &str,
) -> Result<M> {
    verify_encrypted_data_size(data)?;
    verify_encrypted_checksum(data)?;

    let header = Header::from_bytes(&data[..HEADER_SIZE])?;
    verify_encrypted_header(&header, expected_type)?;

    let (salt, nonce, ciphertext) = extract_encrypted_components(data, &header)?;
    let payload_compressed = decrypt_encrypted_payload(password, &salt, &nonce, ciphertext)?;

    let payload_uncompressed = decompress_payload(&payload_compressed, header.compression)?;
    bincode::deserialize(&payload_uncompressed)
        .map_err(|e| AprenderError::Serialization(format!("Failed to deserialize model: {e}")))
}

include!("test_model.rs");

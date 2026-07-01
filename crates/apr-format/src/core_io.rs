//! v1 (`APRN`) core I/O — spike slice for issue #2231 Stage 1.
//!
//! Moved from `aprender-core/src/format/core_io.rs`, rewired to the sovereign
//! [`crate::error::AprFormatError`] and the deduplicated [`crate::crc32::crc32`].
//! Demonstrates the byte-identical save/load path with no dependency on
//! `aprender-core`.

use crate::crc32::crc32;
use crate::error::{AprFormatError, Result};
use crate::types::{Compression, Header, Metadata, ModelInfo, ModelType, SaveOptions, HEADER_SIZE};
use serde::{de::DeserializeOwned, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

/// Threshold for switching to mmap loading (1MB).
///
/// Files larger than this use memory-mapped I/O (when the `mmap` feature is on);
/// smaller files use standard read-to-heap (lower overhead for small data).
pub const MMAP_THRESHOLD: u64 = 1024 * 1024;

/// Compress payload based on algorithm (spec §3.3).
///
/// Without the `compression` feature, compressed variants fall back to `None`
/// (mirrors the legacy `aprender-core` behavior when `format-compression` is off).
#[allow(clippy::unnecessary_wraps)]
fn compress_payload(data: &[u8], compression: Compression) -> Result<(Vec<u8>, Compression)> {
    match compression {
        Compression::None => Ok((data.to_vec(), Compression::None)),
        #[cfg(feature = "compression")]
        Compression::ZstdDefault => {
            let compressed = zstd::encode_all(std::io::Cursor::new(data), 3).map_err(|e| {
                AprFormatError::Serialization(format!("Zstd compression failed: {e}"))
            })?;
            Ok((compressed, Compression::ZstdDefault))
        }
        #[cfg(feature = "compression")]
        Compression::ZstdMax => {
            let compressed = zstd::encode_all(std::io::Cursor::new(data), 19).map_err(|e| {
                AprFormatError::Serialization(format!("Zstd compression failed: {e}"))
            })?;
            Ok((compressed, Compression::ZstdMax))
        }
        #[cfg(not(feature = "compression"))]
        Compression::ZstdDefault | Compression::ZstdMax => Ok((data.to_vec(), Compression::None)),
        #[cfg(feature = "compression")]
        Compression::Lz4 => {
            let compressed = lz4_flex::compress_prepend_size(data);
            Ok((compressed, Compression::Lz4))
        }
        #[cfg(not(feature = "compression"))]
        Compression::Lz4 => Ok((data.to_vec(), Compression::None)),
    }
}

/// Decompress payload based on algorithm (spec §3.3).
fn decompress_payload(data: &[u8], compression: Compression) -> Result<Vec<u8>> {
    match compression {
        Compression::None => Ok(data.to_vec()),
        #[cfg(feature = "compression")]
        Compression::ZstdDefault | Compression::ZstdMax => {
            zstd::decode_all(std::io::Cursor::new(data)).map_err(|e| {
                AprFormatError::Serialization(format!("Zstd decompression failed: {e}"))
            })
        }
        #[cfg(not(feature = "compression"))]
        Compression::ZstdDefault | Compression::ZstdMax => Err(AprFormatError::FormatError {
            message: "Zstd compression not supported (enable `compression` feature)".to_string(),
        }),
        #[cfg(feature = "compression")]
        Compression::Lz4 => lz4_flex::decompress_size_prepended(data)
            .map_err(|e| AprFormatError::Serialization(format!("LZ4 decompression failed: {e}"))),
        #[cfg(not(feature = "compression"))]
        Compression::Lz4 => Err(AprFormatError::FormatError {
            message: "LZ4 compression not supported (enable `compression` feature)".to_string(),
        }),
    }
}

/// Save a model to `.apr` (v1 `APRN`) format.
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
    let path = path.as_ref();

    // APR-POKA-001: Jidoka gate — refuse to write if validation explicitly failed.
    if options.quality_score == Some(0) {
        return Err(AprFormatError::ValidationError {
            message: "Jidoka: Refusing to save model with quality_score=0. \
                      Fix validation errors or use score=None to skip validation."
                .to_string(),
        });
    }

    let payload_uncompressed = bincode::serialize(model)
        .map_err(|e| AprFormatError::Serialization(format!("Failed to serialize model: {e}")))?;

    let (payload_compressed, compression) =
        compress_payload(&payload_uncompressed, options.compression)?;

    let metadata_bytes = rmp_serde::to_vec_named(&options.metadata)
        .map_err(|e| AprFormatError::Serialization(format!("Failed to serialize metadata: {e}")))?;

    let mut header = Header::new(model_type);
    header.compression = compression;
    header.metadata_size = metadata_bytes.len() as u32;
    header.payload_size = payload_compressed.len() as u32;
    header.uncompressed_size = payload_uncompressed.len() as u32;

    if options.metadata.license.is_some() {
        header.flags = header.flags.with_licensed();
    }
    header.quality_score = options.quality_score.unwrap_or(0);

    let mut content = Vec::new();
    content.extend_from_slice(&header.to_bytes());
    content.extend_from_slice(&metadata_bytes);
    content.extend_from_slice(&payload_compressed);

    let checksum = crc32(&content);
    content.extend_from_slice(&checksum.to_le_bytes());

    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    writer.write_all(&content)?;
    writer.flush()?;
    Ok(())
}

/// Load a model from a `.apr` (v1 `APRN`) file.
///
/// # Errors
/// Returns an error on I/O failure, format error, checksum failure, or type mismatch.
pub fn load<M: DeserializeOwned>(path: impl AsRef<Path>, expected_type: ModelType) -> Result<M> {
    let file = File::open(path.as_ref())?;
    let mut reader = BufReader::new(file);
    let mut content = Vec::new();
    reader.read_to_end(&mut content)?;
    load_from_bytes(&content, expected_type)
}

/// Load a model from a byte slice (spec §1.1 — single-binary deployment).
///
/// Enables the `include_bytes!()` pattern for embedding models directly in
/// executables.
///
/// # Errors
/// Returns an error on format error, type mismatch, or checksum failure.
pub fn load_from_bytes<M: DeserializeOwned>(data: &[u8], expected_type: ModelType) -> Result<M> {
    if data.len() < HEADER_SIZE + 4 {
        return Err(AprFormatError::FormatError {
            message: format!("Data too small: {} bytes", data.len()),
        });
    }

    // Verify checksum (Jidoka: stop the line on corruption).
    let stored_checksum = u32::from_le_bytes([
        data[data.len() - 4],
        data[data.len() - 3],
        data[data.len() - 2],
        data[data.len() - 1],
    ]);
    let computed_checksum = crc32(&data[..data.len() - 4]);
    if stored_checksum != computed_checksum {
        return Err(AprFormatError::ChecksumMismatch {
            expected: stored_checksum,
            actual: computed_checksum,
        });
    }

    let header = Header::from_bytes(&data[..HEADER_SIZE])?;
    if header.model_type != expected_type {
        return Err(AprFormatError::FormatError {
            message: format!(
                "Model type mismatch: data contains {:?}, expected {:?}",
                header.model_type, expected_type
            ),
        });
    }

    let metadata_end = HEADER_SIZE + header.metadata_size as usize;
    let payload_end = metadata_end + header.payload_size as usize;
    if payload_end > data.len() - 4 {
        return Err(AprFormatError::InvalidOffset);
    }

    let payload_compressed = &data[metadata_end..payload_end];
    let payload_uncompressed = decompress_payload(payload_compressed, header.compression)?;

    bincode::deserialize(&payload_uncompressed)
        .map_err(|e| AprFormatError::Serialization(format!("Failed to deserialize model: {e}")))
}

/// Build a [`ModelInfo`] from a parsed header + decoded metadata.
fn model_info_from(header: &Header, metadata: Metadata) -> ModelInfo {
    ModelInfo {
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
    }
}

/// Inspect model data without loading the payload (spec §1.1).
///
/// Useful for validating embedded models or checking metadata without
/// deserializing the full model.
///
/// # Errors
/// Returns an error on a too-small buffer, a bad header, or metadata that
/// extends past the data boundary.
pub fn inspect_bytes(data: &[u8]) -> Result<ModelInfo> {
    if data.len() < HEADER_SIZE {
        return Err(AprFormatError::FormatError {
            message: format!("Data too small: {} bytes", data.len()),
        });
    }
    let header = Header::from_bytes(&data[..HEADER_SIZE])?;
    let metadata_end = HEADER_SIZE + header.metadata_size as usize;
    if metadata_end > data.len() {
        return Err(AprFormatError::FormatError {
            message: "Metadata extends beyond data boundary".to_string(),
        });
    }
    let metadata_bytes = &data[HEADER_SIZE..metadata_end];
    let metadata: Metadata = rmp_serde::from_slice(metadata_bytes)
        .map_err(|e| AprFormatError::Serialization(format!("Failed to parse metadata: {e}")))?;
    Ok(model_info_from(&header, metadata))
}

/// Inspect a model file without loading the payload.
///
/// # Errors
/// Returns an error on I/O failure or a format error.
pub fn inspect(path: impl AsRef<Path>) -> Result<ModelInfo> {
    let path = path.as_ref();
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    let mut header_bytes = [0u8; HEADER_SIZE];
    reader.read_exact(&mut header_bytes)?;
    let header = Header::from_bytes(&header_bytes)?;

    let mut metadata_bytes = vec![0u8; header.metadata_size as usize];
    reader.read_exact(&mut metadata_bytes)?;
    let metadata: Metadata = rmp_serde::from_slice(&metadata_bytes)
        .map_err(|e| AprFormatError::Serialization(format!("Failed to parse metadata: {e}")))?;

    Ok(model_info_from(&header, metadata))
}

/// Load a model using memory-mapped I/O (zero-copy where possible).
///
/// Maps the file directly into the address space (via `memmap2`) and parses
/// from the mapped slice, avoiding a read-to-heap copy. Falls back to standard
/// [`load`] when the `mmap` feature is disabled, preserving the same API.
///
/// # Safety
/// Uses OS-level memory mapping; the file must not be modified while loaded.
///
/// # Errors
/// Returns an error on file-not-found, a format error, a type mismatch, or a
/// checksum failure.
#[cfg(feature = "mmap")]
pub fn load_mmap<M: DeserializeOwned>(
    path: impl AsRef<Path>,
    expected_type: ModelType,
) -> Result<M> {
    let file = File::open(path.as_ref())?;
    // SAFETY: standard memmap2 usage; the caller upholds the no-concurrent-write
    // contract documented above (same precondition as the pre-extraction core).
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    load_from_bytes(&mmap, expected_type)
}

/// Load a model using memory-mapped I/O — `mmap`-feature-disabled fallback.
///
/// Without the `mmap` feature this delegates to the standard heap-backed
/// [`load`], keeping the public API identical.
///
/// # Errors
/// Returns an error on file-not-found, a format error, a type mismatch, or a
/// checksum failure.
#[cfg(not(feature = "mmap"))]
pub fn load_mmap<M: DeserializeOwned>(
    path: impl AsRef<Path>,
    expected_type: ModelType,
) -> Result<M> {
    load(path, expected_type)
}

/// Load a model with automatic strategy selection based on file size.
///
/// Files larger than [`MMAP_THRESHOLD`] use [`load_mmap`]; smaller files use
/// [`load`] (lower overhead for small files).
///
/// # Errors
/// Returns an error on file-not-found, a format error, a type mismatch, or a
/// checksum failure.
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

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
        save(
            &model,
            ModelType::LinearRegression,
            &path,
            SaveOptions::default(),
        )
        .expect("save");
        let loaded: TestModel = load(&path, ModelType::LinearRegression).expect("load");
        assert_eq!(model, loaded);
    }

    #[test]
    fn test_save_rejects_quality_score_zero() {
        let model = TestModel {
            name: "bad".to_string(),
            values: vec![],
        };
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("nope.apr");
        let options = SaveOptions {
            quality_score: Some(0),
            ..Default::default()
        };
        assert!(save(&model, ModelType::LinearRegression, &path, options).is_err());
    }

    #[test]
    fn test_load_wrong_model_type() {
        let model = TestModel {
            name: "t".to_string(),
            values: vec![1.0],
        };
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("t.apr");
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
    fn test_load_from_bytes_corrupted_checksum() {
        let model = TestModel {
            name: "c".to_string(),
            values: vec![1.0],
        };
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("c.apr");
        save(
            &model,
            ModelType::LinearRegression,
            &path,
            SaveOptions::default(),
        )
        .expect("save");
        let mut data = std::fs::read(&path).expect("read");
        data[HEADER_SIZE + 2] ^= 0xFF;
        let result: Result<TestModel> = load_from_bytes(&data, ModelType::LinearRegression);
        assert!(matches!(
            result,
            Err(AprFormatError::ChecksumMismatch { .. })
        ));
    }

    #[test]
    fn test_save_with_metadata_license_sets_flag() {
        use crate::types::{LicenseInfo, LicenseTier};
        let model = TestModel {
            name: "l".to_string(),
            values: vec![1.0],
        };
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("l.apr");
        let metadata = Metadata {
            license: Some(LicenseInfo {
                uuid: "u".to_string(),
                hash: "h".to_string(),
                expiry: None,
                seats: None,
                licensee: Some("X".to_string()),
                tier: LicenseTier::Enterprise,
            }),
            ..Metadata::default()
        };
        let options = SaveOptions {
            metadata,
            compression: Compression::None,
            quality_score: None,
        };
        save(&model, ModelType::LinearRegression, &path, options).expect("save");
        let data = std::fs::read(&path).expect("read");
        let header = Header::from_bytes(&data[..HEADER_SIZE]).expect("hdr");
        assert!(header.flags.is_licensed());
    }
}

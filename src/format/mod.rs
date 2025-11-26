//! Aprender Model Format (.apr)
//!
//! Binary format for ML model serialization with built-in quality (Jidoka):
//! - CRC32 checksum (integrity)
//! - Ed25519 signatures (provenance)
//! - AES-256-GCM encryption (confidentiality)
//! - Zstd compression (efficiency)
//! - Streaming/mmap (JIT loading)
//!
//! # Format Structure
//!
//! ```text
//! ┌─────────────────────────────────────────┐
//! │ Header (32 bytes, fixed)                │
//! ├─────────────────────────────────────────┤
//! │ Metadata (variable, MessagePack)        │
//! ├─────────────────────────────────────────┤
//! │ Chunk Index (if STREAMING flag)         │
//! ├─────────────────────────────────────────┤
//! │ Salt + Nonce (if ENCRYPTED flag)        │
//! ├─────────────────────────────────────────┤
//! │ Payload (variable, compressed)          │
//! ├─────────────────────────────────────────┤
//! │ Signature Block (if SIGNED flag)        │
//! ├─────────────────────────────────────────┤
//! │ Checksum (4 bytes, CRC32)               │
//! └─────────────────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use aprender::format::{save, load, ModelType, SaveOptions};
//! use aprender::linear_model::LinearRegression;
//!
//! let model = LinearRegression::new();
//! // ... train model ...
//!
//! // Save with compression
//! save(&model, ModelType::LinearRegression, "model.apr", SaveOptions::default())?;
//!
//! // Load with verification
//! let loaded: LinearRegression = load("model.apr", ModelType::LinearRegression)?;
//! ```

use crate::error::{AprenderError, Result};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
#[cfg(feature = "format-compression")]
use std::io::Cursor;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

/// Magic number: "APRN" in ASCII (0x4150524E)
pub const MAGIC: [u8; 4] = [0x41, 0x50, 0x52, 0x4E];

/// Current format version (1.0)
pub const FORMAT_VERSION: (u8, u8) = (1, 0);

/// Header size in bytes
pub const HEADER_SIZE: usize = 32;

/// Maximum uncompressed size (1GB safety limit)
pub const MAX_UNCOMPRESSED_SIZE: u32 = 1024 * 1024 * 1024;

/// Model type identifiers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u16)]
pub enum ModelType {
    /// Linear regression (OLS/Ridge/Lasso)
    LinearRegression = 0x0001,
    /// Logistic regression (GLM Binomial)
    LogisticRegression = 0x0002,
    /// Decision tree (CART/ID3)
    DecisionTree = 0x0003,
    /// Random forest (Bagging ensemble)
    RandomForest = 0x0004,
    /// Gradient boosting (Boosting ensemble)
    GradientBoosting = 0x0005,
    /// K-means clustering (Lloyd's algorithm)
    KMeans = 0x0006,
    /// Principal component analysis
    Pca = 0x0007,
    /// Gaussian naive bayes
    NaiveBayes = 0x0008,
    /// K-nearest neighbors
    Knn = 0x0009,
    /// Support vector machine
    Svm = 0x000A,
    /// N-gram language model (Markov chains)
    NgramLm = 0x0010,
    /// TF-IDF vectorizer
    Tfidf = 0x0011,
    /// Count vectorizer
    CountVectorizer = 0x0012,
    /// Sequential neural network (Feed-forward)
    NeuralSequential = 0x0020,
    /// Custom neural architecture
    NeuralCustom = 0x0021,
    /// Content-based recommender
    ContentRecommender = 0x0030,
    /// User-defined model
    Custom = 0x00FF,
}

impl ModelType {
    /// Convert from u16 value
    pub fn from_u16(value: u16) -> Option<Self> {
        match value {
            0x0001 => Some(Self::LinearRegression),
            0x0002 => Some(Self::LogisticRegression),
            0x0003 => Some(Self::DecisionTree),
            0x0004 => Some(Self::RandomForest),
            0x0005 => Some(Self::GradientBoosting),
            0x0006 => Some(Self::KMeans),
            0x0007 => Some(Self::Pca),
            0x0008 => Some(Self::NaiveBayes),
            0x0009 => Some(Self::Knn),
            0x000A => Some(Self::Svm),
            0x0010 => Some(Self::NgramLm),
            0x0011 => Some(Self::Tfidf),
            0x0012 => Some(Self::CountVectorizer),
            0x0020 => Some(Self::NeuralSequential),
            0x0021 => Some(Self::NeuralCustom),
            0x0030 => Some(Self::ContentRecommender),
            0x00FF => Some(Self::Custom),
            _ => None,
        }
    }
}

/// Compression algorithm
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum Compression {
    /// No compression (debugging/Genchi Genbutsu)
    None = 0x00,
    /// Zstd level 3 (default, good balance)
    #[default]
    ZstdDefault = 0x01,
    /// Zstd level 19 (maximum compression, archival)
    ZstdMax = 0x02,
    /// LZ4 (high-throughput streaming)
    Lz4 = 0x03,
}

impl Compression {
    /// Convert from u8 value
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0x00 => Some(Self::None),
            0x01 => Some(Self::ZstdDefault),
            0x02 => Some(Self::ZstdMax),
            0x03 => Some(Self::Lz4),
            _ => None,
        }
    }
}

/// Feature flags (bitmask) - spec §3.2
#[derive(Debug, Clone, Copy, Default)]
pub struct Flags(u8);

impl Flags {
    /// Payload is encrypted (AES-256-GCM)
    pub const ENCRYPTED: u8 = 0b0000_0001;
    /// Has digital signature (Ed25519)
    pub const SIGNED: u8 = 0b0000_0010;
    /// Supports chunked/streaming loading
    pub const STREAMING: u8 = 0b0000_0100;
    /// Has commercial license block
    pub const LICENSED: u8 = 0b0000_1000;
    /// 64-byte aligned tensors for zero-copy SIMD (trueno-native)
    pub const TRUENO_NATIVE: u8 = 0b0001_0000;

    /// Create new flags
    pub fn new() -> Self {
        Self(0)
    }

    /// Set encrypted flag
    pub fn with_encrypted(mut self) -> Self {
        self.0 |= Self::ENCRYPTED;
        self
    }

    /// Set signed flag
    pub fn with_signed(mut self) -> Self {
        self.0 |= Self::SIGNED;
        self
    }

    /// Set streaming flag
    pub fn with_streaming(mut self) -> Self {
        self.0 |= Self::STREAMING;
        self
    }

    /// Set licensed flag
    pub fn with_licensed(mut self) -> Self {
        self.0 |= Self::LICENSED;
        self
    }

    /// Set trueno-native flag
    pub fn with_trueno_native(mut self) -> Self {
        self.0 |= Self::TRUENO_NATIVE;
        self
    }

    /// Check if encrypted
    pub fn is_encrypted(self) -> bool {
        self.0 & Self::ENCRYPTED != 0
    }

    /// Check if signed
    pub fn is_signed(self) -> bool {
        self.0 & Self::SIGNED != 0
    }

    /// Check if streaming
    pub fn is_streaming(self) -> bool {
        self.0 & Self::STREAMING != 0
    }

    /// Check if licensed
    pub fn is_licensed(self) -> bool {
        self.0 & Self::LICENSED != 0
    }

    /// Check if trueno-native
    pub fn is_trueno_native(self) -> bool {
        self.0 & Self::TRUENO_NATIVE != 0
    }

    /// Get raw value
    pub fn bits(self) -> u8 {
        self.0
    }

    /// Create from raw value
    pub fn from_bits(bits: u8) -> Self {
        Self(bits & 0b0001_1111) // Mask reserved bits (5-7)
    }
}

/// File header (32 bytes)
#[derive(Debug, Clone)]
pub struct Header {
    /// Magic number (must be "APRN")
    pub magic: [u8; 4],
    /// Format version (major, minor)
    pub version: (u8, u8),
    /// Model type identifier
    pub model_type: ModelType,
    /// Metadata section size in bytes
    pub metadata_size: u32,
    /// Compressed payload size in bytes
    pub payload_size: u32,
    /// Uncompressed payload size (for allocation check)
    pub uncompressed_size: u32,
    /// Compression algorithm
    pub compression: Compression,
    /// Feature flags
    pub flags: Flags,
}

impl Header {
    /// Create a new header
    pub fn new(model_type: ModelType) -> Self {
        Self {
            magic: MAGIC,
            version: FORMAT_VERSION,
            model_type,
            metadata_size: 0,
            payload_size: 0,
            uncompressed_size: 0,
            compression: Compression::default(),
            flags: Flags::default(),
        }
    }

    /// Serialize header to bytes (32 bytes)
    pub fn to_bytes(&self) -> [u8; HEADER_SIZE] {
        let mut bytes = [0u8; HEADER_SIZE];

        // Magic (0-3)
        bytes[0..4].copy_from_slice(&self.magic);

        // Format version (4-5)
        bytes[4] = self.version.0;
        bytes[5] = self.version.1;

        // Model type (6-7, little-endian)
        let model_type = self.model_type as u16;
        bytes[6..8].copy_from_slice(&model_type.to_le_bytes());

        // Metadata size (8-11, little-endian)
        bytes[8..12].copy_from_slice(&self.metadata_size.to_le_bytes());

        // Payload size (12-15, little-endian)
        bytes[12..16].copy_from_slice(&self.payload_size.to_le_bytes());

        // Uncompressed size (16-19, little-endian)
        bytes[16..20].copy_from_slice(&self.uncompressed_size.to_le_bytes());

        // Compression (20)
        bytes[20] = self.compression as u8;

        // Flags (21)
        bytes[21] = self.flags.bits();

        // Reserved (22-31) - already zero

        bytes
    }

    /// Parse header from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < HEADER_SIZE {
            return Err(AprenderError::FormatError {
                message: format!(
                    "Header too short: {} bytes, expected {}",
                    bytes.len(),
                    HEADER_SIZE
                ),
            });
        }

        // Validate magic
        let magic: [u8; 4] = bytes[0..4].try_into().expect("slice length is 4");
        if magic != MAGIC {
            return Err(AprenderError::FormatError {
                message: format!(
                    "Invalid magic number: {:02X}{:02X}{:02X}{:02X}, expected APRN",
                    magic[0], magic[1], magic[2], magic[3]
                ),
            });
        }

        // Parse version
        let version = (bytes[4], bytes[5]);
        if version.0 > FORMAT_VERSION.0 {
            return Err(AprenderError::UnsupportedVersion {
                found: version,
                supported: FORMAT_VERSION,
            });
        }

        // Parse model type
        let model_type_raw = u16::from_le_bytes([bytes[6], bytes[7]]);
        let model_type =
            ModelType::from_u16(model_type_raw).ok_or_else(|| AprenderError::FormatError {
                message: format!("Unknown model type: 0x{model_type_raw:04X}"),
            })?;

        // Parse sizes
        let metadata_size = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        let payload_size = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
        let uncompressed_size = u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);

        // Safety check: prevent compression bombs
        if uncompressed_size > MAX_UNCOMPRESSED_SIZE {
            return Err(AprenderError::FormatError {
                message: format!(
                    "Uncompressed size {uncompressed_size} exceeds maximum {MAX_UNCOMPRESSED_SIZE} (compression bomb protection)"
                ),
            });
        }

        // Parse compression
        let compression =
            Compression::from_u8(bytes[20]).ok_or_else(|| AprenderError::FormatError {
                message: format!("Unknown compression algorithm: 0x{:02X}", bytes[20]),
            })?;

        // Parse flags
        let flags = Flags::from_bits(bytes[21]);

        Ok(Self {
            magic,
            version,
            model_type,
            metadata_size,
            payload_size,
            uncompressed_size,
            compression,
            flags,
        })
    }
}

/// Model metadata (MessagePack-encoded)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metadata {
    /// Creation timestamp (ISO 8601)
    pub created_at: String,
    /// Aprender version that created this model
    pub aprender_version: String,
    /// Optional model name
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_name: Option<String>,
    /// Optional description
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Training information
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub training: Option<TrainingInfo>,
    /// Hyperparameters
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub hyperparameters: HashMap<String, serde_json::Value>,
    /// Model metrics
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub metrics: HashMap<String, serde_json::Value>,
    /// Custom user data
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub custom: HashMap<String, serde_json::Value>,
}

/// Training information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingInfo {
    /// Number of training samples
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub samples: Option<usize>,
    /// Training duration in milliseconds
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>,
    /// Data source description
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
}

impl Default for Metadata {
    fn default() -> Self {
        Self {
            created_at: chrono_lite_now(),
            aprender_version: env!("CARGO_PKG_VERSION").to_string(),
            model_name: None,
            description: None,
            training: None,
            hyperparameters: HashMap::new(),
            metrics: HashMap::new(),
            custom: HashMap::new(),
        }
    }
}

/// Simple ISO 8601 timestamp (no chrono dependency)
fn chrono_lite_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = duration.as_secs();
    // Convert to rough ISO 8601 (good enough for metadata)
    format!("{secs}")
}

/// Options for saving models
#[derive(Debug, Clone, Default)]
pub struct SaveOptions {
    /// Compression algorithm
    pub compression: Compression,
    /// Additional metadata
    pub metadata: Metadata,
}

impl SaveOptions {
    /// Create with default compression
    pub fn new() -> Self {
        Self::default()
    }

    /// Set compression algorithm
    pub fn with_compression(mut self, compression: Compression) -> Self {
        self.compression = compression;
        self
    }

    /// Set model name
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.metadata.model_name = Some(name.into());
        self
    }

    /// Set description
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.metadata.description = Some(desc.into());
        self
    }
}

/// Model information (from inspection)
#[derive(Debug, Clone)]
#[allow(clippy::struct_excessive_bools)] // Bools represent independent flag states
pub struct ModelInfo {
    /// Model type
    pub model_type: ModelType,
    /// Format version
    pub format_version: (u8, u8),
    /// Metadata
    pub metadata: Metadata,
    /// Compressed payload size
    pub payload_size: usize,
    /// Uncompressed payload size
    pub uncompressed_size: usize,
    /// Is encrypted
    pub encrypted: bool,
    /// Is signed
    pub signed: bool,
    /// Is streaming
    pub streaming: bool,
    /// Has commercial license block
    pub licensed: bool,
    /// Uses trueno-native 64-byte aligned tensors
    pub trueno_native: bool,
}

/// Compress payload based on algorithm (spec §3.3)
#[allow(clippy::unnecessary_wraps)] // Returns Result to handle compression errors when feature enabled
fn compress_payload(data: &[u8], compression: Compression) -> Result<(Vec<u8>, Compression)> {
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
        Compression::Lz4 => {
            // LZ4 not yet implemented, fall back to no compression
            Ok((data.to_vec(), Compression::None))
        }
    }
}

/// Decompress payload based on algorithm (spec §3.3)
fn decompress_payload(data: &[u8], compression: Compression) -> Result<Vec<u8>> {
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
        Compression::Lz4 => Err(AprenderError::FormatError {
            message: "LZ4 compression not yet implemented".to_string(),
        }),
    }
}

/// CRC32 checksum (IEEE polynomial)
fn crc32(data: &[u8]) -> u32 {
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

    // Serialize payload with bincode
    let payload_uncompressed = bincode::serialize(model)
        .map_err(|e| AprenderError::Serialization(format!("Failed to serialize model: {e}")))?;

    // Compress payload
    let (payload_compressed, compression) =
        compress_payload(&payload_uncompressed, options.compression)?;

    // Serialize metadata as MessagePack (spec §2)
    let metadata_bytes = rmp_serde::to_vec(&options.metadata)
        .map_err(|e| AprenderError::Serialization(format!("Failed to serialize metadata: {e}")))?;

    // Build header
    let mut header = Header::new(model_type);
    header.compression = compression;
    header.metadata_size = metadata_bytes.len() as u32;
    header.payload_size = payload_compressed.len() as u32;
    header.uncompressed_size = payload_uncompressed.len() as u32;

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

    // Read metadata (MessagePack per spec §2)
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
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_magic_number() {
        assert_eq!(MAGIC, [0x41, 0x50, 0x52, 0x4E]);
        assert_eq!(&MAGIC, b"APRN");
    }

    #[test]
    fn test_header_roundtrip() {
        let mut header = Header::new(ModelType::LinearRegression);
        header.metadata_size = 256;
        header.payload_size = 1024;
        header.uncompressed_size = 2048;
        header.compression = Compression::ZstdDefault;
        header.flags = Flags::new().with_signed();

        let bytes = header.to_bytes();
        assert_eq!(bytes.len(), HEADER_SIZE);

        let parsed = Header::from_bytes(&bytes).expect("valid header");
        assert_eq!(parsed.magic, MAGIC);
        assert_eq!(parsed.version, FORMAT_VERSION);
        assert_eq!(parsed.model_type, ModelType::LinearRegression);
        assert_eq!(parsed.metadata_size, 256);
        assert_eq!(parsed.payload_size, 1024);
        assert_eq!(parsed.uncompressed_size, 2048);
        assert_eq!(parsed.compression, Compression::ZstdDefault);
        assert!(parsed.flags.is_signed());
        assert!(!parsed.flags.is_encrypted());
    }

    #[test]
    fn test_invalid_magic() {
        let mut bytes = [0u8; HEADER_SIZE];
        bytes[0..4].copy_from_slice(b"BAAD");

        let result = Header::from_bytes(&bytes);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Invalid magic"));
    }

    #[test]
    fn test_unsupported_version() {
        let mut header = Header::new(ModelType::LinearRegression);
        header.version = (99, 0); // Future version

        let mut bytes = header.to_bytes();
        bytes[4] = 99; // Major version

        let result = Header::from_bytes(&bytes);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Unsupported"));
    }

    #[test]
    fn test_compression_bomb_protection() {
        let mut header = Header::new(ModelType::LinearRegression);
        header.uncompressed_size = MAX_UNCOMPRESSED_SIZE + 1;

        let bytes = header.to_bytes();
        let result = Header::from_bytes(&bytes);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("compression bomb"));
    }

    #[test]
    fn test_crc32() {
        // Known CRC32 value for "123456789"
        let data = b"123456789";
        let crc = crc32(data);
        assert_eq!(crc, 0xCBF43926);
    }

    #[test]
    fn test_flags() {
        let flags = Flags::new()
            .with_encrypted()
            .with_signed()
            .with_streaming()
            .with_licensed()
            .with_trueno_native();

        assert!(flags.is_encrypted());
        assert!(flags.is_signed());
        assert!(flags.is_streaming());
        assert!(flags.is_licensed());
        assert!(flags.is_trueno_native());
        assert_eq!(flags.bits(), 0b0001_1111);
    }

    #[test]
    fn test_model_type_roundtrip() {
        let types = [
            ModelType::LinearRegression,
            ModelType::LogisticRegression,
            ModelType::DecisionTree,
            ModelType::RandomForest,
            ModelType::KMeans,
            ModelType::NeuralSequential,
            ModelType::Custom,
        ];

        for model_type in types {
            let value = model_type as u16;
            let parsed = ModelType::from_u16(value).expect("valid type");
            assert_eq!(parsed, model_type);
        }
    }

    #[test]
    fn test_save_load_simple() {
        use tempfile::tempdir;

        // Simple serializable struct for testing
        #[derive(Debug, Serialize, Deserialize, PartialEq)]
        struct TestModel {
            weights: Vec<f32>,
            bias: f32,
        }

        let model = TestModel {
            weights: vec![1.0, 2.0, 3.0],
            bias: 0.5,
        };

        let dir = tempdir().expect("create temp dir");
        let path = dir.path().join("test.apr");

        // Save
        save(&model, ModelType::Custom, &path, SaveOptions::default())
            .expect("save should succeed");

        // Load
        let loaded: TestModel = load(&path, ModelType::Custom).expect("load should succeed");
        assert_eq!(loaded, model);

        // Inspect
        let info = inspect(&path).expect("inspect should succeed");
        assert_eq!(info.model_type, ModelType::Custom);
        assert_eq!(info.format_version, FORMAT_VERSION);
        assert!(!info.encrypted);
        assert!(!info.signed);
    }

    #[test]
    fn test_checksum_corruption() {
        use tempfile::tempdir;

        #[derive(Debug, Serialize, Deserialize)]
        struct TestModel {
            value: i32,
        }

        let model = TestModel { value: 42 };
        let dir = tempdir().expect("create temp dir");
        let path = dir.path().join("corrupt.apr");

        save(&model, ModelType::Custom, &path, SaveOptions::default())
            .expect("save should succeed");

        // Corrupt the file
        let mut content = std::fs::read(&path).expect("read file");
        if content.len() > HEADER_SIZE + 10 {
            content[HEADER_SIZE + 5] ^= 0xFF; // Flip some bits in metadata
        }
        std::fs::write(&path, &content).expect("write corrupted file");

        // Load should fail with checksum error
        let result: Result<TestModel> = load(&path, ModelType::Custom);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Checksum"));
    }

    #[test]
    fn test_type_mismatch() {
        use tempfile::tempdir;

        #[derive(Debug, Serialize, Deserialize)]
        struct TestModel {
            value: i32,
        }

        let model = TestModel { value: 42 };
        let dir = tempdir().expect("create temp dir");
        let path = dir.path().join("typed.apr");

        save(
            &model,
            ModelType::LinearRegression,
            &path,
            SaveOptions::default(),
        )
        .expect("save should succeed");

        // Load with wrong type should fail
        let result: Result<TestModel> = load(&path, ModelType::KMeans);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("type mismatch"));
    }

    #[test]
    #[cfg(feature = "format-compression")]
    fn test_zstd_compression_roundtrip() {
        use tempfile::tempdir;

        // Model with repetitive data (compresses well)
        #[derive(Debug, Serialize, Deserialize, PartialEq)]
        struct LargeModel {
            weights: Vec<f32>,
        }

        // 10,000 floats with repetitive pattern (compresses well)
        let model = LargeModel {
            weights: (0..10_000).map(|i| (i % 100) as f32).collect(),
        };

        let dir = tempdir().expect("create temp dir");
        let path = dir.path().join("compressed.apr");

        // Save with default zstd compression
        save(
            &model,
            ModelType::Custom,
            &path,
            SaveOptions::default().with_compression(Compression::ZstdDefault),
        )
        .expect("save should succeed");

        // Load and verify
        let loaded: LargeModel = load(&path, ModelType::Custom).expect("load should succeed");
        assert_eq!(loaded, model);

        // Verify compression reduced size
        let info = inspect(&path).expect("inspect should succeed");
        assert!(
            info.payload_size < info.uncompressed_size,
            "Compressed size {} should be less than uncompressed {}",
            info.payload_size,
            info.uncompressed_size
        );
    }

    #[test]
    #[cfg(feature = "format-compression")]
    fn test_zstd_max_compression() {
        use tempfile::tempdir;

        #[derive(Debug, Serialize, Deserialize, PartialEq)]
        struct TestModel {
            data: Vec<u8>,
        }

        // Highly compressible data (all zeros)
        let model = TestModel {
            data: vec![0u8; 50_000],
        };

        let dir = tempdir().expect("create temp dir");
        let path_default = dir.path().join("default.apr");
        let path_max = dir.path().join("max.apr");

        // Save with default compression
        save(
            &model,
            ModelType::Custom,
            &path_default,
            SaveOptions::default().with_compression(Compression::ZstdDefault),
        )
        .expect("save default should succeed");

        // Save with maximum compression
        save(
            &model,
            ModelType::Custom,
            &path_max,
            SaveOptions::default().with_compression(Compression::ZstdMax),
        )
        .expect("save max should succeed");

        // Both should load correctly
        let loaded_default: TestModel =
            load(&path_default, ModelType::Custom).expect("load default should succeed");
        let loaded_max: TestModel =
            load(&path_max, ModelType::Custom).expect("load max should succeed");

        assert_eq!(loaded_default, model);
        assert_eq!(loaded_max, model);

        // Max compression should be at least as small (often smaller)
        let info_default = inspect(&path_default).expect("inspect default");
        let info_max = inspect(&path_max).expect("inspect max");
        assert!(
            info_max.payload_size <= info_default.payload_size,
            "Max compression {} should be <= default {}",
            info_max.payload_size,
            info_default.payload_size
        );
    }

    #[test]
    fn test_compression_fallback_without_feature() {
        // When feature is disabled, zstd requests should fall back to None
        let data = vec![1u8, 2, 3, 4, 5];
        let (compressed, actual_compression) =
            compress_payload(&data, Compression::ZstdDefault).expect("should fallback");

        #[cfg(not(feature = "format-compression"))]
        {
            assert_eq!(actual_compression, Compression::None);
            assert_eq!(compressed, data);
        }

        #[cfg(feature = "format-compression")]
        {
            assert_eq!(actual_compression, Compression::ZstdDefault);
            // Compressed data will be different (has zstd header)
            assert_ne!(compressed, data);
        }
    }
}

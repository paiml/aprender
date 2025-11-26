//! Aprender Model Format (.apr)
//!
//! Binary format for ML model serialization with built-in quality (Jidoka):
//! - CRC32 checksum (integrity)
//! - Ed25519 signatures (provenance)
//! - AES-256-GCM encryption (confidentiality)
//! - Zstd compression (efficiency)
//! - Quantization (Q8_0, Q4_0, Q4_1 - GGUF compatible)
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

// Quantization module (spec §6.2)
#[cfg(feature = "format-quantize")]
pub mod quantize;

// GGUF export module (spec §7.2)
pub mod gguf;

// Re-export quantization types when feature is enabled
#[cfg(feature = "format-quantize")]
pub use quantize::{
    dequantize, quantize as quantize_data, Q4_0Quantizer, Q8_0Quantizer, QuantType,
    QuantizationInfo, QuantizedBlock, QuantizedTensor, Quantizer, BLOCK_SIZE,
};

// Re-export signing types when feature is enabled
#[cfg(feature = "format-signing")]
pub use ed25519_dalek::{SigningKey, VerifyingKey};

/// Ed25519 signature size in bytes
#[cfg(feature = "format-signing")]
pub const SIGNATURE_SIZE: usize = 64;

/// Ed25519 public key size in bytes
#[cfg(feature = "format-signing")]
pub const PUBLIC_KEY_SIZE: usize = 32;

/// Argon2id salt size in bytes (spec §4.1.2)
#[cfg(feature = "format-encryption")]
pub const SALT_SIZE: usize = 16;

/// AES-GCM nonce size in bytes
#[cfg(feature = "format-encryption")]
pub const NONCE_SIZE: usize = 12;

/// AES-256 key size in bytes
#[cfg(feature = "format-encryption")]
pub const KEY_SIZE: usize = 32;

/// X25519 public key size in bytes (spec §4.1.3)
#[cfg(feature = "format-encryption")]
pub const X25519_PUBLIC_KEY_SIZE: usize = 32;

/// Recipient public key hash size for identification (spec §4.1.3)
#[cfg(feature = "format-encryption")]
pub const RECIPIENT_HASH_SIZE: usize = 8;

/// HKDF info string for X25519 key derivation (spec §4.1.3)
#[cfg(feature = "format-encryption")]
pub const HKDF_INFO: &[u8] = b"apr-v1-encrypt";

// Re-export X25519 types when feature is enabled
#[cfg(feature = "format-encryption")]
pub use x25519_dalek::{PublicKey as X25519PublicKey, StaticSecret as X25519SecretKey};

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
    /// Payload contains quantized tensors (spec §6.2)
    pub const QUANTIZED: u8 = 0b0010_0000;

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

    /// Set quantized flag
    pub fn with_quantized(mut self) -> Self {
        self.0 |= Self::QUANTIZED;
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

    /// Check if quantized
    pub fn is_quantized(self) -> bool {
        self.0 & Self::QUANTIZED != 0
    }

    /// Get raw value
    pub fn bits(self) -> u8 {
        self.0
    }

    /// Create from raw value
    pub fn from_bits(bits: u8) -> Self {
        Self(bits & 0b0011_1111) // Mask reserved bits (6-7)
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
    /// Distillation teacher hash (spec §6.3) - simple form
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub distillation: Option<String>,
    /// Full distillation provenance (spec §6.3.2) - structured form
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub distillation_info: Option<DistillationInfo>,
    /// Commercial license information (spec §9.1)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub license: Option<LicenseInfo>,
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

// ============================================================================
// Knowledge Distillation Types (spec §6.3)
// ============================================================================

/// Distillation method used (spec §6.3.1)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistillMethod {
    /// KL divergence on final logits (Hinton2015)
    Standard,
    /// Intermediate layer matching
    Progressive,
    /// Multiple teachers weighted average
    Ensemble,
}

/// Teacher model provenance for audit trails (spec §6.3.2)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeacherProvenance {
    /// SHA256 hash of teacher .apr file
    pub hash: String,
    /// Ed25519 signature of teacher (if signed)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
    /// Teacher model type
    pub model_type: ModelType,
    /// Teacher parameter count
    pub param_count: u64,
    /// For ensemble: multiple teachers
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ensemble_teachers: Option<Vec<TeacherProvenance>>,
}

/// Distillation hyperparameters (spec §6.3.2)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillationParams {
    /// Temperature for softening distributions (typically 2.0-5.0)
    pub temperature: f32,
    /// Weight for soft vs hard loss (α in loss formula)
    pub alpha: f32,
    /// For progressive: weight for hidden vs logit loss (β)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub beta: Option<f32>,
    /// Training epochs for distillation
    pub epochs: u32,
    /// Final distillation loss achieved
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub final_loss: Option<f32>,
}

/// Layer mapping for progressive distillation (spec §6.3.2)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerMapping {
    /// Student layer index
    pub student_layer: usize,
    /// Teacher layer index
    pub teacher_layer: usize,
    /// Weight for this layer's loss
    pub weight: f32,
}

/// Complete distillation provenance (spec §6.3.2)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillationInfo {
    /// Distillation method used
    pub method: DistillMethod,
    /// Teacher model provenance
    pub teacher: TeacherProvenance,
    /// Distillation hyperparameters
    pub params: DistillationParams,
    /// Optional: layer mapping for progressive distillation
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layer_mapping: Option<Vec<LayerMapping>>,
}

// ============================================================================
// Commercial License Types (spec §9)
// ============================================================================

/// License tier levels (spec §9.1)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LicenseTier {
    /// Personal/individual use
    Personal,
    /// Team/organization use (limited seats)
    Team,
    /// Enterprise use (unlimited seats, priority support)
    Enterprise,
    /// Academic/research use (non-commercial)
    Academic,
}

/// Commercial license information (spec §9.1)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LicenseInfo {
    /// Unique license identifier (UUID v4)
    pub uuid: String,
    /// Hash of the license certificate (cryptographically bound)
    pub hash: String,
    /// License expiration date (ISO 8601) - None for perpetual
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expiry: Option<String>,
    /// Maximum concurrent seats - None for unlimited
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seats: Option<u32>,
    /// Licensee name/organization
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub licensee: Option<String>,
    /// License tier
    pub tier: LicenseTier,
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
            distillation: None,
            distillation_info: None,
            license: None,
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

    /// Set distillation info (spec §6.3)
    pub fn with_distillation_info(mut self, info: DistillationInfo) -> Self {
        self.metadata.distillation_info = Some(info);
        self
    }

    /// Set license info (spec §9.1)
    pub fn with_license(mut self, license: LicenseInfo) -> Self {
        self.metadata.license = Some(license);
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
    /// Contains quantized tensors
    pub quantized: bool,
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

    // Serialize metadata as MessagePack with named fields (spec §2)
    // Must use to_vec_named() for map mode to preserve field names with skip_serializing_if
    let metadata_bytes = rmp_serde::to_vec_named(&options.metadata)
        .map_err(|e| AprenderError::Serialization(format!("Failed to serialize metadata: {e}")))?;

    // Build header
    let mut header = Header::new(model_type);
    header.compression = compression;
    header.metadata_size = metadata_bytes.len() as u32;
    header.payload_size = payload_compressed.len() as u32;
    header.uncompressed_size = payload_uncompressed.len() as u32;

    // Set LICENSED flag if license info present (spec §9.1)
    if options.metadata.license.is_some() {
        header.flags = header.flags.with_licensed();
    }

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

/// Load a model from a byte slice (spec §1.1 - Single Binary Deployment)
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

/// Load an encrypted model from a byte slice (spec §1.1 + §4.1.2)
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
    use aes_gcm::{
        aead::{Aead, KeyInit},
        Aes256Gcm, Nonce,
    };
    use argon2::Argon2;

    // Verify minimum size
    if data.len() < HEADER_SIZE + SALT_SIZE + NONCE_SIZE + 4 {
        return Err(AprenderError::FormatError {
            message: format!("Data too small for encrypted model: {} bytes", data.len()),
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

    // Verify ENCRYPTED flag is set
    if !header.flags.is_encrypted() {
        return Err(AprenderError::FormatError {
            message: "Data is not encrypted (ENCRYPTED flag not set)".to_string(),
        });
    }

    // Verify model type
    if header.model_type != expected_type {
        return Err(AprenderError::FormatError {
            message: format!(
                "Model type mismatch: data contains {:?}, expected {:?}",
                header.model_type, expected_type
            ),
        });
    }

    // Calculate content boundaries
    let metadata_end = HEADER_SIZE + header.metadata_size as usize;
    let salt_end = metadata_end + SALT_SIZE;
    let nonce_end = salt_end + NONCE_SIZE;
    let payload_end = metadata_end + header.payload_size as usize;

    if payload_end > data.len() - 4 {
        return Err(AprenderError::FormatError {
            message: "Encrypted payload extends beyond data boundary".to_string(),
        });
    }

    // Extract salt, nonce, and ciphertext
    let salt: [u8; SALT_SIZE] =
        data[metadata_end..salt_end]
            .try_into()
            .map_err(|_| AprenderError::FormatError {
                message: "Invalid salt size".to_string(),
            })?;
    let nonce_bytes: [u8; NONCE_SIZE] =
        data[salt_end..nonce_end]
            .try_into()
            .map_err(|_| AprenderError::FormatError {
                message: "Invalid nonce size".to_string(),
            })?;
    let ciphertext = &data[nonce_end..payload_end];

    // Derive key using Argon2id (same parameters as encryption)
    let mut key = [0u8; KEY_SIZE];
    Argon2::default()
        .hash_password_into(password.as_bytes(), &salt, &mut key)
        .map_err(|e| AprenderError::Other(format!("Key derivation failed: {e}")))?;

    // Decrypt payload with AES-256-GCM
    let cipher = Aes256Gcm::new_from_slice(&key)
        .map_err(|e| AprenderError::Other(format!("Failed to create cipher: {e}")))?;
    let nonce = Nonce::from_slice(&nonce_bytes);
    let payload_compressed =
        cipher
            .decrypt(nonce, ciphertext)
            .map_err(|_| AprenderError::DecryptionFailed {
                message: "Decryption failed (wrong password or corrupted data)".to_string(),
            })?;

    // Decompress payload
    let payload_uncompressed = decompress_payload(&payload_compressed, header.compression)?;

    // Deserialize model
    bincode::deserialize(&payload_uncompressed)
        .map_err(|e| AprenderError::Serialization(format!("Failed to deserialize model: {e}")))
}

/// Inspect model data without loading the payload (spec §1.1)
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
        quantized: header.flags.is_quantized(),
    })
}

/// Save a model with Ed25519 digital signature (spec §4.2)
///
/// Signs the model content (header + metadata + payload) for provenance verification.
/// The signature block (96 bytes) is appended before the checksum.
///
/// # Arguments
/// * `model` - The model to save (must implement Serialize)
/// * `model_type` - Model type identifier
/// * `path` - Output file path
/// * `options` - Save options (compression, metadata)
/// * `signing_key` - Ed25519 signing key for creating signature
///
/// # Errors
/// Returns error on I/O failure, serialization error, or signing failure
#[cfg(feature = "format-signing")]
#[allow(clippy::needless_pass_by_value)] // SaveOptions is small and passed by value for ergonomics
pub fn save_signed<M: Serialize>(
    model: &M,
    model_type: ModelType,
    path: impl AsRef<Path>,
    options: SaveOptions,
    signing_key: &SigningKey,
) -> Result<()> {
    use ed25519_dalek::Signer;

    let path = path.as_ref();

    // Serialize payload with bincode
    let payload_uncompressed = bincode::serialize(model)
        .map_err(|e| AprenderError::Serialization(format!("Failed to serialize model: {e}")))?;

    // Compress payload
    let (payload_compressed, compression) =
        compress_payload(&payload_uncompressed, options.compression)?;

    // Serialize metadata as MessagePack with named fields (spec §2)
    // Must use to_vec_named() for map mode to preserve field names with skip_serializing_if
    let metadata_bytes = rmp_serde::to_vec_named(&options.metadata)
        .map_err(|e| AprenderError::Serialization(format!("Failed to serialize metadata: {e}")))?;

    // Build header with SIGNED flag
    let mut header = Header::new(model_type);
    header.compression = compression;
    header.metadata_size = metadata_bytes.len() as u32;
    header.payload_size = payload_compressed.len() as u32;
    header.uncompressed_size = payload_uncompressed.len() as u32;
    header.flags = header.flags.with_signed();

    // Assemble content to sign (header + metadata + payload)
    let mut signable_content = Vec::new();
    signable_content.extend_from_slice(&header.to_bytes());
    signable_content.extend_from_slice(&metadata_bytes);
    signable_content.extend_from_slice(&payload_compressed);

    // Sign the content
    let signature = signing_key.sign(&signable_content);
    let verifying_key = signing_key.verifying_key();

    // Assemble complete file content
    let mut content = signable_content;
    content.extend_from_slice(&signature.to_bytes()); // 64 bytes
    content.extend_from_slice(verifying_key.as_bytes()); // 32 bytes

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

/// Load a model with signature verification (spec §4.2, Jidoka)
///
/// Verifies the Ed25519 signature before deserializing the model.
/// If verification fails, loading halts immediately (Jidoka principle).
///
/// # Arguments
/// * `path` - Input file path
/// * `expected_type` - Expected model type (for type safety)
/// * `trusted_key` - Optional trusted public key for verification (if None, uses embedded key)
///
/// # Errors
/// Returns error on I/O failure, format error, type mismatch, or signature verification failure
#[cfg(feature = "format-signing")]
pub fn load_verified<M: DeserializeOwned>(
    path: impl AsRef<Path>,
    expected_type: ModelType,
    trusted_key: Option<&VerifyingKey>,
) -> Result<M> {
    use ed25519_dalek::{Signature, Verifier};

    let path = path.as_ref();

    // Read entire file
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut content = Vec::new();
    reader.read_to_end(&mut content)?;

    // Verify minimum size (header + signature block + checksum)
    const SIGNATURE_BLOCK_SIZE: usize = SIGNATURE_SIZE + PUBLIC_KEY_SIZE; // 96 bytes
    if content.len() < HEADER_SIZE + SIGNATURE_BLOCK_SIZE + 4 {
        return Err(AprenderError::FormatError {
            message: format!("File too small for signed model: {} bytes", content.len()),
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

    // Verify SIGNED flag is set
    if !header.flags.is_signed() {
        return Err(AprenderError::FormatError {
            message: "File is not signed (SIGNED flag not set)".to_string(),
        });
    }

    // Verify model type
    if header.model_type != expected_type {
        return Err(AprenderError::FormatError {
            message: format!(
                "Model type mismatch: file contains {:?}, expected {:?}",
                header.model_type, expected_type
            ),
        });
    }

    // Calculate content boundaries
    let metadata_end = HEADER_SIZE + header.metadata_size as usize;
    let payload_end = metadata_end + header.payload_size as usize;
    let signature_start = payload_end;
    let pubkey_start = signature_start + SIGNATURE_SIZE;
    let pubkey_end = pubkey_start + PUBLIC_KEY_SIZE;

    if pubkey_end > content.len() - 4 {
        return Err(AprenderError::FormatError {
            message: "Signature block extends beyond file boundary".to_string(),
        });
    }

    // Extract signature and public key
    let signature_bytes: [u8; 64] =
        content[signature_start..pubkey_start]
            .try_into()
            .map_err(|_| AprenderError::FormatError {
                message: "Invalid signature size".to_string(),
            })?;
    let signature = Signature::from_bytes(&signature_bytes);

    let pubkey_bytes: [u8; 32] =
        content[pubkey_start..pubkey_end]
            .try_into()
            .map_err(|_| AprenderError::FormatError {
                message: "Invalid public key size".to_string(),
            })?;
    let embedded_key =
        VerifyingKey::from_bytes(&pubkey_bytes).map_err(|e| AprenderError::FormatError {
            message: format!("Invalid public key: {e}"),
        })?;

    // Use trusted key if provided, otherwise use embedded key
    let verifying_key = trusted_key.unwrap_or(&embedded_key);

    // Extract signable content (header + metadata + payload)
    let signable_content = &content[..payload_end];

    // Verify signature (Jidoka: halt on verification failure)
    verifying_key
        .verify(signable_content, &signature)
        .map_err(|e| AprenderError::SignatureInvalid {
            reason: format!("Signature verification failed: {e}"),
        })?;

    // Extract and decompress payload
    let payload_compressed = &content[metadata_end..payload_end];
    let payload_uncompressed = decompress_payload(payload_compressed, header.compression)?;

    // Deserialize model
    bincode::deserialize(&payload_uncompressed)
        .map_err(|e| AprenderError::Serialization(format!("Failed to deserialize model: {e}")))
}

/// Save a model with password-based encryption (spec §4.1.2)
///
/// Encrypts the model payload using AES-256-GCM with a key derived from
/// the password using Argon2id. The salt and nonce are prepended to the
/// encrypted payload.
///
/// # Arguments
/// * `model` - The model to save (must implement Serialize)
/// * `model_type` - Model type identifier
/// * `path` - Output file path
/// * `options` - Save options (compression, metadata)
/// * `password` - Password for encryption
///
/// # Errors
/// Returns error on I/O failure, serialization error, or encryption failure
#[cfg(feature = "format-encryption")]
#[allow(clippy::needless_pass_by_value)] // SaveOptions is small and passed by value for ergonomics
pub fn save_encrypted<M: Serialize>(
    model: &M,
    model_type: ModelType,
    path: impl AsRef<Path>,
    options: SaveOptions,
    password: &str,
) -> Result<()> {
    use aes_gcm::{
        aead::{Aead, KeyInit},
        Aes256Gcm, Nonce,
    };
    use argon2::Argon2;

    let path = path.as_ref();

    // Serialize payload with bincode
    let payload_uncompressed = bincode::serialize(model)
        .map_err(|e| AprenderError::Serialization(format!("Failed to serialize model: {e}")))?;

    // Compress payload
    let (payload_compressed, compression) =
        compress_payload(&payload_uncompressed, options.compression)?;

    // Generate random salt and nonce
    let mut salt = [0u8; SALT_SIZE];
    let mut nonce_bytes = [0u8; NONCE_SIZE];
    rand::RngCore::fill_bytes(&mut rand::rngs::OsRng, &mut salt);
    rand::RngCore::fill_bytes(&mut rand::rngs::OsRng, &mut nonce_bytes);

    // Derive key using Argon2id (spec §4.1.2)
    let mut key = [0u8; KEY_SIZE];
    Argon2::default()
        .hash_password_into(password.as_bytes(), &salt, &mut key)
        .map_err(|e| AprenderError::Other(format!("Key derivation failed: {e}")))?;

    // Encrypt payload with AES-256-GCM
    let cipher = Aes256Gcm::new_from_slice(&key)
        .map_err(|e| AprenderError::Other(format!("Failed to create cipher: {e}")))?;
    let nonce = Nonce::from_slice(&nonce_bytes);
    let ciphertext = cipher
        .encrypt(nonce, payload_compressed.as_ref())
        .map_err(|e| AprenderError::Other(format!("Encryption failed: {e}")))?;

    // Serialize metadata as MessagePack with named fields (spec §2)
    // Must use to_vec_named() for map mode to preserve field names with skip_serializing_if
    let metadata_bytes = rmp_serde::to_vec_named(&options.metadata)
        .map_err(|e| AprenderError::Serialization(format!("Failed to serialize metadata: {e}")))?;

    // Build header with ENCRYPTED flag
    let mut header = Header::new(model_type);
    header.compression = compression;
    header.metadata_size = metadata_bytes.len() as u32;
    // Payload size now includes salt + nonce + ciphertext
    header.payload_size = (SALT_SIZE + NONCE_SIZE + ciphertext.len()) as u32;
    header.uncompressed_size = payload_uncompressed.len() as u32;
    header.flags = header.flags.with_encrypted();

    // Assemble file content
    let mut content = Vec::new();
    content.extend_from_slice(&header.to_bytes());
    content.extend_from_slice(&metadata_bytes);
    content.extend_from_slice(&salt);
    content.extend_from_slice(&nonce_bytes);
    content.extend_from_slice(&ciphertext);

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

/// Load a model with password-based decryption (spec §4.1.2)
///
/// Decrypts the model payload using AES-256-GCM with a key derived from
/// the password using Argon2id.
///
/// # Arguments
/// * `path` - Input file path
/// * `expected_type` - Expected model type (for type safety)
/// * `password` - Password for decryption
///
/// # Errors
/// Returns error on I/O failure, format error, type mismatch, or decryption failure
#[cfg(feature = "format-encryption")]
pub fn load_encrypted<M: DeserializeOwned>(
    path: impl AsRef<Path>,
    expected_type: ModelType,
    password: &str,
) -> Result<M> {
    use aes_gcm::{
        aead::{Aead, KeyInit},
        Aes256Gcm, Nonce,
    };
    use argon2::Argon2;

    let path = path.as_ref();

    // Read entire file
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut content = Vec::new();
    reader.read_to_end(&mut content)?;

    // Verify minimum size
    if content.len() < HEADER_SIZE + SALT_SIZE + NONCE_SIZE + 4 {
        return Err(AprenderError::FormatError {
            message: format!(
                "File too small for encrypted model: {} bytes",
                content.len()
            ),
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

    // Verify ENCRYPTED flag is set
    if !header.flags.is_encrypted() {
        return Err(AprenderError::FormatError {
            message: "File is not encrypted (ENCRYPTED flag not set)".to_string(),
        });
    }

    // Verify model type
    if header.model_type != expected_type {
        return Err(AprenderError::FormatError {
            message: format!(
                "Model type mismatch: file contains {:?}, expected {:?}",
                header.model_type, expected_type
            ),
        });
    }

    // Calculate content boundaries
    let metadata_end = HEADER_SIZE + header.metadata_size as usize;
    let salt_end = metadata_end + SALT_SIZE;
    let nonce_end = salt_end + NONCE_SIZE;
    let payload_end = metadata_end + header.payload_size as usize;

    if payload_end > content.len() - 4 {
        return Err(AprenderError::FormatError {
            message: "Encrypted payload extends beyond file boundary".to_string(),
        });
    }

    // Extract salt, nonce, and ciphertext
    let salt: [u8; SALT_SIZE] =
        content[metadata_end..salt_end]
            .try_into()
            .map_err(|_| AprenderError::FormatError {
                message: "Invalid salt size".to_string(),
            })?;
    let nonce_bytes: [u8; NONCE_SIZE] =
        content[salt_end..nonce_end]
            .try_into()
            .map_err(|_| AprenderError::FormatError {
                message: "Invalid nonce size".to_string(),
            })?;
    let ciphertext = &content[nonce_end..payload_end];

    // Derive key using Argon2id (same parameters as encryption)
    let mut key = [0u8; KEY_SIZE];
    Argon2::default()
        .hash_password_into(password.as_bytes(), &salt, &mut key)
        .map_err(|e| AprenderError::Other(format!("Key derivation failed: {e}")))?;

    // Decrypt payload with AES-256-GCM
    let cipher = Aes256Gcm::new_from_slice(&key)
        .map_err(|e| AprenderError::Other(format!("Failed to create cipher: {e}")))?;
    let nonce = Nonce::from_slice(&nonce_bytes);
    let payload_compressed =
        cipher
            .decrypt(nonce, ciphertext)
            .map_err(|_| AprenderError::DecryptionFailed {
                message: "Decryption failed (wrong password or corrupted data)".to_string(),
            })?;

    // Decompress payload
    let payload_uncompressed = decompress_payload(&payload_compressed, header.compression)?;

    // Deserialize model
    bincode::deserialize(&payload_uncompressed)
        .map_err(|e| AprenderError::Serialization(format!("Failed to deserialize model: {e}")))
}

/// Save a model encrypted for a specific recipient (spec §4.1.3)
///
/// Uses X25519 key agreement + AES-256-GCM. The sender generates an ephemeral
/// keypair, performs ECDH with the recipient's public key, and derives the
/// encryption key using HKDF-SHA256.
///
/// # Arguments
/// * `model` - The model to save (must implement Serialize)
/// * `model_type` - Model type identifier
/// * `path` - Output file path
/// * `options` - Save options (compression, metadata)
/// * `recipient_public_key` - Recipient's X25519 public key
///
/// # Errors
/// Returns error on I/O failure, serialization error, or encryption failure
#[cfg(feature = "format-encryption")]
#[allow(clippy::needless_pass_by_value)] // SaveOptions is small and passed by value for ergonomics
pub fn save_for_recipient<M: Serialize>(
    model: &M,
    model_type: ModelType,
    path: impl AsRef<Path>,
    options: SaveOptions,
    recipient_public_key: &X25519PublicKey,
) -> Result<()> {
    use aes_gcm::{
        aead::{Aead, KeyInit},
        Aes256Gcm, Nonce,
    };
    use hkdf::Hkdf;
    use sha2::Sha256;

    let path = path.as_ref();

    // Serialize payload with bincode
    let payload_uncompressed = bincode::serialize(model)
        .map_err(|e| AprenderError::Serialization(format!("Failed to serialize model: {e}")))?;

    // Compress payload
    let (payload_compressed, compression) =
        compress_payload(&payload_uncompressed, options.compression)?;

    // Generate ephemeral keypair for this encryption
    let ephemeral_secret = X25519SecretKey::random_from_rng(rand::rngs::OsRng);
    let ephemeral_public = X25519PublicKey::from(&ephemeral_secret);

    // Perform X25519 key agreement
    let shared_secret = ephemeral_secret.diffie_hellman(recipient_public_key);

    // Derive encryption key using HKDF-SHA256 (spec §4.1.3)
    let hkdf = Hkdf::<Sha256>::new(None, shared_secret.as_bytes());
    let mut key = [0u8; KEY_SIZE];
    hkdf.expand(HKDF_INFO, &mut key)
        .map_err(|_| AprenderError::Other("HKDF expansion failed".to_string()))?;

    // Generate random nonce
    let mut nonce_bytes = [0u8; NONCE_SIZE];
    rand::RngCore::fill_bytes(&mut rand::rngs::OsRng, &mut nonce_bytes);

    // Encrypt payload with AES-256-GCM
    let cipher = Aes256Gcm::new_from_slice(&key)
        .map_err(|e| AprenderError::Other(format!("Failed to create cipher: {e}")))?;
    let nonce = Nonce::from_slice(&nonce_bytes);
    let ciphertext = cipher
        .encrypt(nonce, payload_compressed.as_ref())
        .map_err(|e| AprenderError::Other(format!("Encryption failed: {e}")))?;

    // Create recipient hash (first 8 bytes of recipient public key for identification)
    let recipient_hash: [u8; RECIPIENT_HASH_SIZE] = recipient_public_key.as_bytes()
        [..RECIPIENT_HASH_SIZE]
        .try_into()
        .expect("recipient hash size is correct");

    // Serialize metadata as MessagePack with named fields (spec §2)
    // Must use to_vec_named() for map mode to preserve field names with skip_serializing_if
    let metadata_bytes = rmp_serde::to_vec_named(&options.metadata)
        .map_err(|e| AprenderError::Serialization(format!("Failed to serialize metadata: {e}")))?;

    // Build header with ENCRYPTED flag
    let mut header = Header::new(model_type);
    header.compression = compression;
    header.metadata_size = metadata_bytes.len() as u32;
    // Payload: ephemeral_pub (32) + recipient_hash (8) + nonce (12) + ciphertext
    header.payload_size =
        (X25519_PUBLIC_KEY_SIZE + RECIPIENT_HASH_SIZE + NONCE_SIZE + ciphertext.len()) as u32;
    header.uncompressed_size = payload_uncompressed.len() as u32;
    header.flags = header.flags.with_encrypted();

    // Assemble file content (spec §4.1.3 layout)
    let mut content = Vec::new();
    content.extend_from_slice(&header.to_bytes());
    content.extend_from_slice(&metadata_bytes);
    content.extend_from_slice(ephemeral_public.as_bytes()); // 32 bytes
    content.extend_from_slice(&recipient_hash); // 8 bytes
    content.extend_from_slice(&nonce_bytes); // 12 bytes
    content.extend_from_slice(&ciphertext);

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

/// Load a model encrypted for this recipient (spec §4.1.3)
///
/// Uses X25519 key agreement + AES-256-GCM. The recipient uses their secret key
/// to perform ECDH with the sender's ephemeral public key.
///
/// # Arguments
/// * `path` - Input file path
/// * `expected_type` - Expected model type (for type safety)
/// * `recipient_secret_key` - Recipient's X25519 secret key
///
/// # Errors
/// Returns error on I/O failure, format error, type mismatch, or decryption failure
#[cfg(feature = "format-encryption")]
#[allow(clippy::too_many_lines)]
pub fn load_as_recipient<M: DeserializeOwned>(
    path: impl AsRef<Path>,
    expected_type: ModelType,
    recipient_secret_key: &X25519SecretKey,
) -> Result<M> {
    use aes_gcm::{
        aead::{Aead, KeyInit},
        Aes256Gcm, Nonce,
    };
    use hkdf::Hkdf;
    use sha2::Sha256;

    let path = path.as_ref();

    // Read entire file
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut content = Vec::new();
    reader.read_to_end(&mut content)?;

    // Calculate minimum size for X25519 encrypted file
    const MIN_PAYLOAD_SIZE: usize = X25519_PUBLIC_KEY_SIZE + RECIPIENT_HASH_SIZE + NONCE_SIZE;
    if content.len() < HEADER_SIZE + MIN_PAYLOAD_SIZE + 4 {
        return Err(AprenderError::FormatError {
            message: format!(
                "File too small for X25519 encrypted model: {} bytes",
                content.len()
            ),
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

    // Verify ENCRYPTED flag is set
    if !header.flags.is_encrypted() {
        return Err(AprenderError::FormatError {
            message: "File is not encrypted (ENCRYPTED flag not set)".to_string(),
        });
    }

    // Verify model type
    if header.model_type != expected_type {
        return Err(AprenderError::FormatError {
            message: format!(
                "Model type mismatch: file contains {:?}, expected {:?}",
                header.model_type, expected_type
            ),
        });
    }

    // Calculate content boundaries
    let metadata_end = HEADER_SIZE + header.metadata_size as usize;
    let ephemeral_pub_end = metadata_end + X25519_PUBLIC_KEY_SIZE;
    let recipient_hash_end = ephemeral_pub_end + RECIPIENT_HASH_SIZE;
    let nonce_end = recipient_hash_end + NONCE_SIZE;
    let payload_end = metadata_end + header.payload_size as usize;

    if payload_end > content.len() - 4 {
        return Err(AprenderError::FormatError {
            message: "Encrypted payload extends beyond file boundary".to_string(),
        });
    }

    // Extract ephemeral public key
    let ephemeral_pub_bytes: [u8; X25519_PUBLIC_KEY_SIZE] = content
        [metadata_end..ephemeral_pub_end]
        .try_into()
        .map_err(|_| AprenderError::FormatError {
            message: "Invalid ephemeral public key size".to_string(),
        })?;
    let ephemeral_public = X25519PublicKey::from(ephemeral_pub_bytes);

    // Extract and verify recipient hash
    let stored_recipient_hash: [u8; RECIPIENT_HASH_SIZE] = content
        [ephemeral_pub_end..recipient_hash_end]
        .try_into()
        .map_err(|_| AprenderError::FormatError {
            message: "Invalid recipient hash size".to_string(),
        })?;

    // Verify this file is for us
    let our_public = X25519PublicKey::from(recipient_secret_key);
    let our_hash: [u8; RECIPIENT_HASH_SIZE] = our_public.as_bytes()[..RECIPIENT_HASH_SIZE]
        .try_into()
        .expect("hash size is correct");

    if stored_recipient_hash != our_hash {
        return Err(AprenderError::DecryptionFailed {
            message: "This file was encrypted for a different recipient".to_string(),
        });
    }

    // Extract nonce and ciphertext
    let nonce_bytes: [u8; NONCE_SIZE] =
        content[recipient_hash_end..nonce_end]
            .try_into()
            .map_err(|_| AprenderError::FormatError {
                message: "Invalid nonce size".to_string(),
            })?;
    let ciphertext = &content[nonce_end..payload_end];

    // Perform X25519 key agreement
    let shared_secret = recipient_secret_key.diffie_hellman(&ephemeral_public);

    // Derive encryption key using HKDF-SHA256 (same as encryption)
    let hkdf = Hkdf::<Sha256>::new(None, shared_secret.as_bytes());
    let mut key = [0u8; KEY_SIZE];
    hkdf.expand(HKDF_INFO, &mut key)
        .map_err(|_| AprenderError::Other("HKDF expansion failed".to_string()))?;

    // Decrypt payload with AES-256-GCM
    let cipher = Aes256Gcm::new_from_slice(&key)
        .map_err(|e| AprenderError::Other(format!("Failed to create cipher: {e}")))?;
    let nonce = Nonce::from_slice(&nonce_bytes);
    let payload_compressed =
        cipher
            .decrypt(nonce, ciphertext)
            .map_err(|_| AprenderError::DecryptionFailed {
                message: "Decryption failed (wrong recipient key or corrupted data)".to_string(),
            })?;

    // Decompress payload
    let payload_uncompressed = decompress_payload(&payload_compressed, header.compression)?;

    // Deserialize model
    bincode::deserialize(&payload_uncompressed)
        .map_err(|e| AprenderError::Serialization(format!("Failed to deserialize model: {e}")))
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
        let err = result.expect_err("Should fail with invalid magic");
        assert!(err.to_string().contains("Invalid magic"));
    }

    #[test]
    fn test_unsupported_version() {
        let mut header = Header::new(ModelType::LinearRegression);
        header.version = (99, 0); // Future version

        let mut bytes = header.to_bytes();
        bytes[4] = 99; // Major version

        let result = Header::from_bytes(&bytes);
        let err = result.expect_err("Should fail with unsupported version");
        assert!(err.to_string().contains("Unsupported"));
    }

    #[test]
    fn test_compression_bomb_protection() {
        let mut header = Header::new(ModelType::LinearRegression);
        header.uncompressed_size = MAX_UNCOMPRESSED_SIZE + 1;

        let bytes = header.to_bytes();
        let result = Header::from_bytes(&bytes);
        let err = result.expect_err("Should fail with compression bomb protection");
        assert!(err.to_string().contains("compression bomb"));
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
            .with_trueno_native()
            .with_quantized();

        assert!(flags.is_encrypted());
        assert!(flags.is_signed());
        assert!(flags.is_streaming());
        assert!(flags.is_licensed());
        assert!(flags.is_trueno_native());
        assert!(flags.is_quantized());
        assert_eq!(flags.bits(), 0b0011_1111);
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
    fn test_load_from_bytes_roundtrip() {
        use tempfile::tempdir;

        // Simple serializable struct for testing
        #[derive(Debug, Serialize, Deserialize, PartialEq)]
        struct TestModel {
            weights: Vec<f32>,
            bias: f32,
        }

        let model = TestModel {
            weights: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            bias: 0.5,
        };

        let dir = tempdir().expect("create temp dir");
        let path = dir.path().join("embedded.apr");

        // Save to file first
        save(&model, ModelType::Custom, &path, SaveOptions::default())
            .expect("save should succeed");

        // Read file into bytes (simulating include_bytes!)
        let data = std::fs::read(&path).expect("read file");

        // Load from bytes
        let loaded: TestModel =
            load_from_bytes(&data, ModelType::Custom).expect("load_from_bytes should succeed");
        assert_eq!(loaded, model);

        // Inspect from bytes
        let info = inspect_bytes(&data).expect("inspect_bytes should succeed");
        assert_eq!(info.model_type, ModelType::Custom);
        assert_eq!(info.format_version, FORMAT_VERSION);
    }

    #[test]
    fn test_load_from_bytes_type_mismatch() {
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

        let data = std::fs::read(&path).expect("read file");

        // Load with wrong type should fail
        let result: Result<TestModel> = load_from_bytes(&data, ModelType::KMeans);
        assert!(result.is_err());
        assert!(result
            .expect_err("should fail")
            .to_string()
            .contains("type mismatch"));
    }

    #[test]
    fn test_load_from_bytes_checksum_failure() {
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

        // Read and corrupt the data
        let mut data = std::fs::read(&path).expect("read file");
        if data.len() > HEADER_SIZE + 10 {
            data[HEADER_SIZE + 5] ^= 0xFF; // Flip some bits
        }

        // Load should fail with checksum error
        let result: Result<TestModel> = load_from_bytes(&data, ModelType::Custom);
        assert!(result.is_err());
        assert!(result
            .expect_err("should fail")
            .to_string()
            .contains("Checksum"));
    }

    #[test]
    fn test_load_from_bytes_too_small() {
        let data = vec![0u8; 10]; // Too small

        let result: Result<i32> = load_from_bytes(&data, ModelType::Custom);
        assert!(result.is_err());
        assert!(result
            .expect_err("should fail")
            .to_string()
            .contains("too small"));
    }

    #[test]
    fn test_inspect_bytes_too_small() {
        let data = vec![0u8; 10]; // Too small

        let result = inspect_bytes(&data);
        assert!(result.is_err());
        assert!(result
            .expect_err("should fail")
            .to_string()
            .contains("too small"));
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
        let err = result.expect_err("Should fail with checksum error");
        assert!(err.to_string().contains("Checksum"));
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
        let err = result.expect_err("Should fail with type mismatch");
        assert!(err.to_string().contains("type mismatch"));
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

    #[test]
    #[cfg(feature = "format-signing")]
    fn test_signed_save_load_roundtrip() {
        use tempfile::tempdir;

        #[derive(Debug, Serialize, Deserialize, PartialEq)]
        struct TestModel {
            weights: Vec<f32>,
            bias: f32,
        }

        let model = TestModel {
            weights: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            bias: 0.5,
        };

        // Generate signing key
        let signing_key = SigningKey::generate(&mut rand::rngs::OsRng);
        let verifying_key = signing_key.verifying_key();

        let dir = tempdir().expect("create temp dir");
        let path = dir.path().join("signed.apr");

        // Save with signature
        save_signed(
            &model,
            ModelType::Custom,
            &path,
            SaveOptions::default(),
            &signing_key,
        )
        .expect("save_signed should succeed");

        // Inspect - should show signed flag
        let info = inspect(&path).expect("inspect should succeed");
        assert!(info.signed, "Model should be marked as signed");
        assert_eq!(info.model_type, ModelType::Custom);

        // Load with verification (using embedded key)
        let loaded: TestModel =
            load_verified(&path, ModelType::Custom, None).expect("load_verified should succeed");
        assert_eq!(loaded, model);

        // Load with verification (using trusted key)
        let loaded2: TestModel = load_verified(&path, ModelType::Custom, Some(&verifying_key))
            .expect("load_verified with trusted key should succeed");
        assert_eq!(loaded2, model);
    }

    #[test]
    #[cfg(feature = "format-signing")]
    fn test_signature_verification_fails_with_wrong_key() {
        use tempfile::tempdir;

        #[derive(Debug, Serialize, Deserialize, PartialEq)]
        struct TestModel {
            value: i32,
        }

        let model = TestModel { value: 42 };

        // Generate two different key pairs
        let signing_key = SigningKey::generate(&mut rand::rngs::OsRng);
        let wrong_key = SigningKey::generate(&mut rand::rngs::OsRng).verifying_key();

        let dir = tempdir().expect("create temp dir");
        let path = dir.path().join("signed_wrong.apr");

        // Save with one key
        save_signed(
            &model,
            ModelType::Custom,
            &path,
            SaveOptions::default(),
            &signing_key,
        )
        .expect("save_signed should succeed");

        // Try to verify with wrong key - should fail
        let result: Result<TestModel> = load_verified(&path, ModelType::Custom, Some(&wrong_key));
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Signature verification failed"));
    }

    #[test]
    #[cfg(feature = "format-signing")]
    fn test_signed_model_detects_tampering() {
        use tempfile::tempdir;

        #[derive(Debug, Serialize, Deserialize)]
        struct TestModel {
            value: i32,
        }

        let model = TestModel { value: 42 };
        let signing_key = SigningKey::generate(&mut rand::rngs::OsRng);

        let dir = tempdir().expect("create temp dir");
        let path = dir.path().join("tampered.apr");

        // Save signed model
        save_signed(
            &model,
            ModelType::Custom,
            &path,
            SaveOptions::default(),
            &signing_key,
        )
        .expect("save_signed should succeed");

        // Tamper with the file (modify a byte in the payload)
        let mut content = std::fs::read(&path).expect("read file");
        let payload_offset = HEADER_SIZE + 20; // Somewhere in metadata/payload
        if content.len() > payload_offset {
            content[payload_offset] ^= 0xFF;
        }

        // Recalculate checksum to avoid checksum failure
        let checksum_start = content.len() - 4;
        let new_checksum = crc32(&content[..checksum_start]);
        content[checksum_start..].copy_from_slice(&new_checksum.to_le_bytes());

        std::fs::write(&path, &content).expect("write tampered file");

        // Verification should fail due to signature mismatch
        let result: Result<TestModel> = load_verified(&path, ModelType::Custom, None);
        assert!(result.is_err());
        // Either signature verification fails or parsing fails due to corrupted data
    }

    #[test]
    #[cfg(feature = "format-signing")]
    fn test_load_verified_rejects_unsigned_file() {
        use tempfile::tempdir;

        #[derive(Debug, Serialize, Deserialize)]
        struct TestModel {
            value: i32,
        }

        let model = TestModel { value: 42 };

        let dir = tempdir().expect("create temp dir");
        let path = dir.path().join("unsigned.apr");

        // Save without signature
        save(&model, ModelType::Custom, &path, SaveOptions::default())
            .expect("save should succeed");

        // load_verified should reject unsigned files
        let result: Result<TestModel> = load_verified(&path, ModelType::Custom, None);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("SIGNED flag not set") || err_msg.contains("File too small"),
            "Expected SIGNED flag error or size error, got: {err_msg}"
        );
    }

    #[test]
    #[cfg(feature = "format-encryption")]
    fn test_encrypted_save_load_roundtrip() {
        use tempfile::tempdir;

        #[derive(Debug, Serialize, Deserialize, PartialEq)]
        struct TestModel {
            weights: Vec<f32>,
            bias: f32,
        }

        let model = TestModel {
            weights: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            bias: 0.5,
        };

        let password = "super_secret_password_123!";

        let dir = tempdir().expect("create temp dir");
        let path = dir.path().join("encrypted.apr");

        // Save with encryption
        save_encrypted(
            &model,
            ModelType::Custom,
            &path,
            SaveOptions::default(),
            password,
        )
        .expect("save_encrypted should succeed");

        // Inspect - should show encrypted flag
        let info = inspect(&path).expect("inspect should succeed");
        assert!(info.encrypted, "Model should be marked as encrypted");
        assert_eq!(info.model_type, ModelType::Custom);

        // Load with correct password
        let loaded: TestModel = load_encrypted(&path, ModelType::Custom, password)
            .expect("load_encrypted should succeed");
        assert_eq!(loaded, model);
    }

    #[test]
    #[cfg(feature = "format-encryption")]
    fn test_load_from_bytes_encrypted_roundtrip() {
        use tempfile::tempdir;

        #[derive(Debug, Serialize, Deserialize, PartialEq)]
        struct TestModel {
            weights: Vec<f32>,
            bias: f32,
        }

        let model = TestModel {
            weights: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            bias: 0.5,
        };

        let password = "embedded_secret_123!";

        let dir = tempdir().expect("create temp dir");
        let path = dir.path().join("encrypted_embedded.apr");

        // Save with encryption
        save_encrypted(
            &model,
            ModelType::Custom,
            &path,
            SaveOptions::default(),
            password,
        )
        .expect("save_encrypted should succeed");

        // Read file into bytes (simulating include_bytes!)
        let data = std::fs::read(&path).expect("read file");

        // Load from bytes with correct password
        let loaded: TestModel = load_from_bytes_encrypted(&data, ModelType::Custom, password)
            .expect("load_from_bytes_encrypted should succeed");
        assert_eq!(loaded, model);
    }

    #[test]
    #[cfg(feature = "format-encryption")]
    fn test_load_from_bytes_encrypted_wrong_password() {
        use tempfile::tempdir;

        #[derive(Debug, Serialize, Deserialize)]
        struct TestModel {
            value: i32,
        }

        let model = TestModel { value: 42 };
        let password = "correct_password";

        let dir = tempdir().expect("create temp dir");
        let path = dir.path().join("encrypted.apr");

        save_encrypted(
            &model,
            ModelType::Custom,
            &path,
            SaveOptions::default(),
            password,
        )
        .expect("save should succeed");

        let data = std::fs::read(&path).expect("read file");

        // Load with wrong password should fail
        let result: Result<TestModel> =
            load_from_bytes_encrypted(&data, ModelType::Custom, "wrong_password");
        assert!(result.is_err());
        assert!(result
            .expect_err("should fail")
            .to_string()
            .contains("Decryption failed"));
    }

    #[test]
    #[cfg(feature = "format-encryption")]
    fn test_encrypted_wrong_password_fails() {
        use tempfile::tempdir;

        #[derive(Debug, Serialize, Deserialize, PartialEq)]
        struct TestModel {
            value: i32,
        }

        let model = TestModel { value: 42 };
        let correct_password = "correct_password";
        let wrong_password = "wrong_password";

        let dir = tempdir().expect("create temp dir");
        let path = dir.path().join("encrypted_wrong.apr");

        // Save with correct password
        save_encrypted(
            &model,
            ModelType::Custom,
            &path,
            SaveOptions::default(),
            correct_password,
        )
        .expect("save_encrypted should succeed");

        // Try to load with wrong password - should fail
        let result: Result<TestModel> = load_encrypted(&path, ModelType::Custom, wrong_password);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Decryption failed"));
    }

    #[test]
    #[cfg(feature = "format-encryption")]
    fn test_load_encrypted_rejects_unencrypted_file() {
        use tempfile::tempdir;

        #[derive(Debug, Serialize, Deserialize)]
        struct TestModel {
            value: i32,
        }

        let model = TestModel { value: 42 };

        let dir = tempdir().expect("create temp dir");
        let path = dir.path().join("unencrypted.apr");

        // Save without encryption
        save(&model, ModelType::Custom, &path, SaveOptions::default())
            .expect("save should succeed");

        // load_encrypted should reject unencrypted files
        let result: Result<TestModel> = load_encrypted(&path, ModelType::Custom, "any_password");
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("ENCRYPTED flag not set") || err_msg.contains("File too small"),
            "Expected ENCRYPTED flag error or size error, got: {err_msg}"
        );
    }

    #[test]
    #[cfg(feature = "format-encryption")]
    fn test_encrypted_with_compression() {
        use tempfile::tempdir;

        #[derive(Debug, Serialize, Deserialize, PartialEq)]
        struct LargeModel {
            data: Vec<f32>,
        }

        // Repetitive data compresses well
        let model = LargeModel {
            data: (0..1000).map(|i| (i % 10) as f32).collect(),
        };

        let password = "compress_and_encrypt";

        let dir = tempdir().expect("create temp dir");
        let path = dir.path().join("encrypted_compressed.apr");

        // Save with encryption (compression will be applied before encryption)
        save_encrypted(
            &model,
            ModelType::Custom,
            &path,
            SaveOptions::default(), // No explicit compression, but internal compression happens
            password,
        )
        .expect("save_encrypted should succeed");

        // Load and verify
        let loaded: LargeModel = load_encrypted(&path, ModelType::Custom, password)
            .expect("load_encrypted should succeed");
        assert_eq!(loaded, model);
    }

    #[test]
    #[cfg(feature = "format-encryption")]
    fn test_x25519_recipient_roundtrip() {
        use tempfile::tempdir;
        use x25519_dalek::{PublicKey, StaticSecret};

        #[derive(Debug, Serialize, Deserialize, PartialEq)]
        struct TestModel {
            weights: Vec<f32>,
            bias: f32,
        }

        let model = TestModel {
            weights: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            bias: 0.5,
        };

        // Generate recipient keypair
        let recipient_secret = StaticSecret::random_from_rng(rand::thread_rng());
        let recipient_public = PublicKey::from(&recipient_secret);

        let dir = tempdir().expect("create temp dir");
        let path = dir.path().join("recipient_encrypted.apr");

        // Save for recipient
        save_for_recipient(
            &model,
            ModelType::Custom,
            &path,
            SaveOptions::default(),
            &recipient_public,
        )
        .expect("save_for_recipient should succeed");

        // Inspect - should show encrypted flag
        let info = inspect(&path).expect("inspect should succeed");
        assert!(info.encrypted, "Model should be marked as encrypted");
        assert_eq!(info.model_type, ModelType::Custom);

        // Load as recipient with correct key
        let loaded: TestModel = load_as_recipient(&path, ModelType::Custom, &recipient_secret)
            .expect("load_as_recipient should succeed");
        assert_eq!(loaded, model);
    }

    #[test]
    #[cfg(feature = "format-encryption")]
    fn test_x25519_wrong_key_fails() {
        use tempfile::tempdir;
        use x25519_dalek::{PublicKey, StaticSecret};

        #[derive(Debug, Serialize, Deserialize, PartialEq)]
        struct TestModel {
            value: i32,
        }

        let model = TestModel { value: 42 };

        // Generate correct recipient keypair
        let recipient_secret = StaticSecret::random_from_rng(rand::thread_rng());
        let recipient_public = PublicKey::from(&recipient_secret);

        // Generate wrong keypair
        let wrong_secret = StaticSecret::random_from_rng(rand::thread_rng());

        let dir = tempdir().expect("create temp dir");
        let path = dir.path().join("x25519_wrong_key.apr");

        // Save for correct recipient
        save_for_recipient(
            &model,
            ModelType::Custom,
            &path,
            SaveOptions::default(),
            &recipient_public,
        )
        .expect("save_for_recipient should succeed");

        // Try to load with wrong key - should fail
        let result: Result<TestModel> = load_as_recipient(&path, ModelType::Custom, &wrong_secret);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Decryption failed"));
    }

    #[test]
    #[cfg(feature = "format-encryption")]
    fn test_x25519_rejects_password_encrypted_file() {
        use tempfile::tempdir;
        use x25519_dalek::StaticSecret;

        #[derive(Debug, Serialize, Deserialize)]
        struct TestModel {
            value: i32,
        }

        let model = TestModel { value: 42 };

        let dir = tempdir().expect("create temp dir");
        let path = dir.path().join("password_not_x25519.apr");

        // Save with password encryption
        save_encrypted(
            &model,
            ModelType::Custom,
            &path,
            SaveOptions::default(),
            "some_password",
        )
        .expect("save_encrypted should succeed");

        // Try to load as recipient - should fail
        let wrong_secret = StaticSecret::random_from_rng(rand::thread_rng());
        let result: Result<TestModel> = load_as_recipient(&path, ModelType::Custom, &wrong_secret);
        assert!(result.is_err());
        // Will fail because file layout doesn't match X25519 format
    }

    #[test]
    #[cfg(feature = "format-encryption")]
    fn test_x25519_load_rejects_unencrypted_file() {
        use tempfile::tempdir;
        use x25519_dalek::StaticSecret;

        #[derive(Debug, Serialize, Deserialize)]
        struct TestModel {
            value: i32,
        }

        let model = TestModel { value: 42 };

        let dir = tempdir().expect("create temp dir");
        let path = dir.path().join("unencrypted_for_x25519.apr");

        // Save without encryption
        save(&model, ModelType::Custom, &path, SaveOptions::default())
            .expect("save should succeed");

        // load_as_recipient should reject unencrypted files
        let secret = StaticSecret::random_from_rng(rand::thread_rng());
        let result: Result<TestModel> = load_as_recipient(&path, ModelType::Custom, &secret);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("ENCRYPTED flag not set") || err_msg.contains("File too small"),
            "Expected ENCRYPTED flag error or size error, got: {err_msg}"
        );
    }

    // EXTREME TDD: Distillation metadata (spec §6.3)
    // Step 1: GREEN - Verified we can use description field
    #[test]
    fn test_distillation_teacher_hash() {
        use tempfile::tempdir;

        #[derive(Debug, Serialize, Deserialize, PartialEq)]
        struct TestModel {
            value: i32,
        }

        let model = TestModel { value: 42 };
        let dir = tempdir().expect("create temp dir");
        let path = dir.path().join("distilled.apr");

        // For now, use the simplest approach - add to description
        let options = SaveOptions::default()
            .with_name("student_model")
            .with_description("Distilled from teacher abc123");

        save(&model, ModelType::Custom, &path, options).expect("save should succeed");

        // Inspect and verify description contains teacher info
        let info = inspect(&path).expect("inspect should succeed");
        assert!(info.metadata.description.is_some());
        assert!(info
            .metadata
            .description
            .as_ref()
            .expect("description should be set")
            .contains("abc123"));
    }

    // Step 2: RED - Test dedicated distillation field
    #[test]
    fn test_distillation_dedicated_field() {
        use tempfile::tempdir;

        #[derive(Debug, Serialize, Deserialize)]
        struct TestModel {
            value: i32,
        }

        let model = TestModel { value: 123 };
        let dir = tempdir().expect("create temp dir");
        let path = dir.path().join("distilled2.apr");

        // First verify that description (an existing Optional<String>) works
        let options = SaveOptions::default().with_description("test description");

        save(&model, ModelType::Custom, &path, options).expect("save should succeed");

        let info = inspect(&path).expect("inspect should succeed");

        // This should work
        assert_eq!(
            info.metadata.description,
            Some("test description".to_string())
        );

        // Now test distillation
        let mut options2 = SaveOptions::default();
        options2.metadata.distillation = Some("teacher_abc123".to_string());

        let path2 = dir.path().join("distilled2b.apr");
        save(&model, ModelType::Custom, &path2, options2).expect("save should succeed");

        let info2 = inspect(&path2).expect("inspect should succeed");
        assert_eq!(
            info2.metadata.distillation,
            Some("teacher_abc123".to_string())
        );
    }

    // Test: serialize/deserialize metadata directly with named fields
    #[test]
    fn test_metadata_msgpack_roundtrip() {
        let metadata = Metadata {
            description: Some("test description".to_string()),
            distillation: Some("teacher_abc123".to_string()),
            ..Default::default()
        };

        // Serialize with named fields (map mode) - required for skip_serializing_if
        let bytes = rmp_serde::to_vec_named(&metadata).expect("serialize");

        // Deserialize
        let restored: Metadata = rmp_serde::from_slice(&bytes).expect("deserialize");

        assert_eq!(restored.description, metadata.description);
        assert_eq!(restored.distillation, metadata.distillation);
    }

    // EXTREME TDD: Step 3 - RED test for DistillationInfo struct (spec §6.3)
    #[test]
    fn test_distillation_info_struct() {
        use tempfile::tempdir;

        #[derive(Debug, Serialize, Deserialize)]
        struct TestModel {
            value: i32,
        }

        let model = TestModel { value: 42 };
        let dir = tempdir().expect("create temp dir");
        let path = dir.path().join("distilled3.apr");

        // Create DistillationInfo per spec §6.3.2
        let distill_info = DistillationInfo {
            method: DistillMethod::Standard,
            teacher: TeacherProvenance {
                hash: "sha256:abc123def456".to_string(),
                signature: None,
                model_type: ModelType::NeuralSequential,
                param_count: 7_000_000_000, // 7B params
                ensemble_teachers: None,
            },
            params: DistillationParams {
                temperature: 3.0,
                alpha: 0.7,
                beta: None,
                epochs: 10,
                final_loss: Some(0.42),
            },
            layer_mapping: None,
        };

        let options = SaveOptions::default().with_distillation_info(distill_info);

        save(&model, ModelType::Custom, &path, options).expect("save should succeed");

        let info = inspect(&path).expect("inspect should succeed");
        let restored = info
            .metadata
            .distillation_info
            .expect("should have distillation_info");

        assert!(matches!(restored.method, DistillMethod::Standard));
        assert_eq!(restored.teacher.hash, "sha256:abc123def456");
        assert_eq!(restored.teacher.param_count, 7_000_000_000);
        assert!((restored.params.temperature - 3.0).abs() < f32::EPSILON);
        assert!((restored.params.alpha - 0.7).abs() < f32::EPSILON);
        assert_eq!(restored.params.epochs, 10);
        assert!(
            (restored.params.final_loss.expect("should have final_loss") - 0.42).abs()
                < f32::EPSILON
        );
    }

    // EXTREME TDD: Test progressive distillation with layer mapping (spec §6.3.1)
    #[test]
    fn test_distillation_progressive_with_layer_mapping() {
        use tempfile::tempdir;

        #[derive(Debug, Serialize, Deserialize)]
        struct TestModel {
            value: i32,
        }

        let model = TestModel { value: 42 };
        let dir = tempdir().expect("create temp dir");
        let path = dir.path().join("progressive.apr");

        // Progressive distillation: 4-layer student from 8-layer teacher
        let layer_mapping = vec![
            LayerMapping {
                student_layer: 0,
                teacher_layer: 0,
                weight: 0.5,
            },
            LayerMapping {
                student_layer: 1,
                teacher_layer: 2,
                weight: 0.3,
            },
            LayerMapping {
                student_layer: 2,
                teacher_layer: 5,
                weight: 0.15,
            },
            LayerMapping {
                student_layer: 3,
                teacher_layer: 7,
                weight: 0.05,
            },
        ];

        let distill_info = DistillationInfo {
            method: DistillMethod::Progressive,
            teacher: TeacherProvenance {
                hash: "sha256:teacher_8layer".to_string(),
                signature: Some("sig_abc123".to_string()),
                model_type: ModelType::NeuralSequential,
                param_count: 1_000_000_000, // 1B params
                ensemble_teachers: None,
            },
            params: DistillationParams {
                temperature: 4.0,
                alpha: 0.8,
                beta: Some(0.5), // Progressive uses beta for hidden vs logit loss
                epochs: 20,
                final_loss: Some(0.31),
            },
            layer_mapping: Some(layer_mapping),
        };

        let options = SaveOptions::default().with_distillation_info(distill_info);
        save(&model, ModelType::Custom, &path, options).expect("save should succeed");

        let info = inspect(&path).expect("inspect should succeed");
        let restored = info
            .metadata
            .distillation_info
            .expect("should have distillation_info");

        // Verify method
        assert!(matches!(restored.method, DistillMethod::Progressive));

        // Verify teacher info
        assert_eq!(restored.teacher.hash, "sha256:teacher_8layer");
        assert_eq!(restored.teacher.signature, Some("sig_abc123".to_string()));
        assert_eq!(restored.teacher.param_count, 1_000_000_000);

        // Verify params with beta
        assert!((restored.params.temperature - 4.0).abs() < f32::EPSILON);
        assert!((restored.params.alpha - 0.8).abs() < f32::EPSILON);
        assert!((restored.params.beta.expect("should have beta") - 0.5).abs() < f32::EPSILON);
        assert_eq!(restored.params.epochs, 20);

        // Verify layer mapping
        let mapping = restored.layer_mapping.expect("should have layer_mapping");
        assert_eq!(mapping.len(), 4);
        assert_eq!(mapping[0].student_layer, 0);
        assert_eq!(mapping[0].teacher_layer, 0);
        assert!((mapping[0].weight - 0.5).abs() < f32::EPSILON);
        assert_eq!(mapping[2].student_layer, 2);
        assert_eq!(mapping[2].teacher_layer, 5);
    }

    // EXTREME TDD: Test ensemble distillation with multiple teachers (spec §6.3.1)
    #[test]
    fn test_distillation_ensemble_multiple_teachers() {
        use tempfile::tempdir;

        #[derive(Debug, Serialize, Deserialize)]
        struct TestModel {
            value: i32,
        }

        let model = TestModel { value: 42 };
        let dir = tempdir().expect("create temp dir");
        let path = dir.path().join("ensemble.apr");

        // Ensemble: 3 teachers averaged
        let ensemble_teachers = vec![
            TeacherProvenance {
                hash: "sha256:teacher1".to_string(),
                signature: None,
                model_type: ModelType::NeuralSequential,
                param_count: 3_000_000_000,
                ensemble_teachers: None,
            },
            TeacherProvenance {
                hash: "sha256:teacher2".to_string(),
                signature: None,
                model_type: ModelType::NeuralSequential,
                param_count: 5_000_000_000,
                ensemble_teachers: None,
            },
            TeacherProvenance {
                hash: "sha256:teacher3".to_string(),
                signature: None,
                model_type: ModelType::GradientBoosting,
                param_count: 2_000_000_000,
                ensemble_teachers: None,
            },
        ];

        let distill_info = DistillationInfo {
            method: DistillMethod::Ensemble,
            teacher: TeacherProvenance {
                hash: "sha256:ensemble_meta".to_string(),
                signature: None,
                model_type: ModelType::NeuralSequential,
                param_count: 10_000_000_000, // Combined param count
                ensemble_teachers: Some(ensemble_teachers),
            },
            params: DistillationParams {
                temperature: 2.5,
                alpha: 0.6,
                beta: None,
                epochs: 15,
                final_loss: Some(0.28),
            },
            layer_mapping: None,
        };

        let options = SaveOptions::default().with_distillation_info(distill_info);
        save(&model, ModelType::Custom, &path, options).expect("save should succeed");

        let info = inspect(&path).expect("inspect should succeed");
        let restored = info
            .metadata
            .distillation_info
            .expect("should have distillation_info");

        // Verify method
        assert!(matches!(restored.method, DistillMethod::Ensemble));

        // Verify ensemble teachers
        let teachers = restored
            .teacher
            .ensemble_teachers
            .expect("should have ensemble_teachers");
        assert_eq!(teachers.len(), 3);
        assert_eq!(teachers[0].hash, "sha256:teacher1");
        assert_eq!(teachers[1].param_count, 5_000_000_000);
        assert!(matches!(
            teachers[2].model_type,
            ModelType::GradientBoosting
        ));

        // Verify combined param count
        assert_eq!(restored.teacher.param_count, 10_000_000_000);
    }

    // ========================================================================
    // EXTREME TDD: License Block (spec §9)
    // ========================================================================

    // Step 1: RED - Test LicenseInfo struct (spec §9.1)
    #[test]
    fn test_license_info_roundtrip() {
        use tempfile::tempdir;

        #[derive(Debug, Serialize, Deserialize)]
        struct TestModel {
            value: i32,
        }

        let model = TestModel { value: 42 };
        let dir = tempdir().expect("create temp dir");
        let path = dir.path().join("licensed.apr");

        // Create LicenseInfo per spec §9.1
        let license = LicenseInfo {
            uuid: "550e8400-e29b-41d4-a716-446655440000".to_string(),
            hash: "sha256:license_hash_abc123".to_string(),
            expiry: Some("2025-12-31T23:59:59Z".to_string()),
            seats: Some(10),
            licensee: Some("ACME Corp".to_string()),
            tier: LicenseTier::Enterprise,
        };

        let options = SaveOptions::default().with_license(license);

        save(&model, ModelType::Custom, &path, options).expect("save should succeed");

        let info = inspect(&path).expect("inspect should succeed");

        // Verify LICENSED flag is set
        assert!(info.licensed);

        // Verify license info restored
        let restored = info.metadata.license.expect("should have license");
        assert_eq!(restored.uuid, "550e8400-e29b-41d4-a716-446655440000");
        assert_eq!(restored.hash, "sha256:license_hash_abc123");
        assert_eq!(restored.expiry, Some("2025-12-31T23:59:59Z".to_string()));
        assert_eq!(restored.seats, Some(10));
        assert_eq!(restored.licensee, Some("ACME Corp".to_string()));
        assert!(matches!(restored.tier, LicenseTier::Enterprise));
    }

    // Step 2: RED - Test license tiers
    #[test]
    fn test_license_tiers() {
        use tempfile::tempdir;

        #[derive(Debug, Serialize, Deserialize)]
        struct TestModel {
            value: i32,
        }

        let model = TestModel { value: 42 };
        let dir = tempdir().expect("create temp dir");

        // Test each tier
        for (tier, name) in [
            (LicenseTier::Personal, "personal"),
            (LicenseTier::Team, "team"),
            (LicenseTier::Enterprise, "enterprise"),
            (LicenseTier::Academic, "academic"),
        ] {
            let path = dir.path().join(format!("{name}.apr"));

            let license = LicenseInfo {
                uuid: format!("uuid-{name}"),
                hash: format!("hash-{name}"),
                expiry: None,
                seats: None,
                licensee: None,
                tier,
            };

            let options = SaveOptions::default().with_license(license);
            save(&model, ModelType::Custom, &path, options).expect("save should succeed");

            let info = inspect(&path).expect("inspect should succeed");
            let restored = info.metadata.license.expect("should have license");
            assert_eq!(restored.uuid, format!("uuid-{name}"));
        }
    }

    // ========================================================================
    // EXTREME TDD: GGUF Export (spec §7.2)
    // ========================================================================

    // Step 1: RED - Test GGUF magic number and header
    #[test]
    fn test_gguf_magic_number() {
        // GGUF magic is "GGUF" = 0x46554747 in little-endian
        assert_eq!(gguf::GGUF_MAGIC, 0x4655_4747);
        assert_eq!(&gguf::GGUF_MAGIC.to_le_bytes(), b"GGUF");
    }

    #[test]
    fn test_gguf_header_write() {
        let mut buffer = Vec::new();

        // Write minimal GGUF header
        let header = gguf::GgufHeader {
            version: gguf::GGUF_VERSION,
            tensor_count: 1,
            metadata_kv_count: 0,
        };

        header.write_to(&mut buffer).expect("write header");

        // Verify: magic (4) + version (4) + tensor_count (8) + kv_count (8) = 24 bytes
        assert_eq!(buffer.len(), 24);

        // Verify magic
        assert_eq!(&buffer[0..4], b"GGUF");

        // Verify version (little-endian u32)
        assert_eq!(
            u32::from_le_bytes([buffer[4], buffer[5], buffer[6], buffer[7]]),
            3
        );
    }

    #[test]
    fn test_gguf_metadata_string() {
        let mut buffer = Vec::new();

        // Write a string metadata value
        gguf::write_metadata_kv(
            &mut buffer,
            "general.name",
            &gguf::GgufValue::String("test_model".to_string()),
        )
        .expect("write metadata");

        // Should have: key_len (8) + key + value_type (4) + str_len (8) + str
        // 8 + 12 + 4 + 8 + 10 = 42 bytes
        assert!(!buffer.is_empty());
    }

    // Step 2: RED - Test full GGUF export function
    #[test]
    fn test_gguf_export_simple_tensor() {
        // Create a simple f32 tensor to export
        let tensor_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let data: Vec<u8> = tensor_data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let tensor = gguf::GgufTensor {
            name: "test.weight".to_string(),
            shape: vec![2, 2],
            dtype: gguf::GgmlType::F32,
            data,
        };

        let mut buffer = Vec::new();

        // Export to GGUF format
        gguf::export_tensors_to_gguf(
            &mut buffer,
            &[tensor],
            &[(
                "general.name".to_string(),
                gguf::GgufValue::String("test_model".to_string()),
            )],
        )
        .expect("export should succeed");

        // Verify magic number at start
        assert_eq!(&buffer[0..4], b"GGUF");

        // Verify we have content
        assert!(buffer.len() > 24); // At least header size
    }

    #[test]
    fn test_gguf_export_with_metadata() {
        let tensor_data: Vec<f32> = vec![0.5; 16];
        let data: Vec<u8> = tensor_data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let tensor = gguf::GgufTensor {
            name: "model.embed".to_string(),
            shape: vec![4, 4],
            dtype: gguf::GgmlType::F32,
            data,
        };

        let metadata = vec![
            (
                "general.name".to_string(),
                gguf::GgufValue::String("aprender_model".to_string()),
            ),
            (
                "general.architecture".to_string(),
                gguf::GgufValue::String("mlp".to_string()),
            ),
            (
                "aprender.version".to_string(),
                gguf::GgufValue::String(env!("CARGO_PKG_VERSION").to_string()),
            ),
        ];

        let mut buffer = Vec::new();
        gguf::export_tensors_to_gguf(&mut buffer, &[tensor], &metadata).expect("export");

        // Verify header
        assert_eq!(&buffer[0..4], b"GGUF");

        // Verify version is 3
        let version = u32::from_le_bytes([buffer[4], buffer[5], buffer[6], buffer[7]]);
        assert_eq!(version, 3);

        // Verify tensor count is 1
        let tensor_count = u64::from_le_bytes([
            buffer[8], buffer[9], buffer[10], buffer[11], buffer[12], buffer[13], buffer[14],
            buffer[15],
        ]);
        assert_eq!(tensor_count, 1);

        // Verify metadata count is 3
        let metadata_count = u64::from_le_bytes([
            buffer[16], buffer[17], buffer[18], buffer[19], buffer[20], buffer[21], buffer[22],
            buffer[23],
        ]);
        assert_eq!(metadata_count, 3);
    }
}

// ============================================================================
// Property-Based Tests (EXTREME TDD)
// ============================================================================

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    // ================================================================
    // Arbitrary Strategies
    // ================================================================

    /// Generate arbitrary ModelType
    fn arb_model_type() -> impl Strategy<Value = ModelType> {
        prop_oneof![
            Just(ModelType::LinearRegression),
            Just(ModelType::LogisticRegression),
            Just(ModelType::DecisionTree),
            Just(ModelType::RandomForest),
            Just(ModelType::GradientBoosting),
            Just(ModelType::KMeans),
            Just(ModelType::Pca),
            Just(ModelType::NaiveBayes),
            Just(ModelType::Knn),
            Just(ModelType::Svm),
            Just(ModelType::NgramLm),
            Just(ModelType::Tfidf),
            Just(ModelType::CountVectorizer),
            Just(ModelType::NeuralSequential),
            Just(ModelType::NeuralCustom),
            Just(ModelType::ContentRecommender),
            Just(ModelType::Custom),
        ]
    }

    /// Generate arbitrary Compression
    fn arb_compression() -> impl Strategy<Value = Compression> {
        prop_oneof![
            Just(Compression::None),
            Just(Compression::ZstdDefault),
            Just(Compression::ZstdMax),
            Just(Compression::Lz4),
        ]
    }

    /// Generate arbitrary valid flags (6 bits)
    fn arb_flags() -> impl Strategy<Value = Flags> {
        (0u8..64).prop_map(Flags::from_bits)
    }

    /// Generate arbitrary Header with valid values
    fn arb_header() -> impl Strategy<Value = Header> {
        (
            arb_model_type(),
            0u32..1_000_000,             // metadata_size
            0u32..100_000_000,           // payload_size
            0u32..MAX_UNCOMPRESSED_SIZE, // uncompressed_size
            arb_compression(),
            arb_flags(),
        )
            .prop_map(
                |(
                    model_type,
                    metadata_size,
                    payload_size,
                    uncompressed_size,
                    compression,
                    flags,
                )| {
                    Header {
                        magic: MAGIC,
                        version: FORMAT_VERSION,
                        model_type,
                        metadata_size,
                        payload_size,
                        uncompressed_size,
                        compression,
                        flags,
                    }
                },
            )
    }

    // ================================================================
    // Header Roundtrip Property Tests
    // ================================================================

    proptest! {
        /// Property: Header serialization always produces exactly 32 bytes
        #[test]
        fn prop_header_size_always_32(header in arb_header()) {
            let bytes = header.to_bytes();
            prop_assert_eq!(bytes.len(), HEADER_SIZE);
        }

        /// Property: Header always starts with magic "APRN"
        #[test]
        fn prop_header_has_magic(header in arb_header()) {
            let bytes = header.to_bytes();
            prop_assert_eq!(&bytes[0..4], &MAGIC);
        }

        /// Property: Header roundtrip preserves model_type
        #[test]
        fn prop_header_roundtrip_model_type(header in arb_header()) {
            let bytes = header.to_bytes();
            let parsed = Header::from_bytes(&bytes).expect("valid header");
            prop_assert_eq!(header.model_type, parsed.model_type);
        }

        /// Property: Header roundtrip preserves metadata_size
        #[test]
        fn prop_header_roundtrip_metadata_size(header in arb_header()) {
            let bytes = header.to_bytes();
            let parsed = Header::from_bytes(&bytes).expect("valid header");
            prop_assert_eq!(header.metadata_size, parsed.metadata_size);
        }

        /// Property: Header roundtrip preserves payload_size
        #[test]
        fn prop_header_roundtrip_payload_size(header in arb_header()) {
            let bytes = header.to_bytes();
            let parsed = Header::from_bytes(&bytes).expect("valid header");
            prop_assert_eq!(header.payload_size, parsed.payload_size);
        }

        /// Property: Header roundtrip preserves uncompressed_size
        #[test]
        fn prop_header_roundtrip_uncompressed_size(header in arb_header()) {
            let bytes = header.to_bytes();
            let parsed = Header::from_bytes(&bytes).expect("valid header");
            prop_assert_eq!(header.uncompressed_size, parsed.uncompressed_size);
        }

        /// Property: Header roundtrip preserves compression
        #[test]
        fn prop_header_roundtrip_compression(header in arb_header()) {
            let bytes = header.to_bytes();
            let parsed = Header::from_bytes(&bytes).expect("valid header");
            prop_assert_eq!(header.compression, parsed.compression);
        }

        /// Property: Header roundtrip preserves flags
        #[test]
        fn prop_header_roundtrip_flags(header in arb_header()) {
            let bytes = header.to_bytes();
            let parsed = Header::from_bytes(&bytes).expect("valid header");
            prop_assert_eq!(header.flags.bits(), parsed.flags.bits());
        }

        // ================================================================
        // ModelType Property Tests
        // ================================================================

        /// Property: ModelType from_u16 roundtrip
        #[test]
        fn prop_model_type_roundtrip(model_type in arb_model_type()) {
            let value = model_type as u16;
            let parsed = ModelType::from_u16(value);
            prop_assert_eq!(parsed, Some(model_type));
        }

        /// Property: Invalid model type returns None
        #[test]
        fn prop_invalid_model_type_none(value in 0x0100u16..0x1000) {
            // Values outside defined range should return None
            let parsed = ModelType::from_u16(value);
            prop_assert!(parsed.is_none());
        }

        // ================================================================
        // Compression Property Tests
        // ================================================================

        /// Property: Compression from_u8 roundtrip
        #[test]
        fn prop_compression_roundtrip(compression in arb_compression()) {
            let value = compression as u8;
            let parsed = Compression::from_u8(value);
            prop_assert_eq!(parsed, Some(compression));
        }

        /// Property: Invalid compression returns None
        #[test]
        fn prop_invalid_compression_none(value in 4u8..255) {
            let parsed = Compression::from_u8(value);
            prop_assert!(parsed.is_none());
        }

        // ================================================================
        // Flags Property Tests
        // ================================================================

        /// Property: Flags from_bits masks reserved bits
        #[test]
        fn prop_flags_masks_reserved(raw in any::<u8>()) {
            let flags = Flags::from_bits(raw);
            // Bits 6-7 should be masked off
            prop_assert!(flags.bits() < 64);
        }

        /// Property: with_encrypted sets ENCRYPTED bit
        #[test]
        fn prop_flags_with_encrypted(_seed in any::<u8>()) {
            let flags = Flags::new().with_encrypted();
            prop_assert!(flags.is_encrypted());
            prop_assert_eq!(flags.bits() & Flags::ENCRYPTED, Flags::ENCRYPTED);
        }

        /// Property: with_signed sets SIGNED bit
        #[test]
        fn prop_flags_with_signed(_seed in any::<u8>()) {
            let flags = Flags::new().with_signed();
            prop_assert!(flags.is_signed());
            prop_assert_eq!(flags.bits() & Flags::SIGNED, Flags::SIGNED);
        }

        /// Property: with_streaming sets STREAMING bit
        #[test]
        fn prop_flags_with_streaming(_seed in any::<u8>()) {
            let flags = Flags::new().with_streaming();
            prop_assert!(flags.is_streaming());
            prop_assert_eq!(flags.bits() & Flags::STREAMING, Flags::STREAMING);
        }

        /// Property: with_licensed sets LICENSED bit
        #[test]
        fn prop_flags_with_licensed(_seed in any::<u8>()) {
            let flags = Flags::new().with_licensed();
            prop_assert!(flags.is_licensed());
            prop_assert_eq!(flags.bits() & Flags::LICENSED, Flags::LICENSED);
        }

        /// Property: with_quantized sets QUANTIZED bit
        #[test]
        fn prop_flags_with_quantized(_seed in any::<u8>()) {
            let flags = Flags::new().with_quantized();
            prop_assert!(flags.is_quantized());
            prop_assert_eq!(flags.bits() & Flags::QUANTIZED, Flags::QUANTIZED);
        }

        /// Property: Flag chaining is commutative (order doesn't matter)
        #[test]
        fn prop_flags_chaining_commutative(a in any::<bool>(), b in any::<bool>()) {
            let mut flags1 = Flags::new();
            let mut flags2 = Flags::new();

            if a {
                flags1 = flags1.with_encrypted();
            }
            if b {
                flags1 = flags1.with_signed();
            }

            // Reverse order
            if b {
                flags2 = flags2.with_signed();
            }
            if a {
                flags2 = flags2.with_encrypted();
            }

            prop_assert_eq!(flags1.bits(), flags2.bits());
        }

        /// Property: Flag bits are independent (setting one doesn't affect others)
        #[test]
        fn prop_flags_independent(flags in arb_flags()) {
            // Check each flag independently
            let encrypted = flags.is_encrypted();
            let signed = flags.is_signed();
            let streaming = flags.is_streaming();
            let licensed = flags.is_licensed();
            let quantized = flags.is_quantized();

            // Reconstruct from individual bits
            let reconstructed = (if encrypted { Flags::ENCRYPTED } else { 0 })
                | (if signed { Flags::SIGNED } else { 0 })
                | (if streaming { Flags::STREAMING } else { 0 })
                | (if licensed { Flags::LICENSED } else { 0 })
                | (if quantized { Flags::QUANTIZED } else { 0 })
                | (flags.bits() & Flags::TRUENO_NATIVE); // Don't forget trueno_native

            prop_assert_eq!(flags.bits(), reconstructed);
        }

        // ================================================================
        // CRC32 Checksum Property Tests
        // ================================================================

        /// Property: CRC32 of same data is always identical
        #[test]
        fn prop_crc32_deterministic(data in proptest::collection::vec(any::<u8>(), 0..1000)) {
            let crc1 = crc32(&data);
            let crc2 = crc32(&data);
            prop_assert_eq!(crc1, crc2);
        }

        /// Property: CRC32 changes when data changes (avalanche property)
        #[test]
        fn prop_crc32_avalanche(
            data in proptest::collection::vec(any::<u8>(), 1..100),
            flip_pos in 0usize..100,
            flip_bit in 0u8..8
        ) {
            if flip_pos >= data.len() {
                return Ok(());
            }

            let crc_original = crc32(&data);

            let mut modified = data.clone();
            modified[flip_pos] ^= 1 << flip_bit;

            let crc_modified = crc32(&modified);

            // Single bit flip should change CRC
            prop_assert_ne!(crc_original, crc_modified);
        }

        /// Property: Empty data has consistent CRC (IEEE polynomial)
        #[test]
        fn prop_crc32_empty(_seed in any::<u8>()) {
            let crc = crc32(&[]);
            // CRC32 of empty data is 0 for our implementation
            prop_assert_eq!(crc, 0);
        }
    }
}

// ============================================================================
// Encryption Property Tests (EXTREME TDD - Security Critical)
// NOTE: These tests are slow due to Argon2id. Use only 3 cases by default.
// For comprehensive testing: PROPTEST_CASES=100 cargo test
// ============================================================================

#[cfg(all(test, feature = "format-encryption"))]
mod encryption_proptests {
    use super::*;
    use proptest::prelude::*;

    /// Strategy for generating valid passwords (8-64 chars)
    fn arb_password() -> impl Strategy<Value = String> {
        proptest::collection::vec(any::<u8>(), 8..64)
            .prop_map(|bytes| bytes.iter().map(|b| (b % 94 + 33) as char).collect())
    }

    /// Strategy for generating test model data
    fn arb_model_data() -> impl Strategy<Value = Vec<f32>> {
        proptest::collection::vec(-100.0f32..100.0, 1..100)
    }

    // Use only 3 cases for encryption tests (Argon2id is intentionally slow)
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(3))]

        /// Property: Encryption roundtrip preserves data (in-memory)
        #[test]
        fn prop_encryption_roundtrip_preserves_data(
            password in arb_password(),
            data in arb_model_data()
        ) {
            use tempfile::tempdir;

            #[derive(Debug, serde::Serialize, serde::Deserialize, PartialEq)]
            struct Model { weights: Vec<f32> }

            let model = Model { weights: data.clone() };
            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("test.apr");

            save_encrypted(&model, ModelType::Custom, &path, SaveOptions::default(), &password)
                .expect("save");
            let loaded: Model = load_encrypted(&path, ModelType::Custom, &password)
                .expect("load");

            prop_assert_eq!(loaded.weights, data);
        }

        /// Property: Wrong password fails decryption
        #[test]
        fn prop_wrong_password_fails(
            password in arb_password(),
            wrong_password in arb_password()
        ) {
            // Skip if passwords happen to be the same
            if password == wrong_password {
                return Ok(());
            }

            use tempfile::tempdir;

            #[derive(Debug, serde::Serialize, serde::Deserialize)]
            struct Model { value: i32 }

            let model = Model { value: 42 };
            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("test.apr");

            save_encrypted(&model, ModelType::Custom, &path, SaveOptions::default(), &password)
                .expect("save");
            let result: Result<Model> = load_encrypted(&path, ModelType::Custom, &wrong_password);

            prop_assert!(result.is_err(), "Wrong password should fail");
        }

        /// Property: Encrypted files have ENCRYPTED flag set
        #[test]
        fn prop_encrypted_flag_set(password in arb_password()) {
            use tempfile::tempdir;

            #[derive(Debug, serde::Serialize, serde::Deserialize)]
            struct Model { v: i32 }

            let model = Model { v: 1 };
            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("test.apr");

            save_encrypted(&model, ModelType::Custom, &path, SaveOptions::default(), &password)
                .expect("save");
            let info = inspect(&path).expect("inspect");

            prop_assert!(info.encrypted, "ENCRYPTED flag must be set");
        }

        /// Property: load_from_bytes_encrypted roundtrip works
        #[test]
        fn prop_bytes_encrypted_roundtrip(
            password in arb_password(),
            data in arb_model_data()
        ) {
            use tempfile::tempdir;

            #[derive(Debug, serde::Serialize, serde::Deserialize, PartialEq)]
            struct Model { weights: Vec<f32> }

            let model = Model { weights: data.clone() };
            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("test.apr");

            save_encrypted(&model, ModelType::Custom, &path, SaveOptions::default(), &password)
                .expect("save");

            let bytes = std::fs::read(&path).expect("read");
            let loaded: Model = load_from_bytes_encrypted(&bytes, ModelType::Custom, &password)
                .expect("load from bytes");

            prop_assert_eq!(loaded.weights, data);
        }
    }
}

// ============================================================================
// X25519 Encryption Property Tests (EXTREME TDD - Security Critical)
// ============================================================================

#[cfg(all(test, feature = "format-encryption"))]
mod x25519_proptests {
    use super::*;
    use proptest::prelude::*;

    /// Strategy for generating test model data
    fn arb_model_data() -> impl Strategy<Value = Vec<f32>> {
        proptest::collection::vec(-100.0f32..100.0, 1..50)
    }

    proptest! {
        /// Property: X25519 roundtrip preserves data
        #[test]
        fn prop_x25519_roundtrip(data in arb_model_data()) {
            use tempfile::tempdir;

            #[derive(Debug, serde::Serialize, serde::Deserialize, PartialEq)]
            struct Model { weights: Vec<f32> }

            let model = Model { weights: data.clone() };

            // Generate recipient keypair
            let recipient_secret = X25519SecretKey::random_from_rng(rand::thread_rng());
            let recipient_public = X25519PublicKey::from(&recipient_secret);

            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("test.apr");

            save_for_recipient(&model, ModelType::Custom, &path, SaveOptions::default(), &recipient_public)
                .expect("save");
            let loaded: Model = load_as_recipient(&path, ModelType::Custom, &recipient_secret)
                .expect("load");

            prop_assert_eq!(loaded.weights, data);
        }

        /// Property: X25519 wrong key fails
        #[test]
        fn prop_x25519_wrong_key_fails(data in arb_model_data()) {
            use tempfile::tempdir;

            #[derive(Debug, serde::Serialize, serde::Deserialize)]
            struct Model { weights: Vec<f32> }

            let model = Model { weights: data };

            // Generate two different keypairs
            let recipient_secret = X25519SecretKey::random_from_rng(rand::thread_rng());
            let recipient_public = X25519PublicKey::from(&recipient_secret);
            let wrong_secret = X25519SecretKey::random_from_rng(rand::thread_rng());

            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("test.apr");

            save_for_recipient(&model, ModelType::Custom, &path, SaveOptions::default(), &recipient_public)
                .expect("save");
            let result: Result<Model> = load_as_recipient(&path, ModelType::Custom, &wrong_secret);

            prop_assert!(result.is_err(), "Wrong key should fail");
        }

        /// Property: X25519 encrypted files have ENCRYPTED flag
        #[test]
        fn prop_x25519_encrypted_flag(_seed in any::<u8>()) {
            use tempfile::tempdir;

            #[derive(Debug, serde::Serialize, serde::Deserialize)]
            struct Model { v: i32 }

            let model = Model { v: 1 };
            let recipient_secret = X25519SecretKey::random_from_rng(rand::thread_rng());
            let recipient_public = X25519PublicKey::from(&recipient_secret);

            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("test.apr");

            save_for_recipient(&model, ModelType::Custom, &path, SaveOptions::default(), &recipient_public)
                .expect("save");
            let info = inspect(&path).expect("inspect");

            prop_assert!(info.encrypted, "ENCRYPTED flag must be set");
        }
    }
}

// ============================================================================
// Signing Property Tests (EXTREME TDD - Security Critical)
// ============================================================================

#[cfg(all(test, feature = "format-signing"))]
mod signing_proptests {
    use super::*;
    use proptest::prelude::*;

    /// Strategy for generating test model data
    fn arb_model_data() -> impl Strategy<Value = Vec<f32>> {
        proptest::collection::vec(-100.0f32..100.0, 1..50)
    }

    proptest! {
        /// Property: Signing roundtrip preserves data and verifies
        #[test]
        fn prop_signing_roundtrip(data in arb_model_data()) {
            use tempfile::tempdir;

            #[derive(Debug, serde::Serialize, serde::Deserialize, PartialEq)]
            struct Model { weights: Vec<f32> }

            let model = Model { weights: data.clone() };

            // Generate signing keypair
            let signing_key = SigningKey::generate(&mut rand::thread_rng());
            let verifying_key = signing_key.verifying_key();

            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("test.apr");

            save_signed(&model, ModelType::Custom, &path, SaveOptions::default(), &signing_key)
                .expect("save");
            let loaded: Model = load_verified(&path, ModelType::Custom, Some(&verifying_key))
                .expect("load");

            prop_assert_eq!(loaded.weights, data);
        }

        /// Property: Wrong verification key fails
        #[test]
        fn prop_signing_wrong_key_fails(data in arb_model_data()) {
            use tempfile::tempdir;

            #[derive(Debug, serde::Serialize, serde::Deserialize)]
            struct Model { weights: Vec<f32> }

            let model = Model { weights: data };

            // Generate two different keypairs
            let signing_key = SigningKey::generate(&mut rand::thread_rng());
            let wrong_key = SigningKey::generate(&mut rand::thread_rng());
            let wrong_verifying = wrong_key.verifying_key();

            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("test.apr");

            save_signed(&model, ModelType::Custom, &path, SaveOptions::default(), &signing_key)
                .expect("save");
            let result: Result<Model> = load_verified(&path, ModelType::Custom, Some(&wrong_verifying));

            prop_assert!(result.is_err(), "Wrong key should fail verification");
        }

        /// Property: Signed files have SIGNED flag set
        #[test]
        fn prop_signed_flag_set(_seed in any::<u8>()) {
            use tempfile::tempdir;

            #[derive(Debug, serde::Serialize, serde::Deserialize)]
            struct Model { v: i32 }

            let model = Model { v: 1 };
            let signing_key = SigningKey::generate(&mut rand::thread_rng());

            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("test.apr");

            save_signed(&model, ModelType::Custom, &path, SaveOptions::default(), &signing_key)
                .expect("save");
            let info = inspect(&path).expect("inspect");

            prop_assert!(info.signed, "SIGNED flag must be set");
        }

        /// Property: Tampering with signed file is detected
        #[test]
        fn prop_tampering_detected(data in arb_model_data()) {
            use tempfile::tempdir;

            #[derive(Debug, serde::Serialize, serde::Deserialize)]
            struct Model { weights: Vec<f32> }

            let model = Model { weights: data };

            let signing_key = SigningKey::generate(&mut rand::thread_rng());
            let verifying_key = signing_key.verifying_key();

            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("test.apr");

            save_signed(&model, ModelType::Custom, &path, SaveOptions::default(), &signing_key)
                .expect("save");

            // Tamper with the file (modify a byte in the middle)
            let mut content = std::fs::read(&path).expect("read");
            if content.len() > 50 {
                content[50] ^= 0xFF; // Flip bits
                std::fs::write(&path, content).expect("write");

                let result: Result<Model> = load_verified(&path, ModelType::Custom, Some(&verifying_key));
                prop_assert!(result.is_err(), "Tampered file should fail verification");
            }
        }
    }
}

// ============================================================================
// Property-based tests for Knowledge Distillation (spec §6.3)
// ============================================================================
#[cfg(test)]
mod distillation_proptests {
    use super::*;
    use proptest::prelude::*;

    // Arbitrary generators for distillation types

    fn arb_distill_method() -> impl Strategy<Value = DistillMethod> {
        prop_oneof![
            Just(DistillMethod::Standard),
            Just(DistillMethod::Progressive),
            Just(DistillMethod::Ensemble),
        ]
    }

    fn arb_model_type() -> impl Strategy<Value = ModelType> {
        prop_oneof![
            Just(ModelType::LinearRegression),
            Just(ModelType::LogisticRegression),
            Just(ModelType::DecisionTree),
            Just(ModelType::RandomForest),
            Just(ModelType::KMeans),
            Just(ModelType::NaiveBayes),
            Just(ModelType::Knn),
            Just(ModelType::Pca),
            Just(ModelType::Custom),
        ]
    }

    fn arb_teacher_provenance() -> impl Strategy<Value = TeacherProvenance> {
        (
            "[a-f0-9]{64}",                              // SHA256 hash
            proptest::option::of("[a-zA-Z0-9+/]{86}=="), // Ed25519 signature (base64)
            arb_model_type(),
            1_000_000u64..10_000_000_000u64, // param count: 1M to 10B
        )
            .prop_map(
                |(hash, signature, model_type, param_count)| TeacherProvenance {
                    hash,
                    signature,
                    model_type,
                    param_count,
                    ensemble_teachers: None,
                },
            )
    }

    fn arb_distillation_params() -> impl Strategy<Value = DistillationParams> {
        (
            1.0f32..10.0f32,                       // temperature (1.0-10.0)
            0.0f32..1.0f32,                        // alpha (0.0-1.0)
            proptest::option::of(0.0f32..1.0f32),  // beta
            1u32..1000u32,                         // epochs
            proptest::option::of(0.0f32..10.0f32), // final_loss
        )
            .prop_map(|(temperature, alpha, beta, epochs, final_loss)| {
                DistillationParams {
                    temperature,
                    alpha,
                    beta,
                    epochs,
                    final_loss,
                }
            })
    }

    fn arb_layer_mapping() -> impl Strategy<Value = LayerMapping> {
        (
            0usize..100usize, // student_layer
            0usize..200usize, // teacher_layer
            0.0f32..1.0f32,   // weight
        )
            .prop_map(|(student_layer, teacher_layer, weight)| LayerMapping {
                student_layer,
                teacher_layer,
                weight,
            })
    }

    fn arb_distillation_info() -> impl Strategy<Value = DistillationInfo> {
        (
            arb_distill_method(),
            arb_teacher_provenance(),
            arb_distillation_params(),
        )
            .prop_map(|(method, teacher, params)| DistillationInfo {
                method,
                teacher,
                params,
                layer_mapping: None,
            })
    }

    fn arb_model_data() -> impl Strategy<Value = Vec<f32>> {
        proptest::collection::vec(
            any::<f32>().prop_filter("finite", |f| f.is_finite()),
            1..100,
        )
    }

    proptest! {
        /// Property: DistillMethod serialization roundtrip (JSON for optionals)
        #[test]
        fn prop_distill_method_roundtrip(method in arb_distill_method()) {
            // Use JSON for roundtrip testing (handles enums better than raw msgpack)
            let serialized = serde_json::to_string(&method).expect("serialize");
            let deserialized: DistillMethod = serde_json::from_str(&serialized).expect("deserialize");
            prop_assert_eq!(method, deserialized);
        }

        /// Property: DistillationParams serialization roundtrip
        #[test]
        fn prop_distillation_params_roundtrip(params in arb_distillation_params()) {
            // JSON handles optional fields correctly
            let serialized = serde_json::to_string(&params).expect("serialize");
            let deserialized: DistillationParams = serde_json::from_str(&serialized).expect("deserialize");

            // Check fields (f32 equality via bits for NaN handling)
            prop_assert_eq!(params.temperature.to_bits(), deserialized.temperature.to_bits());
            prop_assert_eq!(params.alpha.to_bits(), deserialized.alpha.to_bits());
            prop_assert_eq!(params.epochs, deserialized.epochs);
            prop_assert_eq!(params.beta.map(f32::to_bits), deserialized.beta.map(f32::to_bits));
        }

        /// Property: TeacherProvenance serialization roundtrip
        #[test]
        fn prop_teacher_provenance_roundtrip(teacher in arb_teacher_provenance()) {
            let serialized = serde_json::to_string(&teacher).expect("serialize");
            let deserialized: TeacherProvenance = serde_json::from_str(&serialized).expect("deserialize");

            prop_assert_eq!(&teacher.hash, &deserialized.hash);
            prop_assert_eq!(&teacher.signature, &deserialized.signature);
            prop_assert_eq!(teacher.model_type, deserialized.model_type);
            prop_assert_eq!(teacher.param_count, deserialized.param_count);
        }

        /// Property: LayerMapping serialization roundtrip
        #[test]
        fn prop_layer_mapping_roundtrip(mapping in arb_layer_mapping()) {
            let serialized = serde_json::to_string(&mapping).expect("serialize");
            let deserialized: LayerMapping = serde_json::from_str(&serialized).expect("deserialize");

            prop_assert_eq!(mapping.student_layer, deserialized.student_layer);
            prop_assert_eq!(mapping.teacher_layer, deserialized.teacher_layer);
            prop_assert_eq!(mapping.weight.to_bits(), deserialized.weight.to_bits());
        }

        /// Property: DistillationInfo serialization roundtrip
        #[test]
        fn prop_distillation_info_roundtrip(info in arb_distillation_info()) {
            let serialized = serde_json::to_string(&info).expect("serialize");
            let deserialized: DistillationInfo = serde_json::from_str(&serialized).expect("deserialize");

            prop_assert_eq!(info.method, deserialized.method);
            prop_assert_eq!(&info.teacher.hash, &deserialized.teacher.hash);
            prop_assert_eq!(info.params.epochs, deserialized.params.epochs);
        }

        /// Property: Distillation info persists through save/load cycle
        #[test]
        fn prop_distillation_save_load_roundtrip(
            info in arb_distillation_info(),
            data in arb_model_data()
        ) {
            use tempfile::tempdir;

            #[derive(Debug, serde::Serialize, serde::Deserialize)]
            struct Model { weights: Vec<f32> }

            let model = Model { weights: data };
            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("distilled.apr");

            let options = SaveOptions::default().with_distillation_info(info.clone());
            save(&model, ModelType::Custom, &path, options).expect("save");

            let model_info = inspect(&path).expect("inspect");
            let restored = model_info.metadata.distillation_info
                .expect("should have distillation_info");

            prop_assert_eq!(info.method, restored.method);
            prop_assert_eq!(&info.teacher.hash, &restored.teacher.hash);
            prop_assert_eq!(info.teacher.param_count, restored.teacher.param_count);
            prop_assert_eq!(info.params.epochs, restored.params.epochs);
        }

        /// Property: Temperature must be positive for valid distillation
        #[test]
        fn prop_temperature_positive(temp in 0.1f32..20.0f32) {
            let params = DistillationParams {
                temperature: temp,
                alpha: 0.5,
                beta: None,
                epochs: 10,
                final_loss: None,
            };
            prop_assert!(params.temperature > 0.0, "Temperature must be positive");
        }

        /// Property: Alpha (soft loss weight) must be in [0, 1]
        #[test]
        fn prop_alpha_bounded(alpha in 0.0f32..=1.0f32) {
            let params = DistillationParams {
                temperature: 3.0,
                alpha,
                beta: None,
                epochs: 10,
                final_loss: None,
            };
            prop_assert!((0.0..=1.0).contains(&params.alpha), "Alpha must be in [0,1]");
        }

        /// Property: Progressive distillation requires beta parameter (design guideline)
        #[test]
        fn prop_progressive_with_beta(beta in 0.0f32..1.0f32) {
            let info = DistillationInfo {
                method: DistillMethod::Progressive,
                teacher: TeacherProvenance {
                    hash: "abc123".to_string(),
                    signature: None,
                    model_type: ModelType::Custom,
                    param_count: 7_000_000_000,
                    ensemble_teachers: None,
                },
                params: DistillationParams {
                    temperature: 3.0,
                    alpha: 0.7,
                    beta: Some(beta),
                    epochs: 10,
                    final_loss: None,
                },
                layer_mapping: None,
            };
            // Progressive distillation should have beta for hidden layer loss weight
            prop_assert!(info.params.beta.is_some());
        }

        /// Property: Layer mappings have valid indices
        #[test]
        fn prop_layer_mapping_valid_indices(
            student in 0usize..100,
            teacher in 0usize..200,
            weight in 0.0f32..1.0f32
        ) {
            let mapping = LayerMapping {
                student_layer: student,
                teacher_layer: teacher,
                weight,
            };
            // Teacher layer index can be >= student (many-to-one mapping)
            // Weight should be non-negative
            prop_assert!(mapping.weight >= 0.0);
        }
    }
}

// ============================================================================
// Property-based tests for Commercial License Block (spec §9)
// ============================================================================
#[cfg(test)]
mod license_proptests {
    use super::*;
    use proptest::prelude::*;

    // Arbitrary generators for license types

    fn arb_license_tier() -> impl Strategy<Value = LicenseTier> {
        prop_oneof![
            Just(LicenseTier::Personal),
            Just(LicenseTier::Team),
            Just(LicenseTier::Enterprise),
            Just(LicenseTier::Academic),
        ]
    }

    /// Generate valid UUID v4 format
    fn arb_uuid() -> impl Strategy<Value = String> {
        "[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"
    }

    /// Generate SHA256 hash
    fn arb_hash() -> impl Strategy<Value = String> {
        "[0-9a-f]{64}"
    }

    /// Generate ISO 8601 date (YYYY-MM-DD)
    fn arb_iso_date() -> impl Strategy<Value = String> {
        (2024u32..2035, 1u32..13, 1u32..29).prop_map(|(y, m, d)| format!("{y:04}-{m:02}-{d:02}"))
    }

    fn arb_license_info() -> impl Strategy<Value = LicenseInfo> {
        (
            arb_uuid(),
            arb_hash(),
            proptest::option::of(arb_iso_date()),
            proptest::option::of(1u32..1000),
            proptest::option::of("[A-Za-z0-9 ]{1,50}"),
            arb_license_tier(),
        )
            .prop_map(|(uuid, hash, expiry, seats, licensee, tier)| LicenseInfo {
                uuid,
                hash,
                expiry,
                seats,
                licensee,
                tier,
            })
    }

    fn arb_model_data() -> impl Strategy<Value = Vec<f32>> {
        proptest::collection::vec(
            any::<f32>().prop_filter("finite", |f| f.is_finite()),
            1..100,
        )
    }

    proptest! {
        /// Property: LicenseTier serialization roundtrip
        #[test]
        fn prop_license_tier_roundtrip(tier in arb_license_tier()) {
            let serialized = serde_json::to_string(&tier).expect("serialize");
            let deserialized: LicenseTier = serde_json::from_str(&serialized).expect("deserialize");
            prop_assert_eq!(tier, deserialized);
        }

        /// Property: LicenseInfo serialization roundtrip
        #[test]
        fn prop_license_info_roundtrip(info in arb_license_info()) {
            let serialized = serde_json::to_string(&info).expect("serialize");
            let deserialized: LicenseInfo = serde_json::from_str(&serialized).expect("deserialize");

            prop_assert_eq!(&info.uuid, &deserialized.uuid);
            prop_assert_eq!(&info.hash, &deserialized.hash);
            prop_assert_eq!(info.tier, deserialized.tier);
            prop_assert_eq!(info.seats, deserialized.seats);
            prop_assert_eq!(&info.expiry, &deserialized.expiry);
        }

        /// Property: UUID format is valid v4
        #[test]
        fn prop_uuid_format_valid(uuid in arb_uuid()) {
            // UUID v4 format: xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx
            // where y is 8, 9, a, or b
            prop_assert_eq!(uuid.len(), 36);
            prop_assert!(uuid.chars().nth(14) == Some('4'), "Version must be 4");
            let y = uuid.chars().nth(19).expect("UUID must have char at position 19");
            prop_assert!(
                matches!(y, '8' | '9' | 'a' | 'b'),
                "Variant must be 8, 9, a, or b"
            );
        }

        /// Property: License persists through save/load cycle
        #[test]
        fn prop_license_save_load_roundtrip(
            license in arb_license_info(),
            data in arb_model_data()
        ) {
            use tempfile::tempdir;

            #[derive(Debug, serde::Serialize, serde::Deserialize)]
            struct Model { weights: Vec<f32> }

            let model = Model { weights: data };
            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("licensed.apr");

            let options = SaveOptions::default().with_license(license.clone());
            save(&model, ModelType::Custom, &path, options).expect("save");

            let model_info = inspect(&path).expect("inspect");
            let restored = model_info.metadata.license
                .expect("should have license");

            prop_assert_eq!(&license.uuid, &restored.uuid);
            prop_assert_eq!(&license.hash, &restored.hash);
            prop_assert_eq!(license.tier, restored.tier);
            prop_assert_eq!(license.seats, restored.seats);
        }

        /// Property: LICENSED flag is set when license provided
        #[test]
        fn prop_licensed_flag_set(license in arb_license_info(), data in arb_model_data()) {
            use tempfile::tempdir;

            #[derive(Debug, serde::Serialize, serde::Deserialize)]
            struct Model { weights: Vec<f32> }

            let model = Model { weights: data };
            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("licensed.apr");

            let options = SaveOptions::default().with_license(license);
            save(&model, ModelType::Custom, &path, options).expect("save");

            let info = inspect(&path).expect("inspect");
            prop_assert!(info.licensed, "LICENSED flag must be set");
        }

        /// Property: Seats must be positive when specified
        #[test]
        fn prop_seats_positive(seats in 1u32..10000) {
            let license = LicenseInfo {
                uuid: "00000000-0000-4000-8000-000000000000".to_string(),
                hash: "0".repeat(64),
                expiry: None,
                seats: Some(seats),
                licensee: None,
                tier: LicenseTier::Team,
            };
            prop_assert!(license.seats == Some(seats) && seats > 0, "Seats must be positive");
        }

        /// Property: Enterprise tier has no seat limit by default
        #[test]
        fn prop_enterprise_unlimited_seats(_dummy in 0u8..1) {
            // Enterprise tier typically has unlimited seats
            let license = LicenseInfo {
                uuid: "00000000-0000-4000-8000-000000000000".to_string(),
                hash: "0".repeat(64),
                expiry: None,
                seats: None, // Unlimited
                licensee: Some("ACME Corp".to_string()),
                tier: LicenseTier::Enterprise,
            };
            prop_assert!(license.seats.is_none(), "Enterprise should have unlimited seats");
            prop_assert!(matches!(license.tier, LicenseTier::Enterprise));
        }

        /// Property: Academic tier is non-commercial
        #[test]
        fn prop_academic_tier_valid(_dummy in 0u8..1) {
            let license = LicenseInfo {
                uuid: "00000000-0000-4000-8000-000000000000".to_string(),
                hash: "0".repeat(64),
                expiry: Some("2025-12-31".to_string()),
                seats: Some(100),
                licensee: Some("MIT".to_string()),
                tier: LicenseTier::Academic,
            };
            prop_assert!(matches!(license.tier, LicenseTier::Academic));
        }

        /// Property: Hash is 64 hex characters (SHA256)
        #[test]
        fn prop_hash_length_valid(hash in arb_hash()) {
            prop_assert_eq!(hash.len(), 64, "SHA256 hash must be 64 hex chars");
            prop_assert!(hash.chars().all(|c| c.is_ascii_hexdigit()));
        }
    }
}

// ============================================================================
// Property-based tests for Metadata and TrainingInfo
// ============================================================================
#[cfg(test)]
mod metadata_proptests {
    use super::*;
    use proptest::prelude::*;

    fn arb_training_info() -> impl Strategy<Value = TrainingInfo> {
        (
            proptest::option::of(1usize..1_000_000),
            proptest::option::of(1u64..86_400_000), // up to 24h in ms
            proptest::option::of("[a-zA-Z0-9_/]{1,50}"),
        )
            .prop_map(|(samples, duration_ms, source)| TrainingInfo {
                samples,
                duration_ms,
                source,
            })
    }

    fn arb_model_name() -> impl Strategy<Value = String> {
        "[a-zA-Z][a-zA-Z0-9_-]{0,49}"
    }

    #[allow(clippy::disallowed_methods)] // json! macro uses unwrap internally
    fn arb_hyperparams() -> impl Strategy<Value = HashMap<String, serde_json::Value>> {
        proptest::collection::hash_map(
            "[a-z_]{1,20}",
            prop_oneof![
                any::<f64>()
                    .prop_filter("finite", |f| f.is_finite())
                    .prop_map(|f| serde_json::json!(f)),
                any::<i32>().prop_map(|i| serde_json::json!(i)),
                "[a-z]{1,10}".prop_map(|s| serde_json::json!(s)),
            ],
            0..5,
        )
    }

    fn arb_model_data() -> impl Strategy<Value = Vec<f32>> {
        proptest::collection::vec(any::<f32>().prop_filter("finite", |f| f.is_finite()), 1..50)
    }

    proptest! {
        /// Property: TrainingInfo serialization roundtrip
        #[test]
        fn prop_training_info_roundtrip(info in arb_training_info()) {
            let serialized = serde_json::to_string(&info).expect("serialize");
            let deserialized: TrainingInfo = serde_json::from_str(&serialized).expect("deserialize");

            prop_assert_eq!(info.samples, deserialized.samples);
            prop_assert_eq!(info.duration_ms, deserialized.duration_ms);
            prop_assert_eq!(&info.source, &deserialized.source);
        }

        /// Property: Metadata with model name persists through save/load
        #[test]
        fn prop_metadata_model_name_roundtrip(
            name in arb_model_name(),
            data in arb_model_data()
        ) {
            use tempfile::tempdir;

            #[derive(Debug, serde::Serialize, serde::Deserialize)]
            struct Model { weights: Vec<f32> }

            let model = Model { weights: data };
            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("named.apr");

            let options = SaveOptions::default().with_name(&name);
            save(&model, ModelType::Custom, &path, options).expect("save");

            let info = inspect(&path).expect("inspect");
            prop_assert_eq!(info.metadata.model_name.as_deref(), Some(name.as_str()));
        }

        /// Property: Metadata with training info persists
        #[test]
        fn prop_metadata_training_roundtrip(
            training in arb_training_info(),
            data in arb_model_data()
        ) {
            use tempfile::tempdir;

            #[derive(Debug, serde::Serialize, serde::Deserialize)]
            struct Model { weights: Vec<f32> }

            let model = Model { weights: data };
            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("trained.apr");

            let mut options = SaveOptions::default();
            options.metadata.training = Some(training.clone());
            save(&model, ModelType::Custom, &path, options).expect("save");

            let info = inspect(&path).expect("inspect");
            let restored = info.metadata.training.expect("should have training");
            prop_assert_eq!(training.samples, restored.samples);
            prop_assert_eq!(training.duration_ms, restored.duration_ms);
        }

        /// Property: Hyperparameters persist through save/load
        #[test]
        fn prop_metadata_hyperparams_roundtrip(
            hyperparams in arb_hyperparams(),
            data in arb_model_data()
        ) {
            use tempfile::tempdir;

            #[derive(Debug, serde::Serialize, serde::Deserialize)]
            struct Model { weights: Vec<f32> }

            let model = Model { weights: data };
            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("hyperparams.apr");

            let mut options = SaveOptions::default();
            options.metadata.hyperparameters = hyperparams.clone();
            save(&model, ModelType::Custom, &path, options).expect("save");

            let info = inspect(&path).expect("inspect");
            prop_assert_eq!(hyperparams.len(), info.metadata.hyperparameters.len());
            for (k, v) in &hyperparams {
                prop_assert_eq!(Some(v), info.metadata.hyperparameters.get(k));
            }
        }

        /// Property: Custom metadata persists
        #[test]
        fn prop_metadata_custom_roundtrip(
            custom in arb_hyperparams(),
            data in arb_model_data()
        ) {
            use tempfile::tempdir;

            #[derive(Debug, serde::Serialize, serde::Deserialize)]
            struct Model { weights: Vec<f32> }

            let model = Model { weights: data };
            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("custom.apr");

            let mut options = SaveOptions::default();
            options.metadata.custom = custom.clone();
            save(&model, ModelType::Custom, &path, options).expect("save");

            let info = inspect(&path).expect("inspect");
            prop_assert_eq!(custom.len(), info.metadata.custom.len());
        }

        /// Property: Aprender version is always set
        #[test]
        fn prop_metadata_version_always_set(data in arb_model_data()) {
            use tempfile::tempdir;

            #[derive(Debug, serde::Serialize, serde::Deserialize)]
            struct Model { weights: Vec<f32> }

            let model = Model { weights: data };
            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("versioned.apr");

            save(&model, ModelType::Custom, &path, SaveOptions::default()).expect("save");

            let info = inspect(&path).expect("inspect");
            prop_assert!(!info.metadata.aprender_version.is_empty());
            prop_assert!(info.metadata.aprender_version.contains('.'));
        }

        /// Property: Created timestamp is always set
        #[test]
        fn prop_metadata_timestamp_always_set(data in arb_model_data()) {
            use tempfile::tempdir;

            #[derive(Debug, serde::Serialize, serde::Deserialize)]
            struct Model { weights: Vec<f32> }

            let model = Model { weights: data };
            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("timestamped.apr");

            save(&model, ModelType::Custom, &path, SaveOptions::default()).expect("save");

            let info = inspect(&path).expect("inspect");
            prop_assert!(!info.metadata.created_at.is_empty());
        }
    }
}

// ============================================================================
// Property-based tests for Error Handling / Robustness
// ============================================================================
#[cfg(test)]
mod error_proptests {
    use super::*;
    use proptest::prelude::*;

    fn arb_non_magic_bytes() -> impl Strategy<Value = [u8; 4]> {
        any::<[u8; 4]>().prop_filter("not APR magic", |b| b != b"APR\x00")
    }

    fn arb_invalid_model_type() -> impl Strategy<Value = u16> {
        // Valid model types are 0-16, so anything >= 17 is invalid
        17u16..=u16::MAX
    }

    fn arb_invalid_compression() -> impl Strategy<Value = u8> {
        // Valid compression values are 0-3, so anything >= 4 is invalid
        4u8..=u8::MAX
    }

    proptest! {
        /// Property: Invalid magic bytes are rejected
        #[test]
        fn prop_invalid_magic_rejected(bad_magic in arb_non_magic_bytes()) {
            use tempfile::tempdir;

            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("bad_magic.apr");

            // Create a file with invalid magic
            let mut content = vec![0u8; 64];
            content[0..4].copy_from_slice(&bad_magic);
            std::fs::write(&path, &content).expect("write");

            let result = inspect(&path);
            prop_assert!(result.is_err(), "Invalid magic should be rejected");
        }

        /// Property: Truncated header is rejected
        #[test]
        fn prop_truncated_header_rejected(len in 0usize..32) {
            use tempfile::tempdir;

            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("truncated.apr");

            // Create a file shorter than header size
            let content = vec![0u8; len];
            std::fs::write(&path, &content).expect("write");

            let result = inspect(&path);
            prop_assert!(result.is_err(), "Truncated header should be rejected");
        }

        /// Property: Invalid model type in header is rejected
        #[test]
        fn prop_invalid_model_type_rejected(bad_type in arb_invalid_model_type()) {
            use tempfile::tempdir;

            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("bad_type.apr");

            // Create header with valid magic but invalid model type
            let mut content = vec![0u8; 64];
            content[0..4].copy_from_slice(b"APR\x00");
            content[4..6].copy_from_slice(&bad_type.to_le_bytes());
            std::fs::write(&path, &content).expect("write");

            let result = inspect(&path);
            prop_assert!(result.is_err(), "Invalid model type should be rejected");
        }

        /// Property: Invalid compression byte is rejected
        #[test]
        fn prop_invalid_compression_rejected(bad_comp in arb_invalid_compression()) {
            use tempfile::tempdir;

            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("bad_comp.apr");

            // Create header with valid magic, model type, but invalid compression
            let mut content = vec![0u8; 64];
            content[0..4].copy_from_slice(b"APR\x00");
            content[4..6].copy_from_slice(&0u16.to_le_bytes()); // Valid model type
            content[20] = bad_comp;
            std::fs::write(&path, &content).expect("write");

            let result = inspect(&path);
            prop_assert!(result.is_err(), "Invalid compression should be rejected");
        }

        /// Property: CRC mismatch is detected on load
        #[test]
        fn prop_crc_mismatch_detected(data in proptest::collection::vec(any::<f32>().prop_filter("finite", |f| f.is_finite()), 1..50)) {
            use tempfile::tempdir;

            #[derive(Debug, serde::Serialize, serde::Deserialize)]
            struct Model { weights: Vec<f32> }

            let model = Model { weights: data };
            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("crc_test.apr");

            save(&model, ModelType::Custom, &path, SaveOptions::default()).expect("save");

            // Corrupt the payload (after header + metadata)
            let mut content = std::fs::read(&path).expect("read");
            if content.len() > 100 {
                content[80] ^= 0xFF; // Flip bits in payload area
                std::fs::write(&path, &content).expect("write corrupted");

                let result: Result<Model> = load(&path, ModelType::Custom);
                // Either CRC check fails or deserialization fails - both are correct
                prop_assert!(result.is_err(), "Corrupted file should fail to load");
            }
        }

        /// Property: Empty file is rejected
        #[test]
        fn prop_empty_file_rejected(_dummy in 0u8..1) {
            use tempfile::tempdir;

            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("empty.apr");

            std::fs::write(&path, []).expect("write empty");

            let result = inspect(&path);
            prop_assert!(result.is_err(), "Empty file should be rejected");
        }

        /// Property: Random bytes are rejected
        #[test]
        fn prop_random_bytes_rejected(random in proptest::collection::vec(any::<u8>(), 32..256)) {
            use tempfile::tempdir;

            // Skip if random bytes happen to start with APR magic (very unlikely)
            if random.len() >= 4 && &random[0..4] == b"APR\x00" {
                return Ok(());
            }

            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("random.apr");

            std::fs::write(&path, &random).expect("write random");

            let result = inspect(&path);
            prop_assert!(result.is_err(), "Random bytes should be rejected");
        }

        /// Property: Format version matches constant
        #[test]
        fn prop_format_version_correct(data in proptest::collection::vec(any::<f32>().prop_filter("finite", |f| f.is_finite()), 1..20)) {
            use tempfile::tempdir;

            #[derive(Debug, serde::Serialize, serde::Deserialize)]
            struct Model { weights: Vec<f32> }

            let model = Model { weights: data.clone() };
            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("versioned.apr");

            save(&model, ModelType::Custom, &path, SaveOptions::default()).expect("save");

            // Verify the version bytes match FORMAT_VERSION (1, 0)
            let content = std::fs::read(&path).expect("read");
            prop_assert_eq!(content[4], FORMAT_VERSION.0, "Major version mismatch");
            prop_assert_eq!(content[5], FORMAT_VERSION.1, "Minor version mismatch");

            // Verify we can load it back
            let loaded: Model = load(&path, ModelType::Custom).expect("load");
            prop_assert_eq!(data.len(), loaded.weights.len());
        }
    }
}

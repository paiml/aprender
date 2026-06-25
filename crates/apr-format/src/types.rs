//! v1 (`APRN`) container type definitions — spike slice for issue #2231 Stage 1.
//!
//! This is the **representative cut** moved out of `aprender-core/src/format/`
//! (`types.rs` + `spec.rs`) to prove the error-seam works at a crate boundary.
//! It uses [`crate::error::AprFormatError`] instead of `aprender_core::AprenderError`.
//! `aprender-core` wraps these errors via `impl From<AprFormatError> for AprenderError`
//! and re-exports the moved API so existing `aprender_core::format::*` paths keep working.

use crate::error::{AprFormatError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Magic number: "APRN" in ASCII (0x4150524E).
pub const MAGIC: [u8; 4] = [0x41, 0x50, 0x52, 0x4E];

/// Current v1 format version (1.0).
pub const FORMAT_VERSION: (u8, u8) = (1, 0);

/// v1 header size in bytes.
pub const HEADER_SIZE: usize = 32;

/// Maximum uncompressed size (1GB safety limit — compression-bomb protection).
pub const MAX_UNCOMPRESSED_SIZE: u32 = 1024 * 1024 * 1024;

/// Model type identifiers.
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
    /// Mixture of Experts (sparse/dense `MoE`)
    MixtureOfExperts = 0x0040,
    /// User-defined model
    Custom = 0x00FF,
}

impl ModelType {
    /// Convert from u16 value.
    #[must_use]
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
            0x0040 => Some(Self::MixtureOfExperts),
            0x00FF => Some(Self::Custom),
            _ => None,
        }
    }
}

/// Compression algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum Compression {
    /// No compression (debugging / Genchi Genbutsu)
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
    /// Convert from u8 value.
    #[must_use]
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

/// Feature flags (bitmask) — spec §3.2.
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
    /// 64-byte aligned tensors for zero-copy SIMD
    pub const TRUENO_NATIVE: u8 = 0b0001_0000;
    /// Payload contains quantized tensors
    pub const QUANTIZED: u8 = 0b0010_0000;
    /// Has model card metadata
    pub const HAS_MODEL_CARD: u8 = 0b0100_0000;

    /// Create new empty flags.
    #[must_use]
    pub fn new() -> Self {
        Self(0)
    }

    /// Set licensed flag.
    #[must_use]
    pub fn with_licensed(mut self) -> Self {
        self.0 |= Self::LICENSED;
        self
    }

    /// Check if encrypted.
    #[must_use]
    pub fn is_encrypted(self) -> bool {
        self.0 & Self::ENCRYPTED != 0
    }

    /// Check if signed.
    #[must_use]
    pub fn is_signed(self) -> bool {
        self.0 & Self::SIGNED != 0
    }

    /// Check if streaming.
    #[must_use]
    pub fn is_streaming(self) -> bool {
        self.0 & Self::STREAMING != 0
    }

    /// Check if licensed.
    #[must_use]
    pub fn is_licensed(self) -> bool {
        self.0 & Self::LICENSED != 0
    }

    /// Check if has model card.
    #[must_use]
    pub fn has_model_card(self) -> bool {
        self.0 & Self::HAS_MODEL_CARD != 0
    }

    /// Get raw value.
    #[must_use]
    pub fn bits(self) -> u8 {
        self.0
    }

    /// Create from raw value (reserved high bit masked).
    #[must_use]
    pub fn from_bits(bits: u8) -> Self {
        Self(bits & 0b0111_1111)
    }
}

/// File header (32 bytes).
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
    /// Quality score (0-100, Poka-yoke validation)
    pub quality_score: u8,
}

impl Header {
    /// Create a new header.
    #[must_use]
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
            quality_score: 0,
        }
    }

    /// Serialize header to bytes (32 bytes).
    #[must_use]
    pub fn to_bytes(&self) -> [u8; HEADER_SIZE] {
        let mut bytes = [0u8; HEADER_SIZE];
        bytes[0..4].copy_from_slice(&self.magic);
        bytes[4] = self.version.0;
        bytes[5] = self.version.1;
        let model_type = self.model_type as u16;
        bytes[6..8].copy_from_slice(&model_type.to_le_bytes());
        bytes[8..12].copy_from_slice(&self.metadata_size.to_le_bytes());
        bytes[12..16].copy_from_slice(&self.payload_size.to_le_bytes());
        bytes[16..20].copy_from_slice(&self.uncompressed_size.to_le_bytes());
        bytes[20] = self.compression as u8;
        bytes[21] = self.flags.bits();
        bytes[22] = self.quality_score;
        // Reserved (23-31) already zero.
        bytes
    }

    /// Parse header from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < HEADER_SIZE {
            return Err(AprFormatError::FormatError {
                message: format!(
                    "Header too short: {} bytes, expected {}",
                    bytes.len(),
                    HEADER_SIZE
                ),
            });
        }

        let magic: [u8; 4] = bytes[0..4]
            .try_into()
            .map_err(|_| AprFormatError::FormatError {
                message: "header slice too short for magic".to_string(),
            })?;
        if magic != MAGIC {
            return Err(AprFormatError::FormatError {
                message: format!(
                    "Invalid magic number: {:02X}{:02X}{:02X}{:02X}, expected APRN",
                    magic[0], magic[1], magic[2], magic[3]
                ),
            });
        }

        let version = (bytes[4], bytes[5]);
        if version.0 > FORMAT_VERSION.0 {
            return Err(AprFormatError::UnsupportedVersion {
                found: version,
                supported: FORMAT_VERSION,
            });
        }

        let model_type_raw = u16::from_le_bytes([bytes[6], bytes[7]]);
        let model_type =
            ModelType::from_u16(model_type_raw).ok_or_else(|| AprFormatError::FormatError {
                message: format!("Unknown model type: 0x{model_type_raw:04X}"),
            })?;

        let metadata_size = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        let payload_size = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
        let uncompressed_size = u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);

        if uncompressed_size > MAX_UNCOMPRESSED_SIZE {
            return Err(AprFormatError::FormatError {
                message: format!(
                    "Uncompressed size {uncompressed_size} exceeds maximum {MAX_UNCOMPRESSED_SIZE} (compression bomb protection)"
                ),
            });
        }

        let compression =
            Compression::from_u8(bytes[20]).ok_or_else(|| AprFormatError::FormatError {
                message: format!("Unknown compression algorithm: 0x{:02X}", bytes[20]),
            })?;
        let flags = Flags::from_bits(bytes[21]);
        let quality_score = bytes[22];

        Ok(Self {
            magic,
            version,
            model_type,
            metadata_size,
            payload_size,
            uncompressed_size,
            compression,
            flags,
            quality_score,
        })
    }
}

/// Training information.
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

/// License tier levels (spec §9.1).
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

/// Commercial license information (spec §9.1).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LicenseInfo {
    /// Unique license identifier (UUID v4)
    pub uuid: String,
    /// Hash of the license certificate
    pub hash: String,
    /// License expiration date (ISO 8601) — None for perpetual
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expiry: Option<String>,
    /// Maximum concurrent seats — None for unlimited
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seats: Option<u32>,
    /// Licensee name/organization
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub licensee: Option<String>,
    /// License tier
    pub tier: LicenseTier,
}

/// Model metadata (MessagePack-encoded).
///
/// Spike slice keeps the load-bearing fields (`created_at`, `aprender_version`,
/// hyperparameters, training, license). The full distillation/model-card
/// provenance types stay in `aprender-core` for now and move in Stage 2.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metadata {
    /// Creation timestamp (ISO 8601 — seconds since epoch in spike form)
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
    /// Commercial license information (spec §9.1)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub license: Option<LicenseInfo>,
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
            license: None,
        }
    }
}

/// Simple ISO-8601-ish timestamp (seconds since epoch, no chrono dependency).
#[must_use]
pub fn chrono_lite_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format!("{secs}")
}

/// Options for saving models.
#[derive(Debug, Clone, Default)]
pub struct SaveOptions {
    /// Compression algorithm
    pub compression: Compression,
    /// Additional metadata
    pub metadata: Metadata,
    /// Quality score from Poka-yoke validation (APR-POKA-001).
    /// - `None`: no validation performed (score=0 in file)
    /// - `Some(0)`: explicit failure — save will be REFUSED (Jidoka)
    /// - `Some(1-59)`: validation failed but allowed to save
    /// - `Some(60-100)`: validation passed
    pub quality_score: Option<u8>,
}

impl SaveOptions {
    /// Create with default compression.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

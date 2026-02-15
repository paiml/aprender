//! APR format type definitions (spec §2-§9)

use crate::error::{AprenderError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::model_card::ModelCard;
use super::validation::PokaYokeResult;
use super::{FORMAT_VERSION, HEADER_SIZE, MAGIC, MAX_UNCOMPRESSED_SIZE};

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
    /// Mixture of Experts (sparse/dense `MoE`)
    MixtureOfExperts = 0x0040,
    /// User-defined model
    Custom = 0x00FF,
}

impl ModelType {
    /// Convert from u16 value
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
    /// Has model card metadata (spec §11)
    pub const HAS_MODEL_CARD: u8 = 0b0100_0000;

    /// Create new flags
    #[must_use]
    pub fn new() -> Self {
        Self(0)
    }

    /// Set encrypted flag
    #[must_use]
    pub fn with_encrypted(mut self) -> Self {
        self.0 |= Self::ENCRYPTED;
        self
    }

    /// Set signed flag
    #[must_use]
    pub fn with_signed(mut self) -> Self {
        self.0 |= Self::SIGNED;
        self
    }

    /// Set streaming flag
    #[must_use]
    pub fn with_streaming(mut self) -> Self {
        self.0 |= Self::STREAMING;
        self
    }

    /// Set licensed flag
    #[must_use]
    pub fn with_licensed(mut self) -> Self {
        self.0 |= Self::LICENSED;
        self
    }

    /// Set trueno-native flag
    #[must_use]
    pub fn with_trueno_native(mut self) -> Self {
        self.0 |= Self::TRUENO_NATIVE;
        self
    }

    /// Set quantized flag
    #[must_use]
    pub fn with_quantized(mut self) -> Self {
        self.0 |= Self::QUANTIZED;
        self
    }

    /// Set model card flag
    #[must_use]
    pub fn with_model_card(mut self) -> Self {
        self.0 |= Self::HAS_MODEL_CARD;
        self
    }

    /// Check if encrypted
    #[must_use]
    pub fn is_encrypted(self) -> bool {
        self.0 & Self::ENCRYPTED != 0
    }

    /// Check if signed
    #[must_use]
    pub fn is_signed(self) -> bool {
        self.0 & Self::SIGNED != 0
    }

    /// Check if streaming
    #[must_use]
    pub fn is_streaming(self) -> bool {
        self.0 & Self::STREAMING != 0
    }

    /// Check if licensed
    #[must_use]
    pub fn is_licensed(self) -> bool {
        self.0 & Self::LICENSED != 0
    }

    /// Check if trueno-native
    #[must_use]
    pub fn is_trueno_native(self) -> bool {
        self.0 & Self::TRUENO_NATIVE != 0
    }

    /// Check if quantized
    #[must_use]
    pub fn is_quantized(self) -> bool {
        self.0 & Self::QUANTIZED != 0
    }

    /// Check if has model card
    #[must_use]
    pub fn has_model_card(self) -> bool {
        self.0 & Self::HAS_MODEL_CARD != 0
    }

    /// Get raw value
    #[must_use]
    pub fn bits(self) -> u8 {
        self.0
    }

    /// Create from raw value
    #[must_use]
    pub fn from_bits(bits: u8) -> Self {
        Self(bits & 0b0111_1111) // Mask reserved bit (7 only)
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
    /// Quality score (0-100, Poka-yoke validation) - APR-POKA-001
    /// 0 = no validation (F), 1-59 = failing, 60-100 = passing grades
    pub quality_score: u8,
}

impl Header {
    /// Create a new header
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
            quality_score: 0, // Must be set via PokaYoke validation
        }
    }

    /// Serialize header to bytes (32 bytes)
    #[must_use]
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

        // Quality score (22) - APR-POKA-001
        bytes[22] = self.quality_score;

        // Reserved (23-31) - already zero

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

        // Parse quality score (byte 22) - APR-POKA-001
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
    /// Model card metadata (spec §11)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_card: Option<ModelCard>,
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

include!("types_part_02.rs");

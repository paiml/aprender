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
            model_card: None,
        }
    }
}

/// Simple ISO 8601 timestamp (no chrono dependency)
pub(crate) fn chrono_lite_now() -> String {
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
    /// Quality score from Poka-yoke validation (APR-POKA-001)
    /// - None: no validation performed (score=0 in file)
    /// - Some(0): explicit failure - save will be REFUSED (Jidoka)
    /// - Some(1-59): validation failed but allowed to save
    /// - Some(60-100): validation passed
    pub quality_score: Option<u8>,
}

impl SaveOptions {
    /// Create with default compression
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set compression algorithm
    #[must_use]
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
    #[must_use]
    pub fn with_distillation_info(mut self, info: DistillationInfo) -> Self {
        self.metadata.distillation_info = Some(info);
        self
    }

    /// Set license info (spec §9.1)
    #[must_use]
    pub fn with_license(mut self, license: LicenseInfo) -> Self {
        self.metadata.license = Some(license);
        self
    }

    /// Set model card (spec §11)
    #[must_use]
    pub fn with_model_card(mut self, card: ModelCard) -> Self {
        self.metadata.model_card = Some(card);
        self
    }

    /// Set quality score from Poka-yoke validation (APR-POKA-001)
    ///
    /// # Jidoka (Stop the Line)
    /// - Score 0 will cause `save()` to REFUSE the write
    /// - Score 1-59 allows save with warning
    /// - Score 60-100 is passing
    #[must_use]
    pub fn with_quality_score(mut self, score: u8) -> Self {
        self.quality_score = Some(score);
        self
    }

    /// Set quality score from `PokaYokeResult` (APR-POKA-001)
    #[must_use]
    pub fn with_poka_yoke_result(mut self, result: &PokaYokeResult) -> Self {
        self.quality_score = Some(result.score);
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
    /// Has model card metadata (spec §11)
    pub has_model_card: bool,
}

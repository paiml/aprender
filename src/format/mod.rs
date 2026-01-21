//! Aprender Model Format (.apr)
//!
//! Binary format for ML model serialization with built-in quality (Jidoka):
//! - CRC32 checksum (integrity)
//! - Ed25519 signatures (provenance)
//! - AES-256-GCM encryption (confidentiality)
//! - Zstd compression (efficiency)
//! - Quantization (`Q8_0`, `Q4_0`, `Q4_1` - GGUF compatible)
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

// Homomorphic encryption module (spec: homomorphic-encryption-spec.md)
#[cfg(feature = "format-homomorphic")]
pub mod homomorphic;

// Weight comparison module (GH-121, HuggingFace/SafeTensors comparison)
pub mod compare;

// APR v2 format module (GH-119, 64-byte alignment, JSON metadata, sharding)
pub mod v2;

// GGUF export module (spec §7.2)
pub mod gguf;

// Hex dump and data flow visualization (GH-122, Toyota Principle 12: Genchi Genbutsu)
pub mod hexdump;

// Model card module (spec §11)
pub mod model_card;

// Validation module (spec §11 - 100-Point QA Checklist)
#[allow(clippy::case_sensitive_file_extension_comparisons)]
pub mod validation;

// Converter module (spec §13 - Import/Convert Pipeline)
#[allow(
    clippy::unnecessary_wraps,
    clippy::type_complexity,
    clippy::trivially_copy_pass_by_ref,
    clippy::explicit_iter_loop,
    clippy::cast_lossless,
    clippy::needless_pass_by_value,
    clippy::map_unwrap_or,
    clippy::case_sensitive_file_extension_comparisons,
    clippy::uninlined_format_args,
    clippy::derivable_impls
)]
pub mod converter;

// Lint module (spec §4.11 - Best Practices & Conventions)
#[allow(
    clippy::struct_excessive_bools,
    clippy::field_reassign_with_default,
    clippy::uninlined_format_args,
    dead_code
)]
pub mod lint;

// Sharded model import module (GH-127 - multi-tensor repos, streaming import)
pub mod sharded;

// Golden trace verification (spec §7.6.3 - prove model authenticity)
pub mod golden;

// Re-export golden trace types
pub use golden::{
    verify_logits, GoldenTrace, GoldenTraceSet, GoldenVerifyReport, LogitStats, TraceVerifyResult,
};

// Re-export model card types
pub use model_card::{ModelCard, TrainingDataInfo};

// Re-export validation types (spec §11 - 100-Point QA Checklist)
pub use validation::{
    AprHeader, AprValidator, Category, CheckStatus, TensorStats, ValidationCheck, ValidationReport,
};

// Re-export Poka-yoke types (APR-POKA-001 - Toyota Way mistake-proofing)
#[allow(deprecated)]
pub use validation::no_validation_result;
pub use validation::{fail_no_validation_rules, Gate, PokaYoke, PokaYokeResult};

// Re-export converter types (spec §13 - Import/Convert Pipeline)
pub use converter::{
    apr_convert, apr_export, apr_import, apr_merge, AprConverter, Architecture, ConvertOptions,
    ConvertReport, ExportFormat, ExportOptions, ExportReport, ImportError, ImportOptions,
    MergeOptions, MergeReport, MergeStrategy, QuantizationType, Source, TensorExpectation,
    ValidationConfig,
};

// Re-export lint types (spec §4.11 - Best Practices & Conventions)
pub use lint::{
    lint_apr_file, lint_model, LintCategory, LintIssue, LintLevel, LintReport, ModelLintInfo,
    TensorLintInfo,
};

// Re-export sharded import types (GH-127 - multi-tensor repos)
pub use sharded::{
    estimate_shard_memory, get_shard_files, is_sharded_model, CacheStats, CachedShard, ImportPhase,
    ImportProgress, ImportReport, ShardCache, ShardIndex, ShardedImportConfig, ShardedImporter,
};

// Re-export quantization types when feature is enabled
#[cfg(feature = "format-quantize")]
pub use quantize::{
    dequantize, quantize as quantize_data, Q4_0Quantizer, Q8_0Quantizer, QuantType,
    QuantizationInfo, QuantizedBlock, QuantizedTensor, Quantizer, BLOCK_SIZE,
};

// Re-export homomorphic encryption types when feature is enabled
#[cfg(feature = "format-homomorphic")]
pub use homomorphic::{
    Ciphertext, HeContext, HeGaloisKeys, HeParameters, HePublicKey, HeRelinKeys, HeScheme,
    HeSecretKey, Plaintext, SecurityLevel,
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

// ============================================================================
// FILE LOADING HELPER FUNCTIONS (Refactored for reduced complexity)
// ============================================================================

/// Read entire file content into a buffer.
#[cfg(any(feature = "format-signing", feature = "format-encryption"))]
fn read_file_content(path: &Path) -> Result<Vec<u8>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut content = Vec::new();
    reader.read_to_end(&mut content)?;
    Ok(content)
}

/// Verify CRC32 checksum at end of file content.
#[cfg(any(feature = "format-signing", feature = "format-encryption"))]
fn verify_file_checksum(content: &[u8]) -> Result<()> {
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
fn parse_and_validate_header(content: &[u8], expected_type: ModelType) -> Result<Header> {
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
fn verify_signed_flag(header: &Header) -> Result<()> {
    if !header.flags.is_signed() {
        return Err(AprenderError::FormatError {
            message: "File is not signed (SIGNED flag not set)".to_string(),
        });
    }
    Ok(())
}

/// Verify header flag is set for encrypted files.
#[cfg(feature = "format-encryption")]
fn verify_encrypted_flag(header: &Header) -> Result<()> {
    if !header.flags.is_encrypted() {
        return Err(AprenderError::FormatError {
            message: "File is not encrypted (ENCRYPTED flag not set)".to_string(),
        });
    }
    Ok(())
}

/// Verify payload boundary is within file content.
#[cfg(any(feature = "format-signing", feature = "format-encryption"))]
fn verify_payload_boundary(payload_end: usize, content_len: usize) -> Result<()> {
    if payload_end > content_len - 4 {
        return Err(AprenderError::FormatError {
            message: "Payload extends beyond file boundary".to_string(),
        });
    }
    Ok(())
}

/// Decompress and deserialize payload.
#[cfg(feature = "format-signing")]
fn decompress_and_deserialize<M: DeserializeOwned>(
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
/// - Files ≤ 1MB: Standard `load()` (lower overhead for small files)
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

/// Verify encrypted data has minimum required size.
#[cfg(feature = "format-encryption")]
fn verify_encrypted_data_size(data: &[u8]) -> Result<()> {
    if data.len() < HEADER_SIZE + SALT_SIZE + NONCE_SIZE + 4 {
        return Err(AprenderError::FormatError {
            message: format!("Data too small for encrypted model: {} bytes", data.len()),
        });
    }
    Ok(())
}

/// Verify encrypted data checksum.
#[cfg(feature = "format-encryption")]
fn verify_encrypted_checksum(data: &[u8]) -> Result<()> {
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
fn verify_encrypted_header(header: &Header, expected_type: ModelType) -> Result<()> {
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
fn extract_encrypted_components<'a>(
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
fn decrypt_encrypted_payload(
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
        has_model_card: header.flags.has_model_card(),
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
    let path = path.as_ref();

    // Read and validate file
    let content = read_file_content(path)?;
    verify_signed_file_size(&content)?;
    verify_file_checksum(&content)?;

    // Parse and validate header
    let header = parse_and_validate_header(&content, expected_type)?;
    verify_signed_flag(&header)?;

    // Calculate content boundaries
    let metadata_end = HEADER_SIZE + header.metadata_size as usize;
    let payload_end = metadata_end + header.payload_size as usize;
    let signature_start = payload_end;
    let pubkey_start = signature_start + SIGNATURE_SIZE;
    let pubkey_end = pubkey_start + PUBLIC_KEY_SIZE;

    verify_payload_boundary(pubkey_end, content.len())?;

    // Extract and verify signature
    let (signature, embedded_key) =
        extract_signature_and_key(&content, signature_start, pubkey_start, pubkey_end)?;
    let verifying_key = trusted_key.unwrap_or(&embedded_key);
    let signable_content = &content[..payload_end];
    verify_signature(verifying_key, signable_content, &signature)?;

    // Extract and deserialize payload
    decompress_and_deserialize(&content[metadata_end..payload_end], header.compression)
}

/// Verify minimum file size for signed files.
#[cfg(feature = "format-signing")]
fn verify_signed_file_size(content: &[u8]) -> Result<()> {
    const SIGNATURE_BLOCK_SIZE: usize = SIGNATURE_SIZE + PUBLIC_KEY_SIZE;
    if content.len() < HEADER_SIZE + SIGNATURE_BLOCK_SIZE + 4 {
        return Err(AprenderError::FormatError {
            message: format!("File too small for signed model: {} bytes", content.len()),
        });
    }
    Ok(())
}

/// Extract signature and public key from file content.
#[cfg(feature = "format-signing")]
fn extract_signature_and_key(
    content: &[u8],
    signature_start: usize,
    pubkey_start: usize,
    pubkey_end: usize,
) -> Result<(ed25519_dalek::Signature, VerifyingKey)> {
    use ed25519_dalek::Signature;

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

    Ok((signature, embedded_key))
}

/// Verify Ed25519 signature.
#[cfg(feature = "format-signing")]
fn verify_signature(
    verifying_key: &VerifyingKey,
    signable_content: &[u8],
    signature: &ed25519_dalek::Signature,
) -> Result<()> {
    use ed25519_dalek::Verifier;

    verifying_key
        .verify(signable_content, signature)
        .map_err(|e| AprenderError::SignatureInvalid {
            reason: format!("Signature verification failed: {e}"),
        })
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
    let path = path.as_ref();

    // Read and validate file
    let content = read_file_content(path)?;
    verify_password_encrypted_file_size(&content)?;
    verify_file_checksum(&content)?;

    // Parse and validate header
    let header = parse_and_validate_header(&content, expected_type)?;
    verify_encrypted_flag(&header)?;

    // Calculate content boundaries
    let metadata_end = HEADER_SIZE + header.metadata_size as usize;
    let salt_end = metadata_end + SALT_SIZE;
    let nonce_end = salt_end + NONCE_SIZE;
    let payload_end = metadata_end + header.payload_size as usize;

    verify_payload_boundary(payload_end, content.len())?;

    // Extract encryption components and decrypt
    let (salt, nonce_bytes, ciphertext) = extract_password_encryption_components(
        &content,
        metadata_end,
        salt_end,
        nonce_end,
        payload_end,
    )?;
    let payload_compressed = decrypt_password_payload(password, &salt, &nonce_bytes, ciphertext)?;

    // Decompress and deserialize
    let payload_uncompressed = decompress_payload(&payload_compressed, header.compression)?;
    bincode::deserialize(&payload_uncompressed)
        .map_err(|e| AprenderError::Serialization(format!("Failed to deserialize model: {e}")))
}

/// Verify minimum file size for password-encrypted files.
#[cfg(feature = "format-encryption")]
fn verify_password_encrypted_file_size(content: &[u8]) -> Result<()> {
    if content.len() < HEADER_SIZE + SALT_SIZE + NONCE_SIZE + 4 {
        return Err(AprenderError::FormatError {
            message: format!(
                "File too small for encrypted model: {} bytes",
                content.len()
            ),
        });
    }
    Ok(())
}

/// Extract salt, nonce, and ciphertext from password-encrypted file.
#[cfg(feature = "format-encryption")]
fn extract_password_encryption_components(
    content: &[u8],
    metadata_end: usize,
    salt_end: usize,
    nonce_end: usize,
    payload_end: usize,
) -> Result<([u8; SALT_SIZE], [u8; NONCE_SIZE], &[u8])> {
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
    Ok((salt, nonce_bytes, ciphertext))
}

/// Derive key from password and decrypt payload.
#[cfg(feature = "format-encryption")]
fn decrypt_password_payload(
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

    // Derive key using Argon2id
    let mut key = [0u8; KEY_SIZE];
    Argon2::default()
        .hash_password_into(password.as_bytes(), salt, &mut key)
        .map_err(|e| AprenderError::Other(format!("Key derivation failed: {e}")))?;

    // Decrypt with AES-256-GCM
    let cipher = Aes256Gcm::new_from_slice(&key)
        .map_err(|e| AprenderError::Other(format!("Failed to create cipher: {e}")))?;
    let nonce = Nonce::from_slice(nonce_bytes);
    cipher
        .decrypt(nonce, ciphertext)
        .map_err(|_| AprenderError::DecryptionFailed {
            message: "Decryption failed (wrong password or corrupted data)".to_string(),
        })
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
pub fn load_as_recipient<M: DeserializeOwned>(
    path: impl AsRef<Path>,
    expected_type: ModelType,
    recipient_secret_key: &X25519SecretKey,
) -> Result<M> {
    let path = path.as_ref();

    // Read and validate file
    let content = read_file_content(path)?;
    verify_x25519_encrypted_file_size(&content)?;
    verify_file_checksum(&content)?;

    // Parse and validate header
    let header = parse_and_validate_header(&content, expected_type)?;
    verify_encrypted_flag(&header)?;

    // Calculate content boundaries
    let metadata_end = HEADER_SIZE + header.metadata_size as usize;
    let ephemeral_pub_end = metadata_end + X25519_PUBLIC_KEY_SIZE;
    let recipient_hash_end = ephemeral_pub_end + RECIPIENT_HASH_SIZE;
    let nonce_end = recipient_hash_end + NONCE_SIZE;
    let payload_end = metadata_end + header.payload_size as usize;

    verify_payload_boundary(payload_end, content.len())?;

    // Extract and verify recipient components
    let (ephemeral_public, stored_recipient_hash) = extract_x25519_recipient_info(
        &content,
        metadata_end,
        ephemeral_pub_end,
        recipient_hash_end,
    )?;
    verify_recipient(recipient_secret_key, stored_recipient_hash)?;

    // Extract nonce and ciphertext, then decrypt
    let (nonce_bytes, ciphertext) =
        extract_nonce_and_ciphertext(&content, recipient_hash_end, nonce_end, payload_end)?;
    let payload_compressed = decrypt_x25519_payload(
        recipient_secret_key,
        &ephemeral_public,
        &nonce_bytes,
        ciphertext,
    )?;

    // Decompress and deserialize
    let payload_uncompressed = decompress_payload(&payload_compressed, header.compression)?;
    bincode::deserialize(&payload_uncompressed)
        .map_err(|e| AprenderError::Serialization(format!("Failed to deserialize model: {e}")))
}

/// Verify minimum file size for X25519-encrypted files.
#[cfg(feature = "format-encryption")]
fn verify_x25519_encrypted_file_size(content: &[u8]) -> Result<()> {
    const MIN_PAYLOAD_SIZE: usize = X25519_PUBLIC_KEY_SIZE + RECIPIENT_HASH_SIZE + NONCE_SIZE;
    if content.len() < HEADER_SIZE + MIN_PAYLOAD_SIZE + 4 {
        return Err(AprenderError::FormatError {
            message: format!(
                "File too small for X25519 encrypted model: {} bytes",
                content.len()
            ),
        });
    }
    Ok(())
}

/// Extract ephemeral public key and recipient hash from X25519-encrypted file.
#[cfg(feature = "format-encryption")]
fn extract_x25519_recipient_info(
    content: &[u8],
    metadata_end: usize,
    ephemeral_pub_end: usize,
    recipient_hash_end: usize,
) -> Result<(X25519PublicKey, [u8; RECIPIENT_HASH_SIZE])> {
    let ephemeral_pub_bytes: [u8; X25519_PUBLIC_KEY_SIZE] = content
        [metadata_end..ephemeral_pub_end]
        .try_into()
        .map_err(|_| AprenderError::FormatError {
            message: "Invalid ephemeral public key size".to_string(),
        })?;
    let ephemeral_public = X25519PublicKey::from(ephemeral_pub_bytes);

    let stored_recipient_hash: [u8; RECIPIENT_HASH_SIZE] = content
        [ephemeral_pub_end..recipient_hash_end]
        .try_into()
        .map_err(|_| AprenderError::FormatError {
            message: "Invalid recipient hash size".to_string(),
        })?;

    Ok((ephemeral_public, stored_recipient_hash))
}

/// Verify this file was encrypted for the given recipient.
#[cfg(feature = "format-encryption")]
fn verify_recipient(
    recipient_secret_key: &X25519SecretKey,
    stored_recipient_hash: [u8; RECIPIENT_HASH_SIZE],
) -> Result<()> {
    let our_public = X25519PublicKey::from(recipient_secret_key);
    let our_hash: [u8; RECIPIENT_HASH_SIZE] = our_public.as_bytes()[..RECIPIENT_HASH_SIZE]
        .try_into()
        .expect("hash size is correct");

    if stored_recipient_hash != our_hash {
        return Err(AprenderError::DecryptionFailed {
            message: "This file was encrypted for a different recipient".to_string(),
        });
    }
    Ok(())
}

/// Extract nonce and ciphertext from encrypted content.
#[cfg(feature = "format-encryption")]
fn extract_nonce_and_ciphertext(
    content: &[u8],
    recipient_hash_end: usize,
    nonce_end: usize,
    payload_end: usize,
) -> Result<([u8; NONCE_SIZE], &[u8])> {
    let nonce_bytes: [u8; NONCE_SIZE] =
        content[recipient_hash_end..nonce_end]
            .try_into()
            .map_err(|_| AprenderError::FormatError {
                message: "Invalid nonce size".to_string(),
            })?;
    let ciphertext = &content[nonce_end..payload_end];
    Ok((nonce_bytes, ciphertext))
}

/// Perform X25519 key agreement and decrypt payload.
#[cfg(feature = "format-encryption")]
fn decrypt_x25519_payload(
    recipient_secret_key: &X25519SecretKey,
    ephemeral_public: &X25519PublicKey,
    nonce_bytes: &[u8; NONCE_SIZE],
    ciphertext: &[u8],
) -> Result<Vec<u8>> {
    use aes_gcm::{
        aead::{Aead, KeyInit},
        Aes256Gcm, Nonce,
    };
    use hkdf::Hkdf;
    use sha2::Sha256;

    // Perform X25519 key agreement
    let shared_secret = recipient_secret_key.diffie_hellman(ephemeral_public);

    // Derive encryption key using HKDF-SHA256
    let hkdf = Hkdf::<Sha256>::new(None, shared_secret.as_bytes());
    let mut key = [0u8; KEY_SIZE];
    hkdf.expand(HKDF_INFO, &mut key)
        .map_err(|_| AprenderError::Other("HKDF expansion failed".to_string()))?;

    // Decrypt with AES-256-GCM
    let cipher = Aes256Gcm::new_from_slice(&key)
        .map_err(|e| AprenderError::Other(format!("Failed to create cipher: {e}")))?;
    let nonce = Nonce::from_slice(nonce_bytes);
    cipher
        .decrypt(nonce, ciphertext)
        .map_err(|_| AprenderError::DecryptionFailed {
            message: "Decryption failed (wrong recipient key or corrupted data)".to_string(),
        })
}


#[cfg(test)]
mod tests;

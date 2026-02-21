//! Model Inspection Tooling
//!
//! Provides comprehensive inspection capabilities for `.apr` model files,
//! enabling debugging, validation, and quality assessment.
//!
//! # Features
//!
//! - **Header inspection**: View magic, version, flags
//! - **Metadata extraction**: Model type, hyperparameters, provenance
//! - **Weight statistics**: Min/max/mean/std for model parameters
//! - **Diff comparison**: Compare two model versions
//! - **Quality scoring**: 100-point model assessment
//! - **SafeTensors comparison**: Compare against HuggingFace models (GH-121)
//!
//! # Toyota Way Alignment
//!
//! - **Genchi Genbutsu**: Go and see - inspect actual model data
//! - **Visualization**: Make problems visible for debugging

#[cfg(feature = "safetensors-compare")]
pub mod safetensors;

#[cfg(feature = "safetensors-compare")]
pub use safetensors::{BatchComparison, HfSafetensors, TensorComparison, TensorData};

use std::collections::HashMap;
use std::fmt;
use std::time::Duration;

/// Model inspection result containing all extracted information
#[derive(Debug, Clone)]
pub struct InspectionResult {
    /// Header information
    pub header: HeaderInspection,
    /// Metadata information
    pub metadata: MetadataInspection,
    /// Weight statistics
    pub weights: Option<WeightStats>,
    /// Quality score (0-100)
    pub quality_score: Option<u32>,
    /// Inspection duration
    pub duration: Duration,
    /// Any warnings found
    pub warnings: Vec<InspectionWarning>,
    /// Any errors found
    pub errors: Vec<InspectionError>,
}

impl InspectionResult {
    /// Create a new inspection result
    #[must_use]
    pub fn new(header: HeaderInspection, metadata: MetadataInspection) -> Self {
        Self {
            header,
            metadata,
            weights: None,
            quality_score: None,
            duration: Duration::ZERO,
            warnings: Vec::new(),
            errors: Vec::new(),
        }
    }

    /// Check if inspection found any issues
    #[must_use]
    pub fn has_issues(&self) -> bool {
        !self.warnings.is_empty() || !self.errors.is_empty()
    }

    /// Check if inspection is valid (no errors)
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.errors.is_empty()
    }

    /// Get total issue count
    #[must_use]
    pub fn issue_count(&self) -> usize {
        self.warnings.len() + self.errors.len()
    }
}

/// Header inspection details
#[derive(Debug, Clone)]
pub struct HeaderInspection {
    /// Magic bytes (should be "APRN")
    pub magic: [u8; 4],
    /// Format version (major, minor)
    pub version: (u8, u8),
    /// Model type ID
    pub model_type: u16,
    /// Feature flags
    pub flags: HeaderFlags,
    /// Compressed size in bytes
    pub compressed_size: u64,
    /// Uncompressed size in bytes
    pub uncompressed_size: u64,
    /// Checksum value
    pub checksum: u32,
    /// Whether magic is valid
    pub magic_valid: bool,
    /// Whether version is supported
    pub version_supported: bool,
}

impl HeaderInspection {
    /// Create a new header inspection
    #[must_use]
    pub fn new() -> Self {
        Self {
            magic: *b"APRN",
            version: (1, 0),
            model_type: 0,
            flags: HeaderFlags::default(),
            compressed_size: 0,
            uncompressed_size: 0,
            checksum: 0,
            magic_valid: true,
            version_supported: true,
        }
    }

    /// Get magic as string
    #[must_use]
    pub fn magic_string(&self) -> String {
        String::from_utf8_lossy(&self.magic).to_string()
    }

    /// Get version as string
    #[must_use]
    pub fn version_string(&self) -> String {
        format!("{}.{}", self.version.0, self.version.1)
    }

    /// Get compression ratio
    #[must_use]
    pub fn compression_ratio(&self) -> f64 {
        if self.compressed_size == 0 {
            1.0
        } else {
            self.uncompressed_size as f64 / self.compressed_size as f64
        }
    }

    /// Check if header is fully valid
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.magic_valid && self.version_supported
    }
}

impl Default for HeaderInspection {
    fn default() -> Self {
        Self::new()
    }
}

/// Header feature flags
#[derive(Debug, Clone, Copy, Default)]
#[allow(clippy::struct_excessive_bools)] // Flags struct legitimately has independent booleans
pub struct HeaderFlags {
    /// Model is compressed
    pub compressed: bool,
    /// Model is signed
    pub signed: bool,
    /// Model is encrypted
    pub encrypted: bool,
    /// Model supports streaming
    pub streaming: bool,
    /// Model is licensed
    pub licensed: bool,
    /// Model is quantized
    pub quantized: bool,
}

impl HeaderFlags {
    /// Create from raw flags byte
    #[must_use]
    pub fn from_byte(byte: u8) -> Self {
        Self {
            compressed: byte & 0x01 != 0,
            signed: byte & 0x02 != 0,
            encrypted: byte & 0x04 != 0,
            streaming: byte & 0x08 != 0,
            licensed: byte & 0x10 != 0,
            quantized: byte & 0x20 != 0,
        }
    }

    /// Convert to raw flags byte
    #[must_use]
    pub fn to_byte(&self) -> u8 {
        let mut byte = 0u8;
        if self.compressed {
            byte |= 0x01;
        }
        if self.signed {
            byte |= 0x02;
        }
        if self.encrypted {
            byte |= 0x04;
        }
        if self.streaming {
            byte |= 0x08;
        }
        if self.licensed {
            byte |= 0x10;
        }
        if self.quantized {
            byte |= 0x20;
        }
        byte
    }

    /// Get human-readable flag list
    #[must_use]
    pub fn flag_list(&self) -> Vec<&'static str> {
        let mut flags = Vec::new();
        if self.compressed {
            flags.push("COMPRESSED");
        }
        if self.signed {
            flags.push("SIGNED");
        }
        if self.encrypted {
            flags.push("ENCRYPTED");
        }
        if self.streaming {
            flags.push("STREAMING");
        }
        if self.licensed {
            flags.push("LICENSED");
        }
        if self.quantized {
            flags.push("QUANTIZED");
        }
        flags
    }
}

/// Metadata inspection details
#[derive(Debug, Clone)]
pub struct MetadataInspection {
    /// Model type name
    pub model_type_name: String,
    /// Number of parameters
    pub n_parameters: u64,
    /// Number of features
    pub n_features: u32,
    /// Number of outputs
    pub n_outputs: u32,
    /// Hyperparameters
    pub hyperparameters: HashMap<String, String>,
    /// Training info
    pub training_info: Option<TrainingInfo>,
    /// License info
    pub license_info: Option<LicenseInfo>,
    /// Custom metadata
    pub custom: HashMap<String, String>,
}

impl MetadataInspection {
    /// Create a new metadata inspection
    #[must_use]
    pub fn new(model_type_name: impl Into<String>) -> Self {
        Self {
            model_type_name: model_type_name.into(),
            n_parameters: 0,
            n_features: 0,
            n_outputs: 0,
            hyperparameters: HashMap::new(),
            training_info: None,
            license_info: None,
            custom: HashMap::new(),
        }
    }

    /// Check if model has training info
    #[must_use]
    pub fn has_training_info(&self) -> bool {
        self.training_info.is_some()
    }

    /// Check if model is licensed
    #[must_use]
    pub fn is_licensed(&self) -> bool {
        self.license_info.is_some()
    }
}

/// Training information
#[derive(Debug, Clone)]
pub struct TrainingInfo {
    /// Training date
    pub trained_at: Option<String>,
    /// Training duration
    pub duration: Option<Duration>,
    /// Training dataset name
    pub dataset_name: Option<String>,
    /// Number of training samples
    pub n_samples: Option<u64>,
    /// Final loss value
    pub final_loss: Option<f64>,
    /// Training framework
    pub framework: Option<String>,
    /// Framework version
    pub framework_version: Option<String>,
}

impl TrainingInfo {
    /// Create new training info
    #[must_use]
    pub fn new() -> Self {
        Self {
            trained_at: None,
            duration: None,
            dataset_name: None,
            n_samples: None,
            final_loss: None,
            framework: None,
            framework_version: None,
        }
    }
}

impl Default for TrainingInfo {
    fn default() -> Self {
        Self::new()
    }
}

/// License information
#[derive(Debug, Clone)]
pub struct LicenseInfo {
    /// License type (e.g., "MIT", "Apache-2.0", "Proprietary")
    pub license_type: String,
    /// Licensee name
    pub licensee: Option<String>,
    /// Expiration date
    pub expires_at: Option<String>,
    /// Usage restrictions
    pub restrictions: Vec<String>,
}

impl LicenseInfo {
    /// Create new license info
    #[must_use]
    pub fn new(license_type: impl Into<String>) -> Self {
        Self {
            license_type: license_type.into(),
            licensee: None,
            expires_at: None,
            restrictions: Vec::new(),
        }
    }

    /// Check if license has restrictions
    #[must_use]
    pub fn has_restrictions(&self) -> bool {
        !self.restrictions.is_empty()
    }
}

/// Weight statistics
#[derive(Debug, Clone)]
pub struct WeightStats {
    /// Total number of weights
    pub count: u64,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// Mean value
    pub mean: f64,
    /// Standard deviation
    pub std: f64,
    /// Number of zero weights
    pub zero_count: u64,
    /// Number of NaN weights
    pub nan_count: u64,
    /// Number of infinite weights
    pub inf_count: u64,
    /// Sparsity (fraction of zeros)
    pub sparsity: f64,
    /// L1 norm
    pub l1_norm: f64,
    /// L2 norm
    pub l2_norm: f64,
}

include!("weight_stats.rs");
include!("tests.rs");

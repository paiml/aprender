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

impl WeightStats {
    /// Create weight stats from a slice of values
    #[must_use]
    pub fn from_slice(weights: &[f32]) -> Self {
        if weights.is_empty() {
            return Self::empty();
        }

        let count = weights.len() as u64;
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        let mut sum = 0.0_f64;
        let mut zero_count = 0_u64;
        let mut nan_count = 0_u64;
        let mut inf_count = 0_u64;
        let mut l1_sum = 0.0_f64;
        let mut l2_sum = 0.0_f64;

        for &w in weights {
            let w = f64::from(w);
            if w.is_nan() {
                nan_count += 1;
                continue;
            }
            if w.is_infinite() {
                inf_count += 1;
                continue;
            }
            if w == 0.0 {
                zero_count += 1;
            }
            min = min.min(w);
            max = max.max(w);
            sum += w;
            l1_sum += w.abs();
            l2_sum += w * w;
        }

        let valid_count = count - nan_count - inf_count;
        let mean = if valid_count > 0 {
            sum / valid_count as f64
        } else {
            0.0
        };

        // Calculate standard deviation
        let mut variance_sum = 0.0_f64;
        for &w in weights {
            let w = f64::from(w);
            if !w.is_nan() && !w.is_infinite() {
                variance_sum += (w - mean).powi(2);
            }
        }
        let std = if valid_count > 1 {
            (variance_sum / (valid_count - 1) as f64).sqrt()
        } else {
            0.0
        };

        let sparsity = zero_count as f64 / count as f64;

        Self {
            count,
            min: if min == f64::INFINITY { 0.0 } else { min },
            max: if max == f64::NEG_INFINITY { 0.0 } else { max },
            mean,
            std,
            zero_count,
            nan_count,
            inf_count,
            sparsity,
            l1_norm: l1_sum,
            l2_norm: l2_sum.sqrt(),
        }
    }

    /// Create empty weight stats
    #[must_use]
    pub fn empty() -> Self {
        Self {
            count: 0,
            min: 0.0,
            max: 0.0,
            mean: 0.0,
            std: 0.0,
            zero_count: 0,
            nan_count: 0,
            inf_count: 0,
            sparsity: 0.0,
            l1_norm: 0.0,
            l2_norm: 0.0,
        }
    }

    /// Check if weights have issues (NaN or Inf)
    #[must_use]
    pub fn has_issues(&self) -> bool {
        self.nan_count > 0 || self.inf_count > 0
    }

    /// Get health status
    #[must_use]
    pub fn health_status(&self) -> WeightHealth {
        if self.nan_count > 0 || self.inf_count > 0 {
            WeightHealth::Critical
        } else if self.sparsity > 0.99 || self.std < 1e-10 {
            WeightHealth::Warning
        } else {
            WeightHealth::Healthy
        }
    }
}

/// Weight health status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightHealth {
    /// Weights are healthy
    Healthy,
    /// Weights have potential issues
    Warning,
    /// Weights have critical issues
    Critical,
}

impl WeightHealth {
    /// Get description
    #[must_use]
    pub const fn description(&self) -> &'static str {
        match self {
            Self::Healthy => "Weights are within normal parameters",
            Self::Warning => "Weights may have potential issues (high sparsity or low variance)",
            Self::Critical => "Weights have critical issues (NaN or Inf values)",
        }
    }
}

/// Inspection warning
#[derive(Debug, Clone)]
pub struct InspectionWarning {
    /// Warning code
    pub code: String,
    /// Warning message
    pub message: String,
    /// Recommendation
    pub recommendation: Option<String>,
}

impl InspectionWarning {
    /// Create a new warning
    #[must_use]
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
            recommendation: None,
        }
    }

    /// Add recommendation
    #[must_use]
    pub fn with_recommendation(mut self, recommendation: impl Into<String>) -> Self {
        self.recommendation = Some(recommendation.into());
        self
    }
}

impl fmt::Display for InspectionWarning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.code, self.message)?;
        if let Some(rec) = &self.recommendation {
            write!(f, " (Recommendation: {rec})")?;
        }
        Ok(())
    }
}

/// Inspection error
#[derive(Debug, Clone)]
pub struct InspectionError {
    /// Error code
    pub code: String,
    /// Error message
    pub message: String,
    /// Whether error is fatal
    pub fatal: bool,
}

impl InspectionError {
    /// Create a new error
    #[must_use]
    pub fn new(code: impl Into<String>, message: impl Into<String>, fatal: bool) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
            fatal,
        }
    }
}

impl fmt::Display for InspectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let severity = if self.fatal { "FATAL" } else { "ERROR" };
        write!(f, "[{} {}] {}", severity, self.code, self.message)
    }
}

/// Model diff result
#[derive(Debug, Clone)]
pub struct DiffResult {
    /// First model info
    pub model_a: String,
    /// Second model info
    pub model_b: String,
    /// Header differences
    pub header_diff: Vec<DiffItem>,
    /// Metadata differences
    pub metadata_diff: Vec<DiffItem>,
    /// Weight differences
    pub weight_diff: Option<WeightDiff>,
    /// Overall similarity (0.0 - 1.0)
    pub similarity: f64,
}

impl DiffResult {
    /// Create a new diff result
    #[must_use]
    pub fn new(model_a: impl Into<String>, model_b: impl Into<String>) -> Self {
        Self {
            model_a: model_a.into(),
            model_b: model_b.into(),
            header_diff: Vec::new(),
            metadata_diff: Vec::new(),
            weight_diff: None,
            similarity: 1.0,
        }
    }

    /// Check if models are identical
    #[must_use]
    pub fn is_identical(&self) -> bool {
        self.header_diff.is_empty()
            && self.metadata_diff.is_empty()
            && self
                .weight_diff
                .as_ref()
                .map_or(true, WeightDiff::is_identical)
    }

    /// Get total difference count
    #[must_use]
    pub fn diff_count(&self) -> usize {
        let weight_count = self.weight_diff.as_ref().map_or(0, WeightDiff::diff_count);
        self.header_diff.len() + self.metadata_diff.len() + weight_count
    }
}

/// Diff item for scalar values
#[derive(Debug, Clone)]
pub struct DiffItem {
    /// Field name
    pub field: String,
    /// Value in model A
    pub value_a: String,
    /// Value in model B
    pub value_b: String,
}

impl DiffItem {
    /// Create a new diff item
    #[must_use]
    pub fn new(
        field: impl Into<String>,
        value_a: impl Into<String>,
        value_b: impl Into<String>,
    ) -> Self {
        Self {
            field: field.into(),
            value_a: value_a.into(),
            value_b: value_b.into(),
        }
    }
}

impl fmt::Display for DiffItem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {} -> {}", self.field, self.value_a, self.value_b)
    }
}

/// Weight difference statistics
#[derive(Debug, Clone)]
pub struct WeightDiff {
    /// Number of weights that differ
    pub changed_count: u64,
    /// Maximum absolute difference
    pub max_diff: f64,
    /// Mean absolute difference
    pub mean_diff: f64,
    /// L2 distance between weight vectors
    pub l2_distance: f64,
    /// Cosine similarity
    pub cosine_similarity: f64,
}

impl WeightDiff {
    /// Create empty weight diff
    #[must_use]
    pub fn empty() -> Self {
        Self {
            changed_count: 0,
            max_diff: 0.0,
            mean_diff: 0.0,
            l2_distance: 0.0,
            cosine_similarity: 1.0,
        }
    }

    /// Create from two weight slices
    #[must_use]
    pub fn from_slices(a: &[f32], b: &[f32]) -> Self {
        if a.len() != b.len() || a.is_empty() {
            return Self::empty();
        }

        let mut changed_count = 0_u64;
        let mut max_diff = 0.0_f64;
        let mut diff_sum = 0.0_f64;
        let mut l2_sum = 0.0_f64;
        let mut dot_product = 0.0_f64;
        let mut norm_a = 0.0_f64;
        let mut norm_b = 0.0_f64;

        for (&va, &vb) in a.iter().zip(b.iter()) {
            let va = f64::from(va);
            let vb = f64::from(vb);
            let diff = (va - vb).abs();

            if diff > 1e-10 {
                changed_count += 1;
            }
            max_diff = max_diff.max(diff);
            diff_sum += diff;
            l2_sum += diff * diff;
            dot_product += va * vb;
            norm_a += va * va;
            norm_b += vb * vb;
        }

        let count = a.len() as f64;
        let mean_diff = diff_sum / count;
        let l2_distance = l2_sum.sqrt();

        let cosine_similarity = if norm_a > 0.0 && norm_b > 0.0 {
            dot_product / (norm_a.sqrt() * norm_b.sqrt())
        } else {
            1.0
        };

        Self {
            changed_count,
            max_diff,
            mean_diff,
            l2_distance,
            cosine_similarity,
        }
    }

    /// Check if weights are identical
    #[must_use]
    pub fn is_identical(&self) -> bool {
        self.changed_count == 0
    }

    /// Get diff count (treat any changes as a single diff)
    #[must_use]
    pub fn diff_count(&self) -> usize {
        usize::from(self.changed_count > 0)
    }
}

/// Inspection options
#[derive(Debug, Clone)]
pub struct InspectOptions {
    /// Include weight statistics
    pub include_weights: bool,
    /// Include quality scoring
    pub include_quality: bool,
    /// Maximum weights to analyze (for large models)
    pub max_weights: usize,
    /// Verbose output
    pub verbose: bool,
}

impl Default for InspectOptions {
    fn default() -> Self {
        Self {
            include_weights: true,
            include_quality: true,
            max_weights: 10_000_000, // 10M weights
            verbose: false,
        }
    }
}

impl InspectOptions {
    /// Create new inspection options
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Quick inspection (header and metadata only)
    #[must_use]
    pub fn quick() -> Self {
        Self {
            include_weights: false,
            include_quality: false,
            max_weights: 0,
            verbose: false,
        }
    }

    /// Full inspection with all analysis
    #[must_use]
    pub fn full() -> Self {
        Self {
            include_weights: true,
            include_quality: true,
            max_weights: usize::MAX,
            verbose: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_flags_roundtrip() {
        let flags = HeaderFlags {
            compressed: true,
            signed: false,
            encrypted: true,
            streaming: false,
            licensed: true,
            quantized: false,
        };

        let byte = flags.to_byte();
        let restored = HeaderFlags::from_byte(byte);

        assert_eq!(flags.compressed, restored.compressed);
        assert_eq!(flags.signed, restored.signed);
        assert_eq!(flags.encrypted, restored.encrypted);
        assert_eq!(flags.streaming, restored.streaming);
        assert_eq!(flags.licensed, restored.licensed);
        assert_eq!(flags.quantized, restored.quantized);
    }

    #[test]
    fn test_header_flags_list() {
        let flags = HeaderFlags {
            compressed: true,
            signed: true,
            encrypted: false,
            streaming: false,
            licensed: false,
            quantized: false,
        };

        let list = flags.flag_list();
        assert!(list.contains(&"COMPRESSED"));
        assert!(list.contains(&"SIGNED"));
        assert!(!list.contains(&"ENCRYPTED"));
    }

    #[test]
    fn test_header_inspection() {
        let header = HeaderInspection::new();
        assert!(header.is_valid());
        assert_eq!(header.magic_string(), "APRN");
        assert_eq!(header.version_string(), "1.0");
        assert!((header.compression_ratio() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_weight_stats_from_slice() {
        let weights = vec![1.0_f32, 2.0, 3.0, 0.0, 5.0];
        let stats = WeightStats::from_slice(&weights);

        assert_eq!(stats.count, 5);
        assert!((stats.min - 0.0).abs() < 0.001);
        assert!((stats.max - 5.0).abs() < 0.001);
        assert!((stats.mean - 2.2).abs() < 0.001);
        assert_eq!(stats.zero_count, 1);
        assert_eq!(stats.nan_count, 0);
        assert!((stats.sparsity - 0.2).abs() < 0.001);
    }

    #[test]
    fn test_weight_stats_empty() {
        let stats = WeightStats::from_slice(&[]);
        assert_eq!(stats.count, 0);
        assert!(!stats.has_issues());
    }

    #[test]
    fn test_weight_stats_with_nan() {
        let weights = vec![1.0_f32, f32::NAN, 3.0];
        let stats = WeightStats::from_slice(&weights);

        assert_eq!(stats.nan_count, 1);
        assert!(stats.has_issues());
        assert_eq!(stats.health_status(), WeightHealth::Critical);
    }

    #[test]
    fn test_weight_health() {
        assert_eq!(
            WeightHealth::Healthy.description(),
            "Weights are within normal parameters"
        );
    }

    #[test]
    fn test_inspection_warning() {
        let warning =
            InspectionWarning::new("W001", "Test warning").with_recommendation("Fix the issue");

        let display = format!("{}", warning);
        assert!(display.contains("W001"));
        assert!(display.contains("Test warning"));
        assert!(display.contains("Fix the issue"));
    }

    #[test]
    fn test_inspection_error() {
        let error = InspectionError::new("E001", "Test error", true);
        let display = format!("{}", error);
        assert!(display.contains("FATAL"));
        assert!(display.contains("E001"));
    }

    #[test]
    fn test_diff_item() {
        let item = DiffItem::new("version", "1.0", "2.0");
        let display = format!("{}", item);
        assert!(display.contains("version"));
        assert!(display.contains("1.0"));
        assert!(display.contains("2.0"));
    }

    #[test]
    fn test_weight_diff_identical() {
        let a = vec![1.0_f32, 2.0, 3.0];
        let b = vec![1.0_f32, 2.0, 3.0];
        let diff = WeightDiff::from_slices(&a, &b);

        assert!(diff.is_identical());
        assert!((diff.cosine_similarity - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_weight_diff_different() {
        let a = vec![1.0_f32, 2.0, 3.0];
        let b = vec![1.0_f32, 2.0, 4.0];
        let diff = WeightDiff::from_slices(&a, &b);

        assert!(!diff.is_identical());
        assert_eq!(diff.changed_count, 1);
        assert!((diff.max_diff - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_diff_result() {
        let mut diff = DiffResult::new("model_a.apr", "model_b.apr");
        assert!(diff.is_identical());

        diff.header_diff
            .push(DiffItem::new("version", "1.0", "2.0"));
        assert!(!diff.is_identical());
        assert_eq!(diff.diff_count(), 1);
    }

    #[test]
    fn test_inspect_options_default() {
        let opts = InspectOptions::default();
        assert!(opts.include_weights);
        assert!(opts.include_quality);
    }

    #[test]
    fn test_inspect_options_quick() {
        let opts = InspectOptions::quick();
        assert!(!opts.include_weights);
        assert!(!opts.include_quality);
    }

    #[test]
    fn test_inspection_result() {
        let header = HeaderInspection::new();
        let metadata = MetadataInspection::new("LinearRegression");
        let result = InspectionResult::new(header, metadata);

        assert!(!result.has_issues());
        assert!(result.is_valid());
    }

    #[test]
    fn test_training_info() {
        let info = TrainingInfo::new();
        assert!(info.trained_at.is_none());
        assert!(info.dataset_name.is_none());
    }

    #[test]
    fn test_license_info() {
        let info = LicenseInfo::new("MIT");
        assert_eq!(info.license_type, "MIT");
        assert!(!info.has_restrictions());
    }

    #[test]
    fn test_metadata_inspection() {
        let meta = MetadataInspection::new("RandomForest");
        assert_eq!(meta.model_type_name, "RandomForest");
        assert!(!meta.has_training_info());
        assert!(!meta.is_licensed());
    }

    // =========================================================================
    // Additional coverage tests
    // =========================================================================

    #[test]
    fn test_inspection_result_with_warnings() {
        let header = HeaderInspection::new();
        let metadata = MetadataInspection::new("Test");
        let mut result = InspectionResult::new(header, metadata);

        result.warnings.push(InspectionWarning::new("W001", "Test warning"));
        assert!(result.has_issues());
        assert!(result.is_valid()); // Still valid, just has warnings
        assert_eq!(result.issue_count(), 1);
    }

    #[test]
    fn test_inspection_result_with_errors() {
        let header = HeaderInspection::new();
        let metadata = MetadataInspection::new("Test");
        let mut result = InspectionResult::new(header, metadata);

        result.errors.push(InspectionError::new("E001", "Test error", true));
        assert!(result.has_issues());
        assert!(!result.is_valid()); // Invalid due to errors
        assert_eq!(result.issue_count(), 1);
    }

    #[test]
    fn test_inspection_result_with_both() {
        let header = HeaderInspection::new();
        let metadata = MetadataInspection::new("Test");
        let mut result = InspectionResult::new(header, metadata);

        result.warnings.push(InspectionWarning::new("W001", "Warning"));
        result.errors.push(InspectionError::new("E001", "Error", false));
        assert_eq!(result.issue_count(), 2);
    }

    #[test]
    fn test_header_inspection_compression_ratio_nonzero() {
        let mut header = HeaderInspection::new();
        header.compressed_size = 500;
        header.uncompressed_size = 1000;
        assert!((header.compression_ratio() - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_header_inspection_invalid_magic() {
        let mut header = HeaderInspection::new();
        header.magic_valid = false;
        assert!(!header.is_valid());
    }

    #[test]
    fn test_header_inspection_unsupported_version() {
        let mut header = HeaderInspection::new();
        header.version_supported = false;
        assert!(!header.is_valid());
    }

    #[test]
    fn test_header_flags_all_set() {
        let flags = HeaderFlags {
            compressed: true,
            signed: true,
            encrypted: true,
            streaming: true,
            licensed: true,
            quantized: true,
        };
        let list = flags.flag_list();
        assert_eq!(list.len(), 6);
        assert!(list.contains(&"COMPRESSED"));
        assert!(list.contains(&"SIGNED"));
        assert!(list.contains(&"ENCRYPTED"));
        assert!(list.contains(&"STREAMING"));
        assert!(list.contains(&"LICENSED"));
        assert!(list.contains(&"QUANTIZED"));
    }

    #[test]
    fn test_header_flags_byte_roundtrip_all() {
        let byte = 0x3F; // All 6 flags set
        let flags = HeaderFlags::from_byte(byte);
        assert!(flags.compressed);
        assert!(flags.signed);
        assert!(flags.encrypted);
        assert!(flags.streaming);
        assert!(flags.licensed);
        assert!(flags.quantized);
        assert_eq!(flags.to_byte(), byte);
    }

    #[test]
    fn test_header_flags_empty() {
        let flags = HeaderFlags::default();
        assert!(flags.flag_list().is_empty());
        assert_eq!(flags.to_byte(), 0);
    }

    #[test]
    fn test_metadata_with_training_info() {
        let mut meta = MetadataInspection::new("Model");
        meta.training_info = Some(TrainingInfo::new());
        assert!(meta.has_training_info());
    }

    #[test]
    fn test_metadata_with_license_info() {
        let mut meta = MetadataInspection::new("Model");
        meta.license_info = Some(LicenseInfo::new("MIT"));
        assert!(meta.is_licensed());
    }

    #[test]
    fn test_training_info_default() {
        let info = TrainingInfo::default();
        assert!(info.trained_at.is_none());
        assert!(info.duration.is_none());
        assert!(info.dataset_name.is_none());
        assert!(info.n_samples.is_none());
        assert!(info.final_loss.is_none());
        assert!(info.framework.is_none());
        assert!(info.framework_version.is_none());
    }

    #[test]
    fn test_license_info_with_restrictions() {
        let mut info = LicenseInfo::new("Proprietary");
        info.restrictions.push("No commercial use".to_string());
        info.restrictions.push("No redistribution".to_string());
        assert!(info.has_restrictions());
        assert_eq!(info.restrictions.len(), 2);
    }

    #[test]
    fn test_weight_stats_with_inf() {
        let weights = vec![1.0_f32, f32::INFINITY, 3.0];
        let stats = WeightStats::from_slice(&weights);
        assert_eq!(stats.inf_count, 1);
        assert!(stats.has_issues());
        assert_eq!(stats.health_status(), WeightHealth::Critical);
    }

    #[test]
    fn test_weight_stats_high_sparsity() {
        let weights = vec![0.0_f32; 100];
        let stats = WeightStats::from_slice(&weights);
        assert!((stats.sparsity - 1.0).abs() < 0.001);
        assert_eq!(stats.health_status(), WeightHealth::Warning);
    }

    #[test]
    fn test_weight_stats_low_variance() {
        let weights = vec![1.0_f32; 100];
        let stats = WeightStats::from_slice(&weights);
        assert!(stats.std < 1e-10);
        assert_eq!(stats.health_status(), WeightHealth::Warning);
    }

    #[test]
    fn test_weight_stats_single_element() {
        let weights = vec![5.0_f32];
        let stats = WeightStats::from_slice(&weights);
        assert_eq!(stats.count, 1);
        assert!((stats.mean - 5.0).abs() < 0.001);
        assert!((stats.std - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_weight_health_descriptions() {
        assert!(!WeightHealth::Healthy.description().is_empty());
        assert!(!WeightHealth::Warning.description().is_empty());
        assert!(!WeightHealth::Critical.description().is_empty());
        assert!(WeightHealth::Warning.description().contains("potential"));
        assert!(WeightHealth::Critical.description().contains("critical"));
    }

    #[test]
    fn test_inspection_warning_without_recommendation() {
        let warning = InspectionWarning::new("W001", "Test warning");
        assert!(warning.recommendation.is_none());
        let display = format!("{}", warning);
        assert!(display.contains("W001"));
        assert!(!display.contains("Recommendation"));
    }

    #[test]
    fn test_inspection_error_nonfatal() {
        let error = InspectionError::new("E001", "Non-fatal error", false);
        assert!(!error.fatal);
        let display = format!("{}", error);
        assert!(display.contains("ERROR"));
        assert!(!display.contains("FATAL"));
    }

    #[test]
    fn test_diff_result_with_weight_diff() {
        let mut diff = DiffResult::new("a.apr", "b.apr");
        diff.weight_diff = Some(WeightDiff::empty());
        assert!(diff.is_identical()); // Empty diff means identical

        diff.weight_diff = Some(WeightDiff {
            changed_count: 5,
            max_diff: 0.1,
            mean_diff: 0.05,
            l2_distance: 0.2,
            cosine_similarity: 0.99,
        });
        assert!(!diff.is_identical());
        assert_eq!(diff.diff_count(), 1); // One weight diff entry
    }

    #[test]
    fn test_weight_diff_empty_or_mismatched() {
        let empty1: Vec<f32> = vec![];
        let empty2: Vec<f32> = vec![];
        let diff = WeightDiff::from_slices(&empty1, &empty2);
        assert!(diff.is_identical());

        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0]; // Different lengths
        let diff = WeightDiff::from_slices(&a, &b);
        assert!(diff.is_identical()); // Empty diff due to length mismatch
    }

    #[test]
    fn test_weight_diff_zero_norms() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![0.0, 0.0, 0.0];
        let diff = WeightDiff::from_slices(&a, &b);
        assert!((diff.cosine_similarity - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_inspect_options_new() {
        let opts = InspectOptions::new();
        assert!(opts.include_weights);
        assert!(opts.include_quality);
    }

    #[test]
    fn test_inspect_options_full() {
        let opts = InspectOptions::full();
        assert!(opts.include_weights);
        assert!(opts.include_quality);
        assert!(opts.verbose);
        assert_eq!(opts.max_weights, usize::MAX);
    }

    #[test]
    fn test_diff_item_display() {
        let item = DiffItem::new("field", "old", "new");
        let display = format!("{}", item);
        assert!(display.contains("field"));
        assert!(display.contains("old"));
        assert!(display.contains("new"));
        assert!(display.contains("->"));
    }

    #[test]
    fn test_weight_stats_norms() {
        let weights = vec![1.0_f32, 2.0, 3.0];
        let stats = WeightStats::from_slice(&weights);
        // L1 norm = |1| + |2| + |3| = 6
        assert!((stats.l1_norm - 6.0).abs() < 0.001);
        // L2 norm = sqrt(1 + 4 + 9) = sqrt(14)
        assert!((stats.l2_norm - 14.0_f64.sqrt()).abs() < 0.001);
    }

    #[test]
    fn test_header_inspection_default() {
        let header = HeaderInspection::default();
        assert_eq!(header.magic_string(), "APRN");
        assert_eq!(header.version_string(), "1.0");
    }

    #[test]
    fn test_weight_diff_diff_count_zero() {
        let diff = WeightDiff::empty();
        assert_eq!(diff.diff_count(), 0);
    }

    #[test]
    fn test_weight_diff_diff_count_nonzero() {
        let diff = WeightDiff {
            changed_count: 10,
            max_diff: 0.5,
            mean_diff: 0.2,
            l2_distance: 1.0,
            cosine_similarity: 0.95,
        };
        assert_eq!(diff.diff_count(), 1);
    }
}

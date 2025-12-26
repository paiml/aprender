//! Data Embedding Module (spec §4)
//!
//! Provides types for embedding test data and tiny models directly within
//! `.apr` model files. This enables:
//! - Educational demos with bundled sample data
//! - Zero-dependency model validation
//! - Efficient small model representation
//!
//! # Toyota Way Principles
//! - **Traceability**: DataProvenance tracks complete data lineage
//! - **Muda Elimination**: Compression strategies minimize waste
//! - **Kaizen**: TinyModelRepr optimizes for common small model patterns

pub mod tiny;

pub use tiny::TinyModelRepr;

/// Embedded test data for model validation and demos (spec §4.1)
///
/// Enables models to bundle sample datasets for:
/// - Self-contained educational examples
/// - Validation without external data dependencies
/// - Quick model quality verification
///
/// # Example
/// ```
/// use aprender::embed::{EmbeddedTestData, DataProvenance, DataCompression};
///
/// let test_data = EmbeddedTestData::new(
///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
///     (3, 2),  // 3 samples, 2 features
/// )
/// .with_targets(vec![0.0, 1.0, 0.0])
/// .with_feature_names(vec!["feature_a".into(), "feature_b".into()])
/// .with_provenance(DataProvenance::new("UCI Iris").with_license("CC0"));
///
/// assert_eq!(test_data.n_samples(), 3);
/// assert_eq!(test_data.n_features(), 2);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddedTestData {
    /// Feature matrix (flattened, row-major)
    pub x_data: Vec<f32>,

    /// Feature matrix shape (n_samples, n_features)
    pub x_shape: (usize, usize),

    /// Target vector (for supervised models)
    pub y_data: Option<Vec<f32>>,

    /// Feature names (for inspection/debugging)
    pub feature_names: Option<Vec<String>>,

    /// Sample identifiers (for traceability)
    pub sample_ids: Option<Vec<String>>,

    /// Data provenance (Toyota Way: traceability)
    pub provenance: Option<DataProvenance>,

    /// Compression strategy used
    pub compression: DataCompression,
}

impl EmbeddedTestData {
    /// Create new embedded test data with features
    ///
    /// # Arguments
    /// * `x_data` - Flattened feature matrix (row-major order)
    /// * `x_shape` - Shape as (n_samples, n_features)
    ///
    /// # Panics
    /// Panics if `x_data.len() != x_shape.0 * x_shape.1`
    #[must_use]
    pub fn new(x_data: Vec<f32>, x_shape: (usize, usize)) -> Self {
        assert_eq!(
            x_data.len(),
            x_shape.0 * x_shape.1,
            "Data length {} doesn't match shape {:?}",
            x_data.len(),
            x_shape
        );
        Self {
            x_data,
            x_shape,
            y_data: None,
            feature_names: None,
            sample_ids: None,
            provenance: None,
            compression: DataCompression::None,
        }
    }

    /// Add target values
    #[must_use]
    pub fn with_targets(mut self, y_data: Vec<f32>) -> Self {
        assert_eq!(
            y_data.len(),
            self.x_shape.0,
            "Target length {} doesn't match n_samples {}",
            y_data.len(),
            self.x_shape.0
        );
        self.y_data = Some(y_data);
        self
    }

    /// Add feature names
    #[must_use]
    pub fn with_feature_names(mut self, names: Vec<String>) -> Self {
        assert_eq!(
            names.len(),
            self.x_shape.1,
            "Feature names length {} doesn't match n_features {}",
            names.len(),
            self.x_shape.1
        );
        self.feature_names = Some(names);
        self
    }

    /// Add sample identifiers
    #[must_use]
    pub fn with_sample_ids(mut self, ids: Vec<String>) -> Self {
        assert_eq!(
            ids.len(),
            self.x_shape.0,
            "Sample IDs length {} doesn't match n_samples {}",
            ids.len(),
            self.x_shape.0
        );
        self.sample_ids = Some(ids);
        self
    }

    /// Add data provenance
    #[must_use]
    pub fn with_provenance(mut self, provenance: DataProvenance) -> Self {
        self.provenance = Some(provenance);
        self
    }

    /// Set compression strategy
    #[must_use]
    pub fn with_compression(mut self, compression: DataCompression) -> Self {
        self.compression = compression;
        self
    }

    /// Number of samples in the dataset
    #[must_use]
    pub const fn n_samples(&self) -> usize {
        self.x_shape.0
    }

    /// Number of features per sample
    #[must_use]
    pub const fn n_features(&self) -> usize {
        self.x_shape.1
    }

    /// Total size in bytes (uncompressed)
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        let base_size = self.x_data.len() * 4; // f32 = 4 bytes
        let y_size = self.y_data.as_ref().map_or(0, |y| y.len() * 4);
        base_size + y_size
    }

    /// Get feature row by index
    ///
    /// # Arguments
    /// * `idx` - Sample index
    ///
    /// # Returns
    /// Slice of features for the given sample, or None if out of bounds
    #[must_use]
    pub fn get_row(&self, idx: usize) -> Option<&[f32]> {
        if idx >= self.n_samples() {
            return None;
        }
        let start = idx * self.n_features();
        let end = start + self.n_features();
        Some(&self.x_data[start..end])
    }

    /// Get target value by index
    #[must_use]
    pub fn get_target(&self, idx: usize) -> Option<f32> {
        self.y_data.as_ref().and_then(|y| y.get(idx).copied())
    }

    /// Validate data integrity
    ///
    /// Checks:
    /// - Shape consistency
    /// - NaN/Inf values
    /// - Feature name/ID count matching
    pub fn validate(&self) -> Result<(), EmbedError> {
        // Shape check
        if self.x_data.len() != self.x_shape.0 * self.x_shape.1 {
            return Err(EmbedError::ShapeMismatch {
                expected: self.x_shape.0 * self.x_shape.1,
                actual: self.x_data.len(),
            });
        }

        // NaN/Inf check
        for (i, &val) in self.x_data.iter().enumerate() {
            if !val.is_finite() {
                return Err(EmbedError::InvalidValue {
                    index: i,
                    value: val,
                });
            }
        }

        // Target validation
        if let Some(ref y) = self.y_data {
            if y.len() != self.x_shape.0 {
                return Err(EmbedError::TargetMismatch {
                    expected: self.x_shape.0,
                    actual: y.len(),
                });
            }
            for (i, &val) in y.iter().enumerate() {
                if !val.is_finite() {
                    return Err(EmbedError::InvalidValue {
                        index: i,
                        value: val,
                    });
                }
            }
        }

        Ok(())
    }
}

impl Default for EmbeddedTestData {
    fn default() -> Self {
        Self::new(Vec::new(), (0, 0))
    }
}

/// Data provenance for traceability (Toyota Way principle)
///
/// Tracks the complete lineage of embedded data:
/// - Original source dataset
/// - Preprocessing steps applied
/// - Subset selection criteria
/// - License and attribution
///
/// # Reference
/// [Gebru et al. 2021] "Datasheets for Datasets"
#[derive(Debug, Clone, PartialEq)]
pub struct DataProvenance {
    /// Original dataset name (e.g., "UCI Iris")
    pub source: String,

    /// Subset selection criteria
    pub subset_criteria: Option<String>,

    /// Preprocessing steps applied
    pub preprocessing: Vec<String>,

    /// Creation timestamp (ISO 8601)
    pub created_at: String,

    /// License/attribution
    pub license: Option<String>,

    /// Version identifier
    pub version: Option<String>,

    /// Additional metadata
    pub metadata: std::collections::HashMap<String, String>,
}

impl DataProvenance {
    /// Create new provenance with source
    #[must_use]
    pub fn new(source: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            subset_criteria: None,
            preprocessing: Vec::new(),
            created_at: chrono_lite_timestamp(),
            license: None,
            version: None,
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Add subset criteria
    #[must_use]
    pub fn with_subset(mut self, criteria: impl Into<String>) -> Self {
        self.subset_criteria = Some(criteria.into());
        self
    }

    /// Add preprocessing step
    #[must_use]
    pub fn with_preprocessing(mut self, step: impl Into<String>) -> Self {
        self.preprocessing.push(step.into());
        self
    }

    /// Add multiple preprocessing steps
    #[must_use]
    pub fn with_preprocessing_steps(mut self, steps: Vec<String>) -> Self {
        self.preprocessing.extend(steps);
        self
    }

    /// Set license
    #[must_use]
    pub fn with_license(mut self, license: impl Into<String>) -> Self {
        self.license = Some(license.into());
        self
    }

    /// Set version
    #[must_use]
    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.version = Some(version.into());
        self
    }

    /// Add custom metadata
    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Check if provenance is complete (has recommended fields)
    #[must_use]
    pub fn is_complete(&self) -> bool {
        !self.source.is_empty() && self.license.is_some()
    }
}

impl Default for DataProvenance {
    fn default() -> Self {
        Self::new("unknown")
    }
}

/// Data compression strategy (spec §4.2)
///
/// Selects optimal compression based on data characteristics:
/// - **None**: Raw f32 values, zero latency
/// - **Zstd**: General purpose, 2-10x ratio
/// - **DeltaZstd**: Time series/sorted data, 5-20x ratio
/// - **QuantizedEntropy**: ML-specific, 4-8x with minimal accuracy loss
/// - **Sparse**: Sparse features, ratio proportional to sparsity
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DataCompression {
    /// No compression (raw f32 values)
    #[default]
    None,

    /// Zstd compression (general purpose)
    /// Ratio: 2-10x, Speed: 500 MB/s decompress
    Zstd {
        /// Compression level (1-22)
        level: u8,
    },

    /// Delta encoding + Zstd (time series, sorted data)
    /// Ratio: 5-20x for sequential data
    DeltaZstd {
        /// Compression level (1-22)
        level: u8,
    },

    /// Quantization + entropy coding (ML-specific)
    /// Ratio: 4-8x with minimal accuracy loss
    QuantizedEntropy {
        /// Quantization bits (4, 8, 16)
        bits: u8,
    },

    /// Sparse representation (for sparse features)
    /// Ratio: proportional to sparsity
    Sparse {
        /// Threshold below which values are treated as zero
        threshold: u32, // stored as bits of f32
    },
}

impl DataCompression {
    /// Create Zstd compression with default level
    #[must_use]
    pub const fn zstd() -> Self {
        Self::Zstd { level: 3 }
    }

    /// Create Zstd compression with custom level
    #[must_use]
    pub const fn zstd_level(level: u8) -> Self {
        Self::Zstd { level }
    }

    /// Create delta+Zstd compression
    #[must_use]
    pub const fn delta_zstd() -> Self {
        Self::DeltaZstd { level: 3 }
    }

    /// Create quantized entropy compression
    #[must_use]
    pub const fn quantized(bits: u8) -> Self {
        Self::QuantizedEntropy { bits }
    }

    /// Create sparse compression with threshold
    #[must_use]
    pub fn sparse(threshold: f32) -> Self {
        Self::Sparse {
            threshold: threshold.to_bits(),
        }
    }

    /// Human-readable name
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Zstd { .. } => "zstd",
            Self::DeltaZstd { .. } => "delta-zstd",
            Self::QuantizedEntropy { .. } => "quantized-entropy",
            Self::Sparse { .. } => "sparse",
        }
    }

    /// Estimated compression ratio (typical)
    #[must_use]
    pub const fn estimated_ratio(&self) -> f32 {
        match self {
            Self::None => 1.0,
            Self::Zstd { level } => {
                // Higher levels = better compression
                if *level < 5 {
                    2.5
                } else if *level < 10 {
                    4.0
                } else {
                    6.0
                }
            }
            Self::DeltaZstd { level } => {
                if *level < 5 {
                    8.0
                } else {
                    12.0
                }
            }
            Self::QuantizedEntropy { bits } => match bits {
                4 => 8.0,
                8 => 4.0,
                _ => 2.0,
            },
            Self::Sparse { .. } => 5.0, // depends on actual sparsity
        }
    }
}

/// Errors during data embedding operations
#[derive(Debug, Clone)]
pub enum EmbedError {
    /// Data shape doesn't match declared dimensions
    ShapeMismatch { expected: usize, actual: usize },
    /// Target vector length doesn't match samples
    TargetMismatch { expected: usize, actual: usize },
    /// Invalid value (NaN or Inf)
    InvalidValue { index: usize, value: f32 },
    /// Compression error
    CompressionFailed {
        strategy: &'static str,
        message: String,
    },
    /// Decompression error
    DecompressionFailed {
        strategy: &'static str,
        message: String,
    },
}

impl std::fmt::Display for EmbedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ShapeMismatch { expected, actual } => {
                write!(
                    f,
                    "Shape mismatch: expected {expected} elements, got {actual}"
                )
            }
            Self::TargetMismatch { expected, actual } => {
                write!(
                    f,
                    "Target mismatch: expected {expected} samples, got {actual}"
                )
            }
            Self::InvalidValue { index, value } => {
                write!(f, "Invalid value at index {index}: {value}")
            }
            Self::CompressionFailed { strategy, message } => {
                write!(f, "Compression ({strategy}) failed: {message}")
            }
            Self::DecompressionFailed { strategy, message } => {
                write!(f, "Decompression ({strategy}) failed: {message}")
            }
        }
    }
}

impl std::error::Error for EmbedError {}

/// Simple timestamp without chrono dependency
fn chrono_lite_timestamp() -> String {
    // Returns a placeholder; in real code would use actual timestamp
    "2025-01-01T00:00:00Z".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedded_test_data_creation() {
        let data = EmbeddedTestData::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (3, 2));

        assert_eq!(data.n_samples(), 3);
        assert_eq!(data.n_features(), 2);
        assert_eq!(data.size_bytes(), 24); // 6 floats * 4 bytes
    }

    #[test]
    fn test_embedded_test_data_with_targets() {
        let data = EmbeddedTestData::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (3, 2))
            .with_targets(vec![0.0, 1.0, 0.0]);

        assert_eq!(data.y_data, Some(vec![0.0, 1.0, 0.0]));
        assert_eq!(data.size_bytes(), 36); // 6 + 3 floats * 4 bytes
    }

    #[test]
    fn test_embedded_test_data_with_feature_names() {
        let data = EmbeddedTestData::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (3, 2))
            .with_feature_names(vec!["a".into(), "b".into()]);

        assert_eq!(data.feature_names, Some(vec!["a".into(), "b".into()]));
    }

    #[test]
    fn test_embedded_test_data_get_row() {
        let data = EmbeddedTestData::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (3, 2));

        assert_eq!(data.get_row(0), Some(&[1.0, 2.0][..]));
        assert_eq!(data.get_row(1), Some(&[3.0, 4.0][..]));
        assert_eq!(data.get_row(2), Some(&[5.0, 6.0][..]));
        assert_eq!(data.get_row(3), None);
    }

    #[test]
    fn test_embedded_test_data_get_target() {
        let data =
            EmbeddedTestData::new(vec![1.0, 2.0, 3.0, 4.0], (2, 2)).with_targets(vec![0.0, 1.0]);

        assert_eq!(data.get_target(0), Some(0.0));
        assert_eq!(data.get_target(1), Some(1.0));
        assert_eq!(data.get_target(2), None);
    }

    #[test]
    fn test_embedded_test_data_validate() {
        // Valid data
        let valid = EmbeddedTestData::new(vec![1.0, 2.0, 3.0, 4.0], (2, 2));
        assert!(valid.validate().is_ok());

        // Invalid: contains NaN
        let mut invalid_nan = EmbeddedTestData::new(vec![1.0, f32::NAN, 3.0, 4.0], (2, 2));
        invalid_nan.x_data[1] = f32::NAN;
        let err = invalid_nan.validate();
        assert!(matches!(err, Err(EmbedError::InvalidValue { .. })));

        // Invalid: contains Inf
        let mut invalid_inf = EmbeddedTestData::new(vec![1.0, 2.0, 3.0, 4.0], (2, 2));
        invalid_inf.x_data[0] = f32::INFINITY;
        let err = invalid_inf.validate();
        assert!(matches!(err, Err(EmbedError::InvalidValue { .. })));
    }

    #[test]
    fn test_data_provenance() {
        let provenance = DataProvenance::new("UCI Iris")
            .with_subset("first 50 samples")
            .with_preprocessing("normalize")
            .with_preprocessing("pca")
            .with_license("CC0")
            .with_version("1.0")
            .with_metadata("author", "Fisher");

        assert_eq!(provenance.source, "UCI Iris");
        assert_eq!(provenance.subset_criteria, Some("first 50 samples".into()));
        assert_eq!(provenance.preprocessing, vec!["normalize", "pca"]);
        assert_eq!(provenance.license, Some("CC0".into()));
        assert_eq!(provenance.version, Some("1.0".into()));
        assert_eq!(provenance.metadata.get("author"), Some(&"Fisher".into()));
        assert!(provenance.is_complete());
    }

    #[test]
    fn test_data_provenance_incomplete() {
        let incomplete = DataProvenance::new("test");
        assert!(!incomplete.is_complete()); // missing license

        let complete = DataProvenance::new("test").with_license("MIT");
        assert!(complete.is_complete());
    }

    #[test]
    fn test_data_compression_none() {
        let comp = DataCompression::None;
        assert_eq!(comp.name(), "none");
        assert!((comp.estimated_ratio() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_data_compression_zstd() {
        let comp = DataCompression::zstd();
        assert_eq!(comp.name(), "zstd");
        assert!(comp.estimated_ratio() > 1.0);

        let comp_high = DataCompression::zstd_level(15);
        assert!(comp_high.estimated_ratio() > comp.estimated_ratio());
    }

    #[test]
    fn test_data_compression_delta_zstd() {
        let comp = DataCompression::delta_zstd();
        assert_eq!(comp.name(), "delta-zstd");
        assert!(comp.estimated_ratio() > DataCompression::zstd().estimated_ratio());
    }

    #[test]
    fn test_data_compression_quantized() {
        let comp_4bit = DataCompression::quantized(4);
        let comp_8bit = DataCompression::quantized(8);

        assert_eq!(comp_4bit.name(), "quantized-entropy");
        assert!(comp_4bit.estimated_ratio() > comp_8bit.estimated_ratio());
    }

    #[test]
    fn test_data_compression_sparse() {
        let comp = DataCompression::sparse(0.001);
        assert_eq!(comp.name(), "sparse");
        assert!(comp.estimated_ratio() > 1.0);
    }

    #[test]
    fn test_embed_error_display() {
        let err = EmbedError::ShapeMismatch {
            expected: 100,
            actual: 50,
        };
        let msg = format!("{err}");
        assert!(msg.contains("100"));
        assert!(msg.contains("50"));

        let err = EmbedError::InvalidValue {
            index: 5,
            value: f32::NAN,
        };
        let msg = format!("{err}");
        assert!(msg.contains("5"));
    }

    #[test]
    fn test_embedded_test_data_with_provenance() {
        let data = EmbeddedTestData::new(vec![1.0, 2.0], (1, 2))
            .with_provenance(DataProvenance::new("test").with_license("MIT"));

        assert!(data.provenance.is_some());
        assert!(data.provenance.as_ref().unwrap().is_complete());
    }

    #[test]
    fn test_embedded_test_data_with_sample_ids() {
        let data = EmbeddedTestData::new(vec![1.0, 2.0, 3.0, 4.0], (2, 2))
            .with_sample_ids(vec!["sample_1".into(), "sample_2".into()]);

        assert_eq!(
            data.sample_ids,
            Some(vec!["sample_1".into(), "sample_2".into()])
        );
    }

    #[test]
    #[should_panic(expected = "Data length")]
    fn test_embedded_test_data_shape_mismatch_panics() {
        let _ = EmbeddedTestData::new(vec![1.0, 2.0, 3.0], (2, 2)); // 3 != 4
    }

    #[test]
    #[should_panic(expected = "Target length")]
    fn test_embedded_test_data_target_mismatch_panics() {
        let _ = EmbeddedTestData::new(vec![1.0, 2.0, 3.0, 4.0], (2, 2))
            .with_targets(vec![0.0, 1.0, 2.0]); // 3 != 2
    }

    #[test]
    fn test_embedded_test_data_default() {
        let data = EmbeddedTestData::default();
        assert_eq!(data.n_samples(), 0);
        assert_eq!(data.n_features(), 0);
        assert!(data.x_data.is_empty());
    }

    #[test]
    fn test_data_compression_default() {
        let comp = DataCompression::default();
        assert_eq!(comp, DataCompression::None);
    }

    #[test]
    fn test_embedded_test_data_with_compression() {
        let data =
            EmbeddedTestData::new(vec![1.0, 2.0], (1, 2)).with_compression(DataCompression::zstd());

        assert_eq!(data.compression, DataCompression::Zstd { level: 3 });
    }

    // =========================================================================
    // Additional coverage tests
    // =========================================================================

    #[test]
    fn test_data_provenance_default() {
        let prov = DataProvenance::default();
        assert_eq!(prov.source, "unknown");
        assert!(!prov.is_complete()); // no license
    }

    #[test]
    fn test_data_provenance_with_preprocessing_steps() {
        let prov = DataProvenance::new("test")
            .with_preprocessing_steps(vec!["step1".into(), "step2".into()]);
        assert_eq!(prov.preprocessing.len(), 2);
    }

    #[test]
    fn test_data_compression_delta_zstd_high_level() {
        let comp = DataCompression::DeltaZstd { level: 10 };
        assert_eq!(comp.name(), "delta-zstd");
        assert!((comp.estimated_ratio() - 12.0).abs() < 0.1);
    }

    #[test]
    fn test_data_compression_zstd_high_level() {
        let comp = DataCompression::zstd_level(12);
        assert!((comp.estimated_ratio() - 6.0).abs() < 0.1);

        let comp_medium = DataCompression::zstd_level(7);
        assert!((comp_medium.estimated_ratio() - 4.0).abs() < 0.1);
    }

    #[test]
    fn test_data_compression_quantized_16bit() {
        let comp = DataCompression::quantized(16);
        assert!((comp.estimated_ratio() - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_embed_error_compression_failed_display() {
        let err = EmbedError::CompressionFailed {
            strategy: "zstd",
            message: "out of memory".into(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("zstd"));
        assert!(msg.contains("out of memory"));
    }

    #[test]
    fn test_embed_error_decompression_failed_display() {
        let err = EmbedError::DecompressionFailed {
            strategy: "delta-zstd",
            message: "corrupt data".into(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("delta-zstd"));
        assert!(msg.contains("corrupt data"));
    }

    #[test]
    fn test_embed_error_target_mismatch_display() {
        let err = EmbedError::TargetMismatch {
            expected: 10,
            actual: 5,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("10"));
        assert!(msg.contains("5"));
    }

    #[test]
    fn test_embedded_test_data_validate_target_nan() {
        let mut data = EmbeddedTestData::new(vec![1.0, 2.0], (1, 2)).with_targets(vec![0.0]);
        data.y_data = Some(vec![f32::NAN]);

        let err = data.validate();
        assert!(matches!(err, Err(EmbedError::InvalidValue { .. })));
    }

    #[test]
    fn test_embedded_test_data_validate_target_inf() {
        let mut data = EmbeddedTestData::new(vec![1.0, 2.0], (1, 2)).with_targets(vec![0.0]);
        data.y_data = Some(vec![f32::INFINITY]);

        let err = data.validate();
        assert!(matches!(err, Err(EmbedError::InvalidValue { .. })));
    }

    #[test]
    fn test_embedded_test_data_validate_target_mismatch() {
        let mut data = EmbeddedTestData::new(vec![1.0, 2.0, 3.0, 4.0], (2, 2));
        // Manually set mismatched targets
        data.y_data = Some(vec![0.0, 1.0, 2.0]); // 3 targets for 2 samples

        let err = data.validate();
        assert!(matches!(err, Err(EmbedError::TargetMismatch { .. })));
    }

    #[test]
    fn test_embedded_test_data_clone() {
        let data = EmbeddedTestData::new(vec![1.0, 2.0], (1, 2)).with_targets(vec![1.0]);
        let cloned = data.clone();
        assert_eq!(cloned.x_data, data.x_data);
        assert_eq!(cloned.y_data, data.y_data);
    }

    #[test]
    fn test_data_provenance_clone() {
        let prov = DataProvenance::new("test").with_license("MIT");
        let cloned = prov.clone();
        assert_eq!(cloned.source, prov.source);
        assert_eq!(cloned.license, prov.license);
    }

    #[test]
    fn test_data_compression_copy() {
        let comp = DataCompression::zstd();
        let copied = comp;
        assert_eq!(copied.name(), "zstd");
    }

    #[test]
    fn test_embed_error_clone() {
        let err = EmbedError::ShapeMismatch {
            expected: 10,
            actual: 5,
        };
        let cloned = err.clone();
        let msg = format!("{}", cloned);
        assert!(msg.contains("10"));
    }

    #[test]
    #[should_panic(expected = "Feature names length")]
    fn test_embedded_test_data_feature_names_mismatch() {
        let _ = EmbeddedTestData::new(vec![1.0, 2.0, 3.0, 4.0], (2, 2)).with_feature_names(vec![
            "a".into(),
            "b".into(),
            "c".into(),
        ]); // 3 != 2
    }

    #[test]
    #[should_panic(expected = "Sample IDs length")]
    fn test_embedded_test_data_sample_ids_mismatch() {
        let _ = EmbeddedTestData::new(vec![1.0, 2.0, 3.0, 4.0], (2, 2)).with_sample_ids(vec![
            "a".into(),
            "b".into(),
            "c".into(),
        ]); // 3 != 2
    }

    #[test]
    fn test_embedded_test_data_partial_eq() {
        let data1 = EmbeddedTestData::new(vec![1.0, 2.0], (1, 2));
        let data2 = EmbeddedTestData::new(vec![1.0, 2.0], (1, 2));
        let data3 = EmbeddedTestData::new(vec![1.0, 3.0], (1, 2));

        assert_eq!(data1, data2);
        assert_ne!(data1, data3);
    }

    #[test]
    fn test_data_provenance_partial_eq() {
        let prov1 = DataProvenance::new("test");
        let prov2 = DataProvenance::new("test");
        let prov3 = DataProvenance::new("other");

        assert_eq!(prov1.source, prov2.source);
        assert_ne!(prov1.source, prov3.source);
    }
}

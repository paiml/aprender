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

    /// Feature matrix shape (`n_samples`, `n_features`)
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
    /// * `x_shape` - Shape as (`n_samples`, `n_features`)
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
/// - **`DeltaZstd`**: Time series/sorted data, 5-20x ratio
/// - **`QuantizedEntropy`**: ML-specific, 4-8x with minimal accuracy loss
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

include!("embed_error.rs");
include!("tests_embedded_data.rs");

//! Quality Profiles (GH-10)
//!
//! Quality profile for customizing scoring rules per data type.

use std::collections::HashSet;

/// Quality profile for customizing scoring rules per data type.
///
/// Different data types (doctest corpora, ML training sets, time series, etc.)
/// have different expectations. For example:
/// - Doctest corpus: `source` and `version` columns are expected to be constant
/// - ML training: features should have high variance, labels can be categorical
/// - Time series: timestamps should be unique and sequential
///
/// # Example
///
/// ```ignore
/// let profile = QualityProfile::doctest_corpus();
/// let score = profile.score_report(&report);
/// ```
#[derive(Debug, Clone)]
pub struct QualityProfile {
    /// Profile name for display
    pub name: String,
    /// Description of what this profile is for
    pub description: String,
    /// Columns that are expected to be constant (not penalized)
    pub expected_constant_columns: HashSet<String>,
    /// Columns where high null ratio is acceptable
    pub nullable_columns: HashSet<String>,
    /// Maximum acceptable null ratio (default: 0.1)
    pub max_null_ratio: f64,
    /// Maximum acceptable duplicate ratio (default: 0.5)
    pub max_duplicate_ratio: f64,
    /// Minimum cardinality before flagging as low (default: 2)
    pub min_cardinality: usize,
    /// Maximum outlier ratio to report (default: 0.05)
    pub max_outlier_ratio: f64,
    /// Maximum duplicate row ratio (default: 0.01)
    pub max_duplicate_row_ratio: f64,
    /// Whether to penalize constant columns not in expected list
    pub penalize_unexpected_constants: bool,
    /// Whether this profile requires a signature column (for doctest)
    pub require_signature: bool,
}

impl Default for QualityProfile {
    fn default() -> Self {
        Self {
            name: "default".to_string(),
            description: "General-purpose quality profile".to_string(),
            expected_constant_columns: HashSet::new(),
            nullable_columns: HashSet::new(),
            max_null_ratio: 0.1,
            max_duplicate_ratio: 0.5,
            min_cardinality: 2,
            max_outlier_ratio: 0.05,
            max_duplicate_row_ratio: 0.01,
            penalize_unexpected_constants: true,
            require_signature: false,
        }
    }
}

impl QualityProfile {
    /// Create a new profile with custom name
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Default::default()
        }
    }

    /// Get profile by name
    #[must_use]
    pub fn by_name(name: &str) -> Option<Self> {
        match name {
            "default" => Some(Self::default()),
            "doctest-corpus" | "doctest" => Some(Self::doctest_corpus()),
            "ml-training" | "ml" => Some(Self::ml_training()),
            "time-series" | "timeseries" => Some(Self::time_series()),
            _ => None,
        }
    }

    /// List available profile names
    #[must_use]
    pub fn available_profiles() -> Vec<&'static str> {
        vec!["default", "doctest-corpus", "ml-training", "time-series"]
    }

    /// Doctest corpus profile - for Python doctest extraction datasets.
    ///
    /// Expects:
    /// - `source` and `version` columns to be constant (single crate/version)
    /// - `signature` column may have nulls (module-level doctests)
    /// - `input`, `expected`, `function` should be non-null
    #[must_use]
    pub fn doctest_corpus() -> Self {
        let mut expected_constants = HashSet::new();
        expected_constants.insert("source".to_string());
        expected_constants.insert("version".to_string());

        let mut nullable = HashSet::new();
        nullable.insert("signature".to_string()); // Module-level doctests have no signature

        Self {
            name: "doctest-corpus".to_string(),
            description: "Profile for Python doctest extraction datasets".to_string(),
            expected_constant_columns: expected_constants,
            nullable_columns: nullable,
            max_null_ratio: 0.05,     // Stricter for doctest data
            max_duplicate_ratio: 0.3, // Some duplicate inputs are normal
            min_cardinality: 2,
            max_outlier_ratio: 0.05,
            max_duplicate_row_ratio: 0.0, // No exact duplicate rows allowed
            penalize_unexpected_constants: true,
            require_signature: false, // Relaxed - signature nulls are OK for module doctests
        }
    }

    /// ML training profile - for machine learning datasets.
    ///
    /// Expects:
    /// - Features to have reasonable variance
    /// - Labels can be categorical (low cardinality OK)
    /// - No null values in features or labels
    #[must_use]
    pub fn ml_training() -> Self {
        Self {
            name: "ml-training".to_string(),
            description: "Profile for machine learning training datasets".to_string(),
            expected_constant_columns: HashSet::new(),
            nullable_columns: HashSet::new(),
            max_null_ratio: 0.0,      // No nulls allowed in training data
            max_duplicate_ratio: 0.8, // Higher tolerance for categorical features
            min_cardinality: 2,
            max_outlier_ratio: 0.1, // More tolerant of outliers
            max_duplicate_row_ratio: 0.01,
            penalize_unexpected_constants: true,
            require_signature: false,
        }
    }

    /// Time series profile - for temporal data.
    ///
    /// Expects:
    /// - Timestamp column should be unique
    /// - Data should have temporal patterns
    #[must_use]
    pub fn time_series() -> Self {
        Self {
            name: "time-series".to_string(),
            description: "Profile for time series datasets".to_string(),
            expected_constant_columns: HashSet::new(),
            nullable_columns: HashSet::new(),
            max_null_ratio: 0.05,
            max_duplicate_ratio: 0.5,
            min_cardinality: 2,
            max_outlier_ratio: 0.1,       // Time series often have outliers
            max_duplicate_row_ratio: 0.0, // No duplicate rows (each timestamp unique)
            penalize_unexpected_constants: true,
            require_signature: false,
        }
    }

    /// Set description
    #[must_use]
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Add an expected constant column
    #[must_use]
    pub fn with_expected_constant(mut self, column: impl Into<String>) -> Self {
        self.expected_constant_columns.insert(column.into());
        self
    }

    /// Add a nullable column
    #[must_use]
    pub fn with_nullable(mut self, column: impl Into<String>) -> Self {
        self.nullable_columns.insert(column.into());
        self
    }

    /// Set max null ratio
    #[must_use]
    pub fn with_max_null_ratio(mut self, ratio: f64) -> Self {
        self.max_null_ratio = ratio;
        self
    }

    /// Set max duplicate ratio
    #[must_use]
    pub fn with_max_duplicate_ratio(mut self, ratio: f64) -> Self {
        self.max_duplicate_ratio = ratio;
        self
    }

    /// Check if a column is expected to be constant
    #[must_use]
    pub fn is_expected_constant(&self, column: &str) -> bool {
        self.expected_constant_columns.contains(column)
    }

    /// Check if a column is allowed to have nulls
    #[must_use]
    pub fn is_nullable(&self, column: &str) -> bool {
        self.nullable_columns.contains(column)
    }

    /// Get effective null threshold for a column
    #[must_use]
    pub fn null_threshold_for(&self, column: &str) -> f64 {
        if self.is_nullable(column) {
            1.0 // Allow up to 100% nulls for nullable columns
        } else {
            self.max_null_ratio
        }
    }
}

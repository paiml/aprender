//! Feature type enumeration for auto-inference

/// Inferred feature type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FeatureType {
    /// Continuous numeric values (float32/float64)
    Numeric,
    /// Discrete categories with limited cardinality
    Categorical,
    /// Free-form text requiring tokenization
    Text,
    /// Timestamp/datetime values
    DateTime,
    /// Pre-computed embedding vectors
    Embedding,
    /// Binary classification target
    BinaryTarget,
    /// Multi-class classification target
    MultiClassTarget,
    /// Regression target
    RegressionTarget,
    /// Sequence of tokens (for language models)
    TokenSequence,
    /// Unknown/ambiguous type
    Unknown,
}

impl std::fmt::Display for FeatureType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Numeric => write!(f, "numeric"),
            Self::Categorical => write!(f, "categorical"),
            Self::Text => write!(f, "text"),
            Self::DateTime => write!(f, "datetime"),
            Self::Embedding => write!(f, "embedding"),
            Self::BinaryTarget => write!(f, "binary_target"),
            Self::MultiClassTarget => write!(f, "multiclass_target"),
            Self::RegressionTarget => write!(f, "regression_target"),
            Self::TokenSequence => write!(f, "token_sequence"),
            Self::Unknown => write!(f, "unknown"),
        }
    }
}

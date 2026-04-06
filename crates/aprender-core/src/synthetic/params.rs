//! Type-safe synthetic data hyperparameter definitions.
//!
//! Parameter keys are enums rather than strings to catch typos at compile time.
//! This eliminates an entire class of runtime errors (Poka-Yoke principle).

use crate::automl::params::ParamKey;
use std::fmt;

/// Synthetic data generation hyperparameters.
///
/// These parameters can be added to an `AutoML` search space to jointly
/// optimize data augmentation alongside model hyperparameters.
///
/// # Example
///
/// ```
/// use aprender::synthetic::SyntheticParam;
/// use aprender::automl::params::ParamKey;
///
/// let param = SyntheticParam::AugmentationRatio;
/// assert_eq!(param.name(), "augmentation_ratio");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SyntheticParam {
    /// Ratio of synthetic to original data [0.0, 10.0].
    AugmentationRatio,

    /// Minimum quality threshold for accepting samples [0.0, 1.0].
    QualityThreshold,

    /// Weight for diversity vs quality in selection [0.0, 1.0].
    DiversityWeight,

    /// Generation strategy to use (categorical).
    Strategy,

    /// Random seed for reproducibility.
    Seed,

    /// Maximum generation attempts per sample.
    MaxAttempts,
}

impl ParamKey for SyntheticParam {
    fn name(&self) -> &'static str {
        match self {
            Self::AugmentationRatio => "augmentation_ratio",
            Self::QualityThreshold => "quality_threshold",
            Self::DiversityWeight => "diversity_weight",
            Self::Strategy => "strategy",
            Self::Seed => "seed",
            Self::MaxAttempts => "max_attempts",
        }
    }
}

impl fmt::Display for SyntheticParam {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl SyntheticParam {
    /// Get all synthetic parameters.
    #[must_use]
    pub fn all() -> &'static [SyntheticParam] {
        &[
            Self::AugmentationRatio,
            Self::QualityThreshold,
            Self::DiversityWeight,
            Self::Strategy,
            Self::Seed,
            Self::MaxAttempts,
        ]
    }

    /// Get a description of the parameter.
    #[must_use]
    pub fn description(&self) -> &'static str {
        match self {
            Self::AugmentationRatio => "Ratio of synthetic to original data",
            Self::QualityThreshold => "Minimum quality score for sample acceptance",
            Self::DiversityWeight => "Weight given to diversity vs quality",
            Self::Strategy => "Data generation strategy",
            Self::Seed => "Random seed for reproducibility",
            Self::MaxAttempts => "Maximum generation attempts per sample",
        }
    }

    /// Get the default value as a string representation.
    #[must_use]
    pub fn default_value(&self) -> &'static str {
        match self {
            Self::AugmentationRatio => "0.5",
            Self::QualityThreshold => "0.7",
            Self::DiversityWeight => "0.3",
            Self::Strategy => "eda",
            Self::Seed => "42",
            Self::MaxAttempts => "10",
        }
    }

    /// Check if this parameter is continuous (vs discrete/categorical).
    #[must_use]
    pub fn is_continuous(&self) -> bool {
        matches!(
            self,
            Self::AugmentationRatio | Self::QualityThreshold | Self::DiversityWeight
        )
    }

    /// Check if this parameter is categorical.
    #[must_use]
    pub fn is_categorical(&self) -> bool {
        matches!(self, Self::Strategy)
    }

    /// Get valid range as (min, max) for continuous parameters.
    ///
    /// Returns `None` for categorical parameters.
    #[must_use]
    pub fn range(&self) -> Option<(f32, f32)> {
        match self {
            Self::AugmentationRatio => Some((0.0, 10.0)),
            Self::QualityThreshold | Self::DiversityWeight => Some((0.0, 1.0)),
            Self::Seed => Some((0.0, f32::MAX)),
            Self::MaxAttempts => Some((1.0, 100.0)),
            Self::Strategy => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashMap, HashSet};

    #[test]
    fn test_param_names() {
        assert_eq!(
            SyntheticParam::AugmentationRatio.name(),
            "augmentation_ratio"
        );
        assert_eq!(SyntheticParam::QualityThreshold.name(), "quality_threshold");
        assert_eq!(SyntheticParam::DiversityWeight.name(), "diversity_weight");
        assert_eq!(SyntheticParam::Strategy.name(), "strategy");
        assert_eq!(SyntheticParam::Seed.name(), "seed");
        assert_eq!(SyntheticParam::MaxAttempts.name(), "max_attempts");
    }

    #[test]
    fn test_display() {
        assert_eq!(
            format!("{}", SyntheticParam::AugmentationRatio),
            "augmentation_ratio"
        );
        assert_eq!(format!("{}", SyntheticParam::Strategy), "strategy");
    }

    #[test]
    fn test_all_params() {
        let all = SyntheticParam::all();
        assert_eq!(all.len(), 6);

        let unique: HashSet<_> = all.iter().collect();
        assert_eq!(unique.len(), 6);
    }

    #[test]
    fn test_descriptions() {
        for param in SyntheticParam::all() {
            let desc = param.description();
            assert!(!desc.is_empty());
        }
    }

    #[test]
    fn test_default_values() {
        assert_eq!(SyntheticParam::AugmentationRatio.default_value(), "0.5");
        assert_eq!(SyntheticParam::QualityThreshold.default_value(), "0.7");
        assert_eq!(SyntheticParam::Strategy.default_value(), "eda");
    }

    #[test]
    fn test_is_continuous() {
        assert!(SyntheticParam::AugmentationRatio.is_continuous());
        assert!(SyntheticParam::QualityThreshold.is_continuous());
        assert!(SyntheticParam::DiversityWeight.is_continuous());

        assert!(!SyntheticParam::Strategy.is_continuous());
    }

    #[test]
    fn test_is_categorical() {
        assert!(SyntheticParam::Strategy.is_categorical());

        assert!(!SyntheticParam::AugmentationRatio.is_categorical());
        assert!(!SyntheticParam::QualityThreshold.is_categorical());
    }

    #[test]
    fn test_range() {
        assert_eq!(SyntheticParam::AugmentationRatio.range(), Some((0.0, 10.0)));
        // Both threshold and weight have same range [0, 1]
        assert_eq!(SyntheticParam::QualityThreshold.range(), Some((0.0, 1.0)));
        assert_eq!(SyntheticParam::DiversityWeight.range(), Some((0.0, 1.0)));
        assert_eq!(SyntheticParam::Strategy.range(), None);
    }

    #[test]
    fn test_equality() {
        let p1 = SyntheticParam::AugmentationRatio;
        let p2 = SyntheticParam::AugmentationRatio;
        let p3 = SyntheticParam::Strategy;

        assert_eq!(p1, p2);
        assert_ne!(p1, p3);
    }

    #[test]
    fn test_hash() {
        let mut set = HashSet::new();
        set.insert(SyntheticParam::AugmentationRatio);
        set.insert(SyntheticParam::Strategy);
        set.insert(SyntheticParam::AugmentationRatio); // duplicate

        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_hashmap_key() {
        let mut map = HashMap::new();
        map.insert(SyntheticParam::AugmentationRatio, 0.5_f32);
        map.insert(SyntheticParam::QualityThreshold, 0.7);

        assert_eq!(map.get(&SyntheticParam::AugmentationRatio), Some(&0.5));
        assert_eq!(map.get(&SyntheticParam::QualityThreshold), Some(&0.7));
    }

    #[test]
    fn test_clone_copy() {
        let p1 = SyntheticParam::Strategy;
        let p2 = p1; // Copy
        let p3 = p1; // Also copy (Clone is auto-derived for Copy types)

        assert_eq!(p1, p2);
        assert_eq!(p1, p3);
    }

    #[test]
    fn test_debug() {
        let debug = format!("{:?}", SyntheticParam::AugmentationRatio);
        assert!(debug.contains("AugmentationRatio"));
    }
}

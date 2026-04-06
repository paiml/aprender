//! Inferred schema representation

use std::collections::HashMap;

use super::stats::ColumnStats;
use super::types::FeatureType;

/// Inferred schema for a dataset
#[derive(Debug, Clone, Default)]
pub struct InferredSchema {
    /// Feature name -> inferred type
    pub features: HashMap<String, FeatureType>,
    /// Column statistics used for inference
    pub stats: HashMap<String, ColumnStats>,
}

impl InferredSchema {
    /// Get features of a specific type
    pub fn features_of_type(&self, feature_type: FeatureType) -> Vec<&str> {
        self.features
            .iter()
            .filter(|(_, &t)| t == feature_type)
            .map(|(name, _)| name.as_str())
            .collect()
    }

    /// Get target columns
    pub fn targets(&self) -> Vec<&str> {
        self.features
            .iter()
            .filter(|(_, &t)| {
                matches!(
                    t,
                    FeatureType::BinaryTarget
                        | FeatureType::MultiClassTarget
                        | FeatureType::RegressionTarget
                )
            })
            .map(|(name, _)| name.as_str())
            .collect()
    }

    /// Get input feature columns (non-targets)
    pub fn inputs(&self) -> Vec<&str> {
        self.features
            .iter()
            .filter(|(_, &t)| {
                !matches!(
                    t,
                    FeatureType::BinaryTarget
                        | FeatureType::MultiClassTarget
                        | FeatureType::RegressionTarget
                )
            })
            .map(|(name, _)| name.as_str())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_schema() -> InferredSchema {
        let mut schema = InferredSchema::default();
        schema.features.insert("age".to_string(), FeatureType::Numeric);
        schema.features.insert("income".to_string(), FeatureType::Numeric);
        schema.features.insert("category".to_string(), FeatureType::Categorical);
        schema.features.insert("text".to_string(), FeatureType::Text);
        schema.features.insert("is_spam".to_string(), FeatureType::BinaryTarget);
        schema.features.insert("label".to_string(), FeatureType::MultiClassTarget);
        schema.features.insert("price".to_string(), FeatureType::RegressionTarget);
        schema
    }

    #[test]
    fn test_inferred_schema_default() {
        let schema = InferredSchema::default();
        assert!(schema.features.is_empty());
        assert!(schema.stats.is_empty());
    }

    #[test]
    fn test_features_of_type_numeric() {
        let schema = make_schema();
        let numeric = schema.features_of_type(FeatureType::Numeric);
        assert_eq!(numeric.len(), 2);
        assert!(numeric.contains(&"age"));
        assert!(numeric.contains(&"income"));
    }

    #[test]
    fn test_features_of_type_categorical() {
        let schema = make_schema();
        let categorical = schema.features_of_type(FeatureType::Categorical);
        assert_eq!(categorical.len(), 1);
        assert!(categorical.contains(&"category"));
    }

    #[test]
    fn test_features_of_type_text() {
        let schema = make_schema();
        let text = schema.features_of_type(FeatureType::Text);
        assert_eq!(text.len(), 1);
        assert!(text.contains(&"text"));
    }

    #[test]
    fn test_features_of_type_empty() {
        let schema = make_schema();
        let embedding = schema.features_of_type(FeatureType::Embedding);
        assert!(embedding.is_empty());
    }

    #[test]
    fn test_targets() {
        let schema = make_schema();
        let targets = schema.targets();
        assert_eq!(targets.len(), 3);
        assert!(targets.contains(&"is_spam"));
        assert!(targets.contains(&"label"));
        assert!(targets.contains(&"price"));
    }

    #[test]
    fn test_targets_empty() {
        let schema = InferredSchema::default();
        let targets = schema.targets();
        assert!(targets.is_empty());
    }

    #[test]
    fn test_inputs() {
        let schema = make_schema();
        let inputs = schema.inputs();
        assert_eq!(inputs.len(), 4);
        assert!(inputs.contains(&"age"));
        assert!(inputs.contains(&"income"));
        assert!(inputs.contains(&"category"));
        assert!(inputs.contains(&"text"));
    }

    #[test]
    fn test_inputs_excludes_targets() {
        let schema = make_schema();
        let inputs = schema.inputs();
        assert!(!inputs.contains(&"is_spam"));
        assert!(!inputs.contains(&"label"));
        assert!(!inputs.contains(&"price"));
    }

    #[test]
    fn test_inputs_empty() {
        let schema = InferredSchema::default();
        let inputs = schema.inputs();
        assert!(inputs.is_empty());
    }

    #[test]
    fn test_inferred_schema_clone() {
        let schema = make_schema();
        let cloned = schema.clone();
        assert_eq!(schema.features.len(), cloned.features.len());
    }

    #[test]
    fn test_inferred_schema_debug() {
        let schema = make_schema();
        let debug_str = format!("{schema:?}");
        assert!(debug_str.contains("InferredSchema"));
        assert!(debug_str.contains("features"));
    }
}

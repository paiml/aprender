//! Model Lineage Tracking (ENT-046)
//!
//! Track model versions and training derivations.
//! Toyota Way 改善 (Kaizen): Track improvement over time.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Model identifier
    pub model_id: String,
    /// Semantic version
    pub version: String,
    /// Validation accuracy
    pub accuracy: f64,
    /// Creation timestamp
    pub created_at: u64,
    /// Configuration hash
    pub config_hash: String,
    /// Additional tags
    pub tags: HashMap<String, String>,
}

/// What changed between model versions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChangeType {
    /// More training data added
    AddData,
    /// Hyperparameters changed
    Hyperparams,
    /// Architecture modified
    Architecture,
    /// Different training run (same config)
    Retrain,
    /// Fine-tuning applied
    FineTune,
    /// Model merged
    Merge,
}

impl ChangeType {
    pub fn as_str(&self) -> &'static str {
        match self {
            ChangeType::AddData => "add_data",
            ChangeType::Hyperparams => "hyperparams",
            ChangeType::Architecture => "architecture",
            ChangeType::Retrain => "retrain",
            ChangeType::FineTune => "fine_tune",
            ChangeType::Merge => "merge",
        }
    }
}

/// Edge in the lineage graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Derivation {
    /// Parent model ID
    pub parent_id: String,
    /// Child model ID
    pub child_id: String,
    /// What changed
    pub change_type: ChangeType,
    /// Description of change
    pub description: String,
}

/// Model lineage tracker
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct ModelLineage {
    /// All models by ID
    models: HashMap<String, ModelMetadata>,
    /// Derivation edges
    derivations: Vec<Derivation>,
}

impl ModelLineage {
    /// Create a new lineage tracker
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a model to the lineage
    pub fn add_model(&mut self, metadata: ModelMetadata) -> String {
        let id = metadata.model_id.clone();
        self.models.insert(id.clone(), metadata);
        id
    }

    /// Add a derivation edge
    pub fn add_derivation(
        &mut self,
        parent_id: &str,
        child_id: &str,
        change_type: ChangeType,
        description: &str,
    ) {
        self.derivations.push(Derivation {
            parent_id: parent_id.to_string(),
            child_id: child_id.to_string(),
            change_type,
            description: description.to_string(),
        });
    }

    /// Get a model by ID
    pub fn get_model(&self, id: &str) -> Option<&ModelMetadata> {
        self.models.get(id)
    }

    /// Get all models
    pub fn all_models(&self) -> impl Iterator<Item = &ModelMetadata> {
        self.models.values()
    }

    /// Get parent of a model
    pub fn get_parent(&self, child_id: &str) -> Option<&ModelMetadata> {
        self.derivations
            .iter()
            .find(|d| d.child_id == child_id)
            .and_then(|d| self.models.get(&d.parent_id))
    }

    /// Get children of a model
    pub fn get_children(&self, parent_id: &str) -> Vec<&ModelMetadata> {
        self.derivations
            .iter()
            .filter(|d| d.parent_id == parent_id)
            .filter_map(|d| self.models.get(&d.child_id))
            .collect()
    }

    /// Compare two model versions
    pub fn compare(&self, a_id: &str, b_id: &str) -> Option<ModelComparison> {
        let a = self.models.get(a_id)?;
        let b = self.models.get(b_id)?;

        Some(ModelComparison {
            model_a: a_id.to_string(),
            model_b: b_id.to_string(),
            accuracy_delta: b.accuracy - a.accuracy,
            is_improvement: b.accuracy > a.accuracy,
        })
    }

    /// Find what caused a regression
    pub fn find_regression_source(&self, model_id: &str) -> Option<&Derivation> {
        let model = self.models.get(model_id)?;

        // Find parent
        let derivation = self.derivations.iter().find(|d| d.child_id == model_id)?;
        let parent = self.models.get(&derivation.parent_id)?;

        // Check if this is a regression
        if model.accuracy < parent.accuracy {
            Some(derivation)
        } else {
            None
        }
    }

    /// Get lineage chain from root to model
    pub fn get_lineage_chain(&self, model_id: &str) -> Vec<String> {
        let mut chain = vec![model_id.to_string()];
        let mut current = model_id;

        while let Some(derivation) = self.derivations.iter().find(|d| d.child_id == current) {
            chain.push(derivation.parent_id.clone());
            current = &derivation.parent_id;
        }

        chain.reverse();
        chain
    }

    /// Export to JSON
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Load from JSON
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

/// Comparison between two model versions
#[derive(Debug, Clone)]
pub struct ModelComparison {
    pub model_a: String,
    pub model_b: String,
    pub accuracy_delta: f64,
    pub is_improvement: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_model(id: &str, version: &str, accuracy: f64) -> ModelMetadata {
        ModelMetadata {
            model_id: id.to_string(),
            version: version.to_string(),
            accuracy,
            created_at: 0,
            config_hash: String::new(),
            tags: HashMap::new(),
        }
    }

    #[test]
    fn test_lineage_new() {
        let lineage = ModelLineage::new();
        assert_eq!(lineage.models.len(), 0);
    }

    #[test]
    fn test_add_model() {
        let mut lineage = ModelLineage::new();
        let id = lineage.add_model(make_model("v1", "1.0.0", 0.85));
        assert_eq!(id, "v1");
        assert!(lineage.get_model("v1").is_some());
    }

    #[test]
    fn test_add_derivation() {
        let mut lineage = ModelLineage::new();
        lineage.add_model(make_model("v1", "1.0.0", 0.85));
        lineage.add_model(make_model("v2", "2.0.0", 0.87));
        lineage.add_derivation("v1", "v2", ChangeType::AddData, "Added 1000 samples");

        assert_eq!(lineage.derivations.len(), 1);
    }

    #[test]
    fn test_get_parent() {
        let mut lineage = ModelLineage::new();
        lineage.add_model(make_model("v1", "1.0.0", 0.85));
        lineage.add_model(make_model("v2", "2.0.0", 0.87));
        lineage.add_derivation("v1", "v2", ChangeType::AddData, "More data");

        let parent = lineage.get_parent("v2").expect("operation should succeed");
        assert_eq!(parent.model_id, "v1");
    }

    #[test]
    fn test_get_children() {
        let mut lineage = ModelLineage::new();
        lineage.add_model(make_model("v1", "1.0.0", 0.85));
        lineage.add_model(make_model("v2a", "2.0.0", 0.87));
        lineage.add_model(make_model("v2b", "2.1.0", 0.86));
        lineage.add_derivation("v1", "v2a", ChangeType::AddData, "Branch A");
        lineage.add_derivation("v1", "v2b", ChangeType::Hyperparams, "Branch B");

        let children = lineage.get_children("v1");
        assert_eq!(children.len(), 2);
    }

    #[test]
    fn test_compare_improvement() {
        let mut lineage = ModelLineage::new();
        lineage.add_model(make_model("v1", "1.0.0", 0.85));
        lineage.add_model(make_model("v2", "2.0.0", 0.87));

        let cmp = lineage.compare("v1", "v2").expect("operation should succeed");
        assert!(cmp.is_improvement);
        assert!((cmp.accuracy_delta - 0.02).abs() < 1e-6);
    }

    #[test]
    fn test_compare_regression() {
        let mut lineage = ModelLineage::new();
        lineage.add_model(make_model("v1", "1.0.0", 0.87));
        lineage.add_model(make_model("v2", "2.0.0", 0.82));

        let cmp = lineage.compare("v1", "v2").expect("operation should succeed");
        assert!(!cmp.is_improvement);
    }

    #[test]
    fn test_find_regression_source() {
        let mut lineage = ModelLineage::new();
        lineage.add_model(make_model("v1", "1.0.0", 0.87));
        lineage.add_model(make_model("v2", "2.0.0", 0.82));
        lineage.add_derivation("v1", "v2", ChangeType::Hyperparams, "Changed LR");

        let source = lineage.find_regression_source("v2").expect("operation should succeed");
        assert_eq!(source.change_type, ChangeType::Hyperparams);
    }

    #[test]
    fn test_lineage_chain() {
        let mut lineage = ModelLineage::new();
        lineage.add_model(make_model("v1", "1.0.0", 0.80));
        lineage.add_model(make_model("v2", "2.0.0", 0.85));
        lineage.add_model(make_model("v3", "3.0.0", 0.87));
        lineage.add_derivation("v1", "v2", ChangeType::AddData, "");
        lineage.add_derivation("v2", "v3", ChangeType::FineTune, "");

        let chain = lineage.get_lineage_chain("v3");
        assert_eq!(chain, vec!["v1", "v2", "v3"]);
    }

    #[test]
    fn test_json_roundtrip() {
        let mut lineage = ModelLineage::new();
        lineage.add_model(make_model("v1", "1.0.0", 0.85));

        let json = lineage.to_json().expect("operation should succeed");
        let loaded = ModelLineage::from_json(&json).expect("load should succeed");
        assert!(loaded.get_model("v1").is_some());
    }

    // =========================================================================
    // Additional Coverage Tests
    // =========================================================================

    #[test]
    fn test_change_type_as_str() {
        assert_eq!(ChangeType::AddData.as_str(), "add_data");
        assert_eq!(ChangeType::Hyperparams.as_str(), "hyperparams");
        assert_eq!(ChangeType::Architecture.as_str(), "architecture");
        assert_eq!(ChangeType::Retrain.as_str(), "retrain");
        assert_eq!(ChangeType::FineTune.as_str(), "fine_tune");
        assert_eq!(ChangeType::Merge.as_str(), "merge");
    }

    #[test]
    fn test_all_models() {
        let mut lineage = ModelLineage::new();
        lineage.add_model(make_model("v1", "1.0.0", 0.80));
        lineage.add_model(make_model("v2", "2.0.0", 0.85));
        lineage.add_model(make_model("v3", "3.0.0", 0.90));

        let models: Vec<_> = lineage.all_models().collect();
        assert_eq!(models.len(), 3);
    }

    #[test]
    fn test_get_parent_no_parent() {
        let mut lineage = ModelLineage::new();
        lineage.add_model(make_model("v1", "1.0.0", 0.80));

        assert!(lineage.get_parent("v1").is_none());
    }

    #[test]
    fn test_find_regression_source_no_regression() {
        let mut lineage = ModelLineage::new();
        lineage.add_model(make_model("v1", "1.0.0", 0.80));
        lineage.add_model(make_model("v2", "2.0.0", 0.85)); // Improvement
        lineage.add_derivation("v1", "v2", ChangeType::AddData, "More data");

        // v2 is an improvement, so no regression
        assert!(lineage.find_regression_source("v2").is_none());
    }

    #[test]
    fn test_find_regression_source_nonexistent() {
        let lineage = ModelLineage::new();
        assert!(lineage.find_regression_source("v99").is_none());
    }

    #[test]
    fn test_compare_nonexistent_models() {
        let mut lineage = ModelLineage::new();
        lineage.add_model(make_model("v1", "1.0.0", 0.80));

        assert!(lineage.compare("v1", "v99").is_none());
        assert!(lineage.compare("v99", "v1").is_none());
    }

    #[test]
    fn test_get_children_no_children() {
        let mut lineage = ModelLineage::new();
        lineage.add_model(make_model("v1", "1.0.0", 0.80));

        let children = lineage.get_children("v1");
        assert!(children.is_empty());
    }

    #[test]
    fn test_get_model_nonexistent() {
        let lineage = ModelLineage::new();
        assert!(lineage.get_model("v99").is_none());
    }

    #[test]
    fn test_lineage_chain_single() {
        let mut lineage = ModelLineage::new();
        lineage.add_model(make_model("v1", "1.0.0", 0.80));

        let chain = lineage.get_lineage_chain("v1");
        assert_eq!(chain, vec!["v1"]);
    }

    #[test]
    fn test_model_metadata_with_tags() {
        let mut tags = HashMap::new();
        tags.insert("env".to_string(), "production".to_string());
        tags.insert("owner".to_string(), "team-ml".to_string());

        let model = ModelMetadata {
            model_id: "v1".to_string(),
            version: "1.0.0".to_string(),
            accuracy: 0.95,
            created_at: 1700000000,
            config_hash: "abc123".to_string(),
            tags,
        };

        assert_eq!(model.tags.len(), 2);
        assert_eq!(model.created_at, 1700000000);
    }

    #[test]
    fn test_derivation_clone() {
        let d = Derivation {
            parent_id: "v1".to_string(),
            child_id: "v2".to_string(),
            change_type: ChangeType::Merge,
            description: "merged models".to_string(),
        };
        let cloned = d.clone();
        assert_eq!(d.parent_id, cloned.parent_id);
        assert_eq!(d.change_type, cloned.change_type);
    }

    #[test]
    fn test_model_comparison_clone() {
        let cmp = ModelComparison {
            model_a: "v1".to_string(),
            model_b: "v2".to_string(),
            accuracy_delta: 0.05,
            is_improvement: true,
        };
        let cloned = cmp.clone();
        assert_eq!(cmp.accuracy_delta, cloned.accuracy_delta);
    }

    #[test]
    fn test_model_lineage_default() {
        let lineage = ModelLineage::default();
        assert!(lineage.models.is_empty());
        assert!(lineage.derivations.is_empty());
    }
}

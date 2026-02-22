//! Forest Path - Decision path for ensemble models (Random Forest, Gradient Boosting)

use serde::{Deserialize, Serialize};

use super::traits::{ByteReader, DecisionPath, PathError};
use super::tree::TreePath;

/// Decision path for ensemble models (Random Forest, Gradient Boosting)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ForestPath {
    /// Individual tree paths
    pub tree_paths: Vec<TreePath>,
    /// Per-tree predictions
    pub tree_predictions: Vec<f32>,
    /// Aggregated prediction
    pub ensemble_prediction: f32,
    /// Agreement ratio among trees (0.0 - 1.0)
    pub tree_agreement: f32,
    /// Feature importance (averaged across trees)
    pub feature_importance: Vec<f32>,
}

impl ForestPath {
    /// Create a new forest path
    pub fn new(tree_paths: Vec<TreePath>, tree_predictions: Vec<f32>) -> Self {
        let ensemble_prediction = if tree_predictions.is_empty() {
            0.0
        } else {
            tree_predictions.iter().sum::<f32>() / tree_predictions.len() as f32
        };

        let tree_agreement = Self::compute_agreement(&tree_predictions);
        let feature_importance = Vec::new();

        Self {
            tree_paths,
            tree_predictions,
            ensemble_prediction,
            tree_agreement,
            feature_importance,
        }
    }

    /// Compute agreement ratio among trees
    fn compute_agreement(predictions: &[f32]) -> f32 {
        if predictions.len() < 2 {
            return 1.0;
        }

        let mean = predictions.iter().sum::<f32>() / predictions.len().max(1) as f32;
        let variance = predictions.iter().map(|p| (p - mean).powi(2)).sum::<f32>()
            / predictions.len().max(1) as f32;

        // Convert variance to agreement (higher variance = lower agreement)
        1.0 / (1.0 + variance)
    }

    /// Set feature importance
    pub fn with_feature_importance(mut self, importance: Vec<f32>) -> Self {
        self.feature_importance = importance;
        self
    }

    /// Number of trees in the ensemble
    pub fn n_trees(&self) -> usize {
        self.tree_paths.len()
    }
}

impl DecisionPath for ForestPath {
    fn explain(&self) -> String {
        let mut explanation = format!(
            "Ensemble Prediction: {:.4} (n_trees={}, agreement={:.1}%)\n",
            self.ensemble_prediction,
            self.n_trees(),
            self.tree_agreement * 100.0
        );

        explanation.push_str("\nTree predictions:\n");
        for (i, pred) in self.tree_predictions.iter().enumerate() {
            explanation.push_str(&format!("  Tree {i}: {pred:.4}\n"));
        }

        if !self.feature_importance.is_empty() {
            explanation.push_str("\nTop features by importance:\n");
            let mut indexed: Vec<(usize, f32)> = self
                .feature_importance
                .iter()
                .enumerate()
                .map(|(i, &imp)| (i, imp))
                .collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            for (idx, imp) in indexed.iter().take(5) {
                explanation.push_str(&format!("  feature[{idx}]: {imp:.4}\n"));
            }
        }

        explanation
    }

    fn feature_contributions(&self) -> &[f32] {
        &self.feature_importance
    }

    fn confidence(&self) -> f32 {
        self.tree_agreement
    }

    fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.push(1); // version

        // Number of trees
        bytes.extend_from_slice(&(self.tree_paths.len() as u32).to_le_bytes());

        // Each tree path
        for tree_path in &self.tree_paths {
            let tree_bytes = tree_path.to_bytes();
            bytes.extend_from_slice(&(tree_bytes.len() as u32).to_le_bytes());
            bytes.extend_from_slice(&tree_bytes);
        }

        // Tree predictions
        bytes.extend_from_slice(&(self.tree_predictions.len() as u32).to_le_bytes());
        for pred in &self.tree_predictions {
            bytes.extend_from_slice(&pred.to_le_bytes());
        }

        // Ensemble prediction
        bytes.extend_from_slice(&self.ensemble_prediction.to_le_bytes());
        bytes.extend_from_slice(&self.tree_agreement.to_le_bytes());

        // Feature importance
        bytes.extend_from_slice(&(self.feature_importance.len() as u32).to_le_bytes());
        for imp in &self.feature_importance {
            bytes.extend_from_slice(&imp.to_le_bytes());
        }

        bytes
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self, PathError> {
        if bytes.len() < 5 {
            return Err(PathError::InsufficientData {
                expected: 5,
                actual: bytes.len(),
            });
        }

        let mut reader = ByteReader::new(bytes);

        let version = reader.read_u8()?;
        if version != 1 {
            return Err(PathError::VersionMismatch {
                expected: 1,
                actual: version,
            });
        }

        // Tree paths (length-prefixed sub-messages)
        let n_trees = reader.read_u32()? as usize;
        let mut tree_paths = Vec::with_capacity(n_trees);
        for _ in 0..n_trees {
            let tree_path = reader.read_sub_message(TreePath::from_bytes)?;
            tree_paths.push(tree_path);
        }

        let tree_predictions = reader.read_f32_vec()?;
        let ensemble_prediction = reader.read_f32()?;
        let tree_agreement = reader.read_f32()?;
        let feature_importance = reader.read_f32_vec()?;

        Ok(Self {
            tree_paths,
            tree_predictions,
            ensemble_prediction,
            tree_agreement,
            feature_importance,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::tree::{LeafInfo, TreeSplit};
    use super::*;

    fn mock_tree_path() -> TreePath {
        let split = TreeSplit {
            feature_idx: 0,
            threshold: 0.5,
            went_left: true,
            n_samples: 100,
        };
        let leaf = LeafInfo {
            prediction: 0.75,
            n_samples: 50,
            class_distribution: None,
        };
        TreePath::new(vec![split], leaf)
    }

    #[test]
    fn test_forest_path_new_empty() {
        let forest = ForestPath::new(vec![], vec![]);
        assert_eq!(forest.n_trees(), 0);
        assert_eq!(forest.ensemble_prediction, 0.0);
        assert_eq!(forest.tree_agreement, 1.0);
        assert!(forest.feature_importance.is_empty());
    }

    #[test]
    fn test_forest_path_new_single_tree() {
        let tree = mock_tree_path();
        let forest = ForestPath::new(vec![tree], vec![0.5]);
        assert_eq!(forest.n_trees(), 1);
        assert_eq!(forest.ensemble_prediction, 0.5);
        assert_eq!(forest.tree_agreement, 1.0);
    }

    #[test]
    fn test_forest_path_new_multiple_trees() {
        let trees = vec![mock_tree_path(), mock_tree_path(), mock_tree_path()];
        let predictions = vec![0.3, 0.5, 0.7];
        let forest = ForestPath::new(trees, predictions);

        assert_eq!(forest.n_trees(), 3);
        assert!((forest.ensemble_prediction - 0.5).abs() < 1e-6);
        assert!(forest.tree_agreement < 1.0);
        assert!(forest.tree_agreement > 0.0);
    }

    #[test]
    fn test_forest_path_with_feature_importance() {
        let forest = ForestPath::new(vec![mock_tree_path()], vec![0.5])
            .with_feature_importance(vec![0.1, 0.3, 0.6]);

        assert_eq!(forest.feature_importance, vec![0.1, 0.3, 0.6]);
    }

    #[test]
    fn test_forest_path_compute_agreement_identical() {
        let predictions = vec![0.5, 0.5, 0.5];
        let agreement = ForestPath::compute_agreement(&predictions);
        assert_eq!(agreement, 1.0);
    }

    #[test]
    fn test_forest_path_compute_agreement_varied() {
        let predictions = vec![0.0, 1.0];
        let agreement = ForestPath::compute_agreement(&predictions);
        assert!((agreement - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_forest_path_explain() {
        let forest = ForestPath::new(vec![mock_tree_path()], vec![0.75])
            .with_feature_importance(vec![0.1, 0.5, 0.4]);

        let explanation = forest.explain();
        assert!(explanation.contains("Ensemble Prediction: 0.7500"));
        assert!(explanation.contains("n_trees=1"));
        assert!(explanation.contains("agreement=100.0%"));
        assert!(explanation.contains("Tree 0: 0.7500"));
        assert!(explanation.contains("Top features by importance"));
    }

    #[test]
    fn test_forest_path_feature_contributions() {
        let forest = ForestPath::new(vec![], vec![]).with_feature_importance(vec![0.2, 0.8]);

        assert_eq!(forest.feature_contributions(), &[0.2, 0.8]);
    }

    #[test]
    fn test_forest_path_confidence() {
        let forest = ForestPath::new(vec![mock_tree_path(), mock_tree_path()], vec![0.5, 0.5]);
        assert_eq!(forest.confidence(), 1.0);
    }

    #[test]
    fn test_forest_path_serialization_roundtrip() {
        let trees = vec![mock_tree_path(), mock_tree_path()];
        let forest =
            ForestPath::new(trees, vec![0.3, 0.7]).with_feature_importance(vec![0.25, 0.75]);

        let bytes = forest.to_bytes();
        let recovered = ForestPath::from_bytes(&bytes).expect("roundtrip");

        assert_eq!(recovered.n_trees(), forest.n_trees());
        assert!((recovered.ensemble_prediction - forest.ensemble_prediction).abs() < 1e-6);
        assert!((recovered.tree_agreement - forest.tree_agreement).abs() < 1e-6);
        assert_eq!(
            recovered.tree_predictions.len(),
            forest.tree_predictions.len()
        );
        assert_eq!(
            recovered.feature_importance.len(),
            forest.feature_importance.len()
        );
    }

    #[test]
    fn test_forest_path_from_bytes_insufficient_data() {
        let result = ForestPath::from_bytes(&[1, 2]);
        assert!(matches!(result, Err(PathError::InsufficientData { .. })));
    }

    #[test]
    fn test_forest_path_from_bytes_version_mismatch() {
        let bytes = vec![2, 0, 0, 0, 0];
        let result = ForestPath::from_bytes(&bytes);
        assert!(matches!(
            result,
            Err(PathError::VersionMismatch {
                expected: 1,
                actual: 2
            })
        ));
    }

    #[test]
    fn test_forest_path_clone() {
        let forest = ForestPath::new(vec![mock_tree_path()], vec![0.5]);
        let cloned = forest.clone();

        assert_eq!(cloned.n_trees(), forest.n_trees());
        assert_eq!(cloned.ensemble_prediction, forest.ensemble_prediction);
    }

    #[test]
    fn test_forest_path_debug() {
        let forest = ForestPath::new(vec![], vec![]);
        let debug_str = format!("{:?}", forest);
        assert!(debug_str.contains("ForestPath"));
    }
}

//! Tree-based decision path types
//!
//! Decision path for tree-based models (Decision Tree, CART).

use serde::{Deserialize, Serialize};

use super::traits::{ByteReader, DecisionPath, PathError};

/// A single split decision in a tree
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TreeSplit {
    /// Feature index used for split
    pub feature_idx: usize,
    /// Threshold value
    pub threshold: f32,
    /// Direction taken (true = left, false = right)
    pub went_left: bool,
    /// Samples in node before split (from training)
    pub n_samples: usize,
}

/// Information about the leaf node reached
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LeafInfo {
    /// Predicted class or value
    pub prediction: f32,
    /// Samples in training that reached this leaf
    pub n_samples: usize,
    /// Class distribution (for classification)
    pub class_distribution: Option<Vec<f32>>,
}

/// Decision path for tree-based models
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TreePath {
    /// Sequence of splits taken
    pub splits: Vec<TreeSplit>,
    /// Leaf node statistics
    pub leaf: LeafInfo,
    /// Gini impurity at each node (optional)
    pub gini_path: Vec<f32>,
    /// Feature contributions (computed from mean decrease)
    pub(crate) contributions: Vec<f32>,
}

impl TreePath {
    /// Create a new tree path
    pub fn new(splits: Vec<TreeSplit>, leaf: LeafInfo) -> Self {
        let gini_path = Vec::new();
        let contributions = Vec::new();
        Self {
            splits,
            leaf,
            gini_path,
            contributions,
        }
    }

    /// Set Gini impurity path
    pub fn with_gini(mut self, gini_path: Vec<f32>) -> Self {
        self.gini_path = gini_path;
        self
    }

    /// Set feature contributions
    pub fn with_contributions(mut self, contributions: Vec<f32>) -> Self {
        self.contributions = contributions;
        self
    }

    /// Get depth of the tree path
    pub fn depth(&self) -> usize {
        self.splits.len()
    }
}

impl DecisionPath for TreePath {
    fn explain(&self) -> String {
        let depth = self.depth();
        let mut explanation = format!("Decision Path (depth={depth}):\n");

        for (i, split) in self.splits.iter().enumerate() {
            let direction = if split.went_left { "YES" } else { "NO" };
            let comparison = if split.went_left { "<=" } else { ">" };
            let feature_idx = split.feature_idx;
            let threshold = split.threshold;
            let n_samples = split.n_samples;
            explanation.push_str(&format!(
                "  Node {i}: feature[{feature_idx}] {comparison} {threshold:.4}? {direction} (n={n_samples})\n"
            ));
        }

        let prediction = self.leaf.prediction;
        let n_samples = self.leaf.n_samples;
        explanation.push_str(&format!(
            "  LEAF -> prediction={prediction:.4}, n_samples={n_samples}\n"
        ));

        if let Some(dist) = &self.leaf.class_distribution {
            explanation.push_str("         class_distribution: [");
            explanation.push_str(
                &dist
                    .iter()
                    .map(|p| format!("{p:.2}"))
                    .collect::<Vec<_>>()
                    .join(", "),
            );
            explanation.push_str("]\n");
        }

        explanation
    }

    fn feature_contributions(&self) -> &[f32] {
        &self.contributions
    }

    fn confidence(&self) -> f32 {
        if let Some(dist) = &self.leaf.class_distribution {
            // Max probability as confidence
            dist.iter().copied().fold(0.0f32, f32::max)
        } else {
            // For regression, use sample size as proxy
            1.0 - 1.0 / (self.leaf.n_samples as f32 + 1.0)
        }
    }

    fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.push(1); // version

        // Number of splits
        let n_splits = self.splits.len() as u32;
        bytes.extend_from_slice(&n_splits.to_le_bytes());

        // Each split
        for split in &self.splits {
            bytes.extend_from_slice(&(split.feature_idx as u32).to_le_bytes());
            bytes.extend_from_slice(&split.threshold.to_le_bytes());
            bytes.push(u8::from(split.went_left));
            bytes.extend_from_slice(&(split.n_samples as u32).to_le_bytes());
        }

        // Leaf info
        bytes.extend_from_slice(&self.leaf.prediction.to_le_bytes());
        bytes.extend_from_slice(&(self.leaf.n_samples as u32).to_le_bytes());

        // Class distribution
        let has_dist = self.leaf.class_distribution.is_some();
        bytes.push(u8::from(has_dist));
        if let Some(dist) = &self.leaf.class_distribution {
            bytes.extend_from_slice(&(dist.len() as u32).to_le_bytes());
            for p in dist {
                bytes.extend_from_slice(&p.to_le_bytes());
            }
        }

        // Gini path
        bytes.extend_from_slice(&(self.gini_path.len() as u32).to_le_bytes());
        for g in &self.gini_path {
            bytes.extend_from_slice(&g.to_le_bytes());
        }

        // Contributions
        bytes.extend_from_slice(&(self.contributions.len() as u32).to_le_bytes());
        for c in &self.contributions {
            bytes.extend_from_slice(&c.to_le_bytes());
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

        // Splits
        let n_splits = reader.read_u32()? as usize;
        let mut splits = Vec::with_capacity(n_splits);
        for _ in 0..n_splits {
            let feature_idx = reader.read_u32()? as usize;
            let threshold = reader.read_f32()?;
            let went_left = reader.read_bool()?;
            let n_samples = reader.read_u32()? as usize;
            splits.push(TreeSplit {
                feature_idx,
                threshold,
                went_left,
                n_samples,
            });
        }

        // Leaf info
        let prediction = reader.read_f32()?;
        let n_samples = reader.read_u32()? as usize;
        let class_distribution = reader.read_optional(ByteReader::read_f32_vec)?;
        let leaf = LeafInfo {
            prediction,
            n_samples,
            class_distribution,
        };

        let gini_path = reader.read_f32_vec()?;
        let contributions = reader.read_f32_vec()?;

        Ok(Self {
            splits,
            leaf,
            gini_path,
            contributions,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tree_path_new() {
        let splits = vec![
            TreeSplit {
                feature_idx: 0,
                threshold: 35.0,
                went_left: true,
                n_samples: 1000,
            },
            TreeSplit {
                feature_idx: 1,
                threshold: 50000.0,
                went_left: false,
                n_samples: 600,
            },
        ];
        let leaf = LeafInfo {
            prediction: 1.0,
            n_samples: 250,
            class_distribution: Some(vec![0.08, 0.92]),
        };

        let path = TreePath::new(splits, leaf);
        assert_eq!(path.depth(), 2);
    }

    #[test]
    fn test_tree_path_explain() {
        let splits = vec![TreeSplit {
            feature_idx: 0,
            threshold: 35.0,
            went_left: true,
            n_samples: 1000,
        }];
        let leaf = LeafInfo {
            prediction: 1.0,
            n_samples: 250,
            class_distribution: Some(vec![0.1, 0.9]),
        };

        let path = TreePath::new(splits, leaf);
        let explanation = path.explain();
        assert!(explanation.contains("Decision Path (depth=1)"));
        assert!(explanation.contains("feature[0]"));
        assert!(explanation.contains("LEAF"));
    }

    #[test]
    fn test_tree_path_serialization_roundtrip() {
        let splits = vec![
            TreeSplit {
                feature_idx: 0,
                threshold: 35.0,
                went_left: true,
                n_samples: 1000,
            },
            TreeSplit {
                feature_idx: 1,
                threshold: 50000.0,
                went_left: false,
                n_samples: 600,
            },
        ];
        let leaf = LeafInfo {
            prediction: 0.92,
            n_samples: 250,
            class_distribution: Some(vec![0.08, 0.92]),
        };

        let path = TreePath::new(splits, leaf)
            .with_gini(vec![0.5, 0.3, 0.1])
            .with_contributions(vec![0.2, 0.5, 0.3]);

        let bytes = path.to_bytes();
        let restored = TreePath::from_bytes(&bytes).expect("Failed to deserialize");

        assert_eq!(path.splits.len(), restored.splits.len());
        assert_eq!(path.leaf.n_samples, restored.leaf.n_samples);
        assert!((path.leaf.prediction - restored.leaf.prediction).abs() < 1e-6);
        assert_eq!(path.gini_path.len(), restored.gini_path.len());
        assert_eq!(path.contributions.len(), restored.contributions.len());
    }

    #[test]
    fn test_tree_path_confidence() {
        let leaf = LeafInfo {
            prediction: 1.0,
            n_samples: 100,
            class_distribution: Some(vec![0.1, 0.9]),
        };
        let path = TreePath::new(vec![], leaf);
        assert!((path.confidence() - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_tree_path_insufficient_data_at_start() {
        let result = TreePath::from_bytes(&[1u8, 0, 0]);
        assert!(matches!(result, Err(PathError::InsufficientData { .. })));
    }

    #[test]
    fn test_tree_path_invalid_version() {
        let result = TreePath::from_bytes(&[2u8, 0, 0, 0, 0]);
        assert!(matches!(result, Err(PathError::VersionMismatch { .. })));
    }

    #[test]
    fn test_tree_path_insufficient_data_in_splits() {
        let mut bytes = vec![1u8];
        bytes.extend_from_slice(&1u32.to_le_bytes()); // n_splits = 1
        let result = TreePath::from_bytes(&bytes);
        assert!(matches!(result, Err(PathError::InsufficientData { .. })));
    }

    #[test]
    fn test_tree_path_confidence_without_distribution() {
        let leaf = LeafInfo {
            prediction: 0.5,
            n_samples: 100,
            class_distribution: None,
        };
        let path = TreePath::new(vec![], leaf);
        let confidence = path.confidence();
        assert!(confidence > 0.98);
        assert!(confidence < 1.0);
    }

    #[test]
    fn test_tree_path_explain_went_right() {
        let splits = vec![TreeSplit {
            feature_idx: 0,
            threshold: 35.0,
            went_left: false,
            n_samples: 100,
        }];
        let leaf = LeafInfo {
            prediction: 0.5,
            n_samples: 50,
            class_distribution: None,
        };
        let path = TreePath::new(splits, leaf);
        let explanation = path.explain();
        assert!(explanation.contains("NO"));
        assert!(explanation.contains(">"));
    }

    #[test]
    fn test_tree_path_serialization_without_class_distribution() {
        let leaf = LeafInfo {
            prediction: 0.5,
            n_samples: 100,
            class_distribution: None,
        };
        let path = TreePath::new(vec![], leaf);
        let bytes = path.to_bytes();
        let restored = TreePath::from_bytes(&bytes).expect("Failed to deserialize");
        assert!(restored.leaf.class_distribution.is_none());
    }

    #[test]
    fn test_tree_split_clone() {
        let split = TreeSplit {
            feature_idx: 5,
            threshold: 2.5,
            went_left: true,
            n_samples: 500,
        };
        let cloned = split.clone();
        assert_eq!(split.feature_idx, cloned.feature_idx);
        assert_eq!(split.threshold, cloned.threshold);
        assert_eq!(split.went_left, cloned.went_left);
        assert_eq!(split.n_samples, cloned.n_samples);
    }

    #[test]
    fn test_leaf_info_clone() {
        let leaf = LeafInfo {
            prediction: 0.75,
            n_samples: 200,
            class_distribution: Some(vec![0.25, 0.75]),
        };
        let cloned = leaf.clone();
        assert_eq!(leaf.prediction, cloned.prediction);
        assert_eq!(leaf.n_samples, cloned.n_samples);
        assert_eq!(leaf.class_distribution, cloned.class_distribution);
    }

    #[test]
    fn test_tree_path_with_gini() {
        let leaf = LeafInfo {
            prediction: 0.5,
            n_samples: 100,
            class_distribution: None,
        };
        let path = TreePath::new(vec![], leaf).with_gini(vec![0.5, 0.3, 0.1]);
        assert_eq!(path.gini_path, vec![0.5, 0.3, 0.1]);
    }

    #[test]
    fn test_tree_path_with_contributions() {
        let leaf = LeafInfo {
            prediction: 0.5,
            n_samples: 100,
            class_distribution: None,
        };
        let path = TreePath::new(vec![], leaf).with_contributions(vec![0.2, 0.5, 0.3]);
        assert_eq!(path.feature_contributions(), &[0.2, 0.5, 0.3]);
    }

    #[test]
    fn test_tree_path_empty_splits() {
        let leaf = LeafInfo {
            prediction: 0.5,
            n_samples: 100,
            class_distribution: None,
        };
        let path = TreePath::new(vec![], leaf);
        assert_eq!(path.depth(), 0);
        let explanation = path.explain();
        assert!(explanation.contains("Decision Path (depth=0)"));
        assert!(explanation.contains("LEAF"));
    }

    #[test]
    fn test_tree_path_feature_contributions_empty() {
        let leaf = LeafInfo {
            prediction: 0.5,
            n_samples: 100,
            class_distribution: None,
        };
        let path = TreePath::new(vec![], leaf);
        assert!(path.feature_contributions().is_empty());
    }
}

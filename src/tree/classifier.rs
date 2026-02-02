//! Decision Tree Classifier implementation.
//!
//! Uses the CART algorithm with Gini impurity for splitting.

use super::helpers::{build_tree, flatten_tree_node, reconstruct_tree_node};
use super::TreeNode;
use crate::error::Result;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// Decision tree classifier using the CART algorithm.
///
/// Uses Gini impurity for splitting criterion and builds trees recursively.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionTreeClassifier {
    pub(super) tree: Option<TreeNode>,
    pub(super) max_depth: Option<usize>,
    /// Number of features the model was trained on (for validation)
    #[serde(default)]
    pub(super) n_features: Option<usize>,
}

impl DecisionTreeClassifier {
    /// Creates a new decision tree classifier with default parameters.
    #[must_use]
    pub fn new() -> Self {
        Self {
            tree: None,
            max_depth: None,
            n_features: None,
        }
    }

    /// Sets the maximum depth of the tree.
    ///
    /// # Arguments
    ///
    /// * `depth` - Maximum depth (root has depth 0)
    #[must_use]
    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = Some(depth);
        self
    }

    /// Fits the decision tree to training data.
    ///
    /// # Arguments
    ///
    /// * `x` - Training features (`n_samples` × `n_features`)
    /// * `y` - Training labels (`n_samples` class indices)
    ///
    /// # Errors
    ///
    /// Returns an error if the data is invalid.
    pub fn fit(&mut self, x: &crate::primitives::Matrix<f32>, y: &[usize]) -> Result<()> {
        let (n_rows, n_cols) = x.shape();
        if n_rows != y.len() {
            return Err("Number of samples in X and y must match".into());
        }
        if n_rows == 0 {
            return Err("Cannot fit with zero samples".into());
        }

        self.n_features = Some(n_cols);
        self.tree = Some(build_tree(x, y, 0, self.max_depth));
        Ok(())
    }

    /// Predicts class labels for samples.
    ///
    /// # Arguments
    ///
    /// * `x` - Feature matrix (`n_samples` × `n_features`)
    ///
    /// # Returns
    ///
    /// Vector of predicted class labels
    ///
    /// # Panics
    ///
    /// Panics if called before `fit()` or if feature count doesn't match training data
    #[must_use]
    pub fn predict(&self, x: &crate::primitives::Matrix<f32>) -> Vec<usize> {
        let (n_samples, n_features) = x.shape();

        // Validate feature count matches what we trained on
        if let Some(expected) = self.n_features {
            assert!(
                n_features >= expected,
                "Feature count mismatch: model was trained with {expected} features but input has {n_features} features"
            );
        }

        let mut predictions = Vec::with_capacity(n_samples);

        for row in 0..n_samples {
            let mut sample = Vec::with_capacity(n_features);
            for col in 0..n_features {
                sample.push(x.get(row, col));
            }
            predictions.push(self.predict_one(&sample));
        }

        predictions
    }

    /// Predicts the class label for a single sample.
    fn predict_one(&self, x: &[f32]) -> usize {
        let tree = self.tree.as_ref().expect("Model not fitted yet");

        let mut node = tree;
        loop {
            match node {
                TreeNode::Leaf(leaf) => return leaf.class_label,
                TreeNode::Node(internal) => {
                    if x[internal.feature_idx] <= internal.threshold {
                        node = &internal.left;
                    } else {
                        node = &internal.right;
                    }
                }
            }
        }
    }

    /// Computes the accuracy score on test data.
    #[must_use]
    pub fn score(&self, x: &crate::primitives::Matrix<f32>, y: &[usize]) -> f32 {
        let predictions = self.predict(x);
        let correct = predictions
            .iter()
            .zip(y.iter())
            .filter(|(pred, true_label)| pred == true_label)
            .count();
        correct as f32 / y.len() as f32
    }

    /// Saves the model to a binary file using bincode.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization or file writing fails.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> std::result::Result<(), String> {
        let bytes = bincode::serialize(self).map_err(|e| format!("Serialization failed: {e}"))?;
        fs::write(path, bytes).map_err(|e| format!("File write failed: {e}"))?;
        Ok(())
    }

    /// Loads a model from a binary file.
    ///
    /// # Errors
    ///
    /// Returns an error if file reading or deserialization fails.
    pub fn load<P: AsRef<Path>>(path: P) -> std::result::Result<Self, String> {
        let bytes = fs::read(path).map_err(|e| format!("File read failed: {e}"))?;
        let model =
            bincode::deserialize(&bytes).map_err(|e| format!("Deserialization failed: {e}"))?;
        Ok(model)
    }

    /// Saves the model to `SafeTensors` format.
    ///
    /// # Errors
    ///
    /// Returns an error if model is not fitted, serialization fails, or file writing fails.
    pub fn save_safetensors<P: AsRef<Path>>(&self, path: P) -> std::result::Result<(), String> {
        use crate::serialization::safetensors;
        use std::collections::BTreeMap;

        // Verify model is fitted
        let tree = self
            .tree
            .as_ref()
            .ok_or("Cannot save unfitted model. Call fit() first.")?;

        // Flatten tree structure into arrays via pre-order traversal
        let mut node_features = Vec::new();
        let mut node_thresholds = Vec::new();
        let mut node_classes = Vec::new();
        let mut node_samples = Vec::new();
        let mut node_left_child = Vec::new();
        let mut node_right_child = Vec::new();

        flatten_tree_node(
            tree,
            &mut node_features,
            &mut node_thresholds,
            &mut node_classes,
            &mut node_samples,
            &mut node_left_child,
            &mut node_right_child,
        );

        // Prepare tensors
        let mut tensors = BTreeMap::new();

        tensors.insert(
            "node_features".to_string(),
            (node_features.clone(), vec![node_features.len()]),
        );
        tensors.insert(
            "node_thresholds".to_string(),
            (node_thresholds.clone(), vec![node_thresholds.len()]),
        );
        tensors.insert(
            "node_classes".to_string(),
            (node_classes.clone(), vec![node_classes.len()]),
        );
        tensors.insert(
            "node_samples".to_string(),
            (node_samples.clone(), vec![node_samples.len()]),
        );
        tensors.insert(
            "node_left_child".to_string(),
            (node_left_child.clone(), vec![node_left_child.len()]),
        );
        tensors.insert(
            "node_right_child".to_string(),
            (node_right_child.clone(), vec![node_right_child.len()]),
        );

        // Max depth as tensor (-1 for None)
        let max_depth_val = self.max_depth.map_or(-1.0, |d| d as f32);
        tensors.insert("max_depth".to_string(), (vec![max_depth_val], vec![1]));

        // Save to SafeTensors format
        safetensors::save_safetensors(path, &tensors)?;
        Ok(())
    }

    /// Loads a model from `SafeTensors` format.
    ///
    /// # Errors
    ///
    /// Returns an error if file reading fails or format is invalid.
    pub fn load_safetensors<P: AsRef<Path>>(path: P) -> std::result::Result<Self, String> {
        use crate::serialization::safetensors;

        // Load SafeTensors file
        let (metadata, raw_data) = safetensors::load_safetensors(path)?;

        // Extract all tensors
        let node_features = safetensors::extract_tensor(
            &raw_data,
            metadata
                .get("node_features")
                .ok_or("Missing 'node_features' tensor")?,
        )?;
        let node_thresholds = safetensors::extract_tensor(
            &raw_data,
            metadata
                .get("node_thresholds")
                .ok_or("Missing 'node_thresholds' tensor")?,
        )?;
        let node_classes = safetensors::extract_tensor(
            &raw_data,
            metadata
                .get("node_classes")
                .ok_or("Missing 'node_classes' tensor")?,
        )?;
        let node_samples = safetensors::extract_tensor(
            &raw_data,
            metadata
                .get("node_samples")
                .ok_or("Missing 'node_samples' tensor")?,
        )?;
        let node_left_child = safetensors::extract_tensor(
            &raw_data,
            metadata
                .get("node_left_child")
                .ok_or("Missing 'node_left_child' tensor")?,
        )?;
        let node_right_child = safetensors::extract_tensor(
            &raw_data,
            metadata
                .get("node_right_child")
                .ok_or("Missing 'node_right_child' tensor")?,
        )?;
        let max_depth_data = safetensors::extract_tensor(
            &raw_data,
            metadata
                .get("max_depth")
                .ok_or("Missing 'max_depth' tensor")?,
        )?;

        // Validate tensor sizes match
        let n_nodes = node_features.len();
        if node_thresholds.len() != n_nodes
            || node_classes.len() != n_nodes
            || node_samples.len() != n_nodes
            || node_left_child.len() != n_nodes
            || node_right_child.len() != n_nodes
        {
            return Err("Inconsistent node array sizes in SafeTensors file".to_string());
        }

        if n_nodes == 0 {
            return Err("Empty tree in SafeTensors file".to_string());
        }

        // Reconstruct tree from flat arrays
        let tree = Some(reconstruct_tree_node(
            0,
            &node_features,
            &node_thresholds,
            &node_classes,
            &node_samples,
            &node_left_child,
            &node_right_child,
        ));

        // Parse max_depth
        let max_depth = if max_depth_data[0] < 0.0 {
            None
        } else {
            Some(max_depth_data[0] as usize)
        };

        Ok(Self {
            tree,
            max_depth,
            n_features: None,
        })
    }
}

impl Default for DecisionTreeClassifier {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::Matrix;

    fn simple_dataset() -> (Matrix<f32>, Vec<usize>) {
        // 2-class dataset: features [x1, x2], labels {0, 1}
        let x = Matrix::from_vec(
            6,
            2,
            vec![
                1.0, 2.0, // class 0
                1.5, 1.8, // class 0
                2.0, 2.5, // class 0
                5.0, 8.0, // class 1
                6.0, 9.0, // class 1
                7.0, 7.5, // class 1
            ],
        )
        .expect("valid matrix");
        let y = vec![0, 0, 0, 1, 1, 1];
        (x, y)
    }

    #[test]
    fn test_default_impl() {
        let model = DecisionTreeClassifier::default();
        assert!(model.tree.is_none());
        assert!(model.max_depth.is_none());
        assert!(model.n_features.is_none());
    }

    #[test]
    fn test_with_max_depth() {
        let model = DecisionTreeClassifier::new().with_max_depth(3);
        assert_eq!(model.max_depth, Some(3));
    }

    #[test]
    fn test_fit_mismatched_x_y() {
        let x = Matrix::from_vec(3, 2, vec![1.0; 6]).expect("valid matrix");
        let y = vec![0, 1]; // 2 labels for 3 samples
        let mut model = DecisionTreeClassifier::new();
        let err = model.fit(&x, &y);
        assert!(err.is_err());
    }

    #[test]
    fn test_fit_zero_samples() {
        let x = Matrix::from_vec(0, 2, vec![]).expect("valid matrix");
        let y: Vec<usize> = vec![];
        let mut model = DecisionTreeClassifier::new();
        let err = model.fit(&x, &y);
        assert!(err.is_err());
    }

    #[test]
    fn test_fit_and_predict() {
        let (x, y) = simple_dataset();
        let mut model = DecisionTreeClassifier::new();
        model.fit(&x, &y).expect("fit should succeed");
        let predictions = model.predict(&x);
        assert_eq!(predictions.len(), 6);
        // Should get most predictions correct on training data
        let correct = predictions
            .iter()
            .zip(y.iter())
            .filter(|(p, t)| p == t)
            .count();
        assert!(correct >= 4, "Expected at least 4 correct, got {correct}");
    }

    #[test]
    fn test_score() {
        let (x, y) = simple_dataset();
        let mut model = DecisionTreeClassifier::new();
        model.fit(&x, &y).expect("fit should succeed");
        let accuracy = model.score(&x, &y);
        assert!(accuracy >= 0.5, "Expected accuracy >= 0.5, got {accuracy}");
    }

    #[test]
    fn test_save_load_roundtrip() {
        let (x, y) = simple_dataset();
        let mut model = DecisionTreeClassifier::new().with_max_depth(5);
        model.fit(&x, &y).expect("fit should succeed");

        let dir = std::env::temp_dir().join("aprender_test_tree_classifier");
        std::fs::create_dir_all(&dir).ok();
        let path = dir.join("tree_model.bin");

        model.save(&path).expect("save should succeed");
        let loaded = DecisionTreeClassifier::load(&path).expect("load should succeed");
        let orig_preds = model.predict(&x);
        let loaded_preds = loaded.predict(&x);
        assert_eq!(orig_preds, loaded_preds);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_load_nonexistent_file() {
        let result = DecisionTreeClassifier::load("/tmp/aprender_nonexistent_tree_model.bin");
        assert!(result.is_err());
    }

    #[test]
    fn test_save_safetensors_unfitted() {
        let model = DecisionTreeClassifier::new();
        let result = model.save_safetensors("/tmp/aprender_unfitted.safetensors");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("unfitted"));
    }

    #[test]
    fn test_save_load_safetensors_roundtrip() {
        let (x, y) = simple_dataset();
        let mut model = DecisionTreeClassifier::new().with_max_depth(3);
        model.fit(&x, &y).expect("fit should succeed");

        let dir = std::env::temp_dir().join("aprender_test_tree_safetensors");
        std::fs::create_dir_all(&dir).ok();
        let path = dir.join("tree_model.safetensors");

        model.save_safetensors(&path).expect("save should succeed");
        let loaded = DecisionTreeClassifier::load_safetensors(&path).expect("load should succeed");

        let orig_preds = model.predict(&x);
        let loaded_preds = loaded.predict(&x);
        assert_eq!(orig_preds, loaded_preds);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_save_safetensors_no_max_depth() {
        let (x, y) = simple_dataset();
        let mut model = DecisionTreeClassifier::new(); // no max_depth
        model.fit(&x, &y).expect("fit should succeed");

        let dir = std::env::temp_dir().join("aprender_test_tree_no_depth");
        std::fs::create_dir_all(&dir).ok();
        let path = dir.join("tree_model_nodepth.safetensors");

        model.save_safetensors(&path).expect("save should succeed");
        let loaded = DecisionTreeClassifier::load_safetensors(&path).expect("load should succeed");
        assert!(loaded.max_depth.is_none());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_load_safetensors_nonexistent() {
        let result =
            DecisionTreeClassifier::load_safetensors("/tmp/aprender_nonexistent.safetensors");
        assert!(result.is_err());
    }

    #[test]
    fn test_debug_clone() {
        let model = DecisionTreeClassifier::new().with_max_depth(2);
        let debug_str = format!("{:?}", model);
        assert!(debug_str.contains("DecisionTreeClassifier"));

        let cloned = model.clone();
        assert_eq!(cloned.max_depth, Some(2));
    }
}

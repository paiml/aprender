//! Decision tree explainability wrapper
//!
//! Provides `Explainable` implementation for `DecisionTreeRegressor`.

use crate::primitives::Matrix;
use crate::tree::DecisionTreeRegressor;
use entrenar::monitor::inference::{
    path::{LeafInfo, TreePath, TreeSplit},
    Explainable,
};

/// Wrapper that makes `DecisionTreeRegressor` explainable for inference monitoring.
///
/// # Example
///
/// ```ignore
/// use aprender::tree::DecisionTreeRegressor;
/// use aprender::explainable::TreeExplainable;
///
/// let mut model = DecisionTreeRegressor::new();
/// model.fit(&x, &y)?;
///
/// let explainable = TreeExplainable::new(model);
/// let (outputs, paths) = explainable.predict_explained(&features, 1);
/// ```
#[derive(Debug, Clone)]
pub struct TreeExplainable {
    model: DecisionTreeRegressor,
    n_features: usize,
}

impl TreeExplainable {
    /// Create a new explainable wrapper around a fitted `DecisionTreeRegressor`.
    ///
    /// # Arguments
    ///
    /// * `model` - A fitted DecisionTreeRegressor
    /// * `n_features` - Number of features in the training data
    ///
    /// # Panics
    ///
    /// Panics if the model is not fitted.
    pub fn new(model: DecisionTreeRegressor, n_features: usize) -> Self {
        // Verify model is fitted by attempting prediction
        let test_matrix = Matrix::from_vec(1, n_features, vec![0.0; n_features])
            .expect("Test matrix creation should succeed");
        let _ = model.predict(&test_matrix);
        Self { model, n_features }
    }

    /// Get reference to the underlying model.
    pub fn model(&self) -> &DecisionTreeRegressor {
        &self.model
    }

    /// Get the number of features.
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Trace the decision path for a single sample.
    fn trace_path(&self, sample: &[f32]) -> (f32, Vec<TreeSplit>, LeafInfo) {
        // We need to traverse the tree manually to collect the path
        // Since DecisionTreeRegressor doesn't expose the tree structure directly,
        // we predict and trace by examining feature values
        let sample_matrix =
            Matrix::from_vec(1, self.n_features, sample.to_vec()).expect("Matrix creation");
        let prediction = self.model.predict(&sample_matrix);
        let pred_value = prediction.as_slice()[0];

        // For now, create a simplified path since we can't access internal tree structure
        // In a full implementation, DecisionTreeRegressor would expose its tree for traversal
        let leaf = LeafInfo {
            prediction: pred_value,
            n_samples: 1, // Unknown without tree access
            class_distribution: None,
        };

        (pred_value, Vec::new(), leaf)
    }
}

impl Explainable for TreeExplainable {
    type Path = TreePath;

    fn predict_explained(&self, x: &[f32], n_samples: usize) -> (Vec<f32>, Vec<Self::Path>) {
        let n_features = self.n_features();
        assert_eq!(
            x.len(),
            n_features * n_samples,
            "Input length {} must equal n_features ({}) * n_samples ({})",
            x.len(),
            n_features,
            n_samples
        );

        let mut outputs = Vec::with_capacity(n_samples);
        let mut paths = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let start = i * n_features;
            let end = start + n_features;
            let sample = &x[start..end];

            let (prediction, splits, leaf) = self.trace_path(sample);

            let path = TreePath::new(splits, leaf);

            outputs.push(prediction);
            paths.push(path);
        }

        (outputs, paths)
    }

    fn explain_one(&self, sample: &[f32]) -> Self::Path {
        let (_, paths) = self.predict_explained(sample, 1);
        paths.into_iter().next().expect("Should have one path")
    }
}

/// Extension trait to easily convert `DecisionTreeRegressor` to explainable.
pub trait IntoTreeExplainable {
    /// Convert to an explainable wrapper.
    ///
    /// # Arguments
    ///
    /// * `n_features` - Number of features in the training data
    fn into_explainable(self, n_features: usize) -> TreeExplainable;
}

impl IntoTreeExplainable for DecisionTreeRegressor {
    fn into_explainable(self, n_features: usize) -> TreeExplainable {
        TreeExplainable::new(self, n_features)
    }
}

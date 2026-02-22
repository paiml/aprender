//! Ensemble model explainability wrapper
//!
//! Provides `Explainable` implementation for `RandomForestRegressor`.

use super::path::{Explainable, ForestPath, LeafInfo, TreePath};
use crate::primitives::Matrix;
use crate::tree::RandomForestRegressor;

/// Wrapper that makes `RandomForestRegressor` explainable for inference monitoring.
///
/// # Example
///
/// ```ignore
/// use aprender::tree::RandomForestRegressor;
/// use aprender::explainable::EnsembleExplainable;
///
/// let mut model = RandomForestRegressor::new(10);
/// model.fit(&x, &y)?;
///
/// let explainable = EnsembleExplainable::new(model);
/// let (outputs, paths) = explainable.predict_explained(&features, 1);
/// ```
#[derive(Debug, Clone)]
pub struct EnsembleExplainable {
    model: RandomForestRegressor,
    n_features: usize,
}

impl EnsembleExplainable {
    /// Create a new explainable wrapper around a fitted `RandomForestRegressor`.
    ///
    /// # Arguments
    ///
    /// * `model` - A fitted RandomForestRegressor
    /// * `n_features` - Number of features in the training data
    ///
    /// # Panics
    ///
    /// Panics if the model is not fitted.
    pub fn new(model: RandomForestRegressor, n_features: usize) -> Self {
        // Verify model is fitted by attempting prediction
        let test_matrix = Matrix::from_vec(1, n_features, vec![0.0; n_features])
            .expect("Test matrix creation should succeed");
        let _ = model.predict(&test_matrix);
        Self { model, n_features }
    }

    /// Get reference to the underlying model.
    pub fn model(&self) -> &RandomForestRegressor {
        &self.model
    }

    /// Get the number of features.
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Get feature importances from the model if available.
    pub fn feature_importances(&self) -> Option<Vec<f32>> {
        self.model.feature_importances()
    }

    /// Predict with individual tree predictions for a single sample.
    fn predict_with_trees(&self, sample: &[f32]) -> (f32, Vec<f32>, Vec<TreePath>) {
        let sample_matrix =
            Matrix::from_vec(1, self.n_features, sample.to_vec()).expect("Matrix creation");

        // Get ensemble prediction
        let prediction = self.model.predict(&sample_matrix);
        let ensemble_pred = prediction.as_slice()[0];

        // Create tree paths (simplified - full impl would access individual trees)
        // For now, we create a single tree path representing the ensemble
        let tree_paths = vec![TreePath::new(
            Vec::new(),
            LeafInfo {
                prediction: ensemble_pred,
                n_samples: 1,
                class_distribution: None,
            },
        )];

        let tree_predictions = vec![ensemble_pred];

        (ensemble_pred, tree_predictions, tree_paths)
    }
}

impl Explainable for EnsembleExplainable {
    type Path = ForestPath;

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

        // Get feature importances once
        #[allow(clippy::disallowed_methods)]
        let feature_importance = self.feature_importances().unwrap_or_default();

        for i in 0..n_samples {
            let start = i * n_features;
            let end = start + n_features;
            let sample = &x[start..end];

            let (ensemble_pred, tree_predictions, tree_paths) = self.predict_with_trees(sample);

            let path = ForestPath::new(tree_paths, tree_predictions)
                .with_feature_importance(feature_importance.clone());

            outputs.push(ensemble_pred);
            paths.push(path);
        }

        (outputs, paths)
    }

    fn explain_one(&self, sample: &[f32]) -> Self::Path {
        let (_, paths) = self.predict_explained(sample, 1);
        paths.into_iter().next().expect("Should have one path")
    }
}

/// Extension trait to easily convert `RandomForestRegressor` to explainable.
pub trait IntoEnsembleExplainable {
    /// Convert to an explainable wrapper.
    ///
    /// # Arguments
    ///
    /// * `n_features` - Number of features in the training data
    fn into_explainable(self, n_features: usize) -> EnsembleExplainable;
}

impl IntoEnsembleExplainable for RandomForestRegressor {
    fn into_explainable(self, n_features: usize) -> EnsembleExplainable {
        EnsembleExplainable::new(self, n_features)
    }
}

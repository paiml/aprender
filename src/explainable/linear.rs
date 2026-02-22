//! Linear model explainability wrapper
//!
//! Provides `Explainable` implementation for `LinearRegression`.

use super::path::{Explainable, LinearPath};
use crate::linear_model::LinearRegression;

/// Wrapper that makes `LinearRegression` explainable for inference monitoring.
///
/// # Example
///
/// ```ignore
/// use aprender::linear_model::LinearRegression;
/// use aprender::explainable::LinearExplainable;
///
/// let mut model = LinearRegression::new();
/// model.fit(&x, &y)?;
///
/// let explainable = LinearExplainable::new(model);
/// let (outputs, paths) = explainable.predict_explained(&features, 1);
/// ```
#[derive(Debug, Clone)]
pub struct LinearExplainable {
    model: LinearRegression,
}

impl LinearExplainable {
    /// Create a new explainable wrapper around a fitted `LinearRegression`.
    ///
    /// # Panics
    ///
    /// Panics if the model is not fitted.
    pub fn new(model: LinearRegression) -> Self {
        // Verify model is fitted by accessing coefficients
        let _ = model.coefficients();
        Self { model }
    }

    /// Get reference to the underlying model.
    pub fn model(&self) -> &LinearRegression {
        &self.model
    }

    /// Get the number of features.
    pub fn n_features(&self) -> usize {
        self.model.coefficients().len()
    }

    /// Compute feature contributions for a single sample.
    fn compute_contributions(&self, sample: &[f32]) -> Vec<f32> {
        let coefficients = self.model.coefficients();
        coefficients
            .as_slice()
            .iter()
            .zip(sample)
            .map(|(&w, &x)| w * x)
            .collect()
    }
}

impl Explainable for LinearExplainable {
    type Path = LinearPath;

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

        let intercept = self.model.intercept();
        let mut outputs = Vec::with_capacity(n_samples);
        let mut paths = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let start = i * n_features;
            let end = start + n_features;
            let sample = &x[start..end];

            let contributions = self.compute_contributions(sample);
            let logit: f32 = contributions.iter().sum::<f32>() + intercept;
            let output = logit; // Linear regression output is the logit itself

            let path = LinearPath::new(contributions, intercept, logit, output);

            outputs.push(output);
            paths.push(path);
        }

        (outputs, paths)
    }

    fn explain_one(&self, sample: &[f32]) -> Self::Path {
        let (_, paths) = self.predict_explained(sample, 1);
        paths.into_iter().next().expect("Should have one path")
    }
}

/// Extension trait to easily convert `LinearRegression` to explainable.
pub trait IntoExplainable {
    /// Convert to an explainable wrapper.
    fn into_explainable(self) -> LinearExplainable;
}

impl IntoExplainable for LinearRegression {
    fn into_explainable(self) -> LinearExplainable {
        LinearExplainable::new(self)
    }
}

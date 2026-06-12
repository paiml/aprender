//! `Pipeline` — chain transformers then a final estimator (Pillar 1 — beat
//! scikit-learn). Mirrors `sklearn.pipeline.Pipeline`: on `fit`, each transformer
//! is fit then applied in sequence and the final estimator is fit on the fully
//! transformed data; on `predict`/`score` the same transformer chain is applied
//! (transform-only) before delegating to the estimator.
//!
//! Uses trait objects (`Box<dyn Transformer>` / `Box<dyn Estimator>`) so steps
//! can be heterogeneous (e.g. `StandardScaler` then `LogisticRegression`).

use crate::error::Result;
use crate::primitives::{Matrix, Vector};
use crate::traits::{Estimator, Transformer};

/// A sequence of transformers followed by a final estimator.
pub struct Pipeline {
    steps: Vec<Box<dyn Transformer>>,
    estimator: Box<dyn Estimator>,
}

impl Pipeline {
    /// Build a pipeline from transformer `steps` (applied in order) and a final
    /// `estimator`.
    #[must_use]
    pub fn new(steps: Vec<Box<dyn Transformer>>, estimator: Box<dyn Estimator>) -> Self {
        Self { steps, estimator }
    }

    /// Fit each transformer in sequence (fit-then-transform), then fit the final
    /// estimator on the fully transformed features.
    ///
    /// # Errors
    /// Returns an error if any transformer or the estimator fails to fit.
    pub fn fit(&mut self, x: &Matrix<f32>, y: &Vector<f32>) -> Result<()> {
        let mut xt = x.clone();
        for step in &mut self.steps {
            step.fit(&xt)?;
            xt = step.transform(&xt)?;
        }
        self.estimator.fit(&xt, y)
    }

    /// Apply the fitted transformer chain (transform-only), then predict.
    ///
    /// # Errors
    /// Returns an error if any transformer fails.
    pub fn predict(&self, x: &Matrix<f32>) -> Result<Vector<f32>> {
        let xt = self.transform_chain(x)?;
        Ok(self.estimator.predict(&xt))
    }

    /// Apply the fitted transformer chain, then score the estimator
    /// (accuracy for classifiers, R² for regressors).
    ///
    /// # Errors
    /// Returns an error if any transformer fails.
    pub fn score(&self, x: &Matrix<f32>, y: &Vector<f32>) -> Result<f32> {
        let xt = self.transform_chain(x)?;
        Ok(self.estimator.score(&xt, y))
    }

    fn transform_chain(&self, x: &Matrix<f32>) -> Result<Matrix<f32>> {
        let mut xt = x.clone();
        for step in &self.steps {
            xt = step.transform(&xt)?;
        }
        Ok(xt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::classification::LogisticRegression;
    use crate::datasets::make_classification;
    use crate::preprocessing::StandardScaler;

    /// FT-PIPELINE: a StandardScaler -> LogisticRegression pipeline fits end to
    /// end and learns (mirrors sklearn's make_pipeline(StandardScaler(), LogReg)).
    #[test]
    fn pipeline_scaler_then_classifier_learns() {
        let (x, labels) = make_classification(150, 5, 4, 2, 9);
        let y = Vector::from_vec(labels.iter().map(|&l| l as f32).collect());

        let mut pipe = Pipeline::new(
            vec![Box::new(StandardScaler::new())],
            Box::new(LogisticRegression::new().with_max_iter(300)),
        );
        pipe.fit(&x, &y).expect("pipeline fit");

        let preds = pipe.predict(&x).expect("pipeline predict");
        assert_eq!(preds.len(), 150);
        let score = pipe.score(&x, &y).expect("pipeline score");
        assert!(
            score > 0.7,
            "pipeline train accuracy {score} too low to be learning"
        );
    }
}

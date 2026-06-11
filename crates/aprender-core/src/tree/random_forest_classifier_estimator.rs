// `Estimator` impl for `RandomForestClassifier` (Pillar 1 — beat scikit-learn).
//
// Classifiers natively take `y: &[usize]`, but the generic `Estimator` trait
// (and therefore `cross_validate` / `grid_search`) speaks `Vector<f32>`. This
// adapter bridges the two — labels round-trip through `f32` — so classifiers
// work with the generic model-selection machinery, just like in sklearn where
// any estimator drops into `cross_val_score`. Additive: the inherent
// `fit(&[usize])` / `predict() -> Vec<usize>` API is unchanged.

impl crate::traits::Estimator for RandomForestClassifier {
    fn fit(
        &mut self,
        x: &crate::primitives::Matrix<f32>,
        y: &crate::primitives::Vector<f32>,
    ) -> crate::Result<()> {
        let labels: Vec<usize> = y.as_slice().iter().map(|&v| v.round() as usize).collect();
        RandomForestClassifier::fit(self, x, &labels)
    }

    fn predict(&self, x: &crate::primitives::Matrix<f32>) -> crate::primitives::Vector<f32> {
        let labels: Vec<usize> = RandomForestClassifier::predict(self, x);
        crate::primitives::Vector::from_vec(labels.into_iter().map(|l| l as f32).collect())
    }

    fn score(&self, x: &crate::primitives::Matrix<f32>, y: &crate::primitives::Vector<f32>) -> f32 {
        let preds: Vec<usize> = RandomForestClassifier::predict(self, x);
        let n = y.len();
        if n == 0 {
            return 0.0;
        }
        let correct = preds
            .iter()
            .zip(y.as_slice())
            .filter(|(&p, &t)| p == t.round() as usize)
            .count();
        correct as f32 / n as f32
    }
}

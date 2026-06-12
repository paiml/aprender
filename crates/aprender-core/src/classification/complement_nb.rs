//! `ComplementNB` — Complement Naive Bayes (Pillar 1 — beat scikit-learn).
//! Mirrors `sklearn.naive_bayes.ComplementNB` (`norm=False`): a MultinomialNB
//! adaptation that estimates each class's parameters from the *complement* of
//! that class, which is more stable on imbalanced data.
//!
//! `weight[c][j] = −log((comp_count[c][j] + alpha) / (comp_total[c] + alpha·n_features))`
//! where the complement counts sum features over all samples NOT in class `c`;
//! prediction is `argmax_c Σ_j x_j · weight[c][j]`.

use crate::error::Result;
use crate::primitives::Matrix;

/// Complement Naive Bayes classifier (count features; robust to imbalance).
#[derive(Debug, Clone)]
pub struct ComplementNB {
    alpha: f32,
    feature_log_prob: Vec<Vec<f32>>,
    n_features: usize,
}

impl Default for ComplementNB {
    fn default() -> Self {
        Self::new()
    }
}

impl ComplementNB {
    /// Create a new `ComplementNB` with `alpha = 1.0`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            alpha: 1.0,
            feature_log_prob: Vec::new(),
            n_features: 0,
        }
    }

    /// Set the additive smoothing parameter.
    #[must_use]
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }

    /// Fit on count features `x` and integer labels `y` (in `0..n_classes`).
    ///
    /// # Errors
    /// Returns an error if `x`/`y` lengths disagree or there are no samples.
    pub fn fit(&mut self, x: &Matrix<f32>, y: &[usize]) -> Result<()> {
        let (n_samples, n_features) = x.shape();
        if n_samples == 0 {
            return Err("ComplementNB: cannot fit with zero samples".into());
        }
        if y.len() != n_samples {
            return Err("ComplementNB: x/y length mismatch".into());
        }
        let n_classes = y.iter().max().map_or(0, |&m| m + 1);
        // total feature mass and per-class feature mass
        let mut feature_all = vec![0.0f64; n_features];
        let mut feature_count = vec![vec![0.0f64; n_features]; n_classes];
        for (i, &c) in y.iter().enumerate() {
            for j in 0..n_features {
                let v = f64::from(x.get(i, j));
                feature_all[j] += v;
                feature_count[c][j] += v;
            }
        }
        let alpha = f64::from(self.alpha);
        self.feature_log_prob = (0..n_classes)
            .map(|c| {
                // complement = all samples NOT in class c
                let comp: Vec<f64> = (0..n_features)
                    .map(|j| feature_all[j] - feature_count[c][j])
                    .collect();
                let comp_total: f64 = comp.iter().sum::<f64>() + alpha * n_features as f64;
                comp.iter()
                    .map(|&cc| -(((cc + alpha) / comp_total).ln()) as f32)
                    .collect()
            })
            .collect();
        self.n_features = n_features;
        Ok(())
    }

    /// Predict class labels as `argmax_c Σ_j x_j · weight[c][j]`.
    #[must_use]
    pub fn predict(&self, x: &Matrix<f32>) -> Vec<usize> {
        let (n_samples, _) = x.shape();
        (0..n_samples)
            .map(|i| {
                let mut best_c = 0;
                let mut best = f32::NEG_INFINITY;
                for (c, w) in self.feature_log_prob.iter().enumerate() {
                    let mut s = 0.0f32;
                    for j in 0..self.n_features {
                        s += x.get(i, j) * w[j];
                    }
                    if s > best {
                        best = s;
                        best_c = c;
                    }
                }
                best_c
            })
            .collect()
    }
}

impl crate::traits::Estimator for ComplementNB {
    fn fit(&mut self, x: &Matrix<f32>, y: &crate::primitives::Vector<f32>) -> Result<()> {
        let labels: Vec<usize> = y.as_slice().iter().map(|&v| v.round() as usize).collect();
        ComplementNB::fit(self, x, &labels)
    }
    fn predict(&self, x: &Matrix<f32>) -> crate::primitives::Vector<f32> {
        let labels = ComplementNB::predict(self, x);
        crate::primitives::Vector::from_vec(labels.into_iter().map(|l| l as f32).collect())
    }
    fn score(&self, x: &Matrix<f32>, y: &crate::primitives::Vector<f32>) -> f32 {
        let preds = ComplementNB::predict(self, x);
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

#[cfg(test)]
mod tests {
    use super::*;

    /// FT-COMPLEMENTNB: matches sklearn.naive_bayes.ComplementNB on a count fixture.
    #[test]
    fn complement_nb_matches_sklearn() {
        let x = Matrix::from_vec(
            4,
            4,
            vec![
                2.0, 1.0, 0.0, 3.0, 1.0, 1.0, 0.0, 2.0, 0.0, 0.0, 3.0, 1.0, 0.0, 1.0, 2.0, 1.0,
            ],
        )
        .expect("valid");
        let y = [0usize, 0, 1, 1];
        let mut nb = ComplementNB::new();
        nb.fit(&x, &y).expect("fit");
        // weights match sklearn feature_log_prob_ (within 1e-3)
        let expect0 = [2.48491, 1.79176, 0.69315, 1.38629];
        for (j, e) in expect0.iter().enumerate() {
            assert!((nb.feature_log_prob[0][j] - e).abs() < 1e-3, "w0[{j}]");
        }
        assert_eq!(nb.predict(&x), vec![0, 0, 1, 1]);
        let xt =
            Matrix::from_vec(2, 4, vec![3.0, 2.0, 0.0, 4.0, 0.0, 0.0, 4.0, 2.0]).expect("valid");
        assert_eq!(nb.predict(&xt), vec![0, 1]);
    }
}

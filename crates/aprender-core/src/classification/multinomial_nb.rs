//! `MultinomialNB` — Multinomial Naive Bayes for count features (Pillar 1 — beat
//! scikit-learn). Mirrors `sklearn.naive_bayes.MultinomialNB`: the standard
//! classifier for discrete counts (e.g. bag-of-words text features).
//!
//! Training learns `log P(c)` (class log-priors) and `log P(j|c)` with Laplace/
//! Lidstone `alpha` smoothing:
//! `P(j|c) = (count_{c,j} + alpha) / (sum_j count_{c,j} + alpha·n_features)`.
//! Prediction is `argmax_c [ log P(c) + Σ_j x_j · log P(j|c) ]`.

use crate::error::Result;
use crate::primitives::Matrix;

/// Multinomial Naive Bayes classifier (integer/count features).
#[derive(Debug, Clone)]
pub struct MultinomialNB {
    alpha: f32,
    class_log_prior: Vec<f32>,
    feature_log_prob: Vec<Vec<f32>>,
    n_features: usize,
}

impl Default for MultinomialNB {
    fn default() -> Self {
        Self::new()
    }
}

impl MultinomialNB {
    /// Create a new `MultinomialNB` with `alpha = 1.0` (Laplace smoothing).
    #[must_use]
    pub fn new() -> Self {
        Self {
            alpha: 1.0,
            class_log_prior: Vec::new(),
            feature_log_prob: Vec::new(),
            n_features: 0,
        }
    }

    /// Set the additive (Lidstone/Laplace) smoothing parameter.
    #[must_use]
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }

    /// Fit on count features `x` and integer labels `y` (in `0..n_classes`).
    ///
    /// # Errors
    /// Returns an error if `x` and `y` lengths disagree or there are no samples.
    pub fn fit(&mut self, x: &Matrix<f32>, y: &[usize]) -> Result<()> {
        let (n_samples, n_features) = x.shape();
        if n_samples == 0 {
            return Err("MultinomialNB: cannot fit with zero samples".into());
        }
        if y.len() != n_samples {
            return Err("MultinomialNB: x/y length mismatch".into());
        }
        let n_classes = y.iter().max().map_or(0, |&m| m + 1);
        let mut class_count = vec![0usize; n_classes];
        let mut feature_count = vec![vec![0.0f64; n_features]; n_classes];
        for (i, &c) in y.iter().enumerate() {
            class_count[c] += 1;
            for j in 0..n_features {
                feature_count[c][j] += f64::from(x.get(i, j));
            }
        }
        let alpha = f64::from(self.alpha);
        self.class_log_prior = (0..n_classes)
            .map(|c| (class_count[c] as f64 / n_samples as f64).ln() as f32)
            .collect();
        self.feature_log_prob = (0..n_classes)
            .map(|c| {
                let total = feature_count[c].iter().sum::<f64>() + alpha * n_features as f64;
                (0..n_features)
                    .map(|j| ((feature_count[c][j] + alpha) / total).ln() as f32)
                    .collect()
            })
            .collect();
        self.n_features = n_features;
        Ok(())
    }

    /// Predict class labels by maximizing the joint log-likelihood.
    #[must_use]
    pub fn predict(&self, x: &Matrix<f32>) -> Vec<usize> {
        let (n_samples, _) = x.shape();
        (0..n_samples)
            .map(|i| {
                let mut best_c = 0;
                let mut best_ll = f32::NEG_INFINITY;
                for (c, prior) in self.class_log_prior.iter().enumerate() {
                    let mut ll = *prior;
                    for j in 0..self.n_features {
                        ll += x.get(i, j) * self.feature_log_prob[c][j];
                    }
                    if ll > best_ll {
                        best_ll = ll;
                        best_c = c;
                    }
                }
                best_c
            })
            .collect()
    }
}

// Estimator impl so MultinomialNB works with generic cross_validate / grid_search.
impl crate::traits::Estimator for MultinomialNB {
    fn fit(&mut self, x: &Matrix<f32>, y: &crate::primitives::Vector<f32>) -> Result<()> {
        let labels: Vec<usize> = y.as_slice().iter().map(|&v| v.round() as usize).collect();
        MultinomialNB::fit(self, x, &labels)
    }
    fn predict(&self, x: &Matrix<f32>) -> crate::primitives::Vector<f32> {
        let labels = MultinomialNB::predict(self, x);
        crate::primitives::Vector::from_vec(labels.into_iter().map(|l| l as f32).collect())
    }
    fn score(&self, x: &Matrix<f32>, y: &crate::primitives::Vector<f32>) -> f32 {
        let preds = MultinomialNB::predict(self, x);
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

    /// FT-MULTINOMIALNB: matches sklearn.naive_bayes.MultinomialNB on a count fixture.
    #[test]
    fn multinomial_nb_matches_sklearn() {
        let x = Matrix::from_vec(
            4,
            4,
            vec![
                2.0, 1.0, 0.0, 3.0, 1.0, 1.0, 0.0, 2.0, 0.0, 0.0, 3.0, 1.0, 0.0, 1.0, 2.0, 1.0,
            ],
        )
        .expect("valid");
        let y = [0usize, 0, 1, 1];
        let mut nb = MultinomialNB::new();
        nb.fit(&x, &y).expect("fit");
        // perfect train separation (sklearn oracle)
        assert_eq!(nb.predict(&x), vec![0, 0, 1, 1]);
        // balanced class log-priors = ln(0.5)
        for lp in &nb.class_log_prior {
            assert!((lp - (0.5f32).ln()).abs() < 1e-5);
        }
        // test predictions (sklearn oracle: [0, 1])
        let xt =
            Matrix::from_vec(2, 4, vec![3.0, 2.0, 0.0, 4.0, 0.0, 0.0, 4.0, 2.0]).expect("valid");
        assert_eq!(nb.predict(&xt), vec![0, 1]);
    }
}

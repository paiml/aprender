//! `BernoulliNB` — Bernoulli Naive Bayes for binary features (Pillar 1 — beat
//! scikit-learn). Mirrors `sklearn.naive_bayes.BernoulliNB`: features are
//! binarized (`x > binarize`), and unlike MultinomialNB it explicitly models
//! feature *absence* via the `(1 - x)·log(1 - P(j|c))` term.
//!
//! `P(j=1|c) = (present_{c,j} + alpha) / (count_c + 2·alpha)`;
//! prediction is `argmax_c [ logP(c) + Σ_j b_j·logP(j|c) + (1-b_j)·log(1-P(j|c)) ]`.

use crate::error::Result;
use crate::primitives::Matrix;

/// Bernoulli Naive Bayes classifier (binary/binarized features).
#[derive(Debug, Clone)]
pub struct BernoulliNB {
    alpha: f32,
    binarize: f32,
    class_log_prior: Vec<f32>,
    /// Precomputed `ln(P(j=1|c))` per (class, feature) — hoisted out of the predict hot loop.
    feature_log_prob: Vec<Vec<f32>>,
    /// Precomputed `ln(1 - P(j=1|c))` per (class, feature) — the feature-absence term.
    feature_log_neg_prob: Vec<Vec<f32>>,
    n_features: usize,
}

impl Default for BernoulliNB {
    fn default() -> Self {
        Self::new()
    }
}

impl BernoulliNB {
    /// Create a new `BernoulliNB` (`alpha = 1.0`, `binarize = 0.0`).
    #[must_use]
    pub fn new() -> Self {
        Self {
            alpha: 1.0,
            binarize: 0.0,
            class_log_prior: Vec::new(),
            feature_log_prob: Vec::new(),
            feature_log_neg_prob: Vec::new(),
            n_features: 0,
        }
    }

    /// Set the additive smoothing parameter.
    #[must_use]
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set the binarization threshold (features `> binarize` become 1).
    #[must_use]
    pub fn with_binarize(mut self, binarize: f32) -> Self {
        self.binarize = binarize;
        self
    }

    /// Fit on features `x` and integer labels `y` (in `0..n_classes`).
    ///
    /// # Errors
    /// Returns an error if `x`/`y` lengths disagree or there are no samples.
    pub fn fit(&mut self, x: &Matrix<f32>, y: &[usize]) -> Result<()> {
        let (n_samples, n_features) = x.shape();
        if n_samples == 0 {
            return Err("BernoulliNB: cannot fit with zero samples".into());
        }
        if y.len() != n_samples {
            return Err("BernoulliNB: x/y length mismatch".into());
        }
        let n_classes = y.iter().max().map_or(0, |&m| m + 1);
        let mut class_count = vec![0usize; n_classes];
        let mut present = vec![vec![0.0f64; n_features]; n_classes];
        for (i, &c) in y.iter().enumerate() {
            class_count[c] += 1;
            for j in 0..n_features {
                if x.get(i, j) > self.binarize {
                    present[c][j] += 1.0;
                }
            }
        }
        let alpha = f64::from(self.alpha);
        self.class_log_prior = (0..n_classes)
            .map(|c| (class_count[c] as f64 / n_samples as f64).ln() as f32)
            .collect();
        // Precompute ln(p) and ln(1-p) per (class, feature) ONCE here, instead of recomputing both
        // logs for every (sample, class, feature) in predict (O(n·c·d) -> O(c·d) transcendentals).
        self.feature_log_prob = Vec::with_capacity(n_classes);
        self.feature_log_neg_prob = Vec::with_capacity(n_classes);
        for c in 0..n_classes {
            let denom = class_count[c] as f64 + 2.0 * alpha;
            let mut logp = Vec::with_capacity(n_features);
            let mut logn = Vec::with_capacity(n_features);
            for j in 0..n_features {
                let p = (((present[c][j] + alpha) / denom) as f32).clamp(1e-9, 1.0 - 1e-9);
                logp.push(p.ln());
                logn.push((1.0 - p).ln());
            }
            self.feature_log_prob.push(logp);
            self.feature_log_neg_prob.push(logn);
        }
        self.n_features = n_features;
        Ok(())
    }

    /// Predict class labels by maximizing the Bernoulli joint log-likelihood.
    #[must_use]
    pub fn predict(&self, x: &Matrix<f32>) -> Vec<usize> {
        let (n_samples, _) = x.shape();
        (0..n_samples)
            .map(|i| {
                let mut best_c = 0;
                let mut best_ll = f32::NEG_INFINITY;
                for (c, prior) in self.class_log_prior.iter().enumerate() {
                    let mut ll = *prior;
                    let logp = &self.feature_log_prob[c];
                    let logn = &self.feature_log_neg_prob[c];
                    for j in 0..self.n_features {
                        // Precomputed logs; present feature adds ln(p), absent adds ln(1-p).
                        if x.get(i, j) > self.binarize {
                            ll += logp[j];
                        } else {
                            ll += logn[j];
                        }
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

impl crate::traits::Estimator for BernoulliNB {
    fn fit(&mut self, x: &Matrix<f32>, y: &crate::primitives::Vector<f32>) -> Result<()> {
        let labels: Vec<usize> = y.as_slice().iter().map(|&v| v.round() as usize).collect();
        BernoulliNB::fit(self, x, &labels)
    }
    fn predict(&self, x: &Matrix<f32>) -> crate::primitives::Vector<f32> {
        let labels = BernoulliNB::predict(self, x);
        crate::primitives::Vector::from_vec(labels.into_iter().map(|l| l as f32).collect())
    }
    fn score(&self, x: &Matrix<f32>, y: &crate::primitives::Vector<f32>) -> f32 {
        let preds = BernoulliNB::predict(self, x);
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

    /// FT-BERNOULLINB: matches sklearn.naive_bayes.BernoulliNB on a binary fixture.
    #[test]
    fn bernoulli_nb_matches_sklearn() {
        let x = Matrix::from_vec(
            4,
            4,
            vec![
                1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0,
            ],
        )
        .expect("valid");
        let y = [0usize, 0, 1, 1];
        let mut nb = BernoulliNB::new();
        nb.fit(&x, &y).expect("fit");
        assert_eq!(nb.predict(&x), vec![0, 0, 1, 1]);
        let xt =
            Matrix::from_vec(2, 4, vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]).expect("valid");
        assert_eq!(nb.predict(&xt), vec![0, 1]);
    }
}

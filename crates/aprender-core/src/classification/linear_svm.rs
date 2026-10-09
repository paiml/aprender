#[allow(clippy::wildcard_imports)]
use super::*;
use crate::error::Result;
use crate::primitives::Matrix;

/// Samples per block in `GaussianNB::predict`. The block's feature-major copy
/// (`n_features × 128` f32, 15 KiB at 30 features) stays in L1.
const GNB_PREDICT_BLOCK: usize = 128;

/// The rows `idx` of the row-major `data`, each `n_features` wide.
fn gnb_rows<'a>(
    data: &'a [f32],
    n_features: usize,
    idx: impl Iterator<Item = usize> + 'a,
) -> impl Iterator<Item = &'a [f32]> + 'a {
    idx.map(move |s| &data[s * n_features..(s + 1) * n_features])
}

/// Copies `b` row-major rows into `tile` feature-major: `tile[f * b + j] = rows[j * n_features + f]`.
fn gnb_transpose_block(rows: &[f32], b: usize, tile: &mut [f32]) {
    let n_features = tile.len() / b;
    if n_features == 0 {
        return;
    }
    for (j, row) in rows.chunks_exact(n_features).enumerate() {
        for (f, &v) in row.iter().enumerate() {
            tile[f * b + j] = v;
        }
    }
}

/// `lp[j] = const_term − Σ_f (x[j,f] − μ_f)² · inv_2var_f` for one class over a feature-major
/// block (`tile[f * b + j]`, `b = lp.len()`). Each `lp[j]` takes the same operations in the same
/// order as the per-sample scalar loop, so it is bit-identical to it; the update runs across
/// samples, so it vectorizes.
fn gnb_class_log_posterior(
    tile: &[f32],
    const_term: f32,
    mean_c: &[f32],
    inv_c: &[f32],
    lp: &mut [f32],
) {
    lp.fill(const_term);
    for ((col, &mean), &inv) in tile.chunks_exact(lp.len()).zip(mean_c).zip(inv_c) {
        for (l, &v) in lp.iter_mut().zip(col) {
            let diff = v - mean;
            *l -= diff * diff * inv;
        }
    }
}

impl GaussianNB {
    /// Creates a new Gaussian Naive Bayes classifier.
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::classification::GaussianNB;
    ///
    /// let model = GaussianNB::new();
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self {
            class_priors: None,
            means: None,
            variances: None,
            classes: None,
            var_smoothing: 1e-9,
        }
    }

    /// Sets the variance smoothing parameter.
    ///
    /// Matches scikit-learn: the smoothing added to every per-class feature variance is
    /// `epsilon = var_smoothing * X.var(axis=0).max()` (this value scaled by the largest
    /// feature variance), not a raw additive constant. Avoids numerical instability while
    /// staying scale-aware on mixed-scale data.
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::classification::GaussianNB;
    ///
    /// let model = GaussianNB::new().with_var_smoothing(1e-8);
    /// ```
    #[must_use]
    pub fn with_var_smoothing(mut self, var_smoothing: f32) -> Self {
        self.var_smoothing = var_smoothing;
        self
    }

    /// Trains the Gaussian Naive Bayes classifier.
    ///
    /// Computes class priors, feature means, and variances for each class.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Sample count mismatch between X and y
    /// - Empty data
    /// - Less than 2 classes
    // Contract: naive-bayes-v1, equation = "class_prior"
    pub fn fit(&mut self, x: &Matrix<f32>, y: &[usize]) -> Result<()> {
        let (n_samples, n_features) = x.shape();

        if n_samples == 0 {
            return Err("Cannot fit with empty data".into());
        }

        if y.len() != n_samples {
            return Err("Number of samples in X and y must match".into());
        }

        // Find unique classes
        let mut classes: Vec<usize> = y.to_vec();
        classes.sort_unstable();
        classes.dedup();

        if classes.len() < 2 {
            return Err("Need at least 2 classes".into());
        }

        let n_classes = classes.len();

        // scikit-learn `GaussianNB` defines the variance-smoothing term as the single scalar
        //   epsilon = var_smoothing * X.var(axis=0).max()
        // i.e. `var_smoothing` is SCALED by the largest *feature* variance (computed over ALL
        // training rows, biased/population variance `/n`), then that one `epsilon` is added to
        // EVERY per-class feature variance. A raw additive `var_smoothing` would be thousands
        // of times too small on mixed-scale data, distorting the Gaussian log-likelihood.
        // Refs PMAT-890 / F-GAUSSIANNB-EPSILON-003.
        //
        // Every statistic below is accumulated row by row (one pass over the row-major data per
        // sum) instead of one strided pass per feature. Each feature's sum still adds its samples
        // in index order from the same start value, so the result is bit-identical (#4965).
        let data = x.as_slice();
        let max_feature_var = Self::max_feature_variance(data, n_samples, n_features);
        let epsilon = self.var_smoothing * max_feature_var;

        // Initialize storage
        let mut class_priors = vec![0.0; n_classes];
        let mut means = Vec::with_capacity(n_classes);
        let mut variances = Vec::with_capacity(n_classes);

        // Compute class priors and feature statistics
        for (class_idx, &class_label) in classes.iter().enumerate() {
            // Find samples belonging to this class
            let class_samples: Vec<usize> = y
                .iter()
                .enumerate()
                .filter_map(|(i, &label)| if label == class_label { Some(i) } else { None })
                .collect();

            let n_class_samples = class_samples.len() as f32;
            class_priors[class_idx] = n_class_samples / n_samples as f32;

            let (mean_c, variance_c) =
                Self::class_mean_variance(data, n_features, &class_samples, epsilon);
            means.push(mean_c);
            variances.push(variance_c);
        }

        self.class_priors = Some(class_priors);
        self.means = Some(means);
        self.variances = Some(variances);
        self.classes = Some(classes);

        Ok(())
    }

    /// Predicts class labels for samples.
    ///
    /// Returns the class with highest posterior probability for each sample.
    ///
    /// # Errors
    ///
    /// Returns error if model is not fitted or dimension mismatch.
    // Contract: naive-bayes-v1, equation = "log_posterior"
    pub fn predict(&self, x: &Matrix<f32>) -> Result<Vec<usize>> {
        let means = self.means.as_ref().ok_or("Model not fitted")?;
        let variances = self.variances.as_ref().ok_or("Model not fitted")?;
        let class_priors = self.class_priors.as_ref().ok_or("Model not fitted")?;
        let classes = self.classes.as_ref().ok_or("Model not fitted")?;

        let (n_samples, n_features) = x.shape();
        let n_classes = means.len();

        if n_features != means[0].len() {
            return Err("Feature dimension mismatch".into());
        }

        // Hoist the sample-INDEPENDENT terms out of the per-sample hot loop. A naive
        // implementation recomputes `ln(2π·σ²)` for every (sample, class, feature) — O(n·c·d)
        // transcendental calls. These constants depend only on (class, feature), so precompute
        // them ONCE: O(c·d). This is the dominant cost in `predict`.
        let const_term = Self::log_const_terms(class_priors, variances);
        let inv_2var: Vec<Vec<f32>> = variances
            .iter()
            .map(|vc| vc.iter().map(|&v| 1.0 / (2.0 * v)).collect())
            .collect();

        // For class assignment we only need argmax of the log-posterior — softmax normalization
        // (exp / log-sum-exp / per-sample allocation) is wasted work, so skip it entirely.
        //
        // Samples go through in blocks, copied feature-major, so the per-feature update runs
        // across the samples of a block and vectorizes (#4965). Each sample's log-posterior keeps
        // the scalar loop's operation order, so predictions are bit-identical to it.
        let data = x.as_slice();
        let mut tile = vec![0.0_f32; n_features * GNB_PREDICT_BLOCK];
        let mut lp = [0.0_f32; GNB_PREDICT_BLOCK];
        let mut best_lp = [0.0_f32; GNB_PREDICT_BLOCK];
        let mut best_idx = [0usize; GNB_PREDICT_BLOCK];
        let mut predictions = Vec::with_capacity(n_samples);
        for start in (0..n_samples).step_by(GNB_PREDICT_BLOCK) {
            let b = GNB_PREDICT_BLOCK.min(n_samples - start);
            let tile = &mut tile[..n_features * b];
            gnb_transpose_block(&data[start * n_features..(start + b) * n_features], b, tile);
            let (lp, best_lp, best_idx) = (&mut lp[..b], &mut best_lp[..b], &mut best_idx[..b]);
            best_lp.fill(f32::NEG_INFINITY);
            best_idx.fill(0);
            for class_idx in 0..n_classes {
                gnb_class_log_posterior(
                    tile,
                    const_term[class_idx],
                    &means[class_idx],
                    &inv_2var[class_idx],
                    lp,
                );
                for ((&l, bl), bi) in lp.iter().zip(best_lp.iter_mut()).zip(best_idx.iter_mut()) {
                    if l > *bl {
                        *bl = l;
                        *bi = class_idx;
                    }
                }
            }
            predictions.extend(best_idx.iter().map(|&i| classes[i]));
        }

        Ok(predictions)
    }

    /// The largest per-feature (population) variance over all rows of the row-major `data`:
    /// the scale sklearn applies to `var_smoothing`.
    fn max_feature_variance(data: &[f32], n_samples: usize, n_features: usize) -> f32 {
        let mut sums = vec![0.0_f32; n_features];
        for row in gnb_rows(data, n_features, 0..n_samples) {
            for (acc, &v) in sums.iter_mut().zip(row) {
                *acc += v;
            }
        }
        let feature_means: Vec<f32> = sums.iter().map(|&s| s / n_samples as f32).collect();
        let mut sq = vec![0.0_f32; n_features];
        for row in gnb_rows(data, n_features, 0..n_samples) {
            for ((acc, &v), &mean) in sq.iter_mut().zip(row).zip(&feature_means) {
                let diff = v - mean;
                *acc += diff * diff;
            }
        }
        let mut max_feature_var = 0.0_f32;
        for &s in &sq {
            let feature_var = s / n_samples as f32;
            if feature_var > max_feature_var {
                max_feature_var = feature_var;
            }
        }
        max_feature_var
    }

    /// Per-feature mean and smoothed variance of the rows `class_samples` of `data`.
    fn class_mean_variance(
        data: &[f32],
        n_features: usize,
        class_samples: &[usize],
        epsilon: f32,
    ) -> (Vec<f32>, Vec<f32>) {
        // The value `Iterator::sum` folds from, so each sum below equals the iterator sum this
        // replaced bit for bit, the sign of an all-zero sum included.
        let start: f32 = std::iter::empty::<f32>().sum();
        let n_class_samples = class_samples.len() as f32;
        let mut acc = vec![start; n_features];
        for row in gnb_rows(data, n_features, class_samples.iter().copied()) {
            for (a, &v) in acc.iter_mut().zip(row) {
                *a += v;
            }
        }
        let mean_c: Vec<f32> = acc.iter().map(|&s| s / n_class_samples).collect();
        acc.fill(start);
        for row in gnb_rows(data, n_features, class_samples.iter().copied()) {
            for ((a, &v), &mean) in acc.iter_mut().zip(row).zip(&mean_c) {
                let diff = v - mean;
                *a += diff * diff;
            }
        }
        let variance_c = acc.iter().map(|&s| s / n_class_samples + epsilon).collect();
        (mean_c, variance_c)
    }

    /// Precomputes the sample-independent per-class log term
    /// `ln(P(y=c)) + Σ_f −0.5·ln(2π·σ²_{c,f})`, hoisted out of the prediction hot loop so the
    /// O(n·c·d) `ln` evaluations in a naive Gaussian-NB collapse to O(c·d).
    fn log_const_terms(class_priors: &[f32], variances: &[Vec<f32>]) -> Vec<f32> {
        variances
            .iter()
            .zip(class_priors)
            .map(|(vc, &prior)| {
                let mut t = prior.ln();
                for &v in vc {
                    t += -0.5 * (2.0 * std::f32::consts::PI * v).ln();
                }
                t
            })
            .collect()
    }

    /// Returns probability estimates for each class.
    ///
    /// Uses Bayes' theorem with Gaussian likelihood:
    /// P(y=c|X) ∝ P(y=c) * ∏ `P(x_i|y=c)`
    ///
    /// # Errors
    ///
    /// Returns error if model is not fitted or dimension mismatch.
    pub fn predict_proba(&self, x: &Matrix<f32>) -> Result<Vec<Vec<f32>>> {
        let means = self.means.as_ref().ok_or("Model not fitted")?;
        let variances = self.variances.as_ref().ok_or("Model not fitted")?;
        let class_priors = self.class_priors.as_ref().ok_or("Model not fitted")?;

        let (n_samples, n_features) = x.shape();
        let n_classes = means.len();

        if n_features != means[0].len() {
            return Err("Feature dimension mismatch".into());
        }

        // Same hoist as `predict`: the `ln(2π·σ²)` term is sample-independent — precompute once.
        let const_term = Self::log_const_terms(class_priors, variances);

        let mut probabilities = Vec::with_capacity(n_samples);

        for sample_idx in 0..n_samples {
            let mut log_probs = vec![0.0; n_classes];

            // Compute log posterior for each class
            for class_idx in 0..n_classes {
                // Start with the precomputed log-prior + log-normalization constant
                let mut lp = const_term[class_idx];

                // Subtract the per-feature Mahalanobis term: (x-μ)² / (2σ²)
                for feature_idx in 0..n_features {
                    let diff = x.get(sample_idx, feature_idx) - means[class_idx][feature_idx];
                    lp -= (diff * diff) / (2.0 * variances[class_idx][feature_idx]);
                }

                log_probs[class_idx] = lp;
            }

            // Convert log probabilities to probabilities using log-sum-exp trick
            let max_log_prob = log_probs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exp_probs: Vec<f32> = log_probs
                .iter()
                .map(|&log_p| (log_p - max_log_prob).exp())
                .collect();
            let sum: f32 = exp_probs.iter().sum();
            let normalized: Vec<f32> = exp_probs.iter().map(|p| p / sum).collect();

            probabilities.push(normalized);
        }

        Ok(probabilities)
    }
}

impl Default for GaussianNB {
    fn default() -> Self {
        Self::new()
    }
}

/// Linear Support Vector Machine (SVM) classifier.
///
/// Implements binary classification using hinge loss and subgradient descent.
/// For multi-class problems, use One-vs-Rest strategy.
///
/// # Algorithm
///
/// Minimizes the objective:
/// ```text
/// min  λ||w||² + (1/n) Σᵢ max(0, 1 - yᵢ(w·xᵢ + b))
/// ```
///
/// Where λ = 1/(2nC) controls regularization strength.
///
/// # Example
///
/// ```ignore
/// use aprender::classification::LinearSVM;
/// use aprender::primitives::Matrix;
///
/// let x = Matrix::from_vec(4, 2, vec![
///     0.0, 0.0,
///     0.0, 1.0,
///     1.0, 0.0,
///     1.0, 1.0,
/// ])?;
/// let y = vec![0, 0, 1, 1];
///
/// let mut svm = LinearSVM::new();
/// svm.fit(&x, &y)?;
/// let predictions = svm.predict(&x)?;
/// ```
#[derive(Debug, Clone)]
pub struct LinearSVM {
    /// Weights for each feature
    pub(crate) weights: Option<Vec<f32>>,
    /// Bias term
    pub(crate) bias: f32,
    /// Regularization parameter (default: 1.0)
    /// Larger C means less regularization
    pub(crate) c: f32,
    /// Learning rate for subgradient descent (default: 0.01)
    pub(crate) learning_rate: f32,
    /// Maximum iterations (default: 1000)
    pub(crate) max_iter: usize,
    /// Convergence tolerance (default: 1e-4)
    pub(crate) tol: f32,
}

#[cfg(test)]
#[path = "tests_nb_contract.rs"]
mod tests_nb_contract;

#[cfg(test)]
#[path = "tests_gnb_blocked.rs"]
mod tests_gnb_blocked;

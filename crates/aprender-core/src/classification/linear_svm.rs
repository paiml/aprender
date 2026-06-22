#[allow(clippy::wildcard_imports)]
use super::*;
use crate::error::Result;
use crate::primitives::Matrix;

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
        let mut max_feature_var = 0.0_f32;
        for feature_idx in 0..n_features {
            let mut sum = 0.0_f32;
            for sample_idx in 0..n_samples {
                sum += x.get(sample_idx, feature_idx);
            }
            let mean = sum / n_samples as f32;
            let mut sum_sq_diff = 0.0_f32;
            for sample_idx in 0..n_samples {
                let diff = x.get(sample_idx, feature_idx) - mean;
                sum_sq_diff += diff * diff;
            }
            let feature_var = sum_sq_diff / n_samples as f32;
            if feature_var > max_feature_var {
                max_feature_var = feature_var;
            }
        }
        let epsilon = self.var_smoothing * max_feature_var;

        // Initialize storage
        let mut class_priors = vec![0.0; n_classes];
        let mut means = vec![vec![0.0; n_features]; n_classes];
        let mut variances = vec![vec![0.0; n_features]; n_classes];

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

            // Compute mean for each feature
            for (feature_idx, mean_val) in means[class_idx].iter_mut().enumerate() {
                let sum: f32 = class_samples
                    .iter()
                    .map(|&sample_idx| x.get(sample_idx, feature_idx))
                    .sum();
                *mean_val = sum / n_class_samples;
            }

            // Compute variance for each feature
            for (feature_idx, variance_val) in variances[class_idx].iter_mut().enumerate() {
                let mean = means[class_idx][feature_idx];
                let sum_sq_diff: f32 = class_samples
                    .iter()
                    .map(|&sample_idx| {
                        let diff = x.get(sample_idx, feature_idx) - mean;
                        diff * diff
                    })
                    .sum();
                *variance_val = sum_sq_diff / n_class_samples + epsilon;
            }
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
        let mut predictions = Vec::with_capacity(n_samples);
        for sample_idx in 0..n_samples {
            let mut best_idx = 0usize;
            let mut best_lp = f32::NEG_INFINITY;
            for class_idx in 0..n_classes {
                let mean_c = &means[class_idx];
                let inv_c = &inv_2var[class_idx];
                let mut lp = const_term[class_idx];
                for feature_idx in 0..n_features {
                    let diff = x.get(sample_idx, feature_idx) - mean_c[feature_idx];
                    lp -= diff * diff * inv_c[feature_idx];
                }
                if lp > best_lp {
                    best_lp = lp;
                    best_idx = class_idx;
                }
            }
            predictions.push(classes[best_idx]);
        }

        Ok(predictions)
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

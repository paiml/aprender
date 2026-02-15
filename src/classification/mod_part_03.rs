
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
    /// Adds this value to variances to avoid numerical instability.
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
                *variance_val = sum_sq_diff / n_class_samples + self.var_smoothing;
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
    pub fn predict(&self, x: &Matrix<f32>) -> Result<Vec<usize>> {
        let probabilities = self.predict_proba(x)?;
        let classes = self.classes.as_ref().ok_or("Model not fitted")?;

        let predictions: Vec<usize> = probabilities
            .iter()
            .map(|probs| {
                let max_idx = probs
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| {
                        a.partial_cmp(b)
                            .expect("Probabilities are valid f32 (not NaN)")
                    })
                    .map(|(idx, _)| idx)
                    .expect("Probabilities vector is non-empty (n_classes >= 2)");
                classes[max_idx]
            })
            .collect();

        Ok(predictions)
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

        let mut probabilities = Vec::with_capacity(n_samples);

        for sample_idx in 0..n_samples {
            let mut log_probs = vec![0.0; n_classes];

            // Compute log posterior for each class
            for class_idx in 0..n_classes {
                // Start with log prior
                log_probs[class_idx] = class_priors[class_idx].ln();

                // Add log likelihood for each feature (Gaussian PDF)
                for feature_idx in 0..n_features {
                    let x_val = x.get(sample_idx, feature_idx);
                    let mean = means[class_idx][feature_idx];
                    let variance = variances[class_idx][feature_idx];

                    // Log of Gaussian PDF: -0.5 * log(2π*σ²) - (x-μ)² / (2σ²)
                    let diff = x_val - mean;
                    let log_likelihood = -0.5 * (2.0 * std::f32::consts::PI * variance).ln()
                        - (diff * diff) / (2.0 * variance);

                    log_probs[class_idx] += log_likelihood;
                }
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
    weights: Option<Vec<f32>>,
    /// Bias term
    bias: f32,
    /// Regularization parameter (default: 1.0)
    /// Larger C means less regularization
    c: f32,
    /// Learning rate for subgradient descent (default: 0.01)
    learning_rate: f32,
    /// Maximum iterations (default: 1000)
    max_iter: usize,
    /// Convergence tolerance (default: 1e-4)
    tol: f32,
}

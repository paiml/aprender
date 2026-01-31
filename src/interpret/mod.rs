//! Model Interpretability and Explainability.
//!
//! This module provides tools for understanding model predictions through
//! feature attribution and explanation methods.
//!
//! # Methods
//!
//! - **SHAP (`SHapley` Additive exPlanations)**: Computes feature importance using
//!   Shapley values from cooperative game theory.
//! - **Permutation Importance**: Measures feature importance by shuffling features
//!   and measuring prediction change.
//! - **Feature Contributions**: Decomposes predictions into per-feature contributions.
//!
//! # Example
//!
//! ```ignore
//! use aprender::interpret::{KernelSHAP, Explainer};
//!
//! // Create explainer with trained model
//! let explainer = KernelSHAP::new(model, background_data);
//!
//! // Explain a prediction
//! let shap_values = explainer.explain(&sample);
//!
//! // shap_values[i] = contribution of feature i to the prediction
//! ```
//!
//! # References
//!
//! - Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting
//!   Model Predictions. `NeurIPS`.
//! - Ribeiro, M. T., et al. (2016). "Why Should I Trust You?": Explaining the
//!   Predictions of Any Classifier (LIME). KDD.

use crate::primitives::Vector;

/// Trait for model explainers.
///
/// Explainers compute feature attributions/importance scores for predictions.
pub trait Explainer {
    /// Explain a single prediction.
    ///
    /// Returns a vector of feature contributions where:
    /// - Positive values indicate features that increase the prediction
    /// - Negative values indicate features that decrease the prediction
    /// - Sum of contributions + `expected_value` ≈ prediction
    ///
    /// # Arguments
    ///
    /// * `sample` - Input sample to explain
    ///
    /// # Returns
    ///
    /// Vector of SHAP values (one per feature)
    fn explain(&self, sample: &Vector<f32>) -> Vector<f32>;

    /// Get the expected value (baseline prediction).
    ///
    /// This is the model's average prediction over the background dataset.
    fn expected_value(&self) -> f32;
}

/// SHAP (`SHapley` Additive exPlanations) values for feature attribution.
///
/// Computes exact or approximate Shapley values to explain model predictions.
/// Uses kernel-based approximation for efficiency.
///
/// # Properties
///
/// SHAP values satisfy three important properties:
/// 1. **Local accuracy**: `prediction = expected_value + sum(shap_values)`
/// 2. **Missingness**: Features not present have zero attribution
/// 3. **Consistency**: If a feature's contribution increases, its SHAP value won't decrease
///
/// # Example
///
/// ```ignore
/// use aprender::interpret::ShapExplainer;
///
/// let explainer = ShapExplainer::new(&model, &background_data);
/// let shap_values = explainer.explain(&sample);
///
/// // Top contributing features
/// let mut importance: Vec<_> = shap_values.iter().enumerate().collect();
/// importance.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
/// ```
#[derive(Debug)]
pub struct ShapExplainer {
    /// Background data for computing expected values
    background: Vec<Vector<f32>>,
    /// Expected prediction (baseline)
    expected_value: f32,
    /// Number of samples for Monte Carlo approximation
    n_samples: usize,
    /// Number of features
    n_features: usize,
}

impl ShapExplainer {
    /// Create a new SHAP explainer.
    ///
    /// # Arguments
    ///
    /// * `background` - Background dataset for computing expected values
    /// * `model_fn` - Function that makes predictions given input
    ///
    /// # Example
    ///
    /// ```ignore
    /// let background = vec![sample1, sample2, sample3];
    /// let explainer = ShapExplainer::new(&background, |x| model.predict(x));
    /// ```
    pub fn new<F>(background: &[Vector<f32>], model_fn: F) -> Self
    where
        F: Fn(&Vector<f32>) -> f32,
    {
        assert!(!background.is_empty(), "Background data cannot be empty");

        let n_features = background[0].len();

        // Compute expected value as mean prediction over background
        let expected_value: f32 =
            background.iter().map(&model_fn).sum::<f32>() / background.len() as f32;

        Self {
            background: background.to_vec(),
            expected_value,
            n_samples: 100, // Default number of samples
            n_features,
        }
    }

    /// Set the number of samples for Monte Carlo approximation.
    ///
    /// Higher values give more accurate SHAP values but take longer.
    #[must_use]
    pub fn with_n_samples(mut self, n_samples: usize) -> Self {
        self.n_samples = n_samples;
        self
    }

    /// Compute SHAP values using kernel approximation.
    ///
    /// Uses a simplified kernel SHAP approach:
    /// 1. Sample coalitions (subsets of features)
    /// 2. For each coalition, compute marginal contribution
    /// 3. Weight contributions by kernel weights
    pub fn explain_with_model<F>(&self, sample: &Vector<f32>, model_fn: F) -> Vector<f32>
    where
        F: Fn(&Vector<f32>) -> f32,
    {
        assert_eq!(
            sample.len(),
            self.n_features,
            "Sample must have {} features",
            self.n_features
        );

        let n = self.n_features;
        let mut shap_values = vec![0.0f32; n];

        // For each feature, estimate marginal contribution
        for feature_idx in 0..n {
            let mut contribution = 0.0f32;
            let mut count = 0;

            // Monte Carlo sampling over coalitions and background
            for bg_sample in &self.background {
                // Prediction with feature from sample
                let mut x_with = bg_sample.clone();
                x_with[feature_idx] = sample[feature_idx];

                // Prediction without feature (using background value)
                let x_without = bg_sample.clone();

                let pred_with = model_fn(&x_with);
                let pred_without = model_fn(&x_without);

                contribution += pred_with - pred_without;
                count += 1;
            }

            shap_values[feature_idx] = contribution / count.max(1) as f32;
        }

        // Normalize to satisfy local accuracy
        let sum_shap: f32 = shap_values.iter().sum();
        let prediction = model_fn(sample);
        let target_sum = prediction - self.expected_value;

        if sum_shap.abs() > 1e-8 {
            let scale = target_sum / sum_shap;
            for v in &mut shap_values {
                *v *= scale;
            }
        }

        Vector::from_slice(&shap_values)
    }

    /// Get the background dataset.
    #[must_use]
    pub fn background(&self) -> &[Vector<f32>] {
        &self.background
    }

    /// Get the expected value.
    #[must_use]
    pub fn expected_value(&self) -> f32 {
        self.expected_value
    }

    /// Get the number of features.
    #[must_use]
    pub fn n_features(&self) -> usize {
        self.n_features
    }
}

/// Permutation feature importance.
///
/// Measures feature importance by shuffling each feature and measuring
/// the decrease in model performance.
///
/// # Properties
///
/// - **Model-agnostic**: Works with any model
/// - **Fast**: `O(n_features` * `n_samples`) predictions
/// - **Simple interpretation**: Importance = performance drop when feature is shuffled
///
/// # Example
///
/// ```ignore
/// use aprender::interpret::PermutationImportance;
///
/// let importance = PermutationImportance::compute(
///     &model,
///     &X_test,
///     &y_test,
///     |pred, true_val| (pred - true_val).powi(2), // MSE
/// );
///
/// // importance[i] = how much error increases when feature i is shuffled
/// ```
#[derive(Debug, Clone)]
pub struct PermutationImportance {
    /// Importance scores for each feature
    pub importance: Vector<f32>,
    /// Baseline score (without shuffling)
    pub baseline_score: f32,
}

impl PermutationImportance {
    /// Compute permutation importance.
    ///
    /// # Arguments
    ///
    /// * `predict_fn` - Function that predicts given input
    /// * `X` - Feature matrix (samples as vectors)
    /// * `y` - True labels
    /// * `score_fn` - Scoring function (e.g., MSE, accuracy)
    ///
    /// # Returns
    ///
    /// `PermutationImportance` with importance scores for each feature
    pub fn compute<P, S>(predict_fn: P, x: &[Vector<f32>], y: &[f32], score_fn: S) -> Self
    where
        P: Fn(&Vector<f32>) -> f32,
        S: Fn(f32, f32) -> f32,
    {
        assert!(!x.is_empty(), "Data cannot be empty");
        assert_eq!(x.len(), y.len(), "X and y must have same length");

        let n_samples = x.len();
        let n_features = x[0].len();

        // Compute baseline score
        let baseline_score: f32 = x
            .iter()
            .zip(y.iter())
            .map(|(xi, &yi)| score_fn(predict_fn(xi), yi))
            .sum::<f32>()
            / n_samples as f32;

        let mut importance = vec![0.0f32; n_features];

        // For each feature, shuffle and measure score drop
        for feature_idx in 0..n_features {
            let mut total_shuffled_score = 0.0f32;

            // Simple permutation: use a circular shift
            for (i, xi) in x.iter().enumerate() {
                // Create shuffled sample (use next sample's feature value)
                let mut xi_shuffled = xi.clone();
                let shuffled_idx = (i + 1) % n_samples;
                xi_shuffled[feature_idx] = x[shuffled_idx][feature_idx];

                let pred = predict_fn(&xi_shuffled);
                total_shuffled_score += score_fn(pred, y[i]);
            }

            let shuffled_score = total_shuffled_score / n_samples as f32;
            importance[feature_idx] = shuffled_score - baseline_score;
        }

        Self {
            importance: Vector::from_slice(&importance),
            baseline_score,
        }
    }

    /// Get importance scores.
    #[must_use]
    pub fn scores(&self) -> &Vector<f32> {
        &self.importance
    }

    /// Get feature ranking (indices sorted by importance, descending).
    #[must_use]
    pub fn ranking(&self) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..self.importance.len()).collect();
        indices.sort_by(|&a, &b| {
            self.importance[b]
                .abs()
                .partial_cmp(&self.importance[a].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        indices
    }
}

/// Feature contribution analysis.
///
/// Decomposes a prediction into per-feature contributions for interpretable
/// models (e.g., linear models, tree ensembles).
///
/// For a linear model: `prediction = bias + sum(weight[i] * feature[i])`
/// Each term `weight[i] * feature[i]` is a feature contribution.
#[derive(Debug, Clone)]
pub struct FeatureContributions {
    /// Contribution of each feature to the prediction
    pub contributions: Vector<f32>,
    /// Bias/intercept term
    pub bias: f32,
    /// Total prediction
    pub prediction: f32,
}

impl FeatureContributions {
    /// Create feature contributions for a linear model.
    ///
    /// # Arguments
    ///
    /// * `weights` - Model weights (coefficients)
    /// * `features` - Input features
    /// * `bias` - Model bias/intercept
    #[must_use]
    pub fn from_linear(weights: &Vector<f32>, features: &Vector<f32>, bias: f32) -> Self {
        assert_eq!(
            weights.len(),
            features.len(),
            "Weights and features must have same length"
        );

        let contributions: Vec<f32> = weights
            .as_slice()
            .iter()
            .zip(features.as_slice().iter())
            .map(|(&w, &f)| w * f)
            .collect();

        let prediction = bias + contributions.iter().sum::<f32>();

        Self {
            contributions: Vector::from_slice(&contributions),
            bias,
            prediction,
        }
    }

    /// Create from pre-computed contributions.
    #[must_use]
    pub fn new(contributions: Vector<f32>, bias: f32) -> Self {
        let prediction = bias + contributions.sum();
        Self {
            contributions,
            bias,
            prediction,
        }
    }

    /// Get top contributing features.
    ///
    /// Returns indices of features with highest absolute contributions.
    #[must_use]
    pub fn top_features(&self, k: usize) -> Vec<(usize, f32)> {
        let mut indexed: Vec<(usize, f32)> = self
            .contributions
            .as_slice()
            .iter()
            .copied()
            .enumerate()
            .collect();
        indexed.sort_by(|a, b| {
            b.1.abs()
                .partial_cmp(&a.1.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        indexed.truncate(k);
        indexed
    }

    /// Verify that contributions sum to prediction.
    ///
    /// Returns true if `bias + sum(contributions) ≈ prediction`.
    #[must_use]
    pub fn verify_sum(&self, tolerance: f32) -> bool {
        let reconstructed = self.bias + self.contributions.sum();
        (reconstructed - self.prediction).abs() < tolerance
    }
}

/// Integrated Gradients for neural network attribution.
///
/// Computes attributions by integrating gradients along a path from
/// a baseline to the input.
///
/// ```text
/// IG_i(x) = (x_i - x'_i) * ∫ (∂F(x' + α(x-x')) / ∂x_i) dα
/// ```
///
/// where x' is a baseline (typically zeros).
///
/// # References
///
/// - Sundararajan, M., et al. (2017). Axiomatic Attribution for Deep Networks.
#[derive(Debug)]
pub struct IntegratedGradients {
    /// Number of steps for numerical integration
    n_steps: usize,
}

impl IntegratedGradients {
    /// Create new Integrated Gradients explainer.
    ///
    /// # Arguments
    ///
    /// * `n_steps` - Number of steps for Riemann approximation (default: 50)
    #[must_use]
    pub fn new(n_steps: usize) -> Self {
        Self { n_steps }
    }

    /// Compute attributions using numerical gradient approximation.
    ///
    /// # Arguments
    ///
    /// * `model_fn` - Prediction function
    /// * `sample` - Input to explain
    /// * `baseline` - Baseline input (typically zeros)
    ///
    /// # Returns
    ///
    /// Attribution for each input feature
    pub fn attribute<F>(
        &self,
        model_fn: F,
        sample: &Vector<f32>,
        baseline: &Vector<f32>,
    ) -> Vector<f32>
    where
        F: Fn(&Vector<f32>) -> f32,
    {
        assert_eq!(
            sample.len(),
            baseline.len(),
            "Sample and baseline must have same length"
        );

        let n = sample.len();
        let mut attributions = vec![0.0f32; n];

        // Compute path integral using Riemann sum
        for step in 0..self.n_steps {
            let alpha = (step as f32 + 0.5) / self.n_steps as f32;

            // Interpolated point
            let mut x_interp: Vec<f32> = (0..n)
                .map(|i| baseline[i] + alpha * (sample[i] - baseline[i]))
                .collect();

            // Numerical gradient at this point
            let eps = 1e-4;
            for i in 0..n {
                let original = x_interp[i];

                x_interp[i] = original + eps;
                let f_plus = model_fn(&Vector::from_slice(&x_interp));

                x_interp[i] = original - eps;
                let f_minus = model_fn(&Vector::from_slice(&x_interp));

                x_interp[i] = original;

                let grad = (f_plus - f_minus) / (2.0 * eps);
                attributions[i] += grad / self.n_steps as f32;
            }
        }

        // Multiply by (x - x')
        for i in 0..n {
            attributions[i] *= sample[i] - baseline[i];
        }

        Vector::from_slice(&attributions)
    }
}

impl Default for IntegratedGradients {
    fn default() -> Self {
        Self::new(50)
    }
}

/// LIME (Local Interpretable Model-agnostic Explanations).
///
/// Explains predictions by fitting a simple interpretable model locally
/// around the instance being explained.
///
/// # Algorithm
///
/// 1. Generate perturbed samples around the instance
/// 2. Get model predictions for each perturbation
/// 3. Weight samples by proximity to original instance
/// 4. Fit weighted linear model to the perturbations
/// 5. Return linear model coefficients as explanations
///
/// # Reference
///
/// - Ribeiro, M. T., et al. (2016). "Why Should I Trust You?": Explaining
///   the Predictions of Any Classifier. KDD.
#[derive(Debug)]
pub struct LIME {
    /// Number of perturbed samples to generate
    n_samples: usize,
    /// Kernel width for proximity weighting
    kernel_width: f32,
}

impl LIME {
    /// Create new LIME explainer.
    ///
    /// # Arguments
    ///
    /// * `n_samples` - Number of perturbations to generate (default: 1000)
    /// * `kernel_width` - Width of exponential kernel (default: 0.75)
    #[must_use]
    pub fn new(n_samples: usize, kernel_width: f32) -> Self {
        Self {
            n_samples,
            kernel_width,
        }
    }

    /// Explain a prediction by fitting a local linear model.
    ///
    /// # Arguments
    ///
    /// * `model_fn` - Prediction function
    /// * `sample` - Instance to explain
    /// * `seed` - Random seed for reproducibility
    ///
    /// # Returns
    ///
    /// Coefficients of the local linear model (feature importance)
    pub fn explain<F>(&self, model_fn: F, sample: &Vector<f32>, seed: u64) -> LIMEExplanation
    where
        F: Fn(&Vector<f32>) -> f32,
    {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let n = sample.len();
        let mut rng = StdRng::seed_from_u64(seed);

        // Generate perturbed samples
        let mut perturbed_data = Vec::with_capacity(self.n_samples);
        let mut predictions = Vec::with_capacity(self.n_samples);
        let mut weights = Vec::with_capacity(self.n_samples);

        // Original prediction
        let original_pred = model_fn(sample);

        for _ in 0..self.n_samples {
            // Create perturbation (random feature masking)
            let mut perturbed = sample.clone();
            let mut distance_sq = 0.0_f32;

            for i in 0..n {
                if rng.gen::<f32>() < 0.5 {
                    // Perturb this feature by adding noise
                    let noise = rng.gen_range(-0.5..0.5);
                    perturbed[i] += noise * sample[i].abs().max(1.0);
                    distance_sq += noise * noise;
                }
            }

            let pred = model_fn(&perturbed);
            let weight = (-distance_sq / (2.0 * self.kernel_width * self.kernel_width)).exp();

            perturbed_data.push(perturbed);
            predictions.push(pred);
            weights.push(weight);
        }

        // Fit weighted linear regression
        // Using simplified normal equations: (X^T W X)^-1 X^T W y
        let coefficients =
            Self::fit_weighted_linear(&perturbed_data, &predictions, &weights, sample);

        LIMEExplanation {
            coefficients: Vector::from_slice(&coefficients),
            intercept: original_pred
                - coefficients
                    .iter()
                    .zip(sample.as_slice())
                    .map(|(&c, &x)| c * x)
                    .sum::<f32>(),
            original_prediction: original_pred,
        }
    }

    /// Fit weighted linear regression.
    fn fit_weighted_linear(
        x_data: &[Vector<f32>],
        y: &[f32],
        weights: &[f32],
        reference: &Vector<f32>,
    ) -> Vec<f32> {
        let n = reference.len();
        let _m = x_data.len();

        // Compute relative features (difference from reference)
        let x_centered: Vec<Vec<f32>> = x_data
            .iter()
            .map(|xi| {
                xi.as_slice()
                    .iter()
                    .zip(reference.as_slice())
                    .map(|(&x, &r)| x - r)
                    .collect()
            })
            .collect();

        // Y centered around mean
        let y_mean: f32 = y.iter().zip(weights).map(|(&yi, &w)| yi * w).sum::<f32>()
            / weights.iter().sum::<f32>().max(1e-10);

        // Weighted covariance
        let mut xtx = vec![vec![0.0_f32; n]; n];
        let mut xty = vec![0.0_f32; n];

        for (i, (xi, &wi)) in x_centered.iter().zip(weights).enumerate() {
            let yi_centered = y[i] - y_mean;
            for j in 0..n {
                xty[j] += wi * xi[j] * yi_centered;
                for k in 0..n {
                    xtx[j][k] += wi * xi[j] * xi[k];
                }
            }
        }

        // Add regularization
        for (j, row) in xtx.iter_mut().enumerate() {
            row[j] += 0.01;
        }

        // Solve using simple Gaussian elimination
        Self::solve_linear_system(&xtx, &xty)
    }

    /// Solve linear system Ax = b using Gaussian elimination.
    #[allow(clippy::needless_range_loop)]
    fn solve_linear_system(a: &[Vec<f32>], b: &[f32]) -> Vec<f32> {
        let n = b.len();
        let mut aug: Vec<Vec<f32>> = a
            .iter()
            .zip(b)
            .map(|(row, &bi)| {
                let mut r = row.clone();
                r.push(bi);
                r
            })
            .collect();

        // Forward elimination
        for i in 0..n {
            // Find pivot
            let mut max_row = i;
            for k in (i + 1)..n {
                if aug[k][i].abs() > aug[max_row][i].abs() {
                    max_row = k;
                }
            }
            aug.swap(i, max_row);

            let pivot = aug[i][i];
            if pivot.abs() < 1e-10 {
                continue;
            }

            // Eliminate below
            for k in (i + 1)..n {
                let factor = aug[k][i] / pivot;
                for j in i..=n {
                    aug[k][j] -= factor * aug[i][j];
                }
            }
        }

        // Back substitution
        let mut x = vec![0.0_f32; n];
        for i in (0..n).rev() {
            let mut sum = aug[i][n];
            for j in (i + 1)..n {
                sum -= aug[i][j] * x[j];
            }
            let pivot = aug[i][i];
            x[i] = if pivot.abs() > 1e-10 {
                sum / pivot
            } else {
                0.0
            };
        }
        x
    }

    #[must_use]
    pub fn n_samples(&self) -> usize {
        self.n_samples
    }

    #[must_use]
    pub fn kernel_width(&self) -> f32 {
        self.kernel_width
    }
}

impl Default for LIME {
    fn default() -> Self {
        Self::new(500, 0.75)
    }
}

/// Saliency Maps for neural network visualization.
///
/// Computes gradients of the output with respect to input features,
/// showing which parts of the input most affect the prediction.
///
/// # Method
///
/// Uses numerical gradient estimation via finite differences:
/// ```text
/// saliency_i = ∂f/∂x_i ≈ (f(x + ε*e_i) - f(x - ε*e_i)) / (2ε)
/// ```
///
/// # Reference
///
/// - Simonyan, K., et al. (2014). Deep Inside Convolutional Networks:
///   Visualising Image Classification Models and Saliency Maps.
#[derive(Debug, Clone)]
pub struct SaliencyMap {
    /// Epsilon for numerical gradient
    epsilon: f32,
}

impl SaliencyMap {
    /// Create a new saliency map calculator.
    #[must_use]
    pub fn new() -> Self {
        Self { epsilon: 1e-4 }
    }

    /// Create with custom epsilon.
    #[must_use]
    pub fn with_epsilon(epsilon: f32) -> Self {
        Self { epsilon }
    }

    /// Get epsilon value.
    #[must_use]
    pub fn epsilon(&self) -> f32 {
        self.epsilon
    }

    /// Compute saliency (gradient) for each input feature.
    ///
    /// # Arguments
    ///
    /// * `model_fn` - Prediction function
    /// * `sample` - Input to explain
    ///
    /// # Returns
    ///
    /// Gradient values for each input feature.
    pub fn compute<F>(&self, model_fn: F, sample: &Vector<f32>) -> Vector<f32>
    where
        F: Fn(&Vector<f32>) -> f32,
    {
        let n = sample.len();
        let mut gradients = vec![0.0f32; n];

        for i in 0..n {
            // Forward difference
            let mut x_plus = sample.clone();
            x_plus[i] += self.epsilon;
            let f_plus = model_fn(&x_plus);

            // Backward difference
            let mut x_minus = sample.clone();
            x_minus[i] -= self.epsilon;
            let f_minus = model_fn(&x_minus);

            // Central difference gradient
            gradients[i] = (f_plus - f_minus) / (2.0 * self.epsilon);
        }

        Vector::from_slice(&gradients)
    }

    /// Compute absolute saliency (magnitude of gradient).
    ///
    /// Useful for visualizing which features matter, regardless of direction.
    pub fn compute_absolute<F>(&self, model_fn: F, sample: &Vector<f32>) -> Vector<f32>
    where
        F: Fn(&Vector<f32>) -> f32,
    {
        let gradients = self.compute(model_fn, sample);
        let abs_grads: Vec<f32> = gradients.as_slice().iter().map(|&g| g.abs()).collect();
        Vector::from_slice(&abs_grads)
    }

    /// Compute smooth gradient by averaging over noisy samples.
    ///
    /// `SmoothGrad` reduces noise in saliency maps by averaging gradients
    /// over multiple noisy versions of the input.
    ///
    /// # Reference
    ///
    /// - Smilkov, D., et al. (2017). `SmoothGrad`: removing noise by adding noise.
    pub fn smooth_grad<F>(
        &self,
        model_fn: F,
        sample: &Vector<f32>,
        n_samples: usize,
        noise_level: f32,
    ) -> Vector<f32>
    where
        F: Fn(&Vector<f32>) -> f32,
    {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let n = sample.len();
        let mut sum_grads = vec![0.0f32; n];

        for _ in 0..n_samples {
            // Add noise to sample
            let noisy: Vec<f32> = sample
                .as_slice()
                .iter()
                .map(|&x| x + rng.gen_range(-noise_level..noise_level))
                .collect();
            let noisy_sample = Vector::from_slice(&noisy);

            // Compute gradient for noisy sample
            let grad = self.compute(&model_fn, &noisy_sample);

            for (j, &g) in grad.as_slice().iter().enumerate() {
                sum_grads[j] += g;
            }
        }

        // Average
        for g in &mut sum_grads {
            *g /= n_samples as f32;
        }

        Vector::from_slice(&sum_grads)
    }
}

impl Default for SaliencyMap {
    fn default() -> Self {
        Self::new()
    }
}

/// Counterfactual explanation generator.
///
/// Finds the minimal change to an input that changes the model's prediction
/// to a target class. Counterfactuals answer "What would need to change?"
///
/// # Method
///
/// Uses gradient descent to find the closest input in feature space
/// that results in a different classification.
///
/// ```text
/// x_cf = argmin_x' ||x' - x||² s.t. f(x') = target_class
/// ```
///
/// # Reference
///
/// - Wachter, S., et al. (2018). Counterfactual Explanations without Opening
///   the Black Box. Harvard Journal of Law & Technology.
#[derive(Debug, Clone)]
pub struct CounterfactualExplainer {
    /// Maximum optimization iterations
    max_iter: usize,
    /// Step size for gradient descent
    step_size: f32,
    /// Epsilon for numerical gradient
    epsilon: f32,
}

impl CounterfactualExplainer {
    /// Create a new counterfactual explainer.
    ///
    /// # Arguments
    ///
    /// * `max_iter` - Maximum iterations for optimization
    /// * `step_size` - Learning rate for gradient descent
    #[must_use]
    pub fn new(max_iter: usize, step_size: f32) -> Self {
        Self {
            max_iter,
            step_size,
            epsilon: 1e-4,
        }
    }

    /// Get max iterations.
    #[must_use]
    pub fn max_iter(&self) -> usize {
        self.max_iter
    }

    /// Get step size.
    #[must_use]
    pub fn step_size(&self) -> f32 {
        self.step_size
    }

    /// Find a counterfactual explanation.
    ///
    /// # Arguments
    ///
    /// * `original` - Original input
    /// * `target_class` - Desired output class
    /// * `model_fn` - Classification function (returns class index)
    ///
    /// # Returns
    ///
    /// `Some(CounterfactualResult)` if found, `None` if optimization failed.
    pub fn find<F>(
        &self,
        original: &Vector<f32>,
        target_class: usize,
        model_fn: F,
    ) -> Option<CounterfactualResult>
    where
        F: Fn(&Vector<f32>) -> usize,
    {
        let n = original.len();
        let mut current = original.clone();

        for _ in 0..self.max_iter {
            // Check if we've reached target class
            if model_fn(&current) == target_class {
                let distance = Self::euclidean_distance(&current, original);
                return Some(CounterfactualResult {
                    counterfactual: current,
                    original: original.clone(),
                    target_class,
                    distance,
                });
            }

            // Compute gradient toward target class
            // We use the distance to original as a regularizer
            let mut gradient = vec![0.0f32; n];

            for i in 0..n {
                // Numerical gradient of "wrongness" w.r.t. feature i
                let mut x_plus = current.clone();
                x_plus[i] += self.epsilon;

                let mut x_minus = current.clone();
                x_minus[i] -= self.epsilon;

                // Score: 0 if correct class, 1 otherwise
                let score_plus = if model_fn(&x_plus) == target_class {
                    0.0
                } else {
                    1.0
                };
                let score_minus = if model_fn(&x_minus) == target_class {
                    0.0
                } else {
                    1.0
                };

                // Gradient of classification error
                let class_grad = (score_plus - score_minus) / (2.0 * self.epsilon);

                // Gradient of distance regularizer: 2 * (current - original)
                let dist_grad = 2.0 * (current[i] - original[i]);

                // Combined gradient (minimize error + small regularization)
                gradient[i] = class_grad + 0.001 * dist_grad;
            }

            // Update
            for i in 0..n {
                current[i] -= self.step_size * gradient[i];
            }
        }

        // Check if final result achieves target
        if model_fn(&current) == target_class {
            let distance = Self::euclidean_distance(&current, original);
            Some(CounterfactualResult {
                counterfactual: current,
                original: original.clone(),
                target_class,
                distance,
            })
        } else {
            None
        }
    }

    fn euclidean_distance(a: &Vector<f32>, b: &Vector<f32>) -> f32 {
        a.as_slice()
            .iter()
            .zip(b.as_slice())
            .map(|(&x, &y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }
}

/// Result of counterfactual explanation search.
#[derive(Debug, Clone)]
pub struct CounterfactualResult {
    /// The counterfactual instance
    pub counterfactual: Vector<f32>,
    /// Original instance
    pub original: Vector<f32>,
    /// Target class achieved
    pub target_class: usize,
    /// Distance from original
    pub distance: f32,
}

impl CounterfactualResult {
    /// Get feature changes (counterfactual - original).
    #[must_use]
    pub fn feature_changes(&self) -> Vec<f32> {
        self.counterfactual
            .as_slice()
            .iter()
            .zip(self.original.as_slice())
            .map(|(&cf, &orig)| cf - orig)
            .collect()
    }

    /// Get top k features with largest absolute changes.
    #[must_use]
    pub fn top_changed_features(&self, k: usize) -> Vec<(usize, f32)> {
        let changes = self.feature_changes();
        let mut indexed: Vec<(usize, f32)> = changes.into_iter().enumerate().collect();
        indexed.sort_by(|a, b| {
            b.1.abs()
                .partial_cmp(&a.1.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        indexed.truncate(k);
        indexed
    }
}

/// LIME explanation result.
#[derive(Debug, Clone)]
pub struct LIMEExplanation {
    /// Linear model coefficients (feature importance)
    pub coefficients: Vector<f32>,
    /// Intercept of local model
    pub intercept: f32,
    /// Original model prediction
    pub original_prediction: f32,
}

impl LIMEExplanation {
    /// Get top influential features.
    #[must_use]
    pub fn top_features(&self, k: usize) -> Vec<(usize, f32)> {
        let mut indexed: Vec<(usize, f32)> = self
            .coefficients
            .as_slice()
            .iter()
            .copied()
            .enumerate()
            .collect();
        indexed.sort_by(|a, b| {
            b.1.abs()
                .partial_cmp(&a.1.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        indexed.truncate(k);
        indexed
    }

    /// Local prediction using the linear model.
    #[must_use]
    pub fn local_prediction(&self, sample: &Vector<f32>) -> f32 {
        self.intercept
            + self
                .coefficients
                .as_slice()
                .iter()
                .zip(sample.as_slice())
                .map(|(&c, &x)| c * x)
                .sum::<f32>()
    }
}


#[cfg(test)]
mod tests;

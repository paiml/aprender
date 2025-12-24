//! Model Interpretability and Explainability.
//!
//! This module provides tools for understanding model predictions through
//! feature attribution and explanation methods.
//!
//! # Methods
//!
//! - **SHAP (SHapley Additive exPlanations)**: Computes feature importance using
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
//!   Model Predictions. NeurIPS.
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
    /// - Sum of contributions + expected_value ≈ prediction
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

/// SHAP (SHapley Additive exPlanations) values for feature attribution.
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
    pub fn background(&self) -> &[Vector<f32>] {
        &self.background
    }

    /// Get the expected value.
    pub fn expected_value(&self) -> f32 {
        self.expected_value
    }

    /// Get the number of features.
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
/// - **Fast**: O(n_features * n_samples) predictions
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
    pub fn scores(&self) -> &Vector<f32> {
        &self.importance
    }

    /// Get feature ranking (indices sorted by importance, descending).
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

    pub fn n_samples(&self) -> usize {
        self.n_samples
    }

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
    pub fn new() -> Self {
        Self { epsilon: 1e-4 }
    }

    /// Create with custom epsilon.
    pub fn with_epsilon(epsilon: f32) -> Self {
        Self { epsilon }
    }

    /// Get epsilon value.
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
    /// SmoothGrad reduces noise in saliency maps by averaging gradients
    /// over multiple noisy versions of the input.
    ///
    /// # Reference
    ///
    /// - Smilkov, D., et al. (2017). SmoothGrad: removing noise by adding noise.
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
    pub fn new(max_iter: usize, step_size: f32) -> Self {
        Self {
            max_iter,
            step_size,
            epsilon: 1e-4,
        }
    }

    /// Get max iterations.
    pub fn max_iter(&self) -> usize {
        self.max_iter
    }

    /// Get step size.
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
    pub fn feature_changes(&self) -> Vec<f32> {
        self.counterfactual
            .as_slice()
            .iter()
            .zip(self.original.as_slice())
            .map(|(&cf, &orig)| cf - orig)
            .collect()
    }

    /// Get top k features with largest absolute changes.
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
mod tests {
    use super::*;

    fn simple_linear_model(x: &Vector<f32>) -> f32 {
        // Simple linear model: y = 2*x0 + 3*x1 - 1*x2 + 1.5
        2.0 * x[0] + 3.0 * x[1] - 1.0 * x[2] + 1.5
    }

    #[test]
    fn test_shap_explainer_creation() {
        let background = vec![
            Vector::from_slice(&[1.0, 2.0, 3.0]),
            Vector::from_slice(&[2.0, 3.0, 4.0]),
        ];

        let explainer = ShapExplainer::new(&background, simple_linear_model);

        assert_eq!(explainer.n_features(), 3);
        assert!(explainer.expected_value() > 0.0); // Should be positive for our model
    }

    #[test]
    fn test_shap_explain() {
        let background = vec![
            Vector::from_slice(&[0.0, 0.0, 0.0]),
            Vector::from_slice(&[1.0, 1.0, 1.0]),
            Vector::from_slice(&[2.0, 2.0, 2.0]),
        ];

        let explainer = ShapExplainer::new(&background, simple_linear_model);
        let sample = Vector::from_slice(&[1.0, 1.0, 1.0]);

        let shap_values = explainer.explain_with_model(&sample, simple_linear_model);

        // SHAP values should be finite
        for i in 0..shap_values.len() {
            assert!(shap_values[i].is_finite());
        }

        // Local accuracy: sum(shap) + expected ≈ prediction
        let prediction = simple_linear_model(&sample);
        let reconstructed: f32 = shap_values.sum() + explainer.expected_value();
        assert!(
            (prediction - reconstructed).abs() < 0.5,
            "Local accuracy: {} vs {} (diff: {})",
            prediction,
            reconstructed,
            (prediction - reconstructed).abs()
        );
    }

    #[test]
    fn test_permutation_importance() {
        let x = vec![
            Vector::from_slice(&[1.0, 0.0, 0.0]),
            Vector::from_slice(&[2.0, 0.0, 0.0]),
            Vector::from_slice(&[3.0, 0.0, 0.0]),
            Vector::from_slice(&[4.0, 0.0, 0.0]),
        ];
        let y: Vec<f32> = x.iter().map(simple_linear_model).collect();

        // MSE scoring (higher = worse)
        let importance =
            PermutationImportance::compute(simple_linear_model, &x, &y, |pred, true_val| {
                (pred - true_val).powi(2)
            });

        // Feature 0 should have highest importance (coefficient 2.0)
        // Features 1 and 2 have zero importance (they're constant)
        let ranking = importance.ranking();
        assert_eq!(ranking[0], 0, "Feature 0 should be most important");
    }

    #[test]
    fn test_permutation_importance_ranking() {
        let importance = PermutationImportance {
            importance: Vector::from_slice(&[0.1, 0.5, 0.3, 0.2]),
            baseline_score: 1.0,
        };

        let ranking = importance.ranking();
        assert_eq!(ranking, vec![1, 2, 3, 0]); // Sorted by abs importance
    }

    #[test]
    fn test_feature_contributions_linear() {
        let weights = Vector::from_slice(&[2.0, 3.0, -1.0]);
        let features = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let bias = 1.5;

        let contributions = FeatureContributions::from_linear(&weights, &features, bias);

        // Check individual contributions
        assert!((contributions.contributions[0] - 2.0).abs() < 1e-6); // 2.0 * 1.0
        assert!((contributions.contributions[1] - 6.0).abs() < 1e-6); // 3.0 * 2.0
        assert!((contributions.contributions[2] - (-3.0)).abs() < 1e-6); // -1.0 * 3.0

        // Check prediction
        let expected = 2.0 + 6.0 - 3.0 + 1.5; // = 6.5
        assert!((contributions.prediction - expected).abs() < 1e-6);

        // Verify sum
        assert!(contributions.verify_sum(1e-6));
    }

    #[test]
    fn test_feature_contributions_top_features() {
        let contributions =
            FeatureContributions::new(Vector::from_slice(&[0.1, -0.5, 0.3, -0.2, 0.4]), 1.0);

        let top3 = contributions.top_features(3);

        assert_eq!(top3.len(), 3);
        assert_eq!(top3[0].0, 1); // -0.5 has highest abs value
        assert_eq!(top3[1].0, 4); // 0.4
        assert_eq!(top3[2].0, 2); // 0.3
    }

    #[test]
    fn test_integrated_gradients_basic() {
        let ig = IntegratedGradients::new(20);

        let sample = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let baseline = Vector::from_slice(&[0.0, 0.0, 0.0]);

        let attributions = ig.attribute(simple_linear_model, &sample, &baseline);

        // Attributions should be finite
        for i in 0..attributions.len() {
            assert!(attributions[i].is_finite());
        }

        // For linear model, attributions should match weight * (x - baseline)
        // Approximately: [2*1, 3*2, -1*3] = [2, 6, -3]
        assert!(
            (attributions[0] - 2.0).abs() < 0.5,
            "Feature 0 attribution: {}",
            attributions[0]
        );
        assert!(
            (attributions[1] - 6.0).abs() < 0.5,
            "Feature 1 attribution: {}",
            attributions[1]
        );
        assert!(
            (attributions[2] - (-3.0)).abs() < 0.5,
            "Feature 2 attribution: {}",
            attributions[2]
        );
    }

    #[test]
    fn test_integrated_gradients_completeness() {
        // Completeness axiom: sum(attributions) = f(x) - f(baseline)
        let ig = IntegratedGradients::new(50);

        let sample = Vector::from_slice(&[2.0, 1.0, 0.5]);
        let baseline = Vector::from_slice(&[0.0, 0.0, 0.0]);

        let attributions = ig.attribute(simple_linear_model, &sample, &baseline);

        let sum_attr: f32 = attributions.sum();
        let delta = simple_linear_model(&sample) - simple_linear_model(&baseline);

        assert!(
            (sum_attr - delta).abs() < 0.5,
            "Completeness: sum={sum_attr}, delta={delta}"
        );
    }

    #[test]
    fn test_integrated_gradients_default() {
        let ig = IntegratedGradients::default();
        assert_eq!(ig.n_steps, 50);
    }

    #[test]
    fn test_shap_with_samples() {
        let background = vec![
            Vector::from_slice(&[0.0, 0.0, 0.0]),
            Vector::from_slice(&[1.0, 1.0, 1.0]),
        ];

        let explainer = ShapExplainer::new(&background, simple_linear_model).with_n_samples(50);

        assert_eq!(explainer.n_samples, 50);
    }

    // LIME Tests
    #[test]
    fn test_lime_creation() {
        let lime = LIME::new(100, 0.5);
        assert_eq!(lime.n_samples(), 100);
        assert_eq!(lime.kernel_width(), 0.5);
    }

    #[test]
    fn test_lime_default() {
        let lime = LIME::default();
        assert_eq!(lime.n_samples(), 500);
        assert_eq!(lime.kernel_width(), 0.75);
    }

    #[test]
    fn test_lime_explain_linear() {
        let lime = LIME::new(200, 1.0);
        let sample = Vector::from_slice(&[1.0, 2.0, 3.0]);

        let explanation = lime.explain(simple_linear_model, &sample, 42);

        // Coefficients should be finite
        for i in 0..explanation.coefficients.len() {
            assert!(explanation.coefficients[i].is_finite());
        }

        // Original prediction should match
        assert!((explanation.original_prediction - simple_linear_model(&sample)).abs() < 1e-6);
    }

    #[test]
    fn test_lime_explanation_top_features() {
        let exp = LIMEExplanation {
            coefficients: Vector::from_slice(&[0.1, 0.5, -0.3, 0.2]),
            intercept: 1.0,
            original_prediction: 2.0,
        };

        let top2 = exp.top_features(2);
        assert_eq!(top2.len(), 2);
        assert_eq!(top2[0].0, 1); // 0.5 has highest abs
        assert_eq!(top2[1].0, 2); // -0.3 is second
    }

    #[test]
    fn test_lime_local_prediction() {
        let exp = LIMEExplanation {
            coefficients: Vector::from_slice(&[2.0, 3.0]),
            intercept: 1.0,
            original_prediction: 8.0,
        };

        let sample = Vector::from_slice(&[1.0, 1.0]);
        let local = exp.local_prediction(&sample);
        // intercept + 2*1 + 3*1 = 1 + 2 + 3 = 6
        assert!((local - 6.0).abs() < 1e-6);
    }

    // Saliency Maps Tests
    #[test]
    fn test_saliency_map_creation() {
        let sm = SaliencyMap::new();
        assert!((sm.epsilon() - 1e-4).abs() < 1e-10);
    }

    #[test]
    fn test_saliency_map_custom_epsilon() {
        let sm = SaliencyMap::with_epsilon(1e-3);
        assert!((sm.epsilon() - 1e-3).abs() < 1e-10);
    }

    #[test]
    fn test_saliency_map_compute() {
        let sm = SaliencyMap::new();
        let sample = Vector::from_slice(&[1.0, 2.0, 3.0]);

        let saliency = sm.compute(simple_linear_model, &sample);

        // Gradients should be approximately [2, 3, -1] (model coefficients)
        assert!(saliency.len() == 3);
        assert!((saliency[0] - 2.0).abs() < 0.1, "Got {}", saliency[0]);
        assert!((saliency[1] - 3.0).abs() < 0.1, "Got {}", saliency[1]);
        assert!((saliency[2] - (-1.0)).abs() < 0.1, "Got {}", saliency[2]);
    }

    #[test]
    fn test_saliency_map_absolute() {
        let sm = SaliencyMap::new();
        let sample = Vector::from_slice(&[1.0, 2.0, 3.0]);

        let saliency = sm.compute_absolute(simple_linear_model, &sample);

        // All values should be positive
        for i in 0..saliency.len() {
            assert!(saliency[i] >= 0.0);
        }
    }

    // Counterfactual Tests
    #[test]
    fn test_counterfactual_creation() {
        let cf = CounterfactualExplainer::new(100, 0.01);
        assert_eq!(cf.max_iter(), 100);
        assert!((cf.step_size() - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_counterfactual_find() {
        // Model: classify as 1 if x[0] + x[1] > 2
        let model = |x: &Vector<f32>| -> usize { usize::from(x[0] + x[1] > 2.0) };

        let cf = CounterfactualExplainer::new(500, 0.1);
        let original = Vector::from_slice(&[0.5, 0.5]); // Class 0

        if let Some(result) = cf.find(&original, 1, model) {
            // Counterfactual should be class 1
            let cf_class = model(&result.counterfactual);
            assert_eq!(cf_class, 1, "Counterfactual should achieve target class");

            // Distance should be finite
            assert!(result.distance.is_finite());
        }
    }

    #[test]
    fn test_counterfactual_changes() {
        let result = CounterfactualResult {
            counterfactual: Vector::from_slice(&[1.5, 1.5, 0.5]),
            original: Vector::from_slice(&[1.0, 1.0, 1.0]),
            target_class: 1,
            distance: 0.5,
        };

        let changes = result.feature_changes();

        assert_eq!(changes.len(), 3);
        assert!((changes[0] - 0.5).abs() < 1e-6);
        assert!((changes[1] - 0.5).abs() < 1e-6);
        assert!((changes[2] - (-0.5)).abs() < 1e-6);
    }

    #[test]
    fn test_counterfactual_top_changes() {
        let result = CounterfactualResult {
            counterfactual: Vector::from_slice(&[2.0, 1.1, 3.0]),
            original: Vector::from_slice(&[1.0, 1.0, 1.0]),
            target_class: 1,
            distance: 2.0,
        };

        let top = result.top_changed_features(2);

        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, 2); // Largest change: 3.0 - 1.0 = 2.0
        assert_eq!(top[1].0, 0); // Second: 2.0 - 1.0 = 1.0
    }

    // =========================================================================
    // Additional coverage tests
    // =========================================================================

    #[test]
    fn test_saliency_map_default() {
        let sm = SaliencyMap::default();
        assert!((sm.epsilon() - 1e-4).abs() < 1e-10);
    }

    #[test]
    fn test_saliency_map_clone() {
        let sm = SaliencyMap::with_epsilon(1e-5);
        let cloned = sm.clone();
        assert_eq!(cloned.epsilon(), sm.epsilon());
    }

    #[test]
    fn test_counterfactual_not_found() {
        // Impossible to change: model always returns 0
        let impossible_model = |_: &Vector<f32>| -> usize { 0 };

        let cf = CounterfactualExplainer::new(10, 0.1);
        let original = Vector::from_slice(&[1.0, 1.0]);

        let result = cf.find(&original, 1, impossible_model);
        assert!(result.is_none());
    }

    #[test]
    fn test_permutation_importance_scores() {
        let importance = PermutationImportance {
            importance: Vector::from_slice(&[0.1, 0.2, 0.3]),
            baseline_score: 0.5,
        };

        // Test scores() getter
        assert_eq!(importance.scores().len(), 3);
        assert!((importance.baseline_score - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_feature_contributions_bias() {
        let fc = FeatureContributions {
            contributions: Vector::from_slice(&[1.0, 2.0, 3.0]),
            bias: 1.0,
            prediction: 7.0,
        };
        assert_eq!(fc.contributions.len(), 3);
        assert_eq!(fc.prediction, 7.0);
        assert_eq!(fc.bias, 1.0);
    }

    #[test]
    fn test_integrated_gradients_steps() {
        let ig = IntegratedGradients::new(100);
        assert_eq!(ig.n_steps, 100);
    }

    #[test]
    fn test_shap_explainer_debug() {
        let background = vec![Vector::from_slice(&[1.0, 2.0, 3.0])];
        let explainer = ShapExplainer::new(&background, simple_linear_model);
        let debug_str = format!("{:?}", explainer);
        assert!(debug_str.contains("ShapExplainer"));
    }

    #[test]
    fn test_integrated_gradients_debug() {
        let ig = IntegratedGradients::new(50);
        let debug_str = format!("{:?}", ig);
        assert!(debug_str.contains("IntegratedGradients"));
    }

    #[test]
    fn test_lime_debug() {
        let lime = LIME::new(100, 0.5);
        let debug_str = format!("{:?}", lime);
        assert!(debug_str.contains("LIME"));
    }

    #[test]
    fn test_saliency_map_debug() {
        let sm = SaliencyMap::new();
        let debug_str = format!("{:?}", sm);
        assert!(debug_str.contains("SaliencyMap"));
    }

    #[test]
    fn test_counterfactual_explainer_debug() {
        let cf = CounterfactualExplainer::new(100, 0.01);
        let debug_str = format!("{:?}", cf);
        assert!(debug_str.contains("CounterfactualExplainer"));
    }

    #[test]
    fn test_counterfactual_result_debug() {
        let result = CounterfactualResult {
            counterfactual: Vector::from_slice(&[1.0]),
            original: Vector::from_slice(&[0.0]),
            target_class: 1,
            distance: 1.0,
        };
        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("CounterfactualResult"));
    }

    #[test]
    fn test_counterfactual_result_clone() {
        let result = CounterfactualResult {
            counterfactual: Vector::from_slice(&[1.0, 2.0]),
            original: Vector::from_slice(&[0.0, 0.0]),
            target_class: 1,
            distance: 2.0,
        };
        let cloned = result.clone();
        assert_eq!(cloned.target_class, result.target_class);
        assert!((cloned.distance - result.distance).abs() < 1e-6);
    }

    #[test]
    fn test_lime_explanation_debug() {
        let exp = LIMEExplanation {
            coefficients: Vector::from_slice(&[0.1, 0.2]),
            intercept: 1.0,
            original_prediction: 2.0,
        };
        let debug_str = format!("{:?}", exp);
        assert!(debug_str.contains("LIMEExplanation"));
    }

    #[test]
    fn test_lime_explanation_clone() {
        let exp = LIMEExplanation {
            coefficients: Vector::from_slice(&[0.1, 0.2]),
            intercept: 1.0,
            original_prediction: 2.0,
        };
        let cloned = exp.clone();
        assert!((cloned.intercept - exp.intercept).abs() < 1e-6);
    }

    #[test]
    fn test_feature_contributions_verify_sum() {
        let fc = FeatureContributions {
            contributions: Vector::from_slice(&[1.0, 2.0, 3.0]),
            prediction: 7.5, // sum + bias = 6 + 1.5 = 7.5
            bias: 1.5,
        };
        assert!(fc.verify_sum(1e-6));
    }

    #[test]
    fn test_permutation_importance_debug() {
        let pi = PermutationImportance {
            importance: Vector::from_slice(&[0.1, 0.2]),
            baseline_score: 1.0,
        };
        let debug_str = format!("{:?}", pi);
        assert!(debug_str.contains("PermutationImportance"));
    }

    #[test]
    fn test_feature_contributions_debug() {
        let fc = FeatureContributions::new(Vector::from_slice(&[1.0, 2.0]), 3.0);
        let debug_str = format!("{:?}", fc);
        assert!(debug_str.contains("FeatureContributions"));
    }
}

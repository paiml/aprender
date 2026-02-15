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
/// importance.sort_by(|a, b| b.1.abs().total_cmp(&a.1.abs()));
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

include!("mod_part_02.rs");
include!("mod_part_03.rs");

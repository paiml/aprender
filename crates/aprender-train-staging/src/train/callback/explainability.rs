//! Explainability callback for computing feature attributions during training

use super::traits::{CallbackAction, CallbackContext, TrainerCallback};

/// Method for computing feature attributions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExplainMethod {
    /// Permutation importance - fast, model-agnostic
    PermutationImportance,
    /// Integrated gradients - for differentiable models
    IntegratedGradients,
    /// Saliency maps - gradient-based attribution
    Saliency,
}

/// Feature importance result for a single epoch
#[derive(Debug, Clone)]
pub struct FeatureImportanceResult {
    /// Epoch when computed
    pub epoch: usize,
    /// Feature index to importance score
    pub importances: Vec<(usize, f32)>,
    /// Method used
    pub method: ExplainMethod,
}

/// Callback for computing feature attributions during training
///
/// Integrates with aprender's interpret module to provide explainability
/// insights during model evaluation.
///
/// # Example
///
/// ```ignore
/// use entrenar::train::{ExplainabilityCallback, ExplainMethod};
///
/// let callback = ExplainabilityCallback::new(ExplainMethod::PermutationImportance)
///     .with_top_k(5)
///     .with_eval_samples(100);
/// ```
#[derive(Debug)]
pub struct ExplainabilityCallback {
    method: ExplainMethod,
    top_k: usize,
    eval_samples: usize,
    results: Vec<FeatureImportanceResult>,
    feature_names: Option<Vec<String>>,
}

impl ExplainabilityCallback {
    /// Create new explainability callback
    ///
    /// # Arguments
    ///
    /// * `method` - Attribution method to use
    pub fn new(method: ExplainMethod) -> Self {
        Self { method, top_k: 10, eval_samples: 50, results: Vec::new(), feature_names: None }
    }

    /// Set number of top features to track
    pub fn with_top_k(mut self, k: usize) -> Self {
        self.top_k = k;
        self
    }

    /// Set number of samples to use for evaluation
    pub fn with_eval_samples(mut self, n: usize) -> Self {
        self.eval_samples = n;
        self
    }

    /// Set feature names for interpretability
    pub fn with_feature_names(mut self, names: Vec<String>) -> Self {
        self.feature_names = Some(names);
        self
    }

    /// Get attribution method
    pub fn method(&self) -> ExplainMethod {
        self.method
    }

    /// Get top-k setting
    pub fn top_k(&self) -> usize {
        self.top_k
    }

    /// Get eval samples setting
    pub fn eval_samples(&self) -> usize {
        self.eval_samples
    }

    /// Get all computed results
    pub fn results(&self) -> &[FeatureImportanceResult] {
        &self.results
    }

    /// Get feature names if set
    pub fn feature_names(&self) -> Option<&[String]> {
        self.feature_names.as_deref()
    }

    /// Record feature importances for an epoch
    ///
    /// Call this during on_epoch_end with computed importances
    pub fn record_importances(&mut self, epoch: usize, importances: Vec<(usize, f32)>) {
        let mut sorted = importances;
        sorted
            .sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap_or(std::cmp::Ordering::Equal));
        sorted.truncate(self.top_k);

        self.results.push(FeatureImportanceResult {
            epoch,
            importances: sorted,
            method: self.method,
        });
    }

    /// Compute permutation importance using aprender
    ///
    /// # Arguments
    ///
    /// * `predict_fn` - Model prediction function
    /// * `x` - Feature vectors
    /// * `y` - Target values
    pub fn compute_permutation_importance<P>(
        &self,
        predict_fn: P,
        x: &[aprender::primitives::Vector<f32>],
        y: &[f32],
    ) -> Vec<(usize, f32)>
    where
        P: Fn(&aprender::primitives::Vector<f32>) -> f32,
    {
        let importance = aprender::interpret::PermutationImportance::compute(
            predict_fn,
            x,
            y,
            |pred, true_val| (pred - true_val).powi(2), // MSE
        );

        importance.scores().as_slice().iter().enumerate().map(|(i, &v)| (i, v)).collect()
    }

    /// Compute integrated gradients using aprender
    ///
    /// # Arguments
    ///
    /// * `model_fn` - Model prediction function
    /// * `sample` - Input sample to explain
    /// * `baseline` - Baseline input (typically zeros)
    pub fn compute_integrated_gradients<F>(
        &self,
        model_fn: F,
        sample: &aprender::primitives::Vector<f32>,
        baseline: &aprender::primitives::Vector<f32>,
    ) -> Vec<(usize, f32)>
    where
        F: Fn(&aprender::primitives::Vector<f32>) -> f32,
    {
        let ig = aprender::interpret::IntegratedGradients::default();
        let attributions = ig.attribute(model_fn, sample, baseline);

        attributions.as_slice().iter().enumerate().map(|(i, &v)| (i, v)).collect()
    }

    /// Compute saliency map using aprender
    ///
    /// # Arguments
    ///
    /// * `model_fn` - Model prediction function
    /// * `sample` - Input sample to explain
    pub fn compute_saliency<F>(
        &self,
        model_fn: F,
        sample: &aprender::primitives::Vector<f32>,
    ) -> Vec<(usize, f32)>
    where
        F: Fn(&aprender::primitives::Vector<f32>) -> f32,
    {
        let sm = aprender::interpret::SaliencyMap::default();
        let saliency = sm.compute(model_fn, sample);

        saliency.as_slice().iter().enumerate().map(|(i, &v)| (i, v)).collect()
    }

    /// Get top features that have been consistently important across epochs
    pub fn consistent_top_features(&self) -> Vec<(usize, f32)> {
        if self.results.is_empty() {
            return Vec::new();
        }

        // Count frequency of each feature in top-k across epochs
        let mut freq: std::collections::HashMap<usize, (usize, f32)> =
            std::collections::HashMap::new();

        for result in &self.results {
            for (idx, score) in &result.importances {
                let entry = freq.entry(*idx).or_insert((0, 0.0));
                entry.0 += 1;
                entry.1 += score.abs();
            }
        }

        // Average score and sort by frequency then score
        let mut features: Vec<_> = freq
            .into_iter()
            .map(|(idx, (count, total))| (idx, total / count as f32, count))
            .collect();

        features.sort_by(|a, b| {
            b.2.cmp(&a.2).then_with(|| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal))
        });

        features.into_iter().take(self.top_k).map(|(idx, avg_score, _)| (idx, avg_score)).collect()
    }
}

impl TrainerCallback for ExplainabilityCallback {
    fn on_epoch_end(&mut self, ctx: &CallbackContext) -> CallbackAction {
        // Note: Actual computation requires model and data access
        // This callback stores configuration and results
        // Users should call compute_* methods and record_importances externally
        let _ = ctx; // Acknowledge context
        CallbackAction::Continue
    }

    fn name(&self) -> &'static str {
        "ExplainabilityCallback"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_explainability_callback_creation() {
        let cb = ExplainabilityCallback::new(ExplainMethod::PermutationImportance);
        assert_eq!(cb.method(), ExplainMethod::PermutationImportance);
        assert_eq!(cb.top_k(), 10); // Default
        assert_eq!(cb.eval_samples(), 50); // Default
        assert!(cb.results().is_empty());
    }

    #[test]
    fn test_explainability_callback_builder() {
        let cb = ExplainabilityCallback::new(ExplainMethod::IntegratedGradients)
            .with_top_k(5)
            .with_eval_samples(100)
            .with_feature_names(vec!["f1".to_string(), "f2".to_string()]);

        assert_eq!(cb.method(), ExplainMethod::IntegratedGradients);
        assert_eq!(cb.top_k(), 5);
        assert_eq!(cb.eval_samples(), 100);
        assert_eq!(cb.feature_names(), Some(&["f1".to_string(), "f2".to_string()][..]));
    }

    #[test]
    fn test_explainability_callback_record_importances() {
        let mut cb = ExplainabilityCallback::new(ExplainMethod::Saliency).with_top_k(3);

        // Record importances for epoch 0
        let importances = vec![(0, 0.5), (1, 0.3), (2, 0.8), (3, 0.1), (4, 0.6)];
        cb.record_importances(0, importances);

        assert_eq!(cb.results().len(), 1);
        let result = &cb.results()[0];
        assert_eq!(result.epoch, 0);
        assert_eq!(result.method, ExplainMethod::Saliency);
        assert_eq!(result.importances.len(), 3); // Top 3

        // Should be sorted by absolute value descending
        assert_eq!(result.importances[0].0, 2); // 0.8
        assert_eq!(result.importances[1].0, 4); // 0.6
        assert_eq!(result.importances[2].0, 0); // 0.5
    }

    #[test]
    fn test_explainability_callback_consistent_features() {
        let mut cb =
            ExplainabilityCallback::new(ExplainMethod::PermutationImportance).with_top_k(2);

        // Epoch 0: features 0 and 1 are important
        cb.record_importances(0, vec![(0, 0.8), (1, 0.6), (2, 0.1)]);
        // Epoch 1: features 0 and 2 are important
        cb.record_importances(1, vec![(0, 0.7), (2, 0.5), (1, 0.2)]);
        // Epoch 2: feature 0 is important again
        cb.record_importances(2, vec![(0, 0.9), (1, 0.4), (2, 0.3)]);

        let consistent = cb.consistent_top_features();
        // Feature 0 appears in all epochs, should be first
        assert!(!consistent.is_empty());
        assert_eq!(consistent[0].0, 0);
    }

    #[test]
    fn test_explainability_callback_trainer_callback_impl() {
        let mut cb = ExplainabilityCallback::new(ExplainMethod::PermutationImportance);
        let ctx = CallbackContext::default();

        // Should always continue (doesn't auto-compute)
        assert_eq!(cb.on_epoch_end(&ctx), CallbackAction::Continue);
        assert_eq!(cb.name(), "ExplainabilityCallback");
    }

    #[test]
    fn test_explain_method_enum() {
        // Test all variants are distinct
        assert_ne!(ExplainMethod::PermutationImportance, ExplainMethod::IntegratedGradients);
        assert_ne!(ExplainMethod::IntegratedGradients, ExplainMethod::Saliency);
        assert_ne!(ExplainMethod::Saliency, ExplainMethod::PermutationImportance);

        // Test Clone and Copy
        let method = ExplainMethod::Saliency;
        let cloned = method;
        assert_eq!(method, cloned);
    }

    #[test]
    fn test_feature_importance_result_fields() {
        let result = FeatureImportanceResult {
            epoch: 5,
            importances: vec![(0, 0.9), (1, 0.7)],
            method: ExplainMethod::IntegratedGradients,
        };

        assert_eq!(result.epoch, 5);
        assert_eq!(result.importances.len(), 2);
        assert_eq!(result.method, ExplainMethod::IntegratedGradients);
    }

    #[test]
    fn test_explainability_empty_results() {
        let cb = ExplainabilityCallback::new(ExplainMethod::Saliency);
        assert!(cb.consistent_top_features().is_empty());
    }

    #[test]
    fn test_explainability_feature_names_none() {
        let cb = ExplainabilityCallback::new(ExplainMethod::Saliency);
        assert!(cb.feature_names().is_none());
    }

    #[test]
    fn test_explainability_record_importances_negative() {
        let mut cb = ExplainabilityCallback::new(ExplainMethod::Saliency).with_top_k(2);
        let importances = vec![(0, -0.9), (1, 0.5), (2, -0.3)];
        cb.record_importances(0, importances);
        let result = &cb.results()[0];
        assert_eq!(result.importances[0].0, 0);
        assert_eq!(result.importances[1].0, 1);
    }

    #[test]
    fn test_explainability_callback_basic() {
        let mut cb = ExplainabilityCallback::new(ExplainMethod::PermutationImportance);
        assert_eq!(cb.name(), "ExplainabilityCallback");

        let mut ctx = CallbackContext::default();
        ctx.step = 5;
        ctx.loss = 0.5;

        cb.on_step_end(&ctx);
        // Should have recorded something
    }

    #[test]
    fn test_explainability_compute_permutation_importance() {
        let cb = ExplainabilityCallback::new(ExplainMethod::PermutationImportance);

        // Create sample data using aprender's Vector type
        let x = vec![
            aprender::primitives::Vector::from_slice(&[1.0, 2.0, 3.0]),
            aprender::primitives::Vector::from_slice(&[4.0, 5.0, 6.0]),
            aprender::primitives::Vector::from_slice(&[7.0, 8.0, 9.0]),
        ];
        let y = vec![1.0, 2.0, 3.0];

        // Simple linear prediction function
        let predict_fn = |v: &aprender::primitives::Vector<f32>| -> f32 {
            v.as_slice()[0] * 0.1 + v.as_slice()[1] * 0.2
        };

        let importance = cb.compute_permutation_importance(predict_fn, &x, &y);
        assert_eq!(importance.len(), 3);
    }

    #[test]
    fn test_explainability_compute_integrated_gradients() {
        let cb = ExplainabilityCallback::new(ExplainMethod::IntegratedGradients);

        let sample = aprender::primitives::Vector::from_slice(&[1.0, 2.0, 3.0]);
        let baseline = aprender::primitives::Vector::from_slice(&[0.0, 0.0, 0.0]);

        let model_fn =
            |v: &aprender::primitives::Vector<f32>| -> f32 { v.as_slice().iter().sum::<f32>() };

        let attributions = cb.compute_integrated_gradients(model_fn, &sample, &baseline);
        assert_eq!(attributions.len(), 3);
    }

    #[test]
    fn test_explainability_compute_saliency() {
        let cb = ExplainabilityCallback::new(ExplainMethod::Saliency);

        let sample = aprender::primitives::Vector::from_slice(&[1.0, 2.0, 3.0]);

        let model_fn =
            |v: &aprender::primitives::Vector<f32>| -> f32 { v.as_slice().iter().sum::<f32>() };

        let saliency = cb.compute_saliency(model_fn, &sample);
        assert_eq!(saliency.len(), 3);
    }
}

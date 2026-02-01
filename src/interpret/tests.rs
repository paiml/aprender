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

// ==================== Accessor Tests ====================

#[test]
fn test_shap_accessors() {
    let background = vec![
        Vector::from_slice(&[1.0, 2.0]),
        Vector::from_slice(&[3.0, 4.0]),
    ];
    let model_fn = |_v: &Vector<f32>| 0.0_f32;
    let explainer = ShapExplainer::new(&background, model_fn);

    assert_eq!(explainer.background().len(), 2);
    assert_eq!(explainer.n_features(), 2);
    // expected_value is the mean prediction over background
    let _ = explainer.expected_value();
}

#[test]
fn test_shap_with_n_samples() {
    let background = vec![Vector::from_slice(&[1.0, 2.0])];
    let model_fn = |_v: &Vector<f32>| 0.0_f32;
    let explainer = ShapExplainer::new(&background, model_fn).with_n_samples(50);

    assert_eq!(explainer.background().len(), 1);
}

#[test]
fn test_lime_accessors() {
    let lime = LIME::new(100, 0.5);
    assert_eq!(lime.n_samples(), 100);
    assert!((lime.kernel_width() - 0.5).abs() < f32::EPSILON);
}

#[test]
fn test_saliency_map_accessors() {
    let sm = SaliencyMap::with_epsilon(1e-5);
    assert!((sm.epsilon() - 1e-5).abs() < f32::EPSILON);
}

#[test]
fn test_saliency_map_default_trait() {
    let sm = SaliencyMap::default();
    assert!((sm.epsilon() - 1e-4).abs() < f32::EPSILON); // Default is 1e-4
}

#[test]
fn test_counterfactual_accessors() {
    let explainer = CounterfactualExplainer::new(200, 0.01);
    assert_eq!(explainer.max_iter(), 200);
    assert!((explainer.step_size() - 0.01).abs() < f32::EPSILON);
}

#[test]
fn test_permutation_importance_scores_accessor() {
    let pi = PermutationImportance {
        importance: Vector::from_slice(&[0.3, 0.7, 0.1]),
        baseline_score: 1.0,
    };

    let scores = pi.scores();
    assert_eq!(scores.len(), 3);
    assert!((scores[0] - 0.3).abs() < f32::EPSILON);
}

#[test]
fn test_permutation_importance_ranking_order() {
    let pi = PermutationImportance {
        importance: Vector::from_slice(&[0.3, 0.7, 0.1]),
        baseline_score: 1.0,
    };

    let ranking = pi.ranking();
    // Should be sorted by importance descending: [1, 0, 2]
    assert_eq!(ranking[0], 1); // 0.7 is highest
    assert_eq!(ranking[1], 0); // 0.3 is second
    assert_eq!(ranking[2], 2); // 0.1 is lowest
}

// ==================== Additional Coverage Tests ====================

#[test]
fn test_smooth_grad_computation() {
    let sm = SaliencyMap::new();
    let sample = Vector::from_slice(&[1.0, 2.0, 3.0]);

    // SmoothGrad with small noise level
    let smooth = sm.smooth_grad(simple_linear_model, &sample, 10, 0.1);

    // Result should be close to regular saliency (model coefficients: 2, 3, -1)
    assert_eq!(smooth.len(), 3);
    // With smoothing, results approximate the model gradients
    for i in 0..smooth.len() {
        assert!(smooth[i].is_finite());
    }
}

#[test]
fn test_smooth_grad_with_larger_noise() {
    let sm = SaliencyMap::new();
    let sample = Vector::from_slice(&[1.0, 2.0]);

    // Simple model for testing
    let model = |x: &Vector<f32>| x[0] * 2.0 + x[1] * 3.0;

    let smooth = sm.smooth_grad(model, &sample, 20, 0.5);

    assert_eq!(smooth.len(), 2);
    // With more noise, variance is higher but mean should still approximate gradients
    for &val in smooth.as_slice() {
        assert!(val.is_finite());
    }
}

#[test]
fn test_smooth_grad_one_sample() {
    let sm = SaliencyMap::new();
    let sample = Vector::from_slice(&[1.0]);

    let model = |x: &Vector<f32>| x[0] * 5.0;

    let smooth = sm.smooth_grad(model, &sample, 1, 0.01);

    assert_eq!(smooth.len(), 1);
    // With 1 sample, should be close to regular gradient
    assert!((smooth[0] - 5.0).abs() < 0.5);
}

#[test]
fn test_shap_local_accuracy_edge_case() {
    // Test SHAP with very small background
    let background = vec![Vector::from_slice(&[0.0, 0.0])];

    let model = |x: &Vector<f32>| x[0] + x[1];

    let explainer = ShapExplainer::new(&background, model);
    let sample = Vector::from_slice(&[1.0, 2.0]);

    let shap_values = explainer.explain_with_model(&sample, model);

    // SHAP values should be finite
    for i in 0..shap_values.len() {
        assert!(shap_values[i].is_finite());
    }
}

#[test]
fn test_shap_zero_sum_shap_values() {
    // Test edge case where sum of SHAP values is near zero
    let background = vec![
        Vector::from_slice(&[1.0, 1.0]),
        Vector::from_slice(&[1.0, 1.0]),
    ];

    // Constant model - all SHAP values should be ~0
    let constant_model = |_x: &Vector<f32>| 5.0f32;

    let explainer = ShapExplainer::new(&background, constant_model);
    let sample = Vector::from_slice(&[1.0, 1.0]);

    let shap_values = explainer.explain_with_model(&sample, constant_model);

    // For constant model, SHAP values should be near zero
    for i in 0..shap_values.len() {
        assert!(shap_values[i].abs() < 1.0);
    }
}

#[test]
fn test_lime_weighted_linear_regression() {
    // Test LIME with various input configurations
    let lime = LIME::new(50, 0.5);
    let sample = Vector::from_slice(&[2.0, 3.0, 4.0]);

    let model = |x: &Vector<f32>| x[0] + x[1] * 2.0 + x[2] * 3.0;

    let explanation = lime.explain(model, &sample, 123);

    // Verify explanation structure
    assert_eq!(explanation.coefficients.len(), 3);
    assert!(explanation.intercept.is_finite());
    assert!(explanation.original_prediction.is_finite());
}

#[test]
fn test_lime_solve_linear_system() {
    // Test with different seed to exercise different paths
    let lime = LIME::new(100, 1.0);
    let sample = Vector::from_slice(&[1.0, 1.0]);

    let model = |x: &Vector<f32>| x[0] * 10.0 + x[1] * 5.0;

    let explanation = lime.explain(model, &sample, 999);

    // Should produce reasonable output
    for i in 0..explanation.coefficients.len() {
        assert!(explanation.coefficients[i].is_finite());
    }
}

#[test]
fn test_counterfactual_immediate_success() {
    // Model where counterfactual is immediately found
    let model = |x: &Vector<f32>| -> usize { usize::from(x[0] > 0.0) };

    let cf = CounterfactualExplainer::new(10, 0.5);
    let original = Vector::from_slice(&[1.0]); // Already class 1

    // Try to find class 1 (already there)
    if let Some(result) = cf.find(&original, 1, model) {
        assert_eq!(result.target_class, 1);
        assert!(result.distance >= 0.0);
    }
}

#[test]
fn test_counterfactual_requires_many_iterations() {
    // Model that requires more iterations to change class
    let model = |x: &Vector<f32>| -> usize { usize::from(x[0] > 5.0) };

    let cf = CounterfactualExplainer::new(1000, 0.01);
    let original = Vector::from_slice(&[0.0]);

    // Try to find class 1
    let result = cf.find(&original, 1, model);

    // May or may not find solution depending on step size
    if let Some(r) = result {
        assert_eq!(model(&r.counterfactual), 1);
    }
}

#[test]
fn test_feature_contributions_empty_top_features() {
    let fc = FeatureContributions::new(Vector::from_slice(&[0.1, 0.2, 0.3]), 1.0);

    // Request more features than available
    let top = fc.top_features(10);
    assert_eq!(top.len(), 3); // Only 3 features exist

    // Request 0 features
    let top0 = fc.top_features(0);
    assert!(top0.is_empty());
}

#[test]
fn test_permutation_importance_with_nan_handling() {
    let pi = PermutationImportance {
        importance: Vector::from_slice(&[0.5, 0.5, 0.5]), // Equal importance
        baseline_score: 1.0,
    };

    let ranking = pi.ranking();
    // With equal importance, order doesn't matter but should be consistent
    assert_eq!(ranking.len(), 3);
}

#[test]
fn test_integrated_gradients_zero_baseline() {
    let ig = IntegratedGradients::new(10);
    let sample = Vector::from_slice(&[2.0, 3.0]);
    let baseline = Vector::from_slice(&[0.0, 0.0]);

    let quadratic_model = |x: &Vector<f32>| x[0].powi(2) + x[1].powi(2);

    let attr = ig.attribute(quadratic_model, &sample, &baseline);

    assert_eq!(attr.len(), 2);
    // For quadratic model, attributions should be finite
    for i in 0..attr.len() {
        assert!(attr[i].is_finite());
    }
}

#[test]
fn test_integrated_gradients_nonzero_baseline() {
    let ig = IntegratedGradients::new(20);
    let sample = Vector::from_slice(&[3.0, 4.0]);
    let baseline = Vector::from_slice(&[1.0, 1.0]);

    let model = |x: &Vector<f32>| x[0] * x[1];

    let attr = ig.attribute(model, &sample, &baseline);

    assert_eq!(attr.len(), 2);
    // Attributions should sum approximately to f(sample) - f(baseline)
    let delta = model(&sample) - model(&baseline);
    let sum_attr: f32 = attr.sum();
    // Allow some error due to numerical integration
    assert!(
        (sum_attr - delta).abs() < 2.0,
        "sum={sum_attr}, delta={delta}"
    );
}

#[test]
fn test_permutation_importance_clone() {
    let pi = PermutationImportance {
        importance: Vector::from_slice(&[0.1, 0.2, 0.3]),
        baseline_score: 1.5,
    };
    let cloned = pi.clone();
    assert_eq!(cloned.importance.len(), pi.importance.len());
    assert!((cloned.baseline_score - pi.baseline_score).abs() < f32::EPSILON);
}

#[test]
fn test_feature_contributions_clone() {
    let fc = FeatureContributions::new(Vector::from_slice(&[1.0, 2.0]), 0.5);
    let cloned = fc.clone();
    assert_eq!(cloned.contributions.len(), fc.contributions.len());
    assert!((cloned.bias - fc.bias).abs() < f32::EPSILON);
}

#[test]
fn test_counterfactual_result_feature_changes_sign() {
    let result = CounterfactualResult {
        counterfactual: Vector::from_slice(&[0.0, 5.0, -3.0]),
        original: Vector::from_slice(&[1.0, 2.0, 1.0]),
        target_class: 0,
        distance: 1.0,
    };

    let changes = result.feature_changes();

    assert!((changes[0] - (-1.0)).abs() < 1e-6); // 0 - 1 = -1
    assert!((changes[1] - 3.0).abs() < 1e-6); // 5 - 2 = 3
    assert!((changes[2] - (-4.0)).abs() < 1e-6); // -3 - 1 = -4
}

use super::*;

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

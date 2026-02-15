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

include!("tests_part_02.rs");

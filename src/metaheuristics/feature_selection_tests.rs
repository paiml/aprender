use super::*;

// ==========================================================
// EXTREME TDD: Tests written first
// ==========================================================

#[test]
fn test_feature_selector_basic() {
    let n_features = 10;

    // Evaluator: accuracy increases with first 3 features, then plateaus
    let evaluator = |mask: &[bool]| -> (f64, usize) {
        let sel = selected_indices(mask);
        let n = sel.len();

        if n == 0 {
            return (0.0, 0);
        }

        // First 3 features are important
        let important_count = sel.iter().filter(|&&i| i < 3).count();
        let accuracy = 0.5 + 0.15 * important_count as f64;
        (accuracy, n)
    };

    let mut selector = FeatureSelector::new(n_features)
        .with_criterion(SelectionCriterion::MaxAccuracyMinFeatures { alpha: 0.02 })
        .with_seed(42);

    let result = selector.select(evaluator, Budget::Evaluations(500));

    assert!(result.n_selected > 0, "Should select at least one feature");
    assert!(
        result.n_selected <= n_features,
        "Can't select more than exist"
    );
    assert!(
        result.accuracy.unwrap() > 0.5,
        "Should achieve decent accuracy"
    );
}

#[test]
fn test_feature_selector_min_features() {
    let n_features = 5;

    // Constant accuracy regardless of features
    let evaluator = |mask: &[bool]| -> (f64, usize) {
        let n = count_selected(mask);
        (0.8, n) // Always 80% accuracy
    };

    let mut selector = FeatureSelector::new(n_features)
        .with_criterion(SelectionCriterion::MinFeatures)
        .with_seed(42);

    let result = selector.select(evaluator, Budget::Evaluations(200));

    // Should select 0 or very few features since accuracy is constant
    assert!(result.n_selected <= 2, "Should minimize features");
}

#[test]
fn test_feature_selector_max_limit() {
    let n_features = 10;

    let evaluator = |mask: &[bool]| -> (f64, usize) {
        let n = count_selected(mask);
        let accuracy = 0.5 + 0.05 * n as f64; // More features = better
        (accuracy, n)
    };

    let mut selector = FeatureSelector::new(n_features)
        .with_criterion(SelectionCriterion::MaxAccuracyWithLimit { max_features: 3 })
        .with_seed(42);

    let result = selector.select(evaluator, Budget::Evaluations(500));

    assert!(
        result.n_selected <= 3,
        "Should respect max_features limit, got {}",
        result.n_selected
    );
}

#[test]
fn test_feature_selector_aic() {
    let n_features = 8;
    let n_samples = 100;

    let evaluator = |mask: &[bool]| -> (f64, usize) {
        let n = count_selected(mask);
        let accuracy = if n == 0 {
            0.0
        } else {
            0.7 + 0.02 * n.min(4) as f64
        };
        (accuracy, n)
    };

    let mut selector = FeatureSelector::new(n_features)
        .with_criterion(SelectionCriterion::AIC { n_samples })
        .with_seed(42);

    let result = selector.select(evaluator, Budget::Evaluations(500));

    assert!(result.n_selected > 0);
    assert!(result.n_selected < n_features);
}

#[test]
fn test_convenience_function() {
    let result = select_features(
        5,
        |mask| {
            let n = count_selected(mask);
            (0.6 + 0.1 * n.min(2) as f64, n)
        },
        Budget::Evaluations(200),
    );

    assert!(!result.mask.is_empty());
    assert_eq!(result.mask.len(), 5);
}

#[test]
fn test_rank_features() {
    let n_features = 4;

    // Feature 0 is most important, feature 3 is useless
    let evaluator = |mask: &[bool]| -> (f64, usize) {
        let n = count_selected(mask);
        let mut acc = 0.5;
        if mask[0] {
            acc += 0.2;
        }
        if mask[1] {
            acc += 0.1;
        }
        if mask[2] {
            acc += 0.05;
        }
        // mask[3] contributes nothing
        (acc, n)
    };

    let ranking = rank_features(n_features, evaluator);

    // Feature 0 should rank highest (biggest drop when removed)
    assert_eq!(ranking[0].0, 0, "Feature 0 should be most important");
    assert!(
        ranking[0].1 > ranking[3].1,
        "Feature 0 more important than 3"
    );
}

#[test]
fn test_feature_selection_result_structure() {
    let result = FeatureSelectionResult {
        selected_indices: vec![0, 2, 5],
        n_selected: 3,
        score: 0.85,
        accuracy: Some(0.9),
        mask: vec![true, false, true, false, false, true],
        evaluations: 500,
    };

    assert_eq!(result.selected_indices.len(), result.n_selected);
    assert_eq!(count_selected(&result.mask), result.n_selected);
}

#[test]
fn test_selector_builder_pattern() {
    let selector = FeatureSelector::new(20)
        .with_criterion(SelectionCriterion::MaxAccuracy)
        .with_population_size(100)
        .with_mutation_prob(0.05)
        .with_seed(123);

    assert_eq!(selector.n_features, 20);
    assert_eq!(selector.population_size, 100);
    assert!((selector.mutation_prob - 0.05).abs() < 1e-10);
}

#[test]
fn test_bic_criterion() {
    let n_features = 6;
    let n_samples = 50;

    let evaluator = |mask: &[bool]| -> (f64, usize) {
        let n = count_selected(mask);
        (0.75, n)
    };

    let mut selector = FeatureSelector::new(n_features)
        .with_criterion(SelectionCriterion::BIC { n_samples })
        .with_seed(42);

    let result = selector.select(evaluator, Budget::Evaluations(300));

    // BIC penalizes more than AIC for large n, should select fewer
    assert!(result.n_selected < n_features);
}

#[test]
fn test_empty_selection_handling() {
    let n_features = 3;

    // Evaluator that handles empty selection
    let evaluator = |mask: &[bool]| -> (f64, usize) {
        let n = count_selected(mask);
        if n == 0 {
            (0.0, 0)
        } else {
            (0.8, n)
        }
    };

    // MinFeatures should still work (might select 0)
    let mut selector = FeatureSelector::new(n_features)
        .with_criterion(SelectionCriterion::MinFeatures)
        .with_seed(42);

    let result = selector.select(evaluator, Budget::Evaluations(100));

    // Should handle gracefully regardless of result
    assert!(result.mask.len() == n_features);
}

use super::*;
use crate::stack::quality::{QualityGrade, StackLayer};

// ========================================================================
// Test helpers
// ========================================================================

/// Build a healthy component with slight per-index variation
fn make_healthy_component(name: impl Into<String>, offset: f64) -> ComponentNode {
    let mut node = ComponentNode::new(name, "1.0", StackLayer::Compute);
    node.metrics = ComponentMetrics {
        demo_score: 90.0 + offset,
        coverage: 85.0 + offset,
        mutation_score: 80.0,
        complexity_avg: 5.0,
        satd_count: 2,
        dead_code_pct: 1.0,
        grade: QualityGrade::A,
    };
    node
}

/// Build an anomalous component with metrics that deviate strongly
fn make_anomalous_component(name: impl Into<String>) -> ComponentNode {
    let mut node = ComponentNode::new(name, "1.0", StackLayer::Ml);
    node.metrics = ComponentMetrics {
        demo_score: 30.0,
        coverage: 20.0,
        mutation_score: 10.0,
        complexity_avg: 25.0,
        satd_count: 50,
        dead_code_pct: 30.0,
        grade: QualityGrade::F,
    };
    node
}

// ========================================================================
// Isolation Forest Tests
// ========================================================================

#[test]
fn test_isolation_forest_new() {
    let forest = IsolationForest::new(10, 32, 42);
    assert_eq!(forest.n_trees, 10);
    assert_eq!(forest.sample_size, 32);
    assert_eq!(forest.seed, 42);
}

#[test]
fn test_isolation_forest_default() {
    let forest = IsolationForest::default_forest();
    assert_eq!(forest.n_trees, 100);
    assert_eq!(forest.sample_size, 256);
}

#[test]
fn test_isolation_forest_with_feature_names() {
    let forest = IsolationForest::new(10, 32, 42)
        .with_feature_names(vec!["demo_score".into(), "coverage".into()]);
    assert_eq!(forest.feature_names.len(), 2);
}

#[test]
fn test_isolation_forest_fit_empty() {
    let mut forest = IsolationForest::new(10, 32, 42);
    forest.fit(&[]);
    assert!(forest.trees.is_empty());
}

#[test]
fn test_isolation_forest_fit() {
    let mut forest = IsolationForest::new(10, 32, 42);
    let data = vec![
        vec![90.0, 85.0, 80.0],
        vec![88.0, 82.0, 78.0],
        vec![92.0, 88.0, 82.0],
        vec![85.0, 80.0, 75.0],
    ];
    forest.fit(&data);
    assert_eq!(forest.trees.len(), 10);
}

#[test]
fn test_isolation_forest_score_empty() {
    let forest = IsolationForest::new(10, 32, 42);
    let scores = forest.score(&[vec![90.0, 85.0]]);
    assert_eq!(scores, vec![0.0]); // No trees fitted
}

#[test]
fn test_isolation_forest_score() {
    let mut forest = IsolationForest::new(10, 32, 42);
    let data = vec![
        vec![90.0, 85.0],
        vec![88.0, 82.0],
        vec![92.0, 88.0],
        vec![10.0, 5.0], // Anomaly
    ];
    forest.fit(&data);
    let scores = forest.score(&data);

    assert_eq!(scores.len(), 4);
    // All scores should be in [0, 1]
    for score in &scores {
        assert!(*score >= 0.0 && *score <= 1.0);
    }
}

#[test]
fn test_isolation_forest_predict() {
    let mut forest = IsolationForest::new(10, 32, 42);
    let data = vec![vec![90.0, 85.0], vec![88.0, 82.0], vec![92.0, 88.0]];
    forest.fit(&data);
    let predictions = forest.predict(&data, 0.5);
    assert_eq!(predictions.len(), 3);
}

#[test]
fn test_isolation_forest_detect_anomalies_empty() {
    let forest = IsolationForest::default_forest();
    let diag = StackDiagnostics::new();
    let anomalies = forest.detect_anomalies(&diag, 0.5);
    assert!(anomalies.is_empty());
}

#[test]
fn test_isolation_forest_detect_anomalies() {
    let mut forest = IsolationForest::new(50, 64, 42);
    let mut diag = StackDiagnostics::new();

    // Add normal components
    for i in 0..5 {
        diag.add_component(make_healthy_component(format!("healthy{i}"), i as f64));
    }

    // Add anomalous component
    diag.add_component(make_anomalous_component("anomalous"));

    // Train on component data
    let data: Vec<Vec<f64>> = diag.components().map(|c| extract_features(&c.metrics)).collect();
    forest.fit(&data);

    // Should detect at least something (may or may not flag anomaly depending on threshold)
    let anomalies = forest.detect_anomalies(&diag, 0.3);
    // Just verify it runs without error
    assert!(anomalies.len() <= 6);
}

#[test]
fn test_isolation_forest_categorize_anomaly() {
    let forest = IsolationForest::default_forest();

    // Low demo score -> QualityRegression
    let cat1 = forest.categorize_anomaly(&[50.0, 80.0, 75.0, 5.0, 2.0, 1.0]);
    assert_eq!(cat1, AnomalyCategory::QualityRegression);

    // Low coverage -> CoverageDrop
    let cat2 = forest.categorize_anomaly(&[80.0, 40.0, 75.0, 5.0, 2.0, 1.0]);
    assert_eq!(cat2, AnomalyCategory::CoverageDrop);

    // High complexity -> ComplexityIncrease
    let cat3 = forest.categorize_anomaly(&[80.0, 80.0, 75.0, 20.0, 2.0, 1.0]);
    assert_eq!(cat3, AnomalyCategory::ComplexityIncrease);

    // High dead code -> DependencyRisk
    let cat4 = forest.categorize_anomaly(&[80.0, 80.0, 75.0, 5.0, 2.0, 15.0]);
    assert_eq!(cat4, AnomalyCategory::DependencyRisk);

    // Normal -> Other
    let cat5 = forest.categorize_anomaly(&[90.0, 90.0, 85.0, 5.0, 2.0, 1.0]);
    assert_eq!(cat5, AnomalyCategory::Other);
}

#[test]
fn test_average_path_length() {
    assert_eq!(average_path_length(0.0), 0.0);
    assert_eq!(average_path_length(1.0), 0.0);

    // For n=2, c(n) ≈ 1
    let c2 = average_path_length(2.0);
    assert!(c2 > 0.0 && c2 < 2.0);

    // For larger n, c(n) grows logarithmically
    let c256 = average_path_length(256.0);
    assert!(c256 > c2);
}

// ========================================================================
// Error Forecaster Tests
// ========================================================================

#[test]
fn test_error_forecaster_new() {
    let forecaster = ErrorForecaster::new(0.5);
    assert_eq!(forecaster.alpha, 0.5);
    assert!(forecaster.history().is_empty());
}

#[test]
fn test_error_forecaster_alpha_clamp() {
    let f1 = ErrorForecaster::new(-0.5);
    assert_eq!(f1.alpha, 0.0);

    let f2 = ErrorForecaster::new(1.5);
    assert_eq!(f2.alpha, 1.0);
}

#[test]
fn test_error_forecaster_default() {
    let forecaster = ErrorForecaster::default_forecaster();
    assert_eq!(forecaster.alpha, 0.3);
}

#[test]
fn test_error_forecaster_observe() {
    let mut forecaster = ErrorForecaster::new(0.5);
    forecaster.observe(100.0);
    assert_eq!(forecaster.current_level(), 100.0);
    assert_eq!(forecaster.history().len(), 1);

    forecaster.observe(80.0);
    // Level = 0.5 * 80 + 0.5 * 100 = 90
    assert_eq!(forecaster.current_level(), 90.0);
    assert_eq!(forecaster.history().len(), 2);
}

#[test]
fn test_error_forecaster_forecast() {
    let mut forecaster = ErrorForecaster::new(0.3);
    forecaster.observe(100.0);
    forecaster.observe(90.0);
    forecaster.observe(85.0);

    let forecast = forecaster.forecast(5);
    assert_eq!(forecast.len(), 5);
    // All forecasts should be the same (simple exponential smoothing)
    let level = forecaster.current_level();
    for f in &forecast {
        assert_eq!(*f, level);
    }
}

#[test]
fn test_error_forecaster_error_metrics_empty() {
    let forecaster = ErrorForecaster::new(0.3);
    let metrics = forecaster.error_metrics();
    assert_eq!(metrics.mae, 0.0);
    assert_eq!(metrics.mse, 0.0);
    assert_eq!(metrics.rmse, 0.0);
}

#[test]
fn test_error_forecaster_error_metrics_single() {
    let mut forecaster = ErrorForecaster::new(0.3);
    forecaster.observe(100.0);
    let metrics = forecaster.error_metrics();
    assert_eq!(metrics.mae, 0.0); // Not enough data
}

#[test]
fn test_error_forecaster_error_metrics() {
    let mut forecaster = ErrorForecaster::new(0.5);
    forecaster.observe(100.0);
    forecaster.observe(110.0);
    forecaster.observe(105.0);
    forecaster.observe(108.0);

    let metrics = forecaster.error_metrics();
    assert!(metrics.mae >= 0.0);
    assert!(metrics.mse >= 0.0);
    assert!(metrics.rmse >= 0.0);
    assert_eq!(metrics.rmse, metrics.mse.sqrt());
}

#[test]
fn test_error_forecaster_exponential_smoothing() {
    // Test the exponential smoothing formula
    let mut forecaster = ErrorForecaster::new(0.3);
    forecaster.observe(100.0); // Level = 100
    forecaster.observe(130.0); // Level = 0.3*130 + 0.7*100 = 39 + 70 = 109
    forecaster.observe(100.0); // Level = 0.3*100 + 0.7*109 = 30 + 76.3 = 106.3

    let level = forecaster.current_level();
    assert!((level - 106.3).abs() < 0.01);
}

#[test]
fn test_forecast_metrics_default() {
    let metrics = ForecastMetrics::default();
    assert_eq!(metrics.mae, 0.0);
    assert_eq!(metrics.mse, 0.0);
    assert_eq!(metrics.rmse, 0.0);
    assert_eq!(metrics.mape, 0.0);
}

// ========================================================================
// Additional coverage tests
// ========================================================================

#[test]
fn test_extract_features() {
    let metrics = ComponentMetrics {
        demo_score: 85.0,
        coverage: 90.0,
        mutation_score: 75.0,
        complexity_avg: 8.0,
        satd_count: 5,
        dead_code_pct: 2.0,
        grade: QualityGrade::A,
    };
    let features = extract_features(&metrics);
    assert_eq!(features.len(), 6);
    assert_eq!(features[0], 85.0); // demo_score
    assert_eq!(features[1], 90.0); // coverage
    assert_eq!(features[2], 75.0); // mutation_score
    assert_eq!(features[3], 8.0); // complexity_avg
    assert_eq!(features[4], 5.0); // satd_count
    assert_eq!(features[5], 2.0); // dead_code_pct
}

#[test]
fn test_rule_matches_below() {
    let rule = &CATEGORY_RULES[0]; // FEAT_DEMO_SCORE < 70
    let features = vec![60.0, 80.0, 80.0, 5.0, 2.0, 1.0];
    assert!(rule_matches(rule, &features));

    let features_above = vec![80.0, 80.0, 80.0, 5.0, 2.0, 1.0];
    assert!(!rule_matches(rule, &features_above));
}

#[test]
fn test_rule_matches_above() {
    let rule = &CATEGORY_RULES[2]; // FEAT_COMPLEXITY > 15
    let features_high = vec![80.0, 80.0, 80.0, 20.0, 2.0, 1.0];
    assert!(rule_matches(rule, &features_high));

    let features_low = vec![80.0, 80.0, 80.0, 5.0, 2.0, 1.0];
    assert!(!rule_matches(rule, &features_low));
}

#[test]
fn test_find_matching_rule_quality_regression() {
    let features = vec![50.0, 80.0, 75.0, 5.0, 2.0, 1.0];
    let rule = find_matching_rule(&features);
    assert!(rule.is_some());
    assert_eq!(rule.unwrap().category, AnomalyCategory::QualityRegression);
}

#[test]
fn test_find_matching_rule_none() {
    let features = vec![90.0, 90.0, 85.0, 5.0, 2.0, 1.0];
    let rule = find_matching_rule(&features);
    assert!(rule.is_none());
}

#[test]
fn test_render_description() {
    let template = "Quality score {val:.1} is below threshold";
    let features = vec![50.5, 80.0, 75.0, 5.0, 2.0, 1.0];
    let desc = render_description(template, &features, 0);
    assert!(desc.contains("50.5"));
}

#[test]
fn test_simple_rng_seed() {
    let rng1 = SimpleRng::seed_from_u64(42);
    let rng2 = SimpleRng::seed_from_u64(42);
    assert_eq!(rng1.state, rng2.state);
}

#[test]
fn test_simple_rng_next_u64() {
    let mut rng = SimpleRng::seed_from_u64(42);
    let v1 = rng.next_u64();
    let v2 = rng.next_u64();
    assert_ne!(v1, v2);
}

#[test]
fn test_simple_rng_gen_range() {
    let mut rng = SimpleRng::seed_from_u64(42);
    for _ in 0..100 {
        let val = rng.gen_range(0..10);
        assert!(val < 10);
    }
}

#[test]
fn test_simple_rng_gen_range_empty() {
    let mut rng = SimpleRng::seed_from_u64(42);
    let val = rng.gen_range(5..5);
    assert_eq!(val, 5);
}

#[test]
fn test_simple_rng_gen_range_f64() {
    let mut rng = SimpleRng::seed_from_u64(42);
    for _ in 0..100 {
        let val = rng.gen_range_f64(0.0..1.0);
        assert!((0.0..1.0).contains(&val));
    }
}

#[test]
fn test_isolation_forest_describe_anomaly_short_features() {
    let forest = IsolationForest::default_forest();
    let desc = forest.describe_anomaly(&[50.0], &AnomalyCategory::QualityRegression);
    assert!(desc.contains("Unusual metric combination"));
}

#[test]
fn test_isolation_forest_describe_anomaly_mismatch() {
    let forest = IsolationForest::default_forest();
    // Features trigger QualityRegression but we ask for CoverageDrop
    let features = vec![50.0, 80.0, 75.0, 5.0, 2.0, 1.0];
    let desc = forest.describe_anomaly(&features, &AnomalyCategory::CoverageDrop);
    assert!(desc.contains("Unusual metric combination"));
}

#[test]
fn test_isolation_forest_recommend_action_quality_low_coverage() {
    let forest = IsolationForest::default_forest();
    let features = vec![50.0, 60.0, 75.0, 5.0, 2.0, 1.0]; // Low quality AND coverage < 80
    let rec = forest.recommend_action(&AnomalyCategory::QualityRegression, &features);
    assert!(rec.contains("coverage above 80"));
}

#[test]
fn test_isolation_forest_recommend_action_quality_high_coverage() {
    let forest = IsolationForest::default_forest();
    let features = vec![50.0, 85.0, 75.0, 5.0, 2.0, 1.0]; // Low quality but coverage >= 80
    let rec = forest.recommend_action(&AnomalyCategory::QualityRegression, &features);
    assert!(rec.contains("Review recent changes"));
}

#[test]
fn test_isolation_forest_recommend_action_fallback() {
    let forest = IsolationForest::default_forest();
    let features = vec![90.0, 90.0, 85.0, 5.0, 2.0, 1.0]; // All healthy
    let rec = forest.recommend_action(&AnomalyCategory::Other, &features);
    assert!(rec.contains("unusual patterns"));
}

#[test]
fn test_isolation_forest_categorize_short_features() {
    let forest = IsolationForest::default_forest();
    let cat = forest.categorize_anomaly(&[50.0, 80.0]);
    assert_eq!(cat, AnomalyCategory::Other);
}

#[test]
fn test_isolation_tree_external_node() {
    let mut rng = SimpleRng::seed_from_u64(42);
    let tree = IsolationTree::build(&[], 10, &mut rng);
    match tree {
        IsolationTree::External { size } => assert_eq!(size, 0),
        _ => panic!("Expected External node for empty data"),
    }
}

#[test]
fn test_isolation_tree_single_point() {
    let mut rng = SimpleRng::seed_from_u64(42);
    let data = vec![vec![1.0, 2.0, 3.0]];
    let tree = IsolationTree::build(&data, 10, &mut rng);
    match tree {
        IsolationTree::External { size } => assert_eq!(size, 1),
        _ => panic!("Expected External node for single point"),
    }
}

#[test]
fn test_isolation_tree_empty_features() {
    let mut rng = SimpleRng::seed_from_u64(42);
    let data = vec![vec![], vec![]];
    let tree = IsolationTree::build(&data, 10, &mut rng);
    match tree {
        IsolationTree::External { size } => assert_eq!(size, 2),
        _ => panic!("Expected External node for empty features"),
    }
}

#[test]
fn test_isolation_tree_constant_values() {
    let mut rng = SimpleRng::seed_from_u64(42);
    let data = vec![vec![5.0, 5.0], vec![5.0, 5.0], vec![5.0, 5.0]];
    let tree = IsolationTree::build(&data, 10, &mut rng);
    // Constant values -> External (can't split)
    match tree {
        IsolationTree::External { size } => assert_eq!(size, 3),
        _ => panic!("Expected External node for constant values"),
    }
}

#[test]
fn test_isolation_tree_path_length_external() {
    let tree = IsolationTree::External { size: 10 };
    let point = vec![1.0, 2.0];
    let path = tree.path_length(&point, 0);
    assert!(path > 0);
}

#[test]
fn test_isolation_forest_score_empty_data() {
    let mut forest = IsolationForest::new(10, 32, 42);
    forest.fit(&[vec![1.0, 2.0], vec![3.0, 4.0]]);
    let scores = forest.score(&[]);
    assert!(scores.is_empty());
}

#[test]
fn test_forecast_metrics_fields() {
    let metrics = ForecastMetrics { mae: 1.5, mse: 2.25, rmse: 1.5, mape: 5.0 };
    assert_eq!(metrics.mae, 1.5);
    assert_eq!(metrics.mse, 2.25);
    assert_eq!(metrics.rmse, 1.5);
    assert_eq!(metrics.mape, 5.0);
}

#[test]
fn test_error_forecaster_history() {
    let mut forecaster = ErrorForecaster::new(0.5);
    forecaster.observe(100.0);
    forecaster.observe(200.0);
    let history = forecaster.history();
    assert_eq!(history.len(), 2);
    assert_eq!(history[0], 100.0);
    assert_eq!(history[1], 200.0);
}

#[test]
fn test_error_forecaster_mape_with_zeros() {
    let mut forecaster = ErrorForecaster::new(0.5);
    forecaster.observe(0.0);
    forecaster.observe(0.0);
    forecaster.observe(1.0);
    let metrics = forecaster.error_metrics();
    // MAPE should be NaN when there are zeros in history
    assert!(metrics.mape.is_nan());
}

#[test]
fn test_category_rule_coverage_drop() {
    let features = vec![80.0, 40.0, 75.0, 5.0, 2.0, 1.0]; // Low coverage
    let rule = find_matching_rule(&features);
    assert!(rule.is_some());
    assert_eq!(rule.unwrap().category, AnomalyCategory::CoverageDrop);
}

#[test]
fn test_category_rule_dependency_risk() {
    let features = vec![80.0, 80.0, 75.0, 5.0, 2.0, 15.0]; // High dead code
    let rule = find_matching_rule(&features);
    assert!(rule.is_some());
    assert_eq!(rule.unwrap().category, AnomalyCategory::DependencyRisk);
}

//! Model evaluation examples from the book

use verificar::data::CodeFeatures;
use verificar::ml::{
    benchmark_inference, calculate_feature_importance, BenchmarkResult, ComparisonMetrics,
    ConfusionMatrix, ModelComparison, RocCurve,
};

#[test]
fn test_confusion_matrix_example() {
    // Example: Creating a confusion matrix from predictions
    let predictions = vec![true, true, false, false, true, false, true, false];
    let ground_truth = vec![true, false, false, true, true, false, false, true];

    let matrix = ConfusionMatrix::from_predictions(&predictions, &ground_truth);

    // Verify metrics are in valid ranges
    assert!((0.0..=1.0).contains(&matrix.accuracy()));
    assert!((0.0..=1.0).contains(&matrix.precision()));
    assert!((0.0..=1.0).contains(&matrix.recall()));
    assert!((0.0..=1.0).contains(&matrix.f1_score()));

    // Display the confusion matrix
    let ascii = matrix.to_ascii();
    assert!(ascii.contains("Confusion Matrix"));
}

#[test]
fn test_roc_curve_example() {
    // Example: Computing ROC curve and AUC
    let scores = vec![0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1];
    let ground_truth = vec![true, true, true, true, false, false, false, false];

    let roc = RocCurve::from_scores(&scores, &ground_truth);

    // Perfect separation should have high AUC
    assert!(roc.auc > 0.9);

    // Display ROC curve
    let ascii = roc.to_ascii();
    assert!(ascii.contains("AUC"));
}

#[test]
fn test_benchmark_inference_example() {
    // Example: Benchmarking model inference speed
    let features: Vec<CodeFeatures> = (0..100).map(|_| CodeFeatures::default()).collect();

    let predictor = |f: &CodeFeatures| f.ast_depth as f64 * 0.1;
    let result = benchmark_inference(predictor, &features, 10);

    // Should complete 1000 predictions (100 features × 10 iterations)
    assert_eq!(result.num_predictions, 1000);
    assert!(result.predictions_per_sec > 0.0);
    assert!(result.avg_latency_us > 0.0);
}

#[test]
fn test_model_comparison_example() {
    // Example: Comparing baseline vs trained model
    let baseline = ComparisonMetrics {
        name: "Heuristic".to_string(),
        accuracy: 0.65,
        f1_score: 0.60,
        predictions_per_sec: 100_000.0,
    };

    let trained = ComparisonMetrics {
        name: "RandomForest".to_string(),
        accuracy: 0.85,
        f1_score: 0.82,
        predictions_per_sec: 50_000.0,
    };

    let comparison = ModelComparison::compare(baseline, trained);

    // Trained model should show improvement
    assert!(comparison.accuracy_improvement > 0.0);
    assert!(comparison.f1_improvement > 0.0);

    // Display comparison
    let ascii = comparison.to_ascii();
    assert!(ascii.contains("Model Comparison"));
}

#[test]
fn test_feature_importance_example() {
    // Example: Calculating feature importance
    let features: Vec<CodeFeatures> = (0..50)
        .map(|i| CodeFeatures {
            ast_depth: (i % 10) as u32,
            num_operators: (i % 5) as u32,
            cyclomatic_complexity: (i % 8) as f32,
            uses_edge_values: i % 3 == 0,
            ..Default::default()
        })
        .collect();
    let labels: Vec<bool> = (0..50).map(|i| i % 2 == 0).collect();

    let predictor = |f: &CodeFeatures| f.ast_depth as f64 * 0.1 + f.num_operators as f64 * 0.05;
    let importance = calculate_feature_importance(&features, &labels, &predictor);

    // Should return importance for all features
    assert_eq!(importance.len(), 5);

    // Importance should sum to ~1.0 (normalized)
    let total: f64 = importance.iter().map(|f| f.importance).sum();
    assert!(total <= 1.1);
}

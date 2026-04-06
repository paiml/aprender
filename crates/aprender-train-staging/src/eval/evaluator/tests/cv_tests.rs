//! Tests for cross-validation, KFold, and classification evaluation

use crate::eval::classification::Average;
use crate::eval::evaluator::*;

#[test]
fn test_kfold_split() {
    let kfold = KFold::new(5).without_shuffle();
    let folds = kfold.split(100);

    assert_eq!(folds.len(), 5);

    // Each fold should have 20 test samples
    for (train, test) in &folds {
        assert_eq!(test.len(), 20);
        assert_eq!(train.len(), 80);
    }

    // All indices should be covered exactly once across test sets
    let mut all_test: Vec<usize> = folds.iter().flat_map(|(_, t)| t.iter().copied()).collect();
    all_test.sort_unstable();
    assert_eq!(all_test, (0..100).collect::<Vec<_>>());
}

#[test]
fn test_kfold_uneven_split() {
    let kfold = KFold::new(3).without_shuffle();
    let folds = kfold.split(10);

    assert_eq!(folds.len(), 3);

    // With 10 samples and 3 folds: sizes should be 4, 3, 3
    let test_sizes: Vec<usize> = folds.iter().map(|(_, t)| t.len()).collect();
    assert_eq!(test_sizes.iter().sum::<usize>(), 10);
}

#[test]
fn test_kfold_shuffled() {
    let kfold1 = KFold::new(5).with_seed(42);
    let kfold2 = KFold::new(5).with_seed(42);
    let kfold3 = KFold::new(5).with_seed(99);

    let folds1 = kfold1.split(100);
    let folds2 = kfold2.split(100);
    let folds3 = kfold3.split(100);

    // Same seed should produce same splits
    assert_eq!(folds1[0].1, folds2[0].1);

    // Different seed should produce different splits
    assert_ne!(folds1[0].1, folds3[0].1);
}

#[test]
fn test_evaluate_classification() {
    let config = EvalConfig {
        metrics: vec![
            Metric::Accuracy,
            Metric::Precision(Average::Macro),
            Metric::Recall(Average::Macro),
            Metric::F1(Average::Macro),
        ],
        ..Default::default()
    };

    let evaluator = ModelEvaluator::new(config);

    let y_pred = vec![0, 1, 1, 2, 0, 1];
    let y_true = vec![0, 1, 0, 2, 0, 2];

    let result = evaluator
        .evaluate_classification("TestModel", &y_pred, &y_true)
        .expect("operation should succeed");

    assert!(result.get_score(Metric::Accuracy).is_some());
    assert!(result.get_score(Metric::Precision(Average::Macro)).is_some());
    assert!(result.get_score(Metric::Recall(Average::Macro)).is_some());
    assert!(result.get_score(Metric::F1(Average::Macro)).is_some());

    // Accuracy should be 4/6 = 0.667 (positions 0,1,3,4 correct)
    let acc = result.get_score(Metric::Accuracy).expect("operation should succeed");
    assert!((acc - 0.667).abs() < 0.01);
}

#[test]
fn test_compare_classification() {
    let config = EvalConfig { metrics: vec![Metric::Accuracy], ..Default::default() };

    let evaluator = ModelEvaluator::new(config);

    let y_true = vec![0, 1, 1, 0, 1, 0];

    // Model A: 4/6 correct
    let y_pred_a = vec![0, 1, 1, 1, 1, 0];
    // Model B: 6/6 correct
    let y_pred_b = vec![0, 1, 1, 0, 1, 0];
    // Model C: 3/6 correct
    let y_pred_c = vec![1, 0, 1, 1, 0, 0];

    let models: Vec<(&str, &[usize])> =
        vec![("Model A", &y_pred_a), ("Model B", &y_pred_b), ("Model C", &y_pred_c)];

    let leaderboard =
        evaluator.compare_classification(&models, &y_true).expect("operation should succeed");

    // Model B should be first (perfect accuracy)
    assert_eq!(leaderboard.best().expect("operation should succeed").model_name, "Model B");
}

#[test]
fn test_evaluate_cv() {
    let config =
        EvalConfig { metrics: vec![Metric::Accuracy], cv_folds: 5, seed: 42, ..Default::default() };

    let evaluator = ModelEvaluator::new(config);

    // Create simple data: labels 0-9 repeated 10 times
    let y_true: Vec<usize> = (0..100).map(|i| i % 10).collect();

    // Prediction function that returns the true labels for test indices
    let result = evaluator
        .evaluate_cv("PerfectModel", &y_true, |_train_idx, test_idx| {
            test_idx.iter().map(|&i| y_true[i]).collect()
        })
        .expect("operation should succeed");

    // Perfect predictions should have ~1.0 accuracy
    assert!(result.cv_mean.is_some());
    let cv_mean = result.cv_mean.expect("operation should succeed");
    assert!((cv_mean - 1.0).abs() < 0.01);

    // Should have 5 fold scores
    assert_eq!(result.cv_scores.as_ref().expect("operation should succeed").len(), 5);
}

#[test]
fn test_evaluate_cv_no_folds_error() {
    let config = EvalConfig {
        cv_folds: 0, // No CV
        ..Default::default()
    };

    let evaluator = ModelEvaluator::new(config);
    let y_true = vec![0, 1, 0, 1];

    let result = evaluator.evaluate_cv("Test", &y_true, |_, _| vec![0, 1, 0, 1]);
    assert!(result.is_err());
}

#[test]
fn test_evaluate_classification_length_mismatch() {
    let config = EvalConfig::default();
    let evaluator = ModelEvaluator::new(config);

    let y_pred = vec![0, 1, 2];
    let y_true = vec![0, 1];

    let result = evaluator.evaluate_classification("Test", &y_pred, &y_true);
    assert!(result.is_err());
}

#[test]
fn test_evaluate_cv_with_precision_metric() {
    let config = EvalConfig {
        metrics: vec![Metric::Precision(Average::Macro)],
        cv_folds: 3,
        seed: 42,
        ..Default::default()
    };

    let evaluator = ModelEvaluator::new(config);
    let y_true: Vec<usize> = (0..30).map(|i| i % 3).collect();

    let result = evaluator
        .evaluate_cv("TestModel", &y_true, |_train_idx, test_idx| {
            test_idx.iter().map(|&i| y_true[i]).collect()
        })
        .expect("operation should succeed");

    assert!(result.cv_mean.is_some());
}

#[test]
fn test_evaluate_cv_with_recall_metric() {
    let config = EvalConfig {
        metrics: vec![Metric::Recall(Average::Weighted)],
        cv_folds: 3,
        seed: 42,
        ..Default::default()
    };

    let evaluator = ModelEvaluator::new(config);
    let y_true: Vec<usize> = (0..30).map(|i| i % 3).collect();

    let result = evaluator
        .evaluate_cv("TestModel", &y_true, |_train_idx, test_idx| {
            test_idx.iter().map(|&i| y_true[i]).collect()
        })
        .expect("operation should succeed");

    assert!(result.cv_mean.is_some());
}

#[test]
fn test_evaluate_cv_single_fold_zero_std() {
    let config =
        EvalConfig { metrics: vec![Metric::Accuracy], cv_folds: 1, seed: 42, ..Default::default() };

    let evaluator = ModelEvaluator::new(config);
    let y_true: Vec<usize> = vec![0, 1, 0, 1];

    let result = evaluator
        .evaluate_cv("TestModel", &y_true, |_train_idx, test_idx| {
            test_idx.iter().map(|&i| y_true[i]).collect()
        })
        .expect("operation should succeed");

    // With single fold, std should be 0
    assert_eq!(result.cv_std, Some(0.0));
}

#[test]
fn test_evaluate_classification_skips_non_classification_metrics() {
    use crate::eval::evaluator::metric::RougeVariant;

    let config = EvalConfig {
        metrics: vec![
            // Classification (should compute)
            Metric::Accuracy,
            Metric::Precision(Average::Macro),
            Metric::Recall(Average::Micro),
            Metric::F1(Average::Weighted),
            // Non-classification (all should be skipped via continue)
            Metric::R2,
            Metric::MSE,
            Metric::MAE,
            Metric::RMSE,
            Metric::Silhouette,
            Metric::Inertia,
            Metric::WER,
            Metric::RTFx,
            Metric::BLEU,
            Metric::ROUGE(RougeVariant::Rouge1),
            Metric::Perplexity,
            Metric::MMLUAccuracy,
            Metric::PassAtK(5),
            Metric::NDCGAtK(10),
        ],
        ..Default::default()
    };

    let evaluator = ModelEvaluator::new(config);
    let y_pred = vec![0, 1, 1, 0];
    let y_true = vec![0, 1, 0, 0];

    let result = evaluator
        .evaluate_classification("Test", &y_pred, &y_true)
        .expect("operation should succeed");

    // Classification metrics should be present
    assert!(result.get_score(Metric::Accuracy).is_some());
    assert!(result.get_score(Metric::Precision(Average::Macro)).is_some());
    assert!(result.get_score(Metric::Recall(Average::Micro)).is_some());
    assert!(result.get_score(Metric::F1(Average::Weighted)).is_some());
    // All non-classification metrics should be skipped
    assert!(result.get_score(Metric::R2).is_none());
    assert!(result.get_score(Metric::MSE).is_none());
    assert!(result.get_score(Metric::MAE).is_none());
    assert!(result.get_score(Metric::RMSE).is_none());
    assert!(result.get_score(Metric::Silhouette).is_none());
    assert!(result.get_score(Metric::Inertia).is_none());
    assert!(result.get_score(Metric::WER).is_none());
    assert!(result.get_score(Metric::RTFx).is_none());
    assert!(result.get_score(Metric::BLEU).is_none());
    assert!(result.get_score(Metric::ROUGE(RougeVariant::Rouge1)).is_none());
    assert!(result.get_score(Metric::Perplexity).is_none());
    assert!(result.get_score(Metric::MMLUAccuracy).is_none());
    assert!(result.get_score(Metric::PassAtK(5)).is_none());
    assert!(result.get_score(Metric::NDCGAtK(10)).is_none());
}

#[test]
fn test_cv_non_classification_metric_falls_back_to_accuracy() {
    use crate::eval::evaluator::metric::RougeVariant;

    // Using non-classification metrics as primary in CV should fall back to accuracy
    for metric in [
        Metric::R2,
        Metric::MSE,
        Metric::MAE,
        Metric::RMSE,
        Metric::Silhouette,
        Metric::Inertia,
        Metric::WER,
        Metric::RTFx,
        Metric::BLEU,
        Metric::ROUGE(RougeVariant::RougeL),
        Metric::Perplexity,
        Metric::MMLUAccuracy,
        Metric::PassAtK(1),
        Metric::NDCGAtK(5),
    ] {
        let config =
            EvalConfig { metrics: vec![metric], cv_folds: 2, seed: 42, ..Default::default() };

        let evaluator = ModelEvaluator::new(config);
        let y_true: Vec<usize> = (0..20).map(|i| i % 2).collect();

        let result = evaluator
            .evaluate_cv("FallbackTest", &y_true, |_train_idx, test_idx| {
                test_idx.iter().map(|&i| y_true[i]).collect()
            })
            .expect("operation should succeed");

        assert!(
            result.cv_mean.is_some(),
            "CV should succeed with metric {metric:?} falling back to accuracy"
        );
    }
}

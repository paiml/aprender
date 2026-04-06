//! Core tests for config, metrics, eval results, and leaderboard

use crate::eval::classification::Average;
use crate::eval::evaluator::*;

#[test]
fn test_eval_config_default() {
    let config = EvalConfig::default();
    assert_eq!(config.metrics.len(), 2);
    assert_eq!(config.cv_folds, 0);
    assert_eq!(config.seed, 42);
}

#[test]
fn test_metric_higher_is_better() {
    assert!(Metric::Accuracy.higher_is_better());
    assert!(Metric::F1(Average::Macro).higher_is_better());
    assert!(Metric::R2.higher_is_better());
    assert!(!Metric::MSE.higher_is_better());
    assert!(!Metric::MAE.higher_is_better());
    assert!(!Metric::RMSE.higher_is_better());
}

#[test]
fn test_eval_result() {
    let mut result = EvalResult::new("TestModel");
    result.add_score(Metric::Accuracy, 0.95);
    result.add_score(Metric::F1(Average::Weighted), 0.93);

    assert_eq!(result.get_score(Metric::Accuracy), Some(0.95));
    assert_eq!(result.get_score(Metric::F1(Average::Weighted)), Some(0.93));
    assert_eq!(result.get_score(Metric::R2), None);
}

#[test]
fn test_leaderboard() {
    let mut leaderboard = Leaderboard::new(Metric::Accuracy);

    let mut result1 = EvalResult::new("Model A");
    result1.add_score(Metric::Accuracy, 0.85);
    leaderboard.add(result1);

    let mut result2 = EvalResult::new("Model B");
    result2.add_score(Metric::Accuracy, 0.92);
    leaderboard.add(result2);

    let mut result3 = EvalResult::new("Model C");
    result3.add_score(Metric::Accuracy, 0.88);
    leaderboard.add(result3);

    // Best should be Model B (highest accuracy)
    assert_eq!(leaderboard.best().expect("operation should succeed").model_name, "Model B");

    // Order should be B, C, A
    assert_eq!(leaderboard.results[0].model_name, "Model B");
    assert_eq!(leaderboard.results[1].model_name, "Model C");
    assert_eq!(leaderboard.results[2].model_name, "Model A");
}

#[test]
fn test_leaderboard_lower_is_better() {
    let mut leaderboard = Leaderboard::new(Metric::MSE);

    let mut result1 = EvalResult::new("Model A");
    result1.add_score(Metric::MSE, 0.1);
    leaderboard.add(result1);

    let mut result2 = EvalResult::new("Model B");
    result2.add_score(Metric::MSE, 0.05);
    leaderboard.add(result2);

    // Best should be Model B (lowest MSE)
    assert_eq!(leaderboard.best().expect("operation should succeed").model_name, "Model B");
}

#[test]
fn test_leaderboard_display() {
    let mut leaderboard = Leaderboard::new(Metric::Accuracy);

    let mut result = EvalResult::new("TestModel");
    result.add_score(Metric::Accuracy, 0.95);
    result.inference_time_ms = 1.5;
    leaderboard.add(result);

    let display = format!("{leaderboard}");
    assert!(display.contains("TestModel"));
    assert!(display.contains("Accuracy"));
}

#[test]
fn test_leaderboard_markdown() {
    let mut leaderboard = Leaderboard::new(Metric::Accuracy);

    let mut result = EvalResult::new("TestModel");
    result.add_score(Metric::Accuracy, 0.95);
    result.inference_time_ms = 1.5;
    leaderboard.add(result);

    let md = leaderboard.to_markdown();
    assert!(md.contains("| Model |"));
    assert!(md.contains("| TestModel |"));
}

#[test]
fn test_metric_display() {
    assert_eq!(format!("{}", Metric::Accuracy), "Accuracy");
    assert_eq!(format!("{}", Metric::F1(Average::Weighted)), "F1(Weighted)");
}

#[test]
fn test_empty_leaderboard() {
    let leaderboard = Leaderboard::new(Metric::Accuracy);
    assert!(leaderboard.best().is_none());
}

#[test]
fn test_metric_name_all_variants() {
    assert_eq!(Metric::Accuracy.name(), "Accuracy");
    assert_eq!(Metric::Precision(Average::Macro).name(), "Precision");
    assert_eq!(Metric::Recall(Average::Micro).name(), "Recall");
    assert_eq!(Metric::F1(Average::Weighted).name(), "F1");
    assert_eq!(Metric::R2.name(), "R²");
    assert_eq!(Metric::MSE.name(), "MSE");
    assert_eq!(Metric::MAE.name(), "MAE");
    assert_eq!(Metric::RMSE.name(), "RMSE");
    assert_eq!(Metric::Silhouette.name(), "Silhouette");
    assert_eq!(Metric::Inertia.name(), "Inertia");
}

#[test]
fn test_metric_higher_is_better_all_variants() {
    assert!(Metric::Accuracy.higher_is_better());
    assert!(Metric::Precision(Average::Macro).higher_is_better());
    assert!(Metric::Recall(Average::Micro).higher_is_better());
    assert!(Metric::F1(Average::Weighted).higher_is_better());
    assert!(Metric::R2.higher_is_better());
    assert!(Metric::Silhouette.higher_is_better());
    assert!(!Metric::MSE.higher_is_better());
    assert!(!Metric::MAE.higher_is_better());
    assert!(!Metric::RMSE.higher_is_better());
    assert!(!Metric::Inertia.higher_is_better());
}

#[test]
fn test_metric_display_all_variants() {
    assert_eq!(format!("{}", Metric::Precision(Average::Macro)), "Precision(Macro)");
    assert_eq!(format!("{}", Metric::Recall(Average::Micro)), "Recall(Micro)");
    assert_eq!(format!("{}", Metric::MSE), "MSE");
    assert_eq!(format!("{}", Metric::R2), "R²");
    assert_eq!(format!("{}", Metric::Silhouette), "Silhouette");
}

// ─── New generative metric variant tests ─────────────────────────────

#[test]
fn test_metric_higher_is_better_generative() {
    assert!(!Metric::WER.higher_is_better());
    assert!(Metric::RTFx.higher_is_better());
    assert!(Metric::BLEU.higher_is_better());
    assert!(Metric::ROUGE(RougeVariant::Rouge1).higher_is_better());
    assert!(Metric::ROUGE(RougeVariant::Rouge2).higher_is_better());
    assert!(Metric::ROUGE(RougeVariant::RougeL).higher_is_better());
    assert!(!Metric::Perplexity.higher_is_better());
    assert!(Metric::MMLUAccuracy.higher_is_better());
    assert!(Metric::PassAtK(1).higher_is_better());
    assert!(Metric::NDCGAtK(10).higher_is_better());
}

#[test]
fn test_metric_name_generative() {
    assert_eq!(Metric::WER.name(), "WER");
    assert_eq!(Metric::RTFx.name(), "RTFx");
    assert_eq!(Metric::BLEU.name(), "BLEU");
    assert_eq!(Metric::ROUGE(RougeVariant::Rouge1).name(), "ROUGE");
    assert_eq!(Metric::Perplexity.name(), "Perplexity");
    assert_eq!(Metric::MMLUAccuracy.name(), "MMLU");
    assert_eq!(Metric::PassAtK(1).name(), "pass@k");
    assert_eq!(Metric::NDCGAtK(10).name(), "NDCG@k");
}

#[test]
fn test_metric_display_generative() {
    assert_eq!(format!("{}", Metric::WER), "WER");
    assert_eq!(format!("{}", Metric::RTFx), "RTFx");
    assert_eq!(format!("{}", Metric::BLEU), "BLEU");
    assert_eq!(format!("{}", Metric::ROUGE(RougeVariant::Rouge1)), "ROUGE-1");
    assert_eq!(format!("{}", Metric::ROUGE(RougeVariant::Rouge2)), "ROUGE-2");
    assert_eq!(format!("{}", Metric::ROUGE(RougeVariant::RougeL)), "ROUGE-L");
    assert_eq!(format!("{}", Metric::Perplexity), "Perplexity");
    assert_eq!(format!("{}", Metric::MMLUAccuracy), "MMLU");
    assert_eq!(format!("{}", Metric::PassAtK(1)), "pass@1");
    assert_eq!(format!("{}", Metric::PassAtK(100)), "pass@100");
    assert_eq!(format!("{}", Metric::NDCGAtK(5)), "NDCG@5");
    assert_eq!(format!("{}", Metric::NDCGAtK(10)), "NDCG@10");
}

#[test]
fn test_rouge_variant_display() {
    assert_eq!(format!("{}", RougeVariant::Rouge1), "ROUGE-1");
    assert_eq!(format!("{}", RougeVariant::Rouge2), "ROUGE-2");
    assert_eq!(format!("{}", RougeVariant::RougeL), "ROUGE-L");
}

#[test]
fn test_leaderboard_with_generative_metrics() {
    let mut leaderboard = Leaderboard::new(Metric::WER);

    let mut result1 = EvalResult::new("Whisper-large");
    result1.add_score(Metric::WER, 0.08);
    leaderboard.add(result1);

    let mut result2 = EvalResult::new("Whisper-small");
    result2.add_score(Metric::WER, 0.15);
    leaderboard.add(result2);

    // Best WER is lowest → Whisper-large
    assert_eq!(leaderboard.best().expect("operation should succeed").model_name, "Whisper-large");
}

#[test]
fn test_metric_generative_equality() {
    assert_eq!(Metric::PassAtK(1), Metric::PassAtK(1));
    assert_ne!(Metric::PassAtK(1), Metric::PassAtK(10));
    assert_eq!(Metric::NDCGAtK(5), Metric::NDCGAtK(5));
    assert_ne!(Metric::NDCGAtK(5), Metric::NDCGAtK(10));
    assert_eq!(Metric::ROUGE(RougeVariant::Rouge1), Metric::ROUGE(RougeVariant::Rouge1));
    assert_ne!(Metric::ROUGE(RougeVariant::Rouge1), Metric::ROUGE(RougeVariant::Rouge2));
}

#[test]
fn test_eval_result_display() {
    let mut result = EvalResult::new("TestModel");
    result.add_score(Metric::Accuracy, 0.95);
    result.inference_time_ms = 1.5;

    let display = format!("{result}");
    assert!(display.contains("TestModel"));
    assert!(display.contains("0.95"));
    assert!(display.contains("1.50ms"));
}

#[test]
fn test_leaderboard_empty_display() {
    let leaderboard = Leaderboard::new(Metric::Accuracy);
    let display = format!("{leaderboard}");
    assert!(display.contains("empty"));
}

#[test]
fn test_leaderboard_empty_markdown() {
    let leaderboard = Leaderboard::new(Metric::Accuracy);
    let md = leaderboard.to_markdown();
    assert!(md.is_empty());
}

#[test]
fn test_leaderboard_sort_by() {
    let mut leaderboard = Leaderboard::new(Metric::Accuracy);

    let mut result1 = EvalResult::new("Model A");
    result1.add_score(Metric::Accuracy, 0.85);
    result1.add_score(Metric::F1(Average::Macro), 0.90);
    leaderboard.add(result1);

    let mut result2 = EvalResult::new("Model B");
    result2.add_score(Metric::Accuracy, 0.92);
    result2.add_score(Metric::F1(Average::Macro), 0.80);
    leaderboard.add(result2);

    // Initially sorted by Accuracy: B, A
    assert_eq!(leaderboard.results[0].model_name, "Model B");

    // Sort by F1 instead
    leaderboard.sort_by(Metric::F1(Average::Macro));

    // Now sorted by F1: A, B
    assert_eq!(leaderboard.results[0].model_name, "Model A");
}

#[test]
fn test_eval_config_custom() {
    let config = EvalConfig {
        metrics: vec![Metric::MSE, Metric::MAE],
        cv_folds: 10,
        seed: 123,
        parallel: true,
        trace_enabled: true,
    };

    assert_eq!(config.cv_folds, 10);
    assert_eq!(config.seed, 123);
    assert!(config.parallel);
    assert!(config.trace_enabled);
}

#[test]
fn test_model_evaluator_config() {
    let config = EvalConfig::default();
    let evaluator = ModelEvaluator::new(config.clone());
    assert_eq!(evaluator.config().seed, config.seed);
}

#[test]
fn test_eval_result_cv_fields() {
    let mut result = EvalResult::new("TestModel");
    result.cv_scores = Some(vec![0.9, 0.92, 0.88, 0.91, 0.89]);
    result.cv_mean = Some(0.9);
    result.cv_std = Some(0.014);
    result.trace_id = Some("trace-123".to_string());

    assert_eq!(result.cv_scores.as_ref().expect("operation should succeed").len(), 5);
    assert_eq!(result.cv_mean, Some(0.9));
    assert_eq!(result.cv_std, Some(0.014));
    assert_eq!(result.trace_id, Some("trace-123".to_string()));
}

#[test]
fn test_leaderboard_print() {
    let mut leaderboard = Leaderboard::new(Metric::Accuracy);
    let mut result = EvalResult::new("TestModel");
    result.add_score(Metric::Accuracy, 0.95);
    leaderboard.add(result);

    // Just test that print() doesn't panic
    // (output goes to stdout, can't easily capture in tests)
    leaderboard.print();
}

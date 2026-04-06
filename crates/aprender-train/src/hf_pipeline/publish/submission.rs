//! Leaderboard submission formatting
//!
//! Formats `EvalResult` as JSONL for submission to HuggingFace
//! leaderboard-specific result repositories.

use crate::eval::evaluator::{EvalResult, Metric};

/// Format an `EvalResult` as a JSONL line for leaderboard submission
pub fn format_submission_jsonl(result: &EvalResult) -> String {
    let mut obj = serde_json::Map::new();

    obj.insert("model".to_string(), serde_json::Value::String(result.model_name.clone()));

    for (metric, &value) in &result.scores {
        let key = metric_to_submission_key(metric);
        obj.insert(key, serde_json::json!(value));
    }

    if result.inference_time_ms > 0.0 {
        obj.insert("inference_time_ms".to_string(), serde_json::json!(result.inference_time_ms));
    }

    serde_json::Value::Object(obj).to_string()
}

/// Format multiple `EvalResult`s as JSONL
pub fn format_submissions_jsonl(results: &[EvalResult]) -> String {
    results.iter().map(format_submission_jsonl).collect::<Vec<_>>().join("\n")
}

/// Convert a `Metric` to a leaderboard-compatible key name
fn metric_to_submission_key(metric: &Metric) -> String {
    match metric {
        Metric::WER => "wer".to_string(),
        Metric::RTFx => "rtfx".to_string(),
        Metric::BLEU => "bleu".to_string(),
        Metric::ROUGE(v) => format!("{v}").to_lowercase().replace('-', "_"),
        Metric::Perplexity => "perplexity".to_string(),
        Metric::MMLUAccuracy => "mmlu_accuracy".to_string(),
        Metric::PassAtK(k) => format!("pass@{k}"),
        Metric::NDCGAtK(k) => format!("ndcg@{k}"),
        other => other.name().to_lowercase(),
    }
}

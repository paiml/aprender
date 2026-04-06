//! Falsification tests for HF Hub publishing contracts

use crate::eval::evaluator::{EvalResult, Metric};
use crate::eval::RougeVariant;

use super::config::{PublishConfig, RepoType};
use super::model_card::ModelCard;
use super::submission::{format_submission_jsonl, format_submissions_jsonl};

// ═══════════════════════════════════════════════════════════════════════
// CLAIM: Model card YAML front matter is valid YAML
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn falsify_model_card_yaml_parseable() {
    let mut result = EvalResult::new("test-model");
    result.add_score(Metric::WER, 0.05);
    result.add_score(Metric::BLEU, 0.45);

    let mut card = ModelCard::from_eval_result(&result);
    card.license = Some("apache-2.0".to_string());
    card.language = vec!["en".to_string(), "de".to_string()];
    card.base_model = Some("meta-llama/Llama-3-8B".to_string());

    let md = card.to_markdown();

    // Extract YAML between --- delimiters
    let parts: Vec<&str> = md.splitn(3, "---").collect();
    assert_eq!(parts.len(), 3, "Should have exactly 2 --- delimiters");

    let yaml_str = parts[1].trim();
    // Parse the YAML to verify it's valid
    let yaml: serde_yaml::Value = serde_yaml::from_str(yaml_str).unwrap_or_else(|e| {
        panic!("Model card YAML is invalid: {e}\nYAML:\n{yaml_str}");
    });

    // Verify required fields
    assert_eq!(yaml["license"], "apache-2.0");
    assert!(yaml["language"].is_sequence());
    assert_eq!(yaml["base_model"], "meta-llama/Llama-3-8B");
    assert!(yaml["tags"].is_sequence());
    assert!(yaml["model-index"].is_sequence());
}

#[test]
fn falsify_model_card_empty_result() {
    let result = EvalResult::new("empty-model");
    let card = ModelCard::from_eval_result(&result);
    let md = card.to_markdown();

    // Should still produce valid markdown
    assert!(md.starts_with("---\n"));
    assert!(md.contains("# empty-model"));
    // No metrics → no model-index or evaluation results section
    assert!(!md.contains("Evaluation Results"));
}

#[test]
fn falsify_model_card_special_chars_in_name() {
    let result = EvalResult::new("org/model-v2.1_fp16");
    let card = ModelCard::from_eval_result(&result);
    let md = card.to_markdown();

    assert!(md.contains("# org/model-v2.1_fp16"));
}

// ═══════════════════════════════════════════════════════════════════════
// CLAIM: Model card metric keys match HF Hub naming conventions
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn falsify_metric_yaml_keys() {
    let mut result = EvalResult::new("test");
    result.add_score(Metric::WER, 0.1);
    result.add_score(Metric::BLEU, 0.5);
    result.add_score(Metric::ROUGE(RougeVariant::Rouge1), 0.6);
    result.add_score(Metric::ROUGE(RougeVariant::RougeL), 0.55);
    result.add_score(Metric::Perplexity, 10.0);
    result.add_score(Metric::MMLUAccuracy, 0.7);
    result.add_score(Metric::PassAtK(1), 0.3);
    result.add_score(Metric::NDCGAtK(10), 0.8);

    let card = ModelCard::from_eval_result(&result);

    // Check that all metric keys are lowercase, no special chars except underscore
    for (key, _) in &card.metrics {
        assert!(
            key.chars().all(|c| c.is_ascii_lowercase() || c == '_' || c.is_ascii_digit()),
            "Metric key should be lowercase+underscore+digits: got '{key}'"
        );
    }

    // Check specific key mappings
    let keys: Vec<&str> = card.metrics.iter().map(|(k, _)| k.as_str()).collect();
    assert!(keys.contains(&"wer"), "Should have 'wer' key");
    assert!(keys.contains(&"bleu"), "Should have 'bleu' key");
    assert!(keys.contains(&"perplexity"), "Should have 'perplexity' key");
    assert!(keys.contains(&"mmlu_accuracy"), "Should have 'mmlu_accuracy' key");
    assert!(keys.contains(&"pass_at_1"), "Should have 'pass_at_1' key");
    assert!(keys.contains(&"ndcg_at_10"), "Should have 'ndcg_at_10' key");
}

// ═══════════════════════════════════════════════════════════════════════
// CLAIM: JSONL submission is valid JSON per line
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn falsify_jsonl_each_line_valid_json() {
    let mut r1 = EvalResult::new("model-a");
    r1.add_score(Metric::WER, 0.05);
    r1.add_score(Metric::RTFx, 150.0);
    r1.inference_time_ms = 2.5;

    let mut r2 = EvalResult::new("model-b");
    r2.add_score(Metric::BLEU, 0.45);
    r2.add_score(Metric::ROUGE(RougeVariant::Rouge1), 0.6);

    let jsonl = format_submissions_jsonl(&[r1, r2]);
    for (i, line) in jsonl.lines().enumerate() {
        let parsed: Result<serde_json::Value, _> = serde_json::from_str(line);
        assert!(
            parsed.is_ok(),
            "Line {i} is not valid JSON: {line}\nError: {}",
            parsed.expect_err("parsing should succeed")
        );
    }
}

#[test]
fn falsify_jsonl_model_field_always_present() {
    let result = EvalResult::new("test-model");
    let jsonl = format_submission_jsonl(&result);
    let parsed: serde_json::Value =
        serde_json::from_str(&jsonl).expect("JSON deserialization should succeed");
    assert_eq!(parsed["model"], "test-model");
}

#[test]
fn falsify_jsonl_no_inference_time_when_zero() {
    let mut result = EvalResult::new("test");
    result.add_score(Metric::Accuracy, 0.9);
    // inference_time_ms defaults to 0.0

    let jsonl = format_submission_jsonl(&result);
    let parsed: serde_json::Value =
        serde_json::from_str(&jsonl).expect("JSON deserialization should succeed");
    assert!(parsed.get("inference_time_ms").is_none(), "Should omit inference_time_ms when 0.0");
}

#[test]
fn falsify_jsonl_metric_key_format() {
    let mut result = EvalResult::new("test");
    result.add_score(Metric::PassAtK(1), 0.5);
    result.add_score(Metric::NDCGAtK(10), 0.8);

    let jsonl = format_submission_jsonl(&result);
    let parsed: serde_json::Value =
        serde_json::from_str(&jsonl).expect("JSON deserialization should succeed");

    // Submission uses pass@1 format (not pass_at_1)
    assert!(parsed.get("pass@1").is_some(), "Submission should use 'pass@1' key, got: {jsonl}");
    assert!(parsed.get("ndcg@10").is_some(), "Submission should use 'ndcg@10' key, got: {jsonl}");
}

// ═══════════════════════════════════════════════════════════════════════
// CLAIM: PublishConfig validation
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn falsify_publish_config_serde_roundtrip() {
    let config = PublishConfig {
        repo_id: "org/model".to_string(),
        repo_type: RepoType::Model,
        private: true,
        token: None, // skipped in serde
        license: Some("mit".to_string()),
        tags: vec!["nlp".to_string(), "transformer".to_string()],
    };

    let json = serde_json::to_string(&config).expect("JSON serialization should succeed");
    let parsed: PublishConfig =
        serde_json::from_str(&json).expect("JSON deserialization should succeed");

    assert_eq!(parsed.repo_id, "org/model");
    assert_eq!(parsed.repo_type, RepoType::Model);
    assert!(parsed.private);
    assert!(parsed.token.is_none(), "Token should be skipped in serde");
    assert_eq!(parsed.license, Some("mit".to_string()));
    assert_eq!(parsed.tags.len(), 2);
}

#[test]
fn falsify_repo_type_serde() {
    // RepoType should serialize as lowercase strings
    let json = serde_json::to_string(&RepoType::Model).expect("JSON serialization should succeed");
    assert_eq!(json, "\"model\"");

    let json =
        serde_json::to_string(&RepoType::Dataset).expect("JSON serialization should succeed");
    assert_eq!(json, "\"dataset\"");

    let json = serde_json::to_string(&RepoType::Space).expect("JSON serialization should succeed");
    assert_eq!(json, "\"space\"");

    // And deserialize back
    let rt: RepoType =
        serde_json::from_str("\"model\"").expect("JSON deserialization should succeed");
    assert_eq!(rt, RepoType::Model);
}

#[test]
fn falsify_publish_config_default_values() {
    let config = PublishConfig::default();
    assert!(config.repo_id.is_empty());
    assert_eq!(config.repo_type, RepoType::Model);
    assert!(!config.private);
    assert!(config.token.is_none());
    assert!(config.license.is_none());
    assert!(config.tags.is_empty());
}

// ═══════════════════════════════════════════════════════════════════════
// CLAIM: Model card from_eval_result captures all scores
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn falsify_model_card_captures_all_metrics() {
    let mut result = EvalResult::new("multi-metric");
    result.add_score(Metric::Accuracy, 0.95);
    result.add_score(Metric::WER, 0.05);
    result.add_score(Metric::BLEU, 0.45);

    let card = ModelCard::from_eval_result(&result);
    assert_eq!(card.metrics.len(), 3, "Card should have all 3 metrics, got {}", card.metrics.len());
}

// ═══════════════════════════════════════════════════════════════════════
// CLAIM: Publisher rejects invalid repo IDs
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn falsify_publisher_rejects_no_slash() {
    use super::publisher::HfPublisher;
    use super::result::PublishError;

    let config = PublishConfig {
        repo_id: "no-slash-here".to_string(),
        token: Some("fake".to_string()),
        ..Default::default()
    };
    let result = HfPublisher::new(config);
    assert!(matches!(result, Err(PublishError::InvalidRepoId { .. })));
}

#[test]
fn falsify_publisher_rejects_empty() {
    use super::publisher::HfPublisher;
    use super::result::PublishError;

    let config = PublishConfig {
        repo_id: String::new(),
        token: Some("fake".to_string()),
        ..Default::default()
    };
    let result = HfPublisher::new(config);
    assert!(matches!(result, Err(PublishError::InvalidRepoId { .. })));
}

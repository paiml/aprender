//! Tests for HuggingFace Hub publishing

use super::config::{PublishConfig, RepoType};
use super::model_card::ModelCard;
use super::result::PublishError;
use super::submission::{format_submission_jsonl, format_submissions_jsonl};
use crate::eval::evaluator::{EvalResult, Metric};

// ─── PublishConfig tests ─────────────────────────────────────────────

#[test]
fn test_publish_config_default() {
    let config = PublishConfig::default();
    assert!(config.repo_id.is_empty());
    assert_eq!(config.repo_type, RepoType::Model);
    assert!(!config.private);
    assert!(config.token.is_none());
    assert!(config.license.is_none());
    assert!(config.tags.is_empty());
}

#[test]
fn test_repo_type_api_path() {
    assert_eq!(RepoType::Model.api_path(), "models");
    assert_eq!(RepoType::Dataset.api_path(), "datasets");
    assert_eq!(RepoType::Space.api_path(), "spaces");
}

#[test]
fn test_repo_type_display() {
    assert_eq!(format!("{}", RepoType::Model), "model");
    assert_eq!(format!("{}", RepoType::Dataset), "dataset");
    assert_eq!(format!("{}", RepoType::Space), "space");
}

// ─── PublishError tests ──────────────────────────────────────────────

#[test]
fn test_publish_error_display() {
    let err = PublishError::AuthRequired;
    assert!(err.to_string().contains("HF_TOKEN"));

    let err = PublishError::InvalidRepoId { repo_id: "bad".into() };
    assert!(err.to_string().contains("bad"));

    let err = PublishError::RepoCreationFailed { repo_id: "a/b".into(), message: "403".into() };
    assert!(err.to_string().contains("a/b"));
}

// ─── HfPublisher tests ──────────────────────────────────────────────

#[test]
fn test_publisher_requires_auth() {
    // Without token, should fail with AuthRequired
    let config = PublishConfig { repo_id: "test/model".to_string(), ..Default::default() };
    let result = super::publisher::HfPublisher::new(config);
    // Will either fail with AuthRequired or succeed if HF_TOKEN is set
    if std::env::var("HF_TOKEN").is_err() {
        assert!(matches!(result, Err(PublishError::AuthRequired)));
    }
}

#[test]
fn test_publisher_invalid_repo_id() {
    let config = PublishConfig {
        repo_id: "no-slash".to_string(),
        token: Some("fake-token".to_string()),
        ..Default::default()
    };
    let result = super::publisher::HfPublisher::new(config);
    assert!(matches!(result, Err(PublishError::InvalidRepoId { .. })));
}

#[test]
fn test_publisher_empty_repo_id() {
    let config = PublishConfig {
        repo_id: String::new(),
        token: Some("fake-token".to_string()),
        ..Default::default()
    };
    let result = super::publisher::HfPublisher::new(config);
    assert!(matches!(result, Err(PublishError::InvalidRepoId { .. })));
}

// ─── ModelCard tests ─────────────────────────────────────────────────

#[test]
fn test_model_card_from_eval_result() {
    let mut result = EvalResult::new("my-model");
    result.add_score(Metric::WER, 0.05);
    result.add_score(Metric::Accuracy, 0.95);

    let card = ModelCard::from_eval_result(&result);
    assert_eq!(card.model_name, "my-model");
    assert_eq!(card.metrics.len(), 2);
    assert!(card.tags.contains(&"entrenar".to_string()));
}

#[test]
fn test_model_card_markdown() {
    let mut card = ModelCard::from_eval_result(&EvalResult::new("test-model"));
    card.license = Some("apache-2.0".to_string());
    card.language = vec!["en".to_string()];
    card.base_model = Some("meta-llama/Llama-3-8B".to_string());

    let md = card.to_markdown();
    assert!(md.starts_with("---\n"));
    assert!(md.contains("license: apache-2.0"));
    assert!(md.contains("- en"));
    assert!(md.contains("base_model: meta-llama/Llama-3-8B"));
    assert!(md.contains("# test-model"));
    assert!(md.contains("entrenar"));
}

#[test]
fn test_model_card_with_metrics() {
    let mut result = EvalResult::new("whisper-fine-tuned");
    result.add_score(Metric::WER, 0.042);

    let card = ModelCard::from_eval_result(&result);
    let md = card.to_markdown();
    assert!(md.contains("model-index:"));
    assert!(md.contains("wer"));
    assert!(md.contains("0.042"));
    assert!(md.contains("Evaluation Results"));
}

#[test]
fn test_model_card_with_training_details() {
    let mut card = ModelCard::from_eval_result(&EvalResult::new("test"));
    card.training_details = Some("Fine-tuned for 3 epochs on custom data.".to_string());

    let md = card.to_markdown();
    assert!(md.contains("Training Details"));
    assert!(md.contains("Fine-tuned for 3 epochs"));
}

// ─── Submission tests ───────────────────────────────────────────────

#[test]
fn test_format_submission_jsonl() {
    let mut result = EvalResult::new("my-model");
    result.add_score(Metric::WER, 0.05);
    result.inference_time_ms = 10.5;

    let jsonl = format_submission_jsonl(&result);
    let parsed: serde_json::Value =
        serde_json::from_str(&jsonl).expect("JSON deserialization should succeed");

    assert_eq!(parsed["model"], "my-model");
    assert_eq!(parsed["wer"], 0.05);
    assert_eq!(parsed["inference_time_ms"], 10.5);
}

#[test]
fn test_format_submissions_jsonl() {
    let mut r1 = EvalResult::new("model-a");
    r1.add_score(Metric::Accuracy, 0.9);

    let mut r2 = EvalResult::new("model-b");
    r2.add_score(Metric::Accuracy, 0.8);

    let jsonl = format_submissions_jsonl(&[r1, r2]);
    let lines: Vec<&str> = jsonl.lines().collect();
    assert_eq!(lines.len(), 2);

    let p1: serde_json::Value =
        serde_json::from_str(lines[0]).expect("JSON deserialization should succeed");
    let p2: serde_json::Value =
        serde_json::from_str(lines[1]).expect("JSON deserialization should succeed");
    assert_eq!(p1["model"], "model-a");
    assert_eq!(p2["model"], "model-b");
}

#[test]
fn test_format_submission_no_inference_time() {
    let mut result = EvalResult::new("my-model");
    result.add_score(Metric::BLEU, 0.45);

    let jsonl = format_submission_jsonl(&result);
    let parsed: serde_json::Value =
        serde_json::from_str(&jsonl).expect("JSON deserialization should succeed");
    assert!(parsed.get("inference_time_ms").is_none());
}

// ─── PublishResult tests ────────────────────────────────────────────

#[test]
fn test_publish_result_display() {
    let result = super::result::PublishResult {
        repo_url: "https://huggingface.co/models/test/model".to_string(),
        repo_id: "test/model".to_string(),
        files_uploaded: 3,
        model_card_generated: true,
    };
    let display = format!("{result}");
    assert!(display.contains("3 files"));
    assert!(display.contains("model card"));
}

#[test]
fn test_publish_result_display_no_card() {
    let result = super::result::PublishResult {
        repo_url: "https://huggingface.co/models/test/model".to_string(),
        repo_id: "test/model".to_string(),
        files_uploaded: 1,
        model_card_generated: false,
    };
    let display = format!("{result}");
    assert!(!display.contains("model card"));
}

// ─── Integration tests (requires network + HF_TOKEN) ───────────────

#[test]
#[ignore = "Requires network access and HF_TOKEN with write permissions"]
fn test_publish_to_hub() {
    use super::publisher::HfPublisher;

    let config = PublishConfig {
        repo_id: "test-org/entrenar-test-publish".to_string(),
        private: true,
        ..Default::default()
    };
    let publisher = HfPublisher::new(config).expect("Failed to create publisher");

    let result = publisher.create_repo();
    match result {
        Ok(url) => assert!(url.contains("huggingface.co")),
        Err(e) => eprintln!("Publish test failed (expected in CI): {e}"),
    }
}

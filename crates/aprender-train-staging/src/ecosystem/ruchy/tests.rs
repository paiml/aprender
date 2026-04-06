//! Tests for the Ruchy session bridge.

use super::artifact::build_session_description;
use super::*;
use crate::research::ArtifactType;
use std::collections::HashMap;

#[test]
fn test_session_metrics_creation() {
    let mut metrics = SessionMetrics::new();
    assert!(metrics.is_empty());

    metrics.add_loss(0.5);
    metrics.add_loss(0.3);
    metrics.add_loss(0.2);

    assert!(!metrics.is_empty());
    assert_eq!(metrics.total_steps(), 3);
    assert_eq!(metrics.final_loss(), Some(0.2));
    assert_eq!(metrics.best_loss(), Some(0.2));
}

#[test]
fn test_session_metrics_accuracy() {
    let mut metrics = SessionMetrics::new();

    metrics.add_accuracy(0.7);
    metrics.add_accuracy(0.8);
    metrics.add_accuracy(0.85);

    assert_eq!(metrics.final_accuracy(), Some(0.85));
    assert_eq!(metrics.best_accuracy(), Some(0.85));
}

#[test]
fn test_session_metrics_custom() {
    let mut metrics = SessionMetrics::new();

    metrics.add_custom("f1_score", 0.75);
    metrics.add_custom("f1_score", 0.82);

    assert_eq!(metrics.custom.get("f1_score"), Some(&vec![0.75, 0.82]));
}

#[test]
fn test_entrenar_session_creation() {
    let session = EntrenarSession::new("sess-001", "My Training Session")
        .with_user("alice")
        .with_architecture("llama-7b")
        .with_dataset("custom-dataset")
        .with_config("batch_size", "32")
        .with_config("learning_rate", "1e-4")
        .with_tag("fine-tuning")
        .with_notes("Initial experiment");

    assert_eq!(session.id, "sess-001");
    assert_eq!(session.name, "My Training Session");
    assert_eq!(session.user, Some("alice".to_string()));
    assert_eq!(session.model_architecture, Some("llama-7b".to_string()));
    assert_eq!(session.dataset_id, Some("custom-dataset".to_string()));
    assert_eq!(session.config.get("batch_size"), Some(&"32".to_string()));
    assert_eq!(session.tags, vec!["fine-tuning"]);
    assert!(!session.has_training_data());
}

#[test]
fn test_entrenar_session_with_metrics() {
    let mut session = EntrenarSession::new("sess-001", "Training");
    session.metrics.add_loss(0.5);
    session.metrics.add_loss(0.3);

    assert!(session.has_training_data());
    assert_eq!(session.metrics.total_steps(), 2);
}

#[test]
fn test_entrenar_session_duration() {
    let mut session = EntrenarSession::new("sess-001", "Training");
    let start = session.created_at;

    // Simulate session end
    session.ended_at = Some(start + chrono::Duration::hours(2));

    let duration = session.duration().expect("operation should succeed");
    assert_eq!(duration.num_hours(), 2);
}

#[test]
fn test_code_cell_creation() {
    let cell = CodeCell {
        execution_order: 1,
        source: "model.train()".to_string(),
        output: Some("Training started...".to_string()),
        timestamp: chrono::Utc::now(),
        duration_ms: Some(1500),
    };

    assert_eq!(cell.execution_order, 1);
    assert_eq!(cell.source, "model.train()");
    assert!(cell.output.is_some());
}

#[test]
fn test_ruchy_session_conversion() {
    let ruchy = RuchySession {
        session_id: "ruchy-123".to_string(),
        title: "LLaMA Fine-tuning".to_string(),
        username: Some("bob".to_string()),
        start_time: chrono::Utc::now(),
        end_time: None,
        kernel: Some("python3".to_string()),
        cells: vec![RuchyCell {
            id: "cell-1".to_string(),
            cell_type: "code".to_string(),
            source: "import entrenar".to_string(),
            outputs: vec!["OK".to_string()],
            execution_count: Some(1),
            executed_at: Some(chrono::Utc::now()),
        }],
        variables: HashMap::from([("lr".to_string(), "0.001".to_string())]),
        training_runs: vec![TrainingRun {
            run_id: "run-1".to_string(),
            model: "llama".to_string(),
            dataset: Some("alpaca".to_string()),
            epochs: 3,
            losses: vec![0.5, 0.3, 0.2],
            metrics: HashMap::new(),
        }],
    };

    let session: EntrenarSession = ruchy.into();

    assert_eq!(session.id, "ruchy-123");
    assert_eq!(session.name, "LLaMA Fine-tuning");
    assert_eq!(session.user, Some("bob".to_string()));
    assert_eq!(session.model_architecture, Some("python3".to_string()));
    assert_eq!(session.dataset_id, Some("alpaca".to_string()));
    assert_eq!(session.code_history.len(), 1);
    assert_eq!(session.metrics.total_steps(), 3);
    assert_eq!(session.config.get("lr"), Some(&"0.001".to_string()));
}

#[test]
fn test_session_to_artifact_success() {
    let mut session = EntrenarSession::new("sess-001", "My Experiment")
        .with_user("alice")
        .with_architecture("llama-7b")
        .with_dataset("custom-data")
        .with_tag("lora")
        .with_tag("fine-tuning");

    session.metrics.add_loss(0.5);
    session.metrics.add_loss(0.3);
    session.metrics.add_loss(0.2);

    let artifact = session_to_artifact(&session).expect("operation should succeed");

    assert_eq!(artifact.id, "sess-001");
    assert_eq!(artifact.title, "My Experiment");
    assert_eq!(artifact.artifact_type, ArtifactType::Notebook);
    assert_eq!(artifact.authors.len(), 1);
    assert_eq!(artifact.authors[0].name, "alice");
    assert!(artifact.keywords.contains(&"lora".to_string()));
    assert!(artifact.version.contains("steps3"));
}

#[test]
fn test_session_to_artifact_no_training() {
    let session = EntrenarSession::new("sess-001", "Empty Session");
    let result = session_to_artifact(&session);
    assert!(matches!(result, Err(RuchyBridgeError::NoTrainingHistory)));
}

#[test]
fn test_session_to_artifact_with_code_only() {
    let mut session = EntrenarSession::new("sess-001", "Code Session");
    session.add_code_cell(CodeCell {
        execution_order: 1,
        source: "print('hello')".to_string(),
        output: None,
        timestamp: chrono::Utc::now(),
        duration_ms: None,
    });

    let artifact = session_to_artifact(&session).expect("operation should succeed");
    assert_eq!(artifact.id, "sess-001");
}

#[test]
fn test_build_session_description() {
    let mut session =
        EntrenarSession::new("sess-001", "Test").with_architecture("gpt2").with_dataset("wiki");

    session.metrics.add_loss(0.5);
    session.metrics.add_loss(0.2);
    session.metrics.add_accuracy(85.5);

    let desc = build_session_description(&session);

    assert!(desc.contains("gpt2"));
    assert!(desc.contains("wiki"));
    assert!(desc.contains('2')); // steps
    assert!(desc.contains("0.2")); // loss
}

// Issue #75: Session Export API tests
#[test]
fn test_export_json_basic() {
    let session = EntrenarSession::new("export-001", "Export Test")
        .with_user("tester")
        .with_tag("export-test");

    let json = session.export_json().expect("operation should succeed");

    assert_eq!(json["session_id"], "export-001");
    assert_eq!(json["name"], "Export Test");
    assert_eq!(json["user"], "tester");
    assert_eq!(json["metrics"]["total_steps"], 0);
    assert_eq!(json["code_cells_count"], 0);
    assert!(json["tags"]
        .as_array()
        .expect("operation should succeed")
        .contains(&"export-test".into()));
}

#[test]
fn test_export_json_with_metrics() {
    let mut session = EntrenarSession::new("export-002", "Metrics Export")
        .with_architecture("llama-7b")
        .with_dataset("alpaca");

    session.metrics.add_loss(0.5);
    session.metrics.add_loss(0.3);
    session.metrics.add_loss(0.2);
    session.metrics.add_accuracy(0.7);
    session.metrics.add_accuracy(0.85);
    session.metrics.add_custom("f1", 0.78);

    let json = session.export_json().expect("operation should succeed");

    assert_eq!(json["metrics"]["total_steps"], 3);
    assert_eq!(json["metrics"]["final_loss"], 0.2);
    assert_eq!(json["metrics"]["best_loss"], 0.2);
    assert_eq!(json["metrics"]["final_accuracy"], 0.85);
    assert_eq!(json["metrics"]["best_accuracy"], 0.85);
    assert_eq!(
        json["metrics"]["loss_history"].as_array().expect("operation should succeed").len(),
        3
    );
    assert_eq!(
        json["metrics"]["accuracy_history"].as_array().expect("operation should succeed").len(),
        2
    );
    assert!(json["metrics"]["custom_metrics"]["f1"].as_array().is_some());
}

#[test]
fn test_export_json_string() {
    let session =
        EntrenarSession::new("export-003", "String Export").with_config("batch_size", "32");

    let json_str = session.export_json_string().expect("operation should succeed");

    assert!(json_str.contains("\"session_id\": \"export-003\""));
    assert!(json_str.contains("\"batch_size\": \"32\""));
    // Pretty print should have newlines
    assert!(json_str.contains('\n'));
}

#[test]
fn test_export_json_with_duration() {
    let mut session = EntrenarSession::new("export-004", "Duration Export");
    let start = session.created_at;
    session.ended_at = Some(start + chrono::Duration::hours(1) + chrono::Duration::minutes(30));

    let json = session.export_json().expect("operation should succeed");

    assert_eq!(json["duration_seconds"], 5400); // 1.5 hours = 5400 seconds
    assert!(json["ended_at"].as_str().is_some());
}

#[test]
fn test_export_json_roundtrip() {
    let mut session = EntrenarSession::new("export-005", "Roundtrip Test")
        .with_user("alice")
        .with_architecture("transformer")
        .with_dataset("custom-data")
        .with_config("epochs", "10")
        .with_tag("test");

    session.metrics.add_loss(0.4);
    session.metrics.add_accuracy(0.9);

    let json = session.export_json().expect("operation should succeed");
    let export: SessionExport =
        serde_json::from_value(json).expect("JSON deserialization should succeed");

    assert_eq!(export.session_id, "export-005");
    assert_eq!(export.name, "Roundtrip Test");
    assert_eq!(export.user, Some("alice".to_string()));
    assert_eq!(export.model_architecture, Some("transformer".to_string()));
    assert_eq!(export.dataset_id, Some("custom-data".to_string()));
    assert_eq!(export.metrics.total_steps, 1);
    assert_eq!(export.metrics.final_loss, Some(0.4));
    assert_eq!(export.metrics.final_accuracy, Some(0.9));
}

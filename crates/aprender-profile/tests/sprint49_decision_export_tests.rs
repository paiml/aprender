//! Sprint 49: Decision Trace OTLP Export Tests (Ticket #19)
//!
//! EXTREME TDD: RED phase - these tests define the acceptance criteria
//!
//! Reference: paiml/depyler docs/specifications/decision-traces-signal-spec.md

#![allow(deprecated)] // Command::cargo_bin deprecation warning

use renacer::decision_export::{DecisionExportConfig, DecisionExporter, ExportStats, RetryConfig};
use renacer::decision_trace::DecisionTrace;
use tempfile::TempDir;

// =============================================================================
// RED PHASE: Export config tests
// =============================================================================

#[test]
fn test_decision_export_config_default() {
    // AC: Default config with sensible defaults
    let config = DecisionExportConfig::default();

    assert_eq!(config.otlp_endpoint, "http://localhost:4317");
    assert_eq!(config.batch_size, 100);
    assert_eq!(config.flush_interval_ms, 1000);
    assert_eq!(config.queue_size, 10000);
    assert!(config.auth_token.is_none());
}

#[test]
fn test_decision_export_config_from_toml() {
    // AC: Parse config from TOML
    let toml = r#"
otlp_endpoint = "http://entrenar.example.com:4317"
batch_size = 200
flush_interval_ms = 2000

[retry]
max_attempts = 10
initial_backoff_ms = 200
max_backoff_ms = 60000
queue_size = 20000
"#;

    let config: DecisionExportConfig = toml::from_str(toml).expect("test");

    assert_eq!(config.otlp_endpoint, "http://entrenar.example.com:4317");
    assert_eq!(config.batch_size, 200);
    assert_eq!(config.flush_interval_ms, 2000);
    assert_eq!(config.retry.max_attempts, 10);
    assert_eq!(config.retry.initial_backoff_ms, 200);
    assert_eq!(config.retry.max_backoff_ms, 60000);
    assert_eq!(config.retry.queue_size, 20000);
}

#[test]
fn test_retry_config_default() {
    // AC: Retry config with sensible defaults
    let config = RetryConfig::default();

    assert_eq!(config.max_attempts, 5);
    assert_eq!(config.initial_backoff_ms, 100);
    assert_eq!(config.max_backoff_ms, 30000);
    assert_eq!(config.queue_size, 10000);
}

#[test]
fn test_retry_config_exponential_backoff() {
    // AC: Calculate exponential backoff correctly
    let config = RetryConfig {
        max_attempts: 5,
        initial_backoff_ms: 100,
        max_backoff_ms: 30000,
        queue_size: 10000,
    };

    // Attempt 0: 100ms
    assert_eq!(config.backoff_ms(0), 100);
    // Attempt 1: 200ms
    assert_eq!(config.backoff_ms(1), 200);
    // Attempt 2: 400ms
    assert_eq!(config.backoff_ms(2), 400);
    // Attempt 3: 800ms
    assert_eq!(config.backoff_ms(3), 800);
    // Attempt 4: 1600ms
    assert_eq!(config.backoff_ms(4), 1600);
    // Attempt 10: capped at max_backoff_ms
    assert_eq!(config.backoff_ms(10), 30000);
}

// =============================================================================
// RED PHASE: Exporter creation tests
// =============================================================================

#[test]
fn test_decision_exporter_creation() {
    // AC: Create exporter from config
    let config = DecisionExportConfig::default();
    let exporter = DecisionExporter::new(config);

    assert!(exporter.is_ok());
}

#[test]
fn test_decision_exporter_queue_decisions() {
    // AC: Queue decisions for export
    let config = DecisionExportConfig::default();
    let mut exporter = DecisionExporter::new(config).expect("test");

    let decision = DecisionTrace {
        timestamp_us: 1000,
        category: "TypeMapping".to_string(),
        name: "test".to_string(),
        input: serde_json::json!({}),
        result: None,
        source_location: None,
        decision_id: Some(1),
    };

    exporter.queue(decision.clone());
    assert_eq!(exporter.queue_len(), 1);

    exporter.queue(decision);
    assert_eq!(exporter.queue_len(), 2);
}

#[test]
fn test_decision_exporter_queue_overflow() {
    // AC: Queue respects max size, drops oldest
    let config = DecisionExportConfig { queue_size: 5, ..Default::default() };
    let mut exporter = DecisionExporter::new(config).expect("test");

    // Queue 10 decisions
    for i in 0..10 {
        let decision = DecisionTrace {
            timestamp_us: i * 1000,
            category: "Test".to_string(),
            name: format!("decision_{}", i),
            input: serde_json::json!({}),
            result: None,
            source_location: None,
            decision_id: Some(i),
        };
        exporter.queue(decision);
    }

    // Should only have 5 (most recent)
    assert_eq!(exporter.queue_len(), 5);

    // Stats should show dropped
    let stats = exporter.stats();
    assert_eq!(stats.decisions_dropped, 5);
}

// =============================================================================
// RED PHASE: Batch export tests
// =============================================================================

#[test]
fn test_decision_exporter_batch_size() {
    // AC: Export respects batch size
    let config = DecisionExportConfig { batch_size: 3, ..Default::default() };
    let mut exporter = DecisionExporter::new(config).expect("test");

    // Queue 10 decisions
    for i in 0..10 {
        let decision = DecisionTrace {
            timestamp_us: i * 1000,
            category: "Test".to_string(),
            name: format!("decision_{}", i),
            input: serde_json::json!({}),
            result: None,
            source_location: None,
            decision_id: Some(i),
        };
        exporter.queue(decision);
    }

    // Get next batch (should be 3)
    let batch = exporter.next_batch();
    assert_eq!(batch.len(), 3);

    // Queue should have 7 remaining
    assert_eq!(exporter.queue_len(), 7);
}

// =============================================================================
// RED PHASE: CLI command tests
// =============================================================================

#[test]
#[ignore] // Sprint 50: Requires CLI subcommand refactor
fn test_cli_stats_command() {
    // AC: renacer stats <file> shows decision statistics
    use assert_cmd::Command;
    use renacer::decision_trace::DecisionTrace;

    let temp_dir = TempDir::new().expect("test");
    let msgpack_path = temp_dir.path().join("decisions.msgpack");

    // Write test decisions
    let traces: Vec<DecisionTrace> = (0..100)
        .map(|i| DecisionTrace {
            timestamp_us: i * 1000,
            category: if i % 2 == 0 { "TypeMapping" } else { "BorrowStrategy" }.to_string(),
            name: format!("decision_{}", i),
            input: serde_json::json!({}),
            result: None,
            source_location: Some(format!("test.rs:{}", i)),
            decision_id: Some(i),
        })
        .collect();

    std::fs::write(&msgpack_path, rmp_serde::to_vec(&traces).expect("test")).expect("test");

    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    let output = cmd.arg("stats").arg(&msgpack_path).output().expect("Failed to execute");

    assert!(output.status.success(), "stats command should succeed");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("100"), "Should show total count: {}", stdout);
    assert!(stdout.contains("TypeMapping"), "Should show category: {}", stdout);
    assert!(stdout.contains("BorrowStrategy"), "Should show category: {}", stdout);
}

#[test]
#[ignore] // Sprint 50: Requires CLI subcommand refactor
fn test_cli_export_command_help() {
    // AC: renacer export --help shows usage
    use assert_cmd::Command;

    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    let output = cmd.arg("export").arg("--help").output().expect("Failed to execute");

    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("--format"), "Should show format option");
    assert!(stdout.contains("--endpoint"), "Should show endpoint option");
    assert!(stdout.contains("otlp"), "Should mention OTLP format");
}

// =============================================================================
// RED PHASE: Export stats tests
// =============================================================================

#[test]
fn test_export_stats_default() {
    // AC: Export stats start at zero
    let stats = ExportStats::default();

    assert_eq!(stats.decisions_queued, 0);
    assert_eq!(stats.decisions_exported, 0);
    assert_eq!(stats.decisions_dropped, 0);
    assert_eq!(stats.batches_sent, 0);
    assert_eq!(stats.batches_failed, 0);
    assert_eq!(stats.retry_attempts, 0);
}

// =============================================================================
// RED PHASE: Auth token tests
// =============================================================================

#[test]
fn test_decision_export_config_with_auth() {
    // AC: Config supports auth token
    let toml = r#"
otlp_endpoint = "http://secure.example.com:4317"
auth_token = "secret-token-123"
"#;

    let config: DecisionExportConfig = toml::from_str(toml).expect("test");

    assert_eq!(config.auth_token, Some("secret-token-123".to_string()));
}

#[test]
fn test_decision_export_config_auth_from_env() {
    // AC: Auth token can come from environment
    std::env::set_var("RENACER_AUTH_TOKEN", "env-token-456");

    let config = DecisionExportConfig::from_env();

    assert_eq!(config.auth_token, Some("env-token-456".to_string()));

    std::env::remove_var("RENACER_AUTH_TOKEN");
}

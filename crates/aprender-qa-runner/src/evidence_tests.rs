use super::*;

use aprender_qa_gen::{Backend, Format, Modality, ModelId};

fn test_scenario() -> QaScenario {
    QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Cpu,
        Format::Gguf,
        "2+2=".to_string(),
        42,
    )
}

#[test]
fn test_evidence_corroborated() {
    let evidence = Evidence::corroborated("F-TEST-001", test_scenario(), "4", 100);
    assert_eq!(evidence.outcome, Outcome::Corroborated);
    assert!(evidence.outcome.is_pass());
}

#[test]
fn test_evidence_falsified() {
    let evidence = Evidence::falsified("F-TEST-001", test_scenario(), "Wrong answer", "5", 100);
    assert_eq!(evidence.outcome, Outcome::Falsified);
    assert!(evidence.outcome.is_fail());
}

#[test]
fn test_evidence_collector() {
    let mut collector = EvidenceCollector::new();
    collector.add(Evidence::corroborated(
        "F-TEST-001",
        test_scenario(),
        "4",
        100,
    ));
    collector.add(Evidence::falsified(
        "F-TEST-002",
        test_scenario(),
        "Failed",
        "",
        100,
    ));

    assert_eq!(collector.total(), 2);
    assert_eq!(collector.pass_count(), 1);
    assert_eq!(collector.fail_count(), 1);
}

#[test]
fn test_outcome_pass_fail() {
    // Only Corroborated is a pass (Popperian: must survive falsification)
    assert!(Outcome::Corroborated.is_pass());
    assert!(!Outcome::Skipped.is_pass());
    assert!(!Outcome::Falsified.is_pass());
    assert!(Outcome::Falsified.is_fail());
    assert!(Outcome::Timeout.is_fail());
    assert!(Outcome::Crashed.is_fail());
    // Skipped is neither pass nor fail
    assert!(!Outcome::Skipped.is_pass());
    assert!(!Outcome::Skipped.is_fail());
}

#[test]
fn test_evidence_timeout() {
    let evidence = Evidence::timeout("F-TEST-001", test_scenario(), 30000);
    assert_eq!(evidence.outcome, Outcome::Timeout);
    assert!(evidence.outcome.is_fail());
    assert!(evidence.reason.contains("30000"));
    assert_eq!(evidence.metrics.duration_ms, 30000);
}

#[test]
fn test_evidence_crashed() {
    let evidence = Evidence::crashed("F-TEST-001", test_scenario(), "segfault", 139, 100);
    assert_eq!(evidence.outcome, Outcome::Crashed);
    assert!(evidence.outcome.is_fail());
    assert!(evidence.reason.contains("139"));
    assert_eq!(evidence.stderr, Some("segfault".to_string()));
    assert_eq!(evidence.exit_code, Some(139));
}

#[test]
fn test_evidence_skipped() {
    let evidence = Evidence::skipped("F-TEST-001", test_scenario(), "Format not available");
    assert_eq!(evidence.outcome, Outcome::Skipped);
    // Skipped is neither pass nor fail — test was not subjected to falsification
    assert!(!evidence.outcome.is_pass());
    assert!(!evidence.outcome.is_fail());
    assert!(evidence.reason.contains("Format not available"));
    assert!(evidence.output.is_empty());
    assert!(evidence.exit_code.is_none());
}

#[test]
fn test_evidence_with_metrics() {
    let metrics = PerformanceMetrics {
        tokens_per_second: Some(100.0),
        time_to_first_token_ms: Some(50.0),
        total_tokens: Some(1000),
        memory_peak_mb: Some(512),
        duration_ms: 200,
    };
    let evidence =
        Evidence::corroborated("F-TEST-001", test_scenario(), "output", 100).with_metrics(metrics);
    assert_eq!(evidence.metrics.tokens_per_second, Some(100.0));
    assert_eq!(evidence.metrics.total_tokens, Some(1000));
    assert_eq!(evidence.metrics.duration_ms, 200);
}

#[test]
fn test_evidence_add_metadata() {
    let mut evidence = Evidence::corroborated("F-TEST-001", test_scenario(), "output", 100);
    evidence.add_metadata("key1", "value1");
    evidence.add_metadata("key2", "value2");
    assert_eq!(evidence.metadata.get("key1"), Some(&"value1".to_string()));
    assert_eq!(evidence.metadata.get("key2"), Some(&"value2".to_string()));
}

#[test]
fn test_collector_counts() {
    let mut collector = EvidenceCollector::new();
    collector.add(Evidence::corroborated("F-001", test_scenario(), "ok", 100));
    collector.add(Evidence::corroborated("F-002", test_scenario(), "ok", 100));
    collector.add(Evidence::falsified(
        "F-003",
        test_scenario(),
        "fail",
        "bad",
        100,
    ));
    collector.add(Evidence::timeout("F-004", test_scenario(), 5000));

    let counts = collector.counts();
    assert_eq!(counts.get(&Outcome::Corroborated), Some(&2));
    assert_eq!(counts.get(&Outcome::Falsified), Some(&1));
    assert_eq!(counts.get(&Outcome::Timeout), Some(&1));
}

#[test]
fn test_collector_failures() {
    let mut collector = EvidenceCollector::new();
    collector.add(Evidence::corroborated("F-001", test_scenario(), "ok", 100));
    collector.add(Evidence::falsified(
        "F-002",
        test_scenario(),
        "fail",
        "bad",
        100,
    ));
    collector.add(Evidence::crashed("F-003", test_scenario(), "err", -1, 100));

    let failures = collector.failures();
    assert_eq!(failures.len(), 2);
    assert!(failures.iter().all(|e| e.outcome.is_fail()));
}

#[test]
fn test_collector_to_json() {
    let mut collector = EvidenceCollector::new();
    collector.add(Evidence::corroborated("F-001", test_scenario(), "ok", 100));

    let json = collector.to_json().expect("Failed to serialize");
    assert!(json.contains("F-001"));
    assert!(json.contains("Corroborated"));
}

#[test]
fn test_host_info_default() {
    let host = HostInfo::default();
    assert!(!host.hostname.is_empty());
    assert!(!host.os.is_empty());
    assert_eq!(host.apr_version, "unknown");
}

#[test]
fn test_performance_metrics_default() {
    let metrics = PerformanceMetrics::default();
    assert_eq!(metrics.duration_ms, 0);
    assert!(metrics.tokens_per_second.is_none());
    assert!(metrics.memory_peak_mb.is_none());
}

#[test]
fn test_outcome_debug() {
    let outcome = Outcome::Corroborated;
    let debug_str = format!("{outcome:?}");
    assert!(debug_str.contains("Corroborated"));
}

#[test]
fn test_outcome_clone_eq() {
    let outcome1 = Outcome::Falsified;
    let outcome2 = outcome1;
    assert_eq!(outcome1, outcome2);
}

#[test]
fn test_evidence_collector_default() {
    let collector = EvidenceCollector::default();
    assert_eq!(collector.total(), 0);
    assert_eq!(collector.pass_count(), 0);
    assert_eq!(collector.fail_count(), 0);
}

#[test]
fn test_uuid_generation() {
    let uuid1 = uuid_v4();
    let uuid2 = uuid_v4();
    // UUIDs should be generated (not asserting uniqueness as they may be same in fast succession)
    assert!(!uuid1.is_empty());
    assert!(!uuid2.is_empty());
}

#[test]
fn test_evidence_has_all_fields() {
    let evidence = Evidence::corroborated("F-001", test_scenario(), "output", 100);
    assert!(!evidence.id.is_empty());
    assert_eq!(evidence.gate_id, "F-001");
    assert_eq!(evidence.output, "output");
    assert!(evidence.exit_code.is_some());
    assert!(evidence.stderr.is_none());
}

#[test]
fn test_evidence_serialization() {
    let evidence = Evidence::falsified("F-001", test_scenario(), "bad", "out", 100);
    let json = serde_json::to_string(&evidence).expect("serialize");
    let parsed: Evidence = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.gate_id, evidence.gate_id);
    assert_eq!(parsed.outcome, evidence.outcome);
}

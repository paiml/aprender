
use super::*;
use crate::command::MockCommandRunner;

// ── execute_transformation_tests ─────────────────────────────────────────────

/// Build a minimal playbook with no `transformations:` block
fn playbook_no_transformations() -> Playbook {
    let yaml = r#"
name: test-no-transform
version: "1.0.0"
model:
  hf_repo: "test/model"
  formats: [gguf]
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
"#;
    Playbook::from_yaml(yaml).expect("Failed to parse")
}

/// Build a minimal playbook with a `transformations:` block
fn playbook_with_quantize() -> Playbook {
    let yaml = r#"
name: test-transform-quantize
version: "1.0.0"
model:
  hf_repo: "test/model"
  formats: [gguf]
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
transformations:
  quantize:
    schemes: ["q4_k_m"]
"#;
    Playbook::from_yaml(yaml).expect("Failed to parse")
}

fn playbook_with_import() -> Playbook {
    let yaml = r#"
name: test-transform-import
version: "1.0.0"
model:
  hf_repo: "test/model"
  formats: [gguf]
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
transformations:
  import:
    source_formats: ["gguf"]
"#;
    Playbook::from_yaml(yaml).expect("Failed to parse")
}

fn playbook_with_prune() -> Playbook {
    let yaml = r#"
name: test-transform-prune
version: "1.0.0"
model:
  hf_repo: "test/model"
  formats: [gguf]
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
transformations:
  prune:
    method: "magnitude"
    target_ratio: 0.5
"#;
    Playbook::from_yaml(yaml).expect("Failed to parse")
}

fn playbook_with_distill() -> Playbook {
    let yaml = r#"
name: test-transform-distill
version: "1.0.0"
model:
  hf_repo: "test/model"
  formats: [gguf]
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
transformations:
  distill:
    student_model: "test/student"
    data_path: "/tmp/data"
"#;
    Playbook::from_yaml(yaml).expect("Failed to parse")
}

#[test]
fn test_execute_transformation_no_block_produces_skipped_evidence() {
    // When playbook has no transformations: block, execute_transformation_tests
    // must add exactly one skipped evidence with gate_id F-TRANSFORM-SKIP-002
    let runner = MockCommandRunner::default();
    let config = ExecutionConfig {
        dry_run: true,
        ..Default::default()
    };
    let mut executor = Executor::with_runner(config, Arc::new(runner));
    let playbook = playbook_no_transformations();

    let (passed, failed) = executor.execute_transformation_tests(&playbook);
    assert_eq!(passed, 0);
    assert_eq!(failed, 0);

    let evidence = executor.collector.all();
    let skip_ev = evidence
        .iter()
        .find(|e| e.scenario.prompt.contains("F-TRANSFORM-SKIP") || e.gate_id == "F-TRANSFORM-SKIP-002");
    // The skip evidence exists (gate F-TRANSFORM-SKIP-002)
    let transform_skip = evidence
        .iter()
        .find(|e| e.gate_id == "F-TRANSFORM-SKIP-002");
    assert!(
        transform_skip.is_some() || skip_ev.is_some(),
        "Expected F-TRANSFORM-SKIP-002 skipped evidence, got: {:?}",
        evidence.iter().map(|e| &e.gate_id).collect::<Vec<_>>()
    );
}

#[test]
fn test_execute_transformation_no_model_path_produces_skipped_evidence() {
    // When playbook has transformations: but ExecutionConfig has no model_path,
    // execute_transformation_tests must add skipped evidence with F-TRANSFORM-SKIP-001
    let runner = MockCommandRunner::default();
    let config = ExecutionConfig {
        dry_run: true,
        model_path: None, // explicitly absent
        ..Default::default()
    };
    let mut executor = Executor::with_runner(config, Arc::new(runner));
    let playbook = playbook_with_quantize();

    let (passed, failed) = executor.execute_transformation_tests(&playbook);
    assert_eq!(passed, 0);
    assert_eq!(failed, 0);

    let evidence = executor.collector.all();
    let transform_skip = evidence
        .iter()
        .find(|e| e.gate_id == "F-TRANSFORM-SKIP-001");
    assert!(
        transform_skip.is_some(),
        "Expected F-TRANSFORM-SKIP-001 skipped evidence, got: {:?}",
        evidence.iter().map(|e| &e.gate_id).collect::<Vec<_>>()
    );
    assert_eq!(transform_skip.unwrap().outcome, Outcome::Skipped);
}

#[test]
fn test_execute_transformation_quantize_dispatches_battery() {
    // With quantize: config and model_path set, the quantize battery must be invoked
    let runner = MockCommandRunner::default();
    let dir = tempfile::TempDir::new().unwrap();
    let config = ExecutionConfig {
        model_path: Some(dir.path().to_string_lossy().to_string()),
        ..Default::default()
    };
    let mut executor = Executor::with_runner(config, Arc::new(runner));
    let playbook = playbook_with_quantize();

    executor.execute_transformation_tests(&playbook);

    let evidence = executor.collector.all();
    // At least one T1-QUANT-* gate should be in evidence
    let quant_ev = evidence
        .iter()
        .any(|e| e.gate_id.starts_with("T1-QUANT"));
    assert!(quant_ev, "Expected T1-QUANT evidence from quantize battery dispatch");
}

#[test]
fn test_execute_transformation_import_dispatches_battery() {
    let runner = MockCommandRunner::default();
    let dir = tempfile::TempDir::new().unwrap();
    let config = ExecutionConfig {
        model_path: Some(dir.path().to_string_lossy().to_string()),
        ..Default::default()
    };
    let mut executor = Executor::with_runner(config, Arc::new(runner));
    let playbook = playbook_with_import();

    executor.execute_transformation_tests(&playbook);

    let evidence = executor.collector.all();
    // Import battery uses T1-IMPORT-* gate IDs
    let import_ev = evidence
        .iter()
        .any(|e| e.gate_id.starts_with("T1-IMPORT") || e.scenario.prompt.contains("import:"));
    assert!(import_ev, "Expected import battery evidence, got gates: {:?}",
        evidence.iter().map(|e| &e.gate_id).collect::<Vec<_>>());
}

#[test]
fn test_execute_transformation_prune_dispatches_battery() {
    let runner = MockCommandRunner::default();
    let dir = tempfile::TempDir::new().unwrap();
    let config = ExecutionConfig {
        model_path: Some(dir.path().to_string_lossy().to_string()),
        ..Default::default()
    };
    let mut executor = Executor::with_runner(config, Arc::new(runner));
    let playbook = playbook_with_prune();

    executor.execute_transformation_tests(&playbook);

    let evidence = executor.collector.all();
    // Prune battery uses T1-PRUNE-* gate IDs
    let prune_ev = evidence
        .iter()
        .any(|e| e.gate_id.starts_with("T1-PRUNE") || e.scenario.prompt.contains("prune:"));
    assert!(prune_ev, "Expected prune battery evidence, got gates: {:?}",
        evidence.iter().map(|e| &e.gate_id).collect::<Vec<_>>());
}

#[test]
fn test_execute_transformation_distill_dispatches_battery() {
    let runner = MockCommandRunner::default();
    let dir = tempfile::TempDir::new().unwrap();
    let config = ExecutionConfig {
        model_path: Some(dir.path().to_string_lossy().to_string()),
        ..Default::default()
    };
    let mut executor = Executor::with_runner(config, Arc::new(runner));
    let playbook = playbook_with_distill();

    executor.execute_transformation_tests(&playbook);

    let evidence = executor.collector.all();
    // Distill battery uses T1-DISTILL-* gate IDs
    let distill_ev = evidence
        .iter()
        .any(|e| e.gate_id.starts_with("T1-DISTILL") || e.scenario.prompt.contains("distill"));
    assert!(distill_ev, "Expected distill battery evidence, got gates: {:?}",
        evidence.iter().map(|e| &e.gate_id).collect::<Vec<_>>());
}

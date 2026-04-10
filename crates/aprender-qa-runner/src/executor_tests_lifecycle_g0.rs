
// ─── execute() lifecycle gate coverage ───────────────────────────────────────
//
// Covers uncovered early-return branches in execute():
//   • metadata_only = true  → execute_metadata_only()
//   • format_failed > 0     → G0-FORMAT Jidoka stop
//   • integrity_failed > 0  → G0-INTEGRITY Jidoka stop
//
// Also covers execute_metadata_only() function body:
//   • corroborated loop branch (check.passed = true)
//   • falsified loop branch   (check.passed = false)
//   • gateway_failed = Some(…)

/// execute() metadata_only=true routes to execute_metadata_only() (line 45-47).
/// Empty model dir → config_parse fails → falsified evidence + gateway_failed Some(…).
#[test]
fn test_execute_metadata_only_empty_dir() {
    let dir = tempfile::TempDir::new().expect("create temp dir");

    let config = ExecutionConfig {
        metadata_only: true,
        model_path: Some(dir.path().to_string_lossy().to_string()),
        ..Default::default()
    };
    let mut executor = Executor::with_runner(config, Arc::new(MockCommandRunner::new()));

    let yaml = r#"
name: metadata-only-empty
version: "1.0.0"
model:
  hf_repo: "test/model"
  formats: [gguf]
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 2
"#;
    let playbook = Playbook::from_yaml(yaml).expect("parse");
    let result = executor.execute(&playbook).expect("execute");

    // metadata_only branch executed — scenarios NOT run (total != 2)
    // Empty dir: config.json missing → at least config_parse fails
    assert!(result.gateway_failed.is_some(), "Expected gateway_failed for empty dir");
    let gf = result.gateway_failed.as_ref().unwrap();
    assert!(gf.contains("G0-DIM"), "Expected G0-DIM in gateway_failed, got: {gf}");

    // Falsified evidence generated for failing checks
    let evidence = executor.evidence().all();
    assert!(
        evidence.iter().any(|e| e.gate_id.starts_with("G0-DIM") && e.outcome.is_fail()),
        "Expected at least one G0-DIM falsified evidence item"
    );
}

/// execute() metadata_only=true with valid config.json but no safetensors.
/// Covers corroborated branch (config_parse passes) and falsified branch (safetensors_found fails).
#[test]
fn test_execute_metadata_only_config_passes_no_safetensors() {
    let dir = tempfile::TempDir::new().expect("create temp dir");

    // Valid config.json → config_parse check passes (corroborated branch)
    std::fs::write(
        dir.path().join("config.json"),
        r#"{"num_hidden_layers": 4, "hidden_size": 64}"#,
    )
    .expect("write config.json");

    let config = ExecutionConfig {
        metadata_only: true,
        model_path: Some(dir.path().to_string_lossy().to_string()),
        ..Default::default()
    };
    let mut executor = Executor::with_runner(config, Arc::new(MockCommandRunner::new()));

    let yaml = r#"
name: metadata-config-pass
version: "1.0.0"
model:
  hf_repo: "test/model"
  formats: [safetensors]
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
"#;
    let playbook = Playbook::from_yaml(yaml).expect("parse");
    let result = executor.execute(&playbook).expect("execute");

    // config_parse check passes → corroborated evidence should exist
    let evidence = executor.evidence().all();
    let config_check = evidence
        .iter()
        .find(|e| e.gate_id == "G0-DIM-CONFIG_PARSE");
    assert!(config_check.is_some(), "Expected G0-DIM-CONFIG_PARSE evidence");
    assert!(
        config_check.unwrap().outcome.is_pass(),
        "config_parse should be corroborated when config.json is valid"
    );

    // safetensors_found check fails → falsified evidence exists
    assert!(
        evidence.iter().any(|e| e.gate_id.starts_with("G0-DIM") && e.outcome.is_fail()),
        "Expected falsified G0-DIM evidence (safetensors_found fails)"
    );

    // Some checks failed → gateway_failed is Some
    assert!(result.gateway_failed.is_some());
}

/// execute() format_failed > 0 early return (line 102-115).
/// Flat model dir + APR format + convert_failure → format_failed=1 → G0-FORMAT gateway stop.
#[test]
fn test_execute_format_failed_early_return() {
    let source_dir = tempfile::TempDir::new().expect("create source dir");
    let out_dir = tempfile::TempDir::new().expect("create output dir");

    // Flat dir: model.safetensors in root (no apr/ subdir) — triggers is_flat_dir
    std::fs::write(source_dir.path().join("model.safetensors"), b"fake").expect("write model");

    let mock_runner = MockCommandRunner::new().with_convert_failure();

    let config = ExecutionConfig {
        model_path: Some(source_dir.path().to_string_lossy().to_string()),
        output_dir: Some(out_dir.path().to_string_lossy().to_string()),
        run_conversion_tests: false,
        run_golden_rule_test: false,
        run_contract_tests: false,
        ..Default::default()
    };
    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));

    // APR format triggers convert_model → with_convert_failure → format_failed=1
    let yaml = r#"
name: format-fail-test
version: "1.0.0"
model:
  hf_repo: "test/model"
  formats: [apr]
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 2
"#;
    let playbook = Playbook::from_yaml(yaml).expect("parse");
    let result = executor.execute(&playbook).expect("execute");

    assert!(
        result.gateway_failed.is_some(),
        "Expected gateway_failed for format failure"
    );
    let gf = result.gateway_failed.as_ref().unwrap();
    assert!(
        gf.contains("G0-FORMAT"),
        "Expected G0-FORMAT in gateway_failed, got: {gf}"
    );
}

/// execute_metadata_only: model_path=None, pull fails → G0-PULL-001 gateway failure
/// (executor_lifecycle.rs lines 289-301)
#[test]
fn test_execute_metadata_only_pull_fails_gateway() {
    let mock_runner = MockCommandRunner::new().with_pull_failure();

    let config = ExecutionConfig {
        metadata_only: true,
        model_path: None, // Forces HF cache resolve + pull
        ..Default::default()
    };
    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));

    let yaml = r#"
name: metadata-pull-fail
version: "1.0.0"
model:
  hf_repo: "nonexistent/repo-xyz-12345"
  formats: [safetensors]
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
"#;
    let playbook = Playbook::from_yaml(yaml).expect("parse");
    let result = executor.execute(&playbook).expect("execute");

    assert!(
        result.gateway_failed.is_some(),
        "Expected gateway_failed when pull fails"
    );
    let gf = result.gateway_failed.as_ref().unwrap();
    assert!(
        gf.contains("G0-PULL-001"),
        "Expected G0-PULL-001 gateway, got: {gf}"
    );
}

/// execute_metadata_only: model_path=None, pull succeeds but returns empty path
/// → G0-PULL-001 gateway failure (executor_lifecycle.rs lines 302-318)
#[test]
fn test_execute_metadata_only_pull_returns_empty_path() {
    let mock_runner = MockCommandRunner::new().with_pull_model_path("");

    let config = ExecutionConfig {
        metadata_only: true,
        model_path: None,
        ..Default::default()
    };
    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));

    let yaml = r#"
name: metadata-empty-path
version: "1.0.0"
model:
  hf_repo: "nonexistent/repo-xyz-12345"
  formats: [safetensors]
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
"#;
    let playbook = Playbook::from_yaml(yaml).expect("parse");
    let result = executor.execute(&playbook).expect("execute");

    assert!(
        result.gateway_failed.is_some(),
        "Expected gateway_failed when pull returns empty path"
    );
    let gf = result.gateway_failed.as_ref().unwrap();
    assert!(
        gf.contains("G0-PULL-001"),
        "Expected G0-PULL-001 gateway for empty path, got: {gf}"
    );
}

/// execute() integrity_failed > 0 early return (line 138-151).
/// Flat model dir + safetensors format → workspace created (no config.json inside)
/// → integrity check fails with G0-INTEGRITY-CONFIG error → early return.
#[test]
fn test_execute_integrity_failed_early_return() {
    let source_dir = tempfile::TempDir::new().expect("create source dir");
    let out_dir = tempfile::TempDir::new().expect("create output dir");

    // Flat dir: model.safetensors in root (no apr/ subdir, no config.json)
    std::fs::write(source_dir.path().join("model.safetensors"), b"fake").expect("write model");

    // Default mock: validate_model_strict succeeds, convert_model succeeds
    let mock_runner = MockCommandRunner::new();

    let config = ExecutionConfig {
        model_path: Some(source_dir.path().to_string_lossy().to_string()),
        output_dir: Some(out_dir.path().to_string_lossy().to_string()),
        run_conversion_tests: false,
        run_golden_rule_test: false,
        run_contract_tests: false,
        ..Default::default()
    };
    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));

    // safetensors format → no conversion (SafeTensors skipped in convert_requested_formats)
    // format_check passes (0 failed), validate_check passes (mock succeeds),
    // tensor_check skips (no DEFAULT_APRENDER_PATH),
    // integrity_check on workspace: workspace/safetensors/ has no config.json → fails
    let yaml = r#"
name: integrity-fail-test
version: "1.0.0"
model:
  hf_repo: "test/model"
  formats: [safetensors]
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 2
"#;
    let playbook = Playbook::from_yaml(yaml).expect("parse");
    let result = executor.execute(&playbook).expect("execute");

    assert!(
        result.gateway_failed.is_some(),
        "Expected gateway_failed for integrity failure"
    );
    let gf = result.gateway_failed.as_ref().unwrap();
    assert!(
        gf.contains("G0-INTEGRITY"),
        "Expected G0-INTEGRITY in gateway_failed, got: {gf}"
    );
}

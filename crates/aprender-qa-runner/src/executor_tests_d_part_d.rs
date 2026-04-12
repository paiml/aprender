#[test]
fn test_integrity_check_disabled_by_default() {
    // With check_integrity=false (default), integrity checks are skipped
    let config = ExecutionConfig {
        run_conversion_tests: false,
        run_golden_rule_test: false,
        ..Default::default()
    };

    assert!(!config.check_integrity);
    assert!(config.lock_file_path.is_none());

    let mock_runner = MockCommandRunner::new();
    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));
    let yaml = r#"
name: no-integrity
version: "1.0.0"
model:
  hf_repo: "test/model"
  formats: [gguf]
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
"#;
    let playbook = Playbook::from_yaml(yaml).expect("parse");
    let result = executor.execute(&playbook).expect("execute");

    // Should succeed without integrity check
    assert!(result.gateway_failed.is_none());
}

#[test]
fn test_integrity_check_missing_lock_file_warns() {
    // When lock file path is set but file doesn't exist, should warn (not error)
    let mock_runner = MockCommandRunner::new();
    let config = ExecutionConfig {
        check_integrity: true,
        lock_file_path: Some("/nonexistent/playbook.lock.yaml".to_string()),
        run_conversion_tests: false,
        run_golden_rule_test: false,
        ..Default::default()
    };

    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));
    let yaml = r#"
name: missing-lock
version: "1.0.0"
model:
  hf_repo: "test/model"
  formats: [gguf]
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
"#;
    let playbook = Playbook::from_yaml(yaml).expect("parse");
    let result = executor.execute(&playbook).expect("execute");

    // Should proceed (not fail) when lock file is missing — just warn
    assert!(result.gateway_failed.is_none());
}

#[test]
fn test_warn_implicit_skips_flag() {
    // warn_implicit_skips should not crash even when no skip files exist
    let mock_runner = MockCommandRunner::new();
    let config = ExecutionConfig {
        warn_implicit_skips: true,
        run_conversion_tests: false,
        run_golden_rule_test: false,
        ..Default::default()
    };

    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));
    let yaml = r#"
name: skip-warn-test
version: "1.0.0"
model:
  hf_repo: "test/model"
  formats: [gguf]
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
"#;
    let playbook = Playbook::from_yaml(yaml).expect("parse");
    let result = executor.execute(&playbook).expect("execute");

    // Should succeed — implicit skip warnings are informational only
    assert!(result.gateway_failed.is_none());
}

#[test]
fn test_backward_compat_new_flags_off() {
    // Ensure old configs (without new fields) still work via Default
    let config = ExecutionConfig::default();
    assert!(!config.check_integrity);
    assert!(!config.warn_implicit_skips);
    assert!(config.lock_file_path.is_none());
}

// ============================================================
// HF Parity Tests
// ============================================================

#[test]
fn test_hf_parity_disabled_by_default() {
    // HF parity should be disabled by default
    let config = ExecutionConfig::default();
    assert!(!config.run_hf_parity);
    assert!(config.hf_parity_corpus_path.is_none());
    assert!(config.hf_parity_model_family.is_none());
}

#[test]
fn test_hf_parity_skipped_when_missing_config() {
    // When HF parity is enabled but config is incomplete, should skip gracefully
    let mock_runner = MockCommandRunner::new();
    let config = ExecutionConfig {
        run_hf_parity: true,
        hf_parity_corpus_path: None,  // Missing!
        hf_parity_model_family: None, // Missing!
        run_conversion_tests: false,
        run_golden_rule_test: false,
        ..Default::default()
    };

    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));
    let yaml = r#"
name: hf-parity-test
version: "1.0.0"
model:
  hf_repo: "test/model"
  formats: [gguf]
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
"#;
    let playbook = Playbook::from_yaml(yaml).expect("parse");
    let result = executor.execute(&playbook).expect("execute");

    // Should succeed — missing config is handled gracefully
    assert!(result.gateway_failed.is_none());

    // Evidence should contain skip reason
    let has_skip_evidence = result
        .evidence
        .all()
        .iter()
        .any(|e| e.gate_id == "F-HF-PARITY-SKIP");
    assert!(has_skip_evidence, "Expected F-HF-PARITY-SKIP evidence");
}

#[test]
fn test_hf_parity_skipped_when_manifest_missing() {
    // When HF parity config points to non-existent corpus
    let mock_runner = MockCommandRunner::new();
    let config = ExecutionConfig {
        run_hf_parity: true,
        hf_parity_corpus_path: Some("/nonexistent/corpus".to_string()),
        hf_parity_model_family: Some("nonexistent-model/v1".to_string()),
        run_conversion_tests: false,
        run_golden_rule_test: false,
        ..Default::default()
    };

    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));
    let yaml = r#"
name: hf-parity-missing-test
version: "1.0.0"
model:
  hf_repo: "test/model"
  formats: [gguf]
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
"#;
    let playbook = Playbook::from_yaml(yaml).expect("parse");
    let result = executor.execute(&playbook).expect("execute");

    // The executor should still succeed, but have failures (1 from parity, plus scenario failures)
    assert!(
        result.failed >= 1,
        "Expected at least 1 failed test for missing manifest"
    );

    // Evidence should contain the manifest not found error
    let has_parity_evidence = result
        .evidence
        .all()
        .iter()
        .any(|e| e.gate_id == "F-HF-PARITY-001");
    assert!(
        has_parity_evidence,
        "Expected F-HF-PARITY-001 evidence for missing manifest"
    );
}

/// Helper: create a temp HF parity corpus with the given manifest content
fn setup_hf_parity_corpus(manifest_json: &str) -> (tempfile::TempDir, String, String) {
    let dir = tempfile::tempdir().expect("create temp dir");
    let model_family = "test-model/v1";
    let family_dir = dir.path().join("test-model").join("v1");
    std::fs::create_dir_all(&family_dir).expect("create family dir");
    std::fs::write(family_dir.join("manifest.json"), manifest_json)
        .expect("write manifest");
    let corpus_path = dir.path().to_string_lossy().to_string();
    (dir, corpus_path, model_family.to_string())
}

/// HF parity: manifest JSON parse failure → F-HF-PARITY-003
#[test]
fn test_hf_parity_manifest_invalid_json() {
    let (dir, corpus_path, model_family) = setup_hf_parity_corpus("not valid json {{");
    let mock_runner = MockCommandRunner::new();
    let config = ExecutionConfig {
        run_hf_parity: true,
        hf_parity_corpus_path: Some(corpus_path),
        hf_parity_model_family: Some(model_family),
        run_conversion_tests: false,
        run_golden_rule_test: false,
        ..Default::default()
    };
    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));
    let model_id = ModelId::new("test", "model");
    let (passed, failed) = executor.run_hf_parity_tests(&model_id);
    assert_eq!(passed, 0);
    assert_eq!(failed, 1);
    let evidence = executor.evidence().all();
    assert!(
        evidence.iter().any(|e| e.gate_id == "F-HF-PARITY-003"),
        "Expected F-HF-PARITY-003 for JSON parse failure"
    );
    drop(dir);
}

/// HF parity: manifest has empty prompts list → skip
#[test]
fn test_hf_parity_manifest_empty_prompts() {
    let (dir, corpus_path, model_family) = setup_hf_parity_corpus(r#"{"prompts":[]}"#);
    let mock_runner = MockCommandRunner::new();
    let config = ExecutionConfig {
        run_hf_parity: true,
        hf_parity_corpus_path: Some(corpus_path),
        hf_parity_model_family: Some(model_family),
        run_conversion_tests: false,
        run_golden_rule_test: false,
        ..Default::default()
    };
    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));
    let model_id = ModelId::new("test", "model");
    let (passed, failed) = executor.run_hf_parity_tests(&model_id);
    assert_eq!(passed, 0);
    assert_eq!(failed, 0);
    let evidence = executor.evidence().all();
    assert!(
        evidence.iter().any(|e| e.gate_id == "F-HF-PARITY-SKIP"),
        "Expected F-HF-PARITY-SKIP for empty prompts"
    );
    drop(dir);
}

/// HF parity: golden .json file missing → F-HF-PARITY-004
#[test]
fn test_hf_parity_golden_file_missing() {
    let (dir, corpus_path, model_family) =
        setup_hf_parity_corpus(r#"{"prompts":["abc123nonexistent"]}"#);
    let mock_runner = MockCommandRunner::new();
    let config = ExecutionConfig {
        run_hf_parity: true,
        hf_parity_corpus_path: Some(corpus_path),
        hf_parity_model_family: Some(model_family),
        run_conversion_tests: false,
        run_golden_rule_test: false,
        ..Default::default()
    };
    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));
    let model_id = ModelId::new("test", "model");
    let (passed, failed) = executor.run_hf_parity_tests(&model_id);
    assert_eq!(passed, 0);
    assert!(failed >= 1);
    let evidence = executor.evidence().all();
    // F-HF-PARITY-004: golden file read failed
    assert!(
        evidence.iter().any(|e| e.gate_id == "F-HF-PARITY-004"),
        "Expected F-HF-PARITY-004 for missing golden file, got: {:?}",
        evidence.iter().map(|e| &e.gate_id).collect::<Vec<_>>()
    );
    drop(dir);
}

/// HF parity: golden .json file exists but has bad JSON → F-HF-PARITY-004
#[test]
fn test_hf_parity_golden_meta_bad_json() {
    let (dir, corpus_path, model_family) = setup_hf_parity_corpus(r#"{"prompts":["hash001"]}"#);
    // Create the golden JSON file with invalid content
    let family_dir = dir.path().join("test-model").join("v1");
    std::fs::write(family_dir.join("hash001.json"), "not json at all {{ bad")
        .expect("write bad golden json");
    let mock_runner = MockCommandRunner::new();
    let config = ExecutionConfig {
        run_hf_parity: true,
        hf_parity_corpus_path: Some(corpus_path),
        hf_parity_model_family: Some(model_family),
        run_conversion_tests: false,
        run_golden_rule_test: false,
        ..Default::default()
    };
    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));
    let model_id = ModelId::new("test", "model");
    let (passed, failed) = executor.run_hf_parity_tests(&model_id);
    assert_eq!(passed, 0);
    assert!(failed >= 1);
    let evidence = executor.evidence().all();
    assert!(
        evidence.iter().any(|e| e.gate_id == "F-HF-PARITY-004"),
        "Expected F-HF-PARITY-004 for bad JSON meta, got: {:?}",
        evidence.iter().map(|e| &e.gate_id).collect::<Vec<_>>()
    );
    drop(dir);
}

/// HF parity: golden meta valid but oracle.load_golden fails (no .safetensors) → F-HF-PARITY-004
#[test]
fn test_hf_parity_oracle_load_golden_fails() {
    let (dir, corpus_path, model_family) = setup_hf_parity_corpus(r#"{"prompts":["hash002"]}"#);
    // Valid golden JSON with a prompt, but NO .safetensors for the prompt's hash
    let family_dir = dir.path().join("test-model").join("v1");
    std::fs::write(family_dir.join("hash002.json"), r#"{"prompt":"test prompt xyz"}"#)
        .expect("write golden json");
    // No .safetensors file → oracle.load_golden fails
    let mock_runner = MockCommandRunner::new();
    let config = ExecutionConfig {
        run_hf_parity: true,
        hf_parity_corpus_path: Some(corpus_path),
        hf_parity_model_family: Some(model_family),
        run_conversion_tests: false,
        run_golden_rule_test: false,
        ..Default::default()
    };
    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));
    let model_id = ModelId::new("test", "model");
    let (passed, failed) = executor.run_hf_parity_tests(&model_id);
    assert_eq!(passed, 0);
    assert!(failed >= 1);
    let evidence = executor.evidence().all();
    // load_golden fails → F-HF-PARITY-004 "Failed to load golden for prompt"
    assert!(
        evidence.iter().any(|e| e.gate_id == "F-HF-PARITY-004"),
        "Expected F-HF-PARITY-004 for oracle load_golden failure, got: {:?}",
        evidence.iter().map(|e| &e.gate_id).collect::<Vec<_>>()
    );
    drop(dir);
}

// ============================================================
// G0-FORMAT Workspace Tests
// ============================================================

#[test]
fn test_workspace_creates_directory_structure() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let output_dir = dir.path().join("output");

    // Create a fake safetensors file
    let model_file = dir.path().join("abc123.safetensors");
    std::fs::write(&model_file, b"fake-safetensors-content").expect("write model");

    let mock_runner = MockCommandRunner::new();
    let config = ExecutionConfig {
        output_dir: Some(output_dir.to_string_lossy().to_string()),
        run_conversion_tests: false,
        run_golden_rule_test: false,
        run_contract_tests: false,
        ..Default::default()
    };

    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));
    let model_id = ModelId::new("test", "model");
    let formats = vec![Format::SafeTensors, Format::Apr];

    let (workspace, passed, _failed) =
        executor.prepare_model_workspace(&model_file, &model_id, &formats);

    // Verify workspace directory was created
    let ws_path = Path::new(&workspace);
    assert!(ws_path.exists(), "Workspace directory should exist");

    // Verify safetensors subdir exists with symlinked model
    let st_dir = ws_path.join("safetensors");
    assert!(st_dir.exists(), "safetensors subdir should exist");
    let st_link = st_dir.join("model.safetensors");
    assert!(st_link.exists(), "model.safetensors symlink should exist");

    // Verify APR subdir was created with converted model
    let apr_dir = ws_path.join("apr");
    assert!(apr_dir.exists(), "apr subdir should exist");

    // MockCommandRunner.convert_model returns success, so conversion passed
    assert!(passed >= 1, "At least one format conversion should pass");
}

#[test]
fn test_workspace_symlinks_config_files() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let output_dir = dir.path().join("output");

    // Create model file and sibling config files (pacha cache naming)
    let model_file = dir.path().join("abc123.safetensors");
    std::fs::write(&model_file, b"fake-model").expect("write model");
    std::fs::write(
        dir.path().join("abc123.config.json"),
        r#"{"num_hidden_layers": 24}"#,
    )
    .expect("write config");
    std::fs::write(
        dir.path().join("abc123.tokenizer.json"),
        r#"{"version": "1.0"}"#,
    )
    .expect("write tokenizer");

    let mock_runner = MockCommandRunner::new();
    let config = ExecutionConfig {
        output_dir: Some(output_dir.to_string_lossy().to_string()),
        run_conversion_tests: false,
        run_golden_rule_test: false,
        run_contract_tests: false,
        ..Default::default()
    };

    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));
    let model_id = ModelId::new("test", "model");
    let formats = vec![Format::SafeTensors];

    let (workspace, _passed, _failed) =
        executor.prepare_model_workspace(&model_file, &model_id, &formats);

    let ws_path = Path::new(&workspace);
    let st_dir = ws_path.join("safetensors");

    // Verify config files were symlinked with canonical names
    assert!(
        st_dir.join("config.json").exists(),
        "config.json should be symlinked"
    );
    assert!(
        st_dir.join("tokenizer.json").exists(),
        "tokenizer.json should be symlinked"
    );
}

#[test]
fn test_workspace_conversion_failure_nonfatal() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let output_dir = dir.path().join("output");

    let model_file = dir.path().join("test.safetensors");
    std::fs::write(&model_file, b"fake-model").expect("write model");

    // Use a mock runner where conversion fails
    let mock_runner = MockCommandRunner::new().with_convert_failure();
    let config = ExecutionConfig {
        output_dir: Some(output_dir.to_string_lossy().to_string()),
        run_conversion_tests: false,
        run_golden_rule_test: false,
        run_contract_tests: false,
        ..Default::default()
    };

    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));
    let model_id = ModelId::new("test", "model");
    let formats = vec![Format::SafeTensors, Format::Apr, Format::Gguf];

    let (workspace, passed, failed) =
        executor.prepare_model_workspace(&model_file, &model_id, &formats);

    // Workspace should still be created
    assert!(
        Path::new(&workspace).exists(),
        "Workspace should exist even with conversion failures"
    );
    // SafeTensors subdir should exist
    assert!(
        Path::new(&workspace).join("safetensors").exists(),
        "safetensors dir should exist"
    );

    // Conversions should have failed (APR + GGUF = 2 failures)
    assert_eq!(passed, 0, "No conversions should pass");
    assert_eq!(failed, 2, "Both APR and GGUF conversions should fail");

    // Verify evidence was collected for failures
    let evidence = executor.evidence().all();
    let apr_evidence = evidence.iter().any(|e| e.gate_id == "G0-FORMAT-APR-001");
    let gguf_evidence = evidence.iter().any(|e| e.gate_id == "G0-FORMAT-GGUF-001");
    assert!(apr_evidence, "Should have G0-FORMAT-APR-001 evidence");
    assert!(gguf_evidence, "Should have G0-FORMAT-GGUF-001 evidence");
}

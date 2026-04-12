#[test]
fn test_workspace_skipped_for_directory() {
    // When model_path is already a directory, workspace creation should be skipped
    let mock_runner = MockCommandRunner::new();
    let config = ExecutionConfig {
        model_path: Some("/some/directory/path".to_string()),
        run_conversion_tests: false,
        run_golden_rule_test: false,
        run_contract_tests: false,
        ..Default::default()
    };

    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));
    let yaml = r#"
name: workspace-skip-test
version: "1.0.0"
model:
  hf_repo: "test/model"
  formats: [safetensors, apr]
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
"#;
    let playbook = Playbook::from_yaml(yaml).expect("parse");
    let result = executor.execute(&playbook).expect("execute");

    // No G0-FORMAT evidence should be present (workspace was skipped)
    let has_format_evidence = result
        .evidence
        .all()
        .iter()
        .any(|e| e.gate_id.starts_with("G0-FORMAT"));
    assert!(
        !has_format_evidence,
        "No G0-FORMAT evidence expected for directory model path"
    );
}

#[test]
fn test_workspace_evidence_emitted() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let output_dir = dir.path().join("output");

    let model_file = dir.path().join("test.safetensors");
    std::fs::write(&model_file, b"fake-model").expect("write model");

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
    let formats = vec![Format::SafeTensors, Format::Apr, Format::Gguf];

    let (_workspace, passed, failed) =
        executor.prepare_model_workspace(&model_file, &model_id, &formats);

    // Both APR and GGUF conversions should produce evidence
    assert_eq!(passed + failed, 2, "Should have evidence for APR and GGUF");

    let evidence = executor.evidence().all();
    let format_evidence_count = evidence
        .iter()
        .filter(|e| e.gate_id.starts_with("G0-FORMAT"))
        .count();
    assert_eq!(
        format_evidence_count, 2,
        "Should have 2 G0-FORMAT evidence entries"
    );
}

#[test]
fn test_find_sibling_model_files() {
    let dir = tempfile::tempdir().expect("create temp dir");

    // Create pacha cache structure
    let model_file = dir.path().join("abc123.safetensors");
    std::fs::write(&model_file, b"model").expect("write");
    std::fs::write(dir.path().join("abc123.config.json"), b"config").expect("write");
    std::fs::write(dir.path().join("abc123.tokenizer.json"), b"tokenizer").expect("write");
    // Different model (should be excluded)
    std::fs::write(dir.path().join("def456.safetensors"), b"other").expect("write");
    std::fs::write(dir.path().join("def456.config.json"), b"other-config").expect("write");

    let siblings = Executor::find_sibling_model_files(&model_file);

    // Should find config.json and tokenizer.json for abc123 only
    assert_eq!(siblings.len(), 2, "Should find exactly 2 sibling files");

    let canonical_names: Vec<&str> = siblings.iter().map(|(_, n)| n.as_str()).collect();
    assert!(
        canonical_names.contains(&"config.json"),
        "Should find config.json"
    );
    assert!(
        canonical_names.contains(&"tokenizer.json"),
        "Should find tokenizer.json"
    );
}

#[test]
fn test_find_sibling_model_files_no_siblings() {
    let dir = tempfile::tempdir().expect("create temp dir");

    let model_file = dir.path().join("lonely.safetensors");
    std::fs::write(&model_file, b"model").expect("write");

    let siblings = Executor::find_sibling_model_files(&model_file);
    assert!(siblings.is_empty(), "Should find no siblings");
}

#[test]
fn test_find_sibling_model_files_non_safetensors() {
    let dir = tempfile::tempdir().expect("create temp dir");

    let model_file = dir.path().join("model.gguf");
    std::fs::write(&model_file, b"model").expect("write");

    let siblings = Executor::find_sibling_model_files(&model_file);
    assert!(
        siblings.is_empty(),
        "Should return empty for non-safetensors files"
    );
}

#[test]
fn test_workspace_execute_integration_with_single_file() {
    // Integration test: execute() with a real single .safetensors file
    // should trigger workspace creation and resolve all formats
    let dir = tempfile::tempdir().expect("create temp dir");
    let output_dir = dir.path().join("output");

    let model_file = dir.path().join("test.safetensors");
    std::fs::write(&model_file, b"fake-model").expect("write model");

    let mock_runner =
        MockCommandRunner::new().with_pull_model_path(model_file.to_string_lossy().to_string());
    let config = ExecutionConfig {
        model_path: Some(model_file.to_string_lossy().to_string()),
        output_dir: Some(output_dir.to_string_lossy().to_string()),
        run_conversion_tests: false,
        run_golden_rule_test: false,
        run_contract_tests: false,
        ..Default::default()
    };

    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));
    let yaml = r#"
name: workspace-integration
version: "1.0.0"
model:
  hf_repo: "test/model"
  formats: [safetensors, apr]
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
"#;
    let playbook = Playbook::from_yaml(yaml).expect("parse");
    let result = executor.execute(&playbook).expect("execute");

    // Verify the model_path was changed from file to workspace directory
    let final_model_path = executor.config().model_path.as_deref().unwrap_or("");
    assert!(
        final_model_path.contains("workspace"),
        "model_path should point to workspace: {final_model_path}"
    );
    assert!(
        !final_model_path.ends_with(".safetensors"),
        "model_path should not be a file: {final_model_path}"
    );

    // G0-FORMAT evidence should be present (conversion to APR)
    let has_format_evidence = result
        .evidence
        .all()
        .iter()
        .any(|e| e.gate_id.starts_with("G0-FORMAT"));
    assert!(
        has_format_evidence,
        "Should have G0-FORMAT evidence for APR conversion"
    );
}

// ── G0-TENSOR Template Validation Tests (PMAT-271) ─────────────────────────

#[test]
fn test_g0_tensor_no_family_configured() {
    // When family/size_variant are not set, G0-TENSOR should be skipped (0, 0)
    let mock_runner = MockCommandRunner::new();
    let dir = make_temp_model_dir();

    let config = ExecutionConfig {
        model_path: Some(dir.path().to_string_lossy().to_string()),
        run_conversion_tests: false,
        run_golden_rule_test: false,
        run_contract_tests: false,
        ..Default::default()
    };

    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));

    // Playbook without family/size_variant
    let yaml = r#"
name: no-family-test
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

    // No G0-TENSOR evidence when family not configured
    let has_tensor_evidence = result
        .evidence
        .all()
        .iter()
        .any(|e| e.gate_id == "G0-TENSOR-001");
    assert!(
        !has_tensor_evidence,
        "Should NOT have G0-TENSOR evidence when family not configured"
    );
}

#[test]
fn test_g0_tensor_family_contract_not_found() {
    // When family is set but contract doesn't exist, should skip gracefully
    let mock_runner = MockCommandRunner::new();
    let dir = make_temp_model_dir();

    let config = ExecutionConfig {
        model_path: Some(dir.path().to_string_lossy().to_string()),
        run_conversion_tests: false,
        run_golden_rule_test: false,
        run_contract_tests: false,
        ..Default::default()
    };

    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));
    let model_id = ModelId::new("test", "model");

    // Call with a nonexistent family
    let (passed, failed) = executor.run_g0_tensor_template_check(
        dir.path(),
        &model_id,
        "nonexistent-family",
        "1b",
        Some("/nonexistent/path"),
    );

    // Should skip (0, 0) with evidence
    assert_eq!(passed, 0);
    assert_eq!(failed, 0);

    let evidence = executor.evidence().all();
    let tensor_ev = evidence
        .iter()
        .find(|e| e.gate_id == "G0-TENSOR-001")
        .expect("should have G0-TENSOR evidence");
    // Evidence::skipped puts message in reason field (not output)
    assert!(tensor_ev.reason.contains("G0 SKIP"));
    assert!(tensor_ev.reason.contains("Family contract not found"));
}

#[test]
fn test_g0_tensor_no_safetensors_files() {
    // When there are no safetensors files, should skip
    let mock_runner = MockCommandRunner::new();
    let dir = tempfile::TempDir::new().expect("create temp dir");

    let config = ExecutionConfig {
        model_path: Some(dir.path().to_string_lossy().to_string()),
        run_conversion_tests: false,
        run_golden_rule_test: false,
        run_contract_tests: false,
        ..Default::default()
    };

    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));
    let model_id = ModelId::new("test", "model");

    // Call with a valid family name but empty directory
    let (passed, failed) = executor.run_g0_tensor_template_check(
        dir.path(),
        &model_id,
        "qwen2",
        "0.5b",
        Some("/nonexistent/path"), // Will fail to load, but we also don't have safetensors
    );

    // Should skip (0, 0)
    assert_eq!(passed, 0);
    assert_eq!(failed, 0);
}

#[test]
fn test_g0_tensor_inspect_returns_empty_names() {
    // When inspect doesn't return tensor names, should skip
    let mock_runner = MockCommandRunner::new().with_tensor_names(vec![]); // Empty tensor names
    let dir = make_temp_model_dir();

    let config = ExecutionConfig {
        model_path: Some(dir.path().to_string_lossy().to_string()),
        run_conversion_tests: false,
        run_golden_rule_test: false,
        run_contract_tests: false,
        ..Default::default()
    };

    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));
    let model_id = ModelId::new("test", "model");

    // This will fail at registry load since aprender isn't available in tests,
    // but this tests the empty tensor_names path in isolation
    let (passed, failed) = executor.run_g0_tensor_template_check(
        dir.path(),
        &model_id,
        "qwen2",
        "0.5b",
        Some("/nonexistent/path"),
    );

    // Should skip
    assert_eq!(passed, 0);
    assert_eq!(failed, 0);
}

#[test]
fn test_g0_tensor_inspect_failure() {
    // When inspect fails, should report failure
    let mock_runner = MockCommandRunner::new().with_inspect_json_failure();
    let dir = make_temp_model_dir();

    // Create a temp contracts directory with a minimal family contract
    let contracts_dir = tempfile::TempDir::new().expect("create contracts dir");
    let family_yaml = r#"
family: testfamily
size_variants:
  1b:
    parameters: "1B"
    hidden_dim: 1024
    num_layers: 12
    num_heads: 8
tensor_template:
  embedding: "embed.weight"
"#;
    std::fs::write(contracts_dir.path().join("testfamily.yaml"), family_yaml)
        .expect("write family yaml");

    let config = ExecutionConfig {
        model_path: Some(dir.path().to_string_lossy().to_string()),
        run_conversion_tests: false,
        run_golden_rule_test: false,
        run_contract_tests: false,
        ..Default::default()
    };

    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));
    let model_id = ModelId::new("test", "model");

    let (passed, failed) = executor.run_g0_tensor_template_check(
        dir.path(),
        &model_id,
        "testfamily",
        "1b",
        Some(contracts_dir.path().to_str().expect("path")),
    );

    // Should fail
    assert_eq!(passed, 0);
    assert_eq!(failed, 1);

    let evidence = executor.evidence().all();
    let tensor_ev = evidence
        .iter()
        .find(|e| e.gate_id == "G0-TENSOR-001")
        .expect("should have G0-TENSOR evidence");
    assert!(tensor_ev.reason.contains("G0 FAIL"));
    assert!(tensor_ev.reason.contains("Could not inspect"));
}

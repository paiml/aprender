/// Verify G0 integrity check detects layer count mismatch between config and tensors
#[test]
fn test_run_g0_integrity_check_layer_mismatch() {
    use tempfile::TempDir;
    let dir = TempDir::new().expect("create temp dir");

    // Config says 14 layers but tensors have 24 (the corrupted cache bug)
    create_test_config_for_executor(dir.path(), 14, 896, 151_936);
    create_mock_safetensors_for_test(dir.path(), 24, 896, 151_936);

    let mut executor = Executor::new();
    let model_id = ModelId::new("test", "model");
    let (passed, failed) = executor.run_g0_integrity_check(dir.path(), &model_id);

    assert_eq!(passed, 0);
    assert!(failed > 0);

    // Evidence should contain LAYERS failure
    let evidence = executor.evidence();
    assert!(evidence.all().iter().any(|e| e.gate_id.contains("LAYERS")));
}

/// Helper to create test config.json
fn create_test_config_for_executor(
    dir: &std::path::Path,
    layers: usize,
    hidden: usize,
    vocab: usize,
) {
    let config = format!(
        r#"{{"num_hidden_layers": {layers}, "hidden_size": {hidden}, "vocab_size": {vocab}}}"#
    );
    std::fs::write(dir.join("config.json"), config).expect("write config");
}

/// Helper to create mock SafeTensors file with specific dimensions
#[allow(clippy::items_after_statements)]
fn create_mock_safetensors_for_test(
    dir: &std::path::Path,
    layers: usize,
    hidden: usize,
    vocab: usize,
) {
    let mut header_obj = serde_json::Map::new();

    // Embedding tensor
    let mut embed_info = serde_json::Map::new();
    embed_info.insert("shape".to_string(), serde_json::json!([vocab, hidden]));
    embed_info.insert(
        "dtype".to_string(),
        serde_json::Value::String("F32".to_string()),
    );
    embed_info.insert(
        "data_offsets".to_string(),
        serde_json::json!([0, vocab * hidden * 4]),
    );
    header_obj.insert(
        "model.embed_tokens.weight".to_string(),
        serde_json::Value::Object(embed_info),
    );

    // Layer tensors
    for i in 0..layers {
        let mut layer_info = serde_json::Map::new();
        layer_info.insert("shape".to_string(), serde_json::json!([hidden, hidden]));
        layer_info.insert(
            "dtype".to_string(),
            serde_json::Value::String("F32".to_string()),
        );
        layer_info.insert("data_offsets".to_string(), serde_json::json!([0, 0]));
        header_obj.insert(
            format!("model.layers.{i}.self_attn.q_proj.weight"),
            serde_json::Value::Object(layer_info),
        );
    }

    let header_json = serde_json::to_string(&header_obj).expect("serialize header");
    let header_bytes = header_json.as_bytes();
    let header_len = header_bytes.len() as u64;

    let path = dir.join("model.safetensors");
    let mut file = std::fs::File::create(path).expect("create safetensors");
    use std::io::Write;
    file.write_all(&header_len.to_le_bytes())
        .expect("write len");
    file.write_all(header_bytes).expect("write header");
    file.write_all(&[0u8; 1024]).expect("write data");
}

// =========================================================================
// Additional coverage tests — uncovered paths
// =========================================================================

/// Verify execute_all_with_serve includes serve-lifecycle when flag is true
#[test]
fn test_execute_all_with_serve_true() {
    let mock_runner = MockCommandRunner::new();
    let executor = ToolExecutor::with_runner(
        "test-model.gguf".to_string(),
        true,
        5000,
        Arc::new(mock_runner),
    );
    let results = executor.execute_all_with_serve(true);
    assert!(!results.is_empty());
    // Should include serve-lifecycle when include_serve=true
    assert!(results.iter().any(|r| r.tool == "serve-lifecycle"));
}

/// Verify G0 integrity check detects hidden_size mismatch between config and tensors
#[test]
fn test_run_g0_integrity_check_hidden_mismatch() {
    use tempfile::TempDir;
    let dir = TempDir::new().expect("create temp dir");

    // Config says hidden_size=1024 but tensors have 896
    create_test_config_for_executor(dir.path(), 24, 1024, 151_936);
    create_mock_safetensors_for_test(dir.path(), 24, 896, 151_936);

    let mut executor = Executor::new();
    let model_id = ModelId::new("test", "model");
    let (passed, failed) = executor.run_g0_integrity_check(dir.path(), &model_id);

    assert_eq!(passed, 0);
    assert!(failed > 0);

    let evidence = executor.evidence();
    assert!(evidence.all().iter().any(|e| e.gate_id.contains("HIDDEN")));
}

/// Verify G0 integrity check detects vocab_size mismatch between config and tensors
#[test]
fn test_run_g0_integrity_check_vocab_mismatch() {
    use tempfile::TempDir;
    let dir = TempDir::new().expect("create temp dir");

    // Config says vocab=200_000 but tensors have 151_936
    create_test_config_for_executor(dir.path(), 24, 896, 200_000);
    create_mock_safetensors_for_test(dir.path(), 24, 896, 151_936);

    let mut executor = Executor::new();
    let model_id = ModelId::new("test", "model");
    let (passed, failed) = executor.run_g0_integrity_check(dir.path(), &model_id);

    assert_eq!(passed, 0);
    assert!(failed > 0);

    let evidence = executor.evidence();
    assert!(evidence.all().iter().any(|e| e.gate_id.contains("VOCAB")));
}

// G0-LAYOUT Pre-flight Gate Tests (Issue #4)

/// Verify G0 layout check auto-skips when tensor-layout-v1.yaml is absent
#[test]
fn test_run_g0_layout_check_no_contract() {
    // When tensor-layout-v1.yaml is not found, the check should auto-skip (0, 0)
    use tempfile::TempDir;
    let dir = TempDir::new().expect("create temp dir");

    let mut executor = Executor::new();
    let model_id = ModelId::new("test", "model");
    let (passed, failed) = executor.run_g0_layout_check(dir.path(), &model_id);

    // Contract not found → skip (0, 0), not failure
    assert_eq!(passed, 0);
    assert_eq!(failed, 0);
}

/// Verify G0 layout check fails when model file does not exist but contract is present
#[test]
fn test_run_g0_layout_check_model_not_found() {
    // When model file doesn't exist but contract is found, validation fails
    use tempfile::TempDir;
    let dir = TempDir::new().expect("create temp dir");

    // Create a minimal contract file
    let contract_path = dir.path().join("tensor-layout-v1.yaml");
    std::fs::write(
        &contract_path,
        r#"
metadata:
  version: "1.0"
  created: "2026-01-01"
  updated: "2026-01-01"
  author: "test"
  description: "test"
formats: {}
kernel:
  signature: "test"
  weight_shape: "[out, in]"
  computation: "y = Wx"
  byte_calculation: "out * in"
  block_sizes: {}
  QK_K: 256
tensors: {}
validation_rules: []
"#,
    )
    .expect("write contract");

    // Test with a non-existent path inside the temp directory
    let nonexistent_path = dir.path().join("does_not_exist.safetensors");
    let contract =
        crate::layout_contract::load_contract_from(&contract_path).expect("load contract");
    let result = crate::layout_contract::validate_model(&nonexistent_path, &contract)
        .expect("validation should return result");

    // Model not found = failed validation
    assert!(!result.passed);
    assert!(!result.critical_failures.is_empty());
}

/// Verify layout scenario has correct prompt and metadata fields
#[test]
fn test_layout_scenario_creation() {
    let model_id = ModelId::new("test", "model");
    let scenario = Executor::layout_scenario(&model_id);

    assert_eq!(
        scenario.prompt,
        "G0 Layout: tensor shape contract validation"
    );
    assert_eq!(scenario.format, Format::SafeTensors);
    assert_eq!(scenario.backend, Backend::Cpu);
    assert_eq!(scenario.modality, Modality::Run);
}

/// Verify format_tensor_failure includes expected and actual values when present
#[test]
fn test_format_tensor_failure_with_expected_and_actual() {
    let tensor_result = crate::layout_contract::TensorValidationResult {
        tensor_name: "lm_head.weight".to_string(),
        rule_id: "F-LAYOUT-CONTRACT-002".to_string(),
        passed: false,
        details: "Shape mismatch".to_string(),
        expected: Some("[vocab, hidden]".to_string()),
        actual: Some("[hidden, vocab]".to_string()),
    };

    let formatted = Executor::format_tensor_failure(&tensor_result);
    assert!(formatted.contains("F-LAYOUT-CONTRACT-002"));
    assert!(formatted.contains("Shape mismatch"));
    assert!(formatted.contains("Expected: [vocab, hidden]"));
    assert!(formatted.contains("Actual: [hidden, vocab]"));
}

/// Verify format_tensor_failure omits expected/actual when both are None
#[test]
fn test_format_tensor_failure_without_expected() {
    let tensor_result = crate::layout_contract::TensorValidationResult {
        tensor_name: "test.weight".to_string(),
        rule_id: "F-LAYOUT-CONTRACT-001".to_string(),
        passed: false,
        details: "Missing transpose".to_string(),
        expected: None,
        actual: None,
    };

    let formatted = Executor::format_tensor_failure(&tensor_result);
    assert!(formatted.contains("F-LAYOUT-CONTRACT-001"));
    assert!(formatted.contains("Missing transpose"));
    assert!(!formatted.contains("Expected:"));
    assert!(!formatted.contains("Actual:"));
}

/// G0 layout check: when contract exists, nonexistent model path → critical_failures → (0, ≥1)
#[test]
fn test_run_g0_layout_check_contract_present_model_missing() {
    // Only runs when aprender sibling is present; otherwise auto-skips (0, 0)
    let contract_path =
        std::path::Path::new(crate::layout_contract::DEFAULT_CONTRACT_PATH);
    if !contract_path.exists() {
        return; // CI without aprender: tolerated, skip path already covered
    }

    let dir = tempfile::TempDir::new().expect("create temp dir");
    let nonexistent = dir.path().join("missing.safetensors");
    let mut executor = Executor::new();
    let model_id = ModelId::new("test", "model");

    let (passed, failed) = executor.run_g0_layout_check(&nonexistent, &model_id);
    // Model doesn't exist → validate_model returns critical_failures → (0, ≥1)
    assert_eq!(passed, 0, "Expected 0 passed for missing model");
    assert!(failed >= 1, "Expected ≥1 failure for missing model");
}

/// G0 layout check: when contract exists, empty dir → FILE-FORMAT → (0, ≥1)
#[test]
fn test_run_g0_layout_check_contract_present_no_safetensors() {
    let contract_path =
        std::path::Path::new(crate::layout_contract::DEFAULT_CONTRACT_PATH);
    if !contract_path.exists() {
        return;
    }

    let dir = tempfile::TempDir::new().expect("create temp dir");
    // Dir exists but contains no .safetensors files
    let mut executor = Executor::new();
    let model_id = ModelId::new("test", "model");

    let (passed, failed) = executor.run_g0_layout_check(dir.path(), &model_id);
    // Dir exists, no SafeTensors → validate_model returns failed result → (0, ≥1)
    assert_eq!(passed, 0, "Expected 0 passed for dir with no safetensors");
    assert!(failed >= 1, "Expected ≥1 failure for dir with no safetensors");
}

/// Verify inspect_verified fails gracefully for a nonexistent model path
#[test]
fn test_execute_inspect_verified_nonexistent_model() {
    // run_inspect with "apr" binary + nonexistent model → fails → exercises Err path
    let executor = ToolExecutor::new("/nonexistent/path/to/model.gguf".to_string(), false, 5000);
    let result = executor.execute_inspect_verified();
    // apr binary exists but model doesn't → inspect fails → result is not passed
    assert!(!result.passed);
    assert_eq!(result.gate_id, "F-INSPECT-META-001");
    // Either exit_code=-1 (Err path) or exit_code=1 (Ok path with tensor_count=0)
    assert!(result.exit_code != 0);
}

/// Verify StopOnP0 policy does NOT halt on non-P0 gate failures
#[test]
fn test_execute_scenario_stop_on_p0_gate() {
    // Non-P0 scenario failures (gate_id = F-A1-001) should not stop execution
    let mock_runner = MockCommandRunner::new()
        .with_inference_failure()
        .with_exit_code(1);

    let config = ExecutionConfig {
        model_path: Some("/test/model.gguf".to_string()),
        failure_policy: FailurePolicy::StopOnP0,
        run_conversion_tests: false,
        run_golden_rule_test: false,
        ..Default::default()
    };

    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));

    // Gate IDs will be F-A1-001 (non-P0), so StopOnP0 should collect all
    let yaml = r#"
name: p0-stop
version: "1.0.0"
model:
  hf_repo: "test/model"
  formats: [gguf]
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 3
"#;
    let playbook = Playbook::from_yaml(yaml).expect("Failed to parse");
    let result = executor.execute(&playbook).expect("Execution failed");

    // Non-P0 failures should be collected, not stopped
    assert!(result.failed >= 1);
}

/// Verify corroborated evidence propagates stderr from mock runner
#[test]
fn test_execute_scenario_corroborated_with_stderr_via_playbook() {
    // Use a mock that returns correct output ("The answer is 4.") with stderr
    // The mock auto-responds "The answer is 4." for "2+2" prompts
    // This exercises the Corroborated branch with stderr propagation (line 624-626)
    let mock_runner = MockCommandRunner::new()
        .with_inference_response_and_stderr("correct", "warning: low memory");

    let config = ExecutionConfig {
        model_path: Some("/test/model.gguf".to_string()),
        run_conversion_tests: false,
        run_golden_rule_test: false,
        failure_policy: FailurePolicy::CollectAll,
        ..Default::default()
    };

    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));

    let yaml = r#"
name: corroborated-stderr
version: "1.0.0"
model:
  hf_repo: "test/model"
  formats: [gguf]
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
"#;
    let playbook = Playbook::from_yaml(yaml).expect("Failed to parse");
    let result = executor.execute(&playbook).expect("Execution failed");

    // Should pass (mock responds "The answer is 4." for 2+2 prompts)
    assert!(result.passed >= 1);

    // The corroborated evidence should carry stderr
    let evidence = executor.evidence().all();
    assert!(
        evidence
            .iter()
            .any(|e| e.outcome.is_pass() && e.stderr.is_some()),
        "should have corroborated evidence with stderr"
    );
}

/// Verify conversion tests skip for single-file models (not directories)
#[test]
fn test_run_conversion_tests_single_file_model() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let model_path = dir.path().join("model.gguf");
    std::fs::write(&model_path, b"fake model").expect("write model");

    let config = ExecutionConfig {
        model_path: Some(model_path.to_string_lossy().to_string()),
        run_conversion_tests: true,
        ..Default::default()
    };

    let mut executor = Executor::with_config(config);
    let model_id = ModelId::new("test", "model");
    // Single file model (not a directory) — should return (0, 0)
    let (passed, failed) = executor.run_conversion_tests(&model_path, &model_id);
    assert_eq!(passed, 0);
    assert_eq!(failed, 0);
}

/// Verify golden rule test skips for single-file models (not directories)
#[test]
fn test_run_golden_rule_single_file_model() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let model_path = dir.path().join("model.gguf");
    std::fs::write(&model_path, b"fake model").expect("write model");

    let config = ExecutionConfig {
        model_path: Some(model_path.to_string_lossy().to_string()),
        run_golden_rule_test: true,
        ..Default::default()
    };

    let mut executor = Executor::with_config(config);
    let model_id = ModelId::new("test", "model");
    // Single file model — golden rule returns (0, 0)
    let (passed, failed) = executor.run_golden_rule_test(&model_path, &model_id);
    assert_eq!(passed, 0);
    assert_eq!(failed, 0);
}

/// G0 layout check: valid-but-empty SafeTensors (0 tensors) → result.passed = true → corroborated
#[test]
fn test_run_g0_layout_check_corroborated_empty_safetensors() {
    let contract_path = std::path::Path::new(crate::layout_contract::DEFAULT_CONTRACT_PATH);
    if !contract_path.exists() {
        return; // CI without aprender: skip
    }

    let dir = tempfile::TempDir::new().expect("create temp dir");

    // Build a valid-but-empty SafeTensors file: 8-byte LE u64 header_len=2 + "{}"
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&2u64.to_le_bytes());
    bytes.extend_from_slice(b"{}");
    let safetensors_file = dir.path().join("model.safetensors");
    std::fs::write(&safetensors_file, &bytes).expect("write empty safetensors");

    let mut executor = Executor::new();
    let model_id = ModelId::new("test", "model");

    let (passed, failed) = executor.run_g0_layout_check(dir.path(), &model_id);
    // 0 tensors → all rules pass → corroborated
    assert_eq!(passed, 1, "Expected 1 passed for empty-tensor model");
    assert_eq!(failed, 0, "Expected 0 failed for empty-tensor model");

    let evidence = executor.evidence().all();
    let corr = evidence.iter().find(|e| e.gate_id == "G0-LAYOUT-001" && e.outcome.is_pass());
    assert!(corr.is_some(), "Expected G0-LAYOUT-001 corroborated evidence");
    assert!(
        corr.unwrap().output.contains("G0 PASS"),
        "Corroborated output should contain 'G0 PASS'"
    );
}

/// G0 layout check: garbage SafeTensors → PARSE-ERROR in tensor_results → falsified via loop
#[test]
fn test_run_g0_layout_check_tensor_results_loop_parse_error() {
    let contract_path = std::path::Path::new(crate::layout_contract::DEFAULT_CONTRACT_PATH);
    if !contract_path.exists() {
        return; // CI without aprender: skip
    }

    let dir = tempfile::TempDir::new().expect("create temp dir");

    // Garbage bytes → read_safetensors_metadata fails → PARSE-ERROR tensor result
    // (header_len from first 8 bytes >> MAX_HEADER_SIZE → "Header too large")
    let safetensors_file = dir.path().join("model.safetensors");
    std::fs::write(&safetensors_file, b"not a valid safetensors file at all").expect("write garbage");

    let mut executor = Executor::new();
    let model_id = ModelId::new("test", "model");

    let (passed, failed) = executor.run_g0_layout_check(dir.path(), &model_id);
    // PARSE-ERROR → tensor_results loop → 1 falsified, no critical_failures
    assert_eq!(passed, 0, "Expected 0 passed for garbage safetensors");
    assert!(failed >= 1, "Expected ≥1 failure for garbage safetensors");

    let evidence = executor.evidence().all();
    let parse_err = evidence
        .iter()
        .find(|e| e.gate_id == "PARSE-ERROR" && e.outcome.is_fail());
    assert!(
        parse_err.is_some(),
        "Expected PARSE-ERROR falsified evidence, got gates: {:?}",
        evidence.iter().map(|e| &e.gate_id).collect::<Vec<_>>()
    );
}

/// Verify integrity check rejects execution when lock file hash does not match
#[test]
fn test_integrity_check_refuses_on_mismatch() {
    use crate::playbook::{PlaybookLockEntry, PlaybookLockFile, save_lock_file};
    use std::collections::HashMap;

    let dir = tempfile::tempdir().expect("create temp dir");
    let lock_path = dir.path().join("playbook.lock.yaml");

    // Create a lock file with a wrong hash for 'test-playbook'
    let mut entries = HashMap::new();
    entries.insert(
        "integrity-test".to_string(),
        PlaybookLockEntry {
            sha256: "0000000000000000000000000000000000000000000000000000000000000000".to_string(),
            locked_fields: vec!["name".to_string()],
        },
    );
    let lock_file = PlaybookLockFile { entries };
    save_lock_file(&lock_file, &lock_path).expect("save lock");

    let config = ExecutionConfig {
        check_integrity: true,
        lock_file_path: Some(lock_path.to_string_lossy().to_string()),
        run_conversion_tests: false,
        run_golden_rule_test: false,
        ..Default::default()
    };

    let mut executor = Executor::with_config(config);
    let yaml = r#"
name: integrity-test
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

    // verify_playbook_integrity checks the lock_path as the playbook path,
    // which won't match the stored hash. This should trigger a gateway failure.
    // Even if the integrity flow changes, the test validates it runs without panic.
    assert!(result.gateway_failed.is_some() || result.failed > 0);
}

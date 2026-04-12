#[test]
fn test_run_all_validations_end_to_end() {
    let dir = tempfile::tempdir().unwrap();

    // Create safetensors with lm_head + layer tensors + 1D tensors
    create_test_safetensors(
        &dir.path().join("model.safetensors"),
        &[
            ("lm_head.weight", &[32000, 4096]),
            ("model.layers.0.self_attn.q_proj.weight", &[4096, 4096]),
            ("model.layers.0.input_layernorm.weight", &[4096]),
            ("model.norm.weight", &[4096]),
        ],
    );

    // Create config.json
    let config = serde_json::json!({
        "vocab_size": 32000,
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "num_hidden_layers": 1
    });
    std::fs::write(
        dir.path().join("config.json"),
        serde_json::to_string(&config).unwrap(),
    )
    .unwrap();

    // Build a contract with real tensor specs
    let mut contract = make_contract();
    contract.tensors.insert(
        "lm_head".to_string(),
        TensorSpec {
            gguf_name: "output.weight".to_string(),
            apr_name: "lm_head.weight".to_string(),
            gguf_shape: "[hidden, vocab]".to_string(),
            apr_shape: "[vocab, hidden]".to_string(),
            transpose: true,
            kernel: "matmul".to_string(),
            kernel_out_dim: Some("vocab_size".to_string()),
            kernel_in_dim: Some("hidden_dim".to_string()),
            validation: None,
            critical: true,
            note: Some("GH-202".to_string()),
        },
    );
    contract.tensors.insert(
        "q_proj".to_string(),
        make_spec(
            "model.layers.{n}.self_attn.q_proj.weight",
            "[heads*head_dim, hidden]",
            true,
        ),
    );
    contract.tensors.insert(
        "input_layernorm".to_string(),
        make_spec("model.layers.{n}.input_layernorm.weight", "[hidden]", false),
    );
    contract.tensors.insert(
        "final_norm".to_string(),
        make_spec("model.norm.weight", "[hidden]", false),
    );

    let (results, critical_failures) = run_all_validations(dir.path(), &contract);

    // lm_head should pass
    assert!(critical_failures.is_empty());
    // Results: lm_head via validate_lm_head (1) + lm_head via validate_2d_tensors (1)
    //        + q_proj layer 0 (1) + layernorm layer 0 (1) + final_norm (1) = 5
    assert_eq!(results.len(), 5);
    assert!(
        results.iter().all(|r| r.passed),
        "All results should pass: {:?}",
        results
    );
}

#[test]
fn test_run_all_validations_with_critical_failure() {
    let dir = tempfile::tempdir().unwrap();

    // lm_head shape is transposed => should fail
    create_test_safetensors(
        &dir.path().join("model.safetensors"),
        &[("lm_head.weight", &[4096, 32000])],
    );

    let config = serde_json::json!({
        "vocab_size": 32000,
        "hidden_size": 4096
    });
    std::fs::write(
        dir.path().join("config.json"),
        serde_json::to_string(&config).unwrap(),
    )
    .unwrap();

    let mut contract = make_contract();
    contract.tensors.insert(
        "lm_head".to_string(),
        TensorSpec {
            gguf_name: "output.weight".to_string(),
            apr_name: "lm_head.weight".to_string(),
            gguf_shape: "[hidden, vocab]".to_string(),
            apr_shape: "[vocab, hidden]".to_string(),
            transpose: true,
            kernel: "matmul".to_string(),
            kernel_out_dim: None,
            kernel_in_dim: None,
            validation: None,
            critical: true,
            note: None,
        },
    );

    let (results, critical_failures) = run_all_validations(dir.path(), &contract);

    assert!(!critical_failures.is_empty());
    assert!(results.iter().any(|r| !r.passed));
}

// ========================================================================
// 19. check_model_path_preconditions
// ========================================================================

#[test]
fn test_check_model_path_preconditions_missing_path() {
    let result = check_model_path_preconditions(Path::new("/nonexistent/path"));
    assert!(result.is_some());
    let result = result.unwrap();
    assert!(!result.passed);
    assert_eq!(result.rules_failed, 1);
    assert!(!result.critical_failures.is_empty());
}

#[test]
fn test_check_model_path_preconditions_no_safetensors() {
    let dir = tempfile::tempdir().unwrap();
    // Dir exists but has no .safetensors files
    std::fs::write(dir.path().join("something.txt"), "hello").unwrap();

    let result = check_model_path_preconditions(dir.path());
    assert!(result.is_some());
    let result = result.unwrap();
    // No safetensors -> fail (Popper: untested ≠ validated)
    assert!(!result.passed);
    assert_eq!(result.rules_failed, 1);
    assert!(!result.critical_failures.is_empty());
}

#[test]
fn test_check_model_path_preconditions_has_safetensors() {
    let dir = tempfile::tempdir().unwrap();
    create_test_safetensors(&dir.path().join("model.safetensors"), &[("x", &[2, 3])]);

    let result = check_model_path_preconditions(dir.path());
    // Should return None (proceed with validation)
    assert!(result.is_none());
}

// ========================================================================
// 20. validate_model full integration
// ========================================================================

#[test]
fn test_validate_model_full_pass() {
    let dir = tempfile::tempdir().unwrap();

    create_test_safetensors(
        &dir.path().join("model.safetensors"),
        &[("lm_head.weight", &[32000, 4096])],
    );

    let config = serde_json::json!({
        "vocab_size": 32000,
        "hidden_size": 4096
    });
    std::fs::write(
        dir.path().join("config.json"),
        serde_json::to_string(&config).unwrap(),
    )
    .unwrap();

    let contract = make_contract();
    let result = validate_model(dir.path(), &contract).unwrap();

    assert!(result.passed);
    assert!(result.critical_failures.is_empty());
    // lm_head validated
    assert!(result.rules_checked > 0);
}

#[test]
fn test_validate_model_full_fail() {
    let dir = tempfile::tempdir().unwrap();

    // Transposed lm_head
    create_test_safetensors(
        &dir.path().join("model.safetensors"),
        &[("lm_head.weight", &[4096, 32000])],
    );

    let config = serde_json::json!({
        "vocab_size": 32000,
        "hidden_size": 4096
    });
    std::fs::write(
        dir.path().join("config.json"),
        serde_json::to_string(&config).unwrap(),
    )
    .unwrap();

    let contract = make_contract();
    let result = validate_model(dir.path(), &contract).unwrap();

    assert!(!result.passed);
    assert!(!result.critical_failures.is_empty());
}

#[test]
fn test_validate_model_no_config_json() {
    let dir = tempfile::tempdir().unwrap();

    create_test_safetensors(
        &dir.path().join("model.safetensors"),
        &[("lm_head.weight", &[32000, 4096])],
    );

    // No config.json => LayoutModelConfig::default() => lm_head UNVALIDATED (Popper)
    let contract = make_contract();
    let result = validate_model(dir.path(), &contract).unwrap();

    // Popper: missing config.json means lm_head cannot be validated → fail
    assert!(!result.passed);
    assert!(result.rules_failed > 0);
}

// ========================================================================
// 21. read_safetensors_metadata - edge cases
// ========================================================================

#[test]
fn test_read_safetensors_metadata_nonexistent_file() {
    let result = read_safetensors_metadata(Path::new("/nonexistent/file.safetensors"));
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Failed to open"));
}

#[test]
fn test_read_safetensors_metadata_truncated_header() {
    use std::io::Write;
    let dir = tempfile::tempdir().unwrap();
    let file_path = dir.path().join("truncated.safetensors");

    // Write a header_len of 1000 but only provide 5 bytes of header
    let header_len: u64 = 1000;
    let mut file = std::fs::File::create(&file_path).unwrap();
    file.write_all(&header_len.to_le_bytes()).unwrap();
    file.write_all(b"short").unwrap();

    let result = read_safetensors_metadata(&file_path);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Failed to read header"));
}

#[test]
fn test_read_safetensors_metadata_not_json_object() {
    use std::io::Write;
    let dir = tempfile::tempdir().unwrap();
    let file_path = dir.path().join("array.safetensors");

    // Valid JSON but not an object
    let header = b"[1, 2, 3]";
    let header_len = header.len() as u64;
    let mut file = std::fs::File::create(&file_path).unwrap();
    file.write_all(&header_len.to_le_bytes()).unwrap();
    file.write_all(header).unwrap();

    let result = read_safetensors_metadata(&file_path);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("not JSON object"));
}

#[test]
fn test_read_safetensors_metadata_tensor_without_shape() {
    use std::io::Write;
    let dir = tempfile::tempdir().unwrap();
    let file_path = dir.path().join("noshape.safetensors");

    // Valid JSON object but tensor entry has no "shape" field
    let header_json = serde_json::json!({
        "__metadata__": {"format": "pt"},
        "broken_tensor": {"dtype": "F32", "data_offsets": [0, 100]}
    });
    let header_bytes = serde_json::to_string(&header_json).unwrap();
    let header_len = header_bytes.len() as u64;
    let mut file = std::fs::File::create(&file_path).unwrap();
    file.write_all(&header_len.to_le_bytes()).unwrap();
    file.write_all(header_bytes.as_bytes()).unwrap();

    let result = read_safetensors_metadata(&file_path).unwrap();
    // broken_tensor should be skipped (filter_map returns None)
    assert!(result.is_empty());
}

// ========================================================================
// 22. find_safetensors_files - nonexistent path
// ========================================================================

#[test]
fn test_find_safetensors_files_nonexistent_dir() {
    let files = find_safetensors_files(Path::new("/nonexistent/dir"));
    assert!(files.is_empty());
}

// ========================================================================
// 23. validate_model with file path (not directory)
// ========================================================================

#[test]
fn test_validate_model_with_file_path() {
    let dir = tempfile::tempdir().unwrap();
    let file_path = dir.path().join("model.safetensors");

    create_test_safetensors(&file_path, &[("lm_head.weight", &[32000, 4096])]);

    let config = serde_json::json!({
        "vocab_size": 32000,
        "hidden_size": 4096
    });
    std::fs::write(
        dir.path().join("config.json"),
        serde_json::to_string(&config).unwrap(),
    )
    .unwrap();

    let contract = make_contract();
    let result = validate_model(&file_path, &contract).unwrap();

    assert!(result.passed);
    assert!(result.rules_checked > 0);
}

// ========================================================================
// 24. validate_lm_head non-2D through the orchestrator
// ========================================================================

#[test]
fn test_validate_lm_head_not_2d_through_orchestrator() {
    let config = make_config_full();
    let contract = make_contract();

    let mut all_tensors = HashMap::new();
    all_tensors.insert("lm_head.weight".to_string(), vec![4096, 32000, 10]);

    let mut results = Vec::new();
    let mut critical_failures = Vec::new();

    validate_lm_head(
        &all_tensors,
        &config,
        &contract,
        &mut results,
        &mut critical_failures,
    );

    assert_eq!(results.len(), 1);
    assert!(!results[0].passed);
    assert_eq!(critical_failures.len(), 1);
}

// ========================================================================
// 25. Multiple safetensors files merged
// ========================================================================

#[test]
fn test_collect_tensor_metadata_multiple_files() {
    let dir = tempfile::tempdir().unwrap();

    create_test_safetensors(
        &dir.path().join("model-00001-of-00002.safetensors"),
        &[("lm_head.weight", &[32000, 4096])],
    );
    create_test_safetensors(
        &dir.path().join("model-00002-of-00002.safetensors"),
        &[("model.embed_tokens.weight", &[32000, 4096])],
    );

    let mut results = Vec::new();
    let tensors = collect_tensor_metadata(dir.path(), &mut results);

    assert!(results.is_empty());
    assert_eq!(tensors.len(), 2);
    assert!(tensors.contains_key("lm_head.weight"));
    assert!(tensors.contains_key("model.embed_tokens.weight"));
}

// ========================================================================
// 26. read_safetensors_metadata: empty file (< 8 bytes)
// ========================================================================

#[test]
fn test_read_safetensors_metadata_too_short() {
    use std::io::Write;
    let dir = tempfile::tempdir().unwrap();
    let file_path = dir.path().join("tiny.safetensors");

    let mut file = std::fs::File::create(&file_path).unwrap();
    file.write_all(b"tiny").unwrap(); // Only 4 bytes, need 8

    let result = read_safetensors_metadata(&file_path);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Failed to read header length"));
}

// ========================================================================
// 27. resolve_dimension with spaces in expression
// ========================================================================

#[test]
fn test_resolve_dimension_expression_with_spaces() {
    let config = make_config_full();
    // "heads * head_dim" => trimmed parts should work
    assert_eq!(
        resolve_dimension("heads * head_dim", &config),
        Some(32 * 128)
    );
}

// ========================================================================
// 28. find_and_load_config with invalid JSON
// ========================================================================

#[test]
fn test_find_and_load_config_invalid_json() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("config.json"), "not json at all").unwrap();

    let mc = find_and_load_config(dir.path());
    // Should fall through to default
    assert_eq!(mc.vocab_size, None);
}

// ========================================================================
// 29. validate_2d_tensor_shape 3D tensor
// ========================================================================

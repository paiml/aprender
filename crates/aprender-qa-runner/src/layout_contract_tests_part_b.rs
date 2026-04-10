#[test]
fn test_validate_lm_head_shape_no_config() {
    let config = LayoutModelConfig::default();
    let contract = make_contract();
    // Popper: no config dimensions → UNVALIDATED → fails
    let result = validate_lm_head_shape(&[100, 200], &config, &contract);
    assert!(!result.passed);
    assert!(result.details.contains("UNVALIDATED"));
}

// ========================================================================
// 5. validate_2d_tensor_shape
// ========================================================================

#[test]
fn test_validate_2d_tensor_shape_not_2d() {
    let spec = make_spec("test.weight", "[vocab, hidden]", true);
    let config = make_config_full();
    let result = validate_2d_tensor_shape("test", &[4096], &spec, &config);
    assert!(!result.passed);
    assert_eq!(result.rule_id, "F-LAYOUT-CONTRACT-001");
    assert!(result.details.contains("must be 2D"));
}

#[test]
fn test_validate_2d_tensor_shape_valid() {
    let spec = make_spec("test.weight", "[vocab, hidden]", true);
    let config = make_config_full();
    let result = validate_2d_tensor_shape("test", &[32000, 4096], &spec, &config);
    assert!(result.passed);
    assert!(result.details.contains("shape correct"));
}

#[test]
fn test_validate_2d_tensor_shape_invalid() {
    let spec = make_spec("test.weight", "[vocab, hidden]", true);
    let config = make_config_full();
    // Wrong shape
    let result = validate_2d_tensor_shape("test", &[4096, 32000], &spec, &config);
    assert!(!result.passed);
    assert!(result.details.contains("mismatch"));
}

#[test]
fn test_validate_2d_tensor_shape_unresolvable() {
    // Popper: unresolvable shape dims → UNVALIDATED → fails
    let spec = make_spec("test.weight", "[unknown1, unknown2]", true);
    let config = LayoutModelConfig::default();
    let result = validate_2d_tensor_shape("test", &[100, 200], &spec, &config);
    assert!(!result.passed);
    assert!(result.details.contains("UNVALIDATED"));
}

// ========================================================================
// 6. validate_1d_tensor_shape
// ========================================================================

#[test]
fn test_validate_1d_tensor_shape_not_1d() {
    let spec = make_spec("test.bias", "[hidden]", false);
    let config = make_config_full();
    let result = validate_1d_tensor_shape("test.bias", &[4096, 100], &spec, &config);
    assert!(!result.passed);
    assert_eq!(result.rule_id, "F-LAYOUT-CONTRACT-003");
    assert!(result.details.contains("must be 1D"));
}

#[test]
fn test_validate_1d_tensor_shape_valid() {
    let spec = make_spec("test.bias", "[hidden]", false);
    let config = make_config_full();
    let result = validate_1d_tensor_shape("test.bias", &[4096], &spec, &config);
    assert!(result.passed);
    assert!(result.details.contains("shape correct"));
}

#[test]
fn test_validate_1d_tensor_shape_invalid() {
    let spec = make_spec("test.bias", "[hidden]", false);
    let config = make_config_full();
    let result = validate_1d_tensor_shape("test.bias", &[9999], &spec, &config);
    assert!(!result.passed);
    assert!(result.details.contains("shape mismatch"));
}

#[test]
fn test_validate_1d_tensor_shape_no_config() {
    let spec = make_spec("test.bias", "[hidden]", false);
    let config = LayoutModelConfig::default();
    // Popper: no hidden_size → UNVALIDATED → fails
    let result = validate_1d_tensor_shape("test.bias", &[9999], &spec, &config);
    assert!(!result.passed);
    assert!(result.details.contains("UNVALIDATED"));
}

// ========================================================================
// 7. find_safetensors_files
// ========================================================================

#[test]
fn test_find_safetensors_files_single_file() {
    let dir = tempfile::tempdir().unwrap();
    let file_path = dir.path().join("model.safetensors");
    create_test_safetensors(&file_path, &[("x", &[2, 3])]);

    let files = find_safetensors_files(&file_path);
    assert_eq!(files.len(), 1);
    assert_eq!(files[0], file_path);
}

#[test]
fn test_find_safetensors_files_directory() {
    let dir = tempfile::tempdir().unwrap();
    create_test_safetensors(
        &dir.path().join("model-00001-of-00002.safetensors"),
        &[("a", &[2, 3])],
    );
    create_test_safetensors(
        &dir.path().join("model-00002-of-00002.safetensors"),
        &[("b", &[4, 5])],
    );

    let files = find_safetensors_files(dir.path());
    assert_eq!(files.len(), 2);
}

#[test]
fn test_find_safetensors_files_subdir() {
    let dir = tempfile::tempdir().unwrap();
    let st_dir = dir.path().join("safetensors");
    std::fs::create_dir_all(&st_dir).unwrap();
    create_test_safetensors(&st_dir.join("model.safetensors"), &[("x", &[2])]);

    let files = find_safetensors_files(dir.path());
    assert_eq!(files.len(), 1);
}

#[test]
fn test_find_safetensors_files_no_files() {
    let dir = tempfile::tempdir().unwrap();
    let files = find_safetensors_files(dir.path());
    assert!(files.is_empty());
}

#[test]
fn test_find_safetensors_files_non_safetensors_file() {
    let dir = tempfile::tempdir().unwrap();
    let not_st = dir.path().join("model.gguf");
    std::fs::write(&not_st, b"not a safetensors file").unwrap();

    // Passed as file path
    let files = find_safetensors_files(&not_st);
    assert!(files.is_empty());

    // Passed as directory containing only .gguf
    let files = find_safetensors_files(dir.path());
    assert!(files.is_empty());
}

// ========================================================================
// 8. read_safetensors_metadata
// ========================================================================

#[test]
fn test_read_safetensors_metadata_valid() {
    let dir = tempfile::tempdir().unwrap();
    let file_path = dir.path().join("model.safetensors");
    create_test_safetensors(
        &file_path,
        &[
            ("lm_head.weight", &[32000, 4096]),
            ("embed_tokens.weight", &[32000, 4096]),
        ],
    );

    let metadata = read_safetensors_metadata(&file_path).unwrap();
    assert_eq!(metadata.len(), 2);
    assert_eq!(metadata["lm_head.weight"], vec![32000, 4096]);
    assert_eq!(metadata["embed_tokens.weight"], vec![32000, 4096]);
}

#[test]
fn test_read_safetensors_metadata_invalid_json() {
    use std::io::Write;
    let dir = tempfile::tempdir().unwrap();
    let file_path = dir.path().join("bad.safetensors");

    let bad_header = b"this is not json{{{{";
    let header_len = bad_header.len() as u64;
    let mut file = std::fs::File::create(&file_path).unwrap();
    file.write_all(&header_len.to_le_bytes()).unwrap();
    file.write_all(bad_header).unwrap();

    let result = read_safetensors_metadata(&file_path);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("JSON parse error"));
}

#[test]
fn test_read_safetensors_metadata_header_too_large() {
    use std::io::Write;
    let dir = tempfile::tempdir().unwrap();
    let file_path = dir.path().join("huge.safetensors");

    // Write a header_len that exceeds MAX_HEADER_SIZE
    let huge_len: u64 = (MAX_HEADER_SIZE as u64) + 1;
    let mut file = std::fs::File::create(&file_path).unwrap();
    file.write_all(&huge_len.to_le_bytes()).unwrap();

    let result = read_safetensors_metadata(&file_path);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Header too large"));
}

#[test]
fn test_read_safetensors_metadata_skips_metadata_key() {
    let dir = tempfile::tempdir().unwrap();
    let file_path = dir.path().join("model.safetensors");
    // Our helper inserts __metadata__ automatically
    create_test_safetensors(&file_path, &[("weight", &[10, 20])]);

    let metadata = read_safetensors_metadata(&file_path).unwrap();
    // __metadata__ should not appear
    assert!(!metadata.contains_key("__metadata__"));
    assert_eq!(metadata.len(), 1);
}

// ========================================================================
// 9. find_and_load_config
// ========================================================================

#[test]
fn test_find_and_load_config_directory() {
    let dir = tempfile::tempdir().unwrap();
    let config = serde_json::json!({
        "vocab_size": 32000,
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "num_hidden_layers": 24
    });
    std::fs::write(
        dir.path().join("config.json"),
        serde_json::to_string(&config).unwrap(),
    )
    .unwrap();

    let mc = find_and_load_config(dir.path());
    assert_eq!(mc.vocab_size, Some(32000));
    assert_eq!(mc.hidden_size, Some(4096));
    assert_eq!(mc.intermediate_size, Some(11008));
    assert_eq!(mc.num_attention_heads, Some(32));
    assert_eq!(mc.num_key_value_heads, Some(8));
    assert_eq!(mc.num_hidden_layers, Some(24));
}

#[test]
fn test_find_and_load_config_file_mode() {
    let dir = tempfile::tempdir().unwrap();
    // Simulate file mode: model file is "model.safetensors", config is "config.json" in same dir
    let model_file = dir.path().join("model.safetensors");
    create_test_safetensors(&model_file, &[("x", &[2, 3])]);

    let config = serde_json::json!({
        "vocab_size": 50000,
        "hidden_size": 2048
    });
    std::fs::write(
        dir.path().join("config.json"),
        serde_json::to_string(&config).unwrap(),
    )
    .unwrap();

    let mc = find_and_load_config(&model_file);
    assert_eq!(mc.vocab_size, Some(50000));
    assert_eq!(mc.hidden_size, Some(2048));
}

#[test]
fn test_find_and_load_config_file_mode_stem_prefix() {
    let dir = tempfile::tempdir().unwrap();
    let model_file = dir.path().join("mymodel.safetensors");
    create_test_safetensors(&model_file, &[("x", &[2])]);

    let config = serde_json::json!({"vocab_size": 12345});
    // Write stem-prefixed config: "mymodel.config.json"
    std::fs::write(
        dir.path().join("mymodel.config.json"),
        serde_json::to_string(&config).unwrap(),
    )
    .unwrap();

    let mc = find_and_load_config(&model_file);
    assert_eq!(mc.vocab_size, Some(12345));
}

#[test]
fn test_find_and_load_config_missing() {
    let dir = tempfile::tempdir().unwrap();
    let mc = find_and_load_config(dir.path());
    assert_eq!(mc.vocab_size, None);
    assert_eq!(mc.hidden_size, None);
}

#[test]
fn test_find_and_load_config_safetensors_subdir() {
    let dir = tempfile::tempdir().unwrap();
    let st_dir = dir.path().join("safetensors");
    std::fs::create_dir_all(&st_dir).unwrap();

    let config = serde_json::json!({"vocab_size": 99999});
    std::fs::write(
        st_dir.join("config.json"),
        serde_json::to_string(&config).unwrap(),
    )
    .unwrap();

    let mc = find_and_load_config(dir.path());
    assert_eq!(mc.vocab_size, Some(99999));
}

// ========================================================================
// 10. validate_model with empty dir (no safetensors)
// ========================================================================

#[test]
fn test_validate_model_empty_dir() {
    let dir = tempfile::tempdir().unwrap();
    let contract = make_contract();

    let result = validate_model(dir.path(), &contract).unwrap();
    // No safetensors -> fail (Popper: untested ≠ validated)
    assert!(!result.passed);
    assert_eq!(result.rules_failed, 1);
    assert!(!result.critical_failures.is_empty());
}

// ========================================================================
// 11. validate_layer_tensors
// ========================================================================

#[test]
fn test_validate_layer_tensors_with_layers() {
    let config = LayoutModelConfig {
        num_hidden_layers: Some(2),
        vocab_size: Some(32000),
        hidden_size: Some(4096),
        ..LayoutModelConfig::default()
    };

    let spec = make_spec(
        "model.layers.{n}.self_attn.q_proj.weight",
        "[vocab, hidden]",
        true,
    );

    let mut all_tensors = HashMap::new();
    all_tensors.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        vec![32000, 4096],
    );
    all_tensors.insert(
        "model.layers.1.self_attn.q_proj.weight".to_string(),
        vec![32000, 4096],
    );

    let mut results = Vec::new();
    validate_layer_tensors(
        "model.layers.{n}.self_attn.q_proj.weight",
        &all_tensors,
        &config,
        &spec,
        &mut results,
    );

    assert_eq!(results.len(), 2);
    assert!(results.iter().all(|r| r.passed));
}

#[test]
fn test_validate_layer_tensors_missing_layer() {
    let config = LayoutModelConfig {
        num_hidden_layers: Some(3),
        ..LayoutModelConfig::default()
    };

    let spec = make_spec("model.layers.{n}.weight", "[vocab, hidden]", true);

    let mut all_tensors = HashMap::new();
    // Only layer 0 exists; layers 1 and 2 missing => they are skipped
    all_tensors.insert("model.layers.0.weight".to_string(), vec![10, 20]);

    let mut results = Vec::new();
    validate_layer_tensors(
        "model.layers.{n}.weight",
        &all_tensors,
        &config,
        &spec,
        &mut results,
    );

    assert_eq!(results.len(), 1);
}

#[test]
fn test_validate_layer_tensors_zero_layers() {
    let config = LayoutModelConfig::default(); // num_hidden_layers = None
    let spec = make_spec("model.layers.{n}.weight", "[vocab, hidden]", true);
    let all_tensors = HashMap::new();
    let mut results = Vec::new();

    validate_layer_tensors(
        "model.layers.{n}.weight",
        &all_tensors,
        &config,
        &spec,
        &mut results,
    );

    // Popper: missing num_hidden_layers → UNVALIDATED failure
    assert_eq!(results.len(), 1);
    assert!(!results[0].passed);
    assert!(results[0].details.contains("UNVALIDATED"));
    assert!(results[0].details.contains("num_hidden_layers"));
}

// ========================================================================
// 12. validate_1d_layer_tensors
// ========================================================================

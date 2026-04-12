#[test]
fn test_find_model_by_prefix_nonexistent_dir() {
    let result = Executor::find_model_by_prefix(Path::new("/nonexistent/dir"), "model", "gguf");
    assert!(result.is_none());
}

// ── find_clean_model_file (unique) ──────────────────────────────────

#[test]
fn test_find_clean_model_file_wrong_extension() {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(tmp.path().join("model.safetensors"), b"data").unwrap();
    let result = Executor::find_clean_model_file(tmp.path(), "gguf");
    assert!(result.is_none());
}

// ── metadata_only mode ─────────────────────────────────────────────

#[test]
fn test_metadata_only_skips_inference() {
    let tmp = tempfile::tempdir().unwrap();
    // Write config.json with matching dimensions
    let config = serde_json::json!({
        "hidden_size": 896,
        "num_hidden_layers": 24,
        "num_attention_heads": 14,
        "num_key_value_heads": 2,
        "vocab_size": 151_936
    });
    let config_path = tmp.path().join("config.json");
    std::fs::write(&config_path, serde_json::to_string(&config).unwrap()).unwrap();

    // Write minimal safetensors with correct shapes
    {
        use std::collections::HashMap;
        use std::io::Write;
        let mut header_map: HashMap<&str, serde_json::Value> = HashMap::new();
        header_map.insert(
            "model.embed_tokens.weight",
            serde_json::json!({"dtype": "F32", "shape": [151_936, 896], "data_offsets": [0, 8]}),
        );
        header_map.insert(
            "lm_head.weight",
            serde_json::json!({"dtype": "F32", "shape": [151_936, 896], "data_offsets": [8, 16]}),
        );
        let header_json = serde_json::to_string(&header_map).unwrap();
        let header_bytes = header_json.as_bytes();
        let header_len = header_bytes.len() as u64;

        let st_path = tmp.path().join("model.safetensors");
        let mut f = std::fs::File::create(st_path).unwrap();
        f.write_all(&header_len.to_le_bytes()).unwrap();
        f.write_all(header_bytes).unwrap();
        f.write_all(&[0u8; 16]).unwrap();
    }

    // Write tokenizer files for G0-TOKENIZER checks
    std::fs::write(tmp.path().join("tokenizer.json"), b"{}").unwrap();
    std::fs::write(
        tmp.path().join("tokenizer_config.json"),
        br#"{"eos_token":"<|endoftext|>"}"#,
    )
    .unwrap();

    let playbook_yaml = r#"
name: test-dim-smoke
version: "1.0"
model:
  hf_repo: "test/model"
  expected_hidden_dim: 896
  expected_num_layers: 24
  expected_num_heads: 14
  expected_num_kv_heads: 2
  expected_vocab_size: 151936
test_matrix:
  modalities: [run]
  backends: [cpu]
  formats: [safetensors]
  prompts:
    - "hello"
"#;
    let playbook = crate::playbook::Playbook::from_yaml(playbook_yaml).expect("valid playbook");

    let exec_config = ExecutionConfig {
        metadata_only: true,
        model_path: Some(tmp.path().to_string_lossy().to_string()),
        ..Default::default()
    };
    let mut executor = Executor::with_config(exec_config);
    let result = executor.execute(&playbook).expect("should succeed");

    // All dimensional checks should pass
    assert_eq!(
        result.failed, 0,
        "no checks should fail: gateway={:?}",
        result.gateway_failed
    );
    assert!(result.passed > 0, "should have passing checks");
    assert!(result.gateway_failed.is_none());
}

#[test]
fn test_metadata_only_detects_mismatch() {
    let tmp = tempfile::tempdir().unwrap();
    // Write config.json with WRONG hidden_size
    let model_config = serde_json::json!({
        "hidden_size": 512,
        "num_hidden_layers": 24
    });
    std::fs::write(
        tmp.path().join("config.json"),
        serde_json::to_string(&model_config).unwrap(),
    )
    .unwrap();

    let playbook_yaml = r#"
name: test-dim-smoke-fail
version: "1.0"
model:
  hf_repo: "test/model"
  expected_hidden_dim: 896
  expected_num_layers: 24
test_matrix:
  modalities: [run]
  backends: [cpu]
  formats: [safetensors]
  prompts:
    - "hello"
"#;
    let playbook = crate::playbook::Playbook::from_yaml(playbook_yaml).expect("valid playbook");

    let exec_config = ExecutionConfig {
        metadata_only: true,
        model_path: Some(tmp.path().to_string_lossy().to_string()),
        ..Default::default()
    };
    let mut executor = Executor::with_config(exec_config);
    let result = executor.execute(&playbook).expect("should succeed");

    // hidden_size mismatch + no safetensors => failures
    assert!(result.failed > 0, "should have failures");
    assert!(result.gateway_failed.is_some());
}

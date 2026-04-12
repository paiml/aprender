
// ============================================================
// HF Parity Tests — coverage for model_path/inference/oracle branches
// (gates.rs lines 375-432)
// ============================================================

/// Helper: write a minimal SafeTensors file with a `logits` tensor.
fn write_golden_safetensors(path: &Path, logit_values: &[f32]) {
    use std::io::Write;
    let byte_size = logit_values.len() * 4;
    let header = serde_json::json!({
        "__metadata__": {"format": "pt"},
        "logits": {
            "dtype": "F32",
            "shape": [1usize, logit_values.len()],
            "data_offsets": [0usize, byte_size]
        }
    });
    let header_json = serde_json::to_string(&header).expect("serialize header");
    let header_bytes = header_json.as_bytes();
    let header_len = header_bytes.len() as u64;
    let mut file = std::fs::File::create(path).expect("create safetensors file");
    file.write_all(&header_len.to_le_bytes()).expect("write header len");
    file.write_all(header_bytes).expect("write header");
    let raw: Vec<u8> = logit_values
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    file.write_all(&raw).expect("write logit data");
}

/// Helper: set up a complete HF parity corpus that `oracle.load_golden` can succeed with.
///
/// Returns: (TempDir, corpus_path_str, model_family_str)
/// The corpus contains:
/// - `manifest.json`: `{"prompts": ["manifest_key001"]}`
/// - `manifest_key001.json`: `{"prompt": "What is 2+2?"}`
/// - `52cb6b5e4a038af1.safetensors`: golden logits (hash of "What is 2+2?")
/// - `52cb6b5e4a038af1.json`: metadata with expected text (optional)
fn setup_complete_hf_parity_corpus(
    expected_text: Option<&str>,
) -> (tempfile::TempDir, String, String) {
    let dir = tempfile::tempdir().expect("create temp dir");
    let model_family = "test-model/v1";
    let family_dir = dir.path().join("test-model").join("v1");
    std::fs::create_dir_all(&family_dir).expect("create family dir");

    // Manifest with one prompt key
    std::fs::write(
        family_dir.join("manifest.json"),
        r#"{"prompts":["manifest_key001"]}"#,
    )
    .expect("write manifest");

    // Golden JSON file for that key: maps manifest_key001 → "What is 2+2?"
    std::fs::write(
        family_dir.join("manifest_key001.json"),
        r#"{"prompt":"What is 2+2?"}"#,
    )
    .expect("write golden json");

    // Golden SafeTensors for "What is 2+2?" (hash = 52cb6b5e4a038af1)
    let st_path = family_dir.join("52cb6b5e4a038af1.safetensors");
    write_golden_safetensors(&st_path, &[1.0_f32, 2.0, 3.0, 4.0]);

    // Optional metadata JSON with expected text output (field name is "generated_text")
    if let Some(text) = expected_text {
        let meta = serde_json::json!({
            "generated_text": text,
            "model": "test/model",
            "transformers_version": "4.40.0"
        });
        std::fs::write(
            family_dir.join("52cb6b5e4a038af1.json"),
            serde_json::to_string(&meta).expect("serialize meta"),
        )
        .expect("write golden meta");
    }

    let corpus_path = dir.path().to_string_lossy().to_string();
    (dir, corpus_path, model_family.to_string())
}

/// HF parity: model_path not configured → F-HF-PARITY-001 skipped (lines 375-383)
/// oracle.load_golden succeeds (valid .safetensors), but model_path=None → skip
#[test]
fn test_hf_parity_no_model_path_skips_inference() {
    let (dir, corpus_path, model_family) = setup_complete_hf_parity_corpus(None);
    let mock_runner = MockCommandRunner::new();
    let config = ExecutionConfig {
        run_hf_parity: true,
        hf_parity_corpus_path: Some(corpus_path),
        hf_parity_model_family: Some(model_family),
        model_path: None, // ← triggers the skip branch
        run_conversion_tests: false,
        run_golden_rule_test: false,
        run_contract_tests: false,
        ..Default::default()
    };
    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));
    let model_id = ModelId::new("test", "model");
    let (passed, failed) = executor.run_hf_parity_tests(&model_id);
    assert_eq!(passed, 0);
    assert_eq!(failed, 0);
    let evidence = executor.evidence().all();
    // Skipped evidence has is_pass()=true, !is_fail()
    assert!(
        evidence.iter().any(|e| e.gate_id == "F-HF-PARITY-001" && !e.outcome.is_fail()),
        "Expected F-HF-PARITY-001 skipped when model_path=None, got: {:?}",
        evidence.iter().map(|e| (&e.gate_id, &e.outcome)).collect::<Vec<_>>()
    );
    drop(dir);
}

/// HF parity: inference fails → F-HF-PARITY-001 falsified (lines 395-406)
#[test]
fn test_hf_parity_inference_failure() {
    let (dir, corpus_path, model_family) = setup_complete_hf_parity_corpus(None);
    // Mock runner returns inference failure
    let mock_runner = MockCommandRunner::new().with_inference_failure();
    let config = ExecutionConfig {
        run_hf_parity: true,
        hf_parity_corpus_path: Some(corpus_path),
        hf_parity_model_family: Some(model_family),
        model_path: Some("/mock/model.apr".to_string()),
        run_conversion_tests: false,
        run_golden_rule_test: false,
        run_contract_tests: false,
        ..Default::default()
    };
    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));
    let model_id = ModelId::new("test", "model");
    let (passed, failed) = executor.run_hf_parity_tests(&model_id);
    assert_eq!(passed, 0);
    assert_eq!(failed, 1, "Expected 1 failure for inference error");
    let evidence = executor.evidence().all();
    assert!(
        evidence.iter().any(|e| {
            e.gate_id == "F-HF-PARITY-001" && e.outcome.is_fail()
        }),
        "Expected F-HF-PARITY-001 falsified for inference failure, got: {:?}",
        evidence.iter().map(|e| (&e.gate_id, &e.outcome)).collect::<Vec<_>>()
    );
    drop(dir);
}

/// HF parity: oracle evaluates and returns Corroborated (no text in golden → fallthrough) (lines 411-420)
#[test]
fn test_hf_parity_oracle_corroborated_text_match() {
    // No generated_text in golden → oracle cannot text-compare → falls through to Corroborated
    let (dir, corpus_path, model_family) = setup_complete_hf_parity_corpus(None);
    // Default mock runner — output format doesn't matter since there's no golden text to compare
    let mock_runner = MockCommandRunner::new();
    let config = ExecutionConfig {
        run_hf_parity: true,
        hf_parity_corpus_path: Some(corpus_path),
        hf_parity_model_family: Some(model_family),
        model_path: Some("/mock/model.apr".to_string()),
        run_conversion_tests: false,
        run_golden_rule_test: false,
        run_contract_tests: false,
        ..Default::default()
    };
    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));
    let model_id = ModelId::new("test", "model");
    let (passed, failed) = executor.run_hf_parity_tests(&model_id);
    assert_eq!(failed, 0);
    assert_eq!(passed, 1, "Expected 1 passed for oracle corroborated");
    let evidence = executor.evidence().all();
    assert!(
        evidence.iter().any(|e| {
            e.gate_id == "F-HF-PARITY-001" && e.outcome.is_pass()
        }),
        "Expected F-HF-PARITY-001 corroborated, got: {:?}",
        evidence.iter().map(|e| (&e.gate_id, &e.outcome)).collect::<Vec<_>>()
    );
    drop(dir);
}

/// HF parity: oracle evaluates and returns Falsified (text mismatch) (lines 421-432)
#[test]
fn test_hf_parity_oracle_falsified_text_mismatch() {
    // Golden has text="Four" but inference returns "garbage output xyz"
    let (dir, corpus_path, model_family) = setup_complete_hf_parity_corpus(Some("Four"));
    // Mock returns something that doesn't match "Four" and is not a safetensors path
    let mock_runner = MockCommandRunner::new().with_inference_response("completely different output xyz");
    let config = ExecutionConfig {
        run_hf_parity: true,
        hf_parity_corpus_path: Some(corpus_path),
        hf_parity_model_family: Some(model_family),
        model_path: Some("/mock/model.apr".to_string()),
        run_conversion_tests: false,
        run_golden_rule_test: false,
        run_contract_tests: false,
        ..Default::default()
    };
    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));
    let model_id = ModelId::new("test", "model");
    let (passed, failed) = executor.run_hf_parity_tests(&model_id);
    assert_eq!(passed, 0);
    assert_eq!(failed, 1, "Expected 1 failure for oracle falsified");
    let evidence = executor.evidence().all();
    assert!(
        evidence.iter().any(|e| {
            e.gate_id == "F-HF-PARITY-001" && e.outcome.is_fail()
        }),
        "Expected F-HF-PARITY-001 falsified for text mismatch, got: {:?}",
        evidence.iter().map(|e| (&e.gate_id, &e.outcome)).collect::<Vec<_>>()
    );
    drop(dir);
}

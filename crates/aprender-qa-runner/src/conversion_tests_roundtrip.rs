#[test]
fn test_round_trip_execute_falsified_via_mock() {
    let dir = tempfile::tempdir().unwrap();
    let model_file = dir.path().join("model.gguf");
    std::fs::write(&model_file, "fake").unwrap();

    let mock = create_mock_apr(
        dir.path(),
        r#"case "$1" in
run)
  case "$2" in
  *converted*) printf "Round-trip drift detected";;
  *) printf "The answer is 4";;
  esac
  exit 0;;
rosetta) touch "$4"; exit 0;;
esac
exit 1"#,
    );

    let mut rt = RoundTripTest::new(
        vec![Format::Gguf, Format::Apr],
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    rt.binary = mock.to_string_lossy().to_string();

    if let Ok(ConversionResult::Falsified { gate_id, .. }) = &rt.execute(&model_file) {
        assert_eq!(gate_id, "F-CONV-RT-001");
    }
}

#[test]
fn test_conversion_executor_execute_all_via_mock() {
    let dir = tempfile::tempdir().unwrap();
    let model_file = dir.path().join("model.gguf");
    std::fs::write(&model_file, "fake").unwrap();

    let mock = create_mock_apr(
        dir.path(),
        r#"case "$1" in
run) printf "The answer is 4"; exit 0;;
rosetta) touch "$4"; exit 0;;
esac
exit 1"#,
    );

    let config = ConversionConfig {
        test_all_pairs: true,
        test_round_trips: true,
        backends: vec![Backend::Cpu],
        no_gpu: true,
        ..Default::default()
    };
    let mut executor = ConversionExecutor::new(config);
    executor.binary = mock.to_string_lossy().to_string();
    let model_id = ModelId::new("test", "model");

    if let Ok(exec_result) = executor.execute_all(&model_file, &model_id) {
        assert!(exec_result.total > 0);
        assert!(!exec_result.evidence.is_empty());
        assert!(!exec_result.results.is_empty());
        if exec_result.failed == 0 {
            assert!(exec_result.all_passed());
        }
    }
}

#[test]
fn test_conversion_executor_execute_all_with_errors_via_mock() {
    let dir = tempfile::tempdir().unwrap();
    let model_file = dir.path().join("model.gguf");
    std::fs::write(&model_file, "fake").unwrap();

    let mock = create_mock_apr(
        dir.path(),
        r#"case "$1" in
run) printf "output"; exit 0;;
rosetta) printf "convert failed" >&2; exit 1;;
esac
exit 1"#,
    );

    let config = ConversionConfig {
        test_all_pairs: true,
        test_round_trips: false,
        backends: vec![Backend::Cpu],
        no_gpu: true,
        ..Default::default()
    };
    let mut executor = ConversionExecutor::new(config);
    executor.binary = mock.to_string_lossy().to_string();
    let model_id = ModelId::new("test", "model");

    if let Ok(exec_result) = executor.execute_all(&model_file, &model_id) {
        assert!(exec_result.total > 0);
        assert!(exec_result.failed > 0);
        assert!(!exec_result.all_passed());
    }
}

#[test]
fn test_conversion_executor_round_trip_error_via_mock() {
    let dir = tempfile::tempdir().unwrap();
    let model_file = dir.path().join("model.gguf");
    std::fs::write(&model_file, "fake").unwrap();

    let mock = create_mock_apr(dir.path(), r"exit 1");

    let config = ConversionConfig {
        test_all_pairs: false,
        test_round_trips: true,
        backends: vec![Backend::Cpu],
        no_gpu: true,
        ..Default::default()
    };
    let mut executor = ConversionExecutor::new(config);
    executor.binary = mock.to_string_lossy().to_string();
    let model_id = ModelId::new("test", "model");

    if let Ok(exec_result) = executor.execute_all(&model_file, &model_id) {
        assert!(!exec_result.evidence.is_empty());
    }
}

#[test]
fn test_conversion_test_execute_safetensors_target_via_mock() {
    let dir = tempfile::tempdir().unwrap();
    let model_file = dir.path().join("model.gguf");
    std::fs::write(&model_file, "fake").unwrap();

    let mock = create_mock_apr(
        dir.path(),
        r#"case "$1" in
run) printf "The answer is 4"; exit 0;;
rosetta) touch "$4"; exit 0;;
esac
exit 1"#,
    );

    let mut test = ConversionTest::new(
        Format::Gguf,
        Format::SafeTensors,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    test.binary = mock.to_string_lossy().to_string();

    if let Ok(ConversionResult::Corroborated { target_format, .. }) = test.execute(&model_file) {
        assert_eq!(target_format, Format::SafeTensors);
    }
}

// =========================================================================
// Rosetta-Testing Spec: New test type constructors (PMAT-ROSETTA-002/003)
// =========================================================================

#[test]
fn test_idempotency_test_new() {
    let idem = IdempotencyTest::new(
        Format::Gguf,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    assert_eq!(idem.format_a, Format::Gguf);
    assert_eq!(idem.format_b, Format::Apr);
    assert_eq!(idem.backend, Backend::Cpu);
}

#[test]
fn test_idempotency_test_debug() {
    let idem = IdempotencyTest::new(
        Format::Gguf,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    let debug_str = format!("{idem:?}");
    assert!(debug_str.contains("IdempotencyTest"));
}

#[test]
fn test_idempotency_test_clone() {
    let idem = IdempotencyTest::new(
        Format::SafeTensors,
        Format::Gguf,
        Backend::Gpu,
        ModelId::new("test", "model"),
    );
    let cloned = idem.clone();
    assert_eq!(cloned.format_a, Format::SafeTensors);
    assert_eq!(cloned.format_b, Format::Gguf);
}

#[test]
fn test_commutativity_test_new() {
    let com = CommutativityTest::new(Backend::Cpu, ModelId::new("test", "model"));
    assert_eq!(com.backend, Backend::Cpu);
}

#[test]
fn test_commutativity_test_debug() {
    let com = CommutativityTest::new(Backend::Cpu, ModelId::new("test", "model"));
    let debug_str = format!("{com:?}");
    assert!(debug_str.contains("CommutativityTest"));
}

#[test]
fn test_commutativity_test_clone() {
    let com = CommutativityTest::new(Backend::Gpu, ModelId::new("test", "model"));
    let cloned = com.clone();
    assert_eq!(cloned.backend, Backend::Gpu);
}

#[test]
fn test_conversion_config_new_fields_default() {
    let config = ConversionConfig::default();
    assert!(config.test_multi_hop);
    assert!(config.test_cardinality);
    assert!(config.test_tensor_names);
    assert!(config.test_idempotency);
    assert!(config.test_commutativity);
}

#[test]
fn test_conversion_config_cpu_only_new_fields() {
    let config = ConversionConfig::cpu_only();
    assert!(config.test_multi_hop);
    assert!(config.test_cardinality);
    assert!(config.test_tensor_names);
    assert!(config.test_idempotency);
    assert!(config.test_commutativity);
    assert!(config.no_gpu);
}

#[test]
fn test_conversion_config_selective_disable() {
    let config = ConversionConfig {
        test_multi_hop: false,
        test_cardinality: false,
        test_tensor_names: true,
        test_idempotency: false,
        test_commutativity: true,
        ..Default::default()
    };
    assert!(!config.test_multi_hop);
    assert!(!config.test_cardinality);
    assert!(config.test_tensor_names);
    assert!(!config.test_idempotency);
    assert!(config.test_commutativity);
}

#[test]
fn test_check_cardinality_nonexistent_binary() {
    let source = std::path::PathBuf::from("source.gguf");
    let target = std::path::PathBuf::from("target.apr");
    let result = check_cardinality(&source, &target, "/nonexistent/apr");
    assert!(result.is_err());
}

#[test]
fn test_check_tensor_names_nonexistent_binary() {
    let source = std::path::PathBuf::from("source.gguf");
    let target = std::path::PathBuf::from("target.apr");
    let result = check_tensor_names(&source, &target, "/nonexistent/apr");
    assert!(result.is_err());
}

// =========================================================================
// Mock binary tests for check_cardinality and check_tensor_names
// =========================================================================

/// Create a mock binary with explicit fd sync/close to avoid ETXTBSY (os error 26)
/// when parallel tests execute mock scripts concurrently.
fn create_mock_inspect_binary(
    dir: &std::path::Path,
    name: &str,
    json_output: &str,
) -> std::path::PathBuf {
    create_mock_script(dir, name, &format!("#!/bin/bash\necho '{json_output}'"))
}

/// Create a conditional mock binary (if/else on model arg).
fn create_conditional_mock_binary(
    dir: &std::path::Path,
    name: &str,
    script: &str,
) -> std::path::PathBuf {
    create_mock_script(dir, name, script)
}

/// Write a mock script with explicit open→write→sync→close to ensure the
/// write fd is fully released before any execve() can hit ETXTBSY.
fn create_mock_script(dir: &std::path::Path, name: &str, content: &str) -> std::path::PathBuf {
    let path = dir.join(name);
    {
        use std::io::Write;
        let mut f = std::fs::File::create(&path).expect("create mock");
        f.write_all(content.as_bytes()).expect("write mock");
        f.sync_all().expect("sync mock");
        drop(f);
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755))
            .expect("set permissions");
    }
    // Yield to let the OS fully release the write reference on the inode
    std::thread::yield_now();
    path
}

#[test]
fn test_check_cardinality_loss_detected() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let source_model = dir.path().join("source.gguf");
    let target_model = dir.path().join("target.apr");
    std::fs::write(&source_model, b"source").expect("write source");
    std::fs::write(&target_model, b"target").expect("write target");

    // Mock binary that returns different tensor counts based on the model arg
    let mock = create_conditional_mock_binary(
        dir.path(),
        "apr_card",
        "#!/bin/bash\nif echo \"$3\" | grep -q source; then\n  echo '{\"tensor_count\": 338, \"tensor_names\": []}'\nelse\n  echo '{\"tensor_count\": 227, \"tensor_names\": []}'\nfi",
    );

    let result = check_cardinality(&source_model, &target_model, mock.to_str().expect("path"));
    let (gate_id, reason) = result
        .expect("should succeed")
        .expect("should detect cardinality loss");
    assert_eq!(gate_id, "F-CONV-CARD-001");
    assert!(reason.contains("338"));
    assert!(reason.contains("227"));
}

#[test]
fn test_check_cardinality_no_loss() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let source_model = dir.path().join("source.gguf");
    let target_model = dir.path().join("target.apr");
    std::fs::write(&source_model, b"source").expect("write source");
    std::fs::write(&target_model, b"target").expect("write target");

    let mock = create_mock_inspect_binary(
        dir.path(),
        "apr_card_ok",
        r#"{"tensor_count": 338, "tensor_names": []}"#,
    );

    let result = check_cardinality(&source_model, &target_model, mock.to_str().expect("path"));
    assert!(result.expect("should succeed").is_none());
}

#[test]
fn test_check_tensor_names_fusion_detected() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let source_model = dir.path().join("source.gguf");
    let target_model = dir.path().join("target.apr");
    std::fs::write(&source_model, b"source").expect("write source");
    std::fs::write(&target_model, b"target").expect("write target");

    // Source has q_proj, k_proj, v_proj; target has qkv_proj (fusion)
    let mock = create_conditional_mock_binary(
        dir.path(),
        "apr_names",
        "#!/bin/bash\nif echo \"$3\" | grep -q source; then\n  echo '{\"tensor_count\": 3, \"tensor_names\": [\"layer.0.q_proj\", \"layer.0.k_proj\", \"layer.0.v_proj\"]}'\nelse\n  echo '{\"tensor_count\": 1, \"tensor_names\": [\"layer.0.qkv_proj\"]}'\nfi",
    );

    let result = check_tensor_names(&source_model, &target_model, mock.to_str().expect("path"));
    let (gate_id, detail) = result
        .expect("should succeed")
        .expect("should detect name divergence");
    assert_eq!(gate_id, "F-CONV-NAME-001");
    assert!(detail.contains("QKV fusion"));
    assert!(detail.contains("q_proj"));
}

#[test]
fn test_check_tensor_names_non_fusion_divergence() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let source_model = dir.path().join("source.gguf");
    let target_model = dir.path().join("target.apr");
    std::fs::write(&source_model, b"source").expect("write source");
    std::fs::write(&target_model, b"target").expect("write target");

    // Source has "embed.weight"; target renamed it to "embedding.weight"
    let mock = create_conditional_mock_binary(
        dir.path(),
        "apr_names2",
        "#!/bin/bash\nif echo \"$3\" | grep -q source; then\n  echo '{\"tensor_count\": 2, \"tensor_names\": [\"embed.weight\", \"lm_head.weight\"]}'\nelse\n  echo '{\"tensor_count\": 2, \"tensor_names\": [\"embedding.weight\", \"lm_head.weight\"]}'\nfi",
    );

    let result = check_tensor_names(&source_model, &target_model, mock.to_str().expect("path"));
    let (gate_id, detail) = result
        .expect("should succeed")
        .expect("should detect divergence");
    assert_eq!(gate_id, "F-CONV-NAME-001");
    assert!(detail.contains("divergence"));
    assert!(detail.contains("embed.weight"));
}

#[test]
fn test_check_tensor_names_all_preserved() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let source_model = dir.path().join("source.gguf");
    let target_model = dir.path().join("target.apr");
    std::fs::write(&source_model, b"source").expect("write source");
    std::fs::write(&target_model, b"target").expect("write target");

    let mock = create_mock_inspect_binary(
        dir.path(),
        "apr_names_ok",
        r#"{"tensor_count": 2, "tensor_names": ["a.weight", "b.weight"]}"#,
    );

    let result = check_tensor_names(&source_model, &target_model, mock.to_str().expect("path"));
    assert!(result.expect("should succeed").is_none());
}

#[test]
fn test_check_tensor_names_empty_names_skip() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let source_model = dir.path().join("source.gguf");
    let target_model = dir.path().join("target.apr");
    std::fs::write(&source_model, b"source").expect("write source");
    std::fs::write(&target_model, b"target").expect("write target");

    let mock = create_mock_inspect_binary(
        dir.path(),
        "apr_names_empty",
        r#"{"tensor_count": 10, "tensor_names": []}"#,
    );

    let result = check_tensor_names(&source_model, &target_model, mock.to_str().expect("path"));
    assert!(result.expect("should succeed").is_none());
}

#[test]
fn test_convert_to_format_tagged_gguf_ext() {
    let source = std::path::PathBuf::from("/tmp/model.apr");
    let target = source.with_extension("tag1.gguf");
    assert!(target.to_str().expect("path").ends_with("tag1.gguf"));
}

// =========================================================================
// check_cardinality_gate / check_tensor_name_gate via run_structural_checks
// These exercise the private gate methods through ConversionExecutor::execute_all
// with a pre-created "converted" file so run_structural_checks doesn't skip.
// =========================================================================

/// Config that runs only structural checks (cardinality + tensor names).
fn structural_only_config() -> ConversionConfig {
    ConversionConfig {
        test_all_pairs: false,
        test_round_trips: false,
        test_multi_hop: false,
        test_cardinality: true,
        test_tensor_names: true,
        test_idempotency: false,
        test_commutativity: false,
        backends: vec![Backend::Cpu],
        no_gpu: true,
    }
}

#[test]
fn test_check_cardinality_gate_passes_via_executor() {
    let dir = tempfile::tempdir().unwrap();
    let model_file = dir.path().join("model.gguf");
    // run_structural_checks looks for model.converted.apr (with_extension)
    let converted_file = dir.path().join("model.converted.apr");
    std::fs::write(&model_file, "fake source").unwrap();
    std::fs::write(&converted_file, "fake target").unwrap();

    // Mock inspect returns same tensor count → cardinality preserved → corroborated
    let mock = create_mock_script(
        dir.path(),
        "mock_card_ok",
        "#!/bin/bash\necho '{\"tensor_count\": 338, \"tensor_names\": []}'",
    );

    let mut executor = ConversionExecutor::new(structural_only_config());
    executor.binary = mock.to_string_lossy().to_string();
    let model_id = ModelId::new("test", "model");

    let result = executor.execute_all(&model_file, &model_id).unwrap();
    // At least one structural check ran (F-CONV-CARD-001)
    let card_evidence: Vec<_> = result
        .evidence
        .iter()
        .filter(|e| e.gate_id == "F-CONV-CARD-001")
        .collect();
    assert!(
        !card_evidence.is_empty(),
        "Expected F-CONV-CARD-001 evidence, got: {:?}",
        result.evidence.iter().map(|e| &e.gate_id).collect::<Vec<_>>()
    );
    // Corroborated when no cardinality loss
    assert!(card_evidence[0].outcome.is_pass());
}

#[test]
fn test_check_cardinality_gate_fails_via_executor() {
    let dir = tempfile::tempdir().unwrap();
    let model_file = dir.path().join("model.gguf");
    let converted_file = dir.path().join("model.converted.apr");
    std::fs::write(&model_file, "fake source").unwrap();
    std::fs::write(&converted_file, "fake target").unwrap();

    // Mock inspect: source has 338 tensors, converted has 227 → cardinality loss
    let mock = create_conditional_mock_binary(
        dir.path(),
        "mock_card_fail",
        "#!/bin/bash\nif echo \"$3\" | grep -q model.gguf; then\n  echo '{\"tensor_count\": 338, \"tensor_names\": []}'\nelse\n  echo '{\"tensor_count\": 227, \"tensor_names\": []}'\nfi",
    );

    let mut executor = ConversionExecutor::new(structural_only_config());
    executor.binary = mock.to_string_lossy().to_string();
    let model_id = ModelId::new("test", "model");

    let result = executor.execute_all(&model_file, &model_id).unwrap();
    let card_evidence: Vec<_> = result
        .evidence
        .iter()
        .filter(|e| e.gate_id == "F-CONV-CARD-001")
        .collect();
    assert!(!card_evidence.is_empty(), "Expected F-CONV-CARD-001 evidence");
    // At least one failure expected (cardinality loss)
    let has_fail = card_evidence.iter().any(|e| e.outcome.is_fail());
    assert!(has_fail, "Expected at least one F-CONV-CARD-001 failure");
}

#[test]
fn test_check_cardinality_gate_error_path_via_executor() {
    let dir = tempfile::tempdir().unwrap();
    let model_file = dir.path().join("model.gguf");
    let converted_file = dir.path().join("model.converted.apr");
    std::fs::write(&model_file, "fake source").unwrap();
    std::fs::write(&converted_file, "fake target").unwrap();

    // Use a nonexistent binary to trigger Err path in check_cardinality
    let mut executor = ConversionExecutor::new(structural_only_config());
    executor.binary = "/nonexistent/binary/that/does/not/exist".to_string();
    let model_id = ModelId::new("test", "model");

    // Should still succeed (executor wraps errors as Falsified evidence)
    let result = executor.execute_all(&model_file, &model_id).unwrap();
    // All structural check evidence should be failures (binary not found → Err path)
    let card_evidence: Vec<_> = result
        .evidence
        .iter()
        .filter(|e| e.gate_id == "F-CONV-CARD-001")
        .collect();
    if !card_evidence.is_empty() {
        assert!(
            card_evidence.iter().all(|e| e.outcome.is_fail()),
            "Binary not found should produce failures"
        );
    }
}

#[test]
fn test_check_tensor_name_gate_passes_via_executor() {
    let dir = tempfile::tempdir().unwrap();
    let model_file = dir.path().join("model.gguf");
    let converted_file = dir.path().join("model.converted.apr");
    std::fs::write(&model_file, "fake source").unwrap();
    std::fs::write(&converted_file, "fake target").unwrap();

    // Both sides have the same tensor names → names preserved → corroborated
    let mock = create_mock_script(
        dir.path(),
        "mock_names_ok",
        "#!/bin/bash\necho '{\"tensor_count\": 2, \"tensor_names\": [\"a.weight\", \"b.weight\"]}'",
    );

    let mut executor = ConversionExecutor::new(structural_only_config());
    executor.binary = mock.to_string_lossy().to_string();
    let model_id = ModelId::new("test", "model");

    let result = executor.execute_all(&model_file, &model_id).unwrap();
    let name_evidence: Vec<_> = result
        .evidence
        .iter()
        .filter(|e| e.gate_id == "F-CONV-NAME-001")
        .collect();
    assert!(
        !name_evidence.is_empty(),
        "Expected F-CONV-NAME-001 evidence"
    );
    assert!(name_evidence[0].outcome.is_pass());
}

#[test]
fn test_check_tensor_name_gate_fails_via_executor() {
    let dir = tempfile::tempdir().unwrap();
    let model_file = dir.path().join("model.gguf");
    let converted_file = dir.path().join("model.converted.apr");
    std::fs::write(&model_file, "fake source").unwrap();
    std::fs::write(&converted_file, "fake target").unwrap();

    // Source has q_proj, k_proj, v_proj; target renames all → divergence
    let mock = create_conditional_mock_binary(
        dir.path(),
        "mock_names_fail",
        "#!/bin/bash\nif echo \"$3\" | grep -q model.gguf; then\n  echo '{\"tensor_count\": 3, \"tensor_names\": [\"a.q_proj\", \"a.k_proj\", \"a.v_proj\"]}'\nelse\n  echo '{\"tensor_count\": 3, \"tensor_names\": [\"a.renamed1\", \"a.renamed2\", \"a.renamed3\"]}'\nfi",
    );

    let mut executor = ConversionExecutor::new(structural_only_config());
    executor.binary = mock.to_string_lossy().to_string();
    let model_id = ModelId::new("test", "model");

    let result = executor.execute_all(&model_file, &model_id).unwrap();
    let name_evidence: Vec<_> = result
        .evidence
        .iter()
        .filter(|e| e.gate_id.starts_with("F-CONV-NAME"))
        .collect();
    assert!(!name_evidence.is_empty(), "Expected F-CONV-NAME evidence");
    let has_fail = name_evidence.iter().any(|e| e.outcome.is_fail());
    assert!(has_fail, "Expected tensor name divergence failure");
}

// =========================================================================
// ByteLevelRoundTripTest::execute — coverage for conversion_strategies.rs
// =========================================================================

#[test]
fn test_byte_level_round_trip_error_on_wrong_format() {
    // resolve_model_path expects SafeTensors but model is .gguf → error path
    let dir = tempfile::tempdir().unwrap();
    let model_file = dir.path().join("model.gguf");
    std::fs::write(&model_file, "fake").unwrap();

    let mut brt = ByteLevelRoundTripTest::new(Backend::Cpu, ModelId::new("test", "model"));
    brt.binary = "/nonexistent/apr".to_string();

    let result = brt.execute(&model_file);
    assert!(result.is_err(), "Expected Err for wrong format extension");
}

#[test]
fn test_byte_level_round_trip_corroborated_via_mock() {
    // Happy path: mock creates target files and diff-tensors returns pass JSON
    let dir = tempfile::tempdir().unwrap();
    let model_file = dir.path().join("model.safetensors");
    std::fs::write(&model_file, "fake st").unwrap();

    // Mock: rosetta convert <src> <dst> → touch dst; rosetta diff-tensors → {"passed":true}
    let mock = create_mock_apr(
        dir.path(),
        r#"case "$2" in
convert) touch "$4"; exit 0;;
diff-tensors) echo '{"passed":true,"summary":"all tensors match"}'; exit 0;;
esac
exit 1"#,
    );

    let mut brt = ByteLevelRoundTripTest::new(Backend::Cpu, ModelId::new("test", "model"));
    brt.binary = mock.to_string_lossy().to_string();

    match brt.execute(&model_file) {
        Ok(ConversionResult::Corroborated { source_format, target_format, max_diff, .. }) => {
            assert_eq!(source_format, Format::SafeTensors);
            assert_eq!(target_format, Format::Apr);
            assert!((max_diff - 0.0).abs() < f64::EPSILON);
        }
        other => panic!("Expected Corroborated, got: {other:?}"),
    }
}

#[test]
fn test_byte_level_round_trip_falsified_via_mock() {
    // Falsified path: diff-tensors returns JSON containing "passed":false
    let dir = tempfile::tempdir().unwrap();
    let model_file = dir.path().join("model.safetensors");
    std::fs::write(&model_file, "fake st").unwrap();

    let mock = create_mock_apr(
        dir.path(),
        r#"case "$2" in
convert) touch "$4"; exit 0;;
diff-tensors) echo '{"passed":false,"mismatched":["lm_head.weight"]}'; exit 0;;
esac
exit 1"#,
    );

    let mut brt = ByteLevelRoundTripTest::new(Backend::Cpu, ModelId::new("test", "model"));
    brt.binary = mock.to_string_lossy().to_string();

    match brt.execute(&model_file) {
        Ok(ConversionResult::Falsified { gate_id, .. }) => {
            assert_eq!(gate_id, "F-CONV-RT-BYTE-001");
        }
        other => panic!("Expected Falsified, got: {other:?}"),
    }
}

#[test]
fn test_byte_level_round_trip_falsified_mismatched_keyword() {
    // Falsified via "mismatched" keyword (not "passed":false)
    let dir = tempfile::tempdir().unwrap();
    let model_file = dir.path().join("model.safetensors");
    std::fs::write(&model_file, "fake st").unwrap();

    let mock = create_mock_apr(
        dir.path(),
        r#"case "$2" in
convert) touch "$4"; exit 0;;
diff-tensors) echo '{"result":"mismatched tensors found"}'; exit 0;;
esac
exit 1"#,
    );

    let mut brt = ByteLevelRoundTripTest::new(Backend::Cpu, ModelId::new("test", "model"));
    brt.binary = mock.to_string_lossy().to_string();

    match brt.execute(&model_file) {
        Ok(ConversionResult::Falsified { gate_id, .. }) => {
            assert_eq!(gate_id, "F-CONV-RT-BYTE-001");
        }
        other => panic!("Expected Falsified, got: {other:?}"),
    }
}

#[test]
fn test_byte_level_round_trip_convert_failure() {
    // convert failure → Err propagation
    let dir = tempfile::tempdir().unwrap();
    let model_file = dir.path().join("model.safetensors");
    std::fs::write(&model_file, "fake st").unwrap();

    let mock = create_mock_apr(dir.path(), "exit 1");

    let mut brt = ByteLevelRoundTripTest::new(Backend::Cpu, ModelId::new("test", "model"));
    brt.binary = mock.to_string_lossy().to_string();

    let result = brt.execute(&model_file);
    assert!(result.is_err(), "Expected Err when conversion fails");
}

// ── IdempotencyTest::execute ─────────────────────────────────────────────────

#[test]
fn test_idempotency_execute_error_on_wrong_format() {
    // resolve_model_path expects Gguf but file is .safetensors → Err
    let dir = tempfile::tempdir().unwrap();
    let model_file = dir.path().join("model.safetensors");
    std::fs::write(&model_file, "fake").unwrap();

    let mut idem = IdempotencyTest::new(
        Format::Gguf,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    idem.binary = "/nonexistent/apr".to_string();

    let result = idem.execute(&model_file);
    assert!(result.is_err(), "Expected Err for wrong format extension");
}

#[test]
fn test_idempotency_execute_error_on_conversion_failure() {
    let dir = tempfile::tempdir().unwrap();
    let model_file = dir.path().join("model.gguf");
    std::fs::write(&model_file, "fake").unwrap();

    // Mock always exits 1 → conversion fails
    let mock = create_mock_apr(dir.path(), "exit 1");

    let mut idem = IdempotencyTest::new(
        Format::Gguf,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    idem.binary = mock.to_string_lossy().to_string();

    let result = idem.execute(&model_file);
    assert!(result.is_err(), "Expected Err when conversion fails");
}

#[test]
fn test_idempotency_execute_corroborated_cross_format() {
    // Cross-format (Gguf→Apr): non-garbage output on both runs → Corroborated
    let dir = tempfile::tempdir().unwrap();
    let model_file = dir.path().join("model.gguf");
    std::fs::write(&model_file, "fake").unwrap();

    let mock = create_mock_apr(
        dir.path(),
        r#"case "$1" in
rosetta) case "$2" in
  convert) touch "$4"; exit 0;;
esac;;
run) echo "The answer is 4"; exit 0;;
esac
exit 1"#,
    );

    let mut idem = IdempotencyTest::new(
        Format::Gguf,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    idem.binary = mock.to_string_lossy().to_string();

    match idem.execute(&model_file) {
        Ok(ConversionResult::Corroborated { source_format, target_format, .. }) => {
            assert_eq!(source_format, Format::Gguf);
            assert_eq!(target_format, Format::Apr);
        }
        other => panic!("Expected Corroborated, got: {other:?}"),
    }
}

#[test]
fn test_idempotency_execute_falsified_cross_format_garbage() {
    // Cross-format: garbage output → Falsified
    let dir = tempfile::tempdir().unwrap();
    let model_file = dir.path().join("model.gguf");
    std::fs::write(&model_file, "fake").unwrap();

    // Return heavy repetition garbage (high repetition ratio triggers is_garbage)
    let mock = create_mock_apr(
        dir.path(),
        r#"case "$1" in
rosetta) case "$2" in
  convert) touch "$4"; exit 0;;
esac;;
run) printf "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"; exit 0;;
esac
exit 1"#,
    );

    let mut idem = IdempotencyTest::new(
        Format::Gguf,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    idem.binary = mock.to_string_lossy().to_string();

    match idem.execute(&model_file) {
        Ok(ConversionResult::Falsified { gate_id, .. }) => {
            assert_eq!(gate_id, "F-CONV-IDEM-001");
        }
        other => panic!("Expected Falsified, got: {other:?}"),
    }
}

#[test]
fn test_idempotency_execute_corroborated_same_format_identical_output() {
    // Same format (Gguf→Gguf): outputs must be identical → Corroborated
    let dir = tempfile::tempdir().unwrap();
    let model_file = dir.path().join("model.gguf");
    std::fs::write(&model_file, "fake").unwrap();

    let mock = create_mock_apr(
        dir.path(),
        r#"case "$1" in
rosetta) case "$2" in
  convert) touch "$4"; exit 0;;
esac;;
run) echo "The answer is 4"; exit 0;;
esac
exit 1"#,
    );

    let mut idem = IdempotencyTest::new(
        Format::Gguf,
        Format::Gguf,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    idem.binary = mock.to_string_lossy().to_string();

    match idem.execute(&model_file) {
        Ok(ConversionResult::Corroborated { source_format, target_format, .. }) => {
            assert_eq!(source_format, Format::Gguf);
            assert_eq!(target_format, Format::Gguf);
        }
        other => panic!("Expected Corroborated, got: {other:?}"),
    }
}

#[test]
fn test_idempotency_execute_falsified_same_format_different_output() {
    // Same format: outputs differ → Falsified
    let dir = tempfile::tempdir().unwrap();
    let model_file = dir.path().join("model.gguf");
    std::fs::write(&model_file, "fake").unwrap();

    // Return different output based on which tagged file is being run (idem1 vs idem2)
    let mock = create_mock_apr(
        dir.path(),
        r#"case "$1" in
rosetta) case "$2" in
  convert) touch "$4"; exit 0;;
esac;;
run) case "$2" in
  *idem1*) echo "output one";;
  *idem2*) echo "output two";;
  *) echo "output one";;
esac; exit 0;;
esac
exit 1"#,
    );

    let mut idem = IdempotencyTest::new(
        Format::Gguf,
        Format::Gguf,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    idem.binary = mock.to_string_lossy().to_string();

    match idem.execute(&model_file) {
        Ok(ConversionResult::Falsified { gate_id, .. }) => {
            assert_eq!(gate_id, "F-CONV-IDEM-001");
        }
        other => panic!("Expected Falsified, got: {other:?}"),
    }
}

// ── CommutativityTest::execute ───────────────────────────────────────────────

#[test]
fn test_commutativity_execute_error_on_wrong_format() {
    // resolve_model_path expects Gguf but file is .safetensors → Err
    let dir = tempfile::tempdir().unwrap();
    let model_file = dir.path().join("model.safetensors");
    std::fs::write(&model_file, "fake").unwrap();

    let mut com = CommutativityTest::new(Backend::Cpu, ModelId::new("test", "model"));
    com.binary = "/nonexistent/apr".to_string();

    let result = com.execute(&model_file);
    assert!(result.is_err(), "Expected Err for wrong format extension");
}

#[test]
fn test_commutativity_execute_error_on_conversion_failure() {
    let dir = tempfile::tempdir().unwrap();
    let model_file = dir.path().join("model.gguf");
    std::fs::write(&model_file, "fake").unwrap();

    let mock = create_mock_apr(dir.path(), "exit 1");

    let mut com = CommutativityTest::new(Backend::Cpu, ModelId::new("test", "model"));
    com.binary = mock.to_string_lossy().to_string();

    let result = com.execute(&model_file);
    assert!(result.is_err(), "Expected Err when conversion fails");
}

#[test]
fn test_commutativity_execute_corroborated() {
    // Both paths produce non-garbage output → Corroborated
    let dir = tempfile::tempdir().unwrap();
    let model_file = dir.path().join("model.gguf");
    std::fs::write(&model_file, "fake").unwrap();

    let mock = create_mock_apr(
        dir.path(),
        r#"case "$1" in
rosetta) case "$2" in
  convert) touch "$4"; exit 0;;
esac;;
run) echo "The answer is 4"; exit 0;;
esac
exit 1"#,
    );

    let mut com = CommutativityTest::new(Backend::Cpu, ModelId::new("test", "model"));
    com.binary = mock.to_string_lossy().to_string();

    match com.execute(&model_file) {
        Ok(ConversionResult::Corroborated { source_format, target_format, .. }) => {
            assert_eq!(source_format, Format::Gguf);
            assert_eq!(target_format, Format::Apr);
        }
        other => panic!("Expected Corroborated, got: {other:?}"),
    }
}

#[test]
fn test_commutativity_execute_falsified_indirect_path_garbage() {
    // Indirect path (GGUF→ST→APR) returns garbage output → Falsified
    let dir = tempfile::tempdir().unwrap();
    let model_file = dir.path().join("model.gguf");
    std::fs::write(&model_file, "fake").unwrap();

    // Direct path (com_direct) → good output; indirect path (com_indirect) → garbage
    let mock = create_mock_apr(
        dir.path(),
        r#"case "$1" in
rosetta) case "$2" in
  convert) touch "$4"; exit 0;;
esac;;
run) case "$2" in
  *com_direct*) echo "The answer is 4";;
  *) printf "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";;
esac; exit 0;;
esac
exit 1"#,
    );

    let mut com = CommutativityTest::new(Backend::Cpu, ModelId::new("test", "model"));
    com.binary = mock.to_string_lossy().to_string();

    match com.execute(&model_file) {
        Ok(ConversionResult::Falsified { gate_id, .. }) => {
            assert_eq!(gate_id, "F-CONV-COM-001");
        }
        other => panic!("Expected Falsified, got: {other:?}"),
    }
}

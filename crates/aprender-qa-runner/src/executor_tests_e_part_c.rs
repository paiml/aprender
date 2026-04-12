#[test]
fn test_execute_with_profile_ci_perf_gates() {
    let runner = MockCommandRunner::new()
        .with_tps(50.0)
        .with_inference_response("Output:\nHello\nCompleted in 0.5s");
    let config = ExecutionConfig {
        run_profile_ci: true,
        model_path: Some("/mock/model".to_string()),
        no_gpu: true,
        ..Default::default()
    };
    let mut executor = Executor::with_runner(config, Arc::new(runner));

    let yaml = r#"
name: test-perf-integration
version: "1.0.0"
model:
  hf_repo: "test/model"
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
profile_ci:
  enabled: true
  warmup: 1
  measure: 2
  formats: [safetensors]
  backends: [cpu, gpu]
"#;
    let playbook = Playbook::from_yaml(yaml).expect("Failed to parse");
    let result = executor.execute(&playbook).expect("Execution failed");
    assert!(result.total_scenarios >= 1);
    let evidence = executor.evidence().all();
    assert!(evidence.iter().any(|e| e.gate_id == "F-PERF-006")); // GPU/CPU ratio
    assert!(evidence.iter().any(|e| e.gate_id == "F-PERF-005")); // memory profiling
}

// ── Bug 202: Sibling-file lookup in file mode ────────────────────────

#[test]
fn test_resolve_model_path_file_sibling_gguf() {
    // Given a .safetensors file, resolve_model_path should find sibling .gguf
    let temp_dir = tempfile::tempdir().unwrap();
    let st_file = temp_dir.path().join("model.safetensors");
    let gguf_file = temp_dir.path().join("model.gguf");
    std::fs::write(&st_file, b"fake safetensors").unwrap();
    std::fs::write(&gguf_file, b"fake gguf").unwrap();

    let config = ExecutionConfig {
        model_path: Some(st_file.to_string_lossy().to_string()),
        ..Default::default()
    };
    let executor = Executor::with_config(config);

    let scenario = QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Cpu,
        Format::Gguf,
        "test".to_string(),
        0,
    );
    let path = executor.resolve_model_path(&scenario);
    assert!(path.is_some(), "Should find sibling .gguf file");
    assert!(path.unwrap().contains("model.gguf"));
}

#[test]
fn test_resolve_model_path_file_sibling_apr() {
    // Given a .gguf file, resolve_model_path should find sibling .apr
    let temp_dir = tempfile::tempdir().unwrap();
    let gguf_file = temp_dir.path().join("model.gguf");
    let apr_file = temp_dir.path().join("model.apr");
    std::fs::write(&gguf_file, b"fake gguf").unwrap();
    std::fs::write(&apr_file, b"fake apr").unwrap();

    let config = ExecutionConfig {
        model_path: Some(gguf_file.to_string_lossy().to_string()),
        ..Default::default()
    };
    let executor = Executor::with_config(config);

    let scenario = QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Cpu,
        Format::Apr,
        "test".to_string(),
        0,
    );
    let path = executor.resolve_model_path(&scenario);
    assert!(path.is_some(), "Should find sibling .apr file");
    assert!(path.unwrap().contains("model.apr"));
}

#[test]
fn test_resolve_model_path_file_sibling_not_found() {
    // Given a .safetensors file with no sibling .gguf, should return None
    let temp_dir = tempfile::tempdir().unwrap();
    let st_file = temp_dir.path().join("model.safetensors");
    std::fs::write(&st_file, b"fake safetensors").unwrap();

    let config = ExecutionConfig {
        model_path: Some(st_file.to_string_lossy().to_string()),
        ..Default::default()
    };
    let executor = Executor::with_config(config);

    let scenario = QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Cpu,
        Format::Gguf,
        "test".to_string(),
        0,
    );
    assert!(
        executor.resolve_model_path(&scenario).is_none(),
        "No sibling .gguf exists, should return None"
    );
}

#[test]
fn test_resolve_model_path_file_sibling_fallback_different_stem() {
    // Given a .safetensors file with a DIFFERENT-FAMILY .gguf file in same dir,
    // prefix matching should NOT return it (avoids cross-model confusion).
    let temp_dir = tempfile::tempdir().unwrap();
    let st_file = temp_dir.path().join("abc123.safetensors");
    let gguf_file = temp_dir.path().join("other-name.gguf");
    std::fs::write(&st_file, b"fake safetensors").unwrap();
    std::fs::write(&gguf_file, b"fake gguf").unwrap();

    let config = ExecutionConfig {
        model_path: Some(st_file.to_string_lossy().to_string()),
        ..Default::default()
    };
    let executor = Executor::with_config(config);

    let scenario = QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Cpu,
        Format::Gguf,
        "test".to_string(),
        0,
    );
    let path = executor.resolve_model_path(&scenario);
    assert!(path.is_none(), "Should NOT match unrelated model family");
}

#[test]
fn test_resolve_model_path_file_sibling_prefix_match() {
    // Given a GGUF with quantization suffix, should find APR with same family prefix
    let temp_dir = tempfile::tempdir().unwrap();
    let gguf_file = temp_dir.path().join("qwen2.5-coder-7b-instruct-q4k.gguf");
    let apr_file = temp_dir.path().join("qwen2.5-coder-7b-instruct.apr");
    std::fs::write(&gguf_file, b"fake gguf").unwrap();
    std::fs::write(&apr_file, b"fake apr").unwrap();

    let config = ExecutionConfig {
        model_path: Some(gguf_file.to_string_lossy().to_string()),
        ..Default::default()
    };
    let executor = Executor::with_config(config);

    let scenario = QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Cpu,
        Format::Apr,
        "test".to_string(),
        0,
    );
    let path = executor.resolve_model_path(&scenario);
    assert!(
        path.is_some(),
        "Should find APR via model family prefix match"
    );
    assert!(path.unwrap().contains("qwen2.5-coder-7b-instruct.apr"));
}

// ── Bug 200: Modality-aware dispatch ─────────────────────────────────

#[test]
fn test_subprocess_execution_chat_modality() {
    let runner = MockCommandRunner::new();
    let config = ExecutionConfig {
        model_path: Some("/mock/model.gguf".to_string()),
        ..Default::default()
    };
    let executor = Executor::with_runner(config, Arc::new(runner));

    let scenario = QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Chat,
        Backend::Cpu,
        Format::Gguf,
        "What is 2+2?".to_string(),
        0,
    );

    let (text, stderr, exit_code, _tps, skipped) = executor.subprocess_execution(&scenario);
    assert!(!skipped, "Chat scenario should not be skipped");
    assert_eq!(exit_code, 0);
    assert!(stderr.is_none() || stderr.as_deref() == Some(""));
    assert!(text.contains("4"), "Chat should return arithmetic answer");
}

#[test]
fn test_subprocess_execution_serve_modality() {
    let runner = MockCommandRunner::new();
    let config = ExecutionConfig {
        model_path: Some("/mock/model.gguf".to_string()),
        ..Default::default()
    };
    let executor = Executor::with_runner(config, Arc::new(runner));

    let scenario = QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Serve,
        Backend::Cpu,
        Format::Gguf,
        "What is 2+2?".to_string(),
        0,
    );

    let (_text, _stderr, _exit_code, _tps, skipped) = executor.subprocess_execution(&scenario);
    // Serve scenario should not be skipped (spawn_serve mock returns success)
    assert!(!skipped, "Serve scenario should not be skipped");
}

// ── Bug 201: Per-scenario backend ────────────────────────────────────

#[test]
fn test_subprocess_execution_gpu_backend() {
    // GPU scenario should NOT pass --no-gpu
    let runner = MockCommandRunner::new();
    let config = ExecutionConfig {
        model_path: Some("/mock/model.gguf".to_string()),
        no_gpu: true, // Global flag says no GPU — but scenario overrides
        ..Default::default()
    };
    let executor = Executor::with_runner(config, Arc::new(runner));

    let scenario = QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Gpu,
        Format::Gguf,
        "test".to_string(),
        0,
    );

    let (_text, _stderr, exit_code, _tps, skipped) = executor.subprocess_execution(&scenario);
    assert!(!skipped);
    assert_eq!(exit_code, 0);
    // The mock doesn't validate the no_gpu flag directly, but the code path
    // now uses scenario.backend instead of config.no_gpu
}

// =========================================================================
// NEW: Coverage tests for extracted helper methods
// =========================================================================

// ── parse_timing_ms ─────────────────────────────────────────────────

#[test]
fn test_parse_timing_ms_valid() {
    let output = "Loading model...\nCompleted in 1.5s\nDone";
    let ms = parse_timing_ms(output);
    assert!(ms.is_some());
    assert!((ms.unwrap() - 1500.0).abs() < f64::EPSILON);
}

#[test]
fn test_parse_timing_ms_integer_seconds() {
    let output = "Completed in 3s";
    let ms = parse_timing_ms(output);
    assert!(ms.is_some());
    assert!((ms.unwrap() - 3000.0).abs() < f64::EPSILON);
}

#[test]
fn test_parse_timing_ms_no_match() {
    let output = "Some random output without timing";
    assert!(parse_timing_ms(output).is_none());
}

#[test]
fn test_parse_timing_ms_empty() {
    assert!(parse_timing_ms("").is_none());
}

#[test]
fn test_parse_timing_ms_case_insensitive() {
    let output = "COMPLETED IN 2.0s";
    let ms = parse_timing_ms(output);
    assert!(ms.is_some());
    assert!((ms.unwrap() - 2000.0).abs() < f64::EPSILON);
}

#[test]
fn test_parse_timing_ms_invalid_number() {
    let output = "Completed in abcs";
    assert!(parse_timing_ms(output).is_none());
}

// ── parse_throughput ────────────────────────────────────────────────

#[test]
fn test_parse_throughput_valid_decimal() {
    let output = r#"{"throughput_tps":25.5,"other":1}"#;
    let tps = parse_throughput(output);
    assert!(tps.is_some());
    assert!((tps.unwrap() - 25.5).abs() < f64::EPSILON);
}

#[test]
fn test_parse_throughput_at_end_of_json() {
    let output = r#"{"throughput_tps":100}"#;
    // The parse_throughput function looks for a non-digit/non-dot terminator
    // but at end of string this may not find one
    let tps = parse_throughput(output);
    // "100}" - the "}" terminates it
    assert!(tps.is_some());
    assert!((tps.unwrap() - 100.0).abs() < f64::EPSILON);
}

#[test]
fn test_parse_throughput_no_tps_field() {
    let output = r#"{"latency_ms":42}"#;
    assert!(parse_throughput(output).is_none());
}

#[test]
fn test_parse_throughput_empty_string() {
    assert!(parse_throughput("").is_none());
}

// ── classify_integrity_gate ─────────────────────────────────────────

#[test]
fn test_classify_integrity_gate_layers() {
    let gate = Executor::classify_integrity_gate("LAYERS mismatch: expected 24, got 14");
    assert_eq!(gate, integrity::gate_ids::LAYERS);
}

#[test]
fn test_classify_integrity_gate_hidden() {
    let gate = Executor::classify_integrity_gate("HIDDEN size mismatch: expected 896, got 4096");
    assert_eq!(gate, integrity::gate_ids::HIDDEN);
}

#[test]
fn test_classify_integrity_gate_vocab() {
    let gate = Executor::classify_integrity_gate("VOCAB size wrong: 896 vs 151936");
    assert_eq!(gate, integrity::gate_ids::VOCAB);
}

#[test]
fn test_classify_integrity_gate_config_default() {
    let gate = Executor::classify_integrity_gate("Some unknown error");
    assert_eq!(gate, integrity::gate_ids::CONFIG);
}

// ── is_conversion_artifact ──────────────────────────────────────────

#[test]
fn test_is_conversion_artifact_converted() {
    assert!(Executor::is_conversion_artifact("model-converted.gguf"));
}

#[test]
fn test_is_conversion_artifact_byte_rt() {
    assert!(Executor::is_conversion_artifact("model.byte_rt.apr"));
}

#[test]
fn test_is_conversion_artifact_idem() {
    assert!(Executor::is_conversion_artifact("model.idem.safetensors"));
}

#[test]
fn test_is_conversion_artifact_com() {
    assert!(Executor::is_conversion_artifact("model.com_q4k.gguf"));
}

#[test]
fn test_is_conversion_artifact_clean_file() {
    assert!(!Executor::is_conversion_artifact("model.safetensors"));
    assert!(!Executor::is_conversion_artifact("model.gguf"));
    assert!(!Executor::is_conversion_artifact("model.apr"));
    assert!(!Executor::is_conversion_artifact("config.json"));
}

// ── truncate_str ────────────────────────────────────────────────────

#[test]
fn test_truncate_str_short() {
    assert_eq!(Executor::truncate_str("hello", 10), "hello");
}

#[test]
fn test_truncate_str_exact() {
    assert_eq!(Executor::truncate_str("hello", 5), "hello");
}

#[test]
fn test_truncate_str_truncates() {
    assert_eq!(Executor::truncate_str("hello world", 5), "hello");
}

#[test]
fn test_truncate_str_empty() {
    assert_eq!(Executor::truncate_str("", 5), "");
}

// ── run_g0_tensor_template_check: uncovered branches ─────────────────────────

/// No tensor_template in family YAML → skipped (Popperian: untested ≠ corroborated)
#[test]
fn test_g0_tensor_no_tensor_template_skipped() {
    let mock_runner = MockCommandRunner::new();
    let dir = make_temp_model_dir();

    let contracts_dir = tempfile::TempDir::new().expect("create contracts dir");
    // Family YAML without tensor_template section
    let family_yaml = r#"
family: notemplatetest
size_variants:
  1b:
    parameters: "1B"
    hidden_dim: 512
    num_layers: 6
    num_heads: 4
"#;
    std::fs::write(contracts_dir.path().join("notemplatetest.yaml"), family_yaml)
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
        "notemplatetest",
        "1b",
        Some(contracts_dir.path().to_str().expect("path")),
    );

    assert_eq!(passed, 0);
    assert_eq!(failed, 0);
    let evidence = executor.evidence().all();
    let ev = evidence.iter().find(|e| e.gate_id == "G0-TENSOR-001").expect("G0-TENSOR evidence");
    assert!(ev.reason.contains("G0 SKIP"), "Expected skip, got: {}", ev.reason);
    assert!(ev.reason.contains("No tensor template"), "Expected 'No tensor template': {}", ev.reason);
}

/// inspect_model_json returns success=true but invalid JSON → falsified (lines 107-118)
#[test]
fn test_g0_tensor_inspect_invalid_json_falsified() {
    use crate::command::{CommandOutput, CommandRunner};
    use std::path::Path;

    struct InvalidJsonInspector;
    impl CommandRunner for InvalidJsonInspector {
        fn inspect_model_json(&self, _: &Path) -> CommandOutput {
            CommandOutput::success("this is not json {{{{")
        }
        fn run_inference(&self, _: &Path, _: &str, _: u32, _: bool, _: &[&str]) -> CommandOutput { CommandOutput::success("") }
        fn convert_model(&self, _: &Path, _: &Path) -> CommandOutput { CommandOutput::success("") }
        fn inspect_model(&self, _: &Path) -> CommandOutput { CommandOutput::success("") }
        fn validate_model(&self, _: &Path) -> CommandOutput { CommandOutput::success("") }
        fn bench_model(&self, _: &Path) -> CommandOutput { CommandOutput::success("") }
        fn check_model(&self, _: &Path) -> CommandOutput { CommandOutput::success("") }
        fn profile_model(&self, _: &Path, _: u32, _: u32) -> CommandOutput { CommandOutput::success("") }
        fn profile_ci(&self, _: &Path, _: Option<f64>, _: Option<f64>, _: u32, _: u32, _: bool) -> CommandOutput { CommandOutput::success("") }
        fn diff_tensors(&self, _: &Path, _: &Path, _: bool) -> CommandOutput { CommandOutput::success("") }
        fn compare_inference(&self, _: &Path, _: &Path, _: &str, _: u32, _: f64) -> CommandOutput { CommandOutput::success("") }
        fn profile_with_flamegraph(&self, _: &Path, _: &Path, _: bool) -> CommandOutput { CommandOutput::success("") }
        fn profile_with_focus(&self, _: &Path, _: &str, _: bool) -> CommandOutput { CommandOutput::success("") }
        fn validate_model_strict(&self, _: &Path) -> CommandOutput { CommandOutput::success("") }
        fn fingerprint_model(&self, _: &Path, _: bool) -> CommandOutput { CommandOutput::success("") }
        fn validate_stats(&self, _: &Path, _: &Path) -> CommandOutput { CommandOutput::success("") }
        fn pull_model(&self, _: &str) -> CommandOutput { CommandOutput::success("") }
        fn run_ollama_inference(&self, _: &str, _: &str, _: f64) -> CommandOutput { CommandOutput::success("") }
        fn pull_ollama_model(&self, _: &str) -> CommandOutput { CommandOutput::success("") }
        fn create_ollama_model(&self, _: &str, _: &Path) -> CommandOutput { CommandOutput::success("") }
        fn serve_model(&self, _: &Path, _: u16) -> CommandOutput { CommandOutput::success("") }
        fn http_get(&self, _: &str) -> CommandOutput { CommandOutput::success("") }
        fn profile_memory(&self, _: &Path) -> CommandOutput { CommandOutput::success("") }
        fn run_chat(&self, _: &Path, _: &str, _: bool, _: &[&str]) -> CommandOutput { CommandOutput::success("") }
        fn http_post(&self, _: &str, _: &str) -> CommandOutput { CommandOutput::success("") }
        fn spawn_serve(&self, _: &Path, _: u16, _: bool) -> CommandOutput { CommandOutput::success("") }
        fn quantize_model(&self, _: &Path, _: &Path, _: &str) -> CommandOutput { CommandOutput::success("") }
        fn import_model(&self, _: &Path, _: &Path) -> CommandOutput { CommandOutput::success("") }
        fn prune_model(&self, _: &Path, _: &Path, _: &str, _: f64) -> CommandOutput { CommandOutput::success("") }
        fn distill_model(&self, _: &Path, _: &Path, _: &Path, _: &str) -> CommandOutput { CommandOutput::success("") }
    }

    let dir = make_temp_model_dir();
    let contracts_dir = tempfile::TempDir::new().expect("create contracts dir");
    let family_yaml = r#"
family: testfamily2
size_variants:
  1b:
    parameters: "1B"
    hidden_dim: 1024
    num_layers: 1
    num_heads: 8
tensor_template:
  embedding: "embed.weight"
"#;
    std::fs::write(contracts_dir.path().join("testfamily2.yaml"), family_yaml)
        .expect("write family yaml");

    let config = ExecutionConfig {
        model_path: Some(dir.path().to_string_lossy().to_string()),
        run_conversion_tests: false,
        run_golden_rule_test: false,
        run_contract_tests: false,
        ..Default::default()
    };
    let mut executor = Executor::with_runner(config, Arc::new(InvalidJsonInspector));
    let model_id = ModelId::new("test", "model");

    let (passed, failed) = executor.run_g0_tensor_template_check(
        dir.path(),
        &model_id,
        "testfamily2",
        "1b",
        Some(contracts_dir.path().to_str().expect("path")),
    );

    assert_eq!(passed, 0);
    assert_eq!(failed, 1);
    let evidence = executor.evidence().all();
    let ev = evidence.iter().find(|e| e.gate_id == "G0-TENSOR-001").expect("G0-TENSOR evidence");
    assert!(ev.reason.contains("G0 FAIL"), "Expected G0 FAIL, got: {}", ev.reason);
    assert!(ev.reason.contains("invalid JSON") || ev.reason.contains("JSON"),
        "Expected JSON error in reason: {}", ev.reason);
}

/// inspect_model_json returns valid JSON but without tensor_names → falsified (lines 121-132)
#[test]
fn test_g0_tensor_inspect_no_tensor_names_field_falsified() {
    use crate::command::{CommandOutput, CommandRunner};
    use std::path::Path;

    struct NoTensorNamesInspector;
    impl CommandRunner for NoTensorNamesInspector {
        fn inspect_model_json(&self, _: &Path) -> CommandOutput {
            // Valid JSON but no tensor_names field → actual_tensors.is_empty()
            CommandOutput::success(r#"{"format":"SafeTensors","tensor_count":5}"#)
        }
        fn run_inference(&self, _: &Path, _: &str, _: u32, _: bool, _: &[&str]) -> CommandOutput { CommandOutput::success("") }
        fn convert_model(&self, _: &Path, _: &Path) -> CommandOutput { CommandOutput::success("") }
        fn inspect_model(&self, _: &Path) -> CommandOutput { CommandOutput::success("") }
        fn validate_model(&self, _: &Path) -> CommandOutput { CommandOutput::success("") }
        fn bench_model(&self, _: &Path) -> CommandOutput { CommandOutput::success("") }
        fn check_model(&self, _: &Path) -> CommandOutput { CommandOutput::success("") }
        fn profile_model(&self, _: &Path, _: u32, _: u32) -> CommandOutput { CommandOutput::success("") }
        fn profile_ci(&self, _: &Path, _: Option<f64>, _: Option<f64>, _: u32, _: u32, _: bool) -> CommandOutput { CommandOutput::success("") }
        fn diff_tensors(&self, _: &Path, _: &Path, _: bool) -> CommandOutput { CommandOutput::success("") }
        fn compare_inference(&self, _: &Path, _: &Path, _: &str, _: u32, _: f64) -> CommandOutput { CommandOutput::success("") }
        fn profile_with_flamegraph(&self, _: &Path, _: &Path, _: bool) -> CommandOutput { CommandOutput::success("") }
        fn profile_with_focus(&self, _: &Path, _: &str, _: bool) -> CommandOutput { CommandOutput::success("") }
        fn validate_model_strict(&self, _: &Path) -> CommandOutput { CommandOutput::success("") }
        fn fingerprint_model(&self, _: &Path, _: bool) -> CommandOutput { CommandOutput::success("") }
        fn validate_stats(&self, _: &Path, _: &Path) -> CommandOutput { CommandOutput::success("") }
        fn pull_model(&self, _: &str) -> CommandOutput { CommandOutput::success("") }
        fn run_ollama_inference(&self, _: &str, _: &str, _: f64) -> CommandOutput { CommandOutput::success("") }
        fn pull_ollama_model(&self, _: &str) -> CommandOutput { CommandOutput::success("") }
        fn create_ollama_model(&self, _: &str, _: &Path) -> CommandOutput { CommandOutput::success("") }
        fn serve_model(&self, _: &Path, _: u16) -> CommandOutput { CommandOutput::success("") }
        fn http_get(&self, _: &str) -> CommandOutput { CommandOutput::success("") }
        fn profile_memory(&self, _: &Path) -> CommandOutput { CommandOutput::success("") }
        fn run_chat(&self, _: &Path, _: &str, _: bool, _: &[&str]) -> CommandOutput { CommandOutput::success("") }
        fn http_post(&self, _: &str, _: &str) -> CommandOutput { CommandOutput::success("") }
        fn spawn_serve(&self, _: &Path, _: u16, _: bool) -> CommandOutput { CommandOutput::success("") }
        fn quantize_model(&self, _: &Path, _: &Path, _: &str) -> CommandOutput { CommandOutput::success("") }
        fn import_model(&self, _: &Path, _: &Path) -> CommandOutput { CommandOutput::success("") }
        fn prune_model(&self, _: &Path, _: &Path, _: &str, _: f64) -> CommandOutput { CommandOutput::success("") }
        fn distill_model(&self, _: &Path, _: &Path, _: &Path, _: &str) -> CommandOutput { CommandOutput::success("") }
    }

    let dir = make_temp_model_dir();
    let contracts_dir = tempfile::TempDir::new().expect("create contracts dir");
    let family_yaml = r#"
family: testfamily3
size_variants:
  1b:
    parameters: "1B"
    hidden_dim: 1024
    num_layers: 1
    num_heads: 8
tensor_template:
  embedding: "embed.weight"
"#;
    std::fs::write(contracts_dir.path().join("testfamily3.yaml"), family_yaml)
        .expect("write family yaml");

    let config = ExecutionConfig {
        model_path: Some(dir.path().to_string_lossy().to_string()),
        run_conversion_tests: false,
        run_golden_rule_test: false,
        run_contract_tests: false,
        ..Default::default()
    };
    let mut executor = Executor::with_runner(config, Arc::new(NoTensorNamesInspector));
    let model_id = ModelId::new("test", "model");

    let (passed, failed) = executor.run_g0_tensor_template_check(
        dir.path(),
        &model_id,
        "testfamily3",
        "1b",
        Some(contracts_dir.path().to_str().expect("path")),
    );

    assert_eq!(passed, 0);
    assert_eq!(failed, 1);
    let evidence = executor.evidence().all();
    let ev = evidence.iter().find(|e| e.gate_id == "G0-TENSOR-001").expect("G0-TENSOR evidence");
    assert!(ev.reason.contains("G0 FAIL"), "Expected G0 FAIL, got: {}", ev.reason);
    assert!(ev.reason.contains("no tensor names") || ev.reason.contains("tensor names"),
        "Expected 'no tensor names' in reason: {}", ev.reason);
}

// ── F-PERF-006: GPU/CPU ratio falsified branch (ratio < 1.0) ─────────────────

/// When both CPU and GPU return 0.0 tok/s, GPU/CPU ratio = 0.0 < 1.0 → F-PERF-006 falsified.
/// Covers the `ratio < 1.0` branch in `run_perf_gates` (lines 166-176).
#[test]
fn test_perf_006_gpu_slower_than_cpu_falsified() {
    // tps=0.0 → profile_ci returns {"throughput_tps":0.0,...}
    // cpu_tps=Some(0.0), gpu_tps=Some(0.0)
    // ratio = 0.0 / max(0.0, 0.01) = 0.0 / 0.01 = 0.0 < 1.0 → falsified
    let runner = Arc::new(MockCommandRunner::new().with_tps(0.0));
    let config = ExecutionConfig {
        run_profile_ci: true,
        model_path: Some("/mock/model".to_string()),
        ..Default::default()
    };
    let mut executor = Executor::with_runner(config, runner);

    let yaml = r#"
name: perf-006-falsified
version: "1.0.0"
model:
  hf_repo: "test/model"
test_matrix:
  modalities: [run]
  backends: [cpu, gpu]
  scenario_count: 1
profile_ci:
  enabled: true
  warmup: 1
  measure: 2
  formats: [safetensors]
  backends: [cpu, gpu]
"#;
    let playbook: Playbook = serde_yaml::from_str(yaml).expect("parse");
    let model_id = playbook.model_id();
    let (_passed, failed) = executor.run_perf_gates(Path::new("/mock/model"), &model_id, &playbook);

    // F-PERF-006 falsified (ratio < 1.0)
    assert!(failed >= 1, "Expected ≥1 failed for GPU slower than CPU");
    let evidence = executor.evidence().all();
    let perf_ev = evidence
        .iter()
        .find(|e| e.gate_id == "F-PERF-006")
        .expect("F-PERF-006 evidence");
    assert!(
        !perf_ev.outcome.is_pass(),
        "F-PERF-006 should be falsified when GPU/CPU ratio < 1.0"
    );
    assert!(
        perf_ev.reason.contains("ratio") || perf_ev.reason.contains("slower"),
        "Expected ratio/slower in reason: {}",
        perf_ev.reason
    );
}

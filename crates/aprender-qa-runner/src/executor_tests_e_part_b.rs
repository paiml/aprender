#[test]
fn test_g0_tensor_all_tensors_present() {
    // When all expected tensors are present, should pass
    let mock_runner = MockCommandRunner::new().with_tensor_names(vec![
        "embed.weight".to_string(),
        "model.layers.0.self_attn.q_proj.weight".to_string(),
    ]);
    let dir = make_temp_model_dir();

    // Create a temp contracts directory with a minimal family contract
    let contracts_dir = tempfile::TempDir::new().expect("create contracts dir");
    let family_yaml = r#"
family: testfamily
size_variants:
  1b:
    parameters: "1B"
    hidden_dim: 1024
    num_layers: 1
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

    // Should pass
    assert_eq!(passed, 1);
    assert_eq!(failed, 0);

    let evidence = executor.evidence().all();
    let tensor_ev = evidence
        .iter()
        .find(|e| e.gate_id == "G0-TENSOR-001")
        .expect("should have G0-TENSOR evidence");
    assert!(tensor_ev.output.contains("G0 PASS"));
}

#[test]
fn test_g0_tensor_missing_tensors() {
    // When expected tensors are missing, should fail
    let mock_runner = MockCommandRunner::new().with_tensor_names(vec![
        "some.other.tensor".to_string(), // Not the expected one
    ]);
    let dir = make_temp_model_dir();

    // Create a temp contracts directory with a minimal family contract
    let contracts_dir = tempfile::TempDir::new().expect("create contracts dir");
    let family_yaml = r#"
family: testfamily
size_variants:
  1b:
    parameters: "1B"
    hidden_dim: 1024
    num_layers: 1
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
    assert!(tensor_ev.reason.contains("Missing"));
    assert!(tensor_ev.reason.contains("embed.weight"));
}

// ── parse_timing_ms tests ──────────────────────────────────────────

#[test]
fn test_parse_timing_ms_standard() {
    let output = "Output:\nHello\nCompleted in 1.5s\ntok/s: 25.0";
    assert!((parse_timing_ms(output).unwrap() - 1500.0).abs() < 0.1);
}

#[test]
fn test_parse_timing_ms_no_timing() {
    let output = "Just some output without timing";
    assert!(parse_timing_ms(output).is_none());
}

#[test]
fn test_parse_timing_ms_zero() {
    let output = "Completed in 0.0s";
    assert!((parse_timing_ms(output).unwrap()).abs() < 0.1);
}

// ── parse_throughput tests ──────────────────────────────────────────

#[test]
fn test_parse_throughput_json() {
    let output = r#"{"throughput_tps":25.0,"latency_p50_ms":78.2}"#;
    assert!((parse_throughput(output).unwrap() - 25.0).abs() < 0.1);
}

#[test]
fn test_parse_throughput_no_match() {
    let output = "no json here";
    assert!(parse_throughput(output).is_none());
}

#[test]
fn test_parse_throughput_integer() {
    let output = r#"{"throughput_tps":100,"other":0}"#;
    assert!((parse_throughput(output).unwrap() - 100.0).abs() < 0.1);
}

// ── F-OLLAMA-003 TTFT comparison test ──────────────────────────────

#[test]
fn test_ollama_parity_ttft_comparison() {
    let runner = MockCommandRunner::new().with_inference_response("Hello world");
    let runner = Arc::new(runner);

    let config = ExecutionConfig {
        run_ollama_parity: true,
        model_path: Some("/mock/model".to_string()),
        ..Default::default()
    };
    let mut executor = Executor::with_runner(config, runner);

    let yaml = r#"
name: test-ollama-ttft
version: "1.0.0"
model:
  hf_repo: "test/model"
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
ollama_parity:
  enabled: true
  model_tag: "test:latest"
  prompts: ["What is 2+2?"]
  temperature: 0.0
"#;
    let playbook: Playbook = serde_yaml::from_str(yaml).unwrap();
    let (passed, failed) = executor.run_ollama_parity_tests(Path::new("/mock/model"), &playbook);
    // F-OLLAMA-001 + F-OLLAMA-003 (TTFT) + F-OLLAMA-005 + F-OLLAMA-004
    assert!(
        passed + failed >= 2,
        "Expected at least 2 evidence items, got passed={passed} failed={failed}"
    );
}

// ── F-OLLAMA-005 GGUF loadability test ─────────────────────────────

#[test]
fn test_ollama_gguf_loadability_success() {
    let runner = Arc::new(MockCommandRunner::new());
    let config = ExecutionConfig {
        run_ollama_parity: true,
        model_path: Some("/mock/model".to_string()),
        ..Default::default()
    };
    let mut executor = Executor::with_runner(config, runner);

    let yaml = r#"
name: test-ollama-gguf
version: "1.0.0"
model:
  hf_repo: "test/model"
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
ollama_parity:
  enabled: true
  prompts: ["test"]
"#;
    let playbook: Playbook = serde_yaml::from_str(yaml).unwrap();
    let (passed, failed) = executor.run_ollama_parity_tests(Path::new("/mock/model"), &playbook);
    // F-OLLAMA-001 (output match), F-OLLAMA-005 (GGUF loadability), F-OLLAMA-004 (API)
    // F-OLLAMA-001 may now FAIL if APR and Ollama produce different text (Bug #32 fix)
    assert!(passed + failed >= 3, "Expected at least 3 evidence items, got passed={passed} failed={failed}");
    let evidence = executor.evidence().all();
    assert!(evidence.iter().any(|e| e.gate_id == "F-OLLAMA-005"));
    // F-OLLAMA-005 and F-OLLAMA-004 should still pass (ecosystem gates)
    let gguf_ev = evidence.iter().find(|e| e.gate_id == "F-OLLAMA-005").unwrap();
    assert!(gguf_ev.outcome.is_pass());
    let api_ev = evidence.iter().find(|e| e.gate_id == "F-OLLAMA-004").unwrap();
    assert!(api_ev.outcome.is_pass());
}

#[test]
fn test_ollama_gguf_loadability_failure() {
    let runner = Arc::new(MockCommandRunner::new().with_ollama_create_failure());
    let config = ExecutionConfig {
        run_ollama_parity: true,
        model_path: Some("/mock/model".to_string()),
        ..Default::default()
    };
    let mut executor = Executor::with_runner(config, runner);

    let yaml = r#"
name: test-ollama-gguf-fail
version: "1.0.0"
model:
  hf_repo: "test/model"
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
ollama_parity:
  enabled: true
  prompts: ["test"]
"#;
    let playbook: Playbook = serde_yaml::from_str(yaml).unwrap();
    let (_passed, failed) = executor.run_ollama_parity_tests(Path::new("/mock/model"), &playbook);
    assert!(
        failed >= 1,
        "Expected at least 1 failure for create failure"
    );
    let evidence = executor.evidence().all();
    let gguf_ev = evidence
        .iter()
        .find(|e| e.gate_id == "F-OLLAMA-005")
        .unwrap();
    assert!(!gguf_ev.outcome.is_pass());
}

// ── F-OLLAMA-004 API parity test ───────────────────────────────────

#[test]
fn test_ollama_api_parity_success() {
    let runner = Arc::new(MockCommandRunner::new());
    let config = ExecutionConfig {
        run_ollama_parity: true,
        model_path: Some("/mock/model".to_string()),
        ..Default::default()
    };
    let mut executor = Executor::with_runner(config, runner);

    let yaml = r#"
name: test-ollama-api
version: "1.0.0"
model:
  hf_repo: "test/model"
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
ollama_parity:
  enabled: true
  prompts: ["test"]
"#;
    let playbook: Playbook = serde_yaml::from_str(yaml).unwrap();
    let (passed, _failed) = executor.run_ollama_parity_tests(Path::new("/mock/model"), &playbook);
    assert!(passed >= 1);
    let evidence = executor.evidence().all();
    assert!(evidence.iter().any(|e| e.gate_id == "F-OLLAMA-004"));
}

#[test]
fn test_ollama_api_parity_failure() {
    let runner = Arc::new(MockCommandRunner::new().with_http_get_failure());
    let config = ExecutionConfig {
        run_ollama_parity: true,
        model_path: Some("/mock/model".to_string()),
        ..Default::default()
    };
    let mut executor = Executor::with_runner(config, runner);

    let yaml = r#"
name: test-ollama-api-fail
version: "1.0.0"
model:
  hf_repo: "test/model"
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
ollama_parity:
  enabled: true
  prompts: ["test"]
"#;
    let playbook: Playbook = serde_yaml::from_str(yaml).unwrap();
    let (_passed, failed) = executor.run_ollama_parity_tests(Path::new("/mock/model"), &playbook);
    assert!(failed >= 1);
    let evidence = executor.evidence().all();
    let api_ev = evidence
        .iter()
        .find(|e| e.gate_id == "F-OLLAMA-004")
        .unwrap();
    assert!(!api_ev.outcome.is_pass());
}

// ── F-PERF-006 GPU/CPU ratio test ──────────────────────────────────
// Note: F-PERF-003 = Memory Leak (patterns_spec_gates.rs). F-PERF-006 = GPU/CPU ratio.

#[test]
fn test_perf_006_gpu_cpu_ratio() {
    let runner = Arc::new(MockCommandRunner::new().with_tps(50.0));
    let config = ExecutionConfig {
        run_profile_ci: true,
        model_path: Some("/mock/model".to_string()),
        ..Default::default()
    };
    let mut executor = Executor::with_runner(config, runner);

    let yaml = r#"
name: test-perf-003
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
    let playbook: Playbook = serde_yaml::from_str(yaml).unwrap();
    let model_id = playbook.model_id();
    let (passed, _failed) = executor.run_perf_gates(Path::new("/mock/model"), &model_id, &playbook);
    // F-PERF-006 (GPU/CPU ratio) + F-PERF-005 (memory profiling)
    assert!(passed >= 2, "Expected at least 2 passes, got {passed}");
    let evidence = executor.evidence().all();
    assert!(evidence.iter().any(|e| e.gate_id == "F-PERF-006"));
    assert!(evidence.iter().any(|e| e.gate_id == "F-PERF-005"));
}

// ── F-PERF-005 memory profiling test ───────────────────────────────

#[test]
fn test_perf_005_memory_profiling_failure() {
    let runner = Arc::new(MockCommandRunner::new().with_profile_memory_failure());
    let config = ExecutionConfig {
        run_profile_ci: true,
        model_path: Some("/mock/model".to_string()),
        ..Default::default()
    };
    let mut executor = Executor::with_runner(config, runner);

    let yaml = r#"
name: test-perf-005-fail
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
  backends: [cpu]
"#;
    let playbook: Playbook = serde_yaml::from_str(yaml).unwrap();
    let model_id = playbook.model_id();
    let (_passed, failed) = executor.run_perf_gates(Path::new("/mock/model"), &model_id, &playbook);
    assert!(failed >= 1);
    let evidence = executor.evidence().all();
    let mem_ev = evidence.iter().find(|e| e.gate_id == "F-PERF-005").unwrap();
    assert!(!mem_ev.outcome.is_pass());
}

// ── Integration: execute() with ollama parity enabled ─────────────

#[test]
fn test_execute_with_ollama_parity_enabled() {
    let runner =
        MockCommandRunner::new().with_inference_response("Output:\nHello\nCompleted in 0.5s");
    let config = ExecutionConfig {
        run_ollama_parity: true,
        model_path: Some("/mock/model".to_string()),
        no_gpu: true,
        ..Default::default()
    };
    let mut executor = Executor::with_runner(config, Arc::new(runner));

    let yaml = r#"
name: test-ollama-integration
version: "1.0.0"
model:
  hf_repo: "test/model"
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
ollama_parity:
  enabled: true
  prompts: ["What is 2+2?"]
"#;
    let playbook = Playbook::from_yaml(yaml).expect("Failed to parse");
    let result = executor.execute(&playbook).expect("Execution failed");
    assert!(result.total_scenarios >= 1);
    let evidence = executor.evidence().all();
    assert!(evidence.iter().any(|e| e.gate_id == "F-OLLAMA-001"));
}

// ── run_ollama_prompt_gates: uncovered branches ─────────────────────────────

fn ollama_parity_playbook_with_prompt(name: &str, prompt: &str) -> Playbook {
    let yaml = format!(
        r#"
name: {name}
version: "1.0.0"
model:
  hf_repo: "test/model"
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
ollama_parity:
  enabled: true
  model_tag: "test:latest"
  prompts: ["{prompt}"]
  temperature: 0.0
"#
    );
    Playbook::from_yaml(&yaml).expect("Failed to parse ollama playbook")
}

/// F-OLLAMA-001: ollama inference failure → falsified "Ollama inference failed"
#[test]
fn test_ollama_prompt_gates_ollama_inference_failure() {
    let runner = Arc::new(MockCommandRunner::new().with_ollama_failure());
    let config = ExecutionConfig {
        run_ollama_parity: true,
        model_path: Some("/mock/model".to_string()),
        ..Default::default()
    };
    let mut executor = Executor::with_runner(config, runner);
    let playbook = ollama_parity_playbook_with_prompt("ollama-fail", "What is 2+2?");

    let (_passed, failed) = executor.run_ollama_parity_tests(Path::new("/mock/model"), &playbook);
    assert!(failed >= 1, "Expected ≥1 failure for ollama inference failure");

    let evidence = executor.evidence().all();
    let ol_ev = evidence
        .iter()
        .find(|e| e.gate_id == "F-OLLAMA-001" && e.outcome.is_fail());
    assert!(ol_ev.is_some(), "Expected F-OLLAMA-001 falsified evidence");
    assert!(
        ol_ev.unwrap().reason.contains("Ollama inference failed"),
        "Expected 'Ollama inference failed' reason, got: {}",
        ol_ev.unwrap().reason
    );
}

/// F-OLLAMA-001: APR inference failure → falsified "APR inference failed"
#[test]
fn test_ollama_prompt_gates_apr_inference_failure() {
    let runner = Arc::new(MockCommandRunner::new().with_inference_failure());
    let config = ExecutionConfig {
        run_ollama_parity: true,
        model_path: Some("/mock/model".to_string()),
        ..Default::default()
    };
    let mut executor = Executor::with_runner(config, runner);
    let playbook = ollama_parity_playbook_with_prompt("apr-fail", "What is 2+2?");

    let (_passed, failed) = executor.run_ollama_parity_tests(Path::new("/mock/model"), &playbook);
    assert!(failed >= 1, "Expected ≥1 failure for APR inference failure");

    let evidence = executor.evidence().all();
    let ol_ev = evidence
        .iter()
        .find(|e| e.gate_id == "F-OLLAMA-001" && e.outcome.is_fail());
    assert!(ol_ev.is_some(), "Expected F-OLLAMA-001 falsified evidence");
    assert!(
        ol_ev.unwrap().reason.contains("APR inference failed"),
        "Expected 'APR inference failed' reason, got: {}",
        ol_ev.unwrap().reason
    );
}

/// F-OLLAMA-001: identical extracted outputs → corroborated
///
/// APR returns "Output:\nhello\nCompleted in 1.5s" → extract_output_text → "hello"
/// Ollama returns "hello" (plain, no wrapper) → ollama_text = "hello"
/// → apr_text == ollama_text → corroborated
#[test]
fn test_ollama_prompt_gates_corroborated_matching_output() {
    use crate::command::CommandOutput;
    struct MatchingOllamaRunner;
    impl CommandRunner for MatchingOllamaRunner {
        fn run_inference(
            &self, _: &Path, _: &str, _: u32, _: bool, _: &[&str],
        ) -> CommandOutput {
            CommandOutput::success("Output:\nhello\nCompleted in 1.5s")
        }
        fn run_ollama_inference(&self, _: &str, _: &str, _: f64) -> CommandOutput {
            // Return plain text without "Output:" wrapper so ollama_text == apr_text
            CommandOutput::success("hello")
        }
        fn convert_model(&self, _: &Path, _: &Path) -> CommandOutput { CommandOutput::success("") }
        fn inspect_model(&self, _: &Path) -> CommandOutput { CommandOutput::success("") }
        fn validate_model(&self, _: &Path) -> CommandOutput { CommandOutput::success("") }
        fn bench_model(&self, _: &Path) -> CommandOutput { CommandOutput::success("") }
        fn check_model(&self, _: &Path) -> CommandOutput { CommandOutput::success("All checks passed") }
        fn profile_model(&self, _: &Path, _: u32, _: u32) -> CommandOutput { CommandOutput::success("") }
        fn profile_ci(&self, _: &Path, _: Option<f64>, _: Option<f64>, _: u32, _: u32, _: bool) -> CommandOutput { CommandOutput::success("") }
        fn diff_tensors(&self, _: &Path, _: &Path, _: bool) -> CommandOutput { CommandOutput::success("") }
        fn compare_inference(&self, _: &Path, _: &Path, _: &str, _: u32, _: f64) -> CommandOutput { CommandOutput::success("") }
        fn profile_with_flamegraph(&self, _: &Path, _: &Path, _: bool) -> CommandOutput { CommandOutput::success("") }
        fn profile_with_focus(&self, _: &Path, _: &str, _: bool) -> CommandOutput { CommandOutput::success("") }
        fn fingerprint_model(&self, _: &Path, _: bool) -> CommandOutput { CommandOutput::success("") }
        fn validate_stats(&self, _: &Path, _: &Path) -> CommandOutput { CommandOutput::success("") }
        fn validate_model_strict(&self, _: &Path) -> CommandOutput { CommandOutput::success(r#"{"valid":true,"tensors_checked":0,"issues":[]}"#) }
        fn pull_model(&self, _: &str) -> CommandOutput { CommandOutput::success("Path: /mock/model.safetensors") }
        fn inspect_model_json(&self, _: &Path) -> CommandOutput { CommandOutput::success(r#"{"format":"SafeTensors","tensor_count":0,"tensor_names":[]}"#) }
        fn pull_ollama_model(&self, _: &str) -> CommandOutput { CommandOutput::success("done") }
        fn create_ollama_model(&self, _: &str, _: &Path) -> CommandOutput { CommandOutput::success("done") }
        fn serve_model(&self, _: &Path, _: u16) -> CommandOutput { CommandOutput::success(r#"{"status":"listening"}"#) }
        fn http_get(&self, _: &str) -> CommandOutput { CommandOutput::success(r#"{"models":[]}"#) }
        fn profile_memory(&self, _: &Path) -> CommandOutput { CommandOutput::success(r#"{"peak_rss_mb":512}"#) }
        fn run_chat(&self, _: &Path, _: &str, _: bool, _: &[&str]) -> CommandOutput { CommandOutput::success("") }
        fn http_post(&self, _: &str, _: &str) -> CommandOutput { CommandOutput::success("{}") }
        fn spawn_serve(&self, _: &Path, _: u16, _: bool) -> CommandOutput { CommandOutput::success("12345") }
        fn quantize_model(&self, _: &Path, _: &Path, _: &str) -> CommandOutput { CommandOutput::success("") }
        fn import_model(&self, _: &Path, _: &Path) -> CommandOutput { CommandOutput::success("") }
        fn prune_model(&self, _: &Path, _: &Path, _: &str, _: f64) -> CommandOutput { CommandOutput::success("") }
        fn distill_model(&self, _: &Path, _: &Path, _: &Path, _: &str) -> CommandOutput { CommandOutput::success("") }
    }

    let config = ExecutionConfig {
        run_ollama_parity: true,
        model_path: Some("/mock/model".to_string()),
        ..Default::default()
    };
    let mut executor = Executor::with_runner(config, Arc::new(MatchingOllamaRunner));
    let playbook = ollama_parity_playbook_with_prompt("ollama-corr", "hello test");

    let (passed, _failed) = executor.run_ollama_parity_tests(Path::new("/mock/model"), &playbook);
    assert!(passed >= 1, "Expected ≥1 passed for matching output");

    let evidence = executor.evidence().all();
    let corr = evidence
        .iter()
        .find(|e| e.gate_id == "F-OLLAMA-001" && e.outcome.is_pass());
    assert!(corr.is_some(), "Expected F-OLLAMA-001 corroborated evidence");
}

/// F-OLLAMA-003: TTFT ratio > 3.0 → falsified "TTFT ratio exceeds 3.0x threshold"
///
/// APR returns "Completed in 4.0s" (4000ms), Ollama returns "Completed in 1.0s" (1000ms)
/// ratio = 4.0 > 3.0 → TTFT FAILED (lines 696-705)
#[test]
fn test_ollama_prompt_gates_ttft_ratio_exceeded() {
    use crate::command::CommandOutput;
    struct SlowAprRunner;
    impl CommandRunner for SlowAprRunner {
        fn run_inference(
            &self, _: &Path, _: &str, _: u32, _: bool, _: &[&str],
        ) -> CommandOutput {
            // High timing to trigger TTFT failure (4000ms >> 3x Ollama 1000ms)
            CommandOutput::success("Output:\nhello slow\nCompleted in 4.0s")
        }
        fn run_ollama_inference(&self, _: &str, _: &str, _: f64) -> CommandOutput {
            CommandOutput::success("Output:\nhello fast\nCompleted in 1.0s")
        }
        fn convert_model(&self, _: &Path, _: &Path) -> CommandOutput { CommandOutput::success("") }
        fn inspect_model(&self, _: &Path) -> CommandOutput { CommandOutput::success("") }
        fn validate_model(&self, _: &Path) -> CommandOutput { CommandOutput::success("") }
        fn bench_model(&self, _: &Path) -> CommandOutput { CommandOutput::success("") }
        fn check_model(&self, _: &Path) -> CommandOutput { CommandOutput::success("All checks passed") }
        fn profile_model(&self, _: &Path, _: u32, _: u32) -> CommandOutput { CommandOutput::success("") }
        fn profile_ci(&self, _: &Path, _: Option<f64>, _: Option<f64>, _: u32, _: u32, _: bool) -> CommandOutput { CommandOutput::success("") }
        fn diff_tensors(&self, _: &Path, _: &Path, _: bool) -> CommandOutput { CommandOutput::success("") }
        fn compare_inference(&self, _: &Path, _: &Path, _: &str, _: u32, _: f64) -> CommandOutput { CommandOutput::success("") }
        fn profile_with_flamegraph(&self, _: &Path, _: &Path, _: bool) -> CommandOutput { CommandOutput::success("") }
        fn profile_with_focus(&self, _: &Path, _: &str, _: bool) -> CommandOutput { CommandOutput::success("") }
        fn fingerprint_model(&self, _: &Path, _: bool) -> CommandOutput { CommandOutput::success("") }
        fn validate_stats(&self, _: &Path, _: &Path) -> CommandOutput { CommandOutput::success("") }
        fn validate_model_strict(&self, _: &Path) -> CommandOutput { CommandOutput::success(r#"{"valid":true,"tensors_checked":0,"issues":[]}"#) }
        fn pull_model(&self, _: &str) -> CommandOutput { CommandOutput::success("Path: /mock/model.safetensors") }
        fn inspect_model_json(&self, _: &Path) -> CommandOutput { CommandOutput::success(r#"{"format":"SafeTensors","tensor_count":0,"tensor_names":[]}"#) }
        fn pull_ollama_model(&self, _: &str) -> CommandOutput { CommandOutput::success("done") }
        fn create_ollama_model(&self, _: &str, _: &Path) -> CommandOutput { CommandOutput::success("done") }
        fn serve_model(&self, _: &Path, _: u16) -> CommandOutput { CommandOutput::success(r#"{"status":"listening"}"#) }
        fn http_get(&self, _: &str) -> CommandOutput { CommandOutput::success(r#"{"models":[]}"#) }
        fn profile_memory(&self, _: &Path) -> CommandOutput { CommandOutput::success(r#"{"peak_rss_mb":512}"#) }
        fn run_chat(&self, _: &Path, _: &str, _: bool, _: &[&str]) -> CommandOutput { CommandOutput::success("") }
        fn http_post(&self, _: &str, _: &str) -> CommandOutput { CommandOutput::success("{}") }
        fn spawn_serve(&self, _: &Path, _: u16, _: bool) -> CommandOutput { CommandOutput::success("12345") }
        fn quantize_model(&self, _: &Path, _: &Path, _: &str) -> CommandOutput { CommandOutput::success("") }
        fn import_model(&self, _: &Path, _: &Path) -> CommandOutput { CommandOutput::success("") }
        fn prune_model(&self, _: &Path, _: &Path, _: &str, _: f64) -> CommandOutput { CommandOutput::success("") }
        fn distill_model(&self, _: &Path, _: &Path, _: &Path, _: &str) -> CommandOutput { CommandOutput::success("") }
    }

    let config = ExecutionConfig {
        run_ollama_parity: true,
        model_path: Some("/mock/model".to_string()),
        ..Default::default()
    };
    let mut executor = Executor::with_runner(config, Arc::new(SlowAprRunner));
    let playbook = ollama_parity_playbook_with_prompt("ollama-ttft-fail", "hello test");

    let (_passed, failed) = executor.run_ollama_parity_tests(Path::new("/mock/model"), &playbook);
    // Output differs + TTFT ratio > 3.0 → both emit falsified evidence
    assert!(failed >= 1, "Expected ≥1 failure for high TTFT ratio");

    let evidence = executor.evidence().all();
    let ttft_ev = evidence
        .iter()
        .find(|e| e.gate_id == "F-OLLAMA-003" && e.outcome.is_fail());
    assert!(
        ttft_ev.is_some(),
        "Expected F-OLLAMA-003 falsified (TTFT exceeded), got gates: {:?}",
        evidence.iter().map(|e| &e.gate_id).collect::<Vec<_>>()
    );
    assert!(
        ttft_ev.unwrap().reason.contains("ratio"),
        "Expected ratio in reason, got: {}",
        ttft_ev.unwrap().reason
    );
}

// ── run_ollama_parity_tests: uncovered entry branches ──────────────────────

/// run_ollama_parity_tests: pull_ollama_model fails → F-OLLAMA-PULL-001 falsified + early return
/// (gates.rs/golden.rs lines 550-567)
#[test]
fn test_ollama_parity_pull_failure_early_return() {
    let runner = Arc::new(MockCommandRunner::new().with_ollama_pull_failure());
    let config = ExecutionConfig {
        run_ollama_parity: true,
        model_path: Some("/mock/model".to_string()),
        ..Default::default()
    };
    let mut executor = Executor::with_runner(config, runner);

    let yaml = r#"
name: ollama-pull-fail
version: "1.0.0"
model:
  hf_repo: "test/model"
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
ollama_parity:
  enabled: true
  model_tag: "test:latest"
  prompts: ["What is 2+2?"]
"#;
    let playbook: Playbook = serde_yaml::from_str(yaml).unwrap();
    let (passed, failed) =
        executor.run_ollama_parity_tests(Path::new("/mock/model"), &playbook);
    assert_eq!(passed, 0, "Expected 0 passed when pull fails");
    assert_eq!(failed, 1, "Expected 1 failure for pull failure");
    let evidence = executor.evidence().all();
    let pull_ev = evidence
        .iter()
        .find(|e| e.gate_id == "F-OLLAMA-PULL-001" && e.outcome.is_fail());
    assert!(
        pull_ev.is_some(),
        "Expected F-OLLAMA-PULL-001 falsified, got: {:?}",
        evidence.iter().map(|e| &e.gate_id).collect::<Vec<_>>()
    );
}

/// run_ollama_parity_tests: ollama_parity not configured → F-OLLAMA-PARITY-SKIP-001 skipped
/// (golden.rs lines 519-537)
#[test]
fn test_ollama_parity_not_configured_skipped() {
    let runner = Arc::new(MockCommandRunner::new());
    let config = ExecutionConfig {
        run_ollama_parity: true,
        model_path: Some("/mock/model".to_string()),
        ..Default::default()
    };
    let mut executor = Executor::with_runner(config, runner);

    // No ollama_parity section → triggers skip branch
    let yaml = r#"
name: ollama-not-configured
version: "1.0.0"
model:
  hf_repo: "test/model"
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
"#;
    let playbook: Playbook = serde_yaml::from_str(yaml).unwrap();
    let (passed, failed) =
        executor.run_ollama_parity_tests(Path::new("/mock/model"), &playbook);
    assert_eq!(passed, 0);
    assert_eq!(failed, 0);
    let evidence = executor.evidence().all();
    assert!(
        evidence
            .iter()
            .any(|e| e.gate_id == "F-OLLAMA-PARITY-SKIP-001" && !e.outcome.is_fail()),
        "Expected F-OLLAMA-PARITY-SKIP-001 skipped evidence"
    );
}

// ── Integration: execute() with profile_ci (perf gates) enabled ───

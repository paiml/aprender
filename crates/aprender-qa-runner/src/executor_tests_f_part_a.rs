#[test]
fn test_truncate_str_utf8_boundary() {
    // Multi-byte UTF-8: each char is 2+ bytes
    let s = "\u{00e9}\u{00e9}\u{00e9}"; // 3 x 'e-acute' (2 bytes each = 6 bytes)
    let result = Executor::truncate_str(s, 3);
    // Should not split in the middle of a char boundary
    assert_eq!(result.len(), 2); // Only the first complete char fits within 3 bytes
}

// ── strip_ansi ──────────────────────────────────────────────────────

#[test]
fn test_strip_ansi_no_escapes() {
    assert_eq!(Executor::strip_ansi("hello"), "hello");
}

#[test]
fn test_strip_ansi_with_color() {
    let colored = "\x1b[32m/path/to/model\x1b[0m";
    assert_eq!(Executor::strip_ansi(colored), "/path/to/model");
}

#[test]
fn test_strip_ansi_empty() {
    assert_eq!(Executor::strip_ansi(""), "");
}

#[test]
fn test_strip_ansi_multiple_sequences() {
    let s = "\x1b[1m\x1b[34mBold Blue\x1b[0m Normal";
    assert_eq!(Executor::strip_ansi(s), "Bold Blue Normal");
}

// ── extract_model_family_prefix ─────────────────────────────────────

#[test]
fn test_extract_model_family_prefix_with_quant() {
    let prefix = Executor::extract_model_family_prefix("qwen2.5-coder-7b-instruct-q4k");
    assert_eq!(prefix, "qwen2.5-coder-7b");
}

#[test]
fn test_extract_model_family_prefix_no_quant() {
    let prefix = Executor::extract_model_family_prefix("qwen2.5-coder-1.5b");
    assert_eq!(prefix, "qwen2.5-coder-1.5b");
}

#[test]
fn test_extract_model_family_prefix_instruct_suffix() {
    let prefix = Executor::extract_model_family_prefix("qwen2.5-coder-7b-instruct");
    assert_eq!(prefix, "qwen2.5-coder-7b");
}

#[test]
fn test_extract_model_family_prefix_q4_k_m() {
    let prefix = Executor::extract_model_family_prefix("TinyLlama-1.1B-Chat-v1.0-Q4_K_M");
    assert_eq!(prefix, "TinyLlama-1.1B-Chat-v1.0");
}

#[test]
fn test_extract_model_family_prefix_f16() {
    let prefix = Executor::extract_model_family_prefix("model-name-f16");
    assert_eq!(prefix, "model-name");
}

// ── should_stop_on_failure ───────────────────────────────────────────

#[test]
fn test_should_stop_on_failure_stop_on_first() {
    let config = ExecutionConfig {
        failure_policy: FailurePolicy::StopOnFirst,
        ..Default::default()
    };
    let executor = Executor::with_config(config);
    let evidence = Evidence::falsified("G2-BASIC", test_scenario(), "test failure", "output", 0);
    assert!(executor.should_stop_on_failure(&evidence, "test"));
}

#[test]
fn test_should_stop_on_failure_collect_all() {
    let config = ExecutionConfig {
        failure_policy: FailurePolicy::CollectAll,
        ..Default::default()
    };
    let executor = Executor::with_config(config);
    let evidence = Evidence::falsified("G2-BASIC", test_scenario(), "test failure", "output", 0);
    assert!(!executor.should_stop_on_failure(&evidence, "test"));
}

#[test]
fn test_should_stop_on_failure_stop_on_p0_with_p0_gate() {
    let config = ExecutionConfig {
        failure_policy: FailurePolicy::StopOnP0,
        ..Default::default()
    };
    let executor = Executor::with_config(config);
    let evidence = Evidence::falsified("F-INT-001", test_scenario(), "p0 failure", "output", 0);
    assert!(executor.should_stop_on_failure(&evidence, "test"));
}

#[test]
fn test_should_stop_on_failure_stop_on_p0_without_p0_gate() {
    let config = ExecutionConfig {
        failure_policy: FailurePolicy::StopOnP0,
        ..Default::default()
    };
    let executor = Executor::with_config(config);
    let evidence = Evidence::falsified("G2-BASIC", test_scenario(), "non-p0 failure", "output", 0);
    assert!(!executor.should_stop_on_failure(&evidence, "test"));
}

#[test]
fn test_should_stop_on_failure_fail_fast() {
    let config = ExecutionConfig {
        failure_policy: FailurePolicy::FailFast,
        ..Default::default()
    };
    let executor = Executor::with_config(config);
    let evidence = Evidence::falsified("G2-BASIC", test_scenario(), "test failure", "output", 0);
    assert!(executor.should_stop_on_failure(&evidence, "test"));
}

// ── execute_scenarios ───────────────────────────────────────────────

#[test]
fn test_execute_scenarios_dry_run() {
    let mock_runner = MockCommandRunner::new();
    let config = ExecutionConfig {
        dry_run: true,
        model_path: Some("/test/model.gguf".to_string()),
        ..Default::default()
    };
    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));

    let scenarios = vec![test_scenario(), test_scenario(), test_scenario()];
    let (passed, failed, skipped) = executor.execute_scenarios(scenarios, "test");

    assert_eq!(passed, 0);
    assert_eq!(failed, 0);
    assert_eq!(skipped, 3);
}

#[test]
fn test_execute_scenarios_all_pass() {
    let mock_runner = MockCommandRunner::new().with_inference_response("The answer is 4.");
    let config = ExecutionConfig {
        model_path: Some("/test/model.gguf".to_string()),
        failure_policy: FailurePolicy::CollectAll,
        ..Default::default()
    };
    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));

    let scenarios = vec![test_scenario(), test_scenario()];
    let (passed, failed, skipped) = executor.execute_scenarios(scenarios, "test");

    assert_eq!(passed, 2);
    assert_eq!(failed, 0);
    assert_eq!(skipped, 0);
}

#[test]
fn test_execute_scenarios_stop_on_first_failure() {
    let mock_runner = MockCommandRunner::new().with_inference_failure();
    let config = ExecutionConfig {
        model_path: Some("/test/model.gguf".to_string()),
        failure_policy: FailurePolicy::StopOnFirst,
        ..Default::default()
    };
    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));

    let scenarios = vec![test_scenario(), test_scenario(), test_scenario()];
    let (passed, failed, skipped) = executor.execute_scenarios(scenarios, "test");

    // Should stop after first failure
    assert_eq!(passed, 0);
    assert_eq!(failed, 1);
    assert_eq!(skipped, 0);
}

// ── check_pre_flight ────────────────────────────────────────────────

#[test]
fn test_check_pre_flight_no_integrity_check() {
    let config = ExecutionConfig {
        check_integrity: false,
        warn_implicit_skips: false,
        ..Default::default()
    };
    let executor = Executor::with_config(config);
    let playbook = test_playbook();
    let start = Instant::now();
    let result = executor.check_pre_flight(&playbook, 5, start);
    // No integrity check, no implicit skips, gateways pass => None
    assert!(result.is_none());
}

#[test]
fn test_check_pre_flight_integrity_no_lock_file() {
    let config = ExecutionConfig {
        check_integrity: true,
        lock_file_path: None, // No lock file path
        ..Default::default()
    };
    let executor = Executor::with_config(config);
    let playbook = test_playbook();
    let start = Instant::now();
    let result = executor.check_pre_flight(&playbook, 5, start);
    // check_integrity=true but no lock_file_path => skip integrity, proceed
    assert!(result.is_none());
}

#[test]
fn test_check_pre_flight_integrity_missing_lock_file() {
    let config = ExecutionConfig {
        check_integrity: true,
        lock_file_path: Some("/nonexistent/lock.json".to_string()),
        ..Default::default()
    };
    let executor = Executor::with_config(config);
    let playbook = test_playbook();
    let start = Instant::now();
    let result = executor.check_pre_flight(&playbook, 5, start);
    // Lock file doesn't exist => warning printed but None (continues)
    assert!(result.is_none());
}

// ── run_g0_format_check ─────────────────────────────────────────────

#[test]
fn test_run_g0_format_check_no_model_path() {
    let config = ExecutionConfig {
        model_path: None,
        ..Default::default()
    };
    let mut executor = Executor::with_config(config);
    let playbook = test_playbook();
    let (passed, failed) = executor.run_g0_format_check(&playbook);
    assert_eq!(passed, 0);
    assert_eq!(failed, 0);
}

#[test]
fn test_run_g0_format_check_nonexistent_path() {
    let config = ExecutionConfig {
        model_path: Some("/nonexistent/path/model.gguf".to_string()),
        ..Default::default()
    };
    let mut executor = Executor::with_config(config);
    let playbook = test_playbook();
    let (passed, failed) = executor.run_g0_format_check(&playbook);
    // Path doesn't exist, has gguf extension but doesn't match safetensors checks
    assert_eq!(passed, 0);
    assert_eq!(failed, 0);
}

// ── check_g0_tensor ─────────────────────────────────────────────────

#[test]
fn test_check_g0_tensor_no_model_path() {
    let config = ExecutionConfig {
        model_path: None,
        ..Default::default()
    };
    let mut executor = Executor::with_config(config);
    let playbook = test_playbook();
    let (passed, failed) = executor.check_g0_tensor(&playbook);
    assert_eq!(passed, 0);
    assert_eq!(failed, 0);
}

#[test]
fn test_check_g0_tensor_no_family() {
    let config = ExecutionConfig {
        model_path: Some("/test/model.gguf".to_string()),
        ..Default::default()
    };
    let mut executor = Executor::with_config(config);
    // Default test_playbook has no family set
    let playbook = test_playbook();
    let (passed, failed) = executor.check_g0_tensor(&playbook);
    assert_eq!(passed, 0);
    assert_eq!(failed, 0);
}

#[test]
fn test_check_g0_tensor_no_size_variant() {
    let config = ExecutionConfig {
        model_path: Some("/test/model.gguf".to_string()),
        ..Default::default()
    };
    let mut executor = Executor::with_config(config);
    let yaml = r#"
name: test-with-family
version: "1.0.0"
model:
  hf_repo: "test/model"
  family: "qwen2"
  formats: [gguf]
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
"#;
    let playbook = Playbook::from_yaml(yaml).expect("Failed to parse");
    let (passed, failed) = executor.check_g0_tensor(&playbook);
    // No size_variant => (0, 0)
    assert_eq!(passed, 0);
    assert_eq!(failed, 0);
}

// ── run_extended_tests ──────────────────────────────────────────────

#[test]
fn test_run_extended_tests_all_disabled() {
    let mock_runner = MockCommandRunner::new();
    let config = ExecutionConfig {
        model_path: Some("/test/model.gguf".to_string()),
        run_conversion_tests: false,
        run_golden_rule_test: false,
        run_contract_tests: false,
        run_hf_parity: false,
        run_profile_ci: false,
        run_ollama_parity: false,
        ..Default::default()
    };
    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));
    let playbook = test_playbook();
    let (passed, failed) = executor.run_extended_tests(&playbook);
    assert_eq!(passed, 0);
    assert_eq!(failed, 0);
}

#[test]
fn test_run_extended_tests_no_model_path() {
    let mock_runner = MockCommandRunner::new();
    let config = ExecutionConfig {
        model_path: None,
        run_conversion_tests: true,
        run_golden_rule_test: true,
        run_contract_tests: true,
        run_profile_ci: true,
        run_ollama_parity: true,
        ..Default::default()
    };
    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));
    let playbook = test_playbook();
    let (passed, failed) = executor.run_extended_tests(&playbook);
    // No model path => all model-dependent tests skip silently
    assert_eq!(passed, 0);
    assert_eq!(failed, 0);
}

// ── run_integrity_analysis ──────────────────────────────────────────

#[test]
fn test_run_integrity_analysis_nonexistent_path() {
    let result = Executor::run_integrity_analysis(Path::new("/nonexistent/path"));
    // Non-existent directory should return None (no safetensors dir found)
    assert!(result.is_none());
}

#[test]
fn test_run_integrity_analysis_empty_dir() {
    let tmp = tempfile::tempdir().unwrap();
    let result = Executor::run_integrity_analysis(tmp.path());
    // Empty dir => no safetensors files => None
    assert!(result.is_none());
}

// ── clean_stale_artifacts ───────────────────────────────────────────

#[test]
fn test_clean_stale_artifacts_nonexistent_workspace() {
    // Should not panic
    Executor::clean_stale_artifacts(Path::new("/nonexistent/workspace"));
}

#[test]
fn test_clean_stale_artifacts_empty_workspace() {
    let tmp = tempfile::tempdir().unwrap();
    let workspace = tmp.path();
    // Create empty subdirs
    std::fs::create_dir_all(workspace.join("safetensors")).unwrap();
    std::fs::create_dir_all(workspace.join("apr")).unwrap();
    std::fs::create_dir_all(workspace.join("gguf")).unwrap();
    // Should not panic on empty dirs
    Executor::clean_stale_artifacts(workspace);
}

#[test]
fn test_clean_stale_artifacts_removes_artifacts() {
    let tmp = tempfile::tempdir().unwrap();
    let workspace = tmp.path();
    let apr_dir = workspace.join("apr");
    std::fs::create_dir_all(&apr_dir).unwrap();

    // Create a clean file and an artifact
    std::fs::write(apr_dir.join("model.apr"), b"clean").unwrap();
    std::fs::write(apr_dir.join("model-converted.apr"), b"artifact").unwrap();
    std::fs::write(apr_dir.join("model.idem.apr"), b"artifact").unwrap();
    std::fs::write(apr_dir.join("model.byte_rt.apr"), b"artifact").unwrap();

    Executor::clean_stale_artifacts(workspace);

    // Clean file should remain
    assert!(apr_dir.join("model.apr").exists());
    // Artifacts should be removed
    assert!(!apr_dir.join("model-converted.apr").exists());
    assert!(!apr_dir.join("model.idem.apr").exists());
    assert!(!apr_dir.join("model.byte_rt.apr").exists());
}

// ── setup_source_links (sharded vs single) ─────────────────────────

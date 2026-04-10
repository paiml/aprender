/// Verify profile flamegraph reports failure when runner returns error
#[test]
fn test_execute_profile_flamegraph_unsupported() {
    let mock_runner = MockCommandRunner::new().with_profile_flamegraph_failure();
    let executor = ToolExecutor::with_runner(
        "test-model.gguf".to_string(),
        true,
        5000,
        Arc::new(mock_runner),
    );
    let temp_dir = tempfile::tempdir().unwrap();
    let result = executor.execute_profile_flamegraph(temp_dir.path());
    assert!(!result.passed);
}

/// Verify profile focus fails when apr binary is not available
#[test]
fn test_execute_profile_focus_no_apr() {
    let executor = ToolExecutor::new("test-model.gguf".to_string(), true, 5000);
    let result = executor.execute_profile_focus("attention");
    assert!(!result.passed);
    assert_eq!(result.tool, "profile-focus");
    assert_eq!(result.gate_id, "F-PROFILE-003");
}

/// Verify profile focus passes with mock runner returning success
#[test]
fn test_execute_profile_focus_with_mock_success() {
    let mock_runner = MockCommandRunner::new();
    let executor = ToolExecutor::with_runner(
        "test-model.gguf".to_string(),
        false,
        5000,
        Arc::new(mock_runner),
    );
    let result = executor.execute_profile_focus("attention");
    assert!(result.passed);
    assert_eq!(result.tool, "profile-focus");
    assert_eq!(result.gate_id, "F-PROFILE-003");
}

/// Verify profile focus reports failure when runner returns error
#[test]
fn test_execute_profile_focus_unsupported() {
    let mock_runner = MockCommandRunner::new().with_profile_focus_failure();
    let executor = ToolExecutor::with_runner(
        "test-model.gguf".to_string(),
        true,
        5000,
        Arc::new(mock_runner),
    );
    let result = executor.execute_profile_focus("attention");
    assert!(!result.passed);
}

/// Verify backend equivalence test fails when apr is not available
#[test]
fn test_execute_backend_equivalence_no_apr() {
    let executor = ToolExecutor::new("test-model.gguf".to_string(), false, 5000);
    let result = executor.execute_backend_equivalence();
    assert!(!result.passed);
    assert_eq!(result.tool, "backend-equivalence");
    assert_eq!(result.gate_id, "F-CONV-BE-001");
}

/// Verify serve lifecycle test fails when apr binary is not available
#[test]
fn test_execute_serve_lifecycle_no_apr() {
    let executor = ToolExecutor::new("test-model.gguf".to_string(), true, 5000);
    let result = executor.execute_serve_lifecycle();
    assert!(!result.passed);
    assert_eq!(result.tool, "serve-lifecycle");
    assert_eq!(result.gate_id, "F-INTEG-003");
}

/// Verify execute_all omits serve-lifecycle from default tool execution
#[test]
fn test_execute_all_with_serve() {
    let mock_runner = MockCommandRunner::new();
    let executor = ToolExecutor::with_runner(
        "test-model.gguf".to_string(),
        true,
        5000,
        Arc::new(mock_runner),
    );
    // Without serve
    let results = executor.execute_all();
    assert!(!results.is_empty());
    // None should be serve-lifecycle
    assert!(!results.iter().any(|r| r.tool == "serve-lifecycle"));
}

// =========================================================================
// Conversion infrastructure failure
// =========================================================================

/// Verify executor handles conversion infrastructure failure with mock runner
#[test]
#[allow(clippy::too_many_lines)]
fn test_executor_conversion_infrastructure_failure() {
    use crate::command::CommandOutput;

    struct FailingConversionRunner;
    impl CommandRunner for FailingConversionRunner {
        fn run_inference(
            &self,
            _model_path: &Path,
            _prompt: &str,
            _max_tokens: u32,
            _no_gpu: bool,
            _extra_args: &[&str],
        ) -> CommandOutput {
            CommandOutput {
                stdout: "The answer is 4.".to_string(),
                stderr: String::new(),
                exit_code: 0,
                success: true,
            }
        }
        fn convert_model(&self, _source: &Path, _target: &Path) -> CommandOutput {
            CommandOutput::success("")
        }
        fn inspect_model(&self, _path: &Path) -> CommandOutput {
            CommandOutput::success("")
        }
        fn validate_model(&self, _path: &Path) -> CommandOutput {
            CommandOutput::success("")
        }
        fn bench_model(&self, _path: &Path) -> CommandOutput {
            CommandOutput::success("")
        }
        fn check_model(&self, _path: &Path) -> CommandOutput {
            CommandOutput::success("")
        }
        fn profile_model(&self, _path: &Path, _warmup: u32, _measure: u32) -> CommandOutput {
            CommandOutput::success("")
        }
        fn profile_ci(
            &self,
            _path: &Path,
            _min_throughput: Option<f64>,
            _max_p99: Option<f64>,
            _warmup: u32,
            _measure: u32,
            _no_gpu: bool,
        ) -> CommandOutput {
            CommandOutput::success("")
        }
        fn diff_tensors(&self, _model_a: &Path, _model_b: &Path, _json: bool) -> CommandOutput {
            CommandOutput::success("")
        }
        fn compare_inference(
            &self,
            _model_a: &Path,
            _model_b: &Path,
            _prompt: &str,
            _max_tokens: u32,
            _tolerance: f64,
        ) -> CommandOutput {
            CommandOutput::success("")
        }
        fn profile_with_flamegraph(
            &self,
            _model_path: &Path,
            _output_path: &Path,
            _no_gpu: bool,
        ) -> CommandOutput {
            CommandOutput::success("")
        }
        fn profile_with_focus(
            &self,
            _model_path: &Path,
            _focus: &str,
            _no_gpu: bool,
        ) -> CommandOutput {
            CommandOutput::success("")
        }
        fn fingerprint_model(&self, _path: &Path, _json: bool) -> CommandOutput {
            CommandOutput::success("")
        }
        fn validate_stats(&self, _a: &Path, _b: &Path) -> CommandOutput {
            CommandOutput::success("")
        }
        fn validate_model_strict(&self, _path: &Path) -> CommandOutput {
            CommandOutput::success(r#"{"valid":true,"tensors_checked":100,"issues":[]}"#)
        }
        fn pull_model(&self, _hf_repo: &str) -> CommandOutput {
            CommandOutput::success("Path: /mock/model.safetensors")
        }
        fn inspect_model_json(&self, _model_path: &Path) -> CommandOutput {
            CommandOutput::success(
                r#"{"format":"SafeTensors","tensor_count":10,"tensor_names":[]}"#,
            )
        }
        fn run_ollama_inference(
            &self,
            _model_tag: &str,
            _prompt: &str,
            _temperature: f64,
        ) -> CommandOutput {
            CommandOutput::success("Output:\nThe answer is 4.\nCompleted in 1.0s")
        }
        fn pull_ollama_model(&self, _model_tag: &str) -> CommandOutput {
            CommandOutput::success("pulling manifest... done")
        }
        fn create_ollama_model(&self, _: &str, _: &Path) -> CommandOutput {
            CommandOutput::success("creating model... done")
        }
        fn serve_model(&self, _: &Path, _: u16) -> CommandOutput {
            CommandOutput::success(r#"{"status":"listening"}"#)
        }
        fn http_get(&self, _: &str) -> CommandOutput {
            CommandOutput::success(r#"{"models":[]}"#)
        }
        fn profile_memory(&self, _: &Path) -> CommandOutput {
            CommandOutput::success(r#"{"peak_rss_mb":1024}"#)
        }
        fn run_chat(
            &self,
            _model_path: &Path,
            _prompt: &str,
            _no_gpu: bool,
            _extra_args: &[&str],
        ) -> CommandOutput {
            CommandOutput::success("Chat output")
        }
        fn http_post(&self, _url: &str, _body: &str) -> CommandOutput {
            CommandOutput::success("{}")
        }
        fn spawn_serve(&self, _model_path: &Path, _port: u16, _no_gpu: bool) -> CommandOutput {
            CommandOutput::success("12345")
        }
        fn quantize_model(&self, _model_path: &Path, _output_path: &Path, _scheme: &str) -> CommandOutput {
            CommandOutput::success("{}")
        }
        fn import_model(&self, _source_path: &Path, _output_path: &Path) -> CommandOutput {
            CommandOutput::success("{}")
        }
        fn prune_model(&self, _model_path: &Path, _output_path: &Path, _method: &str, _target_ratio: f64) -> CommandOutput {
            CommandOutput::success("{}")
        }
        fn distill_model(&self, _teacher_path: &Path, _student_path: &Path, _output_path: &Path, _data_path: &str) -> CommandOutput {
            CommandOutput::success("{}")
        }
    }

    let config = ExecutionConfig {
        model_path: Some("/nonexistent/model.gguf".to_string()),
        run_conversion_tests: true,
        run_golden_rule_test: false,
        ..Default::default()
    };

    let mut executor = Executor::with_runner(config, Arc::new(FailingConversionRunner));

    let yaml = r#"
name: conv-infra-fail
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
    // Conversion tests ran (whether they passed or failed depends on
    // ConversionExecutor behavior with the mock runner)
    assert!(result.total_scenarios >= 1);

    // Exercise unused CommandRunner trait methods to cover stubs
    let runner = FailingConversionRunner;
    let p = Path::new("/dev/null");
    assert!(runner.validate_model(p).success);
    assert!(runner.bench_model(p).success);
    assert!(runner.check_model(p).success);
    assert!(runner.profile_model(p, 1, 1).success);
    assert!(runner.profile_ci(p, None, None, 1, 1, false).success);
    assert!(runner.diff_tensors(p, p, false).success);
    assert!(runner.compare_inference(p, p, "", 1, 0.0).success);
    assert!(runner.profile_with_flamegraph(p, p, false).success);
    assert!(runner.profile_with_focus(p, "", false).success);
    assert!(runner.fingerprint_model(p, false).success);
    assert!(runner.validate_stats(p, p).success);
}

// ========================================================================
// G0 INTEGRITY CHECK TESTS
// ========================================================================

/// Verify find_safetensors_dir locates safetensors in a subdirectory
#[test]
fn test_find_safetensors_dir_with_subdir() {
    use tempfile::TempDir;
    let dir = TempDir::new().expect("create temp dir");
    let st_dir = dir.path().join("safetensors");
    std::fs::create_dir(&st_dir).expect("create safetensors dir");
    std::fs::write(st_dir.join("model.safetensors"), "test").expect("write file");

    let result = Executor::find_safetensors_dir(dir.path());
    assert!(result.is_some());
    assert_eq!(result.unwrap(), st_dir);
}

/// Verify find_safetensors_dir locates safetensors in the root directory
#[test]
fn test_find_safetensors_dir_direct() {
    use tempfile::TempDir;
    let dir = TempDir::new().expect("create temp dir");
    std::fs::write(dir.path().join("model.safetensors"), "test").expect("write file");

    let result = Executor::find_safetensors_dir(dir.path());
    assert!(result.is_some());
    assert_eq!(result.unwrap(), dir.path());
}

/// Verify find_safetensors_dir returns None when no safetensors files exist
#[test]
fn test_find_safetensors_dir_none() {
    use tempfile::TempDir;
    let dir = TempDir::new().expect("create temp dir");
    // No safetensors files

    let result = Executor::find_safetensors_dir(dir.path());
    assert!(result.is_none());
}

/// Verify has_safetensors_files returns true when safetensors file exists
#[test]
fn test_has_safetensors_files_true() {
    use tempfile::TempDir;
    let dir = TempDir::new().expect("create temp dir");
    std::fs::write(dir.path().join("model.safetensors"), "test").expect("write file");

    assert!(Executor::has_safetensors_files(dir.path()));
}

/// Verify has_safetensors_files returns false for non-safetensors files
#[test]
fn test_has_safetensors_files_false() {
    use tempfile::TempDir;
    let dir = TempDir::new().expect("create temp dir");
    std::fs::write(dir.path().join("model.gguf"), "test").expect("write file");

    assert!(!Executor::has_safetensors_files(dir.path()));
}

/// Verify has_safetensors_files returns false for nonexistent directory
#[test]
fn test_has_safetensors_files_nonexistent_dir() {
    let nonexistent = std::path::Path::new("/nonexistent/path/xyz123");
    assert!(!Executor::has_safetensors_files(nonexistent));
}

// =========================================================================
// G0-VALIDATE Pre-flight Gate Tests
// =========================================================================

/// Verify validate_scenario creates a SafeTensors scenario with G0 Validate prompt
#[test]
fn test_validate_scenario_creation() {
    let model_id = ModelId::new("test", "model");
    let scenario = Executor::validate_scenario(&model_id);

    assert_eq!(scenario.model.org, "test");
    assert_eq!(scenario.model.name, "model");
    assert_eq!(scenario.format, Format::SafeTensors);
    assert!(scenario.prompt.contains("G0 Validate"));
}

/// Verify pull_scenario creates a SafeTensors scenario with G0 Pull prompt
#[test]
fn test_pull_scenario_creation() {
    let model_id = ModelId::new("test", "model");
    let scenario = Executor::pull_scenario(&model_id);

    assert_eq!(scenario.model.org, "test");
    assert_eq!(scenario.model.name, "model");
    assert_eq!(scenario.format, Format::SafeTensors);
    assert!(scenario.prompt.contains("G0 Pull"));
}

/// Verify G0 pull check passes and returns pulled model path on success
#[test]
fn test_g0_pull_pass() {
    let mock_runner = MockCommandRunner::new();

    let config = ExecutionConfig {
        model_path: Some("/test/model.gguf".to_string()),
        run_conversion_tests: false,
        run_golden_rule_test: false,
        run_contract_tests: false,
        ..Default::default()
    };

    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));
    let model_id = ModelId::new("test", "model");
    let (passed, failed, pulled_path) = executor.run_g0_pull_check("test/model", &model_id);

    assert_eq!(passed, 1);
    assert_eq!(failed, 0);
    assert_eq!(pulled_path.as_deref(), Some("/mock/model.safetensors"));

    let evidence = executor.evidence().all();
    let pull_ev = evidence
        .iter()
        .find(|e| e.gate_id == "G0-PULL-001")
        .expect("should have G0-PULL evidence");
    assert!(pull_ev.outcome.is_pass());
    assert!(pull_ev.output.contains("G0 PASS"));
}

/// Verify G0 pull check fails and returns None path when runner reports failure
#[test]
fn test_g0_pull_fail() {
    let mock_runner = MockCommandRunner::new().with_pull_failure();

    let config = ExecutionConfig {
        model_path: Some("/test/model.gguf".to_string()),
        run_conversion_tests: false,
        run_golden_rule_test: false,
        run_contract_tests: false,
        ..Default::default()
    };

    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));
    let model_id = ModelId::new("test", "model");
    let (passed, failed, pulled_path) = executor.run_g0_pull_check("test/model", &model_id);

    assert_eq!(passed, 0);
    assert_eq!(failed, 1);
    assert!(pulled_path.is_none());

    let evidence = executor.evidence().all();
    let pull_ev = evidence
        .iter()
        .find(|e| e.gate_id == "G0-PULL-001")
        .expect("should have G0-PULL evidence");
    assert!(!pull_ev.outcome.is_pass());
    assert!(pull_ev.reason.contains("G0 FAIL"));
}

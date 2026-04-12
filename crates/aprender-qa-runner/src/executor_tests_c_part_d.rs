#[test]
#[allow(clippy::too_many_lines)]
fn test_executor_subprocess_trace_with_stdout() {
    use crate::command::CommandOutput;

    struct TraceStdoutRunner;
    impl CommandRunner for TraceStdoutRunner {
        fn run_inference(
            &self,
            _model_path: &Path,
            _prompt: &str,
            _max_tokens: u32,
            _no_gpu: bool,
            extra_args: &[&str],
        ) -> CommandOutput {
            if extra_args.contains(&"--trace") {
                // Trace run returns both stderr and stdout
                CommandOutput {
                    stdout: "trace data: layer 0 attention".to_string(),
                    stderr: "TRACE: model loading details".to_string(),
                    exit_code: 0,
                    success: true,
                }
            } else {
                // First run fails
                CommandOutput {
                    stdout: String::new(),
                    stderr: "inference error occurred".to_string(),
                    exit_code: 1,
                    success: false,
                }
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
        model_path: Some("/test/model.gguf".to_string()),
        run_conversion_tests: false,
        run_golden_rule_test: false,
        ..Default::default()
    };

    let mut executor = Executor::with_runner(config, Arc::new(TraceStdoutRunner));

    let yaml = r#"
name: trace-stdout-test
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
    assert!(result.failed >= 1);
    // Check that evidence contains trace data
    let evidence = executor.evidence().all();
    assert!(!evidence.is_empty());
    // stderr should contain trace output
    let last = &evidence[evidence.len() - 1];
    if let Some(ref stderr) = last.stderr {
        assert!(stderr.contains("TRACE STDOUT") || stderr.contains("trace"));
    }
}

// =========================================================================
// Model path resolution fallback
// =========================================================================

#[test]
fn test_resolve_model_path_fallback_to_extension() {
    let temp_dir = tempfile::tempdir().unwrap();
    let gguf_dir = temp_dir.path().join("gguf");
    std::fs::create_dir_all(&gguf_dir).unwrap();

    // Create a file with .gguf extension but NOT named "model.gguf"
    let alt_model = gguf_dir.join("custom-name.gguf");
    std::fs::write(&alt_model, b"fake model").unwrap();

    let config = ExecutionConfig {
        model_path: Some(temp_dir.path().to_string_lossy().to_string()),
        ..Default::default()
    };
    let executor = Executor::with_config(config);

    let scenario = apr_qa_gen::QaScenario::new(
        apr_qa_gen::ModelId::new("test", "model"),
        apr_qa_gen::Modality::Run,
        apr_qa_gen::Backend::Cpu,
        apr_qa_gen::Format::Gguf,
        "test prompt".to_string(),
        0,
    );

    let path = executor.resolve_model_path(&scenario);
    // Should find the custom-name.gguf via extension fallback
    assert!(path.unwrap().contains("custom-name.gguf"));
}

#[test]
fn test_resolve_model_path_prefers_model_dot_ext() {
    let temp_dir = tempfile::tempdir().unwrap();
    let apr_dir = temp_dir.path().join("apr");
    std::fs::create_dir_all(&apr_dir).unwrap();

    // Create the canonical model.apr
    let model_file = apr_dir.join("model.apr");
    std::fs::write(&model_file, b"fake model").unwrap();

    let config = ExecutionConfig {
        model_path: Some(temp_dir.path().to_string_lossy().to_string()),
        ..Default::default()
    };
    let executor = Executor::with_config(config);

    let scenario = apr_qa_gen::QaScenario::new(
        apr_qa_gen::ModelId::new("test", "model"),
        apr_qa_gen::Modality::Run,
        apr_qa_gen::Backend::Cpu,
        apr_qa_gen::Format::Apr,
        "test prompt".to_string(),
        0,
    );

    let path = executor.resolve_model_path(&scenario);
    assert!(path.unwrap().contains("model.apr"));
}

// =========================================================================
// File-mode model path resolution
// =========================================================================

#[test]
fn test_resolve_model_path_file_matching_format() {
    let temp_dir = tempfile::tempdir().unwrap();
    let model_file = temp_dir.path().join("abc123.safetensors");
    std::fs::write(&model_file, b"fake model data").unwrap();

    let config = ExecutionConfig {
        model_path: Some(model_file.to_string_lossy().to_string()),
        ..Default::default()
    };
    let executor = Executor::with_config(config);

    // SafeTensors format should match .safetensors file
    let scenario = QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Cpu,
        Format::SafeTensors,
        "test".to_string(),
        0,
    );
    let path = executor.resolve_model_path(&scenario);
    assert!(path.is_some());
    assert!(path.unwrap().contains("abc123.safetensors"));
}

#[test]
fn test_resolve_model_path_file_nonmatching_format() {
    let temp_dir = tempfile::tempdir().unwrap();
    let model_file = temp_dir.path().join("abc123.safetensors");
    std::fs::write(&model_file, b"fake model data").unwrap();

    let config = ExecutionConfig {
        model_path: Some(model_file.to_string_lossy().to_string()),
        ..Default::default()
    };
    let executor = Executor::with_config(config);

    // GGUF format should NOT match .safetensors file
    let scenario_gguf = QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Cpu,
        Format::Gguf,
        "test".to_string(),
        0,
    );
    assert!(executor.resolve_model_path(&scenario_gguf).is_none());

    // APR format should NOT match .safetensors file
    let scenario_apr = QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Cpu,
        Format::Apr,
        "test".to_string(),
        0,
    );
    assert!(executor.resolve_model_path(&scenario_apr).is_none());
}

#[test]
fn test_resolve_model_path_file_gguf() {
    let temp_dir = tempfile::tempdir().unwrap();
    let model_file = temp_dir.path().join("hash123.gguf");
    std::fs::write(&model_file, b"fake gguf").unwrap();

    let config = ExecutionConfig {
        model_path: Some(model_file.to_string_lossy().to_string()),
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
    assert!(path.is_some());
    assert!(path.unwrap().contains("hash123.gguf"));
}

#[test]
fn test_execute_scenario_skips_nonmatching_format() {
    let temp_dir = tempfile::tempdir().unwrap();
    let model_file = temp_dir.path().join("abc123.safetensors");
    std::fs::write(&model_file, b"fake model").unwrap();

    let mock_runner = MockCommandRunner::new().with_inference_response("The answer is 4.");

    let config = ExecutionConfig {
        model_path: Some(model_file.to_string_lossy().to_string()),
        ..Default::default()
    };
    let executor = Executor::with_runner(config, Arc::new(mock_runner));

    // GGUF scenario against .safetensors file should be skipped
    let scenario = QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Cpu,
        Format::Gguf,
        "2+2=".to_string(),
        42,
    );
    let evidence = executor.execute_scenario(&scenario);
    assert_eq!(evidence.outcome, Outcome::Skipped);
    assert!(evidence.reason.contains("Format"));
}

#[test]
fn test_find_safetensors_dir_file_mode() {
    let temp_dir = tempfile::tempdir().unwrap();

    // File with .safetensors extension → returns parent dir
    let st_file = temp_dir.path().join("model.safetensors");
    std::fs::write(&st_file, b"fake").unwrap();
    let result = Executor::find_safetensors_dir(&st_file);
    assert!(result.is_some());
    assert_eq!(result.unwrap(), temp_dir.path());

    // File with non-safetensors extension → returns None
    let gguf_file = temp_dir.path().join("model.gguf");
    std::fs::write(&gguf_file, b"fake").unwrap();
    let result = Executor::find_safetensors_dir(&gguf_file);
    assert!(result.is_none());
}

#[test]
fn test_subprocess_execution_skip_flag() {
    let temp_dir = tempfile::tempdir().unwrap();
    let model_file = temp_dir.path().join("abc.safetensors");
    std::fs::write(&model_file, b"fake").unwrap();

    let mock_runner = MockCommandRunner::new().with_inference_response("The answer is 4.");

    let config = ExecutionConfig {
        model_path: Some(model_file.to_string_lossy().to_string()),
        ..Default::default()
    };
    let executor = Executor::with_runner(config, Arc::new(mock_runner));

    // Matching format → not skipped
    let scenario_st = QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Cpu,
        Format::SafeTensors,
        "test".to_string(),
        0,
    );
    let (_, _, _, _, skipped) = executor.subprocess_execution(&scenario_st);
    assert!(!skipped);

    // Non-matching format → skipped
    let scenario_gguf = QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Cpu,
        Format::Gguf,
        "test".to_string(),
        0,
    );
    let (_, _, _, _, skipped) = executor.subprocess_execution(&scenario_gguf);
    assert!(skipped);
}

// =========================================================================
// Stderr in oracle corroborated evidence
// =========================================================================

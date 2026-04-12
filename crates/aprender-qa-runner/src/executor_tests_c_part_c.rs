#[test]
#[allow(clippy::too_many_lines)]
fn test_executor_golden_rule_converted_inference_fails() {
    use crate::command::CommandOutput;

    // Build a custom runner that succeeds on original, succeeds on convert,
    // but fails on converted inference
    struct ConvertedFailRunner;
    impl CommandRunner for ConvertedFailRunner {
        fn run_inference(
            &self,
            model_path: &Path,
            _prompt: &str,
            _max_tokens: u32,
            _no_gpu: bool,
            _extra_args: &[&str],
        ) -> CommandOutput {
            // Original model succeeds, converted model (.apr) fails
            if model_path.to_string_lossy().contains(".apr") {
                CommandOutput {
                    stdout: String::new(),
                    stderr: "Failed to load converted model".to_string(),
                    exit_code: 1,
                    success: false,
                }
            } else {
                CommandOutput {
                    stdout: "Output:\nThe answer is 4.\nCompleted in 100ms".to_string(),
                    stderr: String::new(),
                    exit_code: 0,
                    success: true,
                }
            }
        }

        fn convert_model(&self, _source: &Path, _target: &Path) -> CommandOutput {
            CommandOutput {
                stdout: "Conversion complete".to_string(),
                stderr: String::new(),
                exit_code: 0,
                success: true,
            }
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
        run_golden_rule_test: true,
        ..Default::default()
    };

    let mut executor = Executor::with_runner(config, Arc::new(ConvertedFailRunner));

    let yaml = r#"
name: golden-conv-fail
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
    // Golden rule test should produce a failure (converted inference failed)
    assert!(result.failed >= 1);
}

// =========================================================================
// Golden Rule: output differs (F-GOLDEN-RULE-001 FAIL)
// =========================================================================

#[test]
#[allow(clippy::too_many_lines)]
fn test_executor_golden_rule_output_differs_with_data() {
    use crate::command::CommandOutput;

    struct DiffOutputRunner;
    impl CommandRunner for DiffOutputRunner {
        fn run_inference(
            &self,
            model_path: &Path,
            _prompt: &str,
            _max_tokens: u32,
            _no_gpu: bool,
            _extra_args: &[&str],
        ) -> CommandOutput {
            if model_path.to_string_lossy().contains(".apr") {
                CommandOutput {
                    stdout: "Output:\nThe answer is 5.\nCompleted in 100ms".to_string(),
                    stderr: String::new(),
                    exit_code: 0,
                    success: true,
                }
            } else {
                CommandOutput {
                    stdout: "Output:\nThe answer is 4.\nCompleted in 100ms".to_string(),
                    stderr: String::new(),
                    exit_code: 0,
                    success: true,
                }
            }
        }

        fn convert_model(&self, _source: &Path, _target: &Path) -> CommandOutput {
            CommandOutput {
                stdout: "ok".to_string(),
                stderr: String::new(),
                exit_code: 0,
                success: true,
            }
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
        run_golden_rule_test: true,
        ..Default::default()
    };

    let mut executor = Executor::with_runner(config, Arc::new(DiffOutputRunner));

    let yaml = r#"
name: golden-diff
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
    // Output differs => falsified
    assert!(result.failed >= 1);
}

// =========================================================================
// Subprocess execution with trace + stdout
// =========================================================================

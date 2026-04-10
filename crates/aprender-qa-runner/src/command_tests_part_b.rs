#[test]
fn test_mock_runner_check_failure() {
    let runner = MockCommandRunner::new().with_check_failure();
    let path = PathBuf::from("model.gguf");
    let output = runner.check_model(&path);
    assert!(!output.success);
    assert!(output.stderr.contains("safety issues"));
}

#[test]
fn test_mock_runner_profile_failure() {
    let runner = MockCommandRunner::new().with_profile_failure();
    let path = PathBuf::from("model.gguf");
    let output = runner.profile_model(&path, 1, 2);
    assert!(!output.success);
    assert!(output.stderr.contains("insufficient memory"));
}

#[test]
fn test_mock_runner_diff_tensors_failure() {
    let runner = MockCommandRunner::new().with_diff_tensors_failure();
    let a = PathBuf::from("a.gguf");
    let b = PathBuf::from("b.apr");
    let output = runner.diff_tensors(&a, &b, true);
    assert!(!output.success);
    assert!(output.stderr.contains("incompatible models"));
}

#[test]
fn test_mock_runner_compare_inference_failure() {
    let runner = MockCommandRunner::new().with_compare_inference_failure();
    let a = PathBuf::from("a.gguf");
    let b = PathBuf::from("b.apr");
    let output = runner.compare_inference(&a, &b, "test", 10, 1e-5);
    assert!(!output.success);
    assert!(output.stderr.contains("output mismatch"));
}

#[test]
fn test_mock_runner_default_new_fields() {
    let runner = MockCommandRunner::default();
    assert!(!runner.profile_ci_unavailable);
    assert!(runner.profile_ci_stderr.is_none());
    assert!(runner.inspect_success);
    assert!(runner.validate_success);
    assert!(runner.bench_success);
    assert!(runner.check_success);
    assert!(runner.profile_success);
    assert!(runner.diff_tensors_success);
    assert!(runner.compare_inference_success);
}

#[test]
fn test_mock_runner_chained_failures() {
    let runner = MockCommandRunner::new()
        .with_inspect_failure()
        .with_validate_failure()
        .with_bench_failure()
        .with_check_failure()
        .with_profile_failure()
        .with_diff_tensors_failure()
        .with_compare_inference_failure();

    assert!(!runner.inspect_success);
    assert!(!runner.validate_success);
    assert!(!runner.bench_success);
    assert!(!runner.check_success);
    assert!(!runner.profile_success);
    assert!(!runner.diff_tensors_success);
    assert!(!runner.compare_inference_success);
}

// Tests for RealCommandRunner using nonexistent binary to exercise error paths
#[test]
fn test_real_runner_execute_nonexistent_binary() {
    let runner = RealCommandRunner::with_binary("/nonexistent/binary/path");
    let path = PathBuf::from("model.gguf");
    let output = runner.run_inference(&path, "test", 32, false, &[]);
    assert!(!output.success);
    assert_eq!(output.exit_code, -1);
    assert!(output.stderr.contains("Failed to execute"));
}

#[test]
fn test_real_runner_run_inference_with_no_gpu() {
    let runner = RealCommandRunner::with_binary("/nonexistent/binary");
    let path = PathBuf::from("model.gguf");
    let output = runner.run_inference(&path, "test", 32, true, &[]);
    assert!(!output.success);
}

#[test]
fn test_real_runner_run_inference_with_extra_args() {
    let runner = RealCommandRunner::with_binary("/nonexistent/binary");
    let path = PathBuf::from("model.gguf");
    let output = runner.run_inference(&path, "test", 32, false, &["--temp", "0.8"]);
    assert!(!output.success);
}

#[test]
fn test_real_runner_convert_model() {
    let runner = RealCommandRunner::with_binary("/nonexistent/binary");
    let source = PathBuf::from("source.gguf");
    let target = PathBuf::from("target.apr");
    let output = runner.convert_model(&source, &target);
    assert!(!output.success);
}

#[test]
fn test_real_runner_inspect_model() {
    let runner = RealCommandRunner::with_binary("/nonexistent/binary");
    let path = PathBuf::from("model.gguf");
    let output = runner.inspect_model(&path);
    assert!(!output.success);
}

#[test]
fn test_real_runner_validate_model() {
    let runner = RealCommandRunner::with_binary("/nonexistent/binary");
    let path = PathBuf::from("model.gguf");
    let output = runner.validate_model(&path);
    assert!(!output.success);
}

#[test]
fn test_real_runner_bench_model() {
    let runner = RealCommandRunner::with_binary("/nonexistent/binary");
    let path = PathBuf::from("model.gguf");
    let output = runner.bench_model(&path);
    assert!(!output.success);
}

#[test]
fn test_real_runner_check_model() {
    let runner = RealCommandRunner::with_binary("/nonexistent/binary");
    let path = PathBuf::from("model.gguf");
    let output = runner.check_model(&path);
    assert!(!output.success);
}

#[test]
fn test_real_runner_profile_model() {
    let runner = RealCommandRunner::with_binary("/nonexistent/binary");
    let path = PathBuf::from("model.gguf");
    let output = runner.profile_model(&path, 5, 10);
    assert!(!output.success);
}

#[test]
fn test_real_runner_profile_ci_all_options() {
    let runner = RealCommandRunner::with_binary("/nonexistent/binary");
    let path = PathBuf::from("model.gguf");
    let output = runner.profile_ci(&path, Some(10.0), Some(100.0), 5, 10, false);
    assert!(!output.success);
}

#[test]
fn test_real_runner_profile_ci_throughput_only() {
    let runner = RealCommandRunner::with_binary("/nonexistent/binary");
    let path = PathBuf::from("model.gguf");
    let output = runner.profile_ci(&path, Some(50.0), None, 1, 1, false);
    assert!(!output.success);
}

#[test]
fn test_real_runner_profile_ci_p99_only() {
    let runner = RealCommandRunner::with_binary("/nonexistent/binary");
    let path = PathBuf::from("model.gguf");
    let output = runner.profile_ci(&path, None, Some(200.0), 1, 1, false);
    assert!(!output.success);
}

#[test]
fn test_real_runner_profile_ci_no_options() {
    let runner = RealCommandRunner::with_binary("/nonexistent/binary");
    let path = PathBuf::from("model.gguf");
    let output = runner.profile_ci(&path, None, None, 1, 1, false);
    assert!(!output.success);
}

#[test]
fn test_real_runner_diff_tensors_json() {
    let runner = RealCommandRunner::with_binary("/nonexistent/binary");
    let a = PathBuf::from("a.gguf");
    let b = PathBuf::from("b.apr");
    let output = runner.diff_tensors(&a, &b, true);
    assert!(!output.success);
}

#[test]
fn test_real_runner_diff_tensors_text() {
    let runner = RealCommandRunner::with_binary("/nonexistent/binary");
    let a = PathBuf::from("a.gguf");
    let b = PathBuf::from("b.apr");
    let output = runner.diff_tensors(&a, &b, false);
    assert!(!output.success);
}

#[test]
fn test_real_runner_compare_inference() {
    let runner = RealCommandRunner::with_binary("/nonexistent/binary");
    let a = PathBuf::from("a.gguf");
    let b = PathBuf::from("b.apr");
    let output = runner.compare_inference(&a, &b, "prompt", 10, 1e-5);
    assert!(!output.success);
}

#[test]
fn test_mock_runner_profile_flamegraph_success() {
    let runner = MockCommandRunner::new();
    let model = PathBuf::from("model.gguf");
    let output_path = PathBuf::from("/tmp/profile.svg");
    let output = runner.profile_with_flamegraph(&model, &output_path, false);
    assert!(output.success);
    assert!(output.stdout.contains("flamegraph"));
}

#[test]
fn test_mock_runner_profile_flamegraph_failure() {
    let runner = MockCommandRunner::new().with_profile_flamegraph_failure();
    let model = PathBuf::from("model.gguf");
    let output_path = PathBuf::from("/tmp/profile.svg");
    let output = runner.profile_with_flamegraph(&model, &output_path, false);
    assert!(!output.success);
    assert!(output.stderr.contains("profiler error"));
}

#[test]
fn test_mock_runner_profile_focus_success() {
    let runner = MockCommandRunner::new().with_tps(42.0);
    let model = PathBuf::from("model.gguf");
    let output = runner.profile_with_focus(&model, "attention", false);
    assert!(output.success);
    assert!(output.stdout.contains("42.0"));
}

#[test]
fn test_mock_runner_profile_focus_failure() {
    let runner = MockCommandRunner::new().with_profile_focus_failure();
    let model = PathBuf::from("model.gguf");
    let output = runner.profile_with_focus(&model, "attention", false);
    assert!(!output.success);
    assert!(output.stderr.contains("invalid focus target"));
}

#[test]
fn test_real_runner_profile_flamegraph() {
    let runner = RealCommandRunner::with_binary("/nonexistent/binary");
    let model = PathBuf::from("model.gguf");
    let output_path = PathBuf::from("/tmp/profile.svg");
    let output = runner.profile_with_flamegraph(&model, &output_path, false);
    assert!(!output.success);
}

#[test]
fn test_real_runner_profile_flamegraph_no_gpu() {
    let runner = RealCommandRunner::with_binary("/nonexistent/binary");
    let model = PathBuf::from("model.gguf");
    let output_path = PathBuf::from("/tmp/profile.svg");
    let output = runner.profile_with_flamegraph(&model, &output_path, true);
    assert!(!output.success);
}

#[test]
fn test_real_runner_profile_focus() {
    let runner = RealCommandRunner::with_binary("/nonexistent/binary");
    let model = PathBuf::from("model.gguf");
    let output = runner.profile_with_focus(&model, "attention", false);
    assert!(!output.success);
}

#[test]
fn test_real_runner_profile_focus_no_gpu() {
    let runner = RealCommandRunner::with_binary("/nonexistent/binary");
    let model = PathBuf::from("model.gguf");
    let output = runner.profile_with_focus(&model, "matmul", true);
    assert!(!output.success);
}

#[test]
fn test_mock_runner_default_new_profile_fields() {
    let runner = MockCommandRunner::default();
    assert!(runner.profile_flamegraph_success);
    assert!(runner.profile_focus_success);
}

#[test]
fn test_mock_runner_chained_profile_failures() {
    let runner = MockCommandRunner::new()
        .with_profile_flamegraph_failure()
        .with_profile_focus_failure();
    assert!(!runner.profile_flamegraph_success);
    assert!(!runner.profile_focus_success);
}

#[test]
fn test_mock_runner_validate_strict_success() {
    let runner = MockCommandRunner::new();
    let path = PathBuf::from("model.gguf");
    let output = runner.validate_model_strict(&path);
    assert!(output.success);
    assert!(output.stdout.contains("\"valid\":true"));
}

#[test]
fn test_mock_runner_validate_strict_failure() {
    let runner = MockCommandRunner::new().with_validate_strict_failure();
    let path = PathBuf::from("model.gguf");
    let output = runner.validate_model_strict(&path);
    assert!(!output.success);
    assert!(output.stdout.contains("\"valid\":false"));
    assert!(output.stdout.contains("all-zeros"));
}

#[test]
fn test_mock_runner_validate_strict_default() {
    let runner = MockCommandRunner::default();
    assert!(runner.validate_strict_success);
}

#[test]
fn test_real_runner_validate_strict() {
    let runner = RealCommandRunner::with_binary("/nonexistent/binary");
    let path = PathBuf::from("model.gguf");
    let output = runner.validate_model_strict(&path);
    assert!(!output.success);
}

#[test]
fn test_mock_runner_pull_success() {
    let runner = MockCommandRunner::new();
    let output = runner.pull_model("test/model");
    assert!(output.success);
    assert!(output.stdout.contains("Path: /mock/model.safetensors"));
}

#[test]
fn test_mock_runner_pull_failure() {
    let runner = MockCommandRunner::new().with_pull_failure();
    let output = runner.pull_model("test/model");
    assert!(!output.success);
    assert!(output.stderr.contains("Pull failed"));
}

#[test]
fn test_mock_runner_pull_custom_path() {
    let runner = MockCommandRunner::new().with_pull_model_path("/custom/path/model.safetensors");
    let output = runner.pull_model("test/model");
    assert!(output.success);
    assert!(
        output
            .stdout
            .contains("Path: /custom/path/model.safetensors")
    );
}

#[test]
fn test_mock_runner_pull_default() {
    let runner = MockCommandRunner::default();
    assert!(runner.pull_success);
    assert_eq!(runner.pull_model_path, "/mock/model.safetensors");
}

#[test]
fn test_real_runner_pull_model() {
    let runner = RealCommandRunner::with_binary("/nonexistent/binary");
    let output = runner.pull_model("test/model");
    assert!(!output.success);
}

// ── Ollama parity tests (GH-6/AC-2) ────────────────────────────────

#[test]
fn test_mock_runner_ollama_inference_success() {
    let runner = MockCommandRunner::new();
    let output = runner.run_ollama_inference("qwen2.5-coder:7b-q4_k_m", "What is 2+2?", 0.0);
    assert!(output.success);
    assert!(output.stdout.contains("The answer is 4."));
}

#[test]
fn test_mock_runner_ollama_inference_custom_response() {
    let runner = MockCommandRunner::new().with_ollama_response("Custom ollama response");
    let output = runner.run_ollama_inference("qwen2.5-coder:7b", "Hello", 0.7);
    assert!(output.success);
    assert!(output.stdout.contains("Custom ollama response"));
}

#[test]
fn test_mock_runner_ollama_inference_failure() {
    let runner = MockCommandRunner::new().with_ollama_failure();
    let output = runner.run_ollama_inference("qwen2.5-coder:7b", "test", 0.0);
    assert!(!output.success);
    assert!(output.stderr.contains("Ollama inference failed"));
}

#[test]
fn test_mock_runner_ollama_pull_success() {
    let runner = MockCommandRunner::new();
    let output = runner.pull_ollama_model("qwen2.5-coder:7b-q4_k_m");
    assert!(output.success);
    assert!(output.stdout.contains("pulling manifest"));
}

#[test]
fn test_mock_runner_ollama_pull_failure() {
    let runner = MockCommandRunner::new().with_ollama_pull_failure();
    let output = runner.pull_ollama_model("nonexistent:model");
    assert!(!output.success);
    assert!(output.stderr.contains("Ollama pull failed"));
}

#[test]
fn test_mock_runner_ollama_default_fields() {
    let runner = MockCommandRunner::default();
    assert!(runner.ollama_success);
    assert!(runner.ollama_pull_success);
    assert_eq!(runner.ollama_response, "The answer is 4.");
}

// ── New gate methods (F-OLLAMA-003/004/005, F-PERF-003/005) ────────

#[test]
fn test_mock_runner_create_ollama_success() {
    let runner = MockCommandRunner::new();
    let path = PathBuf::from("/tmp/Modelfile");
    let output = runner.create_ollama_model("test:latest", &path);
    assert!(output.success);
    assert!(output.stdout.contains("creating model"));
}

#[test]
fn test_mock_runner_create_ollama_failure() {
    let runner = MockCommandRunner::new().with_ollama_create_failure();
    let path = PathBuf::from("/tmp/Modelfile");
    let output = runner.create_ollama_model("test:latest", &path);
    assert!(!output.success);
}

#[test]
fn test_mock_runner_serve_success() {
    let runner = MockCommandRunner::new();
    let path = PathBuf::from("model.gguf");
    let output = runner.serve_model(&path, 8080);
    assert!(output.success);
    assert!(output.stdout.contains("listening"));
}

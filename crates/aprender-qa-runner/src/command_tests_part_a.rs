#[test]
fn test_command_output_success() {
    let output = CommandOutput::success("hello");
    assert!(output.success);
    assert_eq!(output.exit_code, 0);
    assert_eq!(output.stdout, "hello");
    assert!(output.stderr.is_empty());
}

#[test]
fn test_command_output_failure() {
    let output = CommandOutput::failure(1, "error message");
    assert!(!output.success);
    assert_eq!(output.exit_code, 1);
    assert!(output.stdout.is_empty());
    assert_eq!(output.stderr, "error message");
}

#[test]
fn test_command_output_with_output() {
    let output = CommandOutput::with_output("out", "err", 0);
    assert!(output.success);
    assert_eq!(output.stdout, "out");
    assert_eq!(output.stderr, "err");

    let output2 = CommandOutput::with_output("out", "err", 1);
    assert!(!output2.success);
}

#[test]
fn test_mock_runner_default() {
    let runner = MockCommandRunner::new();
    assert!(runner.inference_success);
    assert!(runner.convert_success);
    assert!((runner.tps - 25.0).abs() < f64::EPSILON);
}

#[test]
fn test_mock_runner_inference_2plus2() {
    let runner = MockCommandRunner::new();
    let path = PathBuf::from("model.gguf");
    let output = runner.run_inference(&path, "What is 2+2?", 32, false, &[]);
    assert!(output.success);
    assert!(output.stdout.contains("4"));
}

#[test]
fn test_mock_runner_inference_code() {
    let runner = MockCommandRunner::new();
    let path = PathBuf::from("model.gguf");
    let output = runner.run_inference(&path, "def fibonacci(n):", 32, false, &[]);
    assert!(output.success);
    assert!(output.stdout.contains("return"));
}

#[test]
fn test_mock_runner_inference_empty() {
    let runner = MockCommandRunner::new();
    let path = PathBuf::from("model.gguf");
    let output = runner.run_inference(&path, "", 32, false, &[]);
    assert!(output.success);
    // Empty prompt produces empty response content
}

#[test]
fn test_mock_runner_inference_generic() {
    let runner = MockCommandRunner::new().with_inference_response("Custom response");
    let path = PathBuf::from("model.gguf");
    let output = runner.run_inference(&path, "Hello world", 32, false, &[]);
    assert!(output.success);
    assert!(output.stdout.contains("Custom response"));
}

#[test]
fn test_mock_runner_inference_failure() {
    let runner = MockCommandRunner::new().with_inference_failure();
    let path = PathBuf::from("model.gguf");
    let output = runner.run_inference(&path, "test", 32, false, &[]);
    assert!(!output.success);
    assert_eq!(output.exit_code, 1);
}

#[test]
fn test_mock_runner_convert_success() {
    let runner = MockCommandRunner::new();
    let source = PathBuf::from("source.gguf");
    let target = PathBuf::from("target.apr");
    let output = runner.convert_model(&source, &target);
    assert!(output.success);
}

#[test]
fn test_mock_runner_convert_failure() {
    let runner = MockCommandRunner::new().with_convert_failure();
    let source = PathBuf::from("source.gguf");
    let target = PathBuf::from("target.apr");
    let output = runner.convert_model(&source, &target);
    assert!(!output.success);
}

#[test]
fn test_mock_runner_inspect() {
    let runner = MockCommandRunner::new();
    let path = PathBuf::from("model.gguf");
    let output = runner.inspect_model(&path);
    assert!(output.success);
    assert!(output.stdout.contains("GGUF"));
}

#[test]
fn test_mock_runner_validate() {
    let runner = MockCommandRunner::new();
    let path = PathBuf::from("model.gguf");
    let output = runner.validate_model(&path);
    assert!(output.success);
}

#[test]
fn test_mock_runner_bench() {
    let runner = MockCommandRunner::new().with_tps(30.0);
    let path = PathBuf::from("model.gguf");
    let output = runner.bench_model(&path);
    assert!(output.success);
    assert!(output.stdout.contains("30.0"));
}

#[test]
fn test_mock_runner_check() {
    let runner = MockCommandRunner::new();
    let path = PathBuf::from("model.gguf");
    let output = runner.check_model(&path);
    assert!(output.success);
}

#[test]
fn test_mock_runner_profile() {
    let runner = MockCommandRunner::new();
    let path = PathBuf::from("model.gguf");
    let output = runner.profile_model(&path, 1, 2);
    assert!(output.success);
    assert!(output.stdout.contains("throughput_tps"));
}

#[test]
fn test_mock_runner_profile_ci_pass() {
    let runner = MockCommandRunner::new().with_tps(20.0);
    let path = PathBuf::from("model.gguf");
    let output = runner.profile_ci(&path, Some(10.0), Some(200.0), 1, 2, false);
    assert!(output.success);
    assert!(output.stdout.contains("\"passed\":true"));
}

#[test]
fn test_mock_runner_profile_ci_fail_throughput() {
    let runner = MockCommandRunner::new().with_tps(5.0);
    let path = PathBuf::from("model.gguf");
    let output = runner.profile_ci(&path, Some(10.0), None, 1, 2, false);
    assert!(!output.success);
    assert!(output.stdout.contains("\"passed\":false"));
}

#[test]
fn test_mock_runner_profile_ci_fail_p99() {
    let runner = MockCommandRunner::new();
    let path = PathBuf::from("model.gguf");
    // p99 is 156.5ms, threshold is 100ms
    let output = runner.profile_ci(&path, None, Some(100.0), 1, 2, false);
    assert!(!output.success);
}

#[test]
fn test_mock_runner_diff_tensors_json() {
    let runner = MockCommandRunner::new();
    let a = PathBuf::from("a.gguf");
    let b = PathBuf::from("b.apr");
    let output = runner.diff_tensors(&a, &b, true);
    assert!(output.success);
    assert!(output.stdout.contains("\"passed\":true"));
}

#[test]
fn test_mock_runner_diff_tensors_text() {
    let runner = MockCommandRunner::new();
    let a = PathBuf::from("a.gguf");
    let b = PathBuf::from("b.apr");
    let output = runner.diff_tensors(&a, &b, false);
    assert!(output.success);
    assert!(output.stdout.contains("match"));
}

#[test]
fn test_mock_runner_compare_inference() {
    let runner = MockCommandRunner::new();
    let a = PathBuf::from("a.gguf");
    let b = PathBuf::from("b.apr");
    let output = runner.compare_inference(&a, &b, "test prompt", 10, 1e-5);
    assert!(output.success);
    assert!(output.stdout.contains("\"passed\":true"));
}

#[test]
fn test_real_runner_new() {
    let runner = RealCommandRunner::new();
    assert_eq!(runner.apr_binary, "apr");
}

#[test]
fn test_real_runner_with_binary() {
    let runner = RealCommandRunner::with_binary("/custom/apr");
    assert_eq!(runner.apr_binary, "/custom/apr");
}

#[test]
fn test_mock_runner_with_tps() {
    let runner = MockCommandRunner::new().with_tps(100.0);
    assert!((runner.tps - 100.0).abs() < f64::EPSILON);
}

#[test]
fn test_mock_runner_chained_config() {
    let runner = MockCommandRunner::new()
        .with_tps(50.0)
        .with_inference_response("Custom")
        .with_convert_failure();

    assert!((runner.tps - 50.0).abs() < f64::EPSILON);
    assert_eq!(runner.inference_response, "Custom");
    assert!(!runner.convert_success);
}

#[test]
fn test_command_output_clone() {
    let output = CommandOutput::success("test");
    let cloned = output.clone();
    assert_eq!(cloned.stdout, output.stdout);
    assert_eq!(cloned.success, output.success);
}

#[test]
fn test_command_output_debug() {
    let output = CommandOutput::success("test");
    let debug_str = format!("{output:?}");
    assert!(debug_str.contains("CommandOutput"));
}

#[test]
fn test_mock_runner_clone() {
    let runner = MockCommandRunner::new().with_tps(42.0);
    let cloned = runner.clone();
    assert!((cloned.tps - 42.0).abs() < f64::EPSILON);
}

#[test]
fn test_mock_runner_debug() {
    let runner = MockCommandRunner::new();
    let debug_str = format!("{runner:?}");
    assert!(debug_str.contains("MockCommandRunner"));
}

#[test]
fn test_real_runner_clone() {
    let runner = RealCommandRunner::with_binary("custom");
    let cloned = runner.clone();
    assert_eq!(cloned.apr_binary, "custom");
}

#[test]
fn test_real_runner_debug() {
    let runner = RealCommandRunner::new();
    let debug_str = format!("{runner:?}");
    assert!(debug_str.contains("RealCommandRunner"));
}

#[test]
fn test_real_runner_default() {
    let runner = RealCommandRunner::default();
    assert_eq!(runner.apr_binary, "apr");
}

#[test]
fn test_mock_runner_with_crash() {
    let runner = MockCommandRunner::new().with_crash();
    assert!(runner.crash);
    let path = PathBuf::from("model.gguf");
    let output = runner.run_inference(&path, "test", 32, false, &[]);
    assert!(!output.success);
    assert_eq!(output.exit_code, -11); // SIGSEGV
    assert!(output.stderr.contains("SIGSEGV"));
}

#[test]
fn test_mock_runner_with_inference_response_and_stderr() {
    let runner = MockCommandRunner::new().with_inference_response_and_stderr("Response", "Warning");
    assert_eq!(runner.inference_response, "Response");
    assert_eq!(runner.inference_stderr.as_deref(), Some("Warning"));

    let path = PathBuf::from("model.gguf");
    let output = runner.run_inference(&path, "Hello", 32, false, &[]);
    assert!(output.success);
    assert!(output.stdout.contains("Response"));
    assert_eq!(output.stderr, "Warning");
}

#[test]
fn test_mock_runner_inference_fn_code() {
    let runner = MockCommandRunner::new();
    let path = PathBuf::from("model.gguf");
    let output = runner.run_inference(&path, "fn main() {}", 32, false, &[]);
    assert!(output.success);
    assert!(output.stdout.contains("return"));
}

#[test]
fn test_mock_runner_inference_2_plus_2_spaced() {
    let runner = MockCommandRunner::new();
    let path = PathBuf::from("model.gguf");
    let output = runner.run_inference(&path, "What is 2 + 2?", 32, false, &[]);
    assert!(output.success);
    assert!(output.stdout.contains("4"));
}

#[test]
fn test_mock_runner_crash_takes_priority() {
    // Crash should take priority over inference failure
    let runner = MockCommandRunner::new()
        .with_crash()
        .with_inference_failure();
    let path = PathBuf::from("model.gguf");
    let output = runner.run_inference(&path, "test", 32, false, &[]);
    // Crash should be returned, not inference failure
    assert_eq!(output.exit_code, -11);
}

#[test]
fn test_command_output_with_output_success_on_zero() {
    let output = CommandOutput::with_output("stdout", "stderr", 0);
    assert!(output.success);
    assert_eq!(output.exit_code, 0);
}

#[test]
fn test_command_output_with_output_failure_on_nonzero() {
    let output = CommandOutput::with_output("", "error", 42);
    assert!(!output.success);
    assert_eq!(output.exit_code, 42);
}

#[test]
fn test_mock_runner_profile_ci_no_assertions() {
    let runner = MockCommandRunner::new().with_tps(15.0);
    let path = PathBuf::from("model.gguf");
    // No throughput or p99 assertions
    let output = runner.profile_ci(&path, None, None, 1, 2, false);
    assert!(output.success);
    assert!(output.stdout.contains("\"passed\":true"));
}

#[test]
fn test_mock_runner_fields_after_default() {
    let runner = MockCommandRunner::default();
    assert!(!runner.crash);
    assert!(runner.inference_stderr.is_none());
}

#[test]
fn test_command_output_failure_negative_exit_code() {
    let output = CommandOutput::failure(-9, "killed");
    assert!(!output.success);
    assert_eq!(output.exit_code, -9);
    assert_eq!(output.stderr, "killed");
}

#[test]
fn test_mock_runner_with_all_options() {
    let runner = MockCommandRunner::new()
        .with_tps(100.0)
        .with_inference_response("Custom response")
        .with_crash();

    assert!((runner.tps - 100.0).abs() < f64::EPSILON);
    assert_eq!(runner.inference_response, "Custom response");
    assert!(runner.crash);
}

#[test]
fn test_mock_runner_profile_ci_both_assertions_pass() {
    let runner = MockCommandRunner::new().with_tps(200.0);
    let path = PathBuf::from("model.gguf");
    // Both assertions should pass
    let output = runner.profile_ci(&path, Some(100.0), Some(500.0), 1, 2, false);
    assert!(output.success);
    assert!(output.stdout.contains("\"passed\":true"));
}

#[test]
fn test_mock_runner_profile_ci_both_assertions_fail() {
    let runner = MockCommandRunner::new().with_tps(5.0);
    let path = PathBuf::from("model.gguf");
    // Throughput too low, p99 too high (156.5 > 100)
    let output = runner.profile_ci(&path, Some(100.0), Some(100.0), 1, 2, false);
    assert!(!output.success);
    assert!(output.stdout.contains("\"passed\":false"));
}

#[test]
fn test_mock_runner_profile_ci_unavailable() {
    let runner = MockCommandRunner::new().with_profile_ci_unavailable();
    let path = PathBuf::from("model.gguf");
    let output = runner.profile_ci(&path, Some(10.0), None, 1, 2, false);
    assert!(!output.success);
    assert!(output.stderr.contains("unexpected argument"));
}

#[test]
fn test_mock_runner_profile_ci_custom_stderr() {
    let runner = MockCommandRunner::new()
        .with_profile_ci_unavailable()
        .with_profile_ci_stderr("Custom error: --ci not supported");
    let path = PathBuf::from("model.gguf");
    let output = runner.profile_ci(&path, None, None, 1, 2, false);
    assert!(!output.success);
    assert!(output.stderr.contains("Custom error"));
}

#[test]
fn test_mock_runner_inspect_failure() {
    let runner = MockCommandRunner::new().with_inspect_failure();
    let path = PathBuf::from("model.gguf");
    let output = runner.inspect_model(&path);
    assert!(!output.success);
    assert!(output.stderr.contains("invalid model format"));
}

#[test]
fn test_mock_runner_validate_failure() {
    let runner = MockCommandRunner::new().with_validate_failure();
    let path = PathBuf::from("model.gguf");
    let output = runner.validate_model(&path);
    assert!(!output.success);
    assert!(output.stderr.contains("corrupted tensors"));
}

#[test]
fn test_mock_runner_bench_failure() {
    let runner = MockCommandRunner::new().with_bench_failure();
    let path = PathBuf::from("model.gguf");
    let output = runner.bench_model(&path);
    assert!(!output.success);
    assert!(output.stderr.contains("model load error"));
}

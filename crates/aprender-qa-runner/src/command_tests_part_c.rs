#[test]
fn test_mock_runner_serve_failure() {
    let runner = MockCommandRunner::new().with_serve_failure();
    let path = PathBuf::from("model.gguf");
    let output = runner.serve_model(&path, 8080);
    assert!(!output.success);
}

#[test]
fn test_mock_runner_http_get_success() {
    let runner = MockCommandRunner::new();
    let output = runner.http_get("http://localhost:8080/v1/models");
    assert!(output.success);
    assert!(output.stdout.contains("models"));
}

#[test]
fn test_mock_runner_http_get_failure() {
    let runner = MockCommandRunner::new().with_http_get_failure();
    let output = runner.http_get("http://localhost:8080/v1/models");
    assert!(!output.success);
}

#[test]
fn test_mock_runner_http_get_custom_response() {
    let runner = MockCommandRunner::new().with_http_get_response(r#"{"status":"ok"}"#);
    let output = runner.http_get("http://localhost:8080/health");
    assert!(output.success);
    assert!(output.stdout.contains("ok"));
}

#[test]
fn test_mock_runner_profile_memory_success() {
    let runner = MockCommandRunner::new();
    let path = PathBuf::from("model.gguf");
    let output = runner.profile_memory(&path);
    assert!(output.success);
    assert!(output.stdout.contains("peak_rss_mb"));
}

#[test]
fn test_mock_runner_profile_memory_failure() {
    let runner = MockCommandRunner::new().with_profile_memory_failure();
    let path = PathBuf::from("model.gguf");
    let output = runner.profile_memory(&path);
    assert!(!output.success);
}

#[test]
fn test_mock_runner_new_default_fields() {
    let runner = MockCommandRunner::default();
    assert!(runner.ollama_create_success);
    assert!(runner.serve_success);
    assert!(runner.http_get_success);
    assert!(runner.profile_memory_success);
}

#[test]
fn test_real_runner_create_ollama_model() {
    // create_ollama_model calls `ollama` binary directly (not apr).
    // With a nonexistent modelfile, it should fail regardless.
    let runner = RealCommandRunner::new();
    let path = PathBuf::from("/nonexistent/path/Modelfile");
    let output = runner.create_ollama_model("apr-test-nonexistent:latest", &path);
    // Either ollama isn't installed (failure) or modelfile is missing (failure)
    // This tests the execution path, not the success case
    assert!(output.exit_code != 0 || !output.success || output.stderr.contains("Error"));
}

#[test]
fn test_real_runner_serve_model() {
    let runner = RealCommandRunner::with_binary("/nonexistent/binary");
    let path = PathBuf::from("model.gguf");
    let output = runner.serve_model(&path, 8080);
    assert!(!output.success);
}

#[test]
fn test_real_runner_profile_memory() {
    let runner = RealCommandRunner::with_binary("/nonexistent/binary");
    let path = PathBuf::from("model.gguf");
    let output = runner.profile_memory(&path);
    assert!(!output.success);
}

// ── Bug 200: Chat, HTTP POST, Spawn Serve tests ─────────────────────

#[test]
fn test_mock_runner_chat_success_2plus2() {
    let runner = MockCommandRunner::new();
    let path = PathBuf::from("model.gguf");
    let output = runner.run_chat(&path, "What is 2+2?", false, &[]);
    assert!(output.success);
    assert!(output.stdout.contains("4"));
}

#[test]
fn test_mock_runner_chat_success_generic() {
    let runner = MockCommandRunner::new().with_chat_response("Custom chat response");
    let path = PathBuf::from("model.gguf");
    let output = runner.run_chat(&path, "Hello", false, &[]);
    assert!(output.success);
    assert!(output.stdout.contains("Custom chat response"));
}

#[test]
fn test_mock_runner_chat_failure() {
    let runner = MockCommandRunner::new().with_chat_failure();
    let path = PathBuf::from("model.gguf");
    let output = runner.run_chat(&path, "test", false, &[]);
    assert!(!output.success);
    assert!(output.stderr.contains("Chat failed"));
}

#[test]
fn test_mock_runner_http_post_success() {
    let runner = MockCommandRunner::new();
    let output = runner.http_post("http://localhost:8080/v1/completions", "{}");
    assert!(output.success);
    assert!(output.stdout.contains("choices"));
}

#[test]
fn test_mock_runner_http_post_failure() {
    let runner = MockCommandRunner::new().with_http_post_failure();
    let output = runner.http_post("http://localhost:8080/v1/completions", "{}");
    assert!(!output.success);
}

#[test]
fn test_mock_runner_http_post_custom_response() {
    let runner = MockCommandRunner::new().with_http_post_response(r#"{"text":"custom output"}"#);
    let output = runner.http_post("http://localhost:8080/v1/completions", "{}");
    assert!(output.success);
    assert!(output.stdout.contains("custom output"));
}

#[test]
fn test_mock_runner_spawn_serve_success() {
    let runner = MockCommandRunner::new();
    let path = PathBuf::from("model.gguf");
    let output = runner.spawn_serve(&path, 8080, false);
    assert!(output.success);
    assert!(output.stdout.contains("12345")); // Mock PID
}

#[test]
fn test_mock_runner_spawn_serve_failure() {
    let runner = MockCommandRunner::new().with_spawn_serve_failure();
    let path = PathBuf::from("model.gguf");
    let output = runner.spawn_serve(&path, 8080, false);
    assert!(!output.success);
}

#[test]
fn test_mock_runner_default_new_chat_fields() {
    let runner = MockCommandRunner::default();
    assert!(runner.chat_success);
    assert!(runner.http_post_success);
    assert!(runner.spawn_serve_success);
}

#[test]
fn test_real_runner_chat_nonexistent() {
    let runner = RealCommandRunner::with_binary("/nonexistent/binary");
    let path = PathBuf::from("model.gguf");
    let output = runner.run_chat(&path, "test", false, &[]);
    assert!(!output.success);
}

#[test]
fn test_real_runner_spawn_serve_nonexistent() {
    let runner = RealCommandRunner::with_binary("/nonexistent/binary");
    let path = PathBuf::from("model.gguf");
    let output = runner.spawn_serve(&path, 8080, false);
    assert!(!output.success);
}

// ── Additional coverage tests ──────────────────────────────────────

#[test]
fn test_mock_runner_fingerprint_json_success() {
    let runner = MockCommandRunner::new();
    let path = PathBuf::from("model.gguf");
    let output = runner.fingerprint_model(&path, true);
    assert!(output.success);
    assert!(output.stdout.contains("q_proj.weight"));
    assert!(output.stdout.contains("mean"));
}

#[test]
fn test_mock_runner_fingerprint_text_success() {
    let runner = MockCommandRunner::new();
    let path = PathBuf::from("model.gguf");
    let output = runner.fingerprint_model(&path, false);
    assert!(output.success);
    assert!(output.stdout.contains("Fingerprint"));
    assert!(output.stdout.contains("100 tensors"));
}

#[test]
fn test_mock_runner_fingerprint_failure() {
    let runner = MockCommandRunner::new().with_fingerprint_failure();
    let path = PathBuf::from("model.gguf");
    let output = runner.fingerprint_model(&path, true);
    assert!(!output.success);
    assert!(output.stderr.contains("model load error"));
}

#[test]
fn test_mock_runner_validate_stats_success() {
    let runner = MockCommandRunner::new();
    let a = PathBuf::from("fp_a.json");
    let b = PathBuf::from("fp_b.json");
    let output = runner.validate_stats(&a, &b);
    assert!(output.success);
    assert!(output.stdout.contains("\"passed\":true"));
    assert!(output.stdout.contains("\"failed_tensors\":0"));
}

#[test]
fn test_mock_runner_validate_stats_failure() {
    let runner = MockCommandRunner::new().with_validate_stats_failure();
    let a = PathBuf::from("fp_a.json");
    let b = PathBuf::from("fp_b.json");
    let output = runner.validate_stats(&a, &b);
    assert!(!output.success);
    assert!(output.stderr.contains("3 tensors exceed tolerance"));
}

#[test]
fn test_mock_runner_inspect_json_success() {
    let runner = MockCommandRunner::new();
    let path = PathBuf::from("model.safetensors");
    let output = runner.inspect_model_json(&path);
    assert!(output.success);
    assert!(output.stdout.contains("SafeTensors"));
    assert!(output.stdout.contains("tensor_count"));
    assert!(output.stdout.contains("model.embed_tokens.weight"));
    assert!(output.stdout.contains("lm_head.weight"));
}

#[test]
fn test_mock_runner_inspect_json_failure() {
    let runner = MockCommandRunner::new().with_inspect_json_failure();
    let path = PathBuf::from("model.safetensors");
    let output = runner.inspect_model_json(&path);
    assert!(!output.success);
    assert!(output.stderr.contains("invalid model format"));
}

#[test]
fn test_mock_runner_inspect_json_custom_tensor_names() {
    let runner = MockCommandRunner::new().with_tensor_names(vec![
        "layer.0.weight".to_string(),
        "layer.1.weight".to_string(),
    ]);
    let path = PathBuf::from("model.safetensors");
    let output = runner.inspect_model_json(&path);
    assert!(output.success);
    assert!(output.stdout.contains("\"tensor_count\":2"));
    assert!(output.stdout.contains("layer.0.weight"));
    assert!(output.stdout.contains("layer.1.weight"));
}

#[test]
fn test_mock_runner_inspect_json_empty_tensor_names() {
    let runner = MockCommandRunner::new().with_tensor_names(vec![]);
    let path = PathBuf::from("model.safetensors");
    let output = runner.inspect_model_json(&path);
    assert!(output.success);
    assert!(output.stdout.contains("\"tensor_count\":0"));
    assert!(output.stdout.contains("\"tensor_names\":[]"));
}

#[test]
fn test_mock_runner_custom_exit_code() {
    let runner = MockCommandRunner::new().with_exit_code(137);
    let path = PathBuf::from("model.gguf");
    let output = runner.run_inference(&path, "test", 32, false, &[]);
    assert!(!output.success);
    assert_eq!(output.exit_code, 137);
    assert!(output.stderr.contains("Custom exit code error"));
}

#[test]
fn test_mock_runner_custom_exit_code_zero() {
    let runner = MockCommandRunner::new().with_exit_code(0);
    let path = PathBuf::from("model.gguf");
    let output = runner.run_inference(&path, "test", 32, false, &[]);
    assert!(output.success);
    assert_eq!(output.exit_code, 0);
}

#[test]
fn test_mock_runner_custom_exit_code_priority_over_crash() {
    // Custom exit code takes precedence over crash
    let runner = MockCommandRunner::new().with_exit_code(42).with_crash();
    let path = PathBuf::from("model.gguf");
    let output = runner.run_inference(&path, "test", 32, false, &[]);
    assert_eq!(output.exit_code, 42);
}

#[test]
fn test_mock_runner_chat_2_plus_2_spaced() {
    let runner = MockCommandRunner::new();
    let path = PathBuf::from("model.gguf");
    let output = runner.run_chat(&path, "What is 2 + 2?", false, &[]);
    assert!(output.success);
    assert!(output.stdout.contains("4"));
}

#[test]
fn test_real_runner_fingerprint_model() {
    let runner = RealCommandRunner::with_binary("/nonexistent/binary");
    let path = PathBuf::from("model.gguf");
    let output = runner.fingerprint_model(&path, true);
    assert!(!output.success);
    assert!(output.stderr.contains("Failed to execute"));
}

#[test]
fn test_real_runner_fingerprint_model_text() {
    let runner = RealCommandRunner::with_binary("/nonexistent/binary");
    let path = PathBuf::from("model.gguf");
    let output = runner.fingerprint_model(&path, false);
    assert!(!output.success);
}

#[test]
fn test_real_runner_validate_stats() {
    let runner = RealCommandRunner::with_binary("/nonexistent/binary");
    let a = PathBuf::from("fp_a.json");
    let b = PathBuf::from("fp_b.json");
    let output = runner.validate_stats(&a, &b);
    assert!(!output.success);
    assert!(output.stderr.contains("Failed to execute"));
}

#[test]
fn test_real_runner_inspect_model_json() {
    let runner = RealCommandRunner::with_binary("/nonexistent/binary");
    let path = PathBuf::from("model.safetensors");
    let output = runner.inspect_model_json(&path);
    assert!(!output.success);
    assert!(output.stderr.contains("Failed to execute"));
}

#[test]
fn test_real_runner_spawn_serve_no_gpu() {
    let runner = RealCommandRunner::with_binary("/nonexistent/binary");
    let path = PathBuf::from("model.gguf");
    let output = runner.spawn_serve(&path, 9090, true);
    assert!(!output.success);
    assert!(output.stderr.contains("Failed to spawn serve"));
}

#[test]
fn test_real_runner_chat_no_gpu_with_extra_args() {
    let runner = RealCommandRunner::with_binary("/nonexistent/binary");
    let path = PathBuf::from("model.gguf");
    let output = runner.run_chat(&path, "test", true, &["--temp", "0.5"]);
    assert!(!output.success);
    assert!(output.stderr.contains("Failed to execute chat"));
}

#[test]
fn test_mock_runner_fingerprint_default() {
    let runner = MockCommandRunner::default();
    assert!(runner.fingerprint_success);
    assert!(runner.validate_stats_success);
    assert!(runner.inspect_json_success);
    assert_eq!(runner.inspect_tensor_names.len(), 10);
}

#[test]
fn test_mock_runner_chained_new_failures() {
    let runner = MockCommandRunner::new()
        .with_fingerprint_failure()
        .with_validate_stats_failure()
        .with_inspect_json_failure()
        .with_pull_failure()
        .with_ollama_failure()
        .with_ollama_pull_failure()
        .with_ollama_create_failure()
        .with_serve_failure()
        .with_http_get_failure()
        .with_profile_memory_failure()
        .with_chat_failure()
        .with_http_post_failure()
        .with_spawn_serve_failure();

    assert!(!runner.fingerprint_success);
    assert!(!runner.validate_stats_success);
    assert!(!runner.inspect_json_success);
    assert!(!runner.pull_success);
    assert!(!runner.ollama_success);
    assert!(!runner.ollama_pull_success);
    assert!(!runner.ollama_create_success);
    assert!(!runner.serve_success);
    assert!(!runner.http_get_success);
    assert!(!runner.profile_memory_success);
    assert!(!runner.chat_success);
    assert!(!runner.http_post_success);
    assert!(!runner.spawn_serve_success);
}

#[test]
fn test_mock_runner_chat_tps_in_output() {
    let runner = MockCommandRunner::new().with_tps(88.5);
    let path = PathBuf::from("model.gguf");
    let output = runner.run_chat(&path, "Hello world", false, &[]);
    assert!(output.success);
    assert!(output.stdout.contains("88.5"));
    assert!(output.stdout.contains("tok/s"));
}

#[test]
fn test_command_output_failure_with_empty_stderr() {
    let output = CommandOutput::failure(2, "");
    assert!(!output.success);
    assert_eq!(output.exit_code, 2);
    assert!(output.stdout.is_empty());
    assert!(output.stderr.is_empty());
}

#[test]
fn test_command_output_success_with_empty_stdout() {
    let output = CommandOutput::success("");
    assert!(output.success);
    assert_eq!(output.exit_code, 0);
    assert!(output.stdout.is_empty());
}

#[test]
fn test_command_output_with_output_negative_exit_code() {
    let output = CommandOutput::with_output("out", "err", -1);
    assert!(!output.success);
    assert_eq!(output.exit_code, -1);
    assert_eq!(output.stdout, "out");
    assert_eq!(output.stderr, "err");
}

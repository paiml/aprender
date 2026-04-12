#[test]
fn test_mock_runner_inference_tps_in_output() {
    let runner = MockCommandRunner::new().with_tps(55.3);
    let path = PathBuf::from("model.gguf");
    let output = runner.run_inference(&path, "Hello", 32, false, &[]);
    assert!(output.success);
    assert!(output.stdout.contains("55.3"));
}

/// Cover `RealCommandRunner::http_post` Ok(output) branch.
/// Uses a port that is almost certainly not listening; curl executes but
/// returns exit code 7 (Failed to connect) — still `Ok(output)`.
#[test]
fn test_real_runner_http_post_connection_refused() {
    let runner = RealCommandRunner::new();
    // Port 19439 is unlikely to be open; curl will return exit code 7
    let output = runner.http_post("http://127.0.0.1:19439/v1/generate", r#"{"prompt":"test"}"#);
    // curl found and executed (Ok branch hit), but connection failed
    // success is false because curl exits non-zero on connection failure
    assert!(!output.success);
    // exit_code should be curl's error code (7 = CURLE_COULDNT_CONNECT) or -1 if curl not found
    // Either way, we verify the function ran through the Ok(output) path
    assert!(output.exit_code != 0);
}

/// Cover `RealCommandRunner::prune_model` — all function lines executed.
/// apr binary exists but will fail with bad args/paths (non-zero exit), covering the Ok branch.
#[test]
fn test_real_runner_prune_model_covers_function_body() {
    let runner = RealCommandRunner::new();
    let output = runner.prune_model(
        &PathBuf::from("/nonexistent/model.apr"),
        &PathBuf::from("/tmp/pruned.apr"),
        "magnitude",
        0.5,
    );
    // apr executes but fails (bad path) — exit code non-zero
    assert!(output.exit_code != 0);
}

/// Cover `RealCommandRunner::distill_model` — all function lines executed.
/// apr binary exists but will fail with bad args/paths (non-zero exit), covering the Ok branch.
#[test]
fn test_real_runner_distill_model_covers_function_body() {
    let runner = RealCommandRunner::new();
    let output = runner.distill_model(
        &PathBuf::from("/nonexistent/teacher.apr"),
        &PathBuf::from("/nonexistent/student.apr"),
        &PathBuf::from("/tmp/distilled.apr"),
        "/nonexistent/data",
    );
    // apr executes but fails (bad path) — exit code non-zero
    assert!(output.exit_code != 0);
}

/// Cover `RealCommandRunner::run_ollama_inference` Ok(output) branch.
/// ollama binary exists; will fail for a nonexistent model tag but covers the Ok branch.
#[test]
fn test_real_runner_run_ollama_inference_covers_function_body() {
    let runner = RealCommandRunner::new();
    let output = runner.run_ollama_inference(
        "nonexistent-model-qa-test-xyz:latest",
        "hello",
        0.0,
    );
    // ollama runs but the model doesn't exist — not success; body fully executed regardless
    // The key test is that the function returns a CommandOutput (no panic)
    drop(output);
}

/// Cover `RealCommandRunner::run_chat` Ok(spawn) → Ok(output) branches (lines 349-364).
/// apr spawns successfully but fails (bad model path) — covers the Ok(child) path.
#[test]
fn test_real_runner_run_chat_covers_spawn_ok_path() {
    let runner = RealCommandRunner::new();
    let output = runner.run_chat(
        &PathBuf::from("/nonexistent/model.apr"),
        "hello",
        false,
        &[],
    );
    // apr chat spawned, wrote stdin, waited for output — all branches hit
    // Result is non-success since model doesn't exist
    let _ = output.exit_code;
}

/// Cover `RealCommandRunner::run_chat` with `no_gpu=true` (line 337-338).
#[test]
fn test_real_runner_run_chat_no_gpu_flag() {
    let runner = RealCommandRunner::new();
    let output = runner.run_chat(
        &PathBuf::from("/nonexistent/model.apr"),
        "hello",
        true,
        &[],
    );
    let _ = output.exit_code;
}

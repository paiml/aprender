
use super::*;
use crate::command::MockCommandRunner;
use aprender_qa_gen::{Backend, Format, Modality, ModelId, QaScenario};

fn serve_scenario() -> QaScenario {
    QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Serve,
        Backend::Cpu,
        Format::Gguf,
        "2+2=".to_string(),
        42,
    )
}

/// OpenAI ChatResponse-compatible JSON that also has `text` for /generate extraction.
/// This satisfies both `extract_generated_text` and `ChatResponse` deserialization.
const MOCK_CHAT_RESPONSE: &str = r#"{"id":"test-123","object":"chat.completion","created":0,"model":"test","choices":[{"index":0,"text":"The answer is 4.","message":{"role":"assistant","content":"The answer is 4."},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":5,"total_tokens":10}}"#;

/// GET response that satisfies health ("healthy"), /v1/models ("data" array),
/// and other GET endpoints (non-empty).
const MOCK_GET_RESPONSE: &str = r#"{"status":"healthy","data":[{"id":"test","object":"model"}]}"#;

/// Helper: create a mock runner that passes health checks and returns valid responses.
///
/// Returns OpenAI ChatResponse-compatible JSON for POST and a combined
/// healthy + models response for GET.
fn mock_with_healthy_server() -> MockCommandRunner {
    MockCommandRunner::new()
        .with_http_get_response(MOCK_GET_RESPONSE)
        .with_http_post_response(MOCK_CHAT_RESPONSE)
}

#[test]
fn test_serve_battery_all_endpoints_pass() {
    let mock_runner = mock_with_healthy_server();
    let config = ExecutionConfig {
        model_path: Some("/test/model.gguf".to_string()),
        failure_policy: FailurePolicy::CollectAll,
        ..Default::default()
    };
    let executor = Executor::with_runner(config, Arc::new(mock_runner));
    let scenario = serve_scenario();
    let results = executor.run_serve_battery("/test/model.gguf", &scenario, true);

    // Should produce 19 evidence items (one per check)
    assert_eq!(results.len(), 19, "Expected 19 battery checks, got {}", results.len());

    // Check all 19 gate IDs are present
    let gate_ids: Vec<&str> = results.iter().map(|e| e.gate_id.as_str()).collect();
    // Checks 1-10
    assert!(gate_ids.contains(&"F-A5-001"), "Missing primary generate gate");
    assert!(gate_ids.contains(&"F-A5-COMP-001"), "Missing v1/completions gate");
    assert!(gate_ids.contains(&"F-A5-CHAT-001"), "Missing v1/chat gate");
    assert!(gate_ids.contains(&"F-A5-STREAM-001"), "Missing streaming gate");
    assert!(gate_ids.contains(&"F-A5-STOP-001"), "Missing stop sequence gate");
    assert!(gate_ids.contains(&"F-A5-ERR-001"), "Missing error resilience gate");
    assert!(gate_ids.contains(&"F-A5-INFO-001"), "Missing server info gate");
    assert!(gate_ids.contains(&"F-A5-METRICS-001"), "Missing metrics gate");
    assert!(gate_ids.contains(&"F-A5-EOS-001"), "Missing EOS termination gate");
    assert!(gate_ids.contains(&"F-A5-PERF-001"), "Missing perf floor gate");
    // Checks 11-19
    assert!(gate_ids.contains(&"F-A5-MODELS-001"), "Missing v1/models gate");
    assert!(gate_ids.contains(&"F-A5-TMPL-001"), "Missing template leakage gate");
    assert!(gate_ids.contains(&"F-A5-DETERM-001"), "Missing temp determinism gate");
    assert!(gate_ids.contains(&"F-A5-MULTI-001"), "Missing multi-turn gate");
    assert!(gate_ids.contains(&"F-A5-TOK-001"), "Missing tokenize gate");
    assert!(gate_ids.contains(&"F-A5-CHARS-001"), "Missing special chars gate");
    assert!(gate_ids.contains(&"F-A5-CSTREAM-001"), "Missing chat streaming gate");
    assert!(gate_ids.contains(&"F-A5-MAXTOK-001"), "Missing max_tokens gate");
    assert!(gate_ids.contains(&"F-A5-SCHEMA-001"), "Missing response schema gate");

    // Primary check should pass (mock returns valid response)
    assert!(
        results[0].outcome.is_pass(),
        "Primary generate check should pass"
    );
}

#[test]
fn test_serve_battery_primary_fail_skips_rest() {
    // Health check passes but HTTP POST fails
    let mock_runner = MockCommandRunner::new()
        .with_http_get_response(r#"{"status":"healthy"}"#)
        .with_http_post_failure();
    let config = ExecutionConfig {
        model_path: Some("/test/model.gguf".to_string()),
        failure_policy: FailurePolicy::CollectAll,
        ..Default::default()
    };
    let executor = Executor::with_runner(config, Arc::new(mock_runner));
    let scenario = serve_scenario();
    let results = executor.run_serve_battery("/test/model.gguf", &scenario, true);

    // Primary generate failed → only 1 evidence (rest skipped)
    assert_eq!(results.len(), 1, "Should only have primary evidence when generate fails");
    assert!(results[0].outcome.is_fail(), "Primary check should fail");
    assert_eq!(results[0].gate_id, "F-A5-001");
}

#[test]
fn test_serve_battery_chat_format() {
    // Use the default healthy server mock — all http_post responses return
    // "The answer is 4." which passes the arithmetic oracle for the primary check,
    // allowing the battery to reach the chat endpoint check.
    let mock_runner = mock_with_healthy_server();
    let config = ExecutionConfig {
        model_path: Some("/test/model.gguf".to_string()),
        ..Default::default()
    };
    let executor = Executor::with_runner(config, Arc::new(mock_runner));
    let scenario = serve_scenario();
    let results = executor.run_serve_battery("/test/model.gguf", &scenario, true);

    // Find the chat check
    let chat = results.iter().find(|e| e.gate_id == "F-A5-CHAT-001");
    assert!(chat.is_some(), "Chat evidence should exist");
    assert!(
        chat.unwrap().outcome.is_pass(),
        "Chat check should pass with valid response"
    );
}

#[test]
fn test_serve_battery_sse_valid() {
    let valid_sse = "data: {\"text\":\"hello\"}\n\ndata: {\"text\":\" world\"}\n\ndata: [DONE]\n";
    assert!(Executor::verify_sse_response(valid_sse));
}

#[test]
fn test_serve_battery_sse_invalid_no_done() {
    let invalid = "data: {\"text\":\"hello\"}\n\ndata: {\"text\":\" world\"}\n";
    assert!(!Executor::verify_sse_response(invalid));
}

#[test]
fn test_serve_battery_sse_invalid_no_prefix() {
    let invalid = "{\"text\":\"hello\"}\n{\"text\":\" world\"}\ndata: [DONE]\n";
    assert!(!Executor::verify_sse_response(invalid));
}

#[test]
fn test_serve_battery_sse_empty() {
    assert!(!Executor::verify_sse_response(""));
    assert!(!Executor::verify_sse_response("\n\n\n"));
}

#[test]
fn test_serve_battery_malformed_accepted_is_failure() {
    // mock_with_healthy_server has http_post always succeed, meaning the mock server
    // "accepts" the malformed request. Under the new falsification logic, a server
    // that accepts malformed input FAILS the ERR check — vacuous survival is not enough.
    let mock_runner = mock_with_healthy_server();
    let config = ExecutionConfig {
        model_path: Some("/test/model.gguf".to_string()),
        ..Default::default()
    };
    let executor = Executor::with_runner(config, Arc::new(mock_runner));
    let scenario = serve_scenario();
    let results = executor.run_serve_battery("/test/model.gguf", &scenario, true);

    let malformed = results.iter().find(|e| e.gate_id == "F-A5-ERR-001");
    assert!(malformed.is_some(), "Malformed check evidence should exist");
    // Mock accepts malformed input (http_post succeeds) → fails the check
    assert!(
        malformed.unwrap().outcome.is_fail(),
        "Server that accepts malformed input should fail ERR check"
    );
}

#[test]
fn test_serve_battery_spawn_failure() {
    let mock_runner = MockCommandRunner::new()
        .with_spawn_serve_failure();
    let config = ExecutionConfig {
        model_path: Some("/test/model.gguf".to_string()),
        ..Default::default()
    };
    let executor = Executor::with_runner(config, Arc::new(mock_runner));
    let scenario = serve_scenario();
    let results = executor.run_serve_battery("/test/model.gguf", &scenario, true);

    // Spawn failed → 1 failure evidence
    assert_eq!(results.len(), 1);
    assert!(results[0].outcome.is_fail());
    assert!(results[0].reason.contains("Failed to spawn serve"));
}

#[test]
fn test_serve_battery_gpu_backend_gate_ids() {
    let mock_runner = mock_with_healthy_server();
    let config = ExecutionConfig {
        model_path: Some("/test/model.gguf".to_string()),
        failure_policy: FailurePolicy::CollectAll,
        ..Default::default()
    };
    let executor = Executor::with_runner(config, Arc::new(mock_runner));
    let scenario = QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Serve,
        Backend::Gpu,
        Format::Gguf,
        "2+2=".to_string(),
        42,
    );
    let results = executor.run_serve_battery("/test/model.gguf", &scenario, false);

    // GPU serve scenarios use A6 category
    let gate_ids: Vec<&str> = results.iter().map(|e| e.gate_id.as_str()).collect();
    assert!(gate_ids.contains(&"F-A6-001"), "GPU serve should use A6 category");
    assert!(gate_ids.contains(&"F-A6-CHAT-001"), "GPU chat should use A6");
}

// ── Direct check method tests (failure paths) ──────────────────────

#[test]
fn test_check_serve_generate_oracle_falsified() {
    // Response doesn't satisfy the arithmetic oracle for "2+2="
    let mock_runner = MockCommandRunner::new()
        .with_http_post_response(r#"{"choices":[{"text":"I don't know"}]}"#);
    let config = ExecutionConfig {
        model_path: Some("/test/model.gguf".to_string()),
        ..Default::default()
    };
    let executor = Executor::with_runner(config, Arc::new(mock_runner));
    let scenario = serve_scenario();
    let start = Instant::now();
    let ev = executor.check_serve_generate(8080, &scenario, &start);
    assert!(ev.outcome.is_fail(), "Oracle should falsify wrong answer");
    assert_eq!(ev.gate_id, "F-A5-001");
}

#[test]
fn test_check_serve_v1_completions_failure() {
    let mock_runner = MockCommandRunner::new().with_http_post_failure();
    let config = ExecutionConfig::default();
    let executor = Executor::with_runner(config, Arc::new(mock_runner));
    let scenario = serve_scenario();
    let start = Instant::now();
    let ev = executor.check_serve_v1_completions(8080, &scenario, &start);
    assert!(ev.outcome.is_fail());
    assert_eq!(ev.gate_id, "F-A5-COMP-001");
    assert!(ev.reason.contains("v1/completions failed"));
}

#[test]
fn test_check_serve_v1_chat_failure() {
    let mock_runner = MockCommandRunner::new().with_http_post_failure();
    let config = ExecutionConfig::default();
    let executor = Executor::with_runner(config, Arc::new(mock_runner));
    let scenario = serve_scenario();
    let start = Instant::now();
    let ev = executor.check_serve_v1_chat(8080, &scenario, &start);
    assert!(ev.outcome.is_fail());
    assert_eq!(ev.gate_id, "F-A5-CHAT-001");
    assert!(ev.reason.contains("v1/chat/completions failed"));
}

#[test]
fn test_check_serve_streaming_request_failure() {
    let mock_runner = MockCommandRunner::new().with_http_post_failure();
    let config = ExecutionConfig::default();
    let executor = Executor::with_runner(config, Arc::new(mock_runner));
    let scenario = serve_scenario();
    let start = Instant::now();
    let ev = executor.check_serve_streaming(8080, &scenario, &start);
    assert!(ev.outcome.is_fail());
    assert_eq!(ev.gate_id, "F-A5-STREAM-001");
    assert!(ev.reason.contains("Streaming request failed"));
}

#[test]
fn test_check_serve_streaming_valid_sse() {
    // Return valid SSE so the corroborated branch is hit
    let sse_body = "data: {\"text\":\"hello\"}\n\ndata: [DONE]\n";
    let mock_runner = MockCommandRunner::new()
        .with_http_post_response(sse_body);
    let config = ExecutionConfig::default();
    let executor = Executor::with_runner(config, Arc::new(mock_runner));
    let scenario = serve_scenario();
    let start = Instant::now();
    let ev = executor.check_serve_streaming(8080, &scenario, &start);
    assert!(ev.outcome.is_pass(), "Valid SSE should be corroborated");
    assert_eq!(ev.gate_id, "F-A5-STREAM-001");
}

#[test]
fn test_check_serve_stop_sequence_request_failure() {
    let mock_runner = MockCommandRunner::new().with_http_post_failure();
    let config = ExecutionConfig::default();
    let executor = Executor::with_runner(config, Arc::new(mock_runner));
    let scenario = serve_scenario();
    let start = Instant::now();
    let ev = executor.check_serve_stop_sequence(8080, &scenario, &start);
    assert!(ev.outcome.is_fail());
    assert_eq!(ev.gate_id, "F-A5-STOP-001");
    assert!(ev.reason.contains("Stop sequence request failed"));
}

#[test]
fn test_check_serve_stop_sequence_not_honored() {
    // Response contains "5" — stop sequence was not honored
    let mock_runner = MockCommandRunner::new()
        .with_http_post_response(r#"{"choices":[{"text":"Count: 1, 2, 3, 4, 5, 6"}]}"#);
    let config = ExecutionConfig::default();
    let executor = Executor::with_runner(config, Arc::new(mock_runner));
    let scenario = serve_scenario();
    let start = Instant::now();
    let ev = executor.check_serve_stop_sequence(8080, &scenario, &start);
    assert!(ev.outcome.is_fail());
    assert_eq!(ev.gate_id, "F-A5-STOP-001");
    assert!(ev.reason.contains("Stop sequence not honored"));
}

#[test]
fn test_check_serve_malformed_server_unhealthy() {
    // http_get fails → server unhealthy after malformed request
    let mock_runner = MockCommandRunner::new()
        .with_http_get_failure();
    let config = ExecutionConfig::default();
    let executor = Executor::with_runner(config, Arc::new(mock_runner));
    let scenario = serve_scenario();
    let start = Instant::now();
    let ev = executor.check_serve_malformed(8080, &scenario, &start);
    assert!(ev.outcome.is_fail());
    assert_eq!(ev.gate_id, "F-A5-ERR-001");
    assert!(ev.reason.contains("unhealthy after malformed"));
}

#[test]
fn test_check_serve_malformed_request_rejected_correctly() {
    // http_post fails (malformed rejected), http_get succeeds (server healthy) → PASS
    // This is the correct Jidoka behavior: server rejects bad input and stays healthy
    let mock_runner = MockCommandRunner::new()
        .with_http_post_failure(); // malformed request rejected (non-2xx)
    // http_get defaults to success (server remains healthy)
    let config = ExecutionConfig::default();
    let executor = Executor::with_runner(config, Arc::new(mock_runner));
    let scenario = serve_scenario();
    let start = Instant::now();
    let ev = executor.check_serve_malformed(8080, &scenario, &start);
    assert!(ev.outcome.is_pass(), "Server that rejects malformed input and stays healthy should pass");
    assert_eq!(ev.gate_id, "F-A5-ERR-001");
    assert!(ev.output.contains("rejected malformed request"));
}

#[test]
fn test_check_serve_malformed_request_accepted_is_failure() {
    // http_post succeeds (server accepted malformed request) and server healthy → FAIL
    // Vacuous truth: a server that accepts bad input provides no error-handling guarantee
    let mock_runner = MockCommandRunner::new();
    // Default: http_post succeeds (malformed accepted), http_get succeeds (server healthy)
    let config = ExecutionConfig::default();
    let executor = Executor::with_runner(config, Arc::new(mock_runner));
    let scenario = serve_scenario();
    let start = Instant::now();
    let ev = executor.check_serve_malformed(8080, &scenario, &start);
    assert!(ev.outcome.is_fail(), "Server that accepts malformed input should fail (expected rejection)");
    assert_eq!(ev.gate_id, "F-A5-ERR-001");
    assert!(ev.reason.contains("accepted malformed request"));
}

#[test]
fn test_check_serve_info_failure() {
    let mock_runner = MockCommandRunner::new().with_http_get_failure();
    let config = ExecutionConfig::default();
    let executor = Executor::with_runner(config, Arc::new(mock_runner));
    let scenario = serve_scenario();
    let start = Instant::now();
    let ev = executor.check_serve_info(8080, &scenario, &start);
    assert!(ev.outcome.is_fail());
    assert_eq!(ev.gate_id, "F-A5-INFO-001");
    assert!(ev.reason.contains("GET / failed"));
}

#[test]
fn test_check_serve_info_empty_response() {
    let mock_runner = MockCommandRunner::new()
        .with_http_get_response("");
    let config = ExecutionConfig::default();
    let executor = Executor::with_runner(config, Arc::new(mock_runner));
    let scenario = serve_scenario();
    let start = Instant::now();
    let ev = executor.check_serve_info(8080, &scenario, &start);
    assert!(ev.outcome.is_fail());
    assert_eq!(ev.gate_id, "F-A5-INFO-001");
}

#[test]
fn test_check_serve_metrics_failure() {
    let mock_runner = MockCommandRunner::new().with_http_get_failure();
    let config = ExecutionConfig::default();
    let executor = Executor::with_runner(config, Arc::new(mock_runner));
    let scenario = serve_scenario();
    let start = Instant::now();
    let ev = executor.check_serve_metrics(8080, &scenario, &start);
    assert!(ev.outcome.is_fail());
    assert_eq!(ev.gate_id, "F-A5-METRICS-001");
    assert!(ev.reason.contains("GET /metrics failed"));
}

#[test]
fn test_check_serve_metrics_empty_response() {
    let mock_runner = MockCommandRunner::new()
        .with_http_get_response("");
    let config = ExecutionConfig::default();
    let executor = Executor::with_runner(config, Arc::new(mock_runner));
    let scenario = serve_scenario();
    let start = Instant::now();
    let ev = executor.check_serve_metrics(8080, &scenario, &start);
    assert!(ev.outcome.is_fail());
    assert_eq!(ev.gate_id, "F-A5-METRICS-001");
}

#[test]
fn test_serve_battery_server_not_ready() {
    // Health check never returns "healthy" — server times out
    // Use minimal timeout to avoid slow test
    let mock_runner = MockCommandRunner::new()
        .with_http_get_response(r#"{"status":"loading"}"#);
    let config = ExecutionConfig {
        model_path: Some("/test/model.gguf".to_string()),
        default_timeout_ms: 2000, // 2s → 1 poll iteration at most
        ..Default::default()
    };
    let executor = Executor::with_runner(config, Arc::new(mock_runner));
    let scenario = serve_scenario();
    let results = executor.run_serve_battery("/test/model.gguf", &scenario, true);

    assert_eq!(results.len(), 1);
    assert!(results[0].outcome.is_fail());
    assert!(results[0].reason.contains("Server failed to become ready"));
}

// ── EOS termination check tests ─────────────────────────────────

#[test]
fn test_serve_battery_eos_termination_pass() {
    // Short output → corroborated (model stopped at EOS)
    let mock_runner = MockCommandRunner::new()
        .with_http_post_response(r#"{"choices":[{"text":"That is all."}]}"#);
    let config = ExecutionConfig::default();
    let executor = Executor::with_runner(config, Arc::new(mock_runner));
    let scenario = serve_scenario();
    let start = Instant::now();
    let ev = executor.check_serve_eos_termination(8080, &scenario, &start);
    assert!(ev.outcome.is_pass(), "Short output should be corroborated");
    assert_eq!(ev.gate_id, "F-A5-EOS-001");
}

#[test]
fn test_serve_battery_eos_termination_repetition() {
    // 25+ words with trigram repetition → falsified
    let repeated = "the end is near ".repeat(10); // 40 words, "the end is" repeats 10x
    let body = format!(r#"{{"choices":[{{"text":"{repeated}"}}]}}"#);
    let mock_runner = MockCommandRunner::new()
        .with_http_post_response(&body);
    let config = ExecutionConfig::default();
    let executor = Executor::with_runner(config, Arc::new(mock_runner));
    let scenario = serve_scenario();
    let start = Instant::now();
    let ev = executor.check_serve_eos_termination(8080, &scenario, &start);
    assert!(ev.outcome.is_fail(), "Repetitive long output should be falsified");
    assert!(ev.reason.contains("EOS failure"));
}

#[test]
fn test_serve_battery_eos_termination_long_but_no_repeat() {
    // Long diverse output → corroborated (no repetition pattern)
    let words: Vec<String> = (0..120).map(|i| format!("word{i}")).collect();
    let diverse_text = words.join(" ");
    let body = format!(r#"{{"choices":[{{"text":"{diverse_text}"}}]}}"#);
    let mock_runner = MockCommandRunner::new()
        .with_http_post_response(&body);
    let config = ExecutionConfig::default();
    let executor = Executor::with_runner(config, Arc::new(mock_runner));
    let scenario = serve_scenario();
    let start = Instant::now();
    let ev = executor.check_serve_eos_termination(8080, &scenario, &start);
    assert!(ev.outcome.is_pass(), "Long but diverse output should pass");
}

#[test]
fn test_detect_repetition_short_text() {
    // < 20 words → always false
    assert!(!Executor::detect_repetition("hello world"));
    assert!(!Executor::detect_repetition("one two three four five"));
    assert!(!Executor::detect_repetition(""));
}

#[test]
fn test_detect_repetition_with_trigrams() {
    // Repeated trigrams → true
    let text = "a b c ".repeat(10); // "a b c" repeats 10 times → trigram count > 5
    assert!(Executor::detect_repetition(&text));
}

#[test]
fn test_serve_battery_19_checks_total() {
    let mock_runner = mock_with_healthy_server();
    let config = ExecutionConfig {
        model_path: Some("/test/model.gguf".to_string()),
        failure_policy: FailurePolicy::CollectAll,
        ..Default::default()
    };
    let executor = Executor::with_runner(config, Arc::new(mock_runner));
    let scenario = serve_scenario();
    let results = executor.run_serve_battery("/test/model.gguf", &scenario, true);

    assert_eq!(results.len(), 19, "Expected 19 battery checks, got {}", results.len());

    // Verify all gate ID categories are present
    let gate_ids: Vec<&str> = results.iter().map(|e| e.gate_id.as_str()).collect();
    assert!(gate_ids.contains(&"F-A5-EOS-001"), "Missing EOS termination gate");
    assert!(gate_ids.contains(&"F-A5-PERF-001"), "Missing perf floor gate");
    assert!(gate_ids.contains(&"F-A5-SCHEMA-001"), "Missing schema gate");
    assert!(gate_ids.contains(&"F-A5-DETERM-001"), "Missing determinism gate");
}

#[test]
fn test_execute_scenarios_partitions_serve() {
    let mock_runner = mock_with_healthy_server()
        .with_inference_response("The answer is 4.");
    let config = ExecutionConfig {
        model_path: Some("/test/model.gguf".to_string()),
        failure_policy: FailurePolicy::CollectAll,
        ..Default::default()
    };
    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));

    let run_scenario = QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Cpu,
        Format::Gguf,
        "2+2=".to_string(),
        42,
    );
    let serve_scenario_1 = QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Serve,
        Backend::Cpu,
        Format::Gguf,
        "2+2=".to_string(),
        42,
    );

    let scenarios = vec![run_scenario, serve_scenario_1];
    let (passed, failed, _skipped) = executor.execute_scenarios(scenarios, "test");

    // Run scenario: 1 evidence, Serve battery: 19 evidence
    // Total passed should be > 1 (at minimum the run + battery checks)
    assert!(
        passed >= 2,
        "Should have at least run pass + some battery passes, got passed={passed} failed={failed}"
    );
}

// ── Perf floor check tests ──────────────────────────────────────

#[test]
fn test_serve_battery_perf_floor_pass() {
    // 10.0 tok/s > 0.1 floor → corroborated
    let executor = Executor::with_runner(ExecutionConfig::default(), Arc::new(MockCommandRunner::new()));
    let scenario = serve_scenario();
    let start = Instant::now();
    let ev = executor.check_serve_perf_floor(Some(10.0), &scenario, &start);
    assert!(ev.outcome.is_pass(), "10 tok/s should pass perf floor");
    assert_eq!(ev.gate_id, "F-A5-PERF-001");
    assert_eq!(ev.metrics.tokens_per_second, Some(10.0));
}

#[test]
fn test_serve_battery_perf_floor_fail() {
    // 0.05 tok/s < 0.1 floor → falsified
    let executor = Executor::with_runner(ExecutionConfig::default(), Arc::new(MockCommandRunner::new()));
    let scenario = serve_scenario();
    let start = Instant::now();
    let ev = executor.check_serve_perf_floor(Some(0.05), &scenario, &start);
    assert!(ev.outcome.is_fail(), "0.05 tok/s should fail perf floor");
    assert_eq!(ev.gate_id, "F-A5-PERF-001");
    assert!(ev.reason.contains("too slow"));
    assert!(ev.reason.contains("0.05"));
}

#[test]
fn test_serve_battery_perf_floor_no_tps() {
    // No TPS reported → skipped (Popper: untested ≠ corroborated)
    let executor = Executor::with_runner(ExecutionConfig::default(), Arc::new(MockCommandRunner::new()));
    let scenario = serve_scenario();
    let start = Instant::now();
    let ev = executor.check_serve_perf_floor(None, &scenario, &start);
    assert_eq!(ev.outcome, Outcome::Skipped, "Missing TPS should be Skipped, not Corroborated");
    assert_eq!(ev.gate_id, "F-A5-PERF-001");
}

#[test]
fn test_serve_battery_perf_floor_boundary() {
    // Exactly at floor → pass (not strictly less)
    let executor = Executor::with_runner(ExecutionConfig::default(), Arc::new(MockCommandRunner::new()));
    let scenario = serve_scenario();
    let start = Instant::now();
    let ev = executor.check_serve_perf_floor(Some(0.1), &scenario, &start);
    assert!(ev.outcome.is_pass(), "Exactly at floor should pass");
}

#[test]
fn test_serve_battery_perf_floor_just_below() {
    // Just below floor → fail
    let executor = Executor::with_runner(ExecutionConfig::default(), Arc::new(MockCommandRunner::new()));
    let scenario = serve_scenario();
    let start = Instant::now();
    let ev = executor.check_serve_perf_floor(Some(0.099), &scenario, &start);
    assert!(ev.outcome.is_fail(), "Just below floor should fail");
}

// ── Checks 11-19: individual tests ──────────────────────────────

#[test]
fn test_check_serve_v1_models_pass() {
    let mock_runner = MockCommandRunner::new()
        .with_http_get_response(r#"{"data":[{"id":"test-model","object":"model"}]}"#);
    let executor = Executor::with_runner(ExecutionConfig::default(), Arc::new(mock_runner));
    let scenario = serve_scenario();
    let start = Instant::now();
    let ev = executor.check_serve_v1_models(8080, &scenario, &start);
    assert!(ev.outcome.is_pass(), "Valid /v1/models response should pass");
    assert_eq!(ev.gate_id, "F-A5-MODELS-001");
}

#[test]
fn test_check_serve_v1_models_missing_data() {
    let mock_runner = MockCommandRunner::new()
        .with_http_get_response(r#"{"models":[]}"#);
    let executor = Executor::with_runner(ExecutionConfig::default(), Arc::new(mock_runner));
    let scenario = serve_scenario();
    let start = Instant::now();
    let ev = executor.check_serve_v1_models(8080, &scenario, &start);
    assert!(ev.outcome.is_fail(), "Missing 'data' array should fail");
}

#[test]
fn test_check_serve_v1_models_failure() {
    let mock_runner = MockCommandRunner::new().with_http_get_failure();
    let executor = Executor::with_runner(ExecutionConfig::default(), Arc::new(mock_runner));
    let scenario = serve_scenario();
    let start = Instant::now();
    let ev = executor.check_serve_v1_models(8080, &scenario, &start);
    assert!(ev.outcome.is_fail());
    assert!(ev.reason.contains("GET /v1/models failed"));
}

#[test]
fn test_check_serve_template_leakage_pass() {
    // No template markers in output → pass
    let mock_runner = MockCommandRunner::new()
        .with_http_post_response(MOCK_CHAT_RESPONSE);
    let executor = Executor::with_runner(ExecutionConfig::default(), Arc::new(mock_runner));
    let scenario = serve_scenario();
    let start = Instant::now();
    let ev = executor.check_serve_template_leakage(8080, &scenario, &start);
    assert!(ev.outcome.is_pass(), "Clean output should pass template check");
    assert_eq!(ev.gate_id, "F-A5-TMPL-001");
}

#[test]
fn test_check_serve_template_leakage_detected() {
    // Template markers leaked into output
    let leaked = r#"{"id":"t","object":"chat.completion","created":0,"model":"t","choices":[{"index":0,"message":{"role":"assistant","content":"<|im_start|>assistant Hello"},"finish_reason":"stop"}]}"#;
    let mock_runner = MockCommandRunner::new()
        .with_http_post_response(leaked);
    let executor = Executor::with_runner(ExecutionConfig::default(), Arc::new(mock_runner));
    let scenario = serve_scenario();
    let start = Instant::now();
    let ev = executor.check_serve_template_leakage(8080, &scenario, &start);
    assert!(ev.outcome.is_fail(), "Leaked template markers should fail");
    assert!(ev.reason.contains("Template markers leaked"));
}

#[test]
fn test_check_serve_temp_determinism_pass() {
    // Same response twice → deterministic
    let mock_runner = MockCommandRunner::new()
        .with_http_post_response(MOCK_CHAT_RESPONSE);
    let executor = Executor::with_runner(ExecutionConfig::default(), Arc::new(mock_runner));
    let scenario = serve_scenario();
    let start = Instant::now();
    let ev = executor.check_serve_temp_determinism(8080, &scenario, &start);
    assert!(ev.outcome.is_pass(), "Identical responses should be deterministic");
    assert_eq!(ev.gate_id, "F-A5-DETERM-001");
}

#[test]
fn test_check_serve_temp_determinism_request_failure() {
    let mock_runner = MockCommandRunner::new().with_http_post_failure();
    let executor = Executor::with_runner(ExecutionConfig::default(), Arc::new(mock_runner));
    let scenario = serve_scenario();
    let start = Instant::now();
    let ev = executor.check_serve_temp_determinism(8080, &scenario, &start);
    assert!(ev.outcome.is_fail());
    assert!(ev.reason.contains("one or both requests failed"));
}

#[test]
fn test_check_serve_multi_turn_pass() {
    // Response contains "blue" → context preserved
    let response = r#"{"id":"t","object":"chat.completion","created":0,"model":"t","choices":[{"index":0,"message":{"role":"assistant","content":"Your favorite color is blue."},"finish_reason":"stop"}]}"#;
    let mock_runner = MockCommandRunner::new()
        .with_http_post_response(response);
    let executor = Executor::with_runner(ExecutionConfig::default(), Arc::new(mock_runner));
    let scenario = serve_scenario();
    let start = Instant::now();
    let ev = executor.check_serve_multi_turn(8080, &scenario, &start);
    assert!(ev.outcome.is_pass(), "Response with 'blue' should pass multi-turn");
    assert_eq!(ev.gate_id, "F-A5-MULTI-001");
}

#[test]
fn test_check_serve_multi_turn_context_lost() {
    // Response doesn't mention "blue" → context lost
    let response = r#"{"id":"t","object":"chat.completion","created":0,"model":"t","choices":[{"index":0,"message":{"role":"assistant","content":"I don't know."},"finish_reason":"stop"}]}"#;
    let mock_runner = MockCommandRunner::new()
        .with_http_post_response(response);
    let executor = Executor::with_runner(ExecutionConfig::default(), Arc::new(mock_runner));
    let scenario = serve_scenario();
    let start = Instant::now();
    let ev = executor.check_serve_multi_turn(8080, &scenario, &start);
    assert!(ev.outcome.is_fail(), "Missing 'blue' should fail multi-turn");
    assert!(ev.reason.contains("context lost"));
}

#[test]
fn test_check_serve_tokenize_optional() {
    // /tokenize not available → skipped (Popper: untested ≠ corroborated)
    let mock_runner = MockCommandRunner::new().with_http_post_failure();
    let executor = Executor::with_runner(ExecutionConfig::default(), Arc::new(mock_runner));
    let scenario = serve_scenario();
    let start = Instant::now();
    let ev = executor.check_serve_tokenize(8080, &scenario, &start);
    assert_eq!(ev.outcome, Outcome::Skipped, "Missing /tokenize should be Skipped, not Corroborated");
    assert_eq!(ev.gate_id, "F-A5-TOK-001");
}

#[test]
fn test_check_serve_tokenize_valid() {
    let mock_runner = MockCommandRunner::new()
        .with_http_post_response(r#"{"tokens":[1,2,3],"count":3}"#);
    let executor = Executor::with_runner(ExecutionConfig::default(), Arc::new(mock_runner));
    let scenario = serve_scenario();
    let start = Instant::now();
    let ev = executor.check_serve_tokenize(8080, &scenario, &start);
    assert!(ev.outcome.is_pass(), "Valid tokenize response should pass");
}

#[test]
fn test_check_serve_tokenize_missing_fields() {
    let mock_runner = MockCommandRunner::new()
        .with_http_post_response(r#"{"result":"ok"}"#);
    let executor = Executor::with_runner(ExecutionConfig::default(), Arc::new(mock_runner));
    let scenario = serve_scenario();
    let start = Instant::now();
    let ev = executor.check_serve_tokenize(8080, &scenario, &start);
    assert!(ev.outcome.is_fail(), "Missing tokens/count should fail");
}

#[test]
fn test_check_serve_special_chars_pass() {
    let mock_runner = MockCommandRunner::new()
        .with_http_post_response(r#"{"choices":[{"text":"ok"}]}"#);
    let executor = Executor::with_runner(ExecutionConfig::default(), Arc::new(mock_runner));
    let scenario = serve_scenario();
    let start = Instant::now();
    let ev = executor.check_serve_special_chars(8080, &scenario, &start);
    assert!(ev.outcome.is_pass(), "Successful response to special chars should pass");
    assert_eq!(ev.gate_id, "F-A5-CHARS-001");
}

#[test]
fn test_check_serve_special_chars_failure() {
    let mock_runner = MockCommandRunner::new().with_http_post_failure();
    let executor = Executor::with_runner(ExecutionConfig::default(), Arc::new(mock_runner));
    let scenario = serve_scenario();
    let start = Instant::now();
    let ev = executor.check_serve_special_chars(8080, &scenario, &start);
    assert!(ev.outcome.is_fail());
    assert!(ev.reason.contains("Special char prompt failed"));
}

#[test]
fn test_check_serve_special_chars_empty_response_is_failure() {
    // HTTP success with empty body is vacuous — no actual output was produced
    let mock_runner = MockCommandRunner::new()
        .with_http_post_response(""); // empty response
    let executor = Executor::with_runner(ExecutionConfig::default(), Arc::new(mock_runner));
    let scenario = serve_scenario();
    let start = Instant::now();
    let ev = executor.check_serve_special_chars(8080, &scenario, &start);
    assert!(ev.outcome.is_fail(), "Empty response to special char prompt should fail");
    assert!(ev.reason.contains("empty"), "Reason should mention empty response");
}

#[test]
fn test_check_serve_chat_streaming_valid() {
    let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\ndata: [DONE]\n";
    let mock_runner = MockCommandRunner::new()
        .with_http_post_response(sse);
    let executor = Executor::with_runner(ExecutionConfig::default(), Arc::new(mock_runner));
    let scenario = serve_scenario();
    let start = Instant::now();
    let ev = executor.check_serve_chat_streaming(8080, &scenario, &start);
    assert!(ev.outcome.is_pass(), "Valid chat SSE should pass");
    assert_eq!(ev.gate_id, "F-A5-CSTREAM-001");
}

#[test]
fn test_check_serve_chat_streaming_invalid() {
    let mock_runner = MockCommandRunner::new()
        .with_http_post_response(r#"{"not":"sse"}"#);
    let executor = Executor::with_runner(ExecutionConfig::default(), Arc::new(mock_runner));
    let scenario = serve_scenario();
    let start = Instant::now();
    let ev = executor.check_serve_chat_streaming(8080, &scenario, &start);
    assert!(ev.outcome.is_fail(), "Non-SSE response should fail chat streaming");
}

#[test]
fn test_check_serve_max_tokens_one_pass() {
    // Short response (1 word) → pass
    let mock_runner = MockCommandRunner::new()
        .with_http_post_response(r#"{"choices":[{"text":"world"}]}"#);
    let executor = Executor::with_runner(ExecutionConfig::default(), Arc::new(mock_runner));
    let scenario = serve_scenario();
    let start = Instant::now();
    let ev = executor.check_serve_max_tokens_one(8080, &scenario, &start);
    assert!(ev.outcome.is_pass(), "1 word for max_tokens=1 should pass");
    assert_eq!(ev.gate_id, "F-A5-MAXTOK-001");
}

#[test]
fn test_check_serve_max_tokens_one_violated() {
    // Long response → max_tokens not honored
    let mock_runner = MockCommandRunner::new()
        .with_http_post_response(r#"{"choices":[{"text":"this is way too many words for one token limit"}]}"#);
    let executor = Executor::with_runner(ExecutionConfig::default(), Arc::new(mock_runner));
    let scenario = serve_scenario();
    let start = Instant::now();
    let ev = executor.check_serve_max_tokens_one(8080, &scenario, &start);
    assert!(ev.outcome.is_fail(), "10 words for max_tokens=1 should fail");
    assert!(ev.reason.contains("max_tokens=1 violated"));
}

#[test]
fn test_check_serve_response_schema_pass() {
    // Valid OpenAI ChatResponse → pass
    let mock_runner = MockCommandRunner::new()
        .with_http_post_response(MOCK_CHAT_RESPONSE);
    let executor = Executor::with_runner(ExecutionConfig::default(), Arc::new(mock_runner));
    let scenario = serve_scenario();
    let start = Instant::now();
    let ev = executor.check_serve_response_schema(8080, &scenario, &start);
    assert!(ev.outcome.is_pass(), "Valid ChatResponse should pass schema check");
    assert_eq!(ev.gate_id, "F-A5-SCHEMA-001");
}

#[test]
fn test_check_serve_response_schema_invalid_json() {
    let mock_runner = MockCommandRunner::new()
        .with_http_post_response(r"not json");
    let executor = Executor::with_runner(ExecutionConfig::default(), Arc::new(mock_runner));
    let scenario = serve_scenario();
    let start = Instant::now();
    let ev = executor.check_serve_response_schema(8080, &scenario, &start);
    assert!(ev.outcome.is_fail(), "Invalid JSON should fail schema check");
    assert!(ev.reason.contains("does not match"));
}

#[test]
fn test_check_serve_response_schema_missing_fields() {
    // Valid JSON but missing required ChatResponse fields
    let mock_runner = MockCommandRunner::new()
        .with_http_post_response(r#"{"result":"ok"}"#);
    let executor = Executor::with_runner(ExecutionConfig::default(), Arc::new(mock_runner));
    let scenario = serve_scenario();
    let start = Instant::now();
    let ev = executor.check_serve_response_schema(8080, &scenario, &start);
    assert!(ev.outcome.is_fail(), "Missing fields should fail schema check");
}

#[test]
fn test_check_serve_response_schema_empty_choices() {
    // ChatResponse parses but has no choices → assert_response_valid fails
    let response = r#"{"id":"t","object":"chat.completion","created":0,"model":"t","choices":[]}"#;
    let mock_runner = MockCommandRunner::new()
        .with_http_post_response(response);
    let executor = Executor::with_runner(ExecutionConfig::default(), Arc::new(mock_runner));
    let scenario = serve_scenario();
    let start = Instant::now();
    let ev = executor.check_serve_response_schema(8080, &scenario, &start);
    assert!(ev.outcome.is_fail(), "Empty choices should fail validation");
    assert!(ev.reason.contains("validation failed"));
}

#[test]
fn test_extract_chat_text_probar_format() {
    // OpenAI ChatResponse format
    let json = r#"{"id":"t","object":"chat.completion","created":0,"model":"t","choices":[{"index":0,"message":{"role":"assistant","content":"Hello world"},"finish_reason":"stop"}]}"#;
    assert_eq!(Executor::extract_chat_text(json), "Hello world");
}

#[test]
fn test_extract_chat_text_fallback() {
    // Non-ChatResponse → returns raw
    let raw = r"just plain text";
    assert_eq!(Executor::extract_chat_text(raw), "just plain text");
}

// ── check_serve_temp_determinism: uncovered branches ───────────────────────

/// check_serve_temp_determinism: http_post succeeds but response is not valid ChatResponse
/// → responses.len() < 2 → falsified "could not parse responses" (lines 667-675)
#[test]
fn test_check_serve_temp_determinism_parse_failure() {
    // Return valid HTTP success but with JSON that doesn't match ProbarChatResponse schema
    let mock_runner = MockCommandRunner::new()
        .with_http_post_response(r#"{"foo":"bar"}"#);
    let executor = Executor::with_runner(ExecutionConfig::default(), Arc::new(mock_runner));
    let scenario = serve_scenario();
    let start = Instant::now();
    let ev = executor.check_serve_temp_determinism(8080, &scenario, &start);
    assert!(ev.outcome.is_fail(), "Unparseable responses should fail determinism check");
    assert!(
        ev.reason.contains("could not parse"),
        "Reason should mention parse failure, got: {}",
        ev.reason
    );
}

/// check_serve_temp_determinism: two different valid ChatResponse objects
/// → assert_deterministic returns !passed → falsified (lines 690-697)
#[test]
fn test_check_serve_temp_determinism_nondeterministic() {
    use crate::command::{CommandOutput, CommandRunner};
    use std::sync::atomic::{AtomicU32, Ordering};

    struct AlternatingRunner {
        call_count: AtomicU32,
    }

    impl CommandRunner for AlternatingRunner {
        fn http_post(&self, _url: &str, _body: &str) -> CommandOutput {
            let n = self.call_count.fetch_add(1, Ordering::Relaxed);
            let content = if n % 2 == 0 { "Paris" } else { "London" };
            let json = format!(
                r#"{{"id":"t","object":"chat.completion","created":0,"model":"t","choices":[{{"index":0,"message":{{"role":"assistant","content":"{content}"}},"finish_reason":"stop"}}]}}"#
            );
            CommandOutput::success(json)
        }
        fn run_inference(&self, _: &std::path::Path, _: &str, _: u32, _: bool, _: &[&str]) -> CommandOutput { CommandOutput::success("") }
        fn convert_model(&self, _: &std::path::Path, _: &std::path::Path) -> CommandOutput { CommandOutput::success("") }
        fn inspect_model(&self, _: &std::path::Path) -> CommandOutput { CommandOutput::success("") }
        fn validate_model(&self, _: &std::path::Path) -> CommandOutput { CommandOutput::success("") }
        fn bench_model(&self, _: &std::path::Path) -> CommandOutput { CommandOutput::success("") }
        fn check_model(&self, _: &std::path::Path) -> CommandOutput { CommandOutput::success("All checks passed") }
        fn profile_model(&self, _: &std::path::Path, _: u32, _: u32) -> CommandOutput { CommandOutput::success("") }
        fn profile_ci(&self, _: &std::path::Path, _: Option<f64>, _: Option<f64>, _: u32, _: u32, _: bool) -> CommandOutput { CommandOutput::success("") }
        fn diff_tensors(&self, _: &std::path::Path, _: &std::path::Path, _: bool) -> CommandOutput { CommandOutput::success("") }
        fn compare_inference(&self, _: &std::path::Path, _: &std::path::Path, _: &str, _: u32, _: f64) -> CommandOutput { CommandOutput::success("") }
        fn profile_with_flamegraph(&self, _: &std::path::Path, _: &std::path::Path, _: bool) -> CommandOutput { CommandOutput::success("") }
        fn profile_with_focus(&self, _: &std::path::Path, _: &str, _: bool) -> CommandOutput { CommandOutput::success("") }
        fn fingerprint_model(&self, _: &std::path::Path, _: bool) -> CommandOutput { CommandOutput::success("") }
        fn validate_stats(&self, _: &std::path::Path, _: &std::path::Path) -> CommandOutput { CommandOutput::success("") }
        fn validate_model_strict(&self, _: &std::path::Path) -> CommandOutput { CommandOutput::success(r#"{"valid":true,"tensors_checked":0,"issues":[]}"#) }
        fn pull_model(&self, _: &str) -> CommandOutput { CommandOutput::success("Path: /mock/model.safetensors") }
        fn inspect_model_json(&self, _: &std::path::Path) -> CommandOutput { CommandOutput::success(r#"{"format":"SafeTensors","tensor_count":0,"tensor_names":[]}"#) }
        fn run_ollama_inference(&self, _: &str, _: &str, _: f64) -> CommandOutput { CommandOutput::success("") }
        fn pull_ollama_model(&self, _: &str) -> CommandOutput { CommandOutput::success("done") }
        fn create_ollama_model(&self, _: &str, _: &std::path::Path) -> CommandOutput { CommandOutput::success("done") }
        fn serve_model(&self, _: &std::path::Path, _: u16) -> CommandOutput { CommandOutput::success(r#"{"status":"listening"}"#) }
        fn http_get(&self, _: &str) -> CommandOutput { CommandOutput::success(r#"{"models":[]}"#) }
        fn profile_memory(&self, _: &std::path::Path) -> CommandOutput { CommandOutput::success(r#"{"peak_rss_mb":512}"#) }
        fn run_chat(&self, _: &std::path::Path, _: &str, _: bool, _: &[&str]) -> CommandOutput { CommandOutput::success("") }
        fn spawn_serve(&self, _: &std::path::Path, _: u16, _: bool) -> CommandOutput { CommandOutput::success("12345") }
        fn quantize_model(&self, _: &std::path::Path, _: &std::path::Path, _: &str) -> CommandOutput { CommandOutput::success("") }
        fn import_model(&self, _: &std::path::Path, _: &std::path::Path) -> CommandOutput { CommandOutput::success("") }
        fn prune_model(&self, _: &std::path::Path, _: &std::path::Path, _: &str, _: f64) -> CommandOutput { CommandOutput::success("") }
        fn distill_model(&self, _: &std::path::Path, _: &std::path::Path, _: &std::path::Path, _: &str) -> CommandOutput { CommandOutput::success("") }
    }

    let runner = AlternatingRunner { call_count: AtomicU32::new(0) };
    let executor = Executor::with_runner(ExecutionConfig::default(), Arc::new(runner));
    let scenario = serve_scenario();
    let start = Instant::now();
    let ev = executor.check_serve_temp_determinism(8080, &scenario, &start);
    assert!(
        ev.outcome.is_fail(),
        "Different responses should fail determinism check, got: {:?}",
        ev.outcome
    );
}

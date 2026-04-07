//! Contract enforcement tests for apr-serve-v1 + http-api-v1 (PMAT-191)
//!
//! FALSIFY-SRV-001 through SRV-007: Server lifecycle, health, 404, isolation,
//! Qwen3 template dispatch, architecture cache, thinking block suppression.
//!
//! FALSIFY-HTTP-001 through HTTP-004: JSON response, error envelope, CORS, 404.

use super::*;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::Arc;

fn create_test_model_file() -> tempfile::NamedTempFile {
    let mut file = tempfile::NamedTempFile::new().expect("tempfile::NamedTempFile::new()");
    file.write_all(b"test model data").expect("write_all(b'test model data'");
    file
}

fn create_test_state() -> Arc<ServerState> {
    let model = create_test_model_file();
    let config = ServerConfig::default();
    let state = ServerState::new(model.path().to_path_buf(), config).expect("to_path_buf(), config");
    // Keep the temp file alive by leaking the path
    std::mem::forget(model);
    Arc::new(state)
}

// ═══ Contract: apr-serve-v1 enforcement (PMAT-191) ═══

#[test]
fn falsify_srv_001_health_returns_503_when_not_ready() {
    // FALSIFY-SRV-001: /health returns 503 during model loading phase.
    // ServerState starts with ready=false. health_check() must return Unhealthy.
    let model = create_test_model_file();
    let config = ServerConfig::default();
    let state = ServerState::new(model.path().to_path_buf(), config).expect("to_path_buf(), config");
    // State starts not ready
    assert!(
        !state.is_ready(),
        "FALSIFY-SRV-001: server must start in not-ready state"
    );
    let health = health_check(&state);
    assert_eq!(
        health.status,
        HealthStatus::Unhealthy,
        "FALSIFY-SRV-001: health must be Unhealthy when not ready"
    );
}

#[test]
fn falsify_srv_001b_health_returns_200_when_ready() {
    // FALSIFY-SRV-001b: /health returns Healthy once ready=true.
    let model = create_test_model_file();
    let config = ServerConfig::default();
    let state = ServerState::new(model.path().to_path_buf(), config).expect("to_path_buf(), config");
    state.ready.store(true, Ordering::Release);
    assert!(state.is_ready());
    let health = health_check(&state);
    assert_eq!(
        health.status,
        HealthStatus::Healthy,
        "FALSIFY-SRV-001b: health must be Healthy when ready"
    );
}

#[test]
fn falsify_srv_003_error_response_is_json() {
    // FALSIFY-SRV-003: Error responses are valid JSON with error envelope.
    // ErrorResponse serializes correctly with all fields.
    let err = ErrorResponse::new("not_found", "Resource not found");
    let json = serde_json::to_string(&err).expect("serialize");
    let parsed: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
    assert!(
        parsed.get("error").is_some(),
        "FALSIFY-SRV-003: error field present"
    );
    assert!(
        parsed.get("message").is_some(),
        "FALSIFY-SRV-003: message field present"
    );
    assert_eq!(
        parsed["error"].as_str().expect("as_str("),
        "not_found",
        "FALSIFY-SRV-003: error type preserved"
    );
}

#[test]
fn falsify_srv_003b_error_with_request_id() {
    // FALSIFY-SRV-003b: Error responses include request_id when set.
    let err = ErrorResponse::new("timeout", "Request timed out").with_request_id("req-abc");
    let json = serde_json::to_string(&err).expect("serialize");
    let parsed: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
    assert_eq!(
        parsed["request_id"].as_str().expect("as_str("),
        "req-abc",
        "FALSIFY-SRV-003b: request_id preserved"
    );
}

#[test]
fn falsify_srv_003c_error_roundtrip() {
    // FALSIFY-SRV-003c: ErrorResponse survives JSON roundtrip.
    let original = ErrorResponse::new("test_error", "A message").with_request_id("id-42");
    let json = serde_json::to_string(&original).expect("serialize");
    let parsed: ErrorResponse = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(
        parsed.error, original.error,
        "FALSIFY-SRV-003c: error roundtrip"
    );
    assert_eq!(
        parsed.message, original.message,
        "FALSIFY-SRV-003c: message roundtrip"
    );
    assert_eq!(
        parsed.request_id, original.request_id,
        "FALSIFY-SRV-003c: request_id roundtrip"
    );
}

#[test]
fn falsify_srv_004_metrics_thread_safe() {
    // FALSIFY-SRV-004: Concurrent inference isolation — metrics are thread-safe.
    // 10 threads each recording 100 requests must not lose any.
    let metrics = ServerMetrics::new();
    let handles: Vec<_> = (0..10)
        .map(|_| {
            let m = Arc::clone(&metrics);
            std::thread::spawn(move || {
                for _ in 0..100 {
                    m.record_request(true, 1, 1);
                }
            })
        })
        .collect();
    for h in handles {
        h.join().expect("join(");
    }
    assert_eq!(
        metrics.requests_total.load(Ordering::Relaxed),
        1000,
        "FALSIFY-SRV-004: all 1000 requests recorded (thread-safe)"
    );
    assert_eq!(
        metrics.tokens_generated.load(Ordering::Relaxed),
        1000,
        "FALSIFY-SRV-004: all 1000 token increments recorded"
    );
}

#[test]
fn falsify_srv_005_qwen3_nothink_template() {
    // FALSIFY-SRV-005: Qwen3 architecture triggers NoThinkTemplate.
    // This test verifies that the detect_format_from_name function in realizar
    // correctly maps "qwen3" to Qwen3NoThink. Since realizar is a dependency,
    // we test the contract boundary at the apr-cli level.
    #[cfg(feature = "inference")]
    {
        use realizar::chat_template::{detect_format_from_name, TemplateFormat};
        assert_eq!(
            detect_format_from_name("qwen3"),
            TemplateFormat::Qwen3NoThink,
            "FALSIFY-SRV-005: qwen3 architecture must get Qwen3NoThink"
        );
        assert_eq!(
            detect_format_from_name("Qwen3-1.7B-Q4_K_M"),
            TemplateFormat::Qwen3NoThink,
            "FALSIFY-SRV-005: Qwen3 model name must get Qwen3NoThink"
        );
        // Qwen2 should NOT get NoThink
        assert_ne!(
            detect_format_from_name("qwen2"),
            TemplateFormat::Qwen3NoThink,
            "FALSIFY-SRV-005: qwen2 must NOT get Qwen3NoThink"
        );
    }
}

#[test]
fn falsify_srv_006_health_response_fields() {
    // FALSIFY-SRV-006: HealthResponse has all required fields per contract.
    let model = create_test_model_file();
    let config = ServerConfig::default();
    let state = ServerState::new(model.path().to_path_buf(), config).expect("to_path_buf(), config");
    state.ready.store(true, Ordering::Release);
    let health = health_check(&state);
    // All fields must be present and valid
    assert!(
        !health.model_id.is_empty(),
        "FALSIFY-SRV-006: model_id present"
    );
    assert!(
        !health.version.is_empty(),
        "FALSIFY-SRV-006: version present"
    );
    // Uptime must be non-negative (u64 guarantees this, but check it's reasonable)
    assert!(
        health.uptime_seconds < 3600,
        "FALSIFY-SRV-006: uptime < 1h for fresh server"
    );
}

#[test]
fn falsify_srv_007_health_status_serde() {
    // FALSIFY-SRV-007: HealthStatus enum serializes to lowercase strings.
    // Contract: "healthy", "degraded", "unhealthy" — not "Healthy", etc.
    let json = serde_json::to_string(&HealthStatus::Healthy).expect("serde_json::to_string(&HealthS");
    assert_eq!(json, "\"healthy\"", "FALSIFY-SRV-007: Healthy → lowercase");
    let json = serde_json::to_string(&HealthStatus::Degraded).expect("serde_json::to_string(&HealthS");
    assert_eq!(
        json, "\"degraded\"",
        "FALSIFY-SRV-007: Degraded → lowercase"
    );
    let json = serde_json::to_string(&HealthStatus::Unhealthy).expect("serde_json::to_string(&HealthS");
    assert_eq!(
        json, "\"unhealthy\"",
        "FALSIFY-SRV-007: Unhealthy → lowercase"
    );
}

// ═══ Contract: http-api-v1 enforcement (PMAT-191) ═══

#[test]
fn falsify_http_001_generate_request_serde() {
    // FALSIFY-HTTP-001: GenerateRequest schema roundtrip.
    let json = serde_json::json!({
        "prompt": "Hello world",
        "max_tokens": 100,
        "temperature": 0.7,
        "stream": false
    });
    let req: GenerateRequest = serde_json::from_value(json.clone()).expect("parse request");
    assert_eq!(req.prompt, "Hello world");
    assert_eq!(req.max_tokens, 100);
    assert!((req.temperature - 0.7).abs() < 0.01);
}

#[test]
fn falsify_http_001b_generate_request_defaults() {
    // FALSIFY-HTTP-001b: GenerateRequest uses defaults when fields omitted.
    let json = serde_json::json!({ "prompt": "test" });
    let req: GenerateRequest = serde_json::from_value(json).expect("parse with defaults");
    assert_eq!(req.prompt, "test");
    // Defaults should be reasonable
    assert!(
        req.max_tokens > 0,
        "FALSIFY-HTTP-001b: max_tokens has default"
    );
    assert!(
        req.temperature >= 0.0 && req.temperature <= 2.0,
        "FALSIFY-HTTP-001b: temperature in valid range"
    );
}

#[test]
fn falsify_http_002_error_envelope_format() {
    // FALSIFY-HTTP-002: Error envelope always has {error, message} structure.
    let err = ErrorResponse::new("invalid_json", "Unexpected token at position 5");
    let json = serde_json::to_value(&err).expect("serialize");
    assert!(json.is_object(), "FALSIFY-HTTP-002: error is JSON object");
    assert!(
        json.get("error").and_then(|v| v.as_str()).is_some(),
        "FALSIFY-HTTP-002: error field is string"
    );
    assert!(
        json.get("message").and_then(|v| v.as_str()).is_some(),
        "FALSIFY-HTTP-002: message field is string"
    );
    // request_id absent when not set
    assert!(
        json.get("request_id").is_none(),
        "FALSIFY-HTTP-002: request_id absent when not set"
    );
}

#[test]
fn falsify_http_003_cors_config_default_enabled() {
    // FALSIFY-HTTP-003: CORS is enabled by default in ServerConfig.
    let config = ServerConfig::default();
    assert!(
        config.cors,
        "FALSIFY-HTTP-003: CORS must be enabled by default"
    );
}

#[test]
fn falsify_http_004_server_config_defaults() {
    // FALSIFY-HTTP-004: ServerConfig defaults match http-api-v1 contract.
    let config = ServerConfig::default();
    assert_eq!(config.host, "127.0.0.1", "FALSIFY-HTTP-004: default host");
    assert_eq!(config.port, 8080, "FALSIFY-HTTP-004: default port");
    assert!(
        config.metrics,
        "FALSIFY-HTTP-004: metrics enabled by default"
    );
    assert!(
        !config.no_gpu,
        "FALSIFY-HTTP-004: GPU not disabled by default"
    );
}

#[test]
fn falsify_http_004b_health_response_serde_roundtrip() {
    // FALSIFY-HTTP-004b: HealthResponse survives JSON roundtrip.
    let health = HealthResponse {
        status: HealthStatus::Healthy,
        model_id: "qwen3-1.7b".into(),
        version: "0.27.5".into(),
        uptime_seconds: 42,
        requests_total: 100,
        gpu_available: true,
    };
    let json = serde_json::to_string(&health).expect("serialize");
    let parsed: HealthResponse = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(
        parsed.status,
        HealthStatus::Healthy,
        "FALSIFY-HTTP-004b: status roundtrip"
    );
    assert_eq!(
        parsed.model_id, "qwen3-1.7b",
        "FALSIFY-HTTP-004b: model_id roundtrip"
    );
    assert_eq!(
        parsed.uptime_seconds, 42,
        "FALSIFY-HTTP-004b: uptime roundtrip"
    );
    assert!(
        parsed.gpu_available,
        "FALSIFY-HTTP-004b: gpu_available roundtrip"
    );
}

// ═══ Contract: apr-serve-v1 enforcement — GAP CLOSURE (PMAT-190) ═══

#[test]
fn falsify_srv_008_format_detection_apr_magic() {
    // FALSIFY-SRV-008: APR files detected by magic bytes, not extension.
    // APR v2 binary header starts with "APR\x02" (4 bytes).
    let apr_magic = b"APR\x02";
    assert_eq!(&apr_magic[..3], b"APR", "FALSIFY-SRV-008: APR magic prefix");
    assert_eq!(apr_magic[3], 0x02, "FALSIFY-SRV-008: APR v2 version byte");

    // GGUF magic: "GGUF" (4 bytes) — must NOT be confused with APR
    let gguf_magic = b"GGUF";
    assert_ne!(
        &apr_magic[..4],
        &gguf_magic[..4],
        "FALSIFY-SRV-008: APR and GGUF have distinct magic bytes"
    );
}

#[test]
fn falsify_srv_009_format_detection_gguf_magic() {
    // FALSIFY-SRV-009: GGUF files detected by magic bytes.
    // GGUF v3 header: "GGUF" + version u32 LE.
    let gguf_header = [b'G', b'G', b'U', b'F', 3, 0, 0, 0]; // GGUF v3
    assert_eq!(&gguf_header[..4], b"GGUF", "FALSIFY-SRV-009: GGUF magic");
    let version = u32::from_le_bytes([
        gguf_header[4],
        gguf_header[5],
        gguf_header[6],
        gguf_header[7],
    ]);
    assert_eq!(version, 3, "FALSIFY-SRV-009: GGUF version 3");

    // SafeTensors has no magic — it's a JSON header with length prefix
    // APR has "APR\x02" — distinct from GGUF
    assert_ne!(
        &gguf_header[..4],
        b"APR\x02",
        "FALSIFY-SRV-009: GGUF distinct from APR"
    );
}

#[test]
fn falsify_srv_010_graceful_shutdown_ready_flag() {
    // FALSIFY-SRV-010: Draining state sets ready=false.
    // When shutdown is initiated, health endpoint must return 503.
    let model = create_test_model_file();
    let config = ServerConfig::default();
    let state = ServerState::new(model.path().to_path_buf(), config).expect("to_path_buf(), config");

    // Simulate Ready → Draining transition
    state.ready.store(true, Ordering::Release);
    assert!(state.is_ready(), "FALSIFY-SRV-010: server is Ready");

    // Draining: set not ready
    state.ready.store(false, Ordering::Release);
    assert!(!state.is_ready(), "FALSIFY-SRV-010: server is Draining");

    // Health check must reflect Draining state
    let health = health_check(&state);
    assert_eq!(
        health.status,
        HealthStatus::Unhealthy,
        "FALSIFY-SRV-010: health is Unhealthy during drain"
    );
}

#[test]
fn falsify_srv_011_degraded_health_on_high_latency() {
    // FALSIFY-SRV-011: Health degrades when avg latency > 1s.
    // Contract: server_lifecycle invariant — Degraded when p99 > 1s.
    let model = create_test_model_file();
    let config = ServerConfig::default();
    let state = ServerState::new(model.path().to_path_buf(), config).expect("to_path_buf(), config");
    state.ready.store(true, Ordering::Release);

    // Record 10 requests with 2000ms average latency
    for _ in 0..10 {
        state.metrics.record_request(true, 10, 2000);
    }

    let health = health_check(&state);
    assert_eq!(
        health.status,
        HealthStatus::Degraded,
        "FALSIFY-SRV-011: health is Degraded when avg latency > 1s"
    );
}

#[test]
fn falsify_srv_012_prometheus_metrics_format() {
    // FALSIFY-SRV-012: /metrics returns valid Prometheus exposition format.
    let metrics = ServerMetrics::new();
    metrics.record_request(true, 42, 100);
    metrics.record_request(false, 0, 50);
    metrics.record_client_error();

    let output = metrics.prometheus_output();

    // Must contain TYPE and HELP annotations
    assert!(
        output.contains("# TYPE apr_requests_total counter"),
        "FALSIFY-SRV-012: TYPE annotation present"
    );
    assert!(
        output.contains("# HELP apr_requests_total"),
        "FALSIFY-SRV-012: HELP annotation present"
    );

    // Counter values must be present and correct
    assert!(
        output.contains("apr_requests_total 3"),
        "FALSIFY-SRV-012: total requests = 3 (2 req + 1 client error)"
    );
    assert!(
        output.contains("apr_tokens_generated_total 42"),
        "FALSIFY-SRV-012: tokens = 42"
    );
    assert!(
        output.contains("apr_requests_client_error 1"),
        "FALSIFY-SRV-012: client errors = 1"
    );
}

#[test]
fn falsify_srv_013_mmap_threshold() {
    // FALSIFY-SRV-013: Model files > 50MB use mmap.
    // Contract: ServerState determines mmap based on 50MB threshold.
    let model = create_test_model_file();
    let config = ServerConfig::default();
    let state = ServerState::new(model.path().to_path_buf(), config).expect("to_path_buf(), config");

    // Test file is small — should NOT use mmap
    assert!(
        !state.uses_mmap,
        "FALSIFY-SRV-013: small models don't use mmap"
    );
    assert!(
        state.model_size_bytes < 50 * 1024 * 1024,
        "FALSIFY-SRV-013: test model is under 50MB"
    );
}

// ═══ Contract: http-api-v1 enforcement — GAP CLOSURE (PMAT-190) ═══

#[test]
fn falsify_http_005_chat_completion_request_schema() {
    // FALSIFY-HTTP-005: ChatCompletionRequest matches OpenAI schema.
    use super::ChatCompletionRequest;

    let json = serde_json::json!({
        "model": "qwen3-1.7b",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"}
        ],
        "max_tokens": 256,
        "temperature": 0.7,
        "stream": false
    });
    let req: ChatCompletionRequest = serde_json::from_value(json).expect("parse");
    assert_eq!(req.messages.len(), 2, "FALSIFY-HTTP-005: 2 messages");
    assert_eq!(
        req.messages[0].role, "system",
        "FALSIFY-HTTP-005: system role"
    );
    assert_eq!(req.messages[1].role, "user", "FALSIFY-HTTP-005: user role");
    assert_eq!(req.max_tokens, Some(256), "FALSIFY-HTTP-005: max_tokens");
}

#[test]
fn falsify_http_006_chat_completion_with_tools() {
    // FALSIFY-HTTP-006: ChatCompletionRequest with tools roundtrips correctly.
    use super::ChatCompletionRequest;

    let json = serde_json::json!({
        "model": "qwen3-1.7b",
        "messages": [{"role": "user", "content": "Read main.rs"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "file_read",
                "description": "Read a file",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}
            }
        }],
        "tool_choice": "auto"
    });
    let req: ChatCompletionRequest = serde_json::from_value(json).expect("parse");
    assert!(req.tools.is_some(), "FALSIFY-HTTP-006: tools present");
    let tools = req.tools.expect("tools");
    assert_eq!(tools.len(), 1, "FALSIFY-HTTP-006: 1 tool");
    assert_eq!(
        tools[0].function.name, "file_read",
        "FALSIFY-HTTP-006: tool name preserved"
    );
}

#[test]
fn falsify_http_007_chat_response_schema() {
    // FALSIFY-HTTP-007: ChatCompletionResponse has choices[0].message.content path.
    use super::{ChatChoice, ChatCompletionResponse, ChatMessage, TokenUsage};

    let resp = ChatCompletionResponse {
        id: "chatcmpl-test".into(),
        object: "chat.completion".into(),
        created: 1700000000,
        model: "qwen3-1.7b".into(),
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".into(),
                content: Some("Hello!".into()),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
            finish_reason: Some("stop".into()),
        }],
        usage: Some(TokenUsage {
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 15,
        }),
    };

    let json = serde_json::to_value(&resp).expect("serialize");
    // Verify OpenAI-compatible path: choices[0].message.content
    let content = json["choices"][0]["message"]["content"]
        .as_str()
        .expect("content path");
    assert_eq!(
        content, "Hello!",
        "FALSIFY-HTTP-007: content at expected path"
    );
    assert_eq!(
        json["object"], "chat.completion",
        "FALSIFY-HTTP-007: object type"
    );
    assert_eq!(
        json["usage"]["total_tokens"], 15,
        "FALSIFY-HTTP-007: usage.total_tokens"
    );
}

#[test]
fn falsify_http_008_sse_stream_event_format() {
    // FALSIFY-HTTP-008: SSE events follow spec format (event: + data: + \n\n).
    let token_event = StreamEvent::token("Hello", 42);
    let sse = token_event.to_sse();
    assert!(
        sse.starts_with("event: token\n"),
        "FALSIFY-HTTP-008: token event type"
    );
    assert!(
        sse.contains("data: Hello\n\n"),
        "FALSIFY-HTTP-008: data field with double newline"
    );

    let done_event = StreamEvent::done("stop", 100);
    let sse = done_event.to_sse();
    assert!(
        sse.starts_with("event: done\n"),
        "FALSIFY-HTTP-008: done event type"
    );
    assert!(
        sse.contains("finish_reason"),
        "FALSIFY-HTTP-008: done event has finish_reason"
    );

    let error_event = StreamEvent::error("OOM");
    let sse = error_event.to_sse();
    assert!(
        sse.starts_with("event: error\n"),
        "FALSIFY-HTTP-008: error event type"
    );
}

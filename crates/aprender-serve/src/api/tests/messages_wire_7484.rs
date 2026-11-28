//! aprender#7484 (G-01): wire falsifiers for `POST /v1/messages` on a quantized
//! (`apr serve run model.gguf`) router.

use axum::http::StatusCode;

use super::native_routes_2376::{post, quantized_state};

const SCHEMA: &str = include_str!("../anthropic_message_schema.json");

const OK_BODY: &str =
    r#"{"model":"m","max_tokens":2,"messages":[{"role":"user","content":"token5"}]}"#;

/// The streaming event sequence the Messages reference prescribes, in order.
const STREAM_EVENTS: [&str; 6] = [
    "message_start",
    "content_block_start",
    "content_block_delta",
    "content_block_stop",
    "message_delta",
    "message_stop",
];

fn json(body: &str) -> serde_json::Value {
    serde_json::from_str(body).unwrap_or_else(|e| panic!("not JSON ({e}): {body}"))
}

fn keys(value: &serde_json::Value) -> Vec<String> {
    let mut keys: Vec<String> = value
        .as_object()
        .expect("an object")
        .keys()
        .cloned()
        .collect();
    keys.sort();
    keys
}

fn schema_keys(name: &str) -> Vec<String> {
    serde_json::from_value(json(SCHEMA)[name].clone()).expect("a key list")
}

async fn send(body: &str) -> (StatusCode, String) {
    post(quantized_state(), "/v1/messages", body).await
}

/// `OK_BODY` with one more top-level field.
fn with_field(field: &str) -> String {
    format!("{{{field},{}", &OK_BODY[1..])
}

/// The Anthropic error envelope: `{"type":"error","error":{"type":...,"message":...}}`.
fn assert_error_envelope(resp: &str, kind: &str) {
    let body = json(resp);
    assert_eq!(body["type"], "error", "{resp}");
    assert_eq!(body["error"]["type"], kind, "{resp}");
    assert!(body["error"]["message"].is_string(), "{resp}");
}

/// The nested `error.message` text, for checks that name the offending field.
fn error_message(resp: &str) -> String {
    json(resp)["error"]["message"]
        .as_str()
        .unwrap_or_default()
        .to_string()
}

#[tokio::test]
async fn a_200_body_has_the_vendored_key_set() {
    let (status, resp) = send(OK_BODY).await;
    assert_eq!(status, StatusCode::OK, "{resp}");
    let body = json(&resp);
    assert_eq!(keys(&body), schema_keys("body_keys"));
    assert_eq!(keys(&body["usage"]), schema_keys("usage_keys"));
    assert_eq!(keys(&body["content"][0]), schema_keys("content_block_keys"));
    assert_eq!(body["type"], "message");
    assert_eq!(body["role"], "assistant");
    assert_eq!(body["content"][0]["type"], "text");
    // OK_BODY caps generation at 2 tokens, so the reply stops on the length limit.
    assert_eq!(body["stop_reason"], "max_tokens", "{resp}");
    assert!(body["usage"]["input_tokens"].is_u64(), "{resp}");
    assert!(body["usage"]["output_tokens"].is_u64(), "{resp}");
}

#[tokio::test]
async fn a_stream_emits_the_full_event_sequence_in_order() {
    let (status, resp) = send(&with_field(r#""stream":true"#)).await;
    assert_eq!(status, StatusCode::OK, "{resp}");
    let mut events: Vec<&str> = resp
        .lines()
        .filter_map(|l| l.strip_prefix("event: "))
        .collect();
    // One `content_block_delta` is emitted per text fragment; collapse the repeats.
    events.dedup();
    assert_eq!(events, STREAM_EVENTS, "{resp}");
}

/// aprender#2375: a temperature no sampler can serve must not reach one.
#[tokio::test]
async fn an_unservable_temperature_is_refused() {
    for t in ["-1", "1e40"] {
        let (status, resp) = send(&with_field(&format!("\"temperature\":{t}"))).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "temperature {t}: {resp}");
        assert_error_envelope(&resp, "invalid_request_error");
        assert!(error_message(&resp).contains("temperature must be"), "{resp}");
    }
}

/// A field this route cannot honour is refused by name, never silently dropped.
#[tokio::test]
async fn unsupported_fields_are_refused_by_name() {
    for (name, field) in [
        ("stop_sequences", r#""stop_sequences":["4"]"#),
        ("tools", r#""tools":[{"name":"f","input_schema":{}}]"#),
        ("tool_choice", r#""tool_choice":{"type":"any"}"#),
        (
            "thinking",
            r#""thinking":{"type":"enabled","budget_tokens":1024}"#,
        ),
    ] {
        let (status, resp) = send(&with_field(field)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{name}: {resp}");
        assert_error_envelope(&resp, "invalid_request_error");
        assert!(error_message(&resp).contains(name), "{name}: {resp}");
    }
}

/// Fields that ask for nothing are accepted.
#[tokio::test]
async fn empty_unsupported_fields_are_accepted() {
    for field in [
        r#""stop_sequences":[]"#,
        r#""tools":[]"#,
        r#""thinking":{"type":"disabled"}"#,
    ] {
        let (status, resp) = send(&with_field(field)).await;
        assert_eq!(status, StatusCode::OK, "{field}: {resp}");
    }
}

/// aprender#2376(7): the parser's cursor never reaches a client.
#[tokio::test]
async fn a_parse_error_leaks_no_parser_internals() {
    for body in ["{not json", "{\"model\":"] {
        let (status, resp) = send(body).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{resp}");
        assert_error_envelope(&resp, "invalid_request_error");
        assert!(
            !resp.contains("at line") && !resp.contains("EOF while"),
            "{resp}"
        );
    }
}

/// `model` is required: a made-up name would be echoed back, or 404 on a registry server.
#[tokio::test]
async fn a_missing_or_empty_model_is_refused_by_name() {
    for body in [
        r#"{"max_tokens":2,"messages":[{"role":"user","content":"token5"}]}"#,
        r#"{"model":"","max_tokens":2,"messages":[{"role":"user","content":"token5"}]}"#,
    ] {
        let (status, resp) = send(body).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{resp}");
        assert_error_envelope(&resp, "invalid_request_error");
        assert!(error_message(&resp).contains("model: "), "{resp}");
    }
}

/// A `top_p` no sampler can serve is the client's error (400), never a 500 or a 200.
#[tokio::test]
async fn an_unservable_top_p_is_refused() {
    for p in ["0", "-0.5", "1.5"] {
        let (status, resp) = send(&with_field(&format!("\"top_p\":{p}"))).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "top_p {p}: {resp}");
        assert_error_envelope(&resp, "invalid_request_error");
        assert!(error_message(&resp).contains("top_p: "), "{resp}");
    }
}

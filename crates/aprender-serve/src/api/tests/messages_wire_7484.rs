//! aprender#7484 (G-01): wire falsifiers for `POST /v1/messages` on a quantized
//! (`apr serve run model.gguf`) router.

use axum::http::StatusCode;

use super::native_routes_2376::{post, quantized_state};

const SCHEMA: &str = include_str!("../anthropic_message_schema.json");

const OK_BODY: &str =
    r#"{"model":"m","max_tokens":2,"messages":[{"role":"user","content":"token5"}]}"#;

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
    assert!(body["stop_reason"].is_string(), "{resp}");
    assert!(body["usage"]["input_tokens"].is_u64(), "{resp}");
    assert!(body["usage"]["output_tokens"].is_u64(), "{resp}");
}

#[tokio::test]
async fn a_stream_starts_with_message_start_and_ends_with_message_stop() {
    let (status, resp) = send(&with_field(r#""stream":true"#)).await;
    assert_eq!(status, StatusCode::OK, "{resp}");
    let events: Vec<&str> = resp
        .lines()
        .filter_map(|l| l.strip_prefix("event: "))
        .collect();
    assert_eq!(events.first(), Some(&"message_start"), "{resp}");
    assert_eq!(events.last(), Some(&"message_stop"), "{resp}");
}

/// aprender#2375: a temperature no sampler can serve must not reach one.
#[tokio::test]
async fn an_unservable_temperature_is_refused() {
    for t in ["-1", "1e40"] {
        let (status, resp) = send(&with_field(&format!("\"temperature\":{t}"))).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "temperature {t}: {resp}");
        assert_eq!(json(&resp)["type"], "error", "{resp}");
        assert!(resp.contains("temperature must be"), "{resp}");
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
        assert_eq!(json(&resp)["type"], "error", "{resp}");
        assert!(resp.contains(name), "{name}: {resp}");
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
        assert_eq!(json(&resp)["type"], "error", "{resp}");
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
        assert_eq!(json(&resp)["type"], "error", "{resp}");
        assert!(resp.contains("model: "), "{resp}");
    }
}

/// A `top_p` no sampler can serve is the client's error (400), never a 500 or a 200.
#[tokio::test]
async fn an_unservable_top_p_is_refused() {
    for p in ["0", "-0.5", "1.5"] {
        let (status, resp) = send(&with_field(&format!("\"top_p\":{p}"))).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "top_p {p}: {resp}");
        assert_eq!(json(&resp)["type"], "error", "{resp}");
        assert!(resp.contains("top_p: "), "{resp}");
    }
}

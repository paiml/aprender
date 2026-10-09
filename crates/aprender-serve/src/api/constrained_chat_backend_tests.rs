//! #3568 PR 4: `response_format` on /v1/chat/completions. A constrained request is answered
//! through the one engine and read again by the second reader, or refused by name with its
//! status before any model work; one that asks for no constraint takes the chain unchanged.

use super::*;
use crate::api::{create_router, ResponseFormat};
use crate::constrain::{ConstraintError, ConstraintRequest};
use crate::session::ConstrainedStop;
use axum::body::Body;
use axum::http::Request;
use tower::ServiceExt;

/// A chat request with `extra` merged over a one-message body.
fn chat_request(extra: serde_json::Value) -> ChatCompletionRequest {
    let mut body = serde_json::json!({
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
    });
    for (k, v) in extra.as_object().expect("an object") {
        body[k] = v.clone();
    }
    serde_json::from_value(body).expect("a chat request")
}

fn format(value: serde_json::Value) -> ResponseFormat {
    serde_json::from_value(value).expect("a response_format")
}

async fn post(state: AppState, body: serde_json::Value) -> (StatusCode, serde_json::Value) {
    let response = create_router(state)
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .expect("request"),
        )
        .await
        .expect("the router answers");
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body");
    let json = serde_json::from_slice(&bytes)
        .unwrap_or_else(|_| serde_json::Value::String(String::from_utf8_lossy(&bytes).into()));
    (status, json)
}

// ---------------------------------------------------------------------------------------------
// The status map: a request's fault is 4xx, the server's 5xx, keyed on the variant
// ---------------------------------------------------------------------------------------------

#[test]
fn every_refusal_answers_with_its_status_and_type() {
    let cases = [
        (
            ConstraintError::SchemaInvalid("x".into()),
            400,
            "schema_invalid",
        ),
        (
            ConstraintError::SchemaUnsupported("x".into()),
            400,
            "schema_unsupported",
        ),
        (unsupported_path("p", "r"), 400, "schema_unsupported_path"),
        (
            ConstraintError::WithThinking("x".into()),
            400,
            "schema_with_thinking",
        ),
        (
            ConstraintError::Truncated { max_tokens: 4 },
            422,
            "constraint_truncated",
        ),
        (
            ConstraintError::Violation("x".into()),
            500,
            "schema_violation",
        ),
        (
            ConstraintError::DeadEnd {
                position: 0,
                reason: "x".into(),
            },
            500,
            "constraint_dead_end",
        ),
        (
            ConstraintError::Rejected {
                token: 1,
                position: 0,
                reason: "x".into(),
            },
            500,
            "constraint_rejected",
        ),
        (ConstraintError::Vocab("x".into()), 500, "constraint_vocab"),
        (
            ConstraintError::NotCompiled,
            501,
            "structured_output_not_compiled",
        ),
    ];
    for (e, status, kind) in cases {
        let want = (StatusCode::from_u16(status).expect("a status"), kind);
        assert_eq!(constraint_status(&e), want, "{e}");
    }
}

#[tokio::test]
async fn a_refusal_body_names_the_error_and_its_type_and_counts_a_failure() {
    let state = AppState::demo().expect("demo state");
    let before = state.metrics.snapshot().failed_requests;
    let e = unsupported_path("serve-stream", "send stream: false");
    let response = constraint_refusal(&state, &e);
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body");
    let body: serde_json::Value = serde_json::from_slice(&bytes).expect("json");
    assert_eq!(body["type"], "schema_unsupported_path");
    assert_eq!(body["error"], e.to_string());
    assert_eq!(state.metrics.snapshot().failed_requests, before + 1);
}

// ---------------------------------------------------------------------------------------------
// `response_format` as OpenAI sends it
// ---------------------------------------------------------------------------------------------

#[test]
fn response_format_maps_to_the_constraint_it_asks_for() {
    let schema = serde_json::json!({"type": "boolean"});
    let cases = [
        (serde_json::json!({"type": "text"}), None),
        (
            serde_json::json!({"type": "json_object"}),
            Some(ConstraintRequest::JsonSchema(
                serde_json::json!({"type": "object"}),
            )),
        ),
        (
            serde_json::json!({"type": "json_schema", "json_schema": {"name": "b", "strict": true, "schema": schema}}),
            Some(ConstraintRequest::JsonSchema(schema.clone())),
        ),
        (
            serde_json::json!({"type": "json_schema", "json_schema": {"schema": true}}),
            Some(ConstraintRequest::JsonSchema(serde_json::json!(true))),
        ),
    ];
    for (value, want) in cases {
        let got = format(value.clone()).constraint().expect("a constraint");
        assert_eq!(got, want, "{value}");
    }
}

#[test]
fn a_json_schema_format_without_a_schema_object_is_schema_invalid() {
    for json_schema in [
        serde_json::json!({"name": "nothing"}),
        serde_json::json!({"schema": 12}),
        serde_json::json!({"schema": "{\"type\": \"object\"}"}),
    ] {
        let f = format(serde_json::json!({"type": "json_schema", "json_schema": json_schema}));
        assert!(
            matches!(f.constraint(), Err(ConstraintError::SchemaInvalid(_))),
            "{json_schema}"
        );
    }
}

/// A `type` apr does not know is refused when the request is read, never dropped: dropping it
/// would answer an unconstrained turn to a client that asked for a constrained one.
#[test]
fn a_response_format_type_apr_does_not_know_is_refused_not_dropped() {
    for value in [
        serde_json::json!({"type": "xml"}),
        serde_json::json!({}),
        serde_json::json!({"type": "json_schema"}),
    ] {
        assert!(
            serde_json::from_value::<ResponseFormat>(value.clone()).is_err(),
            "{value} was read"
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Refusals before any model work
// ---------------------------------------------------------------------------------------------

#[test]
fn what_a_constraint_cannot_honour_is_refused_by_name() {
    let cases = [
        (serde_json::json!({"stream": true}), false, "serve-stream"),
        (
            serde_json::json!({"tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}]}),
            false,
            "serve-tools",
        ),
        (serde_json::json!({"stop": ["}"]}), false, "serve-stop"),
        (
            serde_json::json!({"ignore_eos": true}),
            false,
            "serve-ignore-eos",
        ),
        (serde_json::json!({}), true, "serve-trace"),
    ];
    for (extra, traced, want) in cases {
        let request = chat_request(extra.clone());
        match refuse_unconstrainable(&request, traced) {
            Err(ConstraintError::UnsupportedPath { path, .. }) => assert_eq!(path, want, "{extra}"),
            other => panic!("{extra} traced={traced}: {other:?}"),
        }
    }
}

#[test]
fn thinking_on_is_schema_with_thinking() {
    let request = chat_request(serde_json::json!({"think": true}));
    assert!(matches!(
        refuse_unconstrainable(&request, false),
        Err(ConstraintError::WithThinking(_))
    ));
}

/// The planted control for the table above: the same fields set to what OpenAI reads as
/// "none" refuse nothing, so the refusals are keyed on the request, not on the field's
/// presence.
#[test]
fn the_same_fields_set_to_none_refuse_nothing() {
    let request = chat_request(serde_json::json!({
        "stream": false,
        "tools": [],
        "stop": [],
        "ignore_eos": false,
        "think": false,
    }));
    assert_eq!(refuse_unconstrainable(&request, false), Ok(()));
}

/// No `response_format`, or `text`, is no constraint: the chain answers it unchanged, even
/// with every field a constraint would refuse.
#[test]
fn a_request_without_a_constraint_takes_the_chain_unchanged() {
    for extra in [
        serde_json::json!({"stream": true, "stop": ["x"], "ignore_eos": true}),
        serde_json::json!({"stream": true, "response_format": {"type": "text"}}),
    ] {
        let request = chat_request(extra.clone());
        assert_eq!(requested_constraint(&request, true), Ok(None), "{extra}");
    }
}

/// The refusals come before the schema is compiled, so they are the same in every build.
#[test]
fn a_refusal_precedes_the_schema_check() {
    let request = chat_request(serde_json::json!({
        "stream": true,
        "response_format": {"type": "json_schema", "json_schema": {"schema": {"type": 12}}},
    }));
    assert!(matches!(
        requested_constraint(&request, false),
        Err(ConstraintError::UnsupportedPath { path, .. }) if path == "serve-stream"
    ));
}

#[cfg(feature = "structured-output")]
#[test]
fn a_schema_the_second_reader_cannot_read_is_refused_before_the_model() {
    let malformed = chat_request(serde_json::json!({
        "response_format": {"type": "json_schema", "json_schema": {"schema": {"type": 12}}},
    }));
    assert!(matches!(
        requested_constraint(&malformed, false),
        Err(ConstraintError::SchemaInvalid(_))
    ));
    // The schema comes off the network: a `$ref` outside it is refused, never fetched
    for target in ["http://127.0.0.1:9/x.json", "file:///etc/passwd"] {
        let request = chat_request(serde_json::json!({
            "response_format": {"type": "json_schema", "json_schema": {"schema": {"$ref": target}}},
        }));
        assert!(
            matches!(
                requested_constraint(&request, false),
                Err(ConstraintError::SchemaUnsupported(_))
            ),
            "{target}"
        );
    }
}

#[cfg(not(feature = "structured-output"))]
#[tokio::test]
async fn a_build_without_structured_output_answers_501_by_name() {
    let state = AppState::demo().expect("demo state");
    let (status, body) = post(
        state,
        serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {"type": "json_object"},
        }),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED, "{body}");
    assert_eq!(body["type"], "structured_output_not_compiled");
}

/// Over HTTP, on a server whose chain would stream a demo reply: the refusal is answered and
/// the model never runs (no 200, no stream).
#[tokio::test]
async fn a_streamed_constrained_request_is_400_over_http() {
    let state = AppState::demo().expect("demo state");
    let (status, body) = post(
        state,
        serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true,
            "response_format": {"type": "json_object"},
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["type"], "schema_unsupported_path");
}

// ---------------------------------------------------------------------------------------------
// The route: the chain's own backend when it runs the engine, else that backend's refusal
// ---------------------------------------------------------------------------------------------

/// A Q4_K llama whose 32 tokens can spell small JSON documents, end-of-sequence 1.
fn json_model() -> (
    tempfile::NamedTempFile,
    crate::gguf::MappedGGUFModel,
    Vec<String>,
) {
    use crate::gguf::test_factory::build_minimal_llama_gguf_with;
    use std::io::Write;
    let named = [
        "<unk>", "</s>", "{", "}", "\"", ":", ",", "[", "]", "0", "1", "2", "3", "4", "5", "6",
        "7", "8", "9", "true", "false", "null", "a", "b", "-", ".", "e",
    ];
    let vocab: Vec<String> = named
        .into_iter()
        .map(String::from)
        .chain((named.len()..32).map(|i| format!("x{i}")))
        .collect();
    let tokens: Vec<&str> = vocab.iter().map(String::as_str).collect();
    // intermediate = hidden: the builder sizes each Q4_K tensor from its first dim as rows
    let bytes = build_minimal_llama_gguf_with(32, 64, 64, 4, 4, |b| {
        b.add_string("tokenizer.ggml.model", "llama")
            .add_string_array("tokenizer.ggml.tokens", &tokens)
            .add_u32("tokenizer.ggml.eos_token_id", 1)
    });
    let mut f = tempfile::NamedTempFile::with_suffix(".gguf").expect("temp");
    f.write_all(&bytes).expect("write");
    let mapped = crate::gguf::MappedGGUFModel::from_path(f.path()).expect("map");
    (f, mapped, vocab)
}

/// The dense CPU server `apr serve` builds for a GGUF: the model and the mapped file.
fn cpu_state(retain_gguf: bool) -> (tempfile::NamedTempFile, AppState) {
    let (f, mapped, vocab) = json_model();
    let model = crate::gguf::OwnedQuantizedModel::from_mapped(&mapped).expect("load");
    let state = AppState::with_quantized_model_and_vocab(model, vocab).expect("app state");
    let state = if retain_gguf {
        state.with_mapped_gguf_model(Arc::new(mapped))
    } else {
        state
    };
    (f, state)
}

#[test]
fn the_route_is_the_chains_own_backend() {
    let (_f, state) = cpu_state(true);
    assert!(matches!(
        constrained_route(&state),
        Ok(ConstrainedRoute::Cpu(_))
    ));
    // The MoE loop comes before the dense CPU model in the chain, and applies no constraint
    let moe = state.with_architecture("qwen3_moe");
    assert!(matches!(
        constrained_route(&moe),
        Err(ConstraintError::UnsupportedPath { path, .. }) if path == "serve-qwen3-moe"
    ));
    let demo = AppState::demo().expect("demo state");
    assert!(matches!(
        constrained_route(&demo),
        Err(ConstraintError::UnsupportedPath { path, .. }) if path == "serve-registry"
    ));
}

// ---------------------------------------------------------------------------------------------
// The verdict: the budget's cut, an empty reply and the second reader
// ---------------------------------------------------------------------------------------------

#[test]
fn a_reply_the_budget_cut_is_truncated_never_returned() {
    let c = ConstraintRequest::Lark(r#"start: "a""#.to_string());
    assert_eq!(
        constrained_verdict(&c, ConstrainedStop::Length, "a", 7),
        Err(ConstraintError::Truncated { max_tokens: 7 })
    );
    assert!(matches!(
        constrained_verdict(&c, ConstrainedStop::Complete, "  ", 1),
        Err(ConstraintError::Violation(_))
    ));
    assert_eq!(
        constrained_verdict(&c, ConstrainedStop::Complete, "a", 1),
        Ok(())
    );
}

/// The second reader is a reader: a complete reply it refuses does not ship, whatever the
/// engine said. Planted: the engine's "complete" on a document the schema refuses.
#[cfg(feature = "structured-output")]
#[test]
fn a_complete_reply_the_second_reader_refuses_does_not_ship() {
    let c = ConstraintRequest::JsonSchema(serde_json::json!({"type": "boolean"}));
    assert_eq!(
        constrained_verdict(&c, ConstrainedStop::Complete, "true", 1),
        Ok(())
    );
    for planted in ["1", "\"true\"", "true false"] {
        assert!(
            matches!(
                constrained_verdict(&c, ConstrainedStop::Complete, planted, 1),
                Err(ConstraintError::Violation(_))
            ),
            "{planted} shipped"
        );
    }
}

// ---------------------------------------------------------------------------------------------
// End to end over HTTP on the dense CPU route
// ---------------------------------------------------------------------------------------------

#[cfg(feature = "structured-output")]
fn constrained_body(schema: serde_json::Value, max_tokens: usize) -> serde_json::Value {
    serde_json::json!({
        "model": "m",
        "messages": [{"role": "user", "content": "yes or no"}],
        "temperature": 0.0,
        "top_k": 1,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_schema", "json_schema": {"name": "s", "schema": schema}},
    })
}

/// The reply is the document the schema allows, verbatim, finished "stop", and it passes the
/// second reader the server ran.
#[cfg(feature = "structured-output")]
#[tokio::test(flavor = "multi_thread")]
async fn a_constrained_chat_answers_the_document_the_schema_allows() {
    let (_f, state) = cpu_state(true);
    let schema = serde_json::json!({"type": "boolean"});
    let (status, body) = post(state, constrained_body(schema.clone(), 8)).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let content = body["choices"][0]["message"]["content"]
        .as_str()
        .expect("content");
    assert!(matches!(content, "true" | "false"), "{content:?}");
    assert_eq!(crate::constrain::second_reader(&schema, content), Ok(()));
    assert_eq!(body["choices"][0]["finish_reason"], "stop");
    assert_eq!(body["usage"]["completion_tokens"], 1, "{body}");
    assert_eq!(body["used_gpu"], false);
}

/// A budget that cannot hold the document: 422, and no partial text in the body.
#[cfg(feature = "structured-output")]
#[tokio::test(flavor = "multi_thread")]
async fn a_budget_too_small_for_the_document_is_422_with_no_partial_text() {
    let (_f, state) = cpu_state(true);
    let schema = serde_json::json!({"type": "array", "items": {"type": "integer"}});
    let (status, body) = post(state, constrained_body(schema, 1)).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
    assert_eq!(body["type"], "constraint_truncated");
    assert!(body.get("choices").is_none(), "{body}");
}

/// A server that retains no GGUF has no vocabulary to compile against: refused by name, never
/// answered unconstrained.
#[cfg(feature = "structured-output")]
#[tokio::test(flavor = "multi_thread")]
async fn a_server_without_the_gguf_refuses_by_name() {
    let (_f, state) = cpu_state(false);
    let (status, body) = post(
        state,
        constrained_body(serde_json::json!({"type": "boolean"}), 8),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["type"], "schema_unsupported_path");
    assert!(
        body["error"]
            .as_str()
            .is_some_and(|e| e.contains("serve-without-gguf")),
        "{body}"
    );
}

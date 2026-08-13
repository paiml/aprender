//! Falsifiers for aprender#2375 finding 4 — `POST /v1/chat/completions/stream`.
//!
//! The route is registered unconditionally next to `/v1/chat/completions`
//! (`router.rs`) and 0.63.0 answered it with
//! `{"error":"Model registry error: No model available"}` HTTP 404 on a server
//! whose `/health` said `model_loaded:true` and whose `/v1/chat/completions`
//! returned a full completion in the same second. Cause: the handler resolved
//! ONLY the dense f32 `Model` via `AppState::get_model`, which is `None` for
//! every `apr serve run model.gguf` — the weights live in `quantized_model` —
//! and it mapped *every* resolution failure to 404.
//!
//! Two client-observable claims are asserted here:
//!
//! 1. On the standard quantized deployment the route STREAMS (200,
//!    `text/event-stream`, `[DONE]`-terminated) instead of 404ing, and it
//!    delivers the same text as `/v1/chat/completions` with `"stream":true`.
//!    Two chat routes on one server must not disagree about what the model
//!    said.
//! 2. When the server genuinely has no model, the answer is 503 — the status
//!    `/v1/predict` and `/v1/gpu/warmup` already use for that condition, and
//!    the one `model_resolution_status` mandates (aprender#2376 finding 5).
//!    404 told a client "this route does not exist" about a route that does.

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use tower::util::ServiceExt;

use crate::api::{create_router, AppState};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async fn send(state: AppState, uri: &str, json: &str) -> axum::response::Response {
    create_router(state)
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(uri)
                .header("content-type", "application/json")
                .body(Body::from(json.to_string()))
                .expect("build request"),
        )
        .await
        .expect("dispatch")
}

fn content_type(response: &axum::response::Response) -> String {
    response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_string()
}

async fn body_text(response: axum::response::Response) -> String {
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    String::from_utf8_lossy(&bytes).into_owned()
}

/// Concatenate `choices[0].delta.content` across an SSE body, the way every
/// OpenAI SDK reassembles a stream.
fn streamed_text(body: &str) -> String {
    body.lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|payload| payload.trim() != "[DONE]")
        .filter_map(|payload| serde_json::from_str::<serde_json::Value>(payload).ok())
        .filter_map(|frame| {
            frame["choices"][0]["delta"]["content"]
                .as_str()
                .map(str::to_string)
        })
        .collect()
}

/// A request that pins sampling to greedy so the two routes are comparable.
/// `GREEDY_BODY` closes it; `GREEDY_STREAMING_BODY` adds the explicit flag the
/// `/v1/chat/completions` form needs.
const GREEDY_FIELDS: &str = r#"{"model":"default","messages":[{"role":"user","content":"token5 token6"}],"max_tokens":4,"temperature":0.0"#;

fn greedy_body() -> String {
    format!("{GREEDY_FIELDS}}}")
}

fn greedy_streaming_body() -> String {
    format!("{GREEDY_FIELDS},\"stream\":true}}")
}

// ---------------------------------------------------------------------------
// Claim 1: the route serves the deployment `apr serve run model.gguf` builds
// ---------------------------------------------------------------------------

#[cfg(feature = "gpu")]
#[tokio::test]
async fn chat_completions_stream_serves_a_quantized_server() {
    use super::native_routes_2376::quantized_state;

    let response = send(
        quantized_state(),
        "/v1/chat/completions/stream",
        &greedy_body(),
    )
    .await;

    let status = response.status();
    let ctype = content_type(&response);
    let body = body_text(response).await;

    assert_eq!(
        status,
        StatusCode::OK,
        "a quantized server serves /v1/chat/completions; the /stream sibling \
         answered {status} with {body}"
    );
    assert!(
        ctype.starts_with("text/event-stream"),
        "a streaming route must be SSE-framed, got {ctype:?}"
    );
    assert!(
        body.trim_end().ends_with("data: [DONE]"),
        "an OpenAI stream terminates with the [DONE] sentinel; got:\n{body}"
    );
}

/// The two chat routes on one server must not disagree about the model output.
///
/// Pre-fix this could not even be compared: `/v1/chat/completions` answered 200
/// with text while `/v1/chat/completions/stream` answered 404 with an error
/// envelope.
#[cfg(feature = "gpu")]
#[tokio::test]
async fn stream_route_and_stream_flag_deliver_the_same_text() {
    use super::native_routes_2376::quantized_state;

    let via_flag = body_text(
        send(
            quantized_state(),
            "/v1/chat/completions",
            &greedy_streaming_body(),
        )
        .await,
    )
    .await;
    let via_route = body_text(
        send(
            quantized_state(),
            "/v1/chat/completions/stream",
            &greedy_body(),
        )
        .await,
    )
    .await;

    let expected = streamed_text(&via_flag);
    assert!(
        via_flag.trim_end().ends_with("data: [DONE]"),
        "control: the flag form must stream:\n{via_flag}"
    );
    assert_eq!(
        streamed_text(&via_route),
        expected,
        "the /stream route delivered different text than the stream flag on the \
         same server and request:\nroute body:\n{via_route}\nflag body:\n{via_flag}"
    );
}

// ---------------------------------------------------------------------------
// Non-regression: the dense backend must survive the same request
// ---------------------------------------------------------------------------

/// `temperature: 0` — the canonical OpenAI deterministic request — must be
/// served on the dense `Model` path by BOTH chat routes.
///
/// The `/stream` handler carried PMAT-790's `temperature == 0` normalization
/// and `registry_fallback` (the dense backend of `/v1/chat/completions`) did
/// not, so the same request was 200 on one route and 500
/// ("Temperature must be a positive finite number") on the other. Folding the
/// routes together without folding the config resolvers together would have
/// converted the working route into the broken one.
#[tokio::test]
async fn temperature_zero_is_served_on_both_chat_routes() {
    for uri in ["/v1/chat/completions", "/v1/chat/completions/stream"] {
        let state = AppState::demo().expect("dense demo AppState");
        let response = send(state, uri, &greedy_streaming_body()).await;
        let status = response.status();
        let body = body_text(response).await;

        assert_eq!(
            status,
            StatusCode::OK,
            "{uri} refused temperature 0 with {status}: {body}"
        );
        assert!(
            body.trim_end().ends_with("data: [DONE]"),
            "{uri} must stream to completion; got:\n{body}"
        );
    }
}

// ---------------------------------------------------------------------------
// Claim 2: "no model loaded" is a server condition — 503, never 404
// ---------------------------------------------------------------------------

#[tokio::test]
async fn stream_route_with_no_model_is_503_not_404() {
    let state = AppState::demo_mock().expect("model-less AppState");
    let response = send(state, "/v1/chat/completions/stream", &greedy_body()).await;
    let status = response.status();
    let body = body_text(response).await;

    assert_eq!(
        status,
        StatusCode::SERVICE_UNAVAILABLE,
        "a mounted route with nothing to serve is 503 (what /v1/predict and \
         /v1/gpu/warmup answer); 404 claims the route does not exist. Got \
         {status} with {body}"
    );
}

/// An unknown model NAME is still a client error: 404 must survive for the case
/// it is actually correct for, or the fix above would have traded one wrong
/// status for another.
#[tokio::test]
async fn stream_route_with_an_unknown_model_name_is_still_404() {
    let mut state = AppState::demo_mock().expect("model-less AppState");
    state.registry = Some(std::sync::Arc::new(crate::registry::ModelRegistry::new(1)));

    let response = send(
        state,
        "/v1/chat/completions/stream",
        r#"{"model":"no-such-model","messages":[{"role":"user","content":"token5"}],"max_tokens":2,"temperature":0.0}"#,
    )
    .await;
    let status = response.status();
    let body = body_text(response).await;

    assert_eq!(
        status,
        StatusCode::NOT_FOUND,
        "the client named a model this server does not have; that is 404. Got \
         {status} with {body}"
    );
}

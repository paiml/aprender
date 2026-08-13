//! Falsifiers for the #2375 close-out: the streaming chat surface and
//! `/v1/metrics`.
//!
//! Every test here drives the REAL router — `create_router(state)` — because
//! each defect below lives in the difference between calling a handler and
//! serving a request:
//!
//! - **#2375(1), regression.** The whitespace falsifier for the SSE deltas
//!   (`sse_stream_whitespace.rs`) calls `true_streaming_sse_response` directly,
//!   so it kept passing while the router served an event stream with **zero**
//!   content deltas: `cancel_on_disconnect` dropped its `CancelOnDrop` guard as
//!   soon as the handler returned, and a streaming handler returns *before* its
//!   background decode loop has emitted anything. A guard that never scans the
//!   surface where the request is actually served cannot see this.
//! - **#2375(4).** `POST /v1/chat/completions/stream` answered
//!   `404 "Model registry error: No model available"` on every
//!   `apr serve run model.gguf`, because it resolved the dense f32 `Model`
//!   that a quantized deployment does not have.
//! - **temperature 0.** The OpenAI-canonical deterministic request answered
//!   HTTP 500 on the dense backends of `/v1/chat/completions` and
//!   `/v1/completions`.
//! - **#2375(7).** `/v1/metrics` reported `latency_p50/p95/p99 = 0.0` while
//!   `/metrics` on the same process reported a non-zero average over the same
//!   requests, and `model_name` was a constant.

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use tower::util::ServiceExt;

use crate::api::{create_router, AppState};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// A quantized-only server: exactly what `apr serve run model.gguf` builds — a
/// tokenizer plus `quantized_model`, and NO dense f32 `Model`.
#[cfg(feature = "gpu")]
fn quantized_state() -> AppState {
    super::native_routes_2376::quantized_state()
}

async fn send(state: AppState, uri: &str, json: &str) -> (StatusCode, String, String) {
    let response = create_router(state)
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(uri)
                .header("content-type", "application/json")
                .body(Body::from(json.to_string()))
                .expect("build request"),
        )
        .await
        .expect("dispatch");
    let status = response.status();
    let content_type = response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_string();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    (
        status,
        content_type,
        String::from_utf8_lossy(&bytes).into_owned(),
    )
}

async fn get(state: AppState, uri: &str) -> (StatusCode, String) {
    let response = create_router(state)
        .oneshot(
            Request::builder()
                .uri(uri)
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("dispatch");
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    (status, String::from_utf8_lossy(&bytes).into_owned())
}

/// Concatenate `choices[0].delta.content` across an SSE body — what every
/// OpenAI SDK does to reconstruct the message.
fn concat_deltas(sse_body: &str) -> String {
    sse_body
        .lines()
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

/// The non-streamed `choices[0].message.content` for the same request.
async fn buffered_chat_content(state: AppState, request_json: &str) -> String {
    let (status, _, body) = send(state, "/v1/chat/completions", request_json).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "the non-streaming control must succeed, or the streamed comparison below \
         proves nothing: {body}"
    );
    let json: serde_json::Value = serde_json::from_str(&body).expect("chat completion is JSON");
    json["choices"][0]["message"]["content"]
        .as_str()
        .expect("non-streaming content")
        .to_string()
}

// ---------------------------------------------------------------------------
// #2375(1) regression — a streamed reply must carry the reply
// ---------------------------------------------------------------------------

const CHAT_REQUEST: &str =
    r#"{"model":"default","messages":[{"role":"user","content":"token5 token6"}],"max_tokens":6"#;

/// Streaming and non-streaming views of the same request must carry the same
/// text — through the ROUTER, with the cancellation layer mounted.
///
/// Observed before the fix: `STREAMCAT=""` against `NONSTREAM=
/// "token5token21token9token21token34token25"`. The stream was well-formed
/// (opening chunk, terminal chunk, `[DONE]`) and completely empty of content,
/// because the per-request `CancelOnDrop` guard fired the moment the handler
/// returned the SSE response and the decode loop observed it on its first poll.
#[tokio::test]
#[cfg(feature = "gpu")]
async fn streamed_chat_body_carries_the_same_text_as_the_buffered_one() {
    let expected = buffered_chat_content(quantized_state(), &format!("{CHAT_REQUEST}}}")).await;
    assert!(
        !expected.is_empty(),
        "the fixture must generate SOME text, or an empty stream would match it"
    );

    let (status, content_type, body) = send(
        quantized_state(),
        "/v1/chat/completions",
        &format!("{CHAT_REQUEST},\"stream\":true}}"),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert!(
        content_type.starts_with("text/event-stream"),
        "stream:true must be framed as SSE, got {content_type:?}"
    );
    assert_eq!(
        concat_deltas(&body),
        expected,
        "the concatenated SSE deltas must reproduce the buffered message; an empty \
         result is the cancellation guard stopping the decode loop before its first \
         token (#2375(1))"
    );
}

/// The same property over a REAL socket, served by hyper.
///
/// `tower::ServiceExt::oneshot` drives the router directly, so it can be told
/// that it only proves something about the tower stack. This binds a port,
/// serves the router with `axum::serve`, and reads the event stream off TCP —
/// the transport `apr serve run` uses. It fails the same way when the guard is
/// re-armed, which is what makes the cheaper test above trustworthy.
#[tokio::test(flavor = "multi_thread")]
#[cfg(feature = "gpu")]
async fn streamed_chat_body_carries_the_text_over_a_real_socket() {
    let expected = buffered_chat_content(quantized_state(), &format!("{CHAT_REQUEST}}}")).await;
    assert!(!expected.is_empty(), "the fixture must generate SOME text");

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind an ephemeral port");
    let addr = listener.local_addr().expect("local addr");
    let server = tokio::spawn(async move {
        axum::serve(listener, create_router(quantized_state()).into_make_service())
            .await
            .ok();
    });

    let body = reqwest::Client::new()
        .post(format!("http://{addr}/v1/chat/completions"))
        .header("content-type", "application/json")
        .body(format!("{CHAT_REQUEST},\"stream\":true}}"))
        .send()
        .await
        .expect("HTTP request")
        .text()
        .await
        .expect("read the event stream");
    server.abort();

    assert_eq!(
        concat_deltas(&body),
        expected,
        "over a real socket the streamed deltas must reproduce the buffered message; \
         0.63.0-era behaviour with the guard armed on completion delivered an empty \
         stream here too"
    );
}

// ---------------------------------------------------------------------------
// #2375(4) — /v1/chat/completions/stream must serve the standard deployment
// ---------------------------------------------------------------------------

/// The route is mounted and advertised on every server; it must answer on the
/// one deployment `apr serve run` actually produces.
#[tokio::test]
#[cfg(feature = "gpu")]
async fn chat_completions_stream_route_serves_a_quantized_server() {
    let expected = buffered_chat_content(quantized_state(), &format!("{CHAT_REQUEST}}}")).await;

    let (status, content_type, body) = send(
        quantized_state(),
        "/v1/chat/completions/stream",
        &format!("{CHAT_REQUEST}}}"),
    )
    .await;

    assert_eq!(
        status,
        StatusCode::OK,
        "0.63.0 answered 404 \"No model available\" here while /v1/chat/completions \
         on the same server returned text: {body}"
    );
    assert!(
        content_type.starts_with("text/event-stream"),
        "the /stream route must always stream, got {content_type:?}"
    );
    assert_eq!(
        concat_deltas(&body),
        expected,
        "the dedicated stream route must deliver the same text as the endpoint it \
         is the streaming form of"
    );
    assert!(
        body.trim_end().ends_with("data: [DONE]"),
        "an OpenAI stream terminates with the [DONE] sentinel: {body}"
    );
}

/// The dedicated route and `"stream":true` are the same endpoint, so they must
/// produce the same wire format. They were two independent implementations.
#[tokio::test]
#[cfg(feature = "gpu")]
async fn stream_route_and_stream_flag_agree_on_the_frame_shape() {
    let (_, _, via_route) = send(
        quantized_state(),
        "/v1/chat/completions/stream",
        &format!("{CHAT_REQUEST}}}"),
    )
    .await;
    let (_, _, via_flag) = send(
        quantized_state(),
        "/v1/chat/completions",
        &format!("{CHAT_REQUEST},\"stream\":true}}"),
    )
    .await;

    let shape = |body: &str| -> Vec<String> {
        body.lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .filter(|payload| payload.trim() != "[DONE]")
            .filter_map(|payload| serde_json::from_str::<serde_json::Value>(payload).ok())
            .map(|frame| {
                format!(
                    "{}|{}",
                    frame["object"].as_str().unwrap_or("?"),
                    frame["choices"][0]["finish_reason"]
                )
            })
            .collect()
    };
    assert_eq!(
        shape(&via_route),
        shape(&via_flag),
        "the /stream route must not be a second, divergent implementation"
    );
}

// ---------------------------------------------------------------------------
// temperature: 0 — the canonical deterministic request must be servable
// ---------------------------------------------------------------------------

/// `temperature: 0` on the dense backend answered
/// `500 {"error":"Invalid shape: Temperature must be a positive finite number"}`
/// on `/v1/chat/completions` and `/v1/completions`, while
/// `/v1/chat/completions/stream` served it — one handler had been fixed
/// (PMAT-790) and the other two kept its private copy of the bug.
#[tokio::test]
async fn temperature_zero_is_served_on_every_openai_route() {
    let demo = || AppState::demo().expect("demo AppState");
    for (uri, json) in [
        (
            "/v1/chat/completions",
            r#"{"model":"default","messages":[{"role":"user","content":"token5"}],"max_tokens":3,"temperature":0}"#,
        ),
        (
            "/v1/chat/completions/stream",
            r#"{"model":"default","messages":[{"role":"user","content":"token5"}],"max_tokens":3,"temperature":0}"#,
        ),
        (
            "/v1/completions",
            r#"{"model":"default","prompt":"token5","max_tokens":3,"temperature":0}"#,
        ),
    ] {
        let (status, _, body) = send(demo(), uri, json).await;
        assert_eq!(
            status,
            StatusCode::OK,
            "temperature:0 is the OpenAI deterministic request; {uri} refused it: {body}"
        );
        assert!(
            !body.contains("Temperature must be a positive"),
            "{uri} leaked the sampler's rejection of its own config: {body}"
        );
    }
}

// ---------------------------------------------------------------------------
// #2375(7) — /v1/metrics must report measurements
// ---------------------------------------------------------------------------

/// Drive real traffic through one shared state, then compare the two metrics
/// endpoints on that same state at the same instant. 0.63.0 reported
/// `latency_p50_ms 0.0` next to `realizar_avg_latency_ms 626.79`.
#[tokio::test]
#[cfg(feature = "gpu")]
async fn v1_metrics_percentiles_are_measured_alongside_a_nonzero_average() {
    let state = quantized_state();
    for _ in 0..5 {
        let (status, _, body) = send(
            state.clone(),
            "/v1/chat/completions",
            &format!("{CHAT_REQUEST}}}"),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "traffic must succeed: {body}");
    }

    let (status, prometheus) = get(state.clone(), "/metrics").await;
    assert_eq!(status, StatusCode::OK);
    let avg_latency_ms: f64 = prometheus
        .lines()
        .find_map(|line| line.strip_prefix("realizar_avg_latency_ms "))
        .and_then(|v| v.trim().parse().ok())
        .expect("/metrics must expose realizar_avg_latency_ms");
    assert!(
        avg_latency_ms > 0.0,
        "the control is broken: five completed requests took no measurable time"
    );

    let (status, body) = get(state, "/v1/metrics").await;
    assert_eq!(status, StatusCode::OK);
    let metrics: serde_json::Value = serde_json::from_str(&body).expect("/v1/metrics is JSON");
    let p50 = metrics["latency_p50_ms"].as_f64().expect("p50 present");
    let p95 = metrics["latency_p95_ms"].as_f64().expect("p95 present");
    let p99 = metrics["latency_p99_ms"].as_f64().expect("p99 present");

    assert!(
        p50 > 0.0,
        "/v1/metrics reported p50 {p50} ms while /metrics on the SAME state reported \
         an average of {avg_latency_ms} ms over the same requests"
    );
    assert!(
        p50 <= p95 && p95 <= p99,
        "percentiles must be non-decreasing: p50={p50} p95={p95} p99={p99}"
    );
    // The percentiles and the mean describe the same five requests, so they must
    // be the same order of magnitude. This excludes a p50 sourced from some
    // other collector, which is how the 0.0 was reported next to a 626.79.
    assert!(
        p50 >= avg_latency_ms / 10.0 && p50 <= avg_latency_ms * 10.0,
        "p50 {p50} ms is not consistent with the mean {avg_latency_ms} ms of the same traffic"
    );
}

/// `model_name` must name the model this server loaded. It was the literal
/// `"phi-2-q4_k_m"` for any cached GPU model and `"N/A"` for everything else.
#[tokio::test]
#[cfg(feature = "gpu")]
async fn v1_metrics_model_name_is_derived_not_a_constant() {
    let (status, body) = get(quantized_state(), "/v1/metrics").await;
    assert_eq!(status, StatusCode::OK);
    let metrics: serde_json::Value = serde_json::from_str(&body).expect("/v1/metrics is JSON");
    let model_name = metrics["model_name"].as_str().expect("model_name present");

    assert_ne!(
        model_name, "N/A",
        "a server with a model resident must name it; /health reports \
         model_loaded:true on this same state"
    );
    assert_ne!(
        model_name, "phi-2-q4_k_m",
        "that string is a constant from a different model, served for every model"
    );
}

/// A server with NO model must not invent a name for one.
#[tokio::test]
async fn v1_metrics_reports_no_model_name_when_nothing_is_loaded() {
    let (status, body) = get(AppState::demo_mock().expect("mock state"), "/v1/metrics").await;
    assert_eq!(status, StatusCode::OK);
    let metrics: serde_json::Value = serde_json::from_str(&body).expect("/v1/metrics is JSON");
    assert_eq!(
        metrics["model_name"], "N/A",
        "no model is loaded, so there is no name to report"
    );
    assert_eq!(
        metrics["latency_p50_ms"], 0.0,
        "no request has completed, so there is no latency to report"
    );
}

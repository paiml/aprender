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
//!
//!   PROVENANCE, measured rather than assumed: this one never shipped. The
//!   cancellation layer does not exist in v0.63.0 —
//!   `git cat-file -e v0.63.0:crates/aprender-serve/src/api/cancel_scope.rs`
//!   fails and `git grep CancelOnDrop v0.63.0` has zero hits (while the same
//!   grep finds `AppState` there, so the search could have succeeded). The
//!   empty stream was introduced on `main` by the #2376(3) work in `a8c6807d8`
//!   and is fixed here, in the commit that follows it. The 404 and `/v1/metrics`
//!   claims below ARE 0.63.0 behaviour and are cited as such.
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
         with the guard armed on completion this delivered an empty stream too"
    );
}

/// The other half of the guard change: an ABANDONED stream must still stop.
///
/// Disarming the guard on completion (above) removed the cancellation layer's
/// coverage of streaming responses — a streaming handler *completes* while its
/// decode loop is still running, so the layer can no longer tell an abandoned
/// stream from a healthy one. The contract
/// (`contracts/apr-serve-cancellation-v1.yaml`) states the replacement as a
/// discharged property: "an abandoned STREAM is still stopped, by body-drop
/// rather than by this guard". This is that falsifier. It was asserted and
/// tested by nothing when the guard change shipped.
///
/// The chain, driven through the REAL router and the real quantized backend:
///
/// 1. `POST /v1/chat/completions {"stream":true}` with a budget far larger than
///    the 16-slot token channel, so the decode loop is guaranteed to be alive
///    and blocked on a send when the client leaves;
/// 2. the response body is DROPPED without being read — what hyper does to the
///    body of an abandoned request;
/// 3. the SSE receiver drops with it, the next `on_token` send fails, and
///    `streaming_token_sink` records ONE abandonment and returns `false`;
/// 4. `generate_with_cache_streaming` breaks.
///
/// Both assertions are load-bearing. "at least one abandonment" falsifies
/// step 3 — if dropping the body did not reach the decode loop, no send would
/// ever fail. "EXACTLY one" falsifies step 4 — a loop that keeps generating for
/// a client that left fails every subsequent send too, and records every one of
/// them.
#[tokio::test(flavor = "multi_thread")]
#[cfg(feature = "gpu")]
async fn an_abandoned_stream_is_stopped_by_the_body_drop() {
    // > 16 (the channel capacity), so the loop cannot finish before the drop.
    const BUDGET: usize = 64;
    let request = format!(
        r#"{{"model":"default","messages":[{{"role":"user","content":"token5 token6"}}],"max_tokens":{BUDGET},"stream":true}}"#
    );

    // Control FIRST: a stream that is read to the end records NO abandonment,
    // so the counter is not simply firing for every stream.
    let consumed_state = quantized_state();
    let (status, _, body) = send(
        consumed_state.clone(),
        "/v1/chat/completions",
        &request.clone(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        !concat_deltas(&body).is_empty(),
        "the control stream must carry tokens, or 'nothing was abandoned' is trivial"
    );
    assert_eq!(
        consumed_state.metrics.streams_abandoned(),
        0,
        "a stream the client read to the end was reported as abandoned"
    );

    // Now abandon one.
    let state = quantized_state();
    let response = create_router(state.clone())
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(request))
                .expect("build request"),
        )
        .await
        .expect("dispatch");
    assert_eq!(response.status(), StatusCode::OK);
    drop(response.into_body());

    // Step 3: the drop must reach the decode loop.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while state.metrics.streams_abandoned() == 0 && std::time::Instant::now() < deadline {
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
    assert!(
        state.metrics.streams_abandoned() >= 1,
        "dropping the response body did not stop the decode loop: no failed token \
         send was ever observed, so the loop is still generating for a client that \
         is gone (the guard no longer covers this case — body drop is the ONLY \
         mechanism left)"
    );

    // Step 4: and it must have BROKEN the loop, not merely noticed. A loop still
    // running would fail its remaining sends too; this fixture generates a token
    // in well under a millisecond, so the whole remaining budget would be spent
    // inside this settle window.
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    assert_eq!(
        state.metrics.streams_abandoned(),
        1,
        "one abandoned stream must produce exactly one abandonment: more than one \
         means the decode loop kept running (and kept failing to send) after the \
         client went away, which is the {BUDGET}-token burn #2376(3) is about"
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
///
/// The comparison is only worth making on streams that carry something: two
/// empty streams have equal shapes trivially. As first written this test had no
/// such requirement and stayed GREEN under the guard-re-arm mutation — both
/// sides degraded to the same content-free frame list, so it discriminated the
/// stream-route defect only. It now requires content deltas on both sides
/// FIRST, which is what makes the equality below meaningful.
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

    // Both streams must actually carry the reply before their shapes are
    // compared. Without this the assertion below is satisfied by two empty
    // streams, which is exactly what the guard-re-arm mutation produces.
    let route_text = concat_deltas(&via_route);
    let flag_text = concat_deltas(&via_flag);
    assert!(
        !route_text.is_empty(),
        "/v1/chat/completions/stream delivered no content deltas, so comparing \
         frame shapes proves nothing; frames were {:?}",
        shape(&via_route)
    );
    assert!(
        !flag_text.is_empty(),
        "/v1/chat/completions with stream:true delivered no content deltas, so \
         comparing frame shapes proves nothing; frames were {:?}",
        shape(&via_flag)
    );
    assert_eq!(
        route_text, flag_text,
        "the two forms of the same endpoint must carry the same text"
    );

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

/// The REST of the temperature domain: an unservable value must be refused as a
/// client error naming the field — never answered `500` with the sampler's own
/// complaint about a config the server built.
///
/// Fixing `temperature == 0.0` alone left `-1`, `NaN`, `+inf` and any value that
/// narrows to `+inf` as an `f32` reaching `apply_temperature` and producing
/// exactly the body the fix set out to eliminate:
/// `{"error":"Invalid shape: Temperature must be a positive finite number"}`.
///
/// The cases are chosen for what each one defeats:
///
/// * `-1` — the plain negative.
/// * `1e40` — finite as JSON and as `f64`, `+inf` once narrowed to `f32`. A
///   guard that checks the parsed `f64` only lets this through.
/// * `1e400` — beyond `f64` entirely. Measured, not assumed: serde_json's own
///   number parser refuses this one before any of our code runs, so the refusal
///   is the generic sanitized body rather than a message naming the field. It is
///   asserted for the class it does prove — client error, no sampler leak.
///
/// NaN has no JSON literal, so it cannot be sent over the wire at all; it is
/// covered at the resolver instead (`temperature_domain_is_total`), which is
/// where a Rust caller could still produce one.
#[tokio::test]
#[cfg(feature = "gpu")]
async fn unservable_temperature_is_refused_on_every_generating_route() {
    // Every route on `create_router_with_config` that accepts a `temperature`
    // and can generate — enumerated from the router's own route table, not from
    // the routes that were convenient to fix. The two `/api/*` routes take it
    // inside `options` and build their `ChatCompletionRequest` in Rust, so a
    // guard on that struct alone would not cover them; the three native routes
    // validated it only on the QUANTIZED backend, so on a dense server they
    // answered `500 "Temperature must be a positive finite number"` (measured,
    // then fixed, while writing this test).
    let routes: [(&str, &str, &str); 9] = [
        (
            "/v1/chat/completions",
            r#"{"model":"default","messages":[{"role":"user","content":"token5"}],"max_tokens":3,"temperature":"#,
            "}",
        ),
        (
            "/v1/chat/completions/stream",
            r#"{"model":"default","messages":[{"role":"user","content":"token5"}],"max_tokens":3,"temperature":"#,
            "}",
        ),
        (
            "/v1/completions",
            r#"{"model":"default","prompt":"token5","max_tokens":3,"temperature":"#,
            "}",
        ),
        (
            "/v1/batch/completions",
            r#"{"prompts":["token5"],"max_tokens":3,"temperature":"#,
            "}",
        ),
        (
            "/api/chat",
            r#"{"model":"default","messages":[{"role":"user","content":"token5"}],"options":{"num_predict":3,"temperature":"#,
            "}}",
        ),
        (
            "/api/generate",
            r#"{"model":"default","prompt":"token5","options":{"num_predict":3,"temperature":"#,
            "}}",
        ),
        (
            "/generate",
            r#"{"prompt":"token5","max_tokens":3,"temperature":"#,
            "}",
        ),
        (
            "/stream/generate",
            r#"{"prompt":"token5","max_tokens":3,"temperature":"#,
            "}",
        ),
        (
            "/batch/generate",
            r#"{"prompts":["token5"],"max_tokens":3,"temperature":"#,
            "}",
        ),
    ];

    for (uri, head, tail) in routes {
        // `names_the_field` is false only for the value serde_json refuses before
        // our guard is reached (see the doc comment).
        for (unservable, names_the_field) in [("-1", true), ("1e40", true), ("1e400", false)] {
            let (status, _, body) =
                send(quantized_state(), uri, &format!("{head}{unservable}{tail}")).await;

            assert!(
                status.is_client_error(),
                "{uri} with temperature {unservable} must be refused as a client error, \
                 got {status}: {body}"
            );
            if names_the_field {
                assert!(
                    body.contains("temperature"),
                    "{uri} refused temperature {unservable} without saying which field \
                     was wrong: {body}"
                );
            }
            assert!(
                !body.contains("Temperature must be a positive"),
                "{uri} leaked the sampler's rejection of a config the SERVER built from \
                 temperature {unservable}: {body}"
            );
        }

        // Positive control on the SAME route and the same server: a servable
        // temperature is still answered, so the rejections above are about the
        // value and not about the route being broken.
        let (status, _, body) = send(quantized_state(), uri, &format!("{head}0.7{tail}")).await;
        assert!(
            !status.is_client_error(),
            "{uri} refused a servable temperature of 0.7: {status} {body}"
        );
    }
}

/// The resolver itself must be TOTAL: no `f32` may produce a config that
/// `sample_token` rejects. This is the half of the domain a client cannot reach
/// (NaN has no JSON literal) and a Rust caller can.
#[test]
fn temperature_domain_is_total() {
    use crate::api::realize_handlers::resolve_dense_generation_config;
    use crate::generate::{sample_token, SamplingStrategy};
    use crate::tensor::Tensor;

    let logits = Tensor::from_vec(vec![4], vec![0.1, 0.2, 0.9, 0.3]).expect("tensor");

    for temperature in [
        0.0_f32,
        -1.0,
        -0.0,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::MIN,
    ] {
        let config = resolve_dense_generation_config(temperature, Some(0.9), 16);
        assert_eq!(
            config.strategy,
            SamplingStrategy::Greedy,
            "temperature {temperature} is not a positive finite scale, so it must \
             resolve to deterministic decoding"
        );
        let token = sample_token(&logits, &config, 0.5).unwrap_or_else(|e| {
            panic!(
                "temperature {temperature} produced a config the sampler rejects \
                 (this is the HTTP 500): {e:?}"
            )
        });
        assert_eq!(token, 2, "greedy must select the argmax token");
    }

    // The converse: a servable temperature is NOT rewritten to greedy, or this
    // test would pass on a resolver that ignores its argument.
    let config = resolve_dense_generation_config(0.7, Some(0.9), 16);
    assert_eq!(
        config.strategy,
        SamplingStrategy::TopP { p: 0.9 },
        "a positive finite temperature with top_p must still resolve to nucleus sampling"
    );
    assert!(
        (config.temperature - 0.7).abs() < 1e-6,
        "a servable temperature must reach the sampler unchanged, got {}",
        config.temperature
    );
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

/// `model_name` must name the model THIS server loaded. It was the literal
/// `"phi-2-q4_k_m"` for any cached GPU model and `"N/A"` for everything else.
///
/// Two servers, differing only in the model they were pointed at, must report
/// two different names — each its own file stem. That is what excludes a
/// constant; "not `N/A`" does not. The first version of this falsifier asserted
/// only `!= "N/A"` and `!= "phi-2-q4_k_m"` against a fixture that set no model
/// source at all, so it passed on the value `"default"` — another constant, out
/// of the fallback arm — and the deriving branch it is named for
/// (`served_model_name`'s `file_stem`) was executed by no test in the suite.
#[tokio::test]
#[cfg(feature = "gpu")]
async fn v1_metrics_model_name_is_derived_from_the_model_this_server_loaded() {
    use crate::api::ModelSourceInfo;

    async fn reported_name(dir: &std::path::Path, file_name: &str) -> String {
        let path = dir.join(file_name);
        // A real file, so `from_path` measures it the way `apr serve run` does
        // (size and format from the bytes, not from the extension).
        std::fs::write(&path, b"GGUF\0\0\0\0not-a-real-model").expect("write fixture model");
        let state = quantized_state().with_model_source(ModelSourceInfo::from_path(&path));

        let (status, body) = get(state, "/v1/metrics").await;
        assert_eq!(status, StatusCode::OK);
        let metrics: serde_json::Value = serde_json::from_str(&body).expect("/v1/metrics is JSON");
        metrics["model_name"]
            .as_str()
            .expect("model_name present")
            .to_string()
    }

    let dir = tempfile::tempdir().expect("tempdir");
    let first = reported_name(dir.path(), "albor-370m-v1-q4_k_m.gguf").await;
    let second = reported_name(dir.path(), "qwen2.5-coder-1.5b-instruct-q4k.gguf").await;

    assert_eq!(
        first, "albor-370m-v1-q4_k_m",
        "/v1/metrics must report the stem of the model file this server loaded"
    );
    assert_eq!(
        second, "qwen2.5-coder-1.5b-instruct-q4k",
        "a second server on a different model must report ITS model, not the first one's"
    );
    assert_ne!(
        first, second,
        "two servers serving two different models reported the same name, which no \
         derivation can do and every constant does"
    );

    // The fallback arm, pinned by name so it can never again pass as "derived":
    // a resident model with no source path is reported as the id `/v1/models`
    // advertises in single-model mode.
    let (status, body) = get(quantized_state(), "/v1/metrics").await;
    assert_eq!(status, StatusCode::OK);
    let metrics: serde_json::Value = serde_json::from_str(&body).expect("/v1/metrics is JSON");
    assert_eq!(
        metrics["model_name"], "default",
        "with a model resident but no source path, the reported name is the id a \
         client may send back as \"model\" — not \"N/A\", and not a constant naming \
         some other model"
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

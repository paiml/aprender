//! aprender#3718: `/v1/chat/completions` on a GGUF server must never report a cut as
//! a finished reply, and a prompt that cannot fit must be refused as the client's
//! error.
//!
//! The dense CPU backend clamps `max_tokens` to the room left in the context
//! (`effective_max_tokens`, aprender#2376), then judged `finish_reason` against the
//! UNCLAMPED request. So a reply cut at the context edge said `"stop"`, and a
//! consumer could not tell it from a complete answer. A prompt longer than the
//! context was a 500 (non-streaming) or a 200 stream carrying an error event that
//! still ended `"stop"`.
//!
//! The fixture's greedy pick is token 0, which is also the EOS fallback when a
//! model declares none (`chat_gen_params`). With `ignore_eos` a run ends on its
//! budget, which is what the clamp tests vary; without it the run stops at once,
//! which is the converse.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::util::ServiceExt;

use crate::api::{create_router, AppState};

/// A quantized-only server with a `context_length`-token window.
fn quantized_state(context_length: usize) -> AppState {
    use crate::api::test_helpers::create_test_quantized_model;
    use crate::gguf::{ArchConstraints, GGUFConfig};

    let config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: ArchConstraints::from_architecture("llama"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 256,
        context_length,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        explicit_head_dim: None,
        query_pre_attn_scalar: None,
        bos_token_id: None,
        eos_token_id: None,
    };
    AppState::with_quantized_model(create_test_quantized_model(&config))
        .expect("build quantized AppState")
}

async fn chat(
    state: AppState,
    max_tokens: usize,
    stream: bool,
    ignore_eos: bool,
) -> (StatusCode, String) {
    let body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": stream,
        "ignore_eos": ignore_eos,
    });
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
        .expect("response");
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body");
    (status, String::from_utf8_lossy(&bytes).into_owned())
}

/// How many tokens the templated `"Hi"` prompt is, measured through the server
/// itself with a budget small enough that nothing clamps.
async fn templated_prompt_tokens() -> usize {
    let (status, body) = chat(quantized_state(512), 1, false, true).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let v: serde_json::Value = serde_json::from_str(&body).expect("json");
    v["usage"]["prompt_tokens"].as_u64().expect("prompt_tokens") as usize
}

#[tokio::test]
async fn a_context_clamped_cut_is_length_not_stop() {
    let prompt = templated_prompt_tokens().await;
    let context = prompt + 6;
    let (status, body) = chat(quantized_state(context), 1000, false, true).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let v: serde_json::Value = serde_json::from_str(&body).expect("json");
    assert_eq!(v["usage"]["prompt_tokens"], prompt, "{body}");
    assert_eq!(
        v["usage"]["completion_tokens"], 6,
        "control: the run must stop at the context edge (6 tokens of room), or this \
         is not measuring a clamp: {body}"
    );
    assert_eq!(v["usage"]["total_tokens"], context, "{body}");
    assert_eq!(v["choices"][0]["finish_reason"], "length", "{body}");
}

#[tokio::test]
async fn a_context_clamped_stream_ends_length_not_stop() {
    let prompt = templated_prompt_tokens().await;
    let (status, body) = chat(quantized_state(prompt + 6), 1000, true, true).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(
        body.contains(r#""finish_reason":"length""#),
        "the terminal chunk must say the stream was cut: {body}"
    );
    assert!(!body.contains(r#""finish_reason":"stop""#), "{body}");
}

#[tokio::test]
async fn a_prompt_longer_than_the_context_is_a_400() {
    let prompt = templated_prompt_tokens().await;
    for stream in [false, true] {
        let (status, body) = chat(quantized_state(prompt - 1), 8, stream, true).await;
        assert_eq!(
            status,
            StatusCode::BAD_REQUEST,
            "stream={stream}: the request cannot fit, which is the client's error: {body}"
        );
        assert!(!body.contains("finish_reason"), "stream={stream}: nothing ran: {body}");
    }
}

/// The converse, so the clamp tests cannot pass by reporting "length" for
/// everything: with EOS honoured the fixture stops on its first pick, under
/// budget and under the clamp, and that is a finished reply.
#[tokio::test]
async fn a_stop_under_budget_is_stop() {
    let prompt = templated_prompt_tokens().await;
    for stream in [false, true] {
        let (status, body) = chat(quantized_state(prompt + 6), 1000, stream, false).await;
        assert_eq!(status, StatusCode::OK, "stream={stream}: {body}");
        assert!(
            body.contains(r#""completion_tokens":0"#),
            "control: the fixture must stop on its first pick: {body}"
        );
        assert!(body.contains(r#""finish_reason":"stop""#), "stream={stream}: {body}");
        assert!(!body.contains(r#""finish_reason":"length""#), "stream={stream}: {body}");
    }
}

/// With room to spare, the budget that ends the run is the REQUEST's.
#[tokio::test]
async fn an_unclamped_run_is_cut_at_the_request_budget() {
    let (status, body) = chat(quantized_state(512), 4, false, true).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let v: serde_json::Value = serde_json::from_str(&body).expect("json");
    assert_eq!(v["usage"]["completion_tokens"], 4, "{body}");
    assert_eq!(v["choices"][0]["finish_reason"], "length", "{body}");
}

//! #4268: `apr serve`'s dense CPU chat backend answers through the one engine,
//! streaming and not. The session witness names the prompt each turn served,
//! so a request that still ran the old loop leaves no entry for its prompt.

use super::*;
use crate::api::create_router;
use axum::body::Body;
use axum::http::Request;
use tower::ServiceExt;

fn quantized_state() -> AppState {
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
        context_length: 512,
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

fn chat_body(content: &str, stream: bool) -> serde_json::Value {
    serde_json::json!({
        "model": "anything",
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.0,
        "max_tokens": 4,
        "stream": stream,
    })
}

/// The ids the handler will tokenize `body` to — the witness's key.
fn prompt_ids(state: &AppState, body: &serde_json::Value) -> Vec<u32> {
    let request: ChatCompletionRequest =
        serde_json::from_value(body.clone()).expect("a chat request");
    let tokenizer = require_tokenizer(state).expect("tokenizer");
    let arch = state.model_architecture();
    tokenize_chat_prompt(&tokenizer, &request.messages, arch.as_deref(), None, state).expect("prompt ids")
}

async fn served_by_the_engine(content: &str, stream: bool) {
    let state = quantized_state();
    let body = chat_body(content, stream);
    let ids = prompt_ids(&state, &body);
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
    assert_eq!(response.status(), StatusCode::OK);
    // Drain the body so a streamed turn has finished before the witness is read.
    axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body");
    let entries = crate::session::entries_for(&ids);
    assert!(
        entries
            .iter()
            .any(|e| e.kind == crate::session::EntryKind::Generate && !e.on_gpu),
        "the dense CPU chat turn did not run on the engine: {entries:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn serve_chat_answers_a_dense_model_through_the_engine() {
    served_by_the_engine("dense serve 4268, whole reply", false).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn serve_chat_streams_a_dense_model_through_the_engine() {
    served_by_the_engine("dense serve 4268, streamed reply", true).await;
}

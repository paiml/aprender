//! aprender#4339: `/v1/completions` on a GGUF CPU server must stop on an
//! end-of-generation token.
//!
//! `try_quantized_completions` built its config with `stop_tokens: Vec::new()`,
//! so a raw completion ran to `max_tokens` whatever the model emitted. On
//! Qwen2.5-coder-1.5b-instruct that was 13 of 13 CRUX prompts running past the
//! answer into invented `\nHuman:` turns, streaming and not, while llama-server
//! stopped. The Qwen shape is the trap: the GGUF declares `<|im_end|>` as EOS, and a
//! raw completion ends with `<|endoftext|>`, so the EOS id alone is not enough.
//!
//! The fixture's greedy pick is token 0 (see `usage_finish_3718`). Each test
//! decides what token 0 IS: an EOG marker that is NOT the declared EOS (the Qwen
//! shape), the declared EOS itself, or an ordinary token — the positive control,
//! which must still run to its budget, or a stop proves nothing.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::util::ServiceExt;

use crate::api::{create_router, AppState};

const BUDGET: usize = 12;

/// A quantized-only server whose vocabulary puts `token0` at id 0 and declares
/// `eos` as the model's EOS. The rest of the vocabulary is printable ASCII, so the
/// prompt encodes to real ids.
fn state_with(token0: &str, eos: Option<u32>) -> AppState {
    use crate::api::test_helpers::create_test_quantized_model;
    use crate::gguf::{ArchConstraints, GGUFConfig};

    let mut vocab: Vec<String> = vec![token0.to_string(), "<unk>".to_string()];
    vocab.extend((b' '..=b'~').map(|c| (c as char).to_string()));
    vocab.extend((vocab.len()..256).map(|i| format!("token{i}")));
    let config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: ArchConstraints::from_architecture("llama"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: vocab.len(),
        context_length: 512,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        explicit_head_dim: None,
        query_pre_attn_scalar: None,
        bos_token_id: None,
        eos_token_id: eos,
    };
    AppState::with_quantized_model_and_vocab(create_test_quantized_model(&config), vocab)
        .expect("build quantized AppState")
}

async fn complete(state: AppState, stream: bool) -> (StatusCode, String) {
    let body = serde_json::json!({
        "model": "default",
        "prompt": "def f(x): return x",
        "max_tokens": BUDGET,
        "temperature": 0.0,
        "stream": stream,
    });
    let response = create_router(state)
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
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

fn json(body: &str) -> serde_json::Value {
    serde_json::from_str(body).unwrap_or_else(|e| panic!("not JSON ({e}): {body}"))
}

#[tokio::test]
async fn control_an_ordinary_token_runs_to_the_budget() {
    let (status, body) = complete(state_with("tokenZ", Some(7)), false).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let v = json(&body);
    assert_eq!(
        v["usage"]["completion_tokens"], BUDGET,
        "control: the fixture must generate token 0 every step, or the stop tests \
         below measure nothing: {body}"
    );
    assert_eq!(v["choices"][0]["finish_reason"], "length", "{body}");
}

#[tokio::test]
async fn an_eog_marker_that_is_not_the_declared_eos_stops_the_completion() {
    // Qwen shape: EOS is some other id (7); the model emits `<|endoftext|>`.
    let (status, body) = complete(state_with("<|endoftext|>", Some(7)), false).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let v = json(&body);
    assert_eq!(
        v["usage"]["completion_tokens"], 0,
        "#4339: the completion ran past <|endoftext|>: {body}"
    );
    assert_eq!(v["choices"][0]["finish_reason"], "stop", "{body}");
    assert_eq!(
        v["choices"][0]["text"], "",
        "the stop token must not leak into the text: {body}"
    );
}

#[tokio::test]
async fn the_declared_eos_stops_the_completion() {
    let (status, body) = complete(state_with("tokenZ", Some(0)), false).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let v = json(&body);
    assert_eq!(v["usage"]["completion_tokens"], 0, "{body}");
    assert_eq!(v["choices"][0]["finish_reason"], "stop", "{body}");
}

#[tokio::test]
async fn an_im_end_turn_marker_stops_a_streamed_completion() {
    let (status, body) = complete(state_with("<|im_end|>", None), true).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(
        body.contains(r#""finish_reason":"stop""#),
        "#4339: the stream must end on the turn marker: {body}"
    );
    assert!(!body.contains(r#""finish_reason":"length""#), "{body}");
    assert!(!body.contains("im_end"), "the marker leaked into the stream: {body}");
}

#[test]
fn stop_set_is_eos_plus_every_eog_marker_in_the_vocabulary_once() {
    use crate::api::realize_handlers::completion_stop_tokens;
    let vocab: Vec<String> = ["<unk>", "<|endoftext|>", "a", "<|im_end|>", "<|eot_id|>"]
        .iter()
        .map(|s| (*s).to_string())
        .collect();
    let tok = crate::tokenizer::BPETokenizer::new(vocab, vec![], "<unk>").expect("tokenizer");
    assert_eq!(completion_stop_tokens(&tok, Some(3)), vec![3, 1, 4]);
    assert_eq!(completion_stop_tokens(&tok, Some(2)), vec![2, 3, 1, 4]);
    assert_eq!(completion_stop_tokens(&tok, None), vec![3, 1, 4]);
}

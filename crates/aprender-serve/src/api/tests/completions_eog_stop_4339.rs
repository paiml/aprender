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

/// The three model backends that answer `/v1/completions` on a GGUF server:
/// `try_quantized_completions`, `try_cached_completions`, and the batch scheduler
/// `try_cached_completions` hands off to when `apr serve` wired one in (it runs
/// its own decode loop in `batch_processing.rs`, with its own config).
#[derive(Clone, Copy)]
enum Backend {
    Quantized,
    Cached,
    CachedBatch,
}

/// A quantized-only server whose vocabulary puts `token0` at id 0 and declares
/// `eos` as the model's EOS. The rest of the vocabulary is printable ASCII, so the
/// prompt encodes to real ids.
fn state_with(token0: &str, eos: Option<u32>) -> AppState {
    backend_state(Backend::Quantized, token0, eos)
}

fn backend_state(backend: Backend, token0: &str, eos: Option<u32>) -> AppState {
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
    let model = create_test_quantized_model(&config);
    match backend {
        Backend::Quantized => AppState::with_quantized_model_and_vocab(model, vocab)
            .expect("build quantized AppState"),
        Backend::Cached | Backend::CachedBatch => {
            let cached = crate::gguf::OwnedQuantizedModelCachedSync::new(model);
            let state = AppState::with_cached_model_and_vocab(cached, vocab)
                .expect("build cached AppState");
            if matches!(backend, Backend::Cached) {
                return state;
            }
            // What `apr serve` does (apr-cli serve/server.rs): spawn the processor
            // on the state's own model and hand the state its sender.
            let model = state.cached_model().expect("cached model").clone();
            let config = crate::api::gpu_handlers::BatchConfig::default();
            let tx = crate::api::gpu_handlers::spawn_batch_processor(model, config.clone());
            state.with_batch_config(tx, config)
        },
    }
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

/// Every backend: the control runs to its budget, and `<|endoftext|>` (not the
/// declared EOS) stops at once. The batch scheduler built its own config with
/// `stop_tokens: Vec::new()` — fixing the two handler call sites left it running
/// to `max_tokens` on every `apr serve` whose batch processor was wired in.
async fn assert_backend_stops_on_eog(backend: Backend, name: &str) {
    let (status, body) = complete(backend_state(backend, "tokenZ", Some(7)), false).await;
    assert_eq!(status, StatusCode::OK, "{name}: {body}");
    let v = json(&body);
    assert_eq!(
        v["usage"]["completion_tokens"], BUDGET,
        "{name} control: an ordinary token must run to the budget: {body}"
    );
    assert_eq!(v["choices"][0]["finish_reason"], "length", "{name}: {body}");

    let (status, body) = complete(backend_state(backend, "<|endoftext|>", Some(7)), false).await;
    assert_eq!(status, StatusCode::OK, "{name}: {body}");
    let v = json(&body);
    assert_eq!(
        v["usage"]["completion_tokens"], 0,
        "#4339 {name}: the completion ran past <|endoftext|>: {body}"
    );
    assert_eq!(v["choices"][0]["finish_reason"], "stop", "{name}: {body}");
    assert_eq!(v["choices"][0]["text"], "", "{name}: {body}");
}

#[tokio::test]
async fn the_cached_backend_stops_on_an_eog_marker() {
    assert_backend_stops_on_eog(Backend::Cached, "cached").await;
}

#[tokio::test]
async fn the_batch_scheduler_stops_on_an_eog_marker() {
    assert_backend_stops_on_eog(Backend::CachedBatch, "cached+batch").await;
}

/// aprender#4345: `/v1/batch/completions` built its config with `stop_tokens:
/// vec![]`, so every prompt ran to `max_tokens` whatever the model emitted.
async fn batch_generated(token0: &str) -> usize {
    let body = serde_json::json!({
        "prompts": ["def f(x): return x"],
        "max_tokens": BUDGET,
        "temperature": 0.0,
        "top_k": 1,
    });
    let response = create_router(backend_state(Backend::Cached, token0, Some(7)))
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/batch/completions")
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
    let body = String::from_utf8_lossy(&bytes).into_owned();
    assert_eq!(status, StatusCode::OK, "{body}");
    serde_json::from_value(json(&body)["results"][0]["num_generated"].clone())
        .unwrap_or_else(|e| panic!("num_generated ({e}): {body}"))
}

#[tokio::test]
async fn the_batch_completions_route_stops_on_an_eog_marker() {
    assert_eq!(
        batch_generated("tokenZ").await,
        BUDGET,
        "control: an ordinary token must run to the budget"
    );
    assert_eq!(
        batch_generated("<|endoftext|>").await,
        0,
        "#4345: /v1/batch/completions ran past <|endoftext|>"
    );
}

/// aprender#4345: `/v1/logprobs` stopped on the EOS alone. Its handler needs a
/// CUDA model, so the config it builds is checked directly.
#[test]
fn the_logprobs_config_stops_on_the_eos_and_every_eog_marker() {
    use crate::api::realize_handlers::logprobs_config;
    let vocab: Vec<String> = ["<unk>", "<|endoftext|>", "a", "<|im_end|>"]
        .iter()
        .map(|s| (*s).to_string())
        .collect();
    let tok = crate::tokenizer::BPETokenizer::new(vocab, vec![], "<unk>").expect("tokenizer");
    let config = logprobs_config(&tok, Some(3), 9);
    assert_eq!(config.stop_tokens, vec![3, 1], "EOS <|im_end|>, then <|endoftext|>");
    assert_eq!(config.max_tokens, 9);
    assert!(config.logprobs);
    assert_eq!((config.temperature, config.top_k), (0.0, 1), "greedy for perplexity");
}

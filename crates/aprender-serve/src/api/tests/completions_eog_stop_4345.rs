//! aprender#4345: `/v1/completions` on a wgpu `GpuModel` server must stop on an
//! end-of-generation token.
//!
//! #4339 fixed the CPU GGUF backends. `try_gpu_completions` still built its
//! config with `stop_tokens: Vec::new()`, so it ran to `max_tokens` whatever the
//! model emitted — the same Qwen trap: `<|im_end|>` is the declared EOS, and a raw
//! completion ends with `<|endoftext|>`.
//!
//! The fixture's weights are uniform, so every logit ties and the wgpu greedy
//! pick is the LAST id (127). Each test decides what that token IS. The
//! ordinary-token control must still run to its budget, or a stop proves nothing.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::util::ServiceExt;

use crate::api::{create_router, AppState};
use crate::gpu::{GpuModel, GpuModelConfig};

const BUDGET: usize = 6;

/// A wgpu-only server (`gpu_model` set, no quantized/cached model) whose
/// vocabulary puts `picked` at the last id, the one greedy lands on. The rest is
/// printable ASCII, so the prompt encodes to real ids.
fn gpu_state(picked: &str) -> AppState {
    let mut vocab: Vec<String> = vec!["<unk>".to_string()];
    vocab.extend((b' '..=b'~').map(|c| (c as char).to_string()));
    vocab.extend((vocab.len()..127).map(|i| format!("token{i}")));
    vocab.push(picked.to_string());
    let config = GpuModelConfig {
        hidden_dim: 32,
        intermediate_dim: 64,
        num_layers: 1,
        num_heads: 2,
        num_kv_heads: 2,
        vocab_size: vocab.len(),
        eps: 1e-5,
        rope_theta: 10000.0,
        explicit_head_dim: None,
        layer_types: None,
        linear_key_head_dim: None,
        linear_value_head_dim: None,
        linear_num_key_heads: None,
        linear_num_value_heads: None,
        linear_conv_kernel_dim: None,
        constraints: None,
        num_experts: None,
        num_experts_per_tok: None,
        expert_intermediate_size: None,
    };
    let model = GpuModel::new(config).expect("GpuModel");
    AppState::with_gpu_model_and_vocab(model, vocab).expect("gpu AppState")
}

async fn complete(state: AppState) -> serde_json::Value {
    let body = serde_json::json!({
        "model": "default",
        "prompt": "def f(x): return x",
        "max_tokens": BUDGET,
        "temperature": 0.0,
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
    let body = String::from_utf8_lossy(&bytes).into_owned();
    assert_eq!(status, StatusCode::OK, "{body}");
    serde_json::from_str(&body).unwrap_or_else(|e| panic!("not JSON ({e}): {body}"))
}

#[tokio::test]
async fn gpu_control_an_ordinary_token_runs_to_the_budget() {
    let v = complete(gpu_state("tokenZ")).await;
    assert_eq!(
        v["usage"]["completion_tokens"], BUDGET,
        "control: the fixture must generate the last id every step, or the stop tests \
         below measure nothing: {v}"
    );
    assert_eq!(
        v["choices"][0]["text"],
        "tokenZ".repeat(BUDGET),
        "control: the greedy pick must be the last id, where the stop tests put their marker: {v}"
    );
}

#[tokio::test]
async fn gpu_completion_stops_on_endoftext() {
    let v = complete(gpu_state("<|endoftext|>")).await;
    assert_eq!(
        v["usage"]["completion_tokens"], 0,
        "#4345: the wgpu completion ran past <|endoftext|>: {v}"
    );
    assert_eq!(v["choices"][0]["finish_reason"], "stop", "{v}");
    assert_eq!(v["choices"][0]["text"], "", "{v}");
}

#[tokio::test]
async fn gpu_completion_stops_on_im_end() {
    let v = complete(gpu_state("<|im_end|>")).await;
    assert_eq!(
        v["usage"]["completion_tokens"], 0,
        "#4345: the wgpu completion ran past <|im_end|>: {v}"
    );
    assert_eq!(v["choices"][0]["finish_reason"], "stop", "{v}");
}

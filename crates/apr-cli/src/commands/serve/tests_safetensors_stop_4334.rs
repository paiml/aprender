//! #4334: the SafeTensors serve handlers (`chat.rs`, `simple.rs`) stop at the
//! tokenizer's EOS / chat-turn end, like the APR CPU path does since #4265.
//! They hardcoded `stop_tokens: vec![]`, so every reply ran to `max_tokens`.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use super::*;
use axum::body::to_bytes;
use realizar::apr_transformer::{AprTransformer, AprTransformerConfig};

const VOCAB: usize = 100;
const IM_END: u32 = 7;
const EOS: u32 = 9;
const MAX_TOKENS: u64 = 8;

/// A vocab with `<|im_end|>` at `IM_END`; every other entry is an ordinary piece.
fn vocab() -> Vec<String> {
    (0..VOCAB)
        .map(|i| match i {
            0 => "<unk>".to_string(),
            7 => "<|im_end|>".to_string(),
            _ => format!("t{i}"),
        })
        .collect()
}

fn tokenizer_info(eos: Option<u32>) -> SafeTensorsTokenizerInfo {
    let vocab = vocab();
    SafeTensorsTokenizerInfo {
        tokenizer: std::sync::Arc::new(
            realizar::tokenizer::BPETokenizer::new(vocab.clone(), vec![], "<unk>")
                .expect("tiny tokenizer"),
        ),
        vocab,
        bos_token_id: None,
        eos_token_id: eos,
    }
}

/// A tiny zero-weight transformer whose LM-head bias makes greedy decoding emit
/// `always` at every step, so the only way generation ends early is a stop id.
fn state(always: u32, eos: Option<u32>) -> SafeTensorsState {
    let mut t = AprTransformer::new(AprTransformerConfig {
        architecture: "test".to_string(),
        hidden_dim: 16,
        num_layers: 1,
        num_heads: 2,
        num_kv_heads: 2,
        vocab_size: VOCAB,
        intermediate_dim: 32,
        context_length: 128,
        rope_theta: 10000.0,
        eps: 1e-5,
        eos_token_id: None,
        ..Default::default()
    });
    let mut bias = vec![0.0; VOCAB];
    bias[always as usize] = 10.0;
    t.lm_head_bias = Some(bias);
    SafeTensorsState {
        transformer: Some(std::sync::Arc::new(std::sync::Mutex::new(t))),
        tokenizer_info: Some(tokenizer_info(eos)),
        model_path: "test".to_string(),
    }
}

async fn json(resp: axum::response::Response) -> serde_json::Value {
    let bytes = to_bytes(resp.into_body(), 64 * 1024).await.expect("body");
    serde_json::from_slice(&bytes).expect("json")
}

#[test]
fn stop_set_holds_eos_and_turn_end_from_vocab() {
    assert_eq!(
        super::super::handlers::tokenizer_info_stop_tokens(&tokenizer_info(Some(EOS))),
        vec![IM_END, EOS]
    );
    assert_eq!(
        super::super::handlers::tokenizer_info_stop_tokens(&tokenizer_info(None)),
        vec![IM_END]
    );
}

#[tokio::test]
async fn generate_handler_stops_at_eos() {
    let resp = safetensors_generate_handler(
        axum::extract::State(state(EOS, Some(EOS))),
        axum::Json(serde_json::json!({"prompt": "t1", "max_tokens": MAX_TOKENS})),
    )
    .await;
    let v = json(resp).await;
    // One step: the model emits EOS, the loop stops, and the stop id is not reply text.
    assert_eq!(v["tokens_generated"], 0, "{v}");
}

#[tokio::test]
async fn generate_handler_runs_to_max_without_a_stop_id() {
    // Positive control: the same model emitting an ordinary token runs to max_tokens,
    // so the test above measures the stop set, not a broken generation loop.
    let resp = safetensors_generate_handler(
        axum::extract::State(state(11, Some(EOS))),
        axum::Json(serde_json::json!({"prompt": "t1", "max_tokens": MAX_TOKENS})),
    )
    .await;
    assert_eq!(json(resp).await["tokens_generated"], MAX_TOKENS);
}

#[tokio::test]
async fn chat_handler_stops_at_im_end() {
    let resp = safetensors_chat_completions_handler(
        axum::extract::State(state(IM_END, None)),
        axum::Json(serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": MAX_TOKENS,
        })),
    )
    .await;
    let v = json(resp).await;
    assert_eq!(v["usage"]["completion_tokens"], 0, "{v}");
}

//! Falsifiers for aprender#2465 finding 3 — `POST /v1/batch/completions` answered
//! from UTF-8 BYTE VALUES instead of tokens.
//!
//! The shipped handler tokenized with `p.bytes().map(|b| b as u32)` and decoded with
//! `t as u8 as char`. Both halves are provably wrong for every input: a multi-byte
//! character became one id per byte, and even ASCII produced ids that name unrelated
//! vocabulary entries. The response still looked like a completion.
//!
//! Every test below asserts what a client observes, and compares against the SAME
//! server's `/tokenize` route — two routes on one server must not disagree about the
//! tokenization of one string.

#![cfg(feature = "gpu")]

use axum::http::StatusCode;

use super::native_routes_2376::post;
use crate::api::AppState;

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

/// Model vocabulary size. 256 is deliberate: it is large enough that the OLD
/// byte-value mapping still produced in-range ids and a `200 OK`, so these tests
/// fail on the ASSERTION (wrong tokens) rather than on an out-of-range crash.
const VOCAB_SIZE: usize = 256;

/// A vocabulary whose low ids are real multi-byte and multi-character tokens, so
/// "one token" and "one byte" can never be confused for each other.
fn multibyte_vocab() -> Vec<String> {
    let mut vocab: Vec<String> = (0..VOCAB_SIZE).map(|i| format!("tok{i}")).collect();
    vocab[0] = "<unk>".to_string();
    vocab[1] = "世".to_string(); // 3 UTF-8 bytes: E4 B8 96
    vocab[2] = "界".to_string(); // 3 UTF-8 bytes: E7 95 8C
    vocab[3] = "Hello".to_string(); // 5 ASCII bytes
    vocab
}

/// A cached-model server with a real vocabulary — what `apr serve run model.gguf`
/// builds once the GPU cache path is in use, and the only shape in which
/// `/v1/batch/completions` gets past its `SERVICE_UNAVAILABLE` guard.
fn cached_state() -> AppState {
    use crate::api::test_helpers::create_test_quantized_model;
    use crate::gguf::{ArchConstraints, GGUFConfig, OwnedQuantizedModelCachedSync};

    let config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: ArchConstraints::from_architecture("llama"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: VOCAB_SIZE,
        context_length: 128,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        explicit_head_dim: None,
        query_pre_attn_scalar: None,
        bos_token_id: None,
        eos_token_id: None,
    };
    let cached = OwnedQuantizedModelCachedSync::new(create_test_quantized_model(&config));
    AppState::with_cached_model_and_vocab(cached, multibyte_vocab())
        .expect("build cached AppState with a real vocabulary")
}

/// Ask the server's own `/tokenize` route for the ids of `text`.
async fn tokenize_route_ids(text: &str) -> Vec<u32> {
    let body = serde_json::json!({ "text": text }).to_string();
    let (status, body) = post(cached_state(), "/tokenize", &body).await;
    assert_eq!(status, StatusCode::OK, "/tokenize failed: {body}");
    let parsed: serde_json::Value = serde_json::from_str(&body).expect("parse /tokenize body");
    serde_json::from_value(parsed["token_ids"].clone()).expect("token_ids is a u32 array")
}

/// One result of `POST /v1/batch/completions`: `(token_ids, num_generated, text)`.
async fn batch_completion(prompt: &str, max_tokens: usize) -> (Vec<u32>, usize, String) {
    let body = serde_json::json!({ "prompts": [prompt], "max_tokens": max_tokens }).to_string();
    let (status, body) = post(cached_state(), "/v1/batch/completions", &body).await;
    assert_eq!(status, StatusCode::OK, "/v1/batch/completions failed: {body}");
    let parsed: serde_json::Value = serde_json::from_str(&body).expect("parse batch body");
    let result = &parsed["results"][0];
    let token_ids: Vec<u32> =
        serde_json::from_value(result["token_ids"].clone()).expect("token_ids is a u32 array");
    let num_generated: usize =
        serde_json::from_value(result["num_generated"].clone()).expect("num_generated is a usize");
    let text: String = serde_json::from_value(result["text"].clone()).expect("text is a string");
    (token_ids, num_generated, text)
}

// ---------------------------------------------------------------------------
// Finding 3 (P0): the prompt the model is given must be the tokenizer's output
// ---------------------------------------------------------------------------

/// A multi-byte prompt must reach the model as the tokenizer's ids.
///
/// `"世界"` is two vocabulary tokens, ids `[1, 2]`. The byte mapping produced six ids
/// — `[0xE4, 0xB8, 0x96, 0xE7, 0x95, 0x8C]` = `[228, 184, 150, 231, 149, 140]` — each
/// naming an unrelated entry of this vocabulary. The completion was real work done on
/// a sequence the client never asked for.
#[tokio::test]
async fn test_batch_completions_multibyte_prompt_matches_tokenize_route() {
    let expected = tokenize_route_ids("世界").await;
    assert_eq!(
        expected,
        vec![1, 2],
        "fixture check: '世界' must be two vocabulary tokens for this falsifier to bite"
    );

    let (token_ids, num_generated, _text) = batch_completion("世界", 2).await;
    let prompt_len = token_ids.len() - num_generated;

    assert_eq!(
        prompt_len,
        expected.len(),
        "prompt token COUNT disagrees with /tokenize on the same server (ids: {token_ids:?})"
    );
    assert_eq!(
        &token_ids[..prompt_len],
        expected.as_slice(),
        "prompt token IDS disagree with /tokenize on the same server"
    );
}

/// ASCII is wrong too — the defect is not limited to multi-byte input.
///
/// `"Hello"` is the single token id `3`. The byte mapping produced five ids,
/// `[72, 101, 108, 108, 111]`.
#[tokio::test]
async fn test_batch_completions_ascii_prompt_matches_tokenize_route() {
    let expected = tokenize_route_ids("Hello").await;
    assert_eq!(
        expected,
        vec![3],
        "fixture check: 'Hello' must be one vocabulary token for this falsifier to bite"
    );

    let (token_ids, num_generated, _text) = batch_completion("Hello", 2).await;
    let prompt_len = token_ids.len() - num_generated;

    assert_eq!(
        prompt_len,
        expected.len(),
        "prompt token COUNT disagrees with /tokenize on the same server (ids: {token_ids:?})"
    );
    assert_eq!(
        &token_ids[..prompt_len],
        expected.as_slice(),
        "prompt token IDS disagree with /tokenize on the same server"
    );
}

/// The returned `text` must be the tokenizer's decoding of the returned ids.
///
/// `t as u8 as char` reinterpreted each id as a byte and each byte as a codepoint, so
/// the echoed prompt came back as Latin-1 mojibake (`"ä¸..."`) instead of `"世界"`.
#[tokio::test]
async fn test_batch_completions_text_is_decoded_by_the_tokenizer() {
    let (_token_ids, _num_generated, text) = batch_completion("世界", 1).await;

    assert!(
        text.starts_with("世界"),
        "decoded text must begin with the prompt it was generated from, got {text:?}"
    );
}

/// An empty prompt is refused, not generated from.
///
/// Under the byte mapping an empty string silently became an empty token sequence and
/// was handed to the model as a prompt.
#[tokio::test]
async fn test_batch_completions_refuses_an_empty_prompt() {
    let body = serde_json::json!({ "prompts": [""], "max_tokens": 2 }).to_string();
    let (status, body) = post(cached_state(), "/v1/batch/completions", &body).await;

    assert_eq!(
        status,
        StatusCode::BAD_REQUEST,
        "an empty prompt must be refused by status, body: {body}"
    );
    assert!(
        body.contains("Prompt 0"),
        "the refusal must name which prompt was empty, got {body}"
    );
}

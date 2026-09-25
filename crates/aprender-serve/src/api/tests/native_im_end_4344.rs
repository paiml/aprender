//! Falsifier (aprender#4344): a spelled-out special-token marker must never
//! reach the text of a native generate route.
//!
//! `apr serve --no-gpu` on a Qwen GGUF returned `"<answer>7</answer><|im_end|>"`
//! on `/generate`, `/batch/generate`, `/stream/generate`, `/realize/generate` and
//! `/realize/batch`. The model wrote the end-of-turn marker out as ordinary text
//! tokens, so the EOS stop token never fired, and no route cut the text.
//!
//! Fixture: the demo model with a vocabulary in which every token
//! decodes to `"w{i}<|im_end|>"`, so every generated token spells the
//! marker. Each test also checks that the RAW decode of the returned ids
//! contains the marker, so a test cannot pass because the fixture emitted
//! nothing. Removing either cut (`decode_generated` or `cut_pieces_at_marker`)
//! turns the matching tests RED.

use crate::api::{create_router, AppState};
use crate::tokenizer::BPETokenizer;
use axum::body::Body;
use axum::http::Request;
use std::sync::Arc;
use tower::util::ServiceExt;

const MARKER: &str = "<|im_end|>";

fn marker_state() -> (AppState, Arc<BPETokenizer>) {
    let mut state = AppState::demo().expect("demo state");
    // Token 0 is the unk token the demo model generates; it must spell the
    // marker too, and a literal "<unk>" would decode to nothing.
    let vocab: Vec<String> = (0..100).map(|i| format!("w{i}{MARKER}")).collect();
    let unk = vocab[0].clone();
    let tokenizer = Arc::new(BPETokenizer::new(vocab, vec![], unk.as_str()).expect("tokenizer"));
    state.tokenizer = Some(tokenizer.clone());
    (state, tokenizer)
}

async fn post(state: AppState, uri: &str, body: serde_json::Value) -> String {
    let response = create_router(state)
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(uri)
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
    let body = String::from_utf8(bytes.to_vec()).expect("utf-8");
    assert!(status.is_success(), "{uri} answered {status}: {body}");
    body
}

fn generate_body() -> serde_json::Value {
    serde_json::json!({ "prompt": "w1", "max_tokens": 4, "strategy": "greedy" })
}

fn batch_body() -> serde_json::Value {
    serde_json::json!({ "prompts": ["w1", "w2"], "max_tokens": 4, "strategy": "greedy" })
}

/// Assert one `GenerateResponse`. `text` echoes the prompt, and in this
/// fixture the prompt spells the marker too; that echo must stay. What
/// follows it (the completion) generated something whose raw decode has the
/// marker (the fixture engaged), yet carries none.
fn assert_cut(uri: &str, tokenizer: &BPETokenizer, resp: &serde_json::Value) {
    let ids: Vec<u32> = resp["token_ids"]
        .as_array()
        .expect("token_ids")
        .iter()
        .map(|v| v.as_u64().expect("id") as u32)
        .collect();
    let generated = resp["num_generated"].as_u64().expect("num_generated") as usize;
    assert!(generated > 0, "{uri}: nothing generated: {resp}");
    let prompt = tokenizer.decode(&ids[..ids.len() - generated]).expect("decode");
    let raw = tokenizer.decode(&ids[ids.len() - generated..]).expect("decode");
    let text = resp["text"].as_str().expect("text");
    let completion = text
        .strip_prefix(prompt.as_str())
        .unwrap_or_else(|| panic!("{uri}: text {text:?} dropped the prompt echo {prompt:?}"));
    assert!(raw.contains(MARKER), "{uri}: fixture did not engage, raw completion {raw:?}");
    assert!(!completion.contains(MARKER), "{uri}: marker leaked into completion {completion:?}");
    assert!(raw.starts_with(completion), "{uri}: {completion:?} is not a prefix of {raw:?}");
}

async fn check_generate(uri: &str) {
    let (state, tokenizer) = marker_state();
    let body = post(state, uri, generate_body()).await;
    let resp: serde_json::Value = serde_json::from_str(&body).expect("json");
    assert_cut(uri, &tokenizer, &resp);
}

async fn check_batch(uri: &str) {
    let (state, tokenizer) = marker_state();
    let body = post(state, uri, batch_body()).await;
    let resp: serde_json::Value = serde_json::from_str(&body).expect("json");
    let results = resp["results"].as_array().expect("results");
    assert_eq!(results.len(), 2, "{uri}: {resp}");
    for r in results {
        assert_cut(uri, &tokenizer, r);
    }
}

async fn check_stream(uri: &str) {
    let (state, tokenizer) = marker_state();
    let body = post(state, uri, generate_body()).await;
    let mut ids = Vec::new();
    let mut text = String::new();
    for payload in body.lines().filter_map(|l| l.strip_prefix("data: ")) {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(payload) else {
            continue;
        };
        if let (Some(id), Some(t)) = (v["token_id"].as_u64(), v["text"].as_str()) {
            ids.push(id as u32);
            text.push_str(t);
        }
    }
    assert!(!ids.is_empty(), "{uri}: no token events: {body}");
    let raw = tokenizer.decode(&ids).expect("decode");
    assert!(raw.contains(MARKER), "{uri}: fixture did not engage, raw decode {raw:?}");
    assert!(!text.contains(MARKER), "{uri}: marker leaked into the stream {text:?}");
    // The first token spells the marker, so the stream stops at it.
    assert_eq!(ids.len(), 1, "{uri}: tokens after the marker were streamed: {body}");
}

#[tokio::test]
async fn generate_cuts_spelled_out_im_end() {
    check_generate("/generate").await;
}

#[tokio::test]
async fn batch_generate_cuts_spelled_out_im_end() {
    check_batch("/batch/generate").await;
}

#[tokio::test]
async fn realize_batch_cuts_spelled_out_im_end() {
    check_batch("/realize/batch").await;
}

#[tokio::test]
async fn stream_generate_cuts_spelled_out_im_end() {
    check_stream("/stream/generate").await;
}

#[tokio::test]
async fn realize_generate_cuts_spelled_out_im_end() {
    check_stream("/realize/generate").await;
}

#[test]
fn cut_never_searches_the_prompt_echo() {
    use crate::api::realize_handlers::cut_completion_at_marker;
    // A chat-templated prompt carries markers of its own; only the completion is cut.
    let prompt = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n";
    let text = format!("{prompt}<answer>7</answer><|im_end|>\nmore");
    assert_eq!(cut_completion_at_marker(text, prompt), format!("{prompt}<answer>7</answer>"));
    // No marker in the completion: unchanged.
    let clean = format!("{prompt}fine");
    assert_eq!(cut_completion_at_marker(clean.clone(), prompt), clean);
    // Raw-completion text is NOT a special marker and stays.
    let dialogue = "A:\nUser: hi".to_string();
    assert_eq!(cut_completion_at_marker(dialogue.clone(), ""), dialogue);
}

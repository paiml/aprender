//! Falsifier (aprender#4340): a chat stop marker must never reach a live SSE
//! delta.
//!
//! `apr serve --no-gpu` on a Qwen GGUF answered the second turn of a
//! multi-turn chat with `"<answer>7</answer><|im_end|>"` over
//! `POST /v1/chat/completions` `"stream":true`, while the non-streaming call
//! returned `"<answer>7</answer>"`. The non-streaming body goes through
//! `clean_chat_output`, which truncates at `<|im_end|>`; the live stream
//! (`true_streaming_sse_response`, the CPU quantized, CUDA and MoE path) wrote
//! each per-token decode to the wire unfiltered.
//!
//! The marker is spelled out across SEVERAL tokens here (`"<|"`, `"im_end"`,
//! `"|>"`), because that is the case a per-token check cannot see. Removing the
//! `ChatStopFilter` from the live stream turns every test here RED.

use crate::api::openai_handlers::ChatStopFilter;
use crate::tokenizer::BPETokenizer;
use std::sync::Arc;

/// Run `pieces` (each one token) through the live SSE builder; return
/// (concatenated deltas, every delta, `finish_reason`).
async fn stream(pieces: &[&str], stops: Option<&[String]>) -> (String, Vec<String>, String) {
    let vocab: Vec<String> = pieces.iter().map(|s| (*s).to_string()).collect();
    let tokenizer = BPETokenizer::new(vocab.clone(), vec![], vocab[0].as_str()).expect("build tokenizer");
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<u32, String>>(64);
    for id in 0..vocab.len() as u32 {
        tx.send(Ok(id)).await.expect("send token");
    }
    drop(tx);

    let response = crate::api::openai_handlers::true_streaming_sse_response(
        rx,
        Arc::new(tokenizer),
        "chatcmpl-4340".to_string(),
        "test-model".to_string(),
        Arc::new(crate::metrics::MetricsCollector::new()),
        std::time::Instant::now(),
        256,
        0,
        None,
        stops,
    );
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read SSE body");
    let body = String::from_utf8(bytes.to_vec()).expect("SSE body is utf-8");

    let mut deltas = Vec::new();
    let mut finish = String::new();
    for payload in body.lines().filter_map(|l| l.strip_prefix("data: ")) {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(payload) else {
            continue;
        };
        if let Some(t) = v["choices"][0]["delta"]["content"].as_str() {
            deltas.push(t.to_string());
        }
        if let Some(f) = v["choices"][0]["finish_reason"].as_str() {
            finish = f.to_string();
        }
    }
    (deltas.concat(), deltas, finish)
}

#[tokio::test]
async fn im_end_split_across_tokens_never_reaches_a_delta() {
    let (text, deltas, finish) =
        stream(&["<answer>", "7", "</answer>", "<|", "im_end", "|>"], None).await;
    assert_eq!(text, "<answer>7</answer>", "deltas: {deltas:?}");
    for d in &deltas {
        assert!(!d.contains("<|"), "a delta carries a marker fragment: {deltas:?}");
    }
    assert_eq!(finish, "stop");
}

#[tokio::test]
async fn text_after_im_end_is_swallowed() {
    // A model that keeps going past its end-of-turn must not leak the next turn.
    // (A whole `<|im_end|>` vocab entry decodes to "" — the tokenizer drops
    // specials — so the leak is always the SPELLED-OUT marker.)
    let (text, deltas, _) = stream(
        &["Hi", " there", "<|im", "_end|>", "\n", "<|", "im_start|>", "user", " again"],
        None,
    )
    .await;
    assert_eq!(text, "Hi there", "deltas: {deltas:?}");
}

#[tokio::test]
async fn a_false_marker_prefix_is_released_not_eaten() {
    // `<|` that turns out NOT to be a marker is held for one token, then emitted:
    // the filter must not lose text, and the stream ends on EOS, not a stop.
    let (text, deltas, finish) = stream(&["a", " <|", "x", " b"], None).await;
    assert_eq!(text, "a <|x b", "deltas: {deltas:?}");
    assert_eq!(finish, "stop");
}

#[tokio::test]
async fn trailing_marker_prefix_is_flushed_at_end_of_stream() {
    let (text, deltas, _) = stream(&["a", " b", "<|"], None).await;
    assert_eq!(text, "a b<|", "held tail must be flushed at stream end: {deltas:?}");
}

#[tokio::test]
async fn request_stop_applies_to_the_live_stream() {
    let stops = vec!["END".to_string()];
    let (text, deltas, _) = stream(&["one", " two", " E", "ND", " three"], Some(&stops)).await;
    assert_eq!(text, "one two ", "deltas: {deltas:?}");
}

#[test]
fn filter_holds_only_a_possible_marker_prefix() {
    let mut f = ChatStopFilter::new(None);
    assert_eq!(f.push("abc").as_deref(), Some("abc"));
    // "<|im" could still become "<|im_end|>" — held, nothing emitted.
    assert_eq!(f.push("<|im"), None);
    assert_eq!(f.push("_end|>tail"), None);
    assert!(f.stopped());
    assert_eq!(f.push("more"), None);
    assert_eq!(f.finish(), None);
}

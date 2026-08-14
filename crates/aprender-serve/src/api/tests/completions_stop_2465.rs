//! aprender#2465 finding 2: `/v1/completions` could not END a completion.
//!
//! `registry_completions` — the CPU dense backend that answers `apr serve` for every
//! .apr / .safetensors / registry model — never read `request.stop`. The field was
//! accepted (it has been on `CompletionRequest` all along), and had no effect: the
//! generation ran the full `max_tokens` straight past the stop string, which came
//! back inside `choices[0].text` with `finish_reason: "length"`.
//!
//! These are CLIENT-observable falsifiers over the real router and a real (tiny,
//! deterministic) dense model — not assertions about a config field being set.
//! Every one of them runs the UNCONTROLLED request first, so none can pass by
//! generation being broken: the control pins the exact full text that the stopped
//! request must be a strict prefix of.
//!
//! Recorded pre-fix behaviour (verbatim, `stop:["-b"]`, max_tokens 4):
//! ```text
//! {"choices":[{"finish_reason":"length","index":0,
//!   "text":"a0-b0-c0a0-b0-c0a0-b0-c0a0-b0-c0"}], ...}
//! ```
//! i.e. byte-identical to the no-stop control, stop string included.

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use tower::util::ServiceExt;

use crate::api::{create_router, AppState};
use crate::layers::{Model, ModelConfig};
use crate::tokenizer::BPETokenizer;

/// A router over a real dense [`Model`] whose completions are deterministic.
///
/// The weights are the freshly-constructed (uniform) ones, so greedy decoding emits
/// token 0 every step; the vocabulary makes token 0 decode to the structured word
/// `a0-b0-c0`, which gives stop strings that occur at a KNOWN, non-zero offset
/// (`-b` at 2, `-c` at 5). Nothing here depends on the model being smart — only on
/// it being deterministic, which is what makes the control run a valid baseline.
fn stop_app() -> axum::Router {
    let config = ModelConfig {
        vocab_size: 8,
        hidden_dim: 8,
        num_heads: 1,
        num_layers: 1,
        intermediate_dim: 16,
        eps: 1e-5,
    };
    let model = Model::new(config).expect("dense model");
    let vocab: Vec<String> = (0..8).map(|i| format!("a{i}-b{i}-c{i}")).collect();
    let tokenizer = BPETokenizer::new(vocab, vec![], "a0-b0-c0").expect("tokenizer");
    create_router(AppState::new(model, tokenizer))
}

async fn post(uri: &str, body: &str) -> axum::response::Response {
    stop_app()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(uri)
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .expect("request"),
        )
        .await
        .expect("response")
}

async fn body_text(response: axum::response::Response) -> String {
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    String::from_utf8(bytes.to_vec()).expect("body is utf-8")
}

async fn body_json(response: axum::response::Response) -> serde_json::Value {
    let text = body_text(response).await;
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("body is JSON ({e}): {text}"))
}

/// `(text, finish_reason)` of a completion for `prompt` with an optional `stop` clause.
async fn complete(stop_clause: &str) -> (String, String) {
    let body = format!(
        r#"{{"model":"default","prompt":"a1-b1-c1","max_tokens":4,"temperature":0{stop_clause}}}"#
    );
    let response = post("/v1/completions", &body).await;
    assert_eq!(
        response.status(),
        StatusCode::OK,
        "completion must be served; body: {}",
        body
    );
    let json = body_json(response).await;
    let choice = &json["choices"][0];
    (
        choice["text"].as_str().expect("text is a string").to_string(),
        choice["finish_reason"]
            .as_str()
            .expect("finish_reason is a string")
            .to_string(),
    )
}

/// The CONTROL: with no stop clause, the model must produce its full output.
///
/// Every other test in this file compares against this. If generation breaks, this
/// assertion fails first and the stop tests cannot pass vacuously.
#[tokio::test]
async fn control_without_stop_produces_the_full_output() {
    let (text, finish_reason) = complete("").await;

    assert_eq!(
        text, "a0-b0-c0a0-b0-c0a0-b0-c0a0-b0-c0",
        "control: 4 tokens must decode to the full untruncated text"
    );
    assert_eq!(
        finish_reason, "length",
        "control: the budget was exhausted with no stop match"
    );
}

/// THE FALSIFIER (#2465 finding 2): a stop string occurring mid-output truncates
/// the completion at its EARLIEST position and is itself absent from the result.
#[tokio::test]
async fn stop_sequence_truncates_the_completion_at_the_earliest_occurrence() {
    let (full, _) = complete("").await;
    assert!(
        full.contains("-b"),
        "precondition: the control output must CONTAIN the stop string, else this \
         test proves nothing; got {full:?}"
    );
    let cut = full.find("-b").expect("stop occurs in the control output");
    assert!(cut > 0, "the stop must occur MID-output, not at position 0");

    let (stopped, finish_reason) = complete(r#","stop":["-b"]"#).await;

    assert_eq!(
        stopped,
        full[..cut],
        "the completion must be cut at the earliest stop position; pre-fix this \
         returned the whole {full:?}"
    );
    assert!(
        !stopped.contains("-b"),
        "the returned text must not contain the stop string; got {stopped:?}"
    );
    assert!(
        stopped.len() < full.len(),
        "the stopped completion must be SHORTER than the control ({} vs {})",
        stopped.len(),
        full.len()
    );
    assert_eq!(
        finish_reason, "stop",
        "a matched stop beats the token budget; pre-fix this said \"length\""
    );
}

/// Earliest POSITION, not first LISTED: `["-c","-b"]` must cut at `-b` (offset 2),
/// not at `-c` (offset 5).
#[tokio::test]
async fn stop_list_order_does_not_decide_where_the_cut_lands() {
    let (full, _) = complete("").await;
    let earliest = full.find("-b").expect("-b occurs");
    let later = full.find("-c").expect("-c occurs");
    assert!(earliest < later, "the fixture must order the two stops");

    let (stopped, finish_reason) = complete(r#","stop":["-c","-b"]"#).await;

    assert_eq!(
        stopped,
        full[..earliest],
        "with -c listed first, the cut must still land at the earlier -b"
    );
    assert_eq!(finish_reason, "stop");
}

/// A stop string that never occurs must NOT truncate — otherwise "honours stop"
/// could be satisfied by always returning less text.
#[tokio::test]
async fn unmatched_stop_leaves_the_completion_whole() {
    let (full, control_reason) = complete("").await;

    let (text, finish_reason) = complete(r#","stop":["ZZZ-not-in-output"]"#).await;

    assert_eq!(text, full, "an unmatched stop must change nothing");
    assert_eq!(
        finish_reason, control_reason,
        "an unmatched stop must not fake a stop finish"
    );
}

/// An empty stop string is not a match at offset 0: `{"stop":[""]}` must not
/// collapse every completion to `""`.
#[tokio::test]
async fn empty_stop_string_does_not_erase_the_completion() {
    let (full, _) = complete("").await;

    let (text, _) = complete(r#","stop":[""]"#).await;

    assert_eq!(text, full, "an empty stop string must be ignored");
}

/// The SSE surface reads the same completion, so it must be stopped too: the
/// concatenated deltas equal the truncated text and no frame leaks the stop string.
#[tokio::test]
async fn streamed_completion_is_stopped_at_the_same_place() {
    let (expected, _) = complete(r#","stop":["-b"]"#).await;

    let response = post(
        "/v1/completions",
        r#"{"model":"default","prompt":"a1-b1-c1","max_tokens":4,"temperature":0,"stream":true,"stop":["-b"]}"#,
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = body_text(response).await;

    let frames: Vec<serde_json::Value> = body
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|payload| payload.trim() != "[DONE]")
        .map(|payload| serde_json::from_str(payload).expect("SSE frame is JSON"))
        .collect();
    let streamed: String = frames
        .iter()
        .filter_map(|f| f["choices"][0]["text"].as_str())
        .collect();

    assert_eq!(
        streamed, expected,
        "the stream must reassemble to the same stopped text as the buffered body"
    );
    assert!(
        !streamed.contains("-b"),
        "no delta may carry the stop string; got {streamed:?}"
    );
}

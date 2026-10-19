//! `max_completion_tokens` is the newer OpenAI name for the generation cap that
//! older clients call `max_tokens` (OpenAI SDK >= 1.40 sends the new name).
//! Before this alias the unknown key was silently ignored, so the cap was lost
//! and generation ran to the server default.
//!
//! Pinned here: the new name fills `max_tokens`, the old name alone still
//! works, and a body carrying BOTH names is refused by serde with a
//! `duplicate field` error (a deterministic 422, never a silently chosen value).
#![allow(clippy::unwrap_used, clippy::disallowed_methods)]

use super::*;

fn parse(v: serde_json::Value) -> Result<ChatCompletionRequest, serde_json::Error> {
    serde_json::from_value::<ChatCompletionRequest>(v)
}

fn base() -> serde_json::Value {
    serde_json::json!({
        "model": "m",
        "messages": [{"role": "user", "content": "x"}]
    })
}

#[test]
fn max_completion_tokens_aliases_max_tokens() {
    let mut body = base();
    body["max_completion_tokens"] = serde_json::json!(12);
    let req = parse(body).expect("max_completion_tokens must parse");
    assert_eq!(req.max_tokens, Some(12));
}

#[test]
fn max_tokens_alone_still_parses() {
    let mut body = base();
    body["max_tokens"] = serde_json::json!(7);
    let req = parse(body).expect("max_tokens must still parse");
    assert_eq!(req.max_tokens, Some(7));
}

#[test]
fn both_max_tokens_names_present_is_documented() {
    let mut body = base();
    body["max_tokens"] = serde_json::json!(7);
    body["max_completion_tokens"] = serde_json::json!(12);
    let err = parse(body)
        .expect_err("a body carrying both names is refused: serde reports a duplicate field")
        .to_string();
    assert!(
        err.contains("duplicate field"),
        "both names present: serde's derive refuses with `duplicate field` (no value wins), got: {err}"
    );
}

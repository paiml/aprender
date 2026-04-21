//! Ollama `/api/chat` response-shape classifier (CRUX-C-04).
//!
//! Pure, deterministic classifiers that discharge FALSIFY-CRUX-C-04-{001,002,003}
//! at the PARTIAL_ALGORITHM_LEVEL — algorithm-level necessary conditions on
//! already-parsed Ollama chat responses:
//!
//!   * `classify_ollama_chat_schema` — non-streaming response has all 10
//!     Ollama-required top-level keys, `message.role == "assistant"`,
//!     `done == true`.
//!   * `classify_eval_metrics` — `eval_count >= 1` AND `eval_duration > 0`
//!     whenever `message.content` is non-empty.
//!   * `classify_ndjson_stream` — the frame sequence has exactly one
//!     `done == true` frame, and it is the last frame; all non-final frames
//!     have `done == false`.
//!
//! Full discharge blocks on a live `/api/chat` handler in aprender-serve.

use serde_json::Value;

/// The 10 Ollama-required top-level keys on a non-streaming `/api/chat` response
/// (see `https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-chat-completion`).
pub const OLLAMA_CHAT_REQUIRED_KEYS: &[&str] = &[
    "model",
    "created_at",
    "message",
    "done",
    "total_duration",
    "load_duration",
    "prompt_eval_count",
    "prompt_eval_duration",
    "eval_count",
    "eval_duration",
];

/// Outcome of `classify_ollama_chat_schema`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OllamaSchemaOutcome {
    /// Response object contains all 10 required top-level keys,
    /// `message.role == "assistant"`, and `done == true`.
    Ok,
    /// Response is not a JSON object.
    NotAnObject,
    /// One of the 10 required top-level keys is absent.
    MissingRequiredKey { key: &'static str },
    /// `message` is present but not a JSON object.
    MessageNotAnObject,
    /// `message.role` is absent, not a string, or not `"assistant"`.
    WrongMessageRole { got: Option<String> },
    /// `done` is present but not the boolean `true`.
    NotDone { got: Value },
}

/// Non-streaming `/api/chat` response-shape gate.
/// Pure function of `response`; deterministic.
pub fn classify_ollama_chat_schema(response: &Value) -> OllamaSchemaOutcome {
    let obj = match response.as_object() {
        Some(o) => o,
        None => return OllamaSchemaOutcome::NotAnObject,
    };
    for &key in OLLAMA_CHAT_REQUIRED_KEYS {
        if !obj.contains_key(key) {
            return OllamaSchemaOutcome::MissingRequiredKey { key };
        }
    }
    let message = obj.get("message").expect("presence checked above");
    let message_obj = match message.as_object() {
        Some(o) => o,
        None => return OllamaSchemaOutcome::MessageNotAnObject,
    };
    match message_obj.get("role").and_then(Value::as_str) {
        Some("assistant") => {}
        Some(other) => {
            return OllamaSchemaOutcome::WrongMessageRole {
                got: Some(other.to_string()),
            };
        }
        None => return OllamaSchemaOutcome::WrongMessageRole { got: None },
    }
    let done = obj.get("done").expect("presence checked above");
    if done.as_bool() != Some(true) {
        return OllamaSchemaOutcome::NotDone { got: done.clone() };
    }
    OllamaSchemaOutcome::Ok
}

/// Outcome of `classify_eval_metrics`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvalMetricsOutcome {
    /// Metrics are consistent with the emitted content.
    Ok,
    /// `message.content` is non-empty but `eval_count == 0`.
    EvalCountZeroWithContent,
    /// `message.content` is non-empty but `eval_duration == 0`.
    EvalDurationZeroWithContent,
}

/// Gate: any non-empty completion must bill at least one evaluated token
/// and a positive evaluation duration.
pub fn classify_eval_metrics(
    eval_count: u64,
    eval_duration: u64,
    message_content_nonempty: bool,
) -> EvalMetricsOutcome {
    if !message_content_nonempty {
        return EvalMetricsOutcome::Ok;
    }
    if eval_count == 0 {
        return EvalMetricsOutcome::EvalCountZeroWithContent;
    }
    if eval_duration == 0 {
        return EvalMetricsOutcome::EvalDurationZeroWithContent;
    }
    EvalMetricsOutcome::Ok
}

/// Outcome of `classify_ndjson_stream`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NdjsonStreamOutcome {
    /// Exactly one terminal `done=true` frame as the last frame, all others `done=false`.
    Ok,
    /// Frame list is empty.
    NoFrames,
    /// A frame is not a JSON object.
    FrameNotAnObject { frame_index: usize },
    /// A frame is missing the `done` field.
    MissingDoneField { frame_index: usize },
    /// A frame's `done` field is not a boolean.
    DoneNotBool { frame_index: usize },
    /// The last frame is not `done=true`.
    TerminalFrameNotDone { last_frame_index: usize },
    /// A non-final frame is `done=true`.
    EarlyDoneFrame { frame_index: usize },
    /// Zero frames had `done=true`.
    NoTerminalFrame,
}

/// Streaming NDJSON terminal-frame gate. `frames` is the already-parsed
/// ordered list of frames (one JSON object per line). Pure; deterministic.
pub fn classify_ndjson_stream(frames: &[Value]) -> NdjsonStreamOutcome {
    if frames.is_empty() {
        return NdjsonStreamOutcome::NoFrames;
    }
    let last_idx = frames.len() - 1;
    let mut terminal_seen = false;
    for (i, frame) in frames.iter().enumerate() {
        let obj = match frame.as_object() {
            Some(o) => o,
            None => return NdjsonStreamOutcome::FrameNotAnObject { frame_index: i },
        };
        let done_val = match obj.get("done") {
            Some(v) => v,
            None => return NdjsonStreamOutcome::MissingDoneField { frame_index: i },
        };
        let done = match done_val.as_bool() {
            Some(b) => b,
            None => return NdjsonStreamOutcome::DoneNotBool { frame_index: i },
        };
        if done {
            if i != last_idx {
                return NdjsonStreamOutcome::EarlyDoneFrame { frame_index: i };
            }
            terminal_seen = true;
        }
    }
    if !terminal_seen {
        return NdjsonStreamOutcome::TerminalFrameNotDone {
            last_frame_index: last_idx,
        };
    }
    NdjsonStreamOutcome::Ok
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn well_formed_response() -> Value {
        json!({
            "model": "tiny",
            "created_at": "2026-04-21T00:00:00Z",
            "message": {"role": "assistant", "content": "ok"},
            "done": true,
            "total_duration": 1000_u64,
            "load_duration": 100_u64,
            "prompt_eval_count": 2_u64,
            "prompt_eval_duration": 200_u64,
            "eval_count": 1_u64,
            "eval_duration": 500_u64,
        })
    }

    #[test]
    fn schema_ok_on_well_formed_response() {
        assert_eq!(
            classify_ollama_chat_schema(&well_formed_response()),
            OllamaSchemaOutcome::Ok
        );
    }

    #[test]
    fn schema_rejects_non_object() {
        assert_eq!(
            classify_ollama_chat_schema(&json!([1, 2, 3])),
            OllamaSchemaOutcome::NotAnObject
        );
    }

    #[test]
    fn schema_rejects_missing_each_required_key() {
        for &key in OLLAMA_CHAT_REQUIRED_KEYS {
            let mut resp = well_formed_response();
            resp.as_object_mut().unwrap().remove(key);
            let outcome = classify_ollama_chat_schema(&resp);
            assert_eq!(
                outcome,
                OllamaSchemaOutcome::MissingRequiredKey { key },
                "expected MissingRequiredKey({key}), got {outcome:?}"
            );
        }
    }

    #[test]
    fn schema_rejects_non_object_message() {
        let mut resp = well_formed_response();
        resp["message"] = json!("just a string");
        assert_eq!(
            classify_ollama_chat_schema(&resp),
            OllamaSchemaOutcome::MessageNotAnObject
        );
    }

    #[test]
    fn schema_rejects_wrong_role() {
        let mut resp = well_formed_response();
        resp["message"] = json!({"role": "user", "content": "x"});
        assert_eq!(
            classify_ollama_chat_schema(&resp),
            OllamaSchemaOutcome::WrongMessageRole {
                got: Some("user".to_string())
            }
        );
    }

    #[test]
    fn schema_rejects_missing_role() {
        let mut resp = well_formed_response();
        resp["message"] = json!({"content": "x"});
        assert_eq!(
            classify_ollama_chat_schema(&resp),
            OllamaSchemaOutcome::WrongMessageRole { got: None }
        );
    }

    #[test]
    fn schema_rejects_done_false() {
        let mut resp = well_formed_response();
        resp["done"] = json!(false);
        assert_eq!(
            classify_ollama_chat_schema(&resp),
            OllamaSchemaOutcome::NotDone { got: json!(false) }
        );
    }

    #[test]
    fn schema_rejects_done_non_bool() {
        let mut resp = well_formed_response();
        resp["done"] = json!("true");
        assert_eq!(
            classify_ollama_chat_schema(&resp),
            OllamaSchemaOutcome::NotDone {
                got: json!("true")
            }
        );
    }

    #[test]
    fn schema_classifier_is_deterministic() {
        let r = well_formed_response();
        let a = classify_ollama_chat_schema(&r);
        let b = classify_ollama_chat_schema(&r);
        assert_eq!(a, b);
    }

    #[test]
    fn eval_metrics_ok_on_nonempty_reply() {
        assert_eq!(
            classify_eval_metrics(1, 500, true),
            EvalMetricsOutcome::Ok
        );
    }

    #[test]
    fn eval_metrics_ok_on_empty_reply_even_with_zero_metrics() {
        assert_eq!(
            classify_eval_metrics(0, 0, false),
            EvalMetricsOutcome::Ok
        );
    }

    #[test]
    fn eval_metrics_rejects_eval_count_zero_with_content() {
        assert_eq!(
            classify_eval_metrics(0, 500, true),
            EvalMetricsOutcome::EvalCountZeroWithContent
        );
    }

    #[test]
    fn eval_metrics_rejects_eval_duration_zero_with_content() {
        assert_eq!(
            classify_eval_metrics(1, 0, true),
            EvalMetricsOutcome::EvalDurationZeroWithContent
        );
    }

    #[test]
    fn eval_metrics_is_deterministic() {
        let a = classify_eval_metrics(3, 42, true);
        let b = classify_eval_metrics(3, 42, true);
        assert_eq!(a, b);
    }

    fn delta_frame(content: &str) -> Value {
        json!({"message": {"role": "assistant", "content": content}, "done": false})
    }
    fn terminal_frame() -> Value {
        json!({"done": true, "eval_count": 3, "eval_duration": 500})
    }

    #[test]
    fn ndjson_ok_on_well_formed_stream() {
        let frames = vec![
            delta_frame("he"),
            delta_frame("llo"),
            terminal_frame(),
        ];
        assert_eq!(classify_ndjson_stream(&frames), NdjsonStreamOutcome::Ok);
    }

    #[test]
    fn ndjson_ok_on_single_terminal_frame() {
        let frames = vec![terminal_frame()];
        assert_eq!(classify_ndjson_stream(&frames), NdjsonStreamOutcome::Ok);
    }

    #[test]
    fn ndjson_rejects_empty_stream() {
        let frames: Vec<Value> = vec![];
        assert_eq!(classify_ndjson_stream(&frames), NdjsonStreamOutcome::NoFrames);
    }

    #[test]
    fn ndjson_rejects_non_object_frame() {
        let frames = vec![json!([1, 2]), terminal_frame()];
        assert_eq!(
            classify_ndjson_stream(&frames),
            NdjsonStreamOutcome::FrameNotAnObject { frame_index: 0 }
        );
    }

    #[test]
    fn ndjson_rejects_missing_done_field() {
        let frames = vec![json!({"message": {"role":"assistant","content":"x"}}), terminal_frame()];
        assert_eq!(
            classify_ndjson_stream(&frames),
            NdjsonStreamOutcome::MissingDoneField { frame_index: 0 }
        );
    }

    #[test]
    fn ndjson_rejects_non_bool_done() {
        let frames = vec![json!({"done": "false"}), terminal_frame()];
        assert_eq!(
            classify_ndjson_stream(&frames),
            NdjsonStreamOutcome::DoneNotBool { frame_index: 0 }
        );
    }

    #[test]
    fn ndjson_rejects_early_done_true() {
        let frames = vec![
            delta_frame("hi"),
            json!({"done": true}),
            delta_frame("!"),
            terminal_frame(),
        ];
        assert_eq!(
            classify_ndjson_stream(&frames),
            NdjsonStreamOutcome::EarlyDoneFrame { frame_index: 1 }
        );
    }

    #[test]
    fn ndjson_rejects_no_terminal_frame() {
        let frames = vec![delta_frame("a"), delta_frame("b")];
        assert_eq!(
            classify_ndjson_stream(&frames),
            NdjsonStreamOutcome::TerminalFrameNotDone { last_frame_index: 1 }
        );
    }

    #[test]
    fn ndjson_classifier_is_deterministic() {
        let frames = vec![delta_frame("a"), terminal_frame()];
        let a = classify_ndjson_stream(&frames);
        let b = classify_ndjson_stream(&frames);
        assert_eq!(a, b);
    }

    #[test]
    fn ndjson_first_frame_done_true_alone_is_ok() {
        // i.e. single-frame stream where the one frame is terminal
        let frames = vec![terminal_frame()];
        assert_eq!(classify_ndjson_stream(&frames), NdjsonStreamOutcome::Ok);
    }

    #[test]
    fn required_keys_has_expected_count_and_order() {
        assert_eq!(OLLAMA_CHAT_REQUIRED_KEYS.len(), 10);
        assert_eq!(OLLAMA_CHAT_REQUIRED_KEYS[0], "model");
        assert_eq!(OLLAMA_CHAT_REQUIRED_KEYS[9], "eval_duration");
    }
}
